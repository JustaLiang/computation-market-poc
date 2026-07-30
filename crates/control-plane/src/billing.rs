//! The billing ticker (SPEC §5): settle invoices, mark silent hosts offline,
//! and meter running rentals — charging one period in advance.

use vgpu_core::api::Command;
use vgpu_core::model::LedgerKind;

use crate::db::{self, RentalRow};
use crate::state::AppState;

/// One tick: `poll_invoices` → `mark_offline` → `bill_once`.
pub async fn tick(state: &AppState) -> anyhow::Result<()> {
    poll_invoices(state).await?;
    mark_offline(state).await?;
    bill_once(state).await?;
    Ok(())
}

/// Settle any newly-paid invoices: credit the account and record a `deposit`.
pub async fn poll_invoices(state: &AppState) -> anyhow::Result<()> {
    let pending: Vec<(String, String, i64)> =
        sqlx::query_as("SELECT payment_hash, account_id, sats FROM invoices WHERE settled = 0")
            .fetch_all(&state.pool)
            .await?;

    for (payment_hash, account_id, sats) in pending {
        if !state.ln.is_settled(&payment_hash).await? {
            continue;
        }
        let now = state.clock.now();
        let mut conn = db::begin_immediate(&state.pool).await?;
        let res = async {
            sqlx::query("UPDATE invoices SET settled = 1 WHERE payment_hash = ?")
                .bind(&payment_hash)
                .execute(&mut *conn)
                .await?;
            sqlx::query("UPDATE accounts SET balance_sats = balance_sats + ? WHERE id = ?")
                .bind(sats)
                .bind(&account_id)
                .execute(&mut *conn)
                .await?;
            db::record(
                &mut *conn,
                now,
                LedgerKind::Deposit,
                sats,
                Some(&account_id),
                None,
                None,
                Some("deposit"),
            )
            .await?;
            Ok::<_, sqlx::Error>(())
        }
        .await;

        match res {
            Ok(()) => {
                sqlx::query("COMMIT").execute(&mut *conn).await?;
            }
            Err(e) => {
                let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
                return Err(e.into());
            }
        }
    }
    Ok(())
}

/// A host silent for longer than `HEARTBEAT_TIMEOUT` goes offline.
pub async fn mark_offline(state: &AppState) -> anyhow::Result<()> {
    let threshold = state.clock.now() - state.billing.heartbeat_timeout_secs();
    sqlx::query("UPDATE machines SET online = 0 WHERE online = 1 AND last_heartbeat < ?")
        .bind(threshold)
        .execute(&state.pool)
        .await?;
    Ok(())
}

/// Charge every running rental for the period it just entered — or evict it when
/// the balance can't cover one more minute.
pub async fn bill_once(state: &AppState) -> anyhow::Result<()> {
    let period = state.billing.bill_period_secs();
    let hb_timeout = state.billing.heartbeat_timeout_secs();

    let running: Vec<RentalRow> = sqlx::query_as("SELECT * FROM rentals WHERE status = 'running'")
        .fetch_all(&state.pool)
        .await?;

    for r in running {
        let now = state.clock.now();

        let last_heartbeat: i64 =
            sqlx::query_scalar("SELECT last_heartbeat FROM machines WHERE id=?")
                .bind(r.machine_id)
                .fetch_one(&state.pool)
                .await?;
        if now - last_heartbeat > hb_timeout {
            continue; // clock stops for a silent host (SPEC §5)
        }
        if now < r.paid_through {
            continue; // already paid for this period
        }

        let rate = r.rate_sats_per_min;
        let balance: i64 = sqlx::query_scalar("SELECT balance_sats FROM accounts WHERE id=?")
            .bind(&r.account_id)
            .fetch_one(&state.pool)
            .await?;

        let mut conn = db::begin_immediate(&state.pool).await?;
        let res = async {
            if balance < rate {
                sqlx::query(
                    "UPDATE rentals SET status='evicting', error='insufficient balance' WHERE id=?",
                )
                .bind(r.id)
                .execute(&mut *conn)
                .await?;
                let payload = serde_json::to_string(&Command::StopRental { rental_id: r.id })
                    .map_err(|e| sqlx::Error::Protocol(e.to_string()))?;
                sqlx::query("INSERT INTO commands (machine_id, payload, delivered, created_at) VALUES (?, ?, 0, ?)")
                    .bind(r.machine_id)
                    .bind(&payload)
                    .bind(now)
                    .execute(&mut *conn)
                    .await?;
                db::record(
                    &mut *conn,
                    now,
                    LedgerKind::Evict,
                    0,
                    Some(&r.account_id),
                    None,
                    Some(r.id),
                    Some("insufficient balance"),
                )
                .await?;
            } else {
                sqlx::query("UPDATE accounts SET balance_sats = balance_sats - ? WHERE id=?")
                    .bind(rate)
                    .bind(&r.account_id)
                    .execute(&mut *conn)
                    .await?;
                sqlx::query("UPDATE machines SET payout_balance = payout_balance + ? WHERE id=?")
                    .bind(rate)
                    .bind(r.machine_id)
                    .execute(&mut *conn)
                    .await?;
                // Advance by exactly one period; never assign now + period.
                let new_paid_through = std::cmp::max(r.paid_through, now - period) + period;
                sqlx::query(
                    "UPDATE rentals SET sats_charged = sats_charged + ?, \
                     minutes_billed = minutes_billed + 1, paid_through = ? WHERE id=?",
                )
                .bind(rate)
                .bind(new_paid_through)
                .bind(r.id)
                .execute(&mut *conn)
                .await?;
                db::record_charge(&mut conn, now, &r.account_id, r.machine_id, r.id, rate).await?;
            }
            Ok::<_, sqlx::Error>(())
        }
        .await;

        match res {
            Ok(()) => {
                sqlx::query("COMMIT").execute(&mut *conn).await?;
            }
            Err(e) => {
                let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
                return Err(e.into());
            }
        }
    }
    Ok(())
}
