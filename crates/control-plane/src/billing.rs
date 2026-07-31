//! The billing ticker: settle invoices, mark silent hosts offline, and enforce
//! per-rental budgets. Money moves at stop, not per period — the single charge
//! (`elapsed × rate`) is applied in `agent::report` when a rental goes terminal.
//! This ticker moves no money except deposits; it only *evicts* a rental whose
//! accrued cost has outrun its account balance, so the stop-time charge can then
//! settle it.

use vgpu_core::api::Command;
use vgpu_core::model::LedgerKind;

use crate::db::{self, RentalRow};
use crate::state::AppState;

/// One tick: `poll_invoices` → `mark_offline` → `enforce_budgets`.
pub async fn tick(state: &AppState) -> anyhow::Result<()> {
    poll_invoices(state).await?;
    mark_offline(state).await?;
    enforce_budgets(state).await?;
    Ok(())
}

/// Prorated charge for an interval, in whole sats: `elapsed_secs × rate ÷ 60`
/// (rate is per minute). Integer arithmetic — sub-sat fractions are truncated,
/// never rounded up, so the tenant is never overcharged. Negative elapsed (clock
/// skew) clamps to zero. Shared by the eviction check and the stop-time charge.
pub fn accrued_charge(start: i64, end: i64, rate_sats_per_min: i64) -> i64 {
    let elapsed = (end - start).max(0);
    elapsed * rate_sats_per_min / 60
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

/// Evict any running rental whose *accrued* cost `(now − created_at) × rate` has
/// outgrown its account balance: queue `stop_rental` so the single stop-time
/// charge (capped at the balance) settles it and the machine frees. No money
/// moves here. A silent host's clock is stopped (SPEC §5), so it is skipped.
pub async fn enforce_budgets(state: &AppState) -> anyhow::Result<()> {
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

        let accrued = accrued_charge(r.created_at, now, r.rate_sats_per_min);
        let balance: i64 = sqlx::query_scalar("SELECT balance_sats FROM accounts WHERE id=?")
            .bind(&r.account_id)
            .fetch_one(&state.pool)
            .await?;
        if accrued <= balance {
            continue; // still within budget — no charge, no eviction
        }

        // Over budget → evict. The stop-time charge in `agent::report` settles it.
        let mut conn = db::begin_immediate(&state.pool).await?;
        let res = async {
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
