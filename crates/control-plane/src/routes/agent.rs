//! Agent-facing routes (`/agent/*`). All control traffic is agent-initiated
//! (SPEC §1): commands are queued server-side and delivered in the heartbeat
//! response.

use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use vgpu_core::api::{
    Command, HeartbeatRequest, HeartbeatResponse, PayoutRequest, PayoutResponse, RateRequest,
    RegisterRequest, RegisterResponse, ReportRequest,
};
use vgpu_core::model::LedgerKind;

use crate::db::{self, MachineRow, RentalRow};
use crate::error::ApiError;
use crate::routes::{authenticate, new_agent_token};
use crate::state::AppState;

/// `POST /agent/register` — idempotent on `host_id`.
pub async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, ApiError> {
    let now = state.clock.now();

    let existing: Option<MachineRow> = sqlx::query_as("SELECT * FROM machines WHERE host_id = ?")
        .bind(&req.host_id)
        .fetch_optional(&state.pool)
        .await?;

    if let Some(m) = existing {
        if (req.gpu_count as i64) < m.gpu_count {
            tracing::warn!(
                host_id = %req.host_id,
                previous = m.gpu_count,
                now = req.gpu_count,
                "machine reports less hardware than before (de-verification signal)"
            );
        }
        // Re-registration (restart) brings the machine back online.
        sqlx::query(
            "UPDATE machines SET gpu_name=?, gpu_count=?, vram_mb=?, cpu_name=?, cpu_cores=?, \
             ram_mb=?, disk_gb=?, disk_type=?, public_ip=?, port_start=?, port_end=?, \
             inet_down_mbps=?, inet_up_mbps=?, country=?, dlperf=?, rate_sats_per_min=?, \
             hw_fingerprint=?, online=1, last_heartbeat=? WHERE id=?",
        )
        .bind(req.gpu_name.as_str())
        .bind(req.gpu_count)
        .bind(req.vram_mb)
        .bind(req.cpu_name.as_str())
        .bind(req.cpu_cores)
        .bind(req.ram_mb)
        .bind(req.disk_gb)
        .bind(req.disk_type)
        .bind(req.public_ip.as_str())
        .bind(req.port_start)
        .bind(req.port_end)
        .bind(req.inet_down_mbps)
        .bind(req.inet_up_mbps)
        .bind(req.country.as_deref())
        .bind(req.dlperf)
        .bind(req.rate_sats_per_min)
        .bind(req.hw_fingerprint.as_str())
        .bind(now)
        .bind(m.id)
        .execute(&state.pool)
        .await?;
        return Ok(Json(RegisterResponse {
            machine_id: m.id,
            agent_token: m.agent_token,
        }));
    }

    let token = new_agent_token();
    // A newly-registered machine is online with a fresh heartbeat, so it appears
    // in the offer index immediately (SPEC §10 step 2).
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO machines (agent_token, host_id, gpu_name, gpu_count, vram_mb, cpu_name, \
         cpu_cores, ram_mb, disk_gb, disk_type, public_ip, port_start, port_end, inet_down_mbps, \
         inet_up_mbps, country, dlperf, rate_sats_per_min, hw_fingerprint, online, last_heartbeat, \
         created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?) RETURNING id",
    )
    .bind(token.as_str())
    .bind(req.host_id.as_str())
    .bind(req.gpu_name.as_str())
    .bind(req.gpu_count)
    .bind(req.vram_mb)
    .bind(req.cpu_name.as_str())
    .bind(req.cpu_cores)
    .bind(req.ram_mb)
    .bind(req.disk_gb)
    .bind(req.disk_type)
    .bind(req.public_ip.as_str())
    .bind(req.port_start)
    .bind(req.port_end)
    .bind(req.inet_down_mbps)
    .bind(req.inet_up_mbps)
    .bind(req.country.as_deref())
    .bind(req.dlperf)
    .bind(req.rate_sats_per_min)
    .bind(req.hw_fingerprint.as_str())
    .bind(now) // last_heartbeat
    .bind(now) // created_at
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(RegisterResponse {
        machine_id: id,
        agent_token: token,
    }))
}

/// `POST /agent/heartbeat` — refresh liveness, drain the command queue (marking
/// delivered) in one transaction. Delivery is at-most-once.
pub async fn heartbeat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<HeartbeatRequest>,
) -> Result<Json<HeartbeatResponse>, ApiError> {
    let machine = authenticate(&state, &headers).await?;
    let now = state.clock.now();

    let mut conn = db::begin_immediate(&state.pool).await?;
    let outcome = async {
        sqlx::query("UPDATE machines SET last_heartbeat=?, online=? WHERE id=?")
            .bind(now)
            .bind(req.online)
            .bind(machine.id)
            .execute(&mut *conn)
            .await?;

        let rows: Vec<(i64, String)> = sqlx::query_as(
            "SELECT id, payload FROM commands WHERE machine_id=? AND delivered=0 ORDER BY id",
        )
        .bind(machine.id)
        .fetch_all(&mut *conn)
        .await?;

        for (id, _) in &rows {
            sqlx::query("UPDATE commands SET delivered=1 WHERE id=?")
                .bind(id)
                .execute(&mut *conn)
                .await?;
        }

        let mut commands = Vec::with_capacity(rows.len());
        for (_, payload) in &rows {
            commands.push(serde_json::from_str::<Command>(payload)?);
        }
        Ok::<_, ApiError>(commands)
    }
    .await;

    match outcome {
        Ok(commands) => {
            sqlx::query("COMMIT").execute(&mut *conn).await?;
            Ok(Json(HeartbeatResponse { commands }))
        }
        Err(e) => {
            let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
            Err(e)
        }
    }
}

/// `POST /agent/report` — the agent reports a command's outcome.
pub async fn report(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReportRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let machine = authenticate(&state, &headers).await?;

    let rental: RentalRow = sqlx::query_as("SELECT * FROM rentals WHERE id=? AND machine_id=?")
        .bind(req.rental_id)
        .bind(machine.id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or_else(|| ApiError::not_found("no such rental for this machine"))?;

    let now = state.clock.now();
    let terminal = req.status.is_terminal();
    let ended_at: Option<i64> = terminal.then_some(now);
    // Bill the whole rental once, when a still-live rental first reaches a
    // terminal state — for `(ended_at − created_at) × rate` (charge-at-stop).
    // Re-reports of an already-terminal rental never re-charge.
    let settle = terminal && !rental.status.is_terminal();

    let mut conn = db::begin_immediate(&state.pool).await?;
    let res = async {
        sqlx::query(
            "UPDATE rentals SET status=?, kind=COALESCE(?, kind), ssh_host=?, ssh_port=?, \
             container_id=COALESCE(?, container_id), error=?, ended_at=COALESCE(?, ended_at) WHERE id=?",
        )
        .bind(req.status)
        .bind(req.kind)
        .bind(machine.public_ip.as_str())
        .bind(req.ssh_port)
        .bind(req.container_id.as_deref())
        .bind(req.error.as_deref())
        .bind(ended_at)
        .bind(req.rental_id)
        .execute(&mut *conn)
        .await?;

        if settle {
            let accrued =
                crate::billing::accrued_charge(rental.created_at, now, rental.rate_sats_per_min);
            let balance: i64 = sqlx::query_scalar("SELECT balance_sats FROM accounts WHERE id=?")
                .bind(&rental.account_id)
                .fetch_one(&mut *conn)
                .await?;
            // Custodial: never debit below zero. The ticker's eviction valve
            // normally keeps the balance ample; this cap is the backstop.
            let charge = accrued.min(balance);
            if charge > 0 {
                sqlx::query("UPDATE accounts SET balance_sats = balance_sats - ? WHERE id=?")
                    .bind(charge)
                    .bind(&rental.account_id)
                    .execute(&mut *conn)
                    .await?;
                sqlx::query("UPDATE machines SET payout_balance = payout_balance + ? WHERE id=?")
                    .bind(charge)
                    .bind(machine.id)
                    .execute(&mut *conn)
                    .await?;
                db::record_charge(&mut conn, now, &rental.account_id, machine.id, rental.id, charge)
                    .await?;
            }
            let minutes = (now - rental.created_at).max(0) / 60;
            sqlx::query("UPDATE rentals SET sats_charged=?, minutes_billed=? WHERE id=?")
                .bind(charge)
                .bind(minutes)
                .bind(rental.id)
                .execute(&mut *conn)
                .await?;
        }
        Ok::<_, ApiError>(())
    }
    .await;

    match res {
        Ok(()) => {
            sqlx::query("COMMIT").execute(&mut *conn).await?;
            Ok(Json(serde_json::json!({ "ok": true })))
        }
        Err(e) => {
            let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
            Err(e)
        }
    }
}

/// `POST /agent/rate` — update the machine's price. Live rentals are unaffected
/// (their rate was snapshotted at creation).
pub async fn rate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RateRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let machine = authenticate(&state, &headers).await?;
    if req.rate_sats_per_min < 1 {
        return Err(ApiError::bad_request("rate_sats_per_min must be >= 1"));
    }
    sqlx::query("UPDATE machines SET rate_sats_per_min=? WHERE id=?")
        .bind(req.rate_sats_per_min)
        .bind(machine.id)
        .execute(&state.pool)
        .await?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

/// `POST /agent/payout` — pay the host's invoice, then zero `payout_balance`.
/// On payment failure the balance must remain intact (502).
pub async fn payout(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<PayoutRequest>,
) -> Result<Json<PayoutResponse>, ApiError> {
    let machine = authenticate(&state, &headers).await?;
    if machine.payout_balance <= 0 {
        return Err(ApiError::bad_request("nothing owed"));
    }

    // Pay first. If it fails, return 502 without touching the balance.
    state
        .ln
        .pay_invoice(&req.bolt11)
        .await
        .map_err(|e| ApiError::bad_gateway(format!("payout failed: {e}")))?;

    let balance = machine.payout_balance;
    let now = state.clock.now();

    let mut conn = db::begin_immediate(&state.pool).await?;
    let outcome = async {
        sqlx::query("UPDATE machines SET payout_balance=0 WHERE id=?")
            .bind(machine.id)
            .execute(&mut *conn)
            .await?;
        db::record(
            &mut *conn,
            now,
            LedgerKind::Payout,
            -balance,
            None,
            Some(machine.id),
            None,
            Some("payout"),
        )
        .await?;
        Ok::<_, ApiError>(())
    }
    .await;

    match outcome {
        Ok(()) => {
            sqlx::query("COMMIT").execute(&mut *conn).await?;
            Ok(Json(PayoutResponse { paid_sats: balance }))
        }
        Err(e) => {
            let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
            Err(e)
        }
    }
}
