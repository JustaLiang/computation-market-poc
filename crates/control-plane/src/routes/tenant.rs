//! Tenant-facing routes: `/offers`, `/accounts`, `/rentals`, `/health`.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use sqlx::QueryBuilder;
use vgpu_core::api::{
    AccountResponse, Command, CreateAccountResponse, CreateRentalRequest, CreateRentalResponse,
    DepositRequest, DepositResponse, HealthResponse, Offer, OffersResponse, RentalResponse,
};
use vgpu_core::model::{is_http_status_image, RentalKind, RentalStatus};
use vgpu_core::Sats;

use crate::db::{self, AccountRow, MachineRow, RentalRow};
use crate::error::ApiError;
use crate::routes::new_account_id;
use crate::state::AppState;

/// `GET /health`.
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        ln_backend: state.ln_backend_name.clone(),
    })
}

/// Occupying statuses as a SQL list, derived from [`RentalStatus::occupies_machine`]
/// rather than hardcoded strings.
fn occupying_in_list() -> String {
    RentalStatus::ALL
        .iter()
        .filter(|s| s.occupies_machine())
        .map(|s| format!("'{}'", s.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

fn machine_to_offer(m: MachineRow) -> Offer {
    Offer {
        machine_id: m.id,
        gpu_name: m.gpu_name,
        gpu_count: m.gpu_count as i32,
        vram_mb: m.vram_mb,
        cpu_name: m.cpu_name,
        cpu_cores: m.cpu_cores as i32,
        ram_mb: m.ram_mb,
        disk_gb: m.disk_gb,
        disk_type: m.disk_type,
        public_ip: m.public_ip,
        country: m.country,
        inet_down_mbps: m.inet_down_mbps,
        inet_up_mbps: m.inet_up_mbps,
        dlperf: m.dlperf,
        rate_sats_per_min: m.rate_sats_per_min,
        rate_sats_per_hour: m.rate_sats_per_min * 60,
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct OffersQuery {
    gpu_name: Option<String>,
    min_vram_mb: Option<i64>,
    min_gpu_count: Option<i64>,
    max_rate_sats_per_min: Option<i64>,
    sort: Option<String>,
    limit: Option<i64>,
}

/// `GET /offers` — online, idle machines, filtered and sorted. Default sort is
/// `value` = sats per unit of dlperf, ascending (SPEC §7).
pub async fn offers(
    State(state): State<AppState>,
    Query(q): Query<OffersQuery>,
) -> Result<Json<OffersResponse>, ApiError> {
    let mut qb: QueryBuilder<sqlx::Sqlite> = QueryBuilder::new(
        "SELECT * FROM machines WHERE online = 1 AND NOT EXISTS \
         (SELECT 1 FROM rentals r WHERE r.machine_id = machines.id AND r.status IN (",
    );
    qb.push(occupying_in_list()); // trusted, enum-derived
    qb.push("))");

    if let Some(name) = &q.gpu_name {
        qb.push(" AND gpu_name LIKE ")
            .push_bind(format!("%{name}%"));
    }
    if let Some(v) = q.min_vram_mb {
        qb.push(" AND vram_mb >= ").push_bind(v);
    }
    if let Some(c) = q.min_gpu_count {
        qb.push(" AND gpu_count >= ").push_bind(c);
    }
    if let Some(r) = q.max_rate_sats_per_min {
        qb.push(" AND rate_sats_per_min <= ").push_bind(r);
    }

    qb.push(match q.sort.as_deref() {
        Some("rate") => " ORDER BY rate_sats_per_min ASC",
        Some("dlperf") => " ORDER BY dlperf DESC",
        // value (default): sats per unit of performance, ascending.
        _ => " ORDER BY (CAST(rate_sats_per_min AS REAL) / max(dlperf, 0.01)) ASC",
    });

    let limit = q.limit.unwrap_or(200).clamp(1, 200);
    qb.push(" LIMIT ").push_bind(limit);

    let machines: Vec<MachineRow> = qb
        .build_query_as::<MachineRow>()
        .fetch_all(&state.pool)
        .await?;

    Ok(Json(OffersResponse {
        offers: machines.into_iter().map(machine_to_offer).collect(),
    }))
}

/// `POST /accounts`.
pub async fn create_account(
    State(state): State<AppState>,
) -> Result<(StatusCode, Json<CreateAccountResponse>), ApiError> {
    let id = new_account_id();
    let now = state.clock.now();
    sqlx::query("INSERT INTO accounts (id, balance_sats, created_at) VALUES (?, 0, ?)")
        .bind(&id)
        .bind(now)
        .execute(&state.pool)
        .await?;
    Ok((
        StatusCode::CREATED,
        Json(CreateAccountResponse { account_id: id }),
    ))
}

/// `GET /accounts/{id}`.
pub async fn get_account(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<AccountResponse>, ApiError> {
    let acct: AccountRow = sqlx::query_as("SELECT * FROM accounts WHERE id=?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or_else(|| ApiError::not_found("unknown account"))?;

    let burn: i64 = sqlx::query_scalar(
        "SELECT COALESCE(SUM(rate_sats_per_min), 0) FROM rentals WHERE account_id=? AND status=?",
    )
    .bind(&id)
    .bind(RentalStatus::Running.as_str())
    .fetch_one(&state.pool)
    .await?;

    let runway_minutes = (burn > 0).then(|| acct.balance_sats / burn);

    Ok(Json(AccountResponse {
        account_id: acct.id,
        balance_sats: acct.balance_sats,
        burn_sats_per_min: burn,
        runway_minutes,
    }))
}

/// `POST /accounts/{id}/deposit`.
pub async fn deposit(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<DepositRequest>,
) -> Result<Json<DepositResponse>, ApiError> {
    if req.sats < 1 {
        return Err(ApiError::bad_request("deposit must be >= 1 sat"));
    }
    let exists: Option<(String,)> = sqlx::query_as("SELECT id FROM accounts WHERE id=?")
        .bind(&id)
        .fetch_optional(&state.pool)
        .await?;
    if exists.is_none() {
        return Err(ApiError::not_found("unknown account"));
    }

    let invoice = state
        .ln
        .create_invoice(Sats(req.sats), &format!("deposit for {id}"))
        .await
        .map_err(|e| ApiError::bad_gateway(format!("invoice creation failed: {e}")))?;

    let now = state.clock.now();
    sqlx::query(
        "INSERT INTO invoices (payment_hash, account_id, sats, bolt11, settled, created_at) \
         VALUES (?, ?, ?, ?, 0, ?)",
    )
    .bind(&invoice.payment_hash)
    .bind(&id)
    .bind(req.sats)
    .bind(&invoice.bolt11)
    .bind(now)
    .execute(&state.pool)
    .await?;

    Ok(Json(DepositResponse {
        bolt11: invoice.bolt11,
        payment_hash: invoice.payment_hash,
        sats: req.sats,
    }))
}

/// `POST /rentals`. The first minute is charged up front, before any container
/// exists, inside one serialized `BEGIN IMMEDIATE` transaction (SPEC §5).
pub async fn create_rental(
    State(state): State<AppState>,
    Json(req): Json<CreateRentalRequest>,
) -> Result<(StatusCode, Json<CreateRentalResponse>), ApiError> {
    let now = state.clock.now();
    let period = state.billing.bill_period_secs();

    let mut conn = db::begin_immediate(&state.pool).await?;
    let outcome = async {
        let machine: MachineRow = sqlx::query_as("SELECT * FROM machines WHERE id=?")
            .bind(req.machine_id)
            .fetch_optional(&mut *conn)
            .await?
            .ok_or_else(|| ApiError::not_found("unknown machine"))?;

        let account: AccountRow = sqlx::query_as("SELECT * FROM accounts WHERE id=?")
            .bind(&req.account_id)
            .fetch_optional(&mut *conn)
            .await?
            .ok_or_else(|| ApiError::not_found("unknown account"))?;

        // SSH key gates the SSH/container tier; the HTTP-status (`metal:`) tier
        // runs a host-controlled job that ignores it, so accept its absence there.
        let ssh_pubkey = match req.ssh_pubkey.as_deref().map(str::trim) {
            Some(k) if !k.is_empty() => k.to_string(),
            _ if is_http_status_image(&req.image) => String::new(),
            _ => {
                return Err(ApiError::bad_request(
                    "ssh_pubkey is required for this image (only metal: images may omit it)",
                ))
            }
        };

        if !machine.online {
            return Err(ApiError::conflict("machine is offline"));
        }

        let occ_sql = format!(
            "SELECT id FROM rentals WHERE machine_id=? AND status IN ({}) LIMIT 1",
            occupying_in_list()
        );
        let occupied: Option<(i64,)> = sqlx::query_as(&occ_sql)
            .bind(req.machine_id)
            .fetch_optional(&mut *conn)
            .await?;
        if occupied.is_some() {
            return Err(ApiError::conflict("machine already rented"));
        }

        let rate = machine.rate_sats_per_min; // snapshot
        if account.balance_sats < rate {
            return Err(ApiError::payment_required(
                "balance below one minute at the machine's rate",
            ));
        }

        let paid_through = now + period;
        let rental_id: i64 = sqlx::query_scalar(
            "INSERT INTO rentals (machine_id, account_id, image, ssh_pubkey, status, \
             rate_sats_per_min, sats_charged, minutes_billed, paid_through, created_at) \
             VALUES (?, ?, ?, ?, 'provisioning', ?, ?, 1, ?, ?) RETURNING id",
        )
        .bind(req.machine_id)
        .bind(&req.account_id)
        .bind(&req.image)
        .bind(&ssh_pubkey)
        .bind(rate)
        .bind(rate)
        .bind(paid_through)
        .bind(now)
        .fetch_one(&mut *conn)
        .await?;

        sqlx::query("UPDATE accounts SET balance_sats = balance_sats - ? WHERE id=?")
            .bind(rate)
            .bind(&req.account_id)
            .execute(&mut *conn)
            .await?;
        sqlx::query("UPDATE machines SET payout_balance = payout_balance + ? WHERE id=?")
            .bind(rate)
            .bind(req.machine_id)
            .execute(&mut *conn)
            .await?;

        db::record_charge(
            &mut conn,
            now,
            &req.account_id,
            req.machine_id,
            rental_id,
            rate,
        )
        .await?;

        let payload = serde_json::to_string(&Command::StartRental {
            rental_id,
            image: req.image.clone(),
            ssh_pubkey: ssh_pubkey.clone(),
        })?;
        sqlx::query(
            "INSERT INTO commands (machine_id, payload, delivered, created_at) VALUES (?, ?, 0, ?)",
        )
        .bind(req.machine_id)
        .bind(&payload)
        .bind(now)
        .execute(&mut *conn)
        .await?;

        Ok::<_, ApiError>(CreateRentalResponse {
            rental_id,
            status: RentalStatus::Provisioning,
            rate_sats_per_min: rate,
        })
    }
    .await;

    match outcome {
        Ok(resp) => {
            sqlx::query("COMMIT").execute(&mut *conn).await?;
            Ok((StatusCode::CREATED, Json(resp)))
        }
        Err(e) => {
            let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
            Err(e)
        }
    }
}

/// `GET /rentals/{id}`. Never returns `ssh_pubkey`.
pub async fn get_rental(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<RentalResponse>, ApiError> {
    let r: RentalRow = sqlx::query_as("SELECT * FROM rentals WHERE id=?")
        .bind(id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or_else(|| ApiError::not_found("unknown rental"))?;

    let (gpu_name, gpu_count): (String, i64) =
        sqlx::query_as("SELECT gpu_name, gpu_count FROM machines WHERE id=?")
            .bind(r.machine_id)
            .fetch_one(&state.pool)
            .await?;

    // `ssh_command` is meaningful only for the SSH (container) tier; HTTP-kind
    // rentals get their hint rendered by the client from `kind` + host/port.
    let ssh_command = match (r.kind, &r.ssh_host, r.ssh_port) {
        (RentalKind::Ssh, Some(host), Some(port)) => Some(format!("ssh -p {port} root@{host}")),
        _ => None,
    };

    Ok(Json(RentalResponse {
        id: r.id,
        machine_id: r.machine_id,
        account_id: r.account_id,
        image: r.image,
        status: r.status,
        kind: r.kind,
        gpu_name,
        gpu_count: gpu_count as i32,
        ssh_host: r.ssh_host,
        ssh_port: r.ssh_port.map(|p| p as i32),
        ssh_command,
        rate_sats_per_min: r.rate_sats_per_min,
        sats_charged: r.sats_charged,
        minutes_billed: r.minutes_billed,
        paid_through: r.paid_through,
        error: r.error,
        created_at: r.created_at,
        ended_at: r.ended_at,
    }))
}

/// `DELETE /rentals/{id}` — transition to `evicting`, queue `stop_rental`.
/// Idempotent for already-terminal or already-evicting rentals.
pub async fn delete_rental(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let r: RentalRow = sqlx::query_as("SELECT * FROM rentals WHERE id=?")
        .bind(id)
        .fetch_optional(&state.pool)
        .await?
        .ok_or_else(|| ApiError::not_found("unknown rental"))?;

    if r.status.is_terminal() || r.status == RentalStatus::Evicting {
        return Ok(Json(
            serde_json::json!({ "rental_id": r.id, "status": r.status.as_str() }),
        ));
    }

    let now = state.clock.now();
    let mut conn = db::begin_immediate(&state.pool).await?;
    let outcome = async {
        sqlx::query("UPDATE rentals SET status='evicting' WHERE id=?")
            .bind(r.id)
            .execute(&mut *conn)
            .await?;
        let payload = serde_json::to_string(&Command::StopRental { rental_id: r.id })?;
        sqlx::query(
            "INSERT INTO commands (machine_id, payload, delivered, created_at) VALUES (?, ?, 0, ?)",
        )
        .bind(r.machine_id)
        .bind(&payload)
        .bind(now)
        .execute(&mut *conn)
        .await?;
        Ok::<_, ApiError>(())
    }
    .await;

    match outcome {
        Ok(()) => {
            sqlx::query("COMMIT").execute(&mut *conn).await?;
            Ok(Json(
                serde_json::json!({ "rental_id": r.id, "status": "evicting" }),
            ))
        }
        Err(e) => {
            let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
            Err(e)
        }
    }
}
