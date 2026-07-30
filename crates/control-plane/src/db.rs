//! Storage: pool, migrations, row types, and the single ledger-write path.
//!
//! Queries are runtime-checked (`query`/`query_as`), not the compile-time
//! `query!` macro, so the crate builds with no live `DATABASE_URL`. The schema
//! is still embedded at compile time by `migrate!`.

use std::str::FromStr;
use std::time::Duration;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::{Sqlite, SqliteConnection};
use vgpu_core::model::{DiskType, LedgerKind, RentalStatus};

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Open a pool at `url`, enabling foreign keys, and run migrations.
///
/// `max_connections` is 1 for an in-memory test DB (so every query hits the same
/// database) and larger for a file-backed production DB.
pub async fn connect(url: &str, max_connections: u32) -> anyhow::Result<SqlitePool> {
    let opts = SqliteConnectOptions::from_str(url)?
        .create_if_missing(true)
        .foreign_keys(true)
        .busy_timeout(Duration::from_secs(5));

    let pool = SqlitePoolOptions::new()
        .max_connections(max_connections)
        // Keep one live connection so an in-memory DB is not dropped mid-run.
        .min_connections(1)
        .connect_with(opts)
        .await?;

    MIGRATOR.run(&pool).await?;
    Ok(pool)
}

/// Acquire a connection and take a real write lock via `BEGIN IMMEDIATE`.
///
/// SQLite's deferred `BEGIN` (what `pool.begin()` issues) can fail with
/// `SQLITE_BUSY` on lock upgrade and would let two tenants race for one machine;
/// `BEGIN IMMEDIATE` takes the write lock up front (SPEC, CLAUDE.md). The caller
/// must `COMMIT` on success and `ROLLBACK` on error.
pub async fn begin_immediate(
    pool: &SqlitePool,
) -> Result<sqlx::pool::PoolConnection<Sqlite>, sqlx::Error> {
    let mut conn = pool.acquire().await?;
    sqlx::query("BEGIN IMMEDIATE").execute(&mut *conn).await?;
    Ok(conn)
}

/// Current wall-clock time as Unix seconds.
pub fn system_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before 1970")
        .as_secs() as i64
}

// --- Row types (columns decoded by name) ---------------------------------

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct MachineRow {
    pub id: i64,
    pub agent_token: String,
    pub host_id: String,
    pub gpu_name: String,
    pub gpu_count: i64,
    pub vram_mb: i64,
    pub cpu_name: String,
    pub cpu_cores: i64,
    pub ram_mb: i64,
    pub disk_gb: i64,
    pub disk_type: DiskType,
    pub public_ip: String,
    pub port_start: i64,
    pub port_end: i64,
    pub inet_down_mbps: Option<f64>,
    pub inet_up_mbps: Option<f64>,
    pub country: Option<String>,
    pub dlperf: f64,
    pub rate_sats_per_min: i64,
    pub payout_balance: i64,
    pub online: bool,
    pub last_heartbeat: i64,
    pub hw_fingerprint: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct RentalRow {
    pub id: i64,
    pub machine_id: i64,
    pub account_id: String,
    pub image: String,
    pub ssh_pubkey: String,
    pub status: RentalStatus,
    pub ssh_host: Option<String>,
    pub ssh_port: Option<i64>,
    pub container_id: Option<String>,
    pub error: Option<String>,
    pub rate_sats_per_min: i64,
    pub sats_charged: i64,
    pub minutes_billed: i64,
    pub paid_through: i64,
    pub created_at: i64,
    pub ended_at: Option<i64>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct AccountRow {
    pub id: String,
    pub balance_sats: i64,
    pub created_at: i64,
}

// --- Ledger (double-entry, append-only) ----------------------------------

/// The **only** way to write to `ledger`. Never insert from a route or the
/// ticker directly (CLAUDE.md). Generic over the executor so it works against a
/// pool or inside an explicit transaction.
#[allow(clippy::too_many_arguments)]
pub async fn record<'e, E>(
    executor: E,
    ts: i64,
    kind: LedgerKind,
    delta_sats: i64,
    account_id: Option<&str>,
    machine_id: Option<i64>,
    rental_id: Option<i64>,
    note: Option<&str>,
) -> Result<(), sqlx::Error>
where
    E: sqlx::Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        "INSERT INTO ledger (ts, account_id, machine_id, rental_id, delta_sats, kind, note) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(ts)
    .bind(account_id)
    .bind(machine_id)
    .bind(rental_id)
    .bind(delta_sats)
    .bind(kind)
    .bind(note)
    .execute(executor)
    .await?;
    Ok(())
}

/// A metered charge: debit the account, credit the host, in two ledger rows that
/// must be written inside the same transaction as the balance mutations. The two
/// rows sum to zero, preserving `SUM(ledger.delta_sats) == 0`.
pub async fn record_charge(
    conn: &mut SqliteConnection,
    ts: i64,
    account_id: &str,
    machine_id: i64,
    rental_id: i64,
    rate: i64,
) -> Result<(), sqlx::Error> {
    record(
        &mut *conn,
        ts,
        LedgerKind::RentalCharge,
        -rate,
        Some(account_id),
        None,
        Some(rental_id),
        None,
    )
    .await?;
    record(
        &mut *conn,
        ts,
        LedgerKind::HostCredit,
        rate,
        None,
        Some(machine_id),
        Some(rental_id),
        None,
    )
    .await?;
    Ok(())
}
