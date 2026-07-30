//! HTTP routes, split into the agent-facing and tenant-facing surfaces.

pub mod agent;
pub mod tenant;

use std::sync::atomic::{AtomicU64, Ordering};

use axum::http::{header::AUTHORIZATION, HeaderMap};

use crate::db::MachineRow;
use crate::error::ApiError;
use crate::state::AppState;

/// Resolve `Authorization: Bearer <agent_token>` to the owning machine, or 401.
pub(crate) async fn authenticate(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<MachineRow, ApiError> {
    let token = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .ok_or_else(|| ApiError::unauthorized("missing or malformed bearer token"))?;

    let machine: Option<MachineRow> =
        sqlx::query_as("SELECT * FROM machines WHERE agent_token = ?")
            .bind(token)
            .fetch_optional(&state.pool)
            .await?;

    machine.ok_or_else(|| ApiError::unauthorized("invalid agent token"))
}

/// POC-grade unique id source. Not cryptographically strong — agent tokens and
/// account ids are POC-grade bearer secrets (SPEC §8).
fn mix() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17)
}

pub(crate) fn new_agent_token() -> String {
    format!("agt_{:016x}", mix())
}

pub(crate) fn new_account_id() -> String {
    format!("acct_{:016x}", mix())
}
