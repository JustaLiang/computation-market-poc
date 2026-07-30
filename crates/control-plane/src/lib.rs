//! Control plane: a rendezvous and accounting service (SPEC §1) — an offer
//! index, a custodial sats ledger, and a command queue. It is **not** a proxy;
//! tenant workload traffic never traverses it.
//!
//! Exposed as a library so the acceptance test can build the router and drive it
//! with `tower::ServiceExt::oneshot`, and the binary in `main.rs` wires the same
//! pieces to a real port.

pub mod billing;
pub mod db;
pub mod error;
pub mod lightning;
pub mod routes;
pub mod state;

use axum::routing::{get, post};
use axum::Router;

use crate::state::AppState;

/// Build the full router over the given state.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Tenant
        .route("/health", get(routes::tenant::health))
        .route("/offers", get(routes::tenant::offers))
        .route("/accounts", post(routes::tenant::create_account))
        .route("/accounts/:id", get(routes::tenant::get_account))
        .route("/accounts/:id/deposit", post(routes::tenant::deposit))
        .route("/rentals", post(routes::tenant::create_rental))
        .route(
            "/rentals/:id",
            get(routes::tenant::get_rental).delete(routes::tenant::delete_rental),
        )
        // Agent
        .route("/agent/register", post(routes::agent::register))
        .route("/agent/heartbeat", post(routes::agent::heartbeat))
        .route("/agent/report", post(routes::agent::report))
        .route("/agent/rate", post(routes::agent::rate))
        .route("/agent/payout", post(routes::agent::payout))
        .with_state(state)
}
