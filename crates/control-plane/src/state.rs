//! Shared application state and the billing configuration.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sqlx::SqlitePool;

use crate::lightning::LightningBackend;

/// Billing timing (SPEC §5). Held on state, never a `const`, so it has exactly
/// one definition and the acceptance test can compress `bill_period` to 2s.
#[derive(Clone)]
pub struct BillingConfig {
    pub bill_period: Duration,
    pub tick: Duration,
    pub heartbeat_timeout: Duration,
}

impl Default for BillingConfig {
    fn default() -> Self {
        Self {
            bill_period: Duration::from_secs(60),
            tick: Duration::from_secs(5),
            heartbeat_timeout: Duration::from_secs(90),
        }
    }
}

impl BillingConfig {
    pub fn bill_period_secs(&self) -> i64 {
        self.bill_period.as_secs() as i64
    }
    pub fn heartbeat_timeout_secs(&self) -> i64 {
        self.heartbeat_timeout.as_secs() as i64
    }
}

/// Time source. `System` is wall-clock; `Manual` lets the acceptance test
/// advance time deterministically instead of sleeping through real seconds.
#[derive(Clone)]
pub enum Clock {
    System,
    Manual(Arc<AtomicI64>),
}

impl Clock {
    pub fn manual(start: i64) -> Self {
        Clock::Manual(Arc::new(AtomicI64::new(start)))
    }

    /// Unix seconds.
    pub fn now(&self) -> i64 {
        match self {
            Clock::System => crate::db::system_now(),
            Clock::Manual(t) => t.load(Ordering::SeqCst),
        }
    }

    /// Advance a manual clock (no-op for `System`).
    pub fn advance(&self, secs: i64) {
        if let Clock::Manual(t) = self {
            t.fetch_add(secs, Ordering::SeqCst);
        }
    }
}

/// Everything a handler or the ticker needs. Cheap to clone (pool and Arc).
#[derive(Clone)]
pub struct AppState {
    pub pool: SqlitePool,
    pub ln: Arc<dyn LightningBackend>,
    pub billing: BillingConfig,
    pub clock: Clock,
    pub ln_backend_name: String,
}
