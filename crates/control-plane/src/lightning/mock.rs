//! In-process Lightning mock: invoices auto-settle after a delay, so the whole
//! system runs with no node. A zero delay (the acceptance test) settles
//! immediately.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use vgpu_core::Sats;

use super::{Invoice, LightningBackend};

pub struct MockBackend {
    settle_delay: Duration,
    counter: AtomicU64,
    /// payment_hash -> instant at/after which it is considered settled.
    settle_at: Mutex<HashMap<String, Instant>>,
}

impl MockBackend {
    pub fn new(settle_delay: Duration) -> Self {
        Self {
            settle_delay,
            counter: AtomicU64::new(1),
            settle_at: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl LightningBackend for MockBackend {
    async fn create_invoice(&self, sats: Sats, _memo: &str) -> anyhow::Result<Invoice> {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        // 32-byte hex, like a real payment hash.
        let payment_hash = format!("{n:064x}");
        let bolt11 = format!("lnbcmock{}n{}", sats.get(), n);
        self.settle_at
            .lock()
            .expect("mock lock")
            .insert(payment_hash.clone(), Instant::now() + self.settle_delay);
        Ok(Invoice {
            bolt11,
            payment_hash,
        })
    }

    async fn is_settled(&self, payment_hash: &str) -> anyhow::Result<bool> {
        let map = self.settle_at.lock().expect("mock lock");
        Ok(map
            .get(payment_hash)
            .is_some_and(|at| Instant::now() >= *at))
    }

    async fn pay_invoice(&self, _bolt11: &str) -> anyhow::Result<()> {
        // A mock node always pays successfully.
        Ok(())
    }
}
