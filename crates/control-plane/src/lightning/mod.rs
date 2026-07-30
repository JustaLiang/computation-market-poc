//! Lightning payment backends behind one small trait (SPEC "Lightning").

use async_trait::async_trait;
use vgpu_core::Sats;

pub mod lnd;
pub mod mock;

/// A created invoice. We store `payment_hash` as hex everywhere.
#[derive(Debug, Clone)]
pub struct Invoice {
    pub bolt11: String,
    pub payment_hash: String,
}

/// Three methods, no more (CLAUDE.md).
#[async_trait]
pub trait LightningBackend: Send + Sync {
    async fn create_invoice(&self, sats: Sats, memo: &str) -> anyhow::Result<Invoice>;
    async fn is_settled(&self, payment_hash: &str) -> anyhow::Result<bool>;
    async fn pay_invoice(&self, bolt11: &str) -> anyhow::Result<()>;
}
