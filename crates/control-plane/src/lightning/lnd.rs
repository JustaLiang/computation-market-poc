//! lnd REST backend — reserved (CLAUDE.md build step 9).
//!
//! The real client talks to lnd's REST API over `reqwest` with the macaroon hex
//! in `Grpc-Metadata-macaroon`, converting lnd's base64 `r_hash` to the hex we
//! store everywhere. Not wired yet — select `LN_BACKEND=mock`.

use async_trait::async_trait;
use vgpu_core::Sats;

use super::{Invoice, LightningBackend};

const UNIMPLEMENTED: &str = "the lnd backend is not implemented yet; set LN_BACKEND=mock";

pub struct LndRest;

impl LndRest {
    pub fn new() -> anyhow::Result<Self> {
        anyhow::bail!(UNIMPLEMENTED)
    }
}

#[async_trait]
impl LightningBackend for LndRest {
    async fn create_invoice(&self, _sats: Sats, _memo: &str) -> anyhow::Result<Invoice> {
        anyhow::bail!(UNIMPLEMENTED)
    }
    async fn is_settled(&self, _payment_hash: &str) -> anyhow::Result<bool> {
        anyhow::bail!(UNIMPLEMENTED)
    }
    async fn pay_invoice(&self, _bolt11: &str) -> anyhow::Result<()> {
        anyhow::bail!(UNIMPLEMENTED)
    }
}
