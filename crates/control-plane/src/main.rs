//! Control-plane binary: wiring, config, router, and the billing ticker.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;

use control_plane::billing;
use control_plane::build_router;
use control_plane::db;
use control_plane::lightning::{lnd::LndRest, mock::MockBackend, LightningBackend};
use control_plane::state::{AppState, BillingConfig, Clock};

#[derive(Debug, Parser)]
#[command(name = "control-plane", version)]
struct Args {
    /// Address to bind, e.g. 127.0.0.1:8080
    #[arg(long, env = "CP_BIND", default_value = "127.0.0.1:8080")]
    bind: String,

    /// SQLite database URL.
    #[arg(long, env = "DATABASE_URL", default_value = "sqlite:control-plane.db")]
    database_url: String,

    /// Lightning backend: mock | lnd.
    #[arg(long, env = "LN_BACKEND", default_value = "mock")]
    ln_backend: String,

    /// Mock invoice auto-settle delay, seconds (mock backend only).
    #[arg(long, env = "MOCK_SETTLE_SECS", default_value_t = 2)]
    mock_settle_secs: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let pool = db::connect(&args.database_url, 5)
        .await
        .context("opening database")?;

    let (ln, ln_backend_name): (Arc<dyn LightningBackend>, String) = match args.ln_backend.as_str()
    {
        "mock" => (
            Arc::new(MockBackend::new(Duration::from_secs(args.mock_settle_secs))),
            "mock".to_string(),
        ),
        "lnd" => (Arc::new(LndRest::new()?), "lnd".to_string()),
        other => anyhow::bail!("unknown LN_BACKEND: {other}"),
    };

    let state = AppState {
        pool,
        ln,
        billing: BillingConfig::default(),
        clock: Clock::System,
        ln_backend_name,
    };

    // Billing ticker.
    let ticker = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(ticker.billing.tick);
        loop {
            interval.tick().await;
            if let Err(e) = billing::tick(&ticker).await {
                tracing::warn!(error = format!("{e:#}"), "billing tick failed");
            }
        }
    });

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .with_context(|| format!("binding {}", args.bind))?;
    tracing::info!(bind = %args.bind, ln = %args.ln_backend, "control plane listening");
    axum::serve(listener, app).await?;
    Ok(())
}
