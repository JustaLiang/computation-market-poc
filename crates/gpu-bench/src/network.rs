//! Network probe — real internet down/up/latency.
//!
//! Unlike the GPU measurements, this is **not** self-contained: bandwidth only
//! means something against a peer. We use Cloudflare's public speedtest
//! endpoints (`speed.cloudflare.com`), the same signal the marketplace's
//! `inet_down_mbps` / `inet_up_mbps` host fields care about.
//!
//! This makes outbound HTTPS requests, so it is **opt-in** everywhere — the CLI
//! `net` subcommand or `run --network`, never the default `run`.

use std::io::Read;
use std::time::{Duration, Instant};

use crate::backend::NetworkResult;

const DOWN_URL: &str = "https://speed.cloudflare.com/__down";
const UP_URL: &str = "https://speed.cloudflare.com/__up";

/// Probe sizes. Defaults are modest to be light on the connection.
#[derive(Debug, Clone)]
pub struct NetConfig {
    pub down_bytes: u64,
    pub up_bytes: u64,
    pub latency_samples: u32,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self {
            down_bytes: 25_000_000,
            up_bytes: 10_000_000,
            latency_samples: 5,
        }
    }
}

/// Measure latency, then download, then upload throughput over one keep-alive
/// connection.
pub fn probe(cfg: &NetConfig) -> anyhow::Result<NetworkResult> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(60))
        .timeout_write(Duration::from_secs(60))
        .build();

    let latency_ms = measure_latency(&agent, cfg.latency_samples)?;
    let down_mbps = measure_download(&agent, cfg.down_bytes)?;
    let up_mbps = measure_upload(&agent, cfg.up_bytes)?;

    Ok(NetworkResult {
        down_mbps,
        up_mbps,
        latency_ms,
    })
}

/// Best (minimum) round trip over `samples` tiny requests — min rejects noise
/// from scheduling and transient congestion better than the mean.
fn measure_latency(agent: &ureq::Agent, samples: u32) -> anyhow::Result<f64> {
    let url = format!("{DOWN_URL}?bytes=0");
    let mut best = f64::INFINITY;
    for _ in 0..samples.max(1) {
        let start = Instant::now();
        let resp = agent
            .get(&url)
            .call()
            .map_err(|e| anyhow::anyhow!("latency request failed: {e}"))?;
        // Drain so the connection is reusable for the next sample.
        std::io::copy(&mut resp.into_reader(), &mut std::io::sink())?;
        best = best.min(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(best)
}

fn measure_download(agent: &ureq::Agent, bytes: u64) -> anyhow::Result<f64> {
    let url = format!("{DOWN_URL}?bytes={bytes}");
    let resp = agent
        .get(&url)
        .call()
        .map_err(|e| anyhow::anyhow!("download request failed: {e}"))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 16];

    // Time only the body transfer, not connection/TLS setup.
    let start = Instant::now();
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total += n as u64;
    }
    Ok(to_mbps(total, start.elapsed()))
}

fn measure_upload(agent: &ureq::Agent, bytes: u64) -> anyhow::Result<f64> {
    let payload = vec![0u8; bytes as usize];
    let start = Instant::now();
    agent
        .post(UP_URL)
        .send_bytes(&payload)
        .map_err(|e| anyhow::anyhow!("upload request failed: {e}"))?;
    Ok(to_mbps(bytes, start.elapsed()))
}

fn to_mbps(bytes: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64().max(1e-9);
    (bytes as f64) * 8.0 / secs / 1e6
}
