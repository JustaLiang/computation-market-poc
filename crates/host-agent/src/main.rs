//! Host agent binary (`vgpu-agent`).
//!
//! Registers the machine, then runs the heartbeat loop: all control traffic is
//! agent-initiated (SPEC §1), so the agent polls `POST /agent/heartbeat`, and
//! the control plane answers with queued commands. The agent dispatches each to
//! [`runtime`] and reports the outcome via `POST /agent/report`.
//!
//! Layers: [`benchmark`] discovers hardware, [`runtime`] runs containers, and
//! this file is the control glue between them and the control plane.

mod benchmark;
#[cfg(target_os = "macos")]
mod mac_worker;
mod runtime;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use clap::Parser;
use serde::{Deserialize, Serialize};
use sysinfo::{DiskKind, Disks, System};
use vgpu_core::api::{
    Command, HeartbeatRequest, HeartbeatResponse, RateRequest, RegisterRequest, RegisterResponse,
    ReportRequest,
};
use vgpu_core::model::{DiskType, RentalStatus};

use crate::runtime::{HostRuntime, RentalSpec};

/// Must stay below the control plane's `HEARTBEAT_TIMEOUT` (SPEC §5, 90s), or a
/// healthy host is marked offline and its billing clock stops.
const HEARTBEAT_TIMEOUT_SECS: u64 = 90;

/// Host agent for the GPU rental marketplace.
#[derive(Debug, Parser)]
#[command(name = "vgpu-agent", version)]
struct Args {
    /// Base URL of the control plane, e.g. http://cp.example.com:8080
    #[arg(long, env = "VGPU_CONTROL_PLANE")]
    control_plane: String,

    /// Public IP where tenants reach this host (SSH/workload traffic).
    #[arg(long, env = "VGPU_PUBLIC_IP")]
    public_ip: String,

    /// Provider-set price, integer sats per minute (≥ 1).
    #[arg(long, env = "VGPU_RATE")]
    rate_sats_per_min: i64,

    /// Start of the reachable host port range for tenant SSH mapping.
    #[arg(long, default_value_t = 40000)]
    port_start: i32,

    /// End (inclusive) of the reachable host port range.
    #[arg(long, default_value_t = 40099)]
    port_end: i32,

    /// Heartbeat interval in seconds. Kept well under the 90s server timeout.
    #[arg(long, default_value_t = 30)]
    heartbeat_secs: u64,

    /// Where the stable host id and issued agent token are persisted.
    #[arg(long, default_value = "vgpu-agent-state.json")]
    state_file: PathBuf,

    /// ISO-3166 alpha-2 country code (optional).
    #[arg(long)]
    country: Option<String>,

    #[arg(long)]
    inet_down_mbps: Option<f64>,

    #[arg(long)]
    inet_up_mbps: Option<f64>,
}

/// Persisted across restarts. `host_id` is generated once and is the idempotency
/// key for registration; the rest is filled in after the first successful
/// register so a restart re-uses the same identity and token.
#[derive(Debug, Default, Serialize, Deserialize)]
struct AgentState {
    host_id: String,
    #[serde(default)]
    machine_id: Option<i64>,
    #[serde(default)]
    agent_token: Option<String>,
}

fn load_or_init_state(path: &Path) -> anyhow::Result<AgentState> {
    if path.exists() {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading state file {}", path.display()))?;
        Ok(serde_json::from_str(&raw).context("parsing state file")?)
    } else {
        let state = AgentState {
            host_id: uuid::Uuid::new_v4().to_string(),
            ..Default::default()
        };
        save_state(path, &state)?;
        Ok(state)
    }
}

fn save_state(path: &Path, state: &AgentState) -> anyhow::Result<()> {
    let raw = serde_json::to_string_pretty(state)?;
    std::fs::write(path, raw).with_context(|| format!("writing state file {}", path.display()))?;
    Ok(())
}

/// Static host facts NVML doesn't cover: CPU, RAM, and scratch disk.
struct HostSpecs {
    cpu_name: String,
    cpu_cores: i32,
    ram_mb: i64,
    disk_gb: i64,
    disk_type: DiskType,
}

fn collect_host_specs() -> HostSpecs {
    let sys = System::new_all();
    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let cpu_cores = sys.cpus().len() as i32;
    let ram_mb = (sys.total_memory() / (1024 * 1024)) as i64;

    // Largest disk is the scratch volume tenants get. NVMe vs SSD needs Linux
    // /sys probing that sysinfo doesn't do; unmapped kinds fall to Unknown.
    let disks = Disks::new_with_refreshed_list();
    let (disk_gb, disk_type) = disks
        .list()
        .iter()
        .max_by_key(|d| d.total_space())
        .map(|d| {
            let gb = (d.total_space() / 1_000_000_000) as i64;
            let kind = match d.kind() {
                DiskKind::SSD => DiskType::Ssd,
                DiskKind::HDD => DiskType::Hdd,
                _ => DiskType::Unknown,
            };
            (gb, kind)
        })
        .unwrap_or((0, DiskType::Unknown));

    HostSpecs {
        cpu_name,
        cpu_cores,
        ram_mb,
        disk_gb,
        disk_type,
    }
}

/// Thin control-plane client. All `/agent/*` calls except register carry the
/// bearer token.
struct ControlPlane {
    http: reqwest::Client,
    base: String,
    token: String,
}

impl ControlPlane {
    async fn heartbeat(&self, online: bool) -> anyhow::Result<HeartbeatResponse> {
        let resp = self
            .http
            .post(format!("{}/agent/heartbeat", self.base))
            .bearer_auth(&self.token)
            .json(&HeartbeatRequest { online })
            .send()
            .await?;
        Ok(ensure_ok(resp).await?.json().await?)
    }

    async fn report(&self, req: &ReportRequest) -> anyhow::Result<()> {
        let resp = self
            .http
            .post(format!("{}/agent/report", self.base))
            .bearer_auth(&self.token)
            .json(req)
            .send()
            .await?;
        ensure_ok(resp).await?;
        Ok(())
    }

    #[allow(dead_code)] // exposed for a future `set-rate` subcommand
    async fn set_rate(&self, rate_sats_per_min: i64) -> anyhow::Result<()> {
        let resp = self
            .http
            .post(format!("{}/agent/rate", self.base))
            .bearer_auth(&self.token)
            .json(&RateRequest { rate_sats_per_min })
            .send()
            .await?;
        ensure_ok(resp).await?;
        Ok(())
    }
}

async fn register(
    http: &reqwest::Client,
    base: &str,
    req: &RegisterRequest,
) -> anyhow::Result<RegisterResponse> {
    let resp = http
        .post(format!("{base}/agent/register"))
        .json(req)
        .send()
        .await
        .context("sending register request")?;
    Ok(ensure_ok(resp).await?.json().await?)
}

async fn ensure_ok(resp: reqwest::Response) -> anyhow::Result<reqwest::Response> {
    if resp.status().is_success() {
        Ok(resp)
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("control plane returned {status}: {body}");
    }
}

/// Execute one queued command and report the outcome. Runs as its own task so a
/// slow image pull never stalls the heartbeat loop.
async fn dispatch(cp: Arc<ControlPlane>, rt: Arc<dyn HostRuntime>, cmd: Command) {
    let report = match cmd {
        Command::StartRental {
            rental_id,
            image,
            ssh_pubkey,
        } => {
            match rt
                .start_rental(RentalSpec {
                    rental_id,
                    image,
                    ssh_pubkey,
                })
                .await
            {
                Ok(started) => {
                    tracing::info!(rental_id, ssh_port = started.ssh_port, "rental running");
                    ReportRequest {
                        rental_id,
                        status: RentalStatus::Running,
                        kind: Some(started.kind),
                        ssh_port: Some(started.ssh_port),
                        container_id: Some(started.container_id),
                        error: None,
                    }
                }
                Err(e) => {
                    tracing::error!(rental_id, error = format!("{e:#}"), "start_rental failed");
                    ReportRequest {
                        rental_id,
                        status: RentalStatus::Failed,
                        kind: None,
                        ssh_port: None,
                        container_id: None,
                        error: Some(format!("{e:#}")),
                    }
                }
            }
        }
        Command::StopRental { rental_id } => match rt.stop_rental(rental_id).await {
            Ok(()) => {
                tracing::info!(rental_id, "rental stopped");
                ReportRequest {
                    rental_id,
                    status: RentalStatus::Stopped,
                    kind: None,
                    ssh_port: None,
                    container_id: None,
                    error: None,
                }
            }
            Err(e) => {
                tracing::error!(rental_id, error = format!("{e:#}"), "stop_rental failed");
                ReportRequest {
                    rental_id,
                    status: RentalStatus::Failed,
                    kind: None,
                    ssh_port: None,
                    container_id: None,
                    error: Some(format!("{e:#}")),
                }
            }
        },
    };

    if let Err(e) = cp.report(&report).await {
        tracing::error!(error = format!("{e:#}"), "failed to report command outcome");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Hidden re-entrant subcommand: on macOS the runtime launches a rental by
    // re-invoking this binary as `vgpu-agent __mac-worker <n> <port>`, which runs
    // the native Metal GEMM worker and never returns until killed. Intercept it
    // before any arg parsing or control-plane setup.
    #[cfg(target_os = "macos")]
    {
        let a: Vec<String> = std::env::args().collect();
        if a.get(1).map(String::as_str) == Some("__mac-worker") {
            let n = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048u32);
            let port = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(40000u16);
            return mac_worker::run(n, port);
        }
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    anyhow::ensure!(
        args.rate_sats_per_min >= 1,
        "rate_sats_per_min must be >= 1 (SPEC §2)"
    );
    if args.heartbeat_secs >= HEARTBEAT_TIMEOUT_SECS {
        tracing::warn!(
            heartbeat_secs = args.heartbeat_secs,
            "heartbeat interval is at/above the {HEARTBEAT_TIMEOUT_SECS}s server timeout; \
             the host may be marked offline"
        );
    }

    let mut state = load_or_init_state(&args.state_file)?;
    tracing::info!(host_id = %state.host_id, "host identity loaded");

    // Probe hardware. A box with no GPU has nothing to offer and should not run.
    let inventory = benchmark::probe().context("probing GPU inventory")?;
    let specs = collect_host_specs();
    tracing::info!(
        gpu = inventory.gpu_name(),
        gpu_count = inventory.gpu_count(),
        dlperf = inventory.dlperf,
        fingerprint = %inventory.hw_fingerprint,
        "hardware probed"
    );

    let register_req = RegisterRequest {
        host_id: state.host_id.clone(),
        gpu_name: inventory.gpu_name().to_string(),
        gpu_count: inventory.gpu_count(),
        vram_mb: inventory.vram_mb(),
        cpu_name: specs.cpu_name,
        cpu_cores: specs.cpu_cores,
        ram_mb: specs.ram_mb,
        disk_gb: specs.disk_gb,
        disk_type: specs.disk_type,
        public_ip: args.public_ip.clone(),
        port_start: args.port_start,
        port_end: args.port_end,
        inet_down_mbps: args.inet_down_mbps,
        inet_up_mbps: args.inet_up_mbps,
        country: args.country.clone(),
        dlperf: inventory.dlperf,
        rate_sats_per_min: args.rate_sats_per_min,
        hw_fingerprint: inventory.hw_fingerprint.clone(),
    };

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    let registered = register(&http, &args.control_plane, &register_req)
        .await
        .context("registering with control plane")?;
    tracing::info!(machine_id = registered.machine_id, "registered");

    // Persist the issued identity so a restart re-registers idempotently.
    state.machine_id = Some(registered.machine_id);
    state.agent_token = Some(registered.agent_token.clone());
    save_state(&args.state_file, &state)?;

    let cp = Arc::new(ControlPlane {
        http,
        base: args.control_plane.clone(),
        token: registered.agent_token,
    });

    // Connect to Docker only after registering — the machine is a valid offer as
    // soon as it is known, and command dispatch is what needs the runtime.
    let rt: Arc<dyn HostRuntime> = Arc::from(
        runtime::connect(args.port_start, args.port_end)
            .await
            .context("connecting to the host runtime")?,
    );

    // Reconcile on startup: what is already running, and how much is free.
    let running = rt.running_rentals().await.map(|r| r.len()).unwrap_or(0);
    let free = rt.available().await.map(|p| p.len()).unwrap_or(0);
    tracing::info!(running, free_ports = free, "host runtime connected");

    tracing::info!(
        interval_secs = args.heartbeat_secs,
        "entering heartbeat loop"
    );
    let mut ticker = tokio::time::interval(Duration::from_secs(args.heartbeat_secs));

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                match cp.heartbeat(true).await {
                    Ok(resp) => {
                        for cmd in resp.commands {
                            tracing::info!(?cmd, "received command");
                            tokio::spawn(dispatch(cp.clone(), rt.clone(), cmd));
                        }
                    }
                    Err(e) => tracing::warn!(error = format!("{e:#}"), "heartbeat failed"),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("shutdown requested; sending offline heartbeat");
                if let Err(e) = cp.heartbeat(false).await {
                    tracing::warn!(error = format!("{e:#}"), "final heartbeat failed");
                }
                break;
            }
        }
    }

    Ok(())
}
