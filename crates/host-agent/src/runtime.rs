//! Workload runtime — the swappable, platform-specific launcher.
//!
//! The [`HostRuntime`] trait is the narrow surface the agent's control loop
//! depends on (`start_rental` / `stop_rental` / `running_rentals` / `available`).
//! It has two implementations, selected by target OS at compile time:
//!
//! - **Linux** (`docker`): tenant containers via `bollard` with `--gpus all`
//!   GPU passthrough. This is the real host tier (SPEC).
//! - **macOS** (`mac`): an **MLX-only** runtime — the host runs an allowlisted,
//!   host-controlled MLX job (fp16 GEMM) and *never* tenant code. Parameters,
//!   not a shell: that bounded surface is the isolation model in lieu of a
//!   microVM. Needs a Python with `mlx` (`VGPU_MLX_PYTHON`, default `python3`).
//!
//! Same `vgpu-agent` binary, same flags, same control-plane protocol; only the
//! guts differ. `main.rs` holds a `dyn HostRuntime` and never names a backend.
//! No `bollard` type appears outside the `docker` module (CLAUDE.md).

use async_trait::async_trait;
use vgpu_core::model::RentalKind;

/// What to launch. A control-plane `start_rental` command maps onto this.
///
/// `ssh_pubkey` is consumed only by the Docker backend (Linux); the MLX runtime
/// uses `image` (the task spec) and `rental_id` but not the key, so the allow is
/// scoped to non-Linux.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct RentalSpec {
    pub rental_id: i64,
    pub image: String,
    /// Injected as `authorized_keys` (via the `PUBLIC_KEY` convention on Linux).
    pub ssh_pubkey: String,
}

/// Result of a successful launch, reported back to the control plane.
pub struct Started {
    pub container_id: String,
    /// Host-side port, reachable at the machine's `public_ip`. SSH for the
    /// container tier; an HTTP endpoint for MLX jobs (see `kind`).
    pub ssh_port: i32,
    /// How a tenant connects — drives the client's endpoint hint.
    pub kind: RentalKind,
}

/// A workload this agent currently has running, recovered on startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunningRental {
    pub rental_id: i64,
    pub container_id: String,
    pub ssh_port: i32,
}

/// The platform-agnostic launcher surface (CLAUDE.md). `Send + Sync` so the
/// agent can hold it behind `Arc<dyn HostRuntime>` and dispatch across tasks.
#[async_trait]
pub trait HostRuntime: Send + Sync {
    async fn start_rental(&self, spec: RentalSpec) -> anyhow::Result<Started>;
    async fn stop_rental(&self, rental_id: i64) -> anyhow::Result<()>;
    async fn running_rentals(&self) -> anyhow::Result<Vec<RunningRental>>;
    /// Host ports in the machine's range not currently bound to a rental.
    async fn available(&self) -> anyhow::Result<Vec<i32>>;
}

/// Connect the runtime for this platform.
pub async fn connect(port_start: i32, port_end: i32) -> anyhow::Result<Box<dyn HostRuntime>> {
    anyhow::ensure!(
        port_start <= port_end,
        "port_start ({port_start}) must be <= port_end ({port_end})"
    );

    #[cfg(target_os = "linux")]
    let runtime: Box<dyn HostRuntime> =
        Box::new(docker::DockerRuntime::connect(port_start, port_end).await?);

    #[cfg(target_os = "macos")]
    let runtime: Box<dyn HostRuntime> = Box::new(mac::MlxRuntime::connect(port_start, port_end)?);

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let runtime: Box<dyn HostRuntime> = {
        let _ = (port_start, port_end);
        anyhow::bail!("no host runtime is available for this platform");
    };

    Ok(runtime)
}

// --- shared pure helpers (used by the Docker backend; tested on any OS) -----

/// First free port in `start..=end` not present in `used`.
#[cfg(any(target_os = "linux", target_os = "macos", test))]
pub(crate) fn pick_free_port(
    start: i32,
    end: i32,
    used: &std::collections::BTreeSet<i32>,
) -> Option<i32> {
    (start..=end).find(|p| !used.contains(p))
}

#[cfg(any(target_os = "linux", test))]
fn container_name(rental_id: i64) -> String {
    format!("vgpu-rental-{rental_id}")
}

/// Split `repo[:tag]` into `(repo, tag)`, defaulting `tag` to `latest`. A colon
/// only counts as a tag separator after the last `/`, so a registry host-port
/// (`registry:5000/img`) is not mistaken for a tag.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn split_image_tag(image: &str) -> (String, String) {
    let last_segment_start = image.rfind('/').map(|i| i + 1).unwrap_or(0);
    match image[last_segment_start..].rfind(':') {
        Some(rel) => {
            let abs = last_segment_start + rel;
            (image[..abs].to_string(), image[abs + 1..].to_string())
        }
        None => (image.to_string(), "latest".to_string()),
    }
}

// --- Linux: Docker + NVIDIA Container Toolkit ------------------------------

#[cfg(target_os = "linux")]
mod docker {
    use std::collections::{BTreeSet, HashMap};

    use anyhow::Context;
    use async_trait::async_trait;
    use bollard::container::{
        Config, CreateContainerOptions, ListContainersOptions, RemoveContainerOptions,
        StartContainerOptions, StopContainerOptions,
    };
    use bollard::image::CreateImageOptions;
    use bollard::models::{DeviceRequest, HostConfig, PortBinding};
    use bollard::Docker;
    use futures_util::StreamExt;

    use super::{container_name, pick_free_port, split_image_tag};
    use super::{HostRuntime, RentalKind, RentalSpec, RunningRental, Started};

    /// Labels we stamp on our containers so the agent can find and account for
    /// them across its own restarts — the container list is the source of truth
    /// for what is running (reconciliation, roadmap item 3).
    const LABEL_RENTAL_ID: &str = "io.vgpu.rental_id";
    const LABEL_SSH_PORT: &str = "io.vgpu.ssh_port";
    const CONTAINER_SSH_PORT: &str = "22/tcp";

    pub struct DockerRuntime {
        docker: Docker,
        port_start: i32,
        port_end: i32,
    }

    impl DockerRuntime {
        pub async fn connect(port_start: i32, port_end: i32) -> anyhow::Result<Self> {
            let docker =
                Docker::connect_with_local_defaults().context("connecting to the Docker daemon")?;
            docker
                .ping()
                .await
                .context("Docker daemon did not respond to ping — is it running?")?;
            Ok(Self {
                docker,
                port_start,
                port_end,
            })
        }

        async fn used_ports(&self) -> anyhow::Result<BTreeSet<i32>> {
            Ok(self
                .running_rentals()
                .await?
                .into_iter()
                .map(|r| r.ssh_port)
                .collect())
        }

        async fn pull_image(&self, image: &str) -> anyhow::Result<()> {
            let (from_image, tag) = split_image_tag(image);
            let mut stream = self.docker.create_image(
                Some(CreateImageOptions {
                    from_image,
                    tag,
                    ..Default::default()
                }),
                None,
                None,
            );
            while let Some(item) = stream.next().await {
                item.context("image pull stream")?;
            }
            Ok(())
        }

        async fn force_remove(&self, name: &str) -> anyhow::Result<()> {
            match self
                .docker
                .remove_container(
                    name,
                    Some(RemoveContainerOptions {
                        force: true,
                        v: true,
                        ..Default::default()
                    }),
                )
                .await
            {
                Ok(()) => Ok(()),
                Err(e) if is_not_found(&e) => Ok(()),
                Err(e) => Err(e).context("removing container"),
            }
        }
    }

    #[async_trait]
    impl HostRuntime for DockerRuntime {
        /// Pull the image, allocate a free SSH port, launch with all GPUs passed
        /// through. The tenant key is injected via `PUBLIC_KEY` (a common
        /// ML-image convention) — a POC shortcut a microVM would replace.
        async fn start_rental(&self, spec: RentalSpec) -> anyhow::Result<Started> {
            self.pull_image(&spec.image)
                .await
                .with_context(|| format!("pulling image {}", spec.image))?;

            let used = self.used_ports().await?;
            let ssh_port =
                pick_free_port(self.port_start, self.port_end, &used).ok_or_else(|| {
                    anyhow::anyhow!(
                        "no free SSH port in {}..={}",
                        self.port_start,
                        self.port_end
                    )
                })?;

            let name = container_name(spec.rental_id);
            self.force_remove(&name).await?; // clear any stale container

            let mut labels = HashMap::new();
            labels.insert(LABEL_RENTAL_ID.to_string(), spec.rental_id.to_string());
            labels.insert(LABEL_SSH_PORT.to_string(), ssh_port.to_string());

            let mut exposed_ports = HashMap::new();
            exposed_ports.insert(CONTAINER_SSH_PORT.to_string(), HashMap::new());

            let mut port_bindings = HashMap::new();
            port_bindings.insert(
                CONTAINER_SSH_PORT.to_string(),
                Some(vec![PortBinding {
                    host_ip: Some("0.0.0.0".to_string()),
                    host_port: Some(ssh_port.to_string()),
                }]),
            );

            let host_config = HostConfig {
                port_bindings: Some(port_bindings),
                // Equivalent to `docker run --gpus all`.
                device_requests: Some(vec![DeviceRequest {
                    driver: Some(String::new()),
                    count: Some(-1),
                    device_ids: None,
                    capabilities: Some(vec![vec!["gpu".to_string()]]),
                    options: None,
                }]),
                ..Default::default()
            };

            let env = vec![
                format!("PUBLIC_KEY={}", spec.ssh_pubkey),
                format!("SSH_PUBLIC_KEY={}", spec.ssh_pubkey),
            ];

            let config = Config::<String> {
                image: Some(spec.image.clone()),
                env: Some(env),
                exposed_ports: Some(exposed_ports),
                labels: Some(labels),
                host_config: Some(host_config),
                ..Default::default()
            };

            let created = self
                .docker
                .create_container(
                    Some(CreateContainerOptions {
                        name: name.clone(),
                        ..Default::default()
                    }),
                    config,
                )
                .await
                .context("creating container")?;

            self.docker
                .start_container(&name, None::<StartContainerOptions<String>>)
                .await
                .context("starting container")?;

            Ok(Started {
                container_id: created.id,
                ssh_port,
                kind: RentalKind::Ssh,
            })
        }

        async fn stop_rental(&self, rental_id: i64) -> anyhow::Result<()> {
            let name = container_name(rental_id);
            match self
                .docker
                .stop_container(&name, Some(StopContainerOptions { t: 10 }))
                .await
            {
                Ok(()) => {}
                Err(e) if is_not_found(&e) || is_not_modified(&e) => {}
                Err(e) => return Err(e).context("stopping container"),
            }
            self.force_remove(&name).await
        }

        async fn running_rentals(&self) -> anyhow::Result<Vec<RunningRental>> {
            let mut filters = HashMap::new();
            filters.insert("label".to_string(), vec![LABEL_RENTAL_ID.to_string()]);

            let containers = self
                .docker
                .list_containers(Some(ListContainersOptions {
                    all: false,
                    filters,
                    ..Default::default()
                }))
                .await
                .context("listing containers")?;

            let mut out = Vec::new();
            for c in containers {
                let labels = c.labels.unwrap_or_default();
                let (Some(rid), Some(port)) =
                    (labels.get(LABEL_RENTAL_ID), labels.get(LABEL_SSH_PORT))
                else {
                    continue;
                };
                let (Ok(rental_id), Ok(ssh_port)) = (rid.parse::<i64>(), port.parse::<i32>())
                else {
                    tracing::warn!(
                        ?rid,
                        ?port,
                        "container has unparseable vgpu labels; skipping"
                    );
                    continue;
                };
                out.push(RunningRental {
                    rental_id,
                    container_id: c.id.unwrap_or_default(),
                    ssh_port,
                });
            }
            Ok(out)
        }

        async fn available(&self) -> anyhow::Result<Vec<i32>> {
            let used = self.used_ports().await?;
            Ok((self.port_start..=self.port_end)
                .filter(|p| !used.contains(p))
                .collect())
        }
    }

    fn is_not_found(e: &bollard::errors::Error) -> bool {
        matches!(
            e,
            bollard::errors::Error::DockerResponseServerError {
                status_code: 404,
                ..
            }
        )
    }

    fn is_not_modified(e: &bollard::errors::Error) -> bool {
        matches!(
            e,
            bollard::errors::Error::DockerResponseServerError {
                status_code: 304,
                ..
            }
        )
    }
}

// --- macOS: MLX-only runtime (host-controlled MLX jobs) --------------------

#[cfg(target_os = "macos")]
mod mac {
    use std::collections::{BTreeSet, HashMap};
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Mutex;

    use anyhow::Context;
    use async_trait::async_trait;

    use super::{pick_free_port, HostRuntime, RentalKind, RentalSpec, RunningRental, Started};

    /// An allowlisted, host-controlled MLX task. The tenant supplies only
    /// *parameters* (via the rental image), never code — that bounded surface is
    /// the isolation model.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum MlxTask {
        /// fp16 GEMM loop — a GPU-time benchmark workload. `mlx:gemm[:N]`.
        Gemm { n: u32 },
        /// `mlx-lm` text generation, served OpenAI-compatibly on the port.
        /// `mlx:generate:<hf-model-id>` (e.g. mlx-community/Llama-3.2-1B-Instruct-4bit).
        Generate { model: String },
    }

    /// A Hugging Face model id is safe to pass as a bare argument: `org/name`,
    /// conservative charset, no path traversal. (We pass it as an argv, never
    /// through a shell.) `mlx-community/*` repos are MLX-format weights + config —
    /// data, not code.
    fn valid_model_id(s: &str) -> bool {
        !s.is_empty()
            && !s.contains("..")
            && s.split('/').count() == 2
            && s.split('/').all(|seg| !seg.is_empty())
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
    }

    /// Parse a rental image into an allowlisted task. Rejects anything that is
    /// not `mlx:...` — an MLX-only host does not run arbitrary images.
    fn parse_task(image: &str) -> anyhow::Result<MlxTask> {
        let rest = image.strip_prefix("mlx:").ok_or_else(|| {
            anyhow::anyhow!(
                "this host runs MLX jobs only — image must be 'mlx:gemm[:N]' or \
                 'mlx:generate:<model>' (got {image:?})"
            )
        })?;
        let (task, arg) = match rest.split_once(':') {
            Some((t, a)) => (t, Some(a)),
            None => (rest, None),
        };
        match task {
            "gemm" | "selftest" => {
                let n = arg
                    .map(str::parse::<u32>)
                    .transpose()
                    .map_err(|_| anyhow::anyhow!("invalid GEMM size in {image:?}"))?
                    .unwrap_or(2048);
                anyhow::ensure!((256..=16384).contains(&n), "GEMM size out of range: {n}");
                Ok(MlxTask::Gemm { n })
            }
            "generate" => {
                let model = arg.ok_or_else(|| {
                    anyhow::anyhow!("missing model: use 'mlx:generate:<hf-model-id>'")
                })?;
                anyhow::ensure!(
                    valid_model_id(model),
                    "invalid model id {model:?}; expected 'org/name' (e.g. mlx-community/…)"
                );
                Ok(MlxTask::Generate {
                    model: model.to_string(),
                })
            }
            other => anyhow::bail!("unknown MLX task {other:?}; supported: gemm, generate"),
        }
    }

    struct Job {
        child: tokio::process::Child,
        port: i32,
    }

    /// Runs host-controlled MLX jobs as child processes of a configured Python
    /// (`VGPU_MLX_PYTHON`, default `python3`). Only host-controlled code ever
    /// runs: the bundled GEMM worker, or `mlx_lm.server` for generation. `gemm`
    /// needs `mlx`; `generate` needs `mlx-lm`.
    pub struct MlxRuntime {
        python: String,
        script_path: PathBuf,
        mlx_available: bool,
        mlx_lm_available: bool,
        port_start: i32,
        port_end: i32,
        jobs: Mutex<HashMap<i64, Job>>,
    }

    /// The host-controlled MLX worker: an fp16 GEMM loop on the Apple GPU plus a
    /// tiny HTTP status endpoint the tenant can poll for live throughput. Never
    /// runs tenant-supplied code.
    const MLX_WORKER: &str = r#"
import sys, time, threading, json
from http.server import BaseHTTPRequestHandler, HTTPServer
import mlx.core as mx

n = int(sys.argv[1]); port = int(sys.argv[2])
max_secs = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
state = {"task": "gemm-fp16", "n": n, "iters": 0, "tflops": 0.0}

def worker():
    a = mx.random.normal((n, n)).astype(mx.float16)
    b = mx.random.normal((n, n)).astype(mx.float16)
    start = time.time()
    while time.time() - start < max_secs:
        c = a @ b
        mx.eval(c)
        state["iters"] += 1
        el = time.time() - start
        if el > 0:
            state["tflops"] = 2 * n**3 * state["iters"] / el / 1e12

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(state).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

threading.Thread(target=worker, daemon=True).start()
print("[mlx-job] serving status on :%d (task=gemm-fp16 n=%d)" % (port, n), flush=True)
HTTPServer(("0.0.0.0", port), H).serve_forever()
"#;

    impl MlxRuntime {
        pub fn connect(port_start: i32, port_end: i32) -> anyhow::Result<Self> {
            let python = std::env::var("VGPU_MLX_PYTHON").unwrap_or_else(|_| "python3".to_string());

            // Write the bundled worker to a temp file — the only code we run.
            let script_path = std::env::temp_dir().join("vgpu-mlx-worker.py");
            std::fs::File::create(&script_path)
                .with_context(|| format!("writing MLX worker to {}", script_path.display()))?
                .write_all(MLX_WORKER.as_bytes())?;

            // Probe capabilities so rentals fail early — the host still registers.
            let mlx_available = std::process::Command::new(&python)
                .args(["-c", "import mlx.core"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            let mlx_lm_available = std::process::Command::new(&python)
                .args(["-c", "import mlx_lm"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if !mlx_available {
                tracing::warn!(
                    python = %python,
                    "MLX unavailable (import mlx.core failed); host registers but `mlx:gemm` \
                     rentals fail until `pip install mlx` in that Python"
                );
            }
            if !mlx_lm_available {
                tracing::warn!(
                    python = %python,
                    "mlx-lm unavailable; `mlx:generate` rentals fail until `pip install mlx-lm`"
                );
            }

            Ok(Self {
                python,
                script_path,
                mlx_available,
                mlx_lm_available,
                port_start,
                port_end,
                jobs: Mutex::new(HashMap::new()),
            })
        }

        fn used_ports(&self) -> BTreeSet<i32> {
            self.jobs
                .lock()
                .expect("mlx jobs lock")
                .values()
                .map(|j| j.port)
                .collect()
        }

        /// Build the child command for a task. Only host-controlled programs are
        /// spawned; the tenant's input reaches them only as validated arguments.
        fn build_command(&self, task: &MlxTask, port: i32) -> tokio::process::Command {
            let mut cmd = tokio::process::Command::new(&self.python);
            match task {
                MlxTask::Gemm { n } => {
                    cmd.arg(&self.script_path)
                        .arg(n.to_string())
                        .arg(port.to_string());
                }
                MlxTask::Generate { model } => {
                    // mlx-lm's OpenAI-compatible server; model is a validated id.
                    cmd.arg("-m")
                        .arg("mlx_lm.server")
                        .arg("--model")
                        .arg(model)
                        .arg("--host")
                        .arg("0.0.0.0")
                        .arg("--port")
                        .arg(port.to_string());
                }
            }
            cmd
        }
    }

    #[async_trait]
    impl HostRuntime for MlxRuntime {
        async fn start_rental(&self, spec: RentalSpec) -> anyhow::Result<Started> {
            let task = parse_task(&spec.image)?;
            match &task {
                MlxTask::Gemm { .. } => anyhow::ensure!(
                    self.mlx_available,
                    "MLX unavailable: `import mlx.core` failed for {} (pip install mlx)",
                    self.python
                ),
                MlxTask::Generate { .. } => anyhow::ensure!(
                    self.mlx_lm_available,
                    "mlx-lm unavailable: `import mlx_lm` failed for {} (pip install mlx-lm)",
                    self.python
                ),
            }

            let kind = match &task {
                MlxTask::Gemm { .. } => RentalKind::HttpStatus,
                MlxTask::Generate { .. } => RentalKind::HttpOpenai,
            };

            let used = self.used_ports();
            let port = pick_free_port(self.port_start, self.port_end, &used).ok_or_else(|| {
                anyhow::anyhow!("no free port in {}..={}", self.port_start, self.port_end)
            })?;

            // Spawn is synchronous. gemm runs the bundled worker; generate runs
            // mlx-lm's OpenAI-compatible server on the port.
            let child = self
                .build_command(&task, port)
                .spawn()
                .with_context(|| format!("spawning MLX job via {}", self.python))?;

            self.jobs
                .lock()
                .expect("mlx jobs lock")
                .insert(spec.rental_id, Job { child, port });

            Ok(Started {
                container_id: format!("mlx-job-{}", spec.rental_id),
                ssh_port: port, // gemm: status endpoint; generate: OpenAI API
                kind,
            })
        }

        async fn stop_rental(&self, rental_id: i64) -> anyhow::Result<()> {
            let job = self.jobs.lock().expect("mlx jobs lock").remove(&rental_id);
            if let Some(mut job) = job {
                let _ = job.child.kill().await;
                let _ = job.child.wait().await;
            }
            Ok(())
        }

        async fn running_rentals(&self) -> anyhow::Result<Vec<RunningRental>> {
            let mut jobs = self.jobs.lock().expect("mlx jobs lock");
            let mut out = Vec::new();
            let mut dead = Vec::new();
            for (rental_id, job) in jobs.iter_mut() {
                if let Ok(Some(_)) = job.child.try_wait() {
                    dead.push(*rental_id); // process exited
                } else {
                    out.push(RunningRental {
                        rental_id: *rental_id,
                        container_id: format!("mlx-job-{rental_id}"),
                        ssh_port: job.port,
                    });
                }
            }
            for id in dead {
                jobs.remove(&id);
            }
            Ok(out)
        }

        async fn available(&self) -> anyhow::Result<Vec<i32>> {
            let used = self.used_ports();
            Ok((self.port_start..=self.port_end)
                .filter(|p| !used.contains(p))
                .collect())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rejects_non_mlx_images() {
            assert!(parse_task("nvidia/cuda:latest").is_err());
        }

        #[test]
        fn parses_gemm_default_and_explicit_size() {
            assert_eq!(parse_task("mlx:gemm").unwrap(), MlxTask::Gemm { n: 2048 });
            assert_eq!(
                parse_task("mlx:gemm:4096").unwrap(),
                MlxTask::Gemm { n: 4096 }
            );
        }

        #[test]
        fn rejects_out_of_range_size() {
            assert!(parse_task("mlx:gemm:1").is_err());
        }

        #[test]
        fn parses_generate_model() {
            assert_eq!(
                parse_task("mlx:generate:mlx-community/Llama-3.2-1B-Instruct-4bit").unwrap(),
                MlxTask::Generate {
                    model: "mlx-community/Llama-3.2-1B-Instruct-4bit".to_string()
                }
            );
        }

        #[test]
        fn generate_requires_a_valid_model() {
            assert!(parse_task("mlx:generate").is_err()); // no model
            assert!(parse_task("mlx:generate:not-a-repo").is_err()); // no '/'
            assert!(parse_task("mlx:generate:../etc/passwd").is_err()); // traversal
            assert!(parse_task("mlx:generate:a/b/c").is_err()); // too many segments
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn picks_lowest_free_port() {
        let used = BTreeSet::from([40000, 40001, 40003]);
        assert_eq!(pick_free_port(40000, 40010, &used), Some(40002));
    }

    #[test]
    fn exhausted_range_returns_none() {
        let used = BTreeSet::from([40000, 40001]);
        assert_eq!(pick_free_port(40000, 40001, &used), None);
    }

    #[test]
    fn image_tag_defaults_to_latest() {
        assert_eq!(
            split_image_tag("nvidia/cuda"),
            ("nvidia/cuda".to_string(), "latest".to_string())
        );
    }

    #[test]
    fn image_tag_is_parsed() {
        assert_eq!(
            split_image_tag("pytorch/pytorch:2.3.0-cuda12.1"),
            ("pytorch/pytorch".to_string(), "2.3.0-cuda12.1".to_string())
        );
    }

    #[test]
    fn registry_port_is_not_mistaken_for_tag() {
        assert_eq!(
            split_image_tag("registry.local:5000/team/img"),
            (
                "registry.local:5000/team/img".to_string(),
                "latest".to_string()
            )
        );
    }

    #[test]
    fn container_name_is_stable() {
        assert_eq!(container_name(42), "vgpu-rental-42");
    }
}
