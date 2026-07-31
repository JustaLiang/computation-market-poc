//! Workload runtime — the swappable, platform-specific launcher.
//!
//! The [`HostRuntime`] trait is the narrow surface the agent's control loop
//! depends on (`start_rental` / `stop_rental` / `running_rentals` / `available`).
//! It has two implementations, selected by target OS at compile time:
//!
//! - **Linux** (`docker`): tenant containers via `bollard` with `--gpus all`
//!   GPU passthrough. This is the real host tier (SPEC).
//! - **macOS** (`mac`): a stub that registers and benchmarks but refuses to
//!   launch — Apple Silicon has no container-GPU passthrough and no microVM
//!   isolation, so the "run a stranger's workload" story is undecided.
//!
//! Same `vgpu-agent` binary, same flags, same control-plane protocol; only the
//! guts differ. `main.rs` holds a `dyn HostRuntime` and never names a backend.
//! No `bollard` type appears outside the `docker` module (CLAUDE.md).

use async_trait::async_trait;

/// What to launch. A control-plane `start_rental` command maps onto this.
///
/// `image`/`ssh_pubkey` are consumed by the Docker backend (Linux); on macOS the
/// stub only needs `rental_id`, so the allow is scoped to non-Linux.
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
    /// Host-side port, reachable at the machine's `public_ip`, mapped to SSH.
    pub ssh_port: i32,
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
    let runtime: Box<dyn HostRuntime> = Box::new(mac::MacRuntime::new(port_start, port_end));

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let runtime: Box<dyn HostRuntime> = {
        let _ = (port_start, port_end);
        anyhow::bail!("no host runtime is available for this platform");
    };

    Ok(runtime)
}

// --- shared pure helpers (used by the Docker backend; tested on any OS) -----

/// First free port in `start..=end` not present in `used`.
#[cfg(any(target_os = "linux", test))]
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
    use super::{HostRuntime, RentalSpec, RunningRental, Started};

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
        /// through. The tenant key is injected via `PUBLIC_KEY` (the
        /// vast.ai/runpod convention) — a POC shortcut a microVM would replace.
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

// --- macOS: register + benchmark only (no isolation yet) -------------------

#[cfg(target_os = "macos")]
mod mac {
    use async_trait::async_trait;

    use super::{HostRuntime, RentalSpec, RunningRental, Started};

    /// Apple-Silicon runtime stub. The machine registers and benchmarks (so it
    /// appears in the offer index), but launching a tenant workload is refused:
    /// Docker on macOS can't hand a container the Apple GPU, and there is no
    /// microVM/VFIO isolation, so running a stranger's code is unresolved. The
    /// real launcher (a macOS VM, a restricted sandbox, or MLX-only jobs) slots
    /// in here without touching the agent's control loop.
    pub struct MacRuntime {
        port_start: i32,
        port_end: i32,
    }

    impl MacRuntime {
        pub fn new(port_start: i32, port_end: i32) -> Self {
            Self {
                port_start,
                port_end,
            }
        }
    }

    #[async_trait]
    impl HostRuntime for MacRuntime {
        async fn start_rental(&self, spec: RentalSpec) -> anyhow::Result<Started> {
            anyhow::bail!(
                "Apple-Silicon isolation is not implemented: this host registers and \
                 benchmarks, but cannot yet run tenant workload (rental {}). No container-GPU \
                 passthrough or microVM on macOS — see runtime::mac.",
                spec.rental_id
            )
        }

        async fn stop_rental(&self, _rental_id: i64) -> anyhow::Result<()> {
            Ok(()) // nothing was ever started
        }

        async fn running_rentals(&self) -> anyhow::Result<Vec<RunningRental>> {
            Ok(Vec::new())
        }

        async fn available(&self) -> anyhow::Result<Vec<i32>> {
            Ok((self.port_start..=self.port_end).collect())
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
