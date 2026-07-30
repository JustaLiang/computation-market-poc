//! Container lifecycle — **all** Docker/`bollard` code lives here.
//!
//! Docker is *not* a security boundary against a tenant holding GPU device
//! access; an escape via the NVIDIA container runtime owns the host (SPEC §8,
//! BACKGROUND §4.3). This module is deliberately narrow so it can be replaced
//! wholesale by a Firecracker/Cloud-Hypervisor + VFIO launcher without touching
//! the agent's control logic. Public surface: [`Runtime::start_rental`],
//! [`Runtime::stop_rental`], [`Runtime::running_rentals`], [`Runtime::available`].
//!
//! No `bollard` type appears in the signatures below or anywhere outside this
//! file (CLAUDE.md).

use std::collections::{BTreeSet, HashMap};

use anyhow::Context;
use bollard::container::{
    Config, CreateContainerOptions, ListContainersOptions, RemoveContainerOptions,
    StartContainerOptions, StopContainerOptions,
};
use bollard::image::CreateImageOptions;
use bollard::models::{DeviceRequest, HostConfig, PortBinding};
use bollard::Docker;
use futures_util::StreamExt;

/// Labels we stamp on our containers so the agent can find and account for them
/// across its own restarts — the container list is the source of truth for what
/// is running, which is what a future reconciliation pass (roadmap item 3) needs.
const LABEL_RENTAL_ID: &str = "io.vgpu.rental_id";
const LABEL_SSH_PORT: &str = "io.vgpu.ssh_port";
/// SSH inside the container. Host ports from the machine's range map onto this.
const CONTAINER_SSH_PORT: &str = "22/tcp";

/// What to launch. A control-plane [`Command::StartRental`] maps onto this.
///
/// [`Command::StartRental`]: vgpu_core::api::Command::StartRental
pub struct RentalSpec {
    pub rental_id: i64,
    pub image: String,
    /// Injected as `authorized_keys` (see [`Runtime::start_rental`]).
    pub ssh_pubkey: String,
}

/// Result of a successful launch, reported back to the control plane.
pub struct Started {
    pub container_id: String,
    /// Host-side port, reachable at the machine's `public_ip`, mapped to SSH.
    pub ssh_port: i32,
}

/// A container this agent currently has running, recovered from Docker labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunningRental {
    pub rental_id: i64,
    pub container_id: String,
    pub ssh_port: i32,
}

/// Owns the Docker connection and the host's assignable port range.
pub struct Runtime {
    docker: Docker,
    port_start: i32,
    port_end: i32,
}

impl Runtime {
    /// Connect to the local Docker daemon and verify it answers.
    pub async fn connect(port_start: i32, port_end: i32) -> anyhow::Result<Self> {
        anyhow::ensure!(
            port_start <= port_end,
            "port_start ({port_start}) must be <= port_end ({port_end})"
        );
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

    /// Pull the image, allocate a free SSH port, and launch the container with
    /// all GPUs passed through.
    ///
    /// The tenant's public key is injected via the `PUBLIC_KEY` env var, the
    /// vast.ai/runpod convention that ML base images read to seed
    /// `authorized_keys` at boot. That is a POC shortcut — a microVM launcher
    /// would provision the key directly.
    pub async fn start_rental(&self, spec: RentalSpec) -> anyhow::Result<Started> {
        self.pull_image(&spec.image)
            .await
            .with_context(|| format!("pulling image {}", spec.image))?;

        let used = self.used_ports().await?;
        let ssh_port = pick_free_port(self.port_start, self.port_end, &used).ok_or_else(|| {
            anyhow::anyhow!(
                "no free SSH port in {}..={}",
                self.port_start,
                self.port_end
            )
        })?;

        let name = container_name(spec.rental_id);
        // Idempotency: clear any stale container squatting on this name.
        self.force_remove(&name).await?;

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
            // Equivalent to `docker run --gpus all`: expose every GPU through the
            // NVIDIA container runtime (count -1 = all, capability "gpu").
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

    /// Stop and remove the rental's container. Idempotent: a missing or
    /// already-stopped container is success, so a repeated `stop_rental` command
    /// (at-most-once delivery can still redeliver on our side) is safe.
    pub async fn stop_rental(&self, rental_id: i64) -> anyhow::Result<()> {
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

    /// Every rental container this agent currently has running, recovered from
    /// Docker labels. Survives an agent restart and underpins reconciliation.
    pub async fn running_rentals(&self) -> anyhow::Result<Vec<RunningRental>> {
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
            let (Some(rid), Some(port)) = (labels.get(LABEL_RENTAL_ID), labels.get(LABEL_SSH_PORT))
            else {
                continue;
            };
            let (Ok(rental_id), Ok(ssh_port)) = (rid.parse::<i64>(), port.parse::<i32>()) else {
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

    /// Host ports in the machine's range not currently bound to a rental.
    ///
    /// Part of the required runtime surface (CLAUDE.md); consumed by the
    /// reconciliation pass (roadmap item 3), not yet wired into the loop.
    #[allow(dead_code)]
    pub async fn available(&self) -> anyhow::Result<Vec<i32>> {
        let used = self.used_ports().await?;
        Ok((self.port_start..=self.port_end)
            .filter(|p| !used.contains(p))
            .collect())
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
        // Draining the stream is what surfaces a pull failure (auth, no such
        // image); each item is a progress event we otherwise ignore.
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

fn container_name(rental_id: i64) -> String {
    format!("vgpu-rental-{rental_id}")
}

/// First free port in `start..=end` not present in `used`.
fn pick_free_port(start: i32, end: i32, used: &BTreeSet<i32>) -> Option<i32> {
    (start..=end).find(|p| !used.contains(p))
}

/// Split `repo[:tag]` into `(repo, tag)`, defaulting `tag` to `latest`.
///
/// A colon only counts as a tag separator if it appears after the last `/`, so a
/// registry host-port like `registry:5000/img` is not mistaken for a tag.
fn split_image_tag(image: &str) -> (String, String) {
    let last_segment_start = image.rfind('/').map(|i| i + 1).unwrap_or(0);
    match image[last_segment_start..].rfind(':') {
        Some(rel) => {
            let abs = last_segment_start + rel;
            (image[..abs].to_string(), image[abs + 1..].to_string())
        }
        None => (image.to_string(), "latest".to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn single_port_range() {
        let used = BTreeSet::new();
        assert_eq!(pick_free_port(40000, 40000, &used), Some(40000));
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
            split_image_tag("pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"),
            (
                "pytorch/pytorch".to_string(),
                "2.3.0-cuda12.1-cudnn8-runtime".to_string()
            )
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
        assert_eq!(
            split_image_tag("registry.local:5000/team/img:v2"),
            ("registry.local:5000/team/img".to_string(), "v2".to_string())
        );
    }

    #[test]
    fn container_name_is_stable() {
        assert_eq!(container_name(42), "vgpu-rental-42");
    }
}
