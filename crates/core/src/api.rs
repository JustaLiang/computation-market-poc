//! Request/response DTOs shared by the server and its clients (SPEC §7).
//!
//! This build step implements the **agent-facing** surface (`/agent/*`), which
//! is what the host agent speaks. Tenant DTOs (`/offers`, `/accounts`,
//! `/rentals`) are added with the tenant-routes build step.

use serde::{Deserialize, Serialize};

use crate::model::{DiskType, RentalStatus};

/// Body of `POST /agent/register` (SPEC §7).
///
/// Carries every `machines` spec field the agent can know about itself —
/// i.e. all of them except the server-assigned `id`, `agent_token`,
/// `payout_balance`, `online`, `last_heartbeat`, and `created_at`.
///
/// Registration is idempotent on `host_id`: re-registration after a restart
/// refreshes specs and rate and returns the existing token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterRequest {
    /// Stable, agent-generated, survives restarts. The idempotency key.
    pub host_id: String,

    pub gpu_name: String,
    pub gpu_count: i32,
    /// Per GPU.
    pub vram_mb: i64,

    pub cpu_name: String,
    pub cpu_cores: i32,
    pub ram_mb: i64,
    pub disk_gb: i64,
    pub disk_type: DiskType,

    /// Where tenants connect. Workload traffic never traverses the control plane.
    pub public_ip: String,
    /// Reachable port range at `public_ip`, for SSH/workload port mapping.
    pub port_start: i32,
    pub port_end: i32,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inet_down_mbps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inet_up_mbps: Option<f64>,
    /// ISO-3166 alpha-2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,

    /// Scalar performance estimate (SPEC §6).
    pub dlperf: f64,
    /// Provider-set price, integer sats/min, ≥ 1.
    pub rate_sats_per_min: i64,
    /// Hash of the sorted `(gpu_name, gpu_uuid, pci_bus_id)` set (SPEC §6).
    pub hw_fingerprint: String,
}

/// Response to `POST /agent/register`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub machine_id: i64,
    /// Long-lived bearer token for subsequent `/agent/*` calls.
    pub agent_token: String,
}

/// Body of `POST /agent/heartbeat`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub online: bool,
}

/// Response to `POST /agent/heartbeat`: the queued commands, delivered at most
/// once. All control traffic is agent-initiated (SPEC §1), so this heartbeat
/// response is the *only* channel by which the control plane reaches a host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub commands: Vec<Command>,
}

/// A queued instruction delivered in a heartbeat response.
///
/// Wire form matches SPEC §7 exactly, e.g.
/// `{"cmd": "start_rental", "rental_id": 1, "image": "...", "ssh_pubkey": "..."}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    StartRental {
        rental_id: i64,
        image: String,
        /// Injected as `authorized_keys`. Never leaves the agent boundary.
        ssh_pubkey: String,
    },
    StopRental {
        rental_id: i64,
    },
}

/// Body of `POST /agent/report` (SPEC §7). The agent reports the outcome of a
/// command. The server verifies the rental belongs to the authenticated machine
/// and sets `ssh_host` from the machine's `public_ip`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportRequest {
    pub rental_id: i64,
    pub status: RentalStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_port: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Body of `POST /agent/rate`: update the machine's price. Live rentals are
/// unaffected because their rate was snapshotted at creation (BACKGROUND §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateRequest {
    pub rate_sats_per_min: i64,
}

/// Body of `POST /agent/payout`: the host supplies a BOLT11 invoice for the sats
/// it is owed. On success the control plane pays it and zeroes `payout_balance`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PayoutRequest {
    pub bolt11: String,
}

/// Response to `POST /agent/payout`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PayoutResponse {
    pub paid_sats: i64,
}

// ---------------------------------------------------------------------------
// Tenant API (SPEC §7): /offers, /accounts, /rentals, /health.
// ---------------------------------------------------------------------------

/// One row of `GET /offers` — an online, idle machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    pub machine_id: i64,
    pub gpu_name: String,
    pub gpu_count: i32,
    pub vram_mb: i64,
    pub cpu_name: String,
    pub cpu_cores: i32,
    pub ram_mb: i64,
    pub disk_gb: i64,
    pub disk_type: DiskType,
    pub public_ip: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inet_down_mbps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inet_up_mbps: Option<f64>,
    pub dlperf: f64,
    pub rate_sats_per_min: i64,
    /// Derived: `rate_sats_per_min * 60`.
    pub rate_sats_per_hour: i64,
}

/// Response to `GET /offers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OffersResponse {
    pub offers: Vec<Offer>,
}

/// Response to `POST /accounts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateAccountResponse {
    pub account_id: String,
}

/// Response to `GET /accounts/{id}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountResponse {
    pub account_id: String,
    pub balance_sats: i64,
    /// Sum of `rate_sats_per_min` over the account's running rentals.
    pub burn_sats_per_min: i64,
    /// `balance / burn`, or `null` when burn is zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runway_minutes: Option<i64>,
}

/// Body of `POST /accounts/{id}/deposit`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepositRequest {
    pub sats: i64,
}

/// Response to `POST /accounts/{id}/deposit`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepositResponse {
    pub bolt11: String,
    pub payment_hash: String,
    pub sats: i64,
}

/// Body of `POST /rentals`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateRentalRequest {
    pub machine_id: i64,
    pub account_id: String,
    pub image: String,
    pub ssh_pubkey: String,
}

/// Response to `POST /rentals`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateRentalResponse {
    pub rental_id: i64,
    pub status: RentalStatus,
    pub rate_sats_per_min: i64,
}

/// Response to `GET /rentals/{id}`.
///
/// **Never** carries `ssh_pubkey` — the field is simply absent from this type,
/// so omitting it is not something a handler has to remember (SPEC §7, CLAUDE.md).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RentalResponse {
    pub id: i64,
    pub machine_id: i64,
    pub account_id: String,
    pub image: String,
    pub status: RentalStatus,
    pub gpu_name: String,
    pub gpu_count: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_port: Option<i32>,
    /// Derived `ssh -p <port> root@<host>` when host and port are known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_command: Option<String>,
    pub rate_sats_per_min: i64,
    pub sats_charged: i64,
    pub minutes_billed: i64,
    pub paid_through: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<i64>,
}

/// Response to `GET /health`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub ln_backend: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_rental_matches_spec_wire_form() {
        let cmd = Command::StartRental {
            rental_id: 1,
            image: " nvidia/cuda:latest".trim().to_string(),
            ssh_pubkey: "ssh-ed25519 AAAA".to_string(),
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        assert_eq!(v["cmd"], "start_rental");
        assert_eq!(v["rental_id"], 1);
        assert_eq!(v["image"], "nvidia/cuda:latest");
        assert_eq!(v["ssh_pubkey"], "ssh-ed25519 AAAA");
    }

    #[test]
    fn stop_rental_round_trips() {
        let json = r#"{"cmd":"stop_rental","rental_id":7}"#;
        let cmd: Command = serde_json::from_str(json).unwrap();
        assert_eq!(cmd, Command::StopRental { rental_id: 7 });
    }

    #[test]
    fn report_omits_absent_optionals() {
        let r = ReportRequest {
            rental_id: 3,
            status: RentalStatus::Failed,
            ssh_port: None,
            container_id: None,
            error: Some("image pull failed".into()),
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(
            !s.contains("ssh_port"),
            "absent optionals must be omitted: {s}"
        );
        assert!(!s.contains("container_id"));
        assert!(s.contains("\"error\":\"image pull failed\""));
        assert!(s.contains("\"status\":\"failed\""));
    }
}
