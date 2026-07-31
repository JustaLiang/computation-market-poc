//! Domain enums shared across components.
//!
//! Full DB row structs (`Machine`, `Account`, `Rental`, `Invoice`) live with the
//! control plane's storage layer and are added in that build step; this module
//! carries the enums that appear in DTOs and must mean the same thing on every
//! side of the wire.

use serde::{Deserialize, Serialize};

/// Lifecycle of a rental (SPEC §4).
///
/// `provisioning`, `running`, and `evicting` are all **occupied** states — a
/// machine in any of them is excluded from the offer index. Treating only
/// `running` as occupied would allow double-renting during provisioning
/// (BACKGROUND §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "sqlx", derive(sqlx::Type))]
#[cfg_attr(feature = "sqlx", sqlx(rename_all = "lowercase"))]
#[serde(rename_all = "lowercase")]
pub enum RentalStatus {
    Provisioning,
    Running,
    Evicting,
    Stopped,
    Failed,
}

impl RentalStatus {
    /// True while the rental holds the machine off the offer index.
    ///
    /// Use this in the offer query rather than comparing status strings.
    pub fn occupies_machine(&self) -> bool {
        matches!(
            self,
            RentalStatus::Provisioning | RentalStatus::Running | RentalStatus::Evicting
        )
    }

    /// True once the rental has reached a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, RentalStatus::Stopped | RentalStatus::Failed)
    }

    /// Lowercase wire/storage form — matches the serde and sqlx representations.
    pub fn as_str(&self) -> &'static str {
        match self {
            RentalStatus::Provisioning => "provisioning",
            RentalStatus::Running => "running",
            RentalStatus::Evicting => "evicting",
            RentalStatus::Stopped => "stopped",
            RentalStatus::Failed => "failed",
        }
    }

    /// Every status, for building queries from [`Self::occupies_machine`] rather
    /// than hardcoding status strings (CLAUDE.md).
    pub const ALL: [RentalStatus; 5] = [
        RentalStatus::Provisioning,
        RentalStatus::Running,
        RentalStatus::Evicting,
        RentalStatus::Stopped,
        RentalStatus::Failed,
    ];
}

/// Storage medium backing a host's scratch disk (SPEC §3, `machines.disk_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "sqlx", derive(sqlx::Type))]
#[cfg_attr(feature = "sqlx", sqlx(rename_all = "lowercase"))]
#[serde(rename_all = "lowercase")]
pub enum DiskType {
    Nvme,
    Ssd,
    Hdd,
    #[default]
    Unknown,
}

/// How a tenant connects to a running rental — the agent reports it, and clients
/// render the right hint (SSH box vs HTTP endpoint). Defaults to `Ssh`, the
/// container tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "sqlx", derive(sqlx::Type))]
#[cfg_attr(feature = "sqlx", sqlx(rename_all = "snake_case"))]
#[serde(rename_all = "snake_case")]
pub enum RentalKind {
    /// An SSH box (the container tier): `ssh -p <port> <user>@<host>`.
    #[default]
    Ssh,
    /// A plain HTTP status/metrics endpoint: `GET http://<host>:<port>/`.
    HttpStatus,
    /// An OpenAI-compatible HTTP inference API at `/v1/...`.
    HttpOpenai,
}

/// Kind of a ledger entry (SPEC §3, `ledger.kind`). The append-only audit trail
/// is double-entry: `SUM(delta_sats) == 0` across the whole table, always.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "sqlx", derive(sqlx::Type))]
#[cfg_attr(feature = "sqlx", sqlx(rename_all = "snake_case"))]
#[serde(rename_all = "snake_case")]
pub enum LedgerKind {
    Deposit,
    RentalCharge,
    HostCredit,
    Payout,
    Evict,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupied_states_match_spec() {
        assert!(RentalStatus::Provisioning.occupies_machine());
        assert!(RentalStatus::Running.occupies_machine());
        assert!(RentalStatus::Evicting.occupies_machine());
        assert!(!RentalStatus::Stopped.occupies_machine());
        assert!(!RentalStatus::Failed.occupies_machine());
    }

    #[test]
    fn status_wire_format_is_lowercase() {
        assert_eq!(
            serde_json::to_string(&RentalStatus::Provisioning).unwrap(),
            "\"provisioning\""
        );
        assert_eq!(
            serde_json::from_str::<RentalStatus>("\"running\"").unwrap(),
            RentalStatus::Running
        );
    }

    #[test]
    fn ledger_kind_wire_format_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&LedgerKind::RentalCharge).unwrap(),
            "\"rental_charge\""
        );
        assert_eq!(
            serde_json::to_string(&LedgerKind::HostCredit).unwrap(),
            "\"host_credit\""
        );
    }
}
