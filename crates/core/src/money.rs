//! Integer satoshis.
//!
//! SPEC §2 and BACKGROUND §2 require integer satoshis *everywhere*: no floats,
//! no decimals, no BTC-denominated values in logic or storage. A system doing
//! ~1,440 debits per rental-day accumulates float error into an unreconcilable
//! ledger, so we make the mistake unrepresentable at the type level.
//!
//! Deliberately **not** implemented:
//! - `Mul<Sats>` — multiplying money by money is meaningless.
//! - `From<f64>` / any float conversion — the exact way the invariant dies.
//!
//! `Mul<i64>` *is* provided: `rate × minutes` is a legitimate operation.

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use serde::{Deserialize, Serialize};

/// A signed amount of satoshis.
///
/// Signed because the ledger records deltas (a `rental_charge` is negative).
/// Non-negativity is an invariant of specific balances (`accounts.balance_sats`,
/// `machines.payout_balance`), enforced where those live — not on the amount type.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[cfg_attr(feature = "sqlx", derive(sqlx::Type))]
#[cfg_attr(feature = "sqlx", sqlx(transparent))]
#[serde(transparent)]
pub struct Sats(pub i64);

impl Sats {
    pub const ZERO: Sats = Sats(0);

    #[inline]
    pub const fn new(value: i64) -> Self {
        Sats(value)
    }

    /// The raw satoshi count. Use only at a boundary (storage, display).
    #[inline]
    pub const fn get(self) -> i64 {
        self.0
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl Add for Sats {
    type Output = Sats;
    #[inline]
    fn add(self, rhs: Sats) -> Sats {
        Sats(self.0 + rhs.0)
    }
}

impl Sub for Sats {
    type Output = Sats;
    #[inline]
    fn sub(self, rhs: Sats) -> Sats {
        Sats(self.0 - rhs.0)
    }
}

impl AddAssign for Sats {
    #[inline]
    fn add_assign(&mut self, rhs: Sats) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Sats {
    #[inline]
    fn sub_assign(&mut self, rhs: Sats) {
        self.0 -= rhs.0;
    }
}

impl Neg for Sats {
    type Output = Sats;
    #[inline]
    fn neg(self) -> Sats {
        Sats(-self.0)
    }
}

/// `rate × minutes`. The only multiplication money admits.
impl Mul<i64> for Sats {
    type Output = Sats;
    #[inline]
    fn mul(self, rhs: i64) -> Sats {
        Sats(self.0 * rhs)
    }
}

impl Sum for Sats {
    fn sum<I: Iterator<Item = Sats>>(iter: I) -> Sats {
        Sats(iter.map(|s| s.0).sum())
    }
}

/// Displays as a bare integer. Any BTC-denominated formatting lives in `vgpu`.
impl fmt::Display for Sats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        assert_eq!(Sats(3) + Sats(4), Sats(7));
        assert_eq!(Sats(10) - Sats(4), Sats(6));
        assert_eq!(-Sats(5), Sats(-5));

        let mut b = Sats(100);
        b -= Sats(60);
        b += Sats(10);
        assert_eq!(b, Sats(50));
    }

    #[test]
    fn rate_times_minutes() {
        // 6 sats/min for 60 minutes = 360 sats/hr, the RTX 4090 anchor in SPEC §2.
        assert_eq!(Sats(6) * 60, Sats(360));
    }

    #[test]
    fn sum_reconciles_to_zero() {
        // The ledger invariant, in miniature: debits and credits cancel.
        let ledger = [Sats(-6), Sats(6), Sats(-6), Sats(6)];
        assert_eq!(ledger.into_iter().sum::<Sats>(), Sats::ZERO);
    }

    #[test]
    fn display_is_bare_integer() {
        assert_eq!(Sats(360).to_string(), "360");
        assert_eq!(Sats(-6).to_string(), "-6");
    }

    #[test]
    fn serde_is_transparent() {
        // Serializes as a bare number, not {"0": n} — clients and storage agree.
        assert_eq!(serde_json::to_string(&Sats(42)).unwrap(), "42");
        assert_eq!(serde_json::from_str::<Sats>("42").unwrap(), Sats(42));
    }
}
