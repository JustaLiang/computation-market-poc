# CLAUDE.md

Rust implementation of a GPU rental marketplace with Lightning payments.

**Read `docs/SPEC.md` before writing code** — it is the normative behaviour spec
(data model, API, algorithms, acceptance test). `docs/BACKGROUND.md` explains why
the design is what it is; consult it before proposing architectural changes.

Nothing is implemented yet. This file describes the target.

## Stack

- Edition 2021, Rust 1.75+
- `tokio` (rt-multi-thread, macros) — async runtime
- `axum` — HTTP, control plane
- `sqlx` (sqlite, runtime-tokio) — storage, compile-time checked queries
- `serde` / `serde_json`
- `reqwest` (json, rustls-tls) — agent → control plane, LND REST
- `clap` (derive) — CLI
- `bollard` — Docker Engine API, host agent
- `nvml-wrapper` — GPU inventory. Use this, not `nvidia-smi` parsing; NVML gives
  name, VRAM, UUID and PCI bus id directly.
- `tracing` + `tracing-subscriber`
- `thiserror` in libraries, `anyhow` at binary boundaries
- `comfy-table` — CLI output

Do **not** add: an ORM, a message broker, a task queue, a web framework other
than axum, or any blockchain dependency beyond Lightning. Nothing goes on-chain
except payments.

## Workspace layout

```
Cargo.toml                 workspace
crates/
  core/                    shared types. no I/O, no axum, no sqlx queries.
    src/money.rs           Sats newtype
    src/model.rs           Machine, Account, Rental, RentalStatus, LedgerKind
    src/api.rs             request/response DTOs shared by server and clients
  control-plane/           binary: axum server
    src/main.rs            wiring, config, router
    src/routes/agent.rs    /agent/*
    src/routes/tenant.rs   /offers, /accounts, /rentals
    src/billing.rs         ticker: invoices, metering, eviction, liveness
    src/lightning/mod.rs   LightningBackend trait
    src/lightning/mock.rs
    src/lightning/lnd.rs
    src/db.rs              pool, migrations, ledger writes
    migrations/
  host-agent/              binary
    src/main.rs            register, heartbeat, dispatch
    src/runtime.rs         ALL container code. see below.
    src/benchmark.rs       NVML inventory, fp16 GEMM score, fingerprint
  vgpu/                    binary: tenant CLI
tests/
  lifecycle.rs             acceptance test from SPEC §10
```

`crates/core` must stay dependency-light so the CLI doesn't pull in sqlx.

## Rust-specific requirements

These express SPEC invariants in ways the compiler can help enforce. Take
advantage of that — it's the main reason to write this in Rust.

**`Sats(i64)` newtype, in `core::money`.** SPEC requires integer satoshis
everywhere. Make it unrepresentable to get wrong:

- Derive `Add`, `Sub`, `AddAssign`, `SubAssign`, `Ord`, `Serialize`,
  `Deserialize`, `sqlx::Type`.
- Do **not** implement `Mul<Sats>` or any `From<f64>`. Multiplying money by money
  is meaningless; float conversion is how the invariant dies.
- `Mul<i64>` is fine (rate × minutes).
- Display as a bare integer. Any BTC-denominated formatting lives in `vgpu` only.

**`RentalStatus` as an enum**, `#[derive(sqlx::Type)]` with
`#[sqlx(rename_all = "lowercase")]`: `Provisioning`, `Running`, `Evicting`,
`Stopped`, `Failed`. Add `fn occupies_machine(&self) -> bool` returning true for
the first three, and use it in the offer query. Do not compare status strings.

**`BillingConfig` in app state, not a `const`.** SPEC requires `BILL_PERIOD` to
have exactly one definition, and the acceptance test needs it compressed to 2s.
A struct on `AppState` satisfies both:

```rust
pub struct BillingConfig {
    pub bill_period: Duration,      // default 60s
    pub tick: Duration,             // default 5s
    pub heartbeat_timeout: Duration // default 90s
}
```

Both the ticker and rental creation read it from state. Never hardcode 60.

**Rental creation needs a real write lock.** `sqlx`'s `begin()` issues a deferred
`BEGIN` on SQLite, which can fail with `SQLITE_BUSY` on upgrade and would allow
two tenants to race for one machine. Execute `BEGIN IMMEDIATE` explicitly, or
move to Postgres and use `SELECT ... FOR UPDATE`. Do not rely on a plain
transaction plus an optimistic re-check.

**`ssh_pubkey` must never leave the API.** Put it on the DB row type but not on
the `RentalResponse` DTO, so omitting it isn't something you have to remember.

**Ledger writes go through one function.** `db::record(&mut tx, kind, delta, ...)`.
Never write to `ledger` from a route or the ticker directly. Charges write two
rows (`rental_charge` debit + `host_credit` credit) inside the same transaction.

**All container/isolation code stays in `host-agent/src/runtime.rs`.** Keep the
public surface to roughly `start_rental`, `stop_rental`, `running_rentals`,
`available`. Docker is not a security boundary against a tenant with GPU device
access; this module gets replaced by a Firecracker/VFIO launcher. No `bollard`
types in `main.rs`.

**Timestamps are `i64` Unix seconds** in the database. Convert at the boundary.

## Lightning

```rust
#[async_trait]
pub trait LightningBackend: Send + Sync {
    async fn create_invoice(&self, sats: Sats, memo: &str) -> Result<Invoice>;
    async fn is_settled(&self, payment_hash: &str) -> Result<bool>;
    async fn pay_invoice(&self, bolt11: &str) -> Result<()>;
}
```

Three methods, no more. `MockBackend` auto-settles after a configurable delay so
the whole system runs with no node — this is what makes the acceptance test
hardware-free. `LndRest` talks to lnd's REST API over `reqwest` with the macaroon
hex in `Grpc-Metadata-macaroon`; note lnd returns `r_hash` base64-encoded while
we store hex everywhere.

Selected by `LN_BACKEND=mock|lnd`. A gRPC client crate is an option instead of
REST, but check its current maintenance status before adopting it — REST plus
`reqwest` has no extra dependency and is sufficient.

`pay_invoice` failure must leave `payout_balance` intact. Return 502, don't zero
the balance.

## Commands

```bash
cargo build --workspace
cargo test --workspace           # includes tests/lifecycle.rs
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

Run the acceptance test after any change to billing, accounting, or the rental
state machine. Prefer driving the router with `tower::ServiceExt::oneshot` over
binding a real port — faster and no flakiness.

## Ordering

Build in this order; each step is independently testable.

1. `core`: `Sats`, `RentalStatus`, model structs, API DTOs.
2. `control-plane`: migrations, pool, `db::record`.
3. `lightning`: trait + `MockBackend`.
4. Tenant routes: accounts, deposit, offers.
5. Agent routes: register, heartbeat + command queue, report.
6. `billing`: ticker. Now `tests/lifecycle.rs` can pass end to end.
7. `vgpu` CLI.
8. `host-agent`: NVML benchmark, then `runtime.rs`, then the heartbeat loop.
9. `LndRest`.

Steps 1–7 need no GPU and no container runtime. Get the acceptance test green
before touching `host-agent`.

## Deliberate gaps

Documented in SPEC §8. Custodial balances, POC-grade auth, Docker instead of a
microVM, host can read tenant memory, single benchmark at registration. **Do not
fix these opportunistically** — they are scoped out and they interact. If a task
requires crossing one of those lines, say so before starting.

Roadmap and priority order are in SPEC §9. The first item — randomized
re-benchmarking against each machine's own history — is the highest-value
addition, not the isolation work.
