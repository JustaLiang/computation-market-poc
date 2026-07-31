# CLAUDE.md

Rust implementation of a GPU rental marketplace with Lightning payments.

**Read `docs/SPEC.md` before writing code** — it is the normative behaviour spec
(data model, API, algorithms, acceptance test). `docs/BACKGROUND.md` explains why
the design is what it is; consult it before proposing architectural changes.

**Status: implemented.** All crates build; `cargo test --workspace` is green,
including the SPEC §10 acceptance test (`crates/control-plane/tests/lifecycle.rs`,
which asserts `SUM(ledger.delta_sats) == 0`). The marketplace runs end to end on
the mock Lightning backend — see "Running" below. The only remaining gap is
`LndRest` (real Lightning, build step 9), which is a stub. This file still
describes the target; where the code deviates, it is noted under "Deviations".

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
Cargo.toml                 workspace (core, control-plane, host-agent, gpu-bench, vgpu)
crates/
  core/                    package `vgpu-core` (NOT `core`: a crate named `core`
                           shadows libcore and breaks proc-macros in binaries).
                           Imported as `vgpu_core`. No I/O; sqlx derives behind
                           the `sqlx` feature so clients stay dependency-light.
    src/money.rs           Sats newtype
    src/model.rs           RentalStatus, RentalKind, DiskType, LedgerKind (enums)
    src/api.rs             request/response DTOs (agent + tenant), shared
  control-plane/           binary + lib (lib so the acceptance test drives it)
    src/lib.rs             build_router()
    src/main.rs            wiring, config, ticker
    src/state.rs           AppState, BillingConfig, Clock (injectable)
    src/error.rs           ApiError -> HTTP
    src/routes/agent.rs    /agent/*
    src/routes/tenant.rs   /offers, /accounts, /rentals, /health
    src/billing.rs         ticker: invoices, metering, eviction, liveness
    src/lightning/{mod,mock,lnd}.rs   LightningBackend trait + backends
    src/db.rs              pool, migrations, row types, ledger writes
    migrations/0001_init.sql
    tests/lifecycle.rs     acceptance test from SPEC §10
  host-agent/              binary `vgpu-agent`
    src/main.rs            register, heartbeat, dispatch (holds dyn HostRuntime)
    src/runtime.rs         HostRuntime trait: Docker (Linux) + Mac backends
    src/benchmark.rs       NVML (Linux) / Metal (macOS) inventory, dlperf, fingerprint
  gpu-bench/               binary + lib: standalone vendor-agnostic GPU benchmark
                           (wgpu fp32/fp16 GEMM + bandwidth + network + index).
                           Isolated from the marketplace; the agent can consume
                           it to derive dlperf later.
  vgpu/                    binary: tenant CLI (BTC/USD formatting lives here)
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

**Status:** steps 1–8 are done and the acceptance test is green. Only step 9
(`LndRest`) remains — `LN_BACKEND=lnd` errors today; use `mock`.

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

## Running

Hardware-free, mock Lightning (no GPU, no Docker, no node):

```bash
# server
LN_BACKEND=mock control-plane            # binds 127.0.0.1:8080

# tenant (another shell) — vgpu talks to the tenant API
vgpu offers
vgpu create-account
vgpu deposit <acct> 6000                  # mock invoice settles on the next tick
vgpu rent --machine-id <id> --account-id <acct> --image <img> --ssh-key "<pubkey>"
vgpu rental <id>                          # shows the derived ssh command
```

On a Linux + NVIDIA host with Docker + the NVIDIA Container Toolkit, the real
host agent registers (becomes an offer), heartbeats, and runs containers:

```bash
vgpu-agent --control-plane http://<server>:8080 --public-ip <ip> --rate-sats-per-min 60
gpu-bench run                             # real measured GPU numbers on that host
```

## Deviations from this document

- **`core` is the package `vgpu-core`** (imported `vgpu_core`). A Cargo package
  literally named `core` shadows Rust's built-in `core` and breaks
  `#[tokio::main]`/`#[derive(...)]` in binaries. The directory stays `crates/core`.
- **Runtime-checked sqlx** (`query`/`query_as`), not the compile-time `query!`
  macro, so builds need no live `DATABASE_URL`. `migrate!` still embeds the schema.
- **Injectable `Clock`** on `AppState` (`System` | `Manual`) so the acceptance
  test advances time deterministically instead of sleeping.
- **`register` marks the machine online** with a fresh heartbeat, so a freshly
  registered host lists in `/offers` immediately (SPEC §10 step 2).
- **`gpu-bench` crate added** — a standalone GPU benchmark behind a `Backend`
  trait: `wgpu` (portable, default), `metal` (native Metal `simdgroup_matrix`,
  Apple matrix units, macOS-only, via `objc2-metal`), and cuBLAS
  (`--features cuda`, NVIDIA-host only, not compile-checked here). The fp16 GEMM
  in `host-agent/src/benchmark.rs` is still the SPEC §6 name-lookup fallback;
  real measurement lives in `gpu-bench`.
- **Host agent is two-platform** behind a `HostRuntime` trait chosen by target
  OS: Linux = Docker + NVML (the real host tier); macOS = Metal/`system_profiler`
  inventory + a **native Metal** runtime (`mac`) — Rust-only, **no Python, no
  `cmake`**. The host runs one allowlisted, host-controlled program: image
  `metal:gemm[:N]` = a continuous fp16 GEMM via a `simdgroup_matrix` MSL kernel
  (`crates/host-agent/src/mac_worker.rs`, launched by re-invoking the binary as
  `vgpu-agent __mac-worker <n> <port>`), exposing a live-throughput JSON status
  endpoint. Never tenant code — that bounded surface is the isolation model in
  lieu of a microVM. A non-`metal:` image is rejected. Same `vgpu-agent` binary,
  flags, and protocol — only `benchmark.rs`/`runtime.rs`/`mac_worker.rs` differ.
  Both the macOS host runtime and `gpu-bench`'s `metal` backend use `objc2-metal`
  (the older `metal`/metal-rs crate is deprecated); nothing on macOS shells out.
- **`RentalKind` extends `RentalResponse`.** SPEC §7 derives only an SSH
  `ssh_command`; the impl adds a `kind` (`ssh` | `http_status` | `http_openai`)
  reported by the agent so the tenant CLI shows the right endpoint hint (an SSH
  box vs a `curl` to an HTTP endpoint). The Metal tier reports `http_status`; the
  container tier reports `ssh`. `http_openai` is a reserved variant with no
  current producer (it was the removed `mlx:generate` inference endpoint).
  `ssh_command` is still populated for the `ssh` kind; `kind` defaults to `ssh`,
  so the acceptance test is unchanged.

## Deliberate gaps

Documented in SPEC §8. Custodial balances, POC-grade auth, Docker instead of a
microVM, host can read tenant memory, single benchmark at registration. **Do not
fix these opportunistically** — they are scoped out and they interact. If a task
requires crossing one of those lines, say so before starting.

Roadmap and priority order are in SPEC §9. The first item — randomized
re-benchmarking against each machine's own history — is the highest-value
addition, not the isolation work.
