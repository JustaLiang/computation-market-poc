# computation-market-poc

A **GPU rental marketplace** with **Bitcoin/Lightning** payments, in Rust.
Providers list a GPU host and set one price in sats/min; tenants deposit
over Lightning, rent by the minute (billed in advance), and are evicted at zero
balance. The control plane is a rendezvous + accounting service — **not** a proxy;
workload traffic (SSH, or an HTTP endpoint for Metal jobs) goes straight to the host.

Normative behaviour lives in [`docs/SPEC.md`](docs/SPEC.md); the "why" is in
[`docs/BACKGROUND.md`](docs/BACKGROUND.md).

**Status:** implemented and runnable end to end on a mock Lightning backend
(no GPU, no Docker, no node). `cargo test --workspace` is green, including the
SPEC §10 acceptance test which asserts `SUM(ledger.delta_sats) == 0`. The only
unbuilt piece is real Lightning (`LndRest`); use `LN_BACKEND=mock`.

```
   Tenant ──── HTTP ───▶ Control plane ◀─── HTTP (agent-initiated) ─── Host agent
   (vgpu CLI)          (offers · sats ledger ·                        (GPUs, Docker)
       │                command queue)                                     │
       └──────────────── SSH / workload, direct to the host ──────────────┘
```

## Components

| Binary | Crate | Role |
|---|---|---|
| `control-plane` | `crates/control-plane` | axum server: offer index, custodial sats ledger, billing ticker |
| `vgpu-agent` | `crates/host-agent` | runs on a GPU host: registers, heartbeats, launches workloads |
| `vgpu` | `crates/vgpu` | tenant CLI: browse offers, deposit, rent, stop |
| `gpu-bench` | `crates/gpu-bench` | standalone vendor-agnostic GPU benchmark (fp32/fp16 GEMM + bandwidth + network) |
| — | `crates/core` (`vgpu-core`) | shared types (`Sats`, DTOs); no I/O |

## Prerequisites

- **Rust 1.75+** (`rustup`). Everything below builds on macOS or Linux.
- To rent out a **real GPU** (host tier): **Linux + NVIDIA driver + Docker + the
  NVIDIA Container Toolkit** (`docker run --gpus all …` must work).

```bash
git clone https://github.com/JustaLiang/computation-market-poc
cd computation-market-poc
cargo build --release        # binaries land in target/release/
```

## Quickstart — the whole marketplace, hardware-free

No GPU, no Docker, no Lightning node. Invoices auto-settle; the billing period is
compressed only in the acceptance test (here it's the default 60s, but the mock
settles deposits immediately).

**Terminal 1 — the server:**
```bash
LN_BACKEND=mock MOCK_SETTLE_SECS=0 target/release/control-plane   # binds 127.0.0.1:8080
```

**Terminal 2 — register a host** (the DGX would run `vgpu-agent`; for a laptop
demo you can register directly):
```bash
curl -s -XPOST localhost:8080/agent/register -H content-type:application/json -d '{
  "host_id":"box-1","gpu_name":"NVIDIA GeForce RTX 4090","gpu_count":1,"vram_mb":24564,
  "cpu_name":"Ryzen 9","cpu_cores":16,"ram_mb":64000,"disk_gb":2000,"disk_type":"nvme",
  "public_ip":"203.0.113.9","port_start":40000,"port_end":40099,
  "dlperf":42.0,"rate_sats_per_min":6,"hw_fingerprint":"fp-1"}'
```

**Terminal 2 — drive it as a tenant** (`vgpu` talks to the tenant API):
```bash
export VGPU_CONTROL_PLANE=http://localhost:8080
vgpu offers                                   # the machine is listed
ACCT=$(vgpu create-account | sed -n 's/^Created account: //p')
vgpu deposit "$ACCT" 6000                      # get a (mock) invoice; settles on the next tick
sleep 6
vgpu account "$ACCT"                            # balance credited
vgpu rent --machine-id 1 --account-id "$ACCT" \
  --image nvidia/cuda:12.4.1-runtime --ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
vgpu rental 1                                  # status + derived `ssh …` command
vgpu stop 1
```

`vgpu --help` lists every command. All money is integer satoshis; the CLI shows a
BTC/USD hint (~$100k/BTC).

## Running a real GPU host (Linux + NVIDIA)

On the DGX (or any NVIDIA Linux box with Docker + the NVIDIA Container Toolkit),
run the agent pointing at your server. It NVML-inventories the GPUs, benchmarks,
`POST /agent/register`s (the offer appears), then heartbeats and runs containers:

```bash
vgpu-agent \
  --control-plane http://<SERVER_IP>:8080 \
  --public-ip <HOST_PUBLIC_IP> \
  --rate-sats-per-min 60 \
  --port-start 40000 --port-end 40099
```

Control traffic is agent-initiated, so the host only needs outbound access to the
server; tenants reach `port_start..port_end` at `public_ip` for SSH.

## Apple Silicon (native Metal tier)

The **same `vgpu-agent` binary** runs on macOS behind a `HostRuntime` trait: it
inventories via Metal/`system_profiler`, benchmarks, registers (a Mac shows up in
`vgpu offers` as e.g. `Apple M4 Max`), and lends compute as a **native Metal**
tier — **Rust-only, no Python, no `cmake`**. The isolation model is an allowlist,
not a shell: the tenant supplies only *parameters* (the rental image), never
code. The host runs one host-controlled program:

- `metal:gemm[:N]` — a continuous fp16 GEMM on the Apple GPU via a
  `simdgroup_matrix` MSL kernel (the matrix units), with a live-throughput JSON
  status endpoint. The agent launches it by re-invoking its own binary; the
  worker lives in `crates/host-agent/src/mac_worker.rs`.

```bash
# No venv, no Python — just the binary.
vgpu-agent --control-plane http://<server>:8080 --public-ip <ip> --rate-sats-per-min 5

# tenant — rent GPU time (GEMM, live TFLOP/s). No --ssh-key needed: the metal:
# tier serves an HTTP status endpoint, not an SSH box.
vgpu rent --machine-id <id> --account-id <acct> --image metal:gemm:2048
curl http://<host>:<port>/   # {"task":"gemm-fp16","n":2048,"iters":...,"tflops":...}
```

Arbitrary containers + SSH are the Linux tier; on macOS a non-`metal:` image is
rejected. Each rental carries a **kind** (`ssh` | `http_status` | `http_openai`),
so `vgpu rental <id>` prints the right hint — an `ssh` command for the container
tier, or the matching `curl` for the Metal status endpoint. See
`crates/host-agent/src/runtime.rs`.

## gpu-bench (standalone)

A vendor-agnostic GPU benchmark behind a `Backend` trait — independent of the
marketplace:

```bash
gpu-bench list                 # enumerate GPUs
gpu-bench run                  # auto: native peak + portable wgpu, side by side
gpu-bench run --backend wgpu   # force portable wgpu only (apples-to-apples)
gpu-bench run -v               # + resolved-backend line and explanatory note
gpu-bench run --json           # machine-readable
gpu-bench net                  # network-only probe (down/up/latency)
```

`run` defaults to `--backend auto`: it picks the best native backend compiled in
(cuBLAS > native Metal > wgpu) and, on a native backend, *also* measures the
portable wgpu path and prints both columns — the portable number is the
cross-vendor ranking metric. An explicit `--backend X` measures only X. Backends
(behind a `Backend` trait): **wgpu** (portable Metal/Vulkan/DX12 — ALU
throughput), **metal** (native `simdgroup_matrix` on Apple's matrix units via
`objc2-metal` — pure Rust, no Python/`cmake`; ~3.7 vs ~0.6 TFLOP/s fp16 on an M4
Max; macOS only), and **cuBLAS** (`--backend cuda`, built with `--features cuda`
on an NVIDIA host).

## Development

```bash
cargo test --workspace         # 32 tests incl. the SPEC §10 acceptance test
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo audit                    # clean (see .cargo/audit.toml for one accepted, unused-dep advisory)
```

Run the acceptance test after any change to billing, accounting, or the rental
state machine.

## Layout

```
crates/core            shared types (package `vgpu-core`): Sats, enums, API DTOs
crates/control-plane   axum server + sqlx/sqlite; migrations; billing ticker; tests/lifecycle.rs
crates/host-agent      vgpu-agent: benchmark.rs (inventory) + runtime.rs (HostRuntime: Docker/Mac)
crates/gpu-bench       standalone GPU benchmark (lib + CLI)
crates/vgpu            tenant CLI
docs/SPEC.md           normative behaviour spec
docs/BACKGROUND.md     design rationale
```

## Trust model (deliberate POC gaps)

Custodial balances, POC-grade bearer auth, Docker (not a microVM) for isolation,
the host can read tenant memory, and a single benchmark at registration. These are
scoped-out and interacting — see [SPEC §8](docs/SPEC.md) before "fixing" one. The
roadmap (SPEC §9) is led by **randomized re-benchmarking**, not isolation.

## License

MIT.
