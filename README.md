# computation-market-poc

A **GPU rental marketplace** with **Bitcoin/Lightning** payments, in Rust.
Providers list a GPU host and set one price in sats/min; tenants deposit
over Lightning, rent by the minute (billed in advance), and are evicted at zero
balance. The control plane is a rendezvous + accounting service — **not** a proxy;
workload traffic (SSH) goes straight to the host.

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

## Apple Silicon (MLX-only tier)

The **same `vgpu-agent` binary** runs on macOS behind a `HostRuntime` trait: it
inventories via Metal/`system_profiler`, benchmarks, registers (a Mac shows up in
`vgpu offers` as e.g. `Apple M4 Max`), and lends compute as an **MLX-only** tier.
The isolation model is an allowlist, not a shell: the tenant supplies only
*parameters* (the rental image), never code. The host runs one of a fixed set of
host-controlled programs:

- `mlx:gemm[:N]` — an fp16 GEMM loop on the Apple GPU with a live-throughput
  status endpoint.
- `mlx:generate:<hf-model-id>` — `mlx_lm.server` (OpenAI-compatible LLM inference)
  on the mapped port; the tenant picks a model, not code.

```bash
python3 -m venv .mlx && .mlx/bin/pip install mlx mlx-lm
VGPU_MLX_PYTHON="$PWD/.mlx/bin/python" vgpu-agent \
  --control-plane http://<server>:8080 --public-ip <ip> --rate-sats-per-min 5

# tenant — rent GPU time (GEMM benchmark, live TFLOP/s):
vgpu rent --machine-id <id> --account-id <acct> --image mlx:gemm:2048 --ssh-key unused
curl http://<host>:<port>/

# tenant — rent an LLM inference endpoint:
vgpu rent --machine-id <id> --account-id <acct> \
  --image mlx:generate:mlx-community/Llama-3.2-1B-Instruct-4bit --ssh-key unused
curl http://<host>:<port>/v1/chat/completions -H content-type:application/json \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":40}'
```

Arbitrary containers + SSH are the Linux tier; on macOS a non-`mlx:` image is
rejected. The port is an HTTP endpoint (not SSH). See
`crates/host-agent/src/runtime.rs`.

## gpu-bench (standalone)

Runs on any GPU via `wgpu` (Metal / Vulkan / DX12) — independent of the marketplace:

```bash
gpu-bench list                 # enumerate GPUs
gpu-bench run                  # fp32 + fp16 GEMM + memory bandwidth + a provisional compute index
gpu-bench run --network        # also probe internet down/up/latency (outbound HTTPS)
gpu-bench run --json           # machine-readable
```

## Development

```bash
cargo test --workspace         # 33 tests incl. the SPEC §10 acceptance test
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
