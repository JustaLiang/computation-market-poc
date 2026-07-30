# SPEC — GPU rental marketplace with Lightning payments

Implementation-agnostic. Everything here is required behaviour. See
`BACKGROUND.md` for why these choices were made, and `../CLAUDE.md` for the Rust
stack and conventions.

---

## 1. System shape

Three components. The control plane is a rendezvous and accounting service; it
is **not** a proxy.

```
Tenant ──── HTTP ────> Control plane <──── HTTP (agent-initiated) ──── Host agent
   │                  (offers, rentals,                                 (docker,
   │                   sats ledger)                                      GPUs)
   │
   └──────── SSH / workload traffic, direct to host ─────────────────────────┘
```

**Tenant workload traffic never traverses the control plane.** SSH, Jupyter, and
inference go straight to the host's public IP on a mapped port. If this is
violated, egress cost scales with GPU-hours sold and the margin disappears.

**All control traffic is agent-initiated.** The control plane never opens a
connection to a host. Commands are queued server-side and delivered in the
heartbeat response. This is what lets a firewalled host participate in control
flow even though it still needs reachable ports for tenant traffic.

---

## 2. Money

**All monetary values are integer satoshis.** No floats, no decimals, no
BTC-denominated values in code or storage. Conversion happens at the display
layer only, never in logic or persistence.

Providers set exactly one price: `rate_sats_per_min`, an integer ≥ 1. There is no
auction, no bidding, no matching engine.

Sanity anchors at roughly $100k/BTC:

| Rate | Per hour | Realistic for |
|---|---|---|
| 6 sats/min | 360 sats/hr ≈ $0.36 | single RTX 4090 |
| 60 sats/min | 3,600 sats/hr ≈ $3.60 | H100 |
| 1,000 sats/min | 60,000 sats/hr ≈ $60 | 8×H100 |

### Why Lightning and not on-chain

At 1,000 sats/min an on-chain settlement would cost several times the payment
itself and take ~10 minutes to confirm against a 1-minute billing period.
On-chain BTC cannot express per-minute billing. Lightning can, and 1,000 sats is
an ordinary Lightning payment.

### Payment model (POC: custodial prepaid balance)

1. Tenant requests a deposit → control plane issues a BOLT11 invoice.
2. Invoice settles → tenant's `balance_sats` is credited.
3. Billing loop debits `rate_sats_per_min` per minute, **in advance**.
4. Balance falls below one minute's rate → rental is evicted.
5. Host accrues `payout_balance`, supplies an invoice, control plane pays it.

This is custodial and that is the main deliberate flaw. The non-custodial
successor is specified in §8.

---

## 3. Data model

Integer satoshis throughout. Timestamps are Unix seconds.

### machines

Registered hosts. A machine is an *offer* when online and idle — there is no
separate offers table.

| Field | Type | Notes |
|---|---|---|
| `id` | i64 PK | |
| `agent_token` | text unique | bearer token issued at registration |
| `host_id` | text unique | stable, agent-generated, survives restarts |
| `gpu_name` | text | e.g. "NVIDIA GeForce RTX 4090" |
| `gpu_count` | i32 | |
| `vram_mb` | i64 | per GPU |
| `cpu_name`, `cpu_cores`, `ram_mb`, `disk_gb`, `disk_type` | | `disk_type` ∈ {nvme, ssd, hdd, unknown} |
| `public_ip` | text | where tenants connect |
| `port_start`, `port_end` | i32 | must be reachable at `public_ip` |
| `inet_down_mbps`, `inet_up_mbps` | f64 nullable | |
| `country` | text nullable | ISO-3166 alpha-2 |
| `dlperf` | f64 | see §6 |
| `rate_sats_per_min` | i64 ≥ 1 | provider-set |
| `payout_balance` | i64 | sats owed to this host |
| `online` | bool | |
| `last_heartbeat` | i64 | Unix seconds |
| `hw_fingerprint` | text | hash of GPU UUIDs + PCIe topology |
| `created_at` | i64 | |

### accounts

| Field | Type |
|---|---|
| `id` | text PK, e.g. `acct_<16 hex>` |
| `balance_sats` | i64, never negative |
| `created_at` | i64 |

### rentals

| Field | Type | Notes |
|---|---|---|
| `id` | i64 PK | |
| `machine_id` | FK machines | |
| `account_id` | FK accounts | |
| `image` | text | container image |
| `ssh_pubkey` | text | injected as authorized_keys; never returned by the API |
| `status` | enum | see §4 |
| `ssh_host`, `ssh_port` | | populated by the agent's report |
| `container_id` | text nullable | |
| `error` | text nullable | |
| `rate_sats_per_min` | i64 | **snapshotted at creation**; a provider raising their price must not affect a live rental |
| `sats_charged` | i64 | cumulative |
| `minutes_billed` | i64 | |
| `paid_through` | i64 | Unix seconds; billing is settled up to this instant |
| `created_at`, `ended_at` | i64 | |

### invoices

| Field | Type |
|---|---|
| `payment_hash` | text PK, hex |
| `account_id` | FK accounts |
| `sats` | i64 |
| `bolt11` | text |
| `settled` | bool |
| `created_at` | i64 |

### ledger

Append-only audit trail. Never updated, never deleted.

| Field | Type |
|---|---|
| `id` | i64 PK |
| `ts` | i64 |
| `account_id`, `machine_id`, `rental_id` | nullable FKs |
| `delta_sats` | i64, signed |
| `kind` | text: `deposit`, `rental_charge`, `host_credit`, `payout`, `evict` |
| `note` | text nullable |

### commands

Queue of instructions awaiting heartbeat delivery.

| Field | Type |
|---|---|
| `id` | i64 PK |
| `machine_id` | FK machines |
| `payload` | JSON |
| `delivered` | bool |
| `created_at` | i64 |

Indexes: `machines(online, dlperf)`, `rentals(status)`,
`commands(machine_id, delivered)`.

---

## 4. Rental state machine

```
                 ┌──────────────┐
  POST /rentals  │ provisioning │  first minute charged here, before any
  ─────────────> └──────┬───────┘  container exists
                        │ agent reports success
                        v
                 ┌──────────────┐
                 │   running    │  metered per minute
                 └──────┬───────┘
                        │ balance exhausted, or DELETE /rentals/{id}
                        v
                 ┌──────────────┐
                 │   evicting   │  stop_rental queued, awaiting agent
                 └──────┬───────┘
                        │ agent reports
                        v
                 ┌──────────────┐        ┌──────────────┐
                 │   stopped    │        │    failed    │ <── agent reports
                 └──────────────┘        └──────────────┘     failure from
                                                              provisioning
```

`provisioning`, `running`, and `evicting` are all **occupied** states — a machine
in any of them is excluded from the offer index.

---

## 5. Billing algorithm

Constants: `BILL_PERIOD = 60`s, `TICK_SECONDS = 5`, `HEARTBEAT_TIMEOUT = 90`s.

`BILL_PERIOD` must have exactly one definition in the codebase. Duplicating it
between rental creation and the ticker caused a real bug in the reference
implementation.

### On rental creation

Inside a single serialized transaction:

```
lock machine row
assert machine.online
assert no rental on machine in {provisioning, running, evicting}
assert account.balance_sats >= machine.rate_sats_per_min
rate := machine.rate_sats_per_min          # snapshot
insert rental{status: provisioning, rate, sats_charged: rate,
              minutes_billed: 1, paid_through: now + BILL_PERIOD}
account.balance_sats  -= rate
machine.payout_balance += rate
ledger += (rental_charge, -rate, account)
ledger += (host_credit,   +rate, machine)
queue command{start_rental, rental_id, image, ssh_pubkey}
commit
```

The first minute is charged **before the container exists**. This is intentional:
a tenant must never consume compute they have not paid for.

### Ticker, every TICK_SECONDS

```
poll_invoices()      # settle -> credit balance, ledger += deposit
mark_offline()       # last_heartbeat < now - HEARTBEAT_TIMEOUT  =>  online = false
bill_once()
```

### bill_once

```
for each rental where status == running:
    if now - machine.last_heartbeat > HEARTBEAT_TIMEOUT:
        continue                        # clock stops; see below
    if now < rental.paid_through:
        continue                        # already paid for this period
    rate := rental.rate_sats_per_min
    if account.balance_sats < rate:
        rental.status := evicting
        rental.error  := "insufficient balance"
        queue command{stop_rental, rental_id}
        ledger += (evict, 0, account)
        continue
    account.balance_sats   -= rate
    machine.payout_balance += rate
    rental.sats_charged    += rate
    rental.minutes_billed  += 1
    rental.paid_through     = max(paid_through, now - BILL_PERIOD) + BILL_PERIOD
    ledger += (rental_charge, -rate, account)
    ledger += (host_credit,   +rate, machine)
```

Two subtleties that are easy to get wrong:

**`paid_through` advances by exactly one period.** Never assign
`now + BILL_PERIOD`. The `max(paid_through, now - BILL_PERIOD)` clamp means a
delayed tick catches up one period at a time instead of gifting free time, while
a duplicated tick cannot double-charge.

**The clock stops for a silent host.** If a machine has not heartbeat within
`HEARTBEAT_TIMEOUT`, the rental is not billed. The tenant did not receive what
they paid for. This rule matters more than it looks — it is the single place
where marketplace incentives most easily go bad.

---

## 6. Benchmark and fingerprint

`dlperf` is a scalar performance estimate, normalized so a single RTX 4090 lands
near **42**, scaled linearly by GPU count. Measure fp16 GEMM throughput
(8192³ matmul, 3 warmup + 20 timed iterations) and scale; fall back to a
GPU-name lookup table when no CUDA runtime is available.

Its purpose is **not** precision. It is to make hardware misrepresentation
expensive. A host claiming 8×H100 while running 1×3090 must either own the
hardware or produce a self-consistent fake under repeated measurement.

`hw_fingerprint` is a hash of the sorted set of `(gpu_name, gpu_uuid, pci_bus_id)`
tuples. It changes if hardware is swapped. Collect and store it; §9 item 1 acts
on it.

A single measurement at registration does **not** achieve the goal above. That
is a known gap, deliberately left for §9.

---

## 7. HTTP API

Auth: `Authorization: Bearer <agent_token>` for `/agent/*`. Tenant endpoints take
`account_id` in the body or path (POC-grade — see §8).

### Agent

**`POST /agent/register`** → `{machine_id, agent_token}`

Body: all `machines` spec fields except `id`, `agent_token`, `payout_balance`,
`online`, `last_heartbeat`, `created_at`.

Idempotent on `host_id`: re-registration after restart refreshes specs and rate
and returns the existing token. A machine reporting *less* hardware than
previously is a de-verification signal — log it; do not act on it yet.

**`POST /agent/heartbeat`** `{online: bool}` → `{commands: [...]}`

Updates `last_heartbeat`. Returns all undelivered commands and marks them
delivered in the same transaction. Commands:

```json
{"cmd": "start_rental", "rental_id": 1, "image": "...", "ssh_pubkey": "..."}
{"cmd": "stop_rental",  "rental_id": 1}
```

Delivery is at-most-once. Note the consequence: if an agent crashes between
delivery and execution, the command is lost and the rental stalls in
`provisioning`. Acceptable for a POC; a reconciliation pass comparing reported
running containers against `rentals` is the fix.

**`POST /agent/report`** `{rental_id, status, ssh_port?, container_id?, error?}`

Must verify the rental belongs to the authenticated machine. Sets `ssh_host`
from the machine's `public_ip`. Sets `ended_at` when status is `stopped` or
`failed`.

**`POST /agent/rate`** `{rate_sats_per_min}` → updates the machine's price. Live
rentals are unaffected because their rate was snapshotted.

**`POST /agent/payout`** `{bolt11}` → `{paid_sats}`

Pays the invoice, zeroes `payout_balance`, writes a `payout` ledger row. 400 if
nothing owed, 502 if the payment fails (balance must remain intact on failure).

### Tenant

**`GET /offers`** → `{offers: [...]}`

Query params: `gpu_name` (substring), `min_vram_mb`, `min_gpu_count`,
`max_rate_sats_per_min`, `sort` ∈ {`rate`, `dlperf`, `value`}, `limit` (≤200).

Returns online machines with **no** rental in `{provisioning, running,
evicting}`. Each row includes a derived `rate_sats_per_hour`.

`sort=value` orders by `rate_sats_per_min / max(dlperf, 0.01)` ascending — sats
per unit of performance, the only ranking a buyer actually cares about. Default
to it.

**`POST /accounts`** → `{account_id}`

**`GET /accounts/{id}`** → `{account_id, balance_sats, burn_sats_per_min, runway_minutes}`

`burn` is the sum of `rate_sats_per_min` over the account's running rentals.
`runway_minutes = balance / burn`, null when burn is zero.

**`POST /accounts/{id}/deposit`** `{sats}` → `{bolt11, payment_hash, sats}`

**`POST /rentals`** `{machine_id, account_id, image, ssh_pubkey}` → `{rental_id, status, rate_sats_per_min}`

Errors: 404 unknown machine/account, 409 offline or already rented, 402 balance
below one minute at the machine's rate.

**`GET /rentals/{id}`** → rental row plus `gpu_name`, `gpu_count`, and a derived
`ssh_command` when host and port are known. **Must never return `ssh_pubkey`.**

**`DELETE /rentals/{id}`** → transitions to `evicting`, queues `stop_rental`.
Idempotent for already-terminal rentals.

**`GET /health`** → `{ok, ln_backend}`

---

## 8. Trust model

Every item here is a known, deliberate gap. Do not fix opportunistically —
several interact, and fixing one alone can make things worse.

**Custodial.** The control plane holds tenant deposits and owes hosts. This is
the main deliberate flaw. The non-custodial design: replace the prepaid balance
with **streaming keysend** — the tenant's client pushes `rate` sats directly to
the host's node each minute, the agent kills the container after N consecutive
missed payments, and the control plane never touches money. The billing rules in
§5 carry over unchanged; only the custody moves.

**Auth is POC-grade.** The tenant identifier is a bare account id, so anyone who
learns it can spend the balance. Agent tokens are long-lived bearer tokens with
no rotation.

**Containers are not an isolation boundary.** A tenant with GPU device access and
a container escape owns the host. All container code must be confined to one
module so it can be replaced with a microVM (Firecracker or Cloud Hypervisor)
plus VFIO GPU passthrough.

**The host can read everything the tenant runs.** Root on the box means model
weights, datasets, and keys are visible. This is not solved here and is not
solved by vast.ai. The fix is confidential computing — H100/H200 CC mode with
Intel TDX or AMD SEV-SNP, where the tenant verifies an attestation quote with
measurement registers pinned and bound to a fresh nonce before shipping any
secret. An attestation that is not verified tightly is a decorative signature.

**Hosts need reachable ports.** Control traffic is agent-initiated, so a
firewalled host can register and heartbeat. Tenant traffic cannot —
`port_start..port_end` must be reachable at `public_ip`. NAT'd hosts require a
relay tier, which must be priced to cover egress.

---

## 9. Roadmap, priority order

1. **Randomized re-benchmarking.** Highest value. Re-run the benchmark during
   idle windows and after each rental; compare against the machine's *own*
   history rather than an absolute threshold; act on `hw_fingerprint` changes.
   This is the failure mode that damaged io.net's supply credibility, and it
   bites before any isolation problem does.
2. **Real auth.** Signed, rotatable agent tokens; API keys for tenants.
3. **Command reconciliation.** Compare agent-reported running containers against
   `rentals` to recover from lost at-most-once deliveries.
4. **Non-custodial streaming keysend.** Removes the custody liability.
5. **microVM isolation.** Firecracker + VFIO, confined to the runtime module.
6. **Attested confidential compute** as a premium tier.

---

## 10. Acceptance test

One test must exercise the full lifecycle with no GPU, no container runtime, and
no Lightning node — using a mock LN backend that auto-settles and a compressed
`BILL_PERIOD` (2s):

1. Host registers → `machine_id` returned.
2. Offer index lists it with correct `rate_sats_per_hour`.
3. Account created, deposit invoice issued, auto-settles, balance credited.
4. Rental created → first minute charged up front.
5. Heartbeat delivers `start_rental`; agent reports running; `ssh_command` correct.
6. Machine no longer appears in the offer index.
7. Ticker drains the balance one period at a time; rental flips to `evicting` at
   zero; `sats_charged` equals the deposit; **balance never goes negative**.
8. `stop_rental` is delivered; agent reports stopped; machine is re-listed.
9. Host payout succeeds and zeroes `payout_balance`.
10. **`SUM(ledger.delta_sats) == 0`.**

Item 10 is the important one. It caught a real bug in the reference
implementation: tenant debits were recorded but the matching host credits were
not, so the audit trail did not reconcile. In a system custodying funds that is a
bug you want a test to find, not an auditor.
