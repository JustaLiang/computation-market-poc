# BACKGROUND — why this design

Context for judgment calls the spec doesn't cover. If you're about to argue for
putting the offer index on a chain, adding an auction, or verifying computation
cryptographically, read the relevant section first.

---

## 1. The decision that shapes everything: no chain in the control plane

Decentralized compute marketplaces split into two families, and this project is
deliberately in the second.

| | Unit sold | Matching | Verification | Settlement |
|---|---|---|---|---|
| Akash | container / K8s pod | on-chain reverse auction | **none** — reputation only | Cosmos appchain, escrowed lease |
| io.net | Ray / K8s cluster | off-chain, auto-priced | device attestation (PoW → Intel TDX) | Solana, off-chain accounting |
| Render | render frame / GPU job | tiered routing + staking | human review + bilateral reputation | Solana, burn-mint |
| Golem | VM task | P2P demand/offer negotiation | **none** — allocation caps loss | Polygon, streaming debit notes |
| **vast.ai** | **container** | **search + provider-set price** | **benchmark + reliability score** | **fiat, centralized** |

**None of them verify that arbitrary computation was performed correctly**,
because nobody knows how to do it cheaply. Verifiable general computation costs
roughly 10⁴–10⁶× native execution with ZK. Each project therefore picks an escape
hatch:

- Akash and Golem cap your financial downside and let you re-run elsewhere.
- Render exploits the fact that a bad render is *visually obvious* — its
  "proof-of-render" is human spot-checking plus bilateral reputation, which only
  works because the output is human-checkable.
- io.net is moving to TEEs, trading cryptographic verification for trust in a
  hardware vendor's attestation chain.

vast.ai skipped the problem entirely: centralized broker, agent on each host, no
token, no consensus. That is precisely why it has deeper GPU supply and lower
prices than the token projects. It solved the boring logistics problem instead of
the hard distributed-systems problem.

**This project follows vast.ai's shape and uses Bitcoin only for payment.**

A chain in the control plane buys you nothing here. On-chain offer indexing and
scheduling produce a slower, more expensive version of a SQL query with no added
trust — because a chain still cannot tell you whether the GPU is real. What a
chain (specifically Lightning) genuinely buys you is **cross-border settlement to
hosts without banking rails**, which is a real problem that stablecoins and BTC
solve well. That is the entire justification, and it should not expand.

The token projects mostly learned this the expensive way.

---

## 2. Why Lightning specifically

Per-minute billing rules out on-chain BTC on two independent grounds:

- **Fee.** At 1,000 sats/min, an on-chain tx fee is several times the payment.
- **Latency.** ~10 minutes to confirm against a 1-minute billing period.

Lightning handles 1,000-sat payments as a matter of course. It is not a
preference; it is the only Bitcoin layer that can express this workload.

The consequence for code: **integer satoshis everywhere.** Float or decimal
arithmetic on money in a system doing 1,440 debits per rental-day will
accumulate error that shows up as an unreconcilable ledger.

---

## 3. Why the provider sets one number

vast.ai's ranking is driven by price, internet speed, reliability, and DLPerf, in
that order. Once a machine is online its DLPerf is effectively fixed and its
reliability only climbs — so **price is the only lever a host can actually
pull.** That is a deliberate design choice: it forces the marketplace to compete
on price rather than on gaming the ranking.

An auction adds a matching engine, bid state, deposit handling, and a whole class
of race conditions, in exchange for price discovery that a sorted list already
provides. Not worth it at this stage. This is why the spec has no bidding and no
interruptible instances.

Sort by **sats per unit of dlperf** by default. Raw price ranks a T4 above an
H100, which is not what a buyer wants.

---

## 4. The hard parts, ranked by how much they actually hurt

This ordering is from observing what breaks these systems in practice, not from
what sounds most alarming.

### 4.1 Networking — the whole game

Hosts sit behind residential NAT, CGNAT, or a firewall someone else controls.
Tenants need SSH plus arbitrary ports. Two options:

- **Public static IP + open port range** → just record `ip:port`. Cheap, works
  for datacenters.
- **Behind NAT** → you need a relay (WireGuard mesh, `frp`, or your own reverse
  tunnel fleet). Now *you* pay egress for every byte, which destroys margin on
  anything bandwidth-heavy.

vast.ai's verification criteria weight high symmetric bandwidth, open ports, and a
static IP — they pushed the cost onto host *selection* rather than eating it. Do
the same: static IP as a hard requirement for the primary supply tier, relay only
as a degraded tier priced to cover egress.

This is also why control traffic is agent-initiated. It costs nothing and it means
a firewalled host can still register and heartbeat even if it can't serve tenants.

### 4.2 Fraud — and not the kind you'd expect

The attack is **not** "host returns wrong results." It is **misrepresented
hardware**: listing 8×H100 while running 1×3090, or overcommitting one GPU across
four tenants.

io.net was hit by exactly this — a Sybil attack registering fake GPUs to farm
rewards. The attack was on *device identity*, not on computation. The gap
persisted visibly in their numbers: hundreds of thousands of registered devices
against a few thousand daily verified.

So the benchmark's job is to make misrepresentation expensive, not to be accurate.
Spoofing `nvidia-smi` output is easy; maintaining a self-consistent fake PCIe
topology and GPU UUID set under *randomized re-probing* is not. Hence: benchmark
at registration, re-benchmark at random during idle windows and after each rental,
and compare against the machine's **own history** rather than an absolute
threshold. A host quietly halving its GPU allocation shows up as a distribution
shift, not a single bad reading.

This is roadmap item 1 for a reason. It bites before any isolation problem does.

### 4.3 Isolation, in both directions

**Tenant → host.** Docker is not a security boundary against a hostile tenant
holding GPU device access; escapes via the NVIDIA container runtime have been
real. The answer is microVMs (Firecracker, Cloud Hypervisor) with VFIO GPU
passthrough. gVisor is a non-starter — it breaks CUDA. This is why vast.ai treats
VM support as improving a machine's verification likelihood.

**Host → tenant.** The one people forget. The host is root and can read container
memory and disk: model weights, API keys, training data. vast.ai doesn't solve
this, it discloses it. The only real fix is confidential computing — H100/H200 CC
mode plus Intel TDX or AMD SEV-SNP, with the tenant verifying an attestation
quote before shipping any secret. Pin the image measurement, check the measurement
registers, bind the quote to a fresh nonce. An attestation that isn't verified
tightly is a decorative signature.

### 4.4 Metering and disputes

Per-second (or per-minute) billing means the source of truth is a heartbeat from
an untrusted machine. Design so neither side can unilaterally lie: agent reports
usage, the tenant's workload reports liveness independently, control plane
reconciles.

Decide upfront who eats a mid-run host failure. vast.ai's answer is "the tenant,
use checkpoints," which is honest — and it's why the standard advice for that
platform is checkpoint-based training and never production SLAs. This project
takes a slightly more tenant-favourable line: **the billing clock stops when the
host goes silent.** The tenant loses their run but not their sats.

### 4.5 Cold start

A marketplace with no GPUs has no tenants and vice versa. vast.ai's answer is
reputation-gated tiers with a slow on-ramp — verification takes weeks, and offers
below a reliability floor don't appear in the default view at all. Out of scope
here, but it's the reason reputation exists at all, so don't dismiss it as
bureaucracy when you get to it.

---

## 5. Structural decisions worth not re-litigating

**Tenant traffic bypasses the control plane.** The control plane is a rendezvous
and accounting service. If it becomes a proxy, egress cost scales with GPU-hours
sold and the margin is gone.

**Rental rate is snapshotted at creation.** A provider raising their price must
not affect a live rental. Reading the rate from the machine row during billing is
a bug.

**A machine with an active rental is not an offer.** `provisioning`, `running`,
and `evicting` are all occupied states. Treating only `running` as occupied
allows double-renting during provisioning.

**Billing is charged in advance and `paid_through` advances by exactly one
period.** Never assign `now + BILL_PERIOD`. A delayed tick must catch up one
period at a time rather than gifting free time; a duplicated tick must not
double-charge. This was duplicated across two call sites in the reference
implementation and caused a real bug.

**The ledger is double-entry.** `SUM(delta_sats) == 0` across the whole table,
always. This assertion caught a real bug during the reference build: tenant
debits were recorded but the matching host credits were not, so the audit trail
didn't reconcile. In a system custodying funds that is a bug you want a test to
find, not an auditor.

**All container code lives in one module.** It will be replaced by a microVM
launcher. Keeping the surface narrow means that swap doesn't touch the agent's
control logic.
