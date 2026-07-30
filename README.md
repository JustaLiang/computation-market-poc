# gpu-market — spec bundle

Design context for a vast.ai-shaped GPU rental marketplace with Bitcoin payments
over Lightning. No implementation here; this is the handoff for building one.

| File | Purpose |
|---|---|
| `CLAUDE.md` | Read automatically by Claude Code. Rust stack, workspace layout, invariants, build order. |
| `docs/SPEC.md` | Normative spec: data model, HTTP API, billing algorithm, state machine, acceptance test. |
| `docs/BACKGROUND.md` | Why the design is what it is. Market analysis, the hard parts ranked, decisions not to re-litigate. |

## Use with Claude Code

```bash
mkdir gpu-market && cd gpu-market
# copy CLAUDE.md and docs/ in
git init && git add -A && git commit -m "spec"
claude
```

Then: `read docs/SPEC.md and docs/BACKGROUND.md, then implement step 1 from the
build order in CLAUDE.md`.

Build steps 1–7 need no GPU, no container runtime, and no Lightning node. Get
`tests/lifecycle.rs` green before starting the host agent.

## Scope in one paragraph

Host agent is full: registers, self-benchmarks, heartbeats, runs containers with
GPU passthrough, maps SSH ports, tears down on command. Control plane is an offer
index only — no auction, no matching engine, no reputation. Providers set one
integer price in sats per minute. Payment is Lightning with a prepaid balance:
deposit once, metered per minute charged in advance, evicted at zero. Nothing
except payment touches a chain.
