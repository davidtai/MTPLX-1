# Qwen3.8 Adaptive Fixed-M4 Capacity Growth Design

## Scope

Improve long-output TPS by reducing fixed-M4 cache-capacity transitions. For a
large request, reserve 1,024 output tokens at first promotion, then grow the
capacity grant through 2,048, 4,096, 8,192, and 16,384 tokens. Further
overruns grow in 16,384-token chunks. Small declared output budgets remain sized to their actual
budget plus the speculative window rather than being rounded up to 1,024.

The unchanged production policy is the benchmark control. It starts large
requests with 512 tokens and continues in 512-token chunks; the guarded 32K
output required 63 capacity transitions and measured 465.253 seconds wall,
72.603 decode tok/s, and an 81.39 GiB peak.

## Non-goals

- Do not change physical verifier width M=4, native MTP depth, sampling,
  acceptance, commit, rollback, or output tokens.
- Do not predict response length from prior requests or reserve the full
  `max_tokens` allowance.
- Do not add per-token counters, model validation, environment reads, or an
  eager fallback to the enabled fixed-M4 hot path.
- Do not change generic compiled-verifier growth or other model families.

## Capacity Policy

Construction resolves the initial reserve once. The strict Qwen4 fixed-M4
ceiling changes from 512 to 1,024 tokens; the generic compiled verifier remains
at 512. Explicit smaller request budgets still tighten the allocation to
`request_max_tokens + speculative_headroom`.

An installed Qwen4 fixed-M4 dispatch owns a `growth_tokens` grant. When the
host-owned committed-token boundary shows that the next four-row window cannot
fit, it grows every QSA state leaf by the current grant, then advances the
grant using:

```text
next_growth = min(current_growth * 2, 16384)
```

An explicit operator reserve larger than 16,384 remains authoritative and is
never reduced. Growth may be capped at the request's reachable logical end to
avoid allocating capacity the request cannot address. The ordinary compiled
replay remains branch-free within each capacity generation; only the existing
boundary transition performs this calculation.

For a 32K output, the expected large-request grants are an initial 1K followed
by 2K, 4K, 8K, and 16K growth steps, reducing capacity transitions from 63 to
about 5. Maximum unused target-QSA capacity after an overrun is below one 16K
grant, approximately 0.434 GiB for the measured 12-cache geometry.

## Correctness and Failure Handling

All 12 QSA entries must grow to one common capacity before the capacity-bound
fused gather or compiled verifier is rebound. Failure remains loud; the strict
fixed-M4 route must not fall back, demote, or continue with mixed capacities.
Final demotion continues to expose only the accepted logical prefix, so unused
reserve cannot enter SessionBank state.

The policy changes only backing shapes. Identical seeds and inputs must produce
identical token digests, acceptance counts, and verifier work relative to the
512-token control.

## Verification

1. Add focused tests before implementation for the 1,024 default, the
   1K/2K/4K/8K/16K grant sequence, request-end clamping, explicit operator
   overrides, and compact final demotion.
2. Run the fixed-M4, graph-bank, QSA, profile, and server suites under the
   exclusive GPU guard.
3. Run matched guarded control/candidate cells for actual 2K, 16K, and 32K
   outputs on the exact model revision and production stack.
4. Use lowest wall time as the primary metric and report decode TPS, transition
   count, traces, active memory, and peak memory.

## Promotion Gate

Retain the adaptive ladder only when long-output wall time improves repeatably,
token digests and work trajectories match the unchanged control, every M4 call
remains compiled, fallback/demotion/repair stay zero, and peak memory increases
by no more than 0.5 GiB. If the 2K cell regresses materially or the 16K/32K
wall-time gain is noise, keep the proven 512-token policy.

## Failure-Mode Check

- **Critical: shape changes alter arithmetic or sampling.** Require identical
  token digests and work counters before considering TPS.
- **Critical: a growth step leaves mixed QSA capacities or a stale fused-gather
  binding.** Grow all entries first and reinstall the capacity-owned route as
  one boundary transition; focused tests must fail on partial ownership.
- **Minor: long grants waste memory when output stops just after an overrun.**
  Cap grants at 16K and enforce the 0.5 GiB peak-memory promotion limit.
