# System Paged Cache Implementation Plan

> Execute serially.  The generic ownership layer, the standard paged adapter,
> DeepSeek hybrid specs, and DFlash2 chunk lifetime share one evolving storage
> contract.  Stop for the first real correctness or memory failure; do not add
> hypothetical features or tests.

**Goal:** Replace append/concatenate ownership on the exact Mia long-context
route with a reusable fixed-capacity paged cache system, then finish the exact
model benchmarks and PR.

**Design:** `docs/specs/2026-08-21-system-paged-cache-design.md`

## Task 1: Reusable fixed-capacity page ownership

**Files:**

- Create `mtplx/paged_cache.py`.
- Modify `mtplx/cache_state.py`.
- Modify only directly relevant cases in `tests/test_cache_state.py`.

Implement immutable cache specs/plans, a fixed physical pool, request leases,
block tables, slot mappings, paged views, bounded trim, and state handoff.  Parse
configuration and validate geometry when the plan is installed.  The fixed
route must have no geometric growth, concatenation, hot environment reads, or
silent fallback.

Gate: make the existing fixed paged-cache construction/update/trim tests pass
through the shared owner.  Do not add concurrency, eviction, or prefix-sharing
tests.

Commit: `feat: add reusable fixed paged cache ownership`

## Task 2: Move the existing system paged cache onto the owner

**Files:**

- Modify `mtplx/cache_state.py`.
- Modify the runtime construction boundary that currently installs paged cache
  settings.
- Modify existing paged-cache tests only where the delegated owner changes the
  required contract.

Make `VllmMetalPagedKVCache` delegate storage, capacity, slot writes, trim, and
state transfer to the shared layer while preserving its attention interface.
Bind attention and quantization routes once at installation.  Keep the old
non-paged owners explicit; do not fall back to them from an enabled paged lane.

Gate: run the existing cache-state and GraphBank paged subsets that exercise
construction, writes, compiled offsets, trim, and state transfer.

Commit: `refactor: share paged cache ownership system-wide`

## Task 3: Install DeepSeek V4 hybrid page specs

**Files:**

- Modify `mtplx/deepseek_v4_nvfp4_kv.py`.
- Modify `mtplx/models/deepseek_v4.py`.
- Modify `mtplx/models/deepseek_v4_dspark.py`.
- Modify existing DeepSeek NVFP4, target, and DSpark checks.

Keep the proven `stock432` codec and arithmetic.  Replace growing record arrays
with paged views sized from the installed context capacity and real logical-to-
stored ratios.  Page the growing ratio-4/ratio-128 target compressed lanes and
ratio-4 indexer lane; keep target and draft windows bounded.  Preserve current
compressor frontier arithmetic and rollback semantics.

Gate: the current exact record/arithmetic/trim/DSpark checks pass, and a direct
16K cache exercise no longer replaces growing arrays.

Commit: `feat: page DeepSeek V4 native NVFP4 caches`

## Task 4: Stream DFlash2 context into draft pages

**Repositories:** MTPLX and the already-pinned DFlash2 dependency.

Add a construction-selected streaming context-consumer interface.  On the
DeepSeek adapter, project only the scheduled target chunk, write the three
draft-layer context K/V records into their persistent slots, and release chunk
features.  Do not allocate `TargetFeatureStore[prompt_length]` for this route.
Keep the existing retained-feature route for adapters that require it.

Gate: the existing DeepSeek DFlash2 adapter checks pass and the encountered
full-prompt feature-store allocation is absent.  Run one exact-model chunked
prefill/epoch before proceeding.

Commit DFlash2 first, pin that commit in MTPLX, then commit MTPLX as
`feat: stream DeepSeek draft context into paged cache`.

## Task 5: Exact-model execution and first-failure repair

Use the GPU service-restoration guard and `/tmp/mtplx-gpu-exclusive.lock` for
every Metal run.  Run only:

1. one exact-model DSpark epoch with committed-token parity;
2. the requested roughly 100-token Python prompt;
3. 1,024 prefill plus 1,024 decode;
4. cold 16,384 prefill plus 1,024 decode; and
5. cold 65,536 prefill plus 1,024 decode.

Each receipt records source/model revisions, prefill and decode tok/s, DSpark
acceptance, active and peak memory, generated-token digest, and cache-plan
identity.  If a run fails, diagnose that concrete failure and repair only what
blocks the next requested gate.

## Task 6: Review and publish

Run focused lint and only the cache, DeepSeek NVFP4, DSpark, DFlash2, runtime,
and directly affected GraphBank checks.  Inspect the implementation against the
immutable-plan/no-fallback/no-hot-validation constraints.  Push DFlash2 and
MTPLX commits, correct draft PR #312 to name the Mia/Sero artifact and exact
revisions, attach only valid receipts, and make it ready once every requested
gate passes.

