# Hy3 Q4 dynamic-memory hardware campaign

This campaign is the hardware evidence producer for issue #46. It compares a
static 131,072-token Q4 reservation with the dynamic Q4/expert-slab broker at
4K, 32K, 64K, and 128K total context. The schedule is balanced AB/BA, the Qwen
service is restored to its exact captured state, and every arm is rejected
unless its source and inputs are tracked and the worktree is clean.

## Frozen inputs

The checked-in campaign spec is
`benchmarks/specs/issue46-hy3-dynamic-memory.json`. Its hardware hook config is
`benchmarks/specs/issue46-hy3-hardware-hooks.json`. They pin:

- `pipenetwork/Hy3-4bit` revision
  `160619d3f96c8470350b6dac0ef033a8381551e3` and its verified expert manifest;
- plain paged Q4 KV, 16-token blocks, TurboQuant disabled, and a 131,072-token
  full-attention contract; both arms charge the exact 84,480-byte/token Q4
  geometry (10.3125 GiB at 128K) in their expert-memory plans;
- a 110 GiB operating target, 8 GiB runtime reserve, 32 transient slots, and
  32 independently releasable expert-slab slots;
- 16 measured generation tokens plus three 8-token hold samples, with two
  0.5-second gaps so the stable physical hold spans at least one second;
- four balanced repetitions and 10,000 paired bootstrap resamples.

The model snapshot, `expert-manifest-sidecar.json`, its verified 150 GiB
`experts.bin` payload, and every resident tensor payload must exist under the
pinned Hugging Face cache path in the hook config; the campaign deliberately
rejects the non-sidecar manifest. The Qwen launch agent must be
`~/Library/LaunchAgents/com.tea.qwen.plist`, and a loaded service must expose
exactly `mtplx-qwen36-27b-optimized-speed` at
`http://127.0.0.1:8080/v1/models`.

## Preflight and run

Run from the repository root. The plan command is non-mutating and validates
the matrix, balanced order, allocator probe command, arm command template, and
all six Qwen hooks:

```bash
uv run python benchmarks/benchmark_hy3_dynamic_memory.py \
  --spec benchmarks/specs/issue46-hy3-dynamic-memory.json \
  --cwd . \
  --plan-only
```

The hardware run requires a committed, completely clean worktree, including no
untracked files. Put the result outside the repository so it cannot dirty later
arms. The runner invalidates an existing output path before any actual-run
provenance check, so a failed rerun cannot leave stale successful evidence at
the requested path:

```bash
test -z "$(git status --porcelain=v1 --untracked-files=all)"
mkdir -p /tmp/mtplx-issue46
uv run python benchmarks/benchmark_hy3_dynamic_memory.py \
  --spec benchmarks/specs/issue46-hy3-dynamic-memory.json \
  --cwd . \
  --output-json /tmp/mtplx-issue46/campaign.json
```

Do not run an arm command by hand while Qwen is loaded. The campaign runner
owns the exclusive window and restores Qwen in `finally`, including after probe
or arm failure. It holds both `/tmp/mtplx-gpu-exclusive.lock` (the legacy
advisory lock used by older benchmark wrappers) and the owned
`/tmp/mtplx-gpu-exclusive` directory for that complete window. Failure to take
the legacy lock within the spec-pinned wait rejects before directory
acquisition, capture, or unload. Waiting occurs before termination cleanup is
installed, so an operator can still interrupt a queued campaign safely.

Before and after the exclusive window, the runner proves that the campaign
spec and every Python command source are tracked and the worktree is clean. The
artifact-verification, allocator-probe, quality, and performance-arm commands
must also reference one tracked hardware-hooks JSON. The result binds the raw
spec bytes as `campaign_spec_sha256`, that hooks JSON as
`hardware_hooks_config_sha256`, and the full source commit as
`source_git_commit`; any mid-run source, spec, or hooks-config change rejects
the result before it is written. A completed campaign whose acceptance status
is `rejected` still writes its rejection evidence, but exits with status 2.

## Exact Qwen subprocess contract

All commands below emit exactly one JSON object on stdout. `capture` records
both the launchd loaded state and the exact `/v1/models` IDs. The runner passes
that same object on stdin to `unload`, `restore`, and `verify`. Before `unload`
can run, the runner creates and file-plus-directory-syncs
`/tmp/mtplx-gpu-exclusive/issue46-recovery.json` with the owner identity and
captured state. It removes the journal only after exact restoration is verified;
failed restoration retains both the journal and exclusive lane for recovery.
The campaign spec pins `legacy_exclusive_lane_lock` to
`/tmp/mtplx-gpu-exclusive.lock`; the parent campaign holds that advisory lock
from before directory acquisition until after verified restoration and release.

```json
{
  "acquire_lane_command": ["uv", "run", "python", "benchmarks/hy3_qwen_isolation.py", "acquire", "--expected-model", "mtplx-qwen36-27b-optimized-speed"],
  "release_lane_command": ["uv", "run", "python", "benchmarks/hy3_qwen_isolation.py", "release", "--expected-model", "mtplx-qwen36-27b-optimized-speed"],
  "capture_command": ["uv", "run", "python", "benchmarks/hy3_qwen_isolation.py", "capture", "--expected-model", "mtplx-qwen36-27b-optimized-speed"],
  "unload_command": ["uv", "run", "python", "benchmarks/hy3_qwen_isolation.py", "unload", "--expected-model", "mtplx-qwen36-27b-optimized-speed"],
  "restore_command": ["uv", "run", "python", "benchmarks/hy3_qwen_isolation.py", "restore", "--expected-model", "mtplx-qwen36-27b-optimized-speed"],
  "verify_command": ["uv", "run", "python", "benchmarks/hy3_qwen_isolation.py", "verify", "--expected-model", "mtplx-qwen36-27b-optimized-speed"]
}
```

The adapter reuses the hardened issue-30 guard for
`/tmp/mtplx-gpu-exclusive`, launchd/process/API ambiguity checks, service
shutdown, exact-model restoration, and timeouts. Diagnostics are redirected to
stderr so they cannot corrupt the runner's JSON channel. A missing plist,
unexpected model, duplicate model ID, orphan process, API/service disagreement,
stale exclusive lane, incomplete stop, or inexact restoration fails closed.
Successful result JSON includes positive legacy-lock acquire/release, directory
acquire, capture, unload, restore, verify, and release evidence plus the
recovery-journal digest and lifecycle.

## Evidence ordering and interpretation

Each arm emits one schema-v1 observation directly consumable by
`benchmarks/benchmark_hy3_dynamic_memory.py`:

- Static starts with and retains all 8,192 Q4 blocks through prefill,
  invocation, and the stable hold; no expert reclaim or future-demand regrow is
  allowed.
- Dynamic starts with one physical Q4 block. At both the near-final and final
  boundaries, the adapter declares all 80 target-cache members in one real
  broker group. The broker sums steady deltas, reserves only the largest
  serialized replacement transient, reclaims once, and captures the physical
  post-reclaim ledger before member 0 allocates. Every member then commits
  back-to-back before model execution resumes. Any per-entry reservation,
  missing owner, unequal block count, or reconstructed timestamp rejects the
  observation.
- Every dynamic ledger cross-checks retained Q4 arrays against broker KV bytes
  and expert slab registration against broker expert bytes. Both arms report
  MLX active/cache/peak, process RSS/compressed bytes, swap delta, all budget
  pools, slab/record capacity, pin/in-flight state, reclaim/regrow counters,
  resize timing, and failure counters. Exact Q4 bytes must equal physical
  blocks times 1,351,680 bytes.
- Prompt token IDs, generated token IDs, exact route trace, verified manifest
  expert hashes, slot health, cache hit rate, SSD bytes/token, TPS, and p50/p95
  token latency are preserved in the observation. Each timed hold sample also
  retains and hashes its own generated token IDs and route trace, and paired
  arms must match those workloads exactly.
- The integrated quality command runs BF16, static Q4, and dynamic Q4
  full-history retrieval at exact 4K/32K/64K/128K total contexts. It attests
  live storage dtypes and scales, earliest/latest markers, reset and later
  requests, and bounded Q4 streaming attention with zero whole-history
  dequantization, dense fallback, or paging bailout.
- Reset closes the retained physical Q4 cache and releases admission exactly
  once. Dynamic regrow is caused only by a real subsequent `ensure_route`
  demand; the observer never calls the slab-regrow method directly.

The campaign validator additionally requires static/dynamic prompt, token,
route, expert-hash, artifact, manifest, source-commit, and normalized-config
parity. It reports paired bootstrap intervals for stable TPS, expert hit rate,
SSD bytes/token, and p50/p95 latency. It rejects unstable hold TPS, missing
physical allocator release, growth before reclaim, non-quiescent slot health,
failed reset/regrow, normal charged memory above 110 GiB, stress charged memory
at or above 112 GiB, any swap growth, or process-compressed growth above the
frozen 512 MiB runaway threshold. At every short context it also requires more
dynamic expert capacity plus a confidence-bounded improvement in hit rate, SSD
bytes/token, or TPS. At 128K, expert capacity must converge within one slab and
stable TPS/p50/p95 may not regress by more than 5%.

This campaign establishes issue #46's memory, paired-performance, and integrated
issue-#43-style full-history quality evidence. Memory arithmetic or
static/dynamic token parity is not a substitute for the BF16/Q4 retrieval gate.
The dynamic broker remains off by default unless every integrated gate passes.
