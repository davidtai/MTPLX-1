# Hy3 Q4 dynamic-memory hardware campaign

This is the hardware evidence producer for issue #46. It compares the static
131,072-token Q4 reservation with the demand-loaded direct-record cache at 4K,
32K, 64K, and 128K total context. The issue experiment pins a 100 GiB process
limit, 1 GiB allocator headroom, an 8 GiB runtime reserve, and 32 transient
slots.

The checked-in inputs are:

- `benchmarks/specs/issue46-hy3-dynamic-memory.json`;
- `benchmarks/specs/issue46-hy3-hardware-hooks.json`;
- `benchmarks/probe_hy3_direct_cache.py`.

The probe proves empty persistent-cache startup, one-record allocation and
execution, in-place buffer reuse for a different expert, individual release,
and matching allocator-byte reduction. The campaign rejects grouped allocator
fields in configuration or evidence.

## Foreground and lock guarantees

The campaign is foreground and synchronous. The parent process acquires
`/tmp/mtplx-gpu-exclusive.lock`, acquires the owned Qwen lane, captures and
unloads Qwen, waits for every probe/quality/arm child, restores and verifies the
captured Qwen state, releases both locks, and only then exits. Child processes
use their own process groups for reliable cleanup, but they are not detached or
background jobs: the parent blocks in `communicate()` until each child exits.

Never remove or steal a live lock and never kill another benchmark process. A
queued campaign waits up to the spec limit and may be interrupted safely before
Qwen unload. If a crash leaves an owner marker and recovery journal, first prove
the recorded owner and every child are dead, inspect current Qwen state, restore
and verify the journaled state, and only then remove stale ownership files.

## Plan and staged runs

The plan command is non-mutating and does not acquire the GPU lock:

```bash
uv run python benchmarks/benchmark_hy3_dynamic_memory.py \
  --spec benchmarks/specs/issue46-hy3-dynamic-memory.json \
  --cwd . \
  --contexts 4096 \
  --plan-only
```

Actual hardware runs require a committed, completely clean worktree. Put result
JSON outside the repository. Run the 4K gate first:

```bash
test -z "$(git status --porcelain=v1 --untracked-files=all)"
mkdir -p /tmp/mtplx-issue46
uv run python benchmarks/benchmark_hy3_dynamic_memory.py \
  --spec benchmarks/specs/issue46-hy3-dynamic-memory.json \
  --cwd . \
  --contexts 4096 \
  --output-json /tmp/mtplx-issue46/direct-cache-4k.json
```

Only after that passes should the full matrix run:

```bash
uv run python benchmarks/benchmark_hy3_dynamic_memory.py \
  --spec benchmarks/specs/issue46-hy3-dynamic-memory.json \
  --cwd . \
  --contexts 4096,32768,65536,131072 \
  --output-json /tmp/mtplx-issue46/direct-cache-full.json
```

`--contexts` must start with 4096 and may then include an ordered subset of the
remaining frozen matrix. Selection is parsed before any lock or Qwen action,
which prevents a long-context arm from bypassing the 4K safety/performance gate.

## Required evidence and acceptance

Every arm binds the tracked source commit, campaign spec, hooks config, pinned
model revision, manifest, complete expert sidecar, resident payload, prompt,
tokens, route trace, and expert hashes. The integrated quality command compares
BF16, static Q4, and dynamic Q4 full-history retrieval at every context.

Every memory point reports:

- the 100 GiB configured limit and classified limit;
- resident, Q4 KV, expert cache, staging, workspace, and allocator-cache bytes;
- logical, allocated, active, and resident expert records;
- individual record allocations, reuse, eviction, and release counts;
- pinned, in-flight, and speculative expert bytes;
- process RSS, compressed memory, system swap delta, and AGX GPU utilization;
- Python thread CPU for routing, policy, budgeting, broker, and reader work.

The result is rejected if charged or transient memory exceeds 100 GiB,
classified memory exceeds its limit, compression grows by more than 512 MiB,
swap grows, GPU activity stalls during decode, any admission/fail-closed event
occurs, or paired fixed pools differ. At 4K the direct-cache arm may be at most
20% slower than the static control. Longer contexts remain correctness and
memory-safety evidence; they do not require artificial cache-capacity parity.

The output is written even for a cleanly measured rejection (exit status 2),
but never for incomplete provenance or cleanup. Successful startup alone is not
hardware acceptance.
