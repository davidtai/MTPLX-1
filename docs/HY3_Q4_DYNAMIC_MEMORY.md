# Hy3 Q4 dynamic memory serving

The Hy3 Q4 dynamic-memory lane is experimental and off by default. It runs one
full-attention sequence whose Q4 KV may grow to 131,072 total tokens while one
broker keeps the complete process inside a configurable unified-memory limit.

The expert cache uses direct MLX record buffers and positional reads:

- startup allocates no persistent expert buffers;
- a cache miss allocates and fills one demanded record;
- a hit performs no broker query and takes no memory-management lock;
- a full-cache miss overwrites the least-recently-used unpinned record in place;
- KV growth may evict individual unpinned records until the allocation fits;
- no background memory controller or grouped resize lifecycle exists.

The broker has one `memory_limit_bytes` value. `allocator_headroom_bytes` is
reserved inside that limit, so the classified-pool limit is:

```text
classified_limit_bytes = memory_limit_bytes - allocator_headroom_bytes
classified_bytes = resident + KV + expert cache + staging + workspace
charged_bytes = classified_bytes + allocator cache
charged_residual_bytes = memory_limit_bytes - charged_bytes
```

The runtime value is configurable for different machines. The issue #46
hardware experiment pins 100 GiB with 1 GiB of allocator headroom; those values
are benchmark inputs, not constants in the cache implementation.

## Qualified launch shape

Build and verify the Hy3 expert manifest, then launch the target-only AR lane:

```bash
MODEL=/absolute/path/to/pipenetwork/Hy3-4bit
MANIFEST="$MODEL/expert-manifest.json"

uv run mtplx serve --model "$MODEL" --yes \
  --expert-streaming \
  --expert-model-key hy3-q4 \
  --expert-manifest "$MANIFEST" \
  --expert-memory-limit 100GiB \
  --expert-runtime-reserve 8GiB \
  --expert-allocator-headroom 1GiB \
  --expert-max-live-kv-tokens 131072 \
  --expert-transient-slots 32 \
  --expert-cache-policy lru \
  --expert-cache-scope global \
  --expert-slot-layout direct-slots \
  --hy3-q4-dynamic-memory \
  --hy3-q4-dynamic-context \
  --paged-kv-quantization q4 \
  --context-window 131072 \
  --scheduler-mode serial \
  --max-active-requests 1 \
  --decode-batch-max 1 \
  --no-session-bank-live-refs \
  --ssd-session-cache off \
  --generation-mode ar \
  --no-load-mtp
```

The dynamic opt-in forces record-level telemetry, exact 84,480-byte Q4 KV
geometry, global LRU caching, and direct slots. Contradictory combinations are
rejected before model load. A JSON expert config cannot enable the dynamic
cache without the separate serving opt-in.

## Memory and request contract

The 131,072-token limit is total logical context: rendered input plus requested
output. Attention stays full-history. The lane does not use a sliding window,
spill KV to SSD, drop old KV, batch a second sequence, or retain live SessionBank
KV references. A second request receives HTTP 429 while the first owns the
single-sequence lane.

Physical KV starts at one 16-token block and grows from authoritative live
context. A Hy3 aggregate block spans all 80 target caches and occupies
1,351,680 bytes. The full 131,072-token Q4 history is 8,192 blocks, or
10.3125 GiB. KV growth is one aggregate transaction: steady deltas are summed,
the largest serialized replacement is charged as transient memory, enough
individual expert records are evicted, and all cache members commit before
model execution resumes.

Only measured physical release receives budget credit. Moving MLX active memory
into its allocator cache is not release. Missing or contradictory telemetry,
an allocation beyond either limit, compression growth, swap growth, or an
unhealthy route fails closed.

## Health and CPU attribution

`GET /health` always includes `hy3_q4_dynamic_memory`. When disabled it returns
a stable `enabled: false` shape without sampling the expert runtime. When
enabled it reports the configured and classified limits, all charged pools,
Q4 logical/physical ownership, logical/allocated/active/resident expert record
counts, record allocations/reuses/evictions/releases, protected bytes, MLX
allocator values, process RSS/compression/swap, and failure state.

`python_control_cpu` reports cumulative thread CPU for route control, cache
policy, cache budgeting, KV broker work, and the separate reader worker. Cache
policy time is a subset of route-control time and must not be added to it. This
keeps Python overhead attributable if the controller is later moved to Rust.

The serving lane remains off by default until the foreground hardware campaign
passes the direct-record probe, quality, memory-safety, compression, GPU
activity, and paired 4K performance gates.
