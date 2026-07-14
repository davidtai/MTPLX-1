# Hy3 Q4 dynamic memory serving

The Hy3 Q4 dynamic-memory lane is experimental and off by default. It runs one
full-attention sequence whose Q4 KV may grow to 131,072 total tokens while a
single broker shares physical memory with independently releasable expert
slabs. The broker's normal operating target is 110 GiB and its hard ceiling is
112 GiB. One GiB of the operating target is reserved as allocator headroom,
leaving a 109 GiB classified-pool target. Startup pins the physical Q4 geometry
to 84,480 bytes per token including scales; wider analytical KV defaults are
rejected for this lane.

Physical KV capacity starts at one 16-token page and is calculated from live
authoritative context, not the request's maximum context. At every growth
point, `page_slots = ceil(live_context_tokens / 16)` (with a one-page floor).
Each aggregate physical block holds 16 tokens across the 80 target cache
members and occupies 1,351,680 bytes. A partial final block is always allocated:
4,095 and 4,096 tokens require 256 blocks, while 4,097 tokens require 257. The
131,072-token maximum is allocated only if the sequence actually reaches it;
then it is exactly 8,192 blocks, or 10.3125 GiB. Expert record capacity is
planned from the remaining classified budget and the physically allocated slab
count; it is not hard-coded to one context size.

## Exact launch command

Build and verify the Hy3 expert manifest first, then launch the public server
with every lane constraint explicit:

```bash
MODEL=/absolute/path/to/pipenetwork/Hy3-4bit
MANIFEST="$MODEL/expert-manifest.json"

uv run mtplx serve --model "$MODEL" --yes \
  --expert-streaming \
  --expert-model-key hy3-q4 \
  --expert-manifest "$MANIFEST" \
  --expert-memory-limit 110GiB \
  --expert-runtime-reserve 8GiB \
  --expert-allocator-headroom 1GiB \
  --expert-max-live-kv-tokens 131072 \
  --expert-transient-slots 32 \
  --expert-cache-policy lru \
  --expert-cache-scope global \
  --expert-slot-layout component-banks \
  --hy3-q4-dynamic-memory \
  --expert-slab-slots 32 \
  --expert-regrow-hysteresis-slabs 1 \
  --expert-resize-min-interval-ms 1000 \
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

The runtime reserve, transient service bank, cache policy, slab, hysteresis, and
interval values above reproduce the frozen hardware-campaign identity. The
dynamic lane selects the same values when they are omitted, but spelling them
out makes the qualified serving configuration unambiguous. Explicit deviations
are experimental configurations and do not inherit the issue #46 hardware
result. Startup rejects an incomplete or contradictory safety combination
before model weights are loaded. A JSON expert config cannot activate dynamic
slabs without the separate serving opt-in.

## Operator contract

This serving lane is target-only autoregressive generation. SSD expert
streaming intentionally sets `load_mtp=False` and `generation_mode=ar`; MTP
and MTP KV-cache construction are rejected while the strict dynamic broker is
active. Target-plus-draft accounting needs a separately qualified aggregate
lifecycle and is not implied by this lane.

The 131,072-token limit is total logical context (rendered prompt plus requested
output), not a startup reservation or a prefill-only limit. Attention remains full-history. The lane does
not use sliding-window attention, spill KV to SSD, drop old KV, batch a second
sequence, or retain SessionBank snapshots/live KV references for a request. The
existing single-sequence admission gate returns HTTP 429 while another request
owns the lane.

Active KV, resident model weights, current-demand experts, and pinned/in-flight
experts are protected. Before physical Q4 KV growth, the broker must account for
the rounded block allocation for every discovered target attention cache. The
80 Hy3 members enter one transaction: steady deltas are summed, the largest
serialized replacement is the transient peak, expert reclaim is proved once,
and all members materialize back-to-back before model or expert allocation can
resume. Per-entry admission is forbidden in this lane. The five classified
pools are resident model, physical KV, expert slabs, in-flight staging, and
runtime workspace. Their sum must remain at or below 109 GiB. Adding charged
allocator cache must remain at or below the 110 GiB operating target, and every
transition peak must stay strictly below 112 GiB. Missing or contradictory
allocator telemetry fails closed; memory pressure, swap, compression,
allocation failure, and process termination are not resize signals.

Logical request ownership advances to the exact evaluated-token count after
every physical write. Physical page capacity changes only when that count
crosses a 16-token boundary. Q4 attention reads packed `uint8` K/V plus FP16
scales through bounded, explicitly realized
dequantization chunks; whole-history BF16-equivalent materialization, dense
fallback, and stale path claims are rejection conditions.

The startup memory plan is a structural admission check only. After the model
loads, the lane samples the real resident, slab, workspace, staging, KV, and
allocator state and runs the same budget equations again before serving. A
successful analytical plan cannot override a failed measured preflight.

KV close on completion, cancellation, or reset returns its broker ownership.
It does not immediately allocate replacement expert slabs. Expert capacity can
regrow only when a later route demands it, subject to slab-size hysteresis and
the minimum resize interval.

## Health telemetry

`GET /health` always includes `hy3_q4_dynamic_memory`. When disabled it has a
stable `enabled: false` shape and does not sample the expert runtime. When
enabled it reports the broker budget and charged pools, Q4 KV representation and
physical data when sampled, expert capacity/residency/slabs, protected expert
bytes, workspace/staging, allocator memory, reclaim/regrow/resize/admission
counters, cache and decode measurements, and best-effort process RSS,
compression, and swap delta. A field remains JSON `null` when its optional
runtime or operating-system sample is unavailable; the server does not invent a
replacement value.

The central ledger fields obey these identities:

```text
classified_target_bytes = operating_target_bytes - allocator_headroom_bytes
classified_bytes = resident + KV + experts + staging + workspace
charged_bytes = classified_bytes + allocator_cache_bytes
charged_residual_bytes = operating_target_bytes - charged_bytes
```

For this lane, `operating_target_bytes`, `allocator_headroom_bytes`, and
`classified_target_bytes` must be exactly 110 GiB, 1 GiB, and 109 GiB. Negative
residual, an over-target classified sum, a transition at or above 112 GiB, or
missing required allocator data fails admission.

## Evidence and recovery boundary

The issue #46 hardware campaign pins the model config, expert manifest, complete
150 GiB expert sidecar payload, and every resident tensor payload. It verifies
the full sidecar and resident payload hashes before the allocator probe and
again after all paired arms, while also checking shard/path fingerprints and
resident tensor headers. On macOS the verifier requires `F_NOCACHE` for these
hash reads so the verifier itself does not populate the filesystem cache. This
does not prove that the machine cache was cold, and reported SSD bytes are
logical expert-reader bytes rather than physical NAND traffic.

That 150 GiB figure is the on-disk corpus containing all streamed expert
records, not the HTTP request, KV cache, or resident unified-memory payload.
Only the broker-selected expert slabs are resident at a time; the full sidecar
is hashed solely to bind the benchmark to exact model bytes.

The campaign captures the exact Qwen state, acquires the exclusive GPU lane,
unloads Qwen, and restores and verifies the captured state in a `finally` block.
`SIGINT`, `SIGTERM`, and `SIGHUP` therefore enter the cleanup path. `SIGKILL`, a
machine reset, or a process crash that prevents Python cleanup cannot restore an
external service and may leave `/tmp/mtplx-gpu-exclusive` with an
`issue46-owner.json` marker and `issue46-recovery.json` journal. The journal is
file-and-directory-synced before Qwen unload and contains the exact captured
loaded/model state. In that case, do not delete either file first: confirm the
recorded campaign PID and all child arm/probe processes are no longer alive,
inspect Qwen's current model endpoint, restore the journaled pre-campaign state,
verify it, and only then remove the stale journal, owner marker, and empty lane
directory. Never start a second hardware campaign through stale ownership.

The lane remains experimental and off by default until the allocator-release,
correctness, memory-safety, and paired 4K/32K/64K/128K hardware gates in issue
#46 pass. Do not treat successful startup or a single `/health` sample as that
hardware acceptance result.
