# Hy3 Q4 dynamic memory serving

The Hy3 Q4 dynamic-memory lane is experimental and off by default. It runs one
full-attention sequence with a 131,072-token Q4 KV cache while a single broker
shares physical memory with independently releasable expert slabs. The broker's
normal operating target is 110 GiB and its hard ceiling is 112 GiB. Startup
pins the physical Q4 geometry to 84,480 bytes per token including scales, or
10.3125 GiB at 131,072 tokens before block rounding; wider analytical KV
defaults are rejected for this lane.

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
  --expert-max-live-kv-tokens 131072 \
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

The slab, hysteresis, and interval values above are the defaults, but spelling
them out makes benchmark identity unambiguous. Startup rejects an incomplete or
contradictory combination before model weights are loaded. A JSON expert config
cannot activate dynamic slabs without the separate serving opt-in.

## Operator contract

This serving lane is target-only autoregressive generation. SSD expert
streaming intentionally sets `load_mtp=False` and `generation_mode=ar`; MTP
cache support in the lower-level runtime does not enable MTP for this server
configuration.

The 131,072-token limit is total logical context (rendered prompt plus requested
output), not a prefill-only limit. Attention remains full-history. The lane does
not use sliding-window attention, spill KV to SSD, drop old KV, batch a second
sequence, or retain SessionBank snapshots/live KV references for a request. The
existing single-sequence admission gate returns HTTP 429 while another request
owns the lane.

Active KV, resident model weights, current-demand experts, and pinned/in-flight
experts are protected. Before physical Q4 KV growth, the broker must account for
the rounded block allocation and prove enough expert-slab and allocator-cache
memory was reclaimed. Missing or contradictory allocator telemetry fails
closed; memory pressure, swap, compression, allocation failure, and process
termination are not resize signals.

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

The lane remains experimental and off by default until the allocator-release,
correctness, memory-safety, and paired 4K/32K/64K/128K hardware gates in issue
#46 pass. Do not treat successful startup or a single `/health` sample as that
hardware acceptance result.
