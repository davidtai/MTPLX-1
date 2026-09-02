# K-P1 phase 2: removing the draft to target host sync

Design note only. Nothing here is built. Phase 1 is
`MTPLX_FABLE_PLE_CANDIDATE_PREFETCH` (`mtplx/ple_candidate_prefetch.py`),
which hides the *row read*; phase 2 is what would let the *sync* go.

## What the sync is

The fixed-M4 verify graph takes a `[1, 4, ple_embed_dim]` PLE auxiliary as an
explicit compiled input. It is built on the host by
`_FixedM4SidecarAux.__call__`:

```
host_input_ids  ->  _ngram_rows_np  ->  [1, 4, 16] table row ids
                ->  sidecar.gather_np  ->  [1, 4, ple_embed_dim]
```

`host_input_ids` is `[primary, d1, d2, d3]`. The row ids are a *hash* of the
token ids, so the graph cannot be handed the tokens and left to find its own
rows: the table is 32 GB of SSD-backed memmap and the gather is a host read.
That is the structural sync the ledger calls "PLE aux needs host token ids" —
the draft chain must land on the host before the verifier can be enqueued, on
every lane, including the PR391 joint-D3 core (which otherwise syncs nothing
but its packed token vector).

## The phase-2 shape

Phase 1 already proves the only fact phase 2 needs: **at each depth the 20
candidate ids are known before the token is sampled, and the 16 rows for each
are exactly the rows that window position will use** (`candidate_rows`, and
`test_candidate_rows_equal_the_window_rows_for_every_candidate`). So the host
can build, per cycle, a dense candidate tensor

```
    ple_candidates : [4, 20, ple_embed_dim]     (~4 x 20 x 16 x 100 B = 128 KB)
    ple_support    : [4, 20] uint32             the candidate ids, in K20 order
```

and hand *both* to the verify graph. Inside the graph the aux becomes

```
    sel  = argmax(ple_support == sampled_ids[:, None], axis=-1)   # [4]
    aux  = take_along_axis(ple_candidates, sel, axis=1)           # [1, 4, D]
```

where `sampled_ids` is the device token vector the draft core already
produces (`result[0]` of `_pr391_run_float32_d3_core`) and never leaves the
device. The host then no longer needs the sampled tokens to build the aux, so
the `mx.eval(result[0])` between the draft chain and the verifier can move
after the verify enqueue — or go entirely, if the accept bookkeeping is also
restructured.

## What has to be true for it to work

1. **The candidates must reach the host one depth early.** On the retained
   PR391 joint-D3 core they do not: the chain's `raw_ids_by_depth` is built
   and consumed inside one compiled graph and only `result[0]` is evaluated.
   Phase 2 therefore requires the D3 core to *also* return its K20 support as
   a graph output that the aux builder consumes — which is a device-resident
   array, so the host would still have to read it to gather rows. That is the
   real blocker, and it is circular: the rows are a host read keyed by a
   device-computed hash.

   The way out is to move the *hash*, not the gather: `_ngram_rows_np` is
   ~10 integer ops per head over the token ids, so the graph can compute the
   `[4, 20, 16]` row ids itself. The host then reads the K20 *token* ids once
   (which the draft chain already has to expose for the accept bookkeeping's
   `draft_probs`) and gathers 1,280 rows. That is the same sync count as
   today, one depth earlier. **A sync one depth earlier is the whole phase-2
   win** — it is not a sync removal, it is a sync relocation, and the report's
   −0.3..−0.6 ms/window is priced on the relocation, not on removal.

2. **The 128 KB tensor has to be cheaper than what it replaces.** It is a
   compiled-graph input that changes every cycle, so it is a 128 KB host->device
   upload per cycle against the 6.4 KB one it replaces. At ~12 GB/s that is
   ~10 us — under the win, but not free, and it is 20x the graph's current aux
   input. Measure it as its own arm before committing.

3. **The selection must be exact.** `sampled_ids[d]` is by construction one of
   `ple_support[d]` (it was sampled from that support), so `argmax(==)` finds
   it. But the *spill* fallback in `_device_serial_support_arrays` can produce
   a support the device selector did not, and the context-copy streak
   substitutes a token that was never drawn from any support at all. Both must
   route to a host-built aux, or the graph must carry a
   `found ? candidate_row : gathered_row` select with a host-supplied
   fallback slot. Phase 1's all-or-nothing `resolve` is the same guarantee in
   the cheap direction and should be kept as the model.

## Ordering

Do not start phase 2 before phase 1's ABBA reads. If the boundary gap that
report M §A'.3 prices at ≤0.5-1 ms/cycle is not there on the retained stack
(instrument I2 is what decides), the sync relocation has nothing to buy and
the 128 KB upload is a straight cost.
