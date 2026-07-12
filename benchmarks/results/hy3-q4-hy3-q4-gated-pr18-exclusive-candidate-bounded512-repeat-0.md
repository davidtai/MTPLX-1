# Engineering Review: Serving a Very Large MoE LLM on Apple Silicon with NVMe-Backed Routed Experts

## 1. System Overview and Execution Sequence

The proposed design serves a very large mixture-of-experts (MoE) language model on an Apple Silicon workstation. The router, attention layers, embeddings, normalization layers, and shared experts are resident in unified memory (DRAM mapped into the Apple Silicon memory controller). Routed expert weights are too large to fit entirely in unified memory and are stored as affine 4-bit tensors on a local NVMe SSD. At each sparse layer, the router selects eight experts. On a cache miss, selected experts are loaded from NVMe into a fixed, user-bounded memory bank (a reserved region of unified memory). Decode requests populate a frequency-decayed hot cache; prompt prefill uses transient slots that are isolated from the long-lived decode working set.

A single forward pass executes as follows:

1. Token embeddings and positional encodings are read from unified memory.
2. For each transformer block:
   a. Attention, normalization, and shared expert compute execute from unified memory.
   b. The router scores tokens and selects the top-eight experts for that layer.
   c. The scheduler checks the expert cache. Hits are computed in place. Misses are queued for NVMe load into the bounded memory bank.
   d. Once all eight experts are resident, expert compute executes; outputs are combined via the router weights.
3. The final norm and language-head projection run from unified memory.

Because Apple Silicon uses a unified memory architecture, there is no explicit host-to-device copy; "loading into the memory bank" means mapping or copying the affine 4-bit tensor into a reserved virtual region and ensuring it is not paged out (via `mlock` or equivalent advisory locking where permitted).

## 2. Concurrency and Slot Pinning

The bounded memory bank should be partitioned into fixed-size slots, each sized to the largest routed expert (or to a quantized multiple). Slot assignment must be deterministic per layer and expert ID to avoid duplicate loads. We recommend:

- A per-layer slot table mapping `expert_id -> slot_index`.
- A read-write lock per slot: shared for compute, exclusive for load/evict.
- A load coalescing queue: if multiple tokens in a batch miss the same expert, only one NVMe read is issued; subsequent waiters block on a slot-ready futex.
- Pref
