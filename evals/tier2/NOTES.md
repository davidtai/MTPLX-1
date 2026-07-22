# Tier-2 — WikiText-2 perplexity, shipped hy3 q2 vs q4

Run 2026-07-20 in a guarded GPU window (`run_with_qwen_stopped.py` → `compare_streamed_quality.py`).
Output: `ppl_shipped_q2_vs_q4_wikitext2.json`.

## Result — FAIL (severe)

| Lane | model_key | root | PPL | mean_nll |
|------|-----------|------|-----|----------|
| q4 (control) | `hy3-expert-only-q4` | `hy3-expert-only-mlx-q4` | **2.860** | 1.051 |
| q2 (shipped) | `hy3-expert-q2` | `hy3-expert-only-mlx-q2` | **6.747** | 1.909 |

- **relative_perplexity_regression = 1.359 (135.9%)** vs the 5% gate → **FAIL**.
- NLL gap = 0.858 nats. Greedy token agreement = **3.7%** (19/512), first divergence at token 0–6 on all 4 prompts.
- Both lanes finite, `errors=[]`, full `token_count=4095/4096`, distinct roots/keys → genuine two-model comparison, **not** control-vs-control.

## Validity

- Load is **spec-correct**: `HY3_EXPERT_Q2` (`mtplx/expert_streaming_models.py:363`) = quant_bits 2, group_size 64, source `local/hy3-expert-only-mlx-q4` — matches the shipped bank and Tier-1's measured quant. Not a mis-load.
- Consistent with **Tier-1**: median rel-Frobenius error 0.415 (41.5% per-tensor) predicts catastrophic PPL; the 0.858-nat gap fits.
- `EXIT=2` came from the **wrapper's qwen-restore losing the flock race** to the next lane (`51-freq-alloc`), NOT from the comparison, which ran clean to completion and wrote this full JSON. qwen being down afterward = the other lane's window, not this run.

## Reconciliation gap (open)

This 135.9% is ~9× the **15.43%** recorded in `hy3-q2-never-quality-validated.md` for the same bank (that was a 0.143-nat gap). The `-direct` variant scored **125.7%** on this same `wiki.test.raw` corpus. So two independent runs + Tier-1 all agree the bank is severely degraded here, and **15.43% does not reproduce on this config** — likely a different corpus/chunking. A config-matched re-run is needed to explain the 15.43% figure; the *direction* (well over the 5% bar) is robust either way.

## Config used

`--memory-limit 103GiB --expert-cache-limit 60GiB --runtime-reserve 16GiB --max-live-kv-tokens 8192 --evaluation-tokens 4096 --chunk-tokens 64 --greedy-max-tokens 128`, corpus `mlx-kld/corpora/wiki.test.raw`, `--cache-policy frequency --cache-scope layer --slot-layout direct-slots`, full per-record hash verification. Pre-flight fit: q4 94.24 / q2 94.72 GiB of 103, `fits_fixed=True`, ~8–9 GiB headroom, memory knob never raised.

---

# Tier-2 — HumanEval via LiteLLM (2026-07-20): pass@1=0 is a HARNESS ARTIFACT

The guarded run went fully end-to-end — proxy served, `code_eval_gate` connected,
requests reached the handler — but **every request 500'd before any generation**:
`ModuleNotFoundError: No module named 'mtplx.benchmarks.resource_telemetry'`.
So `humaneval_hy3_q2.json` pass@1=0.0 (20/20 `request_error`) says **nothing about
the model** — zero tokens were generated.

Root cause: the campaign venv ships a **PEP660 editable install of `mtplx 2.0.2`**
whose `sys.meta_path` finder points at the stale primary checkout
`mtplx-hy3-ssd/mtplx` (branch codex/…), which lacks `benchmarks/resource_telemetry.py`.
That finder wins over `sys.path`, so the handler's `import mtplx.*` (via
`benchmark_q2_mtp_depth_matrix.py`) resolved to the stale worktree.

Fix (handler.py `_import_benchmark_module`, GPU-free-verified under the campaign
python where the finder is active): drop the mtplx package finder (keep the
native-extension finder), force the eval worktree to the front of `sys.path`,
evict stale cached `mtplx`. Verified: `mtplx` → eval worktree, `resource_telemetry`
imports, native ext still loads.

NOT re-run end-to-end: the box kernel-panicked at 23:10 (GLM freq-alloc buffered
arm — a different lane, not this run), and HumanEval is expected ≈0 from the
Tier-2 PPL (135.9% regression, 3.7% token agreement). The LiteLLM serving path is
proven functional GPU-free except the model-load+generate call itself.

---

# Tier-2 — HumanEval COMPLETED (2026-07-20 23:49): pass@1 = 0.80 (16/20) — "expect ≈0" was WRONG

Two further guarded attempts after the harness-artifact fix:

1. **23:41 attempt — Metal GPU watchdog timeout** during the cold ~90 GiB
   island-fill load on the freshly-panicked box
   (`kIOGPUCommandBufferCallbackErrorTimeout`, proxy SIGABRT). Archived as
   `humaneval_hy3_q2.gpu-timeout.json`. No generation occurred.
2. **23:47 warm retry — CLEAN END-TO-END SUCCESS.** Page cache warm from the
   aborted load; first request (load + generate) 58.9 s, run total 134.8 s
   wall. `EXIT=0`, flock released, qwen restored. No island-count reduction
   needed (champion config as baked).

## Result

- **pass@1 = 0.80 (16/20)**, HumanEvalPlus-v0.1.10 first 20 tasks, 1 sample/task,
  chat endpoint, temp 0.0, seed 42, max_tokens 1024, `no_think`.
- All 20 completions finished at natural `stop`: 25–152 completion tokens
  (median 79, total 1619). No truncation, no empties, no request errors.
- Failures (4): `HumanEval/6` NameError `current`; `/7` NameError `strings`;
  `/10` NameError `is_palindrome` (the prompt-provided helper — possibly a
  chat-extraction artifact dropping prompt context, which would only UNDERcount
  passes); `/12` AssertionError. These are coherent code-shaped mistakes,
  not gibberish.
- Provenance: model `hy3-q2` via LiteLLM proxy :18183, handler default root
  `~/.cache/huggingface/hy3-expert-only-mlx-q2` (no env override in
  `serve_and_eval.sh` — verified), mtplx 2.0.2, dataset sha256 `42526ec0…`.

## Interpretation — revises the lane's headline

The PPL FAIL (135.9% rel regression, 3.7% greedy token agreement vs q4) is
real, but the inference "therefore pass@1 ≈ 0" was wrong: PPL 6.747 absolute
is degraded-but-functional, not gibberish. The bank writes mostly-correct
short Python. Token agreement measures divergence from q4's exact path, not
task competence.

Caveats: n=20 (95% CI on 0.80 is roughly [0.58, 0.93]); first-20 task IDs,
not a random sample; NO q4/bf16 HumanEval baseline exists on this harness
(memory: no published Hy3 HumanEval anchor either) — so the DELTA attributable
to q2 quantization is unmeasured. The 5% quality gate verdict stays FAIL on
PPL; HumanEval says the failure mode is mild-on-code, not catastrophic.

Completions are NOT persisted by `code_eval_gate.py` (rows carry status/usage
only) — capturing sample generations for inspection needs either a gate flag
or a one-off request in a future GPU window.

---

# oQ2e campaign (2026-07-21): mlx-community/Hy3-oQ2e on the mtplx runtime

David's directive: serve as close to stock as possible, requantize nothing.
Integration: spec `hy3-expert-oq2e` (2-bit gs128 experts, q8-gs64 residents via
the config-driven pre-quantized loader path, NO proj-quant), manifest into the
stock shards, experts.bin sidecar (byte-identical extraction, sha
c72fb8c0…). MTP head = our external bf16 layer-80 artifacts (same base
revision).

Local checkpoint corrections (originals kept as *.orig-published):
- index total_size 89,871,151,524 -> 89,870,806,272 (publisher metadata
  overstates true header inventory by 345,252 B; fail-closed check caught it).
- config num_nextn_predict_layers 0 -> 1 (quantizer stripped MTP tensors and
  rewrote the count; base tencent/Hy3@716aa724 declares 1 — restoring the
  architecture-true value lets the trained external head attach).

Script fix en route: depth-matrix `_requests_from_args` binary hy3/glm dispatch
sent any new campaign key down the GLM branch (crossed MTP path). Fixed +
regression tests (71a9f03).

## Decode results (1024/1024, guarded windows, zero requantization)

| envelope | AR | K2 | K3 | notes |
|---|---|---|---|---|
| 96 GiB (limit 103, islands 79, fully resident) | 30.17 | **36.82**† | **31.81** | 0 misses; load peak 90.7 GiB, hard peak 91.9 |
| 88 GiB (limit 95, islands 74 + 5 streamed) | 22.35 | 25.93† | 22.15 | hit rate .26-.33, 101-131 GiB read/cell; hard peak 87.7 |

† K2 shows ONE token-divergence event per envelope vs the AR reference (96:
token 893; both "unclassified" attribution). K3 is bit-exact both envelopes.
Conservative headlines are the K3 numbers. Acceptance identical across
envelopes (1.637@K2 / 2.024@K3) — a model property, unaffected by streaming.

Reference points: our shipped q2 champion 22.74 (K3, islands 60); oMLX
publishes 29.4 tok/s for this checkpoint on the same hardware class.
Fully-resident pre-flight at limit 95 fails by 4.21 GiB (fixed footprint) —
islands 74 is the honest 88 GiB configuration; the >30-at-88 goal is NOT met
by this bank on the current streaming stack (best exact: 22.15).

## Weight fidelity vs bf16 (evals/fidelity_oq2e_vs_bf16.py, CPU, committed)

Median cosine **0.9212** / rel-Frobenius median **0.397** on the exact Tier-1
sample — BEATS the shipped q2 bank (0.9148 / 0.415) at smaller size (2.44 vs
2.50 bpw). Per-projection medians uniform (~0.920-0.9215): the imatrix rescues
gate/up, which our plain affine recipe hurt most (they dipped to ~0.88).

## Open

- K2 divergence attribution (one flip/run; near-tie logit under batched
  verify is the suspect — unproven).
- gs128 disables the fixed-M4 island fast wave path (hardcoded 64; falls back
  to generic gather) — island decode has headroom if a gs128 variant lands.
- HumanEval on oQ2e via the LiteLLM lane (env switch, needs a window).

## oQ2e WikiText-2 PPL (2026-07-21 03:11, config-matched to Tier-2)

**oQ2e 6.443 vs q4 2.860 = 125.3% relative regression** (q4 control reproduced
Tier-2's 2.860 exactly; greedy agreement vs q4 3.9%, diverges at token 2).
Compare: shipped q2 135.9%, `-direct` bf16-derived q2 125.7% — three
independently derived 2-bit banks cluster at 6.4-6.7 on this config while all
evidence says task competence survives (q2 HumanEval 0.80; oQ2e published
benches ≈ its 2.68bpw sibling; oQ2e weight cosine 0.9212). Conclusion
sharpened: the 5%-PPL gate at this config is structurally unpassable at 2-bit
expert precision and mostly measures calibration loss, not task damage. Any
future bank verdicts need a task eval alongside PPL.

Ops notes: shard sha256 provenance is REQUIRED by compare_streamed_quality's
resident check — build manifests with `--hash-shards` (rebuilt + sidecar
--overwrite, byte-identical sha c72fb8c0…). Wrapper exits 1 on the qwen-restore
flock race even after writing full results — launchers must treat the output
artifact as the success signal (attempt-269 window was burned re-learning
this; attempt-1's q4-only receipt kept as *.attempt1-q4-control-only.json).

## oQ2e HumanEval (2026-07-21): pass@1 = 0.95 (19/20)

Same 20-task HumanEvalPlus gate as the shipped-q2 run (greedy, seed 42,
chat endpoint, no_think), served via LiteLLM with env overrides
(MTPLX_HY3_MODEL_KEY=hy3-expert-oq2e, islands 79 fully resident,
proj_quant=none). All completions natural-stop, 77-238 tokens (median 147 —
q2's was 79). Sole failure: HumanEval/10 AssertionError (make_palindrome —
the task q2 also failed). Provenance note: a silent fallback to the q2 bank
under these overrides is structurally impossible (bf16 residents unquantized
+ 79 q2-record islands ≈ 101 GiB > the 103 limit with KV/reserve — pre-flight
would reject); behavioral fingerprint (token profile, score) also differs.

Quality triple for oQ2e vs shipped q2: cosine 0.9212 vs 0.9148; PPL 6.443 vs
6.747 (both far over the 5% gate); HumanEval 0.95 vs 0.80. The bank-quality
ordering is consistent across all three; PPL magnitude remains the outlier
measure (calibration, not competence).

## Wave-port K3 re-measure (2026-07-21): NULL result under AR control

gs128 wave eligibility (489790b) re-measured at the champion envelope:
AR 28.92 / K3 30.54 (both bit-exact, hard peak 91.9 unchanged) vs pre-wave
AR 30.17 / K3 31.81. Raw drop ~4% in BOTH cells — but AR shares no wave code,
so it is the window-drift control: K3/AR ratio 1.0560 post vs 1.0544 pre =
+0.15%, sub-noise. **The wave port recovers ~nothing at K3.** Consistent with
the gather_qmm microbench (gs128 0.92x the time of gs64 at wave shapes —
committed as bench_gather_qmm_gs128.json): both paths call the same kernel;
the wave only restructures surrounding ops. Attribution revision: the q8
residents carry essentially the entire oq2e-vs-q2-champion decode deficit;
wave ineligibility was worth ~0. The port stays (bitwise-locked, correct,
extends to GLM dims) as eligibility hygiene, not as a perf claim.
Cross-window comparisons on this box carry ~4% drift under sustained load —
single-window paired arms only.

## K2 compile-island A/B (2026-07-21, single window, paired arms): NULL

base AR 30.54 / K2 38.00; MTPLX_HY3_COMPILE_ISLAND=1 AR 30.70 / K2 38.31.
K2/AR ratio 1.2443 vs 1.2479 = +0.3%, sub-noise. Compile-island is worth
nothing at full residency and is non-bitwise by design — verdict: leave OFF,
permanently, at this operating point. Third op-restructuring null in a row
(wave, compile, historic compile-the-forward): the gather kernel is the cost
and it is ALU-bound; surrounding-op work does not move it. K2's persistent
single-divergence signature reproduced in both arms (acc/verify 1.637
identical). Window itself ran ~3% faster than last night's (drift confirmed);
paired arms agreed internally.

## oQ2e per-component roofline (2026-07-21, MTPLX_ROOFLINE_PROFILE, K2 window)

Ceiling 502 GB/s measured. attention 84.4 MB/call @313 GB/s (62%) = 21.6
ms/tok DOMINANT; moe batch-1 gather 192 GB/s (38%, occupancy); router 5.5
ms/tok @5% ceiling (occupancy — future lever); shared 469 (93%, saturated);
32-assign wave 463 (92%, saturated). T0a: MoE inefficiency is batch=1-ONLY
(192 vs 463 GB/s at 32 assignments). Attribution: q8 ATTENTION owns the
4 tps deficit vs the q4-attention champion (2x bytes/call on the largest
component); proj-requant q8->q4 over proj_quant_covers scope is aimed
correctly — expected ~9-10 ms/tok raw before overlap. Instrumented tok/s
diagnostic-only.

## proj_requant quality gates (2026-07-21): BOTH PASS — candidate confirmed

Gate 1 HumanEval: 0.95 (19/20) identical to stock-q8 (same lone HumanEval/10
failure, same token profile). Gate 2 WikiText-2 (config-matched): requant
6.547 vs stock 6.443 = +1.6% relative (q4 control 2.8596 reproduced exactly,
third time). Both inside David's "don't lose much" bar; the requant arm still
beats shipped q2 (6.747) on PPL. Remaining: paired K2 speed A/B (stock-q8 vs
requant-q4) to price the win — roofline predicts ~9-10 ms/tok raw attention
savings. FULL-164 CONFIRMATION: see "proj_requant full-164 HumanEval" below —
the 20-task gate's saturation resolved, McNemar p=1.0, verdict unchanged.

## proj_requant speed A/B (2026-07-21, paired single window): +8.6% K2, +26.3% AR

stock-q8 AR 26.89 / K2 32.63 vs requant-q4 AR 33.97 / K2 35.44; hard peak
91.9 -> 88.4 GiB. Window globally slow (drift) — ratios are the evidence;
scaled to fast-window baselines the projections are ~41 K2 (above the 40.59
championship) and ~38 AR. Acceptance 1.637 -> 1.564 (trunk perturbation costs
the MTP head slightly; net K2 still +8.6%). Candidate scorecard complete:
HumanEval identical, wiki +1.6%, K2 +8.6%, AR +26.3%, -3.5 GiB wired.
Adoption as serving default = David's decision.

## Champion-41 reproduction (2026-07-21): K2 42.33 — NEW ALL-TIME CHAMPION

oq2e + proj-requant q4, 96 envelope, islands 79, championship shape (AR+K2):
AR 37.79 / **K2 42.33** (hard peak 88.4 GiB). Beats the q2-bank championship
40.59 by +4.3% on a smaller, higher-quality bank with quality gates passed
(HumanEval 0.95 identical / wiki +1.6%). Projection from the paired-ratio
scaling (~41) confirmed. Known caveats carry: K2 single-divergence signature
(parity False), acceptance 1.564.

## Router M1 scope A/B (2026-07-21 13:10, paired): +4.6% AR; K2 DIVERGENCE ATTRIBUTED

scope-mtp AR 30.60 / K2 37.96 (parity False) vs scope-all AR 32.01 / K2 38.15
(**parity True**). AR +4.6% as predicted (trunk M1 stock->kernel); K2 +0.5% =
control held. The persistent K2 single-token divergence is ATTRIBUTED: a
router-numerics mismatch between lanes (AR reference routed via the stock
host path at M1 while K2 routed via the fp32 split-K kernel; one near-tie
logit forks the sequence). Same kernel numerics both lanes -> parity
restored. Follow-up (one window, when the box frees): champion + requant +
MTPLX_HY3_ROUTER_SPLITK_M1=all — expect 42.33-class K2 with parity True.

## PARITY-STAMPED CHAMPION (2026-07-21): K2 42.18 / AR 40.18, parity True

oq2e + proj_requant q4 + islands 79 + MTPLX_HY3_ROUTER_SPLITK_M1=all at the
96 envelope: AR 40.18 (parity True) / K2 42.18 (parity True), hard peak 88.4
GiB. Reproduces the 42.33 champion within noise WITH the divergence resolved
(router-numerics attribution confirmed by construction: same kernel both
lanes -> exact parity). This is the complete champion config: every caveat
closed — quality gates passed, bit-exact, one envelope tier below the old
champion's memory.

## q4 anchor (2026-07-21): kernel EXONERATED; serving infeasibility QUANTIFIED

Kernel microbench (bench_gather_qmm_q4.json, wave shapes, queued lane):
2-bit/gs64 6.16 µs, 2-bit/gs128 5.99, **4-bit/gs64 7.79 = 2x the bytes for
1.26x the time — per byte the MOST efficient arm measured.** MLX's q4 gather
needs no custom work; every q2-lane optimization (wave, router kernels,
scope-all, proj-quant, cadence, census islands) applies to q4 as-is.

q4 TPS, 96 envelope (islands 42 — ran on the q2-order placement, arms
straddled the census revert): **AR 3.39 / K2 3.00 tok/s**, hit 8-9%,
2.6-2.7 TiB read/cell, hard peak 95.6 GiB. Placement-corrected numbers would
improve marginally; the binding cost is 37 streamed layers x 10.1 MiB
records. **q4 is ~13x slower than oq2e-requant (42.33/40.18) at the same
envelope** — the anchor row justifying the 2-bit program. 80-envelope arm
re-armed with census-explicit islands (its count-resolution exposed a
pre-flight gap for census-only specs — follow-up: resolve_island_placement
before the runtime.py pre-flight plan).

Ops: census-vs-spec precedence matters — q4's own island-placement.json
ranks a DIFFERENT layer set than the q2-derived order (David's catch);
spec pin orders must not shadow per-bank census artifacts.

## q4 anchor complete (2026-07-21): 80-envelope AR 2.74 / K2 2.60

Census islands 34 (own placement, census-first live), hard peak 79.8 GiB,
hit 4-5%, 3.4-3.6 TiB read/cell. Microbench re-run strengthened the kernel
verdict (4-bit 7.19 µs = 1.15x the 2-bit time for 2x bytes). Q4 GOAL CLOSED:
kernel exonerated twice, optimization parity complete, TPS anchored at 96
(3.39/3.00) and 80 (2.74/2.60) — q4 serving is 13-16x under the
oq2e-requant champion at comparable envelopes; the 2-bit program is the only
serveable path on this box.

## q4-OPTIMAL found (2026-07-21, one window, 3 ratio arms x AR/K1/K2/K3)

Same ~80 GiB expert budget, three arrangements (census islands + frequency
cache), full lever stack (proj-quant q4, headers-only, deferred+pin, 8MiB
chunks, cadence, scope-all):

| arm | AR | K1 | K2 | K3 | hit |
|---|---|---|---|---|---|
| island-heavy 40+4GiB | 3.90 | 3.92 | 3.17 | 2.58 | .14-.17 |
| balanced 28+26GiB | 5.02 | 5.18 | 4.61 | 3.73 | .33-.41 |
| **cache-heavy 12+57GiB** | 5.54 | **6.29** | 5.63 | 4.67 | .41-.51 |

VERDICTS: (1) On a bank that doesn't fit, CACHE BEATS ISLANDS per byte
(dynamic frequency adaptation > static census pinning) — inverts the q2-era
C5 conclusion, which held only because that bank nearly fit. (2) K1 > AR in
every arm; K2+ monotonically worse (verify batches amplify misses faster
than acceptance pays — reads grow 1.1->1.5 TiB/cell with depth). Best q4
operating point: cache-heavy K1 = 6.29 tok/s, above the 5-6.4 historical
band. Extrapolation: zero-island max-cache might add ~5%; diminishing.
FINAL ANCHOR: q4-optimal 6.29 vs oq2e-requant champion 42.18 = 6.7x — the
2-bit program's justification at q4's own best configuration.

## proj_requant full-164 HumanEval (2026-07-21, paired one window): NO MEASURABLE COST — McNemar p=1.0

The saturated 20-task gate resolved on HumanEvalPlus-164, both arms in ONE
guarded window (attempt 1, no retries; conditions identical to the 0.95
baselines — MTPLX_HY3_ROUTER_SPLITK_M1 deliberately NOT set):

- stock-q8 **0.8720** (143/164) vs proj-requant-q4 **0.8659** (142/164);
  request_errors 0 both arms.
- Discordant pairs **9 (5:4)** — q8-only passes {83, 102, 130, 147, 148},
  rq4-only passes {26, 32, 100, 108}. **McNemar exact two-sided p = 1.0000**:
  the split is exactly what coin-flip generation divergence predicts; no
  directional quality signal.
- 17 both-fail tasks incl. HumanEval/10 (the lone 20-task failure — screen
  consistent with the full run).
- All 5 rq4-only failures are complete generations (finish=stop) failing
  logic assertions (102/147/83 AssertionError, 130 IndexError, 148 planet-
  ordering) — wrong-answer flips, not truncation or harness artifacts.

VERDICT vs David's "don't lose much" bar: PASS — net −1 task (−0.6 pp),
symmetric flips, p=1.0. The requant candidate's scorecard is now complete at
full task-eval resolution: HumanEval −0.6pp (p=1.0), wiki +1.6%, K2 +8.6%,
AR +26.3%, −3.5 GiB wired. Artifacts:
`humaneval_oq2e_full164_{q8,rq4}.json` (+ per-arm proxy logs
`full164_{q8,rq4}_proxy.log`).

## R1 (per-layer frequency allocation) KILLED at the CPU gate (2026-07-21): the 8x table was leakage

David picked R1 from the q4->10 brainstorm; its zero-GPU kill-test ran the
route-analysis script's issue-#9 HELD-OUT gate (chronological train/eval
split) on the same trace behind the famous 8x claim
(`hy3-q4-route-trace-1k-64.json`, 8058 slots ~= today's 80 GiB budget):

| policy (held-out, split 0.5) | hit | miss/tok |
|---|---|---|
| uniform_per_layer_lru | .8881 | 70.72 |
| **trained_dynamic_quota_lru** | .8880 | **70.78** |
| global_pool_lru | .8917 | 68.47 |
| uniform_per_layer_belady | .9189 | 51.25 |

The deployable trained allocator captures ZERO (identical to uniform; same
at split 0.75: 88.00 vs 87.88). The rebalancer moved 15-18 slots but its own
TRAINING curves offered only +35-47 hits of ~18-27k (~0.2%) — no signal, not
insufficient data. Root cause measured directly on the prefill phase (8184
samples/layer): per-layer top-102 coverage median .862 (concentrated — 53.1%
if uniform) but CROSS-LAYER stdev just .0274 — every layer concentrates the
same amount, so per-layer slot counts have nothing to trade. The legacy
`oracle_decode_frequency_allocation` 8.5 miss/tok (the "8x, beats Belady"
headline in docs/MMAP_VS_PREAD_FINDINGS.md and the mmap memory) is a pure
evaluation-oracle artifact: on a 64-step trace it just pins whatever appears.

Belady ceiling RECOMPUTED (handoff ask): perfect eviction buys 20-28% fewer
misses than LRU at this budget (~25-35 ms/tok at 1.5 ms/miss) — the hard cap
on ALL cache-policy work, unreachable in practice; global pooling adds ~3%.
Receipt: `r1_alloc_killtest_heldout.json`. Consequence for the 10-tps
program: miss COUNT is near its practical floor at this budget — the live
levers are hiding miss latency (R3 overlap), cheapening service (R2
page-cache L2, R4 read_chunk), or batch amortization (R5). Also cautions
GLM W1: the hy3 half of its evidence base is now dead; GLM's own
frequency-vs-uniform A/B remains unmeasured.

## R2-band/R4 replay window (2026-07-21): R4 CONFIRMED ~13-15 ms/tok; band DEMOTED to ~5

Model-free pread replay against the real q4 sidecar (wired ~0, cache fill
19.9 GiB, receipt `r2r4_band_replay_receipt.json`; first launch self-crashed
6x on a watchdog RSS cap that miscounted clean mmap pages as anon memory —
fixed cap 30 GiB + launcher no longer retries watchdog aborts):

- **A1 cold single pread 10.1 MiB: 0.93 ms (10.8 GiB/s).** A2 production
  8 MiB-chunk mirror: 1.05 ms — the 2-preads-per-record tax is 0.13 ms/miss
  = **R4 ≈ 13-15 ms/token for a read_chunk >= record-size change.**
- **B1 warm band via mmap 0.66 ms vs deep tail 0.91 ms alongside it** (no
  interference; band 200/200 resident after replay). Save 0.24 ms/hit x ~20
  band-hits/tok = **~5 ms/token — band DEMOTED** (the ~22 estimate assumed
  1.5 ms cold reads; raw cold is 0.91). Only worth revisiting as zero-copy
  serving that also skips per-miss runtime overhead.
- **Key discovery: ~0.6 ms/miss (~60 ms/token) of NON-READ miss overhead** —
  production wall budget 1.5 ms/miss vs 0.91 raw read, corroborated by raw
  read time 0.98 GiB/tok / 10.8 GiB/s = 91 ms of the 159 ms step. This pool
  (admission/slot/host-sync + serialization) is R3's target; fully-hidden
  reads ceiling ~11 tok/s.
- Cold band populate: 19.9 GiB in 2.1 s (sorted offsets ~sequential);
  warm sweeps 33-40 GiB/s.

Program re-rank: R4 (trivial, confirmed) -> R3 decomposition (hideable vs
deletable split of the 60 ms pool) -> R5 batch (aggregate only) -> band
(parked) . R1/page-cache-LRU remain dead. Awaiting David's pick.

## R4 INVERTED by paired A/B (2026-07-21): chunk16 LOSES −2.1% K1; chunking IS the fanout

One window, receipt-exact cache-heavy config, deterministic-identical routing
across arms (103.4 miss/tok, 0.975 GiB/tok both): chunk8 AR 5.709 / K1 6.270
(champion 6.29 reproduced) vs chunk16 AR 5.635 / K1 6.136 (−1.3%/−2.1%).
The replay's +13 ms/tok prediction for single-pread records was an artifact
of SERIAL microbenching: telemetry (r3_telemetry_chunk16.json, diagnostic
cell −10% overhead) shows the live reader averages **1.15 concurrent reads**
(8-reader pool 14% utilized), I/O active 100% of intervals, realized 5.38
GiB/s vs 10.8 isolated single-stream. Decode misses arrive ~1.5/layer
serially — record-chunk splitting is the ONLY queue depth the read path
gets, so 2 chunks/record beats 1. Attribution field:
'synchronous_fence_or_evaluation' (incomplete); GPU/DRAM counters
unavailable in this lane.

Verdict: read_chunk >= record is DEAD; the live lever is chunk-DESCENT
(more intra-record fanout). Sweep armed: chunk8 control vs 4MiB vs 2MiB,
same window discipline. If realized BW climbs toward the ~12.5 GiB/s
plateau, the ~91 ms/tok I/O share compresses toward ~65-75 ms.
Artifacts: r4_ab_chunk8.json / r4_ab_chunk16.json / r3_telemetry_chunk16.json.
Lesson (method): I/O microbenches must mirror the production SUBMISSION
model (concurrency), not just the syscall pattern.

## Chunk-descent sweep (2026-07-21): FLAT — the chunk lever is exhausted at ~2% total span

Paired one-window (identical routing all arms): chunk8 AR 5.611 / K1 6.282;
chunk4 5.678 / **6.323** (+0.65%); chunk2 5.713 / 6.322. Descent past 4 MiB
buys nothing; the whole read_chunk knob spans 16MiB 6.136 -> 4MiB 6.323
(~2%, chunk4 marginal best). Leans T2 (null-at-IO-layer): live realized read
bandwidth barely responds to chunk shape, so the dynamic-splitter idea is
LOW-EV at K1 (David's test-the-theory-first call — validated before any
build). Discriminating evidence queued in one window: HOL microbench through
the PRODUCTION reader (QD x chunk grid, predictions pre-registered in the
receipt), 2x5.06 MiB even-split arm, and David's pure-mmap probe
(slot_layout metal-mmap, no islands — "islands only work when the bank
nearly fits" — 256-token order-of-magnitude cell, non-gating).
Artifacts: r4_sweep_chunk{8,4,2}.json.

## I/O layer CLOSED (2026-07-21 eve): drive saturates at QD2; the lever is submission, not shape

**HOL theory test** (production PositionalExpertReader, native backend,
F_NOCACHE, predictions pre-registered; receipt hol_theory_test.json):
QD1 whole 0.951 ms (serial chunk8 tax +0.15); at QD2 ALL shapes — whole /
chunk8 / even-halves / even-thirds — within ±2.5% at the aggregate CEILING
12.7-12.8 GiB/s; QD3 same. T2 (null-at-IO-layer) CONFIRMED: no quantum/HOL
effect worth engineering; **two concurrent 10 MiB preads saturate this SSD.**
Live even-split serving A/B agrees: 2x5.06 MiB 6.302 vs chunk8 6.304 (dead
even; r4_even_*.json).

**NVMe research (sonnet web agent, cited in #130):** Apple ANS2/ANS3 exposes
ONE I/O queue (linear submission, per Linux nvme-apple) — no hardware
multi-queue to unlock; 4 independent Apple-silicon MoE-streaming projects
(ds4, hypura, mac-code, SwiftLM) all converged on N-thread sync pread +
F_NOCACHE; MTLIO/dispatch_io/aio: no evidence, do-not-port (MTLIO's GPU-
timeline event gating unmeasured, parked as a 50-line curiosity). F_NOCACHE
is a HINT needing 16 KiB alignment — our offsets/chunks are exact multiples.

**Synthesis:** live decode realizes ~5.4-6.1 GiB/s at mean 1.15 in-flight
reads — between QD1 (10.4) and QD2 (12.7) — so the entire remaining I/O gap
is SUBMISSION-SIDE serialization (the layer loop waits on 1-2 misses), not
read shape, pool size, chunking, or API. read_chunk verdict: keep 8MiB (4MiB
+0.65% marginal, within pairing noise; 16MiB genuinely worse at QD1-ish
depths). The 10-tps program's remaining levers: R3 overlap/host-overhead
(~60 ms/tok pool), batch (R5), fewer bytes (2-bit program).

**mmap probe:** first launch rejected at configuration — metal-mmap requires
--expert-integrity at-open (no per-record hashing on the mapped path);
relaunched with at-open (one-time full-sidecar hash ~15 min). Probe pending.

## mmap probe outcome (2026-07-21): metal-mmap is UNBENCHMARKABLE in this lane today

With at-open integrity the config gate passed; the run then failed closed at
WARMUP: "hy3-q4 d0 decode expert-cache has no routed assignments" — all
expert_streaming_counters zero. Load peak 13.9 GiB (trunk-only; the
mapped-store unwired accounting works). The mapped path executes outside the
expert-cache counters this lane hard-gates on, so pure-mmap serving cannot
produce a valid cell without integration work (wire counters/gates through
MappedExpertStore). Given streamed-mmap physics is independently condemned
(demand-fault 1.4 GiB/s flat; kernel-LRU = worst simulated policy; pread
12.9 GiB/s), further investment is low-EV — David's call whether the
integration is worth doing just to close the measurement.
Artifact: q4_mmap_probe.json (failure record).

## R3 PRICED (2026-07-21 late): 159 ms = ~100 read-wait + ~37 GPU + ~22 host; 10 tok/s is REACHABLE

Instruments: per-layer replay (r3_perlayer_replay.json — production reader,
production per-layer submission; 48 steps, 8173 misses) + roofline-profiled
K1 cell (r3_roofline_cell.{json,log}, d1 6.229 diagnostic ~= headline 6.28).
Caveat: oQ4e download ran concurrently (~1% drive BW; singles 1.06 ms vs
0.93 clean — direction unaffected).

DECOMPOSITION of the 159 ms K1 step:
- **Read-wait ~100 ms** (replay: 165.7 ms/tok at 170 synthetic miss/tok,
  scaled x103.4/170.3). Group walls scale LINEARLY with miss count
  (k=1 1.16 / k=2 2.01 / k=3 2.92 / k=4 3.73 — ~0.97 ms per extra read).
- **GPU ~37 ms**: attention 15.3 (46.7 MB/call, 45% of 539 GB/s ceiling),
  routed experts ~14 (est. via shared-expert calibration 476 GB/s), router
  4.3 (compute/occupancy-bound), shared 1.8 (88% ceiling), MTP head ~2.
- **Host residual ~22 ms** (admission/bookkeeping/dispatch).

KEY MECHANISM — BURST-READ SERIALIZATION (new, reproducible): a burst of k
reads submitted together through the warm production executor completes in
~sum time (pair 1.90 ms, triple 2.71) not ~max time (1.65/2.5 ideal), while
steady-state looping workers achieve true bandwidth-sharing (HOL qd2 1.548
ms/record at 12.7 GiB/s aggregate). Fresh-thread spawn exonerated (0.06 ms).
The live 1.15 mean-active-readers despite 1.5-miss groups is this effect.
Suspects for the fix: chunk-loop GIL crossings on the burst path, or route
misses down the EXISTING batch scatter-preadv path (expert_banked) = ONE
native call per layer group, kernel-level concurrency, no GIL.

LEVER PRICES (stacking):
- **(A) Fix burst concurrency**: group walls sum->max ~= read-wait 100 ->
  ~50-60 ms => ~110-120 ms/tok = **8.4-9.1 tok/s**. Software artifact, not
  hardware (drive+reader both proven QD2-capable).
- **(B) Resident-first same-layer overlap** (David's design; canonical
  accumulation order preserves bit-exactness): hide min(GPU ~0.5 ms/layer,
  read/layer) => additional ~15-25 ms => **~10-11 tok/s** after (A).
- (C) Early submission at route time: few ms, folds into (A)/(B).
VERDICT: q4->10 is priced as reachable via (A)+(B). No implementation
without David's pick.

## CORRECTION (2026-07-21 ~21:50): fix-A re-priced ~9 ms (was 40-50); (B) overlap carries the program

Quiet-disk burst re-probe (sidecar build finished, no background traffic):
single 1.110 ms; bursts k=2/3/4 = 1.880/2.698/3.523 ms vs concurrent-ideal
1.56/2.34/3.11 and serial 2.22/3.33/4.44 — **51/64/69% overlap efficiency
already present**. The earlier "sum-not-max" serialization read used a
download-contaminated single baseline; the R1 lesson (verify before build)
caught it pre-implementation. GIL exonerated separately (spinner probe:
native read releases it).

Re-priced against the REAL-trace burst distribution (per token: 22.0 k=1 /
14.6 k=2 / 6.6 k=3 / 2.4 k=4 / 1.2 k>=5 groups): perfect batching recovers
~9 ms/token. Read-wait stands at ~83-99 ms/token but is mostly IRREDUCIBLE
read time near the drive ceiling, not serialization.

REVISED PROGRAM: **(B) resident-first same-layer overlap is the primary
lever** — hide reads behind ~37 GPU + ~22 host ms => floor max(read, rest)
≈ 100 ms => ~10 tok/s. (A) batched non-blocking group reads DEMOTED to
(B)'s foundation (~9 ms standalone; required so miss reads stop blocking
the layer loop). Design: submit layer's misses at route time (batch
scatter-preadv, non-blocking) -> compute resident experts + shared ->
accumulate partials in canonical expert order on arrival (bit-exactness
preserved) -> parity-gated paired A/B.
Also: oQ4e sidecar COMPLETE (experts.bin 161,036,107,776 B — byte-identical
geometry to old q4; manifest w/ record hashes; spec + benchmark registered
5df7d7d). Eval ladder pending a window.

## FIX (B) IMPLEMENTED (2026-07-21): --moe-overlap = single batched miss part + run-coalesced reads + measured overlap telemetry (commit 9865f0b)

Design per David's pick (resident-first same-layer overlap; batched group
reads as its foundation). Config-gated: `overlap_miss_reads` /
`--moe-overlap`, default OFF; OFF leaves the per-expert split-part decode
path untouched (behavior-locked by test). Requires component-banks.

WHAT CHANGED (ON):
- `begin_split_route` submits a decode layer's misses as ONE part -> one
  future, one admission pass ahead of the wait, one policy commit, one
  miss-gather dispatch. (Same part shape prefill always used; only the
  phase gate is new.) Per-expert parts each paid their own executor task,
  lifecycle claim, admission, pin pass, ReadyRoute, and per-part
  iter/claim/commit bookkeeping — the ~0.6 ms/miss non-read overhead pool
  (~60 ms/token at 103 miss/tok) this lever targets.
- Pool partitions the part's loads into sidecar-adjacency runs: run >= 2
  -> one scatter preadv batch (`_fill_batch`); scattered records keep
  their own reads on the 8-worker pool, so batching never serializes
  non-adjacent misses. (Near-uniform routing makes adjacent same-layer
  miss pairs rare — the concurrency-preserving fallback is the common
  case; coalescing is opportunistic.)
- Hit + shared dispatch before the miss wait was already in place under
  split-route-release=deferred; the availability split and canonical
  accumulation were already partition-invariant — LOCKED by a new
  bitwise test: gather_qmm subset dispatch + position-order reassembly
  reproduces the fused 8-expert wave BIT-EXACTLY on real shapes (hidden
  4096, expert_hidden 1536, gs64 affine q4; partitions 8/0, 6/2, 0/8,
  interleaved 3/5; run-to-run deterministic). The [rows,1,K] gather
  calling-convention shape is asserted inside the test harness.
- Telemetry (the acceptance counter, measured, never tok/s-inferred):
  slot metrics `batched_miss_parts`/`batched_miss_records` prove the
  batch path executed; `overlap_split_routes`,
  `overlap_gpu_dispatch_ns` (host ns of GPU work built+submitted while a
  miss read was open), `overlap_exposed_wait_ns` (residual blocking wait
  after dispatch = the read-wait the overlap could NOT hide). Every
  benchmark observation row carries them as `overlap_telemetry`.

TDD EVIDENCE (tests/test_expert_overlap_split.py, 9 tests, all green):
batch == N single reads bitwise via per-record sha256 verification on a
sidecar-backed component-banks runtime; adjacency coalescing observed via
reader read_operations (adjacent [1,2,3] -> 1 op, scattered [5,7] -> 2);
short-read and pre-set-cancellation FAIL CLOSED; knob OFF keeps 4
per-expert parts and zero overlap counters; telemetry counters recorded
under a slowed reader with a warm hit + cold miss route. Benchmark flag
test: --moe-overlap -> runtime config -> per-row overlap_telemetry.

FULL SUITE: touched-area files 309 passed. Full tests/ shows 2
pre-existing test_settings_audit failures (MTPLX_HY3_ROUTER_SPLITK_M1
unclassified — fails identically at HEAD b9c28c3, another lane's env
var) plus sdpa_gqa_packed/streamed_rans failures that appear ONLY under
the full-suite run and pass in isolation both at HEAD and with this
change (suite-order/GPU-state flakes, statically untouched by this diff;
second full-suite run pending as corroboration).

Acceptance window (paired, one window, champion cache-heavy config,
armA control OFF expect ~6.28 / armB ON, gates + telemetry split): NEXT.

## FIX (B) ACCEPTANCE (2026-07-21 22:28-22:43, one paired window): armB K1 5.86 vs armA 6.25 — BELOW THE 7.0 STOP LINE; telemetry shows the overlap window is structurally starved

Window: run_with_qwen_stopped, armA (overlap OFF, control) then armB
(--moe-overlap), champion cache-heavy config, MTPLX_SUSTAINED_PREFILL=1
MTPLX_HY3_SUBMIT_CADENCE=8. Artifacts overlap_ab_{off,on}.json; gates
14/14 true on every row, both arms; deterministic tokens.

| arm | AR tok/s | K1 tok/s | K1 ms/tok |
|---|---|---|---|
| A control (OFF) | 5.541 | **6.253** | 159.9 |
| B overlap (ON)  | 5.279 | **5.857** | 170.7 |

armA sits in the receipt band (6.27-6.30 cross-window; paired control
stands). armB is **-6.3% at K1, -4.7% at AR** — the mechanism not only
missed the 8.5-10 zone, it is net negative. STOPPING per the brief; no
tuning. Decomposition below is the deliverable.

MEASURED OVERLAP TELEMETRY (armB; pool counters accumulate across rows —
reset() clears cache counters, not slot-pool metrics — so K1 = d1-d0
delta; the per-token record volumes 99.3/99.5 match the 103.4 miss/tok
receipt, validating the instrument):
- AR row: 51.2 batched routes/tok, 99.3 records/tok;
  **overlap_gpu_dispatch 9.5 ms/tok; overlap_exposed_wait 97.4 ms/tok.**
- K1 row: 53.3 routes/tok, 99.5 records/tok;
  **overlap_gpu_dispatch 15.2 ms/tok; overlap_exposed_wait 88.8 ms/tok.**
- Batch path executed on every decode split route (routes == parts);
  knob-OFF arm counters all zero.

READING:
1. The mechanism works as built — every miss group went down as one
   batched part and hit/shared GPU work was dispatched inside the open
   read window (9.5-15.2 ms/tok of it).
2. It cannot hide the reads: exposed read-wait stays at 88.8-97.4
   ms/tok, i.e. the stock 83-99 ms band. The reason is structural, now
   measured rather than assumed: after the router resolves, the ONLY
   work not data-dependent on the layer's misses is that same layer's
   hit gather + shared branch (+ graph build) — the 9.5-15.2 ms/tok.
   Layer L+1's attention/router cannot enter the window (they need L's
   combine, which needs the misses). The Σ max(read_L, other_L) pricing
   assumed the full ~37 GPU + ~22 host ms/tok could slide under reads;
   the dependency structure admits ~1/6 of it.
3. The batching itself costs ~9-11 ms/tok net vs the per-expert path.
   Unmeasured split between suspects: (a) single-task serial admission
   delays first-read submission whenever an early load waits on a
   slot/pin (old path: k concurrent admit+read tasks); (b) the miss
   gather now dispatches after the SLOWEST record of the group instead
   of per-part as-ready; (c) fewer per-part async_eval submissions.

PROGRAM CONSEQUENCE: same-layer overlap is a dead lever at ~1-2
misses/layer-visit — the read window dwarfs the dispatchable work.
Routes to 10 tok/s must change the dependency structure or the byte
volume: fewer bytes per miss (the 2-bit program), more positions per
layer-visit (deeper verify batches raise other_L), or cross-layer
speculation (prefetch program: KILLED separately). The exposed-wait
counter stays available (knob ON) for any future lever to be judged
against.

Knob remains default OFF; the stock path is untouched and the in-window
control confirms it unchanged.

## T3 envelope-matrix cell 88x16k (2026-07-21 23:40): preset hy3-oq2e-rq4-88 FAILS CLOSED at its own memory-limit — zero GPU touched

Guarded window, `--preset hy3-oq2e-rq4-88 --max-live-kv-tokens 16384
--hy3-depths 2,3` (AR+K2+K3). The runtime's own admission preflight
rejected the load before any MLX allocation: `ExpertStreamingConfigurationError:
fixed expert-streaming footprint exceeds limit by 4702360064 bytes`
(4.38 GiB over the preset's declared 95 GiB memory-limit).
`load_count=0`, `hard_peak_memory_bytes=null` — genuinely zero GPU work,
zero risk, box laws intact. Receipt: `oq2e_rq4_88_16k.{json,log}`.

Root cause (code-verified, not guessed): a CPU-only dry run against
`plan_expert_memory` with island_layer_count=79 + the proj-requant q4
discount predicted `fits_fixed=True` (+2.84 GiB margin at 95 GiB), but
that call omits `additional_resident_bytes` — the externally-resident
bf16 MTP head (`hy3-bf16-and-mtp-layer80/layer80-bf16.safetensors`,
~7.0 GiB) plus the non-stock splitk router kernel's incremental bytes,
which `mtplx/runtime.py` adds via `streamed_mtp_resident_bytes +
hy3_router_incremental_bytes` before the real preflight gate. That
~7.22 GiB (99.379 - 92.155 GiB) is present regardless of context size.

Consequence: **the preset's own numbers already overshoot at ITS
DEFAULT max-live-kv-tokens=4096**, not just at 16k. Its own description
text ("20.64 GiB fixed w/ requant credit, 0.949 GiB/island") is
internally consistent with this — 20.64 + 79*0.949 = 95.61 GiB, ~0.61
GiB over the declared 95 GiB — and that 20.64 GiB figure already
correctly includes the MTP head (13.42 GiB non-island/non-MTP fixed +
7.22 GiB MTP/router = 20.64). Commit 004571a0 (David, 2026-07-21 12:56)
bumped island-layer-count 78->79 and shrank expert-cache-limit 2GiB->1GiB
"true full residency" without re-verifying the new total still fit; no
receipt under this preset name exists anywhere before this attempt — it
was never GPU-tested until now.

Fixing to admit 88x16k needs memory-limit >= ~99.4 GiB (fixed_bytes at
16k KV), leaving under 0.6 GiB of headroom against the hard 100 GiB
wired knob — far short of the 7-8+ GiB headroom every other full-
residency (79-island) receipt in this campaign has run with (champion41/
parity: limit 103, hard peak ~94.9 GiB). Per the box laws (never exceed
the 100 GiB knob; stop on any mismatch rather than improvise), this cell
was NOT attempted at a razor-thin margin. No K1/K2/K3 measurement exists
for this cell. Options for whoever picks this up: drop back to
islands=78 (1 streamed layer, restores the pre-004571a0 margin) for the
KV-heavy context columns, or size a dedicated `-88-16k` preset variant
with memory-limit computed from the real fixed_bytes formula (envelope +
context KV + MTP/router + reserve), not the stock 95 GiB carried over
from the 4k-KV baseline.

OPS NOTES: (1) the campaign venv's editable mtplx resolves script-mode
children to the PARENT checkout (mtplx-hy3-ssd root, a different
branch) — the first two launches died pre-lock on the wrapper import
(no collateral; qwen untouched). Window drivers must export
PYTHONPATH=<worktree> and assert mtplx.__file__ in-window (driver does
both now). (2) The 51-c6-mmap-band lane queued windows before and after
this one; flock launch-order coordination worked as designed both times.

## T4 envelope-admission sweep (2026-07-22 00:02): full {preset x
max-live-kv-tokens} feasibility table, hy3-oq2e-rq4-32 added — CPU-only,
zero GPU touched, zero weights loaded

Follow-up to T3. Built `research/envelope_admission_sweep.py`, a CPU-only
harness that drives the SAME preset-resolution/argparse machinery
`scripts/benchmark_q2_mtp_depth_matrix.py` uses for `--preset NAME
--max-live-kv-tokens N`, then replicates the CPU-only prefix of
`mtplx/runtime.py`'s `_load_impl` (`ExpertStreamingConfig` construction ->
open the verified bf16 MTP artifact for its header-declared `payload_bytes`
-> `estimate_hy3_router_kernel_incremental_bytes` -> `resolve_island_placement`
-> `proj_requant_plan_discount` -> `config.memory_plan(...)`) and stops at the
`if not streaming_plan.fits_fixed: raise ExpertStreamingConfigurationError(...)`
check (mtplx/runtime.py, `_load_impl`, the block immediately following the
`streaming_plan = expert_streaming_config.memory_plan(...)` call — same
function/line region T3 already identified as the gate that actually fired;
this is the SAME gate, not the internal duplicate inside
`ExpertStreamingRuntime.open`, since T3's receipt had `load_count=0`, meaning
`.open()` was never reached). `apply_mlx_memory_cap` /
`ExpertStreamingRuntime.open` / `apis.load()` are never called — no MLX
buffer is ever allocated, no weight byte is ever read (the MTP artifact is
opened only far enough to read its safetensors HEADER via `os.pread`, per
`open_verified_hy3_mtp_artifacts`'s own docstring: "Consumers must pass the
yielded file objects directly to mx.load ... "; this script never does).

**Parity proof**: re-run against the exact failing cell from T3
(`--preset hy3-oq2e-rq4-88 --max-live-kv-tokens 16384`) reproduces the
receipt's error BIT-FOR-BIT: `ExpertStreamingConfigurationError: fixed
expert-streaming footprint exceeds limit by 4702360064 bytes` — same error
class, same message template, same byte count as `oq2e_rq4_88_16k.json`.

### Corrected feasibility table

Override rule (per envelope accounting: the envelope is the WEIGHTS budget,
KV stacks on top; the override touches nothing else): `override_bytes =
(kv_tokens - 4096) * 327_680` (`HY3_EXPERT_OQ2E.kv_bytes_per_token`),
`override_limit = preset_memory_limit_bytes + override_bytes`. At this
preset family's fixed 4096-token control, the delta is exact and identical
across every preset: +3.75 GiB at 16384, +8.75 GiB at 32768, +18.75 GiB at
65536.

| envelope (preset) | 4096 (control) | 16384 | 32768 | 65536 |
|---|---|---|---|---|
| **88** (`hy3-oq2e-rq4-88`) | REJECT, excess 0.629 GiB (own footprint over-books its 95 GiB limit before any KV is counted) | REJECT even w/ +3.75 GiB override — excess still 0.629 GiB | REJECT even w/ +8.75 GiB override — excess still 0.629 GiB | REJECT even w/ +18.75 GiB override — excess still 0.629 GiB |
| **80** (`hy3-oq2e-rq4-80`) | ADMIT as-is — implied 86.98 GiB | ADMIT w/ +3.75 GiB override (limit 90.75 GiB) — implied 90.73 GiB | ADMIT w/ +8.75 GiB override (limit 95.75 GiB) — implied 95.73 GiB | ADMIT w/ +18.75 GiB override (limit 105.75 GiB) — implied 105.73 GiB |
| **64** (`hy3-oq2e-rq4-64`) | ADMIT as-is — implied 70.93 GiB | ADMIT w/ +3.75 GiB override (limit 74.75 GiB) — implied 74.68 GiB | ADMIT w/ +8.75 GiB override (limit 79.75 GiB) — implied 79.68 GiB | ADMIT w/ +18.75 GiB override (limit 89.75 GiB) — implied 89.68 GiB |
| **48** (`hy3-oq2e-rq4-48`) | ADMIT as-is — implied 54.81 GiB | ADMIT w/ +3.75 GiB override (limit 58.75 GiB) — implied 58.56 GiB | ADMIT w/ +8.75 GiB override (limit 63.75 GiB) — implied 63.56 GiB | ADMIT w/ +18.75 GiB override (limit 73.75 GiB) — implied 73.56 GiB |
| **32** (`hy3-oq2e-rq4-32`, NEW) | ADMIT as-is — implied 38.97 GiB | ADMIT w/ +3.75 GiB override (limit 42.75 GiB) — implied 42.72 GiB | ADMIT w/ +8.75 GiB override (limit 47.75 GiB) — implied 47.72 GiB | ADMIT w/ +18.75 GiB override (limit 57.75 GiB) — implied 57.72 GiB |

Full byte-exact numbers per cell (fixed_bytes, resident/router/discount
breakdown, override amounts, `unallocated_bytes`) are in
`research/envelope-admission-sweep-2026-07-22.json`.

**88 anomaly (confirms + extends T3)**: every kv column for `hy3-oq2e-rq4-88`
rejects even after the KV-delta override, at a CONSTANT excess of 0.629 GiB
regardless of context size — proof the preset's fixed footprint (islands=79,
independent of KV) over-books its own declared 95 GiB limit, exactly as T3
found at the 16k cell alone. No `-88` cell in this matrix is servable without
first fixing the preset itself (raise memory-limit past ~95.63 GiB fixed, or
drop back to islands=78 per T3's suggested options) — a KV-token override
alone can never rescue it.

**<85 GiB vs >=85 GiB split (box law: ask David before any real GPU attempt
above 85 GiB; the 100 GiB wired knob is never exceeded, full stop):**

- **<85 GiB (11 cells, clear to attempt without asking):** all four `-32`
  cells (38.97 / 42.72 / 47.72 / 57.72 GiB), all four `-48` cells (54.81 /
  58.56 / 63.56 / 73.56 GiB), and the `-64` cells at 4096/16384/32768 (70.93 /
  74.68 / 79.68 GiB).
- **>=85 GiB (5 cells, ASK FIRST):** `-64` at 65536 (89.68 GiB); every `-80`
  cell (86.98 / 90.73 / 95.73 / 105.73 GiB) — note `-80` at its OWN 4096
  control already implies 86.98 GiB, over 85 GiB before any KV override is
  even applied, because the ~7.76 GiB `additional_resident_bytes` (MTP head +
  splitk router) sits outside the "80 GiB" envelope label entirely.
- **Categorically excluded, not just "ask" (>100 GiB hard knob, per the
  never-exceed-the-memory-knob rule):** `-80` at 65536 implies 105.73 GiB —
  this is disqualified outright regardless of who is asked; the admission
  gate itself has no opinion on the 100 GiB physical ceiling (the override is
  a bookkeeping knob, not a hardware check), so passing this gate is NOT
  license to attempt this cell on real GPU.
- All `-88` cells are REJECT (see anomaly above) and contribute to neither
  bucket.

**Sign-off**: 11 of 20 swept cells are clear under 85 GiB with no further
review; 5 are servable only above 85 GiB and require asking David first
(one of those 5, `-80`x65536, is disqualified outright above the 100 GiB
knob); 4 are permanently infeasible under this preset (`-88`, all contexts)
until the preset's own fixed footprint is fixed. The new `hy3-oq2e-rq4-32`
preset (islands=19, 60 streamed) admits at every swept KV level with a
family-consistent margin (0.32 GiB fits_fixed margin at its own 4096
default — inside the family's healthy band of 0.19-1.00 GiB, i.e. neither
88's negative-margin bug nor an oversized, un-verified guess).

**Anomalies**: (1) `-88` is broken across its ENTIRE kv sweep, not just the
16k cell T3 found — see above. (2) `-80`'s own 4096-token default already
implies >85 GiB of real footprint despite its "80 GiB" envelope name, purely
from the externally-resident MTP head + router bytes that the envelope
naming convention doesn't account for; anyone reading "hy3-oq2e-rq4-80" as
"an 80ish GiB run" is off by ~7 GiB before touching KV at all. (3) None of
this required a memory-limit override for `-32`/`-48`/`-64`/`-80` at the
4096 control — those four admit at their preset defaults exactly as
shipped; only the `>4096` KV columns needed the override (15 cells across
all 5 presets: 3 non-control kv levels x 5 presets). 12 of those 15 admit
cleanly once applied (`-32`/`-48`/`-64`/`-80`, 3 each); the exception is
`-88`'s 3 non-control cells, which remain rejected by the same
fixed-footprint over-book as its own control cell — `-88` is the sole
preset where the override is structurally powerless.

---

# T3-pre 64x32k paired kv-mode arms (2026-07-22): kv4 and kv8 KILLED on the acceptance law — MTP draft/trunk calibration breaks almost immediately under KV quantization, even though kv4's HumanEval n=20 screen is untouched (0.95, identical failure to the bf16/q8-KV baseline)

Mid-envelope T3 matrix cell, preset `hy3-oq2e-rq4-64` (islands 52, 27
streamed; proj-requant q4; hy3-router-kernel
`mpp-fp32-splitk-r1-fused-r2`), `--max-live-kv-tokens 32768`, shared
memory-limit override **79.75 GiB** (85,630,910,464 B = 71 GiB preset +
(32768-4096)*327,680 B KV delta — a ceiling identical across all three arms
per the envelope-accounting rule; island/cache sizing never changed) — three
paired arms in ONE guarded window, bf16 KV first as the in-window control
(this row also serves as the T3 matrix-cell receipt for 64x32k bf16 KV),
then kv8, then kv4, `--hy3-depths 1,2,3` (AR auto-runs as depth=0 regardless;
K1/K2/K3 = d1/d2/d3), natural 1024/1024 shape.

**CPU-only preflight** (`research/t3-64x32k/preflight_kv_arms.py`, the exact
`mtplx/runtime.py` `_load_impl` admission-gate replica T4 already validated,
extended to vary `kv_quant`): all three arms ADMIT under the shared 79.75
GiB ceiling — bf16 implied 79.68 GiB, kv8 75.93 GiB, kv4 73.43 GiB, all under
the 85 GiB ask-first line, nowhere near the 100 GiB knob. `verify_expert_manifest`
passed on the `hy3-oq2e-mlx` serving root (`checked_shards=18`); an
independent root-vs-manifest diff (the validator's own truncated-to-4 extras
list is not authoritative) found 2 extra `*.safetensors` files
(`layer80-bf16.safetensors`, `layer80-residents-q.safetensors` — both
accounted for by the checkpoint's own resident/MTP-head layout, not orphans)
and 0 missing. Receipt: `t3_64x32k_admission_preflight.json`.

**Rep protocol**: this harness has no `--reps`/replicate flag
(`retained_replicates` is hardcoded 1 everywhere in this campaign, including
every prior champion receipt) — one retained measurement per cell, matching
established convention, used in place of the brief's suggested 3 reps.

## Per-arm x per-K table

`loads` = `persistent_loads + transient_loads` (verified against `bytes_read`:
loads x ~5.31 MB/expert-record matches `bytes_read` to <0.1%); `svc_ms/load`
= `decode_elapsed_s / loads x 1000`, a blended (read+overhead) per-load cost,
same convention as the campaign's R3/R4 ms/miss figures.

| arm | cell | tok/s | acceptance | hit rate | loads/tok | svc ms/load | peak (hard) |
|---|---|---:|---:|---:|---:|---:|---:|
| bf16 | AR | 8.639 | 0.0000 | 0.1164 | 170.6 | 0.678 | 63.73 GiB |
| bf16 | K1 (d1) | 9.074 | 0.9102 | 0.1216 | 168.8 | 0.653 | 63.73 GiB |
| bf16 | K2 (d2) | 7.995 | 1.5637 | 0.1382 | 193.1 | 0.648 | 63.73 GiB |
| bf16 | K3 (d3) | 6.721 | 1.9038 | 0.1558 | 227.7 | 0.653 | 63.73 GiB |
| kv8 | AR | 9.520 | 0.0000 | 0.2002 | 134.5 | 0.781 | 64.49 GiB |
| kv8 | K1 (d1) | 7.418 | **0.3184** | 0.3397 | 168.3 | 0.801 | 64.49 GiB |
| kv8 | K2 (d2) | 5.771 | **0.1070** | 0.4234 | 207.8 | 0.834 | 64.49 GiB |
| kv8 | K3 (d3) | 7.356 | 1.0160 | 0.3706 | 174.0 | 0.781 | 64.49 GiB |
| kv4 | AR | 9.338 | 0.0000 | 0.1867 | 140.3 | 0.764 | 64.41 GiB |
| kv4 | K1 (d1) | 6.876 | **0.2070** | 0.3465 | 180.0 | 0.808 | 64.41 GiB |
| kv4 | K2 (d2) | 5.741 | **0.1157** | 0.4169 | 211.6 | 0.823 | 64.41 GiB |
| kv4 | K3 (d3) | 12.809 | 2.1667 | 0.3486 | 97.0 | 0.805 | 64.41 GiB |

Peaks are all far under the 79.75 GiB override (the override is a ceiling on
a *static worst-case* KV reservation at 32768 tokens; the benchmark only
ever uses 1024-2048 live tokens, so measured hard peak is dominated by the
fixed island/resident footprint, not KV — expected, not a discrepancy).

## Token-hash / divergence outcomes

Every arm's own depth>0 cells are compared against **that arm's own** AR
(depth=0) row (`ar_comparison`, not cross-arm). All hard gates
(`speculative_event_contract`, `final_state_contract`, `committed_history`,
etc.) are `True` on every row in every arm — no structural/harness fault.

| arm | cell | parity | first_divergence | differing_tokens | observed_token_sha256 (16) |
|---|---|---|---:|---:|---|
| bf16 | AR | True | — | 0 | dcbabe7c2861357b |
| bf16 | K1 | False | 298 | 694 | 2c1e0641f0fb8aa2 |
| bf16 | K2 | False | 298 | 694 | 2c1e0641f0fb8aa2 |
| bf16 | K3 | False | 298 | 697 | 86be119755962790 |
| kv8 | AR | True | — | 0 | 511583febfbe0d13 |
| kv8 | K1 | False | **2** | 1022 | c23ff8fec8b6e79f |
| kv8 | K2 | False | **2** | 1017 | 1f5204b38efaa196 |
| kv8 | K3 | False | **2** | 1017 | 0fb3128aa39be227 |
| kv4 | AR | True | — | 0 | 94b933e405d1dc22 |
| kv4 | K1 | False | **2** | 1015 | 9b9b718e3f7174f6 |
| kv4 | K2 | False | **2** | 1010 | f5edf30a3d9aae8b |
| kv4 | K3 | False | **11** | 1011 | 2bf36ae4f99b426d |

**This cell does not reproduce the "K3 bit-exact" property** the campaign's
full-residency champion/parity-stamped configs established — bf16's own K3
diverges from its own AR at token 298 (differing_tokens 697/1024). That
prior property required `MTPLX_HY3_ROUTER_SPLITK_M1=all` (see "PARITY-STAMPED
CHAMPION"), which neither the `hy3-oq2e-rq4-64` preset nor this mission set;
absent it, this streamed/proj-requant cell shows the same
router-numerics-driven single-region divergence documented there. This is
new information for this specific cell/preset combination, reported as
measured — not the target property.

The bf16 control's divergence pattern (late, single-region, position 298) is
the *normal* campaign signature (batched-verify reassociation / near-tie
logit). kv8 and kv4's pattern is categorically different: **first divergence
at token 2 (kv8, kv4-K1/K2) or 11 (kv4-K3)** — the MTP draft essentially
never matches even that arm's own quantized-trunk AR continuation past the
first couple of tokens, at every depth. AR-vs-AR token sequences also differ
across all three arms (three different sha256), confirming KV quantization
measurably perturbs the trunk's greedy continuation even with **zero**
speculation involved — expected for any argmax-sensitive greedy decode under
numerical perturbation, and the necessary (if not sufficient) precondition
for the acceptance collapse below.

## Acceptance A/B vs the bf16 in-window control

| K | bf16 (control) | kv8 | kv8 delta | kv4 | kv4 delta |
|---|---:|---:|---:|---:|---:|
| K1 | 0.9102 | 0.3184 | **-65.0%** | 0.2070 | **-77.3%** |
| K2 | 1.5637 | 0.1070 | **-93.2%** | 0.1157 | **-92.6%** |
| K3 | 1.9038 | 1.0160 | -46.6% | 2.1667 | +13.8% |

**Noise assessment**: this campaign's established noise bands are sub-1%
in-window (paired, same window: K2 compile-island A/B +0.3%, router M1 AR
+4.6% "as predicted") and ~4% cross-window under sustained load. K1/K2
deltas here are 46-93 percentage points — two orders of magnitude beyond any
noise band this campaign has ever measured. This is **not noise**.

K3's partial "recovery" (kv4 K3 even beats the bf16 control) does not save
the arms: K3's own divergence signature (first_divergence=2 or 11, ~99% of
the sequence differing) shows the SAME broken draft/trunk calibration as
K1/K2, just landing on a higher accepted-per-verify count by chance on this
one 1024-token sample; it is not evidence of a repaired mechanism. AR/K1/K2/K3
run sequentially inside one model load per arm (shared, cumulatively-warmed
frequency cache — hit rate climbs AR->K2 within every arm, matching every
prior receipt in this campaign; this does not affect token content, since
cache hit/miss returns bit-identical weights either way).

## Verdict — acceptance law fires; ALU/speed comparison reported per the brief anyway

**kv4 is KILLED for MTP-speculative serving. kv8 is KILLED too**, by the
same law, at the same severity. Acceptance collapse at K1/K2 is 65-93
percentage points below the in-window bf16 control — far beyond any noise
this campaign has ever measured — so per the acceptance law ("if kv4
acceptance drops beyond noise vs the in-window bf16 control, kv4 is KILLED
for serving regardless of speed"), speed is moot for the K>0 (MTP) serving
path. Root cause (mechanistic, not proven at the tensor level): the MTP
draft head is bf16 and was calibrated against a bf16 trunk; quantizing trunk
KV (even at q8, 17/32 of bf16 bytes) perturbs the trunk's hidden state
enough that under exact-match greedy verify, the draft's proposals miss
almost immediately — this is a draft/trunk calibration break, not a
harness bug (every structural gate passes; the pattern is consistent and
reproducible across both quantization levels and three depths).

**Sub-4-bit ALU-risk framing does not directly apply** (this is trunk-KV
precision, not expert-weight precision), but the requested kv8-vs-kv4 speed
comparison is reported anyway: K1/K2 kv4 is 5-7% slower than kv8 (matching
directionally with "more dequant work"); K3 inverts (kv4 12.81 vs kv8 7.36,
kv4 the fastest cell in the entire sweep) — plausibly the smaller KV
footprint's bandwidth saving outweighing verify overhead at that specific
depth, though this is under-powered (n=1 cell) and moot given both arms are
already killed on acceptance.

**AR-only note** (not gated by the acceptance law, which is specifically
about the MTP mechanism): kv8 (9.52) and kv4 (9.34) AR tok/s both *beat*
the bf16 control's AR (8.64) — plain non-speculative KV quantization is a
real, if modest, win here. If a future serving mode ever runs AR-only (no
MTP), kv-quant remains worth a dedicated look; it must not be paired with
MTP under the current (bf16) draft-head calibration.

## kv4 HumanEval n=20 screen: pass@1 = 0.95 (19/20) — task competence UNTOUCHED despite the acceptance collapse

Run as its own standalone guarded window (window 2), immediately after
window 1 released the flock — never overlapped, per box law. Same runtime
identity as the kv4 speed arm (`hy3-expert-oq2e`, islands 52, proj-requant
q4, `mpp-fp32-splitk-r1-fused-r2`, same 79.75 GiB memory-limit override,
same 32768-token KV admission envelope), `--kv-quant q4`, reached through
`evals/litellm_hy3/handler.py`'s `MTPLX_HY3_*` env overrides. Same
20-task HumanEvalPlus-v0.1.10 gate as every prior screen in this campaign
(greedy, seed 42, chat endpoint, `no_think`, dataset sha256 `42526ec0…`).

- **pass@1 = 0.95 (19/20)**, `request_errors=0`, all 20 completions
  `finish_reason=stop` (77-237 completion tokens). Sole failure:
  **HumanEval/10** (AssertionError) — the SAME task every prior HumanEval
  run in this campaign fails (shipped-q2 0.80, oQ2e 0.95, proj_requant 0.95,
  full-164 both arms).
- **Identical score and identical sole failure to the bf16/q8-KV oQ2e
  baseline** ("oQ2e HumanEval" above: 0.95, HumanEval/10). At n=20
  resolution, kv4's task competence is indistinguishable from the
  non-KV-quantized baseline, even though its MTP acceptance collapsed by
  65-93 points in the paired speed arm above. This is the same pattern this
  campaign has repeatedly found for weight quantization (PPL/exact-token-
  agreement collapses while HumanEval survives): **exact-match speculative
  /calibration metrics are far more sensitive than actual task quality.**
  It does not overturn the acceptance-law verdict — the law gates the
  *speculative-serving mechanism*, not raw model quality, and those are
  measurably different things here.
- "proxy" in `humaneval_t3_64x32k_kv4_n20_proxy.log` = the LiteLLM
  OpenAI-API-compatible proxy server process (`litellm --config config.yaml
  --port 18183`) that fronts `handler.py`'s `Hy3StreamedLLM`, so the generic
  `code_eval_gate.py` harness can hit a normal `/v1/chat/completions`
  endpoint; that file is the proxy SERVER's own stdout/stderr (startup,
  request handling), distinct from the harness's result receipt
  (`humaneval_t3_64x32k_kv4_n20.json`) and distinct from the speed arms'
  `.log` files (those call the runtime in-process via the CLI, no proxy
  involved).
- First request wall time 97.96 s = cold model load (islands 52, 27
  streamed) + first generation; provenance `wall_s=406.76` for all 20 tasks.

## `evals/litellm_hy3/handler.py` change (load-bearing, kept)

`_champion_overrides()` had no `kv_quant` key — every other quant knob
(`proj_quant`, `proj_requant`) was already `MTPLX_HY3_*`-overridable but
`--kv-quant` (commit `8aed1942`, benchmark-CLI-only) was never plumbed into
the LiteLLM lane. Added `"kv_quant"` following the exact `proj_requant`
"none"-sentinel pattern (`MTPLX_HY3_KV_QUANT`, default unset/None = bf16).
Required for the n=20 screen above — without it, kv4's HumanEval run would
silently serve bf16 KV regardless of the env var, which would test the
wrong arm. Verified against `evals/litellm_hy3/test_handler_offline.py`
(12/12 pass, no MLX/mtplx needed) before use.

## Commands (exact)

Window 1 (bf16/kv8/kv4 speed arms), single guarded window:
```
bash research/t3-64x32k/run_speed_arms.sh
```
which (after the CPU-only preflight) runs, per arm, inside
`run_with_qwen_stopped.py --plist ~/Library/LaunchAgents/com.tea.qwen.plist
--lock-timeout-seconds 7200 --child-timeout-seconds 10800`:
```
python scripts/benchmark_q2_mtp_depth_matrix.py \
  --preset hy3-oq2e-rq4-64 --max-live-kv-tokens 32768 \
  --memory-limit 85630910464 --hy3-depths 1,2,3 \
  [--kv-quant q8 | --kv-quant q4] \
  --output-json evals/tier2/t3_64x32k_{bf16,kv8,kv4}.json
```

Window 2 (kv4 HumanEval n=20), standalone, launched only after window 1's
flock was released:
```
bash research/t3-64x32k/run_kv4_humaneval_n20.sh
```
which exports `MTPLX_HY3_MODEL_KEY=hy3-expert-oq2e
MTPLX_HY3_MODEL_ROOT=~/.cache/huggingface/hy3-oq2e-mlx
MTPLX_HY3_ISLAND_LAYER_COUNT=52 MTPLX_HY3_PROJ_QUANT=none
MTPLX_HY3_PROJ_REQUANT=q4 MTPLX_HY3_KV_QUANT=q4
MTPLX_HY3_ROUTER_KERNEL=mpp-fp32-splitk-r1-fused-r2
MTPLX_HY3_MEMORY_LIMIT=85630910464 MTPLX_HY3_MAX_LIVE_KV_TOKENS=32768` then
runs `run_with_qwen_stopped.py -- bash evals/litellm_hy3/serve_and_eval.sh`
(`HUMANEVAL_LIMIT` defaults to 20).

Receipts: `t3_64x32k_admission_preflight.{json,log}`,
`t3_64x32k_{bf16,kv8,kv4}.{json,log}`,
`humaneval_t3_64x32k_kv4_n20.json` + `_proxy.log`. Drivers:
`research/t3-64x32k/{preflight_kv_arms.py,run_speed_arms.sh,
run_speed_arms_inner.sh,run_kv4_humaneval_n20.sh}`.

---

# T3 64 GiB envelope: island-vs-cache A/B (2026-07-22) -- CACHE WINS CLEARLY at every K; new preset `hy3-oq2e-rq4-64-cachehvy` committed; K3 bit-exact patch PARTIALLY succeeds (K1/K2 now fully bit-exact, K3 improved 298->805 but not fully bit-exact)

Three-part mission on the `hy3-oq2e-rq4-64` envelope (islands 52, proj-requant
q4): (1) Arm A -- the preset as shipped, 16k KV, doubling as the official T3
matrix 64x16k bf16 cell; (2) Arm B -- the SAME 64 GiB / 16k-KV envelope with
ZERO islands and the freed budget reallocated to an explicit expert-cache,
two cache-policy sub-arms (frequency, lru), per 0157fae's "cache beats
islands on missing banks" motivation; (3) a patch lane re-running Arm A's
config at 32k KV WITH `MTPLX_HY3_ROUTER_SPLITK_M1=all` set, to test whether
that cures the bit-exact caveat on the existing (unpatched)
`t3_64x32k_bf16.json` receipt. AR + K1/K2/K3, natural 1024/1024, 3 reps per
cell (this harness has no `--reps` flag -- 3 full separate process
invocations per lane, aggregated post-hoc by `research/t3-64-ab/aggregate.py`,
not the campaign's usual 1-retained-measurement convention; the mission
explicitly asked for 3 reps here and the window budget supported it).

## Exact-lane convention confirmed: `MTPLX_HY3_ROUTER_SPLITK_M1=all` is PROCESS-WIDE, not K3-only

Checked the receipts before touching GPU (`mtplx/runtime.py` reads
`os.environ.get("MTPLX_HY3_ROUTER_SPLITK_M1", "mtp")` exactly once inside
`_load_impl`, at model load; `benchmark_q2_mtp_depth_matrix.py` loads the
model ONCE per process and reuses it for every requested depth -- AR/K1/K2/K3
all share one `load_model` call). "PARITY-STAMPED CHAMPION" shows AR 40.18
(parity True) AND K2 42.18 (parity True) together, under one `M1=all`
setting -- proof it was applied to the whole load, not scoped to K2/K3 only
(there is no mechanism to scope it narrower within one process). Set here as
`export MTPLX_HY3_ROUTER_SPLITK_M1=all` once per guarded-window inner script,
covering every lane, every depth, matching that convention exactly.

## Admission: all 4 lane configs ADMIT, sub-85 GiB, manifest diff matches expected state

CPU-only preflight (`research/t3-64-ab/preflight.py`, the same
`_load_impl`-replica machinery as T4/T3-pre) evaluated Arm A, both Arm B
sub-arms, and the patch lane before any GPU touch:

| lane | islands | fixed GiB | cache GiB | implied total GiB | limit GiB |
|---|---:|---:|---:|---:|---:|
| armA | 52 | 73.7505 | 0.9344 | 74.6849 | 74.7500 |
| armB_frequency | 0 | 24.3911 | 49.9922 | 74.3833 | 74.7500 |
| armB_lru | 0 | 24.3911 | 49.9922 | 74.3833 | 74.7500 |
| patch_k3exact | 52 | 78.7505 | 0.9344 | 79.6849 | 79.7500 |

Arm B's expert-cache-limit (53,678,702,592 B = 49.9921875 GiB) was DERIVED,
not guessed: a two-pass plan (oversized placeholder cache-limit to read off
the naturally budget-capped `persistent_cache_bytes`, then a self-consistency
re-plan with that exact byte count as the real limit) confirms islands
52 -> 0 frees exactly 49.359375 GiB (52 * 0.949 GiB/island, matching the
family's own documented island cost to 5 decimal places) and that the cache
absorbs essentially all of it (49.9922 of 49.9922 GiB available, the ~0.37
GiB gap being persistent-cache slot-count floor-division quantization, not a
deliberate margin). `verify_expert_manifest` passed on the `hy3-oq2e-mlx`
root (no error); the independent, non-truncated root-vs-manifest diff found
exactly 2 extra `*.safetensors` (`layer80-bf16.safetensors`,
`layer80-residents-q.safetensors`, both accounted for) and 0 missing --
matches the state the T3-pre validation window already established. All 4
lanes sub-85 GiB; no sign-off needed; the 100 GiB knob untouched.

**Why Arm B needed zero islands reached without `--preset`**:
`ExpertStreamingConfig.__post_init__` rejects `island_layer_count=0` outright
(`minimum=1`); the ONLY way to reach a genuine zero-island static plan is to
never set `--island-layer-count`/`--island-layers` at all, so the hardcoded
argparse defaults (`None` / `""`) apply. Since `--preset hy3-oq2e-rq4-64`
pushes `island-layer-count=52` in as an argparse DEFAULT (there is no CLI
value that resets a preset-set default back to `None`), Arm B's CLI
necessarily bypasses `--preset` and passes every other `hy3-oq2e-rq4-64` flag
explicitly instead (verified byte-identical admission math either way).

## Guarded-window disruption (operational note, not a box-law violation)

Window 1 (`research/t3-64-ab/run_window.sh`) ran Arm A (3/3 reps) and Arm B
frequency (3/3 reps) cleanly, then the BACKGROUND BASH TASK itself was
killed by the harness (~60 minutes of wall time; likely an out-of-range
`timeout` parameter on the launching Bash call -- 18,000,000 ms was passed,
10x the tool's documented 600,000 ms max -- rather than anything in the
guarded-window design) just as Arm B lru rep 1 started loading. Verified
safe before continuing: `/private/tmp/mtplx-gpu-exclusive.lock` was free,
qwen had already been restored to a fresh PID by `run_with_qwen_stopped.py`'s
`finally` block, and the one partial output file
(`t3_64x16k_armB_lru_rep1.json`) was a setup-phase-only checkpoint (`"status":
"running"`, `"models": []`, no measurement) -- deleted, not counted. No
foreign lock holder was ever touched. Split the remainder into two further
guarded windows with an in-range `timeout` (window 2: Arm B lru, 3/3 reps
clean; window 3: the K3-exact patch lane, 3/3 reps clean), each independently
CPU-preflighted before opening. All three windows' `run_with_qwen_stopped.py`
wrappers exited cleanly (flock released, qwen restored, verified by health
check before the next window opened).

## Per-lane x per-cell table (mean of 3 reps; K3-only patch lane also gets an AR reference row)

| lane | cell | tok/s | accept | hit rate | loads/tok | svc ms/load | peak hard GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| armA | AR | 8.731 | 0.0000 | 0.2309 | 171.06 | 0.670 | 63.73 |
| armA | K1 | 8.788 | 0.9102 | 0.2329 | 168.79 | 0.675 | 63.73 |
| armA | K2 | 7.933 | 1.5637 | 0.2406 | 193.12 | 0.653 | 63.73 |
| armA | K3 | 6.831 | 1.9038 | 0.2475 | 227.70 | 0.643 | 63.73 |
| armB_frequency | AR | 9.311 | 0.0000 | 0.9286 | 59.69 | 1.799 | 63.12 |
| armB_frequency | K1 | **11.592** | 0.9102 | 0.9292 | 61.61 | 1.400 | 63.12 |
| armB_frequency | K2 | 11.151 | 1.5637 | 0.9332 | 68.43 | 1.311 | 63.12 |
| armB_frequency | K3 | 9.632 | 1.9038 | 0.9364 | 78.05 | 1.330 | 63.12 |
| armB_lru | AR | 8.645 | 0.0000 | 0.9238 | 62.73 | 1.844 | 63.12 |
| armB_lru | K1 | 10.785 | 0.9102 | 0.9252 | 64.01 | 1.448 | 63.12 |
| armB_lru | K2 | 10.822 | 1.5637 | 0.9326 | 68.25 | 1.354 | 63.12 |
| armB_lru | K3 | 9.446 | 1.9038 | 0.9389 | 74.45 | 1.422 | 63.12 |
| patch_k3exact | AR | 8.507 | 0.0000 | 0.2309 | 171.06 | 0.687 | 63.73 |
| patch_k3exact | K3 | 6.796 | 1.9038 | 0.2475 | 227.70 | 0.646 | 63.73 |

Every rep within every lane/cell agreed on token-parity status and
`observed_token_sha256` (see per-lane JSON `*_consistent` flags) -- fully
deterministic, no rep-to-rep flake anywhere in this mission. `accepted_per_verify`
is identical across armA/armB/patch at matching K (0.9102/1.5637/1.9038): MTP
acceptance is a property of the model+draft head, unaffected by island vs
cache placement, exactly as the campaign's prior receipts establish.

## A/B verdict: Arm B (cache-heavy) wins CLEARLY and CONSISTENTLY at every K -- margin far exceeds the 4% band

| cell | armA | armB_freq | freq margin | armB_lru | lru margin | freq vs lru |
|---|---:|---:|---:|---:|---:|---:|
| AR | 8.731 | 9.311 | **+6.7%** | 8.645 | -1.0% (in-noise) | +7.7% |
| K1 (both arms' own optimum) | 8.788 | **11.592** | **+31.9%** | 10.785 | +22.7% | +7.5% |
| K2 | 7.933 | 11.151 | **+40.6%** | 10.822 | +36.4% | +3.0% (in-noise) |
| K3 | 6.831 | 9.632 | **+41.0%** | 9.446 | +38.3% | +2.0% (in-noise) |

Arm A/B genuinely shares one window (Arm A + Arm B frequency both ran in
window 1), so the <1% in-window band applies to that pair directly -- every
margin (+6.7% to +41.0%) is 7-40x that band. Arm B lru ran in window 2 (a
separate, cross-window measurement vs Arm A), so the ~4% cross-window band is
the right comparison there: the AR margin (-1.0%) is inside that band (not
distinguishable from noise -- lru is a wash with islands at AR only), but
K1/K2/K3 (+22.7% to +38.3%) are 5-9x the cross-window band and stand as real.
frequency-vs-lru is also cross-window (freq in window 1, lru in window 2):
the AR/K1 edge (+7.5%/+7.7%) exceeds the 4% band and looks real; K2/K3
(+2-3%) sit inside it and are not distinguishable from cross-window drift --
reported as a wash, not a frequency win, at deeper K.

**Why**: hit rate 0.93-0.94 (Arm B, both policies) vs 0.23-0.25 (Arm A);
loads/tok 60-78 (Arm B) vs 169-228 (Arm A). Arm A's tiny 2 GiB cache serves
only its 27 streamed layers and churns hard; Arm B's ~50 GiB shared pool
spans all 79 routed layers and, once warm, serves the large majority of
requests without a fresh load -- fewer loads/token even though it covers 3x
the layers. This is the SAME mechanism 0157fae found on the q4/bf16-KV stack
at ~80 GiB ("cache beats islands on missing banks... K1 > AR in every arm");
it reproduces cleanly on the oQ2e/proj-requant-q4 streaming stack at 64 GiB.
K1 is the optimum K for BOTH arms (K2+ monotonically worse), also matching
0157fae's finding.

**Preset committed**: `hy3-oq2e-rq4-64-cachehvy`
(`benchmarks/presets.toml`) -- same envelope family, zero islands, explicit
`--expert-cache-limit 53678702592`, `cache-policy=frequency` (default,
matching 0157fae's precedent and this measurement's AR/K1 edge; lru is
within noise at K2/K3 so not clearly worse, but frequency was never clearly
worse anywhere either). Verified via the same CPU-only harness: admits at
its own 4096-token default (implied 70.63/71 GiB) AND, overridden exactly
like this mission's Arm B test (`--max-live-kv-tokens 16384 --memory-limit
80262201344`), reproduces the measured admission numbers bit-for-bit
(24.3911 GiB fixed / 49.9922 GiB cache / 74.3833 GiB implied). Does NOT
overwrite `hy3-oq2e-rq4-64` -- both exist side by side; making cache-heavy
the committed DEFAULT for this envelope is David's call, not made here.

## K3-exact patch lane outcome: PARTIAL fix -- K1/K2 now fully bit-exact, K3 improved but still diverges

The mission asked whether `MTPLX_HY3_ROUTER_SPLITK_M1=all` cures the
bit-exact caveat on the existing (unpatched) `t3_64x32k_bf16.json` receipt,
which showed K1/K2/K3 all diverging from AR at the SAME early point (token
298, `differing_tokens` 694/694/697) because AR used the stock host router
path at rows==1 while K1/K2/K3's batched verify always used the split-K
kernel -- a router-numerics mismatch, per the "Router M1 scope A/B" entry's
attribution.

With the env set (this mission's Arm A AND the dedicated 32k-KV patch lane,
both showing byte-identical results): **K1 and K2 are now fully bit-exact**
(`token_parity=True` on every rep) -- the fix worked completely for those two
depths, exactly as "PARITY-STAMPED CHAMPION" predicted for a full-residency
config, now confirmed on this streamed/proj-requant-q4 config too. **K3 is
NOT fully bit-exact**: `token_parity=False`, but first divergence moved from
token 298 (unpatched) to **token 805** (patched), and `differing_token_count`
dropped from 697/1024 to 213/1024 -- the divergence region shrank from
"most of the sequence" to "the last ~21%". This is the "normal" late,
single-region campaign signature (batched-verify reassociation / near-tie
logit), NOT the early token-2 divergence the mission flagged as a
stop-and-report condition -- nothing anomalous observed, just an incomplete
fix at K3 specifically.

**Independent cross-check (unplanned, strong evidence)**: all four lanes in
this mission (armA, armB_frequency, armB_lru, patch_k3exact) produced the
EXACT SAME K3 `observed_token_sha256`
(`86be1197559627900a928b581edf6b01fc9238f019d0a68c8c9bc4e8206b5da1`), same
`first_divergence=805`, same `differing_token_count=213` -- across two
different KV budgets (16k, 32k) and two different memory placements (52
islands vs 0 islands + cache). Confirms the campaign's own principle
("cache hit/miss returns bit-identical weights either way") holds here:
island vs cache placement and KV ceiling do not affect model output at all,
only which layers pay a read-vs-cache-hit cost. Also: the patched run's AR
`observed_token_sha256` (`2c1e0641f0fb8aa2...`) exactly equals the
UNPATCHED run's K1/K2 sha256 from the original receipt -- consistent
mechanistic explanation: `M1=all` makes AR use the same split-K router path
K1/K2/K3 always used, so patched-AR now walks the identical numeric path
old-K1/K2 already computed, and the only remaining AR-vs-K3 gap is genuine
speculative-verify reassociation, which surfaces far later (805 vs 298).

**Receipt updated**: `t3_64x32k_k3exact.json` is the mission's stated patch
receipt; `t3_64x32k_bf16.json` (the original, unpatched 64x32k T3-pre
receipt, env NOT set) is left as-is/untouched -- it remains the correct
record of what that specific run measured, now with this entry as the
documented follow-up rather than an in-place rewrite.

## Commands (exact)

Window 1 (Arm A 3 reps + Arm B frequency 3 reps, interrupted after Arm B
frequency completed -- see disruption note):
```
bash research/t3-64-ab/run_window.sh
```
Window 2 (Arm B lru 3 reps, continuation):
```
bash research/t3-64-ab/run_window2.sh
```
Window 3 (K3-exact patch lane 3 reps, continuation):
```
bash research/t3-64-ab/run_window3.sh
```
Each ultimately invokes, per rep, inside `run_with_qwen_stopped.py`:
```
# Arm A / patch lane (via preset):
python scripts/benchmark_q2_mtp_depth_matrix.py \
  --preset hy3-oq2e-rq4-64 --max-live-kv-tokens {16384|32768} \
  --memory-limit {80262201344|85630910464} --hy3-depths {1,2,3|3} \
  --output-json evals/tier2/t3_64x{16k,32k}_{armA,k3exact}_rep{1,2,3}.json

# Arm B (explicit flags, no --preset -- see "Why Arm B needed zero islands"):
python scripts/benchmark_q2_mtp_depth_matrix.py \
  --model hy3-oq2e --contexts 1024 --output-tokens 1024 \
  --memory-limit 80262201344 --runtime-reserve 7GiB \
  --expert-cache-limit 53678702592 --max-live-kv-tokens 16384 \
  --cache-policy {frequency|lru} --cache-scope layer --slot-layout component-banks \
  --hy3-router-kernel mpp-fp32-splitk-r1-fused-r2 --verify-strategy batched \
  --expert-integrity headers-only --proj-requant q4 \
  --split-route-release deferred --hy3-depths 1,2,3 \
  --output-json evals/tier2/t3_64x16k_armB_{frequency,lru}_rep{1,2,3}.json
```
`MTPLX_HY3_ROUTER_SPLITK_M1=all MTPLX_SUSTAINED_PREFILL=1
MTPLX_HY3_SUBMIT_CADENCE=8` exported once per inner script, before any rep.

Receipts: `t3_64_ab_admission_preflight{,_window2,_window3}.{json,log}`,
`t3_64x16k_armA.json` (+ `_rep{1,2,3}.json/.log`),
`t3_64x16k_armB_{frequency,lru}.json` (+ `_rep{1,2,3}.json/.log`),
`t3_64x32k_k3exact.json` (+ `_rep{1,2,3}.json/.log`). Drivers:
`research/t3-64-ab/{preflight.py,aggregate.py,run_window.sh,
run_window_inner.sh,run_window2.sh,run_window2_armBlru_inner.sh,
run_window3.sh,run_window3_patch_inner.sh}`. Preset:
`benchmarks/presets.toml` `[preset.hy3-oq2e-rq4-64-cachehvy]`.
