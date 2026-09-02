# Decode critical path, production lane (Qwen3.8 Flash-Next, 16K, B=1)

Built 2026-09-02 from two sources that agree with each other:

* **GPU timestamps** — `.benchmark-artifacts/pr391/w58-retained-{control,composed}-census-*.jsonl`,
  `record:"cb"` rows. `gpu_start_ns`/`gpu_end_ns` and `encode_start_ns`/`encode_end_ns`
  are real clocks; only the per-op split inside a command buffer is modelled, and
  nothing here uses that split.
* **Production phase timers** — the W64 driver receipts
  (`fable-w64-draft-{body,chain}-*.json`, 12 arms each) and the ABBA summaries.

## 0. The census transfers to production

| | census (instrumented) | production receipts |
| --- | ---: | ---: |
| cycle, raw, incl. copy rounds | 39.703 ms | 39.637 ms |

**0.17 % apart.** The dispatch census buffers its records; it does not measurably
slow the host. Its GPU busy/idle therefore applies to the production lane
directly.

> **Correction.** An earlier W64 note claimed the instrumented build inflates
> host time ~5.1x, reading the census's `per dispatch 2.208 us` fit coefficient
> as a host cost. It is not: that fit models **GPU busy** per command buffer
> from (cb count, dispatch count, bytes). The 5.1x claim was wrong and the
> conclusion it supported — that the idle map overstates gaps by ~5x — is
> withdrawn. The real reason those gaps do not convert to wall time is §2.

## 1. Where the cycle goes

Union of command-buffer GPU intervals, 382 cycles (control) / 394 (composed):

| | control | composed |
| --- | ---: | ---: |
| cycle | 39.703 ms | 38.912 ms |
| **GPU busy** | **32.402 ms (81.6 %)** | **31.798 ms (81.7 %)** |
| **exposed host** | **7.301 ms (18.4 %)** | **7.114 ms (18.3 %)** |
| dispatches / cycle | 4,685 | 3,759 |
| command buffers / cycle | 160.5 | 158.9 |

Splitting the exposed host with the encode clocks — union of
`[encode_start, encode_end]`, and the part of it that overlaps a GPU-busy
interval:

| | control | composed |
| --- | ---: | ---: |
| host encode, union | 15.074 ms | 12.964 ms |
| …overlapped with GPU busy (free) | 13.297 ms (88 %) | 11.288 ms (87 %) |
| **…exposed (GPU idle while encoding)** | **1.776 ms** | **1.676 ms** |
| **exposed host that is NOT encoding** | **5.525 ms** | **5.438 ms** |

The non-encoding remainder is Python between ops, NumPy, sampling, the
blocking syncs and driver turnaround.

**Conversion rate.** tok/window is 2.6806 (control), so

* cycle 39.637 ms -> **67.63 tok/s**;
* **1 ms/cycle removed = +1.75 tok/s**;
* at zero exposed host (cycle == GPU busy == 32.402 ms) -> **82.7 tok/s**.

80 tok/s needs **-6.15 ms/cycle**: 85 % of the entire exposed host budget, or a
16 % cut in GPU busy, or a mix.

## 2. Why removing dispatches and syncs keeps measuring neutral

Each dispatch costs **3.22 us of host encode** (15.074 ms / 4,685) but only
**0.379 us of it is exposed** (1.776 ms / 4,685). 88 % of encoding happens while
the GPU is busy and is free.

> **A dispatch diet returns 0.38 us per dispatch removed, not 3.2 us.**

Retro-check against the program's dead levers:

| lever | dispatches removed | model | measured |
| --- | ---: | ---: | --- |
| W60 `qsa_s1` | 168 | 0.064 ms | predicted 0.307, never converted |
| W59 GDN tiny chain | 72 | 0.027 ms | 0.131 static, inside noise |
| HC_M4 + OPDIET | 926 | GPU busy −0.604 + encode −0.351 = **−0.955 ms** | composed −1.67 % = **−0.66 ms** |

The first two are below the 0.2-0.8 % (0.08-0.32 ms) A/A noise floor by
construction — they could never have been measured. The third is the right
order and slightly optimistic, which is the expected direction (some of the
removed encode was already overlapped).

**This retro-explains the whole dead dispatch-diet class**, and it means no
future launch-count lever is fundable unless it removes >2,000 dispatches.

## 3. Why the draft span shrank 0.34 ms and the cycle did not

Two separate effects, both measured, neither of them "a wait absorbed it":

**(a) The saving was real and it did reach the cycle — once the accounting was
fixed.** The prewarm (cache promotion + `mx.compile` trace + first-ever Metal
shader compile) landed inside `decode_elapsed_s`; `first_primary_sample_time_s`
went 0.003 -> 0.024 s steady state and **1.159 s on the first process ever to
compile the body** (= 2.93 ms/M4win over 387 windows, against a measured
arm1-vs-arm2 gap of 2.90). Corrected, body mode is −0.141 / −0.188 ms/cycle on
the two clean seeds.

**(b) An equal and opposite cost was introduced, not absorbed.** The compiled
body needs a fixed-capacity MTP cache, so the route promotes `QSACache` to
`TensorOffsetQSACache` and two per-cycle offset reads that were free python ints
become `int(mx.array)` — i.e. evals — at two points no timer covers. Measured as
the unattributed remainder (`decode_elapsed` minus every named span):

| per cycle, candidate − control | s29 | s30 | s31 |
| --- | ---: | ---: | ---: |
| chain mode | +0.4103 | +0.4318 | +0.4233 |
| body mode | +0.4225 | +0.4399 | +0.4243 |

Six arms, two modes, spread 0.03 ms, against a draft-span saving of
0.34-0.42 ms. Same size — which is the whole story. Fixed on this branch by
keeping the offset on device (`min(current, target)` in graph).

**(c) The three per-depth draft readbacks are worth ~0.048 ms/cycle in total**,
not the 2.62 ms the idle-gap map assigns to those gaps. Chain minus body, draft
span, per cycle: −0.0546 / −0.0260 / −0.0640. So **one draft readback costs
~24 us**, ~36x less than its GPU-idle gap.

The reason the gap and the wall cost differ: the draft chain is a **serial
dependency**. Depth d+1 cannot start until depth d's token exists, so the GPU
idle there is not recoverable by removing the sync — the host work still has to
happen and there is no GPU work to overlap it with. Only removing the host work
itself converts, which is what the compiled body did (−0.39 ms) and what the
readback collapse barely did (−0.05 ms).

## 4. Re-priced lever table

Pricing rules, from §1:

* **GPU-byte lever** — full credit. The GPU binds 81.7 % of the cycle and the
  host runs ahead (`cap_wait`/`sched_backpressure` 12.6 ms/cycle: the main
  thread is throttled waiting on the GPU, not the reverse). 1 ms of GPU busy
  removed = 1 ms of cycle = +1.75 tok/s.
* **Host lever** — credit only against the exposed host at the sync it touches:
  7.30 ms/cycle total, of which 1.78 ms is encode (0.379 us/dispatch) and
  5.52 ms is everything else.

| lever | class | credit | basis | confidence |
| --- | --- | ---: | --- | --- |
| K-M1 route kernel | GPU | **−0.92 ms** (+1.6 tok/s) | measured micro, exact | high — micro EXACT, all counters 0 |
| W65 shared-expert stream | GPU | full GPU-busy delta; ceiling **1.24 ms** (control MoE-shared family) | census family table | med — ceiling only, needs their micro |
| W66 GDN replay fold | GPU | full; ceiling **4.96 ms** (GDN family) | census family table | med — ceiling only |
| W68 sparse attention | GPU | full; ceiling **1.93 ms** (QSA family) | census family table | med — ceiling only |
| W64 draft chain (this) | host | **−0.44 ms** (−0.39 compiled body, −0.05 readbacks) | measured, 12 arms | high — measured twice |
| W62 PLE boundary | host | **≤0.61 ms** (control) / 0.44 (composed) | retained census PLE boundary | med — its 1.16-1.33 estimate was on the OLD census |
| W42 pre-scatter | host+GPU | **0.1-0.2 ms** | K-D1 micro 0.095-0.123 ms/step ×3, mostly host | med |
| W67 N-layer prefix | host | price against 5.52 ms non-encode exposed at the sync it touches | — | low — needs its own sync identified |
| any dispatch diet | host/encode | **0.379 us × dispatches removed** | §2 | high — retro-fits 3 measured arms |

Composed GPU-lever ceiling (route + shared + GDN + QSA, if each took its whole
family, which none will): −9.05 ms -> 88 tok/s. Realistically each takes
20-40 % of its family.

Composed host-lever total on the table: −1.2 ms -> **+2.1 tok/s**. The host side
cannot reach 80 with the levers now in flight; **the funded path to 80 is
GPU bytes.**

## 5. Uncertainty

* One census run per arm, one seed. GPU busy is a single sample; the
  control/composed pair agree to 1.9 %, which bounds run-to-run drift on the
  busy side but is not a variance estimate.
* The census-to-production transfer rests on the 0.17 % cycle agreement in §0.
  If a future census diverges from its production twin by more than the A/A
  noise floor (0.2-0.8 %), re-derive before using it.
* The encode split assumes `encode_start/end` bracket main-thread Metal
  encoding. If MLX encodes off-thread for some ops, exposed encode is
  overstated and the dispatch-diet rate is worse, not better.
* `sched_worker_wait` (26 ms/cycle) exceeds the cycle and is a worker thread
  idling; it is excluded. `cb_wait_until_completed` records zero events in the
  window — MLX's blocking eval surfaces as `cap_wait`/`sched_backpressure`, so
  the syncs cannot be sized from the wait buckets alone. They are sized from
  production A/B instead (§3c).
* tok/window is held constant. Every lever here is a cycle-time lever; block
  verification is the only retained tok/window lever and composes
  independently.
