# Hy3 Q4 native benchmark matrix

Date: 2026-07-11

## Generation gates

All rows use the same 313-token realistic chat prompt, greedy decoding, a
2,048-token output cap, one repeat, a 112 GiB memory limit, a 78 GiB expert
cache limit, 4,096 live KV tokens, 64 MiB uncached reads, and the
component-bank layout with 32 transient slots. Throughput and memory numbers
are single-run measurements, not confidence intervals.

| Artifact | Trunk | MTP head | Output tokens | Accepted / drafted | Acceptance | End-to-end tok/s | Elapsed | Peak memory | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Community control | Q4 | none (AR) | 1,905 | - | - | 5.835 | 326.5 s | 83.02 GiB | [`hy3-q4-gate-v2-ar-cbank99.json`](hy3-q4-gate-v2-ar-cbank99.json) |
| Community control | Q4 | legacy quantized | 1,721 | 222 / 1,498 | 14.82% | 3.534 | 487.0 s | 86.17 GiB | [`hy3-q4-gate-v2-mtp-cbank99.json`](hy3-q4-gate-v2-mtp-cbank99.json) |
| Community control | Q4 | BF16 | 1,721 | 316 / 1,404 | 22.51% | 3.697 | 465.6 s | 96.00 GiB | [`hy3-q4-gate-v3-mtp-bf16-cbank99.json`](hy3-q4-gate-v3-mtp-bf16-cbank99.json) |
| **Native artifact** | **Q4** | none (AR) | 1,905 | - | - | **5.922** | 321.7 s | 83.02 GiB | [`hy3-q4-native-gate-ar-cbank99.json`](hy3-q4-native-gate-ar-cbank99.json) |
| **Native artifact** | **Q4** | BF16 | 1,721 | 316 / 1,404 | **22.51%** | **3.832** | 449.1 s | 96.00 GiB | [`hy3-q4-native-gate-mtp-bf16-cbank99.json`](hy3-q4-native-gate-mtp-bf16-cbank99.json) |
| **Native artifact** | **Q4** | **Q4** | 1,721 | 296 / 1,424 | **20.79%** | **3.769** | 456.6 s | 86.17 GiB | [`hy3-q4-native-gate-mtp-q4-cbank99.json`](hy3-q4-native-gate-mtp-q4-cbank99.json) |

`hy3-q4-native` is the new quantized model. Every native row uses its Q4
trunk. The BF16 row changes only the layer-80 MTP head; the Q4-head row is the
fully quantized configuration.

## Direct comparisons

| Comparison | Result |
| --- | --- |
| Native AR vs community AR | Token IDs are exactly equal; native was 1.50% faster in these single runs. |
| Native BF16 MTP vs community BF16 MTP | Token IDs, accepted drafts, drafted tokens, acceptance, and peak memory are exactly equal; native was 3.66% faster in these single runs. |
| Native Q4 head vs native BF16 head | Q4 accepted 20 fewer drafts and was 1.72 percentage points lower in acceptance; it used 9.83 GiB less peak memory. |
| Native Q4 MTP vs native AR | MTP was 36.4% slower end-to-end at this acceptance rate. |

The exact native/community token and acceptance parity means the requantized
native trunk does not improve the BF16-head MTP stall on this gate. This
strongly refutes the community trunk hiddens as the dominant cause of the
15-25% acceptance ceiling for this prompt and runtime. The Q4-head A/B also
lands inside the same ceiling, only 1.72 points below BF16.

## Correctness and quality gates

| Gate | Native result | Evidence |
| --- | --- | --- |
| Layer 1 fixed-vector Q4 parity | Pass at `atol=rtol=1e-5`; minimum projection cosine 0.994999 | [`hy3-q4-native-layer1-validation.json`](hy3-q4-native-layer1-validation.json) |
| Layer 80 fixed-vector Q4 parity | Pass at `atol=rtol=1e-5`; minimum projection cosine 0.995785 | [`hy3-q4-native-layer80-validation.json`](hy3-q4-native-layer80-validation.json) |
| Q8 router oracle | Exact stored leaves, IDs, and scores for layers 1 and 80; correction bias bit-exact | Layer validation JSON above |
| 24-token perplexity sanity | Pass; native PPL 82.9092, community PPL 82.9092, ratio 1.0 | [`hy3-q4-native-perplexity-sanity.json`](hy3-q4-native-perplexity-sanity.json) |
| Manifest and physical sidecar | Pass; 15,360 records and full 163,074,539,520-byte sidecar verified | [`hy3-q4-native-handoff.md`](hy3-q4-native-handoff.md) |
| Repository tests | 2,025 pass, 4 expected skips | [`hy3-q4-native-handoff.md`](hy3-q4-native-handoff.md) |

## Earlier diagnostic probe

The community-trunk BF16 draft-rank probe remains useful diagnostic context:
over 300 post-norm positions, the true token had median rank 6, top-5 rate
45.67%, top-20 rate 62.33%, and rank-1 acceptance 25%. The native BF16
generation gate's exact token and acceptance parity shows that replacing the
trunk quantization did not move that near-miss behavior. See
[`hy3-q4-mtp-draft-rank-probe.json`](hy3-q4-mtp-draft-rank-probe.json).

