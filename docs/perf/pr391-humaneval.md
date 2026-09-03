# PR 391 quality screen, HumanEval and HumanEval+

Terms used in this document:

- MTP: multi-token prediction.

Four rows are measured and on record. The mlx-serve row was not run. Nothing in
this table is estimated: a cell either carries a receipt or reads `not run`.

Harness and settings:

- Harness `scripts/fable/humaneval_screen.py`, evalplus 0.3.1, dataset hash `fe585eb4df8c88d844eeb463ea4d0302`.
- All 164 problems, greedy: temperature 0.0, top-p 0.95, top-k 20, `max_tokens` 768, `enable_thinking` false, `n=1`.
- Prompt template `evalplus-0.3.1-openai-chat`, one request at a time, under the exclusive GPU lock.

| Engine or arm | HumanEval pass@1 | HumanEval+ pass@1 | Pooled gen tok/s | Wall | Body parity |
| --- | ---: | ---: | ---: | ---: | --- |
| branch, full set of 20 keys: 12 decode and 8 prefill | **0.951** (156/164) | **0.933** (153/164) | 118.9 | 365.7 s generation, 428.3 s total | sampler and dataset blocks identical to the other 2 measured rows |
| branch, composed decode stack of 12 keys | **0.939** (154/164) | **0.927** (152/164) | 119.0 | 367.9 s generation, 430.6 s total | sampler and dataset blocks identical to the other 2 measured rows |
| branch, every key off | **0.933** (153/164) | **0.921** (151/164) | 110.5 | 391.6 s generation, 449.0 s total | sampler and dataset blocks identical to the other 2 measured rows |
| upstream 2.10.2 | 0.933 (153/164) | 0.921 (151/164) | 130.0 | 408 s generation | request digest `101f4146e731…` over all 164 bodies, every field but `model` |
| mlx-serve 26.8.11 | not run | not run | not run | not run | not run |

Receipts, schema `mtplx-fable-humaneval-screen-v1`:

| Row | Receipt | Label | `arm` | `MTPLX_FABLE_*` keys set |
| --- | --- | --- | --- | ---: |
| branch, full set | `composed-fullset-decode-prefill-2.json` | `composed-fullset-decode-prefill-2` | candidate | 20 |
| branch, 12 keys | `composed-decode-stack-final.json` | `composed-decode-stack-final` | candidate | 12 |
| branch, every key off | `control.json` | `control` | control | 0 |

Two columns are computed and are not receipt fields:

- **Pooled gen tok/s** is the sum of `usage.completion_tokens` over the 164 requests divided by the sum of `elapsed_s`, both read from each run's `generation_receipts.jsonl`. The three sums are 43,453 tokens over 365.4 s, 43,758 tokens over 367.6 s, and 43,226 tokens over 391.3 s. The receipts carry no per-problem rate, so this is not a mean of per-problem rates.
- **Wall** is `timings_s.generation_s` for the 164 generations and `timings_s.total_s` for generation, scoring and server readiness.

Three points on the cross-engine row:

- The upstream 2.10.2 row comes from a separate endpoint screen with the same 164 problems, greedy, thinking off, one request at a time. Its pooled tok/s is token-weighted from the server's own timings block, not from the client clock. It is therefore not comparable with the client-clock figures of the three branch rows.
- The three measured rows use the same pack: artifact fingerprint `sha256:855300a7...`, `arch_id` `qwen4-next`, `mtp_depth_max` 3. The mlx-serve row will use a different pack, so that row measures engine and pack together.
- The branch screen records no request body digest. Body parity across the three branch rows comes from the `sampler` and `dataset` blocks of the receipts, which are byte-identical. The cross-engine screen records a per-request digest, given in the table.
