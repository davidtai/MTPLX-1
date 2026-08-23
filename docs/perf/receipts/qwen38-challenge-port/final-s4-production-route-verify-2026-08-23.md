# Qwen 3.8 S4 production-route verification

- Commit: `cb7f8c9e`
- Model: `~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`
- Guard: `/tmp/mtplx-gpu-exclusive.lock` through `bench/laguna/run_guarded.py`
- Prompt: exact 16,384-token `mtplx/generation.py` context with the intact
  `python_modules_long.jsonl` instruction tail
- Output limit: 1,024
- Sampling: target/draft temperature 1.0, top-p 0.95, top-k 20, seed 42
- Route: `kv_only_history+dual_norm+qmv_final+source_proposal`
- Source artifact raw SHA-256:
  `d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1`

This is a one-route deployment verification, not an additional timed ABBA arm.
The runtime installed S1-S3 at model load, installed S4 immediately after the
ordinary Q4 draft head was created, and released the BF16 control MTP body and
full Q4 control head before generation.

```json
{"control_released":true,"decode_tok_s":52.40679317180268,"engagement":{"dual_norm":{"calls":918},"qmv_final":{"g32_m4":41736},"source_proposal":{"e87_probe_calls":918,"k_island_calls":1207,"q_island_calls":918,"row_top32_calls":918,"selected_q4_rerank_calls":918,"selector_calls":918,"v_island_calls":1207}},"generated_tokens":1024,"peak_memory_gib":25.041209558025002,"prefill_tok_s":826.0810156240761,"route":"kv_only_history+dual_norm+qmv_final+source_proposal","wall_s":39.518471749994205}
```
