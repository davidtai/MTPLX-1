# Qwen 3.8 S3 production-route verification

- Commit: `1674e775d17750ac942f35b92bf56a6905c692a5`
- Model: `~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`
- Guard: `/tmp/mtplx-gpu-exclusive.lock` through `bench/laguna/run_guarded.py`
- Prompt: exact 16,384-token `mtplx/generation.py` context with the intact
  `python_modules_long.jsonl` instruction tail
- Output limit: 1,024
- Sampling: target/draft temperature 1.0, top-p 0.95, top-k 20, seed 42
- Route: `kv_only_history+dual_norm+source_proposal`
- Source artifact raw SHA-256:
  `d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1`

This is a one-route deployment verification, not an additional timed ABBA arm.
The automatic runtime route released the BF16 control MTP body and full Q4
control head before generation.

```json
{"accepted_by_depth":[260,232,195],"control_released":true,"decode_tok_s":54.39821004719359,"drafted_by_depth":[295,295,295],"engagement":{"dual_norm":{"calls":885},"source_proposal":{"e87_probe_calls":885,"k_island_calls":1163,"q_island_calls":885,"row_top32_calls":885,"selected_q4_rerank_calls":885,"selector_calls":885,"v_island_calls":1163}},"generated_tokens":1024,"peak_memory_gib":25.041209558025002,"prefill_tok_s":816.9162816157282,"route":"kv_only_history+dual_norm+source_proposal","token_hash":"de1e70f763a317da3c789f2f27f2033c6f87a64a39984bd92f32ef925bbeef57","wall_s":38.98954608300119}
```
