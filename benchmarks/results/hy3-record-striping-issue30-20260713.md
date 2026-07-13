# Issue #30 Phase 2: exact-demand Hy3 record striping

Decision: reject 2/4/8-way within-record striping. Keep the direct synchronous 1-way `preadv` path; do not integrate this mechanism into the runtime.

## Question

Can issuing disjoint, aligned, exact-demand reads for one required 10,616,832-byte Hy3 expert record increase useful storage throughput without increasing p95 record service time?

The control performs one direct synchronous `preadv`. Each candidate uses a persistent 2-, 4-, or 8-worker executor and publishes the record once only after every range completes and the full record passes SHA-256 validation. Routing remains authoritative; the benchmark adds no speculation, scheduler policy, cache-policy, MLX/Metal, or kernel change.

## Result

All candidates lost useful bandwidth and increased p95 service time. The paired 95% intervals exclude zero in the wrong direction for every width across the full 16 repeats.

| Stripes | Host calls / record | Mean useful GiB/s | Useful delta vs. 1-way, 95% CI | Mean p50 ms | Mean p95 ms | p95 delta ms, 95% CI | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 1 | 10.011 | control | 0.961 | 0.992 | control | retain |
| 2 | 2 | 9.744 | -2.665% [-3.790%, -1.758%] | 0.989 | 1.040 | +0.0479 [+0.0305, +0.0703] | reject |
| 4 | 4 | 9.713 | -2.962% [-4.027%, -1.947%] | 0.988 | 1.026 | +0.0341 [+0.0275, +0.0442] | reject |
| 8 | 8 | 9.610 | -3.996% [-4.674%, -3.320%] | 1.002 | 1.042 | +0.0505 [+0.0449, +0.0575] | reject |

Selection and confirmation were evaluated separately before the full result was summarized:

| Stripes | Selection useful delta, 95% CI | Confirmation useful delta, 95% CI | Selection p95 delta ms, 95% CI | Confirmation p95 delta ms, 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 2 | [-3.989%, -2.152%] | [-4.446%, -0.997%] | [+0.0260, +0.0783] | [+0.0271, +0.0811] |
| 4 | [-4.697%, -2.404%] | [-4.278%, -0.963%] | [+0.0238, +0.0329] | [+0.0283, +0.0580] |
| 8 | [-5.070%, -3.812%] | [-4.759%, -2.516%] | [+0.0446, +0.0531] | [+0.0419, +0.0653] |

The decision does not depend on a fixed percentage threshold: none of the widths established a positive useful-throughput interval, positive milliseconds saved, or non-regressing p95 in either phase.

## Measurement contract

- Commit: `f51c727a2a5006b4022effd82e9c10b142dbf732`; the same clean commit and harness SHA were verified before and after measurement.
- Profile: isolated exact-demand reads from `pipenetwork/Hy3-4bit` at revision `160619d3f96c8470350b6dac0ef033a8381551e3`; no generation, prompt, sampler, or token count applies.
- Workload: 96 unique deterministic records per arm, 32 warmups, 16 balanced repeats, widths 1/2/4/8, seed 30.
- Uncertainty: paired percentile bootstrap, 20,000 resamples, 95% intervals; repeats 0-7 select and repeats 8-15 confirm.
- I/O: `F_NOCACHE`; exact aligned ranges; actual `preadv` attempts and returned bytes counted.
- Validation: full-record SHA-256 before one logical publication.
- Cold-state gate: every measured arm verified 0% residency before and after its reads; all 6,144 eviction requests succeeded.

Across the 64 measured arms, all 6,144 record hashes matched, logical publications equaled record requests, 65,229,815,808 returned bytes equaled logical bytes, and byte amplification remained 1.0. Host calls increased as specified from 1 to 2/4/8 per record.

The campaign ran on an Apple M5 Max with 128 GB unified memory and macOS 26.5.2. Machine lifecycle handling was kept outside this repository and its tests. The exclusive lane was released, and the exact pre-campaign Qwen state (`mtplx-qwen36-27b-optimized-speed`) was restored afterward. Raw machine and lifecycle artifacts are retained under the ignored `benchmarks/raw/hy3-record-stripes/hy3-record-stripes-20260713T213238Z-f51c727a2a50/` directory and were not committed.

## Reproduction

From a clean committed worktree, with machine-level exclusivity handled externally:

```bash
.venv/bin/python scripts/benchmark_hy3_record_stripes.py \
  "$MODEL_ROOT" "$MODEL_ROOT/expert-manifest-sidecar.json" \
  --stripe-counts 1,2,4,8 \
  --operations 96 \
  --warmup-operations 32 \
  --repeats 16 \
  --seed 30 \
  --bootstrap-resamples 20000 \
  --output-json /tmp/hy3-record-stripes.json
```

Machine-readable evidence is in `benchmarks/results/hy3-record-striping-issue30-20260713.json`.

## Boundary

This microbenchmark rejects exact-demand within-record striping for the measured Hy3 record geometry. It does not authorize speculative prefetch, scheduler changes, cache-policy changes, MLX/Metal work, kernel changes, or any runtime integration.
