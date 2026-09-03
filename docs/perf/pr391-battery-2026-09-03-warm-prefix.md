## Three-server comparison

_`n=` is the number of SEEDS behind each mean. The battery runs fewer seeds as the context grows, so the rows are not equally weighted: an `n=1` cell is a single measurement with no spread at all, not a mean. Per-cell counts are in F-seeds._

**Prefill tok/s**  _(higher is better)_

| Cell | upstream v2.10.2 (control) | branch (+FR-Spec, +compiled MTP prepare, +restack) | Δ% | mlx-serve 26.8.11 | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanity | 355±24 _n=3_ | 359±24 _n=3_ | +1.0% | 407±7 _n=3_ | +14.7% |
| 1K | 855±64 _n=3_ | 875±68 _n=3_ | +2.3% | 1,628±41 _n=3_ | +90.4% |
| 8K | 1,185±17 _n=3_ | 1,205±41 _n=3_ | +1.7% | 1,805±21 _n=3_ | +52.3% |
| 16K | 1,191±8 _n=3_ | 1,506±68 _n=3_ | +26.5% | 1,507±5 _n=3_ | +26.5% |
| 32K | 1,077±3 _n=3_ | 1,316±20 _n=3_ | +22.2% | 1,140±0 _n=3_ | +5.9% |
| 64K | 1,016±10 _n=3_ | 1,228±38 _n=3_ | +20.9% | 805±1 _n=3_ | -20.8% |
| 128K | 979±7 _n=2_ | 1,216±0 _n=2_ | +24.3% | 572±0 _n=2_ | -41.6% |
| 255K | 952 _n=1_ | 1,110 _n=1_ | +16.6% | 437 _n=1_ | -54.1% |

**Decode tok/s**  _(higher is better)_

| Cell | upstream v2.10.2 (control) | branch (+FR-Spec, +compiled MTP prepare, +restack) | Δ% | mlx-serve 26.8.11 | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanity | 82.25 (79.55-85.82) _n=3_ | 100.53 (95.66-104.33) _n=3_ | +22.2% | 94.13 (89.85-99.88) _n=3_ | +14.4% |
| 1K | 67.95 (66.20-71.24) _n=3_ | 86.33 (82.37-94.14) _n=3_ | +27.0% | 67.43 (62.80-74.75) _n=3_ | -0.8% |
| 8K | 55.51 (52.34-59.16) _n=3_ | 82.55 (79.54-86.14) _n=3_ | +48.7% | 61.57 (61.22-62.20) _n=3_ | +10.9% |
| 16K | 57.65 (56.14-58.72) _n=3_ | 79.61 (75.63-83.70) _n=3_ | +38.1% | 65.73 (63.40-68.89) _n=3_ | +14.0% |
| 32K | 55.34 (54.01-57.86) _n=3_ | 78.01 (75.76-80.12) _n=3_ | +41.0% | 60.43 (58.73-63.58) _n=3_ | +9.2% |
| 64K | 56.60 (52.75-62.26) _n=3_ | 76.92 (71.81-79.83) _n=3_ | +35.9% | 52.24 (50.33-53.28) _n=3_ | -7.7% |
| 128K | 51.17 (49.95-52.39) _n=2_ | 73.36 (70.69-76.03) _n=2_ | +43.4% | 45.77 (43.24-48.29) _n=2_ | -10.6% |
| 255K | 45.86 _n=1_ | 66.54 _n=1_ | +45.1% | 35.15 _n=1_ | -23.4% |

**TTFT s**  _(lower is better)_

| Cell | upstream v2.10.2 (control) | branch (+FR-Spec, +compiled MTP prepare, +restack) | Δ% | mlx-serve 26.8.11 | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanity | 0.265±0.037 _n=3_ | 0.264±0.037 _n=3_ | -0.6% | 0.238±0.004 _n=3_ | -10.5% |
| 1K | 1.397±0.084 _n=3_ | 1.322±0.091 _n=3_ | -5.4% | 0.654±0.016 _n=3_ | -53.2% |
| 8K | 7.121±0.083 _n=3_ | 6.545±0.535 _n=3_ | -8.1% | 4.572±0.054 _n=3_ | -35.8% |
| 16K | 13.972±0.089 _n=3_ | 5.802±0.255 _n=3_ | -58.5% | 10.914±0.039 _n=3_ | -21.9% |
| 32K | 30.718±0.077 _n=3_ | 13.891±1.252 _n=3_ | -54.8% | 28.796±0.011 _n=3_ | -6.3% |
| 64K | 64.936±0.632 _n=3_ | 44.518±11.414 _n=3_ | -31.4% | 81.485±0.097 _n=3_ | +25.5% |
| 128K | 134.589±1.033 _n=2_ | 108.350±0.032 _n=2_ | -19.5% | 229.331±0.162 _n=2_ | +70.4% |
| 255K | 275.476 _n=1_ | 118.248 _n=1_ | -57.1% | 597.681 _n=1_ | +117.0% |

**Peak mem GB**  _(lower is better)_

| Cell | upstream v2.10.2 (control) | branch (+FR-Spec, +compiled MTP prepare, +restack) | Δ% | mlx-serve 26.8.11 | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanity | 84.26±0.13 _n=3_ | 84.67±0.06 _n=3_ | +0.5% | 73.25±0.00 _n=3_ | -13.1% |
| 1K | 85.63±0.36 _n=3_ | 85.90±0.45 _n=3_ | +0.3% | 74.27±0.00 _n=3_ | -13.3% |
| 8K | 87.87 _n=3_ | 92.08±1.13 _n=3_ | +4.8% | 77.37±0.00 _n=3_ | -12.0% |
| 16K | 89.57±0.00 _n=3_ | 96.89±1.45 _n=3_ | +8.2% | 77.80±0.00 _n=3_ | -13.1% |
| 32K | 92.02±0.00 _n=3_ | 100.77±0.02 _n=3_ | +9.5% | 79.28 _n=3_ | -13.9% |
| 64K | 92.03±0.00 _n=3_ | 101.75±0.13 _n=3_ | +10.6% | 82.11±0.00 _n=3_ | -10.8% |
| 128K | 91.58±0.00 _n=2_ | 97.21±0.01 _n=2_ | +6.2% | 88.07 _n=2_ | -3.8% |
| 255K | 96.33 _n=1_ | 104.11 _n=1_ | +8.1% | 99.54 _n=1_ | +3.3% |

**Peak footprint GB**  _(lower is better)_

| Cell | upstream v2.10.2 (control) | branch (+FR-Spec, +compiled MTP prepare, +restack) | Δ% | mlx-serve 26.8.11 | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanity | 85.58±0.01 _n=3_ | 86.00±0.03 _n=3_ | +0.5% | 74.73±0.46 _n=3_ | -12.7% |
| 1K | 87.69±0.21 _n=3_ | 88.02±0.48 _n=3_ | +0.4% | 75.88±0.55 _n=3_ | -13.5% |
| 8K | 91.67±0.91 _n=3_ | 96.43±1.41 _n=3_ | +5.2% | 79.79±0.02 _n=3_ | -13.0% |
| 16K | 94.65±0.47 _n=3_ | 102.36±0.94 _n=3_ | +8.2% | 81.24±0.02 _n=3_ | -14.2% |
| 32K | 98.02±0.14 _n=3_ | 108.15±0.36 _n=3_ | +10.3% | 84.21±0.04 _n=3_ | -14.1% |
| 64K | 98.80±0.33 _n=3_ | 108.28±0.03 _n=3_ | +9.6% | 88.40±0.03 _n=3_ | -10.5% |
| 128K | 100.79±0.01 _n=2_ | 103.99±0.27 _n=2_ | +3.2% | 96.75±0.00 _n=2_ | -4.0% |
| 255K | 105.24 _n=1_ | 108.34 _n=1_ | +2.9% | 111.44 _n=1_ | +5.9% |

**Wall s**  _(lower is better)_

| Cell | upstream v2.10.2 (control) | upstream v2.10.2 (control) gen tok | branch (+FR-Spec, +compiled MTP prepare, +restack) | branch (+FR-Spec, +compiled MTP prepare, +restack) gen tok | Δ% | mlx-serve 26.8.11 | mlx-serve 26.8.11 gen tok | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vanity | 2.72 (2.16-3.35) _n=3_ | 202 (152-254) | 2.26 (1.96-2.49) _n=3_ | 199 (178-212) | -16.9% | 2.34 (2.04-2.51) _n=3_ | 197 (181-210) | -14.0% |
| 1K | 15.46 (12.79-16.90) _n=3_ | 952 (807-1,024) | 12.22 (9.71-13.63) _n=3_ | 932 (793-1,024) | -20.9% | 9.88 (6.75-15.93) _n=3_ | 613 (396-991) | -36.1% |
| 8K | 22.38 (20.56-25.83) _n=3_ | 846 (704-1,024) | 16.79 (13.97-19.19) _n=3_ | 850 (619-1,024) | -25.0% | 19.94 (17.50-21.34) _n=3_ | 948 (795-1,024) | -10.9% |
| 16K | 29.73 (26.29-31.53) _n=3_ | 910 (696-1,024) | 18.69 (17.84-19.19) _n=3_ | 1,024 (1,024-1,024) | -37.1% | 24.84 (23.60-25.62) _n=3_ | 916 (806-986) | -16.5% |
| 32K | 49.28 (48.38-49.81) _n=3_ | 1,024 (1,024-1,024) | 25.15 (24.77-25.57) _n=3_ | 875 (685-1,004) | -49.0% | 42.14 (38.92-46.14) _n=3_ | 806 (595-1,024) | -14.5% |
| 64K | 75.65 (74.95-76.68) _n=3_ | 596 (557-672) | 54.75 (39.75-65.26) _n=3_ | 780 (435-1,024) | -27.6% | 97.37 (93.31-100.88) _n=3_ | 829 (632-1,024) | +28.7% |
| 128K | 146.87 (145.79-147.94) _n=2_ | 577 (483-671) | 120.25 (117.68-122.83) _n=2_ | 808 (592-1,024) | -18.1% | 251.75 (250.35-253.15) _n=2_ | 1,024 (1,024-1,024) | +71.4% |
| 255K | 280.66 _n=1_ | 225 | 133.36 _n=1_ | 821 | -52.5% | 626.78 _n=1_ | 1,024 | +123.3% |

**Prefill s (server)**  _(lower is better)_

| Cell | upstream v2.10.2 (control) | branch (+FR-Spec, +compiled MTP prepare, +restack) | Δ% | mlx-serve 26.8.11 | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanity | 0.246±0.017 _n=3_ | 0.244±0.017 _n=3_ | -1.0% | 0.214±0.004 _n=3_ | -13.2% |
| 1K | 1.205±0.094 _n=3_ | 1.178±0.096 _n=3_ | -2.2% | 0.629±0.016 _n=3_ | -47.8% |
| 8K | 6.916±0.100 _n=3_ | 6.391±0.533 _n=3_ | -7.6% | 4.540±0.054 _n=3_ | -34.4% |
| 16K | 13.756±0.097 _n=3_ | 5.621±0.264 _n=3_ | -59.1% | 10.875±0.038 _n=3_ | -20.9% |
| 32K | 30.437±0.098 _n=3_ | 13.661±1.243 _n=3_ | -55.1% | 28.754±0.011 _n=3_ | -5.5% |
| 64K | 64.526±0.652 _n=3_ | 44.186±11.378 _n=3_ | -31.5% | 81.426±0.096 _n=3_ | +26.2% |
| 128K | 133.936±0.970 _n=2_ | 107.783±0.037 _n=2_ | -19.5% | 229.238±0.161 _n=2_ | +71.2% |
| 255K | 274.227 _n=1_ | 117.394 _n=1_ | -57.2% | 597.501 _n=1_ | +117.9% |

**Per-seed decode tok/s** (spread, not just the mean)

| Cell | Server | per-seed values | mean | sd | min | max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| vanity | upstream-2.10.2 | 85.82, 79.55, 81.40 | 82.25 | 2.63 | 79.55 | 85.82 |
| vanity | branch-fullstack | 95.66, 104.33, 101.60 | 100.53 | 3.62 | 95.66 | 104.33 |
| vanity | mlx-serve | 92.65, 89.85, 99.88 | 94.13 | 4.23 | 89.85 | 99.88 |
| 1K | upstream-2.10.2 | 71.24, 66.43, 66.20 | 67.95 | 2.32 | 66.20 | 71.24 |
| 1K | branch-fullstack | 82.37, 82.48, 94.14 | 86.33 | 5.52 | 82.37 | 94.14 |
| 1K | mlx-serve | 74.75, 64.75, 62.80 | 67.43 | 5.23 | 62.80 | 74.75 |
| 8K | upstream-2.10.2 | 55.04, 59.16, 52.34 | 55.51 | 2.81 | 52.34 | 59.16 |
| 8K | branch-fullstack | 86.14, 81.97, 79.54 | 82.55 | 2.72 | 79.54 | 86.14 |
| 8K | mlx-serve | 61.28, 61.22, 62.20 | 61.57 | 0.45 | 61.22 | 62.20 |
| 16K | upstream-2.10.2 | 58.72, 56.14, 58.09 | 57.65 | 1.10 | 56.14 | 58.72 |
| 16K | branch-fullstack | 79.49, 83.70, 75.63 | 79.61 | 3.30 | 75.63 | 83.70 |
| 16K | mlx-serve | 68.89, 64.91, 63.40 | 65.73 | 2.32 | 63.40 | 68.89 |
| 32K | upstream-2.10.2 | 54.14, 57.86, 54.01 | 55.34 | 1.78 | 54.01 | 57.86 |
| 32K | branch-fullstack | 75.76, 80.12, 78.16 | 78.01 | 1.78 | 75.76 | 80.12 |
| 32K | mlx-serve | 58.73, 63.58, 58.98 | 60.43 | 2.23 | 58.73 | 63.58 |
| 64K | upstream-2.10.2 | 62.26, 52.75, 54.79 | 56.60 | 4.09 | 52.75 | 62.26 |
| 64K | branch-fullstack | 79.13, 71.81, 79.83 | 76.92 | 3.63 | 71.81 | 79.83 |
| 64K | mlx-serve | 50.33, 53.28, 53.12 | 52.24 | 1.35 | 50.33 | 53.28 |
| 128K | upstream-2.10.2 | 52.39, 49.95 | 51.17 | 1.22 | 49.95 | 52.39 |
| 128K | branch-fullstack | 76.03, 70.69 | 73.36 | 2.67 | 70.69 | 76.03 |
| 128K | mlx-serve | 43.24, 48.29 | 45.77 | 2.52 | 43.24 | 48.29 |
| 255K | upstream-2.10.2 | 45.86 | 45.86 | 0.00 | 45.86 | 45.86 |
| 255K | branch-fullstack | 66.54 | 66.54 | 0.00 | 66.54 | 66.54 |
| 255K | mlx-serve | 35.15 | 35.15 | 0.00 | 35.15 | 35.15 |

## Prefill scaling

**Prefill fit** `prefill_s = intercept + slope x prompt_tokens`, least squares over the cell means.

| Server | Intercept (fixed cost) | Slope (marginal) | R^2 | worst residual |
| --- | ---: | ---: | ---: | ---: |
| upstream-2.10.2 | -2,237 ms | 1052.231 us/token | 0.99958 | +2.392 s |
| branch-fullstack | 4,505 ms | 504.509 us/token | 0.88433 | +37.151 s |
| mlx-serve | -27,249 ms | 2268.901 us/token | 0.97949 | -40.902 s |

```
  upstream-2.10.2:
         87 tok  measured   0.246s  fit  -2.145s  resid +2.392s
      1,024 tok  measured   1.205s  fit  -1.159s  resid +2.364s
      8,192 tok  measured   6.916s  fit   6.383s  resid +0.533s
     16,384 tok  measured  13.756s  fit  15.003s  resid -1.247s
     32,768 tok  measured  30.437s  fit  32.243s  resid -1.806s
     65,536 tok  measured  64.526s  fit  66.722s  resid -2.196s
    131,072 tok  measured 133.936s  fit 135.681s  resid -1.745s
    261,120 tok  measured 274.227s  fit 272.522s  resid +1.705s
  branch-fullstack:
         87 tok  measured   0.244s  fit   4.549s  resid -4.305s
      1,024 tok  measured   1.178s  fit   5.021s  resid -3.844s
      8,192 tok  measured   6.391s  fit   8.638s  resid -2.247s
     16,384 tok  measured   5.621s  fit  12.771s  resid -7.150s
     32,768 tok  measured  13.661s  fit  21.037s  resid -7.375s
     65,536 tok  measured  44.186s  fit  37.568s  resid +6.617s
    131,072 tok  measured 107.783s  fit  70.632s  resid +37.151s
    261,120 tok  measured 117.394s  fit 136.242s  resid -18.848s
  mlx-serve:
         87 tok  measured   0.214s  fit -27.051s  resid +27.265s
      1,024 tok  measured   0.629s  fit -24.925s  resid +25.555s
      8,192 tok  measured   4.540s  fit  -8.662s  resid +13.202s
     16,384 tok  measured  10.875s  fit   9.925s  resid +0.951s
     32,768 tok  measured  28.754s  fit  47.098s  resid -18.345s
     65,536 tok  measured  81.426s  fit 121.446s  resid -40.020s
    131,072 tok  measured 229.238s  fit 270.141s  resid -40.902s
    261,120 tok  measured 597.501s  fit 565.207s  resid +32.295s
```

### Footnotes (generated)

- **F1 packs differ.** MTPLX serves `Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` (137 GB, 4-bit gs32, 8-bit gs64 lm_head/embed); mlx-serve serves `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` (98 GB, uniform 4-bit gs64). ~40% more weight at a finer group size explains much of the memory and part of the decode gap. **MTPLX-vs-mlx-serve is engine PLUS pack; branch-vs-upstream is pack-clean and is what this PR is about.**
- **F2 draft depth differs.** Effective depth per arm: `branch-fullstack`=3, `mlx-serve`=adaptive (up to 6-8), `upstream-2.10.2`=3. MTPLX is hard-pinned; mlx-serve's `--mtp` leaves depth adaptive (logs show depth=6 with an EV controller re-planning per round). Never present as an unqualified engine comparison.
- **F4 prefill scope is comparable.** mlx-serve brackets `runPrefill` alone and reports tokenise separately as `tokenize_ms`; MTPLX starts its timer at generator entry on already-tokenized ids. Both exclude cached tokens from the denominator, so `Prefill s` / `Prefill tok/s` are the same quantity on all three. TTFT is client-measured on all three. This is a footnote, not a caveat.
- **F-parity every engine was asked for the same thing.** All 21 cell(s) run on more than one engine carry an IDENTICAL per-cell request digest across `branch-fullstack`, `mlx-serve`, `upstream-2.10.2` (`request_body_sha256`: every body field but `model`). Response side: thinking observed on `branch-fullstack` 18/18 sweep cells, `mlx-serve` 18/18 sweep cells, `upstream-2.10.2` 18/18 sweep cells; vanity thought on no engine; finish reasons `branch-fullstack` ['length', 'stop'], `mlx-serve` ['length', 'stop'], `upstream-2.10.2` ['length', 'stop']. **`branch-fullstack` accepts `top_p+top_k order` but honours it only yes, but top_p is applied BEFORE top_k**: Both values are applied verbatim, but the FILTER ORDER differs from mlx-serve, which applies top_k first (generate.zig:8026 then :8034 on the AR path, :7663 then :7669 on the MTP verify path). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets, so this is a real difference in the sampled distribution, not a formality. It is inherent to the engines and is not something the request can equalise. (mtplx/sampling.py:121-130 -- 'Local mlx_lm applies top-p before top-k, so MTPLX's NumPy reference path mirrors that order') Remedy: disclose it. Sending only ONE of the two would remove the difference, but it would also change the sampler the model ships with, so the honest move is the footnote rather than a quieter benchmark. **`branch-fullstack` accepts `top_p+top_k order` but honours it only yes, but top_p is applied BEFORE top_k**: Both values are applied verbatim, but the FILTER ORDER differs from mlx-serve, which applies top_k first (generate.zig:8026 then :8034 on the AR path, :7663 then :7669 on the MTP verify path). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets, so this is a real difference in the sampled distribution, not a formality. It is inherent to the engines and is not something the request can equalise. (mtplx/sampling.py:121-130 -- 'Local mlx_lm applies top-p before top-k, so MTPLX's NumPy reference path mirrors that order') Remedy: disclose it. Sending only ONE of the two would remove the difference, but it would also change the sampler the model ships with, so the honest move is the footnote rather than a quieter benchmark. **`branch-fullstack` accepts `top_p+top_k order` but honours it only yes, but top_p is applied BEFORE top_k**: Both values are applied verbatim, but the FILTER ORDER differs from mlx-serve, which applies top_k first (generate.zig:8026 then :8034 on the AR path, :7663 then :7669 on the MTP verify path). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets, so this is a real difference in the sampled distribution, not a formality. It is inherent to the engines and is not something the request can equalise. (mtplx/sampling.py:121-130 -- 'Local mlx_lm applies top-p before top-k, so MTPLX's NumPy reference path mirrors that order') Remedy: disclose it. Sending only ONE of the two would remove the difference, but it would also change the sampler the model ships with, so the honest move is the footnote rather than a quieter benchmark. **`mlx-serve` accepts `top_p+top_k order` but honours it only yes, but top_k is applied BEFORE top_p**: The opposite order to both MTPLX arms, which apply top_p first (mtplx/sampling.py:121-130). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets. Applies on the AR path and on the MTP verify distribution alike. (generate.zig:8026 then :8034 (sampleTokenLazy), and :7663 then :7669 (probsAllPositions) -- mlx-serve v26.8.11 / 5afa398) Remedy: disclose it; no request field reorders an engine's own sampler. **`mlx-serve` accepts `seed` but honours it only partial**: The AR sampler and the accept-test PRNG ARE seeded (generate.zig:8054 seedKey, generate.zig:2422). But the correction/bonus token committed on EVERY MTP round is drawn with a null key -- MLX's process-global RNG, seeded once from the wall clock at main.zig:1078. So under --mtp a seeded request is NOT reproducible, and the harness runs mlx-serve with --mtp. (generate.zig:5268-5277, and the DEFAULT batched arm generate.zig:4827-4831 (mlx-serve v26.8.11 / 5afa398)) Remedy: no request field and no server flag fixes it. The seeded control is --mlxserve-no-mtp, which this harness already offers. Otherwise disclose it: mlx-serve's three seeds are REPEATS, not reproductions, and its spread is a sample of run-to-run variance rather than a seed effect. **`mlx-serve` accepts `top_p+top_k order` but honours it only yes, but top_k is applied BEFORE top_p**: The opposite order to both MTPLX arms, which apply top_p first (mtplx/sampling.py:121-130). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets. Applies on the AR path and on the MTP verify distribution alike. (generate.zig:8026 then :8034 (sampleTokenLazy), and :7663 then :7669 (probsAllPositions) -- mlx-serve v26.8.11 / 5afa398) Remedy: disclose it; no request field reorders an engine's own sampler. **`mlx-serve` accepts `seed` but honours it only partial**: The AR sampler and the accept-test PRNG ARE seeded (generate.zig:8054 seedKey, generate.zig:2422). But the correction/bonus token committed on EVERY MTP round is drawn with a null key -- MLX's process-global RNG, seeded once from the wall clock at main.zig:1078. So under --mtp a seeded request is NOT reproducible, and the harness runs mlx-serve with --mtp. (generate.zig:5268-5277, and the DEFAULT batched arm generate.zig:4827-4831 (mlx-serve v26.8.11 / 5afa398)) Remedy: no request field and no server flag fixes it. The seeded control is --mlxserve-no-mtp, which this harness already offers. Otherwise disclose it: mlx-serve's three seeds are REPEATS, not reproductions, and its spread is a sample of run-to-run variance rather than a seed effect. **`mlx-serve` accepts `top_p+top_k order` but honours it only yes, but top_k is applied BEFORE top_p**: The opposite order to both MTPLX arms, which apply top_p first (mtplx/sampling.py:121-130). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets. Applies on the AR path and on the MTP verify distribution alike. (generate.zig:8026 then :8034 (sampleTokenLazy), and :7663 then :7669 (probsAllPositions) -- mlx-serve v26.8.11 / 5afa398) Remedy: disclose it; no request field reorders an engine's own sampler. **`mlx-serve` accepts `seed` but honours it only partial**: The AR sampler and the accept-test PRNG ARE seeded (generate.zig:8054 seedKey, generate.zig:2422). But the correction/bonus token committed on EVERY MTP round is drawn with a null key -- MLX's process-global RNG, seeded once from the wall clock at main.zig:1078. So under --mtp a seeded request is NOT reproducible, and the harness runs mlx-serve with --mtp. (generate.zig:5268-5277, and the DEFAULT batched arm generate.zig:4827-4831 (mlx-serve v26.8.11 / 5afa398)) Remedy: no request field and no server flag fixes it. The seeded control is --mlxserve-no-mtp, which this harness already offers. Otherwise disclose it: mlx-serve's three seeds are REPEATS, not reproductions, and its spread is a sample of run-to-run variance rather than a seed effect. **`upstream-2.10.2` accepts `top_p+top_k order` but honours it only yes, but top_p is applied BEFORE top_k**: Both values are applied verbatim, but the FILTER ORDER differs from mlx-serve, which applies top_k first (generate.zig:8026 then :8034 on the AR path, :7663 then :7669 on the MTP verify path). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets, so this is a real difference in the sampled distribution, not a formality. It is inherent to the engines and is not something the request can equalise. (mtplx/sampling.py:121-130 -- 'Local mlx_lm applies top-p before top-k, so MTPLX's NumPy reference path mirrors that order') Remedy: disclose it. Sending only ONE of the two would remove the difference, but it would also change the sampler the model ships with, so the honest move is the footnote rather than a quieter benchmark. **`upstream-2.10.2` accepts `top_p+top_k order` but honours it only yes, but top_p is applied BEFORE top_k**: Both values are applied verbatim, but the FILTER ORDER differs from mlx-serve, which applies top_k first (generate.zig:8026 then :8034 on the AR path, :7663 then :7669 on the MTP verify path). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets, so this is a real difference in the sampled distribution, not a formality. It is inherent to the engines and is not something the request can equalise. (mtplx/sampling.py:121-130 -- 'Local mlx_lm applies top-p before top-k, so MTPLX's NumPy reference path mirrors that order') Remedy: disclose it. Sending only ONE of the two would remove the difference, but it would also change the sampler the model ships with, so the honest move is the footnote rather than a quieter benchmark. **`upstream-2.10.2` accepts `top_p+top_k order` but honours it only yes, but top_p is applied BEFORE top_k**: Both values are applied verbatim, but the FILTER ORDER differs from mlx-serve, which applies top_k first (generate.zig:8026 then :8034 on the AR path, :7663 then :7669 on the MTP verify path). At top_k=20 / top_p=0.95 the two orders can admit different candidate sets, so this is a real difference in the sampled distribution, not a formality. It is inherent to the engines and is not something the request can equalise. (mtplx/sampling.py:121-130 -- 'Local mlx_lm applies top-p before top-k, so MTPLX's NumPy reference path mirrors that order') Remedy: disclose it. Sending only ONE of the two would remove the difference, but it would also change the sampler the model ships with, so the honest move is the footnote rather than a quieter benchmark.
- **F-version engine builds, receipt-preferred.** 3/3 arm(s) recorded their own version; the rest fall back to the hand-maintained string in the renderer and are marked as such. `branch-fullstack`: **2.10.1** (mlx 0.32.2, git 5c62e89d9fe6) from the receipt (`importlib.metadata:mtplx`); `mlx-serve`: **26.8.11** (mlx 0.32.2) from the receipt (`mlx-serve --version`); `upstream-2.10.2`: **2.10.2** (mlx 0.32.2, git 8bc4d88e421d) from the receipt (`importlib.metadata:mtplx`).
- **F-mtp mlx-serve's controller switched its own speculative decoding off during some requests.** vanity: speculation active in 12 of 18 requests, pooled accept 49.3% (4680/9487 tok), 0.83 tok/round. Both MTPLX arms ran MTP depth 3 throughout. `runtime_disabled` is the state at the END of a request, so tokens generated before the controller gave up were still speculated -- read this as "speculation disabled at some point during the request", NOT as "speculation abandoned". The mixed picture matters: at 16K, the headline cell, speculation was live in 2 of 3 requests. It does mean the sweep decode column is not a clean MTP-vs-MTP comparison, and it leaves open whether pinning `--mtp-depth 3` would help at all if the controller disables speculation anyway.
- **F-eng engagement provenance is asymmetric, by disclosure not by deletion.** MTPLX drafted/accepted counts were not written into the <=16K records (the harness computed them and dropped them before writing); they are recorded from the 32K cells and the branch full-stack re-run onward. mlx-serve's <=16K engagement IS recoverable and has been scraped from its retained per-cell logs, and is marked **log-derived** in the engagement table. MTPLX's <=16K cells read `not_recorded`. Upstream's <=16K cells stay footnoted rather than re-run.
- **F-cap mlx-serve ran at MEMORY PARITY, not at its own defaults.** `MLX_SERVE_CACHE_LIMIT=107374182400` (100 GiB) was set on every cell at every size, matching the 100 GiB cap on both MTPLX arms. mlx-serve's SHIPPED default pool cap is **8 GB** (its startup line reads `[mem] MLX buffer-pool cap 8192 MB`), so these numbers are not what an out-of-the-box mlx-serve would produce. Note this is MLX's reclaimable buffer pool, matched numerically to the MTPLX wired cap rather than being the same kind of limit.
- **F-auto the branch auto-arms four M4 routes at SERVER DEFAULTS; upstream cannot.** `mtplx/server/openai.py:846-869` stamps `MTPLX_COMPILED_VERIFY`, `MTPLX_QWEN4_FIXED_M4_VERIFY`, `MTPLX_QWEN4_M4_STAGE3` and `MTPLX_QSA_M4_FUSED_KV_GATHER` via `setdefault` whenever `_served_model_is_qwen4_fixed_m4()` holds; the predicate (`qwen4_fixed_verify.py:25`) pins one exact geometry and returns **True** for the served pack. Upstream v2.10.2 has no such block (zero references to any of the four). An explicit operator export still wins. **So both branch rows already contain the M4 stack, and 'server defaults' does NOT mean the same thing on the two arms** -- the difference between the branch rows is frspec plus the MTPLX_FABLE_* switches, not M4. Corroborated without any log: `BRANCH_ENV` sets the three `MTPLX_QWEN4_M4_ROUTED_*` children but not stage3, and `qwen4_m4_stage3.py:231-236` raises `ValueError("qwen4 M4 child routes require M4 stage3")` on that combination via `runtime.py:1104`; the branch server booted cleanly in every cell, so stage3 was on.
  - **Resolved Fable flag set per branch arm, read from each receipt's own `fable_flags` block** -- the `--fable-flags-file` resolution, which is the `MTPLX_FABLE_*` tuning set plus any other `MTPLX_` key a flags file named. Keys sorted, values as launched:
    - `branch-fullstack` -- branch (+FR-Spec, +compiled MTP prepare, +restack)
      - cells vanity, 1K, 8K, 16K, 32K, 64K, 128K, 255K -- from flag file(s): `stack.flags` (sha256 `73b63272`, 12 key(s)), `battery-prefill.flags` (sha256 `f02e6895`, 8 key(s))
        - decode keys (14): `MTPLX_FABLE_BLOCK_VERIFY=1`, `MTPLX_FABLE_DRAFT_K20_PRESCATTER=1`, `MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1`, `MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS=3`, `MTPLX_FABLE_HC_M4=1`, `MTPLX_FABLE_OPDIET=1`, `MTPLX_FABLE_PLE_FIRST_GATHER_EARLY=1`, `MTPLX_FABLE_QSA_SPARSE_DECODE=1`, `MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS=17`, `MTPLX_FABLE_QSA_SPARSE_DECODE_TILE=128:32`, `MTPLX_FABLE_ROUTE_KERNEL=1`, `MTPLX_FABLE_VERIFY_GLUE=1`, `MTPLX_FABLE_VERIFY_GLUE_ITEMS=qsa_rope,qsa_rope_idx`, `MTPLX_SESSION_BANK_MAX_BYTES=8G`
        - prefill keys (6): `MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1`, `MTPLX_FABLE_PREFILL_MASK_FUSE=1`, `MTPLX_FABLE_PREFILL_QSA_QUERY_TILE=2048`, `MTPLX_GDN_BLOCKED_PREFILL=1`, `MTPLX_PREFILL_CHUNK_SIZE=4096`, `MTPLX_QSA_PREFILL_COMPILE_ROWS=4096`
        - boot keys (1, applied to every branch arm; a boot requirement, not retained tuning): `MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH=1`
        - **`dropped_vs_default` (1)**: `MTPLX_FABLE_COMPILED_DRAFT` -- in the harness default set, not named by the flag file(s), so **OFF** on this run.
- **F-fan cooling regime constant across every cell: fans at max.** Recorded per cell and in each session header.
- **F-seeds the rows are NOT equally weighted.** Seeds per cell, from the records: **vanity**: `branch-fullstack` n=3, `mlx-serve` n=3, `upstream-2.10.2` n=3; **1K**: `branch-fullstack` n=3, `mlx-serve` n=3, `upstream-2.10.2` n=3; **8K**: `branch-fullstack` n=3, `mlx-serve` n=3, `upstream-2.10.2` n=3; **16K**: `branch-fullstack` n=3, `mlx-serve` n=3, `upstream-2.10.2` n=3; **32K**: `branch-fullstack` n=3, `mlx-serve` n=3, `upstream-2.10.2` n=3; **64K**: `branch-fullstack` n=3, `mlx-serve` n=3, `upstream-2.10.2` n=3; **128K**: `branch-fullstack` n=2, `mlx-serve` n=2, `upstream-2.10.2` n=2; **255K**: `branch-fullstack` n=1, `mlx-serve` n=1, `upstream-2.10.2` n=1. The battery drops seeds as the context grows, so 6 cell(s) carry fewer than the 3 seeds the smallest cells got: 128K/branch-fullstack, 128K/mlx-serve, 128K/upstream-2.10.2, 255K/branch-fullstack, 255K/mlx-serve, 255K/upstream-2.10.2. A one-seed cell is a single measurement -- it has no spread, its `±`/range is absent by construction rather than by agreement, and it must not be read as the same kind of mean as a three-seed row. Every value in the tables carries its own `n=`.
- **F7 most branch switches are launch-env only.** Only `MTPLX_PREFILL_CHUNK_SIZE` and `MTPLX_GDN_BLOCKED_PREFILL` echo an engagement line; the rest are unverifiable from the server's output, and 15 of 20 are bare `os.environ.get` with defaults, so a typo silently no-ops.
- **PAGING** flagged in: branch-fullstack/128K, branch-fullstack/16K, branch-fullstack/1K, branch-fullstack/255K, branch-fullstack/32K, branch-fullstack/64K, branch-fullstack/8K, branch-fullstack/vanity, mlx-serve/128K, mlx-serve/16K, mlx-serve/1K, mlx-serve/255K, mlx-serve/32K, mlx-serve/64K, mlx-serve/8K, mlx-serve/vanity, upstream-2.10.2/128K, upstream-2.10.2/16K, upstream-2.10.2/1K, upstream-2.10.2/255K, upstream-2.10.2/32K, upstream-2.10.2/64K, upstream-2.10.2/8K — these cells are paging cells and must not be averaged in.

## Charts

### Prefill tok/s

```mermaid
%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '#0072B2, #CC79A7, #009E73'}}}}%%
xychart-beta
    title "Prefill tok/s by context size"
    x-axis "Context size (prompt tokens)" ["vanity", "1K", "8K", "16K", "32K", "64K", "128K", "255K"]
    y-axis "Prefill throughput (tokens/s)" 209.972538 --> 1949.572833
    line [354.939229, 854.856566, 1184.71385, 1191.093893, 1076.605674, 1015.748358, 978.665691, 952.204651]
    line [358.611096, 874.628434, 1205.126503, 1506.200421, 1316.124016, 1228.303881, 1216.07621, 1109.968431]
    line [407.042228, 1627.914111, 1804.606142, 1506.546399, 1139.611669, 804.855206, 571.772776, 437.020065]
```

_Series in plotting order (xychart-beta draws no legend, so the colours are pinned by the `plotColorPalette` directive at the top of the block): **1st line, blue `#0072B2`** = upstream v2.10.2 (control); **2nd line, reddish purple `#CC79A7`** = branch (+FR-Spec, +compiled MTP prepare, +restack); **3rd line, bluish green `#009E73`** = mlx-serve 26.8.11._

### Decode tok/s

```mermaid
%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '#0072B2, #CC79A7, #009E73'}}}}%%
xychart-beta
    title "Decode tok/s by context size"
    x-axis "Context size (prompt tokens)" ["vanity", "1K", "8K", "16K", "32K", "64K", "128K", "255K"]
    y-axis "Decode throughput (tokens/s)" 28.613127 --> 107.068669
    line [82.254438, 67.954169, 55.513053, 57.650043, 55.33565, 56.598386, 51.171819, 45.863632]
    line [100.530707, 86.330899, 82.550667, 79.606198, 78.014522, 76.922128, 73.359383, 66.543884]
    line [94.126155, 67.432898, 61.565385, 65.729766, 60.430379, 52.243511, 45.767877, 35.151088]
```

_Series in plotting order (xychart-beta draws no legend, so the colours are pinned by the `plotColorPalette` directive at the top of the block): **1st line, blue `#0072B2`** = upstream v2.10.2 (control); **2nd line, reddish purple `#CC79A7`** = branch (+FR-Spec, +compiled MTP prepare, +restack); **3rd line, bluish green `#009E73`** = mlx-serve 26.8.11._

### TTFT s

```mermaid
%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '#0072B2, #CC79A7, #009E73'}}}}%%
xychart-beta
    title "TTFT s by context size"
    x-axis "Context size (prompt tokens)" ["vanity", "1K", "8K", "16K", "32K", "64K", "128K", "255K"]
    y-axis "Time to first token (seconds)" 0 --> 657.424931
    line [0.265419, 1.397287, 7.120949, 13.971982, 30.718243, 64.93649, 134.5889, 275.476157]
    line [0.263734, 1.322185, 6.54466, 5.801843, 13.890963, 44.517978, 108.349709, 118.247827]
    line [0.237649, 0.65433, 4.572402, 10.91442, 28.796129, 81.485138, 229.331046, 597.680633]
```

_Series in plotting order (xychart-beta draws no legend, so the colours are pinned by the `plotColorPalette` directive at the top of the block): **1st line, blue `#0072B2`** = upstream v2.10.2 (control); **2nd line, reddish purple `#CC79A7`** = branch (+FR-Spec, +compiled MTP prepare, +restack); **3rd line, bluish green `#009E73`** = mlx-serve 26.8.11._

### Peak mem GB

```mermaid
%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '#0072B2, #CC79A7, #009E73'}}}}%%
xychart-beta
    title "Peak mem GB by context size"
    x-axis "Context size (prompt tokens)" ["vanity", "1K", "8K", "16K", "32K", "64K", "128K", "255K"]
    y-axis "Peak memory (GB)" 70.167526 --> 107.195569
    line [84.256787, 85.633392, 87.869288, 89.56801, 92.02476, 92.025103, 91.575391, 96.326972]
    line [84.669775, 85.895446, 92.080316, 96.894967, 100.769025, 101.751971, 97.208286, 104.109898]
    line [73.253196, 74.272108, 77.366171, 77.800192, 79.277849, 82.114963, 88.068193, 99.536147]
```

_Series in plotting order (xychart-beta draws no legend, so the colours are pinned by the `plotColorPalette` directive at the top of the block): **1st line, blue `#0072B2`** = upstream v2.10.2 (control); **2nd line, reddish purple `#CC79A7`** = branch (+FR-Spec, +compiled MTP prepare, +restack); **3rd line, bluish green `#009E73`** = mlx-serve 26.8.11._

### Peak footprint GB

```mermaid
%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '#0072B2, #CC79A7, #009E73'}}}}%%
xychart-beta
    title "Peak footprint GB by context size"
    x-axis "Context size (prompt tokens)" ["vanity", "1K", "8K", "16K", "32K", "64K", "128K", "255K"]
    y-axis "Peak process footprint (GB)" 71.062501 --> 115.10546
    line [85.583625, 87.686038, 91.665723, 94.648747, 98.023452, 98.797826, 100.787274, 105.237808]
    line [86.003225, 88.024357, 96.431534, 102.363728, 108.145849, 108.281088, 103.986119, 108.335678]
    line [74.732747, 75.878547, 79.788747, 81.243898, 84.211766, 88.404836, 96.746113, 111.435213]
```

_Series in plotting order (xychart-beta draws no legend, so the colours are pinned by the `plotColorPalette` directive at the top of the block): **1st line, blue `#0072B2`** = upstream v2.10.2 (control); **2nd line, reddish purple `#CC79A7`** = branch (+FR-Spec, +compiled MTP prepare, +restack); **3rd line, bluish green `#009E73`** = mlx-serve 26.8.11._

### Wall s

```mermaid
%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '#0072B2, #CC79A7, #009E73'}}}}%%
xychart-beta
    title "Wall s by context size"
    x-axis "Context size (prompt tokens)" ["vanity", "1K", "8K", "16K", "32K", "64K", "128K", "255K"]
    y-axis "Wall time (seconds)" 0 --> 689.236414
    line [2.719716, 15.458054, 22.376582, 29.730938, 49.276787, 75.649087, 146.866463, 280.660673]
    line [2.260207, 12.222829, 16.79265, 18.694188, 25.148988, 54.748626, 120.254065, 133.358422]
    line [2.338172, 9.879804, 19.943227, 24.837218, 42.142039, 97.371385, 251.751711, 626.784032]
```

_Series in plotting order (xychart-beta draws no legend, so the colours are pinned by the `plotColorPalette` directive at the top of the block): **1st line, blue `#0072B2`** = upstream v2.10.2 (control); **2nd line, reddish purple `#CC79A7`** = branch (+FR-Spec, +compiled MTP prepare, +restack); **3rd line, bluish green `#009E73`** = mlx-serve 26.8.11._

### Gen tokens

```mermaid
%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '#0072B2, #CC79A7, #009E73'}}}}%%
xychart-beta
    title "Gen tokens by context size"
    x-axis "Context size (prompt tokens)" ["vanity", "1K", "8K", "16K", "32K", "64K", "128K", "255K"]
    y-axis "Generated tokens (count)" 114.3 --> 1106.7
    line [202.333333, 951.666667, 845.666667, 910.333333, 1024, 596, 577, 225]
    line [198.666667, 932, 850, 1024, 875.333333, 780, 808, 821]
    line [197, 613.333333, 947.666667, 916, 806, 829.333333, 1024, 1024]
```

_Series in plotting order (xychart-beta draws no legend, so the colours are pinned by the `plotColorPalette` directive at the top of the block): **1st line, blue `#0072B2`** = upstream v2.10.2 (control); **2nd line, reddish purple `#CC79A7`** = branch (+FR-Spec, +compiled MTP prepare, +restack); **3rd line, bluish green `#009E73`** = mlx-serve 26.8.11._

### Prefill s (server)

```mermaid
%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '#0072B2, #CC79A7, #009E73'}}}}%%
xychart-beta
    title "Prefill s (server) by context size"
    x-axis "Context size (prompt tokens)" ["vanity", "1K", "8K", "16K", "32K", "64K", "128K", "255K"]
    y-axis "Prefill time, server-side (seconds)" 0 --> 657.229906
    line [0.246285, 1.204518, 6.916175, 13.756106, 30.436713, 64.526466, 133.936321, 274.226764]
    line [0.243783, 1.177865, 6.391059, 5.620815, 13.661416, 44.185799, 107.782731, 117.39433]
    line [0.213797, 0.629224, 4.540128, 10.87534, 28.753658, 81.425939, 229.238029, 597.501169]
```

_Series in plotting order (xychart-beta draws no legend, so the colours are pinned by the `plotColorPalette` directive at the top of the block): **1st line, blue `#0072B2`** = upstream v2.10.2 (control); **2nd line, reddish purple `#CC79A7`** = branch (+FR-Spec, +compiled MTP prepare, +restack); **3rd line, bluish green `#009E73`** = mlx-serve 26.8.11._

## Run conditions (per cell)

| Server | Cell | Power | Fan mode | Fan RPM | Max °C | Pressure | Memory source |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| branch-fullstack | 128K | AC | manual | 5348, 5776 | 79.2 | unknown | server:peak_memory_bytes |
| branch-fullstack | 16K | AC | manual | 5342, 5770 | 64.4 | unknown | server:peak_memory_bytes |
| branch-fullstack | 1K | AC | manual | 5348, 5771 | 59.6 | unknown | server:peak_memory_bytes |
| branch-fullstack | 255K | AC | manual | 5350, 5779 | 79.0 | unknown | server:peak_memory_bytes |
| branch-fullstack | 32K | AC | manual | 5342, 5779 | 72.4 | unknown | server:peak_memory_bytes |
| branch-fullstack | 64K | AC | manual | 5354, 5784 | 74.8 | unknown | server:peak_memory_bytes |
| branch-fullstack | 8K | AC | manual | 5360, 5773 | 62.2 | unknown | server:peak_memory_bytes |
| branch-fullstack | vanity | AC | manual | 5345, 5770 | 45.7 | unknown | server:peak_memory_bytes |
| mlx-serve | 128K | AC | manual | 5351, 5770 | 74.6 | unknown | server:/props peak_bytes |
| mlx-serve | 16K | AC | manual | 5347, 5774 | 66.2 | unknown | server:/props peak_bytes |
| mlx-serve | 1K | AC | manual | 5347, 5778 | 60.9 | unknown | server:/props peak_bytes |
| mlx-serve | 255K | AC | manual | 5354, 5780 | 74.4 | unknown | server:/props peak_bytes |
| mlx-serve | 32K | AC | manual | 5346, 5786 | 72.1 | unknown | server:/props peak_bytes |
| mlx-serve | 64K | AC | manual | 5347, 5777 | 74.7 | unknown | server:/props peak_bytes |
| mlx-serve | 8K | AC | manual | 5348, 5773 | 62.6 | unknown | server:/props peak_bytes |
| mlx-serve | vanity | AC | manual | 5344, 5771 | 45.3 | unknown | server:/props peak_bytes |
| upstream-2.10.2 | 128K | AC | manual | 5344, 5769 | 77.2 | unknown | server:peak_memory_bytes |
| upstream-2.10.2 | 16K | AC | manual | 5347, 5779 | 68.1 | unknown | server:peak_memory_bytes |
| upstream-2.10.2 | 1K | AC | manual | 5351, 5769 | 58.1 | unknown | server:peak_memory_bytes |
| upstream-2.10.2 | 255K | AC | manual | 5343, 5781 | 78.9 | unknown | server:peak_memory_bytes |
| upstream-2.10.2 | 32K | AC | manual | 5343, 5785 | 75.4 | unknown | server:peak_memory_bytes |
| upstream-2.10.2 | 64K | AC | manual | 5352, 5781 | 74.8 | unknown | server:peak_memory_bytes |
| upstream-2.10.2 | 8K | AC | manual | 5351, 5775 | 62.8 | unknown | server:peak_memory_bytes |
| upstream-2.10.2 | vanity | AC | manual | 5347, 5773 | 46.8 | unknown | server:peak_memory_bytes |

## Engagement (from responses, not launch flags)

**mlx-serve speculative decoding, per timed request** (log-derived; MTPLX has no equivalent for these cells, see F-eng)

| Cell | Request | Speculation | Source |
| --- | ---: | --- | --- |
| vanity | 1 | active | log:battery-mlxserve-A.server.log |
| vanity | 2 | active | log:battery-mlxserve-A.server.log |
| vanity | 3 | active | log:battery-mlxserve-A.server.log |
| vanity | 4 | active | log:battery-mlxserve-A.server.log |
| vanity | 5 | active | log:battery-mlxserve-A.server.log |
| vanity | 6 | active | log:battery-mlxserve-A.server.log |
| vanity | 7 | active | log:battery-mlxserve-A.server.log |
| vanity | 8 | active | log:battery-mlxserve-A.server.log |
| vanity | 9 | **disabled by end** | log:battery-mlxserve-A.server.log |
| vanity | 10 | active | log:battery-mlxserve-A.server.log |
| vanity | 11 | **disabled by end** | log:battery-mlxserve-A.server.log |
| vanity | 12 | **disabled by end** | log:battery-mlxserve-A.server.log |
| vanity | 13 | active | log:battery-mlxserve-A.server.log |
| vanity | 14 | active | log:battery-mlxserve-A.server.log |
| vanity | 15 | **disabled by end** | log:battery-mlxserve-A.server.log |
| vanity | 16 | **disabled by end** | log:battery-mlxserve-A.server.log |
| vanity | 17 | active | log:battery-mlxserve-A.server.log |
| vanity | 18 | **disabled by end** | log:battery-mlxserve-A.server.log |
| 128K | 1 | active | log:battery-mlxserve-C.server.log |

| Server | Cell | Engagement | Provenance |
| --- | --- | --- | --- |
| branch-fullstack | 128K | receipt: {"accept_rate": 0.5023, "accepted_drafts": 327, "drafted_tokens": 651, "generation_mode": "mtp"} | receipt |
| branch-fullstack | 16K | receipt: {"accept_rate": 0.4616, "accepted_drafts": 547, "drafted_tokens": 1185, "generation_mode": "mtp"} | receipt |
| branch-fullstack | 1K | receipt: {"accept_rate": 0.5358, "accepted_drafts": 389, "drafted_tokens": 726, "generation_mode": "mtp"} | receipt |
| branch-fullstack | 255K | receipt: {"accept_rate": 0.5313, "accepted_drafts": 467, "drafted_tokens": 879, "generation_mode": "mtp"} | receipt |
| branch-fullstack | 32K | receipt: {"accept_rate": 0.5245, "accepted_drafts": 557, "drafted_tokens": 1062, "generation_mode": "mtp"} | receipt |
| branch-fullstack | 64K | receipt: {"accept_rate": 0.5115, "accepted_drafts": 532, "drafted_tokens": 1040, "generation_mode": "mtp"} | receipt |
| branch-fullstack | 8K | receipt: {"accept_rate": 0.4838, "accepted_drafts": 344, "drafted_tokens": 711, "generation_mode": "mtp"} | receipt |
| branch-fullstack | vanity | receipt: {"accept_rate": 0.7473, "accepted_drafts": 139, "drafted_tokens": 186, "generation_mode": "mtp"} | receipt |
| mlx-serve | 128K | receipt: {"predicted_n": 1024, "prompt_n": 131072} | receipt |
| mlx-serve | 16K | receipt: {"predicted_n": 806, "prompt_n": 16384} | receipt |
| mlx-serve | 1K | receipt: {"predicted_n": 396, "prompt_n": 1024} | receipt |
| mlx-serve | 255K | receipt: {"predicted_n": 1024, "prompt_n": 261120} | receipt |
| mlx-serve | 32K | receipt: {"predicted_n": 1024, "prompt_n": 32768} | receipt |
| mlx-serve | 64K | receipt: {"predicted_n": 1024, "prompt_n": 65536} | receipt |
| mlx-serve | 8K | receipt: {"predicted_n": 1024, "prompt_n": 8192} | receipt |
| mlx-serve | vanity | receipt: {"predicted_n": 181, "prompt_n": 87} | receipt |
| upstream-2.10.2 | 128K | receipt: {"accept_rate": 0.463, "accepted_drafts": 375, "drafted_tokens": 810, "generation_mode": "mtp"} | receipt |
| upstream-2.10.2 | 16K | receipt: {"accept_rate": 0.5026, "accepted_drafts": 585, "drafted_tokens": 1164, "generation_mode": "mtp"} | receipt |
| upstream-2.10.2 | 1K | receipt: {"accept_rate": 0.5, "accepted_drafts": 546, "drafted_tokens": 1092, "generation_mode": "mtp"} | receipt |
| upstream-2.10.2 | 255K | receipt: {"accept_rate": 0.4958, "accepted_drafts": 119, "drafted_tokens": 240, "generation_mode": "mtp"} | receipt |
| upstream-2.10.2 | 32K | receipt: {"accept_rate": 0.4624, "accepted_drafts": 566, "drafted_tokens": 1224, "generation_mode": "mtp"} | receipt |
| upstream-2.10.2 | 64K | receipt: {"accept_rate": 0.4857, "accepted_drafts": 306, "drafted_tokens": 630, "generation_mode": "mtp"} | receipt |
| upstream-2.10.2 | 8K | receipt: {"accept_rate": 0.4437, "accepted_drafts": 378, "drafted_tokens": 852, "generation_mode": "mtp"} | receipt |
| upstream-2.10.2 | vanity | receipt: {"accept_rate": 0.7717, "accepted_drafts": 169, "drafted_tokens": 219, "generation_mode": "mtp"} | receipt |