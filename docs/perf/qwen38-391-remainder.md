# Qwen3.8 Flash-Next: the PR 391 remainder onto MTPLX 2.11.2

MTPLX 2.11.2 re-landed nine of PR 391's twelve Flash-Next decode keys under the
`MTPLX_QWEN4_*` / `MTPLX_QSA_*` namespace. On the 16K decode cell the release
default reaches 71.17 tok/s against 80.92 for the full 391 stack and 57.65 for
2.10.2 on the same instrument — the nine keys recovered about 60% of 391's
gain. This change ports the remaining lanes so the ~9.75 tok/s difference is
reachable. Each is default-on for a served fixed-M4 Flash-Next pack (stamped in
the `mtplx/server/openai.py` auto-arm block, gated on the fixed-M4 config
predicate) with a per-key `=0` opt-out through the existing pop loop, and each
old `MTPLX_FABLE_*` name is honoured as an alias when the new key is unset.

All measurements run under the Sustained profile (the release self-selects it
for these packs); the numbers below are from 391's own campaign on the
production geometry and are re-measured in the GPU phase.

| Lane | Key | What it does | Exactness | 391 recovered |
|------|-----|--------------|-----------|---------------|
| HC_M4 | `MTPLX_QWEN4_HC_M4` | Runs the verify-width (2..8 row) hyper-connection read as one multi-threadgroup GEMV over a kernel-private 8-bit pack instead of the eager norm/down/up chain. | rounding-class | ~-1.1% cycle |
| prefill mask fuse | `MTPLX_QWEN4_PREFILL_MASK_FUSE` | Sends the dense QSA prefill chunk through MLX's fused SDPA (causal string or bool selection) instead of a materialized `[H,S,T]` score tensor MLX's head-dim-256 heuristic otherwise declines. | rounding-class (exact visible set) | part of the prefill share |
| QSA prefill query tile | `MTPLX_QSA_PREFILL_QUERY_TILE` | Tiles only the dense QSA attention query rows so a wide prefill chunk keeps the narrow chunk's attention peak and `sum(rows x context)` cost. | rounding-class (exact visible set) | part of the prefill share |
| QSA sparse split-K decode | `MTPLX_QSA_SPARSE_DECODE` | Native split-K sparse-GQA decode kernel that reads the selected KV rows once instead of materializing a gathered `[1,2,4,2052,256]` K/V pair per QSA layer per verify cycle. | rounding-class | ~-1.46 ms/cycle at 16K |
| graph-build overlap | `MTPLX_QWEN4_GRAPH_BUILD_OVERLAP` | Submits the PLE-independent prefix of the fixed-M4 verify graph early so its GPU work overlaps the ~1.4 ms/cycle host build of the rest. | exact | ~1.4 ms/cycle (~4.5% at N=3) |

Status in this change: HC_M4, the prefill mask fuse, and the QSA prefill query
tile are ported and CPU-tested. The QSA sparse split-K decode and the graph-build
overlap are covered in the port report (`.benchmark-artifacts/over100-reports/
remainder-port-report.md`) — the decode lane's substrate is present upstream and
is a native+wiring port whose kernel needs the GPU lock to build and parity-probe;
the graph-build overlap is blocked because upstream re-landed the fixed-M4 verify
as a single compiled graph, without the prefix/suffix split the lane rides.
