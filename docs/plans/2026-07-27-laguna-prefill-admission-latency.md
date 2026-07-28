# Laguna Prefill Admission Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers-optimized:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound Cline admission latency during leaderboard prefill while
preserving Laguna's unchanged fixed-M2 decode path and real-prompt prefill
throughput.

**Architecture:** A reproducible guarded benchmark starts the canonical Laguna
launcher on an isolated port, varies only the scheduler's construction-time
prefill chunk value, and measures a background-first Cline join. Promotion
changes the canonical launcher only after the largest candidate clears queue,
prefill, decode, parity, and M=2 gates.

**Tech Stack:** Python 3.12, zsh, FastAPI OpenAI-compatible HTTP/SSE API, pytest,
MLX/MLX-LM, launchd, existing `bench/laguna/run_guarded.py`.

**Assumptions:**

- Assumes the Laguna oQ4e artifact and Metal-capable host are available — the
  live gate will not run on a headless or model-free host.
- Assumes the guarded runner can unload and restore
  `com.tea.qwen` — the benchmark must not run concurrently with another GPU
  lock owner.
- Assumes the mutable settings endpoint applies `prefill_chunk_tokens` before
  the next idle pump is constructed — values must not change while requests
  are active or pending.
- Assumes Cline continues sending `X-MTPLX-Client: cline` — this plan does not
  address missing-header traffic-class misclassification.

**Execution result:** Tasks 1 and 2 completed on 2026-07-27. The guarded sweep
selected no candidate, so the conditional promotion/deployment steps in Task 3
were intentionally not executed. Production was restored healthy with the
unchanged 1,024-token launcher value.

---

## File structure

- `scripts/bench_laguna_prefill_admission.py`: isolated server lifecycle,
  deterministic background-first measurement, candidate aggregation, gate
  evaluation, and JSON receipt output.
- `tests/test_bench_laguna_prefill_admission.py`: offline tests for percentile,
  control bracketing, candidate rejection, and largest-passing selection.
- `scripts/start-laguna-s21.sh`: canonical promoted prefill chunk and startup
  report.
- `tests/test_start_laguna_s21_script.py`: launcher contract regression.
- `README.md`: supported default, latency rationale, rollback.
- `benchmarks/results/laguna-prefill-admission-m5max-2026-07-27.json`: exact
  guarded A/B receipt.
- `docs/specs/2026-07-27-laguna-prefill-admission-latency-design.md`: approved
  design and acceptance gates.

### Task 1: Add the reproducible admission benchmark

**Files:**

- Create: `scripts/bench_laguna_prefill_admission.py`
- Create: `tests/test_bench_laguna_prefill_admission.py`

**Security flag:** none

**Does NOT cover:** The benchmark selects a candidate but does not mutate the
canonical launcher or production launchd configuration.

- [x] **Step 1: Write failing aggregation and promotion tests**

```python
from scripts.bench_laguna_prefill_admission import summarize, select_candidate


def row(chunk, queue, prefill, decode, *, m2=True, parity=True):
    return {
        "prefill_chunk_tokens": chunk,
        "cline_queue_wait_s": queue,
        "background_prefill_tok_s": prefill,
        "aggregate_decode_tok_s": decode,
        "both_observed_m2": m2,
        "digest_parity": parity,
        "simultaneous_class_slots": True,
        "healthy_after": True,
    }


def test_selects_largest_candidate_clearing_every_gate():
    rows = [
        *[row(1024, 0.75, 100.0, 75.0) for _ in range(5)],
        *[row(512, 0.35, 99.0, 74.5) for _ in range(5)],
        *[row(256, 0.20, 97.0, 74.0) for _ in range(5)],
        *[row(128, 0.10, 91.0, 75.0) for _ in range(5)],
        *[row(1024, 0.72, 98.0, 74.0) for _ in range(5)],
    ]
    summary = summarize(rows)
    assert select_candidate(summary)["prefill_chunk_tokens"] == 256


def test_rejects_missing_m2_parity_health_or_simultaneous_state():
    rows = [row(1024, 0.75, 100.0, 75.0) for _ in range(10)]
    rows += [row(256, 0.20, 98.0, 74.0, m2=False) for _ in range(5)]
    assert select_candidate(summarize(rows)) is None
```

- [x] **Step 2: Run the focused tests and confirm the missing-module failure**

Run:

```bash
cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_bench_laguna_prefill_admission.py
```

Expected: collection fails because
`scripts.bench_laguna_prefill_admission` does not exist.

- [x] **Step 3: Implement exact benchmark contracts**

Implement:

```python
def percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def summarize(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["prefill_chunk_tokens"]), []).append(row)
    return {
        chunk: {
            "trials": trials,
            "cline_queue_p95_s": percentile(
                [float(row["cline_queue_wait_s"]) for row in trials], 0.95
            ),
            "background_prefill_median_tok_s": statistics.median(
                float(row["background_prefill_tok_s"]) for row in trials
            ),
            "aggregate_decode_median_tok_s": statistics.median(
                float(row["aggregate_decode_tok_s"]) for row in trials
            ),
            "all_receipts_valid": all(
                bool(row["both_observed_m2"])
                and bool(row["digest_parity"])
                and bool(row["simultaneous_class_slots"])
                and bool(row["healthy_after"])
                for row in trials
            ),
        }
        for chunk, trials in grouped.items()
    }


def select_candidate(summary: dict[int, dict[str, object]]) -> dict | None:
    control = summary[1024]
    prefill_floor = 0.95 * float(control["background_prefill_median_tok_s"])
    decode_floor = 0.95 * float(control["aggregate_decode_median_tok_s"])
    for chunk in sorted((value for value in summary if value != 1024), reverse=True):
        candidate = summary[chunk]
        if (
            float(candidate["cline_queue_p95_s"]) < 0.250
            and float(candidate["background_prefill_median_tok_s"]) >= prefill_floor
            and float(candidate["aggregate_decode_median_tok_s"]) >= decode_floor
            and bool(candidate["all_receipts_valid"])
        ):
            return {"prefill_chunk_tokens": chunk, **candidate}
    return None
```

The CLI must:

1. accept `--chunks 1024 512 256 128 1024`, `--trials 5`,
   `--port 8081`, and `--output`;
2. start `scripts/start-laguna-s21.sh` with
   `MTPLX_LAGUNA_PORT=<port>` and wait for `/health`;
3. require an idle scheduler before applying each setting through
   `POST /v1/mtplx/settings`;
4. verify the applied setting and the next snapshot configuration;
5. submit the deterministic cold background request first, wait until its
   class is active, then submit the deterministic Cline request;
6. poll until both classes are simultaneously active;
7. collect full SSE usage/stats receipts without writing generated text;
8. record SHA-256 output digests, real server prompt-token counts, queue wait,
   time to active, TTFT, prefill throughput, decode throughput, M=2 receipts,
   errors, and health;
9. terminate its child server in `finally`;
10. write raw rows, summary, selected candidate, environment, checkout commit,
    and exact command to the output JSON.

- [x] **Step 4: Run focused tests**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_bench_laguna_prefill_admission.py
```

Expected: all tests pass.

### Task 2: Run the guarded real-model A/B and choose the value

**Files:**

- Create:
  `benchmarks/results/laguna-prefill-admission-m5max-2026-07-27.json`

**Security flag:** none

**Does NOT cover:** A candidate that fails any gate must not be rounded,
explained away, or installed.

- [x] **Step 1: Verify the benchmark source and lock target**

Run:

```bash
cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf
git status --short --branch
lsof /tmp/mtplx-gpu-exclusive.lock || true
```

Expected: only this task's scoped files are modified; any existing lock owner
must finish before the benchmark starts.

- [x] **Step 2: Execute the isolated guarded sweep**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --api-url http://127.0.0.1:8080/v1/models \
  --timeout-seconds 300 \
  --lock-path /tmp/mtplx-gpu-exclusive.lock \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 3600 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  scripts/bench_laguna_prefill_admission.py \
  --chunks 1024 512 256 128 1024 \
  --trials 5 \
  --port 8081 \
  --model /Users/davidtai/.mtplx/models/mlx-community--Laguna-S-2.1-oQ4e \
  --output \
  benchmarks/results/laguna-prefill-admission-m5max-2026-07-27.json
```

Expected: exit zero, production restored on port 8080, and the receipt names
either the largest fully passing candidate or `null`.

- [x] **Step 3: Audit the receipt before promotion**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("benchmarks/results/laguna-prefill-admission-m5max-2026-07-27.json")
receipt = json.loads(path.read_text())
print(json.dumps({
    "selected_candidate": receipt["selected_candidate"],
    "summary": receipt["summary"],
    "errors": [row["error"] for row in receipt["rows"] if row.get("error")],
}, indent=2, sort_keys=True))
PY
```

Expected: no errors or contaminated rows. If `selected_candidate` is `null`,
stop with the unchanged 1,024-token launcher.

### Task 3: Promote, test, deploy, and record the measured default

**Files:**

- Modify: `scripts/start-laguna-s21.sh`
- Modify: `tests/test_start_laguna_s21_script.py`
- Modify: `README.md`
- Modify:
  `docs/specs/2026-07-27-laguna-prefill-admission-latency-design.md`

**Security flag:** none

**Does NOT cover:** The promoted value applies only to the supported Laguna
dual-lane launcher. It must not be generalized to Hy3, GLM, Qwen, serial
scheduling, or another prompt geometry.

- [ ] **Step 1: Change the focused launcher test first**

Replace the expected `--prefill-chunk-tokens 1024` fragment with the exact
selected candidate and add assertions that `--print-config` and startup logging
report the same value.

- [ ] **Step 2: Prove the launcher test fails before production code changes**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_start_laguna_s21_script.py
```

Expected: failure showing the launcher still contains 1,024.

- [ ] **Step 3: Make the minimal construction-time launcher change**

Change only the canonical command argument and matching startup log from 1,024
to the selected candidate. Do not add a per-request environment read,
eligibility branch, or fallback.

- [ ] **Step 4: Document the scoped promotion and rollback**

Record the selected value, the receipt path, queue p95, prefill delta, decode
delta, M=2/parity results, and `1024` rollback value in the README and design
execution record. Explicitly state that Blackwellboy's prior operational
changes remain applied and are not the source of this scheduler measurement.

- [ ] **Step 5: Run focused and full Laguna verification**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q \
  tests/test_start_laguna_s21_script.py \
  tests/test_server_openai.py \
  tests/test_laguna_fused.py \
  tests/test_laguna_model.py \
  tests/test_laguna_compiled_step.py
```

Expected: all tests pass.

- [ ] **Step 6: Deploy through the guarded GPU workflow and remeasure Cline**

Acquire `/tmp/mtplx-gpu-exclusive.lock`, restart
`gui/$(id -u)/com.tea.qwen`, verify `/health`, `/v1/models`, and the scheduler
snapshot report the promoted chunk, then run the same background-first Cline
join used by the benchmark.

Expected: Cline queue wait remains below 250 milliseconds, both class slots are
simultaneously active, both receipts report M=2, and service health remains
green.

- [ ] **Step 7: Commit and push the scoped upstream branch**

Run:

```bash
git status --short
git diff --check
git add \
  scripts/bench_laguna_prefill_admission.py \
  tests/test_bench_laguna_prefill_admission.py \
  scripts/start-laguna-s21.sh \
  tests/test_start_laguna_s21_script.py \
  README.md \
  docs/specs/2026-07-27-laguna-prefill-admission-latency-design.md \
  docs/plans/2026-07-27-laguna-prefill-admission-latency.md \
  benchmarks/results/laguna-prefill-admission-m5max-2026-07-27.json
git commit -m "perf: bound Laguna dual-lane admission latency"
git push origin perf/laguna-batch-kernels
```

Expected: the pushed branch head equals local `HEAD`, with no unrelated files
staged or committed.
