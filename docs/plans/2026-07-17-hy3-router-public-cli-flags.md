# Hy3 Router Public CLI Flags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing row-owned MPP router and fast-sigmoid modes through the public expert-streaming CLI, validate the Hy3 path, and regression-test the task branch through Qwen/Cline.

**Architecture:** `mtplx.expert_cli` remains the public parse, override, and daemon-forwarding boundary. The new flags map directly to existing `ExpertStreamingConfig` fields; no kernel or arithmetic code changes. Hy3 hardware provides kernel evidence, while Qwen/Cline is reported separately as a generic server smoke.

**Tech Stack:** Python 3.12, argparse, pytest, MLX/Metal, launchd, Cline.

**Assumptions:** Fast sigmoid remains valid only with `mpp-row-owned-fused`; Hy3 artifacts remain local; the default Qwen model is not treated as Hy3 kernel coverage.

---

## Files

- Modify `mtplx/expert_cli.py` for public parsing, explicit overrides, and child forwarding.
- Modify `tests/test_expert_cli_runtime.py` for TDD coverage.
- Modify `docs/specs/2026-07-17-hy3-router-public-cli-flags-design.md` to mark the controls experimental and document pros/cons.

### Task 1: Add the flags test-first

**Security flag:** `security` — public input validation and parent/child forwarding.

**Does NOT cover:** Kernel arithmetic, defaults, benchmark flags, or automatic promotion.

- [ ] Add failing tests proving:
  - `--expert-hy3-router-kernel mpp-row-owned-fused` and `--expert-hy3-router-sigmoid fast` override JSON.
  - Omitted flags preserve JSON values.
  - Fast sigmoid with an incompatible selector fails before model loading.
  - Both explicit flags survive daemon-child forwarding.
- [ ] Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q tests/test_expert_cli_runtime.py -k 'public_hy3_router or public_fast_sigmoid'
```

  Expected: fail because the public parser does not recognize the flags.
- [ ] Add these argparse choices in `add_expert_streaming_args`:

```python
group.add_argument(
    "--expert-hy3-router-kernel",
    choices=[
        "stock", "steel-r1-fused-r2", "mpp-r1-fused-r2",
        "mpp-fp32-splitk-r1-fused-r2",
        "mpp-r1-last-arrival-fused-r2", "mpp-row-owned-fused",
    ],
)
group.add_argument(
    "--expert-hy3-router-sigmoid",
    choices=["precise", "fast"],
)
```

- [ ] Map `expert_hy3_router_kernel` and `expert_hy3_router_sigmoid` to the existing config fields in `expert_streaming_load_kwargs`, and add both to `append_expert_streaming_child_args`.
- [ ] Run the full focused test file, Ruff, and `git diff --check`; then commit only the CLI and test files.

### Task 2: Verify Hy3 under the canonical GPU lock

**Security flag:** `none`

**Does NOT cover:** Promotion of fast sigmoid or claims that Qwen executes a Hy3 router.

- [ ] Wait for PID 46724 to acquire `/tmp/mtplx-gpu-exclusive.lock`; do not kill the current Hy3/GLM owners.
- [ ] Transfer the reservation into the repository Qwen guard without allowing an unrelated owner to enter the lane.
- [ ] Run row-owned kernel selection/hardware tests with `MTPLX_ROW_OWNED_HARDWARE=1`.
- [ ] Run one short Hy3 depth-3 smoke with `--hy3-router-kernel mpp-row-owned-fused --hy3-router-sigmoid fast` and save `/tmp/issue51-public-router-flags-fast-smoke.json`.
- [ ] Record exact selector, sigmoid mode, routing parity/tolerance, end-to-end gates, and Qwen restoration evidence.

### Task 3: Qwen/Cline smoke and experimental PR

**Security flag:** `security` — temporary local service replacement and agent tool-use boundary.

**Does NOT cover:** Hy3 kernel validation; the Qwen model is dense and cannot exercise this router.

- [ ] Capture the `com.tea.qwen` plist hash, launchd state, process command, model ID, and backend/gateway health without printing credentials.
- [ ] Temporarily start Qwen 3.6 from this task branch on the existing backend address, leaving Cline's default model connection unchanged.
- [ ] In Cline, run a read-only coding task that identifies the two new public flags and their valid pairing. Require one successful file-read tool round trip and clean server logs.
- [ ] Restore the captured managed Qwen service and health-check it; leave no orphan process.
- [ ] Push the branch and open a draft PR against `experiment/issue51-stack-69`.
- [ ] The PR body must label the feature experimental and include:
  - Pros: opt-in production-style selection, row-owned protocol-free dispatch, tile-major locality, potential fast-exp savings.
  - Cons: Hy3-only M2–M8 scope, prepared-weight memory, hardware dependence, fast-exp approximation, and low-bit differences from non-associative FP32 reduction grouping.
  - Precision statement: precision is unchanged, but unchanged precision does not imply bitwise identity.
  - Separate Hy3 kernel evidence from Qwen/Cline server-regression evidence.
