# Hy3 Router Public CLI Flags

Date: 2026-07-17
Status: Proposed for user review
Base: `origin/experiment/issue51-stack-69` at `9bf92ad`

## Problem

`ExpertStreamingConfig` already supports the Hy3 router selector and sigmoid
mode, and the depth-matrix benchmark already exposes them as
`--hy3-router-kernel` and `--hy3-router-sigmoid`. The public `mtplx serve`
expert-streaming CLI does not expose either field, so a production-style
server launch must use a JSON config file to select the row-owned MPP router or
fast sigmoid.

The public CLI needs explicit, validated overrides for these existing runtime
options. This change must not alter kernel implementations or existing default
behavior.

## Scope

Add two public expert-streaming flags:

- `--expert-hy3-router-kernel {stock,steel-r1-fused-r2,mpp-r1-fused-r2,mpp-fp32-splitk-r1-fused-r2,mpp-r1-last-arrival-fused-r2,mpp-row-owned-fused}`
- `--expert-hy3-router-sigmoid {precise,fast}`

Both parser defaults are `None`. An explicit flag overrides the corresponding
field in `--expert-streaming-config`; an omitted flag preserves the JSON value
or `ExpertStreamingConfig` default. Both flags are forwarded unchanged from
the public parent process to the daemon child.

The depth-matrix benchmark flags remain unchanged. No ambiguous `--mpp`
boolean alias will be added because the runtime has multiple distinct MPP
selectors.

## Data Flow

1. `add_expert_streaming_args` registers both public enum flags.
2. `expert_streaming_load_kwargs` maps explicit values to
   `hy3_router_kernel` and `hy3_router_sigmoid` overrides.
3. `ExpertStreamingConfig` performs the existing selector validation.
4. `append_expert_streaming_child_args` forwards explicit values to the
   daemon child.
5. The existing Hy3 load path passes the validated values to
   `configure_hy3_router_kernels`; no new runtime branch is introduced.

## Validation and Errors

- Unknown selector or sigmoid values fail in argument parsing.
- `fast` with any selector other than `mpp-row-owned-fused` fails through the
  existing `ExpertStreamingConfig` validation before model loading.
- Explicit CLI values override JSON config values consistently with the other
  expert-streaming flags.
- Omitted flags do not overwrite JSON config or change defaults.

## Floating-Point Contract and PR Disclosure

The PR body must explicitly discuss floating-point non-associativity:

- The MPP/split-K router changes how FP32 partial sums are grouped and reduced.
  Floating-point addition is non-associative, so legal re-grouping can change
  low bits even when input, accumulator, and output precision are unchanged.
- Fast sigmoid selects the row-owned kernel's FP32 fast exponential
  implementation. It changes the approximation used for the exponential; it
  does not introduce a lower-precision tensor representation, quantized
  storage, or BF16 accumulation.
- "Precision unchanged" does not mean "bitwise identical." Validity is based
  on the router's existing ID/order/weight parity or tolerance tests, plus the
  required end-to-end correctness and quality gates. The PR must state the
  measured result rather than imply associativity or exact equivalence.

This PR only exposes already-existing modes. It does not modify arithmetic,
reduction order, sigmoid code, or model precision.

## Test Strategy

Use test-driven development in `tests/test_expert_cli_runtime.py`:

1. Add failing parser-to-config tests for both explicit flags.
2. Add a failing precedence test proving explicit flags override JSON values.
3. Add a failing default test proving omitted flags preserve JSON/defaults.
4. Add a failing daemon-child forwarding test for both flags.
5. Add a failing invalid-pairing test for `fast` without
   `mpp-row-owned-fused`.
6. Implement the minimal CLI plumbing and make the focused tests pass.

Then run the focused public CLI tests, existing Hy3 router selection and
row-owned router tests, Ruff on changed Python files, and `git diff --check`.

## Hardware and Manual Acceptance

The acceptance sequence distinguishes kernel coverage from server regression:

1. Under exclusive GPU ownership, load a Hy3 model with
   `mpp-row-owned-fused` plus `fast` through the new public flags. Confirm the
   runtime reports the requested selector/mode, then run the existing hardware
   router and end-to-end correctness gates. This is the test that exercises
   the new kernel selection.
2. Capture the current `com.tea.qwen` service state. Temporarily run the task
   branch's `mtplx serve` implementation with the same Qwen 3.6 model and
   default backend address used by the existing LaunchAgent.
3. Health-check the Qwen backend and gateway, then run a manual Cline coding
   task through Cline's existing default-Qwen connection. Verify a coherent
   response, tool-use round trip, and clean server logs.
4. Restore the captured managed Qwen service after the regression smoke and
   verify its health. If the user explicitly asks to retain the task-branch
   server after the smoke, leave it only as a managed LaunchAgent, never as an
   orphan process.

The Qwen/Cline step is a generic public-CLI/server regression test. The
default `Qwen3.6-27B-MTPLX-Optimized-Speed` launch is not a Hy3 MoE model and
cannot exercise the Hy3 MPP or sigmoid kernel; the PR and final report must not
claim otherwise.

## Rollout and PR

- Keep the change on an isolated branch based on
  `experiment/issue51-stack-69`.
- Open a focused PR containing the two public flags, tests, and documentation
  only.
- The PR body includes the floating-point contract above, exact verification
  commands/results, and a clear separation between Hy3 kernel coverage and
  the Qwen/Cline regression smoke.

## Non-Goals

- Changing the default Hy3 router or sigmoid mode.
- Adding new MPP kernels or modifying the existing row-owned kernel.
- Claiming bitwise identity solely because precision is unchanged.
- Treating Qwen/Cline as evidence that a Hy3-only kernel executed.
- Adding aliases for every router selector.

## Failure-Mode Check

1. **Flags parse but never reach the model.** Severity: critical. Prevented by
   parser-to-config and parent-to-child survival tests plus runtime selector
   evidence during the Hy3 load.
2. **Fast sigmoid is paired with an incompatible router.** Severity: critical.
   Prevented by existing configuration validation and an explicit public-CLI
   regression test.
3. **The Qwen/Cline smoke is reported as Hy3 kernel validation.** Severity:
   critical. Prevented by separate acceptance stages and explicit PR wording.
4. **Temporary Qwen testing leaves the default service down or orphaned.**
   Severity: critical. Prevented by captured service state, managed restart,
   rollback-on-failure, and post-restore health verification.
