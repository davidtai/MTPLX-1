# DFlash2 Adaptive Drafting Default

## Scope

Make Qwen3.8 DFlash2 adaptive drafting the default at every supported prompt
length. Add one process-level Boolean option that selects adaptive M=1–8 or
fixed M=8 consistently across CLI commands, serving, direct Python loading,
and the final benchmark matrix.

This removes the current automatic rule that disables adaptive drafting at
16,384 prompt tokens and above. It does not change the adaptive policy's
position-EMA algorithm, proposal rows, target sampling, DFlash checkpoint, or
the retained M5–M8 kernels.

## User interface

All public DFlash2 entry points that use the shared CLI argument helper expose:

- `--dflash2-adaptive`: use adaptive M=1–8 drafting. This is the default.
- `--no-dflash2-adaptive`: disable the adaptive policy and use fixed M=8.

The setting is process-level. It is not mutable per request because the
adaptive policy factory is installed on the shared target model.

Direct Python callers receive the same default through
`DFlash2RuntimeConfig.draft_adaptive=True`. They can construct a runtime with
`draft_adaptive=False` for fixed M=8.

## Architecture and data flow

1. The shared DFlash2 CLI argument helper parses the Boolean option for
   `run`, `ask`, `serve`, and server-start flows.
2. Any command that spawns a child server forwards the explicitly resolved
   Boolean value.
3. DFlash2 runtime construction stores the value in `DFlash2RuntimeConfig`.
4. The measured Qwen3.8 feature installer and per-generation context-route
   refresh read that value. Prompt length no longer changes whether adaptive
   drafting is active.
5. Adaptive mode installs rows 11 and 15 and permits physical proposal widths
   M=1–8. Fixed mode removes the adaptive policy factory and requests M=8.
6. Runtime telemetry and benchmark receipts record the requested mode, the
   effective mode, and observed draft widths.

The existing serialized DFlash2 generation lock remains the ownership
boundary for applying the route. No request may mutate the process-level
selection.

## Contracts

- Default: adaptive enabled for every prompt length.
- Explicit opt-out: `--no-dflash2-adaptive` produces fixed M=8.
- Prompt length affects other measured prefill/decode mechanisms only; it does
  not override the adaptive selection.
- Server subprocesses must preserve the parent selection.
- Direct loads without a CLI namespace default to adaptive.
- Receipts must make a missing or incorrectly propagated selection visible.

## Error handling

The parser rejects malformed Boolean spellings using normal `argparse`
behavior. Runtime configuration normalizes the value to `bool`. An arm that
requests fixed mode but reports adaptive widths, or requests adaptive mode but
does not install the policy, fails its benchmark correctness checks.

## Testing strategy

Tests are written before implementation and cover:

- CLI defaults and both Boolean spellings on every shared entry point.
- Child/server command propagation.
- Runtime-config default and explicit fixed mode.
- Short and long context routing with adaptive enabled by default.
- Fixed M=8 behavior at short and long contexts when explicitly disabled.
- Feature-receipt requested/effective fields.
- Final benchmark child commands and aggregate receipts.
- Existing DFlash2 backend, server, adaptive-policy, and Qwen3.8 focused suites.

Real-model verification runs the corrected greedy `is_palindrome` burst first,
then the 1K/16K/64K/128K cold-prefill matrix. Every PR arm uses adaptive mode
explicitly and records observed widths.

## Rollout

This ships on `perf/qwen38-challenge-port` in the existing PR. There is no
separate PR or compatibility migration. Fixed M=8 remains available through
the opt-out flag for comparisons and operational fallback.

## Failure-mode review

1. **Child launch drops the option.** Severity: critical. A server could look
   adaptive in CLI state but run fixed M=8. Mitigation: command-construction
   tests plus requested/effective telemetry.
2. **Long-context routing silently restores the old cutoff.** Severity:
   critical. Mitigation: route tests at 1K, 16K, 64K, and 128K for both flag
   values.
3. **Concurrent requests mutate a shared policy.** Severity: critical.
   Mitigation: selection is immutable process configuration; route refreshes
   only reapply that selection while holding the existing generation lock.

## Non-goals

- Per-request adaptive toggles.
- Retuning the position-EMA constants.
- Choosing a different fixed width.
- Replacing the DFlash2 checkpoint or target sampler.
