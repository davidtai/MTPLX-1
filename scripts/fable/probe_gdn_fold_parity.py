#!/usr/bin/env python3
"""Per-cycle parity probe for the GDN keep-mask fold (W66b/W66d).

WHY THIS EXISTS
---------------
The fold-alone ABBA bracket (2026-09-02, receipts
``fable-w66b-gdn-fold-alone-*-1788400662..1788400673``) reported every
engagement counter exactly as the ring policy predicts -- installed, 35 folded
layers, ``windows == compiled_calls``, flushes 0.16-0.21/window, ring depth
never above the max -- and three seeds of DIFFERENT TEXT.  Nothing in the
receipt could distinguish "the fold ran and is not exact" from "the fold never
ran and the base was stale", because every counter in
``fable_gdn_keepmask_fold.STATS`` is HOST-SIDE RING BOOKKEEPING.  This probe
adds the two facts the receipt cannot carry:

* did ``prefix_gated_delta_update`` actually enter the compiled verify graph,
  for how many layers (``prefix_kernel_traced`` / ``prefix_consumed``), and
* at which decode cycle do the two arms' TOKEN IDS and each GDN layer's
  RECURRENT STATE first disagree, with the flush/decline/bypass status of that
  cycle and the two before it.

Both arms run in ONE process on ONE model load, greedy (temperature 0), so a
difference is the flag and nothing else.

WHAT IT DOES NOT DO
-------------------
Nothing here touches the GPU on its own account -- it runs the production
decode path, so it must be launched inside the serialized MLX window, as the
guard's DIRECT child (``consume_guard_attestation`` refuses otherwise).

COMMAND LINE
------------
::

    ROOT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w66b-gdn-fold-wire
    VENV=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
    cd $ROOT && env PYTHONPATH=$ROOT $VENV \
      /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
        --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
        --lock-timeout-seconds 1800 --child-timeout-seconds 3600 \
        -- $VENV $ROOT/scripts/fable/probe_gdn_fold_parity.py \
             --prompt-tokens 1024 --max-tokens 512

(``--model`` defaults to the pack ``abba_driver`` pins, which is the one the
server resolves; pass it only to point at a different pack.)

WINDOW BUDGET
-------------
Measured on 2026-09-02: ``--prompt-tokens 1024 --max-tokens 512`` runs **177**
compiled physical-M4 windows per arm (512 tokens at ~2.9 tokens/window),
comfortably over the ``--min-cycles 150`` floor, in ~9 s of decode per arm.
The floor is a REFUSAL, not a warning -- a run that recorded one window once
reported "no divergence" for a comparison it never made -- so if a future
stack accepts more tokens per window and the count falls under it, raise
``--max-tokens`` or move to ``--prompt-tokens 16384`` (the production cell)
rather than lowering the floor.

WHAT THE PROBE PERTURBS
-----------------------
It digests every GDN layer's recurrent state once per window, which FORCES the
fold's deferred lazy replay every cycle instead of letting the next window
drop it.  That changes nothing arithmetic -- ``pending_for`` keys on object
identity, not on whether the leaf is evaluated, so the ring evolves exactly as
it does unarmed -- but it does mean this script is a correctness instrument
and never a timing one.  Read speed off ``abba_window.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fable import abba_driver as driver  # noqa: E402


# --------------------------------------------------------------------------
# Per-cycle record
# --------------------------------------------------------------------------


class ArmRecorder:
    """Hooks that turn one decode into an ordered list of cycle records.

    One record per compiled physical-M4 window: the window's drafted token
    ids, the commit that followed it (or the all-accept branch that skipped
    one), the fold counter deltas that classify the cycle, and a digest of
    every GDN layer's recurrent state as it stands AFTER the commit -- i.e.
    the state the next window will start from, which is exactly the value the
    fold defers.
    """

    def __init__(self, *, label: str, state_dir: Path | None, keep_states: int):
        self.label = label
        self.state_dir = state_dir
        self.keep_states = int(keep_states)
        self.cycles: list[dict[str, Any]] = []
        self.prefix_kernel_traced = 0
        self.prefix_consumed: list[int] = []
        self._cache: Any = None
        self._gdn_indices: tuple[int, ...] = ()
        self._pending: dict[str, Any] | None = None
        self._restore: list[tuple[Any, str, Any]] = []

    # -- installation ----------------------------------------------------

    def install(self, runtime: Any) -> None:
        import numpy as np

        from mtplx import fable_gdn_keepmask_fold as fold
        from mtplx import graphbank as gb
        from mtplx.kernels import gdn_keepmask_fold as foldk
        from mtplx.models import qwen4_exp as qm

        self._np = np
        self._fold = fold

        text = getattr(runtime.model, "language_model", runtime.model)
        inner = text.model
        self._gdn_indices = tuple(
            index
            for index, layer in enumerate(inner.layers)
            if getattr(layer, "is_linear", False)
        )
        self._inner = inner

        # (a) the compiled window
        original_install = gb.CompiledVerifyBank.install_fixed_m4
        original_window = gb.CompiledVerifyBank._forward_installed_fixed_m4
        recorder = self

        def install_fixed_m4(bank, cache, **kwargs):
            recorder._cache = cache
            return original_install(bank, cache, **kwargs)

        def forward_window(bank, input_ids, *args, **kwargs):
            # Close the PREVIOUS window first.  Right now `entry.cache[1]`
            # holds that window's committed state -- the value this window is
            # about to start from, and the one the fold defers -- so this is
            # the only point where the digest means what the record says it
            # means.  Closing only in `finish()` kept ONE record for a whole
            # 177-window run (2026-09-02, gdn-fold-parity-1788400389).
            recorder.close_cycle()
            before = recorder._stats()
            try:
                ids = [int(token) for token in input_ids.reshape(-1).tolist()]
            except Exception:  # pragma: no cover - diagnostics only
                ids = []
            result = original_window(bank, input_ids, *args, **kwargs)
            recorder.open_window(ids, before)
            return result

        self._patch(gb.CompiledVerifyBank, "install_fixed_m4", install_fixed_m4)
        self._patch(
            gb.CompiledVerifyBank, "_forward_installed_fixed_m4", forward_window
        )

        # (b) the commit that follows it (absent on an all-accept cycle)
        original_commit = qm.Qwen4ExpTextModel.commit_verified_window

        def commit(model, cache, snapshot_states, *, keep_tokens, verified_tokens):
            before = recorder._stats()
            ok = original_commit(
                model,
                cache,
                snapshot_states,
                keep_tokens=keep_tokens,
                verified_tokens=verified_tokens,
            )
            recorder.note_commit(
                keep_tokens=keep_tokens,
                verified_tokens=verified_tokens,
                committed=ok,
                before=before,
                after=recorder._stats(),
            )
            return ok

        self._patch(qm.Qwen4ExpTextModel, "commit_verified_window", commit)

        # (c) the two facts the ABBA receipt could not carry: did the prefix
        #     kernel enter the graph, and did every folded layer take one.
        original_prefix = foldk.prefix_gated_delta_update

        def prefix_gated_delta_update(*call_args, **call_kwargs):
            recorder.prefix_kernel_traced += 1
            return original_prefix(*call_args, **call_kwargs)

        self._patch(
            foldk, "prefix_gated_delta_update", prefix_gated_delta_update, module=True
        )
        # GatedDeltaNet resolves the kernel module lazily and caches it, so
        # patch the already-bound reference too when it exists.
        cached = getattr(qm, "_GDN_FOLD_KERNELS", None)
        if cached is not None:
            self._patch(
                cached,
                "prefix_gated_delta_update",
                prefix_gated_delta_update,
                module=True,
            )

        original_assert = fold.assert_prefix_consumed

        def assert_prefix_consumed(scope, *, label):
            if scope is not None:
                recorder.prefix_consumed.append(int(getattr(scope, "consumed", -1)))
            return original_assert(scope, label=label)

        self._patch(
            fold, "assert_prefix_consumed", assert_prefix_consumed, module=True
        )
        self._patch(
            gb._gdn_fold,
            "assert_prefix_consumed",
            assert_prefix_consumed,
            module=True,
        )

    def _patch(self, target: Any, name: str, value: Any, *, module: bool = False):
        self._restore.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def uninstall(self) -> None:
        for target, name, value in reversed(self._restore):
            setattr(target, name, value)
        self._restore.clear()

    # -- per-cycle bookkeeping -------------------------------------------

    def _stats(self) -> dict[str, Any]:
        snapshot = self._fold.STATS
        stats: dict[str, Any] = {
            key: int(snapshot.get(key) or 0)
            for key in (
                "windows",
                "folded_windows",
                "deferred_commits",
                "flushes",
                "declines",
                "bypassed_commits",
            )
        }
        # The ring depth a window ENTERED at is the histogram key that
        # `note_window` incremented, which is the only place it is recorded.
        stats["ring_depth_hist"] = dict(snapshot.get("ring_depth_hist") or {})
        return stats

    @staticmethod
    def _entered_ring_depth(before: dict, after: dict) -> int | None:
        """The one histogram key that moved across this window, or None."""

        one, two = before.get("ring_depth_hist") or {}, after.get(
            "ring_depth_hist"
        ) or {}
        moved = [
            int(key)
            for key in set(one) | set(two)
            if int(two.get(key, 0)) - int(one.get(key, 0)) > 0
        ]
        return moved[0] if len(moved) == 1 else None

    # -- what the hooks call ----------------------------------------------

    def open_window(self, token_ids: list[int], before: dict) -> None:
        """Start a record for the window that just ran."""

        self._pending = {
            "cycle": len(self.cycles),
            "window_token_ids": list(token_ids),
            "stats_before": before,
            "commit": None,
        }

    def note_commit(
        self,
        *,
        keep_tokens: int,
        verified_tokens: int,
        committed: bool,
        before: dict,
        after: dict,
    ) -> None:
        """Attribute one commit to the window record still open, if any."""

        if self._pending is None:
            return
        self._pending["commit"] = {
            "keep_tokens": int(keep_tokens),
            "verified_tokens": int(verified_tokens),
            "committed": bool(committed),
            "flushed": after["flushes"] > before["flushes"],
            "declined": after["declines"] > before["declines"],
            "bypassed": after["bypassed_commits"] > before["bypassed_commits"],
            "deferred": after["deferred_commits"] > before["deferred_commits"],
        }

    def close_cycle(self) -> None:
        """Digest the committed state and close the window opened last."""

        import mlx.core as mx

        pending, self._pending = self._pending, None
        if pending is None:
            return
        # Classification first, and unconditionally: a cycle whose digest
        # cannot be taken (no cache stamped yet) is still a cycle, and losing
        # its flush/decline/depth is how a record becomes unreadable.
        after = self._stats()
        pending["stats_after"] = after
        pending["ring_depth_at_entry"] = self._entered_ring_depth(
            pending["stats_before"], after
        )
        cache = self._cache
        if cache is None:
            pending["state_digests"] = None
            self.cycles.append(pending)
            return
        states = [cache[index][1] for index in self._gdn_indices]
        mx.eval(*states)
        np = self._np
        digests = []
        arrays = []
        for state in states:
            host = np.asarray(state, copy=False)
            flat = host.reshape(-1)
            words = flat.view(np.uint32)
            digests.append(
                {
                    "xor": int(np.bitwise_xor.reduce(words)),
                    "stride": int(
                        np.bitwise_xor.reduce(words[::64] + np.arange(
                            words[::64].size, dtype=np.uint32
                        ))
                    ),
                    "absmax": float(np.abs(flat).max()),
                }
            )
            if self.state_dir is not None and pending["cycle"] < self.keep_states:
                arrays.append(host.copy())
        pending["state_digests"] = digests
        if arrays:
            path = self.state_dir / f"{self.label}-cycle{pending['cycle']:04d}.npz"
            np.savez(path, *arrays)
        self.cycles.append(pending)

    def finish(self) -> None:
        self.close_cycle()


# --------------------------------------------------------------------------
# Environment: the retained stack the ABBA bracket measured
# --------------------------------------------------------------------------

#: Exactly ``abba_window``'s CONTROL arm additions -- both arms carry them, so
#: only the fold flag moves between the two runs here.
EXTRA_ENV = {
    "MTPLX_FABLE_BLOCK_VERIFY": "1",
    "MTPLX_FABLE_DRAFT_K20_PRESCATTER": "1",
    "MTPLX_FABLE_HC_M4": "1",
    "MTPLX_FABLE_OPDIET": "1",
    "MTPLX_FABLE_ROUTE_KERNEL": "1",
    "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE": "1",
    "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL": "1",
    "MTPLX_QWEN4_M4_ROUTED_GLU": "1",
}


def retained_stack_args() -> argparse.Namespace:
    """``abba_driver``'s namespace for the retained arm, defaults elsewhere.

    ``--retain-events`` is the probe's ONE deviation from the measured arm's
    flags, and it goes through the driver's own supported route rather than a
    raw ``os.environ`` write.  The turbo profile sets ``MTPLX_DROP_EVENTS=1``;
    the probe reports ``stats.events`` as the independent cross-check that its
    own per-cycle hooks covered the whole run, and that list is populated only
    at 0.
    ``build_family_overrides`` puts it in ``family_overrides`` when the flag is
    set -- with a comment in the driver saying exactly why ("so the
    effective-environment drift check below still compares equal") -- so the
    expected environment carries the probe's value and the drift check below
    stays strict for every key, with no exclusion list.
    """

    parser = driver.build_parser()
    args = parser.parse_args(
        [
            "--label",
            "probe-gdn-fold-parity",
            "--sequence",
            "0",
            "--seed",
            "0",
            "--target-mode",
            "batched",
            "--require-compiled-verify",
            "--m4-stage3",
            "--qsa-fused-kv-gather",
            "--full-frspec",
            "--compiled-mtp-prepare",
            "--retain-events",
            "--max-tokens",
            "512",
        ]
    )
    return args


#: Keys the probe writes straight to ``os.environ``, outside the turbo
#: profile and outside ``build_family_overrides``.  None of them may collide
#: with the expected construction environment: a key that is BOTH in the
#: profile/family env and written raw here would make the drift check compare
#: the profile's value against the probe's, which is what
#: ``MTPLX_DROP_EVENTS`` did on 2026-09-02 (the profile sets 1, the probe
#: needs 0).  The fix for such a key is to route it through the driver -- see
#: ``retained_stack_args`` and ``--retain-events`` -- not to exclude it from
#: the check.  ``_probe_raw_env`` builds the dict and
#: ``_assert_probe_owns_no_expected_key`` enforces the rule.
def _probe_raw_env() -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "MTPLX_CONTEXT_WINDOW_TOKENS": "262144",
        "MTPLX_NGRAM_HOT_MB": "1024",
        "MTPLX_MEMORY_LIMIT_BYTES": str(driver.MEMORY_LIMIT_BYTES),
        "MTPLX_WIRED_LIMIT_BYTES": str(driver.WIRED_LIMIT_BYTES),
        "MTPLX_ADAPTIVE_DTEMP": "0",
        "MTPLX_STATE_REBASE_EVERY": "0",
        "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD": "0",
        "MTPLX_FRSPEC_DRAFT": "1",
        "MTPLX_FRSPEC_VOCAB": "builtin:qwen38-code-64k",
        "MTPLX_FUSED_HC": "1",
        **EXTRA_ENV,
    }


def _assert_probe_owns_no_expected_key(
    expected: dict[str, str], raw: dict[str, str]
) -> None:
    """Refuse a raw write to a key the expected environment already owns.

    Loud and actionable, because the alternative -- silently excluding the key
    from the drift check -- would weaken the check for a key the probe does
    NOT own the moment the profile's value and the probe's happen to agree.
    """

    collisions = {
        key: (expected[key], raw[key]) for key in raw if key in expected
    }
    if collisions:
        raise RuntimeError(
            "probe writes these keys raw but the turbo profile / family "
            f"overrides already own them: {collisions}; route them through "
            "abba_driver.build_family_overrides (see retained_stack_args and "
            "--retain-events) instead of writing os.environ directly"
        )


def apply_environment(args: argparse.Namespace) -> dict[str, Any]:
    """Build the retained construction environment and prove it is in force.

    The drift check is strict over EVERY key of the expected environment --
    the turbo profile plus ``build_family_overrides``, which already carries
    the probe's ``MTPLX_DROP_EVENTS=0`` because ``retained_stack_args`` passes
    ``--retain-events``.  Nothing is excluded from the comparison.
    """

    from mtplx.profiles import apply_profile_env, get_profile

    family_overrides, candidate_environment = driver.build_family_overrides(args)
    expected = get_profile("turbo").env_dict()
    expected.update(family_overrides)
    if expected.get("MTPLX_DROP_EVENTS") != "0":
        # `stats.events` is the probe's cross-check on its own per-cycle
        # hooks; a driver that stopped routing --retain-events through the
        # family overrides would leave that list empty and the cross-check
        # silently vacuous.
        raise RuntimeError(
            "the probe needs retained events: expected MTPLX_DROP_EVENTS=0 in "
            f"the family overrides, got {expected.get('MTPLX_DROP_EVENTS')!r}"
        )
    raw = _probe_raw_env()
    _assert_probe_owns_no_expected_key(expected, raw)
    for key in expected:
        os.environ.pop(key, None)
    apply_profile_env("turbo", runtime_env_overrides=family_overrides)
    os.environ.update(raw)
    observed = {key: os.environ.get(key) for key in expected}
    drift = {
        key: (expected[key], observed[key])
        for key in expected
        if observed[key] != expected[key]
    }
    if drift:
        raise RuntimeError(f"construction environment drifted: {drift}")
    return {
        "family_overrides": family_overrides,
        "candidate_environment": candidate_environment,
        "extra_environment": dict(EXTRA_ENV),
        "probe_raw_environment": dict(raw),
    }


def set_fold_flag(enabled: bool) -> None:
    """Flip the lane between the two in-process arms, everywhere it is cached."""

    from mtplx import fable_gdn_keepmask_fold as fold
    from mtplx.kernels import gdn_keepmask_fold as foldk
    from mtplx.models import qwen4_exp as qm

    os.environ["MTPLX_FABLE_GDN_KEEPMASK_FOLD"] = "1" if enabled else "0"
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.reset_stats()
    foldk.reset_prefix_caches()
    foldk.reset_exactness_probe_cache()
    # Read once at import in the GDN hot path; the probe is the only caller
    # that ever needs it to change inside a process.
    qm._GDN_KEEPMASK_FOLD_ARMED = bool(enabled)
    assert fold.fable_gdn_keepmask_fold_enabled() is bool(enabled)


# --------------------------------------------------------------------------
# One arm
# --------------------------------------------------------------------------


def run_arm(
    runtime: Any,
    *,
    label: str,
    enabled: bool,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    args: argparse.Namespace,
    state_dir: Path | None,
    keep_states: int,
) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx import fable_gdn_keepmask_fold as fold
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    driver.reset_run_caches(runtime, mx)
    set_fold_flag(enabled)

    recorder = ArmRecorder(
        label=label, state_dir=state_dir, keep_states=keep_states
    )
    recorder.install(runtime)
    sampler = SamplerConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    started = time.perf_counter()
    try:
        output = generate_mtpk(
            runtime,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_sampler=sampler,
            speculative_depth=3,
            seed=seed,
            stop_token_ids=set(),
            mtp_hidden_variant="post_norm",
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy=args.verify_strategy,
            verify_core="linear-gdn-from-conv-tape",
            draft_core=args.draft_core,
        )
    finally:
        recorder.finish()
        recorder.uninstall()
    elapsed = time.perf_counter() - started

    # `stats.events` is populated only with MTPLX_DROP_EVENTS=0 (the turbo
    # profile sets 1; `retained_stack_args` passes --retain-events).  The
    # per-cycle records below come from this probe's own hooks, and the event
    # count is the independent cross-check that they cover the whole run.
    events = list(getattr(output.stats, "events", None) or [])
    # The bank's own count of compiled physical-M4 windows.  The recorder must
    # produce exactly one record per window; `assess` refuses the run when it
    # does not, because a recorder that silently drops cycles reports "no
    # divergence" for a comparison it never made (2026-09-02).
    graphbank = dict(getattr(output.stats, "graphbank", None) or {})
    compiled_calls = int(
        dict(graphbank.get("compiled_verify") or {}).get("compiled_calls", 0)
    )
    return {
        "label": label,
        "compiled_calls": compiled_calls,
        "fold_enabled": bool(enabled),
        "elapsed_s": elapsed,
        "generation_events": len(events),
        "tokens": [int(token) for token in output.tokens],
        "text_head": output.text[:600] if hasattr(output, "text") else None,
        "cycles": recorder.cycles,
        "prefix_kernel_traced": recorder.prefix_kernel_traced,
        "prefix_consumed": recorder.prefix_consumed,
        "fold_stats": fold.stats_snapshot(),
    }


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def first_token_divergence(a: dict, b: dict) -> int | None:
    for index, (left, right) in enumerate(zip(a["cycles"], b["cycles"])):
        if left.get("window_token_ids") != right.get("window_token_ids"):
            return index
    return None


def first_state_divergence(a: dict, b: dict) -> tuple[int, int] | None:
    for index, (left, right) in enumerate(zip(a["cycles"], b["cycles"])):
        ld = left.get("state_digests") or []
        rd = right.get("state_digests") or []
        for layer, (one, two) in enumerate(zip(ld, rd)):
            if one["xor"] != two["xor"] or one["stride"] != two["stride"]:
                return index, layer
    return None


def describe(record: dict, index: int) -> dict[str, Any]:
    if index < 0 or index >= len(record["cycles"]):
        return {}
    cycle = record["cycles"][index]
    commit = cycle.get("commit")
    before = cycle.get("stats_before") or {}
    after = cycle.get("stats_after") or {}
    return {
        "cycle": index,
        "window_token_ids": cycle.get("window_token_ids"),
        "commit": commit,
        "ring_depth_at_entry": cycle.get("ring_depth_at_entry"),
        "windows_before": before.get("windows"),
        "flushes_total": after.get("flushes"),
        "declines_total": after.get("declines"),
        "bypassed_total": after.get("bypassed_commits"),
    }


def assess(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_cycles: int,
    folded_layers: int = 35,
) -> list[str]:
    """Every reason this run does not count.  Empty list == a usable answer.

    Pure: it reads the two arm records and nothing else, so the whole verdict
    is exercised on CPU.  The refusals exist because each one has already been
    a silent pass:

    * a run that recorded ONE cycle reported "no divergence" for a comparison
      it never made (gdn-fold-parity-1788400389, 2026-09-02) -- hence both the
      ``min_cycles`` floor and the recorder-vs-bank equality below;
    * an arm whose prefix kernel never entered the graph is the W66b defect
      itself, which every host-side counter reported as perfect.
    """

    problems: list[str] = []
    for name, record in (("control", control), ("candidate", candidate)):
        cycles = len(record.get("cycles") or [])
        compiled = int(record.get("compiled_calls") or 0)
        if cycles < int(min_cycles):
            problems.append(
                f"{name} recorded {cycles} windows, under --min-cycles "
                f"{int(min_cycles)} (the bank ran {compiled})"
            )
        # The instrument's own must-have-happened consequence.
        if compiled and cycles != compiled:
            problems.append(
                f"{name} recorded {cycles} cycles for {compiled} compiled M4 "
                "windows: the probe's per-window hook dropped records, so any "
                "per-cycle verdict below is vacuous"
            )

    traced = int(candidate.get("prefix_kernel_traced") or 0)
    if traced == 0:
        problems.append(
            "the candidate arm never called prefix_gated_delta_update: the "
            "fold's prefix did not enter the compiled verify graph"
        )
    consumed = list(candidate.get("prefix_consumed") or [])
    if not consumed:
        problems.append(
            "no traced verify reported a keep-mask prefix scope on the "
            "candidate arm"
        )
    elif any(int(value) != int(folded_layers) for value in consumed):
        problems.append(
            f"a traced verify bound only {consumed} of {int(folded_layers)} "
            "folded GDN layers to a prefix"
        )
    if int(control.get("prefix_kernel_traced") or 0) != 0:
        problems.append(
            "the control arm called prefix_gated_delta_update: the fold flag "
            "did not actually clear between the two in-process arms"
        )

    if list(control.get("tokens") or []) != list(candidate.get("tokens") or []):
        problems.append("the two arms produced different tokens")
    return problems


def max_abs_between(state_dir: Path, control: str, candidate: str, cycle: int):
    import numpy as np

    left = state_dir / f"{control}-cycle{cycle:04d}.npz"
    right = state_dir / f"{candidate}-cycle{cycle:04d}.npz"
    if not left.exists() or not right.exists():
        return None
    with np.load(left) as one, np.load(right) as two:
        worst = []
        for name in one.files:
            a, b = one[name], two[name]
            worst.append(float(np.abs(a.astype(np.float64) - b).max()))
    return {
        "per_layer_max_abs": worst,
        "max_abs": max(worst) if worst else 0.0,
        "layers_differing": sum(1 for value in worst if value != 0.0),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=driver.MODEL)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=1_024,
        choices=driver.PROMPT_TOKEN_CHOICES,
        help="one of abba_driver's pinned lengths; 16384 is the "
        "production cell and takes ~4x as long to prefill",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--min-cycles",
        type=int,
        default=150,
        help="fail if the run did not reach this many compiled M4 windows",
    )
    parser.add_argument(
        "--keep-states",
        type=int,
        default=12,
        help="cycles whose full GDN states are kept on disk for a max-abs diff",
    )
    parser.add_argument("--out", type=Path, default=None)
    options = parser.parse_args()

    guard = driver.consume_guard_attestation()

    args = retained_stack_args()
    args.max_tokens = options.max_tokens
    environment = apply_environment(args)

    from mtplx.server.openai import _apply_metal_memory_caps

    caps = _apply_metal_memory_caps(
        minimum_resident_bytes=driver.MINIMUM_RESIDENT_BYTES
    )
    if not caps.get("applied"):
        raise RuntimeError(f"Metal memory caps did not apply: {caps}")

    import mlx.core as mx

    from mtplx.draft_lm_head import _install_draft_lm_head
    from mtplx.runtime import load

    # The lane is decided per fixed-M4 installation, but qwen4_exp reads its
    # gate at IMPORT.  Load with the flag OFF so the control arm's import-time
    # constant is the shipped one; `set_fold_flag` moves it for arm B.
    set_fold_flag(False)
    runtime = load(options.model, mtp=True)
    _install_draft_lm_head(runtime, bits=4, group_size=64, mode="affine")

    prompt_ids = driver.build_production_prompt_ids(
        runtime.tokenizer, prompt_tokens=options.prompt_tokens
    )

    state_dir = Path(tempfile.mkdtemp(prefix="gdn-fold-parity-"))
    try:
        control = run_arm(
            runtime,
            label="control-fold-off",
            enabled=False,
            prompt_ids=prompt_ids,
            max_tokens=options.max_tokens,
            seed=options.seed,
            args=args,
            state_dir=state_dir,
            keep_states=options.keep_states,
        )
        candidate = run_arm(
            runtime,
            label="candidate-fold-on",
            enabled=True,
            prompt_ids=prompt_ids,
            max_tokens=options.max_tokens,
            seed=options.seed,
            args=args,
            state_dir=state_dir,
            keep_states=options.keep_states,
        )

        token_cycle = first_token_divergence(control, candidate)
        state_hit = first_state_divergence(control, candidate)
        report: dict[str, Any] = {
            "schema": "gdn-fold-parity-probe/1",
            "guard": guard,
            "environment": environment,
            "model": str(options.model),
            "prompt_tokens": len(prompt_ids),
            "max_tokens": options.max_tokens,
            "seed": options.seed,
            "arms": {
                name: {
                    "fold_enabled": record["fold_enabled"],
                    "cycles": len(record["cycles"]),
                    "compiled_m4_calls": record["compiled_calls"],
                    "elapsed_s": record["elapsed_s"],
                    "tokens": len(record["tokens"]),
                    "generation_events": record["generation_events"],
                    "text_head": record["text_head"],
                    # The two facts the ABBA receipt could not carry.
                    "prefix_kernel_traced": record["prefix_kernel_traced"],
                    "prefix_consumed": record["prefix_consumed"],
                    "fold_stats": record["fold_stats"],
                }
                for name, record in (
                    ("control", control),
                    ("candidate", candidate),
                )
            },
            "identical_tokens": control["tokens"] == candidate["tokens"],
            "first_token_divergence_cycle": token_cycle,
            "first_state_divergence": (
                None
                if state_hit is None
                else {"cycle": state_hit[0], "layer_position": state_hit[1]}
            ),
        }
        for name, index in (
            ("token", token_cycle),
            ("state", None if state_hit is None else state_hit[0]),
        ):
            if index is None:
                continue
            report[f"{name}_divergence_context"] = {
                "control": [
                    describe(control, index + offset) for offset in (-2, -1, 0)
                ],
                "candidate": [
                    describe(candidate, index + offset) for offset in (-2, -1, 0)
                ],
                "max_abs": max_abs_between(
                    state_dir, "control-fold-off", "candidate-fold-on", index
                ),
            }

        problems = assess(control, candidate, min_cycles=options.min_cycles)
        report["problems"] = problems
        report["ok"] = not problems

        text = json.dumps(report, indent=1, sort_keys=True)
        print("[gdn-fold-parity] " + json.dumps(
            {
                key: report[key]
                for key in (
                    "identical_tokens",
                    "first_token_divergence_cycle",
                    "first_state_divergence",
                    "problems",
                )
            },
            sort_keys=True,
        ), flush=True)
        out = options.out or (
            driver.OUT_DIR / f"gdn-fold-parity-{int(time.time())}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"[gdn-fold-parity] receipt {out}", flush=True)
        # Per-cycle detail is large; keep it beside the summary.
        detail = out.with_suffix(".cycles.json")
        detail.write_text(
            json.dumps(
                {"control": control["cycles"], "candidate": candidate["cycles"]},
                indent=1,
            )
        )
        print(f"[gdn-fold-parity] cycles  {detail}", flush=True)
        return 0 if report["ok"] else 1
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
