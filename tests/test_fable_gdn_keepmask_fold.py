"""Pure-Python tests for the GDN keep-mask fold (MTPLX_FABLE_GDN_KEEPMASK_FOLD).

Nothing here imports ``mlx`` or evaluates an array: the ring policy, the mask,
the byte accounting and the contract gate are all host logic, and the one place
that has to agree with a Metal kernel is checked by comparing SOURCE TEXT
against the installed ``mlx_lm``.  The arithmetic parity that needs a GPU lives
in ``scripts/fable/micro_gdn_keepmask_fold.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mtplx import fable_gdn_keepmask_fold as fold

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(fold.ENV_FLAG, raising=False)
    monkeypatch.delenv(fold.ENV_WINDOWS, raising=False)
    monkeypatch.delenv(fold.ENV_LOG, raising=False)
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.reset_stats()
    yield
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.reset_stats()


# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------


def test_flag_defaults_off():
    assert fold.fable_gdn_keepmask_fold_enabled() is False


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_flag_arms(monkeypatch, raw):
    monkeypatch.setenv(fold.ENV_FLAG, raw)
    fold.reset_fable_gdn_keepmask_fold_cache()
    assert fold.fable_gdn_keepmask_fold_enabled() is True


def test_flag_is_read_once(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    assert fold.fable_gdn_keepmask_fold_enabled() is True
    monkeypatch.setenv(fold.ENV_FLAG, "0")
    # A mid-cycle environment change must not move the lane: two traces of the
    # same compiled graph would then disagree about their recurrence.
    assert fold.fable_gdn_keepmask_fold_enabled() is True


def test_default_ring_depth_and_prefix_width():
    assert fold.fable_gdn_keepmask_fold_windows() == fold.DEFAULT_MAX_WINDOWS
    assert fold.prefix_rows() == fold.VERIFY_WIDTH * fold.DEFAULT_MAX_WINDOWS


@pytest.mark.parametrize("value", [0, 5, -1])
def test_bad_ring_depth_raises(monkeypatch, value):
    monkeypatch.setenv(fold.ENV_WINDOWS, str(value))
    fold.reset_fable_gdn_keepmask_fold_cache()
    with pytest.raises(ValueError, match="not one of"):
        fold.fable_gdn_keepmask_fold_windows()


def test_non_integer_ring_depth_raises(monkeypatch):
    monkeypatch.setenv(fold.ENV_WINDOWS, "two")
    fold.reset_fable_gdn_keepmask_fold_cache()
    with pytest.raises(ValueError, match="not an integer"):
        fold.fable_gdn_keepmask_fold_windows()


# --------------------------------------------------------------------------
# Ring policy
# --------------------------------------------------------------------------


def test_ring_grows_until_full_then_flushes():
    decision = fold.ring_after_commit((), 2, max_windows=2)
    assert decision == fold.RingDecision(flush=False, keeps=(2,))
    decision = fold.ring_after_commit(decision.keeps, 1, max_windows=2)
    assert decision == fold.RingDecision(flush=False, keeps=(2, 1))
    decision = fold.ring_after_commit(decision.keeps, 3, max_windows=2)
    assert decision == fold.RingDecision(flush=True, keeps=(3,))


def test_ring_depth_one_flushes_every_partial():
    decision = fold.ring_after_commit((), 1, max_windows=1)
    assert decision.flush is False and decision.keeps == (1,)
    decision = fold.ring_after_commit(decision.keeps, 2, max_windows=1)
    assert decision.flush is True and decision.keeps == (2,)


def test_whole_window_never_commits_through_the_fold():
    # generation.py returns on the all-accept branch before
    # commit_verified_window, so keep == VERIFY_WIDTH must never reach here.
    with pytest.raises(ValueError, match="whole-window accept"):
        fold.ring_after_commit((), fold.VERIFY_WIDTH, max_windows=2)


@pytest.mark.parametrize("keep", [0, -1, 5])
def test_out_of_range_keep_raises(keep):
    with pytest.raises(ValueError):
        fold.ring_after_commit((), keep, max_windows=2)


# --------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------


def test_mask_pads_in_front_and_marks_kept_rows():
    mask = fold.prefix_mask_rows((2,), max_windows=2)
    assert len(mask) == 8
    assert mask == [False] * 4 + [True, True, False, False]


def test_mask_two_windows_oldest_first():
    mask = fold.prefix_mask_rows((1, 3), max_windows=2)
    assert mask == [
        True, False, False, False,      # oldest window kept 1 row
        True, True, True, False,        # newer window kept 3 rows
    ]


def test_empty_ring_mask_is_all_false():
    assert fold.prefix_mask_rows((), max_windows=2) == [False] * 8


def test_mask_live_count_equals_committed_rows():
    for keeps in [(1,), (3,), (1, 1), (2, 3), (3, 3)]:
        mask = fold.prefix_mask_rows(keeps, max_windows=2)
        assert sum(mask) == sum(keeps)
        assert len(mask) == fold.VERIFY_WIDTH * 2


def test_mask_rejects_overfull_ring():
    with pytest.raises(ValueError, match="exceeds max_windows"):
        fold.prefix_mask_rows((1, 2, 3), max_windows=2)


# --------------------------------------------------------------------------
# Pass accounting: what the fold is worth at the measured accept law
# --------------------------------------------------------------------------


def test_stationary_rate_matches_a_direct_simulation():
    p = 0.295
    for max_windows in fold.MAX_WINDOWS_CHOICES:
        closed = fold.expected_state_passes_per_cycle(p, max_windows=max_windows)
        # Deterministic chain iteration to the fixed point (no RNG).
        pi = [0.0] * (max_windows + 1)
        pi[0] = 1.0
        for _ in range(4000):
            nxt = [0.0] * (max_windows + 1)
            nxt[0] = p
            for length, mass in enumerate(pi):
                if length < max_windows:
                    nxt[length + 1] += (1.0 - p) * mass
                else:
                    nxt[1] += (1.0 - p) * mass
            pi = nxt
        simulated = (1.0 - p) * pi[max_windows]
        assert closed == pytest.approx(simulated, abs=1e-9)


def test_fold_always_beats_the_eager_replay_rate():
    p = 0.295
    eager = 1.0 - p
    rates = [
        fold.expected_state_passes_per_cycle(p, max_windows=n)
        for n in fold.MAX_WINDOWS_CHOICES
    ]
    assert all(rate < eager for rate in rates)
    # deeper ring -> strictly fewer passes, with diminishing returns
    assert rates == sorted(rates, reverse=True)


def test_documented_rate_table():
    p = 0.295
    expected = {1: 0.497, 2: 0.206, 3: 0.112, 4: 0.068}
    for windows, value in expected.items():
        assert fold.expected_state_passes_per_cycle(
            p, max_windows=windows
        ) == pytest.approx(value, abs=5e-4)


def test_state_bytes_per_layer_pass():
    # 48 value heads x 128 x 128 float32 = 3,145,728 B; read + write = 6.29 MB.
    assert fold.STATE_BYTES == 3_145_728
    assert 2 * fold.STATE_BYTES * fold.GDN_LAYERS == pytest.approx(
        226_492_416, rel=0
    )


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_rate_rejects_degenerate_accept_law(p):
    with pytest.raises(ValueError):
        fold.expected_state_passes_per_cycle(p, max_windows=2)


# --------------------------------------------------------------------------
# Contract gate + install
# --------------------------------------------------------------------------


class _Gdn:
    def __init__(self, **overrides):
        self.num_v_heads = fold.NUM_V_HEADS
        self.num_k_heads = fold.NUM_K_HEADS
        self.head_v_dim = fold.HEAD_DIM
        self.head_k_dim = fold.HEAD_DIM
        self.training = False
        for key, value in overrides.items():
            setattr(self, key, value)


class _Layer:
    def __init__(self, gdn=None):
        self.linear_attn = gdn


def _production_layers():
    layers = []
    for index in range(48):
        # 36 GDN + 12 QSA; the exact partition does not matter to the gate,
        # only which indices are handed in as GDN.
        layers.append(_Layer(_Gdn()) if index % 4 != 3 else _Layer(None))
    return layers


def _production_indices():
    return tuple(index for index in range(48) if index % 4 != 3)


def test_install_arms_and_counts_foldable_layers():
    report = fold.install_gdn_keepmask_fold(
        gdn_layer_indices=_production_indices(),
        ple_layer_index=2,
        layer_modules=_production_layers(),
    )
    assert report["installed"] is True
    assert report["install_status"] == "installed"
    assert report["folded_layers"] == fold.FOLDABLE_LAYERS == 35
    assert report["prefix_rows"] == 8


def test_install_raises_on_wrong_layer_count():
    with pytest.raises(fold.GdnKeepMaskFoldContractError, match="36 GDN layers"):
        fold.install_gdn_keepmask_fold(
            gdn_layer_indices=(0, 1, 2),
            ple_layer_index=2,
            layer_modules=_production_layers(),
        )


def test_install_raises_when_ple_is_not_a_gdn_layer():
    with pytest.raises(fold.GdnKeepMaskFoldContractError, match="PLE layer"):
        fold.install_gdn_keepmask_fold(
            gdn_layer_indices=_production_indices(),
            ple_layer_index=3,
            layer_modules=_production_layers(),
        )


def test_install_raises_on_wrong_head_geometry():
    layers = _production_layers()
    layers[6].linear_attn = _Gdn(num_v_heads=32)
    with pytest.raises(fold.GdnKeepMaskFoldContractError, match="wired for"):
        fold.install_gdn_keepmask_fold(
            gdn_layer_indices=_production_indices(),
            ple_layer_index=2,
            layer_modules=layers,
        )


def test_install_disables_but_does_not_raise_on_exactness_failure(capsys):
    report = fold.install_gdn_keepmask_fold(
        gdn_layer_indices=_production_indices(),
        ple_layer_index=2,
        layer_modules=_production_layers(),
        exactness_probe=lambda: (False, "max|delta| 3.05e-05 at layer 0"),
    )
    assert report["installed"] is False
    assert report["install_status"] == "exactness_failed"
    assert "3.05e-05" in report["install_error"]
    assert "DISABLED" in capsys.readouterr().out


def test_install_disables_when_the_probe_itself_throws():
    def _boom():
        raise RuntimeError("no metal")

    report = fold.install_gdn_keepmask_fold(
        gdn_layer_indices=_production_indices(),
        ple_layer_index=2,
        layer_modules=_production_layers(),
        exactness_probe=_boom,
    )
    assert report["installed"] is False
    assert "no metal" in report["install_error"]


def test_engagement_line_is_printed_once(capsys):
    for _ in range(3):
        fold.install_gdn_keepmask_fold(
            gdn_layer_indices=_production_indices(),
            ple_layer_index=2,
            layer_modules=_production_layers(),
        )
    out = capsys.readouterr().out
    assert out.count("[fable] gdn_keepmask_fold") == 1
    assert "35 foldable GDN layers" in out
    assert "step T=12" in out


def test_engagement_line_can_be_silenced(monkeypatch, capsys):
    monkeypatch.setenv(fold.ENV_LOG, "0")
    fold.install_gdn_keepmask_fold(
        gdn_layer_indices=_production_indices(),
        ple_layer_index=2,
        layer_modules=_production_layers(),
    )
    assert capsys.readouterr().out == ""


def test_state_contract_rejects_bf16_state():
    class _State:
        shape = (1, 48, 128, 128)
        dtype = "bfloat16"

    with pytest.raises(fold.GdnKeepMaskFoldContractError, match="bit-exact"):
        fold.validate_state_contract(_State(), label="probe")


def test_state_contract_accepts_the_production_state():
    class _State:
        shape = (1, 48, 128, 128)
        dtype = "mlx.core.float32"

    fold.validate_state_contract(_State(), label="probe")


# --------------------------------------------------------------------------
# Counters
# --------------------------------------------------------------------------


def test_counters_track_windows_and_saved_passes():
    fold.note_window(0, folded=False)
    fold.note_window(1, folded=True)
    fold.note_window(2, folded=True)
    fold.note_deferred_commit(layers=35, flushed=False)
    fold.note_deferred_commit(layers=35, flushed=True)
    snapshot = fold.stats_snapshot()
    assert snapshot["windows"] == 3
    assert snapshot["folded_windows"] == 2
    assert snapshot["ring_depth_hist"] == {"0": 1, "1": 1, "2": 1}
    assert snapshot["deferred_commits"] == 2
    assert snapshot["flushes"] == 1
    assert snapshot["state_passes_saved"] == 35
    assert snapshot["state_bytes_saved"] == 35 * 2 * fold.STATE_BYTES


def test_stats_snapshot_is_a_copy():
    fold.note_window(1, folded=True)
    snapshot = fold.stats_snapshot()
    snapshot["ring_depth_hist"]["1"] = 999
    assert fold.STATS["ring_depth_hist"]["1"] == 1


# --------------------------------------------------------------------------
# Pending descriptor is fail-safe
# --------------------------------------------------------------------------


class _Entry:
    def __init__(self, leaf):
        self.cache = [None, leaf]


def test_pending_is_honoured_while_it_owns_the_leaf():
    leaf = object()
    entry = _Entry(leaf)
    entry._mtplx_fold_pending = fold.FoldPending(base=object(), state=leaf)
    assert fold.pending_for(entry) is entry._mtplx_fold_pending


def test_pending_is_dropped_when_something_else_rebinds_the_leaf():
    leaf = object()
    entry = _Entry(leaf)
    entry._mtplx_fold_pending = fold.FoldPending(base=object(), state=leaf)
    entry.cache[1] = object()  # a rollback / trim / detach / re-forward
    assert fold.pending_for(entry) is None
    assert entry._mtplx_fold_pending is None


def test_pending_absent_by_default():
    assert fold.pending_for(_Entry(object())) is None


def test_clear_pending_is_idempotent():
    entry = _Entry(object())
    fold.clear_pending(entry)
    entry._mtplx_fold_pending = fold.FoldPending(base=object())
    fold.clear_pending(entry)
    assert entry._mtplx_fold_pending is None


# --------------------------------------------------------------------------
# The kernel's arithmetic body must stay identical to the stock kernel's
# --------------------------------------------------------------------------


def _stock_gated_delta_source() -> str:
    import mlx_lm.models.gated_delta as stock

    return Path(stock.__file__).read_text()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_prefix_kernel_body_matches_the_stock_recurrence():
    """The fold's exactness rests on the loop body being the stock body.

    Compare TEXT, not behaviour: the stock body is read out of the installed
    ``mlx_lm`` and the module's template is substituted back to the stock
    names.  A drift in either (an mlx_lm upgrade, an edit here) fails loudly
    instead of turning into a silent rounding change 36 layers deep.
    """

    module = (ROOT / "mtplx" / "kernels" / "gdn_keepmask_fold.py").read_text()
    template = module.split('_STEP_BODY = """', 1)[1].split('"""', 1)[0]
    ours = template.format(
        mask_="MASK",
        g_="g_",
        k_="k_",
        v_="v_",
        q_="q_",
        beta_="beta_",
        y_="y",
    )
    stock = _stock_gated_delta_source()
    body = stock.split("for (int t = 0; t < T; ++t) {{", 1)[1]
    body = body.split("// Increment data pointers", 1)[0]
    stock_body = body.replace("{mask_source}", "MASK").replace(
        "{g_access}", "g_[hv_idx]"
    )
    # The stock body lives inside an f-string, so its braces are doubled; the
    # module's body is a ``str.format`` template, and ``.format`` above has
    # already un-doubled ours.
    stock_body = stock_body.replace("{{", "{").replace("}}", "}")
    assert _normalise(ours) == _normalise(stock_body)


def test_prefix_kernel_declares_one_state_in_and_one_state_out():
    module = (ROOT / "mtplx" / "kernels" / "gdn_keepmask_fold.py").read_text()
    # One load of state_in, one store to state_out: that is the whole point.
    assert module.count("state[i] = static_cast<float>(i_state[s_idx]);") == 1
    assert module.count("o_state[s_idx] = static_cast<StT>(state[i]);") == 1
    assert '"state_in",' in module and '"state_out"' in module


def test_prefix_kernel_window_rows_are_unconditional():
    """The window half must compile to the stock UNMASKED kernel's arithmetic.

    Today's verify runs ``gated_delta_update(..., mask=None)`` -> the unmasked
    kernel.  Emitting the window half with ``mask_="true"`` keeps that: the
    compiler folds the branch, and there is no mask buffer to disagree with.
    """

    module = (ROOT / "mtplx" / "kernels" / "gdn_keepmask_fold.py").read_text()
    assert 'mask_="true"' in module
    assert 'mask_="mask_pre[b_idx * Tpre + t]"' in module


# --------------------------------------------------------------------------
# The guarded micro: CLI, pass model, and the no-GPU contract of --dry-run
# --------------------------------------------------------------------------

MICRO = ROOT / "scripts" / "fable" / "micro_gdn_keepmask_fold.py"


def _load_micro():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_micro_keepmask_fold", MICRO)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_micro_carries_the_gpu_window_banner():
    head = MICRO.read_text().splitlines()[:6]
    joined = "\n".join(head)
    assert "GPU WINDOW REQUIRED" in joined
    assert "/tmp/mtplx-gpu-exclusive.lock" in joined


def test_micro_dry_run_touches_no_mlx(capsys):
    import sys

    micro = _load_micro()
    assert micro.mx is None
    assert micro.main(["--dry-run"]) == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["pass_model"]["step_T_folded"] == 12
    assert micro.mx is None, "--dry-run must not import mlx"
    assert "mlx.core" not in sys.modules or micro.mx is None


def test_micro_pass_model_agrees_with_the_module():
    micro = _load_micro()
    model = micro.pass_model(35, 2, 0.295)
    assert model["commit_passes_today"] == pytest.approx(0.705)
    assert model["commit_passes_folded"] == pytest.approx(
        fold.expected_state_passes_per_cycle(0.295, max_windows=2)
    )
    # One pass = 35 layers x (3.146 MB read + 3.146 MB write).
    assert model["bytes_per_pass"] == 35 * 2 * fold.STATE_BYTES


def test_micro_pass_model_uses_the_measured_accept_law():
    micro = _load_micro()
    # 112 all-accept of 374 classified cycles in the retained control census.
    assert micro.P_ALL_ACCEPT == pytest.approx(112 / 374)


def test_micro_t_sweep_brackets_every_supported_ring_depth():
    micro = _load_micro()
    for windows in fold.MAX_WINDOWS_CHOICES:
        folded_t = fold.VERIFY_WIDTH * (windows + 1)
        assert folded_t in micro.T_SWEEP, (
            f"the falsifier must time the T the ring actually runs at "
            f"({folded_t} for {windows} window(s))"
        )
    assert fold.VERIFY_WIDTH in micro.T_SWEEP


def test_micro_keep_cases_cover_every_partial_width_and_depth():
    micro = _load_micro()
    cases = micro._keep_cases(2)
    assert (1,) in cases and (2,) in cases and (3,) in cases
    assert max(len(case) for case in cases) == 2
    assert all(1 <= keep <= 3 for case in cases for keep in case)
    # A whole-window accept is never committed, so 4 must not appear.
    assert all(keep != fold.VERIFY_WIDTH for case in cases for keep in case)
    assert max(len(case) for case in micro._keep_cases(3)) == 3


def test_micro_rejects_an_unsupported_ring_depth():
    micro = _load_micro()
    with pytest.raises(SystemExit):
        micro.main(["--dry-run", "--max-windows", "5"])


def test_micro_reports_the_concat_tax_of_the_mlx_fold():
    micro = _load_micro()
    model = micro.pass_model(35, 2, 0.295)
    # 5 tensors x (ring + 1) pieces x 35 layers -- the reason the kernel arm
    # exists at all.
    assert model["concat_dispatches_fold_mlx_per_cycle"] == 5 * 3 * 35
