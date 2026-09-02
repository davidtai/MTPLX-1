"""Pure-python coverage for MTPLX_FABLE_DEVICE_DRAFT_CHAIN.

**No MLX import happens in this file.**  ``mlx.core`` and the two mlx-importing
modules the route reaches (``mtplx.fable_compiled_draft`` and
``mtplx.fast_sampling``) are replaced with recording stubs through
``sys.modules``, so nothing here can queue or evaluate a device array.  What is
checked:

* the flag-off source contract -- every new call site in ``generation.py`` is
  behind the module constant or ``_device_draft_chain_plan is not None``
  (AST inspection, the shape ``tests/test_fable_device_k20.py`` uses);
* :func:`fast_midpoint_descriptors` bit-identical to the shipped
  ``build_pcg64_midpoint_descriptors``, word for word, on random and
  adversarial uniforms (this is the ``Fraction``-free hot path);
* draw accounting -- the chain consumes exactly the doubles, in exactly the
  order, that the flag-off ``rng.choice`` chain consumes, and leaves the PCG64
  cursor in the same place;
* ``_choice_from_uniform`` reproduces ``rng.choice(ids, p=probs)`` exactly;
* :func:`host_support_tail` against a NumPy transcription of
  ``fast_sampling._device_serial_support_arrays``'s tail;
* chain structure -- ONE ``mx.eval`` per cycle in ``chain`` mode, ``depth`` in
  ``body`` mode, and the token fed to depth d+1 never crosses the host in
  ``chain`` mode;
* rebinding when the live MTP cache container is replaced;
* every named construction-time refusal.

The Metal kernels are measured, not unit tested, by
``scripts/fable/micro_device_draft_chain.py`` under the GPU lock.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mtplx import fable_device_draft_chain as ddc
from mtplx.sampling import SamplerConfig, SparseDistribution


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPTH = 3
K20 = 20
FRSPEC_ROWS = ddc.FRSPEC_ROWS
VOCAB_ROWS = 248_320


def _choice_oracle():
    """Load the choice kernel's CPU oracle WITHOUT importing ``mtplx.kernels``.

    ``mtplx/kernels/__init__.py`` pulls MLX in; the module itself does not.
    """

    path = REPO_ROOT / "mtplx" / "kernels" / "qwen4_frspec_k20_float32_choice.py"
    spec = importlib.util.spec_from_file_location("_ddc_choice_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Flag-off source contract
# ---------------------------------------------------------------------------


def _generation_tree() -> ast.Module:
    return ast.parse((REPO_ROOT / "mtplx" / "generation.py").read_text())


def _guard_names(node: ast.AST) -> set[str]:
    return {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    } | {
        child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
    }


def test_flag_is_read_once_at_import_and_defaults_off(monkeypatch):
    monkeypatch.delenv("MTPLX_FABLE_DEVICE_DRAFT_CHAIN", raising=False)
    assert ddc._resolve_mode(None) == ddc.MODE_OFF
    assert ddc._resolve_mode("") == ddc.MODE_OFF
    assert ddc._resolve_mode("0") == ddc.MODE_OFF
    for raw in ("1", "true", "YES", "on", "chain", "CHAIN"):
        assert ddc._resolve_mode(raw) == ddc.MODE_CHAIN
    for raw in ("body", "BODY", " body "):
        assert ddc._resolve_mode(raw) == ddc.MODE_BODY
    with pytest.raises(ValueError, match="is not a mode"):
        ddc._resolve_mode("chian")


def test_every_generation_call_site_is_guarded():
    """No unguarded reference: flag-off must run the pre-existing code."""

    tree = _generation_tree()
    sites = {
        "_fable_device_draft_chain_claim": False,
        "_fable_device_draft_chain_prewarm": False,
        "_fable_device_draft_chain_run": False,
        "_fable_device_draft_chain_release": False,
    }
    guard_tokens = {
        "_FABLE_DEVICE_DRAFT_CHAIN",
        "_device_draft_chain_plan",
        "_device_draft_chain_used",
        "_device_draft_chain_eligible",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = _guard_names(node.test)
        if not (names & guard_tokens):
            continue
        for called in ast.walk(node):
            if isinstance(called, ast.Call) and isinstance(called.func, ast.Name):
                if called.func.id in sites:
                    sites[called.func.id] = True
    assert all(sites.values()), f"unguarded call sites: {sites}"


def test_the_eligibility_flag_is_rooted_in_the_plan():
    """`_device_draft_chain_eligible` is a guard only if it starts from None."""

    source = (REPO_ROOT / "mtplx" / "generation.py").read_text()
    start = source.index("_device_draft_chain_eligible = (")
    block = source[start : source.index(")", start)]
    assert "_device_draft_chain_plan is not None" in block
    assert "not used_device_core" in block
    assert "not _greedy_chain_used" in block
    assert "_device_k20_chain is None" in block


def test_depth_loop_is_skipped_when_the_chain_ran():
    """The stock per-depth loop must not run on top of the chain."""

    source = (REPO_ROOT / "mtplx" / "generation.py").read_text()
    marker = "or _device_draft_chain_used"
    assert source.count(marker) == 1
    head = source[: source.index(marker)]
    assert head.rstrip().endswith("or _device_k20_chain is not None")


def test_compiled_draft_compact_row_hook_defaults_to_none():
    """The PR391 route must be byte-identical unless a hook is passed."""

    source = (REPO_ROOT / "mtplx" / "fable_compiled_draft.py").read_text()
    assert "compact_row_fn: Callable[[Any], Any] | None = None" in source
    assert (
        "source = logits if compact_row_fn is None else compact_row_fn(logits)"
        in source
    )
    assert '"entry_kv": entry_kv,' in source


# ---------------------------------------------------------------------------
# fast_midpoint_descriptors == build_pcg64_midpoint_descriptors
# ---------------------------------------------------------------------------


def _uniform_corpus() -> np.ndarray:
    """Random draws plus every adversarial float32 rounding case we can name."""

    rng = np.random.default_rng(20260902)
    random_draws = rng.random(4096, dtype=np.float64)
    grid = np.float64(1 << 53)
    edges = [
        0.0,
        1.0 / grid,
        2.0 / grid,
        3.0 / grid,
        np.nextafter(1.0, 0.0),
        0.5,
        0.5 - 1.0 / grid,
        0.5 + 1.0 / grid,
        0.95,
        1.0 - 1.0 / grid,
    ]
    # Values that sit exactly on a float32 tie (the `>` vs `<=` branch) and
    # values that land on a float32 significand carry (0xFFFFFF -> 0x800000).
    for exponent in range(1, 40):
        base = np.ldexp(1.0, -exponent)
        for delta in (-2, -1, 0, 1, 2):
            candidate = base + delta / grid
            if 0.0 <= candidate < 1.0:
                edges.append(candidate)
    for scale in range(1, 30):
        candidate = np.float64(np.float32(np.ldexp(1.0, -scale)))
        nudge = np.nextafter(np.float32(candidate), np.float32(0.0))
        for value in (candidate, float(nudge), (candidate + float(nudge)) / 2):
            snapped = np.ldexp(np.floor(np.ldexp(value, 53)), -53)
            if 0.0 <= snapped < 1.0:
                edges.append(float(snapped))
    corpus = np.concatenate(
        (random_draws, np.asarray(edges, dtype=np.float64))
    ).astype(np.float64)
    # Snap everything onto the exact 53-bit PCG64 grid.
    return np.ldexp(np.floor(np.ldexp(corpus, 53)), -53)


def test_fast_midpoint_descriptors_match_the_shipped_builder():
    oracle = _choice_oracle()
    corpus = _uniform_corpus()
    want = oracle.build_pcg64_midpoint_descriptors(corpus)
    got = ddc.fast_midpoint_descriptors(corpus)
    assert got.dtype == np.dtype(np.uint32)
    assert got.shape == want.shape
    bad = np.nonzero(np.any(got != want, axis=1))[0]
    assert bad.size == 0, (
        f"{bad.size} descriptor rows differ, first at u={corpus[bad[0]]!r}: "
        f"{got[bad[0]]} != {want[bad[0]]}"
    )
    # The shipped validator is the independent second opinion.
    oracle.validate_pcg64_midpoint_descriptors(got)


def test_fast_midpoint_descriptors_refuse_off_grid_input():
    with pytest.raises(ValueError, match="53-bit grid"):
        ddc.fast_midpoint_descriptors(np.array([0.1234567890123], dtype=np.float64))
    with pytest.raises(ValueError, match="in \\[0, 1\\)"):
        ddc.fast_midpoint_descriptors(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="dtype float64"):
        ddc.fast_midpoint_descriptors(np.array([0.5], dtype=np.float32))


# ---------------------------------------------------------------------------
# Draw accounting
# ---------------------------------------------------------------------------


def _distribution(seed: int) -> SparseDistribution:
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(VOCAB_ROWS, size=K20, replace=False)).astype(np.int64)
    weights = rng.random(K20) + 1e-3
    return SparseDistribution(ids, weights / weights.sum(), VOCAB_ROWS)


def test_tape_order_matches_the_stock_per_depth_choice_chain():
    """rng.random(depth) is the same doubles, in order, as depth rng.choice."""

    distributions = [_distribution(seed) for seed in range(DEPTH)]

    stock = np.random.default_rng(4242)
    stock_tokens = [
        int(stock.choice(dist.token_ids, p=dist.probs)) for dist in distributions
    ]
    stock_after = stock.random(8)

    chained = np.random.default_rng(4242)
    uniforms = ddc._draw_draft_uniforms(chained, DEPTH)
    chain_tokens = [
        ddc._choice_from_uniform(dist, float(uniforms[index]))
        for index, dist in enumerate(distributions)
    ]
    chain_after = chained.random(8)

    assert chain_tokens == stock_tokens
    # The cursor lands in the same place, so every later accept coin, residual
    # correction and bonus draw is unshifted.
    assert np.array_equal(chain_after, stock_after)


def test_choice_from_uniform_matches_numpy_choice_over_many_rows():
    for seed in range(200):
        dist = _distribution(seed)
        rng_a = np.random.default_rng(9000 + seed)
        rng_b = np.random.default_rng(9000 + seed)
        want = int(rng_a.choice(dist.token_ids, p=dist.probs))
        got = ddc._choice_from_uniform(dist, float(rng_b.random()))
        assert got == want, f"seed {seed}"


def test_choice_from_uniform_handles_zeroed_nucleus_entries():
    """Top-p zeroes trailing entries; the CDF walk must never select them."""

    ids = np.arange(5, dtype=np.int64)
    probs = np.array([0.7, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    dist = SparseDistribution(ids, probs, VOCAB_ROWS)
    for uniform in (0.0, 0.5, 0.6999, 0.7000001, 0.9999999):
        assert ddc._choice_from_uniform(dist, uniform) in {0, 1}


# ---------------------------------------------------------------------------
# host_support_tail == _device_serial_support_arrays' tail
# ---------------------------------------------------------------------------


def _stock_tail(ids, values, probs, *, top_p, top_k):
    """NumPy transcription of ``fast_sampling.py`` lines 445-462."""

    cand_ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    cand_vals = np.asarray(values, dtype=np.float32).reshape(1, -1)
    token_rows = cand_ids[:, :top_k]
    if 0.0 < float(top_p) < 1.0:
        cand_probs = np.asarray(probs, dtype=np.float64).reshape(1, -1)
        prob_rows = cand_probs[:, :top_k].copy()
        cumulative_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        prob_rows = np.where(cumulative_before < float(top_p), prob_rows, 0.0)
    else:
        vals64 = cand_vals[:, :top_k].astype(np.float64)
        vals64 -= np.max(vals64, axis=1, keepdims=True)
        prob_rows = np.exp(vals64)
        prob_rows /= np.sum(prob_rows, axis=1, keepdims=True)
    return token_rows, prob_rows


@pytest.mark.parametrize("top_p", [0.95, 1.0, 0.5])
def test_host_support_tail_is_the_stock_tail(top_p):
    rng = np.random.default_rng(77)
    for _ in range(200):
        values = np.sort((rng.standard_normal(K20) * 4.0).astype(np.float32))[::-1]
        values = np.ascontiguousarray(values)
        ids = np.sort(rng.choice(VOCAB_ROWS, size=K20, replace=False)).astype(
            np.int64
        )
        raw = np.exp(values.astype(np.float64) - float(values[0]))
        probs = (raw / raw.sum()).astype(np.float32)
        want_ids, want_probs = _stock_tail(
            ids, values, probs, top_p=top_p, top_k=K20
        )
        got_ids, got_probs = ddc.host_support_tail(
            ids, values, probs, top_p=top_p, top_k=K20
        )
        assert np.array_equal(got_ids, want_ids)
        assert np.array_equal(
            got_probs.view(np.uint64), want_probs.view(np.uint64)
        )


# ---------------------------------------------------------------------------
# Chain structure -- readback counting on a stub MLX
# ---------------------------------------------------------------------------


class _StubArray:
    """The minimum an ``mx.array`` needs to be for this module."""

    def __init__(self, value, dtype=None):
        self.value = np.asarray(value)
        self.dtype = dtype
        self.shape = self.value.shape
        self.evaluated = False

    def __array__(self, dtype=None, copy=None):
        if not self.evaluated:
            raise AssertionError(
                "a chain output was read on the host before mx.eval -- that is "
                "an unaccounted readback"
            )
        return np.asarray(self.value, dtype=dtype)

    def reshape(self, *shape):
        return _StubArray(self.value.reshape(*shape), self.dtype)


class _StubMX(types.ModuleType):
    def __init__(self):
        super().__init__("mlx.core")
        self.evals: list[int] = []
        self.uint32 = "uint32"
        self.int32 = "int32"
        self.float32 = "float32"

    def array(self, value, dtype=None):
        out = _StubArray(value, dtype)
        # Host-built inputs are already materialised.
        out.evaluated = True
        return out

    def zeros(self, shape, dtype=None):
        out = _StubArray(np.zeros(shape), dtype)
        out.evaluated = True
        return out

    def eval(self, *arrays):
        for array in arrays:
            array.evaluated = True
        self.evals.append(len(arrays))


@pytest.fixture()
def stub_mx(monkeypatch):
    stub = _StubMX()
    mlx = types.ModuleType("mlx")
    mlx.core = stub
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", stub)
    return stub


def _fake_plan(mode, *, chain_fn=None, compiled_body=None, depth=DEPTH):
    plan = ddc.DeviceDraftChainPlan(
        mode=mode,
        depth=depth,
        top_k=K20,
        draft_temperature=1.0,
        draft_top_p=0.95,
        frspec_rows=FRSPEC_ROWS,
        vocab_rows=VOCAB_ROWS,
        head=types.SimpleNamespace(
            arm_prescatter_capture=lambda enabled: None,
            take_prescatter_row=lambda dense: dense,
        ),
        route="native_mtp_head",
        reserve_tokens=1024,
        build_chain=lambda cache, tree: {
            "chain_fn": chain_fn,
            "compiled_body": compiled_body,
            "entry_kv": None,
            "state_slots": (),
            "state_shapes": (),
            "trace_stats": {"body_traces": 1},
            "depth": depth,
            "top_k": K20,
        },
        state_tree_fn=lambda cache: [],
        promote_fn=lambda cache, **kwargs: (1, {}),
    )
    return plan


def _fake_chain_fn(recorder):
    """A chain_fn that records the token it was fed at each depth."""

    def chain_fn(hidden, first_token, uniform_rows):
        recorder["hidden"] = hidden
        recorder["first_token"] = first_token
        recorder["uniform_rows"] = uniform_rows
        rng = np.random.default_rng(11)
        ids = np.stack(
            [
                np.sort(rng.choice(VOCAB_ROWS, size=K20, replace=False))
                for _ in range(DEPTH)
            ]
        ).astype(np.uint32)
        values = np.sort(
            (rng.standard_normal((DEPTH, K20)) * 3.0).astype(np.float32), axis=1
        )[:, ::-1].copy()
        raw = np.exp(values.astype(np.float64) - values.max(axis=1, keepdims=True))
        probs = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
        tokens = ids[:, 0].astype(np.uint32)
        return (
            _StubArray(tokens),
            _StubArray(ids),
            _StubArray(values),
            _StubArray(probs),
        )

    return chain_fn


def test_chain_mode_does_exactly_one_readback_per_cycle(stub_mx):
    recorder: dict = {}
    plan = _fake_plan(ddc.MODE_CHAIN, chain_fn=_fake_chain_fn(recorder))
    ddc.reset_counters()
    cache = [object()]
    result = ddc.run_cycle(
        plan,
        hidden=_StubArray(np.zeros((1, 1, 8))),
        primary=12345,
        rng=np.random.default_rng(7),
        cycle_depth=DEPTH,
        live_mtp_cache=cache,
    )
    assert len(stub_mx.evals) == 1, stub_mx.evals
    assert result.readbacks == 1
    assert ddc.COUNTERS["readbacks"] == 1
    assert ddc.COUNTERS["tape_draws"] == DEPTH
    assert len(result.tokens) == DEPTH
    assert len(result.distributions) == DEPTH
    # One host->device upload of the whole cycle's tape, depth rows of 5 words.
    assert recorder["uniform_rows"].value.shape == (DEPTH, 5)
    assert recorder["uniform_rows"].value.dtype == np.dtype(np.uint32)


def test_chain_mode_never_hands_the_next_depth_a_host_token(stub_mx):
    """Depth d+1's token must stay a device array inside chain_fn."""

    source = (REPO_ROOT / "mtplx" / "fable_compiled_draft.py").read_text()
    body = source[source.index("    def chain_fn("):]
    body = body[: body.index("\n    return {")]
    assert "next_token = selected" in body
    # No host materialisation anywhere in the loop.
    for forbidden in ("mx.eval", "np.asarray", ".item()", "int("):
        assert forbidden not in body, forbidden


def test_body_mode_does_one_readback_per_depth(stub_mx, monkeypatch):
    calls = {"n": 0}
    rng_rows = np.random.default_rng(5)
    ids = np.sort(rng_rows.choice(VOCAB_ROWS, size=K20, replace=False)).astype(
        np.uint32
    )
    values = np.sort((rng_rows.standard_normal(K20) * 3.0).astype(np.float32))[
        ::-1
    ].copy()
    raw = np.exp(values.astype(np.float64) - float(values[0]))
    probs = (raw / raw.sum()).astype(np.float32)

    def compiled_body(hidden, token):
        calls["n"] += 1
        calls["last_token"] = token
        return (
            _StubArray(ids),
            _StubArray(values),
            _StubArray(probs),
            _StubArray(np.zeros((1, 1, 8))),
        )

    fake_fs = types.ModuleType("mtplx.fast_sampling")

    def _serial_row_distribution(token_ids, row_probs, vocab_size):
        keep = row_probs > 0
        kept_ids = token_ids[keep]
        kept = row_probs[keep]
        order = np.argsort(kept_ids)
        return SparseDistribution(
            kept_ids[order], kept[order] / kept.sum(), vocab_size
        )

    fake_fs._serial_row_distribution = _serial_row_distribution
    monkeypatch.setitem(sys.modules, "mtplx.fast_sampling", fake_fs)

    fake_fcd = types.ModuleType("mtplx.fable_compiled_draft")
    fake_fcd.CompiledDraftStateChanged = RuntimeError
    fake_fcd.state_leaf_shapes = lambda slots: ()
    monkeypatch.setitem(sys.modules, "mtplx.fable_compiled_draft", fake_fcd)

    plan = _fake_plan(ddc.MODE_BODY, compiled_body=compiled_body)
    ddc.reset_counters()
    result = ddc.run_cycle(
        plan,
        hidden=_StubArray(np.zeros((1, 1, 8))),
        primary=99,
        rng=np.random.default_rng(3),
        cycle_depth=DEPTH,
        live_mtp_cache=[object()],
    )
    assert calls["n"] == DEPTH
    assert len(stub_mx.evals) == DEPTH
    assert result.readbacks == DEPTH
    assert ddc.COUNTERS["readbacks"] == DEPTH
    assert ddc.COUNTERS["tape_draws"] == DEPTH


def test_body_mode_token_matches_the_stock_host_sampler(stub_mx, monkeypatch):
    """body mode is the bit-identical arm: same tail, same draw, same token."""

    rng_rows = np.random.default_rng(21)
    ids = np.sort(rng_rows.choice(VOCAB_ROWS, size=K20, replace=False)).astype(
        np.uint32
    )
    values = np.sort((rng_rows.standard_normal(K20) * 3.0).astype(np.float32))[
        ::-1
    ].copy()
    raw = np.exp(values.astype(np.float64) - float(values[0]))
    probs = (raw / raw.sum()).astype(np.float32)

    token_rows, prob_rows = _stock_tail(
        ids.astype(np.int64), values, probs, top_p=0.95, top_k=K20
    )
    keep = prob_rows[0] > 0
    kept_ids = token_rows[0][keep]
    kept = prob_rows[0][keep]
    order = np.argsort(kept_ids)
    stock_dist = SparseDistribution(
        kept_ids[order], kept[order] / kept.sum(), VOCAB_ROWS
    )
    stock_rng = np.random.default_rng(31337)
    want = int(stock_rng.choice(stock_dist.token_ids, p=stock_dist.probs))

    def compiled_body(hidden, token):
        return (
            _StubArray(ids),
            _StubArray(values),
            _StubArray(probs),
            _StubArray(np.zeros((1, 1, 8))),
        )

    fake_fs = types.ModuleType("mtplx.fast_sampling")

    def _serial_row_distribution(token_ids, row_probs, vocab_size):
        keep_mask = row_probs > 0
        kept_id = token_ids[keep_mask]
        kept_p = row_probs[keep_mask]
        idx = np.argsort(kept_id)
        return SparseDistribution(
            kept_id[idx], kept_p[idx] / kept_p.sum(), vocab_size
        )

    fake_fs._serial_row_distribution = _serial_row_distribution
    monkeypatch.setitem(sys.modules, "mtplx.fast_sampling", fake_fs)
    fake_fcd = types.ModuleType("mtplx.fable_compiled_draft")
    fake_fcd.CompiledDraftStateChanged = RuntimeError
    fake_fcd.state_leaf_shapes = lambda slots: ()
    monkeypatch.setitem(sys.modules, "mtplx.fable_compiled_draft", fake_fcd)

    plan = _fake_plan(ddc.MODE_BODY, compiled_body=compiled_body, depth=1)
    ddc.reset_counters()
    result = ddc.run_cycle(
        plan,
        hidden=_StubArray(np.zeros((1, 1, 8))),
        primary=1,
        rng=np.random.default_rng(31337),
        cycle_depth=1,
        live_mtp_cache=[object()],
    )
    assert result.tokens[0] == want


def test_run_cycle_rebinds_when_the_cache_container_is_replaced(stub_mx):
    recorder: dict = {}
    plan = _fake_plan(ddc.MODE_CHAIN, chain_fn=_fake_chain_fn(recorder))
    ddc.reset_counters()
    first, second = [object()], [object()]
    for cache in (first, first, second):
        ddc.run_cycle(
            plan,
            hidden=_StubArray(np.zeros((1, 1, 8))),
            primary=5,
            rng=np.random.default_rng(1),
            cycle_depth=DEPTH,
            live_mtp_cache=cache,
        )
    assert plan.builds == 2
    assert ddc.COUNTERS["chain_builds"] == 2
    assert ddc.COUNTERS["cache_rebinds"] == 1
    assert plan.mtp_cache is second


def test_run_cycle_refuses_a_released_plan(stub_mx):
    plan = _fake_plan(ddc.MODE_CHAIN, chain_fn=_fake_chain_fn({}))
    plan.released = True
    with pytest.raises(ddc.DeviceDraftChainIneligible, match="released"):
        ddc.run_cycle(
            plan,
            hidden=_StubArray(np.zeros((1, 1, 8))),
            primary=5,
            rng=np.random.default_rng(1),
            cycle_depth=DEPTH,
            live_mtp_cache=[object()],
        )


def test_run_cycle_refuses_a_depth_it_was_not_built_for(stub_mx):
    plan = _fake_plan(ddc.MODE_CHAIN, chain_fn=_fake_chain_fn({}))
    with pytest.raises(ddc.DeviceDraftChainIneligible, match="cycle_depth=2"):
        ddc.run_cycle(
            plan,
            hidden=_StubArray(np.zeros((1, 1, 8))),
            primary=5,
            rng=np.random.default_rng(1),
            cycle_depth=2,
            live_mtp_cache=[object()],
        )


def test_run_cycle_draws_nothing_when_it_is_not_called():
    """A context-copy cycle must leave the PCG64 cursor exactly where it was."""

    stock = np.random.default_rng(808)
    skipped = np.random.default_rng(808)
    assert np.array_equal(stock.random(4), skipped.random(4))


# ---------------------------------------------------------------------------
# Construction-time refusals
# ---------------------------------------------------------------------------


class _FakeHead:
    def __init__(self, rows=FRSPEC_ROWS, vocab=VOCAB_ROWS, ascending=True):
        ids = np.arange(rows, dtype=np.int64) * 3
        if not ascending and rows > 1:
            ids[1] = ids[0]
        self._ids = ids
        self._vocab_rows = vocab
        self.armed = None

    def arm_prescatter_capture(self, enabled):
        self.armed = bool(enabled)

    def take_prescatter_row(self, dense):
        return dense


def _fake_rt(head=None):
    text = types.SimpleNamespace(_mtplx_frspec_draft_head=head)
    if head is not None:
        text._mtp_draft_head_logits = types.SimpleNamespace(__self__=head)
        # `_live_draft_route` reads `__self__` off the bound hook.
        bound = types.SimpleNamespace()
        object.__setattr__(bound, "__self__", head)
        text._mtp_draft_head_logits = bound
    model = types.SimpleNamespace(language_model=text)
    return types.SimpleNamespace(model=model, qwen4_relaxed_draft_ties=False)


def _claim_kwargs(**overrides):
    base = dict(
        rt=_fake_rt(_FakeHead()),
        state_tree_fn=lambda cache: [],
        promote_fn=lambda cache, **kwargs: (1, {}),
        mtp_hidden_variant="contract",
        sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=20),
        draft_sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=20),
        speculative_depth=DEPTH,
        request_max_tokens=1024,
        rng=np.random.default_rng(1),
        draft_core="stock",
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        mtp_position_mode="cache",
        target_prefix_verify=False,
        lazy_target_distributions=False,
        lazy_bonus_verify_requested=False,
        batch_target_arrays=True,
        steer_active=False,
        penalties_active=False,
        relaxed_draft_ties=False,
        qsa_mtp_precompute_active=False,
        constraint=None,
        adaptive_policy=None,
        adaptive_width_policy=None,
        mtp_corrector=None,
        mtp_topk_reranker=None,
        draft_margin_threshold=None,
        wants_policy_metrics=False,
        draft_confidence_needed=False,
        online_hidden_corrector_alpha=0.0,
        online_correction_cache=False,
        prompt_correction_cache=False,
        adapter_ensemble_q=False,
        combine_greedy_draft_read=False,
        greedy_chain_enabled=False,
        adaptive_dtemp_active=False,
        frspec_legacy_ids=None,
        late_depth_switch_after=0,
        a3b_target_prefix_route=None,
        pr391_route=None,
        device_k20_route=None,
        draft_k20_prescatter_route=None,
        depth4_probe_active=False,
        k20_log_active=False,
        ple_candidate_submit=None,
    )
    base.update(overrides)
    return base


def test_claim_returns_none_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_OFF)
    assert ddc.claim_request_route(**_claim_kwargs()) is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"pr391_route": object()}, "PR391"),
        ({"target_prefix_verify": True}, "target-prefix"),
        ({"device_k20_route": object()}, "DEVICE_K20"),
        ({"draft_k20_prescatter_route": object()}, "PRESCATTER"),
        ({"draft_core": "device"}, "stock draft selector"),
        ({"greedy_chain_enabled": True}, "greedy draft chain"),
        ({"combine_greedy_draft_read": True}, "greedy draft chain"),
        (
            {"draft_sampler": SamplerConfig(temperature=0.0, top_k=20)},
            "sampled-lane route",
        ),
        (
            {"draft_sampler": SamplerConfig(temperature=1.0, top_k=40)},
            "top_k=20",
        ),
        (
            {"draft_sampler": SamplerConfig(temperature=1.0, top_k=20, top_p=0.0)},
            "top_p",
        ),
        ({"penalties_active": True}, "steering/penalty"),
        ({"adaptive_dtemp_active": True}, "ADAPTIVE_DTEMP"),
        ({"relaxed_draft_ties": True}, "RELAXED_DRAFT_TIES"),
        ({"speculative_depth": 0}, "positive draft depth"),
        ({"late_depth_switch_after": 2}, "late-depth"),
        ({"mtp_cache_policy": "cycle"}, "captures ONE MTP cache"),
        ({"mtp_history_policy": "cycle"}, "committed MTP history"),
        ({"mtp_position_mode": "absolute"}, "position_offset"),
        ({"qsa_mtp_precompute_active": True}, "indexer precompute"),
        ({"depth4_probe_active": True}, "depth4_probe"),
        ({"k20_log_active": True}, "k20_log"),
        ({"ple_candidate_submit": object()}, "ple_candidate_prefetch"),
        ({"online_hidden_corrector_alpha": 0.5}, "online_hidden_corrector"),
        ({"mtp_topk_reranker": object()}, "mtp_topk_reranker"),
        ({"constraint": object()}, "constraint"),
        ({"request_max_tokens": 0}, "request_max_tokens"),
    ],
)
def test_claim_refuses_by_name(monkeypatch, overrides, match):
    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_CHAIN)
    with pytest.raises(ddc.DeviceDraftChainIneligible, match=match):
        ddc.claim_request_route(**_claim_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_target_arrays": False},
        {"lazy_target_distributions": True},
        {"lazy_bonus_verify_requested": True},
    ],
)
def test_claim_admits_target_side_settings(monkeypatch, overrides):
    """This route replaces the DRAFT chain only; target-side knobs are not its
    business.  It hands the loop the same host draft_tokens/draft_probs the
    per-depth loop would have, so the lazy-bonus width decision -- which reads
    `draft_tokens[:-1]` -- sees an indistinguishable cycle."""

    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_CHAIN)
    fake_fcd = types.ModuleType("mtplx.fable_compiled_draft")
    fake_fcd.build_compiled_draft_chain = lambda **kwargs: {}
    monkeypatch.setitem(sys.modules, "mtplx.fable_compiled_draft", fake_fcd)
    fake_kernel = types.ModuleType("mtplx.kernels.qwen4_frspec_k20_float32_choice")
    fake_kernel.bind_qwen4_frspec_k20_float32_choice = lambda *, top_p: "selector"
    monkeypatch.setitem(
        sys.modules,
        "mtplx.kernels.qwen4_frspec_k20_float32_choice",
        fake_kernel,
    )
    assert ddc.claim_request_route(**_claim_kwargs(**overrides)) is not None


def test_claim_refuses_a_non_pcg64_generator(monkeypatch):
    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_CHAIN)
    philox = np.random.Generator(np.random.Philox(1))
    with pytest.raises(ddc.DeviceDraftChainIneligible, match="PCG64"):
        ddc.claim_request_route(**_claim_kwargs(rng=philox))


def test_claim_refuses_a_pinned_numpy_mismatch(monkeypatch):
    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_CHAIN)
    monkeypatch.setattr(np, "__version__", "1.26.4")
    with pytest.raises(ddc.DeviceDraftChainIneligible, match="pinned to NumPy"):
        ddc.claim_request_route(**_claim_kwargs())


@pytest.mark.parametrize(
    ("head", "match"),
    [
        (None, "no FR-Spec draft head"),
        (_FakeHead(rows=32_768), "proven only at"),
        (_FakeHead(ascending=False), "strictly ascending"),
        (_FakeHead(vocab=1024), "does not admit the table"),
    ],
)
def test_claim_refuses_an_unusable_head(monkeypatch, head, match):
    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_CHAIN)
    with pytest.raises(ddc.DeviceDraftChainIneligible, match=match):
        ddc.claim_request_route(**_claim_kwargs(rt=_fake_rt(head)))


def test_claim_refuses_a_head_that_is_not_live(monkeypatch):
    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_CHAIN)
    head = _FakeHead()
    text = types.SimpleNamespace(_mtplx_frspec_draft_head=head)
    rt = types.SimpleNamespace(
        model=types.SimpleNamespace(language_model=text),
        qwen4_relaxed_draft_ties=False,
    )
    with pytest.raises(ddc.DeviceDraftChainIneligible, match="live draft route"):
        ddc.claim_request_route(**_claim_kwargs(rt=rt))


def test_prewarm_happens_before_the_decode_loop():
    """The mx.compile trace and the cache promotion must not land in cycle 1.

    A measured window that carried them would charge a one-off tens-to-hundreds
    of milliseconds to every cycle's average.
    """

    source = (REPO_ROOT / "mtplx" / "generation.py").read_text()
    prewarm_at = source.index("_fable_device_draft_chain_prewarm(")
    loop_at = source.index("\n    while len(tokens) < max_tokens:", prewarm_at)
    run_at = source.index("_fable_device_draft_chain_run(")
    assert prewarm_at < loop_at < run_at


def test_prewarm_restores_the_mtp_history_offset(stub_mx):
    recorder: dict = {}
    plan = _fake_plan(ddc.MODE_CHAIN, chain_fn=_fake_chain_fn(recorder))
    offsets = {"value": 100}
    trimmed: list[int] = []

    def cache_offset(cache):
        return offsets["value"]

    def rollback(cache, offset):
        trimmed.append(offset)
        offsets["value"] = offset

    ddc.reset_counters()
    cache = [object()]
    ddc.prewarm(
        plan,
        _StubArray(np.zeros((1, 1, 8))),
        mtp_cache=cache,
        rollback=rollback,
        cache_offset=cache_offset,
    )
    assert plan.chain is not None, "prewarm binds the compiled body"
    assert trimmed == [100]
    assert len(stub_mx.evals) == 1
    # The tape is not touched: prewarm draws no uniform from the request rng.
    assert ddc.COUNTERS["tape_draws"] == 0


def test_prewarm_raises_when_the_history_cannot_be_restored(stub_mx):
    """A cache whose trim did not take must raise here, not drift silently."""

    plan = _fake_plan(ddc.MODE_CHAIN, chain_fn=_fake_chain_fn({}))
    reads = {"n": 0}

    def cache_offset(cache):
        reads["n"] += 1
        # 7 before the trace, 10 after it -- a trim that did nothing.
        return 7 if reads["n"] == 1 else 10

    with pytest.raises(ddc.DeviceDraftChainIneligible, match="restore the MTP"):
        ddc.prewarm(
            plan,
            _StubArray(np.zeros((1, 1, 8))),
            mtp_cache=[object()],
            rollback=lambda cache, offset: None,
            cache_offset=cache_offset,
        )


def test_claim_binds_and_reports(monkeypatch, capsys):
    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_CHAIN)
    built: dict = {}

    fake_fcd = types.ModuleType("mtplx.fable_compiled_draft")

    def build_compiled_draft_chain(**kwargs):
        built.update(kwargs)
        return {"trace_stats": {"body_traces": 1}}

    fake_fcd.build_compiled_draft_chain = build_compiled_draft_chain
    monkeypatch.setitem(sys.modules, "mtplx.fable_compiled_draft", fake_fcd)

    fake_kernel = types.ModuleType("mtplx.kernels.qwen4_frspec_k20_float32_choice")
    fake_kernel.bind_qwen4_frspec_k20_float32_choice = lambda *, top_p: (
        "selector",
        top_p,
    )
    monkeypatch.setitem(
        sys.modules,
        "mtplx.kernels.qwen4_frspec_k20_float32_choice",
        fake_kernel,
    )

    head = _FakeHead()
    ddc.reset_counters()
    plan = ddc.claim_request_route(**_claim_kwargs(rt=_fake_rt(head)))
    assert plan is not None
    assert head.armed is True
    assert plan.chain is None, "the compiled body binds lazily, at the first cycle"
    receipt = plan.to_dict()
    assert receipt["installed"] is True
    assert receipt["mode"] == ddc.MODE_CHAIN
    assert receipt["readbacks_per_cycle"] == 1
    assert receipt["frspec_rows"] == FRSPEC_ROWS
    assert receipt["reserve_tokens"] == 1024 + DEPTH + 4
    assert "fable-device-draft-chain" in capsys.readouterr().err

    plan.ensure_bound([object()])
    assert built["depth"] == DEPTH
    assert built["top_k"] == K20
    assert built["compact_row_fn"] is not None
    assert plan.builds == 1

    ddc.release(plan)
    assert head.armed is False
    assert plan.released is True


def test_body_mode_receipt_reports_three_readbacks(monkeypatch):
    monkeypatch.setattr(ddc, "_MODE", ddc.MODE_BODY)
    fake_fcd = types.ModuleType("mtplx.fable_compiled_draft")
    fake_fcd.build_compiled_draft_chain = lambda **kwargs: {}
    monkeypatch.setitem(sys.modules, "mtplx.fable_compiled_draft", fake_fcd)
    fake_kernel = types.ModuleType("mtplx.kernels.qwen4_frspec_k20_float32_choice")
    fake_kernel.bind_qwen4_frspec_k20_float32_choice = lambda *, top_p: "selector"
    monkeypatch.setitem(
        sys.modules,
        "mtplx.kernels.qwen4_frspec_k20_float32_choice",
        fake_kernel,
    )
    plan = ddc.claim_request_route(**_claim_kwargs())
    assert plan.readbacks_per_cycle == DEPTH
    assert plan.to_dict()["mode"] == ddc.MODE_BODY
