"""Pure-python / CPU coverage for MTPLX_FABLE_DEVICE_K20.

**No MLX import happens in this file.**  Everything the device lane owns that
can be checked without a GPU is checked here:

* the flag-off source contract (every call site guarded, by AST inspection --
  the same shape ``tests/test_fable_k20_log.py`` uses for the PR391 wiring);
* the kernel-free NumPy oracle for the selector against a brute-force sort,
  including value ties, signed zero and NaN;
* the oracle against a NumPy model of the production support builder's own hot
  path (``argpartition`` to 80 + ``np.lexsort``), so "exact vs stock" is a
  measured claim;
* the host tail (``finalize_target_support``) against a NumPy model of
  ``_device_serial_support_arrays``'s top-p mask;
* ``prepare_draft_row_f32`` bit-identical to the choice kernel's OWN CPU
  oracle ``_prepare_reference_row`` -- the thing the Metal sampler walks;
* draw accounting: the flag-on lane consumes exactly the doubles, in exactly
  the order, that the flag-off ``rng.choice`` chain consumes;
* the K20 logger round trip for the ``stock_device_k20`` layout.

The Metal kernel itself is measured, not unit tested, by
``scripts/fable/micro_k20_select.py`` under the GPU lock.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mtplx import fable_claim_contract as contract
from mtplx import fable_device_k20 as dk
from mtplx import fable_k20_log as log_mod
from mtplx.sampling import SamplerConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
K20 = 20
DEPTH = 3


def _choice_oracle():
    """Load the choice kernel's CPU oracle WITHOUT importing ``mtplx.kernels``.

    ``mtplx/kernels/__init__.py`` pulls MLX in; the module itself does not.
    """

    path = REPO_ROOT / "mtplx" / "kernels" / "qwen4_frspec_k20_float32_choice.py"
    spec = importlib.util.spec_from_file_location("_k20_choice_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Flag-off source contract
# ---------------------------------------------------------------------------


def test_flag_defaults_off_and_claim_returns_none():
    assert dk.is_enabled() is False
    assert (
        dk.claim_request_route(
            sampler=SamplerConfig(),
            draft_sampler=SamplerConfig(),
            speculative_depth=DEPTH,
            rng=np.random.default_rng(0),
            fused_verify_input=False,
            target_prefix_verify=False,
            lazy_target_distributions=False,
            lazy_bonus_verify_requested=False,
            batch_target_arrays=True,
            steer_active=False,
            penalties_active=False,
            constraint=None,
            adaptive_policy=None,
            adaptive_width_policy=None,
            mtp_corrector=None,
            mtp_topk_reranker=None,
            draft_margin_threshold=None,
            online_hidden_corrector_alpha=0.0,
            online_correction_cache=False,
            prompt_correction_cache=False,
            adapter_ensemble_q=False,
            combine_greedy_draft_read=False,
            greedy_chain_enabled=False,
            draft_confidence_needed=False,
            frspec_legacy_ids=None,
            late_depth_switch_after=0,
            a3b_target_prefix_route=None,
            pr391_route=None,
        )
        is None
    )


def _generation_source() -> str:
    return (REPO_ROOT / "mtplx" / "generation.py").read_text()


def test_env_var_is_read_exactly_once_at_import():
    source = (REPO_ROOT / "mtplx" / "fable_device_k20.py").read_text()
    assert source.count('_ENV_VAR = "MTPLX_FABLE_DEVICE_K20"') == 1
    assert source.count("_ENABLED = _env_truthy(_ENV_VAR)") == 1
    # Nothing else may read the variable -- a second read could disagree with
    # the constant the call sites are gated on.
    assert "os.environ" not in source.split("def is_enabled")[1]
    generation = _generation_source()
    assert '_env_truthy("MTPLX_FABLE_DEVICE_K20")' not in generation
    assert 'os.environ.get("MTPLX_FABLE_DEVICE_K20"' not in generation
    assert generation.count("_FABLE_DEVICE_K20 = _fable_device_k20_enabled()") == 1


def test_every_generation_call_site_is_gated_on_the_plan():
    """The flag-off lane must reach none of the device code."""

    source = _generation_source()
    tree = ast.parse(source)
    generate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_mtpk"
    )
    body = ast.get_source_segment(source, generate)

    # The plan is claimed once, behind the import-time constant.
    assert body.count("_fable_device_k20_claim(") == 1
    assert "if _FABLE_DEVICE_K20:" in body
    # The chain only ever runs behind `_device_k20_plan is not None`.
    assert body.count("_FableDeviceDraftChain(") == 1
    chain_at = body.index("_FableDeviceDraftChain(")
    guard_at = body.rindex("_device_k20_plan is not None", 0, chain_at)
    assert chain_at - guard_at < 800
    # Both batched target-support call sites carry the plan (None => stock).
    assert body.count("device_k20_plan=_device_k20_plan") == 2
    # The stock draft loop is skipped exactly when the chain ran.
    assert "or _device_k20_chain is not None" in body


def test_the_device_chain_feeds_the_depth4_probe():
    """MTPLX_FABLE_DEPTH4_PROBE must still fire when the device chain drafts.

    The chain skips the per-depth loop that normally captures the probe's
    inputs, so it captures them itself -- from the MATERIALISED result, after
    its single sync, so the token the hook wraps in `mx.array([[...]])` is a
    host int exactly as on the stock lane.
    """

    source = _generation_source()
    tree = ast.parse(source)
    generate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_mtpk"
    )
    body = ast.get_source_segment(source, generate)

    reset = body.index("_d4_probe_state: tuple[Any, int, Any, bool] | None = None")
    chain = body.index("_device_k20_chain = _FableDeviceDraftChain(")
    materialize = body.index("_dk_result = _device_k20_chain.materialize()")
    capture = body.index("_d4_probe_state = (", materialize)
    stock_loop = body.index("for depth_index in range(\n            0\n")
    # Reset, then the chain, then its capture -- all before the stock loop, so
    # the stock lane's own capture still wins when it is the one that ran.
    assert reset < chain < materialize < capture < stock_loop
    block = body[capture : capture + 400]
    assert "int(_dk_result.tokens[-1])" in block   # materialised, not lazy
    assert "step_mtp_cache" in block               # the hook checks `is mtp_cache`
    assert "draft_hidden" in block
    assert "_frspec_legacy_ids" in block


def test_the_device_layouts_are_stock_layouts_for_every_consumer():
    """`gate_q` / `probe_*` and both offline scorers must accept a device log.

    ``MTPLX_FABLE_DEVICE_K20`` changes where a K20 row was selected and how the
    drafted token was sampled -- not what a row MEANS.  A device log is a stock
    log, and the two offline scorers keep their own copy of the layout names
    (they import no ``mtplx``), so the two lists are pinned to each other here.
    """

    from scripts.fable.offline_block_verification import (
        STOCK_BV_LAYOUTS as SCORER_BV,
        STOCK_LAYOUTS as SCORER_STOCK,
    )

    assert log_mod.LAYOUT_STOCK_DEVICE_K20 in log_mod.STOCK_LAYOUTS
    assert log_mod.LAYOUT_STOCK_DEVICE_K20_BV in log_mod.STOCK_LAYOUTS
    assert log_mod.LAYOUT_STOCK_DEVICE_K20 not in log_mod.STOCK_BV_LAYOUTS
    assert log_mod.LAYOUT_STOCK_DEVICE_K20_BV in log_mod.STOCK_BV_LAYOUTS
    assert set(SCORER_STOCK) == set(log_mod.STOCK_LAYOUTS)
    assert set(SCORER_BV) == set(log_mod.STOCK_BV_LAYOUTS)


def test_batched_builder_delegates_to_stock_when_no_plan():
    source = _generation_source()
    start = source.index("def _batched_distributions_from_mlx_logits(")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    assert "if device_k20_plan is None:" in body
    first = body.index("if device_k20_plan is None:")
    assert "return batched_sparse_distributions_from_mlx_logits(logits, config)" in (
        body[first : first + 200]
    )


# ---------------------------------------------------------------------------
# The selector oracle
# ---------------------------------------------------------------------------


def _brute_force_top_k(row: np.ndarray, k: int, id_map=None) -> list[int]:
    real = list(range(row.shape[0])) if id_map is None else [int(v) for v in id_map]
    order = sorted(
        range(row.shape[0]),
        key=lambda token: (
            (1, 0) if np.isnan(row[token]) else (0, -float(row[token])),
            real[token],
        ),
    )
    return [real[token] for token in order[:k]]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_reference_top_k_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((4, 2048)).astype(np.float32)
    ids, values = dk.reference_top_k(rows, top_k=K20)
    for row in range(rows.shape[0]):
        assert list(ids[row]) == _brute_force_top_k(rows[row], K20)
        assert np.array_equal(values[row], rows[row][ids[row]])


def test_reference_top_k_tie_ownership_and_signed_zero():
    row = np.full((1, 64), -5.0, dtype=np.float32)
    # A value tie spanning the cutoff: ids 40..59 all at 1.0, so the top-20
    # must be the LOWEST twenty ids of the tie group.
    row[0, 40:60] = np.float32(1.0)
    row[0, 5] = np.float32(1.0)
    ids, _ = dk.reference_top_k(row, top_k=K20)
    assert list(ids[0]) == [5] + list(range(40, 59))

    signed = np.full((1, 32), -1.0, dtype=np.float32)
    signed[0, 3] = np.float32(0.0)
    signed[0, 9] = np.float32(-0.0)
    ids, values = dk.reference_top_k(signed, top_k=2)
    assert list(ids[0]) == [3, 9]          # +0.0 and -0.0 are ONE value
    assert list(values[0]) == [0.0, -0.0]


def test_reference_top_k_puts_nan_last():
    row = np.arange(64, dtype=np.float32)
    row[63] = np.nan
    ids, _ = dk.reference_top_k(row[None, :], top_k=3)
    assert list(ids[0]) == [62, 61, 60]


def test_reference_top_k_remaps_ids_for_a_compact_domain():
    """A 65,536-wide FRSpec row must select exactly like its 248,320 scatter."""

    rng = np.random.default_rng(9)
    full_vocab = 4096
    compact = 512
    id_map = rng.choice(full_vocab, size=compact, replace=False).astype(np.int64)
    compact_row = rng.standard_normal((1, compact)).astype(np.float32)
    # Force ties so the real-id tie key is exercised.
    compact_row[0, :8] = np.float32(3.0)

    scattered = np.full((1, full_vocab), np.float32(-1.0e30), dtype=np.float32)
    scattered[0, id_map] = compact_row[0]

    compact_ids, compact_vals = dk.reference_top_k(
        compact_row, top_k=K20, id_map=id_map
    )
    full_ids, full_vals = dk.reference_top_k(scattered, top_k=K20)
    assert list(compact_ids[0]) == list(full_ids[0])
    assert np.array_equal(compact_vals, full_vals)


# ---------------------------------------------------------------------------
# The oracle vs a NumPy model of the production support builder
# ---------------------------------------------------------------------------


def _stock_serial_support(rows: np.ndarray, config: SamplerConfig):
    """NumPy model of ``fast_sampling._device_serial_support_arrays``.

    Literal transcription of the hot path (one ``argpartition`` to an
    ``m = 4k`` superset, ``np.lexsort((ids, -values))``, top-p mask) plus the
    exact spill fallback, so the assertions below are against the shipped
    arithmetic and not against a paraphrase of it.
    """

    rows = np.asarray(rows, dtype=np.float32)
    vocab = rows.shape[-1]
    k = min(int(config.top_k), vocab)
    scaled = (rows * np.float32(1.0 / float(config.temperature))).astype(np.float32)
    m = min(max(4 * k, k), vocab)
    cand_idx = np.argpartition(-scaled, kth=m - 1, axis=-1)[:, :m]
    cand_vals = np.take_along_axis(scaled, cand_idx, axis=-1)
    top_p_active = 0.0 < float(config.top_p) < 1.0
    if top_p_active:
        row_max = scaled.max(axis=-1, keepdims=True)
        log_total = (
            row_max + np.log(np.exp(scaled - row_max).sum(axis=-1, keepdims=True))
        ).astype(np.float32)
        cand_probs = np.exp(cand_vals - log_total).astype(np.float32).astype(np.float64)
    else:
        cand_probs = None
    cand_ids = cand_idx.astype(np.int64)
    order = np.lexsort((cand_ids, -cand_vals), axis=1)
    cand_ids = np.take_along_axis(cand_ids, order, axis=1)
    cand_vals = np.take_along_axis(cand_vals, order, axis=1)
    if cand_probs is not None:
        cand_probs = np.take_along_axis(cand_probs, order, axis=1)
    token_rows = cand_ids[:, :k]
    spill = np.nanmin(cand_vals, axis=1) >= cand_vals[:, k - 1]
    if top_p_active:
        prob_rows = cand_probs[:, :k].copy()
        before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        prob_rows = np.where(before < float(config.top_p), prob_rows, 0.0)
    else:
        vals64 = cand_vals[:, :k].astype(np.float64)
        vals64 -= np.max(vals64, axis=1, keepdims=True)
        prob_rows = np.exp(vals64)
        prob_rows /= np.sum(prob_rows, axis=1, keepdims=True)
    return token_rows, prob_rows, spill


def _plan(**overrides) -> dk.DeviceK20Plan:
    base = dict(
        depth=DEPTH,
        top_k=K20,
        vocab_size=4096,
        target_rows=DEPTH + 1,
        temperature=0.6,
        top_p=0.95,
        draft_temperature=0.6,
        draft_top_p=0.95,
        draft_vocab_size=4096,
        fused_verify_input=False,
    )
    base.update(overrides)
    return dk.DeviceK20Plan(**base)


@pytest.mark.parametrize("top_p", [0.95, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_device_support_equals_stock_hot_path(top_p, seed):
    """Selection AND the host tail, bit for bit, against the stock model."""

    rng = np.random.default_rng(seed)
    vocab = 4096
    rows = (rng.standard_normal((4, vocab)) * 3.0).astype(np.float32)
    config = SamplerConfig(temperature=0.6, top_p=top_p, top_k=K20)
    plan = _plan(vocab_size=vocab, top_p=top_p, draft_top_p=top_p)

    want_ids, want_probs, spill = _stock_serial_support(rows, config)
    assert not spill.any(), "random rows should not hit the cutoff-spill path"

    scaled = (rows * np.float32(1.0 / 0.6)).astype(np.float32)
    ids, values = dk.reference_top_k(scaled, top_k=K20)
    if plan.top_p_active:
        row_max = scaled.max(axis=-1, keepdims=True)
        log_total = (
            row_max + np.log(np.exp(scaled - row_max).sum(axis=-1, keepdims=True))
        ).astype(np.float32)
        probs = np.exp(values - log_total).astype(np.float32).astype(np.float64)
    else:
        probs = None
    got_ids, got_probs = dk.finalize_target_support(ids, values, probs, plan)

    assert np.array_equal(got_ids, want_ids)
    assert np.array_equal(got_probs, want_probs)


def _deterministic_support_set(row: np.ndarray, k: int) -> set[int]:
    """The SET ``_deterministic_mlx_top_k_support`` selects.

    Docstring contract: everything strictly above the k-th largest value, then
    the LOWEST vocabulary ids among the values tied with it, exactly filling k.
    """

    order = np.argpartition(-row, kth=k - 1)[:k]
    cutoff = row[order].min()
    higher = np.flatnonzero(row > cutoff)
    tied = np.flatnonzero(row == cutoff)
    return set(int(t) for t in higher) | set(
        int(t) for t in np.sort(tied)[: k - higher.size]
    )


def test_device_selector_is_exact_where_the_stock_hot_path_spills():
    """A cutoff tie wider than the 80-candidate superset.

    This is the row the stock builder's ``spill`` flag exists for: its
    ``argpartition``-to-80 answer is NOT the contract answer, so it throws the
    superset away and re-derives with ``_deterministic_mlx_top_k_support``.
    The device selector is exact over the whole vocabulary and needs no
    fallback, so it must land on the deterministic selector's SET directly.
    """

    vocab = 1024
    rows = np.zeros((1, vocab), dtype=np.float32)      # every value tied
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=K20)
    hot_ids, _, spill = _stock_serial_support(rows, config)
    assert bool(spill[0]), "an all-tied row must be flagged as a spill"
    assert list(hot_ids[0]) != list(range(K20)), (
        "the hot path is wrong here -- that is exactly why spill re-derives"
    )

    ids, _ = dk.reference_top_k(rows, top_k=K20)
    assert list(ids[0]) == list(range(K20))
    assert set(int(v) for v in ids[0]) == _deterministic_support_set(rows[0], K20)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_device_selector_set_equals_the_deterministic_selector_with_ties(seed):
    """Random rows with heavy value ties, against the fallback's own contract."""

    rng = np.random.default_rng(seed)
    vocab = 2048
    row = rng.choice(
        np.linspace(-4.0, 4.0, 24).astype(np.float32), size=vocab
    ).astype(np.float32)
    ids, _ = dk.reference_top_k(row[None, :], top_k=K20)
    assert set(int(v) for v in ids[0]) == _deterministic_support_set(row, K20)
    # And the emitted order really is value desc / id asc.
    keys = [(-float(row[t]), int(t)) for t in ids[0]]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# The float32 draft-row preparation the device sampler walks
# ---------------------------------------------------------------------------


def test_numpy_reductions_match_the_kernels_own_emulations():
    """``prepare_draft_row_f32`` leans on two NumPy reduction identities."""

    oracle = _choice_oracle()
    rng = np.random.default_rng(5)
    for _ in range(400):
        n = int(rng.integers(1, K20 + 1))
        x = (rng.random(n) * 10.0 ** rng.integers(-8, 1, size=n)).astype(np.float32)
        assert int(
            np.float32(np.sum(x, dtype=np.float32)).view(np.uint32)
        ) == int(oracle._numpy_pairwise_sum(list(x)).view(np.uint32))
        sequential = []
        acc = np.float32(0.0)
        for value in x:
            sequential.append(acc)
            acc = np.float32(acc + np.float32(value))
        assert np.array_equal(
            np.asarray(sequential, dtype=np.float32).view(np.uint32),
            np.concatenate(
                (np.float32([0.0]), np.cumsum(x, dtype=np.float32)[:-1])
            ).view(np.uint32),
        )


@pytest.mark.parametrize("top_p", [0.95, 1.0, 0.5])
def test_prepare_draft_row_matches_the_choice_kernel_oracle(top_p):
    oracle = _choice_oracle()
    rng = np.random.default_rng(3)
    for _ in range(250):
        ids = rng.choice(1 << 17, size=K20, replace=False).astype(np.uint32)
        values = (rng.standard_normal(K20) * 4.0).astype(np.float32)
        raw = np.exp(values.astype(np.float64) - values.max())
        probs = (raw / raw.sum() * rng.uniform(0.1, 1.0)).astype(np.float32)

        want_ids, want_cdf = oracle._prepare_reference_row(
            ids, values, probs, np.float32(top_p)
        )
        got_ids, got_norm = dk.prepare_draft_row_f32(ids, values, probs, top_p)
        got_cdf = np.cumsum(got_norm, dtype=np.float32)
        assert np.array_equal(np.asarray(want_ids, dtype=np.uint32), got_ids)
        assert np.array_equal(
            np.asarray(want_cdf, dtype=np.float32).view(np.uint32),
            got_cdf.view(np.uint32),
        )


def test_prepare_draft_row_handles_value_ties():
    oracle = _choice_oracle()
    rng = np.random.default_rng(21)
    for _ in range(200):
        ids = rng.choice(1 << 12, size=K20, replace=False).astype(np.uint32)
        values = rng.choice(
            np.array([0.0, -0.0, 1.0, 2.0, -3.5], dtype=np.float32), size=K20
        ).astype(np.float32)
        raw = np.exp(values.astype(np.float64) - values.max())
        probs = (raw / raw.sum()).astype(np.float32)
        want_ids, want_cdf = oracle._prepare_reference_row(
            ids, values, probs, np.float32(0.95)
        )
        got_ids, got_norm = dk.prepare_draft_row_f32(ids, values, probs, 0.95)
        assert np.array_equal(np.asarray(want_ids, dtype=np.uint32), got_ids)
        assert np.array_equal(
            np.asarray(want_cdf, dtype=np.float32).view(np.uint32),
            np.cumsum(got_norm, dtype=np.float32).view(np.uint32),
        )


def test_draft_distribution_is_the_device_selection_law():
    """``q_test`` must be the law the device's float32 CDF walk realises."""

    oracle = _choice_oracle()
    rng = np.random.default_rng(31)
    ids = rng.choice(1 << 16, size=K20, replace=False).astype(np.uint32)
    values = (rng.standard_normal(K20) * 3.0).astype(np.float32)
    raw = np.exp(values.astype(np.float64) - values.max())
    probs = (raw / raw.sum()).astype(np.float32)

    distribution, normalized = dk.draft_distribution(
        ids, values, probs, top_p=0.95, vocab_size=1 << 17
    )
    assert np.all(np.diff(distribution.token_ids) > 0)     # ascending support
    assert abs(float(distribution.probs.sum()) - 1.0) < 1e-12

    ordered_ids, _ = dk.prepare_draft_row_f32(ids, values, probs, 0.95)
    assert list(distribution.token_ids) == [int(v) for v in ordered_ids]

    # Every token the device can emit carries positive mass in q, and the
    # mass matches the CDF gap the exact-rational comparison hands it.
    cdf = np.cumsum(normalized, dtype=np.float32)
    grid = np.asarray(
        [np.ldexp(float(n), -53) for n in (0, 1, 1 << 40, (1 << 53) - 1)],
        dtype=np.float64,
    )
    for uniform in grid:
        token = oracle.reference_literal_divided_cdf_token(
            ids, values, probs, np.float64(uniform), top_p=0.95
        )
        assert token in set(int(v) for v in distribution.token_ids)
        index = int(np.nonzero(distribution.token_ids == token)[0][0])
        gap = float(cdf[index]) - (0.0 if index == 0 else float(cdf[index - 1]))
        assert gap >= 0.0
        assert distribution.probs[index] == pytest.approx(
            gap / float(cdf[-1]), rel=1e-12, abs=1e-18
        )


# ---------------------------------------------------------------------------
# Draw accounting
# ---------------------------------------------------------------------------


def test_draft_uniforms_consume_the_same_stream_as_rng_choice():
    """Flag-on must leave the PCG64 cursor exactly where flag-off leaves it."""

    ids = np.array([3, 9, 27], dtype=np.int64)
    probs = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    flag_off = np.random.default_rng(4242)
    off_tokens = [int(flag_off.choice(ids, p=probs)) for _ in range(DEPTH)]
    off_next = flag_off.random(4)

    flag_on = np.random.default_rng(4242)
    uniforms = dk.draw_draft_uniforms(flag_on, DEPTH)
    on_next = flag_on.random(4)

    assert uniforms.shape == (DEPTH,)
    assert np.array_equal(off_next, on_next), "the accept coins must be unshifted"

    # And each uniform is exactly the double that CDF walk consumed.
    cdf = np.cumsum(probs)
    cdf = cdf / cdf[-1]
    replayed = [int(ids[int(np.searchsorted(cdf, u, side="right"))]) for u in uniforms]
    assert replayed == off_tokens


def test_draft_uniforms_equal_independent_random_calls():
    a = np.random.default_rng(7)
    b = np.random.default_rng(7)
    assert np.array_equal(
        dk.draw_draft_uniforms(a, 5),
        np.asarray([b.random() for _ in range(5)], dtype=np.float64),
    )


def test_uniform_descriptors_round_trip_through_the_audited_builder():
    oracle = _choice_oracle()
    uniforms = np.random.default_rng(8).random(DEPTH)
    descriptors = dk.build_uniform_descriptors(uniforms)
    assert descriptors.shape == (DEPTH, oracle.MIDPOINT_DESCRIPTOR_WORDS)
    assert descriptors.dtype == np.uint32
    oracle.validate_pcg64_midpoint_descriptors(descriptors)


# ---------------------------------------------------------------------------
# Construction-time eligibility -- fail closed, never silently
# ---------------------------------------------------------------------------


def _claim(monkeypatch, **overrides):
    monkeypatch.setattr(dk, "_ENABLED", True)
    kwargs = dict(
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=K20),
        draft_sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=K20),
        speculative_depth=DEPTH,
        rng=np.random.default_rng(0),
        fused_verify_input=False,
        target_prefix_verify=False,
        lazy_target_distributions=False,
        lazy_bonus_verify_requested=False,
        batch_target_arrays=True,
        steer_active=False,
        penalties_active=False,
        constraint=None,
        adaptive_policy=None,
        adaptive_width_policy=None,
        mtp_corrector=None,
        mtp_topk_reranker=None,
        draft_margin_threshold=None,
        online_hidden_corrector_alpha=0.0,
        online_correction_cache=False,
        prompt_correction_cache=False,
        adapter_ensemble_q=False,
        combine_greedy_draft_read=False,
        greedy_chain_enabled=False,
        draft_confidence_needed=False,
        frspec_legacy_ids=None,
        late_depth_switch_after=0,
        a3b_target_prefix_route=None,
        pr391_route=None,
    )
    kwargs.update(overrides)
    return dk.claim_request_route(**kwargs)


@pytest.mark.parametrize(
    "override",
    [
        {"target_prefix_verify": True},
        {"batch_target_arrays": False},
        {"lazy_target_distributions": True},
        {"lazy_bonus_verify_requested": True},
        {"steer_active": True},
        {"penalties_active": True},
        {"constraint": object()},
        {"adaptive_policy": object()},
        {"adaptive_width_policy": object()},
        {"mtp_corrector": object()},
        {"mtp_topk_reranker": object()},
        {"draft_margin_threshold": 0.5},
        {"online_hidden_corrector_alpha": 0.1},
        {"online_correction_cache": True},
        {"prompt_correction_cache": True},
        {"adapter_ensemble_q": True},
        {"combine_greedy_draft_read": True},
        {"greedy_chain_enabled": True},
        {"draft_confidence_needed": True},
        {"frspec_legacy_ids": np.arange(4)},
        {"late_depth_switch_after": 32},
        {"a3b_target_prefix_route": object()},
        {"pr391_route": object()},
        {"adaptive_dtemp_active": True},
        {"sampler": SamplerConfig(temperature=0.0, top_p=0.95, top_k=K20)},
        {"draft_sampler": SamplerConfig(temperature=0.0, top_p=0.95, top_k=K20)},
        {"sampler": SamplerConfig(temperature=0.6, top_p=0.95, top_k=40)},
        {"sampler": SamplerConfig(temperature=0.6, top_p=0.0, top_k=K20)},
        {"sampler": SamplerConfig(temperature=0.6, top_p=0.95, top_k=K20,
                                  presence_penalty=0.5)},
        {"speculative_depth": 0},
        {"draft_core": "device"},
    ],
)
def test_unsupported_requests_decline_to_the_stock_lane(monkeypatch, override):
    """Request-shaped ineligibility stands aside; it does not raise.

    Every override here is a property of ONE REQUEST.  Raising made each of
    them an HTTP 500 in serving even though the stock selector serves them
    perfectly (composed-stack HumanEval gate, 2026-09-02).
    """

    monkeypatch.setattr(contract, "_STRICT", False)
    contract.reset_for_test()
    receipt: dict[str, object] = {}
    assert _claim(monkeypatch, receipt=receipt, **override) is None
    assert receipt["installed"] is False
    assert receipt["declined"]
    assert contract.decline_counts(dk._ENV_VAR)[receipt["declined"]] == 1


@pytest.mark.parametrize(
    "override",
    [
        {"greedy_chain_enabled": True},
        {"sampler": SamplerConfig(temperature=0.0, top_p=0.95, top_k=K20)},
        {"draft_core": "device"},
    ],
)
def test_strict_claims_turns_a_decline_back_into_a_failure(monkeypatch, override):
    """A measured arm still fails closed under MTPLX_FABLE_STRICT_CLAIMS."""

    monkeypatch.setattr(contract, "_STRICT", True)
    with pytest.raises(dk.DeviceK20Ineligible):
        _claim(monkeypatch, **override)


def test_install_time_contract_violations_still_raise(monkeypatch):
    """A wrong bit generator is not a request shape -- no request could work.

    The rng comes from `generate_mtpk`'s own seeding, so this can only mean
    the process is built wrong.  Every request would fail identically, so the
    first one fails loudly instead of silently running a slower lane forever.
    """

    monkeypatch.setattr(contract, "_STRICT", False)
    with pytest.raises(dk.DeviceK20Ineligible, match="PCG64"):
        _claim(monkeypatch, rng=np.random.Generator(np.random.Philox(1)))


def test_supported_request_builds_a_plan(monkeypatch):
    monkeypatch.setattr(dk, "_ENABLED", True)
    # The choice-kernel bind imports MLX; stub it so this stays CPU-only.
    import sys
    import types

    stub = types.ModuleType("mtplx.kernels.qwen4_frspec_k20_float32_choice")
    stub.bind_qwen4_frspec_k20_float32_choice = lambda **_kw: (lambda *a: a)
    monkeypatch.setitem(
        sys.modules, "mtplx.kernels.qwen4_frspec_k20_float32_choice", stub
    )
    plan = _claim(monkeypatch)
    assert isinstance(plan, dk.DeviceK20Plan)
    assert plan.depth == DEPTH
    assert plan.top_k == K20
    assert plan.top_p_active is True
    assert plan.fused_verify_input is False
    assert plan.to_dict()["layout"] == dk.K20_LOG_LAYOUT


# ---------------------------------------------------------------------------
# K20 logger round trip
# ---------------------------------------------------------------------------


class _Sparse:
    def __init__(self, ids, probs):
        self.token_ids = np.asarray(ids, dtype=np.int64)
        self.probs = np.asarray(probs, dtype=np.float64)


class _Batched:
    def __init__(self, rows):
        self.token_ids = np.stack([row.token_ids for row in rows])
        self.probs = np.stack([row.probs for row in rows])


class _SimpleSampler:
    def __init__(self, temperature, top_p, top_k):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


def test_device_k20_layout_round_trips_with_the_draft_uniforms(tmp_path):
    log_mod._configure_for_test(str(tmp_path / "rows.npz"))
    try:
        logger = log_mod.k20_log
        logger.set_sampler(
            sampler=_SimpleSampler(0.6, 0.95, K20),
            draft_sampler=_SimpleSampler(0.6, 0.95, K20),
        )
        uniforms = [0.125, 0.5, 0.875]
        logger.stock_open(
            primary=7,
            draft_tokens=[11, 22, 33],
            draft_probs=[
                _Sparse([11, 12], [0.7, 0.3]),
                _Sparse([22, 23], [0.6, 0.4]),
                _Sparse([33, 34], [0.9, 0.1]),
            ],
            target_batch=_Batched(
                [
                    _Sparse([11, 12], [0.7, 0.3]),
                    _Sparse([22, 23], [0.5, 0.5]),
                    _Sparse([33, 34], [0.8, 0.2]),
                    _Sparse([44, 45], [0.6, 0.4]),
                ]
            ),
            bonus_allowed=True,
            greedy=False,
            rng=np.random.default_rng(1),
            device_k20=True,
            draft_uniforms=uniforms,
        )
        for depth in range(DEPTH):
            logger.stock_depth(
                depth,
                accept_prob=0.5,
                coin=0.1,
                accepted=True,
                correction=[11, 22, 33][depth],
            )
        logger.stock_bonus(44)
        logger.stock_close()
        out = logger.flush()
        with np.load(out) as handle:
            data = {key: handle[key] for key in handle.files}
    finally:
        log_mod._configure_for_test(None)

    assert str(data["layout"]) == log_mod.LAYOUT_STOCK_DEVICE_K20
    assert int(data["has_raw_logits"]) == 0
    np.testing.assert_allclose(data["draft_uniforms"][0], uniforms)
    assert data["draft_ids"].shape == (1, DEPTH, K20)
    assert data["target_ids"].shape == (1, DEPTH + 1, K20)
    # The device layout is NOT the block-verify layout: no ladder columns.
    assert "block_valid" not in data
    # ...but it IS a stock layout, so the depth-4 gate feature rides it, and
    # it is `q(x_d)` under the law the DEVICE sampled from (the rows handed to
    # `stock_open` are the device chain's own `draft_distribution` output).
    np.testing.assert_allclose(data["gate_q"][0], [0.7, 0.6, 0.9])
    # No probe ran, so the optional probe block is absent.
    assert "probe_valid" not in data


def test_the_depth4_probe_columns_ride_a_device_k20_log(tmp_path):
    """A probed device-lane window must carry the same probe block as stock."""

    log_mod._configure_for_test(str(tmp_path / "rows.npz"))
    try:
        logger = log_mod.k20_log
        logger.set_sampler(
            sampler=_SimpleSampler(0.6, 0.95, K20),
            draft_sampler=_SimpleSampler(0.6, 0.95, K20),
        )
        logger.stock_open(
            primary=7,
            draft_tokens=[11, 22, 33],
            draft_probs=[
                _Sparse([11, 12], [0.7, 0.3]),
                _Sparse([22, 23], [0.6, 0.4]),
                _Sparse([33, 34], [0.9, 0.1]),
            ],
            target_batch=_Batched(
                [
                    _Sparse([11, 12], [0.7, 0.3]),
                    _Sparse([22, 23], [0.5, 0.5]),
                    _Sparse([33, 34], [0.8, 0.2]),
                    _Sparse([44, 45], [0.6, 0.4]),
                ]
            ),
            bonus_allowed=True,
            greedy=False,
            rng=np.random.default_rng(1),
            device_k20=True,
            draft_uniforms=[0.1, 0.2, 0.3],
        )
        for depth in range(DEPTH):
            logger.stock_depth(
                depth,
                accept_prob=1.0,
                coin=0.0,
                accepted=True,
                correction=[11, 22, 33][depth],
            )
        # The all-accept branch's hook, on the device lane.
        logger.stock_depth4(ids=[44, 45], probs=[0.55, 0.45], trimmed=False)
        logger.stock_bonus(44)
        logger.stock_close()
        out = logger.flush()
        with np.load(out) as handle:
            data = {key: handle[key] for key in handle.files}
    finally:
        log_mod._configure_for_test(None)

    assert str(data["layout"]) == log_mod.LAYOUT_STOCK_DEVICE_K20
    assert int(data["probe_valid"][0]) == 1
    assert list(data["probe_ids"][0][:2]) == [44, 45]
    np.testing.assert_allclose(data["probe_probs"][0][:2], [0.55, 0.45])
    assert int(data["probe_trimmed"][0]) == 0
    np.testing.assert_allclose(data["gate_q"][0], [0.7, 0.6, 0.9])


def test_device_k20_and_block_verify_compose_into_one_layout(tmp_path):
    log_mod._configure_for_test(str(tmp_path / "rows.npz"))
    try:
        logger = log_mod.k20_log
        logger.set_sampler(
            sampler=_SimpleSampler(0.6, 0.95, K20),
            draft_sampler=_SimpleSampler(0.6, 0.95, K20),
        )
        logger.stock_open(
            primary=1,
            draft_tokens=[11],
            draft_probs=[_Sparse([11, 12], [0.7, 0.3])],
            target_batch=_Batched([_Sparse([11, 12], [0.6, 0.4])]),
            bonus_allowed=False,
            greedy=False,
            rng=np.random.default_rng(2),
            block_verify=True,
            block={
                "coin": [0.5],
                "scale": [1.0],
                "budget": [1.0],
                "realised": [1.0],
                "clipped": [0],
            },
            device_k20=True,
            draft_uniforms=[0.25],
        )
        logger.stock_close()
        out = logger.flush()
        with np.load(out) as handle:
            layout = str(handle["layout"])
            has_block = "block_valid" in handle.files
    finally:
        log_mod._configure_for_test(None)

    assert layout == log_mod.LAYOUT_STOCK_DEVICE_K20_BV
    assert has_block


def test_stock_layout_still_leaves_the_draft_uniforms_nan(tmp_path):
    log_mod._configure_for_test(str(tmp_path / "rows.npz"))
    try:
        logger = log_mod.k20_log
        logger.set_sampler(
            sampler=_SimpleSampler(0.6, 0.95, K20),
            draft_sampler=_SimpleSampler(0.6, 0.95, K20),
        )
        logger.stock_open(
            primary=1,
            draft_tokens=[11],
            draft_probs=[_Sparse([11, 12], [0.7, 0.3])],
            target_batch=_Batched([_Sparse([11, 12], [0.6, 0.4])]),
            bonus_allowed=False,
            greedy=False,
            rng=np.random.default_rng(3),
        )
        logger.stock_close()
        out = logger.flush()
        with np.load(out) as handle:
            layout = str(handle["layout"])
            uniforms = handle["draft_uniforms"].copy()
    finally:
        log_mod._configure_for_test(None)

    assert layout == log_mod.LAYOUT_STOCK
    assert np.all(np.isnan(uniforms))


# --------------------------------------------------------------------------
# Greedy vs temperature-1: which shape gets which path
# --------------------------------------------------------------------------


def test_a_greedy_request_is_routed_to_the_greedy_chain(monkeypatch):
    """This is a SAMPLED-lane route; greedy has its own optimised path.

    A greedy request consumes no PCG64 uniform and builds no top-20 support,
    so there is nothing here for it to use. It is not left unoptimised
    either: `generate_mtpk`'s one-sync greedy chain serves it, and with
    MTPLX_FABLE_DRAFT_K20_PRESCATTER armed that chain reads the same
    65,536-row pre-scatter output this route would have.
    """

    monkeypatch.setattr(contract, "_STRICT", False)
    contract.reset_for_test()
    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    receipt: dict[str, object] = {}
    assert (
        _claim(monkeypatch, receipt=receipt, sampler=greedy, draft_sampler=greedy)
        is None
    )
    assert receipt["declined"] == "greedy_request"
    assert "temperature > 0" in str(receipt["declined_detail"])


def test_a_temperature_one_request_still_claims(monkeypatch):
    monkeypatch.setattr(contract, "_STRICT", False)
    monkeypatch.setattr(dk, "_ENABLED", True)
    import sys
    import types

    stub = types.ModuleType("mtplx.kernels.qwen4_frspec_k20_float32_choice")
    stub.bind_qwen4_frspec_k20_float32_choice = lambda **_kw: (lambda *a: a)
    monkeypatch.setitem(
        sys.modules, "mtplx.kernels.qwen4_frspec_k20_float32_choice", stub
    )
    sampled = SamplerConfig(temperature=1.0, top_p=0.95, top_k=K20)
    plan = _claim(monkeypatch, sampler=sampled, draft_sampler=sampled)
    assert isinstance(plan, dk.DeviceK20Plan)
