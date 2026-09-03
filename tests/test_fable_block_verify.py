"""Pure-python coverage for block verification on the stock native-MTP lane.

No MLX import happens here.  ``mtplx.fable_block_verify`` and
``mtplx.sampling`` are plain NumPy, so the law can be driven directly; the
generation-side wiring is checked by source inspection, which is how the rest
of this lane is guarded.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from mtplx import fable_block_verify as bv_mod
from mtplx.fable_block_verify import (
    BlockVerifier,
    build_verifier,
    prepared_pair,
    water_fill_lambda,
)
from mtplx.sampling import (
    SparseDistribution,
    acceptance_probability,
    residual_distribution,
    sample_from_distribution,
)



REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Row helpers -- one set of prepared rows, shared by both implementations.
# ---------------------------------------------------------------------------


def _sparse(pairs: dict[int, float]) -> SparseDistribution:
    ids = np.asarray(sorted(pairs), dtype=np.int64)
    probs = np.asarray([pairs[int(token)] for token in ids], dtype=np.float64)
    return SparseDistribution(ids, probs, 100000)


def _rows(draft_pairs, target_pairs):
    draft = [_sparse(pairs) for pairs in draft_pairs]
    target = [_sparse(pairs) for pairs in target_pairs]
    return draft, target




def _verifier(draft, target, tokens) -> BlockVerifier:
    verifier = build_verifier(
        draft_tokens=tokens, draft_probs=draft, target_list=target
    )
    assert verifier is not None
    return verifier


# A ladder that drops under water at depth 1 (the target dislikes x1), then
# meets a depth-2 token the target likes far more than the draft did -- the
# shape block verification exists to exploit.
_UNDERWATER_DRAFT = [
    {1: 0.22, 2: 0.36, 3: 0.42},
    {4: 0.40, 5: 0.30, 6: 0.30},
    {7: 0.55, 8: 0.35, 9: 0.10},
]
_UNDERWATER_TARGET = [
    {1: 0.20, 2: 0.46, 3: 0.34},
    {4: 0.30, 5: 0.26, 6: 0.44},
    {7: 0.52, 8: 0.30, 9: 0.18},
    {10: 0.60, 11: 0.40},
]
_UNDERWATER_TOKENS = [1, 4, 7]

# A ladder that stays at 1 through depths 1-2 (rho >= 1 there, so the running
# product never dips) and only then meets a token the target likes less.  The
# block law must be the shipped law token for token on such a window: both the
# coin and the residual scale collapse.  The last depth is the ONLY place a
# rejection can happen on a unit ladder -- rho >= 1 everywhere else means
# a_d = 1 there -- which is exactly what makes this the partial parity check.
_UNIT_DRAFT = [
    {1: 0.50, 2: 0.50},
    {4: 0.60, 5: 0.40},
    {7: 0.30, 8: 0.70},
]
_UNIT_TARGET = [
    {1: 0.80, 2: 0.20},
    {4: 0.75, 5: 0.25},
    {7: 0.20, 8: 0.80},
    {10: 1.0},
]
_UNIT_TOKENS = [1, 4, 7]

# Five-wide rows whose depth-2 residual keeps more than one token, so the
# scale's effect on the correction's SHAPE (not just its support) is visible.
_WIDE_DRAFT = [
    {1: 0.32, 2: 0.01, 3: 0.33, 4: 0.06, 5: 0.29},
    {6: 0.28, 7: 0.05, 8: 0.23, 9: 0.10, 10: 0.34},
    {11: 0.23, 12: 0.20, 13: 0.24, 14: 0.06, 15: 0.26},
]
_WIDE_TARGET = [
    {1: 0.19, 2: 0.24, 3: 0.08, 4: 0.02, 5: 0.47},
    {6: 0.05, 7: 0.21, 8: 0.32, 9: 0.19, 10: 0.24},
    {11: 0.13, 12: 0.24, 13: 0.18, 14: 0.19, 15: 0.25},
    {20: 0.60, 21: 0.40},
]
_WIDE_TOKENS = [1, 6, 11]


# ---------------------------------------------------------------------------
# A faithful replica of the stock accept loop, so both laws can be driven and
# their RNG traffic counted without importing generation.py (which needs MLX).
# The AST contract below pins the real loop to this shape.
# ---------------------------------------------------------------------------


class _CountingRNG:
    """Wraps a Generator and counts the two draw kinds the loop can make."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.randoms = 0
        self.choices = 0

    def random(self) -> float:
        self.randoms += 1
        return float(self._rng.random())

    def choice(self, values, *, p):
        self.choices += 1
        return self._rng.choice(values, p=p)

    @property
    def draws(self) -> int:
        return self.randoms + self.choices








# ---------------------------------------------------------------------------
# (a) The flag-off contract on the real accept loop.
# ---------------------------------------------------------------------------


def _generation_source() -> str:
    return (REPO_ROOT / "mtplx" / "generation.py").read_text()


def _accept_loop_source(source: str) -> str:
    """The stock native-MTP accept loop, from its `for` to the round's end."""

    start = source.index(
        "for depth_index, draft_token in enumerate(_host_accept_drafts):"
    )
    end = source.index("elapsed_accept = max(", start)
    return source[start:end]


def test_the_gate_is_read_once_at_import_and_only_in_one_place():
    source = _generation_source()
    assert "_FABLE_BLOCK_VERIFY = _fable_block_verify_enabled()" in source
    # generation.py never reads the variable itself.
    assert 'os.environ.get("MTPLX_FABLE_BLOCK_VERIFY"' not in source
    assert '_env_truthy("MTPLX_FABLE_BLOCK_VERIFY")' not in source
    module = (REPO_ROOT / "mtplx" / "fable_block_verify.py").read_text()
    assert module.count('_ENV_VAR = "MTPLX_FABLE_BLOCK_VERIFY"') == 1
    assert module.count("_env_truthy(_ENV_VAR)") == 1
    assert "os.environ" not in inspect.getsource(bv_mod.BlockVerifier)
    assert "os.environ" not in inspect.getsource(bv_mod.build_verifier)


def test_the_shipped_law_survives_verbatim_in_the_accept_loop():
    """Flag off => the loop evaluates exactly what it evaluated before."""

    loop = _accept_loop_source(_generation_source())
    # Both sampled branches keep their original acceptance expression...
    assert (
        "accept_prob = (\n"
        "                    1.0 if q <= 0 and p > 0 else "
        "(0.0 if q <= 0 else min(1.0, p / q))\n"
        "                )" in loop
    )
    assert "accept_prob = compute_acceptance_probability(\n" in loop
    # ...and their original residual, reached whenever `_bv` is None.
    assert "residual_distribution(target_p, draft_q), rng" in loop
    assert "residual_distribution(\n" in loop
    assert "else target_distribution_batch.to_distribution(depth_index)," in loop
    # The accept coin is still drawn once per depth and compared with `<=`,
    # so arming block verification cannot shift the PCG64 stream.
    assert loop.count("_k20_coin = float(rng.random())") == 2
    assert loop.count("accepted_now = _k20_coin <= accept_prob") == 2
    assert loop.count("rng.random()") == 2


def test_every_block_verification_read_is_guarded():
    loop = _accept_loop_source(_generation_source())
    # Each mention of the verifier is inside an `if/elif _bv is not None`.
    reads = [line.strip() for line in loop.splitlines() if "_bv." in line]
    assert reads, "the loop must read the ladder somewhere"
    for index, line in enumerate(loop.splitlines()):
        if "_bv." not in line:
            continue
        guard = "\n".join(loop.splitlines()[max(0, index - 6) : index])
        assert "_bv is not None" in guard, line
    # The two overrides and the two scaled residuals, and nothing else.
    assert loop.count("_bv.accept_probability[depth_index]") == 2
    assert loop.count("_bv.scaled_residual(depth_index)") == 2
    assert loop.count("elif _bv is not None:") == 2
    assert loop.count("if _bv is not None:") - loop.count(
        "elif _bv is not None:"
    ) == 2


def test_the_verifier_is_built_once_before_the_loop_and_is_gated():
    source = _generation_source()
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_mtpk"
    )
    body = ast.get_source_segment(source, function)
    built = body.index("_bv = _fable_build_block_verifier(")
    loop = body.index(
        "for depth_index, draft_token in enumerate(_host_accept_drafts):"
    )
    assert built < loop
    assert body.count("_fable_build_block_verifier(") == 1
    gate = body.rindex("_FABLE_BLOCK_VERIFY", 0, built)
    # The construction sits under the module-level gate, and under the
    # preconditions that make a block law meaningful at all.
    guard = body[gate:built]
    assert "sampler.temperature > 0" in guard
    assert "target_prefix_tokens is None" in guard
    assert "_bv = None" in body[max(0, gate - 200) : gate]


def test_no_rng_reaches_the_block_verification_module():
    """The law draws nothing, so arming it cannot shift the PCG64 stream."""

    path = REPO_ROOT / "mtplx" / "fable_block_verify.py"
    module = path.read_text()
    assert "import mlx" not in module and "mlx.core" not in module
    assert "np.random" not in module
    tree = ast.parse(module)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [argument.arg for argument in node.args.args]
            names += [argument.arg for argument in node.args.kwonlyargs]
            assert "rng" not in names, node.name
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"random", "choice", "default_rng"}, node.attr
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "random" not in ast.dump(node)


# ---------------------------------------------------------------------------
# (b) The in-loop law equals the offline reference.
# ---------------------------------------------------------------------------










def test_the_scaled_residual_reshapes_the_correction_under_water():
    """(c*p - q)+ is not (p - q)+ rescaled: the scale moves support AND shape."""

    draft, target = _rows(_WIDE_DRAFT, _WIDE_TARGET)
    verifier = _verifier(draft, target, _WIDE_TOKENS)
    assert verifier.residual_scale[1] < 1.0
    scaled = verifier.scaled_residual(1)
    shipped = residual_distribution(target[1], draft[1])
    assert scaled.token_ids.size >= 2
    assert set(scaled.token_ids.tolist()) < set(shipped.token_ids.tolist())
    assert not np.allclose(scaled.probs[:1], shipped.probs[:1])

    # ... and it IS the shipped residual where the ladder is still at 1.
    assert verifier.residual_scale[0] == 1.0
    at_depth_zero = verifier.scaled_residual(0)
    reference = residual_distribution(target[0], draft[0])
    np.testing.assert_array_equal(at_depth_zero.token_ids, reference.token_ids)
    np.testing.assert_allclose(at_depth_zero.probs, reference.probs, rtol=0, atol=1e-15)












def test_the_verifier_declines_when_a_row_is_missing():
    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    assert (
        build_verifier(
            draft_tokens=_UNDERWATER_TOKENS,
            draft_probs=[draft[0], None, draft[2]],
            target_list=target,
        )
        is None
    )
    assert (
        build_verifier(
            draft_tokens=_UNDERWATER_TOKENS,
            draft_probs=draft,
            target_list=target[:2],
        )
        is None
    )
    assert (
        build_verifier(
            draft_tokens=_UNDERWATER_TOKENS, draft_probs=draft, target_list=None
        )
        is None
    )


def test_the_verifier_reads_a_batched_support_the_same_way():
    class _Batch:
        vocab_size = 100000

        def __init__(self, rows):
            width = max(row.token_ids.size for row in rows)
            self.token_ids = np.zeros((len(rows), width), dtype=np.int64)
            self.probs = np.zeros((len(rows), width), dtype=np.float64)
            for index, row in enumerate(rows):
                size = int(row.token_ids.size)
                self.token_ids[index, :size] = row.token_ids
                self.probs[index, :size] = row.probs

    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    from_list = _verifier(draft, target, _UNDERWATER_TOKENS)
    from_batch = build_verifier(
        draft_tokens=_UNDERWATER_TOKENS,
        draft_probs=draft,
        target_batch=_Batch(target),
    )
    assert from_batch is not None
    np.testing.assert_array_equal(
        from_batch.accept_probability, from_list.accept_probability
    )
    np.testing.assert_array_equal(
        from_batch.residual_scale, from_list.residual_scale
    )


def test_the_water_fill_holds_the_budget_in_expectation():
    """E_{x_{d+1}~q}[w_d] = A_d -- the property that keeps the law exact."""

    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    ids, probs = prepared_pair(draft[1])
    realised = []
    for token in ids:
        tokens = [_UNDERWATER_TOKENS[0], int(token), _UNDERWATER_TOKENS[2]]
        verifier = _verifier(draft, target, tokens)
        realised.append(verifier.realised[0])
    expectation = float(np.sum(probs * np.asarray(realised, dtype=np.float64)))
    reference = _verifier(draft, target, _UNDERWATER_TOKENS)
    assert expectation == pytest.approx(reference.budget[0], abs=1e-12)


def test_water_fill_lambda_is_the_reference_solver():
    q = np.array([0.5, 0.3, 0.2])
    base = np.array([0.1, 0.4, 0.9])
    level = water_fill_lambda(q, base, np.float64(1.0), np.float64(0.6))
    assert float(np.sum(q * np.minimum(1.0, base + level))) == pytest.approx(0.6)
    # Already at or above the budget: no water is added.
    assert float(water_fill_lambda(q, base, np.float64(1.0), np.float64(0.2))) == 0.0


# ---------------------------------------------------------------------------
# (c) Draw accounting.
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# (d) The K20 log round-trips through the offline replay.
# ---------------------------------------------------------------------------




class _SimpleSampler:
    def __init__(self, temperature=1.0, top_p=0.95, top_k=20):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k














def test_the_guard_routes_greedy_windows_instead_of_raising():
    """Block verification is a SPECULATIVE-SAMPLING acceptance scheme.

    A greedy window has no draft distributions at all, so there is no ladder
    to build. The guard says so with a plain `sampler.temperature > 0` term:
    the greedy window takes the shipped accept path, nothing raises, and the
    temperature-1 window still gets the ladder.
    """

    import ast
    import inspect

    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    tree = ast.parse(
        "def f():\n" + "\n".join("    " + line for line in source.splitlines())
    )
    condition = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "_FABLE_BLOCK_VERIFY" in ast.unparse(node.test):
            condition = ast.unparse(node.test)
            break
    assert condition is not None, "the block-verify guard moved"
    assert "sampler.temperature > 0" in condition
    # ...and the whole flag surface raises nothing per request.
    assert "raise" not in inspect.getsource(
        __import__("mtplx.fable_block_verify", fromlist=["build_verifier"]).build_verifier
    )
