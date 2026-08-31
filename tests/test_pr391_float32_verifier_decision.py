from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


decision = importlib.import_module(
    "mtplx.kernels.pr391_float32_verifier_decision"
)


def _fadd(left: np.float32, right: np.float32) -> np.float32:
    return np.float32(left + right)


def _lookup(ids: np.ndarray, probs: np.ndarray, token: int) -> np.float32:
    for index in range(ids.size):
        if int(ids[index]) == token:
            return np.float32(probs[index])
    return np.float32(0.0)


def _sample_literal(
    ids: np.ndarray,
    probs: np.ndarray,
    uniform: np.float32,
) -> int:
    ordered_ids = [
        int(token) for token, prob in zip(ids, probs, strict=True) if prob > 0
    ]
    weights = [
        np.float32(prob) for prob in probs if np.float32(prob) > np.float32(0.0)
    ]
    total = np.float32(0.0)
    for weight in weights:
        total = _fadd(total, weight)
    cumulative = np.float32(0.0)
    for token, weight in zip(ordered_ids[:-1], weights[:-1], strict=True):
        cumulative = _fadd(cumulative, weight)
        if uniform < np.float32(cumulative / total):
            return token
    return ordered_ids[-1]


def _literal_oracle(
    draft_tokens: np.ndarray,
    draft_ids: np.ndarray,
    draft_probs: np.ndarray,
    target_ids: np.ndarray,
    target_probs: np.ndarray,
    uniforms: np.ndarray,
    stop_ids: np.ndarray,
    stop_count: int,
    bonus_allowed: bool,
) -> tuple[int, int, int, int, int, int, np.ndarray]:
    accept_probs = np.zeros(3, dtype=np.float32)
    active_stops = {int(token) for token in stop_ids[:stop_count]}
    for depth in range(3):
        token = int(draft_tokens[depth])
        p_value = _lookup(target_ids[depth], target_probs[depth], token)
        q_value = _lookup(draft_ids[depth], draft_probs[depth], token)
        if q_value <= np.float32(0.0):
            accept_prob = np.float32(1.0 if p_value > np.float32(0.0) else 0.0)
        else:
            accept_prob = np.minimum(
                np.float32(1.0), np.float32(p_value / q_value)
            )
        accept_probs[depth] = accept_prob
        if uniforms[depth] <= accept_prob:
            if token in active_stops:
                return (
                    depth + 1,
                    -1,
                    0,
                    decision.SELECTED_NONE,
                    0,
                    depth + 1,
                    accept_probs,
                )
            continue

        union_ids = np.array(
            sorted(
                set(map(int, target_ids[depth]))
                | set(map(int, draft_ids[depth]))
            ),
            dtype=np.uint32,
        )
        residual = np.array(
            [
                max(
                    np.float32(
                        _lookup(target_ids[depth], target_probs[depth], int(token))
                        - _lookup(draft_ids[depth], draft_probs[depth], int(token))
                    ),
                    np.float32(0.0),
                )
                for token in union_ids
            ],
            dtype=np.float32,
        )
        if not np.any(residual > np.float32(0.0)):
            selected = _sample_literal(
                target_ids[depth], target_probs[depth], uniforms[depth + 1]
            )
        else:
            selected = _sample_literal(union_ids, residual, uniforms[depth + 1])
        return (
            depth,
            depth,
            selected,
            decision.SELECTED_CORRECTION,
            1,
            depth + 2,
            accept_probs,
        )

    if not bonus_allowed:
        return (3, -1, 0, decision.SELECTED_NONE, 0, 3, accept_probs)
    bonus = _sample_literal(target_ids[3], target_probs[3], uniforms[3])
    return (3, -1, bonus, decision.SELECTED_BONUS, 1, 4, accept_probs)


def _row(
    entries: list[tuple[int, float]], *, filler_base: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(entries) > 20:
        raise AssertionError("fixture row exceeds K20")
    used = {token for token, _prob in entries}
    fillers: list[int] = []
    candidate = filler_base
    while len(entries) + len(fillers) < 20:
        if candidate not in used:
            fillers.append(candidate)
        candidate += 1
    ids = np.array([token for token, _prob in entries] + fillers, dtype=np.uint32)
    probs = np.array(
        [prob for _token, prob in entries] + [0.0] * len(fillers),
        dtype=np.float32,
    )
    return ids, probs


def _fixture() -> tuple[np.ndarray, ...]:
    draft_tokens = np.array([11, 22, 33], dtype=np.uint32)
    draft_rows = [
        _row([(11, 0.5), (101, 0.5)], filler_base=1000),
        _row([(22, 0.5), (102, 0.5)], filler_base=1100),
        _row([(33, 0.5), (103, 0.5)], filler_base=1200),
    ]
    target_rows = [
        _row([(11, 0.25), (201, 0.75)], filler_base=2000),
        _row([(22, 0.25), (202, 0.75)], filler_base=2100),
        _row([(33, 0.25), (203, 0.75)], filler_base=2200),
        _row([(301, 0.25), (302, 0.75)], filler_base=2300),
    ]
    draft_ids = np.stack([row[0] for row in draft_rows])
    draft_probs = np.stack([row[1] for row in draft_rows])
    target_ids = np.stack([row[0] for row in target_rows])
    target_probs = np.stack([row[1] for row in target_rows])
    uniforms = np.array([0.25, 0.25, 0.25, 0.5], dtype=np.float32)
    stop_ids = np.array([999, 998], dtype=np.uint32)
    return (
        draft_tokens,
        draft_ids,
        draft_probs,
        target_ids,
        target_probs,
        uniforms,
        stop_ids,
    )


def _observed(
    fixture: tuple[np.ndarray, ...],
    *,
    stop_count: int = 0,
    bonus_allowed: bool = True,
) -> tuple[int, int, int, int, int, int, np.ndarray]:
    result = decision.reference_pr391_float32_verifier_decision(
        *fixture,
        stop_count=stop_count,
        bonus_allowed=bonus_allowed,
    )
    return (
        int(result[0][0]),
        int(result[1][0]),
        int(result[2][0]),
        int(result[3][0]),
        int(result[4][0]),
        int(result[5][0]),
        result[6],
    )


def _expected(
    fixture: tuple[np.ndarray, ...],
    *,
    stop_count: int = 0,
    bonus_allowed: bool = True,
) -> tuple[int, int, int, int, int, int, np.ndarray]:
    return _literal_oracle(
        *fixture,
        stop_count=stop_count,
        bonus_allowed=bonus_allowed,
    )


@pytest.mark.parametrize("reject_depth", [0, 1, 2])
def test_rejects_at_each_depth_and_consumes_correction_draw(reject_depth: int) -> None:
    fixture = list(_fixture())
    uniforms = np.array([0.25, 0.25, 0.25, 0.5], dtype=np.float32)
    uniforms[reject_depth] = np.float32(0.75)
    fixture[5] = uniforms
    fixed_fixture = tuple(fixture)

    observed = _observed(fixed_fixture)
    expected = _expected(fixed_fixture)

    assert observed[:6] == expected[:6]
    np.testing.assert_array_equal(observed[6], expected[6])
    assert observed[0] == reject_depth
    assert observed[1] == reject_depth
    assert observed[3:6] == (
        decision.SELECTED_CORRECTION,
        1,
        reject_depth + 2,
    )


def test_all_accepted_samples_bonus_with_fourth_uniform() -> None:
    fixture = _fixture()

    observed = _observed(fixture)
    expected = _expected(fixture)

    assert observed[:6] == expected[:6]
    np.testing.assert_array_equal(observed[6], expected[6])
    assert observed[:6] == (3, -1, 302, decision.SELECTED_BONUS, 1, 4)


def test_bonus_sampling_preserves_runtime_target_support_order() -> None:
    fixture = list(_fixture())
    fixture[3][3], fixture[4][3] = _row(
        [(302, 0.25), (301, 0.75)], filler_base=6000
    )
    fixture[5] = np.array([0.25, 0.25, 0.25, 0.1], dtype=np.float32)
    fixed_fixture = tuple(fixture)

    observed = _observed(fixed_fixture)
    expected = _expected(fixed_fixture)

    assert observed[:6] == expected[:6]
    np.testing.assert_array_equal(observed[6], expected[6])
    assert observed[2] == 302


def test_accepted_stop_returns_without_sampling_or_consuming_later_draws() -> None:
    fixture = list(_fixture())
    fixture[6] = np.array([22, 777], dtype=np.uint32)
    fixed_fixture = tuple(fixture)

    observed = _observed(fixed_fixture, stop_count=1)
    expected = _expected(fixed_fixture, stop_count=1)

    assert observed[:6] == expected[:6]
    np.testing.assert_array_equal(observed[6], expected[6])
    assert observed[:6] == (2, -1, 0, decision.SELECTED_NONE, 0, 2)
    assert observed[6][2] == np.float32(0.0)


def test_bonus_disabled_returns_only_the_three_accepted_drafts() -> None:
    fixture = _fixture()

    observed = _observed(fixture, bonus_allowed=False)
    expected = _expected(fixture, bonus_allowed=False)

    assert observed[:6] == expected[:6]
    np.testing.assert_array_equal(observed[6], expected[6])
    assert observed[:6] == (3, -1, 0, decision.SELECTED_NONE, 0, 3)


def test_target_support_miss_forces_reject_and_uses_target_residual() -> None:
    fixture = list(_fixture())
    target_ids = fixture[3].copy()
    target_probs = fixture[4].copy()
    target_ids[0], target_probs[0] = _row(
        [(201, 0.25), (202, 0.75)], filler_base=3000
    )
    fixture[3] = target_ids
    fixture[4] = target_probs
    fixture[5] = np.array([0.01, 0.5, 0.25, 0.5], dtype=np.float32)
    fixed_fixture = tuple(fixture)

    observed = _observed(fixed_fixture)
    expected = _expected(fixed_fixture)

    assert observed[:6] == expected[:6]
    np.testing.assert_array_equal(observed[6], expected[6])
    assert observed[0:2] == (0, 0)
    assert observed[6][0] == np.float32(0.0)


def test_residual_sampling_uses_ascending_union_token_order() -> None:
    fixture = list(_fixture())
    fixture[1][0], fixture[2][0] = _row(
        [(90, 0.5), (10, 0.5)], filler_base=4000
    )
    fixture[3][0], fixture[4][0] = _row(
        [(70, 0.4), (30, 0.35), (90, 0.25)], filler_base=5000
    )
    fixture[0][0] = np.uint32(90)
    fixture[5] = np.array([0.75, 0.0, 0.25, 0.5], dtype=np.float32)
    fixed_fixture = tuple(fixture)

    observed = _observed(fixed_fixture)
    expected = _expected(fixed_fixture)

    assert observed[:6] == expected[:6]
    np.testing.assert_array_equal(observed[6], expected[6])
    assert observed[2] == 30


def test_empty_residual_falls_back_to_runtime_target_support_order() -> None:
    fixture = list(_fixture())
    fixture[1][0], fixture[2][0] = _row(
        [(11, 0.5), (91, 0.5)], filler_base=7000
    )
    fixture[3][0], fixture[4][0] = _row(
        [(91, 0.4), (11, 0.4)], filler_base=8000
    )
    fixture[5] = np.array([0.9, 0.1, 0.25, 0.5], dtype=np.float32)
    fixed_fixture = tuple(fixture)

    observed = _observed(fixed_fixture)
    expected = _expected(fixed_fixture)

    assert observed[:6] == expected[:6]
    np.testing.assert_array_equal(observed[6], expected[6])
    assert observed[:6] == (
        0,
        0,
        91,
        decision.SELECTED_CORRECTION,
        1,
        2,
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (0, lambda value: value.astype(np.int64), "draft_tokens must be uint32"),
        (1, lambda value: value[:, :-1], r"draft_ids must have shape \[3, 20\]"),
        (2, lambda value: value.astype(np.float64), "draft_probs must be float32"),
        (3, lambda value: value[:3], r"target_ids must have shape \[4, 20\]"),
        (4, lambda value: value.astype(np.float64), "target_probs must be float32"),
        (5, lambda value: value[:3], r"uniforms must have shape \[4\]"),
        (6, lambda value: value.astype(np.int64), "stop_ids must be uint32"),
    ],
)
def test_reference_validates_fixed_abi(
    field: int,
    replacement,
    message: str,
) -> None:
    fixture = list(_fixture())
    fixture[field] = replacement(fixture[field])

    with pytest.raises(ValueError, match=message):
        decision.reference_pr391_float32_verifier_decision(
            *fixture, stop_count=0, bonus_allowed=True
        )


def test_reference_validates_runtime_count_and_probability_rows() -> None:
    fixture = list(_fixture())
    with pytest.raises(ValueError, match="stop_count must be in"):
        decision.reference_pr391_float32_verifier_decision(
            *fixture, stop_count=3, bonus_allowed=True
        )

    fixture[2] = fixture[2].copy()
    fixture[2][0, 0] = np.float32(-0.1)
    with pytest.raises(ValueError, match="probabilities must be finite and non-negative"):
        decision.reference_pr391_float32_verifier_decision(
            *fixture, stop_count=0, bonus_allowed=True
        )


class _FakeKernel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> tuple[str, ...]:
        self.calls.append(kwargs)
        return tuple(f"output-{index}" for index in range(7))


def test_binder_constructs_fixed_kernel_and_hot_call_only_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: dict[str, object] = {}
    fake_kernel = _FakeKernel()

    def metal_kernel(**kwargs: object) -> _FakeKernel:
        built.update(kwargs)
        return fake_kernel

    fake_core = types.ModuleType("mlx.core")
    fake_core.fast = types.SimpleNamespace(metal_kernel=metal_kernel)
    fake_core.uint32 = "uint32"
    fake_core.int32 = "int32"
    fake_core.float32 = "float32"
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    apply = decision.bind_pr391_float32_verifier_decision(depth=3, top_k=20)
    fixture = _fixture()
    stop_count = object()
    bonus_allowed = object()

    assert apply(*fixture, stop_count, bonus_allowed) == tuple(
        f"output-{index}" for index in range(7)
    )
    assert built["input_names"] == [
        "draft_tokens",
        "draft_ids",
        "draft_probs",
        "target_ids",
        "target_probs",
        "uniforms",
        "stop_ids",
        "stop_count",
        "bonus_allowed",
    ]
    assert built["output_names"] == [
        "accepted_count",
        "first_reject",
        "selected_token",
        "selected_kind",
        "selected_present",
        "draws_used",
        "accept_probs",
    ]
    assert fake_kernel.calls == [
        {
            "inputs": [*fixture, stop_count, bonus_allowed],
            "grid": (1, 1, 1),
            "threadgroup": (1, 1, 1),
            "output_shapes": [(1,), (1,), (1,), (1,), (1,), (1,), (3,)],
            "output_dtypes": [
                "uint32",
                "int32",
                "uint32",
                "uint32",
                "uint32",
                "uint32",
                "float32",
            ],
        }
    ]


@pytest.mark.parametrize(
    ("depth", "top_k", "message"),
    [(2, 20, "depth=3"), (4, 20, "depth=3"), (3, 19, "top_k=20")],
)
def test_binder_fails_closed_before_importing_mlx(
    depth: int, top_k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decision.bind_pr391_float32_verifier_decision(depth=depth, top_k=top_k)


def test_metal_source_exposes_fixed_abi_without_rng_or_hard_coded_tokens() -> None:
    source = decision.METAL_SOURCE

    assert "constexpr uint D = 3" in source
    assert "constexpr uint K = 20" in source
    assert "stop_ids[stop_index]" in source
    assert "uniforms[depth]" in source
    assert "uniforms[depth + 1u]" in source
    assert "target_ids[3u * K" in source
    assert "bonus_allowed[0]" in source
    assert "mx.random" not in source
    assert "getenv" not in source
    assert source.count("{") == source.count("}")
