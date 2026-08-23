from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mtplx import gdn_capture


def _gdn() -> SimpleNamespace:
    return SimpleNamespace(
        A_log=mx.array([-1.5, -0.5, 0.25], dtype=mx.float32),
        dt_bias=mx.array([0.1, -0.2, 0.3], dtype=mx.float32),
    )


def test_row18_memoized_decay_gate_is_exact_at_fixed_d3_width() -> None:
    from mlx_lm.models.gated_delta import compute_g

    gdn = _gdn()
    a = mx.array(
        [[[0.2, -0.1, 0.4], [0.7, -0.6, 0.5], [-0.2, 0.9, -0.8], [0.0, 0.1, 0.2]]],
        dtype=mx.bfloat16,
    )
    expected = compute_g(gdn.A_log, a, gdn.dt_bias)
    gdn._mtplx_qwen38_neg_exp_a_log = -mx.exp(gdn.A_log.astype(mx.float32))
    actual = gdn_capture._qwen38_compute_g(gdn, a)
    mx.eval(expected, actual)

    assert mx.array_equal(expected, actual).item()


def test_row18_configuration_materializes_and_toggles_target_layers() -> None:
    gdns = [_gdn(), _gdn()]
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(linear_attn=gdn) for gdn in gdns]
            )
        )
    )

    active = gdn_capture.configure_qwen38_row18_gdn_decay_memo(model, active=True)
    inactive = gdn_capture.configure_qwen38_row18_gdn_decay_memo(model, active=False)

    assert active == {"configured_modules": 2, "active_modules": 2}
    assert inactive == {"configured_modules": 2, "active_modules": 0}
    assert all(gdn._mtplx_qwen38_neg_exp_a_log is None for gdn in gdns)
