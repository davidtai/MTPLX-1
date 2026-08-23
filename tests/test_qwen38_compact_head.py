from __future__ import annotations

import mlx.core as mx

from mtplx.qwen38_compact_head import (
    QWEN38_COMPACT_CONTROL_END,
    QWEN38_COMPACT_CONTROL_START,
    QWEN38_COMPACT_PADDED_ROWS,
    QWEN38_COMPACT_PREFIX_ROWS,
    QWEN38_COMPACT_REAL_ROWS,
    QWEN38_CLUSTER_PROBE_FRACTION,
    QWEN38_CLUSTER_PROBES,
    compact_token_ids_to_full,
)


def test_compact_vocabulary_contract_and_mapping() -> None:
    assert QWEN38_COMPACT_REAL_ROWS == 98_330
    assert QWEN38_COMPACT_PADDED_ROWS == 98_336
    assert QWEN38_COMPACT_CONTROL_END - QWEN38_COMPACT_CONTROL_START == 26

    compact = mx.array(
        [0, QWEN38_COMPACT_PREFIX_ROWS - 1, QWEN38_COMPACT_PREFIX_ROWS, 98_329]
    )
    full = compact_token_ids_to_full(compact)
    mx.eval(full)

    assert full.tolist() == [0, 98_303, 248_044, 248_069]


def test_final_cluster_probe_fraction_is_fifteen_percent() -> None:
    assert QWEN38_CLUSTER_PROBE_FRACTION == 0.15
    assert QWEN38_CLUSTER_PROBES == 1_844

