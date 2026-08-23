from __future__ import annotations

from mtplx.qwen38_qmv import qwen38_qmv_active_input_groups


def test_active_input_groups_match_rows_78_and_80_width_table() -> None:
    assert {
        width: qwen38_qmv_active_input_groups(width)
        for width in range(2, 10)
    } == {2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}
