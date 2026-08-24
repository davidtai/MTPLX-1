from __future__ import annotations

import pytest

from scripts import qwen38_challenge_dflash_gate as gate


def test_dflash_survivors_are_unique_chronological_and_dependency_closed() -> None:
    assert gate._parse_dflash_survivors("") == ()
    assert gate._parse_dflash_survivors("18,21,24,26,48") == (
        18,
        21,
        24,
        26,
        48,
    )

    for invalid in ("24", "21,18", "18,18", "17"):
        with pytest.raises(ValueError):
            gate._parse_dflash_survivors(invalid)


def test_dflash_flat_counter_delta_tracks_only_current_arm() -> None:
    assert gate._flat_counter_delta(
        {"memo": 11, "qk": 3},
        {"memo": 18, "qk": 3, "boundary": 4},
    ) == {"boundary": 4, "memo": 7, "qk": 0}
