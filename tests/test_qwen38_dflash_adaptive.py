from types import SimpleNamespace

import pytest

from mtplx.qwen38_dflash_adaptive import (
    configure_qwen38_dflash_adaptive_policy,
)


def test_row11_position_ema_maps_to_dflash_physical_blocks_one_through_eight() -> None:
    target = SimpleNamespace()

    report = configure_qwen38_dflash_adaptive_policy(
        target,
        active=True,
        proposal_rows=(11,),
    )
    policy = target._dflash_adaptive_block_policy_factory(
        full_block_tokens=8,
        verify_len_cap=8,
        prompt_len=16_384,
    )

    assert report == {
        "active": True,
        "proposal_rows": [11],
        "min_block_tokens": 1,
        "max_block_tokens": 8,
        "base_draft_cap": 4,
        "deep_draft_cap": 4,
        "head_step_cost_ratio": 0.20,
        "streak_gate": None,
    }
    assert 1 <= policy.block_limit() <= 5
    offered = policy.block_limit()
    policy.record(
        block_len=offered,
        acceptance_len=offered - 1,
        cycle_cost_ns=1_000_000,
    )
    metrics = policy.metrics()
    assert metrics["kind"] == "qwen38_position_ema"
    assert metrics["proposal_rows"] == [11]
    assert metrics["cycles"] == 1
    assert metrics["cycles_by_block"] == {str(offered): 1}

    inactive = configure_qwen38_dflash_adaptive_policy(
        target,
        active=False,
        proposal_rows=(),
    )
    assert inactive == {"active": False, "proposal_rows": []}
    assert not hasattr(target, "_dflash_adaptive_block_policy_factory")


def test_dflash_adaptive_rows_are_dependency_closed() -> None:
    with pytest.raises(ValueError, match="requires row 11"):
        configure_qwen38_dflash_adaptive_policy(
            SimpleNamespace(),
            active=True,
            proposal_rows=(15,),
        )
