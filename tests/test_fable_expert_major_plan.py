"""Plan invariants for the expert-major M4 routed GLU kernel.

The kernel recomputes its plan inside every threadgroup rather than taking one
as a tensor, so what is testable off-GPU is the *specification* of that scan --
``expert_id_patterns.expert_major_plan`` -- and the property that decides
whether the kernel is correct: every one of the 40 (row, expert) lanes is owned
by exactly one leader entry, and that entry carries the lane's expert.

Nothing here imports MLX or Metal.  These run on any box, lock or no lock.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

FABLE = Path(__file__).resolve().parents[1] / "scripts" / "fable"
if str(FABLE) not in sys.path:
    sys.path.insert(0, str(FABLE))

import expert_id_patterns as eip  # noqa: E402

SLOTS = eip.SLOTS
MEMBERS = eip.MEMBERS_PER_ENTRY


def _plan(ids):
    expert, member = eip.expert_major_plan(ids)
    return expert, member


def _leaders(member):
    return [entry for entry in range(SLOTS) if int(member[entry, 0]) >= 0]


def test_members_matches_rows():
    # The kernel's `member[MEMBERS]` array is sized by this; a top-k row holds
    # distinct experts, so a run cannot exceed one lane per row.
    assert MEMBERS == eip.ROWS == 4


def test_plan_shape_and_dtype():
    ids = eip.make_expert_ids(28, rng=np.random.default_rng(7))
    expert, member = _plan(ids)
    assert expert.shape == (SLOTS,)
    assert member.shape == (SLOTS, MEMBERS)
    assert expert.dtype == np.int32
    assert member.dtype == np.int32


def test_expert_column_mirrors_flat_ids():
    # Non-leader entries still carry a legal expert id so the kernel's address
    # arithmetic is in range before it masks itself off.
    ids = eip.make_expert_ids(16, rng=np.random.default_rng(3))
    expert, _ = _plan(ids)
    assert np.array_equal(expert, ids.reshape(-1).astype(np.int32))
    assert expert.min() >= 0
    assert expert.max() < eip.NUM_EXPERTS


@pytest.mark.parametrize("unique", eip.UNIQUE_CHOICES)
def test_random_decisions_cover_every_lane_once(unique):
    rng = np.random.default_rng(unique)
    for _ in range(64):
        ids = eip.make_expert_ids(unique, rng=rng)
        expert, member = _plan(ids)
        eip.validate_expert_major_plan(ids, expert, member)
        assert len(_leaders(member)) == unique
        assert eip.expert_major_active_entries(ids) == unique


def test_leader_count_equals_distinct_experts_on_census_shaped_duplicates():
    rng = np.random.default_rng(11)
    for _ in range(200):
        unique = int(rng.integers(eip.TOP_K, SLOTS + 1))
        ids = eip.make_expert_ids(unique, rng=rng)
        _, member = _plan(ids)
        assert len(_leaders(member)) == eip.unique_count(ids)


def test_all_distinct_gives_forty_single_member_leaders():
    ids = np.arange(SLOTS, dtype=np.int32).reshape(eip.ROWS, eip.TOP_K)
    expert, member = _plan(ids)
    eip.validate_expert_major_plan(ids, expert, member)
    assert len(_leaders(member)) == SLOTS
    assert np.array_equal(member[:, 0], np.arange(SLOTS))
    assert (member[:, 1:] == -1).all()


def test_all_same_gives_ten_four_member_leaders():
    row = np.arange(eip.TOP_K, dtype=np.int32)
    ids = np.stack([row] * eip.ROWS)
    expert, member = _plan(ids)
    eip.validate_expert_major_plan(ids, expert, member)
    leaders = _leaders(member)
    assert leaders == list(range(eip.TOP_K))
    for entry in leaders:
        assert list(member[entry]) == [entry + eip.TOP_K * r for r in range(4)]
    # Every non-leader entry is fully masked: the kernel returns before it
    # touches a single weight byte.
    for entry in range(eip.TOP_K, SLOTS):
        assert (member[entry] == -1).all()


def test_one_expert_in_all_four_rows():
    ids = (np.arange(SLOTS, dtype=np.int32) + 1).reshape(eip.ROWS, eip.TOP_K)
    ids[:, 0] = 0
    expert, member = _plan(ids)
    eip.validate_expert_major_plan(ids, expert, member)
    assert eip.unique_count(ids) == SLOTS - 3
    assert len(_leaders(member)) == SLOTS - 3
    assert list(member[0]) == [0, 10, 20, 30]


def test_members_are_ascending_and_row_disjoint():
    rng = np.random.default_rng(5)
    for _ in range(64):
        ids = eip.make_expert_ids(int(rng.integers(10, 41)), rng=rng)
        _, member = _plan(ids)
        for entry in range(SLOTS):
            lanes = [int(v) for v in member[entry] if int(v) >= 0]
            assert lanes == sorted(lanes)
            assert len({lane // eip.TOP_K for lane in lanes}) == len(lanes)


def test_padding_holes_only_trail():
    rng = np.random.default_rng(13)
    for _ in range(64):
        ids = eip.make_expert_ids(int(rng.integers(10, 41)), rng=rng)
        _, member = _plan(ids)
        for entry in range(SLOTS):
            row = [int(v) for v in member[entry]]
            first_hole = next((i for i, v in enumerate(row) if v < 0), MEMBERS)
            assert all(v < 0 for v in row[first_hole:])


def test_runs_longer_than_members_still_cover_every_lane():
    # Not reachable through a legal top-k decision, but the kernel's leader rule
    # is "occurrence index is a multiple of MEMBERS" rather than "first
    # occurrence" precisely so this degenerate case cannot silently drop work.
    ids = np.zeros((eip.ROWS, eip.TOP_K), dtype=np.int32)
    _, member = _plan(ids)
    covered = sorted(int(v) for r in member for v in r if int(v) >= 0)
    assert covered == list(range(SLOTS))
    assert len(_leaders(member)) == SLOTS // MEMBERS

    mixed = np.zeros((eip.ROWS, eip.TOP_K), dtype=np.int32)
    mixed[:, 5:] = np.arange(20, dtype=np.int32).reshape(eip.ROWS, 5)
    _, member = _plan(mixed)
    covered = sorted(int(v) for r in member for v in r if int(v) >= 0)
    assert covered == list(range(SLOTS))


def test_validator_rejects_a_dropped_lane():
    ids = eip.make_expert_ids(28, rng=np.random.default_rng(1))
    expert, member = _plan(ids)
    entry = _leaders(member)[0]
    member[entry, 0] = -1
    with pytest.raises(ValueError):
        eip.validate_expert_major_plan(ids, expert, member)


def test_validator_rejects_a_lane_claimed_twice():
    ids = eip.make_expert_ids(40, rng=np.random.default_rng(2))
    expert, member = _plan(ids)
    member[1, 0] = int(member[0, 0])
    with pytest.raises(ValueError):
        eip.validate_expert_major_plan(ids, expert, member)


def test_validator_rejects_a_wrong_expert_claim():
    ids = eip.make_expert_ids(40, rng=np.random.default_rng(4))
    expert, member = _plan(ids)
    member[0, 1] = 7  # lane 7 holds a different expert in an all-distinct set
    with pytest.raises(ValueError):
        eip.validate_expert_major_plan(ids, expert, member)


def test_plan_rejects_wrong_slot_count():
    with pytest.raises(ValueError):
        eip.expert_major_plan(np.zeros((4, 9), dtype=np.int32))
