"""Pure-numpy expert-id patterns for the Flash-Next M=4 MoE microbenchmarks.

Deliberately free of any MLX import so the invariants can be unit tested on a
box that is not holding the GPU lock.  The physical verifier runs M=4 rows with
top_k=10 over 512 routed experts, so one routing decision is a [4, 10] integer
array with 10 DISTINCT ids per row and between 10 and 40 distinct ids overall.
The gap between 40 (what ``gather_qmm`` streams today) and that distinct count
is the dedup headroom the benchmark prices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROWS = 4
TOP_K = 10
NUM_EXPERTS = 512
SLOTS = ROWS * TOP_K
UNIQUE_CHOICES = (40, 34, 28, 22, 16)


def unique_count(ids: np.ndarray) -> int:
    """Number of distinct experts across all ``ROWS * TOP_K`` slots."""

    return int(np.unique(np.asarray(ids)).size)


def validate_ids(
    ids: np.ndarray,
    *,
    rows: int = ROWS,
    top_k: int = TOP_K,
    num_experts: int = NUM_EXPERTS,
) -> np.ndarray:
    """Raise unless ``ids`` is a legal top-k routing decision."""

    ids = np.asarray(ids)
    if ids.shape != (rows, top_k):
        raise ValueError(f"expert ids must be {(rows, top_k)}, got {ids.shape}")
    if ids.min() < 0 or ids.max() >= num_experts:
        raise ValueError(f"expert ids must lie in [0, {num_experts})")
    for r in range(rows):
        if np.unique(ids[r]).size != top_k:
            raise ValueError(f"row {r} repeats an expert; top-k rows are distinct")
    return ids


def make_expert_ids(
    unique: int,
    *,
    rows: int = ROWS,
    top_k: int = TOP_K,
    num_experts: int = NUM_EXPERTS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A [rows, top_k] routing decision with exactly ``unique`` distinct ids.

    Coverage first: the pool is laid down round-robin across rows so every
    pooled id is used at least once and no row can receive it twice.  The
    leftover slots are then filled from the pool with ids the row does not
    already hold, which keeps rows distinct without adding new experts.
    """

    slots = rows * top_k
    if not top_k <= unique <= slots:
        raise ValueError(f"unique must be in [{top_k}, {slots}], got {unique}")
    if unique > num_experts:
        raise ValueError("unique cannot exceed the expert count")
    rng = np.random.default_rng() if rng is None else rng

    pool = rng.choice(num_experts, size=unique, replace=False)
    row_sets: list[list[int]] = [[] for _ in range(rows)]
    for i, expert in enumerate(pool):
        row_sets[i % rows].append(int(expert))

    pool_list = [int(e) for e in pool]
    for r in range(rows):
        held = set(row_sets[r])
        available = [e for e in pool_list if e not in held]
        needed = top_k - len(row_sets[r])
        if needed > len(available):  # pragma: no cover - guarded by `unique >= top_k`
            raise ValueError("cannot fill rows without repeating within a row")
        if needed:
            picked = rng.choice(len(available), size=needed, replace=False)
            row_sets[r].extend(available[int(p)] for p in picked)
        rng.shuffle(row_sets[r])

    ids = np.array(row_sets, dtype=np.int32)
    return validate_ids(ids, rows=rows, top_k=top_k, num_experts=num_experts)


def make_layer_id_sets(
    unique: int,
    layers: int,
    *,
    seed: int = 0,
    rows: int = ROWS,
    top_k: int = TOP_K,
    num_experts: int = NUM_EXPERTS,
) -> list[np.ndarray]:
    """One independent routing decision per layer at a fixed unique count."""

    rng = np.random.default_rng(seed)
    return [
        make_expert_ids(
            unique, rows=rows, top_k=top_k, num_experts=num_experts, rng=rng
        )
        for _ in range(layers)
    ]


def load_census_id_sets(
    path: str | Path,
    layers: int,
    *,
    seed: int = 0,
    rows: int = ROWS,
    top_k: int = TOP_K,
    num_experts: int = NUM_EXPERTS,
) -> list[np.ndarray]:
    """Replay a real routing census: a JSON list of [rows, top_k] id arrays.

    Sampling with replacement from the census reproduces its unique-count
    distribution rather than one hand-picked point on it.
    """

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("census must be a non-empty JSON list of [4, 10] arrays")
    census = [
        validate_ids(
            np.asarray(entry, dtype=np.int32),
            rows=rows,
            top_k=top_k,
            num_experts=num_experts,
        )
        for entry in payload
    ]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(census), size=layers)
    return [census[int(p)] for p in picks]


# ---------------------------------------------------------------------------
# expert-major plan
# ---------------------------------------------------------------------------
#
# The expert-major routed-GLU kernel (mtplx/kernels/qwen4_m4_expert_major_glu.py)
# does NOT take a precomputed plan tensor: it recomputes the plan below inside
# every threadgroup from the same ``[ROWS, TOP_K]`` id array the row-major kernel
# already binds, which costs two 40-iteration scalar scans over 160 L1-resident
# bytes and zero extra dispatches.  This function is the executable statement of
# what that in-kernel scan computes, so the contract can be argued about (and
# unit tested) with no MLX and no Metal.
#
# Plan ABI, for a flattened lane index ``i = row * TOP_K + slot``:
#
#   entry i is a LEADER  iff  i is the (n * MEMBERS_PER_ENTRY)-th occurrence of
#                             ``ids[i]`` in ascending lane order, for some n >= 0
#   member[i, m]         =    the m-th lane (ascending) in that occurrence run,
#                             or -1 when the run is shorter, or -1 for every m
#                             when i is not a leader
#   expert[i]            =    ids[i], always a legal expert id (so a non-leader
#                             threadgroup's address arithmetic can never go out
#                             of range before it masks itself off)
#
# Every lane appears in exactly one leader's member list, so the union of the
# leaders' work is exactly the 40 (row, expert) products the row-major kernel
# computes -- while the weight tile of a duplicated expert is streamed once per
# leader instead of once per lane.  The ``n * MEMBERS_PER_ENTRY`` rule (rather
# than "first occurrence only") keeps that true even if a run were ever longer
# than MEMBERS_PER_ENTRY, which top-k rows being internally distinct already
# forbids -- it removes the silent-corruption branch rather than relying on the
# invariant.

MEMBERS_PER_ENTRY = ROWS


def expert_major_plan(
    ids: np.ndarray,
    *,
    rows: int = ROWS,
    top_k: int = TOP_K,
    members: int = MEMBERS_PER_ENTRY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(expert[slots], member[slots, members])`` for one routing decision."""

    flat = np.asarray(ids, dtype=np.int64).reshape(-1)
    slots = rows * top_k
    if flat.size != slots:
        raise ValueError(f"expert ids must hold {slots} slots, got {flat.size}")

    expert = flat.astype(np.int32)
    member = np.full((slots, members), -1, dtype=np.int32)
    for i in range(slots):
        e = int(flat[i])
        position = int(np.count_nonzero(flat[:i] == e))
        if position % members:
            continue
        run = [int(j) for j in range(i, slots) if int(flat[j]) == e][:members]
        member[i, : len(run)] = run
    return expert, member


def validate_expert_major_plan(
    ids: np.ndarray,
    expert: np.ndarray,
    member: np.ndarray,
    *,
    rows: int = ROWS,
    top_k: int = TOP_K,
) -> None:
    """Raise unless the plan covers every lane exactly once with the right expert."""

    flat = np.asarray(ids, dtype=np.int64).reshape(-1)
    slots = rows * top_k
    expert = np.asarray(expert).reshape(-1)
    member = np.asarray(member).reshape(slots, -1)

    if not np.array_equal(expert, flat.astype(expert.dtype)):
        raise ValueError("plan expert column must mirror the flattened ids")

    covered: list[int] = []
    for entry in range(slots):
        lanes = [int(v) for v in member[entry] if int(v) >= 0]
        if lanes != sorted(lanes):
            raise ValueError(f"entry {entry} lists members out of lane order")
        if len(set(lanes)) != len(lanes):
            raise ValueError(f"entry {entry} repeats a lane")
        for lane in lanes:
            if not 0 <= lane < slots:
                raise ValueError(f"entry {entry} names lane {lane} out of range")
            if int(flat[lane]) != int(flat[entry]):
                raise ValueError(
                    f"entry {entry} claims lane {lane} with a different expert"
                )
        if len({lane // top_k for lane in lanes}) != len(lanes):
            raise ValueError(f"entry {entry} claims two lanes from one row")
        # A trailing -1 may only follow real members, never precede one.
        seen_hole = False
        for value in member[entry]:
            if int(value) < 0:
                seen_hole = True
            elif seen_hole:
                raise ValueError(f"entry {entry} has a hole before a member")
        covered.extend(lanes)

    if sorted(covered) != list(range(slots)):
        raise ValueError("plan does not cover every lane exactly once")

    active = int(sum(1 for entry in range(slots) if int(member[entry, 0]) >= 0))
    expected = int(np.unique(flat).size)
    if active != expected:
        raise ValueError(
            f"plan activated {active} entries for {expected} distinct experts"
        )


def expert_major_active_entries(ids: np.ndarray) -> int:
    """Leader count for one routing decision (== distinct experts, top-k rows)."""

    _, member = expert_major_plan(ids)
    return int(np.count_nonzero(member[:, 0] >= 0))
