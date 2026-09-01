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
