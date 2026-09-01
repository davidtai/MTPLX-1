#!/usr/bin/env python3
"""Report the M4 routed-expert dedup headroom from an expert-id census.

Input is the ``.npz`` written by ``mtplx.fable_expert_census`` (see that
module's docstring for how to produce one -- it needs both
``MTPLX_FABLE_EXPERT_CENSUS=<path>`` and ``MLX_DISABLE_COMPILE=1``):

* ``ids``       -- ``int16 [cycles, layers, 4, 10]``
* ``layer_ids`` -- ``int16 [layers]``

The number that prices the lever is ``U``: how many DISTINCT experts the
four physical rows name in one layer-cycle.  ``U = 40`` means every one of
the ``4 x 10`` routed weight slices is a different expert and dedup buys
nothing; ``U = 20`` means half the routed bytes are read twice today.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def distinct_counts(flat: np.ndarray) -> np.ndarray:
    """Count distinct values along the last axis of an integer array."""

    ordered = np.sort(flat, axis=-1)
    changes = ordered[..., 1:] != ordered[..., :-1]
    return 1 + changes.sum(axis=-1)


def union_counts(ids: np.ndarray) -> np.ndarray:
    """``U`` per layer-cycle: distinct experts across all rows.

    Shape ``[cycles, layers, rows, top_k]`` -> ``[cycles, layers]``.
    """

    cycles, layers, rows, top_k = ids.shape
    return distinct_counts(ids.reshape(cycles, layers, rows * top_k))


def consecutive_row_jaccard(ids: np.ndarray) -> np.ndarray:
    """Jaccard of each adjacent row pair: ``[cycles, layers, rows-1]``.

    A row's ``top_k`` ids are distinct, so for two rows of ``k`` ids the
    intersection is ``2k - distinct(concat)`` and the union is ``2k -
    intersection``.
    """

    top_k = ids.shape[-1]
    pairs = np.concatenate([ids[:, :, :-1, :], ids[:, :, 1:, :]], axis=-1)
    distinct = distinct_counts(pairs).astype(np.float64)
    intersection = 2 * top_k - distinct
    union = 2 * top_k - intersection
    return intersection / union


def _fmt_stats(values: np.ndarray) -> str:
    return (
        f"min={values.min():.0f}  median={np.median(values):.1f}  "
        f"mean={values.mean():.3f}  p90={np.percentile(values, 90):.1f}  "
        f"max={values.max():.0f}"
    )


def report(path: str, sample_out: str | None, sample_count: int) -> None:
    with np.load(path) as data:
        ids = np.asarray(data["ids"], dtype=np.int32)
        layer_ids = np.asarray(data["layer_ids"], dtype=np.int32)
    if ids.ndim != 4:
        raise SystemExit(f"{path}: expected ids [cycles, layers, rows, top_k]")
    cycles, layers, rows, top_k = ids.shape
    slots = rows * top_k

    u = union_counts(ids)
    flat_u = u.reshape(-1)
    jaccard = consecutive_row_jaccard(ids)

    print(f"source            {path}")
    print(
        f"shape             cycles={cycles} layers={layers} "
        f"rows={rows} top_k={top_k}  ({slots} routed slots per layer-cycle)"
    )
    print(f"layer-cycles      {flat_u.size}")
    print()
    print("distinct experts per layer-cycle (U)")
    print(f"  {_fmt_stats(flat_u)}")
    print()
    print("  histogram of U")
    values, counts = np.unique(flat_u, return_counts=True)
    total = float(flat_u.size)
    peak = int(counts.max())
    for value, count in zip(values.tolist(), counts.tolist()):
        bar = "#" * max(1, round(40 * count / peak))
        print(f"    U={value:3d}  {count:8d}  {100 * count / total:6.2f}%  {bar}")
    print()

    per_layer = u.mean(axis=0)
    print("mean U per layer index")
    for start in range(0, layers, 6):
        chunk = range(start, min(start + 6, layers))
        print(
            "    "
            + "   ".join(
                f"L{int(layer_ids[i]):02d} {per_layer[i]:6.3f}" for i in chunk
            )
        )
    order = np.argsort(per_layer)
    print(
        f"    lowest  L{int(layer_ids[order[0]]):02d} {per_layer[order[0]]:.3f}"
        f"   highest L{int(layer_ids[order[-1]]):02d} "
        f"{per_layer[order[-1]]:.3f}"
    )
    print()

    print("pairwise row overlap (Jaccard, adjacent rows)")
    for pair in range(rows - 1):
        column = jaccard[:, :, pair].reshape(-1)
        print(
            f"    rows {pair}-{pair + 1}   mean={column.mean():.4f}  "
            f"median={np.median(column):.4f}  max={column.max():.4f}"
        )
    print(f"    all pairs   mean={jaccard.mean():.4f}")
    print()

    saving = 1.0 - float(flat_u.mean()) / slots
    print("implied routed-weight-bytes saving from perfect dedup")
    print(
        f"    1 - mean(U)/{slots} = 1 - {flat_u.mean():.3f}/{slots} "
        f"= {saving:.4f}  ({100 * saving:.2f}%)"
    )
    best = 1.0 - float(flat_u.min()) / slots
    print(f"    best layer-cycle observed: {100 * best:.2f}%")

    if sample_out:
        take = min(sample_count, cycles * layers)
        sample = ids.reshape(cycles * layers, rows, top_k)[:take]
        with open(sample_out, "w", encoding="utf-8") as handle:
            json.dump(sample.astype(int).tolist(), handle)
        print()
        print(
            f"sample            wrote {take} [{rows},{top_k}] arrays "
            f"(cycle-major order) to {sample_out}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", help="census .npz written by ExpertCensus.flush")
    parser.add_argument(
        "--sample-out",
        default=None,
        help="write the first N layer-cycles as a JSON list of [4,10] arrays "
        "(consumed by scripts/fable/micro_moe_dedup.py --from-census)",
    )
    parser.add_argument("--sample-count", type=int, default=200)
    args = parser.parse_args(argv)
    report(args.npz, args.sample_out, args.sample_count)


if __name__ == "__main__":
    main()
