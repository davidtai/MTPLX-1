#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price the FR-Spec draft K20 support: scattered [1,248320] vs compact [1,65536].

The question
------------
With ``MTPLX_FRSPEC_DRAFT=1`` every draft step runs the pruned 65,536-row q8
head, SCATTERS its logits into a 248,320-wide row padded with ``-1e30``
(``mtplx/frspec_draft.py`` ``_FullVocabDraftHead.__call__``), and then builds
its K20 support over that padded row through
``fast_sampling._device_serial_support_arrays``: one ``argpartition`` to an
80-candidate superset, one full-vocabulary ``logsumexp``, one sync, and a host
``np.lexsort`` over 80.  73.6% of the lanes it reads are the sentinel.

``MTPLX_FABLE_DRAFT_K20_PRESCATTER`` (``mtplx/fable_draft_k20_prescatter.py``)
does the same selection on the compact row and maps the selected LOCAL rows to
real token ids through the ranked table, so the scatter is never evaluated and
both device passes shrink 3.79x.  This bench prices that and re-measures the
one exactness claim the CPU tests cannot settle on the Metal stream.

Variants (all on the same synthetic compact row + its scatter)
--------------------------------------------------------------
``stock_scatter_serial``
    What the lane pays per draft step: build the scatter, then
    ``_device_serial_support_arrays`` over 248,320.  Host-terminated (the
    builder syncs and calls ``np.asarray`` itself), because that is what the
    lane pays.
``stock_serial_only``
    The same builder on a row that is ALREADY materialized -- the scatter is
    hoisted out of the timed region.  ``stock_scatter_serial`` minus this is
    the scatter's own price.
``scatter_only``
    ``mx.full`` + ``put_along_axis`` alone, device-terminated.  The other half
    of the same split, measured directly rather than by subtraction.
``prescatter_serial``
    ``fable_draft_k20_prescatter.prescatter_serial_support_arrays`` on the
    compact row.  Host-terminated, same one sync, same host tail over 80.
``prescatter_read``
    ``prescatter_serial`` plus ``_serial_row_distribution`` and the one
    ``rng.choice`` draw -- the whole ``read_draft`` the decode loop calls.
``stock_read``
    ``sparse_distribution_from_mlx_logits`` on the scattered row plus the same
    draw: ``prescatter_read``'s apples-to-apples partner.
``read_floor_full`` / ``read_floor_compact``
    ``mx.sum`` over each row: the bandwidth floor for touching it once.

Lanes
-----
``--lane queued`` (default) issues ``--reps`` calls and evaluates ONCE, then
divides; ``--lane eager`` evaluates every call.  Per the
``queued-vs-eager-metal-microbench`` note an eager lane charges every call a
host sync and can INVERT a verdict for microsecond kernels, so promotions are
decided on the queued lane.  The host-terminated variants (every ``serial``
and ``read`` variant, which sync and ``np.asarray`` by construction) are only
ever timed in the eager sense -- and that IS the production shape: the draft
site's ``np.asarray`` follows immediately.

Parity
------
Nothing here is asserted; everything is counted.  Per shape the report prints
* ``support_ids_differing`` / ``support_prob_bits_differing`` -- the shipped
  builder on the scattered row vs the pre-scatter builder on the compact row,
  compared as RAW BITS,
* ``logsumexp_ulp`` -- the signed float32 ULP distance between
  ``logsumexp(scaled_full)`` and ``logsumexp(scaled_compact)``.  This is the
  ONE claim ``tests/test_fable_draft_k20_prescatter.py`` can only pin on the
  CPU stream: the two reductions are different WIDTHS, float32 addition is not
  associative, and the sentinel terms are exactly ``+0.0`` but need not be
  partitioned across threads the same way.  A nonzero count here is not a bug
  -- it bounds the residual, which can only move a knife-edge top-p cut, never
  the support (see the module docstring),
* ``draw_rows_differing`` -- the sampled draft token from an identically
  seeded ``rng``, over ``--parity-rows`` independent rows.

Decision rule
-------------
The draft chain is 3 sync-terminated steps per window.  Report M row K-D1
expects ``stock_read - prescatter_read`` in 0.3-0.6 ms, i.e. 1.0-1.8 ms per
window.  Below ~0.15 ms the lever is not worth the gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

# MLX is imported lazily so ``--self-test`` and the shape table stay runnable
# on a box that is not holding the GPU lock.
mx = None


LOCK_PATH = "/tmp/mtplx-gpu-exclusive.lock"
BANNER = (
    "[micro_draft_k20] GPU WINDOW REQUIRED -- run under "
    f"{LOCK_PATH} via bench/laguna/run_guarded.py"
)

FULL_VOCAB = 248_320        # Qwen3.8-Flash-Next full vocabulary
FRSPEC_ROWS = 65_536        # the ranked draft head's compact domain
TOP_K = 20
SENTINEL = -1.0e30


def _require_mlx() -> None:
    global mx
    import mlx.core

    mx = mlx.core


def _sampler(temperature: float, top_p: float):
    from mtplx.sampling import SamplerConfig

    return SamplerConfig(temperature=temperature, top_p=top_p, top_k=TOP_K)


def ranked_ids(rows: int, vocab_rows: int, seed: int) -> np.ndarray:
    """A strictly ascending ranked table, like the shipped artifact."""

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(vocab_rows, size=rows, replace=False)).astype(np.int64)


def make_compact(rows: int, seed: int) -> np.ndarray:
    """A draft logit row shaped like a real one: a heavy head, a long tail."""

    rng = np.random.default_rng(seed)
    base = (rng.standard_normal(rows) * 2.0).astype(np.float32)
    peaks = rng.choice(rows, size=64, replace=False)
    base[peaks] += rng.uniform(6.0, 14.0, size=64).astype(np.float32)
    return base


def scatter(compact, ids_dev, vocab_rows):
    """Exactly ``_FullVocabDraftHead.__call__``'s scatter, on device."""

    subset = compact.reshape(1, -1)
    output = mx.full((1, vocab_rows), SENTINEL, dtype=subset.dtype)
    index = mx.broadcast_to(ids_dev.reshape(1, -1), subset.shape)
    return mx.put_along_axis(output, index, subset, axis=-1).reshape(-1)


def build_plan(ids_np, rows, vocab_rows, sampler):
    from mtplx.fable_draft_k20_prescatter import DraftK20PrescatterPlan

    return DraftK20PrescatterPlan(
        head=None,
        ids_np=ids_np,
        rows=rows,
        vocab_rows=vocab_rows,
        top_k=TOP_K,
        temperature=float(sampler.temperature),
        top_p=float(sampler.top_p),
    )


def build_variants(sampler, plan, compact, ids_dev, dense):
    from mtplx.fable_draft_k20_prescatter import (
        prescatter_serial_support_arrays,
        prescatter_sparse_distribution,
    )
    from mtplx.fast_sampling import (
        _device_serial_support_arrays,
        sparse_distribution_from_mlx_logits,
    )
    from mtplx.sampling import sample_from_distribution

    vocab_rows = int(plan.vocab_rows)

    def stock_scatter_serial(_):
        row = scatter(compact, ids_dev, vocab_rows)
        return _device_serial_support_arrays(row.astype(mx.float32), sampler)

    def stock_serial_only(_):
        return _device_serial_support_arrays(dense.astype(mx.float32), sampler)

    def scatter_only(_):
        return (scatter(compact, ids_dev, vocab_rows),)

    def prescatter_serial(_):
        return prescatter_serial_support_arrays(
            plan, compact.astype(mx.float32), sampler
        )

    def stock_read(_):
        row = scatter(compact, ids_dev, vocab_rows)
        distribution = sparse_distribution_from_mlx_logits(
            row.astype(mx.float32), sampler
        )
        return (sample_from_distribution(distribution, np.random.default_rng(1)),)

    def prescatter_read(_):
        distribution = prescatter_sparse_distribution(
            plan, compact.astype(mx.float32), sampler
        )
        return (sample_from_distribution(distribution, np.random.default_rng(1)),)

    def read_floor_full(_):
        return (mx.sum(dense.astype(mx.float32), axis=-1),)

    def read_floor_compact(_):
        return (mx.sum(compact.astype(mx.float32), axis=-1),)

    return {
        "stock_scatter_serial": (stock_scatter_serial, True),
        "stock_serial_only": (stock_serial_only, True),
        "scatter_only": (scatter_only, False),
        "prescatter_serial": (prescatter_serial, True),
        "stock_read": (stock_read, True),
        "prescatter_read": (prescatter_read, True),
        "read_floor_full": (read_floor_full, False),
        "read_floor_compact": (read_floor_compact, False),
    }


def _leaves(out):
    if isinstance(out, tuple):
        return [leaf for leaf in out if hasattr(leaf, "dtype")]
    return [out] if hasattr(out, "dtype") else []


def time_variant(fn, *, reps, warmup, lane, host_terminated):
    for _ in range(warmup):
        out = fn(None)
        if not host_terminated:
            mx.eval(*_leaves(out))
    if host_terminated or lane == "eager":
        samples = []
        for _ in range(reps):
            start = time.perf_counter()
            out = fn(None)
            if not host_terminated:
                mx.eval(*_leaves(out))
            samples.append((time.perf_counter() - start) * 1e3)
        samples.sort()
        return {
            "median_ms": statistics.median(samples),
            "p10_ms": samples[max(0, int(0.10 * (len(samples) - 1)))],
            "p90_ms": samples[min(len(samples) - 1, int(0.90 * (len(samples) - 1)))],
            "lane": "eager",
        }
    outs = []
    start = time.perf_counter()
    for _ in range(reps):
        outs.extend(_leaves(fn(None)))
    mx.eval(*outs)
    total = (time.perf_counter() - start) * 1e3
    return {
        "median_ms": total / reps,
        "p10_ms": total / reps,
        "p90_ms": total / reps,
        "lane": "queued",
    }


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed float32 ULP distance, monotone-ordered integer view."""

    def key(x):
        bits = np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, np.int64(-2147483648) - bits, bits)

    return key(a) - key(b)


def parity(sampler, plan, ids_np, rows, vocab_rows, *, seed, parity_rows):
    """Differing-row counters: support bits, logsumexp ULP, sampled token."""

    from mtplx.fable_draft_k20_prescatter import (
        prescatter_serial_support_arrays,
        prescatter_sparse_distribution,
    )
    from mtplx.fast_sampling import (
        _device_serial_support_arrays,
        sparse_distribution_from_mlx_logits,
    )
    from mtplx.sampling import sample_from_distribution

    ids_dev = mx.array(ids_np, dtype=mx.int32)
    scale = 1.0 / float(sampler.temperature)
    ids_bad = 0
    prob_bad = 0
    draw_bad = 0
    first_bad = -1
    ulps: list[int] = []
    for index in range(parity_rows):
        host = make_compact(rows, seed + 7919 * index)
        compact = mx.array(host)
        dense = scatter(compact, ids_dev, vocab_rows)
        want_ids, want_probs, want_vocab = _device_serial_support_arrays(
            dense.astype(mx.float32), sampler
        )
        got_ids, got_probs, got_vocab = prescatter_serial_support_arrays(
            plan, compact.astype(mx.float32), sampler
        )
        assert want_vocab == got_vocab == vocab_rows
        row_bad = False
        if not np.array_equal(want_ids, got_ids):
            ids_bad += 1
            row_bad = True
        if not np.array_equal(
            want_probs.view(np.uint64), got_probs.view(np.uint64)
        ):
            prob_bad += 1
            row_bad = True

        full_lse = mx.logsumexp(dense.astype(mx.float32) * scale, axis=-1)
        compact_lse = mx.logsumexp(compact.astype(mx.float32) * scale, axis=-1)
        mx.eval(full_lse, compact_lse)
        ulps.append(
            int(
                _ulp_distance(
                    np.asarray(full_lse, dtype=np.float32).reshape(-1),
                    np.asarray(compact_lse, dtype=np.float32).reshape(-1),
                )[0]
            )
        )

        want_draw = sample_from_distribution(
            sparse_distribution_from_mlx_logits(dense.astype(mx.float32), sampler),
            np.random.default_rng(4242 + index),
        )
        got_draw = sample_from_distribution(
            prescatter_sparse_distribution(plan, compact.astype(mx.float32), sampler),
            np.random.default_rng(4242 + index),
        )
        if want_draw != got_draw:
            draw_bad += 1
            row_bad = True
        if row_bad and first_bad < 0:
            first_bad = index

    return {
        "parity_rows": parity_rows,
        "support_ids_differing": ids_bad,
        "support_prob_bits_differing": prob_bad,
        "draw_rows_differing": draw_bad,
        "first_differing_row": first_bad,
        "logsumexp_ulp_nonzero_rows": int(sum(1 for u in ulps if u != 0)),
        "logsumexp_ulp_max_abs": int(max((abs(u) for u in ulps), default=0)),
    }


def self_test() -> int:
    """CPU-only, no MLX: the ranked table and the ULP helper."""

    ids = ranked_ids(4096, 16384, 3)
    assert ids.shape == (4096,)
    assert bool(np.all(ids[1:] > ids[:-1])), "ranked table must be ascending"
    one = np.float32(1.0)
    nxt = np.nextafter(one, np.float32(2.0), dtype=np.float32)
    assert int(_ulp_distance(np.array([nxt]), np.array([one]))[0]) == 1
    assert int(_ulp_distance(np.array([one]), np.array([one]))[0]) == 0
    neg = np.float32(-2.5)
    neg_next = np.nextafter(neg, np.float32(0.0), dtype=np.float32)
    assert int(_ulp_distance(np.array([neg_next]), np.array([neg]))[0]) == 1
    # The mapping is a total order over floats; it deliberately collapses
    # +0.0 and -0.0, which a logsumexp of a strictly positive sum never sees.
    print("[micro_draft_k20] self-test ok (ranked table ascending, ULP helper sane)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--lane", choices=("queued", "eager"), default="queued")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--parity-rows", type=int, default=32)
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help=(
            "rowsxvocab, e.g. 65536x248320 (repeatable; default: the production "
            "FR-Spec shape)"
        ),
    )
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    print(BANNER)
    _require_mlx()

    shapes = ((FRSPEC_ROWS, FULL_VOCAB),)
    if args.shape:
        shapes = tuple(
            tuple(int(part) for part in spec.lower().split("x"))
            for spec in args.shape
        )

    sampler = _sampler(args.temperature, args.top_p)
    report: dict[str, object] = {
        "reps": args.reps,
        "lane": args.lane,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": TOP_K,
        "shapes": {},
    }
    for rows, vocab_rows in shapes:
        ids_np = ranked_ids(rows, vocab_rows, args.seed)
        ids_dev = mx.array(ids_np, dtype=mx.int32)
        compact = mx.array(make_compact(rows, args.seed + rows))
        plan = build_plan(ids_np, rows, vocab_rows, sampler)
        dense = scatter(compact, ids_dev, vocab_rows)
        mx.eval(compact, ids_dev, dense)

        key = f"{rows}x{vocab_rows}"
        entry: dict[str, object] = {
            "parity": parity(
                sampler,
                plan,
                ids_np,
                rows,
                vocab_rows,
                seed=args.seed,
                parity_rows=args.parity_rows,
            )
        }
        variants = build_variants(sampler, plan, compact, ids_dev, dense)
        for name, (fn, host_terminated) in variants.items():
            entry[name] = time_variant(
                fn,
                reps=args.reps,
                warmup=args.warmup,
                lane=args.lane,
                host_terminated=host_terminated,
            )
        report["shapes"][key] = entry

        print(f"\n[{key}]  parity: {entry['parity']}")
        floor = float(entry["read_floor_compact"]["median_ms"])
        for name in variants:
            row = entry[name]
            print(
                f"  {name:<22} {row['median_ms']:8.4f} ms"
                f"  (p10 {row['p10_ms']:.4f} / p90 {row['p90_ms']:.4f},"
                f" lane={row['lane']}, x{row['median_ms'] / floor:6.2f} compact floor)"
            )
        read_delta = (
            float(entry["stock_read"]["median_ms"])
            - float(entry["prescatter_read"]["median_ms"])
        )
        support_delta = (
            float(entry["stock_scatter_serial"]["median_ms"])
            - float(entry["prescatter_serial"]["median_ms"])
        )
        print(
            f"  --> stock_read - prescatter_read      = {read_delta:+.4f} ms/step"
            f"  ({3 * read_delta:+.4f} ms/window at depth 3)"
        )
        print(
            f"  --> stock_scatter_serial - prescatter = {support_delta:+.4f} ms/step"
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[micro_draft_k20] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
