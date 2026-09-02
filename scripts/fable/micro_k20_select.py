#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price the stock K20 support builder against the exact device selector.

The question
------------
``L-fable-decode-ideas.md`` §A claims the stock lane's four per-cycle top-20
selections are exposed (each ends in a hard ``np.asarray``), and §0.1 sizes
them at 0.4-1.6 ms each.  ``K-novel-decode-ideas.md`` §K1 argues most of the
3.15 ms/window ``verify_target_distribution_time_s`` is the compiled verify
graph's TAIL being awaited at the first sync, not selection work -- its
MLX_DISABLE_COMPILE arm shows the same code at 0.70 ms.  The two readings
predict very different wins, and this bench separates them: with no model and
no verify graph in front of it, whatever the selector costs here is selection.

Variants (all on the same synthetic float32 logit block)
--------------------------------------------------------
``stock_serial``
    ``mtplx.fast_sampling._device_serial_support_arrays`` -- production.  One
    ``argpartition`` to an 80-candidate superset, ``take_along_axis``, a full
    ``logsumexp``, ``mx.eval``, then a host ``np.lexsort`` over [rows, 80] and
    the top-p mask.  INCLUDES the host tail and the sync, because that is what
    the lane pays.
``stock_deterministic``
    ``_deterministic_mlx_top_k_support`` + ``_order_bounded_mlx_top_k_support``
    -- the all-device selector the serial builder falls back to on a cutoff
    spill: two full-vocabulary ``argpartition`` families, a full-vocabulary
    ``cumsum``, six full-width compares.  Device-only, no host tail.
``device_k20``
    ``mtplx.kernels.fable_device_k20.device_top_k`` -- the parked PR391
    two-stage exact selector (970 tiles of 256 -> one 256-lane radix merge),
    shape-parameterised.  Device-only, no host tail.
``device_k20_host``
    ``device_k20`` plus the same logsumexp/exp, the same single sync, and the
    same float64 top-p mask the production route runs
    (``fable_device_k20.finalize_target_support``).  This is the apples-to-
    apples partner for ``stock_serial``.
``read_floor``
    ``mx.sum`` over the block: the bandwidth floor for touching the logits
    once.  Anything above this is selection overhead, not reading.

Lanes
-----
``--lane queued`` (default) issues ``--reps`` calls and evaluates ONCE, then
divides.  ``--lane eager`` evaluates every call.  Per the
``queued-vs-eager-metal-microbench`` note the eager lane charges every call a
host sync and can INVERT a verdict for microsecond kernels, so promotions are
decided on the queued lane; the eager column is here because the production
draft site really is eager (its ``np.asarray`` follows immediately) and the
production target site really does pay one sync.  ``stock_serial`` and
``device_k20_host`` are host-terminated by construction and are only ever
timed in the eager sense.

Parity
------
Every device variant is checked against ``reference_top_k`` (kernel-free
NumPy, value desc / real id asc) AND against ``stock_serial``'s token rows,
and the report prints the number of DIFFERING rows and the first differing
row, so "exact" is a measurement here rather than an assertion.

Decision rule (from L §A / K §K1)
---------------------------------
If ``stock_serial - device_k20_host`` at ``[4, 248320]`` is >= 0.5 ms, the
target-side lever is real and the receipts' 3.15 ms is selection, not graph
tail.  If it is <= 0.15 ms, K1's discount holds and only the draft side
(three ``[1, V]`` selections, each followed by a sync) is worth building.
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
    "[micro_k20_select] GPU WINDOW REQUIRED -- run under "
    f"{LOCK_PATH} via bench/laguna/run_guarded.py"
)

TARGET_VOCAB = 248_320      # Qwen3.8-Flash-Next full vocabulary
FRSPEC_VOCAB = 65_536       # the ranked draft head's compact domain
TOP_K = 20
DEFAULT_SHAPES = ((4, TARGET_VOCAB), (1, TARGET_VOCAB), (1, FRSPEC_VOCAB))


def _require_mlx() -> None:
    global mx
    import mlx.core

    mx = mlx.core


def _sampler(temperature: float, top_p: float):
    from mtplx.sampling import SamplerConfig

    return SamplerConfig(temperature=temperature, top_p=top_p, top_k=TOP_K)


def make_block(rows: int, vocab: int, seed: int):
    """A logit block shaped like a real one: a heavy head, a long tail."""

    rng = np.random.default_rng(seed)
    base = rng.standard_normal((rows, vocab)).astype(np.float32) * 2.0
    # Real rows have a few dozen tokens far above the bulk.
    for row in range(rows):
        peaks = rng.choice(vocab, size=64, replace=False)
        base[row, peaks] += rng.uniform(6.0, 14.0, size=64).astype(np.float32)
    return mx.array(base), base


def build_variants(sampler, plan):
    from mtplx.fable_device_k20 import (
        finalize_target_support,
        target_support_device,
    )
    from mtplx.fast_sampling import (
        _deterministic_mlx_top_k_support,
        _device_serial_support_arrays,
        _order_bounded_mlx_top_k_support,
    )
    from mtplx.kernels.fable_device_k20 import device_top_k

    scale = 1.0 / float(sampler.temperature)

    def stock_serial(block):
        return _device_serial_support_arrays(block, sampler)

    def stock_deterministic(block):
        scaled = block.astype(mx.float32) * scale
        idx, vals = _deterministic_mlx_top_k_support(scaled, TOP_K)
        return _order_bounded_mlx_top_k_support(idx, vals)

    def device_k20(block):
        scaled = block.astype(mx.float32) * scale
        return device_top_k(scaled, top_k=TOP_K)

    def device_k20_host(block):
        ids, values, probs = target_support_device(block, plan)
        mx.eval(*[leaf for leaf in (ids, values, probs) if leaf is not None])
        return finalize_target_support(
            np.asarray(ids, dtype=np.int64),
            np.asarray(values, dtype=np.float32),
            None if probs is None else np.asarray(probs, dtype=np.float64),
            plan,
        )

    def read_floor(block):
        return (mx.sum(block.astype(mx.float32), axis=-1),)

    return {
        "stock_serial": (stock_serial, True),
        "stock_deterministic": (stock_deterministic, False),
        "device_k20": (device_k20, False),
        "device_k20_host": (device_k20_host, True),
        "read_floor": (read_floor, False),
    }


def _leaves(out):
    if isinstance(out, tuple):
        return [leaf for leaf in out if leaf is not None]
    return [out]


def time_variant(fn, block, *, reps, warmup, lane, host_terminated):
    for _ in range(warmup):
        out = fn(block)
        if not host_terminated:
            mx.eval(*_leaves(out))
    if host_terminated or lane == "eager":
        samples = []
        for _ in range(reps):
            start = time.perf_counter()
            out = fn(block)
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
    # Queued lane: build `reps` independent graphs, evaluate once.
    outs = []
    start = time.perf_counter()
    for _ in range(reps):
        outs.extend(_leaves(fn(block)))
    mx.eval(*outs)
    total = (time.perf_counter() - start) * 1e3
    return {
        "median_ms": total / reps,
        "p10_ms": total / reps,
        "p90_ms": total / reps,
        "lane": "queued",
    }


def parity(block, host_block, sampler, plan):
    """Differing-row counts against the NumPy oracle and against stock."""

    from mtplx.fable_device_k20 import (
        finalize_target_support,
        target_support_device,
    )
    from mtplx.fast_sampling import _device_serial_support_arrays
    from mtplx.fable_device_k20 import reference_top_k
    from mtplx.kernels.fable_device_k20 import device_top_k

    scale = np.float32(1.0 / float(sampler.temperature))
    scaled_host = (host_block.astype(np.float32) * scale).astype(np.float32)
    want_ids, want_vals = reference_top_k(scaled_host, top_k=TOP_K)

    scaled = block.astype(mx.float32) * float(scale)
    got_ids_dev, got_vals_dev = device_top_k(scaled, top_k=TOP_K)
    mx.eval(got_ids_dev, got_vals_dev)
    got_ids = np.asarray(got_ids_dev, dtype=np.int64)
    got_vals = np.asarray(got_vals_dev, dtype=np.float32)

    stock_ids, stock_probs, _ = _device_serial_support_arrays(block, sampler)
    ids, values, probs = target_support_device(block, plan)
    mx.eval(*[leaf for leaf in (ids, values, probs) if leaf is not None])
    lane_ids, lane_probs = finalize_target_support(
        np.asarray(ids, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
        None if probs is None else np.asarray(probs, dtype=np.float64),
        plan,
    )

    def first_diff(a, b):
        rows = np.nonzero(~np.all(a == b, axis=1))[0]
        return int(rows[0]) if rows.size else -1

    return {
        "oracle_id_rows_differing": int(
            np.count_nonzero(~np.all(got_ids == want_ids, axis=1))
        ),
        "oracle_value_rows_differing": int(
            np.count_nonzero(
                ~np.all(got_vals.view(np.uint32) == want_vals.view(np.uint32), axis=1)
            )
        ),
        "oracle_first_differing_row": first_diff(got_ids, want_ids),
        "stock_id_rows_differing": int(
            np.count_nonzero(~np.all(lane_ids == stock_ids, axis=1))
        ),
        "stock_prob_rows_differing": int(
            np.count_nonzero(~np.all(lane_probs == stock_probs, axis=1))
        ),
        "stock_first_differing_row": first_diff(lane_ids, stock_ids),
    }


def choice_parity(block, sampler, plan, *, seed: int, rows: int) -> dict:
    """The device draft sampler against its own CPU oracle, on real shapes.

    The PR391 branch validated ``qwen4_frspec_k20_float32_choice`` on ITS
    lane; this lane feeds it different rows (the FRSpec scatter's top-20 under
    a full-vocabulary logsumexp), so the kernel-vs-oracle agreement is
    re-measured here rather than inherited.  It also checks the host mirror
    ``prepare_draft_row_f32`` against the oracle's preparation, because that
    mirror is what becomes the accept loop's ``q``.
    """

    from mtplx.fable_device_k20 import (
        build_uniform_descriptors,
        draft_distribution,
        prepare_draft_row_f32,
    )
    from mtplx.kernels.fable_device_k20 import device_top_k
    from mtplx.kernels.qwen4_frspec_k20_float32_choice import (
        _prepare_reference_row,
        bind_qwen4_frspec_k20_float32_choice,
        reference_qwen4_frspec_k20_float32_choice,
    )

    scale = 1.0 / float(sampler.temperature)
    scaled = block.astype(mx.float32) * scale
    ids, values = device_top_k(scaled, top_k=TOP_K)
    probs = mx.exp(values - mx.logsumexp(scaled, axis=-1, keepdims=True))
    uniforms = np.random.default_rng(seed).random(rows)
    descriptors = build_uniform_descriptors(uniforms)
    apply = bind_qwen4_frspec_k20_float32_choice(top_p=float(sampler.top_p))
    selected, _, _, _ = apply(
        ids, values, probs, mx.array(descriptors, dtype=mx.uint32)
    )
    mx.eval(ids, values, probs, selected)

    host_ids = np.asarray(ids, dtype=np.uint32)
    host_values = np.asarray(values, dtype=np.float32)
    host_probs = np.asarray(probs, dtype=np.float32)
    want, _, _, _ = reference_qwen4_frspec_k20_float32_choice(
        host_ids, host_values, host_probs, descriptors, top_p=float(sampler.top_p)
    )
    got = np.asarray(selected, dtype=np.uint32)

    mirror_bad = 0
    support_bad = 0
    for row in range(rows):
        want_ids, want_cdf = _prepare_reference_row(
            host_ids[row], host_values[row], host_probs[row],
            np.float32(sampler.top_p),
        )
        got_ids, got_norm = prepare_draft_row_f32(
            host_ids[row], host_values[row], host_probs[row], float(sampler.top_p)
        )
        if not np.array_equal(
            np.asarray(want_ids, dtype=np.uint32), got_ids
        ) or not np.array_equal(
            np.asarray(want_cdf, dtype=np.float32).view(np.uint32),
            np.cumsum(got_norm, dtype=np.float32).view(np.uint32),
        ):
            mirror_bad += 1
        distribution, _ = draft_distribution(
            host_ids[row], host_values[row], host_probs[row],
            top_p=float(sampler.top_p), vocab_size=int(block.shape[-1]),
        )
        hits = np.nonzero(distribution.token_ids == int(got[row]))[0]
        if hits.size == 0 or distribution.probs[int(hits[0])] <= 0.0:
            support_bad += 1

    return {
        "choice_rows_differing": int(np.count_nonzero(got != want)),
        "host_mirror_rows_differing": int(mirror_bad),
        "sampled_token_outside_q_support": int(support_bad),
    }


def self_test() -> int:
    """CPU-only: the NumPy oracle against a brute-force sort.  No MLX."""

    from mtplx.fable_device_k20 import reference_top_k

    rng = np.random.default_rng(11)
    rows = rng.standard_normal((3, 4096)).astype(np.float32)
    rows[0, 7] = rows[0, 9] = np.float32(5.0)      # value tie
    rows[1, 3] = np.float32(0.0)
    rows[1, 4] = np.float32(-0.0)                  # signed-zero tie
    ids, values = reference_top_k(rows, top_k=TOP_K)
    for row in range(rows.shape[0]):
        brute = sorted(
            range(rows.shape[1]),
            key=lambda token: (-float(rows[row, token]), token),
        )[:TOP_K]
        assert list(ids[row]) == brute, (row, list(ids[row])[:5], brute[:5])
        assert np.array_equal(values[row], rows[row][brute])
    print("[micro_k20_select] self-test ok (oracle == brute-force sort)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--lane", choices=("queued", "eager"), default="queued")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="rowsxvocab, e.g. 4x248320 (repeatable; default: the three lane shapes)",
    )
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    print(BANNER)
    _require_mlx()

    from mtplx.fable_device_k20 import DeviceK20Plan

    shapes = DEFAULT_SHAPES
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
        "shapes": {},
    }
    for rows, vocab in shapes:
        block, host_block = make_block(rows, vocab, args.seed + rows * 31 + vocab)
        mx.eval(block)
        plan = DeviceK20Plan(
            depth=3,
            top_k=TOP_K,
            vocab_size=vocab,
            target_rows=rows,
            temperature=args.temperature,
            top_p=args.top_p,
            draft_temperature=args.temperature,
            draft_top_p=args.top_p,
            draft_vocab_size=vocab,
            fused_verify_input=False,
        )
        variants = build_variants(sampler, plan)
        key = f"{rows}x{vocab}"
        entry: dict[str, object] = {
            "parity": parity(block, host_block, sampler, plan),
            "choice_parity": choice_parity(
                block, sampler, plan, seed=args.seed + rows, rows=rows
            ),
        }
        for name, (fn, host_terminated) in variants.items():
            entry[name] = time_variant(
                fn,
                block,
                reps=args.reps,
                warmup=args.warmup,
                lane=args.lane,
                host_terminated=host_terminated,
            )
        report["shapes"][key] = entry

        print(f"\n[{key}]  parity: {entry['parity']}")
        print(f"         choice: {entry['choice_parity']}")
        floor = float(entry["read_floor"]["median_ms"])
        for name in (
            "stock_serial",
            "stock_deterministic",
            "device_k20",
            "device_k20_host",
            "read_floor",
        ):
            row = entry[name]
            print(
                f"  {name:<22} {row['median_ms']:8.4f} ms"
                f"  (p10 {row['p10_ms']:.4f} / p90 {row['p90_ms']:.4f},"
                f" lane={row['lane']}, x{row['median_ms'] / floor:5.2f} read floor)"
            )
        delta = (
            float(entry["stock_serial"]["median_ms"])
            - float(entry["device_k20_host"]["median_ms"])
        )
        print(f"  --> stock_serial - device_k20_host = {delta:+.4f} ms")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[micro_k20_select] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
