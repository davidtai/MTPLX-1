#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price MTPLX_FABLE_DEVICE_DRAFT_CHAIN's three halves, and check its parity.

The claim
---------
Two dispatch censuses put 2.62 ms/cycle in three ``v_Exp -> gather_front`` gaps
-- one per draft depth.  Each gap is the whole host round trip between the end
of depth d's K20 support and the first op of depth d+1's forward:

    A  GPU drain + ``np.asarray`` of the 80-candidate superset
    B  host lexsort / top-p mask / ``_serial_row_distribution`` / ``rng.choice``
    C  the PYTHON construction of depth d+1's MTP graph (~300 MLX ops)

``MTPLX_FABLE_DEVICE_K20`` (W24/W28b) removed A and B and cost +1.14%.  This
lane removes A, B **and C** -- C via one ``mx.compile``d per-depth body -- and
shrinks what is left by selecting on the FR-Spec head's 65,536 pre-scatter row
instead of the 248,320 scattered one.  Three separately measurable halves, so
this script measures them separately rather than asserting the sum.

Arms
----
``stock_serial``
    What the lane pays per cycle for selection + sampling: ``depth`` x
    [scatter -> ``_device_serial_support_arrays`` over 248,320 ->
    ``_serial_row_distribution`` -> ``rng.choice``].  Host-terminated by
    construction (the builder syncs and calls ``np.asarray`` itself), which is
    the production shape.  This is A + B.
``device_chain``
    ``depth`` x [exact 65,536-row K20 -> ranked-id map -> choice kernel],
    queued, then ONE ``mx.eval``, then the host ``draft_distribution`` per
    depth.  The same A + B, on this lane.
``body_eager`` / ``body_compiled``
    C, in isolation.  A STAND-IN chain of ``--body-ops`` MLX ops at the MTP
    layer's real hidden width -- not the real DecoderLayer, which needs the
    model -- issued from Python every call (``body_eager``) versus traced once
    and replayed (``body_compiled``).  The gap between them is the host
    issuance cost per depth that ``mx.compile`` removes, and it scales with
    ``--body-ops``; the census's ~300 ops/depth is the default.  Read this as
    a mechanism measurement with a slope, not as the production number -- only
    the ABBA can give that.
``descriptor_fraction`` / ``descriptor_fast``
    Host only, no Metal.  The shipped ``build_pcg64_midpoint_descriptors``
    (two ``Fraction`` descriptor derivations per uniform: one to build, one to
    re-validate) versus ``fable_device_draft_chain.fast_midpoint_descriptors``.
    W24/W28b paid the former on the serial critical path every cycle.

Parity (counted, never asserted)
--------------------------------
Over ``--parity-rows`` independent rows, against the shipped stock lane:

* ``support_ids_differing`` -- the 20 selected token ids, raw bits.
* ``support_prob_bits_differing`` -- the stock float64 support probabilities
  versus this lane's, raw bits.  EXPECTED NONZERO in ``chain`` mode: the
  device sampler prepares its row in float32 (float32 ``cumulative_before``
  for the nucleus cut, float32 normalisation) where the host prepares it in
  float64.  That is W24's documented, accepted divergence -- a different but
  valid proposal ``q``, which the accept/correct law admits for free.  The
  count bounds it.
* ``logsumexp_ulp`` -- signed float32 ULP distance between the 248,320-lane
  and 65,536-lane reductions.  W42's one claim the CPU tests cannot settle;
  ``micro_draft_k20.py`` measured one 1-ULP row in 32.
* ``token_rows_differing`` -- the drafted token from an identically seeded
  ``rng``: the stock ``rng.choice`` on the float64 row versus the device
  kernel's exact-rational walk of the float32 CDF, over the SAME uniform.
  This is the number that says how often a real run's digest could move.
* ``body_mode_token_rows_differing`` -- the same comparison for ``body`` mode.
  Zero here means the HOST TAIL is exact; it does NOT mean the lane is
  bit-identical.  This script feeds both paths the same synthetic compact row,
  so it cannot see the two rounding sources that actually move a real run's
  digest: the `mx.compile` fusion inside the MTP DecoderLayer forward (which
  `fable_compiled_draft` documents as expected) and, through
  ``logsumexp_ulp``, the 65,536-vs-248,320 reduction width.  Window
  1788400641 measured differing digests on 3/3 seeds in ``body`` mode.  Both
  modes are ROUNDING CLASS and are judged on HumanEval / the long-prompt
  agreement screen, never on digest equality.

Decision rule
-------------
Per cycle at depth 3 the lane's own arithmetic is

    saved  =  (stock_serial - device_chain)
            + depth * (body_eager - body_compiled)
            + (descriptor_fraction - descriptor_fast)

The census leaves 2.62 ms/cycle in those three gaps and the target is >= 1.5
ms/cycle removed.  ``body_mode_token_rows_differing`` must be 0 (it checks the host tail on an
identical row); ``token_rows_differing`` is a rate to report, not a gate.
Neither number gates the lane: both modes are rounding class, so the gate is a
task eval.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

# MLX is imported lazily so ``--host-only`` stays runnable off the lock.
mx = None


LOCK_PATH = "/tmp/mtplx-gpu-exclusive.lock"
BANNER = (
    "[micro_device_draft_chain] GPU WINDOW REQUIRED -- run under "
    f"{LOCK_PATH} via bench/laguna/run_guarded.py"
)

FULL_VOCAB = 248_320        # Qwen3.8-Flash-Next full vocabulary
FRSPEC_ROWS = 65_536        # the ranked draft head's compact domain
HIDDEN_WIDTH = 2_560        # MTP recursion state width (stand-in body only)
TOP_K = 20
DEPTH = 3
SENTINEL = -1.0e30


def _require_mlx() -> None:
    global mx
    import mlx.core

    mx = mlx.core


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


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


def _sampler(temperature: float, top_p: float):
    from mtplx.sampling import SamplerConfig

    return SamplerConfig(temperature=temperature, top_p=top_p, top_k=TOP_K)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_call(fn, *, reps: int, warmup: int, host_terminated: bool):
    """Median/p10/p90 wall milliseconds per call.

    Host-terminated arms sync inside the call, so every rep is timed on its
    own -- which is the production shape.  Device-terminated arms are issued
    ``reps`` times and evaluated once, then divided: the queued lane, per the
    ``queued-vs-eager-metal-microbench`` note (an eager lane charges every call
    a host sync and can invert a verdict for microsecond kernels).
    """

    for _ in range(warmup):
        out = fn(0)
        if not host_terminated:
            mx.eval(out)

    samples: list[float] = []
    if host_terminated:
        for index in range(reps):
            started = time.perf_counter()
            fn(index)
            samples.append((time.perf_counter() - started) * 1e3)
    else:
        outputs = []
        started = time.perf_counter()
        for index in range(reps):
            outputs.append(fn(index))
        mx.eval(*outputs)
        elapsed = (time.perf_counter() - started) * 1e3
        samples = [elapsed / reps]
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[max(0, int(0.1 * (len(samples) - 1)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * (len(samples) - 1)))],
        "reps": reps,
        "lane": "eager" if host_terminated else "queued",
    }


# ---------------------------------------------------------------------------
# Arm A/B: selection + sampling
# ---------------------------------------------------------------------------


def stock_cycle(sampler, compact_rows, ids_dev, vocab_rows, uniforms):
    """One cycle's worth of stock draft reads: depth syncs, depth choices."""

    from mtplx.fast_sampling import (
        _device_serial_support_arrays,
        _serial_row_distribution,
    )

    tokens = []
    for depth in range(len(compact_rows)):
        dense = scatter(compact_rows[depth], ids_dev, vocab_rows)
        token_rows, prob_rows, _ = _device_serial_support_arrays(
            dense.reshape(1, -1).astype(mx.float32), sampler
        )
        distribution = _serial_row_distribution(
            token_rows[0], prob_rows[0], vocab_rows
        )
        probs = np.asarray(distribution.probs, dtype=np.float64)
        cdf = probs.cumsum()
        cdf /= cdf[-1]
        index = int(cdf.searchsorted(np.float64(uniforms[depth]), side="right"))
        tokens.append(
            int(np.asarray(distribution.token_ids, dtype=np.int64)[min(index, probs.size - 1)])
        )
    return tokens


def device_cycle(plan_top_p, temperature, choice, compact_rows, ids_dev, descriptors):
    """One cycle's worth of device chain: queued selects+samples, ONE eval."""

    from mtplx.fast_sampling import (
        _deterministic_mlx_top_k_support,
        _order_bounded_mlx_top_k_support,
    )

    leaves = []
    for depth in range(len(compact_rows)):
        flat = compact_rows[depth].astype(mx.float32) * (1.0 / float(temperature))
        local_ids, values = _deterministic_mlx_top_k_support(flat, TOP_K)
        local_ids, values = _order_bounded_mlx_top_k_support(local_ids, values)
        probs = mx.exp(values - mx.logsumexp(flat, axis=-1, keepdims=True))
        real_ids = mx.take(ids_dev, local_ids)
        selected, raw_ids, raw_values, raw_probs = choice(
            real_ids.astype(mx.uint32).reshape(1, TOP_K),
            values.astype(mx.float32).reshape(1, TOP_K),
            probs.astype(mx.float32).reshape(1, TOP_K),
            descriptors[depth : depth + 1],
        )
        leaves.append((selected, raw_ids, raw_values, raw_probs))
    flat_leaves = [leaf for group in leaves for leaf in group]
    mx.eval(*flat_leaves)          # THE one readback
    return leaves


def device_cycle_host_tail(leaves, top_p, vocab_rows):
    """The host work after the single eval: the accept loop's ``q`` rows."""

    from mtplx.fable_device_k20 import draft_distribution

    tokens = []
    distributions = []
    for selected, raw_ids, raw_values, raw_probs in leaves:
        distribution, _ = draft_distribution(
            np.asarray(raw_ids, dtype=np.uint32).reshape(-1),
            np.asarray(raw_values, dtype=np.float32).reshape(-1),
            np.asarray(raw_probs, dtype=np.float32).reshape(-1),
            top_p=top_p,
            vocab_size=vocab_rows,
        )
        tokens.append(int(np.asarray(selected).reshape(-1)[0]))
        distributions.append(distribution)
    return tokens, distributions


# ---------------------------------------------------------------------------
# Arm C: host graph construction (stand-in body)
# ---------------------------------------------------------------------------


def make_body(ops: int, width: int):
    """A stand-in for one MTP depth's host issuance.

    ``ops`` MLX ops at the MTP recursion width, chained so nothing is dead.
    This is NOT the real DecoderLayer (that needs the model); it is a knob that
    isolates ONE variable -- how many ops Python issues per depth -- so the
    difference between the eager and compiled arms is the issuance cost and
    nothing else.  The census counts ~300 dispatches per draft depth outside
    the MoE, which is the default.
    """

    scale = mx.array(np.float32(1.0009))
    bias = mx.array(np.zeros(width, dtype=np.float32))

    def body(state):
        value = state
        for index in range(ops):
            if index % 3 == 0:
                value = value * scale
            elif index % 3 == 1:
                value = value + bias
            else:
                value = mx.maximum(value, value * np.float32(0.5))
        return value

    return body


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def parity(sampler, ids_np, ids_dev, rows, vocab_rows, *, seed, parity_rows, choice):
    from mtplx.fable_device_draft_chain import (
        fast_midpoint_descriptors,
        host_support_tail,
    )
    from mtplx.fable_device_k20 import draft_distribution
    from mtplx.fast_sampling import (
        _deterministic_mlx_top_k_support,
        _device_serial_support_arrays,
        _order_bounded_mlx_top_k_support,
        _serial_row_distribution,
    )

    counters = {
        "rows": int(parity_rows),
        "support_ids_differing": 0,
        "support_prob_bits_differing": 0,
        "logsumexp_ulp_nonzero": 0,
        "logsumexp_ulp_max": 0,
        "token_rows_differing": 0,
        "body_mode_token_rows_differing": 0,
    }
    temperature = float(sampler.temperature)
    top_p = float(sampler.top_p)

    for row in range(int(parity_rows)):
        compact_np = make_compact(rows, seed + 5_000 + row)
        compact = mx.array(compact_np)
        dense = scatter(compact, ids_dev, vocab_rows)

        # -- stock lane -----------------------------------------------------
        stock_ids, stock_probs, _ = _device_serial_support_arrays(
            dense.reshape(1, -1).astype(mx.float32), sampler
        )
        stock_dist = _serial_row_distribution(stock_ids[0], stock_probs[0], vocab_rows)
        stock_rng = np.random.default_rng(seed + row)
        stock_token = int(stock_rng.choice(stock_dist.token_ids, p=stock_dist.probs))

        # -- this lane's exact selector on the compact row --------------------
        flat = compact.astype(mx.float32) * (1.0 / temperature)
        local_ids, values = _deterministic_mlx_top_k_support(flat, TOP_K)
        local_ids, values = _order_bounded_mlx_top_k_support(local_ids, values)
        log_total_compact = mx.logsumexp(flat, axis=-1, keepdims=True)
        probs = mx.exp(values - log_total_compact)
        real_ids = mx.take(ids_dev, local_ids)
        log_total_full = mx.logsumexp(
            dense.reshape(1, -1).astype(mx.float32) * (1.0 / temperature),
            axis=-1,
            keepdims=True,
        )
        mx.eval(real_ids, values, probs, log_total_compact, log_total_full)

        got_ids = np.asarray(real_ids, dtype=np.int64).reshape(-1)
        got_values = np.asarray(values, dtype=np.float32).reshape(-1)
        got_probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        if not np.array_equal(got_ids, np.asarray(stock_ids[0], dtype=np.int64)):
            counters["support_ids_differing"] += 1

        ulp = int(
            np.asarray(log_total_compact, dtype=np.float32).view(np.int32).reshape(-1)[0]
        ) - int(
            np.asarray(log_total_full, dtype=np.float32).view(np.int32).reshape(-1)[0]
        )
        if ulp:
            counters["logsumexp_ulp_nonzero"] += 1
            counters["logsumexp_ulp_max"] = max(
                counters["logsumexp_ulp_max"], abs(ulp)
            )

        # -- body mode: the stock host tail on the exact support --------------
        body_token_rows, body_prob_rows = host_support_tail(
            got_ids, got_values, got_probs, top_p=top_p, top_k=TOP_K
        )
        body_dist = _serial_row_distribution(
            body_token_rows[0], body_prob_rows[0], vocab_rows
        )
        body_rng = np.random.default_rng(seed + row)
        body_token = int(body_rng.choice(body_dist.token_ids, p=body_dist.probs))
        if body_token != stock_token:
            counters["body_mode_token_rows_differing"] += 1
        if not np.array_equal(
            np.asarray(body_dist.probs, dtype=np.float64).view(np.uint64),
            np.asarray(stock_dist.probs, dtype=np.float64).view(np.uint64),
        ):
            counters["support_prob_bits_differing"] += 1

        # -- chain mode: the device sampler on the same uniform ---------------
        chain_rng = np.random.default_rng(seed + row)
        uniform = np.asarray(chain_rng.random(1, dtype=np.float64))
        descriptor = mx.array(fast_midpoint_descriptors(uniform))
        selected, raw_ids, raw_values, raw_probs = choice(
            real_ids.astype(mx.uint32).reshape(1, TOP_K),
            values.astype(mx.float32).reshape(1, TOP_K),
            probs.astype(mx.float32).reshape(1, TOP_K),
            descriptor,
        )
        mx.eval(selected, raw_ids, raw_values, raw_probs)
        chain_token = int(np.asarray(selected).reshape(-1)[0])
        if chain_token != stock_token:
            counters["token_rows_differing"] += 1
        # Prove the row the accept loop scores is the row the device sampled.
        draft_distribution(
            np.asarray(raw_ids, dtype=np.uint32).reshape(-1),
            np.asarray(raw_values, dtype=np.float32).reshape(-1),
            np.asarray(raw_probs, dtype=np.float32).reshape(-1),
            top_p=top_p,
            vocab_size=vocab_rows,
        )
    return counters


# ---------------------------------------------------------------------------
# Host-only arm: the descriptor build
# ---------------------------------------------------------------------------


def descriptor_arms(depth: int, reps: int, seed: int) -> dict[str, object]:
    from mtplx.fable_device_draft_chain import fast_midpoint_descriptors
    from mtplx.kernels.qwen4_frspec_k20_float32_choice import (
        build_pcg64_midpoint_descriptors,
    )

    rng = np.random.default_rng(seed)
    tapes = [rng.random(depth, dtype=np.float64) for _ in range(reps)]

    shipped = [build_pcg64_midpoint_descriptors(tape) for tape in tapes[:4]]
    fast = [fast_midpoint_descriptors(tape) for tape in tapes[:4]]
    identical = all(np.array_equal(a, b) for a, b in zip(shipped, fast))

    out: dict[str, object] = {"identical": bool(identical), "depth": int(depth)}
    for name, fn in (
        ("descriptor_fraction", build_pcg64_midpoint_descriptors),
        ("descriptor_fast", fast_midpoint_descriptors),
    ):
        samples = []
        for tape in tapes:
            started = time.perf_counter()
            fn(tape)
            samples.append((time.perf_counter() - started) * 1e3)
        samples.sort()
        out[name] = {
            "median_ms": statistics.median(samples),
            "p90_ms": samples[min(len(samples) - 1, int(0.9 * (len(samples) - 1)))],
            "reps": len(samples),
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--rows", type=int, default=FRSPEC_ROWS)
    parser.add_argument("--vocab-rows", type=int, default=FULL_VOCAB)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--parity-rows", type=int, default=64)
    parser.add_argument("--body-ops", type=int, default=300)
    parser.add_argument("--body-width", type=int, default=HIDDEN_WIDTH)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="descriptor arms only; issues no Metal work and needs no lock",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    report: dict[str, object] = {
        "script": "micro_device_draft_chain",
        "args": vars(args),
    }

    report["descriptors"] = descriptor_arms(args.depth, 200, args.seed)
    dsc = report["descriptors"]
    print(
        f"[descriptors] identical={dsc['identical']}  "
        f"fraction {dsc['descriptor_fraction']['median_ms']:.4f} ms/cycle  "
        f"fast {dsc['descriptor_fast']['median_ms']:.4f} ms/cycle  "
        f"--> {dsc['descriptor_fraction']['median_ms'] - dsc['descriptor_fast']['median_ms']:+.4f} ms/cycle"
    )
    if args.host_only:
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(report, handle, indent=2)
        return 0

    print(BANNER)
    _require_mlx()

    from mtplx.kernels.qwen4_frspec_k20_float32_choice import (
        bind_qwen4_frspec_k20_float32_choice,
    )
    from mtplx.fable_device_draft_chain import fast_midpoint_descriptors

    sampler = _sampler(args.temperature, args.top_p)
    choice = bind_qwen4_frspec_k20_float32_choice(top_p=float(args.top_p))

    ids_np = ranked_ids(args.rows, args.vocab_rows, args.seed)
    ids_dev = mx.array(ids_np.astype(np.uint32))
    compact_rows = [
        mx.array(make_compact(args.rows, args.seed + depth))
        for depth in range(args.depth)
    ]
    tape_rng = np.random.default_rng(args.seed + 999)
    uniforms = tape_rng.random(args.depth, dtype=np.float64)
    descriptors = mx.array(fast_midpoint_descriptors(uniforms))
    mx.eval(ids_dev, descriptors, *compact_rows)

    # -- A + B ---------------------------------------------------------------
    report["stock_serial"] = time_call(
        lambda _: stock_cycle(
            sampler, compact_rows, ids_dev, args.vocab_rows, uniforms
        ),
        reps=args.reps,
        warmup=args.warmup,
        host_terminated=True,
    )

    def _device_cycle(_):
        leaves = device_cycle(
            args.top_p, args.temperature, choice, compact_rows, ids_dev, descriptors
        )
        device_cycle_host_tail(leaves, float(args.top_p), args.vocab_rows)

    report["device_chain"] = time_call(
        _device_cycle,
        reps=args.reps,
        warmup=args.warmup,
        host_terminated=True,
    )

    # -- C -------------------------------------------------------------------
    state = mx.array(np.random.default_rng(args.seed).standard_normal(
        args.body_width
    ).astype(np.float32))
    mx.eval(state)
    body = make_body(args.body_ops, args.body_width)
    compiled = mx.compile(body)

    def _body_eager(_):
        out = body(state)
        mx.eval(out)

    def _body_compiled(_):
        out = compiled(state)
        mx.eval(out)

    report["body_eager"] = time_call(
        _body_eager, reps=args.reps, warmup=args.warmup, host_terminated=True
    )
    report["body_compiled"] = time_call(
        _body_compiled, reps=args.reps, warmup=max(args.warmup, 3), host_terminated=True
    )

    # -- parity --------------------------------------------------------------
    report["parity"] = parity(
        sampler,
        ids_np,
        ids_dev,
        args.rows,
        args.vocab_rows,
        seed=args.seed,
        parity_rows=args.parity_rows,
        choice=choice,
    )

    # -- report --------------------------------------------------------------
    print(f"\n[{args.rows} compact / {args.vocab_rows} full, depth {args.depth}]")
    for name in ("stock_serial", "device_chain", "body_eager", "body_compiled"):
        row = report[name]
        print(
            f"  {name:<16} {row['median_ms']:8.4f} ms"
            f"  (p10 {row['p10_ms']:.4f} / p90 {row['p90_ms']:.4f}, lane={row['lane']})"
        )
    select_delta = (
        float(report["stock_serial"]["median_ms"])
        - float(report["device_chain"]["median_ms"])
    )
    body_delta = args.depth * (
        float(report["body_eager"]["median_ms"])
        - float(report["body_compiled"]["median_ms"])
    )
    desc_delta = float(dsc["descriptor_fraction"]["median_ms"]) - float(
        dsc["descriptor_fast"]["median_ms"]
    )
    print(f"\n  select+sample (A+B)   {select_delta:+.4f} ms/cycle")
    print(
        f"  compiled body (C)     {body_delta:+.4f} ms/cycle"
        f"  ({args.body_ops} stand-in ops x {args.depth} depths)"
    )
    print(f"  descriptor build      {desc_delta:+.4f} ms/cycle (vs the W24 builder)")
    print(f"  --> chain-mode total  {select_delta + body_delta:+.4f} ms/cycle")
    print(f"      body-mode total   {body_delta:+.4f} ms/cycle")
    print(f"\n  parity: {report['parity']}")
    if report["parity"]["body_mode_token_rows_differing"]:
        print(
            "  !! the body-mode HOST TAIL disagrees with the stock sampler on "
            "an identical support row -- that IS a bug (the lane's overall "
            "rounding-class divergence lives upstream, in the compiled "
            "forward, and this arm cannot see it)"
        )
    report["summary"] = {
        "select_delta_ms_per_cycle": select_delta,
        "body_delta_ms_per_cycle": body_delta,
        "descriptor_delta_ms_per_cycle": desc_delta,
        "chain_total_ms_per_cycle": select_delta + body_delta,
        "body_total_ms_per_cycle": body_delta,
    }

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[micro_device_draft_chain] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
