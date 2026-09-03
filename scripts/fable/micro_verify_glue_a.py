#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price W70's ``MTPLX_FABLE_VERIFY_GLUE`` items at the verifier's real shapes.

Two items, both fusing rope glue inside the twelve QSA layers of the compiled
fixed-M4 verify body (``docs/perf/verify-node-census.md`` group ``qsa_rope``):

  qsa_rope      attention query+key rotation.  Per layer the stock chain is
                the RoPE table build (positions, angles, cos, sin) plus two
                five-dispatch rotations (two half multiply-adds and a 3-copy
                concatenate each).  ``kernels/qwen4_m4_rope.rope_qk`` issues
                the identical arithmetic as ONE dispatch that rotates both
                tensors from one table.
  qsa_rope_idx  the indexer's query preparation (RMSNorm + partial RoPE)
                through the SHIPPED ``qsa_indexer_prepare_queries_metal``,
                which the fixed-M4 lane never called.

WHAT THIS BENCH HAS TO SETTLE, and why a microbench is the right instrument

1. **Is it bit-exact under mx.compile?**  The kernel keeps its four products
   as distinct fp32 locals so Metal cannot contract ``a*b - c*d`` into an FMA.
   MLX's own fused elementwise kernel for the same expression is compiled by
   the same Metal front end and we do not control ITS contraction.  So parity
   is reported against the stock spelling in BOTH lanes -- eager (where the
   ops are separate kernels, the regime the shipped indexer kernel's tests
   pin) and compiled (the regime the production verify body actually runs).
   ``differing`` must be 0 in the compiled lane before this ships.
2. **Is it faster on the queued lane?**  Removing dispatches is not removing
   GPU time.  Each arm builds one verify cycle's worth of work and evaluates
   it ONCE, so the timing is queued, not a per-kernel host round trip -- an
   eager per-kernel lane costs >10x host sync at these sizes and can invert
   the verdict.
3. **How many dispatches does it actually remove?**  ``disp`` counts launches
   off the built graph (``mx.export_to_dot``), so the node claim in the
   engagement line is measured here rather than asserted.

ARMS

  rope_prediet  the pre-op-diet spelling (_rope_cos_sin + _apply_partial_rope)
  rope_stock    TODAY'S STACK: _shared_rope_cos_sin_half +
                _apply_partial_rope_half, WITHOUT the table memo -- the fixed
                M4 suffix does not enter _rope_table_scope(), so every QSA
                layer rebuilds its own table.  This is the arm to beat.
  rope_scoped   rope_stock inside _rope_table_scope().  Not an item: it is the
                free finding that the op diet's table sharing is inert on the
                fixed-M4 verify route, priced so the parent can decide whether
                a two-line hoist is worth its own arm.
  rope_fused    MTPLX_FABLE_VERIFY_GLUE item qsa_rope
  prep_stock    q_layernorm + the live half-width rotation (the fixed lane's
                _prepare_queries_eager)
  prep_fused    MTPLX_FABLE_VERIFY_GLUE item qsa_rope_idx

Every layer gets its OWN position leaf.  That is not cosmetic: mx.compile runs
a common-subexpression pass, and twelve layers fed one shared position would
collapse into a single table build -- flattering the stock arm with a sharing
the production graph does not have (each QSA layer's ``pos_start`` is its own
cache-offset leaf).  ``inv_freq`` IS shared, because the op diet's
``_rope_inv_freq_and_scaling_shared`` really does make it one object.

Imports mtplx on purpose: the stock arms call the PRODUCTION helpers, so this
bench cannot drift from the thing it certifies.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import time
from pathlib import Path

# MLX and mtplx are imported lazily so the CLI surface and the shape table can
# be read on a box that is not holding the GPU lock.
mx = None
_model = None
_rope = None
_prepare = None


def _require_mlx() -> None:
    global mx, _model, _rope, _prepare
    import mlx.core

    mx = mlx.core
    from mtplx.models import qwen4_exp
    from mtplx.kernels import qsa_indexer_prepare, qwen4_m4_rope

    _model = qwen4_exp
    _rope = qwen4_m4_rope
    _prepare = qsa_indexer_prepare


# --- Qwen3.8-Flash-Next fixed-M4 QSA geometry (TextArgs defaults) ----------
ROWS = 4                # fixed-M4 verify width
QSA_LAYERS = 12         # every 4th of 48
N_HEADS = 24
N_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64         # head_dim * partial_rotary_factor
IDX_HEADS = 4
IDX_HEAD_DIM = 128
RMS_EPS = 1e-6
ROPE_THETA = 10_000_000.0
#: The 16 K decode cell sits here; large positions are where bf16 cutoffs bite,
#: so a parity check at pos_start=0 alone proves very little.
POS_START = 17_405

FAMILIES = {
    "rope": ("rope_prediet", "rope_stock", "rope_scoped", "rope_fused"),
    "prep": ("prep_stock", "prep_fused"),
}
STOCK = {"rope": "rope_stock", "prep": "prep_stock"}
FAMILY_OF = {v: f for f, vs in FAMILIES.items() for v in vs}


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def rope_layer_prediet(q, k, pos, inv_freq, scaling):
    """Pre-diet: full-width table, standalone negate, two concatenates."""

    positions = pos + mx.arange(ROWS, dtype=mx.int32)
    cos, sin = _model._rope_cos_sin(positions, inv_freq, scaling)
    return (
        _model._apply_partial_rope(q, cos, sin),
        _model._apply_partial_rope(k, cos, sin),
    )


def rope_layer_stock(q, k, pos, inv_freq, scaling):
    """Today's stack: half-width table, split-half rotation, no memo hit."""

    cos, sin = _model._shared_rope_cos_sin_half(pos, ROWS, inv_freq, scaling)
    return (
        _model._apply_partial_rope_half(q, cos, sin),
        _model._apply_partial_rope_half(k, cos, sin),
    )


def rope_layer_fused(q, k, pos, inv_freq, scaling):
    """MTPLX_FABLE_VERIFY_GLUE item ``qsa_rope``: one dispatch."""

    return _rope.rope_qk(
        q, k, inv_freq, pos_start=pos, attention_scaling=scaling
    )


def prep_layer_stock(q_idx, pos, norm_weight, inv_freq, scaling):
    """The fixed lane's ``_prepare_queries_eager``, op diet armed."""

    normed = mx.fast.rms_norm(q_idx, norm_weight, RMS_EPS)
    cos, sin = _model._shared_rope_cos_sin_half(pos, ROWS, inv_freq, scaling)
    return (_model._apply_partial_rope_half(normed, cos, sin),)


def prep_layer_fused(q_idx, pos, norm_weight, inv_freq, scaling):
    """MTPLX_FABLE_VERIFY_GLUE item ``qsa_rope_idx``: the shipped kernel."""

    return (
        _prepare.qsa_indexer_prepare_queries_metal(
            q_idx,
            norm_weight,
            inv_freq,
            pos_start=pos,
            eps=RMS_EPS,
            attention_scaling=scaling,
        ),
    )


def make_cycle(name, data, layers):
    """``fn(*inputs) -> outputs`` for one verify cycle's worth of ``name``."""

    inv_freq = data["inv_freq"]
    scaling = data["scaling"]
    norm_weight = data["norm_weight"]
    family = FAMILY_OF[name]

    if family == "rope":
        body = {
            "rope_prediet": rope_layer_prediet,
            "rope_stock": rope_layer_stock,
            "rope_scoped": rope_layer_stock,
            "rope_fused": rope_layer_fused,
        }[name]
        scoped = name == "rope_scoped"

        def rope_cycle(*flat):
            def run():
                outs = []
                for i in range(layers):
                    q, k, pos = flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]
                    outs.extend(body(q, k, pos, inv_freq, scaling))
                return outs

            if scoped:
                with _model._rope_table_scope():
                    return run()
            return run()

        return rope_cycle

    body = prep_layer_stock if name == "prep_stock" else prep_layer_fused

    def prep_cycle(*flat):
        outs = []
        for i in range(layers):
            q_idx, pos = flat[2 * i], flat[2 * i + 1]
            outs.extend(body(q_idx, pos, norm_weight, inv_freq, scaling))
        return outs

    return prep_cycle


def cycle_inputs(family, data, layers):
    flat = []
    for i in range(layers):
        if family == "rope":
            flat.extend((data["q"][i], data["k"][i], data["pos"][i]))
        else:
            flat.extend((data["q_idx"][i], data["pos"][i]))
    return tuple(flat)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
FREE_PRIMITIVES = {
    "Reshape", "ExpandDims", "Squeeze", "Slice", "Transpose", "AsStrided",
}
EXTRA_LAUNCHES = {"DynamicSlice": 1, "DynamicSliceUpdate": 2}


def count_launches(outputs):
    """Approximate Metal dispatches for a built graph (upper bound).

    Same accounting as ``scripts/fable/micro_opdiet.py`` so the two benches'
    ``disp`` columns are comparable.
    """

    buffer = io.StringIO()
    mx.export_to_dot(buffer, *outputs)
    text = buffer.getvalue()
    labels = re.findall(r'label ="([^"]+)"', text)
    total = 0
    for label in labels:
        if label in FREE_PRIMITIVES:
            continue
        total += 1 + EXTRA_LAUNCHES.get(label, 0)
    for match in re.finditer(r'\{ (\d+) \[label ="Concatenate"', text):
        node = match.group(1)
        total += max(0, len(re.findall(rf'-> {node}\b', text)) - 1)
    return total


def time_cycle(fn, inputs, reps, warmup, clear_cache):
    """QUEUED lane: build the whole cycle, then ONE ``mx.eval``.

    ``eval_ms`` is encode + GPU for a verify cycle's worth of the arm -- the
    number the verdict rests on.  Evaluating each kernel on its own would add
    a host round trip per dispatch, which at these sizes is larger than the
    kernels and has inverted a verdict before.
    """

    for _ in range(warmup):
        mx.eval(fn(*inputs))
    evals, builds = [], []
    for _ in range(reps):
        if clear_cache:
            mx.clear_cache()
        t0 = time.perf_counter()
        out = fn(*inputs)
        t1 = time.perf_counter()
        mx.eval(out)
        evals.append((time.perf_counter() - t1) * 1e3)
        builds.append((t1 - t0) * 1e3)
    evals.sort()
    return {
        "median_ms": statistics.median(evals),
        "p10_ms": evals[max(0, int(0.10 * (len(evals) - 1)))],
        "p90_ms": evals[min(len(evals) - 1, int(0.90 * (len(evals) - 1)))],
        "build_ms": statistics.median(builds),
    }


def compare(got, ref):
    """``(max_abs_diff, differing_elements, total_elements)``.

    A differing-element COUNT, not just a max: one flipped bf16 ulp on a
    query element can move a top-k tie, and a max-abs column alone hides how
    many elements moved.

    Both failure modes below are HARNESS bugs, not numerical results, so they
    raise rather than returning a number that would be read as parity:

    * ``ref is None`` -- a candidate was evaluated before the stock arm.  The
      first spelling of this bench iterated ``FAMILIES[family]`` in table
      order, whose first rope entry is ``rope_prediet``, not the stock arm, so
      ``zip(got, None)`` died at the first comparison.  ``arm_order`` now puts
      the reference first; this is the belt to that braces.
    * a length mismatch -- ``zip`` truncates silently, so an arm that emitted
      fewer outputs than the reference would score parity on the prefix it did
      emit and report a smaller ``elements`` count that nothing checks.
    """

    if ref is None:
        raise RuntimeError(
            "compare() has no reference: the stock arm must be evaluated "
            "before any candidate (see arm_order). This is a harness "
            "ordering bug, not a numerical verdict."
        )
    if len(got) != len(ref):
        raise RuntimeError(
            f"compare() got {len(got)} outputs against {len(ref)} reference "
            "outputs; an arm that emits a different number of tensors is not "
            "the same computation and cannot be scored for parity."
        )
    worst = 0.0
    differing = 0
    total = 0
    for a, b in zip(got, ref):
        af = a.astype(mx.float32)
        bf = b.astype(mx.float32)
        worst = max(worst, float(mx.max(mx.abs(af - bf)).item()))
        differing += int(mx.sum(af != bf).item())
        total += int(a.size)
    return worst, differing, total


def arm_order(family):
    """Execution order for ``family``: the STOCK arm first, then the rest.

    ``FAMILIES`` is the PRINT order (pre-diet, stock, scoped, fused reads as a
    progression).  Execution has to start at the reference every other arm is
    compared against, and the two orders are not the same for ``rope``.
    """

    stock = STOCK[family]
    if stock not in FAMILIES[family]:
        raise RuntimeError(
            f"family {family!r} has no stock arm {stock!r} in {FAMILIES[family]}"
        )
    return (stock, *(name for name in FAMILIES[family] if name != stock))


def build_data(args, layers):
    dtype = mx.bfloat16
    inv_freq = 1.0 / (
        float(args.rope_theta)
        ** (mx.arange(0, ROTARY_DIM, 2, dtype=mx.float32) / ROTARY_DIM)
    )
    data = {
        # ONE inv_freq object: the op diet's _rope_inv_freq_and_scaling_shared
        # really does share it across the indexer and every layer.
        "inv_freq": inv_freq,
        "scaling": float(args.attention_scaling),
        "norm_weight": mx.random.normal((IDX_HEAD_DIM,)).astype(dtype),
        "q": [
            mx.random.normal((1, ROWS, N_HEADS, HEAD_DIM)).astype(dtype)
            for _ in range(layers)
        ],
        "k": [
            mx.random.normal((1, ROWS, N_KV_HEADS, HEAD_DIM)).astype(dtype)
            for _ in range(layers)
        ],
        "q_idx": [
            mx.random.normal((1, ROWS, IDX_HEADS, IDX_HEAD_DIM)).astype(dtype)
            for _ in range(layers)
        ],
        # One position leaf per layer: distinct objects, same value, so
        # mx.compile's CSE cannot collapse twelve table builds into one.
        "pos": [
            mx.array([args.pos_start], dtype=mx.int32) for _ in range(layers)
        ],
    }
    flat = [inv_freq, data["norm_weight"]]
    for key in ("q", "k", "q_idx", "pos"):
        flat.extend(data[key])
    mx.eval(*flat)
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--families", type=str, default="rope,prep",
        help="comma list from: rope, prep",
    )
    p.add_argument(
        "--layers", type=int, default=QSA_LAYERS,
        help="QSA layers per cycle (12 = one verify cycle; 48 amplifies)",
    )
    p.add_argument("--pos-start", type=int, default=POS_START)
    p.add_argument("--attention-scaling", type=float, default=1.0)
    p.add_argument("--rope-theta", type=float, default=ROPE_THETA)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lanes", type=str, default="eager,compiled")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--out", type=str, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    lanes = [l.strip() for l in args.lanes.split(",") if l.strip()]
    for family in families:
        if family not in FAMILIES:
            raise SystemExit(
                f"unknown family {family!r}; expected {list(FAMILIES)}"
            )
    layers = int(args.layers)
    if layers < 1:
        raise SystemExit("--layers must be positive")

    print(
        "[micro-verify-glue-a] must run under /tmp/mtplx-gpu-exclusive.lock",
        flush=True,
    )
    _require_mlx()
    mx.random.seed(args.seed)
    print(
        f"[build] layers={layers} rows={ROWS} "
        f"q=[1,{ROWS},{N_HEADS},{HEAD_DIM}] k=[1,{ROWS},{N_KV_HEADS},{HEAD_DIM}] "
        f"q_idx=[1,{ROWS},{IDX_HEADS},{IDX_HEAD_DIM}] rot={ROTARY_DIM} "
        f"pos_start={args.pos_start} scaling={args.attention_scaling}",
        flush=True,
    )

    data = build_data(args, layers)
    results: dict[str, dict] = {}
    numerics: dict[str, dict] = {}

    for family in families:
        inputs = cycle_inputs(family, data, layers)
        for lane in lanes:
            # Stock FIRST: it is the reference. Printing still follows
            # FAMILIES[family]; only execution is reordered.
            ref_out = None
            for name in arm_order(family):
                eager = make_cycle(name, data, layers)
                fn = eager if lane == "eager" else mx.compile(eager)
                # Count on a FRESHLY BUILT graph: export_to_dot walks the
                # unevaluated tape, and an evaluated array is a leaf.
                launches = count_launches(fn(*inputs))
                out = fn(*inputs)
                mx.eval(out)
                if name == STOCK[family]:
                    ref_out = out
                else:
                    worst, differing, total = compare(out, ref_out)
                    numerics[f"{name}/{lane}"] = {
                        "max_abs_diff": worst,
                        "differing": differing,
                        "elements": total,
                    }
                stats = time_cycle(
                    fn, inputs, args.reps, args.warmup, args.clear_cache
                )
                stats["launches"] = launches
                stats["per_layer_us"] = stats["median_ms"] * 1e3 / layers
                stats["dispatches_per_layer"] = launches / layers
                stats["family"] = family
                stats["lane"] = lane
                results[f"{name}/{lane}"] = stats

    for lane in lanes:
        for family in families:
            base = results.get(f"{STOCK[family]}/{lane}")
            if base is None:
                continue
            for name in FAMILIES[family]:
                r = results.get(f"{name}/{lane}")
                if r is not None:
                    r["delta_pct_vs_stock"] = (
                        (r["median_ms"] - base["median_ms"])
                        / base["median_ms"] * 100.0
                    )
                    r["delta_ms_vs_stock"] = r["median_ms"] - base["median_ms"]

    hdr = (
        f"{'variant':<15}{'lane':<10}{'eval ms':>10}{'p10':>9}{'p90':>9}"
        f"{'us/layer':>10}{'delta%':>9}{'delta ms':>10}{'disp':>7}{'d/layer':>9}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for lane in lanes:
        for family in families:
            for name in FAMILIES[family]:
                r = results.get(f"{name}/{lane}")
                if r is None:
                    continue
                print(
                    f"{name:<15}{lane:<10}{r['median_ms']:>10.3f}"
                    f"{r['p10_ms']:>9.3f}{r['p90_ms']:>9.3f}"
                    f"{r['per_layer_us']:>10.1f}"
                    f"{r.get('delta_pct_vs_stock', 0.0):>+9.2f}"
                    f"{r.get('delta_ms_vs_stock', 0.0):>+10.3f}"
                    f"{r['launches']:>7d}{r['dispatches_per_layer']:>9.2f}"
                )
        print()

    print("parity vs the live stock spelling (0 differing = bit-exact):")
    for name, stats in sorted(numerics.items()):
        print(
            f"  {name:<24} differing={stats['differing']:>8d} / "
            f"{stats['elements']:<9d} max_abs={stats['max_abs_diff']:.6g}"
        )
    print(
        "\nThe COMPILED lane decides both columns: the production verify body "
        "is one mx.compile'd graph."
    )

    summary = {
        "shapes": {
            "rows": ROWS, "layers": layers, "n_heads": N_HEADS,
            "n_kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM,
            "rotary_dim": ROTARY_DIM, "idx_heads": IDX_HEADS,
            "idx_head_dim": IDX_HEAD_DIM,
        },
        "pos_start": int(args.pos_start),
        "attention_scaling": float(args.attention_scaling),
        "reps": args.reps, "seed": args.seed,
        "variants": results, "numerics": numerics,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {args.out}")
    else:
        print("\n" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
