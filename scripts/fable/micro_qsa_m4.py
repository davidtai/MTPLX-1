#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price each MTPLX_FABLE_QSA_M4 rewrite at the fixed-M4 verifier's shapes.

Twelve QSA layers issue about half of the compiled verify graph's dispatches,
and the QSA block is a dependent chain, not a byte problem.  This bench prices
the five sub-chains the lane touches, each over ONE verify cycle's worth of
calls (12 QSA layers), separately -- because dispatch count is not GPU time
and a fused kernel can lose (see scripts/fable/micro_opdiet.py, where the
fewest-dispatch bank spelling was the slowest of three).

Families (stock spelling first; the stock spellings are the production code
paths, imported, not copied, so they cannot drift):

  prep    q_layernorm + partial RoPE on the indexer queries [1,4,4,128]
          prep_stock   QSAIndexer._prepare_queries_eager (op diet armed)
          prep_m4      kernels.qsa_indexer_prepare.qsa_indexer_prepare_queries_metal
                       -- the SHIPPED kernel, bit-exact per
                       tests/test_qsa_indexer_prepare_metal.py; the fixed lane
                       simply never called it

  bank    the fixed pooled bank's newly completed row, bank [1,4352,128] bf16
          bank_stock   QSAIndexer._extend_pooled_fixed with the op diet's
                       ``bank`` + ``rope`` items armed (what ships today)
          bank_m4      qsa_m4_pooled_row + the same mx.slice_update

  score   the scoring epilogue after the [1,4,4,4352] fp32 score GEMM
          score_stock  maximum -> sum(axis=2) -> /sqrt(128) -> where(valid)
                       -> tie-break
          score_m4     qsa_m4_index_scores

  tokens  the rows-gather token list, top_idx [4,512] -> [4,2052]
          tokens_stock take_along_axis + repeat + 2 concatenates + where
          tokens_m4    qsa_m4_row_tokens

  gather  the selected K/V read + the score operand's layout
          gather_stock fused K/V gather (both [1,2,4,2052,256]) then
                       k.swapaxes(-1,-2).reshape(...) for the score GEMM
          gather_kt    the same gather emitting K as [1,2,4,256,2052], then
                       mx.expand_dims -- the transpose happens at the source

Both lanes are timed: eager, and wrapped in ``mx.compile``.  **The compiled
number is the one that matters** -- the production verify step is one
mx.compile'd graph, and MLX only fuses elementwise chains under compile, so
the eager lane systematically flatters the stock spellings.

Numerics: every family prints max-abs-diff against its stock spelling AND the
count of differing elements.  Four of the five must print 0/0.  ``score`` is
the one place where a nonzero count would be informative rather than a bug:
its 4-term fp32 head sum assumes MLX's column reduce walks the axis in order
(see kernels/qwen4_qsa_m4_indexer.py).  Treat a nonzero ``score`` count as a
finding, not a rounding allowance.

Per-layer input leaves are DISTINCT on purpose: mx.compile runs a
common-subexpression pass, and 12 layers fed identical arrays would collapse
into one.  Every bank is an argument of the compiled function and is returned,
so no bank is donatable -- the production regime, where the verifier still
holds every pooled leaf.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import statistics
import time
from pathlib import Path

# --- Qwen3.8-Flash-Next fixed-M4 QSA geometry (TextArgs defaults) ----------
ROWS = 4                    # fixed-M4 verify width
QSA_LAYERS = 12             # every 4th of 48
IDX_HEADS = 4
IDX_HEAD_DIM = 128
IDX_ROTARY_DIM = 64
HEAD_DIM = 256
N_HEADS = 24
N_KV_HEADS = 2
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
POOLED_BLOCKS = 4352        # census window (16k prompt + 1k out)
SELECTED = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO  # 2052
RMS_EPS = 1e-6
ROPE_THETA = 10_000_000.0

FAMILIES = {
    "prep": ("prep_stock", "prep_m4"),
    "bank": ("bank_stock", "bank_m4"),
    "score": ("score_stock", "score_m4"),
    "tokens": ("tokens_stock", "tokens_m4"),
    "gather": ("gather_stock", "gather_kt"),
}
STOCK = {f: v[0] for f, v in FAMILIES.items()}
FAMILY_OF = {v: f for f, vs in FAMILIES.items() for v in vs}

mx = None
_mods: dict = {}


def _require_mlx() -> None:
    """Import MLX and mtplx lazily: everything below runs on the GPU."""

    global mx
    import mlx.core

    mx = mlx.core
    import mtplx.runtime_options as runtime_options
    import mtplx.models.qwen4_exp as qwen4_exp
    from mtplx.kernels import (
        qsa_indexer_prepare,
        qwen4_qsa_m4_fused_kv_gather,
        qwen4_qsa_m4_indexer,
    )

    _mods.update(
        runtime_options=runtime_options,
        qwen4_exp=qwen4_exp,
        prepare=qsa_indexer_prepare,
        m4=qwen4_qsa_m4_indexer,
        gather=qwen4_qsa_m4_fused_kv_gather,
    )


# --------------------------------------------------------------------------
# dispatch counting (same approximation as scripts/fable/micro_opdiet.py)
# --------------------------------------------------------------------------
FREE_PRIMITIVES = {
    "Reshape", "ExpandDims", "Squeeze", "Slice", "Transpose", "AsStrided",
    "Broadcast", "StopGradient",
}
#: Metal launches a primitive does not account for on its own: Concatenate
#: copies once PER INPUT; a dynamic slice/update also computes its offset, and
#: a non-donatable slice_update copies the destination first; ArgPartition is
#: MLX's 5-dispatch multi-block partition at this width.
EXTRA_LAUNCHES = {"DynamicSlice": 1, "DynamicSliceUpdate": 2, "ArgPartition": 4}


def count_launches(outputs):
    """Approximate Metal dispatches for a built graph (upper bound)."""

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    buffer = io.StringIO()
    mx.export_to_dot(buffer, *outputs)
    text = buffer.getvalue()
    total = 0
    for label in re.findall(r'label ="([^"]+)"', text):
        if label in FREE_PRIMITIVES:
            continue
        total += 1 + EXTRA_LAUNCHES.get(label, 0)
    for match in re.finditer(r'\{ (\d+) \[label ="Concatenate"', text):
        node = match.group(1)
        total += max(0, len(re.findall(rf'-> {node}\b', text)) - 1)
    return total


# --------------------------------------------------------------------------
# a minimal stand-in for the installed fixed-capacity QSA cache
# --------------------------------------------------------------------------
class _FixedCache:
    """Just the surface ``_extend_pooled_fixed`` reads, nothing more."""

    fixed_capacity = True

    def __init__(self, raw_keys, pooled, offset, ratio, rows):
        self.raw_keys = raw_keys
        self.pooled = pooled
        self.offset = offset
        self.ratio = ratio
        self._last_write_rows = rows


# --------------------------------------------------------------------------
# family bodies -- one call per QSA layer, per verify cycle
# --------------------------------------------------------------------------
def make_cycle(name, data, indexer):
    family = FAMILY_OF[name]
    m4 = _mods["m4"]
    prepare = _mods["prepare"]

    if family == "prep":
        def prep_cycle(*flat):
            q = flat[:QSA_LAYERS]
            pos = flat[QSA_LAYERS:]
            outs = []
            for i in range(QSA_LAYERS):
                if name == "prep_stock":
                    outs.append(indexer._prepare_queries_eager(q[i], pos[i]))
                else:
                    outs.append(
                        prepare.qsa_indexer_prepare_queries_metal(
                            q[i],
                            indexer.q_layernorm.weight,
                            indexer._inv_freq,
                            pos_start=pos[i],
                            eps=RMS_EPS,
                            attention_scaling=indexer._rope_attention_scaling,
                        )
                    )
            return outs

        return prep_cycle

    if family == "bank":
        def bank_cycle(*flat):
            banks = flat[0:QSA_LAYERS]
            raws = flat[QSA_LAYERS:2 * QSA_LAYERS]
            offs = flat[2 * QSA_LAYERS:]
            outs = []
            for i in range(QSA_LAYERS):
                cache = _FixedCache(raws[i], banks[i], offs[i],
                                    COMPRESS_RATIO, ROWS)
                outs.append(
                    indexer._extend_pooled_fixed(
                        cache,
                        offs[i] + ROWS,
                        fused_m4=(name == "bank_m4"),
                    )
                )
            return outs

        return bank_cycle

    if family == "score":
        blk = data["blk"]

        def score_stock(scores, off):
            qpos = off + mx.arange(ROWS, dtype=mx.int32)
            nb_q = (qpos + 1) // COMPRESS_RATIO
            valid = blk[None, :] < nb_q[:, None]
            neg = mx.array(-mx.inf, dtype=mx.float32)
            s = mx.maximum(scores, 0.0).sum(axis=2) / math.sqrt(IDX_HEAD_DIM)
            s = s[0]
            s = mx.where(valid, s, neg)
            return s - blk.astype(mx.float32)[None, :] * 1e-12

        def score_cycle(*flat):
            scores = flat[0:QSA_LAYERS]
            offs = flat[QSA_LAYERS:]
            outs = []
            for i in range(QSA_LAYERS):
                if name == "score_stock":
                    outs.append(score_stock(scores[i], offs[i]))
                else:
                    outs.append(
                        m4.qsa_m4_index_scores(
                            scores[i],
                            offs[i],
                            compress_ratio=COMPRESS_RATIO,
                            head_dim=IDX_HEAD_DIM,
                        )
                    )
            return outs

        return score_cycle

    if family == "tokens":
        blk = data["blk"]

        def tokens_stock(top_idx, off):
            qpos = off + mx.arange(ROWS, dtype=mx.int32)
            nb_q = (qpos + 1) // COMPRESS_RATIO
            valid = blk[None, :] < nb_q[:, None]
            blk_ok = mx.take_along_axis(valid, top_idx.astype(mx.int64), axis=-1)
            tok_blocks = (
                top_idx.astype(mx.int32)[:, :, None] * COMPRESS_RATIO
                + mx.arange(COMPRESS_RATIO, dtype=mx.int32)
            ).reshape(ROWS, -1)
            blocks_ok = mx.repeat(blk_ok, COMPRESS_RATIO, axis=1)
            tail_tok = nb_q[:, None] * COMPRESS_RATIO + mx.arange(
                COMPRESS_RATIO, dtype=mx.int32
            )
            tail_ok = tail_tok <= qpos[:, None]
            token_idx = mx.concatenate([tok_blocks, tail_tok], axis=1)
            token_ok = mx.concatenate([blocks_ok, tail_ok], axis=1)
            token_idx = mx.where(token_ok, token_idx, mx.array(0, dtype=mx.int32))
            return token_idx, token_ok

        def tokens_cycle(*flat):
            tops = flat[0:QSA_LAYERS]
            offs = flat[QSA_LAYERS:]
            outs = []
            for i in range(QSA_LAYERS):
                if name == "tokens_stock":
                    a, b = tokens_stock(tops[i], offs[i])
                else:
                    a, b = m4.qsa_m4_row_tokens(
                        tops[i], offs[i], compress_ratio=COMPRESS_RATIO
                    )
                outs.extend((a, b.astype(mx.int32)))
            return outs

        return tokens_cycle

    stock_gather = data["gather_stock"]
    kt_gather = data["gather_kt"]
    flat_q = data["attn_q"]

    def gather_cycle(*flat):
        keys = flat[0:QSA_LAYERS]
        values = flat[QSA_LAYERS:2 * QSA_LAYERS]
        toks = flat[2 * QSA_LAYERS:]
        outs = []
        for i in range(QSA_LAYERS):
            if name == "gather_stock":
                k_sel, v_sel = stock_gather(keys[i], values[i], toks[i])
                k_view = k_sel.swapaxes(-1, -2).reshape(
                    1, N_KV_HEADS, 1, ROWS, HEAD_DIM, SELECTED
                )
            else:
                k_sel, v_sel = kt_gather(keys[i], values[i], toks[i])
                k_view = mx.expand_dims(k_sel, 2)
            # Consume the transposed operand exactly as attention does, so a
            # layout that only LOOKS free is charged for its copy.
            q = flat_q[i]
            scores = mx.matmul(q, k_view).squeeze(-2)
            outs.extend((scores, v_sel))
        return outs

    return gather_cycle


def cycle_inputs(family, data):
    if family == "prep":
        return (*data["q_idx"], *data["offs"])
    if family == "bank":
        return (*data["banks"], *data["raws"], *data["offs"])
    if family == "score":
        return (*data["scores"], *data["offs"])
    if family == "tokens":
        return (*data["top_idx"], *data["offs"])
    return (*data["keys"], *data["values"], *data["token_idx"])


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def time_cycle(fn, inputs, reps, warmup, clear_cache):
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


def numerics(got, ref):
    """(max abs diff, differing element count) over every paired output."""

    worst, differing = 0.0, 0
    for a, b in zip(got, ref):
        if a.shape != b.shape:
            return float("inf"), -1
        af = a.astype(mx.float32)
        bf = b.astype(mx.float32)
        # -inf - -inf is nan; compare those slots as equal-if-both-inf.
        both_inf = mx.isinf(af) & mx.isinf(bf) & (mx.sign(af) == mx.sign(bf))
        diff = mx.where(both_inf, mx.zeros_like(af), mx.abs(af - bf))
        worst = max(worst, float(mx.max(diff).item()))
        differing += int(mx.sum(diff != 0).item())
    return worst, differing


def build_data(args, indexer):
    dtype = mx.bfloat16
    blocks = args.pooled_blocks
    raw_capacity = blocks * COMPRESS_RATIO
    base = args.pos_start
    offs = [
        mx.array(base + 4 * i, dtype=mx.int32) for i in range(QSA_LAYERS)
    ]
    gather_mod = _mods["gather"]
    data = {
        "blk": mx.arange(blocks, dtype=mx.int32),
        "offs": offs,
        "q_idx": [
            mx.random.normal((1, ROWS, IDX_HEADS, IDX_HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "banks": [
            mx.random.normal((1, blocks, IDX_HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "raws": [
            mx.random.normal((1, raw_capacity, IDX_HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        # The score GEMM's raw fp32 output. Centred on zero so the relu masks
        # about half the entries and exact-zero ties are common -- the regime
        # the tie-break exists for.
        "scores": [
            mx.random.normal((1, ROWS, IDX_HEADS, blocks)).astype(mx.float32)
            for _ in range(QSA_LAYERS)
        ],
        "top_idx": [
            (mx.random.uniform(shape=(ROWS, BLOCK_TOPK)) * blocks)
            .astype(mx.uint32)
            for _ in range(QSA_LAYERS)
        ],
        "keys": [
            mx.random.normal((1, N_KV_HEADS, raw_capacity, HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "values": [
            mx.random.normal((1, N_KV_HEADS, raw_capacity, HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "token_idx": [
            (mx.random.uniform(shape=(ROWS, SELECTED)) * raw_capacity)
            .astype(mx.int32)
            for _ in range(QSA_LAYERS)
        ],
        "attn_q": [
            mx.random.normal(
                (1, N_KV_HEADS, N_HEADS // N_KV_HEADS, ROWS, 1, HEAD_DIM)
            ).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "gather_stock": gather_mod.bind_qwen4_qsa_m4_fused_kv_gather(
            capacity=raw_capacity, transposed_keys=False
        ),
        "gather_kt": gather_mod.bind_qwen4_qsa_m4_fused_kv_gather(
            capacity=raw_capacity, transposed_keys=True
        ),
    }
    flat = []
    for value in data.values():
        if isinstance(value, list):
            flat.extend(value)
        elif isinstance(value, mx.array):
            flat.append(value)
    mx.eval(*flat, indexer.parameters(), indexer._inv_freq)
    return data


def build_indexer(args):
    qwen4_exp = _mods["qwen4_exp"]
    text_args = qwen4_exp.TextArgs(rope_theta=args.rope_theta)
    indexer = qwen4_exp.QSAIndexer(text_args)
    dtype = mx.bfloat16
    indexer.update(
        {
            "q_layernorm": {
                "weight": mx.random.uniform(
                    low=0.5, high=1.5, shape=(IDX_HEAD_DIM,)
                ).astype(dtype)
            },
            "k_layernorm": {
                "weight": mx.random.uniform(
                    low=0.5, high=1.5, shape=(IDX_HEAD_DIM,)
                ).astype(dtype)
            },
        }
    )
    return indexer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--families", type=str, default="prep,bank,score,tokens,gather",
        help="comma list from: " + ", ".join(FAMILIES),
    )
    p.add_argument("--pooled-blocks", type=int, default=POOLED_BLOCKS)
    p.add_argument("--pos-start", type=int, default=16_384)
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
            raise SystemExit(f"unknown family {family!r}; expected {list(FAMILIES)}")

    print("[micro-qsa-m4] must run under /tmp/mtplx-gpu-exclusive.lock", flush=True)
    _require_mlx()
    runtime_options = _mods["runtime_options"]
    # The stock spellings are the op-diet-armed production code; that is the
    # baseline this lane has to beat, not the pre-diet one.
    runtime_options._FABLE_OPDIET = True
    runtime_options._FABLE_OPDIET_SELECTED = frozenset(
        runtime_options.FABLE_OPDIET_ITEMS
    )
    mx.random.seed(args.seed)

    print(
        f"[build] rows={ROWS} layers={QSA_LAYERS} "
        f"bank=[1,{args.pooled_blocks},{IDX_HEAD_DIM}] bf16 "
        f"({args.pooled_blocks * IDX_HEAD_DIM * 2 / 1e6:.3f} MB)  "
        f"scores=[1,{ROWS},{IDX_HEADS},{args.pooled_blocks}] f32  "
        f"selected={SELECTED}  kv=[1,{N_KV_HEADS},"
        f"{args.pooled_blocks * COMPRESS_RATIO},{HEAD_DIM}] bf16  "
        f"pos_start={args.pos_start}",
        flush=True,
    )
    indexer = build_indexer(args)
    data = build_data(args, indexer)

    results: dict[str, dict] = {}
    diffs: dict[str, tuple[float, int]] = {}
    for family in families:
        inputs = cycle_inputs(family, data)
        refs = None
        for name in FAMILIES[family]:
            eager = make_cycle(name, data, indexer)
            for lane in lanes:
                fn = eager if lane == "eager" else mx.compile(eager)
                # Count on a FRESHLY BUILT graph: export_to_dot walks the
                # unevaluated tape, and an evaluated array is a leaf.
                launches = count_launches(fn(*inputs))
                out = fn(*inputs)
                mx.eval(out)
                if name == STOCK[family] and lane == lanes[0]:
                    refs = out
                elif name != STOCK[family]:
                    diffs[f"{name}/{lane}"] = numerics(out, refs)
                stats = time_cycle(fn, inputs, args.reps, args.warmup,
                                   args.clear_cache)
                stats["launches"] = launches
                stats["per_layer_us"] = stats["median_ms"] * 1e3 / QSA_LAYERS
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
                        (r["median_ms"] - base["median_ms"]) / base["median_ms"]
                        * 100.0
                    )

    hdr = (f"{'variant':<15}{'lane':<10}{'eval ms':>10}{'p10':>9}{'p90':>9}"
           f"{'us/layer':>10}{'delta%':>9}{'disp':>7}{'build':>9}")
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
                    f"{r['launches']:>7d}{r['build_ms']:>9.3f}"
                )
        print()

    print("numerics vs stock (max abs diff / differing elements):")
    for name, (worst, count) in sorted(diffs.items()):
        flag = "" if count == 0 else "   <-- NOT EXACT"
        print(f"  {name:<24} {worst:.6g}  {count}{flag}")

    summary = {
        "shapes": {
            "rows": ROWS, "qsa_layers": QSA_LAYERS,
            "pooled_blocks": args.pooled_blocks,
            "idx_head_dim": IDX_HEAD_DIM, "head_dim": HEAD_DIM,
            "selected": SELECTED, "block_topk": BLOCK_TOPK,
            "pos_start": args.pos_start,
        },
        "reps": args.reps, "seed": args.seed,
        "variants": results,
        "numerics": {k: {"max_abs_diff": v[0], "differing": v[1]}
                     for k, v in diffs.items()},
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {args.out}")
    else:
        print("\n" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
