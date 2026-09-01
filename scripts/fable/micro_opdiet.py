#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price each MTPLX_FABLE_OPDIET rewrite at the verifier's real shapes.

The op diet removes ~300 dispatches/cycle from the compiled fixed-M4 verify
graph, and every rewrite is bitwise exact -- yet armed together the four items
measured **-2.9% tok/s** against control on seed 20260829 (66.46 vs 68.42),
with verify_forward +0.4 s over 1,024 tokens (~+1.0 ms per verify window).
Draft and target-dist were unchanged and acceptance was identical, so the
cost is inside the compiled verify graph: one of ``bank`` / ``rope`` /
``resid`` costs more GPU time than the dispatches it removes.

Removing a dispatch is not free of consequence: it can replace CONTIGUOUS
vectorized kernels with BROADCAST/general ones, whose per-element index
arithmetic runs well below the copy engine's bandwidth. That is the leading
hypothesis for ``bank``, and this bench is built to confirm or kill it.

Variants, each timed as ONE verify cycle's worth of calls:

  bank (12 QSA layers, pooled bank [1, 4352, 128] bf16 = 1.11 MB)
    bank_stock   mx.slice_update(bank, row) then mx.where(cond, updated, bank)
                 -- two full-bank passes (~5.6 MB/layer), 4 dispatches, but
                 both passes are contiguous vectorized kernels
    bank_select  mx.where(row_id == blk & cond, row, bank) -- ONE full-bank
                 pass (~2.2 MB/layer), 2 dispatches, but the mask [1,4352,1]
                 and the row [1,1,128] both broadcast, so MLX must emit a
                 general (strided) select over 557,056 elements   [SHIPPED]
    bank_rowsel  dynamic-slice the old row, select on the ROW, slice_update --
                 one full-bank pass like bank_select, but the full-bank pass
                 is the contiguous slice_update copy and the broadcast work
                 is 128 elements wide. 6 dispatches. The fix, if the
                 hypothesis holds.

  rope (12 QSA layers: indexer q [1,4,4,128], attn q [1,4,24,256],
        attn k [1,4,2,256], pooled block [1,1,1,128]; rot 64, half 32)
    rope_stock   3 full-width cos/sin tables (cos over concat([a, a])) and 4
                 x _apply_partial_rope (standalone negate + 2 concatenates)
    rope_half    1 SHARED + 1 unshared half-width table and 4 x split-half
                 rotation (negate folds into the fused kernel, one 3-way
                 concatenate)                                     [SHIPPED]

  resid (96 sites, hyper [1,4,10240], block_out [1,4,2560], inject [1,4,4])
    resid_stock  hyper + (out[...,None,:] * inj[...,:,None]).reshape(...)
                 -- the reshape splits the chain, so multiply and add are two
                 kernels
    resid_fused  add on the [..,4,2560] VIEW of hyper, so mx.compile fuses
                 the broadcast multiply and the add into one kernel [SHIPPED]

Both lanes are timed: eager, and wrapped in ``mx.compile``. **The compiled
number is the one that matters** -- the production verify step is one
mx.compile'd graph, and MLX only fuses elementwise chains under compile, so
the eager lane systematically flatters the stock spellings (it charges them
for fusions the real path already has).

Realism note for ``bank``: in the real graph the pooled bank is a compiled
state LEAF that the verifier bank still holds, so ``mx.slice_update`` cannot
donate its input and pays a full-bank copy. This bench reproduces that by
passing all 12 banks in as arguments and returning all 12, so no bank is ever
donatable. ``--donatable-bank`` shows the other regime (a bank that is a
loop-local temporary), where slice_update donates and the stock spelling gets
much cheaper -- that is the regime to avoid drawing conclusions from.

Standalone by construction: imports no mtplx. The variant bodies are literal
copies of the two spellings in mtplx/models/qwen4_exp.py;
tests/test_fable_opdiet.py pins them to the production helpers so the copies
cannot drift.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import time
from pathlib import Path

# MLX is imported lazily by ``main`` so the CLI surface and the shape table can
# be unit tested on a box that is not holding the GPU lock -- importing MLX is
# the first step toward touching the device, and every code path below that
# uses ``mx`` runs only inside the guarded window.
mx = None


def _require_mlx() -> None:
    global mx
    import mlx.core

    mx = mlx.core


# --- Qwen3.8-Flash-Next verifier geometry (TextArgs defaults) --------------
ROWS = 4               # fixed-M4 verify width
HIDDEN = 2560
HC_COUNT = 4
HC_HIDDEN = HC_COUNT * HIDDEN          # 10240
N_HEADS = 24
N_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64                        # head_dim * partial_rotary_factor
ROT_HALF = ROTARY_DIM // 2             # 32 = inv_freq.size
IDX_HEADS = 4
IDX_HEAD_DIM = 128
COMPRESS_RATIO = 4
QSA_LAYERS = 12                        # every 4th of 48
RESID_SITES = 96                       # 48 layers x (attn half + mlp half)
POOLED_BLOCKS = 4352                   # census window (16k prompt + 1k out)

FAMILIES = {
    "bank": ("bank_stock", "bank_select", "bank_rowsel"),
    "rope": ("rope_stock", "rope_half"),
    "resid": ("resid_stock", "resid_fused"),
}
STOCK = {"bank": "bank_stock", "rope": "rope_stock", "resid": "resid_stock"}
FAMILY_OF = {v: k for k, vs in FAMILIES.items() for v in vs}
UNITS = {"bank": QSA_LAYERS, "rope": QSA_LAYERS, "resid": RESID_SITES}
UNIT_NAME = {"bank": "QSA layer", "rope": "QSA layer", "resid": "site"}


# --------------------------------------------------------------------------
# item 1 -- QSA fixed pooled-bank conditional write
# --------------------------------------------------------------------------


def bank_stock(bank, row, blk, cond):
    """mtplx/models/qwen4_exp.py _extend_pooled_fixed, pre-diet spelling."""

    updated = mx.slice_update(bank, row, blk, axes=(1,))
    return mx.where(cond, updated, bank)


def bank_select(bank, row, blk, cond, row_ids):
    """The shipped diet: one broadcast select over the whole bank."""

    write_row = mx.logical_and(row_ids == blk, cond)
    return mx.where(write_row[..., None], row.astype(bank.dtype), bank)


def bank_rowsel(bank, row, blk, cond, row_ids):
    """Candidate fix: select on the ROW, then one contiguous slice_update.

    Same single full-bank pass as ``bank_select`` (the slice_update copy),
    but that pass is a flat vector copy instead of a broadcast select, and
    the conditional work is 128 elements wide instead of 557,056.
    """

    del row_ids
    old = mx.slice(bank, blk, axes=(1,), slice_size=(1, 1, bank.shape[2]))
    merged = mx.where(cond, row.astype(bank.dtype), old)
    return mx.slice_update(bank, merged, blk, axes=(1,))


# --------------------------------------------------------------------------
# item 2 -- RoPE tables and the partial rotation
# --------------------------------------------------------------------------


def rope_cos_sin(positions, inv_freq, scaling=1.0):
    angles = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([angles, angles], axis=-1)
    cosine, sine = mx.cos(emb), mx.sin(emb)
    if scaling != 1.0:
        cosine, sine = cosine * float(scaling), sine * float(scaling)
    return cosine, sine


def apply_partial_rope(x, cos, sin):
    rot = cos.shape[-1]
    x_rope, x_pass = x[..., :rot], x[..., rot:]
    half = rot // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    x_rope = (
        x_rope.astype(mx.float32) * cos + rotated.astype(mx.float32) * sin
    ).astype(x.dtype)
    return mx.concatenate([x_rope, x_pass], axis=-1)


def rope_cos_sin_half(positions, inv_freq, scaling=1.0):
    angles = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    cosine, sine = mx.cos(angles), mx.sin(angles)
    if scaling != 1.0:
        cosine, sine = cosine * float(scaling), sine * float(scaling)
    return cosine, sine


def apply_partial_rope_half(x, cos_h, sin_h):
    half = cos_h.shape[-1]
    rot = 2 * half
    x_rope, x_pass = x[..., :rot], x[..., rot:]
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    cos_h, sin_h = cos_h[:, None, :], sin_h[:, None, :]
    lo = (x1.astype(mx.float32) * cos_h + (-x2).astype(mx.float32) * sin_h).astype(
        x.dtype
    )
    hi = (x2.astype(mx.float32) * cos_h + x1.astype(mx.float32) * sin_h).astype(
        x.dtype
    )
    if x_pass.shape[-1] == 0:
        return mx.concatenate([lo, hi], axis=-1)
    return mx.concatenate([lo, hi, x_pass], axis=-1)


def rope_layer_stock(q_idx, q, k, pooled_row, pos, blk_pos, freqs):
    """One QSA layer: indexer table + attention table + pooled-block table.

    ``freqs`` is (indexer inv_freq, attention inv_freq) as SEPARATE arrays.
    That is the pre-diet reality -- QSAIndexer and Attention each build their
    own from the same TextArgs -- and it matters here: mx.compile runs a
    common-subexpression pass, so feeding both calls one shared array would
    silently give the stock spelling the diet's table sharing for free and
    make this comparison meaningless.
    """

    inv_idx, inv_attn = freqs
    c1, s1 = rope_cos_sin(pos, inv_idx)
    out_idx = apply_partial_rope(q_idx, c1, s1)
    c2, s2 = rope_cos_sin(pos, inv_attn)
    out_q = apply_partial_rope(q, c2, s2)
    out_k = apply_partial_rope(k, c2, s2)
    c3, s3 = rope_cos_sin(blk_pos, inv_idx)
    out_pool = apply_partial_rope(pooled_row, c3, s3)
    return out_idx, out_q, out_k, out_pool


def rope_layer_half(q_idx, q, k, pooled_row, pos, blk_pos, freqs):
    """Same layer, shipped diet: ONE half table serves all three query/key
    consumers (the diet also makes the two inv_freq arrays one object)."""

    inv_idx, _ = freqs
    c1, s1 = rope_cos_sin_half(pos, inv_idx)
    out_idx = apply_partial_rope_half(q_idx, c1, s1)
    out_q = apply_partial_rope_half(q, c1, s1)
    out_k = apply_partial_rope_half(k, c1, s1)
    c3, s3 = rope_cos_sin_half(blk_pos, inv_idx)
    out_pool = apply_partial_rope_half(pooled_row, c3, s3)
    return out_idx, out_q, out_k, out_pool


# --------------------------------------------------------------------------
# item 4 -- hyper-connection residual write
# --------------------------------------------------------------------------


def resid_stock(hyper, block_out, inject):
    return hyper + (block_out[..., None, :] * inject[..., :, None]).reshape(
        *hyper.shape
    )


def resid_fused(hyper, block_out, inject):
    grouped = hyper.reshape(*hyper.shape[:-1], inject.shape[-1], block_out.shape[-1])
    return (grouped + block_out[..., None, :] * inject[..., :, None]).reshape(
        *hyper.shape
    )


# --------------------------------------------------------------------------
# cycle bodies -- one call per production site, per verify cycle
# --------------------------------------------------------------------------


def make_cycle(name, data, *, donatable_bank=False):
    """Return ``fn(*inputs) -> outputs`` for one cycle's worth of ``name``.

    Every LAYER gets its own input leaves. That is not cosmetic: mx.compile
    runs a common-subexpression pass, so 12 layers fed identical arrays would
    collapse into one and the measurement would be of a single layer. In the
    real graph each QSA layer carries its own cache-offset and bank leaves,
    which is what these distinct inputs reproduce.
    """

    family = FAMILY_OF[name]

    if family == "bank":
        row_ids = data["row_ids"]

        def one(bank, row, blk, cond):
            if name == "bank_stock":
                return bank_stock(bank, row, blk, cond)
            if name == "bank_select":
                return bank_select(bank, row, blk, cond, row_ids)
            return bank_rowsel(bank, row, blk, cond, row_ids)

        def bank_cycle(*flat):
            banks = flat[0:QSA_LAYERS]
            rows = flat[QSA_LAYERS:2 * QSA_LAYERS]
            blks = flat[2 * QSA_LAYERS:3 * QSA_LAYERS]
            conds = flat[3 * QSA_LAYERS:4 * QSA_LAYERS]
            if donatable_bank:
                # One bank threaded through 12 updates: every intermediate is
                # a temporary, so slice_update can donate. NOT production.
                bank = banks[0]
                for i in range(QSA_LAYERS):
                    bank = one(bank, rows[i], blks[i], conds[i])
                return [bank]
            return [
                one(banks[i], rows[i], blks[i], conds[i])
                for i in range(QSA_LAYERS)
            ]

        return bank_cycle

    if family == "rope":
        freqs = (data["inv_freq_idx"], data["inv_freq_attn"])
        if name == "rope_half":
            layer, freqs = rope_layer_half, (data["inv_freq_idx"],) * 2
        else:
            layer = rope_layer_stock

        def rope_cycle(*flat):
            outs = []
            for i in range(QSA_LAYERS):
                base = i * 6
                outs.extend(layer(*flat[base:base + 6], freqs))
            return outs

        return rope_cycle

    write = resid_stock if name == "resid_stock" else resid_fused

    def resid_cycle(*flat):
        hypers = flat[0:RESID_SITES]
        block_outs = flat[RESID_SITES:2 * RESID_SITES]
        injects = flat[2 * RESID_SITES:]
        return [
            write(hypers[i], block_outs[i], injects[i])
            for i in range(RESID_SITES)
        ]

    return resid_cycle


def cycle_inputs(family, data):
    if family == "bank":
        return (*data["banks"], *data["rows"], *data["blks"], *data["conds"])
    if family == "rope":
        flat = []
        for i in range(QSA_LAYERS):
            flat.extend(
                (
                    data["q_idx"][i],
                    data["q"][i],
                    data["k"][i],
                    data["pooled_row"][i],
                    data["pos"][i],
                    data["blk_pos"][i],
                )
            )
        return tuple(flat)
    return (*data["hyper"], *data["block_out"], *data["inject"])


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


FREE_PRIMITIVES = {
    "Reshape", "ExpandDims", "Squeeze", "Slice", "Transpose", "AsStrided",
}
#: Metal launches a primitive does not account for on its own:
#: Concatenate copies once PER INPUT; a dynamic slice/update also computes its
#: offset, and a non-donatable slice_update copies the destination first.
EXTRA_LAUNCHES = {"DynamicSlice": 1, "DynamicSliceUpdate": 2}


def count_launches(outputs):
    """Approximate Metal dispatches for a built graph (upper bound)."""

    buffer = io.StringIO()
    mx.export_to_dot(buffer, *outputs)
    text = buffer.getvalue()
    labels = re.findall(r'label ="([^"]+)"', text)
    total = 0
    for label in labels:
        if label in FREE_PRIMITIVES:
            continue
        total += 1 + EXTRA_LAUNCHES.get(label, 0)
    # Concatenate copies once per input; count its in-edges.
    for match in re.finditer(r'\{ (\d+) \[label ="Concatenate"', text):
        node = match.group(1)
        total += max(0, len(re.findall(rf'-> {node}\b', text)) - 1)
    return total


def time_cycle(fn, inputs, reps, warmup, clear_cache):
    """MLX is lazy: ``fn()`` builds the graph, ``mx.eval`` runs it.

    ``eval_ms`` is encode+GPU -- the number the verdict rests on. ``build_ms``
    is the Python graph-construction tax, reported separately (a compiled
    variant replays in C++, so its build time is not the decode cost).
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


def max_abs_diff(got, ref):
    worst = 0.0
    for a, b in zip(got, ref):
        d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
        worst = max(worst, float(mx.max(d).item()))
    return worst


def build_data(args):
    """Synthetic tensors at the production shapes, one set per site.

    Distinct leaves per layer/site on purpose -- see ``make_cycle``.
    """

    dtype = mx.bfloat16
    blocks = args.pooled_blocks
    inv_freq = 1.0 / (
        float(args.rope_theta)
        ** (mx.arange(0, ROTARY_DIM, 2, dtype=mx.float32) / ROTARY_DIM)
    )
    data = {
        # Two arrays, identical values: the pre-diet QSAIndexer and Attention
        # each build their own. The diet makes them one object.
        "inv_freq_idx": inv_freq,
        "inv_freq_attn": inv_freq + 0.0,
        "row_ids": mx.arange(blocks, dtype=mx.int32).reshape(1, blocks),
        # bank: 12 independent banks, each an ARGUMENT of the compiled
        # function, so MLX can never donate one (the production regime: the
        # verifier bank still holds every pooled leaf).
        "banks": [
            mx.random.normal((1, blocks, IDX_HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "rows": [
            mx.random.normal((1, 1, IDX_HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "blks": [
            mx.array(blocks // 2 + i, dtype=mx.int32) for i in range(QSA_LAYERS)
        ],
        "conds": [mx.array(True) for _ in range(QSA_LAYERS)],
        # rope: one q/k/indexer/pooled set and one position leaf per layer
        "q_idx": [
            mx.random.normal((1, ROWS, IDX_HEADS, IDX_HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "q": [
            mx.random.normal((1, ROWS, N_HEADS, HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "k": [
            mx.random.normal((1, ROWS, N_KV_HEADS, HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "pooled_row": [
            mx.random.normal((1, 1, 1, IDX_HEAD_DIM)).astype(dtype)
            for _ in range(QSA_LAYERS)
        ],
        "pos": [
            mx.arange(1024, 1024 + ROWS, dtype=mx.int32)
            for _ in range(QSA_LAYERS)
        ],
        "blk_pos": [mx.array([1024], dtype=mx.int32) for _ in range(QSA_LAYERS)],
        # resid: one INDEPENDENT hyper per site. Threading one stream
        # through 96 writes would let mx.compile fuse across sites, which the
        # real decoder cannot do (an attention or MoE block sits between any
        # two residual writes).
        "hyper": [
            mx.random.normal((1, ROWS, HC_HIDDEN)).astype(dtype)
            for _ in range(RESID_SITES)
        ],
        "block_out": [
            mx.random.normal((1, ROWS, HIDDEN)).astype(dtype)
            for _ in range(RESID_SITES)
        ],
        "inject": [
            mx.random.normal((1, ROWS, HC_COUNT)).astype(dtype)
            for _ in range(RESID_SITES)
        ],
    }
    flat = []
    for value in data.values():
        flat.extend(value if isinstance(value, list) else [value])
    mx.eval(*flat)
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--families", type=str, default="bank,rope,resid",
        help="comma list from: bank, rope, resid",
    )
    p.add_argument("--pooled-blocks", type=int, default=POOLED_BLOCKS)
    p.add_argument("--rope-theta", type=float, default=10_000_000.0)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lanes", type=str, default="eager,compiled")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument(
        "--donatable-bank", action="store_true",
        help="chain the 12 bank updates so slice_update can donate "
             "(NOT the production regime; the real bank is a held state leaf)",
    )
    p.add_argument("--out", type=str, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    lanes = [l.strip() for l in args.lanes.split(",") if l.strip()]
    for family in families:
        if family not in FAMILIES:
            raise SystemExit(f"unknown family {family!r}; expected {list(FAMILIES)}")

    print("[micro-opdiet] must run under /tmp/mtplx-gpu-exclusive.lock", flush=True)
    _require_mlx()
    mx.random.seed(args.seed)

    print(
        f"[build] rows={ROWS} pooled=[1,{args.pooled_blocks},{IDX_HEAD_DIM}] bf16 "
        f"({args.pooled_blocks * IDX_HEAD_DIM * 2 / 1e6:.3f} MB)  "
        f"q=[1,{ROWS},{N_HEADS},{HEAD_DIM}]  hyper=[1,{ROWS},{HC_HIDDEN}]x{RESID_SITES}  "
        f"rot={ROTARY_DIM}",
        flush=True,
    )
    if args.donatable_bank:
        print("[warn] --donatable-bank is NOT the production regime", flush=True)
    data = build_data(args)

    results: dict[str, dict] = {}
    numerics: dict[str, float] = {}
    for family in families:
        inputs = cycle_inputs(family, data)
        refs = None
        for name in FAMILIES[family]:
            eager = make_cycle(
                name, data, donatable_bank=args.donatable_bank
            )
            for lane in lanes:
                fn = eager if lane == "eager" else mx.compile(eager)
                # Count on a FRESHLY BUILT graph: export_to_dot walks the
                # unevaluated tape, and an already-evaluated array is a leaf.
                launches = count_launches(fn(*inputs))
                out = fn(*inputs)
                mx.eval(out)
                if name == STOCK[family] and lane == lanes[0]:
                    refs = out
                elif name != STOCK[family]:
                    numerics[f"{name}/{lane}"] = max_abs_diff(out, refs)
                stats = time_cycle(fn, inputs, args.reps, args.warmup,
                                   args.clear_cache)
                stats["launches"] = launches
                stats["per_unit_us"] = stats["median_ms"] * 1e3 / UNITS[family]
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
                        (r["median_ms"] - base["median_ms"]) / base["median_ms"] * 100.0
                    )

    hdr = (f"{'variant':<14}{'lane':<10}{'units':>6}{'eval ms':>10}{'p10':>9}"
           f"{'p90':>9}{'us/unit':>10}{'delta%':>9}{'disp':>7}{'build':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for lane in lanes:
        for family in families:
            for name in FAMILIES[family]:
                key = f"{name}/{lane}"
                r = results.get(key)
                if r is None:
                    continue
                print(
                    f"{name:<14}{lane:<10}{UNITS[family]:>6}"
                    f"{r['median_ms']:>10.3f}{r['p10_ms']:>9.3f}{r['p90_ms']:>9.3f}"
                    f"{r['per_unit_us']:>10.1f}{r['delta_pct_vs_stock']:>+9.2f}"
                    f"{r['launches']:>7d}{r['build_ms']:>9.3f}"
                )
        print()

    print("numerics vs stock (max abs diff; every shipped rewrite is exact):")
    for name, diff in sorted(numerics.items()):
        print(f"  {name:<24} {diff:.6g}")

    summary = {
        "shapes": {
            "rows": ROWS, "pooled_blocks": args.pooled_blocks,
            "idx_head_dim": IDX_HEAD_DIM, "head_dim": HEAD_DIM,
            "rotary_dim": ROTARY_DIM, "qsa_layers": QSA_LAYERS,
            "resid_sites": RESID_SITES, "hc_hidden": HC_HIDDEN,
        },
        "reps": args.reps, "seed": args.seed,
        "donatable_bank": bool(args.donatable_bank),
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
