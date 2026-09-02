#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price ONE dependent kernel launch on this M5 Max, in three regimes.

Why this bench exists
---------------------
The whole decode-fusion program rests on a number nobody has measured on this
box: what a single extra *dependent* tiny kernel costs when it sits between two
weight-bound GEMVs.  The Flash-Next verify cycle issues ~5,415 dispatches, ~80%
of them zero-byte.  If a dependent launch costs >= 1 us, then folding roughly
half of those zero-byte dispatches away is worth ~2-3 ms per window and the
fusion program is the lever.  If it costs <= 0.5 us, the launches are hiding
under the GEMV reads and only byte-count and sync-count levers can pay.

``micro_dispatch_overhead.py`` prices dispatches on a chain with no memory
traffic in front of them (a fit over a real cycle put ~2.0 us/dispatch and
~10.4 us/command-buffer).  That is an upper bound: it never asks whether a
launch is *hidden* by a concurrent 26 MB weight read.  This bench interposes
the tiny kernels into a chain that is doing the real GDN's work, so what comes
out is the marginal cost the decode window would actually recover.

The chain (report M section A'.4 I1)
------------------------------------
One "repetition" is one Flash-Next GDN layer's two matmuls with the candidate
tiny kernels between them::

    y = quantized_matmul(x[4, 2560], W_i[16480, 2560])   q4 affine, gs 32
    h = contiguous(y[:, :2560])         hidden-width tensor, materialized ONCE
    t = <N dependent tiny kernels>      (the thing being priced)
    z = y[:, 10240:16384]               (free slice -- the real z branch)
    g = max(t, axis=-1, keepdims=True)  \\  fixed join, CONSTANT in N
    v = z * g                           /
    o = quantized_matmul(v[4, 6144], W_o[2560, 6144])    q8 affine, gs 64
    x = rms_norm(o)                     fixed, keeps 36 chained GEMVs bounded

``16480 = 10240 (qkv) + 6144 (z) + 48 (b) + 48 (a)`` is the fused GDN in_proj
of ``mtplx/models/qwen4_exp.py``; ``[2560, 6144]`` is its out_proj; the two
quantization recipes are the model's own (in_proj q4/gs32, out_proj q8/gs64).
36 repetitions per graph is the model's ``linear_attention`` layer count.

The fixed join, the ``contiguous`` on ``h`` and the terminating rms_norm are
the same at every N, so they cancel exactly in ``d(ms)/dN`` -- the slope is the
only number quoted, never the absolute graph time.  ``h`` is materialized
outside the tiny chain on purpose: a column slice of [4, 16480] is strided, and
``mx.fast.rms_norm`` would insert its own copy at N>0 but not at N=0, putting a
step change into the series the slope is fitted to.

``W_i`` and ``W_o`` are DISTINCT per repetition (36 x 26.37 MB + 36 x 16.71 MB
= 1.551 GB): a shared weight would sit in the SLC after the first repetition
and the GEMVs would stop being memory-bound, which is the regime that decides
whether a launch is hidden or exposed.

The three arms
--------------
``A  compiled, dependent``
    The chain above, inside one ``mx.compile``'d graph, ``mx.eval`` once.  The
    tiny kernels are on the critical path between the two GEMVs.  This is the
    verify-window regime and the arm the verdict rests on.
``B  compiled, independent``
    Identical main chain, but the N tiny kernels are fed from a per-repetition
    CONSTANT leaf that is not on the GEMV chain, and their results are returned
    as side outputs.  Nothing orders them against the GEMVs, so MLX may overlap
    them.  Expect a slope near zero; a slope near arm A's means the launches
    are serialized by the encoder, not by the data dependency.
``C  eager, sync-terminated``
    The same body run eagerly with ``mx.eval`` after every repetition -- the
    draft-chain regime, where each step ends in a host sync.  Reported, never
    used for the verdict: per the ``queued-vs-eager-metal-microbench`` note the
    eager lane charges every call a host sync and can INVERT a verdict for
    microsecond kernels, so promotions are decided on the queued/compiled lane.

The tiny kernel set is the real GDN one, cycled in this order::

    gate   h * mx.sigmoid(h)                          the GDN output gate
    norm   mx.fast.rms_norm(h, gamma, 1e-6)          on [4, 2560]
    add    h + bias
    copy   contiguous copy of a strided view (transpose out and back)
    cast   h.astype(float32).astype(bfloat16)         the mamba_ssm_dtype trip

The order is not cosmetic.  ``mx.compile`` fuses adjacent elementwise chains
into one kernel, so a run of elementwise steps turns "N ops" into fewer than N
launches and halves the price the fit reports.  Three of the five kinds are
plain elementwise (gate, add, cast) and two are not (norm and copy are separate
Metal kernels), and three items cannot be separated by two in a cycle of five
-- exactly one adjacent elementwise pair is unavoidable.  This order puts that
one pair on the WRAP (cast -> gate), so N <= 5 has none at all and only the
N=10 and N=20 points carry it.

Because of that, N is a count of *ops*, not of launches.  Every N also reports
a measured dispatch count from ``mx.export_to_dot`` and the fit is done twice:
``us_per_op`` (the spec's number, slope / repetitions) and ``us_per_dispatch``
(slope against the measured dispatch count).  When the two disagree, the second
is the one to build a fusion budget from.

Verdict
-------
From arm A's ``us_per_op``:

    >= 1.0 us  ->  "launch-count fusions worth 2-3 ms/window"
    <= 0.5 us  ->  "only byte/sync levers pay"
    between    ->  inconclusive; the report prints both bounds

The 2-3 ms arithmetic: ~5,415 dispatches/cycle x ~80% zero-byte x roughly half
of those foldable x 1 us ~= 2.2 ms, and the window is ~1 cycle.

Methodology and caveats (same as micro_k20_select.py / micro_opdiet.py)
----------------------------------------------------------------------
* The GPU must be otherwise IDLE.  This script does not verify that; the lock
  does.  Run it under ``bench/laguna/run_guarded.py``.
* MLX is lazy.  ``build_ms`` (Python graph construction, or the C++ replay for
  a compiled graph) is timed SEPARATELY from ``eval_ms`` (encode + GPU).  Only
  ``eval_ms`` feeds the fit -- at ~1-3 us of Python per op, folding the two
  together would measure Python, not Metal.
* Every arm/N pair gets ``--warmup`` untimed reps first, so the compile trace,
  the Metal pipeline-state build, and first-touch of the weight allocations are
  never charged to a timed rep.  Weights are built and ``mx.eval``'d once,
  before any timing, for the same reason.
* Thermals drift across a sweep.  The N grid is swept INSIDE each arm and each
  arm's fit prints r^2; a fit with r^2 < ``--min-r2`` (default 0.90) is a
  thermal or noise problem and the report says LOW-CONFIDENCE rather than
  quoting a per-launch price from it.  Re-run a low-r^2 arm on a cool box.
* MLX closes a command buffer every ``MLX_MAX_OPS_PER_BUFFER`` ops (50 by
  default in mlx 0.32.2, read once at device init), and a buffer costs ~10.4 us
  on this box.  Adding tiny kernels adds command buffers too, so part of the
  measured slope is buffer cost, not launch cost.  The table prints the implied
  ``cmdbuf`` count at every N and the report says how much of the slope that
  can be worth; if it is a material share, re-run with
  ``MLX_MAX_OPS_PER_BUFFER`` raised in the environment so the buffer count is
  flat across the grid and the slope is launches only.
* Weight residency is the point, not an accident: ``--max-weight-bytes``
  (default 2 GiB) refuses a configuration that would not fit beside nothing
  else on the box.

``--shapes real`` re-derives the geometry from the served model's
``config.json`` (the JSON only -- no weights are loaded, nothing is mmap'd).
It must reproduce the constants at the top of this file exactly;
``tests/test_fable_micro_args.py::test_real_config_reproduces_the_hardcoded_defaults``
is the guard that says so when it does not.

Standalone by construction: imports no mtplx, and MLX is imported lazily, so
the CLI surface, the shape derivation, the fit and the verdict are unit-tested
off-GPU in tests/test_fable_micro_args.py.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# MLX is imported lazily by ``main`` so the CLI surface, the shape table, the
# linear fit and the verdict can be unit tested on a box that is not holding
# the GPU lock -- importing MLX is the first step toward touching the device.
mx = None


def _require_mlx() -> None:
    global mx
    import mlx.core

    mx = mlx.core


LOCK_PATH = "/tmp/mtplx-gpu-exclusive.lock"
BANNER = (
    "[micro_dependent_launch] GPU WINDOW REQUIRED -- run under "
    f"{LOCK_PATH} via bench/laguna/run_guarded.py"
)

# --- Qwen3.8-Flash-Next GDN geometry (TextArgs defaults; see --shapes real) --
ROWS = 4                       # fixed-M4 verify width
HIDDEN = 2560
LINEAR_NUM_KEY_HEADS = 16
LINEAR_KEY_HEAD_DIM = 128
LINEAR_NUM_VALUE_HEADS = 48
LINEAR_VALUE_HEAD_DIM = 128
KEY_DIM = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM        # 2048
VALUE_DIM = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM  # 6144
QKV_DIM = 2 * KEY_DIM + VALUE_DIM                           # 10240
#: fused in_proj rows: qkv + z + b + a
IN_PROJ_ROWS = QKV_DIM + VALUE_DIM + 2 * LINEAR_NUM_VALUE_HEADS  # 16480
IN_BITS, IN_GROUP = 4, 32
OUT_BITS, OUT_GROUP = 8, 64
#: ``layer_types.count("linear_attention")`` -- 36 of the 48 layers.
GDN_LAYERS = 36
RMS_EPS = 1e-6

DEFAULT_N_GRID = (0, 2, 5, 10, 20)
#: Ordered so the one unavoidable adjacent-elementwise pair falls on the wrap.
#: See ELEMENTWISE_KINDS and the docstring.
TINY_CYCLE = ("gate", "norm", "add", "copy", "cast")
#: Kinds mx.compile can fuse into a neighbouring elementwise kernel.
ELEMENTWISE_KINDS = frozenset({"gate", "add", "cast"})

DEFAULT_CONFIG = (
    Path.home()
    / ".mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed/config.json"
)

#: Verdict thresholds, on arm A's microseconds per dependent tiny op.
FUSION_PAYS_US = 1.0
BYTES_ONLY_US = 0.5
VERDICT_FUSION = "launch-count fusions worth 2-3 ms/window"
VERDICT_BYTES = "only byte/sync levers pay"

#: MLX closes a command buffer every MLX_MAX_OPS_PER_BUFFER ops (default 50 in
#: mlx 0.32.2), read once at device init.  A command buffer costs ~10.4 us on
#: this box (micro_dispatch_overhead.py), so a slope measured across a GROWING
#: buffer count carries buffer cost as well as launch cost -- see the table's
#: ``cmdbuf`` column.
DEFAULT_OPS_PER_BUFFER = 50

#: Primitives Metal never launches a kernel for (pure view arithmetic).
FREE_PRIMITIVES = {
    "Reshape", "ExpandDims", "Squeeze", "Slice", "Transpose", "AsStrided",
}
#: Launches a primitive does not account for on its own.
EXTRA_LAUNCHES = {"DynamicSlice": 1, "DynamicSliceUpdate": 2}


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shapes:
    """The GDN geometry one repetition of the chain is built from."""

    rows: int
    hidden: int
    in_proj_rows: int
    qkv_dim: int
    value_dim: int
    in_bits: int
    in_group: int
    out_bits: int
    out_group: int
    gdn_layers: int
    source: str

    def validate(self) -> "Shapes":
        if self.rows < 1 or self.hidden < 1:
            raise ValueError(f"degenerate rows/hidden: {self.rows}x{self.hidden}")
        if self.hidden % self.in_group or self.value_dim % self.out_group:
            raise ValueError(
                "quantized_matmul needs the contracted dim divisible by the "
                f"group size: hidden {self.hidden} % {self.in_group}, "
                f"value_dim {self.value_dim} % {self.out_group}"
            )
        if self.in_proj_rows < self.qkv_dim + self.value_dim:
            raise ValueError(
                f"in_proj_rows {self.in_proj_rows} cannot hold qkv "
                f"{self.qkv_dim} + z {self.value_dim}"
            )
        if self.hidden > self.in_proj_rows:
            raise ValueError(
                f"hidden {self.hidden} exceeds in_proj_rows {self.in_proj_rows}; "
                "the tiny-kernel slice would not fit"
            )
        if self.gdn_layers < 2:
            raise ValueError(f"need >= 2 repetitions to fit a slope, got {self.gdn_layers}")
        return self


def default_shapes() -> Shapes:
    return Shapes(
        rows=ROWS,
        hidden=HIDDEN,
        in_proj_rows=IN_PROJ_ROWS,
        qkv_dim=QKV_DIM,
        value_dim=VALUE_DIM,
        in_bits=IN_BITS,
        in_group=IN_GROUP,
        out_bits=OUT_BITS,
        out_group=OUT_GROUP,
        gdn_layers=GDN_LAYERS,
        source="default",
    ).validate()


_GDN_QUANT_RE = re.compile(r"\.layers\.(\d+)\.linear_attn\.(in_proj_qkv|out_proj)$")


def _quant_entry(quant: dict, suffix: str, fallback: dict) -> tuple[int, int]:
    """bits/group_size for the first GDN ``suffix`` entry, else the top level."""

    for key, value in quant.items():
        match = _GDN_QUANT_RE.search(key)
        if match and match.group(2) == suffix and isinstance(value, dict):
            return int(value["bits"]), int(value["group_size"])
    if not isinstance(fallback, dict) or "bits" not in fallback:
        raise ValueError(f"no quantization entry for linear_attn.{suffix}")
    return int(fallback["bits"]), int(fallback["group_size"])


def derive_shapes_from_config(config: dict, *, rows: int = ROWS) -> Shapes:
    """The real GDN in_proj/out_proj geometry, from the model config JSON.

    Reads the JSON only -- no safetensors are opened and no weights are loaded.
    The in_proj row count is the FUSED one MTPLX builds at sanitize time
    (``_fuse_gdn_in_proj_sanitize``): qkv + z + b + a concatenated on the
    output-rows axis, which is the single GEMV the decode path issues.
    """

    text = config.get("text_config", config)
    hidden = int(text["hidden_size"])
    key_dim = int(text["linear_num_key_heads"]) * int(text["linear_key_head_dim"])
    v_heads = int(text["linear_num_value_heads"])
    value_dim = v_heads * int(text["linear_value_head_dim"])
    qkv_dim = 2 * key_dim + value_dim
    in_proj_rows = qkv_dim + value_dim + 2 * v_heads

    layer_types = text.get("layer_types") or []
    gdn_layers = sum(1 for t in layer_types if t == "linear_attention")
    if not gdn_layers:
        # No layer_types (or an all-attention config): fall back to the
        # full-attention interval, which is how the family spaces them.
        total = int(text.get("num_hidden_layers", 0))
        interval = int(text.get("full_attention_interval", 0) or 0)
        gdn_layers = total - (total // interval if interval else 0)

    quant = config.get("quantization_config") or config.get("quantization") or {}
    in_bits, in_group = _quant_entry(quant, "in_proj_qkv", quant)
    out_bits, out_group = _quant_entry(quant, "out_proj", quant)

    return Shapes(
        rows=rows,
        hidden=hidden,
        in_proj_rows=in_proj_rows,
        qkv_dim=qkv_dim,
        value_dim=value_dim,
        in_bits=in_bits,
        in_group=in_group,
        out_bits=out_bits,
        out_group=out_group,
        gdn_layers=gdn_layers,
        source="real",
    ).validate()


def load_shapes(mode: str, config_path) -> Shapes:
    if mode == "default":
        return default_shapes()
    if mode != "real":
        raise ValueError(f"unknown --shapes {mode!r}")
    path = Path(config_path)
    if not path.is_file():
        raise SystemExit(
            f"--shapes real needs the model config at {path} (JSON only, no "
            "weights are read); pass --config to point elsewhere"
        )
    return derive_shapes_from_config(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Memory budget
# ---------------------------------------------------------------------------


def quantized_bytes(rows: int, cols: int, bits: int, group: int) -> int:
    """Bytes an affine-quantized [rows, cols] weight occupies (bf16 s/b)."""

    if cols % group:
        raise ValueError(f"cols {cols} not divisible by group {group}")
    packed = rows * cols * bits // 8
    scales = rows * (cols // group) * 2
    return packed + 2 * scales


def weight_budget(shapes: Shapes) -> dict:
    """Per-repetition and total weight bytes for one graph."""

    in_bytes = quantized_bytes(
        shapes.in_proj_rows, shapes.hidden, shapes.in_bits, shapes.in_group
    )
    out_bytes = quantized_bytes(
        shapes.hidden, shapes.value_dim, shapes.out_bits, shapes.out_group
    )
    per_rep = in_bytes + out_bytes
    return {
        "in_proj_bytes": in_bytes,
        "out_proj_bytes": out_bytes,
        "per_repetition_bytes": per_rep,
        "repetitions": shapes.gdn_layers,
        "total_bytes": per_rep * shapes.gdn_layers,
    }


def check_budget(shapes: Shapes, max_bytes: int) -> dict:
    budget = weight_budget(shapes)
    if budget["total_bytes"] > max_bytes:
        raise SystemExit(
            f"weights would be {budget['total_bytes'] / 2**30:.3f} GiB, over the "
            f"{max_bytes / 2**30:.3f} GiB cap; lower --repetitions or raise "
            "--max-weight-bytes (and confirm the box is free first)"
        )
    return budget


# ---------------------------------------------------------------------------
# Fit + verdict (pure python; no MLX)
# ---------------------------------------------------------------------------


def linear_fit(xs, ys) -> dict:
    """Least-squares ``y = slope*x + intercept`` with r^2.

    A flat series (ss_tot == 0) has r^2 = 1.0 by definition here: a horizontal
    line explains it perfectly, and that is the *expected* arm B result, so it
    must not be reported as an unfittable one.
    """

    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points to fit a slope")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("all x values are identical; slope is undefined")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"slope": slope, "intercept": intercept, "r2": r2, "points": n}


def per_unit_us(slope_ms_per_unit: float, units_per_graph: int) -> float:
    """Slope in ms per unit of N -> microseconds per unit per repetition."""

    if units_per_graph < 1:
        raise ValueError("units_per_graph must be >= 1")
    return slope_ms_per_unit * 1e3 / units_per_graph


def verdict_line(us_per_launch: float, r2: float, min_r2: float = 0.90) -> str:
    """The one line report M asks for, gated on the fit actually being linear."""

    if r2 < min_r2:
        return (
            f"LOW-CONFIDENCE FIT (r2 {r2:.3f} < {min_r2:.2f}) -- "
            f"{us_per_launch:.3f} us/launch is not quotable; re-run on a cool box"
        )
    if us_per_launch >= FUSION_PAYS_US:
        return f"{us_per_launch:.3f} us/launch >= {FUSION_PAYS_US:.1f} => {VERDICT_FUSION}"
    if us_per_launch <= BYTES_ONLY_US:
        return f"{us_per_launch:.3f} us/launch <= {BYTES_ONLY_US:.1f} => {VERDICT_BYTES}"
    return (
        f"{us_per_launch:.3f} us/launch is between {BYTES_ONLY_US:.1f} and "
        f"{FUSION_PAYS_US:.1f} => INCONCLUSIVE: a full fold of the foldable "
        f"zero-byte dispatches is worth "
        f"{us_per_launch * 5415 * 0.8 * 0.5 / 1e3:.2f} ms/window; decide against "
        "the byte levers on the same window"
    )


def tiny_plan(n: int, cycle=TINY_CYCLE) -> list:
    """The kind of each of the N tiny kernels, cycled."""

    if n < 0:
        raise ValueError("N must be >= 0")
    return [cycle[i % len(cycle)] for i in range(n)]


def parse_int_list(raw: str, *, name: str) -> tuple:
    values = tuple(sorted({int(part) for part in str(raw).split(",") if part.strip()}))
    if not values:
        raise ValueError(f"{name} is empty")
    if len(values) < 2:
        raise ValueError(f"{name} needs >= 2 distinct values to fit a slope")
    if any(v < 0 for v in values):
        raise ValueError(f"{name} must be non-negative")
    return values


# ---------------------------------------------------------------------------
# Graph construction (MLX)
# ---------------------------------------------------------------------------


def _tiny_step(kind, h, gamma, bias):
    """One tiny kernel from the real GDN set, on a [rows, hidden] bf16 array."""

    if kind == "norm":
        return mx.fast.rms_norm(h, gamma, RMS_EPS)
    if kind == "gate":
        return h * mx.sigmoid(h)
    if kind == "add":
        return h + bias
    if kind == "cast":
        return h.astype(mx.float32).astype(mx.bfloat16)
    if kind == "copy":
        # A contiguous copy of a strided view, and back: two Copy kernels, so
        # the step's output is contiguous and the next step's cost does not
        # depend on whether this one ran.
        return mx.contiguous(mx.contiguous(mx.swapaxes(h, 0, 1)).swapaxes(0, 1))
    raise ValueError(f"unknown tiny kernel {kind!r}")


def _tiny_chain(kinds, h, gamma, bias):
    for kind in kinds:
        h = _tiny_step(kind, h, gamma, bias)
    return h


def build_weights(shapes: Shapes, seed: int):
    """Distinct q4 in_proj / q8 out_proj per repetition, plus the tiny leaves.

    Built one repetition at a time and evaluated immediately so the bf16 source
    of ``mx.quantize`` is a transient, not 36 live copies.
    """

    mx.random.seed(seed)
    scale = 1.0 / (shapes.hidden ** 0.5)
    in_w, out_w = [], []
    for _ in range(shapes.gdn_layers):
        src = (
            mx.random.normal((shapes.in_proj_rows, shapes.hidden)) * scale
        ).astype(mx.bfloat16)
        in_w.append(
            mx.quantize(src, group_size=shapes.in_group, bits=shapes.in_bits)
        )
        del src
        src = (
            mx.random.normal((shapes.hidden, shapes.value_dim)) * scale
        ).astype(mx.bfloat16)
        out_w.append(
            mx.quantize(src, group_size=shapes.out_group, bits=shapes.out_bits)
        )
        del src
        mx.eval(in_w[-1], out_w[-1])
    gamma = mx.ones((shapes.hidden,), dtype=mx.bfloat16)
    bias = (mx.random.normal((shapes.hidden,)) * 0.1).astype(mx.bfloat16)
    # Arm B's off-chain feed: DISTINCT per repetition, or mx.compile's CSE pass
    # collapses 36 identical tiny chains into one.
    constants = [
        mx.random.normal((shapes.rows, shapes.hidden)).astype(mx.bfloat16)
        for _ in range(shapes.gdn_layers)
    ]
    x0 = mx.random.normal((shapes.rows, shapes.hidden)).astype(mx.bfloat16)
    mx.eval(gamma, bias, x0, *constants)
    return {
        "in_w": in_w,
        "out_w": out_w,
        "gamma": gamma,
        "bias": bias,
        "constants": constants,
        "x0": x0,
    }


def repetition(shapes: Shapes, weights, kinds, idx: int, x, *, dependent: bool):
    """One GDN layer's worth of the chain.  The ONLY place the body is spelled.

    ``dependent=True`` puts the tiny kernels on the GEMV-to-GEMV critical path
    (arm A / arm C).  ``dependent=False`` feeds them an off-chain constant leaf
    and hands them back as a side output (arm B), so nothing orders them
    against the GEMVs and MLX may overlap them.  The main chain is identical in
    both cases -- byte-for-byte arm A's N=0 chain -- which is what makes the
    two intercepts comparable.

    Returns ``(x_next, side_or_None)``.
    """

    gamma, bias = weights["gamma"], weights["bias"]
    wq, sq, bq = weights["in_w"][idx]
    y = mx.quantized_matmul(
        x, wq, sq, bq, transpose=True,
        group_size=shapes.in_group, bits=shapes.in_bits,
    )
    # Materialize the hidden-width slice ONCE, outside the tiny chain: a column
    # slice of [rows, 16480] is strided, and mx.fast.rms_norm would otherwise
    # insert its own copy at N>0 but not at N=0, putting a step change in the
    # series the slope is fitted to.  Fixed cost, cancels in the slope.
    h = mx.contiguous(y[:, : shapes.hidden])
    side = None
    if dependent:
        t = _tiny_chain(kinds, h, gamma, bias)
    else:
        t = h
        if kinds:
            # At N=0 there is nothing to overlap, so arm B emits no side output
            # at all rather than handing mx.compile a captured constant as a
            # graph output.  That makes arm B's N=0 graph identical to arm A's.
            side = _tiny_chain(kinds, weights["constants"][idx], gamma, bias)
    z = y[:, shapes.qkv_dim : shapes.qkv_dim + shapes.value_dim]
    # Fixed join, identical at every N: a reduce and a broadcast multiply.
    # ``max`` and not ``mean`` because a near-zero gate would collapse the
    # terminating rms_norm into eps-dominated garbage.
    v = z * mx.max(t, axis=-1, keepdims=True)
    wo, so, bo = weights["out_w"][idx]
    o = mx.quantized_matmul(
        v, wo, so, bo, transpose=True,
        group_size=shapes.out_group, bits=shapes.out_bits,
    )
    # Renormalize between repetitions so 36 chained GEMVs stay bounded in bf16.
    # Fixed cost, cancels in the slope.
    return mx.fast.rms_norm(o, gamma, RMS_EPS), side


def make_body(shapes: Shapes, weights: dict, kinds, *, dependent: bool):
    """The full ``repetitions``-long chain as one uncompiled callable of x."""

    def body(x):
        side = []
        for idx in range(shapes.gdn_layers):
            x, extra = repetition(
                shapes, weights, kinds, idx, x, dependent=dependent
            )
            if extra is not None:
                side.append(extra)
        # A list, not a tuple: mx.compile's documented output contract is an
        # array or a list of arrays.
        return [x, *side]

    return body


def count_launches(outputs) -> int:
    """Approximate Metal dispatches for a built graph (upper bound)."""

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
        total += max(0, len(re.findall(rf"-> {node}\b", text)) - 1)
    return total


def ops_per_buffer(env=None) -> int:
    """MLX_MAX_OPS_PER_BUFFER as this process will see it (read at device init)."""

    import os

    raw = (env if env is not None else os.environ).get("MLX_MAX_OPS_PER_BUFFER")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_OPS_PER_BUFFER
    return value if value > 0 else DEFAULT_OPS_PER_BUFFER


def command_buffers(launches: int, per_buffer: int) -> int:
    """Ceiling division: the buffer count a launch count implies."""

    if per_buffer < 1:
        raise ValueError("per_buffer must be >= 1")
    return -(-int(launches) // per_buffer)


def _percentiles(samples) -> dict:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "p10_ms": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
    }


def time_graph(fn, x0, reps: int, warmup: int) -> dict:
    """One compiled graph, ``mx.eval`` once per rep.  Build time kept apart."""

    for _ in range(warmup):
        mx.eval(*fn(x0))
    evals, builds = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(x0)
        t1 = time.perf_counter()
        mx.eval(*out)
        evals.append((time.perf_counter() - t1) * 1e3)
        builds.append((t1 - t0) * 1e3)
    stats = _percentiles(evals)
    stats["build_ms"] = statistics.median(builds)
    return stats


def time_eager(shapes: Shapes, weights, kinds, reps: int, warmup: int) -> dict:
    """Arm C: the same body, eager, with a host sync after every repetition.

    Not the same graph as arm A even at the same N: each repetition is encoded,
    submitted and waited on, so every tiny kernel is bracketed by the host
    round trip a draft step really pays.

    Unlike arms A and B this timing INCLUDES Python graph construction, and
    that is deliberate: the sync at the end of repetition i means the Python
    that builds repetition i+1 runs with the GPU idle, so it is genuinely on
    the critical path here.  ``build_ms`` is reported as 0.0 for this arm
    because there is no separable build phase to report.
    """

    def one_pass():
        x = weights["x0"]
        for idx in range(shapes.gdn_layers):
            x, _ = repetition(shapes, weights, kinds, idx, x, dependent=True)
            mx.eval(x)
        return x

    for _ in range(warmup):
        one_pass()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        one_pass()
        samples.append((time.perf_counter() - t0) * 1e3)
    stats = _percentiles(samples)
    stats["build_ms"] = 0.0
    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


ARMS = {
    "A": "compiled, dependent (verify-window regime)",
    "B": "compiled, independent (overlap headroom)",
    "C": "eager, sync-terminated (draft-chain regime)",
}


def summarize_arm(rows: list, repetitions: int, min_r2: float) -> dict:
    """Fit ms-per-graph against N and against the measured dispatch count."""

    ns = [r["n"] for r in rows]
    ms = [r["median_ms"] for r in rows]
    op_fit = linear_fit(ns, ms)
    out = {
        "op_fit": op_fit,
        "us_per_op": per_unit_us(op_fit["slope"], repetitions),
        "ms_per_graph_slope": op_fit["slope"],
    }
    disp = [r["launches"] for r in rows]
    if len(set(disp)) >= 2:
        d_fit = linear_fit(disp, ms)
        out["dispatch_fit"] = d_fit
        out["us_per_dispatch"] = d_fit["slope"] * 1e3
    else:
        out["dispatch_fit"] = None
        out["us_per_dispatch"] = None
    out["quotable"] = bool(op_fit["r2"] >= min_r2)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shapes", choices=("default", "real"), default="default",
                   help="'real' re-derives the GDN geometry from the model "
                        "config JSON (no weights are loaded)")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                   help="config.json for --shapes real")
    p.add_argument("--n-grid", type=str,
                   default=",".join(str(n) for n in DEFAULT_N_GRID),
                   help="dependent tiny-kernel counts to sweep")
    p.add_argument("--repetitions", type=int, default=None,
                   help="GEMV pairs per graph (default: the model's GDN layer count)")
    p.add_argument("--reps", type=int, default=200, help="timed graphs per N")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--arms", type=str, default="A,B,C")
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--min-r2", type=float, default=0.90,
                   help="below this the per-launch price is not quoted")
    p.add_argument("--max-weight-bytes", type=int, default=2 * 2**30,
                   help="refuse a configuration whose weights exceed this")
    p.add_argument("--json", type=str, default=None, help="receipt path")
    p.add_argument("--print-shapes", action="store_true",
                   help="resolve the shapes, print them, and exit (no MLX)")
    return p


def resolve(args) -> tuple:
    try:
        shapes = load_shapes(args.shapes, args.config)
        if args.repetitions is not None:
            shapes = Shapes(
                **{**asdict(shapes), "gdn_layers": args.repetitions}
            ).validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        n_grid = parse_int_list(args.n_grid, name="--n-grid")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    arms = tuple(a.strip().upper() for a in args.arms.split(",") if a.strip())
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; expected {sorted(ARMS)}")
    budget = check_budget(shapes, args.max_weight_bytes)
    return shapes, n_grid, arms, budget


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    shapes, n_grid, arms, budget = resolve(args)

    print(BANNER)
    print(
        f"[shapes/{shapes.source}] in_proj [{shapes.in_proj_rows}, {shapes.hidden}] "
        f"q{shapes.in_bits}/gs{shapes.in_group} "
        f"({budget['in_proj_bytes'] / 1e6:.2f} MB)  "
        f"out_proj [{shapes.hidden}, {shapes.value_dim}] "
        f"q{shapes.out_bits}/gs{shapes.out_group} "
        f"({budget['out_proj_bytes'] / 1e6:.2f} MB)  "
        f"rows={shapes.rows}  reps/graph={shapes.gdn_layers}  "
        f"weights={budget['total_bytes'] / 2**30:.3f} GiB"
    )
    print(f"[grid] N={list(n_grid)}  arms={list(arms)}  reps={args.reps} "
          f"warmup={args.warmup}")
    if args.print_shapes:
        print(json.dumps({"shapes": asdict(shapes), "budget": budget}, indent=2))
        return 0

    _require_mlx()
    per_buffer = ops_per_buffer()
    weights = build_weights(shapes, args.seed)

    report = {
        "shapes": asdict(shapes),
        "budget": budget,
        "n_grid": list(n_grid),
        "reps": args.reps,
        "warmup": args.warmup,
        "seed": args.seed,
        "min_r2": args.min_r2,
        "tiny_cycle": list(TINY_CYCLE),
        "ops_per_buffer": per_buffer,
        "arms": {},
        "fits": {},
    }

    for arm in arms:
        rows = []
        for n in n_grid:
            kinds = tiny_plan(n)
            if arm == "C":
                stats = time_eager(shapes, weights, kinds, args.reps, args.warmup)
                launches = 0
            else:
                body = make_body(shapes, weights, kinds, dependent=(arm == "A"))
                fn = mx.compile(body)
                # Count on a FRESHLY BUILT graph: export_to_dot walks the
                # unevaluated tape, and an evaluated array is a leaf.
                launches = count_launches(fn(weights["x0"]))
                stats = time_graph(fn, weights["x0"], args.reps, args.warmup)
            stats["n"] = n
            stats["launches"] = launches
            stats["command_buffers"] = command_buffers(launches, per_buffer)
            stats["kinds"] = kinds
            rows.append(stats)
        report["arms"][arm] = rows
        report["fits"][arm] = summarize_arm(rows, shapes.gdn_layers, args.min_r2)

        print(f"\narm {arm}: {ARMS[arm]}")
        hdr = (f"{'N':>4}{'eval ms':>11}{'p10':>10}{'p90':>10}"
               f"{'build ms':>11}{'disp':>8}{'cmdbuf':>8}{'us/rep/N':>11}")
        print(hdr)
        print("-" * len(hdr))
        base, base_n = rows[0]["median_ms"], rows[0]["n"]
        for r in rows:
            span = (r["n"] - base_n) * shapes.gdn_layers
            marginal = (
                (r["median_ms"] - base) * 1e3 / span if span else float("nan")
            )
            print(f"{r['n']:>4}{r['median_ms']:>11.3f}{r['p10_ms']:>10.3f}"
                  f"{r['p90_ms']:>10.3f}{r['build_ms']:>11.3f}"
                  f"{r['launches']:>8d}{r['command_buffers']:>8d}"
                  f"{marginal:>11.3f}")
        spread = rows[-1]["command_buffers"] - rows[0]["command_buffers"]
        if spread > 0:
            print(f"  note: command buffers grow {rows[0]['command_buffers']} -> "
                  f"{rows[-1]['command_buffers']} across the grid; at ~10.4 us "
                  f"each (micro_dispatch_overhead.py) that is up to "
                  f"{spread * 10.4 / ((n_grid[-1] - n_grid[0]) * shapes.gdn_layers):.3f} us/op "
                  "of BUFFER cost inside the slope below.  Re-run with "
                  "MLX_MAX_OPS_PER_BUFFER raised to separate it.")
        fit = report["fits"][arm]
        line = (f"  fit: {fit['ms_per_graph_slope'] * 1e3:+.3f} us/graph per N, "
                f"r2 {fit['op_fit']['r2']:.4f}  ->  "
                f"{fit['us_per_op']:.3f} us per dependent tiny op")
        if fit["us_per_dispatch"] is not None:
            line += f"  ({fit['us_per_dispatch']:.3f} us per measured dispatch)"
        print(line)

    print("\n" + "=" * 72)
    if "A" in report["fits"]:
        a = report["fits"]["A"]
        verdict = verdict_line(a["us_per_op"], a["op_fit"]["r2"], args.min_r2)
        report["verdict"] = verdict
        print("VERDICT: " + verdict)
        if "B" in report["fits"]:
            b = report["fits"]["B"]
            print(f"  arm B (independent) {b['us_per_op']:.3f} us/op: the "
                  f"{a['us_per_op'] - b['us_per_op']:+.3f} us gap is the part of a "
                  "launch that only a DEPENDENT kernel pays")
        if "C" in report["fits"]:
            c = report["fits"]["C"]
            print(f"  arm C (eager+sync) {c['us_per_op']:.3f} us/op: the "
                  "draft-chain regime, not the verdict lane")
    else:
        print("VERDICT: arm A not run; no verdict")
    print("=" * 72)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n[out] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
