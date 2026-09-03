#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This loads the served 125B model and runs a
# real prefill; outside the serialized window it interrupts whatever else
# holds the box.  It never starts a server and never touches :8080.
# ---------------------------------------------------------------------------
"""Attribute the unexplained +4.9 GB prefill peak at chunk 4096.

The problem (M §B1, §B9)
------------------------
Widening the prefill chunk from the shipped 2,048 to 4,096 is measured to be
worth **~1.0 s of a 13.5 s prefill** (W32, ``scratchpad/w32/tile.log``:
control 13.53 s / peak 88.69 GB, candidate 12.53 s / peak **92.22 GB**), and
B1's super-chunk needs the same widening to reach 320 rows/expert.  What
blocks both is the **+3.5 GB the widening costs**, which nothing in the tree
predicts:

* the guard's own model (``mtplx/fable_prefill_chunk.plan_prefill_chunk_memory``
  over ``memory_plan.QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM`` = 12.75 B per
  chunk-row x context token x 4 live layers) says **+1.7 GB**;
* **W32 proved it is not the QSA score tensor.**  Arm B1 (chunk 4096) and arm
  B2 (chunk 4096 + ``MTPLX_FABLE_PREFILL_QSA_QUERY_TILE=2048``, which caps the
  live query rows and therefore the whole ``[24, S, T]`` score/mask/softmax
  chain at the 2,048-row value) recorded peaks of 92,219,331,638 and
  92,219,385,422 bytes -- **the same 92.2 GB to five significant figures**.
  A term the query tile does not move is not the score tensor.

So the peak is somewhere else, and until it is named, B1's group size and B9's
chunk geometry are both being sized by a model that is known to be wrong.

What this measures
------------------
One real prefill, with ``mx.get_active_memory`` / ``mx.get_peak_memory``
bracketed around every block of every layer, in two modes:

``serialized``
    ``mx.eval`` at each block boundary.  Each block's ``active`` delta is then
    the memory *that block's outputs* hold, and its ``peak`` delta is the
    high-water it reaches internally.  This is an attribution, not a
    reproduction: forcing an eval per block removes the lazy scheduler's
    freedom and therefore lowers the whole-chunk peak.
``lazy``
    the production schedule, one ``mx.eval`` per chunk.  Gives the true peak.

``lazy_peak - serialized_peak`` is the part of the peak that is *scheduler
concurrency* -- tensors from different blocks alive at the same time because
MLX chose to keep them -- rather than any single block's footprint.  If the
+3.5 GB lives there, the fix is an eval barrier or a chunk-local
``clear_cache``, not a smaller chunk.

Families bracketed (the J §2.2 taxonomy, so the numbers are comparable):
``attention`` (QSA), ``indexer``, ``gdn``, ``hyper``, ``ple``,
``moe_router``, ``moe_gate_up``, ``moe_glu``, ``moe_down``, ``moe_shared``,
``moe_block``.  The three ``moe_*`` GEMM families come from a shim over
``mx.gather_qmm`` keyed on the output width, which is how the routed path is
separated from the shared expert without editing the model.

Sweep and the answer it produces
--------------------------------
``--chunk-sizes 2048,4096`` runs the same prompt at both widths and prints the
**per-family delta**.  Whatever family carries ~3.5 GB of the difference is the
term the guard is missing.  The arithmetic to check it against, per layer, at
``R`` chunk rows (top_k 10, hidden 2560, moe_intermediate 640, bf16)::

    sorted x      [10R, 2560]  = R x  51,200 B      839 MB at R=16384
    gate+up out   [10R, 1280]  = R x  25,600 B      419 MB
    silu*up       [10R,  640]  = R x  12,800 B      210 MB
    down out      [10R, 2560]  = R x  51,200 B      839 MB
    unsorted out  [10R, 2560]  = R x  51,200 B      839 MB
                                 --------------
                       total     R x 192,000 B     3.15 GB at R=16384
                                                   786 MB at R= 4096
                                                   393 MB at R= 2048

i.e. the routed MoE transient alone accounts for **+393 MB** of a 2048->4096
widening if only one layer's chain is ever live, and **+3.1 GB** if eight
layers' chains are live at once.  That factor -- how many layers of MoE
transient the lazy scheduler keeps alive -- is what this census reads off the
allocator, and it is the number B1's group-size budget needs.

Guard rails: no server, no :8080, no launchctl.  ``--self-test`` exercises the
recorder and the report with a fake allocator; no MLX, no model, no lock.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL = Path(
    "/Users/davidtai/.mtplx/models/"
    "Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
)
LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
#: abba_driver.MINIMUM_RESIDENT_BYTES -- the resident floor the served pack
#: needs before a prefill of this size is safe to start.
MINIMUM_RESIDENT_BYTES = 89_480_048_859
DEFAULT_TOKENS = Path(
    "/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/"
    "1b1e4a52-8af8-4acc-a173-0bf81c785447/scratchpad/prompt_ids.npy"
)

FAMILIES = (
    "ple",
    "hyper",
    "gdn",
    "attention",
    "indexer",
    "moe_block",
    "moe_router",
    "moe_gate_up",
    "moe_glu",
    "moe_down",
    "moe_shared",
)


#: Evidence that ``bench/laguna/run_guarded.py`` launched this process.  The
#: lock FILE always exists and is normally held by whoever owns the box, so
#: ``LOCK.exists()`` proves nothing and a flock probe only says "somebody" --
#: it cannot say "this run's guard".  ``abba_driver.acquire_guard(mode="auto")``
#: keys on exactly these two env vars, and so does this.
GUARD_ENV_VARS = ("MTPLX_DSV4_GUARD_WINDOW_PATH", "MTPLX_GUARD_ATTEST_FD")


def require_guard_window(allow_unlocked: bool) -> None:
    """Refuse to issue Metal work outside a guarded window."""

    import os

    if allow_unlocked:
        return
    if any(os.environ.get(name) for name in GUARD_ENV_VARS):
        return
    raise SystemExit(
        "no GPU guard evidence ("
        + " / ".join(GUARD_ENV_VARS)
        + " unset): run this under bench/laguna/run_guarded.py.  The lock file "
        "existing is NOT evidence -- it is normally held by another job, and "
        "starting Metal work here interrupts it.  --allow-unlocked overrides."
    )



# ---------------------------------------------------------------------------
# Recorder (pure python; the allocator is injected so this is unit-testable)
# ---------------------------------------------------------------------------


class MemoryRecorder:
    """Per-family active/peak accounting around bracketed blocks.

    ``active`` is what the allocator currently holds; ``peak`` is its
    high-water since the last reset.  For each bracketed call we keep:

    ``active_delta_sum``
        sum over calls of (active after - active before).  For a block whose
        output survives, this is what the block *added*; for one whose inputs
        die inside it, it can be negative.  Summed over a layer it is the
        layer's net retention.
    ``peak_delta_max``
        the largest (peak after - active before) any single call reached --
        the block's own internal high-water, independent of what else is live.
    ``active_high_water``
        the largest ``active`` observed at any bracket of this family.  This is
        the one that answers "how much was live while this family ran".
    """

    def __init__(self, active_fn, peak_fn):
        self._active = active_fn
        self._peak = peak_fn
        self.calls: dict[str, int] = defaultdict(int)
        self.active_delta_sum: dict[str, int] = defaultdict(int)
        self.active_delta_max: dict[str, int] = defaultdict(int)
        self.peak_delta_max: dict[str, int] = defaultdict(int)
        self.active_high_water: dict[str, int] = defaultdict(int)
        self.bytes_out: dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def bracket(self, family: str):
        before = self._active()
        peak_before = self._peak()
        yield
        after = self._active()
        peak_after = self._peak()
        self.calls[family] += 1
        delta = after - before
        self.active_delta_sum[family] += delta
        if delta > self.active_delta_max[family]:
            self.active_delta_max[family] = delta
        reach = peak_after - before if peak_after > peak_before else 0
        if reach > self.peak_delta_max[family]:
            self.peak_delta_max[family] = reach
        if after > self.active_high_water[family]:
            self.active_high_water[family] = after

    def note_bytes(self, family: str, nbytes: int) -> None:
        self.bytes_out[family] += int(nbytes)

    def as_dict(self) -> dict:
        return {
            family: {
                "calls": self.calls[family],
                "active_delta_sum": self.active_delta_sum[family],
                "active_delta_max": self.active_delta_max[family],
                "peak_delta_max": self.peak_delta_max[family],
                "active_high_water": self.active_high_water[family],
                "bytes_out": self.bytes_out[family],
            }
            for family in sorted(set(self.calls) | set(self.bytes_out))
        }


# ---------------------------------------------------------------------------
# Model instrumentation
# ---------------------------------------------------------------------------


def _eval_tree(mx, value):
    """Evaluate whatever a block returned, without assuming its shape."""

    if isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, (tuple, list)):
        arrays = [v for v in value if isinstance(v, mx.array)]
        if arrays:
            mx.eval(arrays)


class _BracketShim:
    """Callable stand-in for one child module.

    Python resolves ``obj(...)`` through ``type(obj).__call__``, so patching
    ``__call__`` on an *instance* does nothing.  The house pattern (see
    ``tests/test_fable_mtp_kv_only.py::_install_shims``) is to swap the child
    out on its owner with a plain callable; ``nn.Module`` is a dict, so
    ``owner[name] = shim`` is the swap and the original goes straight back.
    """

    def __init__(self, wrapped, family, recorder, serialize, eval_tree):
        self._wrapped = wrapped
        self._family = family
        self._recorder = recorder
        self._serialize = serialize
        self._eval_tree = eval_tree

    def __call__(self, *a, **kw):
        with self._recorder.bracket(self._family):
            out = self._wrapped(*a, **kw)
            if self._serialize:
                self._eval_tree(out)
        return out

    def __getattr__(self, name):  # keep isinstance-free attribute reads working
        return getattr(self._wrapped, name)


def _wrap_child(mx, owner, name, family, recorder, serialize):
    """Bracket ``owner[name]``.  Returns the undo callable."""

    original = owner[name]
    owner[name] = _BracketShim(
        original, family, recorder, serialize, lambda v: _eval_tree(mx, v)
    )

    def undo():
        owner[name] = original

    return undo


def _wrap_gather_qmm(mx, recorder, serialize, moe_intermediate):
    """Split the routed grouped GEMMs by output width.

    ``2 * moe_intermediate`` is the fused gate+up pack, ``hidden`` is the down
    projection; anything else (the shared expert runs ``quantized_matmul``,
    not ``gather_qmm``) is charged to ``moe_shared``.
    """

    original = mx.gather_qmm
    gate_up_width = 2 * moe_intermediate

    def wrapped(x, w, *a, **kw):
        width = int(w.shape[-2]) if hasattr(w, "shape") and len(w.shape) >= 2 else 0
        # w is [E, out, in_packed] for transpose=True.
        family = (
            "moe_gate_up"
            if width == gate_up_width
            else "moe_down" if width and width != gate_up_width else "moe_shared"
        )
        with recorder.bracket(family):
            out = original(x, w, *a, **kw)
            if serialize:
                mx.eval(out)
        recorder.note_bytes(family, out.size * out.dtype.size)
        return out

    mx.gather_qmm = wrapped
    return lambda: setattr(mx, "gather_qmm", original)


def resolve_text_model(model):
    """Walk down to the module that owns ``layers`` (Model -> ... -> text)."""

    inner = model
    for _ in range(4):
        if hasattr(inner, "layers") and hasattr(inner, "args"):
            return inner
        inner = getattr(inner, "language_model", None) or getattr(inner, "model", None)
        if inner is None:
            break
    raise RuntimeError("could not find the layer list on this model")


def instrument(mx, model, recorder, serialize: bool):
    """Bracket every block of every layer.  Returns an undo callable."""

    undos = []
    inner = resolve_text_model(model)
    args = inner.args

    for layer in inner.layers:
        if "ple" in layer:
            undos.append(_wrap_child(mx, layer, "ple", "ple", recorder, serialize))
        if layer.is_linear:
            undos.append(
                _wrap_child(mx, layer, "linear_attn", "gdn", recorder, serialize)
            )
        else:
            attn = layer.self_attn
            if attn.get("indexer") is not None:
                undos.append(
                    _wrap_child(mx, attn, "indexer", "indexer", recorder, serialize)
                )
            undos.append(
                _wrap_child(mx, layer, "self_attn", "attention", recorder, serialize)
            )
        for name in ("attn_hyper_connection", "mlp_hyper_connection"):
            undos.append(_wrap_child(mx, layer, name, "hyper", recorder, serialize))
        mlp = layer.mlp
        undos.append(
            _wrap_child(mx, mlp, "gate", "moe_router", recorder, serialize)
        )
        undos.append(
            _wrap_child(mx, mlp, "shared_expert", "moe_shared", recorder, serialize)
        )
        undos.append(
            _wrap_child(mx, mlp, "switch_mlp", "moe_glu", recorder, serialize)
        )
        # Wrapped LAST so the inner shims are already installed when the block
        # runs, and undone FIRST (undo walks in reverse).
        undos.append(_wrap_child(mx, layer, "mlp", "moe_block", recorder, serialize))

    undos.append(
        _wrap_gather_qmm(mx, recorder, serialize, int(args.moe_intermediate_size))
    )

    def undo():
        for fn in reversed(undos):
            fn()

    return undo


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_table(per_chunk_families: dict, chunk_sizes: list[int]) -> str:
    """Per-family table with the chunk-width delta in the last column."""

    gib = 1024**3
    lines = []
    widths = [str(c) for c in chunk_sizes]
    header = f"{'family':<14}" + "".join(f"{w + ' peakΔ':>16}" for w in widths)
    if len(chunk_sizes) == 2:
        header += f"{'delta':>16}"
    lines.append(header)
    lines.append("-" * len(header))
    for family in FAMILIES:
        cells = []
        for chunk in chunk_sizes:
            stats = per_chunk_families.get(chunk, {}).get(family)
            cells.append(0 if stats is None else stats["peak_delta_max"])
        row = f"{family:<14}" + "".join(f"{c / gib:>15.3f}G" for c in cells)
        if len(chunk_sizes) == 2:
            row += f"{(cells[1] - cells[0]) / gib:>15.3f}G"
        lines.append(row)
    return "\n".join(lines)


def load_tokens(path: Path | None, count: int, vocab: int, seed: int) -> np.ndarray:
    if path is not None and path.exists():
        ids = np.load(path).astype(np.int64).reshape(-1)
        if ids.size < count:
            raise SystemExit(f"{path} has {ids.size} tokens, need {count}")
        return ids[:count]
    print(
        "[census] WARNING: no token file; using random ids.  Routing balance "
        "and PLE hit rates will not match production.",
        flush=True,
    )
    rng = np.random.default_rng(seed)
    return rng.integers(1, vocab, size=count, dtype=np.int64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--tokens-npy", type=Path, default=DEFAULT_TOKENS)
    p.add_argument("--prompt-tokens", type=int, default=16384)
    p.add_argument("--chunk-sizes", type=str, default="2048,4096")
    p.add_argument(
        "--modes",
        type=str,
        default="lazy,serialized",
        help="lazy = production schedule (true peak); serialized = attribution",
    )
    p.add_argument(
        "--chunks",
        type=int,
        default=0,
        help="stop after N chunks (0 = the whole prompt).  The peak is on the "
        "LAST chunk, so a truncated run under-reports it.",
    )
    p.add_argument("--seed", type=int, default=20260829)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--allow-unlocked", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def self_test() -> int:
    """Recorder + report with a fake allocator.  No MLX, no model, no lock."""

    state = {"active": 0, "peak": 0}

    def bump(delta):
        state["active"] += delta
        state["peak"] = max(state["peak"], state["active"])

    rec = MemoryRecorder(lambda: state["active"], lambda: state["peak"])
    with rec.bracket("moe_down"):
        bump(1000)  # transient
        bump(-600)  # freed inside the block
    assert rec.calls["moe_down"] == 1
    assert rec.active_delta_sum["moe_down"] == 400
    assert rec.peak_delta_max["moe_down"] == 1000, rec.peak_delta_max["moe_down"]
    assert rec.active_high_water["moe_down"] == 400

    with rec.bracket("moe_down"):
        bump(100)
    assert rec.active_delta_sum["moe_down"] == 500
    assert rec.peak_delta_max["moe_down"] == 1000  # the earlier reach still stands
    assert rec.active_delta_max["moe_down"] == 400

    rec.note_bytes("moe_down", 4096)
    dumped = rec.as_dict()
    assert dumped["moe_down"]["bytes_out"] == 4096
    assert dumped["moe_down"]["calls"] == 2

    table = format_table(
        {2048: {"moe_down": {"peak_delta_max": 1 << 30}},
         4096: {"moe_down": {"peak_delta_max": 3 << 30}}},
        [2048, 4096],
    )
    assert "moe_down" in table and "2.000G" in table, table
    print("[self-test] ok: recorder brackets, byte notes, delta table")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    require_guard_window(args.allow_unlocked)
    print(
        "[micro-prefill-memory-census] must run under "
        "/tmp/mtplx-gpu-exclusive.lock",
        flush=True,
    )

    chunk_sizes = [int(c) for c in args.chunk_sizes.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in ("lazy", "serialized"):
            raise SystemExit(f"unknown mode {mode!r}")

    from mtplx.server.openai import _apply_metal_memory_caps

    # Same startup contract as the benchmark driver: the allocator caps are
    # pinned BEFORE the model loads.  Never raised -- a run that grows the
    # wired pool past the box's limit collapses ~4x and can take the box with
    # it (memory: never-exceed-the-memory-knob).
    caps = _apply_metal_memory_caps(minimum_resident_bytes=MINIMUM_RESIDENT_BYTES)
    if not caps.get("applied"):
        raise SystemExit(f"Metal memory caps did not apply: {caps}")
    print(f"[census] metal caps {caps}", flush=True)

    import mlx.core as mx

    from mtplx.generation import prefill_chunk_size_override
    from mtplx.runtime import load

    runtime = load(args.model, mtp=True)
    model = runtime.model
    text = resolve_text_model(model)
    vocab = int(text.args.vocab_size)
    ids = load_tokens(args.tokens_npy, args.prompt_tokens, vocab, args.seed)
    print(f"[census] prompt {ids.size} tokens from {args.tokens_npy}", flush=True)

    mx.eval(mx.array([0]))
    after_load = mx.get_active_memory()
    print(f"[census] active after load {after_load / 1024**3:.3f} GiB", flush=True)

    results: dict[str, dict] = {}
    per_chunk_families: dict[int, dict] = {}
    for chunk in chunk_sizes:
        for mode in modes:
            serialize = mode == "serialized"
            recorder = MemoryRecorder(mx.get_active_memory, mx.get_peak_memory)
            cache = runtime.model.make_cache()
            mx.clear_cache()
            mx.reset_peak_memory()
            base_active = mx.get_active_memory()
            undo = instrument(mx, model, recorder, serialize)
            chunk_rows = []
            started = time.perf_counter()
            try:
                spans = [
                    (s, min(ids.size, s + chunk))
                    for s in range(0, ids.size, chunk)
                ]
                if args.chunks:
                    spans = spans[: args.chunks]
                with prefill_chunk_size_override(chunk):
                    for start, end in spans:
                        block = mx.array(ids[start:end].reshape(1, -1))
                        t0 = time.perf_counter()
                        peak_before = mx.get_peak_memory()
                        out = runtime.forward_ar(
                            block,
                            cache=cache,
                            return_hidden=True,
                            hidden_variant="post_norm",
                            emit_logits=False,
                        )
                        _eval_tree(mx, out)
                        chunk_rows.append(
                            {
                                "start": start,
                                "end": end,
                                "wall_s": time.perf_counter() - t0,
                                "peak_before": peak_before,
                                "peak_after": mx.get_peak_memory(),
                                "active_after": mx.get_active_memory(),
                                "cache_after": mx.get_cache_memory(),
                            }
                        )
                        del out
            finally:
                undo()
            wall = time.perf_counter() - started
            key = f"{chunk}:{mode}"
            results[key] = {
                "chunk_size": chunk,
                "mode": mode,
                "wall_s": wall,
                "tokens": int(spans[-1][1]),
                "base_active": base_active,
                "peak_bytes": mx.get_peak_memory(),
                "active_bytes": mx.get_active_memory(),
                "chunks": chunk_rows,
                "families": recorder.as_dict(),
            }
            if serialize:
                per_chunk_families[chunk] = recorder.as_dict()
            print(
                f"[census] chunk={chunk} mode={mode} wall={wall:.2f}s "
                f"peak={mx.get_peak_memory() / 1024**3:.3f} GiB "
                f"active={mx.get_active_memory() / 1024**3:.3f} GiB",
                flush=True,
            )
            del cache
            mx.clear_cache()

    print()
    for chunk in chunk_sizes:
        lazy = results.get(f"{chunk}:lazy")
        ser = results.get(f"{chunk}:serialized")
        if lazy and ser:
            gap = lazy["peak_bytes"] - ser["peak_bytes"]
            print(
                f"[concurrency] chunk={chunk}: lazy peak "
                f"{lazy['peak_bytes'] / 1024**3:.3f} GiB - serialized "
                f"{ser['peak_bytes'] / 1024**3:.3f} GiB = "
                f"{gap / 1024**3:.3f} GiB held by scheduler concurrency"
            )
    if per_chunk_families:
        print()
        print(format_table(per_chunk_families, chunk_sizes))

    payload = {
        "model": str(args.model),
        "prompt_tokens": int(ids.size),
        "chunk_sizes": chunk_sizes,
        "modes": modes,
        "after_load_active": after_load,
        "runs": results,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\n[out] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
