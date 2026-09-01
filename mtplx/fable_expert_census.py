"""Opt-in diagnostic census of the physical-M4 routed expert selection.

Why
---
The Qwen3.8 Flash-Next verifier runs ``M=4`` physical rows through 48 MoE
layers.  Every row independently picks ``top_k=10`` of 512 experts, so a
layer names up to ``4 * 10 = 40`` routed expert slices per cycle -- and an
expert named by two rows is currently streamed from memory once per row.
This census answers the only question that prices the dedup lever: how
many DISTINCT experts do the four rows select, per layer, per cycle?

Cost when off
-------------
``MTPLX_FABLE_EXPERT_CENSUS`` is read exactly once, at import.  When it is
unset :data:`_ENABLED` is ``False``, ``census.enabled`` is ``False``, and
every method returns on its first statement.  Call sites additionally
guard on ``census.enabled`` before touching anything else, so the retained
perf path pays one attribute load and one predicted-not-taken branch per
MoE layer and nothing else -- no ``getattr`` default lookup, no argument
tuple, no call frame.

Compiled-graph caveat -- READ BEFORE RUNNING A CENSUS
-----------------------------------------------------
The retained fixed-M4 verify forward runs inside ``mx.compile``
(``CompiledVerifyBank._make_verify_step`` -> ``mx.compile`` ->
``_fixed_m4_dispatch["fn"]``).  Inside a compiled function the Python body
executes at TRACE time only; replays skip it.  A side-effect hook such as
:meth:`ExpertCensus.record` would therefore capture the trace's expert ids
once and nothing after that.

Three routes were considered:

1. Return the stacked ids as extra compiled outputs.  Correct, but it
   threads a diagnostic tensor through ``_make_verify_step``,
   ``_unpack_fixed_m4_outputs``, the split prefix/suffix graphs and the
   capture layout -- a large edit to the retained perf path for a
   throwaway measurement.
2. Route the diagnostic through an uncompiled verifier.  Not available:
   ``MTPLX_COMPILED_VERIFY`` accepts ``off|on|parity|parity2``, and both
   ``install_fixed_m4`` (parity modes refused) and the PR391 float32
   verifier gate ("PR391 float32 verifier requires an installed physical
   -M4 verifier") demand mode ``on``.  There is no uncompiled fixed-M4
   route.
3. **Chosen:** disable MLX compilation for the diagnostic run with
   ``MLX_DISABLE_COMPILE=1``.  MLX's ``compile()`` returns the original
   callable when compilation is disabled, so ``verify_step`` -- and every
   M4 MoE forward inside it -- runs eagerly on every call and the Python
   side effects fire per call, with correct ids.  The diagnostic run is
   slower than the retained lane; that does not matter, because this
   measures ROUTING, which is bit-identical either way.

So a census run is::

    MLX_DISABLE_COMPILE=1 MTPLX_FABLE_EXPERT_CENSUS=/path/census.npz <bench>

Enabling the census without ``MLX_DISABLE_COMPILE`` warns on stderr at
startup, and the cycle count printed by :meth:`flush` (plus the ``cycles``
dimension the report prints) is the confirmation that the run captured
per-call ids rather than a single trace.

Cycle framing
-------------
:meth:`record` buffers one (layer index, lazy expert-id array) pair per MoE
layer without evaluating anything.  :meth:`end_cycle`, hooked where a
verify window completes, issues ONE ``mx.eval`` over the whole buffer and
converts to ``int16``.  Cycles whose layer roster differs from the modal
roster are dropped at :meth:`flush` -- that is what discards the
construction-time self-check calls in ``install_qwen4_m4_stage3``, which
run ``_m4_forward`` per layer before any verify window exists.

Output
------
``flush()`` (also registered with ``atexit`` when enabled) writes
``numpy.savez_compressed`` with:

* ``ids``       -- ``int16 [cycles, layers, 4, 10]`` selected expert ids
* ``layer_ids`` -- ``int16 [layers]`` the layer index of each ``ids`` slot

The env var may name either a ``.npz`` or a ``.json`` path.  The arrays
always land in the ``.npz`` sibling; a ``.json`` path additionally gets a
small metadata document pointing at it.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any

import numpy as np

_ENV_VAR = "MTPLX_FABLE_EXPERT_CENSUS"

_TOP_K = 10
_ROWS = 4


def _env_path() -> str | None:
    raw = (os.environ.get(_ENV_VAR) or "").strip()
    return raw or None


def _npz_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return path if ext == ".npz" else root + ".npz"


class ExpertCensus:
    """Buffer M4 expert ids per cycle and write them out once at the end."""

    def __init__(self, path: str | None = None) -> None:
        self.configure(path)

    # -- lifecycle ---------------------------------------------------
    def configure(self, path: str | None) -> None:
        """(Re)point the census.  Also the test seam; not a hot path."""

        self.path = path
        self.enabled = path is not None
        if self.enabled and not (os.environ.get("MLX_DISABLE_COMPILE") or "").strip():
            print(
                "[fable-expert-census] MLX_DISABLE_COMPILE is unset: the M4 "
                "verify forward runs inside mx.compile, so record() will fire "
                "at trace time only and the census will be near-empty. "
                "Re-run with MLX_DISABLE_COMPILE=1.",
                file=sys.stderr,
            )
        self._pending_layers: list[int] = []
        self._pending_ids: list[Any] = []
        self._cycles: list[tuple[tuple[int, ...], np.ndarray]] = []
        self.dropped_cycles = 0

    # -- recording ---------------------------------------------------
    def record(self, layer_idx: int, expert_ids: Any) -> None:
        """Buffer one layer's lazy expert-id array.  Never evaluates."""

        if not _ENABLED or not self.enabled:
            return
        self._pending_layers.append(int(layer_idx))
        self._pending_ids.append(expert_ids)

    def end_cycle(self) -> None:
        """Close the current cycle: one ``mx.eval`` over the whole buffer."""

        if not _ENABLED or not self.enabled:
            return
        pending = self._pending_ids
        if not pending:
            return
        layers = tuple(self._pending_layers)
        self._pending_ids = []
        self._pending_layers = []

        import mlx.core as mx  # local: keeps this module import-cheap/pure

        mx.eval(pending)
        rows = np.stack(
            [
                np.asarray(ids).reshape(_ROWS, _TOP_K).astype(np.int16)
                for ids in pending
            ]
        )
        self._cycles.append((layers, rows))

    # -- output ------------------------------------------------------
    def _modal_group(self):
        counts: dict[tuple[int, ...], int] = {}
        for layers, _rows in self._cycles:
            counts[layers] = counts.get(layers, 0) + 1
        if not counts:
            return None, []
        modal = max(counts, key=lambda key: (counts[key], len(key)))
        kept = [rows for layers, rows in self._cycles if layers == modal]
        self.dropped_cycles = len(self._cycles) - len(kept)
        return modal, kept

    def flush(self) -> str | None:
        """Write the census out.  Safe to call repeatedly; no-op when off."""

        if not _ENABLED or not self.enabled:
            return None
        self.end_cycle()
        modal, kept = self._modal_group()
        if modal is None or not kept:
            print(
                f"[fable-expert-census] nothing recorded for {self.path!r}; "
                "did you forget MLX_DISABLE_COMPILE=1?",
                file=sys.stderr,
            )
            return None
        ids = np.stack(kept).astype(np.int16)
        layer_ids = np.asarray(modal, dtype=np.int16)
        out = _npz_path(self.path)
        directory = os.path.dirname(os.path.abspath(out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(out, ids=ids, layer_ids=layer_ids)
        if self.path.endswith(".json"):
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "npz": out,
                        "cycles": int(ids.shape[0]),
                        "layers": int(ids.shape[1]),
                        "rows": int(ids.shape[2]),
                        "top_k": int(ids.shape[3]),
                        "dropped_cycles": int(self.dropped_cycles),
                    },
                    handle,
                    indent=2,
                )
        print(
            f"[fable-expert-census] wrote {out} "
            f"ids={tuple(int(dim) for dim in ids.shape)} "
            f"dropped_cycles={self.dropped_cycles}",
            file=sys.stderr,
        )
        return out


_CENSUS_PATH = _env_path()
_ENABLED = _CENSUS_PATH is not None

census = ExpertCensus(_CENSUS_PATH)

if _ENABLED:
    atexit.register(census.flush)


def _configure_for_test(path: str | None) -> None:
    """Re-point the module-level singleton and enable flag (tests only)."""

    global _ENABLED, _CENSUS_PATH
    _CENSUS_PATH = path
    _ENABLED = path is not None
    census.configure(path)


__all__ = ["ExpertCensus", "census"]
