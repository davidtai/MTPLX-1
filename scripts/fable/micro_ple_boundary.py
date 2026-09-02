#!/usr/bin/env python3
# GPU WINDOW REQUIRED.  This script issues no Metal work and imports no MLX,
# but it READS THE REAL 32 GB n-gram table's pages, which moves page-cache
# state a live benchmark depends on.  Run it inside the window, as a child of
# bench/laguna/run_guarded.py, exactly like a Metal micro.
"""Host attribution of the decode-cycle PLE boundary, stock vs each W62 item.

What this measures, and why it is the right measurement
-------------------------------------------------------
The 2410 census puts 4.002 ms/cycle of GPU idle on the PLE boundary, 87.1 %
of it host-late, and splits it as::

    gap A  (D3 id copy -> PLE dequant)    1890.7 us host + 22.0 enc + 362.8 submit
    gap B  (PLE dequant -> target gather) 1640.8 us host + 15.7 enc + 166.5 submit

Gap A is the host running the n-gram row arithmetic, the sidecar row read and
the MLX array construction.  This micro times the row read -- the part
``MTPLX_FABLE_PLE_BOUNDARY`` changes -- against the real table, at the real
decode geometry, arm by arm.  Gap B is the compiled verify graph's own host
construction; nothing here touches it (arm it with
``MTPLX_FABLE_PLE_BOUNDARY_ITEMS=timing`` in a real run instead).

The stock arm is not a transcription
------------------------------------
``_SidecarGather._rows_matrices``, ``_stack_hot_rows`` and
``_bind_fixed_m4_owned_row_prefetch`` are lifted out of the shipped SOURCE with
``ast`` and executed in a NumPy-only namespace, so the control arm is the code
production runs and the micro breaks loudly if either is edited underneath it.
That is also what keeps MLX out of this process entirely.

Fairness
--------
Every arm's missing rows come from one shared, never-repeating draw, so no arm
reads pages another arm warmed.  Arms are interleaved round-robin over reps,
so thermal or page-cache drift falls on all of them equally.  The measured
residency (``mincore``) is printed with the result: an arm set measured on a
COLD table is a different experiment from one measured after
``--prewarm-ngram-table``, and the receipt has to say which.

RUN IT (guarded)::

    W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w62-ple-boundary
    PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
    env PYTHONPATH=$W $PY /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \\
        --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \\
        --lock-timeout-seconds 3600 --child-timeout-seconds 900 \\
        -- env PYTHONPATH=$W $PY $W/scripts/fable/micro_ple_boundary.py \\
            --reps 200 --json $W/.benchmark-artifacts/fable/micro-ple-boundary.json

``--self-test`` runs the same arms over a synthetic 64 MB temp file and needs
no window, no model and no lock.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import struct
import sys
import tempfile
import textwrap
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QWEN4_EXP = ROOT / "mtplx" / "models" / "qwen4_exp.py"
FIXED_VERIFY = ROOT / "mtplx" / "qwen4_fixed_verify.py"

LOCK_PATH = "/tmp/mtplx-gpu-exclusive.lock"
GUARD_ATTEST_FD_ENV = "MTPLX_GUARD_ATTEST_FD"
BANNER = (
    "[micro_ple_boundary] GPU WINDOW REQUIRED -- this reads the real n-gram "
    f"table's pages; run under {LOCK_PATH} via bench/laguna/run_guarded.py "
    "(or pass --allow-unguarded, or --self-test)"
)

DEFAULT_MODEL_DIR = Path(
    "/Users/davidtai/.mtplx/models/"
    "Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
)
TABLE_NAME = "ngram-table.safetensors"
NAMES = ("weight", "scales", "biases")

#: The fixed-M4 decode window: 4 positions x (ngram_size-1) x heads_per_ngram
#: = 4 x 16 = 64 table rows, of which position 0's 16 are prefetched.
WINDOW_ROWS = 64
PRIMARY_ROWS = 16
#: Census/ledger figure: 85-93 % of a cycle's rows have never been seen, so
#: the hot LRU serves the rest.  The default sits in the middle.
DEFAULT_NOVEL = 0.89


# --------------------------------------------------------------------------
# Lifting the shipped source (no MLX import)
# --------------------------------------------------------------------------
def _lift(path: Path, names, *, klass: str | None = None, namespace=None):
    source = path.read_text()
    tree = ast.parse(source)
    scope = tree.body
    if klass is not None:
        scope = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == klass
        )
    found = {
        node.name: textwrap.dedent(ast.get_source_segment(source, node))
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = [name for name in names if name not in found]
    if missing:
        raise SystemExit(f"{path.name} no longer defines {missing}")
    if namespace is None:
        namespace = {"np": np, "os": os}
    for name in names:
        exec(compile(ast.parse(found[name]), str(path), "exec"), namespace)
    return namespace


def shipped_namespace():
    namespace = {"np": np, "os": os}
    _lift(QWEN4_EXP, ["_stack_hot_rows"], namespace=namespace)
    _lift(QWEN4_EXP, ["_rows_matrices"], klass="_SidecarGather", namespace=namespace)
    text = ast.get_source_segment(
        FIXED_VERIFY.read_text(),
        next(
            node
            for node in ast.parse(FIXED_VERIFY.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_bind_fixed_m4_owned_row_prefetch"
        ),
    )
    if "ple_boundary" in text:
        raise SystemExit(
            "the W62 swap moved INTO _bind_fixed_m4_owned_row_prefetch; "
            "this micro's control arm is no longer the shipped code"
        )
    exec(compile(text, str(FIXED_VERIFY), "exec"), namespace)
    return namespace


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------
def read_header(path: Path):
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return header, 8 + length


class Sidecar:
    """Every attribute the shipped hot-row branch and the W62 lane touch."""

    _HOT_PATH_MAX_ROWS = 4096

    def __init__(self, path: Path, *, hot_cap_rows: int, workers: int = 16):
        from concurrent.futures import ThreadPoolExecutor

        header, data_start = read_header(path)
        self.path = path
        self._maps = {}
        self._row_meta = []
        self.n_rows = None
        for name in NAMES:
            info = header[f"ngram.{name}"]
            dtype = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16}[
                info["dtype"]
            ]
            shape = tuple(info["shape"])
            offset = data_start + info["data_offsets"][0]
            memmap = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)
            self._maps[name] = (memmap, info["dtype"])
            itemsize = 4 if info["dtype"] == "U32" else 2
            self._row_meta.append((offset, int(shape[1]) * itemsize))
            self.n_rows = int(shape[0]) if self.n_rows is None else self.n_rows
        self._fd = os.open(str(path), os.O_RDONLY)
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="ngram-prefetch")
        self._hot: OrderedDict = OrderedDict()
        self._hot_row_bytes = max(1, sum(rb for _, rb in self._row_meta))
        self._hot_cap_rows = hot_cap_rows
        self.hot_hits = 0
        self.hot_misses = 0
        self.prefetch_batches = 0
        self.lookahead_batches = 0
        self.vectorized_gathers = 0
        self.pread_gathers = 0

    # Verbatim in shape from `_SidecarGather._warm` / `_submit_warm`.
    def _warm(self, rows, *, counted: bool = True) -> None:
        for future in self._submit_warm(rows, counted=counted):
            future.result()

    def _submit_warm(self, rows, *, counted: bool):
        fd = self._fd
        metas = self._row_meta

        def touch(chunk):
            for r in chunk:
                for base, rb in metas:
                    os.pread(fd, rb, base + int(r) * rb)

        step = max(1, min(64, (len(rows) + 31) // 32))
        chunks = [rows[i : i + step] for i in range(0, len(rows), step)]
        futures = tuple(self._pool.submit(touch, chunk) for chunk in chunks)
        if counted:
            self.prefetch_batches += 1
        else:
            self.lookahead_batches += 1
        return futures

    def reset(self):
        self._hot.clear()
        self.hot_hits = self.hot_misses = 0

    def close(self):
        self._pool.shutdown(wait=True)
        os.close(self._fd)


def build_synthetic(path: Path, n_rows: int) -> None:
    """A safetensors file at the production row geometry, for --self-test."""

    widths = {"weight": 20, "scales": 5, "biases": 5}
    dtypes = {"weight": "U32", "scales": "BF16", "biases": "BF16"}
    itemsize = {"U32": 4, "BF16": 2}
    header = {"__metadata__": {"ngram_bits": "4"}}
    offset = 0
    blobs = {}
    rng = np.random.default_rng(20260902)
    for name in NAMES:
        dtype = np.uint32 if dtypes[name] == "U32" else np.uint16
        high = 2**32 if dtype is np.uint32 else 2**16
        data = rng.integers(0, high, size=(n_rows, widths[name])).astype(dtype)
        size = data.nbytes
        header[f"ngram.{name}"] = {
            "dtype": dtypes[name],
            "shape": [n_rows, widths[name]],
            "data_offsets": [offset, offset + size],
        }
        blobs[name] = data
        offset += size
        assert itemsize[dtypes[name]] * widths[name] * n_rows == size
    blob = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for name in NAMES:
            handle.write(blobs[name].tobytes())


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------
def make_arms(sidecar, shipped, boundary):
    """``{name: (setup, run)}``.  Every arm takes one cycle's row ids."""

    from mtplx import ple_boundary as _pb  # noqa: F401  (module identity)

    stock_rows = shipped["_rows_matrices"]

    def stock(flat):
        return stock_rows(sidecar, flat, NAMES)

    def armed(skip_warm, block):
        def run(flat):
            return boundary._boundary_rows_matrices(
                sidecar,
                stock_rows,
                flat,
                NAMES,
                skip_warm=skip_warm,
                block=block,
                timing=False,
                stack_hot_rows=shipped["_stack_hot_rows"],
            )

        return run

    shipped_submit, _missing, shipped_install = shipped[
        "_bind_fixed_m4_owned_row_prefetch"
    ](sidecar)
    lane_submit, lane_install, _line = boundary.bind_owned_row_prefetch(
        sidecar, submit_primary=shipped_submit, install=shipped_install,
        names=NAMES,
    )

    def primary_stock(flat):
        shipped_install(shipped_submit(flat))
        return None

    def primary_lane(flat):
        lane_install(lane_submit(flat))
        return None

    return {
        "gather:stock": (WINDOW_ROWS, stock),
        "gather:warm_skip": (WINDOW_ROWS, armed(True, False)),
        "gather:hot_block": (WINDOW_ROWS, armed(False, True)),
        "gather:warm_skip+hot_block": (WINDOW_ROWS, armed(True, True)),
        "primary:stock": (PRIMARY_ROWS, primary_stock),
        "primary:primary_vectorized": (PRIMARY_ROWS, primary_lane),
    }


def measure(sidecar, arms, *, reps, novel, seed, resident_sample):
    """Round-robin the arms over `reps` cycles, disjoint miss rows throughout."""

    from mtplx.ple_row_gather import resident_fraction

    rng = np.random.default_rng(seed)
    order = sorted(arms)
    samples = {name: [] for name in order}
    residency = {name: [] for name in order}
    n_rows = sidecar.n_rows
    # One never-repeating stream of miss rows, shared by every arm: an arm can
    # never read a page another arm warmed.
    drawn: set[int] = set()
    #: The same ids in draw order.  Sampling "rows an earlier cycle read" from
    #: here instead of from `sidecar._hot.keys()` keeps the untimed part of the
    #: loop O(1) as the LRU grows -- with a 3.5M-row cap nothing is ever
    #: evicted, so the two are the same set.
    history: list[int] = []

    def fresh(count):
        out = []
        while len(out) < count:
            candidate = int(rng.integers(0, n_rows))
            if candidate in drawn:
                continue
            drawn.add(candidate)
            history.append(candidate)
            out.append(candidate)
        return out

    for rep in range(reps):
        for name in order:
            width, run = arms[name]
            misses = fresh(max(1, int(round(width * novel))))
            hits = []
            warm = len(history) - len(misses)
            if warm > 0:
                take = min(width - len(misses), warm)
                hits = [history[int(rng.integers(0, warm))] for _ in range(take)]
            flat = np.asarray(
                (misses + hits + misses)[:width], dtype=np.int64
            )
            rng.shuffle(flat)
            if rep == 0 and resident_sample:
                fractions = [
                    resident_fraction(sidecar._maps[n][0], flat, sample=resident_sample)
                    for n in NAMES
                ]
                residency[name] = [f for f in fractions if f is not None]
            started = time.perf_counter()
            run(flat)
            samples[name].append((time.perf_counter() - started) * 1000.0)
    return samples, residency


def summarize(samples, residency, *, drop):
    rows = []
    for name in sorted(samples):
        values = sorted(samples[name])[: max(1, len(samples[name]) - drop)]
        rows.append(
            {
                "arm": name,
                "reps": len(samples[name]),
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "p10_ms": values[len(values) // 10],
                "p90_ms": values[len(values) * 9 // 10],
                "min_ms": values[0],
                "resident_fraction": (
                    min(residency[name]) if residency.get(name) else None
                ),
            }
        )
    return rows


def report(rows) -> str:
    by = {row["arm"]: row for row in rows}
    lines = [
        f"{'arm':32} {'median ms':>10} {'mean ms':>9} {'p90 ms':>8} "
        f"{'min ms':>8} {'resident':>9}",
    ]
    for row in rows:
        resident = (
            "n/a" if row["resident_fraction"] is None
            else f"{row['resident_fraction']:.3f}"
        )
        lines.append(
            f"{row['arm']:32} {row['median_ms']:10.4f} {row['mean_ms']:9.4f} "
            f"{row['p90_ms']:8.4f} {row['min_ms']:8.4f} {resident:>9}"
        )
    lines.append("")
    lines.append("deltas vs the shipped arm (median ms/cycle, negative = removed):")
    for base, arm in (
        ("gather:stock", "gather:warm_skip"),
        ("gather:stock", "gather:hot_block"),
        ("gather:stock", "gather:warm_skip+hot_block"),
        ("primary:stock", "primary:primary_vectorized"),
    ):
        if base in by and arm in by:
            delta = by[arm]["median_ms"] - by[base]["median_ms"]
            lines.append(
                f"  {arm:32} {delta:+9.4f}  ({delta / 37.4 * 100:+.2f} % of a "
                "37.4 ms window)"
            )
    if "gather:warm_skip" in by and "primary:primary_vectorized" in by:
        # The DEFAULT arm -- what MTPLX_FABLE_PLE_BOUNDARY=1 alone runs --
        # is warm_skip + primary_vectorized.  hot_block is selectable but out
        # of it, so it must not be folded into the headline total.
        total = (
            by["gather:warm_skip"]["median_ms"] - by["gather:stock"]["median_ms"]
        ) + (
            by["primary:primary_vectorized"]["median_ms"] - by["primary:stock"]["median_ms"]
        )
        lines.append(
            f"  {'DEFAULT ARM (warm_skip+primary)':32} {total:+9.4f}  "
            f"({total / 37.4 * 100:+.2f} % of a 37.4 ms window; the census puts "
            "4.002 ms/cycle on this boundary)"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--drop-slowest", type=int, default=0,
                        help="trim this many slowest reps per arm before summarising")
    parser.add_argument("--novel", type=float, default=DEFAULT_NOVEL,
                        help="fraction of a cycle's rows that miss the hot LRU")
    parser.add_argument("--hot-cap-rows", type=int, default=3_500_000,
                        help="MTPLX_NGRAM_HOT_MB=1024 at 300 B/row")
    parser.add_argument("--probe-rows", type=int, default=8)
    parser.add_argument("--resident-sample", type=int, default=64,
                        help="mincore draws used to report the table's residency")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true",
                        help="synthetic table, no window and no model needed")
    parser.add_argument("--self-test-rows", type=int, default=200_000)
    parser.add_argument("--allow-unguarded", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.novel <= 1.0:
        raise SystemExit("--novel must be in (0, 1]")
    if not args.self_test and not args.allow_unguarded:
        if not os.environ.get(GUARD_ATTEST_FD_ENV):
            print(BANNER, file=sys.stderr)
            return 2

    # The lane is configured for the whole process; each arm selects its items
    # by calling `_boundary_rows_matrices` directly, so this only fixes the
    # probe width the armed paths use.
    os.environ.setdefault("MTPLX_FABLE_PLE_BOUNDARY", "1")
    os.environ["MTPLX_FABLE_PLE_BOUNDARY_PROBE_ROWS"] = str(args.probe_rows)
    from mtplx import ple_boundary as boundary

    boundary.enabled.cache_clear()
    boundary.items.cache_clear()
    boundary.probe_rows.cache_clear()

    shipped = shipped_namespace()

    temporary = None
    if args.self_test:
        temporary = tempfile.TemporaryDirectory()
        table = Path(temporary.name) / TABLE_NAME
        build_synthetic(table, args.self_test_rows)
    else:
        table = args.table or (args.model_dir / TABLE_NAME)
        if not table.is_file():
            raise SystemExit(f"n-gram table not found: {table}")

    sidecar = Sidecar(table, hot_cap_rows=args.hot_cap_rows)
    try:
        arms = make_arms(sidecar, shipped, boundary)
        samples, residency = measure(
            sidecar, arms, reps=args.reps, novel=args.novel, seed=args.seed,
            resident_sample=args.resident_sample,
        )
        rows = summarize(samples, residency, drop=args.drop_slowest)
        payload = {
            "table": str(table),
            "table_rows": sidecar.n_rows,
            "self_test": bool(args.self_test),
            "reps": args.reps,
            "novel_fraction": args.novel,
            "probe_rows": args.probe_rows,
            "hot_cap_rows": args.hot_cap_rows,
            "window_rows": WINDOW_ROWS,
            "primary_rows": PRIMARY_ROWS,
            "seed": args.seed,
            "census_ple_boundary_ms_per_cycle": 4.0021,
            "arms": rows,
        }
        print(report(rows))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2))
            print(f"\nwrote {args.json}")
    finally:
        sidecar.close()
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
