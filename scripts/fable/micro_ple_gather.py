#!/usr/bin/env python3
"""Host-only attribution of the PLE prefill gather: I/O, syscall, or NumPy?

No MLX, no GPU, no flock, no model load -- this touches nothing the GPU
benchmark lock protects, and it never reads the real 32 GB n-gram table (which
would evict page cache under a live benchmark).  It rebuilds
``_SidecarGather``'s big-gather branch over a SYNTHETIC file at the production
row geometry and times the three stages separately.

Geometry (Qwen3.8 Flash-Next, config.json): ``ngram_size`` 3 x
``heads_per_ngram`` 8 = 16 rows per token, ``ple_embed_dim`` 2560 / 16 = 160
per head, q4/g32 -> weight ``uint32[20]`` (80 B), scales ``bf16[5]`` (10 B),
biases ``bf16[5]`` (10 B) = 100 B/row.  One 2,048-token prefill chunk asks for
16 x 2,048 = 32,768 rows.

The result that matters (M5 Max, 2026-09-01, warm page cache, one map)::

    shipped warm (step<=64, 512 tasks)   164.8 ms   memmap fancy index  0.62 ms
    16 big batches                       175.6 ms                       0.52 ms
    one pread per distinct page          183.6 ms                       0.92 ms
    no warm, memmap only                   0.0 ms                       0.44 ms

``np.unique`` on the 32,768 ids is 1.4 ms.  Times three maps, that reproduces
the 157-346 ms per-chunk host-late stalls the census measures -- and it is
~5 us of GIL-contended Python per ``os.pread``, not disk and not NumPy.  No
batching formulation moves it, which is why MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD
overlaps the pass instead of trying to make it cheaper.

Usage::

    python scripts/fable/micro_ple_gather.py [--rows 32768] [--reps 5]
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROW_U32 = 20  # weight row: 160 head dims * 4 bits / 32
PAGE = 16 * 1024
RB = ROW_U32 * 4


def build(path: str, n_rows: int) -> None:
    if os.path.exists(path) and os.path.getsize(path) == n_rows * RB:
        return
    rng = np.random.default_rng(20260901)
    with open(path, "wb") as fh:
        step = 1 << 20
        for start in range(0, n_rows, step):
            k = min(step, n_rows - start)
            fh.write(rng.integers(0, 2**32, size=k * ROW_U32, dtype=np.uint32))


def shipped_warm(pool, fd, rows) -> None:
    """`_SidecarGather._submit_warm` + `_warm`, verbatim in shape."""

    def touch(chunk):
        for r in chunk:
            os.pread(fd, RB, int(r) * RB)

    step = max(1, min(64, (len(rows) + 31) // 32))
    chunks = [rows[i : i + step] for i in range(0, len(rows), step)]
    for future in tuple(pool.submit(touch, c) for c in chunks):
        future.result()


def big_batches(pool, fd, rows, workers: int = 16) -> None:
    """Same preads, ~`workers` tasks, offsets precomputed as plain ints."""

    offsets = (rows.astype(np.int64) * RB).tolist()
    step = max(1, (len(offsets) + workers - 1) // workers)

    def touch(lo, hi):
        pread = os.pread
        for i in range(lo, hi):
            pread(fd, RB, offsets[i])

    futures = [
        pool.submit(touch, i, min(i + step, len(offsets)))
        for i in range(0, len(offsets), step)
    ]
    for future in futures:
        future.result()


def page_dedup(pool, fd, rows, workers: int = 16) -> None:
    """One pread per distinct 16 KiB page instead of per row."""

    pages = (np.unique((rows.astype(np.int64) * RB) // PAGE) * PAGE).tolist()
    step = max(1, (len(pages) + workers - 1) // workers)

    def touch(lo, hi):
        pread = os.pread
        for i in range(lo, hi):
            pread(fd, 1, pages[i])

    futures = [
        pool.submit(touch, i, min(i + step, len(pages)))
        for i in range(0, len(pages), step)
    ]
    for future in futures:
        future.result()


def no_warm(pool, fd, rows) -> None:
    """Let the memmap fancy index fault on its own."""


VARIANTS = (
    ("shipped (step<=64, 512 tasks)", shipped_warm),
    ("big batches (16 tasks)", big_batches),
    ("page-dedup preads", page_dedup),
    ("no warm (memmap only)", no_warm),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=32_768,
        help="rows one prefill chunk requests (default 32768 = 2048 x 16)",
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to build the synthetic file (default: a temp dir)",
    )
    args = parser.parse_args()

    stride = PAGE // RB  # one touched row per 16 KiB page, as in the real table
    n_rows = args.rows * stride
    tmp = None
    if args.out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="mtplx-ple-micro-")
        out_dir = tmp.name
    else:
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ple-micro-weight.bin")
    print(
        f"synthetic map: {n_rows} rows x {RB} B = "
        f"{n_rows * RB / 2**20:.0f} MiB, one touched row per {PAGE // 1024} KiB page"
    )
    try:
        build(path, n_rows)
        mm = np.memmap(path, mode="r", dtype=np.uint32, shape=(n_rows, ROW_U32))
        rng = np.random.default_rng(7)
        ids = np.sort(
            rng.permutation(args.rows)[: args.rows].astype(np.int64) * stride
        )
        fd = os.open(path, os.O_RDONLY)
        pool = ThreadPoolExecutor(max_workers=16)
        np.asarray(mm[ids]).sum()  # prime: production prewarms the table
        try:
            t0 = time.perf_counter()
            for _ in range(args.reps):
                np.unique(ids, return_inverse=True)
            unique_ms = (time.perf_counter() - t0) / args.reps * 1e3
            print(f"np.unique on {len(ids)} ids: {unique_ms:.2f} ms")
            for name, fn in VARIANTS:
                warm_s = gather_s = 0.0
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    fn(pool, fd, ids)
                    t1 = time.perf_counter()
                    out = np.ascontiguousarray(mm[ids])
                    t2 = time.perf_counter()
                    warm_s += t1 - t0
                    gather_s += t2 - t1
                assert out.shape == (len(ids), ROW_U32)
                print(
                    f"{name:32s} warm={warm_s / args.reps * 1e3:7.2f} ms  "
                    f"gather={gather_s / args.reps * 1e3:6.2f} ms  "
                    f"per map={(warm_s + gather_s) / args.reps * 1e3:7.2f} ms"
                )
        finally:
            pool.shutdown(wait=True)
            os.close(fd)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
