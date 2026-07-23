#!/usr/bin/env python3
"""Re-shard a safetensors checkpoint into resident-only and expert-only groups.

Byte-level repack: tensor payloads are copied as raw ranges out of the source
shards, so dtypes (BF16, packed U32) round-trip exactly with no framework
load. Emits a fresh ``model.safetensors.index.json`` covering both groups and
a JSON report with per-tensor sha256 receipts.

Order of operations (crash-safe: the root stays consistent at every step):
  1. write all new shards (temp names, fsync, rename into place)
  2. verify: re-read every new shard, per-tensor sha256 vs the source hash,
     dtype/shape equality, contiguous non-overlapping offsets
  3. atomically replace the index
  4. move the retired source shards into --retire-dir (never deleted)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

CHUNK = 128 * 1024 * 1024
INDEX_NAME = "model.safetensors.index.json"


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path}: truncated safetensors header length")
        (n,) = struct.unpack("<Q", raw)
        header = json.loads(f.read(n))
    return header, 8 + n


def copy_range(src, dst, offset: int, length: int) -> str:
    src.seek(offset)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        chunk = src.read(min(CHUNK, remaining))
        if not chunk:
            raise SystemExit("unexpected EOF while copying tensor payload")
        digest.update(chunk)
        dst.write(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def hash_range(src, offset: int, length: int) -> str:
    src.seek(offset)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        chunk = src.read(min(CHUNK, remaining))
        if not chunk:
            raise SystemExit("unexpected EOF while verifying tensor payload")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def pack_shards(records: list[dict], target: int) -> list[list[dict]]:
    shards: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    for record in records:
        if current and current_bytes + record["length"] > target:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += record["length"]
    if current:
        shards.append(current)
    return shards


def write_shard(root: Path, name: str, members: list[dict], metadata, sources) -> None:
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    cursor = 0
    for record in members:
        header[record["name"]] = {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "data_offsets": [cursor, cursor + record["length"]],
        }
        cursor += record["length"]
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-(8 + len(blob)) % 8)
    tmp = root / (name + ".tmp")
    with tmp.open("wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for record in members:
            record["sha256"] = copy_range(
                sources[record["src"]], out, record["abs_offset"], record["length"]
            )
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, root / name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_root", type=Path)
    ap.add_argument(
        "--expert-pattern", default=r"^model\.layers\.\d+\.mlp\.switch_mlp\."
    )
    ap.add_argument("--target-shard-bytes", type=int, default=5 * 1024**3)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--retire-dir", type=Path, required=True)
    args = ap.parse_args()

    root = args.model_root.expanduser().resolve()
    expert_re = re.compile(args.expert_pattern)
    index_path = root / INDEX_NAME
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    declared = index.get("metadata", {}).get("total_size")

    # Inventory every tensor from the source shard headers.
    records: list[dict] = []
    seen: set[str] = set()
    src_names = sorted(set(weight_map.values()))
    for src_name in src_names:
        header, data_start = read_header(root / src_name)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise SystemExit(f"tensor {name!r} appears in multiple shards")
            seen.add(name)
            if weight_map.get(name) != src_name:
                raise SystemExit(f"index maps {name!r} elsewhere; refusing")
            begin, end = info["data_offsets"]
            records.append(
                {
                    "name": name,
                    "src": src_name,
                    "abs_offset": data_start + begin,
                    "length": end - begin,
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                }
            )
    if seen != set(weight_map):
        missing = sorted(set(weight_map) - seen)
        raise SystemExit(f"index tensors missing from headers: {missing[:4]}")
    total = sum(r["length"] for r in records)
    if declared is not None and declared != total:
        print(f"warning: index total_size {declared} != tensor sum {total}")

    metadata_header = read_header(root / src_names[0])[0].get("__metadata__")
    src_order = {name: i for i, name in enumerate(src_names)}
    records.sort(key=lambda r: (src_order[r["src"]], r["abs_offset"]))
    resident = [r for r in records if not expert_re.match(r["name"])]
    expert = [r for r in records if expert_re.match(r["name"])]
    if not resident or not expert:
        raise SystemExit("partition produced an empty group; check --expert-pattern")

    plan: list[tuple[str, list[dict]]] = []
    for group_name, group in (("resident", resident), ("expert", expert)):
        packed = pack_shards(group, args.target_shard_bytes)
        for i, members in enumerate(packed, start=1):
            plan.append(
                (
                    f"model-{group_name}-{i:05d}-of-{len(packed):05d}.safetensors",
                    members,
                )
            )

    stat = os.statvfs(root)
    free = stat.f_bavail * stat.f_frsize
    if free < total + 2 * 1024**3:
        raise SystemExit(f"need {total} bytes free plus margin, have {free}")

    print(
        f"{len(records)} tensors, {total / 1024**3:.1f} GiB -> "
        f"{sum(1 for n, _ in plan if 'resident' in n)} resident + "
        f"{sum(1 for n, _ in plan if 'expert' in n)} expert shards"
    )
    sources = {name: (root / name).open("rb") for name in src_names}
    try:
        for shard_name, members in plan:
            write_shard(root, shard_name, members, metadata_header, sources)
            size = sum(m["length"] for m in members)
            print(f"wrote {shard_name}: {len(members)} tensors, {size / 1024**3:.2f} GiB")
    finally:
        for f in sources.values():
            f.close()

    # Verify: re-read every new shard independently.
    failures = 0
    for shard_name, members in plan:
        header, data_start = read_header(root / shard_name)
        cursor = 0
        with (root / shard_name).open("rb") as f:
            for record in members:
                info = header[record["name"]]
                begin, end = info["data_offsets"]
                ok = (
                    info["dtype"] == record["dtype"]
                    and info["shape"] == record["shape"]
                    and begin == cursor
                    and end - begin == record["length"]
                    and hash_range(f, data_start + begin, end - begin)
                    == record["sha256"]
                )
                if not ok:
                    failures += 1
                    print(f"VERIFY FAIL: {record['name']} in {shard_name}")
                cursor = end
        expected = (root / shard_name).stat().st_size - data_start
        if cursor != expected:
            failures += 1
            print(f"VERIFY FAIL: {shard_name} data section {expected} != {cursor}")
        print(f"verified {shard_name}")
    if failures:
        raise SystemExit(f"{failures} verification failures; index NOT replaced")

    new_index = {
        "metadata": {"total_size": total},
        "weight_map": {
            record["name"]: shard_name
            for shard_name, members in plan
            for record in members
        },
    }
    tmp = root / (INDEX_NAME + ".tmp")
    tmp.write_text(json.dumps(new_index, indent=2, sort_keys=True))
    retire = args.retire_dir.expanduser().resolve()
    retire.mkdir(parents=True, exist_ok=True)
    (retire / (INDEX_NAME + ".pre-reshard")).write_text(index_path.read_text())
    os.replace(tmp, index_path)
    for src_name in src_names:
        os.replace(root / src_name, retire / src_name)
    print(f"retired {len(src_names)} source shards -> {retire}")

    report = {
        "root": str(root),
        "total_size": total,
        "tensors": len(records),
        "shards": {
            shard_name: {
                "tensors": len(members),
                "bytes": sum(m["length"] for m in members),
            }
            for shard_name, members in plan
        },
        "tensor_sha256": {r["name"]: r["sha256"] for r in records},
        "verified": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
