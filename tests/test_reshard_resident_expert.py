"""Round-trip test for scripts/reshard_resident_expert.py on a synthetic checkpoint."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reshard_resident_expert.py"


def _write_shard(path: Path, tensors, metadata=None) -> None:
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    cursor = 0
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-(8 + len(blob)) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, payload in tensors:
            f.write(payload)


def _read_tensor(root: Path, index: dict, name: str) -> tuple[bytes, str, list]:
    shard = root / index["weight_map"][name]
    with shard.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        info = header[name]
        begin, end = info["data_offsets"]
        f.seek(8 + n + begin)
        return f.read(end - begin), info["dtype"], info["shape"]


def test_reshard_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    # Interleave resident and expert tensors across two source shards, the
    # same shape the real checkpoint has.
    tensors = [
        ("model.embed_tokens.weight", "BF16", [4, 2], b"embd" * 4),
        ("model.layers.0.mlp.down_proj.weight", "U32", [2, 2], b"l0dn" * 4),
        ("model.layers.1.mlp.switch_mlp.up_proj.weight", "U32", [8], b"e1up" * 8),
        ("model.layers.1.mlp.shared_mlp.up_proj.weight", "BF16", [4], b"s1up" * 2),
        ("model.layers.2.mlp.switch_mlp.down_proj.scales", "BF16", [16], b"e2dn" * 8),
        ("model.layers.2.self_attn.q_proj.weight", "U32", [4], b"a2qp" * 4),
        ("lm_head.weight", "BF16", [4, 2], b"head" * 4),
    ]
    shard_a, shard_b = tensors[:4], tensors[4:]
    _write_shard(root / "model-00001-of-00002.safetensors", shard_a, {"format": "mlx"})
    _write_shard(root / "model-00002-of-00002.safetensors", shard_b, {"format": "mlx"})
    total = sum(len(p) for _, _, _, p in tensors)
    index = {
        "metadata": {"total_size": total},
        "weight_map": {
            name: f"model-0000{1 if (name, d, s, p) in shard_a else 2}-of-00002.safetensors"
            for name, d, s, p in tensors
        },
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))

    report = tmp_path / "report.json"
    retire = tmp_path / "retired"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(root),
            "--report",
            str(report),
            "--retire-dir",
            str(retire),
            "--target-shard-bytes",
            "40",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    new_index = json.loads((root / "model.safetensors.index.json").read_text())
    assert new_index["metadata"]["total_size"] == total
    assert set(new_index["weight_map"]) == {name for name, _, _, _ in tensors}
    for name, shard in new_index["weight_map"].items():
        group = "expert" if ".switch_mlp." in name else "resident"
        assert shard.startswith(f"model-{group}-"), (name, shard)

    # Byte, dtype, and shape round-trip for every tensor.
    for name, dtype, shape, payload in tensors:
        data, new_dtype, new_shape = _read_tensor(root, new_index, name)
        assert data == payload, name
        assert new_dtype == dtype and new_shape == shape, name

    # The 40-byte target must split the 96-byte expert group into >1 shard.
    expert_shards = {
        s for n, s in new_index["weight_map"].items() if ".switch_mlp." in n
    }
    assert len(expert_shards) > 1

    # Old shards retired, not deleted, alongside the pre-reshard index.
    assert not list(root.glob("model-0000?-of-00002.safetensors"))
    assert len(list(retire.glob("model-0000?-of-00002.safetensors"))) == 2
    assert (retire / "model.safetensors.index.json.pre-reshard").is_file()

    payload = json.loads(report.read_text())
    assert payload["verified"] is True
    assert payload["tensors"] == len(tensors)
