"""Sharded expert sidecars for Hugging Face upload (per-file size limits).

``experts.bin`` sidecars are far larger than the Hub's per-file limits, so
``plan_sidecar_shards``/``write_sidecar_shards`` cut them at record
boundaries into ``experts-00001-of-000NN.bin`` files and re-manifest the
records with per-record ``sidecar_shard`` names and shard-relative offsets.
These tests pin the cut rules, the byte-exact copy, the manifest round trip,
and that the positional reader serves identical record bytes from the
sharded layout without any reassembly step.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from mtplx.expert_io import (
    ExpertIOIntegrityError,
    PositionalExpertReader,
)
from mtplx.expert_manifest import (
    DEFAULT_ALIGNMENT,
    ExpertManifestError,
    build_expert_sidecar,
    load_expert_manifest,
    plan_sidecar_shards,
    save_expert_manifest,
    write_sidecar_shards,
)
from tests.test_expert_slots_runtime import _global_artifact

# Fixture records are 6,912 bytes at 16 KiB-aligned offsets 0/16384/32768/
# 49152; this limit forces a two-records-per-shard split (record 2 would end
# past the limit relative to shard offset 0).
SPLIT_LIMIT = 24_000


class _RecordSlot:
    """Minimal component-slot destination: one flat buffer per record."""

    def __init__(self, record) -> None:
        self.buffer = bytearray(record.logical_bytes)

    def record_views(self, record) -> tuple[memoryview, ...]:
        views = []
        cursor = 0
        for segment in record.segments:
            views.append(memoryview(self.buffer)[cursor : cursor + segment.length])
            cursor += segment.length
        return tuple(views)


def _sidecar_artifact(tmp_path):
    root, spec, manifest, expected = _global_artifact(tmp_path)
    manifest = build_expert_sidecar(manifest, root, root / "experts.bin")
    return root, spec, manifest, expected


def _sharded_artifact(tmp_path, *, max_shard_bytes: int = SPLIT_LIMIT):
    root, spec, manifest, expected = _sidecar_artifact(tmp_path)
    plans = plan_sidecar_shards(manifest, max_shard_bytes=max_shard_bytes)
    sharded = write_sidecar_shards(manifest, root, plans)
    return root, spec, manifest, sharded, plans, expected


def test_plan_cuts_at_record_boundaries_and_covers_every_record(tmp_path) -> None:
    _root, _spec, manifest, _expected = _sidecar_artifact(tmp_path)
    plans = plan_sidecar_shards(manifest, max_shard_bytes=SPLIT_LIMIT)
    assert len(plans) == 2
    covered = [index for plan in plans for index in plan.record_indices]
    assert covered == list(range(len(manifest.records)))
    for plan in plans:
        first = manifest.records[plan.record_indices[0]]
        last = manifest.records[plan.record_indices[-1]]
        assert plan.source_offset == first.sidecar_offset
        assert plan.length == (
            (last.sidecar_offset or 0)
            + (last.sidecar_length or 0)
            - plan.source_offset
        )
        assert plan.length <= SPLIT_LIMIT
        # Shard bases sit on record boundaries, so the sidecar alignment
        # survives rebasing (a record is never split across shards).
        assert plan.source_offset % DEFAULT_ALIGNMENT == 0
    assert plans[0].name == "experts-00001-of-00002.bin"
    assert plans[1].name == "experts-00002-of-00002.bin"


def test_plan_single_shard_when_limit_is_large(tmp_path) -> None:
    _root, _spec, manifest, _expected = _sidecar_artifact(tmp_path)
    plans = plan_sidecar_shards(manifest, max_shard_bytes=1 << 30)
    assert len(plans) == 1
    assert plans[0].name == "experts-00001-of-00001.bin"
    assert len(plans[0].record_indices) == len(manifest.records)


def test_plan_rejects_record_larger_than_limit(tmp_path) -> None:
    _root, _spec, manifest, _expected = _sidecar_artifact(tmp_path)
    with pytest.raises(ExpertManifestError, match="larger than"):
        plan_sidecar_shards(manifest, max_shard_bytes=100)


def test_plan_requires_single_file_sidecar(tmp_path) -> None:
    root, _spec, manifest, sharded, _plans, _expected = _sharded_artifact(tmp_path)
    with pytest.raises(ExpertManifestError, match="single-file sidecar"):
        plan_sidecar_shards(sharded, max_shard_bytes=SPLIT_LIMIT)


def test_written_shards_are_byte_exact_slices_of_the_sidecar(tmp_path) -> None:
    root, _spec, manifest, _sharded, plans, _expected = _sharded_artifact(tmp_path)
    source = (root / "experts.bin").read_bytes()
    for plan in plans:
        payload = (root / plan.name).read_bytes()
        assert payload == source[plan.source_offset : plan.source_offset + plan.length]


def test_sharded_manifest_round_trips_with_digest(tmp_path) -> None:
    root, _spec, _manifest, sharded, plans, _expected = _sharded_artifact(tmp_path)
    assert sharded.sidecar is None
    assert sharded.sidecar_alignment == DEFAULT_ALIGNMENT
    sidecar_shards = [shard for shard in sharded.shards if shard.kind == "sidecar"]
    assert [shard.name for shard in sidecar_shards] == [plan.name for plan in plans]
    for shard in sidecar_shards:
        assert shard.sha256 is not None
    for record in sharded.records:
        assert record.sidecar_shard in {plan.name for plan in plans}
        assert all(
            segment.shard == record.sidecar_shard for segment in record.segments
        )
    path = root / "expert-manifest-sharded.json"
    save_expert_manifest(sharded, path)
    loaded = load_expert_manifest(path)
    assert loaded == sharded


def test_reader_serves_identical_record_bytes_from_shards(tmp_path) -> None:
    root, spec, _manifest, sharded, _plans, expected = _sharded_artifact(tmp_path)
    with PositionalExpertReader(root) as reader:
        for record in sharded.records:
            destination = bytearray(record.logical_bytes)
            digest = reader.read_record_into(sharded, record, destination)
            assert bytes(destination) == expected[(record.layer, record.expert)]
            assert digest == record.sha256


def test_batch_component_read_matches_across_layouts(tmp_path) -> None:
    root, _spec, manifest, sharded, _plans, expected = _sharded_artifact(tmp_path)
    for source_manifest, chunk in (
        (sharded, None),
        (sharded, 1_000),
        (manifest, 1_000),
    ):
        slots = tuple(
            (record, _RecordSlot(record)) for record in source_manifest.records
        )
        with PositionalExpertReader(root) as reader:
            digests = reader.read_component_records_into(
                source_manifest,
                slots,
                coalesce_chunk_bytes=chunk,
            )
        for (record, slot), digest in zip(slots, digests, strict=True):
            assert bytes(slot.buffer) == expected[(record.layer, record.expert)]
            assert digest == record.sha256


def test_corrupted_shard_fails_hash_verification(tmp_path) -> None:
    root, _spec, _manifest, sharded, plans, _expected = _sharded_artifact(tmp_path)
    victim = root / plans[0].name
    payload = bytearray(victim.read_bytes())
    payload[0] ^= 0xFF
    victim.write_bytes(payload)
    record = sharded.records[0]
    with PositionalExpertReader(root) as reader:
        with pytest.raises(ExpertIOIntegrityError, match="hash mismatch"):
            reader.read_record_into(
                sharded, record, bytearray(record.logical_bytes)
            )


def test_write_requires_record_hashes(tmp_path) -> None:
    root, _spec, manifest, _expected = _sidecar_artifact(tmp_path)
    plans = plan_sidecar_shards(manifest, max_shard_bytes=SPLIT_LIMIT)
    unhashed = replace(
        manifest,
        records=tuple(
            replace(record, sha256=None) for record in manifest.records
        ),
        manifest_sha256=None,
    )
    with pytest.raises(ExpertManifestError, match="record hashes"):
        write_sidecar_shards(unhashed, root, plans)


def test_validate_rejects_partial_or_conflicting_sharded_layouts(tmp_path) -> None:
    root, _spec, manifest, sharded, _plans, _expected = _sharded_artifact(tmp_path)
    mixed = replace(
        sharded,
        records=(replace(sharded.records[0], sidecar_shard=None),)
        + sharded.records[1:],
    )
    with pytest.raises(ExpertManifestError):
        mixed.validate_structure()
    conflicted = replace(sharded, sidecar=manifest.sidecar)
    with pytest.raises(ExpertManifestError, match="conflict"):
        conflicted.validate_structure()
    unaligned = replace(sharded, sidecar_alignment=None)
    with pytest.raises(ExpertManifestError, match="sidecar_alignment"):
        unaligned.validate_structure()
    stray_alignment = replace(manifest, sidecar_alignment=DEFAULT_ALIGNMENT)
    with pytest.raises(ExpertManifestError, match="sidecar_alignment"):
        stray_alignment.validate_structure()
    unknown_shard = replace(
        sharded,
        records=(
            replace(sharded.records[0], sidecar_shard="experts-missing.bin"),
        )
        + sharded.records[1:],
    )
    with pytest.raises(ExpertManifestError, match="not a sidecar shard"):
        unknown_shard.validate_structure()


def _load_shard_cli():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "shard_expert_sidecar.py"
    )
    spec = importlib.util.spec_from_file_location("shard_expert_sidecar", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("shard_expert_sidecar", module)
    spec.loader.exec_module(module)
    return module


def test_cli_dry_run_plans_without_writing(tmp_path, capsys) -> None:
    root, _spec, manifest, _expected = _sidecar_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    cli = _load_shard_cli()
    exit_code = cli.main(
        [
            str(root),
            "--max-shard-bytes",
            str(SPLIT_LIMIT),
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "experts-00001-of-00002.bin" in output
    assert not (root / "experts-00001-of-00002.bin").exists()
    assert not (root / "expert-manifest-sharded.json").exists()


def test_cli_execute_writes_shards_and_manifest(tmp_path, capsys) -> None:
    root, _spec, manifest, expected = _sidecar_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    cli = _load_shard_cli()
    exit_code = cli.main(
        [
            str(root),
            "--max-shard-bytes",
            str(SPLIT_LIMIT),
            "--execute",
        ]
    )
    assert exit_code == 0
    sharded_path = root / "expert-manifest-sharded.json"
    assert sharded_path.is_file()
    sharded = load_expert_manifest(sharded_path)
    with PositionalExpertReader(root) as reader:
        record = sharded.records[-1]
        destination = bytearray(record.logical_bytes)
        reader.read_record_into(sharded, record, destination)
        assert bytes(destination) == expected[(record.layer, record.expert)]
