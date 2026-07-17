#!/usr/bin/env python3
"""Validate a local model artifact and stage it for Hugging Face upload.

Given a model root (an expert-streaming artifact with ``expert-manifest.json``
or an MTP-layer artifact with ``mtp-artifact-manifest.json``), this tool:

  1. validates the artifact: manifest present and digest-verified, declared
     file sizes match the bytes on disk, and a deterministic sample of record
     (or tensor) hashes re-verified from the payload — sampled, not exhaustive;
  2. assembles an upload staging layout: README.md model card, manifest,
     config/tokenizer files, and the weight shard list with sizes;
  3. emits a dry-run plan (what would upload, total bytes, per-file limit
     violations).  Nothing is written or uploaded by default.

``--execute`` builds the staging directory (small files copied, large weight
files symlinked) and uploads it with ``huggingface_hub``'s
``HfApi``/``upload_folder``.  It requires ``--repo-id`` and reads ``HF_TOKEN``
from the environment at call time; the token is never stored, echoed, or
logged.  Uploads are refused while validation errors are outstanding.

Monolithic ``experts.bin`` sidecars above the Hub's 50 GB per-file hard cap
are flagged with the exact ``scripts/shard_expert_sidecar.py`` invocation
that produces an uploadable record-aligned sharded layout.

Examples:

    python scripts/prepare_hf_upload.py ~/.cache/huggingface/hy3-expert-only-mlx-q2
    python scripts/prepare_hf_upload.py <root> --write-card hf/<name>/README.md
    HF_TOKEN=... python scripts/prepare_hf_upload.py <root> \
        --repo-id you/model --staging-dir /tmp/stage --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mtplx.expert_io import ExpertIOError, PositionalExpertReader  # noqa: E402
from mtplx.expert_manifest import (  # noqa: E402
    DEFAULT_MAX_SIDECAR_SHARD_BYTES,
    ExpertManifest,
    ExpertManifestError,
    load_expert_manifest,
    plan_sidecar_shards,
)

HUB_HARD_LIMIT_BYTES = 50 * 1000**3  # Hub hard cap per file (50 GB)
HUB_RECOMMENDED_BYTES = 20 * 1000**3  # Hub recommended ceiling per file
COPY_LIMIT_BYTES = 64 * 1024 * 1024  # staging: copy below, symlink above

EXPERT_MANIFEST_NAME = "expert-manifest.json"
SHARDED_MANIFEST_NAME = "expert-manifest-sharded.json"
MTP_MANIFEST_NAME = "mtp-artifact-manifest.json"

# Small metadata files staged verbatim when present in the artifact root.
METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "conversion-manifest.json",
    "conversion-provenance.json",
    "conversion-validation.json",
    "conversion-checkpoint.jsonl",
)

# Deprecated experiment leftovers: never staged, reported for reclamation.
STRAY_PREFIXES = ("experts-huff-", "experts-banked-")


@dataclass
class PlanEntry:
    path_in_repo: str
    source: Path | None
    bytes: int
    role: str
    note: str = ""


@dataclass
class Finding:
    severity: str  # "error" | "warn" | "info"
    message: str


@dataclass
class UploadPlan:
    root: Path
    kind: str  # "expert-streaming" | "mtp-layer"
    entries: list[PlanEntry] = field(default_factory=list)
    excluded: list[tuple[str, int, str]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries)

    @property
    def errors(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "error"]

    def add(self, severity: str, message: str) -> None:
        self.findings.append(Finding(severity, message))


def _format_bytes(value: int) -> str:
    if value >= 1000**3:
        return f"{value / 1000**3:.2f} GB"
    if value >= 1000**2:
        return f"{value / 1000**2:.2f} MB"
    return f"{value} B"


def _check_size(plan: UploadPlan, path: Path, expected: int, label: str) -> bool:
    if not path.is_file():
        plan.add("error", f"{label}: missing file {path.name}")
        return False
    actual = path.stat().st_size
    if actual != expected:
        plan.add(
            "error",
            f"{label}: {path.name} is {actual} bytes; manifest says {expected}",
        )
        return False
    return True


def _flag_file_limits(plan: UploadPlan, entry: PlanEntry) -> None:
    if entry.bytes > HUB_HARD_LIMIT_BYTES:
        plan.add(
            "error",
            f"{entry.path_in_repo} is {_format_bytes(entry.bytes)}, above the "
            f"Hub 50 GB per-file hard cap",
        )
    elif entry.bytes > HUB_RECOMMENDED_BYTES:
        plan.add(
            "warn",
            f"{entry.path_in_repo} is {_format_bytes(entry.bytes)}, above the "
            f"Hub ~20 GB recommended ceiling",
        )


def _sample_expert_records(
    plan: UploadPlan,
    manifest: ExpertManifest,
    *,
    sample_records: int,
    seed: int,
) -> None:
    if sample_records <= 0:
        plan.add("info", "record hash spot-check skipped (--sample-records 0)")
        return
    rng = random.Random(seed)
    count = min(sample_records, len(manifest.records))
    picks = rng.sample(range(len(manifest.records)), count)
    try:
        with PositionalExpertReader(plan.root) as reader:
            for index in sorted(picks):
                record = manifest.records[index]
                destination = bytearray(record.logical_bytes)
                reader.read_record_into(manifest, record, destination)
    except (ExpertIOError, ExpertManifestError, OSError) as exc:
        plan.add("error", f"record hash spot-check failed: {exc}")
        return
    plan.add(
        "info",
        f"record hash spot-check passed: {count} of {len(manifest.records)} "
        f"records re-hashed (seed {seed})",
    )


def _classify_leftovers(plan: UploadPlan, staged_names: set[str]) -> None:
    for path in sorted(plan.root.iterdir()):
        if path.name in staged_names or path.name.startswith("."):
            continue
        if path.is_dir():
            plan.excluded.append((path.name + "/", 0, "directory (never staged)"))
            continue
        size = path.stat().st_size
        if path.name.startswith(STRAY_PREFIXES):
            plan.excluded.append(
                (path.name, size, "deprecated experiment leftover (reclamation candidate)")
            )
        else:
            plan.excluded.append((path.name, size, "not part of the upload set"))


def build_expert_plan(
    root: Path,
    *,
    sample_records: int = 4,
    seed: int = 51,
    card_source: Path | None = None,
) -> UploadPlan:
    plan = UploadPlan(root=root, kind="expert-streaming")
    manifest_path = root / SHARDED_MANIFEST_NAME
    manifest_name = SHARDED_MANIFEST_NAME
    if not manifest_path.is_file():
        manifest_path = root / EXPERT_MANIFEST_NAME
        manifest_name = EXPERT_MANIFEST_NAME
    try:
        manifest = load_expert_manifest(manifest_path)
    except (ExpertManifestError, OSError) as exc:
        plan.add("error", f"manifest is unusable: {exc}")
        return plan
    plan.add(
        "info",
        f"manifest {manifest_name}: format {manifest.format}, model "
        f"{manifest.model_key}, digest verified",
    )

    staged: set[str] = {EXPERT_MANIFEST_NAME, SHARDED_MANIFEST_NAME}
    # The manifest always uploads under the canonical runtime name.
    plan.entries.append(
        PlanEntry(
            path_in_repo=EXPERT_MANIFEST_NAME,
            source=manifest_path,
            bytes=manifest_path.stat().st_size,
            role="expert manifest",
            note=(
                "renamed from expert-manifest-sharded.json"
                if manifest_name == SHARDED_MANIFEST_NAME
                else ""
            ),
        )
    )

    for name in METADATA_FILES:
        path = root / name
        if path.is_file():
            staged.add(name)
            plan.entries.append(
                PlanEntry(name, path, path.stat().st_size, "metadata")
            )
    if not (root / "config.json").is_file():
        plan.add("warn", "config.json is missing")
    if not (root / "tokenizer.json").is_file():
        plan.add("warn", "tokenizer.json is missing")

    resident_shards = sorted({tensor.shard for tensor in manifest.resident_tensors})
    shard_info = {shard.name: shard for shard in manifest.shards}
    for name in resident_shards:
        info = shard_info.get(name)
        path = root / name
        expected = info.size if info is not None else -1
        if _check_size(plan, path, expected, "resident shard"):
            staged.add(name)
            entry = PlanEntry(name, path, expected, "resident weights")
            plan.entries.append(entry)
            _flag_file_limits(plan, entry)
    if resident_shards and not (root / "model.safetensors.index.json").is_file():
        plan.add("error", "model.safetensors.index.json is missing")

    sidecar_shards = [shard for shard in manifest.shards if shard.kind == "sidecar"]
    if manifest.sidecar is not None:
        name = manifest.sidecar.file
        path = root / name
        if _check_size(plan, path, manifest.sidecar.size, "sidecar"):
            staged.add(name)
            entry = PlanEntry(name, path, manifest.sidecar.size, "expert sidecar")
            plan.entries.append(entry)
            _flag_file_limits(plan, entry)
            if manifest.sidecar.size > HUB_HARD_LIMIT_BYTES:
                shard_count = len(
                    plan_sidecar_shards(
                        manifest,
                        max_shard_bytes=DEFAULT_MAX_SIDECAR_SHARD_BYTES,
                    )
                )
                plan.add(
                    "error",
                    "shard the sidecar before uploading (would produce "
                    f"{shard_count} files at <=16 GiB): python "
                    f"scripts/shard_expert_sidecar.py {root} --execute",
                )
    elif sidecar_shards:
        for shard in sidecar_shards:
            path = root / shard.name
            if _check_size(plan, path, shard.size, "sidecar shard"):
                staged.add(shard.name)
                entry = PlanEntry(
                    shard.name, path, shard.size, "expert sidecar (sharded)"
                )
                plan.entries.append(entry)
                _flag_file_limits(plan, entry)
    else:
        plan.add("error", "manifest has neither a sidecar nor sidecar shards")

    _sample_expert_records(
        plan, manifest, sample_records=sample_records, seed=seed
    )
    _stage_card(plan, card_source, staged)
    _classify_leftovers(plan, staged)
    return plan


def _stage_card(
    plan: UploadPlan, card_source: Path | None, staged: set[str]
) -> None:
    staged.add("README.md")
    if card_source is not None and card_source.is_file():
        plan.entries.insert(
            0,
            PlanEntry(
                "README.md",
                card_source,
                card_source.stat().st_size,
                "model card",
            ),
        )
    elif (plan.root / "README.md").is_file():
        path = plan.root / "README.md"
        plan.entries.insert(
            0, PlanEntry("README.md", path, path.stat().st_size, "model card")
        )
    else:
        plan.add(
            "warn",
            "no model card: pass --card <hf/<name>/README.md> (see the hf/ "
            "directory in the mtplx repo)",
        )


def _load_mtp_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_mtp_plan(
    root: Path,
    *,
    sample_records: int = 4,
    seed: int = 51,
    card_source: Path | None = None,
) -> UploadPlan:
    plan = UploadPlan(root=root, kind="mtp-layer")
    staged: set[str] = set()
    manifest_path = root / MTP_MANIFEST_NAME
    manifest: dict | None = None
    if manifest_path.is_file():
        try:
            manifest = _load_mtp_manifest(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            plan.add("error", f"mtp manifest is unreadable: {exc}")
        else:
            schema = manifest.get("schema", "<missing>")
            plan.add("info", f"mtp manifest schema: {schema}")
            staged.add(MTP_MANIFEST_NAME)
            plan.entries.append(
                PlanEntry(
                    MTP_MANIFEST_NAME,
                    manifest_path,
                    manifest_path.stat().st_size,
                    "mtp manifest",
                )
            )
    else:
        plan.add(
            "warn",
            "no mtp-artifact-manifest.json: provenance/integrity metadata is "
            "missing for a clean upload",
        )

    for name in METADATA_FILES:
        if name == "model.safetensors.index.json":
            # In MTP artifact dirs this is the SOURCE checkpoint's shard
            # index; the source shards themselves are never uploaded.
            continue
        path = root / name
        if path.is_file():
            staged.add(name)
            plan.entries.append(
                PlanEntry(name, path, path.stat().st_size, "metadata")
            )

    if not (root / "config.json").is_file():
        plan.add(
            "warn",
            "config.json is missing (for extracted layers, copy it from the "
            "matching -source checkpoint directory)",
        )

    for path in sorted(root.glob("layer*.safetensors")):
        staged.add(path.name)
        entry = PlanEntry(
            path.name, path, path.stat().st_size, "mtp layer weights"
        )
        plan.entries.append(entry)
        _flag_file_limits(plan, entry)

    if manifest is not None:
        artifact = manifest.get("artifact", {})
        name = artifact.get("file")
        if isinstance(name, str):
            path = root / name
            declared = int(artifact.get("file_bytes", -1))
            if _check_size(plan, path, declared, "mtp artifact"):
                _verify_mtp_header(plan, path, artifact)
                _sample_mtp_tensors(
                    plan,
                    path,
                    artifact,
                    sample_records=sample_records,
                    seed=seed,
                )
    _stage_card(plan, card_source, staged)
    _classify_leftovers(plan, staged)
    return plan


def _verify_mtp_header(plan: UploadPlan, path: Path, artifact: dict) -> None:
    header_bytes = int(artifact.get("header_bytes", 0))
    declared = artifact.get("header_sha256")
    if not header_bytes or not isinstance(declared, str):
        return
    with path.open("rb") as handle:
        handle.seek(8)
        header = handle.read(header_bytes)
    if hashlib.sha256(header).hexdigest() != declared:
        plan.add("error", f"{path.name}: safetensors header hash mismatch")
    else:
        plan.add("info", f"{path.name}: safetensors header hash verified")


def _sample_mtp_tensors(
    plan: UploadPlan,
    path: Path,
    artifact: dict,
    *,
    sample_records: int,
    seed: int,
) -> None:
    tensors = artifact.get("tensors")
    if not isinstance(tensors, list) or not tensors or sample_records <= 0:
        return
    data_start = 8 + int(artifact.get("header_bytes", 0))
    # Sample among modest tensors so the spot-check stays a light read.
    small = [
        tensor
        for tensor in tensors
        if isinstance(tensor.get("sha256"), str)
        and tensor["output_data_offsets"][1] - tensor["output_data_offsets"][0]
        <= 256 * 1024 * 1024
    ]
    if not small:
        return
    rng = random.Random(seed)
    picks = rng.sample(small, min(sample_records, len(small)))
    with path.open("rb") as handle:
        for tensor in picks:
            start, end = tensor["output_data_offsets"]
            handle.seek(data_start + int(start))
            payload = handle.read(int(end) - int(start))
            if hashlib.sha256(payload).hexdigest() != tensor["sha256"]:
                plan.add(
                    "error",
                    f"{path.name}: tensor {tensor.get('name')} hash mismatch",
                )
                return
    plan.add(
        "info",
        f"tensor hash spot-check passed: {len(picks)} of {len(tensors)} "
        f"tensors re-hashed (seed {seed})",
    )


def build_plan(
    root: Path,
    *,
    sample_records: int = 4,
    seed: int = 51,
    card_source: Path | None = None,
) -> UploadPlan:
    if (root / EXPERT_MANIFEST_NAME).is_file() or (
        root / SHARDED_MANIFEST_NAME
    ).is_file():
        return build_expert_plan(
            root,
            sample_records=sample_records,
            seed=seed,
            card_source=card_source,
        )
    return build_mtp_plan(
        root,
        sample_records=sample_records,
        seed=seed,
        card_source=card_source,
    )


def print_plan(plan: UploadPlan) -> None:
    print(f"[DRY RUN] upload plan for {plan.root} ({plan.kind})")
    for entry in plan.entries:
        note = f"  ({entry.note})" if entry.note else ""
        print(
            f"  UPLOAD {entry.path_in_repo:44s} {_format_bytes(entry.bytes):>11s}"
            f"  {entry.role}{note}"
        )
    print(
        f"  total: {len(plan.entries)} file(s), {_format_bytes(plan.total_bytes)}"
    )
    if plan.excluded:
        print("  excluded from upload:")
        for name, size, reason in plan.excluded:
            print(f"    SKIP {name:44s} {_format_bytes(size):>11s}  {reason}")
    for finding in plan.findings:
        print(f"  {finding.severity.upper():5s} {finding.message}")


def build_staging(plan: UploadPlan, staging_dir: Path) -> Path:
    staging_dir = staging_dir.expanduser().resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    for entry in plan.entries:
        if entry.source is None:
            continue
        target = staging_dir / entry.path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if entry.bytes <= COPY_LIMIT_BYTES:
            shutil.copyfile(entry.source, target)
        else:
            target.symlink_to(entry.source)
    return staging_dir


def execute_upload(plan: UploadPlan, staging_dir: Path, repo_id: str) -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "error: --execute requires HF_TOKEN in the environment "
            "(the token is read at call time and never logged)",
            file=sys.stderr,
        )
        return 2
    if plan.errors:
        print(
            "error: refusing to upload while validation errors are "
            "outstanding (see the findings above)",
            file=sys.stderr,
        )
        return 1
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "error: huggingface_hub is not installed in this environment",
            file=sys.stderr,
        )
        return 2
    staging = build_staging(plan, staging_dir)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {plan.root.name} via mtplx prepare_hf_upload",
    )
    print(f"uploaded {len(plan.entries)} file(s) to {repo_id}")
    return 0


SIDECAR_FORMAT_SECTION = """\
## Expert sidecar format

`experts-*.bin` (or a monolithic `experts.bin`) is a **record-major** sidecar
of the routed expert weights, described and trust-anchored by
`expert-manifest.json` (`mtplx-expert-manifest-v1`):

- one record per `(layer, expert)`; records are sorted by `(layer, expert)`
  and each record's bytes are contiguous;
- every record starts on a **16 KiB-aligned** offset (`sidecar_alignment`);
- a record is the nine quantized components in fixed order
  `gate_proj/up_proj/down_proj` x `weight/scales/biases`
  (`U32` packed weights, `BF16` scales/biases), laid out back to back;
- manifest **segment offsets are shard-absolute**: each segment names its
  file and the byte offset inside that file;
- the sharded layout cuts the sidecar at record boundaries into
  `experts-00001-of-000NN.bin`; each record then carries `sidecar_shard`
  plus its shard-relative offset, and alignment is preserved because shards
  begin exactly on record boundaries;
- every record has a SHA-256 in the manifest; the runtime verifies it on
  read (fail-closed), and the manifest itself is digest-pinned
  (`manifest_sha256`).
"""

USAGE_SECTION = """\
## Usage

These weights are served by the [mtplx](https://github.com/davidtai/MTPLX)
expert-streaming runtime for Apple Silicon (MLX): resident weights load into
memory while routed experts stream on demand from the sidecar via bounded
positional I/O (dense per-layer "island" banks, slot caches, or mmap-banked
execution). Download the repository contents into one directory and point
the runtime at it; the sharded sidecar layout is read directly - no
reassembly step. This artifact is **not** loadable with plain
`mlx_lm.load()`.
"""


def _front_matter(tags: list[str], base_model: str | None) -> str:
    lines = ["---"]
    if base_model:
        lines.append(f"base_model: {base_model}")
    lines.append("tags:")
    lines.extend(f"- {tag}" for tag in tags)
    lines.append("---")
    # License intentionally unset: inherit/choose one before publishing.
    return "\n".join(lines) + "\n"


def _inventory_section(plan: UploadPlan) -> str:
    lines = [
        "## Files",
        "",
        "| file | size | role |",
        "| --- | ---: | --- |",
    ]
    for entry in plan.entries:
        note = f" ({entry.note})" if entry.note else ""
        lines.append(
            f"| `{entry.path_in_repo}` | {_format_bytes(entry.bytes)} | "
            f"{entry.role}{note} |"
        )
    lines.append("")
    lines.append(f"Total: {_format_bytes(plan.total_bytes)}.")
    return "\n".join(lines) + "\n"


def _planned_shard_section(manifest: ExpertManifest) -> str:
    if manifest.sidecar is None or manifest.sidecar.size <= HUB_HARD_LIMIT_BYTES:
        return ""
    plans = plan_sidecar_shards(
        manifest, max_shard_bytes=DEFAULT_MAX_SIDECAR_SHARD_BYTES
    )
    lines = [
        "## Upload sharding (planned)",
        "",
        f"`{manifest.sidecar.file}` is "
        f"{_format_bytes(manifest.sidecar.size)}, above the Hub's 50 GB "
        "per-file cap, so the published repository ships it as "
        f"{len(plans)} record-aligned shards produced by "
        "`scripts/shard_expert_sidecar.py` (<= 16 GiB each):",
        "",
    ]
    for shard in plans:
        lines.append(f"- `{shard.name}` ({_format_bytes(shard.length)})")
    lines.append("")
    lines.append(
        "The mtplx runtime reads the sharded layout directly through the "
        "updated `expert-manifest.json`."
    )
    return "\n".join(lines) + "\n\n"


def render_expert_card(root: Path, plan: UploadPlan) -> str:
    manifest_path = root / SHARDED_MANIFEST_NAME
    if not manifest_path.is_file():
        manifest_path = root / EXPERT_MANIFEST_NAME
    manifest = load_expert_manifest(manifest_path)
    layers = sorted({record.layer for record in manifest.records})
    expert_count = sum(
        1 for record in manifest.records if record.layer == layers[0]
    )
    record = manifest.records[0]
    conversion = {}
    for name in ("conversion-manifest.json", "conversion-provenance.json"):
        path = root / name
        if path.is_file():
            try:
                conversion = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                conversion = {}
            break
    upstream = ""
    source_block = conversion.get("source")
    if isinstance(source_block, dict) and isinstance(
        source_block.get("repo"), str
    ):
        upstream = (
            f"- Upstream checkpoint: `{source_block['repo']}` @ "
            f"`{source_block.get('revision', 'unknown')}`\n"
        )
    producer = conversion.get("producer")
    producer_line = ""
    if isinstance(producer, dict):
        commit = producer.get("git_commit") or producer.get("commit")
        if commit:
            producer_line = (
                f"- Converted by mtplx commit `{commit}` "
                f"(schema `{conversion.get('schema', conversion.get('format', 'n/a'))}`)\n"
            )
    base_model = None
    if isinstance(source_block, dict) and isinstance(
        source_block.get("repo"), str
    ):
        base_model = source_block["repo"]
    elif "/" in manifest.source_repo and not manifest.source_repo.startswith(
        "local/"
    ):
        base_model = manifest.source_repo

    title = f"{manifest.model_key} (mtplx expert-streaming artifact)"
    body = f"""\
# {title}

Expert-only affine Q{manifest.quant_bits} MoE artifact for the mtplx
expert-streaming runtime: dense (resident) weights as safetensors shards plus
all routed expert weights repacked into a record-major sidecar for SSD
streaming on Apple Silicon.

## Provenance

- Quantized source (manifest): `{manifest.source_repo}` @
  `{manifest.source_revision}`
{upstream}{producer_line}- Manifest digest: `{manifest.manifest_sha256}`

## Quantization

- Routed experts: **{manifest.quant_bits}-bit affine**, group size
  {manifest.quant_group_size}, mode `{manifest.quant_mode}`
- {len(layers)} routed layers (layers {layers[0]}-{layers[-1]}),
  {expert_count} experts per layer, {len(manifest.records)} expert records
- Expert record: {_format_bytes(record.logical_bytes)}
  ({record.logical_bytes} bytes); routed total
  {_format_bytes(manifest.routed_expert_bytes)}
- Resident (non-expert) weights: {_format_bytes(manifest.resident_tensor_bytes)}

{SIDECAR_FORMAT_SECTION}
{_planned_shard_section(manifest)}{_inventory_section(plan)}
{USAGE_SECTION}
> License note: this repository inherits the upstream model license; set the
> `license` front-matter field before publishing.
"""
    tags = ["mlx", "moe", "mtplx", "expert-streaming", f"q{manifest.quant_bits}"]
    return _front_matter(tags, base_model) + body


def render_mtp_card(root: Path, plan: UploadPlan) -> str:
    manifest_path = root / MTP_MANIFEST_NAME
    manifest = (
        _load_mtp_manifest(manifest_path) if manifest_path.is_file() else None
    )
    config = {}
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
    if manifest is not None:
        source = manifest.get("source", {})
        files = source.get("files", []) if isinstance(source, dict) else []
        artifact = manifest.get("artifact", {})
        producer = manifest.get("producer", {})
        provenance = (
            f"- Extracted from `zai-org/GLM-5.2` source shards (see "
            f"`{MTP_MANIFEST_NAME}` for per-file SHA-256 pins over "
            f"{len(files)} source files)\n"
            f"- Producer: mtplx commit `{producer.get('commit', 'n/a')}`\n"
            f"- Artifact SHA-256: `{artifact.get('sha256', 'n/a')}`\n"
            f"- Manifest digest: `{manifest.get('manifest_sha256', 'n/a')}`\n"
        )
        integrity = (
            "Every tensor carries a SHA-256 in the manifest "
            "(`artifact.tensors[]`), alongside its exact source shard and "
            "byte range; the manifest itself is digest-pinned."
        )
        base_model = "zai-org/GLM-5.2"
    else:
        provenance = (
            "- Extracted from the pinned upstream checkpoint (see the mtplx "
            "extraction scripts); no artifact manifest is present yet\n"
        )
        integrity = (
            "No per-tensor integrity manifest accompanies this artifact yet; "
            "generate one before relying on the upload."
        )
        base_model = "tencent/Hy3" if "hy" in root.name.lower() else None
    arch = config.get("architectures", ["unknown"])
    title = f"{root.name} (mtplx MTP layer artifact)"
    body = f"""\
# {title}

Multi-token-prediction (MTP) layer weights extracted from the upstream
checkpoint so the mtplx runtime can attach speculative decoding to the
expert-streaming models without downloading the full source checkpoint.

## Provenance

{provenance}
## Contents

- Architecture: `{arch[0] if arch else "unknown"}`
- MTP layer tensors as safetensors (see file list below)

{_inventory_section(plan)}
## Integrity

{integrity}

{USAGE_SECTION}
> License note: this repository inherits the upstream model license; set the
> `license` front-matter field before publishing.
"""
    tags = ["mlx", "mtplx", "mtp", "speculative-decoding"]
    return _front_matter(tags, base_model) + body


def render_card(root: Path) -> str:
    plan = build_plan(root, sample_records=0)
    if plan.kind == "expert-streaming":
        return render_expert_card(root, plan)
    return render_mtp_card(root, plan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="model artifact root directory")
    parser.add_argument(
        "--card",
        type=Path,
        default=None,
        help="model card README.md to stage (see the repo hf/ directory)",
    )
    parser.add_argument(
        "--write-card",
        type=Path,
        default=None,
        help="render a model card for this artifact to the given path and exit",
    )
    parser.add_argument(
        "--sample-records",
        type=int,
        default=4,
        help="records/tensors to re-hash as a spot check (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=51,
        help="deterministic sampling seed (default: 51)",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="write the machine-readable plan to this file",
    )
    parser.add_argument("--repo-id", default=None, help="target Hub repo id")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="staging directory for --execute (default: ./hf-staging/<name>)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="build the staging directory and upload (default: dry run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2
    if args.write_card is not None:
        payload = render_card(root)
        args.write_card.parent.mkdir(parents=True, exist_ok=True)
        args.write_card.write_text(payload, encoding="utf-8")
        print(f"wrote {args.write_card}")
        return 0
    plan = build_plan(
        root,
        sample_records=args.sample_records,
        seed=args.seed,
        card_source=args.card,
    )
    print_plan(plan)
    if args.plan_json:
        payload = {
            "root": str(plan.root),
            "kind": plan.kind,
            "total_bytes": plan.total_bytes,
            "entries": [
                {
                    "path_in_repo": entry.path_in_repo,
                    "source": None if entry.source is None else str(entry.source),
                    "bytes": entry.bytes,
                    "role": entry.role,
                    "note": entry.note,
                }
                for entry in plan.entries
            ],
            "excluded": [
                {"name": name, "bytes": size, "reason": reason}
                for name, size, reason in plan.excluded
            ],
            "findings": [
                {"severity": item.severity, "message": item.message}
                for item in plan.findings
            ],
        }
        args.plan_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.plan_json}")
    if not args.execute:
        if plan.errors:
            print(
                "dry run finished with validation errors; fix them before "
                "--execute",
                file=sys.stderr,
            )
            return 1
        return 0
    if not args.repo_id:
        print("error: --execute requires --repo-id", file=sys.stderr)
        return 2
    staging_dir = args.staging_dir or Path("hf-staging") / root.name
    return execute_upload(plan, staging_dir, args.repo_id)


if __name__ == "__main__":
    sys.exit(main())
