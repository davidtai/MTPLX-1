"""Construction-time contract for the DeepSeek V4 Flash DSpark artifact."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


_CONFIG_NAME = "config.json"
_INDEX_NAME = "model.safetensors.index.json"
_STAGE_KEY = re.compile(r"^mtp\.(\d+)\.")

_PHASE1_BLOCK_SIZE = 5
_PHASE1_MARKOV_RANK = 256
_PHASE1_NOISE_TOKEN_ID = 128799
_PHASE1_TARGET_LAYER_IDS = (40, 41, 42)
_PHASE1_STAGE_IDS = (0, 1, 2)

_REQUIRED_WEIGHT_KEYS = (
    "mtp.0.main_proj.weight",
    "mtp.0.attn.wq_a.weight",
    "mtp.1.attn.wq_a.weight",
    "mtp.2.attn.wq_a.weight",
    "mtp.2.hc_head.base",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.confidence_head.proj.weight",
)


class DSparkArtifactError(ValueError):
    """The selected model directory cannot install the Phase 1 DSpark lane."""


@dataclass(frozen=True)
class DSparkConfig:
    block_size: int
    markov_rank: int
    noise_token_id: int
    target_layer_ids: tuple[int, int, int]
    stage_ids: tuple[int, int, int]


@dataclass(frozen=True)
class VerifiedDSparkArtifact:
    root: Path
    config: DSparkConfig
    config_sha256: str
    index_sha256: str
    weight_map: Mapping[str, str]
    shards: tuple[str, ...]


def _read_json(path: Path, *, label: str) -> tuple[bytes, dict]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DSparkArtifactError(f"cannot read DSpark {label} at {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DSparkArtifactError(f"invalid JSON in DSpark {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DSparkArtifactError(f"DSpark {label} must be a JSON object: {path}")
    return raw, value


def _phase1_config(config: Mapping[str, object], stage_ids: tuple[int, ...]) -> DSparkConfig:
    try:
        block_size = int(config["dspark_block_size"])
        markov_rank = int(config["dspark_markov_rank"])
        noise_token_id = int(config["dspark_noise_token_id"])
        target_layer_ids = tuple(int(v) for v in config["dspark_target_layer_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DSparkArtifactError(f"incomplete DSpark configuration: {exc}") from exc

    observed = (
        block_size,
        markov_rank,
        noise_token_id,
        target_layer_ids,
        stage_ids,
    )
    expected = (
        _PHASE1_BLOCK_SIZE,
        _PHASE1_MARKOV_RANK,
        _PHASE1_NOISE_TOKEN_ID,
        _PHASE1_TARGET_LAYER_IDS,
        _PHASE1_STAGE_IDS,
    )
    if observed != expected:
        raise DSparkArtifactError(
            "unsupported DSpark Phase 1 contract: "
            f"observed={observed!r}, expected={expected!r}"
        )

    return DSparkConfig(
        block_size=block_size,
        markov_rank=markov_rank,
        noise_token_id=noise_token_id,
        target_layer_ids=_PHASE1_TARGET_LAYER_IDS,
        stage_ids=_PHASE1_STAGE_IDS,
    )


def open_verified_dspark_artifact(root: Path) -> VerifiedDSparkArtifact:
    """Open and qualify the fixed-K5 DSpark checkpoint before model execution."""

    try:
        artifact_root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise DSparkArtifactError(f"DSpark artifact directory is unavailable: {root}") from exc
    if not artifact_root.is_dir():
        raise DSparkArtifactError(f"DSpark artifact root is not a directory: {artifact_root}")

    config_raw, config = _read_json(artifact_root / _CONFIG_NAME, label="config")
    index_raw, index = _read_json(artifact_root / _INDEX_NAME, label="weight index")

    weight_map_value = index.get("weight_map")
    if not isinstance(weight_map_value, dict) or not weight_map_value:
        raise DSparkArtifactError("DSpark weight index has no non-empty weight_map")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map_value.items()):
        raise DSparkArtifactError("DSpark weight_map keys and shard names must be strings")
    weight_map = dict(weight_map_value)

    missing = tuple(key for key in _REQUIRED_WEIGHT_KEYS if key not in weight_map)
    if missing:
        raise DSparkArtifactError(f"DSpark artifact is missing required weights: {missing!r}")

    stage_ids = tuple(
        sorted(
            {
                int(match.group(1))
                for key in weight_map
                if (match := _STAGE_KEY.match(key)) is not None
            }
        )
    )
    dspark_config = _phase1_config(config, stage_ids)

    shards = tuple(sorted(set(weight_map.values())))
    for shard in shards:
        try:
            shard_path = (artifact_root / shard).resolve(strict=True)
        except OSError as exc:
            raise DSparkArtifactError(f"DSpark shard is unavailable: {shard}") from exc
        if not shard_path.is_relative_to(artifact_root) or not shard_path.is_file():
            raise DSparkArtifactError(f"DSpark shard is outside the artifact root: {shard}")

    return VerifiedDSparkArtifact(
        root=artifact_root,
        config=dspark_config,
        config_sha256=sha256(config_raw).hexdigest(),
        index_sha256=sha256(index_raw).hexdigest(),
        weight_map=MappingProxyType(weight_map),
        shards=shards,
    )
