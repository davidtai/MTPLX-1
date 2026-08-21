from pathlib import Path

import pytest

from mtplx.deepseek_v4_dspark_artifact import open_verified_dspark_artifact


MODEL = Path("/Users/davidtai/models/DeepSeek-V4-Flash-0731-2.4bit-mixed")

CONFIG_SHA256 = "44735712733fcf8f299bdf1faa1d87fac88f1917efe1d3876d6d4c582f79a68f"
INDEX_SHA256 = "f1332b2b209769c2db335954c2651652a8048e7d7dbf60296c2f2c0198715861"

REQUIRED_WEIGHT_KEYS = (
    "mtp.0.main_proj.weight",
    "mtp.0.attn.wq_a.weight",
    "mtp.1.attn.wq_a.weight",
    "mtp.2.attn.wq_a.weight",
    "mtp.2.hc_head.base",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.confidence_head.proj.weight",
)


def test_real_dspark_artifact_contract() -> None:
    artifact = open_verified_dspark_artifact(MODEL)

    assert artifact.root == MODEL.resolve()
    assert artifact.config.block_size == 5
    assert artifact.config.markov_rank == 256
    assert artifact.config.noise_token_id == 128799
    assert artifact.config.target_layer_ids == (40, 41, 42)
    assert artifact.config.stage_ids == (0, 1, 2)
    assert artifact.config_sha256 == CONFIG_SHA256
    assert artifact.index_sha256 == INDEX_SHA256
    assert all(key in artifact.weight_map for key in REQUIRED_WEIGHT_KEYS)
    assert all((artifact.root / shard).is_file() for shard in artifact.shards)

    with pytest.raises(TypeError):
        artifact.weight_map[REQUIRED_WEIGHT_KEYS[0]] = "replacement.safetensors"
