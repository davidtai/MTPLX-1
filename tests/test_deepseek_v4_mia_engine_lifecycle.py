from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import mlx.core as mx

import mtplx.deepseek_v4_exl3 as exl3
import mtplx.deepseek_v4_mia_engine as mia_engine
from mtplx.models import deepseek_v4 as target_module
from mtplx.deepseek_v4_mia_engine import MiaDeepseekV4EnginePlan


class _FakeArray:
    def __init__(self, shape=(1,)):
        self.shape = tuple(shape)

    def __getitem__(self, _key):
        return self

    def astype(self, _dtype):
        return self


class _FakeTargetArena:
    def __init__(self, events):
        self.events = events
        self.cache = [object()]

    def acquire(self, _layers):
        self.events.append("target.acquire")
        return self.cache

    def release(self, caches):
        assert caches is self.cache
        self.events.append("target.release")


def _plan(events):
    return MiaDeepseekV4EnginePlan(
        context_capacity_tokens=384_000,
        max_batch_tokens=8_224,
        max_sequences=1,
        page_geometry=(),
        workspace_geometry=(),
        indexer_workspace=None,
        indexer_rope_table=None,
        mla_workspace=None,
        target_cache_arena=_FakeTargetArena(events),
        prewarm_signatures=(),
        installed_routes=(),
        target_artifact="target",
        draft_artifact="draft",
        artifact_small_file_sha256=(),
        identity="test-plan",
    )


def _patch_fake_mx(monkeypatch):
    monkeypatch.setattr(mx, "zeros", lambda shape, **_kwargs: _FakeArray(shape))
    monkeypatch.setattr(mx, "arange", lambda size, **_kwargs: _FakeArray((size,)))
    monkeypatch.setattr(mx, "array", lambda value, **_kwargs: _FakeArray((len(value),)))
    monkeypatch.setattr(mx, "argmax", lambda *_args, **_kwargs: _FakeArray((1,)))
    monkeypatch.setattr(mx, "concatenate", lambda *_args, **_kwargs: _FakeArray((1, 6)))
    monkeypatch.setattr(mx, "eval", lambda *_args, **_kwargs: None)


def test_shared_indexer_rope_table_is_built_once_and_owned_by_every_ratio4_layer(
    monkeypatch,
):
    inv_freq = object()
    shared_table = object()
    workspace = object()
    build_calls = []
    install_calls = []

    class FakeIndexer:
        def __init__(self):
            self._inv_freq = inv_freq

        def install_mia_paged_topk(self, installed_workspace, rope_table):
            install_calls.append((installed_workspace, rope_table))

    layers = tuple(
        SimpleNamespace(attn=SimpleNamespace(indexer=FakeIndexer()))
        for _ in range(3)
    )
    monkeypatch.setattr(
        mia_engine,
        "precompute_indexer_rope_table",
        lambda frequencies, *, max_positions: (
            build_calls.append((frequencies, max_positions)) or shared_table
        ),
    )

    actual = mia_engine._install_shared_indexer_resources(
        layers,
        (4, 128, 4),
        workspace,
        inv_freq,
    )

    assert actual is shared_table
    assert build_calls == [(inv_freq, 384_000)]
    assert install_calls == [
        (workspace, shared_table),
        (workspace, shared_table),
    ]


def test_exact_engine_binds_one_base_and_one_compress_rope_provider() -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        bindings = []

        class FakeAttention:
            def __init__(self, ratio):
                self.compress_ratio = ratio

            def install_mia_rope_provider(self, provider):
                bindings.append((self.compress_ratio, provider))

        model = SimpleNamespace(
            args=SimpleNamespace(
                qk_rope_head_dim=64,
                rope_theta=10_000.0,
                compress_rope_theta=160_000.0,
                original_seq_len=4_096,
                rope_factor=16.0,
                beta_fast=32,
                beta_slow=1,
            ),
            layers=tuple(
                SimpleNamespace(attn=FakeAttention(ratio))
                for ratio in (0, 4, 128, 4, 0)
            ),
            dspark=SimpleNamespace(
                stages=tuple(
                    SimpleNamespace(attn=FakeAttention(0)) for _ in range(3)
                )
            ),
        )

        base, compress = target_module.install_mia_target_rope_providers(
            model,
            max_positions=384_000,
        )

        assert model._mia_base_rope_provider is base
        assert model._mia_compress_rope_provider is compress
        assert base is not compress
        assert base.max_positions == 384_000
        assert compress.max_positions == 384_000
        draft = model._mia_draft_rope_provider
        assert draft.max_positions == 384_005
        assert draft is not base
        assert [
            (ratio, provider is base, provider is compress, provider is draft)
            for ratio, provider in bindings
        ] == [
            (0, True, False, False),
            (4, False, True, False),
            (128, False, True, False),
            (4, False, True, False),
            (0, True, False, False),
            (0, False, False, True),
            (0, False, False, True),
            (0, False, False, True),
        ]
        positions, _cos, _sin = draft.token_tables(384_000, 5)
        assert positions.tolist() == list(range(384_000, 384_005))
        with pytest.raises(ValueError, match="384k capacity"):
            base.token_tables(384_000, 5)
    finally:
        mx.set_default_device(previous)


def test_exact_target_forward_starts_one_shared_rope_epoch_per_chunk() -> None:
    events = []

    class Provider:
        def begin_forward(self):
            events.append("rope")

    model = target_module.Model.__new__(target_module.Model)
    model._mia_base_rope_provider = Provider()
    model._mia_compress_rope_provider = Provider()
    model._mia_draft_rope_provider = Provider()
    model.model = SimpleNamespace(
        _run_mia_hc_target_tail_taps=lambda inputs, cache: (
            events.append((inputs, cache)) or "result"
        )
    )
    inputs = object()
    cache = object()

    assert model._mia_target_forward(inputs, cache) == "result"
    assert events == ["rope", "rope", "rope", (inputs, cache)]


def test_exact_stacked_projection_installer_binds_all_named_owners(monkeypatch):
    validated = []
    built = []

    class FakeStack:
        @staticmethod
        def validate_pair(first, second):
            validated.append((first, second))

        def __init__(self, first, second):
            built.append((first, second))

    monkeypatch.setattr(target_module, "MiaStackedMXFP8Projection", FakeStack)
    monkeypatch.setattr(target_module, "MiaStackedDenseProjection", FakeStack)

    class FakeCompressor:
        def __init__(self, name):
            self.wkv = f"{name}.wkv"
            self.wgate = f"{name}.wgate"
            self.owner = None

        def install_mia_stacked_projection(self, owner):
            self.owner = owner

    class FakeAttention:
        def __init__(self, name, ratio):
            self.compress_ratio = ratio
            self.wq_a = f"{name}.wq_a"
            self.wkv = f"{name}.wkv"
            self.owner = None
            if ratio:
                self.compressor = FakeCompressor(f"{name}.compressor")
            if ratio == 4:
                self.indexer = SimpleNamespace(
                    compressor=FakeCompressor(f"{name}.indexer.compressor")
                )

        def install_mia_stacked_projection(self, owner):
            self.owner = owner

    ratios = (0, 0) + (4,) * 21 + (128,) * 20
    layers = tuple(
        SimpleNamespace(attn=FakeAttention(f"target.{index}", ratio))
        for index, ratio in enumerate(ratios)
    )
    stages = tuple(
        SimpleNamespace(attn=FakeAttention(f"draft.{index}", 0))
        for index in range(3)
    )
    model = SimpleNamespace(
        layers=layers,
        dspark=SimpleNamespace(stages=stages),
    )

    receipt = target_module.install_mia_stacked_projections(model)

    assert receipt == {
        "target_attention": 43,
        "draft_attention": 3,
        "main_compressor": 41,
        "indexer_compressor": 21,
    }
    assert len(validated) == len(built) == 108
    assert all(layer.attn.owner is not None for layer in layers)
    assert all(stage.attn.owner is not None for stage in stages)
    assert all(
        layer.attn.compressor.owner is not None
        for layer in layers
        if layer.attn.compress_ratio
    )
    assert all(
        layer.attn.indexer.compressor.owner is not None
        for layer in layers
        if layer.attn.compress_ratio == 4
    )


def test_ratio_specialized_mla_route_contract_rejects_generic_callables():
    expected = {
        0: (
            "_mia_cached_forward_uncompressed",
            "_mia_cached_attention_ratio0",
            "_mia_uncached_compressed",
            "_run_installed_window_nvfp4_sparse_mla",
            "_run_installed_window_nvfp4_prefill_mla",
        ),
        4: (
            "_mia_cached_forward_ratio4",
            "_mia_cached_attention_ratio4",
            "_mia_uncached_compressed",
            "_run_installed_indexed_paged_nvfp4_sparse_mla",
            "_run_installed_indexed_paged_nvfp4_prefill_mla",
        ),
        128: (
            "_mia_cached_forward_ratio128",
            "_mia_cached_attention_ratio128",
            "_mia_uncached_compressed",
            "_run_installed_sequential_paged_nvfp4_sparse_mla",
            "_run_installed_sequential_paged_nvfp4_prefill_mla",
        ),
    }

    assert mia_engine._MIA_ATTENTION_ROUTE_CONTRACTS == expected
    installed_names = {
        name
        for route_contract in expected.values()
        for name in route_contract
    }
    assert installed_names.isdisjoint(
        {
            "_mia_cached_attention",
            "_run_nvfp4_sparse_mla",
            "_run_paged_nvfp4_sparse_mla",
            "_run_nvfp4_prefill_mla",
            "_run_paged_nvfp4_prefill_mla",
        }
    )


def test_prewarm_releases_target_lease_when_first_forward_fails(monkeypatch):
    _patch_fake_mx(monkeypatch)
    events = []

    class FailingModel:
        layers = (object(),)

        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("target compile failed")

    with pytest.raises(RuntimeError, match="target compile failed"):
        _plan(events).prewarm(FailingModel())

    assert events == ["target.acquire", "target.release"]


def test_prewarm_releases_both_leases_when_proposal_fails(monkeypatch):
    _patch_fake_mx(monkeypatch)
    events = []
    fake = _FakeArray()

    class FakeSelector:
        def _select_rows(self, *_args, **_kwargs):
            return SimpleNamespace(indices=fake, lengths=fake)

    class FakeMHC:
        def post_pre(self, *_args, **_kwargs):
            return fake, fake, fake, fake

    class FakeDraftOwner:
        def release_mia_cache(self, _cache):
            events.append("draft.release")

    class FailingModel:
        def __init__(self):
            first = SimpleNamespace(
                attn_hc=fake,
                attn_norm=fake,
                attn=SimpleNamespace(
                    _output_projection_impl=lambda *_args: (
                        events.append("wo.m16") or fake
                    ),
                    _mia_qkv_plan=SimpleNamespace(
                        prefill_records=lambda *_args: (
                            events.append("qkv.m1024") or (fake, fake)
                        )
                    ),
                ),
            )
            selector_layer = SimpleNamespace(
                attn=SimpleNamespace(indexer=FakeSelector())
            )
            self.layers = (first, first, selector_layer)
            self.model = SimpleNamespace(
                layers=(first,),
                _mia_mhc=FakeMHC(),
            )
            self.dspark = FakeDraftOwner()
            self._mia_base_rope_provider = SimpleNamespace(
                token_tables=lambda *_args: (fake, fake, fake)
            )

        def __call__(self, *_args, **_kwargs):
            return fake, (fake,)

        def make_dspark_cache(self):
            events.append("draft.acquire")
            return [SimpleNamespace(ring=SimpleNamespace(records=fake))]

        def prefill_dspark(self, *_args, **_kwargs):
            return None

        def propose_dspark_k5(self, *_args, **_kwargs):
            raise RuntimeError("proposal compile failed")

    with pytest.raises(RuntimeError, match="proposal compile failed"):
        _plan(events).prewarm(FailingModel())

    assert events == [
        "target.acquire",
        "wo.m16",
        "qkv.m1024",
        "draft.acquire",
        "draft.release",
        "target.release",
    ]


def test_prewarm_release_failures_do_not_replace_the_primary_error():
    class FailingPlan:
        def release_target_cache(self, _cache):
            raise RuntimeError("target release failed")

    class FailingDraftOwner:
        def release_mia_cache(self, _cache):
            raise RuntimeError("draft release failed")

    primary = RuntimeError("proposal failed")

    mia_engine._release_prewarm_leases(
        FailingPlan(),
        SimpleNamespace(dspark=FailingDraftOwner()),
        [object()],
        [object()],
        primary,
    )

    assert str(primary) == "proposal failed"
    assert primary.__notes__ == [
        "prewarm cache release also failed: RuntimeError: draft release failed",
        "prewarm cache release also failed: RuntimeError: target release failed",
    ]


def test_verified_safetensors_rejects_path_swap_and_loads_only_hashed_fd(
    monkeypatch,
    tmp_path,
):
    shard = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    pinned = b"pinned-shard-bytes"
    swapped = b"swapped-shard-byte"
    assert len(pinned) == len(swapped)
    shard.write_bytes(pinned)
    replacement.write_bytes(swapped)

    observed_payloads = []

    def fake_load(stream, *, format):
        assert format == "safetensors"
        os.replace(replacement, shard)
        observed_payloads.append(stream.read())
        return {"payload": observed_payloads[-1]}

    monkeypatch.setattr(exl3.mx, "load", fake_load)

    with pytest.raises(ValueError, match="changed while loading"):
        exl3._load_verified_safetensors(
            shard,
            expected_bytes=len(pinned),
            expected_sha256=hashlib.sha256(pinned).hexdigest(),
        )

    assert observed_payloads == [pinned]
    assert shard.read_bytes() == swapped


def test_verified_safetensors_rejects_digest_before_loading(monkeypatch, tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"wrong bytes")
    load_calls = 0

    def fake_load(*_args, **_kwargs):
        nonlocal load_calls
        load_calls += 1
        return {}

    monkeypatch.setattr(exl3.mx, "load", fake_load)

    with pytest.raises(ValueError, match="checksum changed"):
        exl3._load_verified_safetensors(
            shard,
            expected_bytes=shard.stat().st_size,
            expected_sha256=hashlib.sha256(b"expected bytes").hexdigest(),
        )

    assert load_calls == 0


def test_verified_safetensors_loads_tiny_real_file_from_same_descriptor(tmp_path):
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), {"value": mx.array([7], dtype=mx.int32)})
    expected_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()

    loaded = exl3._load_verified_safetensors(
        shard,
        expected_bytes=shard.stat().st_size,
        expected_sha256=expected_sha256,
    )
    mx.eval(loaded["value"])

    assert loaded["value"].item() == 7


def test_verified_safetensors_rejects_in_place_change_during_load(
    monkeypatch,
    tmp_path,
):
    shard = tmp_path / "model.safetensors"
    pinned = b"pinned-shard-bytes"
    changed = b"changed-shard-byte"
    assert len(pinned) == len(changed)
    shard.write_bytes(pinned)

    def fake_load(stream, *, format):
        assert format == "safetensors"
        shard.write_bytes(changed)
        stream.seek(0)
        return {"payload": stream.read()}

    monkeypatch.setattr(exl3.mx, "load", fake_load)

    with pytest.raises(ValueError, match="changed while loading"):
        exl3._load_verified_safetensors(
            shard,
            expected_bytes=len(pinned),
            expected_sha256=hashlib.sha256(pinned).hexdigest(),
        )


def test_artifact_metadata_preserves_exact_manifest_and_defers_shard_digest(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    target_names = [f"carried-{index:03d}.safetensors" for index in range(1, 6)]
    target_names.extend(
        f"exl3-layer-{layer:03d}-tp1-rank0.safetensors"
        for layer in range(43)
    )
    for name in target_names:
        (target / name).write_bytes(b"t")
    (draft / "dspark-draft.safetensors").write_bytes(b"d")

    target_weight_map = {
        f"tensor.{index}": target_names[index % len(target_names)]
        for index in range(117_005)
    }
    target_documents = {
        "config.json": {},
        "tokenizer.json": {"version": "1.0", "model": {}},
        "tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"},
        "model.safetensors.index.json": {
            "metadata": {"total_size": 106_084_465_528},
            "weight_map": target_weight_map,
        },
        "rank-sliced-tp1-manifest.json": {
            "format": "rank-sliced-exl3-tp1-v1",
            "source_tp": 4,
            "target_tp": 1,
            "tensor_count": 117_005,
            "tensor_bytes": 106_084_465_528,
            "files": [
                {
                    "name": name,
                    "bytes": 1,
                    "sha256": hashlib.sha256(b"not-the-shard").hexdigest(),
                }
                for name in target_names
            ],
        },
        "EXL3_MANIFEST.json": {},
    }
    draft_weight_map = {
        f"tensor.{index}": "dspark-draft.safetensors"
        for index in range(1_249)
    }
    draft_documents = {
        "config.json": {},
        "model.safetensors.index.json": {
            "metadata": {"total_size": 1},
            "weight_map": draft_weight_map,
        },
        "DSPARK_DRAFT_PLAN.json": {
            "draft_experts": 64,
            "source_experts": 216,
            "tensor_count": 1_249,
            "total_size": 1,
            "sha256": {
                "dspark-draft.safetensors": hashlib.sha256(
                    b"not-the-draft"
                ).hexdigest()
            },
        },
    }

    def write_documents(root, documents):
        for name, document in documents.items():
            (root / name).write_text(json.dumps(document), encoding="utf-8")
        return {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in documents
        }

    monkeypatch.setattr(
        mia_engine,
        "_TARGET_SMALL_FILE_PINS",
        write_documents(target, target_documents),
    )
    monkeypatch.setattr(
        mia_engine,
        "_DRAFT_SMALL_FILE_PINS",
        write_documents(draft, draft_documents),
    )
    monkeypatch.setattr(mia_engine, "MIA_DRAFT_SHARD_BYTES", 1)

    validation = mia_engine.validate_pinned_mia_artifacts(target, draft)

    assert len(validation.target_shards) == 48
    assert len(validation.target_weight_map) == 117_005
    assert len(validation.draft_weight_map) == 1_249
    target_small_files = dict(validation.target_small_file_sha256)
    assert target_small_files["tokenizer.json"] == hashlib.sha256(
        (target / "tokenizer.json").read_bytes()
    ).hexdigest()
    assert target_small_files["tokenizer_config.json"] == hashlib.sha256(
        (target / "tokenizer_config.json").read_bytes()
    ).hexdigest()
    assert validation.target_shards[0].sha256 == hashlib.sha256(
        b"not-the-shard"
    ).hexdigest()

    replacement = tmp_path / "replacement-tokenizer.json"
    replacement.write_bytes((target / "tokenizer.json").read_bytes())
    os.replace(replacement, target / "tokenizer.json")
    with pytest.raises(ValueError, match="tokenizer file identity changed"):
        mia_engine.revalidate_pinned_mia_tokenizer_files(validation)

    (target / "tokenizer.json").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned Mia target file changed: tokenizer.json"):
        mia_engine.validate_pinned_mia_artifacts(target, draft)


def test_engine_identity_includes_every_pinned_small_file(monkeypatch):
    original = mia_engine._mia_engine_identity(384_000, 8_224)
    changed = dict(mia_engine._TARGET_SMALL_FILE_PINS)
    changed["tokenizer.json"] = "0" * 64
    monkeypatch.setattr(mia_engine, "_TARGET_SMALL_FILE_PINS", changed)

    assert mia_engine._mia_engine_identity(384_000, 8_224) != original
