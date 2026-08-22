"""Construction-owned execution plan for the pinned Mia/Sero DSpark model.

The exact lane is deliberately closed over one artifact and one serving
geometry.  Validation happens here, before a cache or request exists.  The
request path receives already-bound target, draft, cache, and phase callables;
it never re-reads launcher settings or probes whether the exact lane applies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


MIA_CONTEXT_CAPACITY = 384_000
MIA_MAX_BATCH_TOKENS = 8_224
MIA_LONG_PREFILL_CHUNK = 1_024
MIA_MAX_SEQUENCES = 1
MIA_TARGET_LAYERS = 43
MIA_DSPARK_STAGES = 3
MIA_DSPARK_BLOCK = 5
MIA_TARGET_TAPS = (40, 41, 42)
MIA_TARGET_EXPERTS = 216
MIA_DRAFT_EXPERTS = 64
MIA_TOPK = 6
MIA_MXFP8_MODULES = 390
MIA_INDEX_TOPK = 512
MIA_WINDOW = 128
MIA_HEADS = 64
MIA_HEAD_DIM = 512
MIA_ROPE_DIM = 64
MIA_HIDDEN = 4096
MIA_HC = 4
MIA_DRAFT_SHARD_BYTES = 3_157_508_012
MIA_DFLASH_COMMIT = "d67e6e4788f82c114b8f4efee8c62501d6cf3386"

_TARGET_SMALL_FILE_PINS = {
    "config.json": "39f3a9e158019dc34dd943b64f874cfc43e9e392e6ce9215a56f2e183d661d90",
    "model.safetensors.index.json": "b7a450f88c99aee7f6d44ecb127e91e45ab5ccb1a0dad49ca9eabb90b400c304",
    "rank-sliced-tp1-manifest.json": "cee5b97698e16433f88e7ca23ab529acaa13628ae4af3ea18590ba4060c1203e",
    "EXL3_MANIFEST.json": "1e35cbbc33a977606a950928fba4c6660c7df0134bfab9472dd6d851be894125",
}
_DRAFT_SMALL_FILE_PINS = {
    "config.json": "8dcd2ae923a8e3149454f4db1f1e03109625b19f137995d26f45d357212ba306",
    "model.safetensors.index.json": "c0d0e18e8c84fe6f1b7dc6991a4ba5765d1965f21f8892887aa01169fc2ba2b3",
    "DSPARK_DRAFT_PLAN.json": "d7a45cc065363ec79516593d8910d0be36e6e589d093ad6ab4a3603dbf92b426",
}


def _callable_name(value: Any) -> str:
    """Return the installed implementation name, unwrapping bound partials."""

    current = value
    while callable(getattr(current, "func", None)):
        current = current.func
    return str(getattr(current, "__name__", type(current).__name__))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_small_files(root: Path, pins: dict[str, str], label: str) -> None:
    for name, expected in pins.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"pinned {label} file is absent: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"pinned {label} file changed: {name} "
                f"observed={observed}, expected={expected}"
            )


def validate_pinned_mia_artifacts(target_root: Path, draft_root: Path) -> None:
    """Reject a different model before allocating model or cache storage."""

    target_root = Path(target_root).resolve()
    draft_root = Path(draft_root).resolve()
    _validate_small_files(target_root, _TARGET_SMALL_FILE_PINS, "Mia target")
    _validate_small_files(draft_root, _DRAFT_SMALL_FILE_PINS, "K64 DSpark")

    target_manifest = json.loads(
        (target_root / "rank-sliced-tp1-manifest.json").read_text()
    )
    manifest_contract = (
        target_manifest.get("format"),
        int(target_manifest.get("source_tp", 0)),
        int(target_manifest.get("target_tp", 0)),
        int(target_manifest.get("tensor_count", 0)),
        int(target_manifest.get("tensor_bytes", 0)),
    )
    if manifest_contract != (
        "rank-sliced-exl3-tp1-v1",
        4,
        1,
        117_005,
        106_084_465_528,
    ):
        raise ValueError(
            "pinned Mia TP1 manifest contract changed: "
            f"{manifest_contract!r}"
        )
    files = target_manifest.get("files")
    if not isinstance(files, list) or len(files) != 48:
        raise ValueError("pinned Mia TP1 manifest must own exactly 48 shards")
    verified_bytes = 0
    for entry in files:
        shard = target_root / str(entry["name"])
        expected_bytes = int(entry["bytes"])
        if not shard.is_file() or shard.stat().st_size != expected_bytes:
            raise ValueError(
                f"pinned Mia TP1 shard size changed: {shard.name}"
            )
        expected_sha256 = str(entry.get("sha256", ""))
        if len(expected_sha256) != 64 or _sha256(shard) != expected_sha256:
            raise ValueError(
                f"pinned Mia TP1 shard checksum changed: {shard.name}"
            )
        verified_bytes += expected_bytes
    if verified_bytes != int(target_manifest["tensor_bytes"]):
        raise ValueError("pinned Mia TP1 manifest byte total changed")

    draft_plan = json.loads((draft_root / "DSPARK_DRAFT_PLAN.json").read_text())
    draft_index = json.loads(
        (draft_root / "model.safetensors.index.json").read_text()
    )
    if (
        int(draft_plan.get("draft_experts", 0)) != MIA_DRAFT_EXPERTS
        or int(draft_plan.get("source_experts", 0)) != MIA_TARGET_EXPERTS
        or int(draft_plan.get("tensor_count", 0)) != 1249
        or int(draft_index.get("metadata", {}).get("total_size", 0))
        != int(draft_plan.get("total_size", -1))
    ):
        raise ValueError("pinned K64 DSpark derivation manifest changed")
    weight_map = draft_index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 1249:
        raise ValueError("pinned K64 DSpark weight index changed")
    filenames = set(weight_map.values())
    if filenames != {"dspark-draft.safetensors"}:
        raise ValueError("pinned K64 DSpark must own one packaged weight shard")
    draft_shard = draft_root / "dspark-draft.safetensors"
    if not draft_shard.is_file():
        raise FileNotFoundError(draft_shard)
    if draft_shard.stat().st_size != MIA_DRAFT_SHARD_BYTES:
        raise ValueError("pinned K64 DSpark shard size changed")
    draft_sha256 = str(draft_plan.get("sha256", {}).get(draft_shard.name, ""))
    if len(draft_sha256) != 64 or _sha256(draft_shard) != draft_sha256:
        raise ValueError("pinned K64 DSpark shard checksum changed")


@dataclass(frozen=True, slots=True)
class MiaPageGeometry:
    layer_id: int
    compress_ratio: int
    compressed_capacity: int
    attention_record_bytes: int
    index_record_bytes: int


@dataclass(frozen=True, slots=True)
class MiaWorkspaceGeometry:
    name: str
    shape: tuple[int, ...]
    dtype: str
    ownership: str


@dataclass(frozen=True, slots=True)
class MiaPrewarmSignature:
    name: str
    rows: int
    phase: str


class MiaTargetCacheArena:
    """One persistent vLLM-style page lease for the single-sequence model."""

    def __init__(
        self,
        layers: tuple[Any, ...],
        *,
        capacity_tokens: int,
        max_batch_tokens: int,
    ) -> None:
        import mlx.core as mx

        from mtplx.models.deepseek_v4 import DeepseekV4NVFP4Cache

        self._layer_identity = tuple(id(layer) for layer in layers)
        self._caches = tuple(
            DeepseekV4NVFP4Cache(
                window_size=layer.attn.window_size,
                compress_ratio=layer.attn.compress_ratio,
                head_dim=layer.attn.head_dim,
                capacity_tokens=int(capacity_tokens),
                max_batch_tokens=int(max_batch_tokens),
            )
            for layer in layers
        )
        expected_window_capacity = (
            int(max_batch_tokens)
            + int(layers[0].attn.window_size)
            + int(self._caches[0].rollback_capacity)
        )
        if any(
            getattr(cache.window, "mode", None)
            != "nvfp4_stock432_fixed_window"
            or int(getattr(cache.window, "capacity", 0))
            != expected_window_capacity
            for cache in self._caches
        ):
            raise ValueError("Mia target cache arena did not install fixed SWA pages")
        journal_buffers = []
        for layer, cache in zip(layers, self._caches, strict=True):
            ratio = int(layer.attn.compress_ratio)
            if ratio == 0:
                continue
            expected_rows = (2 if ratio == 4 else 1) * ratio + int(
                cache.rollback_capacity
            )
            expected_width = (2 if ratio == 4 else 1) * int(cache.head_dim)
            if (
                getattr(cache.comp, "mode", None) != "mia_fixed_compressor_state"
                or int(getattr(cache.comp, "rollback_rows", 0)) != expected_rows
                or int(getattr(cache.comp, "state_width", 0)) != expected_width
                or getattr(cache.compressed, "mode", None)
                != "nvfp4_stock432_paged"
                or int(getattr(cache.compressed, "capacity", 0))
                != (int(capacity_tokens) + ratio - 1) // ratio
            ):
                raise ValueError(
                    "Mia target cache arena did not install fixed compressor pages"
                )
            journal_buffers.extend(cache.comp.journal_buffers)
            if ratio == 4:
                if (
                    getattr(cache.index_comp, "mode", None)
                    != "mia_fixed_compressor_state"
                    or int(getattr(cache.index_comp, "rollback_rows", 0))
                    != expected_rows
                    or int(getattr(cache.index_comp, "state_width", 0)) != 256
                    or getattr(cache.index_compressed, "mode", None)
                    != "fp8_e4m3_ue8m0_scale132_paged"
                    or int(getattr(cache.index_compressed, "capacity", 0))
                    != (int(capacity_tokens) + ratio - 1) // ratio
                ):
                    raise ValueError(
                        "Mia target cache arena did not install fixed indexer state"
                    )
                journal_buffers.extend(cache.index_comp.journal_buffers)
        mx.eval(*journal_buffers)
        self._leased = False

    @property
    def layer_count(self) -> int:
        return len(self._caches)

    @property
    def leased(self) -> bool:
        return self._leased

    def _reset(self) -> None:
        for cache in self._caches:
            cache.state = None

    def acquire(self, layers: tuple[Any, ...]) -> list[Any]:
        if tuple(id(layer) for layer in layers) != self._layer_identity:
            raise ValueError("Mia target cache arena belongs to a different model")
        if self._leased:
            raise RuntimeError("Mia target cache arena already owns the active request")
        self._reset()
        self._leased = True
        return list(self._caches)

    def release(self, caches: list[Any]) -> None:
        if not self._leased:
            raise RuntimeError("Mia target cache arena has no active request")
        if len(caches) != len(self._caches) or any(
            observed is not expected
            for observed, expected in zip(caches, self._caches, strict=True)
        ):
            raise ValueError("Mia target cache release does not match its page lease")
        self._reset()
        self._leased = False


@dataclass(frozen=True, slots=True)
class MiaDeepseekV4EnginePlan:
    """Immutable plan installed after all exact callables and weights exist."""

    context_capacity_tokens: int
    max_batch_tokens: int
    max_sequences: int
    page_geometry: tuple[MiaPageGeometry, ...]
    workspace_geometry: tuple[MiaWorkspaceGeometry, ...]
    indexer_workspace: Any
    mla_workspace: Any
    target_cache_arena: MiaTargetCacheArena
    prewarm_signatures: tuple[MiaPrewarmSignature, ...]
    installed_routes: tuple[str, ...]
    target_artifact: str
    draft_artifact: str
    identity: str

    def make_target_cache(self, layers) -> list[Any]:
        return self.target_cache_arena.acquire(tuple(layers))

    def release_target_cache(self, caches: list[Any]) -> None:
        self.target_cache_arena.release(caches)

    def prewarm(self, model) -> dict[str, Any]:
        """Compile every serving phase with bounded, disposable cache state.

        A 128-row prefill reaches both ratio-4 and ratio-128 compressors and the
        large-M Trellis plan.  One physical M6 target verify reaches the small-M
        target plan, while the packaged K5 proposal reaches all three K64 draft
        stages and the sequential Markov head.  Custom Metal kernels carry their
        row counts as runtime arguments, so these dispatches compile the finite
        pipeline set without a model-length-sized warmup.
        """

        import mlx.core as mx

        from mtplx.attention_context import attention_phase
        from mtplx.deepseek_v4_paged_indexer import (
            INDEXER_RECORD_BYTES,
            MiaIndexerQueryRecords,
            PagedMiaIndexerRecords,
        )

        target_cache = self.make_target_cache(model.layers)
        prefill_ids = mx.zeros((1, MIA_WINDOW), dtype=mx.uint32)
        with attention_phase("prefill"):
            prefill_logits, target_taps = model(
                prefill_ids,
                cache=target_cache,
                return_hidden=True,
                logits_keep=1,
            )
        mx.eval(prefill_logits, *target_taps)

        # SparkInfer changes the repeated post-pre projection at M=384.  Warm
        # that installed component directly so startup covers the large-M mHC
        # signature without a second full target-model pass.
        mhc_rows = 384
        first_layer = model.model.layers[0]
        mhc_residual, mhc_post, mhc_comb, mhc_y = model.model._mia_mhc.post_pre(
            mx.zeros((mhc_rows, MIA_HIDDEN), dtype=mx.bfloat16),
            mx.zeros((mhc_rows, MIA_HC, MIA_HIDDEN), dtype=mx.bfloat16),
            mx.zeros((mhc_rows, MIA_HC), dtype=mx.float32),
            mx.zeros((mhc_rows, MIA_HC, MIA_HC), dtype=mx.float32),
            first_layer.attn_hc,
            first_layer.attn_norm,
        )
        mx.eval(mhc_residual, mhc_post, mhc_comb, mhc_y)

        # The real sparse selector starts only after 512 ratio-4 rows. Compile
        # its prefill and decode engines against a tiny synthetic paged view so
        # startup does not need a 2K-token model forward merely to reach that
        # phase boundary.
        selector_rows = MIA_INDEX_TOPK + 1
        selector_block = 64
        selector_blocks = (selector_rows + selector_block - 1) // selector_block
        selector_view = PagedMiaIndexerRecords(
            records=mx.zeros(
                (selector_blocks, selector_block, INDEXER_RECORD_BYTES),
                dtype=mx.uint8,
            ),
            block_table=mx.arange(selector_blocks, dtype=mx.int32),
            length=selector_rows,
            block_size=selector_block,
        )
        selector_q = MiaIndexerQueryRecords(
            mx.zeros((1, 1, MIA_HEADS, INDEXER_RECORD_BYTES), dtype=mx.uint8)
        )
        selector_weights = mx.zeros((1, 1, MIA_HEADS), dtype=mx.float32)
        selector_positions = mx.array([selector_rows * 4 - 1], dtype=mx.int32)
        selector = model.layers[2].attn.indexer
        with attention_phase("prefill"):
            prefill_selection = selector._select_rows(
                selector_q,
                selector_weights,
                selector_positions,
                selector_view,
            )
        with attention_phase("decode_verify"):
            decode_selection = selector._select_rows(
                selector_q,
                selector_weights,
                selector_positions,
                selector_view,
            )
        mx.eval(
            prefill_selection.indices,
            prefill_selection.lengths,
            decode_selection.indices,
            decode_selection.lengths,
        )

        draft_cache = model.make_dspark_cache()
        model.prefill_dspark(target_taps, draft_cache)
        mx.eval(*(cache.ring.records for cache in draft_cache))
        primary = mx.argmax(prefill_logits[:, -1], axis=-1).astype(mx.uint32)
        proposal = model.propose_dspark_k5(
            primary,
            draft_cache,
            start_pos=MIA_WINDOW,
        )
        mx.eval(proposal.future_tokens, proposal.neural_logits)

        verify_ids = mx.concatenate(
            [primary[:, None], proposal.future_tokens], axis=1
        )
        with attention_phase("decode_verify"):
            verify_logits, verify_taps = model(
                verify_ids,
                cache=target_cache,
                return_hidden=True,
            )
        mx.eval(verify_logits, *verify_taps)
        model.dspark.release_mia_cache(draft_cache)
        self.release_target_cache(target_cache)
        return {
            "identity": self.identity,
            "signatures": tuple(signature.name for signature in self.prewarm_signatures),
            "prefill_rows": MIA_WINDOW,
            "mhc_prefill_rows": mhc_rows,
            "verify_rows": MIA_DSPARK_BLOCK + 1,
            "draft_rows": MIA_DSPARK_BLOCK,
        }


def build_mia_engine_plan(
    model,
    *,
    target_root: Path,
    draft_root: Path,
    context_capacity_tokens: int = MIA_CONTEXT_CAPACITY,
    max_batch_tokens: int = MIA_MAX_BATCH_TOKENS,
) -> MiaDeepseekV4EnginePlan:
    """Validate the installed graph once and seal its exact serving geometry."""

    context_capacity_tokens = int(context_capacity_tokens)
    max_batch_tokens = int(max_batch_tokens)
    if (
        context_capacity_tokens != MIA_CONTEXT_CAPACITY
        or max_batch_tokens != MIA_MAX_BATCH_TOKENS
    ):
        raise ValueError(
            "the pinned Mia launcher requires context=384000 and "
            "max_num_batched_tokens=8224"
        )

    args = model.args
    layers = tuple(model.layers)
    stages = tuple(getattr(model.dspark, "stages", ()))
    ratios = tuple(int(layer.attn.compress_ratio) for layer in layers)
    from mtplx.deepseek_v4_paged_indexer import MiaIndexerWorkspace
    from mtplx.kernels.deepseek_v4_nvfp4_mla import mia_mla_workspace
    import mlx.core as mx
    from mlx.utils import tree_flatten

    observed = (
        len(layers),
        len(stages),
        int(args.hidden_size),
        int(args.hc_mult),
        int(args.num_attention_heads),
        int(args.head_dim),
        int(args.qk_rope_head_dim),
        int(args.window_size),
        int(args.index_topk),
        int(args.n_routed_experts),
        int(args.num_experts_per_tok),
        int(model.dspark.args.n_routed_experts),
        int(model.dspark.block_size),
        tuple(int(value) for value in model.dspark.target_layer_ids),
        ratios.count(0),
        ratios.count(4),
        ratios.count(128),
    )
    expected = (
        MIA_TARGET_LAYERS,
        MIA_DSPARK_STAGES,
        MIA_HIDDEN,
        MIA_HC,
        MIA_HEADS,
        MIA_HEAD_DIM,
        MIA_ROPE_DIM,
        MIA_WINDOW,
        MIA_INDEX_TOPK,
        MIA_TARGET_EXPERTS,
        MIA_TOPK,
        MIA_DRAFT_EXPERTS,
        MIA_DSPARK_BLOCK,
        MIA_TARGET_TAPS,
        2,
        21,
        20,
    )
    if observed != expected:
        raise ValueError(
            f"installed Mia engine topology changed: {observed!r} != {expected!r}"
        )

    ratio4_capacity = (context_capacity_tokens + 3) // 4
    indexer_workspace = MiaIndexerWorkspace.allocate(
        max_query_rows=max_batch_tokens,
        topk=MIA_INDEX_TOPK,
        sentinel=ratio4_capacity,
    )
    for layer, ratio in zip(layers, ratios, strict=True):
        if ratio == 4:
            layer.attn.indexer.install_mia_paged_topk(indexer_workspace)
    mla_workspace = mia_mla_workspace()

    load_receipt = dict(getattr(model, "_mia_target_load_receipt", {}))
    target_parameter_count = sum(
        not name.startswith("mtp.")
        for name, _value in tree_flatten(model.parameters())
    )
    if (
        load_receipt.get("mode") != "bounded_one_shard"
        or int(load_receipt.get("source_shards", 0)) != 48
        or int(load_receipt.get("carried_shards", 0)) != 5
        or int(load_receipt.get("exl3_layer_shards", 0)) != MIA_TARGET_LAYERS
        or int(load_receipt.get("installed_parameters", 0))
        != target_parameter_count
    ):
        raise ValueError("the Mia target was not installed by the bounded shard loader")

    # These flags are installed at construction and never checked again by a
    # token path.  They certify that no generic arithmetic owner remains bound.
    router_contract_prefix = (
        "bf16xbf16_fp32_k4096",
        "softplus_threshold20_sqrt",
    )
    for layer_id, layer in enumerate(layers):
        ratio = int(layer.attn.compress_ratio)
        expected_gate = (
            "_mia_hash_route"
            if layer_id < int(args.num_hash_layers)
            else "_mia_score_route"
        )
        expected_sparse = (
            "_run_paged_nvfp4_sparse_mla" if ratio else "_run_nvfp4_sparse_mla"
        )
        expected_prefill = (
            "_run_paged_nvfp4_prefill_mla" if ratio else "_run_nvfp4_prefill_mla"
        )
        expected_compressed = (
            "_mia_uncached_compressed"
            if ratio
            else "_legacy_uncached_compressed"
        )
        expected_forward = {
            0: "_mia_cached_forward_uncompressed",
            4: "_mia_cached_forward_ratio4",
            128: "_mia_cached_forward_ratio128",
        }[ratio]
        installed = (
            bool(getattr(layer.ffn.switch_mlp, "_trellis_installed", False)),
            _callable_name(getattr(layer.ffn, "_forward_impl", None)),
            _callable_name(getattr(layer.ffn, "_input_rows_impl", None)),
            _callable_name(getattr(layer.ffn.gate, "_route_impl", None)),
            _callable_name(getattr(layer.attn, "_forward_impl", None)),
            _callable_name(getattr(layer.attn, "_cached_attention_impl", None)),
            _callable_name(getattr(layer.attn, "_uncached_kv_impl", None)),
            _callable_name(
                getattr(layer.attn, "_uncached_compressed_impl", None)
            ),
            _callable_name(getattr(layer.attn, "_cached_mask_impl", None)),
            _callable_name(getattr(layer.attn, "_nvfp4_sparse_mla", None)),
            _callable_name(getattr(layer.attn, "_nvfp4_prefill_mla", None)),
            type(getattr(layer.attn, "_output_projection_impl", None)).__name__,
            getattr(layer.attn, "_mia_mla_workspace", None) is mla_workspace,
            getattr(getattr(layer.attn, "_mia_attn_sink", None), "dtype", None)
            == mx.float32,
            getattr(layer.ffn.gate, "_mia_router_contract", None),
        )
        expected_router_contract = router_contract_prefix + (
            "hash_tid2eid_int32"
            if layer_id < int(args.num_hash_layers)
            else "bias_selection_fp32",
            "unbiased_normalize_scale1p5",
        )
        required = (
            True,
            "_mia_exl3_forward",
            "_required_input_rows",
            expected_gate,
            expected_forward,
            "_mia_cached_attention",
            "_mia_uncached_kv",
            expected_compressed,
            "_no_additive_mask",
            expected_sparse,
            expected_prefill,
            "_MiaInverseRopeGatherOLora",
            True,
            True,
            expected_router_contract,
        )
        if installed != required:
            raise ValueError(
                f"Mia target layer {layer_id} route changed: "
                f"{installed!r} != {required!r}"
            )
        if ratio:
            compressor_route = _callable_name(
                getattr(layer.attn.compressor, "_mia_record_impl", None)
            )
            if compressor_route != "_nvfp4_record_impl":
                raise ValueError(
                    f"Mia target layer {layer_id} compressor route changed"
                )
        if ratio == 4:
            indexer = layer.attn.indexer
            indexer_routes = (
                getattr(indexer, "_mia_workspace", None) is indexer_workspace,
                _callable_name(getattr(indexer, "_query_components_impl", None)),
                _callable_name(getattr(indexer, "_mia_query_records", None)),
                _callable_name(getattr(indexer, "_prepare_query_rows", None)),
                _callable_name(getattr(indexer, "_select_rows", None)),
                _callable_name(
                    getattr(indexer.compressor, "_mia_record_impl", None)
                ),
            )
            if indexer_routes != (
                True,
                "_mia_query_components",
                "_run_installed_indexer_query_records",
                "_native_query_rows",
                "_run_installed_paged_indexer_phase_topk",
                "_indexer_record_impl",
            ):
                raise ValueError(
                    f"Mia target layer {layer_id} indexer route changed: "
                    f"{indexer_routes!r}"
                )

    for stage_id, stage in enumerate(stages):
        switch = stage.ffn.switch_mlp
        installed = (
            type(stage.attn).__name__,
            type(switch).__name__,
            _callable_name(getattr(stage.attn, "_forward_impl", None)),
            _callable_name(getattr(stage.attn, "_pack_draft_records", None)),
            _callable_name(getattr(stage.ffn, "_forward_impl", None)),
            _callable_name(getattr(stage.ffn, "_input_rows_impl", None)),
            _callable_name(getattr(stage.ffn.gate, "_route_impl", None)),
            tuple(
                str(getattr(getattr(switch, name, None), "mode", ""))
                for name in ("gate_proj", "up_proj", "down_proj")
            ),
            _callable_name(getattr(stage.attn, "_dspark_k5_mla", None)),
            tuple(
                int(value)
                for value in getattr(
                    getattr(
                        stage.attn,
                        "_mia_draft_position_offsets",
                        None,
                    ),
                    "shape",
                    (),
                )
            ),
            getattr(
                getattr(
                    stage.attn,
                    "_mia_draft_position_offsets",
                    None,
                ),
                "dtype",
                None,
            ),
            getattr(getattr(stage.attn, "_mia_attn_sink", None), "dtype", None)
            == mx.float32,
            type(getattr(stage.attn, "_output_projection_impl", None)).__name__,
            getattr(stage.ffn.gate, "_mia_router_contract", None),
        )
        required = (
            "DeepseekV4DSparkAttention",
            "SwitchGLU",
            "_run_k5",
            "_run_pack_stock432",
            "_stock_forward",
            "_required_input_rows",
            "_mia_score_route",
            ("mxfp4", "mxfp4", "mxfp4"),
            "_run_dspark_k5_nvfp4_mla",
            (MIA_DSPARK_BLOCK,),
            mx.int32,
            True,
            "_MiaInverseRopeGatherOLora",
            router_contract_prefix
            + ("bias_selection_fp32", "unbiased_normalize_scale1p5"),
        )
        if installed != required:
            raise ValueError(
                f"Mia K64 DSpark stage {stage_id} route changed: "
                f"{installed!r} != {required!r}"
            )
    target_mhc = getattr(model.model, "_mia_mhc", None)
    draft_mhc = getattr(model.dspark, "_mia_mhc", None)
    target_connections = tuple(
        connection
        for layer in layers
        for connection in (layer.attn_hc, layer.ffn_hc)
    )
    draft_connections = tuple(
        connection
        for stage in stages
        for connection in (stage.attn_hc, stage.ffn_hc)
    )
    target_broadcast = getattr(
        getattr(target_connections[0], "_mia_mhc_weight", None),
        "fn_broadcast",
        None,
    )
    draft_broadcast = getattr(
        getattr(draft_connections[0], "_mia_mhc_weight", None),
        "fn_broadcast",
        None,
    )
    mhc_route_contract = (
        "broadcast_fn_fp32",
        "tiny_split32_fp32",
        "prefill_post_pre_bf16_mma_bm64_fp32",
        "compact_gram_finalize",
    )
    mhc_installed = (
        target_mhc is not None,
        draft_mhc is not None,
        getattr(model.model.embed_tokens.weight, "dtype", None) == mx.bfloat16,
        getattr(target_mhc, "max_tokens", None),
        getattr(draft_mhc, "max_tokens", None),
        getattr(target_mhc, "prefill_min_rows", None),
        getattr(draft_mhc, "prefill_min_rows", None),
        getattr(target_mhc, "prefill_block_m", None),
        getattr(draft_mhc, "prefill_block_m", None),
        getattr(target_mhc, "route_contract", None),
        getattr(draft_mhc, "route_contract", None),
        getattr(target_mhc, "bound_hyper_connections", None),
        getattr(draft_mhc, "bound_hyper_connections", None),
        all(
            getattr(connection, "_mia_mhc_weight", None) is not None
            and connection._mia_mhc_weight.fn_bf16.dtype == mx.bfloat16
            for connection in target_connections + draft_connections
        ),
        tuple(getattr(target_broadcast, "shape", ())) == (24, MIA_HIDDEN),
        getattr(target_broadcast, "dtype", None) == mx.float32,
        tuple(getattr(draft_broadcast, "shape", ())) == (24, MIA_HIDDEN),
        getattr(draft_broadcast, "dtype", None) == mx.float32,
    )
    expected_mhc = (
        True,
        True,
        True,
        MIA_MAX_BATCH_TOKENS,
        MIA_MAX_BATCH_TOKENS,
        384,
        384,
        64,
        64,
        mhc_route_contract,
        mhc_route_contract,
        MIA_TARGET_LAYERS * 2,
        MIA_DSPARK_STAGES * 2,
        True,
        True,
        True,
        True,
        True,
    )
    if mhc_installed != expected_mhc:
        raise ValueError(
            f"the Mia mHC execution plan changed: {mhc_installed!r} "
            f"!= {expected_mhc!r}"
        )
    if (
        target_mhc is None
        or draft_mhc is None
        or _callable_name(getattr(model.model, "_hc_hidden_impl", None))
        != "_mia_hc_hidden"
        or _callable_name(getattr(model.model, "_collapse_impl", None))
        != "_mia_collapse"
        or _callable_name(
            getattr(model.model, "_run_mia_hc_target_tail_taps", None)
        )
        != "_run_mia_hc_target_tail_taps"
        or _callable_name(getattr(model, "_target_forward_route", None))
        != "_mia_target_forward"
        or _callable_name(getattr(model, "mia_dflash_forward", None))
        != "mia_dflash_forward"
        or _callable_name(getattr(model.dspark, "_propose_impl", None))
        != "_mia_propose_k5"
        or _callable_name(getattr(model.dspark, "_make_cache_impl", None))
        != "_acquire_mia_cache"
        or len(tuple(getattr(model.dspark, "_mia_cache_arena", ())))
        != MIA_DSPARK_STAGES
        or any(
            getattr(getattr(cache, "ring", None), "mode", None)
            != "nvfp4_stock432_fixed_ring"
            or int(getattr(getattr(cache, "ring", None), "_capacity_rows", 0))
            != MIA_WINDOW
            for cache in tuple(getattr(model.dspark, "_mia_cache_arena", ()))
        )
        or bool(getattr(model.dspark, "_mia_cache_leased", True))
        or _callable_name(getattr(model.dspark, "_draft_input_ids_k5", None))
        != "_draft_input_ids_k5"
        or _callable_name(getattr(model.dspark, "_mia_draft_input_ids_k5", None))
        != "_mia_draft_input_ids_k5"
        or tuple(getattr(getattr(model.dspark, "_mia_noise_tail", None), "shape", ()))
        != (1, MIA_DSPARK_BLOCK - 1)
        or getattr(getattr(model.dspark, "_mia_noise_tail", None), "dtype", None)
        != mx.uint32
        or type(getattr(stages[-1], "markov_head", None)).__name__
        != "DSparkMarkovHead"
    ):
        raise ValueError("the carried Mia target/DSpark state machines are not installed")
    quantized_receipt = dict(getattr(model, "_mia_quantized_modules", {}))
    installed_quantized = {
        path: str(module.mode)
        for path, module in model.named_modules()
        if str(getattr(module, "mode", "")) in {"mxfp4", "mxfp8"}
    }
    if installed_quantized != quantized_receipt:
        raise ValueError(
            "the pinned Mia FP8/FP4 module route is not fully installed"
        )
    if (
        sum(mode == "mxfp4" for mode in quantized_receipt.values()) != 9
        or sum(mode == "mxfp8" for mode in quantized_receipt.values())
        != MIA_MXFP8_MODULES
    ):
        raise ValueError("the pinned K64 draft/general FP8 storage contract changed")

    # Physical pages are the final construction allocation. Every artifact,
    # topology, storage, callable, and quantization seal above must pass first.
    target_cache_arena = MiaTargetCacheArena(
        layers,
        capacity_tokens=context_capacity_tokens,
        max_batch_tokens=max_batch_tokens,
    )

    page_geometry = tuple(
        MiaPageGeometry(
            layer_id=layer_id,
            compress_ratio=ratio,
            compressed_capacity=(
                0
                if ratio == 0
                else (context_capacity_tokens + ratio - 1) // ratio
            ),
            attention_record_bytes=432,
            index_record_bytes=132 if ratio == 4 else 0,
        )
        for layer_id, ratio in enumerate(ratios)
    )
    workspace_geometry = (
        MiaWorkspaceGeometry(
            "compressor_c4_state_rings",
            (21, 72, 2, 1024),
            "float32",
            "absolute-position circular KV/score state",
        ),
        MiaWorkspaceGeometry(
            "indexer_c4_state_rings",
            (21, 72, 2, 256),
            "float32",
            "absolute-position circular KV/score state",
        ),
        MiaWorkspaceGeometry(
            "compressor_c128_state_rings",
            (20, 192, 2, 512),
            "float32",
            "absolute-position circular KV/score state",
        ),
        MiaWorkspaceGeometry(
            "indexer_top512_carry",
            (1, max_batch_tokens, MIA_INDEX_TOPK),
            "float32+int32",
            "shared immutable seeds plus bounded functional Metal outputs",
        ),
        MiaWorkspaceGeometry(
            "nvfp4_prefill_nax_threadgroup",
            (28 * 1024,),
            "uint8",
            "threadgroup local per 16-head query group",
        ),
        MiaWorkspaceGeometry(
            "mhc_fp32_partials",
            (max_batch_tokens, 32, 35),
            "float32",
            "initial/head FP32 plus M<384 post-pre split-32 output",
        ),
        MiaWorkspaceGeometry(
            "mhc_prefill_compact",
            (max_batch_tokens, 11 + 24),
            "float32",
            "M>=384 compact Gram plus BF16-MMA FP32 projection outputs",
        ),
        MiaWorkspaceGeometry(
            "exl3_route_arena",
            (max_batch_tokens * MIA_TOPK,),
            "uint32",
            "functional Metal outputs owned by the fused MoE call",
        ),
    )
    signatures = (
        MiaPrewarmSignature("target_prefill_bm64", MIA_WINDOW, "prefill"),
        MiaPrewarmSignature("mhc_post_pre_bf16_mma_bm64", 384, "prefill"),
        MiaPrewarmSignature("indexer_sparse_prefill", 1, "prefill"),
        MiaPrewarmSignature("indexer_sparse_decode", 1, "decode_verify"),
        MiaPrewarmSignature("target_verify_m6_bm8", 6, "decode_verify"),
        MiaPrewarmSignature("dspark_k5_bm8", MIA_DSPARK_BLOCK, "decode_verify"),
    )
    installed_routes = (
        "target_bounded_one_shard_loader",
        "target_mhc_carried_post_pre_bf16_mma_bm64",
        "compressor_stock432_mia132",
        "compressor_fixed_absolute_state_rings",
        "indexer_radix_top512",
        "mla_decode_direct_stock432",
        "mla_prefill_nax_mg16_tile32",
        "target_exl3_trellis_bm8_bm64",
        "wo_inverse_rope_mxfp8",
        "nonexpert_native_mxfp8",
        "dspark_k5_direct_stock432_k64_native_mxfp4",
        "target_fixed_swa_page_arena_m8224",
        "target_persistent_compressed_page_arena_384k",
        "dspark_persistent_fixed_ring_arena_128",
        "dflash2_structured_taps_fixed_linear_m6_copyspec_zero_owner",
    )
    identity_source = "|".join(
        (
            _TARGET_SMALL_FILE_PINS["model.safetensors.index.json"],
            _DRAFT_SMALL_FILE_PINS["model.safetensors.index.json"],
            str(context_capacity_tokens),
            str(max_batch_tokens),
            "stock432",
            "mia132",
            "k5-k64",
            "bounded-one-shard-loader",
            "mhc-post-pre-m384-bm64-bf16mma",
            "long-prefill-chunk1024",
            "compressor-absolute-state-rings",
            "fixed-target-window-m8224",
            "persistent-target-draft-page-arenas",
            MIA_DFLASH_COMMIT,
        )
    )
    identity = "mia-dsv4-" + hashlib.sha256(
        identity_source.encode("utf-8")
    ).hexdigest()[:16]
    return MiaDeepseekV4EnginePlan(
        context_capacity_tokens=context_capacity_tokens,
        max_batch_tokens=max_batch_tokens,
        max_sequences=MIA_MAX_SEQUENCES,
        page_geometry=page_geometry,
        workspace_geometry=workspace_geometry,
        indexer_workspace=indexer_workspace,
        mla_workspace=mla_workspace,
        target_cache_arena=target_cache_arena,
        prewarm_signatures=signatures,
        installed_routes=installed_routes,
        target_artifact=str(Path(target_root).resolve()),
        draft_artifact=str(Path(draft_root).resolve()),
        identity=identity,
    )
