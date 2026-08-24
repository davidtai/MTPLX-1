"""First-class Qwen3.8 DFlash2 runtime using the measured dflash-mlx stack."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from mtplx.dflash2_bundle import (
    DFLASH2_ARCH_ID,
    DFLASH2_BACKEND,
    DFLASH2_DEFAULT_BLOCK_SIZE,
    DFLASH2_TARGET_LAYER_IDS,
    dflash2_bundle_inspection,
    load_dflash2_metadata,
    resolve_dflash2_bundle_paths,
)

BACKEND_NAME = DFLASH2_BACKEND
ARCH_ID = DFLASH2_ARCH_ID
# The checkpoint has five feature-extraction layers but proposes one physical
# block of up to eight tokens.  Those are different axes.
DEFAULT_DFLASH2_BLOCK_SIZE = DFLASH2_DEFAULT_BLOCK_SIZE
DEFAULT_DRAFT_BLOCK_SIZE = DEFAULT_DFLASH2_BLOCK_SIZE
DFlash2Quantization = Literal["4bit", "8bit", "unquantized"]
_DFLASH2_QUANTIZATIONS = {"4bit", "8bit", "unquantized"}
_DFLASH2_GENERATION_LOCK = RLock()


class DFlash2Unsupported(RuntimeError):
    """Raised when an MTPLX feature is not supported by DFlash2."""


def _install_checkpoint_capabilities(draft_model: Any, *, block_size: int) -> None:
    """Expose the checkpoint's physical width to the DFlash scheduler."""

    try:
        checkpoint_block_size = draft_model.block_size
        target_layer_ids = tuple(draft_model.target_layer_ids)
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "Qwen3.8 DFlash2 draft must expose checkpoint geometry"
        ) from error
    if checkpoint_block_size != block_size:
        raise ValueError(
            "Qwen3.8 DFlash2 checkpoint block size must match the bundle manifest: "
            f"{checkpoint_block_size} != {block_size}"
        )
    if target_layer_ids != DFLASH2_TARGET_LAYER_IDS:
        raise ValueError(
            "Qwen3.8 DFlash2 checkpoint target layer IDs must be "
            f"{DFLASH2_TARGET_LAYER_IDS}, got {target_layer_ids}"
        )
    try:
        draft_model.capabilities = replace(
            draft_model.capabilities,
            default_block_tokens=block_size,
            max_block_tokens=block_size,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Qwen3.8 DFlash2 draft must expose replaceable runtime capabilities"
        ) from error


def _normalise_quantization(value: Any) -> DFlash2Quantization:
    if value is None:
        return "unquantized"
    if isinstance(value, int):
        value = f"{value}bit"
    text = str(value).strip().lower().replace("-", "").replace(" ", "")
    aliases = {
        "4": "4bit",
        "q4": "4bit",
        "4bit": "4bit",
        "8": "8bit",
        "q8": "8bit",
        "8bit": "8bit",
        "bf16": "unquantized",
        "fp16": "unquantized",
        "fp32": "unquantized",
        "none": "unquantized",
        "unquantized": "unquantized",
    }
    result = aliases.get(text)
    if result is None or result not in _DFLASH2_QUANTIZATIONS:
        raise ValueError(
            "DFlash2 draft_quantization must be one of '4bit', '8bit', or 'unquantized'"
        )
    return result  # type: ignore[return-value]


def _manifest_settings(metadata: dict[str, Any]) -> tuple[DFlash2Quantization, int]:
    draft = metadata.get("draft")
    draft = draft if isinstance(draft, dict) else {}
    quantization = metadata.get(
        "draft_quantization",
        metadata.get("draft_precision", draft.get("quantization", draft.get("precision"))),
    )
    raw_block_size = metadata.get(
        "draft_block_size",
        metadata.get("block_size", draft.get("draft_block_size", draft.get("block_size"))),
    )
    try:
        block_size = DEFAULT_DFLASH2_BLOCK_SIZE if raw_block_size is None else int(raw_block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("DFlash2 block_size must be an integer") from exc
    if not 1 <= block_size <= 8:
        raise ValueError("DFlash2 block_size must be in the physical range 1..8")
    return _normalise_quantization(quantization), block_size


resolve_dflash2_paths = resolve_dflash2_bundle_paths
load_dflash2_manifest = load_dflash2_metadata


@dataclass(frozen=True)
class DFlash2RuntimeConfig:
    target_model_path: Path
    draft_model_path: Path
    draft_block_size: int = DEFAULT_DFLASH2_BLOCK_SIZE
    draft_quantization: DFlash2Quantization = "unquantized"
    prefill_step_size: int = 2048
    backend: str = "dflash2"

    @classmethod
    def from_paths(
        cls,
        *,
        target_model_path: str | Path,
        draft_model_path: str | Path,
        draft_block_size: int = DEFAULT_DFLASH2_BLOCK_SIZE,
        draft_quantization: Any = "unquantized",
        quantization: Any | None = None,
        prefill_step_size: int = 2048,
    ) -> DFlash2RuntimeConfig:
        return cls(
            target_model_path=Path(target_model_path),
            draft_model_path=Path(draft_model_path),
            draft_block_size=int(draft_block_size),
            draft_quantization=_normalise_quantization(
                draft_quantization if quantization is None else quantization
            ),
            prefill_step_size=int(prefill_step_size),
        )

    @property
    def quantization(self) -> DFlash2Quantization:
        return self.draft_quantization

    @property
    def backend_id(self) -> str:
        return self.backend

    @property
    def target_path(self) -> Path:
        return self.target_model_path

    @property
    def draft_path(self) -> Path:
        return self.draft_model_path

    def validate_static(self) -> None:
        if self.backend != "dflash2":
            raise ValueError("DFlash2 backend must be 'dflash2'")
        if not 1 <= self.draft_block_size <= 8:
            raise ValueError("DFlash2 draft_block_size must be in the physical range 1..8")
        _normalise_quantization(self.draft_quantization)
        if self.prefill_step_size < 1:
            raise ValueError("DFlash2 prefill_step_size must be positive")
        if not self.target_model_path.is_dir():
            raise FileNotFoundError(f"DFlash2 target path does not exist: {self.target_model_path}")
        if not self.draft_model_path.is_dir():
            raise FileNotFoundError(f"DFlash2 draft path does not exist: {self.draft_model_path}")


@dataclass
class DFlash2Telemetry:
    chunks: int = 0
    generated_tokens: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_bytes: int = 0
    finish_reason: str | None = None
    adaptive_metrics: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "dflash2",
            "chunks": self.chunks,
            "generated_tokens": self.generated_tokens,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tps": self.prompt_tps,
            "generation_tps": self.generation_tps,
            "peak_memory_bytes": self.peak_memory_bytes,
            "finish_reason": self.finish_reason,
            "adaptive_metrics": dict(self.adaptive_metrics),
            "events": list(self.events),
        }


class DFlash2Runtime:
    """MTPLX wrapper around the target-only Optimized-Speed and DFlash2 draft."""

    def __init__(
        self,
        *,
        target_model: Any,
        tokenizer: Any,
        draft_model: Any,
        target_runtime: Any,
        target_ops: Any,
        draft_backend: Any,
        runtime_context: Any,
        config: DFlash2RuntimeConfig,
    ) -> None:
        self.model = target_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.draft = draft_model
        self.target_runtime = target_runtime
        self.target_ops = target_ops
        self.draft_backend = draft_backend
        self.runtime_context = runtime_context
        self.config = config
        self.model_path = config.target_model_path
        self.path = config.target_model_path
        self.bundle_path: Path | None = None
        self.backend_id = "dflash2"
        self.mtp_enabled = True
        self.dflash2_external_draft = True
        self.telemetry = DFlash2Telemetry()
        self.diagnostic_counters: dict[str, int] = {}
        self.qwen38_feature_receipt: dict[str, Any] = {}

    def _count(self, key: str, amount: int = 1) -> None:
        self.diagnostic_counters[key] = self.diagnostic_counters.get(key, 0) + int(amount)


def load_dflash2_bundle(
    bundle_root: str | Path,
    *,
    draft_block_size: int | None = None,
    draft_quantization: Any | None = None,
) -> DFlash2Runtime:
    """Load the measured target-only Optimized-Speed plus dflash-mlx stack."""

    # MLX reads these once when its Metal device is constructed.  The package
    # bootstrap handles CLI/server argv before importing MLX; direct Python
    # callers must establish the same process contract before importing MTPLX.
    required_environment = {
        "MLX_MAX_MB_PER_BUFFER": "512",
        "MLX_MAX_OPS_PER_BUFFER": "50",
    }
    if any(os.environ.get(name) != value for name, value in required_environment.items()):
        raise RuntimeError(
            "DFlash2 row-53 MLX_MAX_MB_PER_BUFFER=512 and "
            "MLX_MAX_OPS_PER_BUFFER=50 must be set before importing MLX"
        )
    from mtplx.profiles import apply_profile_env

    apply_profile_env("turbo")
    os.environ["MTPLX_QWEN38_DISABLE_SOURCE_AUTO"] = "1"
    root = Path(bundle_root).expanduser()
    resolved = resolve_dflash2_bundle_paths(root)
    if resolved is None:
        raise ValueError(f"not a DFlash2 bundle: {bundle_root}")
    inspection = dflash2_bundle_inspection(
        model_ref=str(root),
        bundle_root=root,
        paths=resolved,
    )
    compatibility = inspection.get("compatibility", {})
    if not isinstance(compatibility, dict) or not compatibility.get("can_run"):
        message = compatibility.get("message") if isinstance(compatibility, dict) else None
        raise ValueError(message or f"DFlash2 bundle rejected: {root}")
    metadata = resolved["metadata"]
    manifest_quantization, manifest_block_size = _manifest_settings(metadata)
    config = DFlash2RuntimeConfig.from_paths(
        target_model_path=resolved["target_model"],
        draft_model_path=resolved["draft_model"],
        draft_block_size=(manifest_block_size if draft_block_size is None else int(draft_block_size)),
        draft_quantization=(
            manifest_quantization if draft_quantization is None else draft_quantization
        ),
    )
    config.validate_static()
    try:
        from dflash_mlx.draft_backend import EagerDraftBackend
        from dflash_mlx.engine.target_ops import bind_draft_to_target
        from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
        from dflash_mlx.runtime.context import build_offline_runtime_context
        from dflash_mlx.runtime.loading import load_draft_bundle
    except ImportError as exc:
        raise RuntimeError(
            "DFlash2 requires the optional 'dflash2' dependency (dflash-mlx)"
        ) from exc
    from mtplx.runtime import load as load_mtplx_runtime

    # This is the benchmarked ownership boundary: native MTP is never built.
    target_runtime = load_mtplx_runtime(config.target_model_path, mtp=False)
    target_model = target_runtime.model
    target_ops = QwenGdnTargetOps()
    if not target_ops.supports_model(target_model):
        raise ValueError("DFlash2 target must be the Qwen3.8 hybrid-GDN model")
    draft_quant = {
        "4bit": "w4:gs64",
        "8bit": "w8:gs64",
        "unquantized": None,
    }[config.draft_quantization]
    draft_model, draft_meta = load_draft_bundle(
        config.draft_model_path,
        lazy=True,
        draft_quant=draft_quant,
    )
    _install_checkpoint_capabilities(
        draft_model,
        block_size=config.draft_block_size,
    )
    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)
    draft_backend = EagerDraftBackend()
    runtime_context = build_offline_runtime_context(
        quantize_kv_cache=False,
        verify_mode="dflash",
        copyspec_mode="off",
        prefill_step_size=config.prefill_step_size,
        verify_len_cap=config.draft_block_size,
    )
    runtime = DFlash2Runtime(
        target_model=target_model,
        tokenizer=target_runtime.tokenizer,
        draft_model=draft_model,
        target_runtime=target_runtime,
        target_ops=target_ops,
        draft_backend=draft_backend,
        runtime_context=runtime_context,
        config=config,
    )
    runtime.bundle_path = root
    runtime.dflash2_metadata = metadata
    runtime.dflash2_draft_metadata = dict(draft_meta)
    runtime.qwen38_feature_receipt = _install_measured_qwen38_dflash_stack(runtime)
    return runtime


load_dflash2 = load_dflash2_bundle


def _install_measured_qwen38_dflash_stack(runtime: DFlash2Runtime) -> dict[str, Any]:
    """Install only DFlash mechanisms retained by the chronological 16K gates."""

    from mtplx.gdn_capture import configure_qwen38_dflash_row48_boundary
    from mtplx.qwen38_challenge import configure_qwen38_row50_wired_residency
    from mtplx.qwen38_challenge_kernels import (
        configure_qwen38_dflash_row24_eval_ladder,
        configure_qwen38_dflash_gqa_widths,
        configure_qwen38_dflash_m8_nax_island,
        configure_qwen38_row21_qk_rms_rope,
        configure_qwen38_row24_qk_length_limit,
    )
    from mtplx.qwen38_dflash_adaptive import (
        configure_qwen38_dflash_adaptive_policy,
    )
    from mtplx.nax_verify import (
        configure_qwen38_m6_barrier_free_kp1,
    )

    model = runtime.target_model
    receipt = {
        "r21_qk_rms_rope": configure_qwen38_row21_qk_rms_rope(model, active=True),
        "r24_qk_length_limit": configure_qwen38_row24_qk_length_limit(
            model, active=True, max_length=32
        ),
        "r24_eval_ladder": configure_qwen38_dflash_row24_eval_ladder(
            model, active=True, prefill_stride=3
        ),
        "r26_prefill_ladder_3": {"active": 1},
        "r48_boundary_fused": configure_qwen38_dflash_row48_boundary(
            model, active=True
        ),
        "dflash_gqa_widths": configure_qwen38_dflash_gqa_widths(
            model, active=True, widths=(6, 7, 8)
        ),
        "dflash_m8_nax_island": configure_qwen38_dflash_m8_nax_island(
            model,
            active=True,
            include_m8_output=False,
            include_m7_output=True,
            include_m7_linear_z=True,
            include_m8_kv=True,
            include_m8_qkv=True,
            include_m8_mlp=True,
            include_m5_exact=True,
            include_m6_kp1=True,
        ),
        "dflash_m6_barrier_free_kp1": configure_qwen38_m6_barrier_free_kp1(
            active=True
        ),
        "adaptive_policy": configure_qwen38_dflash_adaptive_policy(
            model, active=True, proposal_rows=(11, 15)
        ),
        "r50_wired_residency": configure_qwen38_row50_wired_residency(
            runtime.target_runtime, active=True
        ),
        "r53_command_buffers": {
            "active": True,
            "installed": True,
            "max_mb_per_buffer": 512,
            "max_ops_per_buffer": 50,
            "process_latched": True,
        },
        "native_mtp_release": {
            "native_mtp_released": True,
            "native_mtp_loaded": False,
        },
    }
    if not bool(receipt["r50_wired_residency"].get("installed")):
        raise RuntimeError("retained DFlash row 50 residency policy did not install")
    return receipt


def _unsupported_options(
    *,
    constraint: Any = None,
    session_bank: Any = None,
    session_id: str | None = None,
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
    session_restore_mode: str = "clone",
    commit_prompt_state_to_bank: bool = False,
) -> None:
    if constraint is not None:
        raise DFlash2Unsupported("constrained decoding is not supported on the dflash2 backend")
    if (
        session_bank is not None
        or capture_final_state
        or commit_prompt_state_to_bank
    ):
        raise DFlash2Unsupported("sessions and final-state capture are not supported on the dflash2 backend")


def _decode_tokens(tokenizer: Any, tokens: list[int], fallback: str) -> str:
    try:
        return str(tokenizer.decode(tokens))
    except (AttributeError, TypeError, ValueError):
        return fallback


def _default_stop_token_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values is None:
        value = getattr(tokenizer, "eos_token_id", None)
        values = [] if value is None else [value]
    try:
        return {int(value) for value in values}
    except (TypeError, ValueError):
        return set()


def _drafted_token_count(summary: Any, cycle_events: list[dict[str, Any]]) -> int:
    if cycle_events:
        return sum(max(0, int(event["block_len"]) - 1) for event in cycle_events)
    adaptive_metrics = summary.adaptive_metrics or {}
    cycles_by_block = adaptive_metrics.get("cycles_by_block")
    if isinstance(cycles_by_block, dict):
        try:
            return sum(
                max(0, int(block_size) - 1) * max(0, int(cycles))
                for block_size, cycles in cycles_by_block.items()
            )
        except (TypeError, ValueError):
            pass
    if summary.block_tokens is not None:
        return max(0, int(summary.block_tokens) - 1) * max(
            0, int(summary.cycles_completed)
        )
    return 0


@contextmanager
def _dflash_target_sampling(sampler: Any, *, seed: int):
    """Use MTPLX's exact target sampler at dflash-mlx's posterior seam."""

    import mlx.core as mx

    from dflash_mlx.engine import spec_epoch
    from mtplx.fast_sampling import sample_token_ids_from_mlx_logits

    original = spec_epoch.greedy_tokens_with_mask

    def sample_target_rows(logits: Any, suppress_token_mask: Any = None):
        if suppress_token_mask is not None:
            raise DFlash2Unsupported("DFlash2 token suppression is not supported")
        sampled = sample_token_ids_from_mlx_logits(logits, sampler)
        if sampled is None:
            raise RuntimeError("DFlash2 target sampler could not stay on device")
        return sampled.astype(mx.uint32)

    with _DFLASH2_GENERATION_LOCK:
        mx.random.seed(int(seed))
        spec_epoch.greedy_tokens_with_mask = sample_target_rows
        try:
            yield
        finally:
            spec_epoch.greedy_tokens_with_mask = original


def _generate_stream(
    runtime: DFlash2Runtime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: Any,
    seed: int,
    stop_token_ids: set[int] | None,
    token_callback: Callable[[list[int]], None] | None,
    abort_check: Callable[[], bool] | None,
) -> Any:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    telemetry = DFlash2Telemetry(prompt_tokens=len(prompt_ids))
    effective_stop_ids = (
        _default_stop_token_ids(runtime.tokenizer)
        if stop_token_ids is None
        else {int(value) for value in stop_token_ids}
    )
    try:
        from dflash_mlx.engine.events import CycleCompleteEvent, SummaryEvent, TokenEvent
        from dflash_mlx.runtime import stream_dflash_generate
    except ImportError as exc:
        raise RuntimeError("DFlash2 requires the optional dflash-mlx runtime") from exc
    summary = None
    cycle_events: list[dict[str, Any]] = []
    with _dflash_target_sampling(sampler, seed=seed):
        stream = stream_dflash_generate(
            target_model=runtime.target_model,
            target_ops=runtime.target_ops,
            tokenizer=runtime.tokenizer,
            draft_model=runtime.draft_model,
            draft_backend=runtime.draft_backend,
            prompt_tokens_override=list(prompt_ids),
            prompt="",
            use_chat_template=False,
            max_new_tokens=max_tokens,
            block_tokens=runtime.config.draft_block_size,
            stop_token_ids=sorted(effective_stop_ids),
            runtime_context=runtime.runtime_context,
            should_cancel=abort_check,
        )
        for event in stream:
            if isinstance(event, TokenEvent):
                token_id = int(event.token_id)
                if token_callback is not None and token_id not in effective_stop_ids:
                    token_callback([token_id])
            elif isinstance(event, CycleCompleteEvent):
                cycle_events.append(event.to_payload())
            elif isinstance(event, SummaryEvent):
                if summary is not None:
                    raise RuntimeError("DFlash2 produced multiple summary events")
                summary = event
    if summary is None:
        raise RuntimeError("DFlash2 ended without a summary event")
    if summary.fallback_ar:
        raise RuntimeError(
            "DFlash2 unexpectedly fell back to AR: "
            f"{summary.fallback_reason or 'unspecified reason'}"
        )
    generated = [int(token) for token in summary.generated_token_ids]
    elapsed = max(float(summary.elapsed_us) / 1_000_000.0, 1e-12)
    prefill_s = max(
        float(summary.phase_timings_us.get("prefill", 0.0)) / 1_000_000.0,
        0.0,
    )
    prompt_tps = len(prompt_ids) / prefill_s if prefill_s > 0 else 0.0
    decode_elapsed = max(elapsed - prefill_s, 1e-12)
    drafted = _drafted_token_count(summary, cycle_events)
    accepted = int(summary.accepted_from_draft)
    finish_reason = "length" if len(generated) >= max_tokens else "stop"
    # dflash-mlx reports decimal GB (mx.get_peak_memory() / 1e9).
    peak_memory_bytes = int(float(summary.peak_memory_gb or 0.0) * 1_000_000_000)
    telemetry.generated_tokens = len(generated)
    telemetry.drafted_tokens = drafted
    telemetry.accepted_tokens = accepted
    telemetry.chunks = int(summary.cycles_completed)
    telemetry.prompt_tps = prompt_tps
    telemetry.generation_tps = len(generated) / decode_elapsed
    telemetry.peak_memory_bytes = peak_memory_bytes
    telemetry.finish_reason = finish_reason
    telemetry.adaptive_metrics = dict(summary.adaptive_metrics or {})
    telemetry.events = cycle_events
    runtime.telemetry = telemetry
    runtime._count("dflash2_chunks", telemetry.chunks)
    runtime._count("dflash2_generated_tokens", telemetry.generated_tokens)

    from mtplx.generation import GenerationOutput, GenerationStats

    stats = GenerationStats(
        mode="mtpk",
        generated_tokens=len(generated),
        elapsed_s=elapsed,
        tok_s=len(generated) / decode_elapsed,
        decode_elapsed_s=decode_elapsed,
        decode_tok_s=len(generated) / decode_elapsed,
        end_to_end_tok_s=len(generated) / elapsed,
        runtime_mtp_enabled=True,
        prompt_eval_time_s=prefill_s,
        prompt_tps=prompt_tps,
        accepted_drafts=accepted,
        drafted_tokens=drafted,
        draft_time_s=decode_elapsed,
        verify_calls=telemetry.chunks,
        peak_memory_bytes=telemetry.peak_memory_bytes,
        draft_core={"backend": "dflash2", **telemetry.to_dict()},
        events=list(telemetry.events),
    )
    return GenerationOutput(
        tokens=generated,
        text=_decode_tokens(
            runtime.tokenizer,
            generated[:-1] if generated and generated[-1] in effective_stop_ids else generated,
            "",
        ),
        stats=stats,
        finish_reason=telemetry.finish_reason,
    )


def generate_dflash2(
    runtime: DFlash2Runtime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: Any,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
    abort_check: Callable[[], bool] | None = None,
    constraint: Any = None,
    session_bank: Any = None,
    session_id: str | None = None,
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
    session_restore_mode: str = "clone",
    commit_prompt_state_to_bank: bool = False,
    speculative_depth: int | None = None,
) -> Any:
    _unsupported_options(
        constraint=constraint,
        session_bank=session_bank,
        session_id=session_id,
        session_template_hash=session_template_hash,
        session_draft_head_identity=session_draft_head_identity,
        session_policy_fingerprint=session_policy_fingerprint,
        capture_final_state=capture_final_state,
        session_restore_mode=session_restore_mode,
        commit_prompt_state_to_bank=commit_prompt_state_to_bank,
    )
    return _generate_stream(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        seed=seed,
        stop_token_ids=stop_token_ids,
        token_callback=token_callback,
        abort_check=abort_check,
    )


def generate_dflash2_ar(
    runtime: DFlash2Runtime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: Any,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
    abort_check: Callable[[], bool] | None = None,
    constraint: Any = None,
    session_bank: Any = None,
    session_id: str | None = None,
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
    session_restore_mode: str = "clone",
    commit_prompt_state_to_bank: bool = False,
    **_: Any,
) -> Any:
    _unsupported_options(
        constraint=constraint,
        session_bank=session_bank,
        session_id=session_id,
        session_template_hash=session_template_hash,
        session_draft_head_identity=session_draft_head_identity,
        session_policy_fingerprint=session_policy_fingerprint,
        capture_final_state=capture_final_state,
        session_restore_mode=session_restore_mode,
        commit_prompt_state_to_bank=commit_prompt_state_to_bank,
    )
    from mtplx.generation import generate_ar

    return generate_ar(
        runtime.target_runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        seed=seed,
        stop_token_ids=stop_token_ids,
        token_callback=token_callback,
        abort_check=abort_check,
    )
