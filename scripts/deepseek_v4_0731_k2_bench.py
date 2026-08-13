"""Official-wheel bracket for the isolated 0731 scheduler evaluation boundary.

Each clean source tree is one arm.  The harness derives ``lazy_joint_eval`` or
``materialize_first`` from the reviewed scheduler source, loads the unchanged
generic ``mtp=True`` runtime once, proves unmeasured prompt-cache parity, and
then runs the same AR/K2 primers and five measured repetitions.  There is no
runtime or environment arm selector.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable
from urllib.parse import urlsplit

import numpy as np


PROMPT_TEXT = "Explain why speculative decoding can preserve greedy output."
PROMPT_TOKEN_COUNT = 9
REPETITIONS = 5
EXPECTED_MLX_VERSION = "0.32.0"
EXPECTED_MLX_CORE_SHA256 = (
    "f96aede5d6eee539d4826a52690914e79794e2ad2c691935d02dca6b0c421c56"
)
EXPECTED_MLX_LIB_SHA256 = (
    "1876795e05b3434925e745fbf6e9f0c8c0446b666224c9d881609ab353e94e51"
)
EXPECTED_MLX_METALLIB_SHA256 = (
    "1518c08860738b08dc4563ddcf380a08dec4e6ad146c0d54888790e80656e9e3"
)
EXPECTED_MODEL_CONFIG_SHA256 = (
    "44735712733fcf8f299bdf1faa1d87fac88f1917efe1d3876d6d4c582f79a68f"
)
EXPECTED_MODEL_INDEX_SHA256 = (
    "f1332b2b209769c2db335954c2651652a8048e7d7dbf60296c2f2c0198715861"
)
EXPECTED_MODEL_METADATA_REVISION = "10001e0065f8394e03e968e652cbbe7cd2ca122c"
_SCHEDULER_SOURCE = "mtplx/native_block_speculation.py"
_BRACKET_SOURCE_PATHS = (
    "mtplx/models/deepseek_v4.py",
    "mtplx/deepseek_v4_dspark_generation.py",
    _SCHEDULER_SOURCE,
    "mtplx/runtime.py",
    "mtplx/generation.py",
    "mtplx/sampling.py",
    "scripts/deepseek_v4_guard_window.py",
    "scripts/deepseek_v4_0731_k2_bench.py",
)
_REQUIRED_IMPORTED_MODULES = (
    "mtplx.models.deepseek_v4",
    "mtplx.deepseek_v4_dspark_generation",
    "mtplx.native_block_speculation",
    "mtplx.runtime",
    "mtplx.generation",
    "mtplx.sampling",
)
_ARM_EVENTS = {
    "lazy_joint_eval": [
        "proposal_graph",
        "target_row_graph",
        "joint_eval",
        "draft_materialize",
    ],
    "materialize_first": [
        "proposal_graph",
        "proposal_eval",
        "draft_materialize",
        "target_row_graph",
    ],
}
_ARM_LABELS = frozenset(_ARM_EVENTS)
EXPECTED_NORMALIZED_SCHEDULER_SHA256 = (
    "10f7a52f59044ca7e7600156626b28826773886657e68201644f8b50385ba2e1"
)
EXPECTED_SCHEDULER_BOUNDARY_PATCH_SHA256 = (
    "f09d68378f940eb948a58cf4f9b24e90bfb9d40119483348b3e6f5d8b849205e"
)
_LAZY_BOUNDARY_BLOCK = """        # Build the authoritative primary M1 before forcing proposal IDs to the
        # host.  The two graphs are independent given carried hidden/current_top,
        # so they share one evaluation boundary without changing target math.
        with attention_phase("ar_decode"):
            row_logits, row_hidden = target_forward(
                mx.array([[current_top]], dtype=mx.int32),
                cache=target_cache,
                return_hidden=True,
            )
        if future is None:
            _eval(row_logits, row_hidden)
        else:
            _eval(future, row_logits, row_hidden)

        accepted_hidden = [row_hidden[:, -1:]]
        next_logits = row_logits[:, -1, :]
        future_tokens: list[int] = []
        if future is not None:
            future_tokens = [int(token) for token in np.asarray(future)[0]]
            for index, token in enumerate(future_tokens):
                drafted_by_depth[index] += 1
                if _is_stop(token, stop_token_ids):
                    future_tokens = future_tokens[: index + 1]
                    width = index + 2
                    break
            drafted_tokens += len(future_tokens)
"""
_MATERIALIZE_BOUNDARY_BLOCK = """        # Settle and materialize proposal IDs before constructing target row zero.
        future_tokens: list[int] = []
        if future is not None:
            _eval(future)
            future_tokens = [int(token) for token in np.asarray(future)[0]]
            for index, token in enumerate(future_tokens):
                drafted_by_depth[index] += 1
                if _is_stop(token, stop_token_ids):
                    future_tokens = future_tokens[: index + 1]
                    width = index + 2
                    break
            drafted_tokens += len(future_tokens)

        with attention_phase("ar_decode"):
            row_logits, row_hidden = target_forward(
                mx.array([[current_top]], dtype=mx.int32),
                cache=target_cache,
                return_hidden=True,
            )
        _eval(row_logits, row_hidden)

        accepted_hidden = [row_hidden[:, -1:]]
        next_logits = row_logits[:, -1, :]
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def attest_official_mlx(mx: Any, distribution: Any) -> dict[str, Any]:
    """Attest an installed wheel and reject editable/source import overlays."""

    package_version = str(distribution.version)
    if package_version != EXPECTED_MLX_VERSION:
        raise ValueError(
            f"expected MLX {EXPECTED_MLX_VERSION}, got {package_version or 'unknown'}"
        )
    installer = str(distribution.read_text("INSTALLER") or "").strip()
    if not installer:
        raise ValueError("MLX distribution has no INSTALLER attestation")
    distribution_root = Path(distribution.locate_file("")).resolve()
    core_path = Path(getattr(mx, "__file__", "")).resolve()
    if not core_path.is_file():
        raise ValueError(f"MLX core module is unreadable: {core_path}")
    try:
        core_path.relative_to(distribution_root)
    except ValueError as exc:
        raise ValueError(
            f"MLX core import is outside installed distribution: {core_path}"
        ) from exc
    core_sha = _sha256(core_path)
    if core_sha != EXPECTED_MLX_CORE_SHA256:
        raise ValueError(f"MLX core SHA mismatch: {core_sha}")

    libmlx_path = core_path.parent / "lib" / "libmlx.dylib"
    metallib_path = core_path.parent / "lib" / "mlx.metallib"
    if not libmlx_path.is_file() or not metallib_path.is_file():
        raise ValueError("MLX dylib/metallib wheel artifacts are missing")
    libmlx_sha = _sha256(libmlx_path)
    metallib_sha = _sha256(metallib_path)
    if libmlx_sha != EXPECTED_MLX_LIB_SHA256:
        raise ValueError(f"MLX libmlx SHA mismatch: {libmlx_sha}")
    if metallib_sha != EXPECTED_MLX_METALLIB_SHA256:
        raise ValueError(f"MLX metallib SHA mismatch: {metallib_sha}")

    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = None
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        url = str(direct_url.get("url") or "")
        is_wheel_archive = urlsplit(url).path.lower().endswith(".whl")
        if (
            direct_url.get("dir_info") is not None
            or direct_url.get("vcs_info") is not None
            or not is_wheel_archive
        ):
            raise ValueError(
                "MLX source/direct overlay is not an official installed wheel"
            )

    return {
        "version": package_version,
        "core_path": str(core_path),
        "core_sha256": core_sha,
        "libmlx": {"path": str(libmlx_path), "sha256": libmlx_sha},
        "metallib": {"path": str(metallib_path), "sha256": metallib_sha},
        "distribution_root": str(distribution_root),
        "installer": installer,
        "direct_url": direct_url,
    }


def attest_model(model_path: Path) -> dict[str, Any]:
    root = model_path.expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config_sha = _sha256(config_path)
    index_sha = _sha256(index_path)
    if config_sha != EXPECTED_MODEL_CONFIG_SHA256:
        raise ValueError(f"model config SHA mismatch: {config_sha}")
    if index_sha != EXPECTED_MODEL_INDEX_SHA256:
        raise ValueError(f"model index SHA mismatch: {index_sha}")
    metadata_root = root / ".cache" / "huggingface" / "download"
    metadata_paths = {
        "config": metadata_root / "config.json.metadata",
        "index": metadata_root / "model.safetensors.index.json.metadata",
    }
    model_metadata = {}
    for name, path in metadata_paths.items():
        try:
            revision = path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError) as exc:
            raise ValueError(f"model {name} metadata is unreadable: {exc}") from exc
        if revision != EXPECTED_MODEL_METADATA_REVISION:
            raise ValueError(f"model {name} metadata revision mismatch: {revision!r}")
        model_metadata[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "revision": revision,
        }
    return {
        "path": str(root),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "index_path": str(index_path),
        "index_sha256": index_sha,
        "metadata": model_metadata,
    }


def attest_git(repo: Path) -> dict[str, Any]:
    """Require a clean committed tree before any MLX import can occur."""

    root = repo.resolve()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status_text = git("status", "--porcelain=v1", "--untracked-files=all")
    status = status_text.splitlines() if status_text else []
    if status:
        raise RuntimeError(
            "scheduler bracket requires a clean committed worktree: "
            + "; ".join(status)
        )
    head_tree_rows = git("ls-tree", "-r", "--full-tree", "HEAD").splitlines()
    head_tree_files = {}
    for row in head_tree_rows:
        metadata_text, path = row.split("\t", 1)
        mode, object_type, object_id = metadata_text.split(" ", 2)
        head_tree_files[path] = {
            "mode": mode,
            "type": object_type,
            "object": object_id,
        }
    head_python_sha256 = {
        path: _sha256(root / path)
        for path in head_tree_files
        if path.startswith("mtplx/") and path.endswith(".py")
    }
    return {
        "repository": str(root),
        "commit": git("rev-parse", "HEAD"),
        "head_tree": git("rev-parse", "HEAD^{tree}"),
        "head_tree_files": head_tree_files,
        "head_tree_files_sha256": hashlib.sha256(
            _canonical_bytes(head_tree_files)
        ).hexdigest(),
        "head_python_sha256": head_python_sha256,
        "head_python_set_sha256": hashlib.sha256(
            _canonical_bytes(head_python_sha256)
        ).hexdigest(),
        "dirty": False,
        "status": [],
    }


def attest_sources(repo: Path) -> dict[str, Any]:
    root = repo.resolve()
    files = {relative: _sha256(root / relative) for relative in _BRACKET_SOURCE_PATHS}
    importable_mtplx_files = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted((root / "mtplx").rglob("*.py"))
    }
    return {
        "files": files,
        "source_set_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
        "importable_mtplx_files": importable_mtplx_files,
        "importable_mtplx_set_sha256": hashlib.sha256(
            _canonical_bytes(importable_mtplx_files)
        ).hexdigest(),
    }


def _scheduler_patch_digest() -> str:
    payload = (
        _LAZY_BOUNDARY_BLOCK.encode() + b"\0" + _MATERIALIZE_BOUNDARY_BLOCK.encode()
    )
    return hashlib.sha256(payload).hexdigest()


def _normalize_scheduler_source(source: str) -> tuple[str, str]:
    """Normalize only the exact reviewed lazy/materialize boundary motion."""

    lazy_count = source.count(_LAZY_BOUNDARY_BLOCK)
    materialize_count = source.count(_MATERIALIZE_BOUNDARY_BLOCK)
    if (lazy_count, materialize_count) == (1, 0):
        label = "lazy_joint_eval"
        normalized = source
    elif (lazy_count, materialize_count) == (0, 1):
        label = "materialize_first"
        normalized = source.replace(
            _MATERIALIZE_BOUNDARY_BLOCK,
            _LAZY_BOUNDARY_BLOCK,
            1,
        )
    else:
        raise ValueError("scheduler source is not an exact sanctioned bracket arm")
    normalized_sha = hashlib.sha256(normalized.encode()).hexdigest()
    if normalized_sha != EXPECTED_NORMALIZED_SCHEDULER_SHA256:
        raise ValueError(
            "scheduler source contains changes outside reviewed boundary motion"
        )
    patch_sha = _scheduler_patch_digest()
    if patch_sha != EXPECTED_SCHEDULER_BOUNDARY_PATCH_SHA256:
        raise RuntimeError("scheduler reviewed boundary patch constant is invalid")
    return label, normalized_sha


def _classify_scheduler_source(source: str) -> tuple[str, list[str]]:
    label, _normalized_sha = _normalize_scheduler_source(source)
    return label, list(_ARM_EVENTS[label])


def attest_scheduler_arm(repo: Path) -> dict[str, Any]:
    path = repo.resolve() / _SCHEDULER_SOURCE
    source_sha = _sha256(path)
    source = path.read_bytes().decode("utf-8")
    label, normalized_sha = _normalize_scheduler_source(source)
    events = list(_ARM_EVENTS[label])
    return {
        "label": label,
        "source_path": _SCHEDULER_SOURCE,
        "source_sha256": source_sha,
        "arm_id": f"{label}:{source_sha}",
        "normalized_source_sha256": normalized_sha,
        "reviewed_boundary_patch_sha256": _scheduler_patch_digest(),
        "sanctioned_event_sequence": events,
    }


def attest_imported_mtplx_modules(
    repo: Path,
    *,
    git_identity: dict[str, Any],
    source_identity: dict[str, Any],
    modules: dict[str, ModuleType] | None = None,
) -> dict[str, Any]:
    """Bind every imported MTPLX Python module to the reviewed worktree."""

    root = repo.resolve()
    observed_modules = sys.modules if modules is None else modules
    files = {}
    for name, module in sorted(observed_modules.items()):
        if name != "mtplx" and not name.startswith("mtplx."):
            continue
        path_text = getattr(module, "__file__", None)
        if not path_text:
            continue
        path = Path(path_text).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"imported reviewed module {name} is outside worktree: {path}"
            ) from exc
        relative_text = str(relative)
        actual_sha = _sha256(path)
        head_sha = git_identity.get("head_python_sha256", {}).get(relative_text)
        if actual_sha != head_sha:
            raise RuntimeError(
                f"imported reviewed module {name} does not match preflight HEAD"
            )
        source_sha = source_identity.get("importable_mtplx_files", {}).get(
            relative_text
        )
        if actual_sha != source_sha:
            raise RuntimeError(
                f"imported reviewed module {name} does not match source attestation"
            )
        files[name] = {
            "path": relative_text,
            "sha256": actual_sha,
            "head_sha256": head_sha,
            "reviewed_source_sha256": source_sha,
        }
    missing = sorted(set(_REQUIRED_IMPORTED_MODULES) - set(files))
    if missing:
        raise RuntimeError(f"reviewed MTPLX modules were not imported: {missing}")
    return {
        "files": files,
        "module_set_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
        "preflight_head_bound": True,
        "preflight_sources_bound": True,
    }


def _state_manifest(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "nbytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        payload = array.tobytes(order="C")
        return {
            "kind": "array",
            "shape": [int(dimension) for dimension in array.shape],
            "dtype": str(array.dtype),
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "items": [_state_manifest(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {
                str(key): _state_manifest(item)
                for key, item in sorted(value.items(), key=lambda row: str(row[0]))
            },
        }
    try:
        attributes = vars(value)
    except TypeError as exc:
        raise TypeError(f"unsupported state value: {type(value)!r}") from exc
    return {
        "kind": "object",
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "attributes": {
            key: _state_manifest(item) for key, item in sorted(attributes.items())
        },
    }


def _metadata_manifest(value: Any) -> Any:
    if isinstance(value, list):
        return [_metadata_manifest(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _metadata_manifest(item)
            for key, item in value.items()
            if not (value.get("kind") in {"array", "bytes"} and key == "sha256")
        }
    return value


def _array_totals(value: Any) -> tuple[int, int]:
    if isinstance(value, list):
        rows = [_array_totals(item) for item in value]
        return sum(row[0] for row in rows), sum(row[1] for row in rows)
    if isinstance(value, dict):
        if value.get("kind") == "array":
            return 1, int(value["nbytes"])
        rows = [_array_totals(item) for item in value.values()]
        return sum(row[0] for row in rows), sum(row[1] for row in rows)
    return 0, 0


def _state_receipt(value: Any) -> dict[str, Any]:
    manifest = _state_manifest(value)
    metadata_manifest = _metadata_manifest(manifest)
    array_count, array_bytes = _array_totals(manifest)
    return {
        "state_sha256": hashlib.sha256(_canonical_bytes(manifest)).hexdigest(),
        "metadata_sha256": hashlib.sha256(
            _canonical_bytes(metadata_manifest)
        ).hexdigest(),
        "array_count": array_count,
        "array_bytes": array_bytes,
    }


class _BackendCapture:
    def __init__(self, backend: Any, proposal_caches: list[Any]):
        self._backend = backend
        self._proposal_caches = proposal_caches

    def make_cache(self, rt: Any) -> Any:
        cache = self._backend.make_cache(rt)
        self._proposal_caches.append(cache)
        return cache

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def prove_prefill_state(
    runtime: Any,
    prompt_ids: list[int],
    *,
    generate_ar: Callable[..., Any],
    generate_mtpk: Callable[..., Any],
    sampler: Any,
) -> dict[str, Any]:
    """Compare state after three target rows and one complete K2 cycle."""

    backend = runtime.block_speculative_backend
    target_caches: list[Any] = []
    proposal_caches: list[Any] = []
    original_make_cache = runtime.make_cache
    absent = object()
    original_instance_make_cache = vars(runtime).get("make_cache", absent)

    def capture_target_cache() -> Any:
        cache = original_make_cache()
        target_caches.append(cache)
        return cache

    runtime.make_cache = capture_target_cache
    runtime.block_speculative_backend = _BackendCapture(backend, proposal_caches)
    try:
        ar_output = generate_ar(
            runtime,
            list(prompt_ids),
            **_generation_kwargs(4, sampler),
        )
        if len(target_caches) != 1:
            raise RuntimeError("AR state proof did not create exactly one target cache")
        ar_target_cache = target_caches[0]
        k2_output = generate_mtpk(
            runtime,
            list(prompt_ids),
            speculative_depth=2,
            **_generation_kwargs(3, sampler),
        )
        if len(target_caches) != 2 or len(proposal_caches) != 1:
            raise RuntimeError("K2 state proof did not expose exact cache ownership")
        k2_target_cache = target_caches[1]
        proposal_snapshot = backend.snapshot(proposal_caches[0])
    finally:
        if original_instance_make_cache is absent:
            del runtime.make_cache
        else:
            runtime.make_cache = original_instance_make_cache
        runtime.block_speculative_backend = backend

    if len(ar_output.tokens) != 4 or len(k2_output.tokens) != 3:
        raise RuntimeError("state proof did not emit the required AR/K2 control rows")
    if list(ar_output.tokens[:3]) != list(k2_output.tokens):
        raise RuntimeError("state proof AR/K2 target token prefix is not exact")
    k2_stats = k2_output.stats
    drafted_by_depth = [int(value) for value in k2_stats.drafted_by_depth]
    complete_k2_cycle = (
        int(k2_stats.verify_calls) >= 1
        and len(drafted_by_depth) >= 2
        and drafted_by_depth[0] >= 1
        and drafted_by_depth[1] >= 1
    )
    if not complete_k2_cycle:
        raise RuntimeError("state proof did not execute one complete K2 cycle")
    ar_target = _state_receipt(ar_target_cache)
    k2_target = _state_receipt(k2_target_cache)
    target_equal = ar_target == k2_target
    if not target_equal:
        raise RuntimeError("AR/K2 target cache state is not bit-exact after K2 cycle")
    return {
        "measured": False,
        "ar_max_tokens": 4,
        "k2_max_tokens": 3,
        "target_rows_consumed": 3,
        "semantic_boundary": "three_serial_target_rows_after_prompt",
        "complete_k2_cycle": True,
        "k2_verify_calls": int(k2_stats.verify_calls),
        "k2_drafted_by_depth": drafted_by_depth,
        "target_token_prefix": [int(token) for token in k2_output.tokens],
        "wrappers_restored_before_primers": True,
        "ar_target": ar_target,
        "k2_target": k2_target,
        "target_state_equal": True,
        "proposal_snapshot": _state_receipt(proposal_snapshot),
    }


def _generation_kwargs(max_tokens: int, sampler: Any) -> dict[str, Any]:
    return {
        "max_tokens": int(max_tokens),
        "sampler": sampler,
        "seed": 0,
        "stop_token_ids": set(),
    }


def _measurement(output: Any, mx: Any) -> dict[str, Any]:
    stats = output.stats
    decode_tok_s = float(stats.decode_tok_s)
    end_to_end_tok_s = float(stats.end_to_end_tok_s)
    if decode_tok_s <= 0.0 or end_to_end_tok_s <= 0.0:
        raise RuntimeError("measured throughput must be positive")
    return {
        "tokens": [int(token) for token in output.tokens],
        "generated_tokens": len(output.tokens),
        "decode_tok_s": decode_tok_s,
        "end_to_end_tok_s": end_to_end_tok_s,
        "prompt_eval_time_s": float(stats.prompt_eval_time_s),
        "prompt_target_prefill_time_s": float(
            getattr(stats, "prompt_target_prefill_time_s", 0.0)
        ),
        "prompt_mtp_history_time_s": float(
            getattr(stats, "prompt_mtp_history_time_s", 0.0)
        ),
        "prompt_target_prefill_tok_s": float(
            getattr(stats, "prompt_target_prefill_tok_s", 0.0)
        ),
        "accepted_drafts": int(getattr(stats, "accepted_drafts", 0)),
        "rejected_drafts": int(getattr(stats, "rejected_drafts", 0)),
        "drafted_tokens": int(getattr(stats, "drafted_tokens", 0)),
        "accepted_by_depth": [
            int(value) for value in getattr(stats, "accepted_by_depth", [])
        ],
        "drafted_by_depth": [
            int(value) for value in getattr(stats, "drafted_by_depth", [])
        ],
        "verify_calls": int(getattr(stats, "verify_calls", 0)),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "active_memory_bytes": int(mx.get_active_memory()),
    }


def _acceptance_signature(measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        key: measurement[key]
        for key in (
            "accepted_drafts",
            "rejected_drafts",
            "drafted_tokens",
            "accepted_by_depth",
            "drafted_by_depth",
            "verify_calls",
        )
    }


def run_benchmark(
    args: argparse.Namespace,
    *,
    mx: Any,
    runtime_load: Callable[..., Any],
    generate_ar: Callable[..., Any],
    generate_mtpk: Callable[..., Any],
    sampler_factory: Callable[..., Any],
    imported_modules_attestation: Callable[[], dict[str, Any]],
    post_run_git_attestation: Callable[[], dict[str, Any]],
    mlx_identity: dict[str, Any],
    model_identity: dict[str, Any],
    git_identity: dict[str, Any],
    source_identity: dict[str, Any],
    scheduler_arm: dict[str, Any],
    guard_attestation: dict[str, Any],
) -> dict[str, Any]:
    """Run one source-derived arm with one load and fixed repetitions."""

    if int(args.max_tokens) <= 0:
        raise ValueError("--max-tokens must be positive")
    if git_identity.get("dirty") is not False:
        raise ValueError("benchmark provenance must be a clean HEAD tree")
    _label, scheduler_sha = _arm_from_receipt({"scheduler_arm": scheduler_arm})
    if source_identity.get("files", {}).get(_SCHEDULER_SOURCE) != scheduler_sha:
        raise ValueError("scheduler arm hash does not match source provenance")

    runtime = runtime_load(args.model, mtp=True)
    backend = getattr(runtime, "block_speculative_backend", None)
    if getattr(backend, "backend_id", None) != "deepseek_v4_dspark_0731":
        raise ValueError("loaded runtime has no native DeepSeek-V4 DSpark backend")
    if getattr(runtime, "deepseek_v4_0731_k2_receipt", None) is not None:
        raise ValueError("scheduler bracket must keep explicit 0731 kernels stock")
    imported_modules_pre_run = imported_modules_attestation()
    if not (
        imported_modules_pre_run.get("preflight_head_bound") is True
        and imported_modules_pre_run.get("preflight_sources_bound") is True
    ):
        raise RuntimeError("imported MTPLX modules lack preflight source binding")
    prompt_ids = [int(token) for token in runtime.tokenizer.encode(PROMPT_TEXT)]
    if len(prompt_ids) != PROMPT_TOKEN_COUNT:
        raise RuntimeError(
            "fixed prompt tokenizer drift: expected "
            f"{PROMPT_TOKEN_COUNT} tokens, got {len(prompt_ids)}"
        )

    sampler = sampler_factory(temperature=0.0, top_p=1.0, top_k=0)
    kwargs = _generation_kwargs(args.max_tokens, sampler)
    state_proof = prove_prefill_state(
        runtime,
        prompt_ids,
        generate_ar=generate_ar,
        generate_mtpk=generate_mtpk,
        sampler=sampler,
    )

    generate_ar(runtime, list(prompt_ids), **kwargs)
    generate_mtpk(
        runtime,
        list(prompt_ids),
        speculative_depth=2,
        **kwargs,
    )

    samples = []
    for repetition in range(1, REPETITIONS + 1):
        mx.reset_peak_memory()
        ar_output = generate_ar(runtime, list(prompt_ids), **kwargs)
        ar_measurement = _measurement(ar_output, mx)

        mx.reset_peak_memory()
        k2_output = generate_mtpk(
            runtime,
            list(prompt_ids),
            speculative_depth=2,
            **kwargs,
        )
        k2_measurement = _measurement(k2_output, mx)
        samples.append(
            {
                "repetition": repetition,
                "ar": ar_measurement,
                "k2": k2_measurement,
                "exact_vs_ar": ar_measurement["tokens"] == k2_measurement["tokens"],
                "acceptance_signature": _acceptance_signature(k2_measurement),
            }
        )

    ar_reference = samples[0]["ar"]["tokens"]
    k2_reference = samples[0]["k2"]["tokens"]
    acceptance_reference = samples[0]["acceptance_signature"]
    ar_deterministic = all(row["ar"]["tokens"] == ar_reference for row in samples)
    k2_deterministic = all(row["k2"]["tokens"] == k2_reference for row in samples)
    acceptance_deterministic = all(
        row["acceptance_signature"] == acceptance_reference for row in samples
    )
    exact_all_samples = all(row["exact_vs_ar"] for row in samples)
    gates = {
        "state_proof_target_equal": state_proof["target_state_equal"],
        "tokens_exact_all_samples": exact_all_samples,
        "ar_deterministic": ar_deterministic,
        "k2_deterministic": k2_deterministic,
        "acceptance_signature_identical_all_samples": acceptance_deterministic,
    }
    passed = all(gates.values())
    ar_decode_median = float(median(row["ar"]["decode_tok_s"] for row in samples))
    k2_decode_median = float(median(row["k2"]["decode_tok_s"] for row in samples))
    ar_end_to_end_median = float(
        median(row["ar"]["end_to_end_tok_s"] for row in samples)
    )
    k2_end_to_end_median = float(
        median(row["k2"]["end_to_end_tok_s"] for row in samples)
    )
    imported_modules = imported_modules_attestation()
    if not (
        imported_modules.get("preflight_head_bound") is True
        and imported_modules.get("preflight_sources_bound") is True
    ):
        raise RuntimeError("post-run MTPLX imports lack preflight source binding")
    for name, identity in imported_modules_pre_run["files"].items():
        if imported_modules.get("files", {}).get(name) != identity:
            raise RuntimeError("imported MTPLX module identity changed during bracket")
    post_run_git = post_run_git_attestation()
    if post_run_git != git_identity:
        raise RuntimeError("repository provenance changed during scheduler bracket")

    return {
        "schema_version": 2,
        "kind": "deepseek_v4_0731_scheduler_boundary_benchmark",
        "scheduler_arm": scheduler_arm,
        "single_model_load": True,
        "baseline": "generic_mtp_true_stock",
        "load_kwargs": {"mtp": True},
        "prompt": {
            "text": PROMPT_TEXT,
            "token_ids": prompt_ids,
            "tokens": len(prompt_ids),
        },
        "max_tokens": int(args.max_tokens),
        "repetitions": REPETITIONS,
        "speculative_depth": 2,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 0,
            "stop_token_ids": [],
        },
        "provenance": {
            "mlx": mlx_identity,
            "model": model_identity,
            "git": git_identity,
            "git_post_run": post_run_git,
            "sources": source_identity,
            "imported_mtplx_modules_pre_run": imported_modules_pre_run,
            "imported_mtplx_modules": imported_modules,
        },
        "guard_attestation": guard_attestation,
        "state_proof": state_proof,
        "primers": {
            "ar": {"executed": True, "measured": False},
            "k2": {
                "executed": True,
                "measured": False,
                "speculative_depth": 2,
            },
        },
        "measurements": {
            "samples": samples,
            "summary": {
                "ar": {
                    "median_decode_tok_s": ar_decode_median,
                    "median_end_to_end_tok_s": ar_end_to_end_median,
                },
                "k2": {
                    "median_decode_tok_s": k2_decode_median,
                    "median_end_to_end_tok_s": k2_end_to_end_median,
                },
                "k2_over_ar_decode_ratio": k2_decode_median / ar_decode_median,
                "k2_over_ar_end_to_end_ratio": (
                    k2_end_to_end_median / ar_end_to_end_median
                ),
            },
        },
        "deterministic": {
            "ar": ar_deterministic,
            "k2": k2_deterministic,
            "acceptance_signature": acceptance_deterministic,
        },
        "exact_vs_ar": exact_all_samples and ar_deterministic and k2_deterministic,
        "gates": gates,
        "passed": passed,
    }


def _arm_from_receipt(receipt: dict[str, Any]) -> tuple[str, str]:
    arm = receipt.get("scheduler_arm") or {}
    label = arm.get("label")
    source_sha = arm.get("source_sha256")
    if (
        label not in _ARM_LABELS
        or arm.get("arm_id") != f"{label}:{source_sha}"
        or arm.get("normalized_source_sha256") != EXPECTED_NORMALIZED_SCHEDULER_SHA256
        or arm.get("reviewed_boundary_patch_sha256")
        != EXPECTED_SCHEDULER_BOUNDARY_PATCH_SHA256
        or arm.get("sanctioned_event_sequence") != _ARM_EVENTS.get(label)
    ):
        raise ValueError("receipt scheduler arm attribution is invalid")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("receipt scheduler source hash is invalid")
    return label, source_sha


def compare_receipts(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Gate a clean lazy/materialize pair without importing MLX."""

    receipts = [first, second]
    arms = {_arm_from_receipt(receipt)[0]: receipt for receipt in receipts}
    if set(arms) != _ARM_LABELS:
        raise ValueError("comparison requires one lazy and one materialize receipt")
    lazy = arms["lazy_joint_eval"]
    materialize = arms["materialize_first"]
    lazy_sha = lazy["scheduler_arm"]["source_sha256"]
    materialize_sha = materialize["scheduler_arm"]["source_sha256"]

    source_keys = set(lazy["provenance"]["sources"]["files"]) | set(
        materialize["provenance"]["sources"]["files"]
    )
    source_differences = sorted(
        key
        for key in source_keys
        if lazy["provenance"]["sources"]["files"].get(key)
        != materialize["provenance"]["sources"]["files"].get(key)
    )
    head_file_keys = set(lazy["provenance"]["git"]["head_tree_files"]) | set(
        materialize["provenance"]["git"]["head_tree_files"]
    )
    head_tree_differences = sorted(
        key
        for key in head_file_keys
        if lazy["provenance"]["git"]["head_tree_files"].get(key)
        != materialize["provenance"]["git"]["head_tree_files"].get(key)
    )
    lazy_imports = lazy["provenance"]["imported_mtplx_modules"]["files"]
    materialize_imports = materialize["provenance"]["imported_mtplx_modules"]["files"]
    imported_names = set(lazy_imports) | set(materialize_imports)
    imported_module_differences = sorted(
        name
        for name in imported_names
        if lazy_imports.get(name) != materialize_imports.get(name)
    )
    state_keys = ("ar_target", "k2_target", "proposal_snapshot")
    state_equal = {
        key: lazy["state_proof"][key] == materialize["state_proof"][key]
        for key in state_keys
    }
    common_fields = ("load_kwargs", "prompt", "max_tokens", "repetitions", "sampling")
    common_configuration = all(lazy[key] == materialize[key] for key in common_fields)
    lazy_samples = lazy["measurements"]["samples"]
    materialize_samples = materialize["measurements"]["samples"]
    cross_arm_tokens = all(
        left[lane]["tokens"] == right[lane]["tokens"]
        for left, right in zip(lazy_samples, materialize_samples, strict=True)
        for lane in ("ar", "k2")
    )
    cross_arm_acceptance = all(
        left["acceptance_signature"] == right["acceptance_signature"]
        for left, right in zip(lazy_samples, materialize_samples, strict=True)
    )
    mlx_fields = ("version", "core_sha256", "libmlx", "metallib")
    same_mlx = all(
        lazy["provenance"]["mlx"].get(key) == materialize["provenance"]["mlx"].get(key)
        for key in mlx_fields
    )
    model_fields = ("config_sha256", "index_sha256", "metadata")
    same_model = all(
        lazy["provenance"]["model"].get(key)
        == materialize["provenance"]["model"].get(key)
        for key in model_fields
    )
    gates = {
        "both_arms_passed": bool(lazy.get("passed") and materialize.get("passed")),
        "source_hashes_distinct": lazy_sha != materialize_sha,
        "head_trees_distinct": (
            lazy["provenance"]["git"]["head_tree"]
            != materialize["provenance"]["git"]["head_tree"]
        ),
        "only_scheduler_source_differs": source_differences == [_SCHEDULER_SOURCE],
        "only_scheduler_head_blob_differs": head_tree_differences
        == [_SCHEDULER_SOURCE],
        "only_scheduler_import_differs": imported_module_differences
        == ["mtplx.native_block_speculation"],
        "normalized_scheduler_source_identical": (
            lazy["scheduler_arm"]["normalized_source_sha256"]
            == materialize["scheduler_arm"]["normalized_source_sha256"]
            == EXPECTED_NORMALIZED_SCHEDULER_SHA256
        ),
        "reviewed_boundary_patch_identical": (
            lazy["scheduler_arm"]["reviewed_boundary_patch_sha256"]
            == materialize["scheduler_arm"]["reviewed_boundary_patch_sha256"]
            == EXPECTED_SCHEDULER_BOUNDARY_PATCH_SHA256
        ),
        "preflight_postrun_git_identical_within_arms": all(
            receipt["provenance"]["git"] == receipt["provenance"]["git_post_run"]
            for receipt in (lazy, materialize)
        ),
        "common_configuration": common_configuration,
        "official_mlx_identical": same_mlx,
        "model_identity_identical": same_model,
        "ar_target_state_identical": state_equal["ar_target"],
        "k2_target_state_identical": state_equal["k2_target"],
        "proposal_snapshot_identical": state_equal["proposal_snapshot"],
        "tokens_identical_cross_arm": cross_arm_tokens,
        "acceptance_signature_identical_cross_arm": cross_arm_acceptance,
    }
    return {
        "schema_version": 1,
        "kind": "deepseek_v4_0731_scheduler_boundary_comparison",
        "arms": {
            "lazy_joint_eval": lazy["scheduler_arm"],
            "materialize_first": materialize["scheduler_arm"],
        },
        "source_differences": source_differences,
        "head_tree_differences": head_tree_differences,
        "imported_module_differences": imported_module_differences,
        "state_digests_equal": state_equal,
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_receipt(receipt: dict[str, Any], output_path: Path) -> int:
    path = output_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if receipt.get("passed", receipt.get("exact_vs_ar", False)) else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--model", type=Path)
    action.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("LAZY_RECEIPT", "MATERIALIZE_RECEIPT"),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.compare is not None:
        first, second = (
            json.loads(path.expanduser().read_text(encoding="utf-8"))
            for path in args.compare
        )
        return write_receipt(compare_receipts(first, second), args.out)

    repo = Path(__file__).resolve().parents[1]
    # These source/provenance checks intentionally precede the guard bridge and
    # MLX imports.  A dirty arm never initializes Metal or loads model weights.
    git_identity = attest_git(repo)
    source_identity = attest_sources(repo)
    scheduler_arm = attest_scheduler_arm(repo)

    from deepseek_v4_guard_window import (
        WINDOW_PATH_ENV,
        WINDOW_SHA256_ENV,
        issue_guard_window,
        load_verified_guard_window,
    )

    guard_path, guard_digest = issue_guard_window()
    try:
        guard_attestation = load_verified_guard_window(
            environment={
                WINDOW_PATH_ENV: str(guard_path),
                WINDOW_SHA256_ENV: guard_digest,
            }
        )
        import mlx.core as mx

        mlx_identity = attest_official_mlx(mx, metadata.distribution("mlx"))
        model_identity = attest_model(args.model)

        from mtplx import deepseek_v4_dspark_generation as _adapter_module  # noqa: F401
        from mtplx import generation as generation_module
        from mtplx import native_block_speculation as _scheduler_module  # noqa: F401
        from mtplx import runtime as runtime_module
        from mtplx import sampling as sampling_module
        from mtplx.models import deepseek_v4 as _model_module  # noqa: F401

        receipt = run_benchmark(
            args,
            mx=mx,
            runtime_load=runtime_module.load,
            generate_ar=generation_module.generate_ar,
            generate_mtpk=generation_module.generate_mtpk,
            sampler_factory=sampling_module.SamplerConfig,
            imported_modules_attestation=lambda: attest_imported_mtplx_modules(
                repo,
                git_identity=git_identity,
                source_identity=source_identity,
            ),
            post_run_git_attestation=lambda: attest_git(repo),
            mlx_identity=mlx_identity,
            model_identity=model_identity,
            git_identity=git_identity,
            source_identity=source_identity,
            scheduler_arm=scheduler_arm,
            guard_attestation=guard_attestation,
        )
        return write_receipt(receipt, args.out)
    finally:
        try:
            guard_path.unlink()
        except FileNotFoundError:
            pass
        try:
            guard_path.parent.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
