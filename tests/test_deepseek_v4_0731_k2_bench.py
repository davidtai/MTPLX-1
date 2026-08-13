"""CPU-only contracts for the source-isolated 0731 scheduler bracket."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_0731_k2_bench.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("deepseek_v4_0731_k2_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeMX:
    def __init__(self):
        self.reset_calls = 0
        self.memory_reads = 0

    def reset_peak_memory(self):
        self.reset_calls += 1

    def get_peak_memory(self):
        self.memory_reads += 1
        return 1_000 + self.memory_reads

    def get_active_memory(self):
        return 900 + self.memory_reads


def _stats(*, speculative: bool, signature_variant: int = 0):
    return SimpleNamespace(
        decode_tok_s=41.25 if speculative else 31.5,
        end_to_end_tok_s=30.5 if speculative else 25.0,
        prompt_eval_time_s=0.8,
        prompt_target_prefill_time_s=0.5,
        prompt_mtp_history_time_s=0.3 if speculative else 0.0,
        prompt_target_prefill_tok_s=18.0,
        accepted_drafts=(4 + signature_variant) if speculative else 0,
        rejected_drafts=2 if speculative else 0,
        drafted_tokens=(6 + signature_variant) if speculative else 0,
        accepted_by_depth=([3 + signature_variant, 1] if speculative else []),
        drafted_by_depth=([4 + signature_variant, 2] if speculative else []),
        verify_calls=3 if speculative else 0,
    )


class _FakeBackend:
    backend_id = "deepseek_v4_dspark_0731"

    def make_cache(self, _runtime):
        return SimpleNamespace(ring=b"proposal-ring", prefill_length=9)

    def snapshot(self, cache):
        return ((cache.ring, cache.prefill_length),)


class _FakeRuntime:
    def __init__(self):
        self.model = object()
        self.tokenizer = SimpleNamespace(
            encode=lambda text: list(range(11, 20)) if text else []
        )
        self.block_speculative_backend = _FakeBackend()
        self.deepseek_v4_0731_k2_receipt = None
        self.target_cache_calls = 0

    def make_cache(self):
        self.target_cache_calls += 1
        return SimpleNamespace(
            offset=9,
            state=(b"target-state",),
            metadata_version="fake-v1",
        )


def _scheduler_arm(
    label: str = "lazy_joint_eval",
    source_sha: str = "a" * 64,
    *,
    normalized_sha: str = "10f7a52f59044ca7e7600156626b28826773886657e68201644f8b50385ba2e1",
    patch_sha: str = "f09d68378f940eb948a58cf4f9b24e90bfb9d40119483348b3e6f5d8b849205e",
):
    events = (
        ["proposal_graph", "target_row_graph", "joint_eval", "draft_materialize"]
        if label == "lazy_joint_eval"
        else [
            "proposal_graph",
            "proposal_eval",
            "draft_materialize",
            "target_row_graph",
        ]
    )
    return {
        "label": label,
        "source_path": "mtplx/native_block_speculation.py",
        "source_sha256": source_sha,
        "arm_id": f"{label}:{source_sha}",
        "normalized_source_sha256": normalized_sha,
        "reviewed_boundary_patch_sha256": patch_sha,
        "sanctioned_event_sequence": events,
    }


def _identities(harness, scheduler_sha: str = "a" * 64):
    sources = {path: "e" * 64 for path in harness._BRACKET_SOURCE_PATHS}
    sources[harness._SCHEDULER_SOURCE] = scheduler_sha
    importable = {
        name.replace(".", "/") + ".py": (
            scheduler_sha if name == "mtplx.native_block_speculation" else "e" * 64
        )
        for name in harness._REQUIRED_IMPORTED_MODULES
    }
    return {
        "mlx_identity": {
            "version": "0.32.0",
            "core_sha256": "1" * 64,
            "libmlx": {"sha256": "2" * 64},
            "metallib": {"sha256": "3" * 64},
        },
        "model_identity": {
            "config_sha256": "4" * 64,
            "index_sha256": "5" * 64,
            "metadata": {"revision": "6" * 40},
        },
        "git_identity": {
            "commit": "7" * 40,
            "head_tree": "8" * 40,
            "head_tree_files": {
                path: {"mode": "100644", "object": digest}
                for path, digest in sources.items()
            },
            "head_python_sha256": importable,
            "dirty": False,
            "status": [],
        },
        "source_identity": {
            "source_set_sha256": "9" * 64,
            "files": sources,
            "importable_mtplx_files": importable,
        },
        "guard_attestation": {
            "window_id": "b" * 64,
            "attestation": {"lock_device": 1, "lock_inode": 2},
            "lock_identity": {"device": 1, "inode": 2},
        },
    }


def _run_fake_benchmark(
    harness,
    *,
    ar_tokens=None,
    k2_tokens=None,
    k2_signature_variant=None,
    post_run_git_mutator=None,
):
    calls = []
    loads = []
    mx = _FakeMX()
    runtime = _FakeRuntime()
    original_backend = runtime.block_speculative_backend
    ar_tokens = ar_tokens or (lambda _call: [101, 102, 103])
    k2_tokens = k2_tokens or (lambda _call: [101, 102, 103])
    k2_signature_variant = k2_signature_variant or (lambda _call: 0)
    ar_calls = 0
    k2_calls = 0

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return runtime

    def generate_ar(active_runtime, prompt_ids, **kwargs):
        nonlocal ar_calls
        ar_calls += 1
        cache = active_runtime.make_cache()
        calls.append(
            (
                "ar",
                prompt_ids,
                tuple(prompt_ids),
                kwargs,
                active_runtime.block_speculative_backend is original_backend,
                "make_cache" in vars(active_runtime),
            )
        )
        tokens = (
            [101, 102, 103, 104] if kwargs["max_tokens"] == 4 else ar_tokens(ar_calls)
        )
        assert cache.offset == 9
        return SimpleNamespace(tokens=tokens, stats=_stats(speculative=False))

    def generate_mtpk(active_runtime, prompt_ids, **kwargs):
        nonlocal k2_calls
        k2_calls += 1
        target_cache = active_runtime.make_cache()
        active_runtime.block_speculative_backend.make_cache(active_runtime)
        calls.append(
            (
                "k2",
                prompt_ids,
                tuple(prompt_ids),
                kwargs,
                active_runtime.block_speculative_backend is original_backend,
                "make_cache" in vars(active_runtime),
            )
        )
        tokens = k2_tokens(k2_calls)
        assert target_cache.offset == 9
        return SimpleNamespace(
            tokens=tokens,
            stats=_stats(
                speculative=True,
                signature_variant=k2_signature_variant(k2_calls),
            ),
        )

    identities = _identities(harness)
    post_run_git = deepcopy(identities["git_identity"])
    if post_run_git_mutator is not None:
        post_run_git_mutator(post_run_git)
    receipt = harness.run_benchmark(
        argparse.Namespace(
            model=Path("/model").resolve(),
            max_tokens=64,
            out=Path("/receipt.json"),
        ),
        mx=mx,
        runtime_load=load,
        generate_ar=generate_ar,
        generate_mtpk=generate_mtpk,
        sampler_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        imported_modules_attestation=lambda: {
            "module_set_sha256": "c" * 64,
            "files": {
                name: {
                    "path": name.replace(".", "/") + ".py",
                    "sha256": (
                        "a" * 64
                        if name == "mtplx.native_block_speculation"
                        else "e" * 64
                    ),
                }
                for name in harness._REQUIRED_IMPORTED_MODULES
            },
            "preflight_head_bound": True,
            "preflight_sources_bound": True,
        },
        post_run_git_attestation=lambda: post_run_git,
        scheduler_arm=_scheduler_arm(),
        **identities,
    )
    return receipt, runtime, original_backend, calls, loads, mx


def test_run_is_stock_one_load_with_unmeasured_state_proof_and_fixed_five():
    harness = _load_harness()
    receipt, runtime, original_backend, calls, loads, mx = _run_fake_benchmark(harness)

    assert loads == [(Path("/model").resolve(), {"mtp": True})]
    assert [call[0] for call in calls] == [
        "ar",
        "k2",
        "ar",
        "k2",
        *(lane for _ in range(5) for lane in ("ar", "k2")),
    ]
    assert [call[3]["max_tokens"] for call in calls[:2]] == [4, 3]
    assert all(call[3]["max_tokens"] == 64 for call in calls[2:])
    assert all(call[3]["stop_token_ids"] == set() for call in calls)
    assert len({id(call[1]) for call in calls}) == len(calls)
    assert all(call[2] == tuple(range(11, 20)) for call in calls)
    assert calls[0][4:] == (False, True)
    assert calls[1][4:] == (False, True)
    assert all(call[4:] == (True, False) for call in calls[2:])
    assert runtime.block_speculative_backend is original_backend
    assert "make_cache" not in vars(runtime)
    assert mx.reset_calls == 10

    assert receipt["baseline"] == "generic_mtp_true_stock"
    assert receipt["load_kwargs"] == {"mtp": True}
    assert receipt["repetitions"] == harness.REPETITIONS == 5
    assert receipt["scheduler_arm"] == _scheduler_arm()
    assert receipt["provenance"]["git"] == receipt["provenance"]["git_post_run"]
    assert receipt["state_proof"]["measured"] is False
    assert receipt["state_proof"]["complete_k2_cycle"] is True
    assert receipt["state_proof"]["target_rows_consumed"] == 3
    assert receipt["state_proof"]["k2_drafted_by_depth"] == [4, 2]
    assert receipt["state_proof"]["target_state_equal"] is True
    assert receipt["state_proof"]["wrappers_restored_before_primers"] is True
    assert receipt["state_proof"]["ar_target"] == receipt["state_proof"]["k2_target"]
    assert len(receipt["state_proof"]["proposal_snapshot"]["state_sha256"]) == 64
    assert len(receipt["measurements"]["samples"]) == 5
    assert all(
        row["acceptance_signature"]
        == receipt["measurements"]["samples"][0]["acceptance_signature"]
        for row in receipt["measurements"]["samples"]
    )
    assert receipt["gates"] == {
        "state_proof_target_equal": True,
        "tokens_exact_all_samples": True,
        "ar_deterministic": True,
        "k2_deterministic": True,
        "acceptance_signature_identical_all_samples": True,
    }
    assert receipt["passed"] is True


def test_state_proof_rejects_target_cache_drift_and_restores_wrappers():
    harness = _load_harness()
    runtime = _FakeRuntime()
    original_backend = runtime.block_speculative_backend

    def ar(active_runtime, _prompt, **_kwargs):
        active_runtime.make_cache()
        return SimpleNamespace(tokens=[101, 102, 103, 104])

    def k2(active_runtime, _prompt, **_kwargs):
        target_cache = active_runtime.make_cache()
        target_cache.offset = 10
        active_runtime.block_speculative_backend.make_cache(active_runtime)
        return SimpleNamespace(tokens=[101, 102, 103], stats=_stats(speculative=True))

    with pytest.raises(RuntimeError, match="target cache state is not bit-exact"):
        harness.prove_prefill_state(
            runtime,
            list(range(9)),
            generate_ar=ar,
            generate_mtpk=k2,
            sampler=SimpleNamespace(),
        )
    assert runtime.block_speculative_backend is original_backend
    assert "make_cache" not in vars(runtime)


def test_state_proof_requires_both_k2_draft_depths():
    harness = _load_harness()
    runtime = _FakeRuntime()

    def ar(active_runtime, _prompt, **_kwargs):
        active_runtime.make_cache()
        return SimpleNamespace(tokens=[101, 102, 103, 104])

    def k2(active_runtime, _prompt, **_kwargs):
        active_runtime.make_cache()
        active_runtime.block_speculative_backend.make_cache(active_runtime)
        stats = _stats(speculative=True)
        stats.drafted_by_depth = [1, 0]
        return SimpleNamespace(tokens=[101, 102, 103], stats=stats)

    with pytest.raises(RuntimeError, match="one complete K2 cycle"):
        harness.prove_prefill_state(
            runtime,
            list(range(9)),
            generate_ar=ar,
            generate_mtpk=k2,
            sampler=SimpleNamespace(),
        )


def test_all_samples_gate_tokens_determinism_and_acceptance_signature():
    harness = _load_harness()
    receipt, *_ = _run_fake_benchmark(
        harness,
        ar_tokens=lambda call: [999] if call == 7 else [101, 102, 103],
        k2_signature_variant=lambda call: 1 if call == 7 else 0,
    )
    assert receipt["gates"]["ar_deterministic"] is False
    assert receipt["gates"]["tokens_exact_all_samples"] is False
    assert receipt["gates"]["acceptance_signature_identical_all_samples"] is False
    assert receipt["passed"] is False


def test_post_run_git_must_match_exact_preflight_tree():
    harness = _load_harness()

    def change_tree(identity):
        identity["head_tree"] = "0" * 40

    with pytest.raises(RuntimeError, match="provenance changed during"):
        _run_fake_benchmark(harness, post_run_git_mutator=change_tree)


def test_fixed_prompt_fails_before_state_proof():
    harness = _load_harness()
    runtime = _FakeRuntime()
    runtime.tokenizer.encode = lambda _text: list(range(8))
    with pytest.raises(RuntimeError, match="fixed prompt tokenizer drift"):
        harness.run_benchmark(
            argparse.Namespace(model=Path("/model"), max_tokens=64),
            mx=_FakeMX(),
            runtime_load=lambda *_args, **_kwargs: runtime,
            generate_ar=lambda *_args, **_kwargs: None,
            generate_mtpk=lambda *_args, **_kwargs: None,
            sampler_factory=lambda **kwargs: SimpleNamespace(**kwargs),
            imported_modules_attestation=lambda: {
                "preflight_head_bound": True,
                "preflight_sources_bound": True,
            },
            post_run_git_attestation=lambda: {},
            scheduler_arm=_scheduler_arm(),
            **_identities(harness),
        )


def test_write_receipt_returns_nonzero_for_failed_gate(tmp_path):
    harness = _load_harness()
    receipt = {"passed": False, "gates": {"tokens": False}}
    output = tmp_path / "receipt.json"
    assert harness.write_receipt(receipt, output) == 1
    assert json.loads(output.read_text()) == receipt


def test_scheduler_arm_is_derived_from_only_two_sanctioned_source_orders():
    harness = _load_harness()
    actual = harness.attest_scheduler_arm(ROOT)
    assert actual["label"] in harness._ARM_LABELS
    assert actual["arm_id"] == f"{actual['label']}:{actual['source_sha256']}"
    assert actual["source_sha256"] == harness._sha256(
        ROOT / "mtplx/native_block_speculation.py"
    )
    assert (
        actual["normalized_source_sha256"]
        == harness.EXPECTED_NORMALIZED_SCHEDULER_SHA256
    )
    assert (
        actual["reviewed_boundary_patch_sha256"]
        == harness.EXPECTED_SCHEDULER_BOUNDARY_PATCH_SHA256
    )

    source = (ROOT / harness._SCHEDULER_SOURCE).read_text()
    lazy_source = source.replace(
        harness._MATERIALIZE_BOUNDARY_BLOCK,
        harness._LAZY_BOUNDARY_BLOCK,
    )
    materialize_source = lazy_source.replace(
        harness._LAZY_BOUNDARY_BLOCK,
        harness._MATERIALIZE_BOUNDARY_BLOCK,
    )
    assert harness._classify_scheduler_source(lazy_source) == (
        "lazy_joint_eval",
        harness._ARM_EVENTS["lazy_joint_eval"],
    )
    assert harness._classify_scheduler_source(materialize_source) == (
        "materialize_first",
        harness._ARM_EVENTS["materialize_first"],
    )
    with pytest.raises(ValueError, match="outside reviewed boundary motion"):
        harness._classify_scheduler_source(lazy_source + "# unrelated edit\n")
    with pytest.raises(ValueError, match="exact sanctioned bracket arm"):
        harness._classify_scheduler_source(
            materialize_source.replace("Settle and materialize", "Changed materialize")
        )


def test_scheduler_arm_rejects_crlf_byte_drift(tmp_path):
    harness = _load_harness()
    source = (ROOT / harness._SCHEDULER_SOURCE).read_bytes()
    scheduler_path = tmp_path / harness._SCHEDULER_SOURCE
    scheduler_path.parent.mkdir(parents=True)
    scheduler_path.write_bytes(source.replace(b"\n", b"\r\n"))

    with pytest.raises(ValueError, match="exact sanctioned bracket arm"):
        harness.attest_scheduler_arm(tmp_path)


def _comparison_pair(harness):
    lazy, *_ = _run_fake_benchmark(harness)
    materialize = deepcopy(lazy)
    materialize_sha = "d" * 64
    materialize["scheduler_arm"] = _scheduler_arm("materialize_first", materialize_sha)
    materialize["provenance"]["git"] = {
        **materialize["provenance"]["git"],
        "commit": "e" * 40,
        "head_tree": "f" * 40,
    }
    materialize["provenance"]["git"]["head_tree_files"] = deepcopy(
        materialize["provenance"]["git"]["head_tree_files"]
    )
    materialize["provenance"]["git"]["head_tree_files"][harness._SCHEDULER_SOURCE][
        "object"
    ] = materialize_sha
    materialize["provenance"]["git"]["head_python_sha256"][
        harness._SCHEDULER_SOURCE
    ] = materialize_sha
    materialize["provenance"]["sources"]["files"][harness._SCHEDULER_SOURCE] = (
        materialize_sha
    )
    materialize["provenance"]["sources"]["source_set_sha256"] = "0" * 64
    materialize["provenance"]["sources"]["importable_mtplx_files"][
        harness._SCHEDULER_SOURCE
    ] = materialize_sha
    materialize["provenance"]["imported_mtplx_modules"]["files"][
        "mtplx.native_block_speculation"
    ]["sha256"] = materialize_sha
    materialize["provenance"]["git_post_run"] = deepcopy(
        materialize["provenance"]["git"]
    )
    return lazy, materialize


def test_comparator_gates_source_isolation_state_tokens_and_acceptance():
    harness = _load_harness()
    lazy, materialize = _comparison_pair(harness)
    comparison = harness.compare_receipts(materialize, lazy)
    assert comparison["source_differences"] == [harness._SCHEDULER_SOURCE]
    assert comparison["head_tree_differences"] == [harness._SCHEDULER_SOURCE]
    assert comparison["imported_module_differences"] == [
        "mtplx.native_block_speculation"
    ]
    assert all(comparison["state_digests_equal"].values())
    assert all(comparison["gates"].values())
    assert comparison["passed"] is True

    materialize["state_proof"]["proposal_snapshot"]["state_sha256"] = "1" * 64
    materialize["measurements"]["samples"][0]["acceptance_signature"][
        "accepted_drafts"
    ] += 1
    failed = harness.compare_receipts(lazy, materialize)
    assert failed["gates"]["proposal_snapshot_identical"] is False
    assert failed["gates"]["acceptance_signature_identical_cross_arm"] is False
    assert failed["passed"] is False

    materialize = _comparison_pair(harness)[1]
    materialize["provenance"]["git"]["head_tree_files"]["unrelated.txt"] = {
        "mode": "100644",
        "object": "2" * 40,
    }
    unrelated = harness.compare_receipts(lazy, materialize)
    assert unrelated["gates"]["only_scheduler_head_blob_differs"] is False

    materialize = _comparison_pair(harness)[1]
    materialize["provenance"]["imported_mtplx_modules"]["files"]["mtplx.runtime"][
        "sha256"
    ] = "3" * 64
    imported = harness.compare_receipts(lazy, materialize)
    assert imported["gates"]["only_scheduler_import_differs"] is False


def test_comparator_rejects_arm_label_not_bound_to_source_hash():
    harness = _load_harness()
    lazy, materialize = _comparison_pair(harness)
    materialize["scheduler_arm"]["arm_id"] = "materialize_first:wrong"
    with pytest.raises(ValueError, match="arm attribution is invalid"):
        harness.compare_receipts(lazy, materialize)


def test_git_attestation_requires_clean_committed_head_tree(tmp_path):
    harness = _load_harness()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True
    )
    (tmp_path / "tracked.txt").write_text("clean\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "clean"], check=True
    )
    identity = harness.attest_git(tmp_path)
    assert identity["dirty"] is False
    assert len(identity["commit"]) == 40
    assert len(identity["head_tree"]) == 40

    (tmp_path / "dirty.txt").write_text("dirty\n")
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        harness.attest_git(tmp_path)


def test_imported_module_attestation_rejects_any_reviewed_overlay(tmp_path):
    harness = _load_harness()
    modules = {}
    preflight_hashes = {}
    for name in harness._REQUIRED_IMPORTED_MODULES:
        path = tmp_path / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        module = ModuleType(name)
        module.__file__ = str(path)
        modules[name] = module
        preflight_hashes[str(path.relative_to(tmp_path))] = harness._sha256(path)
    git_identity = {"head_python_sha256": dict(preflight_hashes)}
    source_identity = {"importable_mtplx_files": dict(preflight_hashes)}
    identity = harness.attest_imported_mtplx_modules(
        tmp_path,
        git_identity=git_identity,
        source_identity=source_identity,
        modules=modules,
    )
    assert set(identity["files"]) == set(harness._REQUIRED_IMPORTED_MODULES)
    assert identity["preflight_head_bound"] is True
    assert identity["preflight_sources_bound"] is True

    first_path = next(iter(preflight_hashes))
    git_identity["head_python_sha256"][first_path] = "0" * 64
    with pytest.raises(RuntimeError, match="does not match preflight HEAD"):
        harness.attest_imported_mtplx_modules(
            tmp_path,
            git_identity=git_identity,
            source_identity=source_identity,
            modules=modules,
        )
    git_identity["head_python_sha256"] = dict(preflight_hashes)
    source_identity["importable_mtplx_files"][first_path] = "1" * 64
    with pytest.raises(RuntimeError, match="does not match source attestation"):
        harness.attest_imported_mtplx_modules(
            tmp_path,
            git_identity=git_identity,
            source_identity=source_identity,
            modules=modules,
        )
    source_identity["importable_mtplx_files"] = dict(preflight_hashes)

    outside = tmp_path.parent / "overlay.py"
    outside.write_text("overlay")
    modules[harness._REQUIRED_IMPORTED_MODULES[0]].__file__ = str(outside)
    with pytest.raises(RuntimeError, match="outside worktree"):
        harness.attest_imported_mtplx_modules(
            tmp_path,
            git_identity=git_identity,
            source_identity=source_identity,
            modules=modules,
        )


def test_official_mlx_attestation_rejects_editable_or_import_overlay(tmp_path):
    harness = _load_harness()
    site = tmp_path / "site-packages"
    core = site / "mlx" / "core.cpython.so"
    core.parent.mkdir(parents=True)
    core.write_bytes(b"official wheel core")
    libmlx = site / "mlx" / "lib" / "libmlx.dylib"
    metallib = site / "mlx" / "lib" / "mlx.metallib"
    libmlx.parent.mkdir()
    libmlx.write_bytes(b"official wheel dylib")
    metallib.write_bytes(b"official wheel metallib")
    harness.EXPECTED_MLX_CORE_SHA256 = harness._sha256(core)
    harness.EXPECTED_MLX_LIB_SHA256 = harness._sha256(libmlx)
    harness.EXPECTED_MLX_METALLIB_SHA256 = harness._sha256(metallib)

    class Distribution:
        version = "0.32.0"

        def __init__(self, direct_url=None):
            self.direct_url = direct_url

        def locate_file(self, value):
            return site / value

        def read_text(self, name):
            if name == "INSTALLER":
                return "uv\n"
            if name == "direct_url.json":
                return self.direct_url
            return None

    mx = SimpleNamespace(__file__=str(core))
    identity = harness.attest_official_mlx(mx, Distribution())
    assert identity["version"] == "0.32.0"
    assert identity["installer"] == "uv"
    assert identity["core_path"] == str(core.resolve())
    assert identity["libmlx"]["sha256"] == harness._sha256(libmlx)
    assert identity["metallib"]["sha256"] == harness._sha256(metallib)

    editable = json.dumps(
        {"url": "file:///tmp/mlx-source", "dir_info": {"editable": True}}
    )
    with pytest.raises(ValueError, match="source/direct overlay"):
        harness.attest_official_mlx(mx, Distribution(editable))

    outside = tmp_path / "mlx-overlay" / "core.cpython.so"
    outside.parent.mkdir()
    outside.write_bytes(b"overlay")
    with pytest.raises(ValueError, match="outside installed distribution"):
        harness.attest_official_mlx(
            SimpleNamespace(__file__=str(outside)), Distribution()
        )


def test_model_attestation_requires_pinned_0731_hashes(tmp_path):
    harness = _load_harness()
    (tmp_path / "config.json").write_bytes(b"config")
    (tmp_path / "model.safetensors.index.json").write_bytes(b"index")
    harness.EXPECTED_MODEL_CONFIG_SHA256 = harness._sha256(tmp_path / "config.json")
    harness.EXPECTED_MODEL_INDEX_SHA256 = harness._sha256(
        tmp_path / "model.safetensors.index.json"
    )
    metadata_root = tmp_path / ".cache" / "huggingface" / "download"
    metadata_root.mkdir(parents=True)
    for name in ("config.json.metadata", "model.safetensors.index.json.metadata"):
        (metadata_root / name).write_text(
            harness.EXPECTED_MODEL_METADATA_REVISION + "\nmetadata\n"
        )

    identity = harness.attest_model(tmp_path)
    assert identity["config_sha256"] == harness.EXPECTED_MODEL_CONFIG_SHA256
    assert identity["index_sha256"] == harness.EXPECTED_MODEL_INDEX_SHA256
    assert {row["revision"] for row in identity["metadata"].values()} == {
        harness.EXPECTED_MODEL_METADATA_REVISION
    }

    (tmp_path / "config.json").write_bytes(b"drift")
    with pytest.raises(ValueError, match="config SHA mismatch"):
        harness.attest_model(tmp_path)


def test_cli_has_fixed_repetitions_and_source_only_arm(tmp_path):
    harness = _load_harness()
    args = harness._parse_args(
        ["--model", str(tmp_path), "--out", str(tmp_path / "receipt.json")]
    )
    assert args.model == tmp_path
    assert not hasattr(args, "repetitions")
    assert not hasattr(args, "mode")
    assert harness.REPETITIONS == 5

    compare = harness._parse_args(
        [
            "--compare",
            str(tmp_path / "lazy.json"),
            str(tmp_path / "materialize.json"),
            "--out",
            str(tmp_path / "comparison.json"),
        ]
    )
    assert compare.compare == [tmp_path / "lazy.json", tmp_path / "materialize.json"]


def test_source_is_scheduler_only_clean_before_mlx_and_includes_guard_bridge():
    source = SCRIPT.read_text()
    assert "os.environ" not in source
    assert "MLX_DISPATCH_CENSUS" not in source
    assert "prepare_dspark_q3_packed_gate_up_m5" not in source
    assert "dspark-ffn-q3-m5" not in source
    assert 'add_argument("--mode"' not in source
    assert '"--repetitions"' not in source
    assert "deepseek_v4_0731_k2=" not in source
    assert "REPETITIONS = 5" in source
    assert "attest_scheduler_arm(" in source
    assert "scripts/deepseek_v4_guard_window.py" in source
    assert source.index("git_identity = attest_git(repo)") < source.index(
        "import mlx.core as mx"
    )
    assert source.index("issue_guard_window()") < source.index("import mlx.core as mx")
    assert source.index("load_verified_guard_window(") < source.index(
        "import mlx.core as mx"
    )
