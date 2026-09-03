"""Family layer-owned capture-commit: repair-free verify rollback.

A rejected speculative window must commit by replaying ONLY the GDN
recurrences (and trimming trimmable entries) from the pre-verify snapshot,
matching a run that never saw the rejected rows to fp32 ulp-class tolerance.
Not bitwise: the chunked gated-delta scan reassociates when the kept rows
ride a wider verify window, so captured activations differ from a fresh
narrow forward's at the last float — the same noise class the fallback
path's own rollback+re-forward produces relative to the verify pass. The
acceptance decision itself always uses the verify pass's own logits, so
this tolerance never touches sampling exactness.

CPU-only (parity surface).
"""

import mlx.core as mx
import numpy as np
import pytest
import tempfile
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

from mtplx.cache_state import snapshot_untrimmable_cache_lazy
from mtplx.models.qwen4_exp import (
    TextArgs,
    TextModel,
    verify_capture_scope,
)


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[2],
        ngram_vocab_size_base=512,
        heads_per_ngram=2,
        ple_embed_dim=64,
    )


@pytest.fixture()
def tm():
    import mlx_lm.models.cache as cache_module
    import mtplx.models.qwen4_exp as qwen4_exp

    prev = mx.default_device()
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    model = TextModel(_tiny_args())
    mx.eval(model.parameters())
    yield model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(prev)


@pytest.fixture()
def tm_fixed_m4_split():
    """Small-width production layer topology for guarded split parity."""

    import mlx_lm.models.cache as cache_module
    import mtplx.models.qwen4_exp as qwen4_exp

    prev = mx.default_device()
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.set_default_device(mx.cpu)
    mx.random.seed(391)
    args = replace(
        _tiny_args(),
        num_hidden_layers=48,
        layer_types=None,
        full_attention_interval=4,
        indexer_compress_ratio=4,
    )
    model = TextModel(args)
    mx.eval(model.parameters())
    yield model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(prev)


def _ids(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(0, 128, (1, tokens))


def _host_ids(ids: mx.array) -> list[int]:
    return [int(token) for token in np.asarray(ids).reshape(-1)]


PREFILL = 12
WINDOW = 4
KEEP = 2


class _FakeSidecar:
    def __init__(self):
        self.direct_inputs = []
        self.warm_inputs = []
        self._file = tempfile.TemporaryFile()
        self._file.truncate(1 << 20)
        self._fd = self._file.fileno()
        self._hot = OrderedDict()
        self._hot_cap_rows = 16
        self.bits = 4
        self.group_size = 32
        self._maps = {
            "weight": (
                SimpleNamespace(
                    offset=0,
                    shape=(4096, 20),
                    dtype=np.dtype(np.uint32),
                ),
                "U32",
            ),
            "scales": (
                SimpleNamespace(
                    offset=4096 * 20 * 4,
                    shape=(4096, 5),
                    dtype=np.dtype(np.uint16),
                ),
                "BF16",
            ),
            "biases": (
                SimpleNamespace(
                    offset=4096 * (20 * 4 + 5 * 2),
                    shape=(4096, 5),
                    dtype=np.dtype(np.uint16),
                ),
                "BF16",
            ),
        }
        self._pool = self._Pool(self)

    class _Warm:
        def __init__(self, value=None):
            self._value = value

        def result(self):
            return self._value

    class _Pool:
        def __init__(self, owner):
            self._owner = owner

        def submit(self, function, *args):
            self._owner.warm_inputs.append(int(args[0]))
            return _FakeSidecar._Warm(function(*args))

    def submit_warm(self, flat):
        self.warm_inputs.append(np.asarray(flat, dtype=np.int64).copy())
        return (self._Warm(),)

    def gather_raw_np(self, flat):
        flat = np.asarray(flat, dtype=np.int64)
        self.direct_inputs.append(flat.copy())
        count = len(flat)
        return (
            mx.zeros((count, 2), dtype=mx.uint32),
            mx.zeros((count, 1), dtype=mx.bfloat16),
            mx.zeros((count, 1), dtype=mx.bfloat16),
        )

    def gather_np(self, flat):
        weight, scales, biases = self.gather_raw_np(flat)
        return mx.dequantize(
            weight,
            scales,
            biases,
            group_size=self.group_size,
            bits=self.bits,
        )

    def __call__(self, ids, dim):
        flat = np.asarray(ids.reshape(-1), dtype=np.int64)
        return self.gather_np(flat).reshape(*ids.shape, dim)


def _run(tm, chunks, cache):
    out = None
    for ids in chunks:
        out = tm.model(ids, cache)
    return out


def test_capture_commit_matches_fresh_run_eager(tm):
    ids_pre = _ids(PREFILL, seed=1)
    ids_verify = _ids(WINDOW, seed=2)
    ids_next = _ids(3, seed=3)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_verify, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [ids_next], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_verify[:, :KEEP], golden_cache)
    golden = _run(tm, [ids_next], golden_cache)

    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()


def test_capture_commit_matches_fresh_run_compiled(tm):
    tm.model._gdn_compiled_env = True

    ids_pre = _ids(PREFILL, seed=4)
    ids_verify = _ids(WINDOW, seed=5)
    ids_next = _ids(3, seed=6)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_verify, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [ids_next], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_verify[:, :KEEP], golden_cache)
    golden = _run(tm, [ids_next], golden_cache)

    # The compiled/eager boundary may reorder float ops; the commit itself
    # must still be exact relative to the same-lane golden (golden ran
    # eager S=2 which the compiled gate also serves) — compare through the
    # compiled lane end to end.
    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()


def test_commit_refuses_without_capture_and_leaves_cache_intact(tm):
    ids_pre = _ids(PREFILL, seed=7)
    ids_verify = _ids(WINDOW, seed=8)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    tm.model(ids_verify, cache)  # NOT captured
    offsets_before = [getattr(c, "offset", None) for c in cache]
    assert not tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    assert [getattr(c, "offset", None) for c in cache] == offsets_before


def test_full_accept_needs_no_commit_and_next_round_overwrites_rows(tm):
    ids_pre = _ids(PREFILL, seed=9)
    ids_v1 = _ids(WINDOW, seed=10)
    ids_v2 = _ids(WINDOW, seed=11)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    with verify_capture_scope():
        tm.model(ids_v1, cache)  # full accept: no commit
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_v2, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [_ids(2, seed=12)], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_v1, golden_cache)
    tm.model(ids_v2[:, :KEEP], golden_cache)
    golden = _run(tm, [_ids(2, seed=12)], golden_cache)

    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()


def test_fixed_m4_capture_route_returns_family_commit_rows(tm):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    report = install_qwen4_fixed_verify_route(runtime)
    cache = tm.make_cache()
    ids_pre = _ids(PREFILL, seed=18)
    ids_verify = _ids(WINDOW, seed=19)
    tm(ids_pre, cache=cache)

    aux = runtime.prepare_compiled_verify_aux(ids_verify, cache)
    logits, hidden, captures = runtime.forward_ar_capture(
        ids_verify,
        cache=cache,
        return_hidden=True,
        compiled_aux=aux,
    )
    mx.eval(logits, hidden)

    linear = [i for i, layer in enumerate(tm.model.layers) if layer.is_linear]
    ple_index = next(
        i for i, layer in enumerate(tm.model.layers) if getattr(layer, "ple", None)
    )
    assert report == {"installed": True, "linear_layers": len(linear), "rows": 4}
    assert all(
        tuple(captures[i])[:6] == ("qkv", "q", "k", "v", "a", "b") for i in linear
    )
    assert {"ple_hidden", "ple_ids"}.issubset(captures[ple_index])

    tm.model.clear_verify_capture(cache)
    runtime.commit_compiled_verify_captures(cache, captures)
    assert all(cache[i]._mtplx_verify_rows is not None for i in linear)
    assert cache[ple_index]._mtplx_verify_ple is not None


@pytest.mark.parametrize(
    ("tokens", "previous"),
    (
        ((3, 4, 5, 6), None),
        ((0, 4, 5, 6), None),
        ((3, 0, 5, 6), None),
        ((3, 4, 0, 6), None),
        ((3, 4, 5, 0), None),
        ((3, 4, 5, 6), (0, 9)),
    ),
)
def test_fixed_m4_sidecar_aux_stages_exact_rows_without_mutating_history(
    tm, tokens, previous
):
    from mtplx.qwen4_fixed_verify import (
        _build_fixed_m4_compiled_verify_aux,
        _dequantize_fixed_m4_ple,
        _prepare_compiled_verify_aux,
    )

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=28)
    tm(prefill, cache=cache)
    ple_index = next(
        i for i, layer in enumerate(tm.model.layers) if getattr(layer, "ple", None)
    )
    ple = tm.model.layers[ple_index].ple
    if previous is not None:
        cache[ple_index][ple.NGRAM_IDX] = mx.array([previous], dtype=mx.int64)
    sidecar = _FakeSidecar()
    ple.ple_embedding.ngram_embedding._sidecar = sidecar

    prompt_ids = list(previous) if previous is not None else _host_ids(prefill)
    prepare = _build_fixed_m4_compiled_verify_aux(runtime, cache, prompt_ids)
    ids = mx.array([tokens])
    history_before = cache[ple_index][ple.NGRAM_IDX]
    reference = _prepare_compiled_verify_aux(runtime, ids, cache)
    prepare.prefetch_primary(
        int(tokens[0]),
        [int(tokens[0])],
        0,
    )
    raw_candidate = prepare(ids, list(tokens), [int(tokens[0])], 0)
    candidate = _dequantize_fixed_m4_ple(raw_candidate, output_dim=64)
    mx.eval(reference, candidate, *raw_candidate)

    assert mx.array_equal(candidate, reference).item()
    assert candidate.shape == (1, WINDOW, 64)
    assert tuple(array.dtype for array in raw_candidate) == (
        mx.uint32,
        mx.bfloat16,
        mx.bfloat16,
    )
    assert cache[ple_index][ple.NGRAM_IDX] is history_before
    assert len(sidecar.warm_inputs) == len(sidecar.direct_inputs[0]) // WINDOW
    assert np.array_equal(
        np.asarray(sidecar.warm_inputs, dtype=np.int64),
        sidecar.direct_inputs[0][: len(sidecar.warm_inputs)],
    )
    assert len(sidecar.direct_inputs) == 2
    assert np.array_equal(sidecar.direct_inputs[0], sidecar.direct_inputs[1])

    rebound_history = mx.array([[0, 11]])
    cache[ple_index][ple.NGRAM_IDX] = rebound_history
    second_ids = mx.array([[9, 8, 7, 6]])
    second_reference = _prepare_compiled_verify_aux(runtime, second_ids, cache)
    prepare.prefetch_primary(9, [0, 11, 9], 2)
    second_raw = prepare(second_ids, [9, 8, 7, 6], [0, 11, 9], 2)
    second_candidate = _dequantize_fixed_m4_ple(second_raw, output_dim=64)
    mx.eval(second_reference, second_candidate, *second_raw)

    assert mx.array_equal(second_candidate, second_reference).item()
    assert cache[ple_index][ple.NGRAM_IDX] is rebound_history
    assert len(sidecar.direct_inputs) == 4
    assert np.array_equal(sidecar.direct_inputs[2], sidecar.direct_inputs[3])


def test_compiled_fixed_m4_route_preserves_family_prefix_commit(tm, monkeypatch):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)

    cache = tm.make_cache()
    ids_pre = _ids(PREFILL, seed=20)
    ids_verify = _ids(WINDOW, seed=21)
    tm(ids_pre, cache=cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16,
    )

    logits, hidden, captures = bank.forward_ar_capture(ids_verify, cache=cache)
    mx.eval(logits, hidden)

    assert bank.stats["fallback_calls"] == 0, bank.stats["fallback_reasons"]
    assert bank.stats["compiled_calls"] == 1
    assert captures
    assert tm.model.commit_verified_window(
        cache,
        snap.states,
        keep_tokens=KEEP,
        verified_tokens=WINDOW,
    )


def test_installed_fixed_m4_replay_preserves_compiled_gdn_schedule(tm, monkeypatch):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)

    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=23)
    tm(prefill, cache=cache)
    tm.model._gdn_compiled_env = True

    compiled_gdn_calls = []
    original_compiled_gdn = tm.model._decode_layers_compiled

    def observed_compiled_gdn(*args, **kwargs):
        compiled_gdn_calls.append(True)
        return original_compiled_gdn(*args, **kwargs)

    monkeypatch.setattr(tm.model, "_decode_layers_compiled", observed_compiled_gdn)
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16,
    )
    bank.install_fixed_m4(
        cache,
        prompt_ids=_host_ids(prefill),
        hidden_variant=None,
    )

    def repeated_check(*_args, **_kwargs):
        raise AssertionError("installed M4 replay re-entered generic dispatch")

    monkeypatch.setattr(bank, "_fallback_reason", repeated_check)
    monkeypatch.setattr(bank, "_resolve_bucket", repeated_check)
    monkeypatch.setattr(bank, "_ensure_shadow", repeated_check)
    monkeypatch.setattr(bank, "_paged_ineligibility", repeated_check)

    completion_tokens = []
    for seed in (24, 25):
        snap = snapshot_untrimmable_cache_lazy(cache)
        ids = _ids(WINDOW, seed=seed)
        host_ids = _host_ids(ids)
        completion_tokens.append(host_ids[0])
        logits, hidden, captures = bank.forward_fixed_m4(
            ids,
            host_input_ids=host_ids,
            completion_tokens=completion_tokens,
            committed_count=len(completion_tokens) - 1,
            cache=cache,
        )
        mx.eval(logits, hidden)
        assert captures == {}
        assert tm.model.commit_verified_window(
            cache,
            snap.states,
            keep_tokens=KEEP,
            verified_tokens=WINDOW,
        )
        completion_tokens.extend(host_ids[1:KEEP])

    assert bank.stats["compiled_calls"] == 2
    assert compiled_gdn_calls


def test_fixed_m4_split_matches_monolithic_across_two_independent_windows_guarded(
    tm_fixed_m4_split,
    monkeypatch,
):
    """Guarded MLX gate: split and monolithic mutate independent state trees."""

    import mtplx.generation as generation
    import mtplx.graphbank as graphbank
    import mtplx.qwen4_fixed_verify as fixed_verify
    from mtplx.kernels.qwen4_m4_state_handoff import (
        QWEN4_M4_GDN_CONV_ROWS,
        QWEN4_M4_VERIFY_WIDTH,
        replay_qwen4_m4_gdn_state,
    )
    from mtplx.kernels.pr391_softfloat64_verifier_decision import (
        SELECTED_BONUS,
        SELECTED_CORRECTION,
        reference_pr391_softfloat64_verifier_decision,
    )
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    tm = tm_fixed_m4_split
    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_BOUNDARY", "both")

    prefill = _ids(PREFILL, seed=391)
    prompt_ids = _host_ids(prefill)
    monolithic_cache = tm.make_cache()
    split_cache = tm.make_cache()
    legacy_cache = tm.make_cache()
    tm(prefill, cache=monolithic_cache)
    tm(prefill, cache=split_cache)
    tm(prefill, cache=legacy_cache)

    monolithic = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=WINDOW, request_max_tokens=32
    )
    split = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=WINDOW, request_max_tokens=32
    )
    legacy = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=WINDOW, request_max_tokens=32
    )
    monolithic.install_fixed_m4(
        monolithic_cache, prompt_ids=prompt_ids, hidden_variant=None
    )
    split.install_fixed_m4(split_cache, prompt_ids=prompt_ids, hidden_variant=None)
    legacy.install_fixed_m4(
        legacy_cache, prompt_ids=prompt_ids, hidden_variant=None
    )
    split.install_fixed_m4_split()

    # Expose layer 0 as one additional output of a full, monolithic compiled
    # probe. Comparing the split prefix to an eager layer-0 call is not an
    # exact gate: the compile boundary may reassociate fp32 operations by one
    # ULP even when the full split and monolithic graphs are bit-identical.
    # This test-only recorder roots the actual intermediate produced inside
    # the otherwise unchanged monolithic graph.
    layer0 = tm.model.layers[0]
    layer_type = type(layer0)
    original_layer_call = layer_type.__call__
    recorded_layer0 = {}

    def record_layer0(self, *args, **kwargs):
        output = original_layer_call(self, *args, **kwargs)
        if self is layer0:
            recorded_layer0["hidden"] = output
        return output

    monkeypatch.setattr(layer_type, "__call__", record_layer0)
    original_forward_ar_capture = runtime.forward_ar_capture

    def forward_ar_capture_with_layer0(
        input_ids,
        *,
        cache=None,
        return_hidden=True,
        hidden_variant=None,
        capture_backend=None,
        compiled_aux=None,
    ):
        logits, hidden, captures = original_forward_ar_capture(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            capture_backend=capture_backend,
            compiled_aux=compiled_aux,
        )
        captures[0]["test_layer0_hidden"] = recorded_layer0["hidden"]
        return logits, hidden, captures

    runtime.forward_ar_capture = forward_ar_capture_with_layer0
    runtime._mtplx_capture_extra_layout = (
        *runtime._mtplx_capture_extra_layout,
        (0, ("test_layer0_hidden",)),
    )
    probe_cache = tm.make_cache()
    tm(prefill, cache=probe_cache)
    probe = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=WINDOW, request_max_tokens=32
    )
    probe.install_fixed_m4(
        probe_cache,
        prompt_ids=prompt_ids,
        hidden_variant=None,
    )
    probe_dispatch = probe._fixed_m4_dispatch

    def leaves(value):
        if isinstance(value, (tuple, list)):
            return tuple(leaf for child in value for leaf in leaves(child))
        if isinstance(value, dict):
            return tuple(leaf for child in value.values() for leaf in leaves(child))
        return () if value is None else (value,)

    def assert_tree_equal(left, right):
        left_leaves = leaves(left)
        right_leaves = leaves(right)
        assert len(left_leaves) == len(right_leaves)
        for observed, expected in zip(left_leaves, right_leaves, strict=True):
            assert mx.array_equal(observed, expected).item()

    def assert_array_equal(label, observed, expected):
        observed_np = np.asarray(observed)
        expected_np = np.asarray(expected)
        if np.array_equal(observed_np, expected_np):
            return
        mismatch = observed_np != expected_np
        first = tuple(int(index) for index in np.argwhere(mismatch)[0])
        max_abs = float(
            np.max(
                np.abs(
                    observed_np.astype(np.float64)
                    - expected_np.astype(np.float64)
                )
            )
        )
        raise AssertionError(
            f"{label}: mismatches={int(mismatch.sum())}/{mismatch.size}, "
            f"max_abs={max_abs:.9g}, first={first}, "
            f"observed={observed_np[first]!r}, expected={expected_np[first]!r}"
        )

    def live_identity(bank):
        state = bank._fixed_m4_state_inputs(bank._fixed_m4_dispatch["state_plan"])
        captures = []
        for entry, _start, _count in bank._fixed_m4_dispatch["capture_plan"]:
            captures.extend(
                (
                    getattr(entry, "_mtplx_verify_rows", None),
                    getattr(entry, "_mtplx_verify_ple", None),
                    getattr(entry, "_mtplx_verify_compiled_aux", None),
                )
            )
        return tuple(map(id, state)), tuple(map(id, captures))

    def production_decision(logits, *, force_reject):
        target_ids, target_values, target_probs = (
            generation._pr391_float32_target_support(logits)
        )
        mx.eval(target_ids, target_values, target_probs)
        target_ids = np.asarray(target_ids, dtype=np.uint32)
        target_values = np.asarray(target_values, dtype=np.float32)
        target_probs = np.asarray(target_probs, dtype=np.float32)
        draft_ids = target_ids[:3].copy()
        draft_values = target_values[:3].copy()
        draft_probs = target_probs[:3].copy()
        draft_tokens = draft_ids[:, 0].copy()
        uniforms = np.zeros(4, dtype=np.float64)
        if force_reject:
            occupied = set(int(token) for token in target_ids[0])
            missing = next(token for token in range(128) if token not in occupied)
            draft_tokens[0] = np.uint32(missing)
            draft_ids[0, 0] = np.uint32(missing)
            draft_values[0, 0] = np.float32(draft_values[0].max() + 1.0)
            draft_probs[0, 0] = np.float32(1.0)
            uniforms[:] = np.float64(0.75)

        def reference_kernel(
            draft_token_rows,
            draft_support,
            draft_scores,
            draft_masses,
            target_support,
            target_scores,
            target_masses,
            uniform_bits,
            stop_ids,
            stop_count,
            bonus_allowed,
        ):
            return reference_pr391_softfloat64_verifier_decision(
                np.asarray(draft_token_rows, dtype=np.uint32),
                np.asarray(draft_support, dtype=np.uint32),
                np.asarray(draft_scores, dtype=np.float32),
                np.asarray(draft_masses, dtype=np.float32),
                np.asarray(target_support, dtype=np.uint32),
                np.asarray(target_scores, dtype=np.float32),
                np.asarray(target_masses, dtype=np.float32),
                np.asarray(uniform_bits, dtype=np.uint64).view(np.float64),
                np.asarray(stop_ids, dtype=np.uint32),
                stop_count=int(np.asarray(stop_count).reshape(-1)[0]),
                bonus_allowed=bool(np.asarray(bonus_allowed).reshape(-1)[0]),
            )

        # This guarded model fixture is CPU-owned, so use the exact reference
        # kernel through the same helper that production uses to wire Metal.
        return generation._pr391_apply_softfloat64_decision(
            verifier_kernel=reference_kernel,
            draft_result=(draft_tokens, draft_ids, draft_values, draft_probs),
            target_support=(target_ids, target_values, target_probs),
            uniform_bits=uniforms.view(np.uint64),
            stop_ids=np.zeros(1, dtype=np.uint32),
            stop_count=np.zeros(1, dtype=np.uint32),
            bonus_allowed=np.ones(1, dtype=np.uint32),
        )

    def legacy_live_device_commit(
        cache,
        accepted_count,
        snapshot_states,
        verify_hidden,
    ):
        """Execute the retained pre-split live-cache commit schedule exactly."""

        accepted = accepted_count.reshape(-1)[0].astype(mx.int32)
        keep = accepted + 1
        conv_indices = keep + mx.arange(
            QWEN4_M4_GDN_CONV_ROWS, dtype=mx.int32
        )
        binding = runtime._mtplx_qwen4_m4_state_handoff_binding
        plan = tuple(
            (
                "gdn" if layer.is_linear else "qsa",
                index,
                entry,
                getattr(layer, "linear_attn", None),
                getattr(layer, "ple", None),
            )
            for index, (layer, entry) in enumerate(zip(tm.model.layers, cache))
        )

        ple_entry = cache[binding.ple_layer_index]
        ple_pre = snapshot_states[binding.ple_layer_index]
        ple_qkv, *_ple_gdn_rows = ple_entry._mtplx_verify_rows
        ple_hidden, ple_ids, ple_conv_rows = ple_entry._mtplx_verify_ple
        compiled_aux = ple_entry._mtplx_verify_compiled_aux
        ple_layer = tm.model.layers[binding.ple_layer_index]
        logical_states = []
        for logical_width in range(1, QWEN4_M4_VERIFY_WIDTH):
            logical_cache = type(ple_entry)(len(ple_entry.cache))
            for slot, leaf in enumerate(ple_pre):
                logical_cache[slot] = leaf
            with (
                fixed_verify.verify_capture_disabled_scope(),
                fixed_verify.compiled_verify_ple_scope(
                    compiled_aux[:, :logical_width]
                ),
            ):
                logical_hidden = ple_hidden[:, :logical_width] + ple_layer.ple(
                    ple_hidden[:, :logical_width],
                    ple_ids[:, :logical_width],
                    logical_cache,
                )
                logical_mixed, _logical_hyper, _logical_inject = (
                    ple_layer.attn_hyper_connection(logical_hidden)
                )
                ple_layer.linear_attn(logical_mixed, None, logical_cache)
            logical_states.append(tuple(logical_cache.cache))
        ple_logical_states = tuple(logical_states)
        (
            ple_gdn_conv,
            selected_ple_conv,
            selected_ple_history,
            selected_hidden,
            gdn_keep_mask,
        ) = binding.select_windows(
            accepted_count,
            ple_pre[0],
            ple_qkv,
            ple_pre[2],
            ple_conv_rows,
            ple_pre[3],
            ple_ids.astype(mx.int32),
            verify_hidden,
        )

        for kind, index, entry, gdn, ple in plan:
            if kind == "qsa":
                entry.kv.cache[2] = entry.kv.cache[2] - (
                    QWEN4_M4_VERIFY_WIDTH - keep
                )
                entry.kv.rollback_state[:] = [None, None, None]
                continue

            pre = snapshot_states[index]
            qkv, q, k, v, a, b = entry._mtplx_verify_rows
            next_conv = (
                ple_gdn_conv
                if ple is not None
                else mx.take(
                    mx.concatenate((pre[0], qkv), axis=1),
                    conv_indices,
                    axis=1,
                )
            )
            next_delta = replay_qwen4_m4_gdn_state(
                q,
                k,
                v,
                a,
                b,
                gdn.A_log,
                gdn.dt_bias,
                pre[1],
                gdn_keep_mask,
            )
            if ple is not None:

                def select_logical(slot, physical_m4):
                    selected = physical_m4
                    for width_index in range(2, -1, -1):
                        selected = mx.where(
                            accepted == width_index,
                            ple_logical_states[width_index][slot],
                            selected,
                        )
                    return selected

                entry[0] = select_logical(0, next_conv)
                entry[1] = select_logical(1, next_delta)
            else:
                entry[0] = next_conv
                entry[1] = next_delta
            entry._mtplx_verify_rows = None
            if ple is not None:
                entry[2] = select_logical(2, selected_ple_conv)
                entry[3] = select_logical(3, selected_ple_history)
                entry._mtplx_verify_ple = None
                entry._mtplx_verify_compiled_aux = None

        state_roots = legacy._fixed_m4_state_inputs(
            legacy._fixed_m4_dispatch["state_plan"]
        )
        mx.async_eval(selected_hidden, *state_roots)
        return selected_hidden

    committed_windows = []
    completion_tokens = []
    for window_index, seed in enumerate((392, 393)):
        ids = _ids(WINDOW, seed=seed)
        host_ids = _host_ids(ids)
        monolithic_snapshot = snapshot_untrimmable_cache_lazy(monolithic_cache)
        split_snapshot = snapshot_untrimmable_cache_lazy(split_cache)
        legacy_snapshot = snapshot_untrimmable_cache_lazy(legacy_cache)
        assert_tree_equal(split_snapshot.states, legacy_snapshot.states)
        monolithic_state_before = monolithic._fixed_m4_state_inputs(
            monolithic._fixed_m4_dispatch["state_plan"]
        )
        split_live_before = live_identity(split)

        mono_logits, mono_hidden, mono_captures = monolithic.forward_fixed_m4(
            ids,
            host_input_ids=host_ids,
            completion_tokens=completion_tokens,
            committed_count=len(completion_tokens),
            cache=monolithic_cache,
        )
        legacy_logits, legacy_hidden, legacy_captures = legacy.forward_fixed_m4(
            ids,
            host_input_ids=host_ids,
            completion_tokens=completion_tokens,
            committed_count=len(completion_tokens),
            cache=legacy_cache,
        )
        prefix = split.enqueue_fixed_m4_prefix(ids, cache=split_cache)
        probe_aux = probe_dispatch["prepare_aux"](
            ids,
            host_ids,
            completion_tokens,
            len(completion_tokens),
        )
        probe_outputs = tuple(
            probe_dispatch["fn"](ids, probe_aux, *monolithic_state_before)
        )
        probe_capture_base = 3 if probe_dispatch["returns_aux"] else 2
        probe_entry, probe_start, probe_count = probe_dispatch["capture_plan"][-1]
        assert probe_entry is probe_cache[0]
        assert probe_count == 1
        reference_prefix_hidden = probe_outputs[probe_capture_base + probe_start]
        split_logits, split_hidden, split_captures, split_result = (
            split.forward_fixed_m4_suffix(
                prefix,
                host_input_ids=host_ids,
                completion_tokens=completion_tokens,
                committed_count=len(completion_tokens),
                cache=split_cache,
            )
        )
        mx.eval(
            mono_logits,
            mono_hidden,
            legacy_logits,
            legacy_hidden,
            prefix.hidden,
            reference_prefix_hidden,
            split_logits,
            split_hidden,
        )
        assert live_identity(split) == split_live_before

        assert mono_captures == split_captures == {}
        assert legacy_captures == {}
        assert_array_equal("legacy_logits", legacy_logits, mono_logits)
        assert_array_equal("legacy_hidden", legacy_hidden, mono_hidden)
        assert_tree_equal(
            legacy._fixed_m4_state_inputs(legacy._fixed_m4_dispatch["state_plan"]),
            monolithic._fixed_m4_state_inputs(
                monolithic._fixed_m4_dispatch["state_plan"]
            ),
        )
        assert_array_equal("final_logits", split_logits, mono_logits)
        assert_array_equal("final_hidden", split_hidden, mono_hidden)
        assert_tree_equal(
            prefix.state_out,
            monolithic._fixed_m4_state_inputs(
                monolithic._fixed_m4_dispatch["state_plan"][:1]
            ),
        )
        mono_prefix_entry = monolithic._fixed_m4_dispatch["capture_plan"][0][0]
        assert_tree_equal(prefix.captures, mono_prefix_entry._mtplx_verify_rows)
        assert_tree_equal(
            split_result.state_out,
            monolithic._fixed_m4_state_inputs(
                monolithic._fixed_m4_dispatch["state_plan"][1:]
            ),
        )
        mono_suffix_captures = []
        mono_aux = None
        for entry, _start, count in monolithic._fixed_m4_dispatch["capture_plan"][1:]:
            mono_suffix_captures.extend(entry._mtplx_verify_rows)
            if count > 6:
                mono_suffix_captures.extend(entry._mtplx_verify_ple)
                mono_aux = entry._mtplx_verify_compiled_aux
        assert_tree_equal(split_result.captures, tuple(mono_suffix_captures))
        assert_tree_equal(split_result.returned_aux, mono_aux)
        legacy_capture_leaves = []
        for entry, _start, count in legacy._fixed_m4_dispatch["capture_plan"]:
            legacy_capture_leaves.extend(entry._mtplx_verify_rows)
            if count > 6:
                legacy_capture_leaves.extend(entry._mtplx_verify_ple)
        assert_tree_equal(
            (*prefix.captures, *split_result.captures),
            tuple(legacy_capture_leaves),
        )
        assert_array_equal(
            "layer0_hidden",
            prefix.hidden,
            reference_prefix_hidden,
        )

        mono_decision = production_decision(
            mono_logits, force_reject=window_index == 0
        )
        split_decision = production_decision(
            split_logits, force_reject=window_index == 0
        )
        for mono_value, split_value in zip(
            mono_decision[:6], split_decision[:6], strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(split_value), np.asarray(mono_value)
            )
        decision_summary = tuple(
            int(np.asarray(value).reshape(-1)[0]) for value in mono_decision[:6]
        )
        expected_kind = SELECTED_CORRECTION if window_index == 0 else SELECTED_BONUS
        expected_accepted = 0 if window_index == 0 else 3
        assert decision_summary[0] == expected_accepted
        assert decision_summary[3] == expected_kind
        assert decision_summary[4] == 1
        accepted_count = mx.array(mono_decision[0], dtype=mx.uint32)
        mono_prefix = graphbank.FixedM4Prefix(
            input_ids=ids,
            hidden=reference_prefix_hidden,
            captures=tuple(mono_prefix_entry._mtplx_verify_rows),
            state_in=(),
            state_out=monolithic._fixed_m4_state_inputs(
                monolithic._fixed_m4_dispatch["state_plan"][:1]
            ),
            outputs=(),
        )
        mono_split = graphbank.FixedM4Split(
            prefix=mono_prefix,
            returned_aux=mono_aux,
            captures=tuple(mono_suffix_captures),
            state_in=(),
            state_out=monolithic._fixed_m4_state_inputs(
                monolithic._fixed_m4_dispatch["state_plan"][1:]
            ),
            outputs=(),
        )
        mono_selected, mono_plan, mono_roots = monolithic._fixed_m4_dispatch[
            "device_commit"
        ](
            accepted_count,
            monolithic_snapshot.states,
            mono_hidden,
            mono_split,
        )
        split_selected = split.commit_fixed_m4_device_window(
            accepted_count,
            split_snapshot.states,
            split_hidden,
            split_result,
        )
        mx.eval(mono_selected, *mono_roots, split_selected)
        monolithic._publish_fixed_m4_selected_state(mono_plan)
        legacy_selected = legacy_live_device_commit(
            legacy_cache,
            accepted_count,
            legacy_snapshot.states,
            legacy_hidden,
        )
        mx.eval(legacy_selected)
        assert mx.array_equal(split_selected, mono_selected).item()
        assert_tree_equal(
            split._fixed_m4_state_inputs(split._fixed_m4_dispatch["state_plan"]),
            monolithic._fixed_m4_state_inputs(
                monolithic._fixed_m4_dispatch["state_plan"]
            ),
        )
        split_committed_state = split._fixed_m4_state_inputs(
            split._fixed_m4_dispatch["state_plan"]
        )
        legacy_committed_state = legacy._fixed_m4_state_inputs(
            legacy._fixed_m4_dispatch["state_plan"]
        )
        assert len(split_committed_state) == len(legacy_committed_state)
        state_labels = []
        for layer_index, kind, leaf_count in split._spec:
            state_labels.extend(
                f"layer{layer_index}.{kind}[{slot}]"
                for slot in range(leaf_count)
            )
        for leaf_index, (observed, expected) in enumerate(
            zip(split_committed_state, legacy_committed_state, strict=True)
        ):
            assert_array_equal(
                f"committed_state[{leaf_index}]={state_labels[leaf_index]}",
                observed,
                expected,
            )

        accepted = decision_summary[0]
        committed = host_ids[: accepted + 1]
        committed_windows.append(tuple(committed))
        completion_tokens.extend(committed)

    assert len(committed_windows) == 2


@pytest.mark.parametrize(
    ("boundary", "staged_builds"),
    (("both", 1), ("pre", 1), ("post", 0), ("none", 0)),
)
def test_fixed_m4_staged_aux_requires_pre_schedule(
    tm, monkeypatch, boundary, staged_builds
):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_BOUNDARY", boundary)
    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=30)
    tm(prefill, cache=cache)
    ple_index = next(
        i for i, layer in enumerate(tm.model.layers) if getattr(layer, "ple", None)
    )
    tm.model.layers[
        ple_index
    ].ple.ple_embedding.ngram_embedding._sidecar = _FakeSidecar()
    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    real_build = runtime.build_fixed_m4_compiled_verify_aux
    builds = []

    def build(_cache, prompt_ids):
        builds.append(True)
        return real_build(_cache, prompt_ids)

    runtime.build_fixed_m4_compiled_verify_aux = build
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16,
    )
    bank.install_fixed_m4(
        cache,
        prompt_ids=_host_ids(prefill),
        hidden_variant=None,
    )

    ids = _ids(WINDOW, seed=31)
    host_ids = _host_ids(ids)
    if boundary in ("both", "pre"):
        bank.prefetch_fixed_m4_primary(host_ids[0], [host_ids[0]], 0)
    logits, hidden, captures = bank.forward_fixed_m4(
        ids,
        host_input_ids=host_ids,
        completion_tokens=[host_ids[0]],
        committed_count=0,
        cache=cache,
    )
    mx.eval(logits, hidden)

    assert len(builds) == staged_builds
    assert captures == {}
    assert bank.stats["compiled_calls"] == 1


def test_fixed_m4_staged_sidecar_matches_materialized_route_across_windows(
    tm, monkeypatch
):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    prefill = _ids(PREFILL, seed=32)
    staged_cache = tm.make_cache()
    materialized_cache = tm.make_cache()
    tm(prefill, cache=staged_cache)
    tm(prefill, cache=materialized_cache)
    ple_index = next(
        i for i, layer in enumerate(tm.model.layers) if getattr(layer, "ple", None)
    )
    ple = tm.model.layers[ple_index].ple
    ple.ple_embedding.ngram_embedding._sidecar = _FakeSidecar()
    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_BOUNDARY", "both")

    staged_bank = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=WINDOW, request_max_tokens=16
    )
    prompt_ids = _host_ids(prefill)
    staged_bank.install_fixed_m4(
        staged_cache,
        prompt_ids=prompt_ids,
        hidden_variant=None,
    )
    materialized_bank = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=WINDOW, request_max_tokens=16
    )
    materialized_bank._build_fixed_m4_aux = None
    materialized_bank.install_fixed_m4(
        materialized_cache,
        prompt_ids=prompt_ids,
        hidden_variant=None,
    )

    completion_tokens = []
    for seed in (33, 34):
        ids = _ids(WINDOW, seed=seed)
        host_ids = _host_ids(ids)
        completion_tokens.append(host_ids[0])
        staged_bank.prefetch_fixed_m4_primary(
            host_ids[0],
            completion_tokens,
            len(completion_tokens) - 1,
        )
        staged_logits, staged_hidden, staged_captures = staged_bank.forward_fixed_m4(
            ids,
            host_input_ids=host_ids,
            completion_tokens=completion_tokens,
            committed_count=len(completion_tokens) - 1,
            cache=staged_cache,
        )
        reference_logits, reference_hidden, reference_captures = (
            materialized_bank.forward_fixed_m4(
                ids,
                host_input_ids=host_ids,
                completion_tokens=completion_tokens,
                committed_count=len(completion_tokens) - 1,
                cache=materialized_cache,
            )
        )
        mx.eval(staged_logits, staged_hidden, reference_logits, reference_hidden)

        assert mx.array_equal(staged_logits, reference_logits).item()
        assert mx.array_equal(staged_hidden, reference_hidden).item()
        assert mx.array_equal(
            staged_cache[ple_index][ple.NGRAM_IDX],
            materialized_cache[ple_index][ple.NGRAM_IDX],
        ).item()
        assert staged_captures == reference_captures == {}
        completion_tokens.extend(host_ids[1:])

    assert staged_bank.stats["compiled_calls"] == 2
    assert materialized_bank.stats["compiled_calls"] == 2


def test_installed_fixed_m4_routes_shorter_windows_to_family_capture(tm, monkeypatch):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)

    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=26)
    tm(prefill, cache=cache)
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16,
    )
    bank.install_fixed_m4(
        cache,
        prompt_ids=_host_ids(prefill),
        hidden_variant=None,
    )

    logits, hidden, captures = bank.forward_ar_capture(_ids(2, seed=27), cache=cache)
    mx.eval(logits, hidden)

    assert captures
    assert bank.stats["compiled_calls"] == 0
    assert bank.stats["fallback_calls"] == 0


def test_fixed_m4_bank_selection_is_not_limited_by_request_or_restore_size():
    from mtplx.generation import _qwen4_fixed_m4_compiled_verify_requested

    runtime = SimpleNamespace(qwen4_fixed_m4_compiled_verify=True)
    assert _qwen4_fixed_m4_compiled_verify_requested(
        runtime,
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=1024,
        cached_tokens=0,
    )
    assert not _qwen4_fixed_m4_compiled_verify_requested(
        runtime,
        verify_strategy="capture_commit",
        compiled_mode="on",
        max_tokens=1024,
        cached_tokens=0,
    )
    assert _qwen4_fixed_m4_compiled_verify_requested(
        runtime,
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=16384,
        cached_tokens=0,
    )
    assert _qwen4_fixed_m4_compiled_verify_requested(
        runtime,
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=1024,
        cached_tokens=512,
    )


def test_fixed_m4_capacity_grows_without_leaving_the_installed_lane(
    tm, monkeypatch
):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "18")

    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=40)
    tm(prefill, cache=cache)
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16_384,
        restored_tokens=4096,
    )
    bank.install_fixed_m4(
        cache,
        prompt_ids=_host_ids(prefill),
        hidden_variant=None,
    )
    assert bank._fixed_m4_dispatch["growth_tokens"] == 8
    qsa_index, qsa = next(
        (index, entry)
        for index, entry in enumerate(cache)
        if hasattr(entry, "raw_keys")
    )
    initial_capacity = int(qsa.raw_keys.shape[1])

    completion_tokens = []
    for ordinal, seed in enumerate((41, 42)):
        ids = _ids(WINDOW, seed=seed)
        host_ids = _host_ids(ids)
        completion_tokens.append(host_ids[0])
        snap = snapshot_untrimmable_cache_lazy(cache)
        logits, hidden, captures = bank.forward_fixed_m4(
            ids,
            host_input_ids=host_ids,
            completion_tokens=completion_tokens,
            committed_count=ordinal * KEEP,
            cache=cache,
        )
        mx.eval(logits, hidden)
        assert captures == {}
        assert tm.model.commit_verified_window(
            cache,
            snap.states,
            keep_tokens=KEEP,
            verified_tokens=WINDOW,
        )
        completion_tokens.extend(host_ids[1:KEEP])

    assert int(qsa.raw_keys.shape[1]) > initial_capacity
    assert bank.stats["fixed_m4_capacity_transitions"] == 1
    assert bank._fixed_m4_dispatch["growth_tokens"] == 16
    assert bank.stats["fixed_m4_route_transitions"] == 1
    assert bank.stats["compiled_calls"] == 2
    assert bank.stats["fallback_calls"] == 0

    bank.demote(cache)
    _keys, _values, raw, pooled = cache[qsa_index].state
    logical_end = PREFILL + 2 * KEEP
    assert int(raw.shape[1]) == logical_end
    assert int(pooled.shape[1]) == logical_end // cache[qsa_index].ratio


def test_fixed_m4_bank_fails_loud_instead_of_falling_back(tm, monkeypatch):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    bank = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=4, request_max_tokens=16
    )
    monkeypatch.setattr(bank, "_fallback_reason", lambda *args, **kwargs: "forced")

    with pytest.raises(RuntimeError, match="fixed-M4 verifier refused: forced"):
        bank.forward_ar_capture(_ids(WINDOW, seed=22), cache=tm.make_cache())
