from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest


class _FakeArray:
    def __init__(self, label: str, shape=(1,), dtype="float32", data=None) -> None:
        self.label = label
        self.shape = tuple(shape)
        self.dtype = dtype
        self.data = None if data is None else np.asarray(data).reshape(self.shape)

    def __array__(self, dtype=None):
        if self.data is None:
            raise TypeError(f"{self.label} has no materialized test data")
        return np.asarray(self.data, dtype=dtype)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        data = None if self.data is None else self.data.reshape(shape)
        return _FakeArray(f"reshape({self.label})", shape, self.dtype, data)

    def astype(self, dtype):
        data = None if self.data is None else self.data.astype(dtype)
        return _FakeArray(f"astype({self.label})", self.shape, str(dtype), data)

    def __sub__(self, other):
        if not isinstance(other, _FakeArray):
            other = _FakeArray(str(other), self.shape, self.dtype)
        data = (
            None
            if self.data is None or getattr(other, "data", None) is None
            else self.data - other.data
        )
        return _FakeArray(
            f"sub({self.label},{other.label})", self.shape, self.dtype, data
        )

    def __getitem__(self, index):
        if index is Ellipsis:
            data = None if self.data is None else self.data.view()
            return _FakeArray(f"view({self.label})", self.shape, self.dtype, data)
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = self.shape[0] if index.stop is None else index.stop
            return _FakeArray(
                f"{self.label}[{start}:{stop}]",
                (max(0, stop - start), *self.shape[1:]),
                self.dtype,
            )
        if isinstance(index, tuple):
            return _FakeArray(f"{self.label}[row]", (248320,), self.dtype)
        return _FakeArray(f"{self.label}[{index}]", self.shape[1:], self.dtype)


class _FakeTensorOffsetQSACache:
    def __init__(self, leaves: int = 5) -> None:
        self.offset = 0
        self.kv = SimpleNamespace(rollback_state=["old-start", "old-k", "old-v"])
        self.compile_state = [
            [_FakeArray(f"leaf-{index}", (1,), "float32") for index in range(leaves)]
        ]

    def trim(self, count):
        self.offset -= int(count)


class _FakeMX:
    float32 = "float32"
    int32 = "int32"
    uint32 = "uint32"
    uint64 = "uint64"

    def __init__(self) -> None:
        self.compile_calls = []
        self.eval_calls = []
        self.async_eval_calls = []

    def compile(self, function, *, inputs, outputs):
        self.compile_calls.append((function, inputs, outputs))
        return function

    def eval(self, *values):
        self.eval_calls.append(values)

    def async_eval(self, *values):
        self.async_eval_calls.append(values)

    @staticmethod
    def array(value, dtype=None):
        array = np.asarray(value)
        return _FakeArray("array", array.shape, str(dtype or array.dtype))

    @staticmethod
    def take(values, indices):
        return _FakeArray(f"take({values.label})", indices.shape, values.dtype)

    @staticmethod
    def exp(values):
        return _FakeArray(f"exp({values.label})", values.shape, values.dtype)

    @staticmethod
    def logsumexp(values, axis=-1, keepdims=False):
        shape = (*values.shape[:-1], 1) if keepdims else values.shape[:-1]
        return _FakeArray(f"logsumexp({values.label})", shape, values.dtype)

    @staticmethod
    def concatenate(values, axis=0):
        values = tuple(values)
        shape = list(values[0].shape)
        shape[axis] = sum(value.shape[axis] for value in values)
        data = (
            np.concatenate([value.data for value in values], axis=axis)
            if all(value.data is not None for value in values)
            else None
        )
        return _FakeArray(
            "concat(" + ",".join(value.label for value in values) + ")",
            tuple(shape),
            values[0].dtype,
            data,
        )


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls = []
        self.model = SimpleNamespace(language_model=SimpleNamespace())

    def draft_mtp(
        self,
        hidden,
        token,
        *,
        mtp_cache,
        return_hidden,
        mtp_hidden_variant,
        mtp_depth,
    ):
        del return_hidden, mtp_hidden_variant
        mtp_cache[0].offset += 1
        self.calls.append((mtp_depth, token.label))
        return (
            _FakeArray(f"logits-{mtp_depth}", (1, 1, 248320)),
            _FakeArray(f"hidden-{mtp_depth}", (1, 1, 10240)),
        )


def _sampler(**changes):
    from mtplx.sampling import SamplerConfig

    values = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    values.update(changes)
    return SamplerConfig(**values)


def _install_fakes(monkeypatch, *, promoted=1, failures=None, leaves=5):
    import mtplx.generation as generation
    import mtplx.pr391_mtp_handoff as mtp_handoff

    fake_mx = _FakeMX()

    def promote(cache, **kwargs):
        promote.kwargs = kwargs
        cache[:] = [_FakeTensorOffsetQSACache(leaves)]
        return promoted, {} if failures is None else failures

    promote.kwargs = {}
    monkeypatch.setattr(generation, "mx", fake_mx)
    monkeypatch.setattr(generation, "TensorOffsetQSACache", _FakeTensorOffsetQSACache)
    monkeypatch.setattr(generation, "promote_kv_cache_offsets", promote)
    monkeypatch.setattr(
        generation,
        "_deterministic_mlx_top_k_support",
        lambda row, _top_k: (
            _FakeArray(f"q-ids({row.label})", (20,), "uint32"),
            _FakeArray(f"q-values({row.label})", (20,), "float32"),
        ),
    )
    monkeypatch.setattr(
        generation,
        "_order_bounded_mlx_top_k_support",
        lambda ids, values: (ids, values),
    )
    monkeypatch.setattr(
        mtp_handoff,
        "bind_pr391_mtp_device_replay",
        lambda _cache, *, append_rows: append_rows,
    )
    return generation, fake_mx, promote


def test_d3_chain_materializes_only_packed_tokens(monkeypatch) -> None:
    generation, fake_mx, promote = _install_fakes(monkeypatch)
    runtime = _FakeRuntime()
    cache = [object()]
    choice_calls = []

    def choice(ids, values, probs, uniform_bits):
        level = len(choice_calls) + 1
        selected = _FakeArray(
            f"selected-{level}",
            (1,),
            "uint32",
            np.asarray([level], dtype=np.uint32),
        )
        choice_calls.append((ids, values, probs, uniform_bits, selected))
        return selected, ids, values, probs

    prebound = SimpleNamespace(selector=choice)

    core = generation._pr391_make_float32_d3_core(
        runtime,
        depth=3,
        mtp_hidden_variant="post_norm",
        mtp_cache=cache,
        draft_sampler=_sampler(),
        request_max_tokens=16384,
        prebound_kernel=prebound,
        preserve_paged=True,
    )

    # The retained exact D3 route deliberately keeps the already optimized
    # model/selector chain bare. An added outer compile changed the production
    # trajectory even when the target verifier remained monolithic.
    assert fake_mx.compile_calls == []
    assert promote.kwargs == {
        "reserve_tokens": 16388,
        "initial_reserve_tokens": 16388,
        "preserve_paged": True,
    }

    generation._pr391_prewarm_float32_d3_core(
        core,
        _FakeArray("warm-hidden", (1, 1, 10240)),
        _FakeArray("warm-primary", (1, 1), "uint32"),
        _FakeArray("uniform-bit-tape", (3,), "uint64"),
    )
    assert cache[0].offset == 0
    assert len(fake_mx.eval_calls) == 1
    runtime.calls.clear()
    choice_calls.clear()
    fake_mx.eval_calls.clear()

    result = generation._pr391_run_float32_d3_core(
        core,
        _FakeArray("live-hidden", (1, 1, 10240)),
        17,
        _FakeArray("uniform-bit-tape", (3,), "uint64"),
    )

    assert [depth for depth, _token in runtime.calls] == [1, 2, 3]
    assert runtime.calls[0][1] == "array"
    assert runtime.calls[1][1] == "reshape(selected-1)"
    assert runtime.calls[2][1] == "reshape(selected-2)"
    assert [call[3].label for call in choice_calls] == [
        "uniform-bit-tape[0:1]",
        "uniform-bit-tape[1:2]",
        "uniform-bit-tape[2:3]",
    ]
    assert fake_mx.eval_calls == [(result[0],)]
    assert cache[0].kv.rollback_state == [None, None, None]
    assert len(result) == 4
    assert result[0].shape == (1, 3)
    assert result[1].shape == (3, 20)
    assert result[2].shape == (3, 20)
    assert result[3].shape == (3, 20)
    assert "selected-1" in result[0].label
    assert "selected-2" in result[0].label
    assert "selected-3" in result[0].label
    assert all(call[0].label in result[1].label for call in choice_calls)
    assert all(call[1].label in result[2].label for call in choice_calls)
    assert all(call[2].label in result[3].label for call in choice_calls)



def test_d3_packed_decode_materializes_tokens_without_q_arrays(monkeypatch) -> None:
    import mtplx.generation as generation

    fake_mx = _FakeMX()
    monkeypatch.setattr(generation, "mx", fake_mx)
    result = (
        np.asarray([[11, 12, 13]], dtype=np.uint32),
        _FakeArray("device-q-ids", (3, 20), "uint32"),
        _FakeArray("device-q-values", (3, 20), "float32"),
        _FakeArray("device-q-probs", (3, 20), "float32"),
    )

    tokens = generation._pr391_decode_float32_d3_tokens(result)

    assert fake_mx.eval_calls == []
    assert tokens == [11, 12, 13]


@pytest.mark.parametrize(
    ("changes", "depth", "prebound", "message"),
    [
        ({}, 2, lambda *_args: None, "depth=3"),
        ({"temperature": 0.9}, 3, lambda *_args: None, "temperature=1"),
        ({"top_k": 19}, 3, lambda *_args: None, "top_k=20"),
        ({"top_p": 1.0}, 3, lambda *_args: None, "top_p=0.95"),
        ({}, 3, None, "selector"),
    ],
)
def test_construction_rejects_nonfixed_contract_before_promotion(
    monkeypatch, changes, depth, prebound, message
) -> None:
    generation, fake_mx, promote = _install_fakes(monkeypatch)

    with pytest.raises((TypeError, ValueError), match=message):
        generation._pr391_make_float32_d3_core(
            _FakeRuntime(),
            depth=depth,
            mtp_hidden_variant="post_norm",
            mtp_cache=[object()],
            draft_sampler=_sampler(**changes),
            request_max_tokens=1024,
            prebound_kernel=(
                None if prebound is None else SimpleNamespace(selector=prebound)
            ),
            preserve_paged=True,
        )

    assert promote.kwargs == {}
    assert fake_mx.compile_calls == []


@pytest.mark.parametrize(
    ("promoted", "failures", "leaves", "message"),
    [
        (0, None, 5, "exactly one"),
        (1, {"auxiliary_qsa_state": 1}, 5, "promotion failures"),
        (1, None, 4, "five tensor leaves"),
        (1, None, 6, "five tensor leaves"),
    ],
)
def test_promotion_fails_closed_before_compile(
    monkeypatch, promoted, failures, leaves, message
) -> None:
    generation, fake_mx, _promote = _install_fakes(
        monkeypatch,
        promoted=promoted,
        failures=failures,
        leaves=leaves,
    )

    with pytest.raises(RuntimeError, match=message):
        generation._pr391_make_float32_d3_core(
            _FakeRuntime(),
            depth=3,
            mtp_hidden_variant="post_norm",
            mtp_cache=[object()],
            draft_sampler=_sampler(),
            request_max_tokens=1024,
            prebound_kernel=SimpleNamespace(selector=lambda *_args: None),
            preserve_paged=True,
        )

    assert fake_mx.compile_calls == []


def test_stage_a_source_has_one_packed_eval_and_no_rng_fallback_or_env() -> None:
    import mtplx.generation as generation

    make_source = inspect.getsource(generation._pr391_make_float32_d3_core)
    run_source = inspect.getsource(generation._pr391_run_float32_d3_core)
    decode_source = inspect.getsource(generation._pr391_decode_float32_d3_tokens)
    make_tree = ast.parse(make_source)

    assert "mx.random" not in make_source
    assert "os.environ" not in make_source
    assert "fallback" not in make_source.lower()
    assert "_eval(" not in make_source + run_source
    assert run_source.count("mx.eval(") == 1
    assert "mx.eval(result[0])" in run_source
    assert "_device_draft_q_arrays(" not in make_source
    assert "_deterministic_mlx_top_k_support(" in make_source
    assert "_order_bounded_mlx_top_k_support(" in make_source
    assert "selected, raw_ids, raw_values, raw_probs = selector(" in make_source
    assert "uniform_bit_rows[level - 1 : level]" in make_source
    assert decode_source.count("np.asarray(") == 1
    assert "result[0]" in decode_source
    assert "result[1]" not in decode_source
    assert "result[2]" not in decode_source
    assert "result[3]" not in decode_source
    assert "mx.eval(" not in decode_source
    assert ".item()" not in decode_source
    assert not any(isinstance(node, ast.Try) for node in ast.walk(make_tree))


def test_d3_packed_decode_precedes_host_built_fixed_m4_input() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    run = source.index("_pr391_run_float32_d3_core(")
    decode = source.index("_pr391_decode_float32_d3_tokens(")
    fixed_m4 = source.index("compiled_verify_bank.forward_fixed_m4(")
    verifier = source.index("_pr391_apply_softfloat64_decision(")

    assert run < decode < fixed_m4 < verifier
    assert "verify_input = [int(primary), *draft_tokens]" in source
    assert "verify_input_array = mx.array([verify_input])" in source
    assert "_pr391_verify_input_array" not in source


def test_pr391_uses_only_the_retained_monolithic_fixed_m4_route() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    assert "install_fixed_m4_split(" not in source
    assert "enqueue_fixed_m4_prefix(" not in source
    assert "forward_fixed_m4_suffix(" not in source
    assert source.count("forward_fixed_m4(") == 1


def test_primary_ple_prefetch_is_started_for_initial_and_carried_d3() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    initial_primary = source.index("prefetch_fixed_m4_primary(")
    initial_d3 = source.index("_pr391_run_float32_d3_core(", initial_primary)
    queued_d3 = source.index("_pr391_queue_device_canonical_d3(")
    carried_primary = source.index("prefetch_fixed_m4_primary(", queued_d3)

    assert initial_primary < initial_d3
    assert queued_d3 < carried_primary


def test_stage_a_does_not_modify_legacy_qsa_or_d2_guards() -> None:
    import mtplx.generation as generation

    d2_source = inspect.getsource(generation._make_device_d2_draft_core)
    legacy_source = inspect.getsource(generation._make_device_draft_core_inner)
    new_source = inspect.getsource(generation._pr391_make_float32_d3_core)

    assert "qsa_mtp_outer_device_core_supported(mtp_cache)" in d2_source
    assert "mx.random.uniform" in legacy_source
    assert "_make_device_d2_draft_core" not in new_source
    assert "qsa_mtp_outer_device_core_supported" not in new_source


def test_softfloat_target_candidates_stay_raw_lazy_and_deterministic() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation._pr391_float32_target_support)

    assert "_deterministic_mlx_top_k_support" in source
    assert "_order_bounded_mlx_top_k_support" in source
    assert "mx.logsumexp" in source
    assert "cumulative_before" not in source
    assert "id_order" not in source
    assert "return (" in source
    assert "np.asarray" not in source
    assert "mx.eval" not in source


def test_softfloat_decision_uses_one_production_wiring_helper() -> None:
    import mtplx.generation as generation

    helper = inspect.getsource(generation._pr391_apply_softfloat64_decision)
    generate = inspect.getsource(generation.generate_mtpk)

    assert "return tuple(" in helper
    assert "verifier_kernel(" in helper
    assert "draft_result[0].reshape(_PR391_FLOAT32_D3_DEPTH)" in helper
    assert "*target_support" in helper
    assert "_pr391_apply_softfloat64_decision(" in generate


def test_softfloat_target_candidates_preserve_ranked_raw_rows(
    monkeypatch,
) -> None:
    import mtplx.generation as generation

    class FakeMX:
        float32 = np.float32
        uint32 = np.uint32
        exp = staticmethod(np.exp)
        cumsum = staticmethod(np.cumsum)
        where = staticmethod(np.where)
        zeros_like = staticmethod(np.zeros_like)
        sum = staticmethod(np.sum)
        argsort = staticmethod(np.argsort)
        take_along_axis = staticmethod(np.take_along_axis)

        @staticmethod
        def logsumexp(rows, *, axis, keepdims):
            return np.zeros(
                np.sum(rows, axis=axis, keepdims=keepdims).shape,
                dtype=np.float32,
            )

    ids = np.array(
        [[9, 2, 7, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 8, 6, 5, 4, 3, 1, 0]],
        dtype=np.uint32,
    )
    values = np.full((1, 20), np.float32(-100.0), dtype=np.float32)
    values[0, :3] = np.log(np.array([0.6, 0.3, 0.1], dtype=np.float32))
    monkeypatch.setattr(generation, "mx", FakeMX())
    monkeypatch.setattr(
        generation,
        "_deterministic_mlx_top_k_support",
        lambda _rows, _top_k: (ids, values),
    )
    monkeypatch.setattr(
        generation,
        "_order_bounded_mlx_top_k_support",
        lambda support_ids, support_values: (support_ids, support_values),
    )

    observed_ids, observed_values, observed_probs = generation._pr391_float32_target_support(
        np.zeros((1, 20), dtype=np.float32)
    )

    np.testing.assert_array_equal(observed_ids, ids)
    np.testing.assert_array_equal(observed_values, values)
    assert observed_probs[0, 0] == pytest.approx(0.6)
    assert observed_probs[0, 1] == pytest.approx(0.3)
    assert observed_probs[0, 2] == pytest.approx(0.1)


def test_verifier_decision_decode_uses_one_sync_and_reported_tape_advance(
    monkeypatch,
) -> None:
    import mtplx.generation as generation
    from mtplx.pcg64_tape import PCG64UniformTape

    fake_mx = _FakeMX()
    monkeypatch.setattr(generation, "mx", fake_mx)
    tape = PCG64UniformTape.build(np.random.default_rng(391), max_output_tokens=4)
    reservation = tape.peek_device_choices(4)
    result = (
        np.array([1], dtype=np.uint32),
        np.array([1], dtype=np.int32),
        np.array([17], dtype=np.uint32),
        np.array([1], dtype=np.uint32),
        np.array([1], dtype=np.uint32),
        np.array([3], dtype=np.uint32),
        np.array([1.0, 0.25, 0.0], dtype=np.float64).view(np.uint64),
    )

    decoded = generation._pr391_decode_float32_verifier_decision(
        result,
        uniform_tape=tape,
        reservation=reservation,
    )

    assert len(fake_mx.eval_calls) == 1
    assert fake_mx.eval_calls[0] == result
    assert tape.cursor == 3
    assert decoded == (1, 1, 17, 1, True, 3, [1.0, 0.25, 0.0])


def test_stage_b_route_reserves_joint_d3_and_keeps_tail_on_shared_tape() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    claim_source = inspect.getsource(generation._pr391_claim_float32_d3_request_route)

    assert "_pr391_route.uniform_tape" in source
    assert "reserve_device_choices(3)" in source
    assert "cycle_depth == _PR391_FLOAT32_D3_DEPTH" in source
    assert "if _pr391_joint_result is not None:" in source
    assert "reservation.offset : reservation.offset + 3" in source
    assert source.index("_pr391_prewarm_float32_d3_core(") < source.index(
        "pre_first_token_setup_s = time.perf_counter()"
    )
    prewarm_source = inspect.getsource(generation._pr391_prewarm_float32_d3_core)
    assert "reserve_device_choices" not in prewarm_source
    assert "_rollback_mtp_cache" in prewarm_source
    assert "except" not in claim_source
    assert "os.environ" not in claim_source
    assert "qsa_mtp_outer_device_core_supported" not in claim_source


def test_active_pr391_route_uses_device_replay_not_host_selected_reference() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    verifier = source.index("verifier_result = _pr391_apply_softfloat64_decision(")
    target_commit = source.index(
        "compiled_verify_bank.commit_fixed_m4_device_window(", verifier
    )
    replay = source.index("_pr391_queue_device_verifier_mtp_replay(", target_commit)
    decode = source.index("_pr391_decode_float32_verifier_decision(", replay)

    assert verifier < target_commit < replay < decode
    assert "_pr391_fixed_m4_split" not in source[target_commit:replay]
    replay_call = source[replay:decode]
    assert "accepted_count=verifier_result[0]," in replay_call
    assert "verify_hidden=verify_hidden," in replay_call
    assert "draft_token_ids=_pr391_joint_result[0]," in replay_call
    assert "_pr391_mtp_handoff_owns_cycle = True" in replay_call
    assert "_pr391_queue_verifier_mtp_replay(" not in source


def test_device_mtp_replay_is_queued_before_host_decision_decode() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    verifier = source.index("verifier_result = _pr391_apply_softfloat64_decision(")
    replay = source.index("_pr391_queue_device_verifier_mtp_replay(", verifier)
    decode = source.index("_pr391_decode_float32_verifier_decision(", replay)

    assert verifier < replay < decode


def test_device_next_d3_is_queued_before_host_decision_decode() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    verifier = source.index("verifier_result = _pr391_apply_softfloat64_decision(")
    replay = source.index("_pr391_queue_device_verifier_mtp_replay(", verifier)
    queued = source.index("_pr391_queue_device_canonical_d3(", replay)
    decode = source.index("_pr391_decode_float32_verifier_decision(", queued)

    assert verifier < replay < queued < decode
    call = source[queued:decode]
    assert "hidden=_pr391_selected_target_hidden" in call
    assert "primary=verifier_result[2]" in call
    assert "descriptor_offset=(" in call
    assert "verifier_result[5]" in call


def test_device_replay_flows_into_dependent_d3_without_intermediate_sync() -> None:
    import mtplx.generation as generation

    replay_source = inspect.getsource(
        generation._pr391_queue_device_verifier_mtp_replay
    )
    assert 'core["device_replay"](' in replay_source
    assert "accepted_count," in replay_source
    assert "verify_hidden[:, :_PR391_FLOAT32_D3_DEPTH, :]" in replay_source
    assert "draft_token_ids," in replay_source
    assert "mx.eval" not in replay_source
    assert "mx.async_eval" not in replay_source

    source = inspect.getsource(generation.generate_mtpk)
    replay = source.index("_pr391_queue_device_verifier_mtp_replay(")
    queued = source.index("_pr391_queue_device_canonical_d3(", replay)
    decode = source.index("_pr391_decode_float32_verifier_decision(", queued)
    assert replay < queued < decode
    assert "mx.eval" not in source[replay:decode]

    queue_source = inspect.getsource(generation._pr391_queue_device_canonical_d3)
    finish_source = inspect.getsource(generation._pr391_finish_canonical_d3_queue)
    assert 'core["fn"](' in queue_source
    assert "_pr391_finish_canonical_d3_queue(core, result)" in queue_source
    assert "mx.async_eval(*result, *entry.state_leaves)" in finish_source


def test_canonical_d3_queue_keeps_one_bank_owner_and_logically_rewinds(monkeypatch) -> None:
    import mtplx.generation as generation

    fake_mx = _FakeMX()
    monkeypatch.setattr(generation, "mx", fake_mx)
    calls = []

    class Entry:
        def __init__(self):
            self.kv = SimpleNamespace(
                cache=[
                    "replay-k",
                    "replay-v",
                    _FakeArray("replay-offset", (), "int32"),
                ],
                rollback_state=["replay-start", "replay-k-tail", "replay-v-tail"],
            )
            self.aux = ["replay-raw", "replay-pooled"]

        @property
        def state_leaves(self):
            return [*self.kv.cache, *self.aux]

    entry = Entry()
    def compiled(hidden, primary, uniform_bits):
        calls.append((hidden, primary, uniform_bits))
        entry.kv.cache[:] = [
            "future-k",
            "future-v",
            _FakeArray("future-offset", (), "int32"),
        ]
        entry.aux[:] = ["future-raw", "future-pooled"]
        return (
            _FakeArray("tokens", (1, 3), "uint32"),
            _FakeArray("ids", (3, 20), "uint32"),
            _FakeArray("values", (3, 20), "float32"),
            _FakeArray("probs", (3, 20), "float32"),
        )

    hidden = _FakeArray("committed-hidden", (1, 1, 10240))
    uniform_bits = _FakeArray("uniform-bit-tape", (64,), "uint64")
    result, future_offset = generation._pr391_queue_canonical_d3(
        {
            "fn": compiled,
            "cache": [entry],
            "state_tree": [entry.kv.cache, entry.aux],
        },
        hidden=hidden,
        primary=17,
        uniform_bit_rows=uniform_bits,
        descriptor_offset=9,
    )

    assert calls[0][0] is hidden
    assert calls[0][1].shape == (1, 1)
    assert calls[0][2].label == "uniform-bit-tape[9:12]"
    assert entry.state_leaves[:2] == ["future-k", "future-v"]
    assert entry.state_leaves[2].label == "sub(future-offset,3)"
    assert entry.state_leaves[3:] == ["future-raw", "future-pooled"]
    assert entry.kv.rollback_state == [None, None, None]
    assert future_offset.label == "future-offset"
    assert fake_mx.async_eval_calls == [(*result, *entry.state_leaves)]
    queue_source = inspect.getsource(generation._pr391_queue_canonical_d3)
    assert "_pr391_install_float32_d3_state(" not in queue_source


def test_host_canonical_d3_is_not_retained_as_a_carried_path() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    device_queue = source.index("_pr391_queue_device_canonical_d3(")
    decode = source.index("_pr391_decode_float32_verifier_decision(")
    carried = source.index("_pr391_carried_d3 = {", decode)

    assert device_queue < decode < carried
    assert "_pr391_device_prequeued_d3 is not None" in source[decode:carried]
    assert "_pr391_queue_canonical_d3(" not in source
    assert "_pr391_queue_verifier_mtp_replay(" not in source


def test_context_copy_drops_carried_d3_without_live_cache_repair() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    capture = source.index("# ---- context-copy round:")
    batched = source.index("# ---- context-copy block rounds, BATCHED lane", capture)
    draft = source.index("draft_hidden = hidden", batched)

    capture_branch = source[capture:batched]
    batched_branch = source[batched:draft]
    for branch in (capture_branch, batched_branch):
        assert "_pr391_abandon_canonical_d3(" not in branch
        assert "_rollback_mtp_cache(" not in branch
        assert "discard_fixed_m4_prefix(" not in branch
        assert "_pr391_carried_d3 = None" in branch
        assert "_pr391_queue_device_verifier_mtp_replay(" not in branch
        assert "_pr391_queue_device_canonical_d3(" not in branch
        assert "forward_fixed_m4(" not in branch
        assert "verifier_kernel(" not in branch
        assert "continue" in branch

    assert "compiled_verify_bank.forward_ar_capture(" in capture_branch
    assert "rt.forward_ar_capture(" in capture_branch
    assert "rt.forward_ar(" in batched_branch


def test_terminal_lookahead_drops_the_isolated_carried_d3() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    decode = source.index("_pr391_decode_float32_verifier_decision(")
    carried = source.index("_pr391_carried_d3 = {", decode)
    final = source[source.rindex("if _pr391_carried_d3 is not None:") :]

    assert "fixed_m4_prefix" not in source[decode:carried]
    assert "_pr391_carried_d3 = None" in final
    assert "discard_fixed_m4_prefix(" not in final


def test_d3_core_is_installed_only_after_fixed_m4_admission() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)

    install = source.index("compiled_verify_bank.install_fixed_m4(")
    admission = source.index("if _pr391_route is not None and (")
    construct = source.index("_pr391_device_core = (")
    prewarm = source.index("_pr391_prewarm_float32_d3_core(")
    timer = source.index("pre_first_token_setup_s = time.perf_counter()")
    assert install < admission < construct < prewarm < timer


def _claim_route_kwargs(route, *, sampler, draft_sampler):
    return {
        "seed": route.expected_seed,
        "max_tokens": route.max_output_tokens,
        "sampler": sampler,
        "draft_sampler": draft_sampler,
        "speculative_depth": 3,
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "draft_core": "stock",
        "constraint": None,
        "adaptive_policy": None,
        "mtp_corrector": None,
        "adaptive_width_policy": None,
        "mtp_position_mode": "cache",
        "draft_margin_threshold": None,
        "online_hidden_corrector_alpha": 0.0,
        "online_correction_cache": False,
        "prompt_correction_cache": False,
        "adapter_ensemble_q": False,
        "mtp_topk_reranker": None,
        "loop_guard": False,
        "thinking_guard": None,
        "late_depth_switch_after": 0,
    }


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"mtp_corrector": object()}, "unsupported feature"),
        ({"adaptive_width_policy": object()}, "unsupported feature"),
        ({"mtp_position_mode": "absolute"}, "position mode"),
        ({"late_depth_switch_after": 1}, "late-depth switching"),
    ],
)
def test_request_claim_rejects_unrepresented_d3_semantics(
    monkeypatch, changed, message
) -> None:
    import mtplx.generation as generation

    sampler = _sampler()
    draft_sampler = _sampler()
    route = SimpleNamespace(
        claimed=False,
        expected_seed=391,
        max_output_tokens=1024,
        sampler=sampler,
        draft_sampler=draft_sampler,
        preserve_paged=True,
    )
    monkeypatch.setattr(generation, "_pr391_float32_d3_request_route", route)
    kwargs = _claim_route_kwargs(
        route,
        sampler=sampler,
        draft_sampler=draft_sampler,
    )
    kwargs.update(changed)

    with pytest.raises(RuntimeError, match=message):
        generation._pr391_claim_float32_d3_request_route(**kwargs)

    assert route.claimed is False


def test_request_claim_receives_resolved_position_and_all_semantic_blockers() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    resolve = source.index("mtp_position_mode = _resolve_runtime_mtp_position_mode(rt)")
    claim = source.index("_pr391_claim_float32_d3_request_route(")
    construct = source.index("_pr391_device_core = (")

    assert resolve < claim < construct
    claim_call = source[claim:source.index("    )", claim)]
    assert "mtp_corrector=mtp_corrector" in claim_call
    assert "adaptive_width_policy=adaptive_width_policy" in claim_call
    assert "mtp_position_mode=mtp_position_mode" in claim_call


@pytest.mark.parametrize(
    ("donate", "boundary"),
    [(True, "pre"), (False, "both"), (False, "post")],
)
def test_fixed_m4_async_enqueue_admission_accepts_proven_dispatches(
    donate, boundary
) -> None:
    import mtplx.generation as generation

    bank = SimpleNamespace(
        _fixed_m4_dispatch={
            "donate": donate,
            "boundary": boundary,
            "device_commit": lambda *_args: None,
            "prefetch_aux": lambda *_args: None,
            "prefetch_window_aux": lambda *_args: None,
        }
    )

    generation._pr391_require_fixed_m4_async_enqueue(bank)


@pytest.mark.parametrize("boundary", ["pre", "off", None])
def test_fixed_m4_async_enqueue_admission_rejects_synchronous_dispatch(
    boundary,
) -> None:
    import mtplx.generation as generation

    bank = SimpleNamespace(
        _fixed_m4_dispatch={"donate": False, "boundary": boundary}
    )

    with pytest.raises(RuntimeError, match="async-enqueue"):
        generation._pr391_require_fixed_m4_async_enqueue(bank)


def test_async_enqueue_is_proved_after_fixed_m4_install_before_d3_prewarm() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    install = source.index("compiled_verify_bank.install_fixed_m4(")
    enqueue_admission = source.index("_pr391_require_fixed_m4_async_enqueue(")
    construct = source.index("_pr391_device_core = (")
    prewarm = source.index("_pr391_prewarm_float32_d3_core(")

    assert install < enqueue_admission < construct < prewarm


def test_promoted_d3_history_uses_exact_bank_demotion() -> None:
    import mtplx.generation as generation

    promoted_cache = [object()]
    core = {"cache": promoted_cache}

    class FakeBank:
        def __init__(self):
            self.calls = []

        def demote(self, cache):
            self.calls.append(cache)
            cache[:] = ["stock-qsa"]
            return 1

    bank = FakeBank()

    generation._pr391_demote_float32_d3_core(core, bank)

    assert bank.calls == [promoted_cache]
    assert promoted_cache == ["stock-qsa"]


def test_promoted_d3_history_is_demoted_before_final_state_exposure() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    demote = source.index("_pr391_demote_float32_d3_core(")
    final_state = source.index("final_state = GenerationFinalState(", demote)

    assert demote < final_state


def test_terminal_d3_d2_d1_cycles_drop_joint_state_and_keep_generic_routes() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation.generate_mtpk)
    depth = source.index(
        "cycle_depth = min(planned_depth, max_tokens - len(tokens))"
    )
    reset = source.index("_pr391_joint_result: tuple[Any, ...] | None = None", depth)
    d3 = source.index("cycle_depth == _PR391_FLOAT32_D3_DEPTH", reset)
    host_tail = source.index(
        "0 if (used_device_core or _greedy_chain_used) else cycle_depth", d3
    )
    generic_verify = source.index("elif compiled_verify_bank is not None:", host_tail)

    assert depth < reset < d3 < host_tail < generic_verify
    assert "and verified_token_count == 4" in source[d3:generic_verify]
