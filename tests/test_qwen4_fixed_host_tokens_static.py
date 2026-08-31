"""Pure-Python gates for fixed-M4 host-owned n-gram inputs."""

from __future__ import annotations

import ast
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_function(path: Path, name: str, namespace=None):
    tree = ast.parse(path.read_text())
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    assert function is not None, f"missing {name} in {path}"
    namespace = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _load_method(path: Path, class_name: str, name: str, namespace=None):
    tree = ast.parse(path.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class _Done:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _ImmediatePool:
    def __init__(self):
        self.submissions = 0

    def submit(self, function, *args):
        self.submissions += 1
        return _Done(function(*args))


def test_fixed_m4_previous_tokens_come_from_committed_host_ledger():
    previous = _load_function(
        ROOT / "mtplx/qwen4_fixed_verify.py", "_fixed_m4_previous_tokens"
    )
    prompt_tail = (101, 102)

    assert previous(prompt_tail, [201], 0) == prompt_tail
    assert previous(prompt_tail, [201, 202], 1) == (102, 201)
    assert previous(prompt_tail, [201, 202, 203], 2) == (201, 202)
    # A deferred correction or bonus is present in the emitted ledger but is
    # the current primary, so committed_count excludes it from prior history.
    assert previous(prompt_tail, [201, 202, 999], 2) == (201, 202)
    # A correction already re-forwarded into the target cache remains inside
    # the committed prefix when the next fresh primary is appended.
    assert previous(prompt_tail, [201, 202, 999, 301], 3) == (202, 999)


def test_fixed_m4_host_entrypoint_is_explicitly_plumbed():
    graphbank = ast.parse((ROOT / "mtplx/graphbank.py").read_text())
    bank = next(
        node
        for node in ast.walk(graphbank)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    entrypoint = next(
        (
            node
            for node in bank.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward_fixed_m4_suffix"
        ),
        None,
    )
    assert entrypoint is not None
    arguments = {
        argument.arg for argument in (*entrypoint.args.args, *entrypoint.args.kwonlyargs)
    }
    assert {
        "prefix",
        "host_input_ids",
        "completion_tokens",
        "committed_count",
        "cache",
    } <= arguments

    generation = ast.parse((ROOT / "mtplx/generation.py").read_text())
    calls = [
        node
        for node in ast.walk(generation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward_fixed_m4_suffix"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert {
        "host_input_ids",
        "completion_tokens",
        "committed_count",
        "cache",
    } <= keywords

    # Rejected full-window PLE read-ahead remains available as a
    # construction-time experiment, but must not add work to the retained
    # primary-only measured route.
    prefetch_calls = [
        node
        for node in ast.walk(generation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prefetch_fixed_m4_window"
    ]
    assert prefetch_calls == []

    enabled_window_prefetch = [
        node
        for node in ast.walk(generation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enable_qwen4_fixed_verify_owned_window_prefetch"
    ]
    assert enabled_window_prefetch == []


def test_fixed_m4_split_partitions_layer0_and_roots_prefix_until_commit():
    graphbank_source = (ROOT / "mtplx/graphbank.py").read_text()
    graphbank = ast.parse(graphbank_source)
    bank = next(
        node
        for node in ast.walk(graphbank)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in bank.body
        if isinstance(node, ast.FunctionDef)
    }
    install = methods["install_fixed_m4_split"]
    enqueue = methods["enqueue_fixed_m4_prefix"]
    suffix = methods["forward_fixed_m4_suffix"]
    commit = methods["commit_fixed_m4_device_window"]

    assert install.count("mx.compile(") == 2
    assert "prefix_state_leaves != 2" in install
    assert "prefix_capture_leaves != 6" in install
    assert "!= 132" in install
    assert "!= 213" in install
    assert "mx.async_eval(*outputs)" in enqueue
    assert "donate" not in enqueue
    assert "mx.async_eval(*prefix.outputs, *outputs)" in suffix
    assert "_held_fixed_m4_split_refs.append" in enqueue
    assert "_held_fixed_m4_split_refs.append" in suffix
    assert commit.count("self._held_fixed_m4_split_refs.clear()") == 2
    assert "prefix_entry.cache[slot] = leaf" not in suffix
    assert "entry.kv.cache[0] = state_out[state_pos]" not in suffix
    assert "entry._mtplx_verify_rows =" not in suffix
    assert "entry._mtplx_verify_ple =" not in suffix
    assert "FixedM4Split(" in suffix
    assert "except Exception" in suffix
    assert "self.discard_fixed_m4_prefix(prefix)" in suffix
    assert "split: FixedM4Split" in commit
    assert "self._publish_fixed_m4_selected_state(commit_plan)" in commit
    assert commit.index("mx.async_eval") < commit.index(
        "self._publish_fixed_m4_selected_state(commit_plan)"
    )


@pytest.mark.parametrize("failure", ["selection", "enqueue"])
def test_fixed_m4_split_commit_failure_keeps_live_identity(failure):
    path = ROOT / "mtplx/graphbank.py"
    live_state = object()
    live_capture = object()
    published = []

    class FakeMX:
        @staticmethod
        def async_eval(*_values):
            if failure == "enqueue":
                raise RuntimeError("injected enqueue failure")

    def device_commit(*_args):
        if failure == "selection":
            raise RuntimeError("injected selection failure")
        return "selected-hidden", ("selected-state",), ("selected-root",)

    fake = SimpleNamespace(
        _fixed_m4_dispatch={"device_commit": device_commit},
        _held_fixed_m4_split_refs=[object()],
        live_state=live_state,
        live_capture=live_capture,
    )

    def publish(plan):
        published.append(plan)
        fake.live_state = plan[0]
        fake.live_capture = plan[0]

    fake._publish_fixed_m4_selected_state = publish
    commit = _load_method(
        path,
        "CompiledVerifyBank",
        "commit_fixed_m4_device_window",
        {"mx": FakeMX, "FixedM4Split": object},
    )

    with pytest.raises(RuntimeError, match="injected"):
        commit(fake, "accepted", "snapshots", "hidden", "split")

    assert fake.live_state is live_state
    assert fake.live_capture is live_capture
    assert published == []
    assert fake._held_fixed_m4_split_refs == []


@pytest.mark.parametrize(
    "class_name",
    ["_FixedM4SidecarAux", "_FixedM4ExperimentalSidecarAux"],
)
def test_fixed_m4_sidecar_optional_primary_prefetch_defaults_to_empty(class_name):
    import mtplx.qwen4_fixed_verify as fixed_verify

    installed = []

    def rows(ids, _previous):
        return np.asarray(ids, dtype=np.int64), np.asarray(ids, dtype=np.int64)

    common = {
        "prompt_tail": (0, 0),
        "rows": rows,
        "gather": lambda gathered: np.asarray(gathered).reshape(1, 4, 1),
        "submit_warm": lambda _rows: (),
        "install_owned_rows": lambda pending: installed.append(tuple(pending)),
    }
    if class_name == "_FixedM4SidecarAux":
        auxiliary = fixed_verify._FixedM4SidecarAux(output_dim=1, **common)
    else:
        auxiliary = fixed_verify._FixedM4ExperimentalSidecarAux(
            prefetch_window_rows=lambda *_args: None,
            resolve_window_rows=lambda *_args: np.arange(4, dtype=np.int64),
            **common,
        )

    auxiliary(None, [1, 2, 3, 4], [], 0)

    assert installed == [()]

    verifier = ast.parse((ROOT / "mtplx/qwen4_fixed_verify.py").read_text())
    verifier_methods = {
        node.name: ast.unparse(node)
        for node in verifier.body
        if isinstance(node, ast.FunctionDef)
    }
    prefix = verifier_methods["_forward_fixed_m4_prefix"]
    suffix_runtime = verifier_methods["_forward_fixed_m4_suffix"]
    assert "inner.layers[0]" in prefix
    assert "norm_query" not in prefix
    assert "range(1, len(inner.layers))" in suffix_runtime


def test_fixed_m4_sidecar_uses_the_installed_dispatch_contract():
    module = ast.parse((ROOT / "mtplx/qwen4_fixed_verify.py").read_text())
    sidecar = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "_FixedM4SidecarAux"
    )
    call = next(
        node
        for node in sidecar.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    assert [argument.arg for argument in call.args.args] == [
        "self",
        "_input_ids",
        "host_input_ids",
        "completion_tokens",
        "committed_count",
    ]


def test_owned_all_miss_prefetch_installs_exact_raw_rows_in_request_order(
    tmp_path, monkeypatch
):
    rows = 4
    weight = np.arange(rows * 20, dtype=np.uint32).reshape(rows, 20)
    scales = (1000 + np.arange(rows * 5, dtype=np.uint16)).reshape(rows, 5)
    biases = (2000 + np.arange(rows * 5, dtype=np.uint16)).reshape(rows, 5)
    path = tmp_path / "owned-rows.bin"
    path.write_bytes(weight.tobytes() + scales.tobytes() + biases.tobytes())

    offsets = {
        "weight": 0,
        "scales": weight.nbytes,
        "biases": weight.nbytes + scales.nbytes,
    }
    maps = {
        "weight": (
            np.memmap(path, mode="r", dtype=np.uint32, offset=offsets["weight"], shape=weight.shape),
            "U32",
        ),
        "scales": (
            np.memmap(path, mode="r", dtype=np.uint16, offset=offsets["scales"], shape=scales.shape),
            "BF16",
        ),
        "biases": (
            np.memmap(path, mode="r", dtype=np.uint16, offset=offsets["biases"], shape=biases.shape),
            "BF16",
        ),
    }
    pool = _ImmediatePool()
    hot = OrderedDict()
    sidecar = SimpleNamespace(
        _fd=os.open(path, os.O_RDONLY),
        _maps=maps,
        _pool=pool,
        _hot=hot,
        _hot_cap_rows=64,
    )
    calls = []
    real_pread = os.pread

    def counted_pread(fd, size, offset):
        calls.append((size, offset))
        return real_pread(fd, size, offset)

    helper = _load_function(
        ROOT / "mtplx/qwen4_fixed_verify.py",
        "_bind_fixed_m4_owned_row_prefetch",
        {"np": np, "os": SimpleNamespace(pread=counted_pread)},
    )
    try:
        submit, submit_window, install = helper(sidecar, all_miss=True)
        pending = submit(np.asarray([2, 0], dtype=np.int64))

        assert not hot
        assert pool.submissions == 2
        assert len(calls) == 6

        install(pending)

        assert list(hot) == [2, 0]
        for row_id in (2, 0):
            observed = hot[row_id]
            expected = (weight[row_id], scales[row_id], biases[row_id])
            for actual, reference in zip(observed, expected, strict=True):
                assert actual.dtype == reference.dtype
                assert actual.shape == reference.shape
                assert np.array_equal(actual, reference)
                owner = actual
                while getattr(owner, "base", None) is not None:
                    owner = owner.base
                assert isinstance(owner, bytes)

        install(submit_window(np.asarray([2, 0, 1, 3, 1], dtype=np.int64)))
        assert pool.submissions == 4
        assert len(calls) == 12
        assert list(hot) == [2, 0, 1, 3]
    finally:
        os.close(sidecar._fd)


def test_owned_full_window_prefetch_is_construction_bound_and_installed_before_gather():
    helper = _load_function(
        ROOT / "mtplx/qwen4_fixed_verify.py",
        "_bind_fixed_m4_owned_row_prefetch",
        {"np": np, "os": os},
    )
    base = dict(_fd=-1, _maps={}, _hot=OrderedDict())

    try:
        helper(SimpleNamespace(**base, _pool=None, _hot_cap_rows=16))
    except ValueError as exc:
        assert "worker pool" in str(exc)
    else:
        raise AssertionError("missing worker-pool construction refusal")

    try:
        helper(
            SimpleNamespace(
                **base,
                _pool=_ImmediatePool(),
                _hot_cap_rows=63,
            ),
            all_miss=True,
        )
    except ValueError as exc:
        assert "64" in str(exc)
    else:
        raise AssertionError("missing hot-capacity construction refusal")

    source = (ROOT / "mtplx/qwen4_fixed_verify.py").read_text()
    module = ast.parse(source)
    sidecar = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_FixedM4ExperimentalSidecarAux"
    )
    call = next(
        node
        for node in sidecar.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    call_source = ast.unparse(call)
    assert call_source.index("self._install_owned_rows") < call_source.index(
        "self._gather"
    )
    assert "self._resolve_window_rows" in call_source
    assert "os.environ" not in call_source
    assert "fallback" not in call_source.lower()


def test_fixed_m4_raw_q4_gather_preserves_storage_views_before_dequantization():
    module = ast.parse((ROOT / "mtplx/models/qwen4_exp.py").read_text())
    sidecar = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "_SidecarGather"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in sidecar.body
        if isinstance(node, ast.FunctionDef)
    }

    raw = methods["gather_raw_np"]
    assert all(repr(name) in raw for name in ("weight", "scales", "biases"))
    assert "mx.array" in raw
    assert "rows.view(mx.bfloat16)" in raw
    assert "mx.dequantize" not in raw
    # The packed uint32 weight must remain the direct mx.array result; only the
    # uint16 BF16 payloads are bit-viewed.
    assert "parts.append(rows)" in raw

    materialized = methods["gather_np"]
    assert "self.gather_raw_np(flat)" not in materialized
    assert all(repr(name) in materialized for name in ("weight", "scales", "biases"))
    assert "mx.array" in materialized
    assert "rows.view(mx.bfloat16)" in materialized
    assert "mx.dequantize" in materialized


def test_fixed_m4_graph_dequantizes_exact_raw_inputs_and_returns_same_bf16_node():
    calls = []

    class Tensor:
        def __init__(self, name):
            self.name = name

        def reshape(self, *shape):
            calls.append(("reshape", self, shape))
            return ("embedding", self, shape)

    class FakeMX:
        @staticmethod
        def dequantize(weight, scales, biases, *, group_size, bits):
            calls.append(
                (
                    "dequantize",
                    weight,
                    scales,
                    biases,
                    group_size,
                    bits,
                )
            )
            return Tensor("dequantized")

    helper = _load_function(
        ROOT / "mtplx/qwen4_fixed_verify.py",
        "_dequantize_fixed_m4_ple",
        {"mx": FakeMX},
    )
    raw = (Tensor("weight"), Tensor("scales"), Tensor("biases"))

    embedding = helper(raw, output_dim=2560)

    assert calls[0] == ("dequantize", *raw, 32, 4)
    assert calls[1][0] == "reshape"
    assert calls[1][2] == (1, 4, 2560)
    assert embedding == ("embedding", calls[1][1], (1, 4, 2560))


def test_fixed_m4_materialized_gather_restores_logical_window_shape():
    calls = []

    class Tensor:
        def reshape(self, *shape):
            calls.append(shape)
            return ("reshaped", shape)

    def gather(flat):
        calls.append(tuple(flat))
        return Tensor()

    helper = _load_function(
        ROOT / "mtplx/qwen4_fixed_verify.py",
        "_gather_fixed_m4_materialized",
    )
    result = helper((7, 8), gather=gather, output_dim=2560)

    assert calls == [(7, 8), (1, 4, 2560)]
    assert result == ("reshaped", (1, 4, 2560))


def test_fixed_m4_output_layout_returns_embedding_before_capture_and_state_leaves():
    unpack = _load_function(
        ROOT / "mtplx/graphbank.py", "_unpack_fixed_m4_outputs"
    )
    outputs = tuple(f"leaf-{index}" for index in range(9))

    logits, hidden, embedding, captures, state = unpack(
        outputs,
        capture_leaves=2,
        returns_aux=True,
    )

    assert (logits, hidden, embedding) == ("leaf-0", "leaf-1", "leaf-2")
    assert captures == outputs[3:5]
    assert state == outputs[5:]


def test_fixed_m4_returned_embedding_owns_selected_width_commit_lifetime():
    graphbank_source = (ROOT / "mtplx/graphbank.py").read_text()
    graphbank = ast.parse(graphbank_source)
    bank = next(
        node
        for node in ast.walk(graphbank)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    forward = next(
        node
        for node in bank.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_forward_installed_fixed_m4"
    )
    forward_source = ast.unparse(forward)
    assert "returned_aux" in forward_source
    assert "entry._mtplx_verify_compiled_aux = returned_aux" in forward_source
    assert "(state_in, compiled_aux)" in forward_source
    assert "mx.async_eval(*state_in)" in forward_source
    assert "mx.async_eval(*aux_inputs" not in forward_source
    assert (
        "self._held_aux_refs.append((compiled_aux, returned_aux))"
        in forward_source
    )

    verifier_source = (ROOT / "mtplx/qwen4_fixed_verify.py").read_text()
    verifier = ast.parse(verifier_source)
    binder = next(
        node
        for node in verifier.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_bind_fixed_m4_device_commit"
    )
    commit = next(
        node
        for node in ast.walk(binder)
        if isinstance(node, ast.FunctionDef) and node.name == "commit"
    )
    commit_source = ast.unparse(commit)
    assert "compiled_aux = split.returned_aux" in commit_source
    assert commit_source.index("compiled_aux = split.returned_aux") < commit_source.index(
        "compiled_aux[:, :logical_width]"
    )
    assert "entry._mtplx_verify_compiled_aux =" not in commit_source
    assert "entry._mtplx_verify_rows =" not in commit_source
    assert "entry._mtplx_verify_ple =" not in commit_source


def test_fixed_m4_transition_preserves_raw_q4_compile_contract():
    graphbank = ast.parse((ROOT / "mtplx/graphbank.py").read_text())
    bank = next(
        node
        for node in ast.walk(graphbank)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    install = next(
        node
        for node in bank.body
        if isinstance(node, ast.FunctionDef) and node.name == "install_fixed_m4"
    )
    transition = next(
        node
        for node in bank.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_transition_fixed_m4_generation"
    )
    install_source = ast.unparse(install)
    transition_source = ast.unparse(transition)

    assert "'aux_contract': aux_contract" in install_source
    assert "'graph_aux': graph_aux" in install_source
    assert "dispatch['aux_contract']" in transition_source
    assert "graph_aux=dispatch['graph_aux']" in transition_source
    assert "return_compiled_aux=dispatch['returns_aux']" in transition_source


def test_fixed_m4_materialized_prefetch_installs_callable_for_production_caller():
    path = ROOT / "mtplx/graphbank.py"
    no_op = _load_function(path, "_fixed_m4_materialized_prefetch")
    assert no_op(7, [8, 9], 2) is None

    graphbank = ast.parse(path.read_text())
    bank = next(
        node
        for node in ast.walk(graphbank)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    install = next(
        node
        for node in bank.body
        if isinstance(node, ast.FunctionDef) and node.name == "install_fixed_m4"
    )
    install_source = ast.unparse(install)
    assert "prefetch_aux = _fixed_m4_materialized_prefetch" in install_source
    assert "'prefetch_aux': prefetch_aux" in install_source


def test_compiled_verify_receipt_formats_materialized_and_raw_q4_keys():
    path = ROOT / "mtplx/graphbank.py"
    formatter = _load_function(path, "_format_compiled_verify_key")
    assert formatter((4, "post_norm", 1)) == "m4:post_norm:b1"
    assert formatter((4, "post_norm", 1, "raw_q4")) == (
        "m4:post_norm:b1:raw_q4"
    )

    graphbank = ast.parse(path.read_text())
    bank = next(
        node
        for node in ast.walk(graphbank)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    receipt = next(
        node
        for node in bank.body
        if isinstance(node, ast.FunctionDef) and node.name == "to_dict"
    )
    assert "_format_compiled_verify_key(key)" in ast.unparse(receipt)


def test_raw_q4_auxiliary_route_is_explicit_opt_in_at_construction():
    path = ROOT / "mtplx/qwen4_fixed_verify.py"
    module = ast.parse(path.read_text())
    install = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "install_qwen4_fixed_verify_route"
    )
    keyword_defaults = {
        argument.arg: default
        for argument, default in zip(
            install.args.kwonlyargs,
            install.args.kw_defaults,
            strict=True,
        )
    }
    assert isinstance(keyword_defaults["raw_q4_aux"], ast.Constant)
    assert keyword_defaults["raw_q4_aux"].value is False
    assert isinstance(keyword_defaults["owned_all_miss_rows"], ast.Constant)
    assert keyword_defaults["owned_all_miss_rows"].value is False
    assert isinstance(keyword_defaults["early_window_prefetch"], ast.Constant)
    assert keyword_defaults["early_window_prefetch"].value is False

    install_source = ast.unparse(install)
    assert "if raw_q4_aux:" in install_source
    assert (
        "runtime.build_fixed_m4_compiled_verify_aux = "
        "partial(_build_fixed_m4_compiled_verify_aux, runtime, "
        "raw_q4_aux=raw_q4_aux, owned_all_miss_rows=owned_all_miss_rows, "
        "early_window_prefetch=early_window_prefetch)"
        in install_source
    )
    assert install_source.index(
        "runtime.build_fixed_m4_compiled_verify_aux"
    ) < install_source.index("if raw_q4_aux:")
    assert (
        "runtime.dequantize_fixed_m4_compiled_verify_aux"
        in install_source[install_source.index("if raw_q4_aux:") :]
    )

    graphbank = ast.parse((ROOT / "mtplx/graphbank.py").read_text())
    bank = next(
        node
        for node in ast.walk(graphbank)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    fixed_install = next(
        node
        for node in bank.body
        if isinstance(node, ast.FunctionDef) and node.name == "install_fixed_m4"
    )
    fixed_source = ast.unparse(fixed_install)
    assert "if graph_aux is None:" in fixed_source
    assert "returns_aux = False" in fixed_source
    assert "aux_contract = 'materialized'" in fixed_source


def test_retained_materialized_aux_is_monomorphic_in_the_hot_path():
    qwen_path = ROOT / "mtplx/qwen4_fixed_verify.py"
    qwen_module = ast.parse(qwen_path.read_text())
    materialized = next(
        node
        for node in qwen_module.body
        if isinstance(node, ast.ClassDef) and node.name == "_FixedM4SidecarAux"
    )
    materialized_call = next(
        node
        for node in materialized.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    materialized_source = ast.unparse(materialized_call)
    assert "_resolve_window_rows" not in materialized_source
    assert "gather_raw_np" not in materialized_source
    assert "self._gather(rows.reshape(-1)).reshape" in materialized_source

    builder = next(
        node
        for node in qwen_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_fixed_m4_compiled_verify_aux"
    )
    builder_source = ast.unparse(builder)
    assert "return _FixedM4SidecarAux(" in builder_source
    assert "return _FixedM4ExperimentalSidecarAux(" in builder_source

    model_path = ROOT / "mtplx/models/qwen4_exp.py"
    model_module = ast.parse(model_path.read_text())
    sidecar = next(
        node
        for node in ast.walk(model_module)
        if isinstance(node, ast.ClassDef) and node.name == "_SidecarGather"
    )
    gather_np = next(
        node
        for node in sidecar.body
        if isinstance(node, ast.FunctionDef) and node.name == "gather_np"
    )
    assert "self.gather_raw_np" not in ast.unparse(gather_np)
