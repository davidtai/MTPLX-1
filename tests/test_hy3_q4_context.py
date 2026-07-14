from types import SimpleNamespace

import pytest

from mtplx.hy3_q4_context import (
    HY3_Q4_CONTEXT_WINDOW,
    ContextLimitExceeded,
    SequenceBusyError,
    SingleSequenceGate,
    admit_hy3_q4_context,
)
from mtplx.runtime_options import validate_hy3_q4_dynamic_context_options


def _options(**overrides):
    values = {
        "hy3_q4_dynamic_context": True,
        "paged_kv_quantization": "q4",
        "context_window": HY3_Q4_CONTEXT_WINDOW,
        "scheduler_mode": "serial",
        "batching_preset": "latency",
        "max_active_requests": 1,
        "decode_batch_max": 1,
        "session_bank_live_refs": False,
        "generation_mode": "ar",
        "load_mtp": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _expert_config(**overrides):
    values = {
        "model_key": "hy3-q4",
        "cache_scope": "global",
        "slot_layout": "component-banks",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("rendered_input_tokens", "requested_output_tokens", "expected_total"),
    [
        (130_943, 128, 131_071),
        (130_944, 128, 131_072),
        (131_072, 0, 131_072),
    ],
)
def test_hy3_q4_context_admits_exact_total_boundary(
    rendered_input_tokens, requested_output_tokens, expected_total
):
    admission = admit_hy3_q4_context(
        rendered_input_tokens=rendered_input_tokens,
        requested_output_tokens=requested_output_tokens,
    )

    assert admission.rendered_input_tokens == rendered_input_tokens
    assert admission.requested_output_tokens == requested_output_tokens
    assert admission.admitted_total_tokens == expected_total
    assert admission.model_context_limit_tokens == HY3_Q4_CONTEXT_WINDOW
    assert admission.to_dict() == {
        "model_context_limit_tokens": 131_072,
        "rendered_input_tokens": rendered_input_tokens,
        "requested_output_tokens": requested_output_tokens,
        "admitted_total_tokens": expected_total,
    }


def test_hy3_q4_context_rejects_131073_total_tokens():
    with pytest.raises(ContextLimitExceeded, match=r"131073.*131072"):
        admit_hy3_q4_context(
            rendered_input_tokens=130_945,
            requested_output_tokens=128,
        )


@pytest.mark.parametrize(
    ("rendered_input_tokens", "requested_output_tokens"), [(-1, 0), (0, -1)]
)
def test_hy3_q4_context_rejects_negative_token_counts(
    rendered_input_tokens, requested_output_tokens
):
    with pytest.raises(ValueError, match="must be >= 0"):
        admit_hy3_q4_context(
            rendered_input_tokens=rendered_input_tokens,
            requested_output_tokens=requested_output_tokens,
        )


@pytest.mark.parametrize("invalid", [True, 1.0, "1"])
@pytest.mark.parametrize(
    ("field", "valid_other"),
    [
        ("rendered_input_tokens", {"requested_output_tokens": 0}),
        ("requested_output_tokens", {"rendered_input_tokens": 0}),
    ],
)
def test_hy3_q4_context_requires_exact_integer_counts(field, valid_other, invalid):
    with pytest.raises(TypeError, match=rf"{field} must be an exact int"):
        admit_hy3_q4_context(**valid_other, **{field: invalid})


def test_hy3_q4_context_limit_cannot_be_overridden_by_caller():
    with pytest.raises(TypeError, match="unexpected keyword argument 'limit'"):
        admit_hy3_q4_context(
            rendered_input_tokens=1,
            requested_output_tokens=1,
            limit=2,
        )


def test_single_sequence_gate_releases_exactly_once():
    gate = SingleSequenceGate()
    admission = admit_hy3_q4_context(
        rendered_input_tokens=4_000,
        requested_output_tokens=96,
    )
    first = gate.acquire("request-1", admission=admission)

    with pytest.raises(SequenceBusyError, match="request-1"):
        gate.acquire("request-2", admission=admission)

    active = gate.snapshot()
    assert active["admission_state"] == "active"
    assert active["active_request_id"] == "request-1"
    assert active["rendered_input_tokens"] == 4_000
    assert active["requested_output_tokens"] == 96
    assert active["admitted_total_tokens"] == 4_096

    assert first.release("complete") is True
    assert first.release("cancel") is False

    released = gate.snapshot()
    assert released["admission_state"] == "last"
    assert released["active_request_id"] is None
    assert released["last_admission"] == admission.to_dict()

    second = gate.acquire("request-2", admission=admission)
    assert second.release("complete") is True


def test_single_sequence_worker_hold_defers_owner_release_until_terminal():
    gate = SingleSequenceGate()
    admission = admit_hy3_q4_context(
        rendered_input_tokens=4_000,
        requested_output_tokens=96,
    )
    owner = gate.acquire("streaming-request", admission=admission)
    worker = owner.hold()

    assert owner.release("http_disconnected") is True
    assert owner.release("duplicate_http_terminal") is False
    assert gate.active_request_id == "streaming-request"
    with pytest.raises(SequenceBusyError, match="streaming-request"):
        gate.acquire("second-request", admission=admission)

    assert worker.release("generation_worker_terminal") is True
    assert worker.release("duplicate_worker_terminal") is False
    assert gate.active_request_id is None
    assert gate.snapshot()["last_admission"] == admission.to_dict()


def test_hy3_q4_dynamic_context_is_disabled_by_default():
    args = _options(hy3_q4_dynamic_context=False, paged_kv_quantization="off")

    assert validate_hy3_q4_dynamic_context_options(args, None) is False


def test_hy3_q4_dynamic_context_accepts_only_the_pinned_lane():
    assert validate_hy3_q4_dynamic_context_options(_options(), _expert_config()) is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"generation_mode": "mtp"}, "generation-mode ar"),
        ({"load_mtp": True}, "no-load-mtp"),
    ],
)
def test_hy3_q4_dynamic_context_rejects_mtp_startup_options(override, message):
    with pytest.raises(ValueError, match=message):
        validate_hy3_q4_dynamic_context_options(
            _options(**override),
            _expert_config(),
        )


@pytest.mark.parametrize("mode", ["off", "q8"])
def test_hy3_q4_dynamic_context_rejects_non_q4_kv(mode):
    with pytest.raises(ValueError, match="paged-kv-quantization q4"):
        validate_hy3_q4_dynamic_context_options(
            _options(paged_kv_quantization=mode), _expert_config()
        )


@pytest.mark.parametrize("context_window", [0, 131_071, 131_073])
def test_hy3_q4_dynamic_context_rejects_wrong_context_window(context_window):
    with pytest.raises(ValueError, match="context-window 131072"):
        validate_hy3_q4_dynamic_context_options(
            _options(context_window=context_window), _expert_config()
        )


def test_hy3_q4_dynamic_context_rejects_loaded_model_with_shorter_context():
    with pytest.raises(ValueError, match="loaded model.*131072-token"):
        validate_hy3_q4_dynamic_context_options(
            _options(),
            _expert_config(),
            resolved_context_window=32_768,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"scheduler_mode": "ar_batch"}, "scheduler-mode serial"),
        ({"max_active_requests": None}, "max-active-requests 1 exactly"),
        ({"max_active_requests": 0}, "max-active-requests 1 exactly"),
        ({"max_active_requests": -1}, "max-active-requests 1 exactly"),
        ({"max_active_requests": 2}, "max-active-requests 1 exactly"),
        ({"decode_batch_max": None}, "decode-batch-max 1 exactly"),
        ({"decode_batch_max": 0}, "decode-batch-max 1 exactly"),
        ({"decode_batch_max": -1}, "decode-batch-max 1 exactly"),
        ({"decode_batch_max": 2}, "decode-batch-max 1 exactly"),
    ],
)
def test_hy3_q4_dynamic_context_rejects_concurrent_scheduling(override, message):
    with pytest.raises(ValueError, match=message):
        validate_hy3_q4_dynamic_context_options(_options(**override), _expert_config())


def test_hy3_q4_dynamic_context_rejects_session_bank_live_refs_at_startup():
    with pytest.raises(ValueError, match="no-session-bank-live-refs"):
        validate_hy3_q4_dynamic_context_options(
            _options(session_bank_live_refs=True),
            _expert_config(),
        )


@pytest.mark.parametrize(
    ("config_override", "message"),
    [
        ({"model_key": "glm52-q4"}, "Hy3 Q4 expert streaming"),
        ({"cache_scope": "layer"}, "global component-bank"),
        ({"slot_layout": "direct-slots"}, "global component-bank"),
    ],
)
def test_hy3_q4_dynamic_context_rejects_non_global_component_bank_lane(
    config_override, message
):
    with pytest.raises(ValueError, match=message):
        validate_hy3_q4_dynamic_context_options(
            _options(), _expert_config(**config_override)
        )
