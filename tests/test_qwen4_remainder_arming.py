"""The PR 391 remainder lanes must arm when SERVED, not freeze at import.

This reproduces the exact ``mtplx serve`` order that hid two arming failures
the battery caught (2026-09-07): the reader modules are imported BEFORE the
fixed-M4 auto-arm stamps the lane keys into the environment, so an
import-time-frozen reader returns its default (off) forever -- the served lane
is then absent from /health with no verdict line. Every remainder-lane reader
must resolve the environment at USE (the install/route path, which runs after
the overrides are applied), so a stamp that lands after import is still seen.

Nothing here dispatches Metal; it drives the pure-Python auto-arm block and the
env readers.
"""

from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace

# Import the server + runtime modules FIRST, exactly as `mtplx serve` does, so
# any import-time reader cache would be populated with the (unset) defaults
# before the auto-arm runs. If a reader froze here, the asserts below fail.
import mtplx.runtime  # noqa: F401  (import-order fidelity)
import mtplx.server.openai as openai
from mtplx import runtime_options as ro
from mtplx.models import qwen4_exp
from mtplx.native import native_qsa_available
from mtplx.profiles import normalize_runtime_env_overrides
from mtplx.qwen4_prefill_chunk import resolve_query_tile_rows

_LANE_ENV_KEYS = (
    "MTPLX_QWEN4_HC_M4",
    "MTPLX_QWEN4_PREFILL_MASK_FUSE",
    "MTPLX_QSA_PREFILL_QUERY_TILE",
    "MTPLX_QSA_SPARSE_DECODE",
    "MTPLX_QSA_SPARSE_DECODE_TILE",
    "MTPLX_QSA_SPARSE_DECODE_SPLITS",
    "MTPLX_FABLE_HC_M4",
    "MTPLX_FABLE_PREFILL_MASK_FUSE",
    "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE",
    "MTPLX_FABLE_QSA_SPARSE_DECODE",
)


def test_remainder_lanes_arm_when_served_not_frozen_at_import(tmp_path, monkeypatch):
    # The state at a fresh import: reader globals unforced (None -> read env),
    # every lane key unset.
    for name in (
        "_QWEN4_HC_M4",
        "_QSA_SPARSE_DECODE",
        "_QSA_SPARSE_DECODE_TILE",
        "_QSA_SPARSE_DECODE_SPLITS",
    ):
        monkeypatch.setattr(ro, name, None)
    for key in _LANE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    # Before the auto-arm: every reader off, exactly the served import-time
    # state. (A reader frozen True/False at import would already fail here.)
    assert ro.qwen4_hc_m4_enabled() is False
    assert qwen4_exp._prefill_mask_fuse_enabled() is False
    assert resolve_query_tile_rows() == 0
    assert ro.qsa_sparse_decode_enabled() is False

    # Run the fixed-M4 auto-arm and APPLY it to the environment, as the server
    # does via apply_profile_env. Force the fixed-M4 predicate on rather than
    # crafting a full fixed-verify config; the remainder defaults gate on that
    # predicate (and, for the decode lane, the built native extension).
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )
    args = SimpleNamespace(
        generation_mode="mtp",
        verify_strategy="capture_commit",
        model=str(tmp_path),
    )
    monkeypatch.setattr(openai, "_served_model_is_qwen4_fixed_m4", lambda a: True)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        overrides = openai._server_runtime_env_overrides(args, {})
    log = err.getvalue()

    # Every stamped key survives the boot-time validator (the check that only
    # runs inside apply_profile_env), then apply them as the server would.
    assert normalize_runtime_env_overrides(overrides) == overrides
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    # The three unconditional remainder lanes are armed, and each reader --
    # read at USE, after the stamp was applied -- sees it. This is the class
    # that failed when served: an import-frozen reader returns its default.
    assert overrides.get("MTPLX_QWEN4_HC_M4") == "1"
    assert ro.qwen4_hc_m4_enabled() is True
    assert overrides.get("MTPLX_QWEN4_PREFILL_MASK_FUSE") == "1"
    assert qwen4_exp._prefill_mask_fuse_enabled() is True
    assert overrides.get("MTPLX_QSA_PREFILL_QUERY_TILE") == "2048"
    assert resolve_query_tile_rows() == 2048

    # The native-gated decode lane: armed (install attempted) when the
    # extension is built, else an explicit declined-to-stock verdict so a
    # wheel without it still serves -- never a silent absence.
    if native_qsa_available():
        assert overrides.get("MTPLX_QSA_SPARSE_DECODE") == "1"
        assert ro.qsa_sparse_decode_enabled() is True
        # its companions resolve to the measured geometry at use, too
        assert ro.qsa_sparse_decode_tile() == (128, 32)
        assert ro.qsa_sparse_decode_splits() == 17
    else:
        assert "MTPLX_QSA_SPARSE_DECODE" not in overrides
        assert "MTPLX_QSA_SPARSE_DECODE declined to stock" in log
        assert ro.qsa_sparse_decode_enabled() is False
