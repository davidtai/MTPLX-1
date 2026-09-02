"""KV-only MTP history append -- ``MTPLX_FABLE_MTP_KV_ONLY_APPEND``.

``L-fable-decode-ideas.md`` §B.  Once per decode cycle the stock native-MTP
lane appends the committed rows to the MTP head's own QSA cache
(``generation.py::_append_mtp_history`` -> ``runtime.update_mtp_cache`` ->
``qwen4_exp.TextModel.mtp_update_cache`` ->
``Qwen4ExpMTP.fuse_and_run_history``).  That call runs the head's ENTIRE
``DecoderLayer``:

===================================  ==========================================
today                                consumed by anything?
===================================  ==========================================
fused prepare (norms + fc + embed)   yes -- it is the layer input
attn hyper-connection read           yes -- ``mixed`` is the attention input
``index_qk_proj`` + k norm/rope       **yes -- written to the indexer streams**
indexer query prep + top-512 scorer  no
``q_proj`` (the layer's widest read) no
``k_proj``/``v_proj`` + norms + rope  **yes -- written to the KV cache**
sparse attention product             no
output gate + ``o_proj``             no
attn residual write                  no
mlp hyper-connection read            no
512-expert MoE + shared expert       no
mlp residual write                   no
===================================  ==========================================

The right column is the whole contract: **every caller discards the returned
hidden.**  ``_append_mtp_history`` binds it only to feed ``_eval(hidden)``, and
that eval is itself already skipped on the production profile
(``MTPLX_LAZY_MTP_HISTORY_APPEND=1`` in ``profiles.NATIVE_MTP_60_FAST_PATH_ENV``;
``force_eval`` is True only on the two *prefill* call sites and on the
``MTPLX_MTP_HISTORY_MATERIALIZE_EVERY`` counter, which defaults to 0).  So on
the retained lane the append is a queued write whose results are never read --
and it still costs a full QSA-attention block plus a 512-expert MoE of
dependent GPU work on the serial segment between two verify graphs.

What this lane changes
----------------------
Armed, the decode-lane append routes to :meth:`Qwen4ExpMTP.append_kv_only`,
which runs the same fused input preparation and the same attention hyper read,
then calls the layer's attention with ``kv_only=True``.  That path evaluates
exactly the expressions that feed a cache write, in the same order, and
returns None:

* indexer: ``index_qk_proj`` -> split -> ``write_raw`` -> ``_extend_pooled``.
  On the compiled route this is the *existing* ``update_only`` mode, whose
  graph computes ``raw_next``/``pooled_next`` before it branches on the
  selector (``kernels/qsa_indexer_compile.py::_make_compiled``), so the two
  modes cannot disagree about cache contents.
* attention: ``k_proj``/``v_proj`` -> reshape -> ``k_norm`` -> the same rope
  branch -> ``cache.kv.update_and_fetch``.

Bit-exactness is therefore structural, not empirical: no expression that
reaches a cache leaf is rewritten, reordered against another write, or given a
different dtype or rope position.  ``tests/test_fable_mtp_kv_only.py`` pins it
on random inputs anyway, and pins that the skipped work is not executed.

Eligibility is decided ONCE, at request construction, and RAISES rather than
falling back: an armed flag that quietly ran the full append would make the
receipts lie about which lane produced the number.

Not armed here (deliberately)
-----------------------------
* **Prefill appends.**  ``generation.py`` lines ~2921 / ~4608 / ~6524 pass
  ``force_eval=True`` and their hidden is equally unused, so the same saving
  is available there; it is a separate change with a separate ABBA, and this
  lane keeps the decode cycle as its only variable.
* **The PR391 fixed-D3 handoff** (``pr391_mtp_handoff.bind_pr391_mtp_device_replay``)
  calls ``rt.update_mtp_cache`` directly, not through ``_append_mtp_history``,
  and reads the cache leaves it produces.  It is untouched.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

_ENV_VAR = "MTPLX_FABLE_MTP_KV_ONLY_APPEND"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


#: Read ONCE, at import.  Everything downstream keys on this, so a mid-run
#: ``os.environ`` mutation cannot split one request across two lanes.
_ENABLED = _env_truthy(_ENV_VAR)


def is_enabled() -> bool:
    """True when ``MTPLX_FABLE_MTP_KV_ONLY_APPEND`` was set at import."""

    return _ENABLED


def _configure_for_test(enabled: bool) -> None:
    """Flip the module gate (tests only); never a hot path."""

    global _ENABLED
    _ENABLED = bool(enabled)


def _accepts_keyword(fn: Any, name: str) -> bool:
    """True when ``fn`` names ``name`` explicitly (never via ``**kwargs``).

    ``**kwargs`` is not evidence: ``qwen4_exp.Model.mtp_update_cache`` forwards
    blindly, so a wrapper would look eligible for any backend it wraps.
    """

    if not callable(fn):
        return False
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def claim_model_route(model: Any) -> bool:
    """Construction-time eligibility for one loaded model.

    Returns False when the flag is off (the only quiet outcome).  When the
    flag is ON this either returns True or RAISES, naming what is missing.
    """

    if not _ENABLED:
        return False

    inner = getattr(model, "language_model", model)
    update = getattr(inner, "mtp_update_cache", None)
    if not callable(update):
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs a model with mtp_update_cache; "
            f"{type(inner).__name__} has none"
        )
    if not _accepts_keyword(update, "kv_only"):
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs an mtp_update_cache that takes kv_only; "
            f"{type(inner).__name__}.mtp_update_cache does not"
        )

    head = getattr(inner, "mtp", None)
    if head is None:
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs an attached MTP head; this model has none"
        )
    if not callable(getattr(head, "append_kv_only", None)):
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs Qwen4ExpMTP.append_kv_only; "
            f"{type(head).__name__} has no KV-only append"
        )

    layers = getattr(head, "layers", None)
    if layers is None or len(layers) != 1:
        raise RuntimeError(
            f"{_ENV_VAR}=1 assumes the one-DecoderLayer MTP head; got "
            f"{0 if layers is None else len(layers)} layers"
        )
    layer = layers[0]
    if getattr(layer, "is_linear", False):
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs the head's layer to be QSA attention; this "
            "one is linear attention (recurrent state, not a KV cache)"
        )
    if not _accepts_keyword(layer.__call__, "kv_only"):
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs a DecoderLayer that takes kv_only; the "
            "head's layer does not (a generic rewrite may have replaced it)"
        )
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs the head layer's self_attn; it has none"
        )
    if not _accepts_keyword(attention.__call__, "kv_only"):
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs an Attention that takes kv_only; "
            f"{type(attention).__name__}.__call__ does not"
        )
    indexer = getattr(attention, "indexer", None)
    if indexer is None:
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs the QSA indexer whose raw/pooled writes the "
            "append exists for; this attention has none"
        )
    if not _accepts_keyword(indexer.__call__, "write_only"):
        raise RuntimeError(
            f"{_ENV_VAR}=1 needs a QSA indexer that takes write_only; "
            f"{type(indexer).__name__}.__call__ does not"
        )
    return True
