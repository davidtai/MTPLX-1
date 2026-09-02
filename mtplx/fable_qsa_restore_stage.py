"""Wire ``mtplx.qsa_restore_staging`` into the near-prefix restore path.

Why
---
``mtplx/qsa_restore_staging.py`` was written and unit-tested but imported by
nothing outside its own test, so the cost it exists to remove was still being
paid on every warm restore.  Today the first restored-suffix forward grows
three independent QSA backings *lazily*, inside the prefill chunks:

* ``QSACache.write_raw`` reallocates and full-copies ``raw_keys`` the first
  time the frontier passes its capacity (``QSACache._grown_cap``,
  ``mtplx/models/qwen4_exp.py``),
* ``QSACache.write_pooled`` does the same for ``pooled`` *and* its fp32
  transposed mirror, and
* ``KVCache.update_and_fetch`` does the same for the attention K/V.

A restored entry lands with backings sized to the *stored* prefix, so a warm
agent turn pays every one of those promotions inside its first suffix chunk --
directly on the time-to-first-token path this flag is being measured on.
Staging reserves all three once, behind one grouped evaluation barrier, before
the suffix forward starts.

Arming it
---------
``MTPLX_FABLE_QSA_RESTORE_STAGING=1``, default off.  The gate is read once and
memoized (a warm restore must not pay a repeated environment lookup), and it is
resolved lazily rather than at import so a serving profile that arms env flags
after ``mtplx.generation`` is imported is still observed.

Fail-closed
-----------
There is no silent fallback.  Eligibility is decided once, at the top of the
restored-suffix prefill (the construction point for this request's staged
backings), and an armed flag that cannot be honoured raises
:class:`QSARestoreStagingUnsupported` instead of quietly running the unstaged
path.  A silent fallback would let an A/B report a staged arm that never
staged anything.

The one deliberate no-op is a cache stack that has QSA layers but needs no
promotion (already wide enough): that is staging succeeding, and it is
reported with zero promotions rather than raising.

Scope
-----
Only the *trunk* cache is staged.  The committed-MTP history cache grows on a
different schedule (``_append_mtp_history`` windows, not the prompt suffix), so
planning it against ``len(suffix)`` would over-reserve; it is left alone.

Nothing here changes restored state.  ``apply_restored_qsa_stage`` copies the
live prefix rows into wider zero-filled backings and calls
``QSACache.reserve_indexer_capacity``; ``kv.offset``, ``pooled_len``, the raw
index keys, the pooled bank contents, and every positional/MRoPE field are
carried through untouched.  ``tests/test_fable_qsa_restore_stage.py`` pins that
against an unstaged twin.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from .qsa_restore_staging import (
    QSARestoreStagingError,
    RestoredQSAStageReport,
    stage_restored_qsa_suffix,
)

FABLE_QSA_RESTORE_STAGING_ENV = "MTPLX_FABLE_QSA_RESTORE_STAGING"

_TRUTHY = {"1", "true", "yes", "on"}
_ENABLED_CACHE: bool | None = None

#: Process-wide engagement counters.  The counters law: a lane that reads flat
#: in an A/B must still be able to prove it ran.
COUNTERS: dict[str, int] = {
    "stage_calls": 0,
    "staged_layers": 0,
    "kv_promotions": 0,
    "raw_promotions": 0,
    "pooled_promotions": 0,
}


class QSARestoreStagingUnsupported(RuntimeError):
    """The staged restore was armed where it cannot be honoured.

    Raised at the staging point only.  The flag is opt-in and fails loudly:
    a restore that asks for staged backings and cannot have them is an error,
    never a silent return to the lazy-growth path (which would quietly
    invalidate any measurement taken under the flag).
    """


def qsa_restore_staging_enabled() -> bool:
    """Return the ``MTPLX_FABLE_QSA_RESTORE_STAGING`` gate; read once, default off."""

    global _ENABLED_CACHE
    if _ENABLED_CACHE is None:
        raw = os.environ.get(FABLE_QSA_RESTORE_STAGING_ENV)
        _ENABLED_CACHE = bool(raw) and raw.strip().lower() in _TRUTHY
    return _ENABLED_CACHE


def reset_qsa_restore_staging_flag_cache() -> None:
    """Drop the memoized gate. Test-support only."""

    global _ENABLED_CACHE
    _ENABLED_CACHE = None


def reset_counters() -> None:
    """Zero the engagement counters. Test-support only."""

    for key in COUNTERS:
        COUNTERS[key] = 0


def _is_qsa_cache(cache: Any) -> bool:
    # Same deliberately narrow, model-independent marker the planner uses.
    return callable(getattr(cache, "reserve_indexer_capacity", None))


def restore_staging_eligibility(
    caches: Any, *, suffix_tokens: int
) -> str | None:
    """Return why this restore cannot be staged, or ``None`` when it can.

    Checked once per restore, before any promotion.  ``no-qsa-cache`` is an
    operator error rather than a compatibility path: this flag exists for the
    Flash-Next QSA lane, and arming it against a cache stack with no QSA layer
    means the arm measured nothing.
    """

    if int(suffix_tokens) <= 0:
        return "empty-suffix"
    if not isinstance(caches, Sequence) or isinstance(caches, (str, bytes)):
        return "caches-not-a-sequence"
    if not caches:
        return "empty-cache-stack"
    if not any(_is_qsa_cache(cache) for cache in caches):
        return "no-qsa-cache"
    return None


def _default_allocate_zeros(shape: tuple[int, ...], dtype: Any) -> Any:
    import mlx.core as mx

    return mx.zeros(shape, dtype)


def _default_materialize_cache(caches: Sequence[Any]) -> None:
    # One grouped barrier over the promoted roots.  Deferring it would move
    # the copy cost back into the first suffix chunk -- exactly the cost this
    # module removes -- and make the staging receipt a lie.
    from .generation import _eval_cache_roots

    _eval_cache_roots(list(caches))


def stage_restored_suffix(
    caches: Any,
    *,
    suffix_tokens: int,
    allocate_zeros: Callable[[tuple[int, ...], Any], Any] | None = None,
    materialize_cache: Callable[[Sequence[Any]], None] | None = None,
) -> RestoredQSAStageReport:
    """Promote every restored QSA backing to its end-of-suffix capacity.

    Raises :class:`QSARestoreStagingUnsupported` when the armed flag cannot be
    honoured.  Callers must not swallow it.
    """

    reason = restore_staging_eligibility(caches, suffix_tokens=suffix_tokens)
    if reason is not None:
        raise QSARestoreStagingUnsupported(
            f"{FABLE_QSA_RESTORE_STAGING_ENV} cannot stage this restore: {reason}"
        )
    try:
        report = stage_restored_qsa_suffix(
            caches,
            suffix_tokens=int(suffix_tokens),
            allocate_zeros=(
                _default_allocate_zeros if allocate_zeros is None else allocate_zeros
            ),
            materialize_cache=(
                _default_materialize_cache
                if materialize_cache is None
                else materialize_cache
            ),
        )
    except QSARestoreStagingError as exc:
        raise QSARestoreStagingUnsupported(
            f"{FABLE_QSA_RESTORE_STAGING_ENV} refused a restored cache: {exc}"
        ) from exc

    COUNTERS["stage_calls"] += 1
    COUNTERS["staged_layers"] += report.qsa_entries
    COUNTERS["kv_promotions"] += report.kv_promotions
    COUNTERS["raw_promotions"] += report.raw_promotions
    COUNTERS["pooled_promotions"] += report.pooled_promotions
    if os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
        import sys

        print(
            "[mtplx] qsa-restore-staging: "
            f"layers={report.qsa_entries} suffix={int(suffix_tokens)} "
            f"kv={report.kv_promotions} raw={report.raw_promotions} "
            f"pooled={report.pooled_promotions}",
            file=sys.stderr,
            flush=True,
        )
    return report


__all__ = [
    "COUNTERS",
    "FABLE_QSA_RESTORE_STAGING_ENV",
    "QSARestoreStagingUnsupported",
    "qsa_restore_staging_enabled",
    "reset_counters",
    "reset_qsa_restore_staging_flag_cache",
    "restore_staging_eligibility",
    "stage_restored_suffix",
]
