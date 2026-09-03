"""Per-token provenance for one ``generate_mtpk`` request.

Why
---
Every fable ABBA verdict rests on a receipt, and until now a receipt could not
answer *which lane produced token i*.  The generated ids themselves were kept
only on a parity failure (``abba_driver`` writes ``response_token_ids`` inside
``softfloat64-parity-failure-*.json``), and the only surviving trace of the
text was ``response_text_head`` / ``response_text_tail``, 600 characters each.
On 2026-09-02 that cost hours twice over: a duplicated subword in one
candidate's output had to be reconstructed after the fact from n-gram probes,
and "this candidate produced the control's output" had to be proved by hashing
two text fragments by hand.

This module is the fix.  It records, per request, the **commit order** of the
decode loop as a list of ``(source code, count)`` spans, and turns that into

* ``output_ids_sha256`` -- sha256 over the little-endian ``uint32`` id array,
* ``output_ids_b64``    -- those ids, base64,
* ``token_sources_b64`` -- one ``uint8`` source code per token, base64.

For a 1,024-token arm that is 5,464 + 1,368 characters of base64: ~6.8 KiB on
a receipt whose measured row is already tens of KiB.

Spans, not a parallel list
--------------------------
``generate_mtpk`` trims its committed stream in three places
(``_truncate_after_first_stop`` and two ``tokens[:stop_index + 1]`` slices),
and every one of them is a **prefix** trim.  A parallel per-token list would
have to mirror each trim by hand -- three more places to forget.  Spans are
expanded against ``len(tokens)`` at the end instead, so a suffix trim needs no
code here at all, and an expansion that runs *short* is reported
(``complete: false``) rather than silently padded with a lie.

Coverage
--------
The call sites are the emission sites, and they are the same set the K20 log
had to grow on ``worker/w51b-shadow-segments`` (00ac2690) after it turned out
that ``mtplx/context_copy.py`` block rounds commit accepted prefixes *and*
their residual correction without ever entering the accept loop the K20 hooks
live in.  That commit's carry accounting is the map; this module mirrors its
coverage and duplicates none of its logger:

===============================  =======================================
emission site                    call
===============================  =======================================
fresh primary sample             :meth:`primary`
context-copy round, eager        :meth:`copy_block` + :meth:`copy_correction`
context-copy round, batched      :meth:`copy_block` + :meth:`copy_correction`
all-accept draft block           :meth:`draft_run`
speculative bonus                :meth:`bonus`
partial accept + correction      :meth:`mtp_commit`
===============================  =======================================

A token that reaches the stream through none of these expands to
:data:`SOURCE_UNKNOWN`, which is what an uninstrumented lane looks like -- a
positive signal, not a gap the reader has to guess at.

One request at a time
---------------------
The singleton holds ONE request's spans, cleared by :meth:`begin_request`.
That is exactly right for the fable drivers, which generate one request at a
time in a process they own, and it is wrong for a server handling concurrent
generations: two interleaved requests would share one span list and both
receipts would be nonsense.  So this is bench instrumentation, off by default,
and a concurrent serving process must not arm it.  ``abba_driver`` arms the
singleton directly rather than through the environment, so the flag never
reaches a process that did not ask for it.

Cost
----
Zero when disabled.  ``generate_mtpk`` snapshots ``enabled`` into a local bool
once per request and every call site is behind that local; the methods
themselves are also self-guarded so a stray call from anywhere else is free.
Armed, one request costs a handful of tuple appends per decode cycle and one
NumPy expansion at the end -- host-only, no ``mx.eval``, no device work.
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Sequence

import numpy as np


#: Receipt schema tag.  Bump when the code table or the encoding changes.
SCHEMA = "mtplx-fable-token-source-v1"

#: Env flag, read ONCE at import.  ``abba_driver`` does not need it (it arms
#: the singleton directly, in-process, right before it generates), but a
#: manual run or a serving process can set it to get the same accounting.
ENV_FLAG = "MTPLX_FABLE_TOKEN_SOURCE"

# -- source codes ----------------------------------------------------------
# uint8.  Everything below 16 is a named lane; 16 + d is an MTP draft token
# accepted at depth d, so depths 1..239 fit and the depth reads straight off
# the code.

#: No instrumented site claimed this position.
SOURCE_UNKNOWN = 0
#: Freshly sampled primary: the cycle's own target-distribution draw.
SOURCE_PRIMARY = 1
#: All-accept speculative bonus.  Becomes the next cycle's primary.
SOURCE_BONUS = 2
#: Residual correction emitted at an MTP rejection boundary.
SOURCE_CORRECTION = 3
#: Accepted token of a context-copy block round (either lane).
SOURCE_COPY = 4
#: Residual correction closing a context-copy round.
SOURCE_COPY_CORRECTION = 5
#: ``SOURCE_DRAFT_BASE + depth`` == a draft token accepted at that depth.
SOURCE_DRAFT_BASE = 16
#: Largest depth the uint8 encoding can carry.
MAX_DRAFT_DEPTH = 255 - SOURCE_DRAFT_BASE

SOURCE_NAMES: dict[int, str] = {
    SOURCE_UNKNOWN: "unknown",
    SOURCE_PRIMARY: "primary",
    SOURCE_BONUS: "bonus",
    SOURCE_CORRECTION: "correction",
    SOURCE_COPY: "copy",
    SOURCE_COPY_CORRECTION: "copy_correction",
}


def draft_code(depth: int) -> int:
    """The source code for a draft token accepted at ``depth`` (1-based)."""

    depth = int(depth)
    if depth < 1 or depth > MAX_DRAFT_DEPTH:
        raise ValueError(f"draft depth {depth} outside 1..{MAX_DRAFT_DEPTH}")
    return SOURCE_DRAFT_BASE + depth


def source_name(code: int) -> str:
    """Human name for one code; ``draft_d3`` for a depth-3 draft token."""

    code = int(code)
    if code in SOURCE_NAMES:
        return SOURCE_NAMES[code]
    if code > SOURCE_DRAFT_BASE:
        return f"draft_d{code - SOURCE_DRAFT_BASE}"
    return f"code_{code}"


class TokenSourceLog:
    """Commit-order spans for one request, and the receipt block they make."""

    __slots__ = ("enabled", "_spans", "_recorded", "_requests")

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._spans: list[tuple[int, int]] = []
        self._recorded = 0
        self._requests = 0

    # -- lifecycle ---------------------------------------------------------

    def begin_request(self) -> None:
        """Start a fresh record.

        One process serves many requests (the fable drivers run three seeds
        back to back) and the committed stream does not run across the join,
        so the spans are cleared here rather than accumulated.  The receipt is
        read out before the next request starts.
        """

        if not self.enabled:
            return
        self._spans = []
        self._recorded = 0
        self._requests += 1

    # -- emission sites ----------------------------------------------------

    def _mark(self, code: int, count: int = 1) -> None:
        if count <= 0:
            return
        spans = self._spans
        if spans and spans[-1][0] == code:
            # Adjacent runs of one lane collapse, which is what a long
            # context-copy stretch looks like.
            spans[-1] = (code, spans[-1][1] + count)
        else:
            spans.append((code, count))
        self._recorded += count

    def primary(self) -> None:
        """``tokens.append(primary)`` -- the cycle's fresh target draw."""

        if not self.enabled:
            return
        self._mark(SOURCE_PRIMARY)

    def bonus(self) -> None:
        """``tokens.append(bonus)`` on the all-accept path."""

        if not self.enabled:
            return
        self._mark(SOURCE_BONUS)

    def correction(self) -> None:
        """The residual correction at an MTP rejection boundary."""

        if not self.enabled:
            return
        self._mark(SOURCE_CORRECTION)

    def copy_block(self, count: int) -> None:
        """``tokens.extend(_cc_acc)`` / ``tokens.extend(_cb_acc)``.

        Pass the length of the slice that was actually extended, i.e. AFTER
        the round's stop-token truncation, so the span matches the stream.
        """

        if not self.enabled:
            return
        self._mark(SOURCE_COPY, int(count))

    def copy_correction(self) -> None:
        """The residual correction closing a context-copy round."""

        if not self.enabled:
            return
        self._mark(SOURCE_COPY_CORRECTION)

    def draft_run(self, count: int) -> None:
        """``count`` accepted draft tokens, in depth order 1..count."""

        if not self.enabled:
            return
        count = int(count)
        if count > MAX_DRAFT_DEPTH:
            raise ValueError(
                f"accepted draft run of {count} exceeds the uint8 depth "
                f"encoding ({MAX_DRAFT_DEPTH})"
            )
        for depth in range(1, count + 1):
            self._mark(SOURCE_DRAFT_BASE + depth)

    def mtp_commit(self, accepted: int, *, correction: bool) -> None:
        """One ``tokens.extend(committed[1:])`` at a rejection boundary.

        ``committed`` is ``[primary] + draft_tokens[:accepted]`` plus, when
        the cycle emitted one, the residual correction -- and the primary was
        already recorded by :meth:`primary` (or by the :meth:`bonus` /
        :meth:`correction` that became this cycle's ``pending_primary``), so
        only the tail is marked here, in exactly that order.
        """

        if not self.enabled:
            return
        self.draft_run(accepted)
        if correction:
            self.correction()

    # -- read-out ----------------------------------------------------------

    @property
    def recorded_tokens(self) -> int:
        """Tokens the instrumented sites claimed, before any stop trim."""

        return self._recorded

    @property
    def spans(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._spans)

    def expand(self, total: int) -> np.ndarray:
        """The per-token ``uint8`` code array for a stream of ``total`` tokens.

        Spans are laid down in commit order and clipped at ``total``: every
        trim ``generate_mtpk`` performs is a prefix trim, so clipping IS the
        trim.  A stream longer than the recorded spans keeps
        :data:`SOURCE_UNKNOWN` in the tail rather than inventing a lane.
        """

        total = int(total)
        out = np.zeros(max(total, 0), dtype=np.uint8)
        position = 0
        for code, count in self._spans:
            if position >= total:
                break
            take = min(int(count), total - position)
            out[position : position + take] = np.uint8(code)
            position += take
        return out

    def receipt(self, tokens: Sequence[Any]) -> dict[str, Any]:
        """The receipt block for one request's generated ids.

        ``tokens`` is ``GenerationResult.tokens`` -- the FINAL committed
        stream, after every stop trim.
        """

        ids = np.asarray([int(token) for token in tokens], dtype="<u4")
        sources = self.expand(ids.size)
        covered = min(self._recorded, int(ids.size))
        counts: dict[str, int] = {}
        if sources.size:
            values, occurrences = np.unique(sources, return_counts=True)
            for code, occurrence in zip(values.tolist(), occurrences.tolist()):
                counts[source_name(int(code))] = int(occurrence)
        return {
            "schema": SCHEMA,
            # The ids and their digest are always real; `available` says
            # whether the SOURCE column means anything.  A disabled recorder
            # produces an all-`unknown` column, and saying so is the point:
            # a receipt then cannot be mistaken for one that observed the
            # lanes and found nothing.
            "available": bool(self.enabled),
            "tokens": int(ids.size),
            "output_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "output_ids_b64": base64.b64encode(ids.tobytes()).decode("ascii"),
            "token_sources_b64": base64.b64encode(
                sources.tobytes()
            ).decode("ascii"),
            "ids_dtype": "uint32",
            "ids_byteorder": "little",
            "sources_dtype": "uint8",
            "recorded_tokens": int(self._recorded),
            # False means an emission site this module does not know about put
            # tokens in the stream: the tail reads SOURCE_UNKNOWN and the
            # provenance is incomplete.  It is NOT set by a stop trim, which
            # only ever makes `recorded_tokens` exceed `tokens`.
            "complete": bool(covered == int(ids.size)),
            "codes": {
                "unknown": SOURCE_UNKNOWN,
                "primary": SOURCE_PRIMARY,
                "bonus": SOURCE_BONUS,
                "correction": SOURCE_CORRECTION,
                "copy": SOURCE_COPY,
                "copy_correction": SOURCE_COPY_CORRECTION,
                "draft_base": SOURCE_DRAFT_BASE,
            },
            "counts": counts,
        }


def unavailable(reason: str) -> dict[str, Any]:
    """The block a receipt carries when nothing was recorded."""

    return {"schema": SCHEMA, "available": False, "reason": str(reason)}


def decode_receipt(block: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(ids, sources)`` from a receipt block, for offline readers and tests."""

    ids = np.frombuffer(
        base64.b64decode(block["output_ids_b64"]), dtype="<u4"
    )
    sources = np.frombuffer(
        base64.b64decode(block["token_sources_b64"]), dtype=np.uint8
    )
    if ids.size != sources.size:
        raise ValueError(
            f"receipt holds {ids.size} ids but {sources.size} source codes"
        )
    return ids, sources


def sha256_ids(tokens: Sequence[Any]) -> str:
    """``output_ids_sha256`` for a token sequence, without a recorder.

    The digest is over the raw little-endian ``uint32`` bytes, so it is
    reproducible from ``output_ids_b64`` alone and does not depend on any
    textual formatting of the ids.
    """

    return hashlib.sha256(
        np.asarray([int(token) for token in tokens], dtype="<u4").tobytes()
    ).hexdigest()


def _env_enabled() -> bool:
    return (os.environ.get(ENV_FLAG) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_ENABLED = _env_enabled()

#: Module-level singleton.  ``generate_mtpk`` reads ``.enabled`` once per
#: request; ``abba_driver`` sets it directly before it generates.
token_source = TokenSourceLog(_ENABLED)


def is_enabled() -> bool:
    """True when :data:`ENV_FLAG` was set at import."""

    return _ENABLED


__all__ = [
    "ENV_FLAG",
    "MAX_DRAFT_DEPTH",
    "SCHEMA",
    "SOURCE_BONUS",
    "SOURCE_COPY",
    "SOURCE_COPY_CORRECTION",
    "SOURCE_CORRECTION",
    "SOURCE_DRAFT_BASE",
    "SOURCE_NAMES",
    "SOURCE_PRIMARY",
    "SOURCE_UNKNOWN",
    "TokenSourceLog",
    "decode_receipt",
    "draft_code",
    "is_enabled",
    "sha256_ids",
    "source_name",
    "token_source",
    "unavailable",
]
