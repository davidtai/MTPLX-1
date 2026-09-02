"""Request-time DECLINE vs install-time RAISE for the armed fable lanes.

The problem this fixes
----------------------
An armed fable flag names a fast path for a SHAPE of request: the pre-scatter
draft read wants a sampled top-k draft; the device draft chain wants
temperature > 0 at depth >= 1; device K20 wants the stock selector.  Each of
them checked those terms in a ``claim`` at the top of ``generate_mtpk`` and
RAISED when the request did not match, on the reasoning that an armed flag
which quietly ran the shipped path would make a benchmark receipt lie about
which code produced the number.

That reasoning is right for a benchmark arm and wrong for a server.  A
benchmark runs one request shape it chose; a server runs whatever arrives.
On 2026-09-02 the composed-decode-stack HumanEval gate launched a server with
``MTPLX_FABLE_DRAFT_K20_PRESCATTER=1`` and the very first request -- HumanEval
is GREEDY -- came back HTTP 500:
``DraftK20PrescatterIneligible: the greedy device chain owns the draft read``.
Every ABBA window had been temperature 1, so no window ever reached it.  The
flag was not broken; its *contract* was: a precondition that raises turns
every ineligible request into an outage.

The contract
------------
Split the checks by whether ANY request could be served:

* **Install-time contract violation -> RAISE.**  The armed flag cannot work in
  this process at all: FR-Spec is not installed, the head is not on the live
  draft route, the ranked table is the wrong width or not ascending, the head
  has no capture surface, the pinned NumPy is missing.  Every request would
  fail identically, so failing the first one loudly is the honest report --
  this is a deployment error, not a request.
* **Request-time ineligibility -> DECLINE.**  This particular request's shape
  is not served: greedy, penalties, a steering guard, a correction cache, a
  reranker, adaptive width, target-prefix verify, a competing owner of the
  same chain.  The shipped path serves it correctly and by construction
  produces the same tokens, so the lane stands aside, logs one line, counts
  it, and stamps the receipt.

A decline is NOT a silent fallback.  ``installed`` stays False, the receipt
carries ``declined`` (a stable key) and ``declined_detail`` (the sentence),
and :data:`DECLINE_COUNTS` accumulates per flag and reason for the life of the
process, so an operator can see how often a flag stood aside and why.

Benchmarks keep their fail-closed guarantee
-------------------------------------------
``MTPLX_FABLE_STRICT_CLAIMS=1`` turns every decline back into the lane's own
``*Ineligible`` exception.  A measured arm that must PROVE its lane ran sets
it; a server never does.  Read once at import, like every other fable gate.

NO device work happens here, and nothing in this module imports MLX.
"""

from __future__ import annotations

import os
import sys
from typing import NoReturn


STRICT_ENV = "MTPLX_FABLE_STRICT_CLAIMS"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


#: Read exactly once, at import.
_STRICT = _env_truthy(STRICT_ENV)


def strict_claims() -> bool:
    """True when ``MTPLX_FABLE_STRICT_CLAIMS`` armed this process at import."""

    return _STRICT


def _configure_for_test(strict: bool) -> None:
    """Flip the import-time strict gate (tests only)."""

    global _STRICT
    _STRICT = bool(strict)


#: ``flag -> reason key -> count``, cumulative for the life of the process.
#: A server reads this to see how often an armed lane stood aside; the tests
#: read it to prove a decline was recorded rather than swallowed.
DECLINE_COUNTS: dict[str, dict[str, int]] = {}

#: ``(flag, reason key)`` pairs already logged.  One line per reason per
#: process: a 164-problem eval must not print 164 identical warnings.
_LOGGED: set[tuple[str, str]] = set()


class ClaimDeclined(Exception):
    """Internal control flow: this request's shape is not served.

    Raised by :func:`decline` from anywhere inside a claim body and caught by
    :func:`declined_receipt` at the claim's boundary.  It never escapes a
    claim -- callers see ``None`` and a receipt, or, under strict claims, the
    lane's own ``*Ineligible``.
    """

    def __init__(self, key: str, detail: str) -> None:
        super().__init__(detail)
        self.key = str(key)
        self.detail = str(detail)


def decline(key: str, detail: str) -> NoReturn:
    """Stand aside for this request.  ``key`` is the stable receipt reason."""

    raise ClaimDeclined(key, detail)


def declined_receipt(
    flag: str,
    exc: ClaimDeclined,
    *,
    ineligible: type[BaseException],
) -> dict[str, object]:
    """Record one decline and return its receipt.

    Under ``MTPLX_FABLE_STRICT_CLAIMS`` the decline is re-raised as ``flag``'s
    own ineligibility error instead, so a measured arm still fails closed.
    """

    if _STRICT:
        raise ineligible(
            f"{flag}: {exc.detail} "
            f"[{STRICT_ENV}=1 turns request-time declines into failures]"
        ) from None
    counts = DECLINE_COUNTS.setdefault(flag, {})
    counts[exc.key] = counts.get(exc.key, 0) + 1
    if (flag, exc.key) not in _LOGGED:
        _LOGGED.add((flag, exc.key))
        # stderr, not `logging`: this is the same channel the lanes' own
        # install/engagement receipts use (`[frspec] install report:`,
        # `[MTPLX_FABLE_GRAPH_BUILD_OVERLAP] armed:`), so a decline lands in
        # `server.log` next to them without depending on a logging config the
        # server does not set.
        print(
            f"[{flag}] declined ({exc.key}): {exc.detail} -- this request "
            f"runs the shipped path; set {STRICT_ENV}=1 to fail closed "
            "instead",
            file=sys.stderr,
            flush=True,
        )
    return {
        "installed": False,
        "declined": exc.key,
        "declined_detail": exc.detail,
        "declines": dict(counts),
    }


def decline_counts(flag: str) -> dict[str, int]:
    """This process's decline tally for ``flag`` (a copy)."""

    return dict(DECLINE_COUNTS.get(flag, {}))


def reset_for_test() -> None:
    """Drop the tallies and the log-once memory (tests only)."""

    DECLINE_COUNTS.clear()
    _LOGGED.clear()
