"""Crossover arithmetic for the Fable 16K sparse-QSA prefill candidate.

Pure Python and pure functions of an environment mapping: no MLX, no device
probe, no import side effects.  ``mtplx.models.qwen4_exp`` binds these once at
construction time; the tests exercise the arithmetic without a GPU.

Why this exists
---------------
The shipped lane gates on ``total_tokens - rows`` -- the EARLIEST query's
history in the chunk -- against a 32,768 crossover.  The production cell is a
16,384-token prompt cut into 8 chunks of 2,048, so that expression takes the
values 0, 2048, ... 14336 and never reaches 32,768: the 2026-08-29 lightning
lane (Metal selector -> ``flash_prefill`` block contract -> NAX
``qsa_prefill_flash``) cannot engage at 16K, not once, at any setting of
``MTPLX_QSA_PREFILL_MIN_CONTEXT`` that the shipped floor allows for the first
two chunks (the floor is 2049 and chunk 1's history is exactly 2048).

``MTPLX_FABLE_QSA_PREFILL_16K=1`` switches the comparison to ``total_tokens``
-- the chunk's own final context -- and, when the operator did not name a
crossover explicitly, drops the default to :data:`CROSSOVER`.  That is the
smallest change that puts every chunk whose selection is not already the full
causal mask on the sparse lane at 16K.

Known prior result (do not re-measure blind)
--------------------------------------------
``.benchmark-artifacts/pr391/rebench3-2760-mlxserve-qsa-prefill-x8k-candidate-seeds-16k-1k.json``
armed ``MTPLX_QSA_PREFILL_MIN_CONTEXT=8192`` +
``MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT=8192`` on the same 16K cell and measured
1,085 prefill tok/s against a matched 1,233 (rebench3-2740, same source commit
and seed) -- a 12% LOSS -- with a changed response digest.  With the history
form that receipt's tree used, 8192 engaged only chunks 4-7; this module's
16K form engages chunks 1-7.  Engaging MORE chunks on a lane that lost is
expected to lose harder, not less: arm this only together with
``MTPLX_QSA_PREFILL_DEBUG=1`` and read the engagement counters.

The receipt's tree already contained the whole 2026-08-29 lightning stack
(d9b1f84b flash kernel, a978a86f tiled scorer, a21f3f91 portable gather tier,
9f0cf04b NAX auto default are all ancestors of its ``source_commit``
e8b39c92), and its gate functions are byte-identical to the shipped ones --
so that arm DID run the lightning lane, not an older compact-QSA path.  The
loss is the crossover itself: ``qsa_prefill_flash`` launches one 32-thread
threadgroup per (query row, KV head) with a 16 KiB threadgroup allocation, so
its per-row cost is fixed at ``block_topk * ratio + ratio - 1`` = 2,051
visible keys and runs far below the ~56 TFLOP/s the dense ``steel_gemm``
lane reaches.  Dense cost grows with T while sparse cost does not, which is
why the shipped 32,768 default is where it starts to pay.

How to A/B it (single flag; digest change is expected -- see below)::

    scripts/fable/abba_window.py \
        --candidate-env MTPLX_QSA_PREFILL=1 \
        --candidate-env MTPLX_FABLE_QSA_PREFILL_16K=1 \
        --candidate-env MTPLX_QSA_PREFILL_DEBUG=1 \
        --prefill-only

Add ``--candidate-env MTPLX_QSA_PREFILL_MIN_CONTEXT=<n>`` and
``--candidate-env MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT=<n>`` to sweep the
crossover (an explicit value still wins over :data:`CROSSOVER`); add
``--candidate-env MTPLX_QSA_PREFILL_GATHER=1`` to route the same block
contract through the portable gathered tier instead of the NAX kernel.

Digest: the selection is bit-identical (the same exact top-512 blocks, from
the same scores), but the sparse consumers reduce in a different order than
dense SDPA, so ``response_token_sha256`` may change.  That is the rounding
class, not a selection change -- rebench3-2760 shows it.  A digest change is
acceptable evidence-wise; a change in which blocks are selected is not.
"""

from __future__ import annotations

from typing import Mapping

__all__ = [
    "CROSSOVER",
    "ENV_FLAG",
    "ENV_FLASH_MIN_CONTEXT",
    "ENV_MIN_CONTEXT",
    "SHIPPED_FLOOR",
    "crossover_from_env",
    "engages",
    "flag_from_env",
    "history_for_gate",
]

ENV_FLAG = "MTPLX_FABLE_QSA_PREFILL_16K"
ENV_MIN_CONTEXT = "MTPLX_QSA_PREFILL_MIN_CONTEXT"
ENV_FLASH_MIN_CONTEXT = "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT"

#: The shipped ``max(2049, ...)`` floor on both crossover knobs.  Below
#: 2049 blocks of 4 tokens the selection IS the full causal mask, so the
#: selector short-circuits before any of this is consulted.
SHIPPED_FLOOR = 2049

#: Crossover the flag installs when the operator named none.  Equal to the
#: shipped floor on purpose: the candidate is "as early as the lane is
#: mathematically distinguishable from dense", not a new tuned threshold.
CROSSOVER = SHIPPED_FLOOR

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


def flag_from_env(environ: Mapping[str, str]) -> bool:
    """Resolve :data:`ENV_FLAG`; raise rather than guess.

    An unparseable value is a configuration error, never a silent "off":
    a candidate arm that quietly measured the control is the one failure
    mode an A/B harness cannot detect afterwards.
    """

    raw = (environ.get(ENV_FLAG) or "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(
        f"{ENV_FLAG} must be one of {accepted}, got {environ.get(ENV_FLAG)!r}"
    )


def crossover_from_env(
    environ: Mapping[str, str], env_key: str, shipped: int
) -> int:
    """The crossover the 16K candidate compares ``total_tokens`` against.

    An explicit ``MTPLX_QSA_PREFILL*_MIN_CONTEXT`` still wins -- the flag
    changes WHICH number the chunk contributes, not the operator's right to
    name the threshold (that is how a crossover sweep stays possible).
    """

    if (environ.get(env_key) or "").strip():
        return int(shipped)
    return CROSSOVER


def history_for_gate(rows: int, total_tokens: int, *, fable_16k: bool) -> int:
    """History the crossover is compared against for one prefill chunk."""

    if fable_16k:
        return int(total_tokens)
    return int(total_tokens) - int(rows)


def engages(
    rows: int, total_tokens: int, *, crossover: int, fable_16k: bool
) -> bool:
    """Whether one chunk's history clears the crossover."""

    return history_for_gate(rows, total_tokens, fable_16k=fable_16k) >= int(
        crossover
    )
