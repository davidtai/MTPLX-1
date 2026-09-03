#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# ONE function in this file touches the GPU: `build_replay_hooks`.  It imports
# MLX lazily, inside its own body, and everything else here -- the trajectory
# reconstruction, the replay orchestration, the scorer, the statistics and the
# report -- is pure NumPy and is exercised by
# tests/test_fable_shadow_draft_harness.py with no MLX in the process.
# `score` / `report` never import MLX at all, so a captured shadow-rows file
# can be re-scored on any box, off the lock, as many times as you like.
# ---------------------------------------------------------------------------
"""Judge a change to the DRAFT PROPOSAL on acceptance, offline, without a live A/B.

The problem this exists to solve
--------------------------------
A live A/B on acceptance costs a full guarded run per arm and still cannot
resolve the effects we care about: the per-run acceptance spread is +-4.2%
(1 sigma, H Sec.7), so a +2% proposal change is invisible under a 3-seed A/B.
Meanwhile a whole class of candidate changes -- indexer reuse across draft
depths, row K-D2, any cheaper way to compute the *same* draft step -- change
``q`` and nothing else.  They do not touch the target's ``p``, they do not
touch the accept law, and they do not touch the emitted-token distribution's
correctness argument.  For that class the honest question is one number per
depth, and the number is computable from rows on disk::

    alpha_d = sum_y min(p_d(y), q_d(y))  =  E_{x ~ q_d}[min(1, p_d(x)/q_d(x))]

That is the Leviathan-Chen acceptance probability with **both** noise sources
integrated out -- the accept coin and the draw of the drafted token.  Scored
against the *same* logged ``p_d`` for the stock chain and the candidate chain,
the difference is a paired, common-random-numbers estimate whose standard error
over ~1,110 windows is one to two orders of magnitude tighter than a live A/B's.

How it works
------------
Two phases, deliberately split so the expensive one runs once:

**capture** (needs the GPU; ``--capture-to``)
    Replays a ``MTPLX_FABLE_K20_LOG`` trajectory through the model.  The log
    records, per verify window, the window's ``primary``, the full draft chain
    ``draft_tokens``, how many of them were accepted, the correction/bonus
    token that was emitted, and -- since 2026-09-02 -- the ``carry``: the
    tokens a lane with no verify window of its own (a ``context_copy`` block
    round) committed between two windows.  Together those are the whole
    committed stream, so the trajectory is reconstructible from the log alone
    (:func:`segment_windows`).  A log written before the carry accounting is
    NOT reconstructible whenever the copy lane fired, and
    :func:`segment_windows` refuses it rather than replaying a stream with
    holes in it.
    The replay re-runs the prompt, then walks that committed stream window by
    window, and at every window runs the draft chain **once per variant** from
    the identical hidden state, **teacher-forced to the logged draft tokens**.
    Each chain's rows land in a shadow-rows ``.npz``.

**score** (pure NumPy, no GPU, the default)
    Loads the K20 log and the shadow rows and reports, per variant and per
    depth, ``alpha_d`` above, the reach ladder, ``E[tokens/window]``, and the
    realised accept count under the logged uniform tape -- with paired standard
    errors against the stock chain, i.i.d. and moving-block bootstrap.

Why teacher forcing is not a shortcut but the definition
--------------------------------------------------------
``p_d`` is ``p(. | primary, x_1 .. x_{d-1})``: the target row logged at depth
``d`` is conditioned on the tokens the **logged** chain drafted.  If a
candidate chain were left free-running and drafted a different token at depth
1, the logged ``p_2`` would be the wrong distribution to verify its depth-2
proposal against, and every number past the divergence would be fiction.  So
the candidate chain is forced onto the logged tokens: only the **row** ``q_d``
differs, ``p_d`` stays exactly the distribution that row is verified against,
and every depth is scored exactly.

Forcing does not hide the divergence, it isolates it.  The report carries
``P(diverge)`` per depth -- the probability that the candidate would have
drafted a different token, computed from its own row and, where the layout
recorded them, the logged draft uniforms.  Read that as the scope of the
extrapolation, not as a correction to it: a candidate with ``P(diverge)`` near
zero is measured, one with ``P(diverge)`` of 0.3 has a depth-2 and depth-3
number that describes a chain the model would often not have taken.

What this harness can and cannot judge
--------------------------------------
**Can** -- a change that alters only the draft proposal ``q``:

* the drafter's arithmetic, fusion, caching or scheduling
  (``MTPLX_FABLE_INDEXER_REUSE``, row K-D2, compiled draft support, a cheaper
  QSA indexer, a different MTP hidden variant),
* draft-side shaping (draft temperature, draft top-k) -- though
  ``offline_draft_temperature.py`` already does that one from the log alone,
  with no replay and no GPU at all, and should be preferred for it,
* anything whose entire claim is "same output law, cheaper or better ``q``".

**Cannot** -- and it does not pretend to:

* anything that changes ``p``: the target model, its quantisation, its shaping
  (``temperature`` / ``top_p`` / ``top_k`` on the *target* sampler), the verify
  graph's numerics.  The logged ``p`` rows would no longer be the rows the
  candidate faces, and nothing here would notice.
* anything that changes the **accept law**: block verification, a different
  clip, a residual change.  Score those with
  ``scripts/fable/offline_block_verification.py``, which replays laws against
  fixed rows -- the mirror image of this harness.
* anything that changes the **window shape**: depth, adaptive width, gated
  stop, a 4th draft step.  ``scripts/fable/offline_depth4_gate.py`` owns the
  depth-4 question.
* **wall time.**  Acceptance is not tok/s.  A candidate that wins ``alpha`` and
  costs 2 ms/window loses.  This harness prices nothing; it hands you the
  acceptance term of the product and the ledger's ``ms/window`` supplies the
  rest.
* a change that only fires **off** the logged trajectory -- a different prompt,
  a different sampler, a longer context.  Three seeds of one cell is three
  seeds of one cell.
* a **greedy** run.  ``temperature <= 0`` builds no distributions, so there are
  no rows to score; those windows are skipped and counted.

The self-check that makes the rest trustworthy
----------------------------------------------
Variant index 0 is always ``stock``: the *same* proposal path the logged run
used, re-run through the replay.  Its rows must reproduce the logged ``q``
rows.  :func:`score` measures that as a total-variation distance per row and
**withholds the verdict** above ``--fidelity-tol``.  A replay whose stock chain
does not reproduce the log is a broken replay -- wrong hidden state, wrong
cache offset, wrong shaping, wrong id space -- and its candidate numbers mean
nothing.  The same gate catches the opposite failure: a candidate whose rows
are bit-identical to stock on every window did not arm, which is what happens
when the variant's flag is read at import or at runtime construction rather
than per draft call.

Usage::

    # capture (guarded, GPU) then score, in one command
    python scripts/fable/shadow_draft_harness.py rows.npz \\
        --capture-to shadow.npz --model <path> \\
        --variant indexer-reuse=MTPLX_FABLE_INDEXER_REUSE=1

    # re-score an existing capture, anywhere, no GPU, no lock
    python scripts/fable/shadow_draft_harness.py rows.npz --rows shadow.npz --budget
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from scripts.fable.offline_block_verification import (
    STOCK_LAYOUTS,
    load_log,
    prepared_row,
    window_uniforms,
)
from scripts.fable.offline_depth4_gate import overlap

#: Log row width, matching ``mtplx.fable_k20_log.K20``.
K20 = 20

#: Variant 0, always.  The re-drafted control whose rows must reproduce the log.
STOCK_VARIANT = "stock"

#: Padding convention, mirroring ``fable_k20_log._row_arrays``: ids well above
#: any vocabulary, probability 0, so :func:`prepared_row` drops them on load.
PAD_ID_BASE = 0xFFFFFFFF

#: Above this total-variation distance the re-drafted stock rows are not the
#: logged ones and the replay is broken.  Loose enough for the float32
#: non-determinism of a re-run reduction, far tighter than any effect worth
#: measuring (a +2% acceptance change moves a row's TV by ~1e-2).
FIDELITY_TOL = 1e-6

#: At or below this a candidate's rows ARE the stock rows: it did not arm.
ARMED_TOL = 0.0

#: Cost-model defaults, from L Sec.3's retained frame and L Sec.D's draft-step
#: row.  Every one is an ASSUMPTION, printed with the estimate, not something
#: this harness measures.
BUDGET_MS_PER_WINDOW = 38.7
BUDGET_CHAIN_MS = 5.0
BUDGET_PREFILL_S = 4.0

#: Moving-block bootstrap defaults for the clustered standard error.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK = 32
BOOTSTRAP_SEED = 20260901


# ---------------------------------------------------------------------------
# Proposal variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalVariant:
    """One draft-proposal arm: a name and how to arm it around a chain call.

    ``env`` is applied with :func:`variant_env_scope` for the duration of a
    single draft-chain call and removed afterwards, so both arms run in one
    process from one hidden state and the comparison stays paired.  That works
    for a flag the draft path reads **per call**.

    A flag read at *import* (``_ENABLED = _env_truthy(...)`` at module scope)
    or at *runtime construction* (``mtplx.runtime.load`` branching on it)
    cannot be flipped this way, and setting it here would silently produce two
    identical chains.  :func:`score` catches exactly that and says ``DID NOT
    ARM``.  For those flags supply ``call`` -- a zero-argument factory
    returning a context manager that reconfigures the path itself -- or
    capture two shadow-rows files from two processes and score both against the
    same log.
    """

    name: str
    env: Mapping[str, str] = field(default_factory=dict)
    call: Callable[[], Any] | None = None

    @property
    def is_stock(self) -> bool:
        return self.name == STOCK_VARIANT


def parse_variant(spec: str) -> ProposalVariant:
    """``NAME=KEY=VAL[,KEY=VAL...]`` -> a :class:`ProposalVariant`.

    ``NAME`` may not be ``stock``: the stock arm is the harness's own control
    and is prepended to every run, never configured.
    """

    name, separator, assignments = spec.partition("=")
    name = name.strip()
    if not name or not separator or not assignments.strip():
        raise ValueError(
            f"variant {spec!r} must be NAME=KEY=VALUE[,KEY=VALUE]; the stock "
            "arm is added automatically and needs no spec"
        )
    if name == STOCK_VARIANT:
        raise ValueError(
            "variant name 'stock' is reserved for the harness's own control arm"
        )
    env: dict[str, str] = {}
    for pair in assignments.split(","):
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"variant {spec!r} has a malformed KEY=VALUE in {pair!r}")
        env[key.strip()] = value.strip()
    return ProposalVariant(name=name, env=env)


def load_variant_module(spec: str) -> list[ProposalVariant]:
    """``dotted.module:factory`` -> the variants that factory returns.

    The escape hatch for a proposal change an env var around a call cannot arm.
    The factory takes no arguments and returns a sequence of
    :class:`ProposalVariant`.
    """

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"variant module {spec!r} must be dotted.module:factory")
    from importlib import import_module

    factory = getattr(import_module(module_name), attribute)
    variants = list(factory())
    for variant in variants:
        if not isinstance(variant, ProposalVariant):
            raise TypeError(f"{spec} returned {variant!r}, not a ProposalVariant")
        if variant.is_stock:
            raise ValueError(f"{spec} returned a variant named 'stock'")
    return variants


@contextmanager
def variant_env_scope(env: Mapping[str, str]) -> Iterator[None]:
    """Apply ``env`` for the body and restore the previous values exactly.

    An absent key is restored to absent rather than to the empty string, so a
    downstream ``_env_truthy`` sees precisely what it saw before.
    """

    previous: dict[str, str | None] = {}
    try:
        for key, value in env.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def variant_scope(variant: ProposalVariant) -> Any:
    """The context manager that arms ``variant`` for one chain call."""

    if variant.call is not None:
        return variant.call()
    if variant.env:
        return variant_env_scope(variant.env)
    return nullcontext()


# ---------------------------------------------------------------------------
# Trajectory reconstruction -- pure, from the log alone.
# ---------------------------------------------------------------------------


def window_emission(log: Mapping[str, np.ndarray], index: int) -> list[int]:
    """The tokens window ``index`` appends to the emitted stream.

    ``generate_mtpk`` commits ``[primary] + draft_tokens`` to the model but
    extends the emitted stream only by the accepted drafts plus the
    correction/bonus: ``primary`` entered the stream in the *previous* window
    as that window's ``selected_token``.  So this is the accepted prefix plus
    at most one token -- exactly ``Outcome.tokens`` in
    ``offline_block_verification``.
    """

    accepted = int(log["accepted"][index])
    tokens = [int(token) for token in log["draft_tokens"][index][:accepted]]
    if bool(log["selected_present"][index]):
        tokens.append(int(log["selected_token"][index]))
    return tokens


@dataclass(frozen=True)
class Segment:
    """One request's contiguous run of windows, and its committed stream.

    ``tokens`` is the full generated continuation: the first window's
    ``primary`` (the token prefill produced) followed by every window's
    emission.  The replay teacher-forces exactly this stream, so the hidden
    state each draft chain starts from is the one the logged run had.
    """

    index: int
    start: int
    stop: int
    tokens: tuple[int, ...]

    @property
    def windows(self) -> int:
        return self.stop - self.start


class TrajectoryGapError(RuntimeError):
    """The log is missing tokens the run committed; it cannot be replayed.

    Raised when the emitted stream reconstructed from the rows of ONE request
    does not join up: window ``c + 1``'s ``primary`` is not the token window
    ``c`` left at the end of the stream, and no ``carry`` accounts for the
    difference.  That is a hole in the log, not a request boundary, and a
    replay teacher-forced on the truncated stream would run the model from the
    wrong hidden state at every window after it.
    """


def window_carry(log: Mapping[str, np.ndarray], index: int) -> list[int]:
    """Tokens committed AFTER window ``index`` by a lane that wrote no row.

    ``mtplx/context_copy.py`` block rounds commit a verbatim slice of the
    prompt (plus the residual correction, plus the freshly sampled primary when
    the block was accepted whole) in their own verify forward, with no K20 row.
    ``carry_len`` / ``carry_tokens`` are the logger's record of them; an older
    log has neither column and this returns ``[]`` for every window.
    """

    lengths = log.get("carry_len")
    if lengths is None:
        return []
    count = int(lengths[index])
    if count <= 0:
        return []
    return [int(token) for token in log["carry_tokens"][index][:count]]


def request_ids(log: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """Per-window request id from the PCG64 stream, or ``None`` when unusable.

    A stock log records the window's PCG64 state as four uint64 words --
    ``state`` high/low then ``inc`` high/low.  ``inc`` is the *stream* id, a
    property of the ``Generator`` and therefore of the request: every window of
    one ``generate_mtpk`` call carries the same one, and a second request
    through the same process gets a different one from its own seeding.  That
    is an exact request marker, and it does not care what the emitted stream
    does -- which is the whole point, because a lane that commits without a row
    breaks the emitted stream's continuity without ending the request.

    ``None`` only when the column is not there at all: the PR391 lane and the
    test stubs leave it zero.  A **constant** stream is an answer, not a
    refusal -- it says "one request" -- and that is the safe reading: were a
    driver ever to reuse one ``Generator`` across requests, the segments would
    merge and the continuity check would then raise at the join, which is loud,
    where guessing from the token stream is silent and wrong.  A stream id that
    reappears after a different one is not a request id (interleaved requests),
    so that is refused.
    """

    state = log.get("rng_state")
    if state is None:
        return None
    words = np.asarray(state)
    if words.ndim != 2 or words.shape[1] < 4:
        return None
    stream = words[:, 2:4]
    if not stream.any():
        return None
    changed = np.zeros(len(stream), dtype=bool)
    changed[0] = True
    changed[1:] = (stream[1:] != stream[:-1]).any(axis=1)
    firsts = stream[changed]
    if len(np.unique(firsts, axis=0)) != len(firsts):
        return None
    return (np.cumsum(changed) - 1).astype(np.int64)


def segment_windows(log: Mapping[str, np.ndarray]) -> list[Segment]:
    """Split a multi-request log into per-request segments.

    A K20 log written by a driver that runs three seeds in one process holds
    all three trajectories concatenated.  Where the request boundary is is a
    property of the *run*, not of the emitted stream, and the stock layout
    records it: :func:`request_ids` reads the window's PCG64 stream id, which
    changes exactly at a request boundary and nowhere else.  That is the split.

    Only when that column cannot answer -- the PR391 lane and the test stubs
    leave it zero -- does this fall back to reading the emitted stream, ending
    a request where ``selected_present`` is 0 (a stop token) or where the next
    window's ``primary`` is not this window's ``selected_token``.

    Either way the reconstructed stream is then **checked**, per segment: every
    window's ``primary`` must be the token the previous window left at the end
    of the stream, or the gap must be accounted for by a logged
    :func:`window_carry`.  An unexplained break is a hole in the log --
    :class:`TrajectoryGapError` -- and never a segment boundary.  Pass
    ``--expect-segments`` to assert the count you know you ran on top of that.
    """

    cycles = int(log["draft_tokens"].shape[0])
    if not cycles:
        return []
    primary = np.asarray(log["primary"], dtype=np.int64)
    selected = np.asarray(log["selected_token"], dtype=np.int64)
    present = np.asarray(log["selected_present"], dtype=np.uint8)

    ids = request_ids(log)
    bounds: list[int] = [0]
    if ids is not None:
        bounds.extend(index for index in range(1, cycles) if ids[index] != ids[index - 1])
    else:
        bounds.extend(
            index
            for index in range(1, cycles)
            if not bool(present[index - 1])
            or int(primary[index]) != int(selected[index - 1])
        )
    bounds.append(cycles)

    segments: list[Segment] = []
    breaks: list[tuple[int, int, int]] = []
    for position in range(len(bounds) - 1):
        start, stop = bounds[position], bounds[position + 1]
        tokens: list[int] = [int(primary[start])]
        for index in range(start, stop):
            if index > start and int(primary[index]) != tokens[-1]:
                breaks.append((index, tokens[-1], int(primary[index])))
            tokens.extend(window_emission(log, index))
            tokens.extend(window_carry(log, index))
        segments.append(
            Segment(index=len(segments), start=start, stop=stop, tokens=tuple(tokens))
        )
    if breaks:
        raise TrajectoryGapError(_gap_message(log, breaks, len(segments)))
    return segments


def _gap_message(
    log: Mapping[str, np.ndarray],
    breaks: Sequence[tuple[int, int, int]],
    segments: int,
) -> str:
    shown = ", ".join(
        f"window {index} expects {expected} but its primary is {found}"
        for index, expected, found in breaks[:5]
    )
    if len(breaks) > 5:
        shown += f", ... ({len(breaks)} in all)"
    if log.get("carry_len") is None:
        cause = (
            "This log has no `carry_len` column, so it was written before the "
            "logger accounted for lanes that commit tokens WITHOUT a K20 row "
            "-- mtplx/context_copy.py block rounds are one, and they commit "
            "the accepted slice, the residual correction and the next freshly "
            "sampled primary between two windows. Those tokens are simply not "
            "in this file and no reconstruction can recover them. Re-record "
            "the log with a build that carries mtplx/fable_k20_log.py's "
            "`carry` accounting, or with the copy lane off "
            "(MTPLX_CONTEXT_COPY=0)."
        )
    else:
        cause = (
            "This log DOES carry `carry_len`, so an unaccounted break means "
            "the record itself is inconsistent -- a lane committed tokens "
            "without calling K20RowLog.carry, or the rows were reordered. Do "
            "not replay it."
        )
    return (
        f"{len(breaks)} window(s) of {segments} reconstructed request "
        f"segment(s) do not join up: {shown}. {cause}"
    )


def segment_of_window(segments: Sequence[Segment], index: int) -> int:
    """Segment id owning window ``index``; ``-1`` when none does."""

    for segment in segments:
        if segment.start <= index < segment.stop:
            return segment.index
    return -1


# ---------------------------------------------------------------------------
# Shadow rows: the capture phase's on-disk product.
# ---------------------------------------------------------------------------


def pad_row(
    ids: Sequence[int], probs: Sequence[float], *, width: int = K20
) -> tuple[np.ndarray, np.ndarray]:
    """One support padded to ``width`` with zero-probability sentinel ids."""

    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if ids.size != probs.size:
        raise ValueError("shadow row has mismatched ids/probs")
    if ids.size > width:
        raise ValueError(f"shadow row support {ids.size} exceeds width {width}")
    padding = width - ids.size
    out_ids = np.concatenate(
        (
            ids.astype(np.uint32, copy=False),
            np.array([PAD_ID_BASE - n for n in range(padding)], dtype=np.uint32),
        )
    )
    out_probs = np.concatenate((probs, np.zeros(padding, dtype=np.float64)))
    return out_ids, out_probs


@dataclass
class ShadowRows:
    """Every variant's draft rows for every replayed window.

    ``ids``/``probs`` are ``[C, V, D, K]``; ``valid`` is ``[C, V, D]``;
    ``tokens`` is ``[C, V, D]`` and, in the only mode this harness replays,
    equals the logged ``draft_tokens`` for every variant -- stored anyway so a
    future free-running capture can never be mistaken for this one.
    """

    variants: list[str]
    ids: np.ndarray
    probs: np.ndarray
    valid: np.ndarray
    tokens: np.ndarray
    mode: str = "forced"
    source: str = ""
    variant_env: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def cycles(self) -> int:
        return int(self.ids.shape[0])

    @property
    def depth(self) -> int:
        return int(self.ids.shape[2])

    def variant_index(self, name: str) -> int:
        return self.variants.index(name)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            shadow_variants=np.asarray(self.variants),
            shadow_ids=self.ids,
            shadow_probs=self.probs,
            shadow_valid=self.valid,
            shadow_tokens=self.tokens,
            shadow_mode=np.asarray(self.mode),
            shadow_source=np.asarray(self.source),
            shadow_variant_env=np.asarray(json.dumps(self.variant_env, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str) -> "ShadowRows":
        with np.load(path, allow_pickle=False) as handle:
            data = {key: handle[key] for key in handle.files}
        missing = [
            key
            for key in (
                "shadow_variants",
                "shadow_ids",
                "shadow_probs",
                "shadow_valid",
            )
            if key not in data
        ]
        if missing:
            raise KeyError(
                f"{path} is missing {missing}; was it written by "
                "shadow_draft_harness --capture-to?"
            )
        return cls(
            variants=[str(name) for name in data["shadow_variants"]],
            ids=data["shadow_ids"],
            probs=data["shadow_probs"],
            valid=data["shadow_valid"],
            tokens=data.get(
                "shadow_tokens", np.zeros(data["shadow_valid"].shape, dtype=np.uint32)
            ),
            mode=str(data["shadow_mode"]) if "shadow_mode" in data else "forced",
            source=str(data["shadow_source"]) if "shadow_source" in data else "",
            variant_env=(
                json.loads(str(data["shadow_variant_env"]))
                if "shadow_variant_env" in data
                else {}
            ),
        )


def empty_shadow_rows(
    *, cycles: int, variants: Sequence[str], depth: int, width: int = K20
) -> ShadowRows:
    return ShadowRows(
        variants=list(variants),
        ids=np.zeros((cycles, len(variants), depth, width), dtype=np.uint32),
        probs=np.zeros((cycles, len(variants), depth, width), dtype=np.float64),
        valid=np.zeros((cycles, len(variants), depth), dtype=np.uint8),
        tokens=np.zeros((cycles, len(variants), depth), dtype=np.uint32),
    )


# ---------------------------------------------------------------------------
# Replay bookkeeping -- pure, and the reason the commit protocol is testable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitStep:
    """One forward-and-commit the replay owes the model.

    A window is one step; a window followed by a ``carry`` is two, because the
    carried tokens went through their own verify forward in the logged run and
    the replay has to advance the cache over them before the next window's
    primary means anything.

    ``fed`` is what the forward consumes, ``keep`` how many of its rows stay
    committed, ``offset`` the absolute cache offset afterwards, ``hidden_row``
    which row becomes the next chain's starting hidden, and ``primary`` the
    token left DEFERRED -- its KV is computed by whichever forward runs next,
    exactly as ``generate_mtpk`` defers a correction.  So ``offset`` counts
    every emitted token except that one.

    ``primary`` is ``-1`` when the request ENDED in this step (the window
    emitted nothing after its accepted prefix): no token is deferred, nothing
    follows, and a ``-1`` fed to a forward is a loud failure rather than a
    plausible-looking wrong token.
    """

    NONE = -1

    kind: str
    fed: tuple[int, ...]
    keep: int
    offset: int
    hidden_row: int
    primary: int


def commit_steps(
    window: Mapping[str, Any], *, offset: int, primary: int
) -> list[CommitStep]:
    """The steps ``advance`` owes for ``window``, in order.

    ``primary`` is the deferred token the replay is holding when the window
    starts -- ``window['primary']`` on a sound trajectory, and the carry step
    below is why it has to be threaded rather than re-read.
    """

    fed = (int(window["primary"]), *(int(token) for token in window["draft_tokens"]))
    accepted = int(window["accepted"])
    keep = accepted + 1  # the primary plus the accepted drafts
    if keep > len(fed):
        raise ValueError(
            f"window {window['index']} accepted {accepted} of {len(fed) - 1} drafts"
        )
    present = bool(window["selected_present"])
    steps = [
        CommitStep(
            kind="window",
            fed=fed,
            keep=keep,
            offset=offset + keep,
            hidden_row=accepted,
            primary=int(window["selected_token"]) if present else CommitStep.NONE,
        )
    ]
    carry = [int(token) for token in window.get("carry", ())]
    if not carry or not present:
        # No carry, or the request ended in this window (nothing was emitted
        # after the accepted prefix, so there is no selection to feed and
        # nothing follows the gap).
        return steps
    # The carry's last token is the NEXT window's primary and stays deferred
    # like a selection; the rest are committed here, led by this window's own
    # selection, which the window step deferred.
    carry_fed = (steps[0].primary, *carry[:-1])
    steps.append(
        CommitStep(
            kind="carry",
            fed=carry_fed,
            keep=len(carry_fed),
            offset=steps[0].offset + len(carry_fed),
            hidden_row=len(carry_fed) - 1,
            primary=carry[-1],
        )
    )
    return steps


def untrimmable_entries(cache: Sequence[Any]) -> list[int]:
    """Indices of cache entries that cannot roll back by trimming an offset.

    A hybrid model's recurrent (GDN) state is one of these: ``mtplx`` builds
    ``ArraysCache`` for every linear layer of ``qwen4_exp`` and ``QSACache``
    for every full-attention layer, and only the latter trims.  A replay that
    commits a verified prefix by trimming offsets therefore works on a
    KV-only model and cannot work here -- which is what
    ``_trim_cache_to_offset`` refusing a NO-OP trim after window 0 of the W51
    capture meant.  The commit has to go through
    ``gdn_capture.commit_captured_prefix`` instead, and this is the predicate
    that says so.
    """

    out: list[int] = []
    for index, entry in enumerate(cache):
        is_trimmable = getattr(entry, "is_trimmable", None)
        if not callable(is_trimmable) or not bool(is_trimmable()):
            out.append(index)
    return out


def trimmable_offsets(cache: Sequence[Any]) -> list[int]:
    """``offset`` of every trimmable entry -- the replay's commit postcondition."""

    out: list[int] = []
    for entry in cache:
        is_trimmable = getattr(entry, "is_trimmable", None)
        if not callable(is_trimmable) or not bool(is_trimmable()):
            continue
        offset = getattr(entry, "offset", None)
        if offset is not None:
            out.append(int(offset))
    return out


# ---------------------------------------------------------------------------
# Replay orchestration -- pure, every device-touching piece injected.
# ---------------------------------------------------------------------------


class ReplayHooks:
    """The calls the replay needs from a model, and nothing else.

    :func:`build_replay_hooks` is the only implementation that touches MLX;
    the tests drive :func:`replay_windows` with a stub, which is how the
    ordering contract below is checked without a GPU.

    Contract, per segment, per window, in this order:

    ``start_segment(segment)``
        prefill the prompt and position the state so that the next
        ``draft_rows`` call starts from the hidden state that produced
        ``segment.tokens[0]`` -- the first window's ``primary``.
    ``draft_rows(variant=..., forced_tokens=...)``
        run ``len(forced_tokens)`` draft steps from the *current* window's
        hidden state and primary, feeding ``forced_tokens[d - 1]`` as step
        ``d``'s source token instead of whatever this chain would have drafted,
        and return one shaped ``(ids, probs)`` row per depth.  Called once per
        variant from the identical state; the implementation MUST restore the
        MTP cache offset it found, or the second call is not paired with the
        first.
    ``advance(window=...)``
        run the window's verify forward over ``[primary] + draft_tokens`` and
        commit the primary plus ``accepted`` drafts, then do the same for
        ``window['carry']`` -- the tokens a lane with no K20 row of its own
        (a context-copy block round) put in the stream before the next
        window's primary -- and leave the state on the next window.
        :func:`commit_steps` is that schedule, and is pure.
    """

    def start_segment(self, segment: Segment) -> None:  # pragma: no cover - iface
        raise NotImplementedError

    def draft_rows(
        self, *, variant: ProposalVariant, forced_tokens: Sequence[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:  # pragma: no cover - iface
        raise NotImplementedError

    def advance(self, *, window: Mapping[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - iface
        return None


def window_record(log: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    """The window fields the replay needs, as plain Python."""

    depth = int(log["draft_tokens"].shape[1])
    return {
        "index": int(index),
        "primary": int(log["primary"][index]),
        "draft_tokens": [int(token) for token in log["draft_tokens"][index][:depth]],
        "accepted": int(log["accepted"][index]),
        "selected_token": int(log["selected_token"][index]),
        "selected_present": bool(log["selected_present"][index]),
        "emission": window_emission(log, index),
        # Tokens a lane committed after this window with no row of its own
        # (context-copy block rounds).  They belong to the stream BEFORE the
        # next window's primary, so `advance` must commit them too.
        "carry": window_carry(log, index),
    }


def replay_windows(
    log: Mapping[str, np.ndarray],
    hooks: ReplayHooks,
    variants: Sequence[ProposalVariant],
    *,
    segments: Sequence[Segment] | None = None,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ShadowRows:
    """Walk the logged trajectory, drafting every variant at every window.

    Pure orchestration: no MLX, no arrays beyond the ones it fills.  The whole
    reason the ordering contract above is testable is that this loop lives here
    and the device work lives behind ``hooks``.
    """

    if not variants or not variants[0].is_stock:
        raise ValueError("variant 0 must be the stock control arm")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate variant names: {names}")

    segments = list(segments if segments is not None else segment_windows(log))
    cycles = int(log["draft_tokens"].shape[0])
    depth = int(log["draft_tokens"].shape[1])
    if limit is not None:
        cycles = min(cycles, int(limit))
    shadow = empty_shadow_rows(cycles=cycles, variants=names, depth=depth)
    shadow.variant_env = {
        variant.name: dict(variant.env) for variant in variants if variant.env
    }

    done = 0
    for segment in segments:
        if segment.start >= cycles:
            break
        hooks.start_segment(segment)
        for index in range(segment.start, min(segment.stop, cycles)):
            window = window_record(log, index)
            forced = window["draft_tokens"]
            for position, variant in enumerate(variants):
                rows = hooks.draft_rows(variant=variant, forced_tokens=forced)
                if len(rows) != depth:
                    raise RuntimeError(
                        f"variant {variant.name!r} returned {len(rows)} rows at "
                        f"window {index}, expected {depth}"
                    )
                for level, (ids, probs) in enumerate(rows):
                    padded_ids, padded_probs = pad_row(ids, probs)
                    shadow.ids[index, position, level] = padded_ids
                    shadow.probs[index, position, level] = padded_probs
                    shadow.valid[index, position, level] = 1
                    shadow.tokens[index, position, level] = np.uint32(forced[level])
            hooks.advance(window=window)
            done += 1
            if progress is not None:
                progress(done, cycles)
    hooks.close()
    return shadow


# ---------------------------------------------------------------------------
# THE ONE GPU FUNCTION.  Everything above and below this block is pure NumPy.
# ---------------------------------------------------------------------------


def build_replay_hooks(
    *,
    model_path: str,
    prompt_ids: Sequence[int] | None = None,
    prompt_builder: Callable[[Any], Sequence[int]] | None = None,
    sampler_settings: Mapping[str, Any] | None = None,
    mtp_hidden_variant: str | None = None,
) -> ReplayHooks:
    """Bind :class:`ReplayHooks` to a live MTPLX runtime.  **Touches the GPU.**

    Every MLX import is inside this body, so importing this module -- and
    calling every other function in it -- stays pure NumPy.

    What it binds, and to what
    --------------------------
    ``start_segment``
        ``rt.forward_ar(prompt_ids, cache, return_hidden=True)``.  The last
        hidden row is the one whose logits produced the segment's first token,
        which is the row the first draft chain starts from.
    ``draft_rows``
        the stock chain of ``generation.py`` (the ``depth_index`` loop at
        ``:11894``) reduced to its proposal: for depth ``d``,
        ``rt.draft_mtp(hidden, [[source]], mtp_cache=..., return_hidden=True,
        mtp_depth=d + 1, position_offset=...)``, the last-position row shaped by
        ``_distribution_from_mlx_logits`` with the **draft** sampler, and the
        ids remapped through the run's frspec table when the draft head emits a
        compact vocabulary -- so the rows land in the target's id space and
        intersect the logged target rows.  ``source`` is ``primary`` at depth 1
        and the *forced* token after, never the token this chain would have
        drafted.  The MTP cache offset is read before the chain and restored in
        a ``finally``, so variant 2 starts where variant 1 started.
    ``advance``
        ``rt.forward_ar_capture([primary] + draft_tokens, cache,
        return_hidden=True, capture_backend=...)`` -- the verify forward --
        then ``gdn_capture.commit_captured_prefix`` to keep ``primary`` plus
        the accepted drafts, take the next window's starting hidden from row
        ``accepted`` of this forward (the row that produced the emitted
        selection), and re-stage the MTP history over the committed tokens.
        A carry runs the same step again over the tokens the copy lane
        committed.

    Why the commit is a capture and not a trim
    ------------------------------------------
    The first hardware run (2026-09-02) died on window 0 with "target cache
    would not trim to 16388", on a **full accept** where nothing needed
    trimming.  ``_trim_cache_to_offset`` walks every cache entry and refuses
    the whole cache if any one of them has no ``trim``, before it ever looks at
    how much is being trimmed.  Flash-Next's ``make_cache`` builds a
    ``QSACache`` per full-attention layer (trimmable) and an ``ArraysCache``
    per linear layer -- the GDN recurrent state, which ``cache_state`` marks
    untrimmable on purpose: "Attention KV caches can roll back by trimming
    their offset. GDN recurrent caches cannot."  So an offset trim is not a
    prefix commit on any hybrid model, and never was; it only ever worked
    because nothing had run this on one.

    ``generate_mtpk`` never trims that cache either.  It captures the
    recurrence during the verify forward and rebuilds each recurrent leaf at
    the kept row (``gdn_capture.commit_captured_prefix``), trimming the
    trimmable entries in the same pass, with rollback + re-forward as the
    fallback.  ``advance`` now does exactly that, and asserts the resulting
    offset rather than assuming it.

    Not verified on hardware
    ------------------------
    The piece most likely to need adjustment next is the MTP-history restage in
    ``advance``: production does it through ``generate_mtpk``'s nested
    ``reconcile_mtp_indexer_history`` (``generation.py:9650``), which keeps the
    QSA indexer's raw and pooled frontiers in lockstep with the rollback.  What
    is here is the same intent expressed through the public
    ``rt.update_mtp_cache``.

    That is not a reason to distrust the numbers, because the fidelity gate in
    :func:`score` is not a smoke test: the ``stock`` arm re-drafts the same
    chain the logged run drafted, and every row it returns must reproduce the
    logged ``q`` row.  A wrong hidden state, a wrong cache offset, a wrong
    shaping, a wrong id space, a drifted MTP history -- each moves that distance
    far above ``--fidelity-tol``, and the harness withholds the verdict rather
    than reporting a number.  Never read a candidate delta from a run whose
    fidelity line says FAIL.
    """

    import mlx.core as mx

    from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache
    from mtplx.gdn_capture import commit_captured_prefix, resolve_gdn_capture_backend
    from mtplx.generation import (
        _distribution_from_mlx_logits,
        _mtp_cache_offset,
        _mtp_position_offset,
        _rollback_mtp_cache,
    )
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    runtime = load(model_path, mtp=True)
    settings = dict(sampler_settings or {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
    draft_sampler = SamplerConfig(**settings)

    # The stock run's draft-side RoPE policy, read from the same env keys
    # `generate_mtpk` reads. "default" returns None, which is what the shipped
    # lane passes and what leaves the offset with the KV cache.
    # The GDN capture backend, from the same env key `generate_mtpk` reads
    # (`MTPLX_CAPTURE_CUSTOM_KERNEL`); unset resolves to "stock", which is the
    # capture path every hybrid family supports.  A uniform full-attention
    # model has no recurrent state and `forward_ar_capture` degrades to the
    # plain forward with empty captures, so this is family-generic.
    capture_backend = resolve_gdn_capture_backend(None)
    position_mode = os.environ.get("MTPLX_MTP_POSITION_MODE", "default")
    position_cap = int(os.environ.get("MTPLX_MTP_POSITION_CAP", "0") or 0)
    position_period = int(os.environ.get("MTPLX_MTP_POSITION_PERIOD", "0") or 0)

    def _position_offset(mtp_cache: Any) -> int | None:
        return _mtp_position_offset(
            _mtp_cache_offset(mtp_cache),
            mode=position_mode,
            cap=position_cap,
            period=position_period,
            base=0,
        )

    if prompt_ids is None:
        if prompt_builder is None:
            raise ValueError("build_replay_hooks needs prompt_ids or prompt_builder")
        prompt_ids = list(prompt_builder(runtime))
    prompt = [int(token) for token in prompt_ids]

    text_model = getattr(runtime.model, "language_model", runtime.model)
    frspec_stamp = getattr(text_model, "_mtplx_frspec_ids", None)
    frspec_ids = None if frspec_stamp is None else np.asarray(frspec_stamp)

    def _shaped(row: Any) -> tuple[np.ndarray, np.ndarray]:
        """One draft row, shaped by the draft sampler, in the target id space."""

        distribution = _distribution_from_mlx_logits(row, draft_sampler)
        ids = getattr(distribution, "token_ids", None)
        probs = getattr(distribution, "probs", None)
        if ids is None or probs is None:
            dense = np.asarray(distribution, dtype=np.float64).reshape(-1)
            keep = np.flatnonzero(dense > 0.0)
            ids, probs = keep, dense[keep]
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if frspec_ids is not None and int(row.shape[-1]) == int(frspec_ids.shape[0]):
            ids = np.asarray(frspec_ids[ids], dtype=np.int64)
        keep = probs > 0.0
        return ids[keep], probs[keep]

    class _RuntimeHooks(ReplayHooks):
        def __init__(self) -> None:
            self.cache: Any = None
            self.mtp_cache: Any = None
            self.hidden: Any = None
            self.primary: int = 0
            self.committed: int = 0

        def start_segment(self, segment: Segment) -> None:
            self.cache = runtime.make_cache()
            self.mtp_cache = runtime.make_mtp_cache()
            _logits, hidden = runtime.forward_ar(
                mx.array([prompt]), cache=self.cache, return_hidden=True
            )
            mx.eval(hidden)
            self.hidden = hidden[:, -1:, :]
            self.primary = int(segment.tokens[0])
            self.committed = len(prompt)

        def draft_rows(
            self, *, variant: ProposalVariant, forced_tokens: Sequence[int]
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            offset = int(_mtp_cache_offset(self.mtp_cache))
            hidden = self.hidden
            source = int(self.primary)
            rows: list[tuple[np.ndarray, np.ndarray]] = []
            try:
                with variant_scope(variant):
                    for level, forced in enumerate(forced_tokens):
                        logits, hidden_next = runtime.draft_mtp(
                            hidden,
                            mx.array([[source]]),
                            mtp_cache=self.mtp_cache,
                            return_hidden=True,
                            mtp_hidden_variant=mtp_hidden_variant,
                            mtp_depth=level + 1,
                            position_offset=_position_offset(self.mtp_cache),
                        )
                        row = logits[:, -1, :][0]
                        mx.eval(row, hidden_next)
                        rows.append(_shaped(row))
                        hidden = hidden_next[:, -1:, :]
                        source = int(forced)
            finally:
                # Both arms must start from the state this call found, and the
                # production commit below must see the offset it expects.
                _rollback_mtp_cache(self.mtp_cache, offset)
            return rows

        def advance(self, *, window: Mapping[str, Any]) -> None:
            for step in commit_steps(
                window, offset=self.committed, primary=int(self.primary)
            ):
                self._commit_step(step, index=int(window["index"]))

        def _commit_step(self, step: CommitStep, *, index: int) -> None:
            """One verify forward and its prefix commit, the way production does it.

            NOT a trim.  ``_trim_cache_to_offset`` only commits a prefix on an
            all-trimmable (KV-only) cache; a hybrid model's recurrent layers
            hold state that cannot be trimmed at all, so on Flash-Next it
            refuses even a no-op.  The prefix commit that works on both is the
            one ``generate_mtpk`` uses: capture the recurrence during the
            verify forward and rebuild each recurrent leaf at the kept row
            (``commit_captured_prefix``), trimming the trimmable entries in the
            same pass.  The pre-verify snapshot pays for the fallback, which is
            the copy round's: roll the whole window back and re-forward the
            kept prefix.
            """

            fed = list(step.fed)
            before = snapshot_untrimmable_cache(self.cache)
            _logits, hidden, captures = runtime.forward_ar_capture(
                mx.array([fed]),
                cache=self.cache,
                return_hidden=True,
                capture_backend=capture_backend,
            )
            mx.eval(hidden)
            kept = fed[: step.keep]
            if not commit_captured_prefix(
                self.cache,
                captures,
                keep_tokens=step.keep,
                verified_tokens=len(fed),
            ):
                # Same fallback the context-copy round takes when its capture
                # commit refuses: undo the whole verify window and re-forward
                # the committed prefix.  `hidden` is still the authoritative
                # target hidden for those rows -- it was evaluated above, and
                # the re-forward recomputes the same rows.
                rollback_after_verify(self.cache, before, verified_tokens=len(fed))
                _repair_logits, repair_hidden = runtime.forward_ar(
                    mx.array([kept]), cache=self.cache, return_hidden=True
                )
                mx.eval(repair_hidden)
            offsets = trimmable_offsets(self.cache)
            if any(offset != step.offset for offset in offsets):
                raise RuntimeError(
                    f"after the {step.kind} step of window {index} the target "
                    f"cache sits at {sorted(set(offsets))}, not {step.offset}; "
                    f"the prefix commit did not land "
                    f"(untrimmable entries: {untrimmable_entries(self.cache)})"
                )
            self.hidden = hidden[:, step.hidden_row : step.hidden_row + 1, :]
            self.primary = step.primary
            self.committed = step.offset
            # Restage the MTP history over exactly the committed tokens, from
            # authoritative target hidden. Production does this through
            # `reconcile_mtp_indexer_history` (generation.py:9650); this is the
            # same intent through the public API, and the fidelity gate is what
            # proves it right.
            _rollback_mtp_cache(self.mtp_cache, 0)
            runtime.update_mtp_cache(
                hidden[:, : step.keep, :],
                mx.array([kept]),
                mtp_cache=self.mtp_cache,
                mtp_hidden_variant=mtp_hidden_variant,
                position_offset=_position_offset(self.mtp_cache),
            )

        def close(self) -> None:
            self.cache = None
            self.mtp_cache = None
            self.hidden = None

    return _RuntimeHooks()


# ---------------------------------------------------------------------------
# Scoring -- pure NumPy.
# ---------------------------------------------------------------------------


def total_variation(
    left_ids: np.ndarray,
    left_probs: np.ndarray,
    right_ids: np.ndarray,
    right_probs: np.ndarray,
) -> float:
    """``0.5 * sum |left - right|`` over the union of two prepared rows.

    Equal to ``1 - sum min(left, right)`` for two rows that each sum to 1, but
    computed directly so a row that does *not* sum to 1 (a broken capture)
    shows up as a large distance rather than as a plausible one.
    """

    union = np.union1d(np.asarray(left_ids), np.asarray(right_ids))
    left = np.zeros(union.size, dtype=np.float64)
    right = np.zeros(union.size, dtype=np.float64)
    if union.size:
        left[np.searchsorted(union, left_ids)] = left_probs
        right[np.searchsorted(union, right_ids)] = right_probs
    return float(0.5 * np.sum(np.abs(left - right), dtype=np.float64))


def lookup_prob(ids: np.ndarray, probs: np.ndarray, token: int) -> float:
    """``row(token)``; 0 when the token is outside the support.

    ``ids`` is a prepared row: ascending, unique, so the search is exact.
    """

    ids = np.asarray(ids)
    if ids.size == 0:
        return 0.0
    hit = int(np.searchsorted(ids, np.asarray(token, dtype=ids.dtype)))
    if hit >= ids.size or int(ids[hit]) != int(token):
        return 0.0
    return float(probs[hit])


def reach_from_alpha(alpha: np.ndarray) -> np.ndarray:
    """``w_d = prod_{j <= d} alpha_j`` along the last axis.

    The reach ladder under the shipped per-token law.  Chaining the *marginal*
    per-depth acceptances treats the depths as conditionally independent given
    the logged rows; they are not exactly (``p_{d+1}`` is conditioned on the
    token drawn at depth ``d``).  It is the estimator L Sec.D and
    ``offline_depth4_gate`` already report, so the columns are comparable, and
    for a PAIRED comparison against the same ``p`` rows the approximation is
    common to both arms and cancels to first order in the difference.
    """

    return np.cumprod(np.asarray(alpha, dtype=np.float64), axis=-1)


def expected_tokens(reach: np.ndarray, bonus_allowed: np.ndarray) -> np.ndarray:
    """``E[tokens/window]`` from a reach ladder.

    A window emits its accepted prefix plus one more token -- the correction
    when a depth rejects, the bonus when every depth accepts and the bonus is
    allowed::

        E[l] = sum_d w_d + (1 - w_D) + w_D * bonus_allowed

    Stop tokens are the one simplification: a stop inside the accepted prefix
    ends the request with no selection.  That happens at most once per segment
    and identically for every variant, so it cannot move a paired delta.
    """

    reach = np.asarray(reach, dtype=np.float64)
    bonus = np.asarray(bonus_allowed, dtype=np.float64)
    if reach.size == 0:
        return np.zeros(reach.shape[:-1], dtype=np.float64)
    final = reach[..., -1]
    return reach.sum(axis=-1) + (1.0 - final) + final * bonus


def realised_accepts(rho: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Accepted depth count under a logged uniform tape.

    ``rho[c, d]`` is ``min(1, p_d(x_d) / q_d(x_d))`` at the drafted token; the
    window accepts depth ``d`` iff ``u_d <= rho_d`` and stops at the first
    depth that does not.  Both arms are driven by the *same* tape, so this is a
    common-random-numbers realisation -- noisier than ``alpha`` by construction,
    and here mainly as the exact reproduction check against the log's own
    ``accepted``.
    """

    rho = np.asarray(rho, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if rho.size == 0:
        return np.zeros((rho.shape[0],), dtype=np.int64)
    accept = uniforms[:, : rho.shape[1]] <= rho
    # A sentinel False column makes argmin the count of leading Trues even when
    # every depth accepted.
    padded = np.concatenate(
        (accept, np.zeros((accept.shape[0], 1), dtype=bool)), axis=1
    )
    return np.asarray(np.argmin(padded, axis=1), dtype=np.int64)


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan")
    return mean, float(np.std(values, ddof=1) / np.sqrt(values.size))


def block_bootstrap_se(
    values: np.ndarray,
    owners: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    block: int = BOOTSTRAP_BLOCK,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    """Moving-block bootstrap SE of ``mean(values)``, blocks drawn inside segments.

    Windows within one request are serially correlated -- a hard stretch of
    text is hard for several windows running -- so the i.i.d. paired SE is a
    *floor*, not the interval.  Blocks are drawn within each segment so a
    resample never splices two requests together, and each segment contributes
    its own length back, keeping the resampled mean's weighting identical to
    the observed one.
    """

    values = np.asarray(values, dtype=np.float64)
    owners = np.asarray(owners, dtype=np.int64)
    if values.size < 2 or resamples < 2:
        return float("nan")
    block = max(1, min(int(block), values.size))
    rng = np.random.default_rng(seed)
    groups = [values[owners == owner] for owner in np.unique(owners)]
    groups = [member for member in groups if member.size]
    if not groups:
        return float("nan")
    means = np.empty(int(resamples), dtype=np.float64)
    for draw in range(int(resamples)):
        parts = []
        for member in groups:
            width = min(block, member.size)
            count = int(np.ceil(member.size / width))
            starts = rng.integers(0, member.size - width + 1, size=count)
            parts.append(
                np.concatenate([member[start : start + width] for start in starts])[
                    : member.size
                ]
            )
        means[draw] = float(np.mean(np.concatenate(parts)))
    return float(np.std(means, ddof=1))


def score(
    log: Mapping[str, np.ndarray],
    shadow: ShadowRows,
    *,
    segments: Sequence[Segment] | None = None,
    fidelity_tol: float = FIDELITY_TOL,
    bootstrap: int = BOOTSTRAP_RESAMPLES,
    block: int = BOOTSTRAP_BLOCK,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    limit: int | None = None,
) -> dict[str, Any]:
    """Every variant's acceptance against the same logged target rows.

    The per-depth headline is ``alpha_d = sum_y min(p_d(y), q_d(y))``, computed
    on the *prepared* rows -- id-ascending, positive, renormalised: exactly the
    preparation ``offline_block_verification.prepared_row`` gives every stock
    row and ``offline_depth4_gate`` gives the depth-4 probe row, so these
    columns line up with those scorers' columns entry for entry.
    """

    layout = str(log["layout"]) if "layout" in log else ""
    if layout and layout not in STOCK_LAYOUTS:
        raise ValueError(
            f"shadow-draft harness: layout {layout!r} is not a stock-lane log. "
            "The replay re-runs the stock native-MTP draft chain; the PR391 "
            "device lane's proposal is a different code path."
        )
    depth = int(log["draft_tokens"].shape[1])
    if shadow.depth != depth:
        raise ValueError(
            f"shadow rows have depth {shadow.depth}, log has {depth}: the "
            "capture and the log are not the same run"
        )
    if STOCK_VARIANT not in shadow.variants:
        raise ValueError(
            f"shadow rows carry {shadow.variants}; there is no 'stock' control "
            "arm, so nothing here can be validated or paired"
        )
    cycles = min(int(log["draft_tokens"].shape[0]), shadow.cycles)
    if limit is not None:
        cycles = min(cycles, int(limit))
    segments = list(segments if segments is not None else segment_windows(log))

    variants = list(shadow.variants)
    stock_at = shadow.variant_index(STOCK_VARIANT)
    count = len(variants)

    target_valid = log.get("target_valid")
    draft_valid = log.get("draft_valid")
    greedy = log.get("greedy")
    bonus_allowed = np.asarray(log["bonus_allowed"], dtype=np.float64)

    alpha = np.full((cycles, count, depth), np.nan, dtype=np.float64)
    rho = np.full((cycles, count, depth), np.nan, dtype=np.float64)
    q_at_token = np.full((cycles, count, depth), np.nan, dtype=np.float64)
    diverge = np.full((cycles, count, depth), np.nan, dtype=np.float64)
    fidelity = np.full((cycles, depth), np.nan, dtype=np.float64)
    identical = np.ones((count,), dtype=bool)
    uniforms = np.zeros((cycles, depth + 1), dtype=np.float64)
    owners = np.full((cycles,), -1, dtype=np.int64)
    scored: list[int] = []
    skipped: list[int] = []

    tape = log.get("draft_uniforms")
    have_tape = (
        tape is not None
        and cycles > 0
        and bool(np.all(np.isfinite(np.asarray(tape)[:cycles, :depth])))
    )

    for index in range(cycles):
        if target_valid is not None and not np.all(target_valid[index, :depth]):
            skipped.append(index)
            continue
        if draft_valid is not None and not np.all(draft_valid[index, :depth]):
            skipped.append(index)
            continue
        if greedy is not None and bool(greedy[index]):
            skipped.append(index)
            continue
        if not np.all(shadow.valid[index]):
            skipped.append(index)
            continue
        try:
            targets = [
                prepared_row(
                    log["target_ids"][index, level],
                    log["target_values"][index, level],
                    log["target_probs"][index, level],
                )
                for level in range(depth)
            ]
            logged_q = [
                prepared_row(
                    log["draft_ids"][index, level],
                    log["draft_values"][index, level],
                    log["draft_probs"][index, level],
                )
                for level in range(depth)
            ]
        except ValueError:
            skipped.append(index)
            continue

        uniforms[index] = window_uniforms(log, index, depth)[: depth + 1]
        owners[index] = segment_of_window(segments, index)
        drafted = [int(token) for token in log["draft_tokens"][index][:depth]]

        replayed: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for position in range(count):
            per_depth: list[tuple[np.ndarray, np.ndarray]] = []
            for level in range(depth):
                try:
                    ids, probs = prepared_row(
                        shadow.ids[index, position, level],
                        None,
                        shadow.probs[index, position, level],
                    )
                except ValueError:
                    ids = np.zeros(0, dtype=np.uint32)
                    probs = np.zeros(0, dtype=np.float64)
                per_depth.append((ids, probs))
                p_ids, p_probs = targets[level]
                alpha[index, position, level] = overlap(p_ids, p_probs, ids, probs)
                p_at = lookup_prob(p_ids, p_probs, drafted[level])
                q_at = lookup_prob(ids, probs, drafted[level])
                q_at_token[index, position, level] = q_at
                rho[index, position, level] = (
                    (1.0 if p_at > 0.0 else 0.0)
                    if q_at <= 0.0
                    else min(1.0, p_at / q_at)
                )
                diverge[index, position, level] = float(
                    _would_draft(ids, probs, index, level, tape if have_tape else None)
                    != drafted[level]
                )
            replayed.append(per_depth)

        for level in range(depth):
            # Fidelity is the replayed STOCK row against the LOGGED row: is
            # this the chain the run drafted?
            fidelity[index, level] = total_variation(
                *logged_q[level], *replayed[stock_at][level]
            )
            # Arming is each candidate against the replayed STOCK row, not
            # against the log: two arms that drifted identically are still one
            # arm, and that is the failure this catches.
            for position in range(count):
                if position == stock_at or not identical[position]:
                    continue
                if (
                    total_variation(
                        *replayed[stock_at][level], *replayed[position][level]
                    )
                    > ARMED_TOL
                ):
                    identical[position] = False
        scored.append(index)

    rows = np.asarray(scored, dtype=np.int64)
    n = int(rows.size)
    identical[stock_at] = False  # the control arm is never "did not arm"

    per_variant: list[dict[str, Any]] = []
    stock_tokens: np.ndarray | None = None
    stock_alpha: np.ndarray | None = None
    for position, name in enumerate(variants):
        a = alpha[rows, position, :] if n else np.zeros((0, depth))
        r = reach_from_alpha(a)
        tokens = expected_tokens(r, bonus_allowed[rows])
        r_x = reach_from_alpha(rho[rows, position, :] if n else np.zeros((0, depth)))
        tokens_x = expected_tokens(r_x, bonus_allowed[rows])
        accepts = (
            realised_accepts(rho[rows, position, :], uniforms[rows])
            if n
            else np.zeros(0, dtype=np.int64)
        )
        entry: dict[str, Any] = {
            "variant": name,
            "armed": bool(not identical[position]),
            "alpha": [list(_mean_se(a[:, d])) for d in range(depth)],
            "reach": [
                float(np.mean(r[:, d])) if n else float("nan") for d in range(depth)
            ],
            "tokens_per_window": list(_mean_se(tokens)),
            "tokens_per_window_at_drafted": list(_mean_se(tokens_x)),
            "realised_accepted": list(_mean_se(accepts.astype(np.float64))),
            "diverge": [
                float(np.mean(diverge[rows, position, d])) if n else float("nan")
                for d in range(depth)
            ],
            "q_at_drafted": [
                float(np.mean(q_at_token[rows, position, d])) if n else float("nan")
                for d in range(depth)
            ],
            "paired": None,
        }
        if position == stock_at:
            stock_tokens = tokens
            stock_alpha = a
        else:
            assert stock_tokens is not None and stock_alpha is not None
            delta_tokens = tokens - stock_tokens
            entry["paired"] = {
                "delta_tokens": list(_mean_se(delta_tokens)),
                "delta_tokens_block_se": block_bootstrap_se(
                    delta_tokens,
                    owners[rows],
                    resamples=bootstrap,
                    block=block,
                    seed=bootstrap_seed,
                ),
                "delta_alpha": [
                    list(_mean_se(a[:, d] - stock_alpha[:, d])) for d in range(depth)
                ],
            }
        per_variant.append(entry)

    logged_accepted = (
        np.asarray(log["accepted"], dtype=np.int64)[rows]
        if n
        else np.zeros(0, dtype=np.int64)
    )
    stock_accepts = (
        realised_accepts(rho[rows, stock_at, :], uniforms[rows])
        if n
        else np.zeros(0, dtype=np.int64)
    )
    fidelity_rows = fidelity[rows] if n else np.zeros((0, depth))
    max_tv = float(np.nanmax(fidelity_rows)) if n else float("nan")

    return {
        "layout": layout,
        "mode": shadow.mode,
        "source": shadow.source,
        "depth": depth,
        "cycles": cycles,
        "cycles_scored": n,
        "cycles_skipped": len(skipped),
        "segments": [
            {"index": s.index, "start": s.start, "stop": s.stop, "windows": s.windows}
            for s in segments
        ],
        "variants": per_variant,
        "variant_env": dict(shadow.variant_env),
        "draft_tape_used": bool(have_tape),
        "fidelity": {
            "max_tv": max_tv,
            "mean_tv": float(np.nanmean(fidelity_rows)) if n else float("nan"),
            "tol": float(fidelity_tol),
            "pass": bool(n and np.isfinite(max_tv) and max_tv <= fidelity_tol),
            "accepted_matches": (
                int(np.sum(stock_accepts == logged_accepted)) if n else 0
            ),
            "accepted_total": n,
        },
        "per_segment": _per_segment(
            segments, rows, alpha, bonus_allowed, variants, stock_at, depth
        ),
    }


def _would_draft(
    ids: np.ndarray,
    probs: np.ndarray,
    index: int,
    level: int,
    tape: np.ndarray | None,
) -> int:
    """The token this row would have drafted, for the divergence diagnostic.

    With a logged draft tape (the ``stock_device_k20`` / ``pr391_raw`` layouts
    record one draw per depth), this is the lane's own inverse-CDF over the
    id-ordered prepared row -- the same walk ``sample_prepared`` does.  Without
    one (the host stock layouts consume the draw inside ``rng.choice`` and never
    surface it), it falls back to the row's argmax, which answers a weaker
    question and is labelled as such in the report.
    """

    if ids.size == 0:
        return -1
    if tape is None:
        return int(ids[int(np.argmax(probs))])
    coin = float(np.asarray(tape)[index, level])
    cdf = np.cumsum(probs, dtype=np.float64)
    total = float(cdf[-1])
    if total > 0.0:
        cdf = cdf / total
    position = min(int(np.searchsorted(cdf, coin, side="right")), int(ids.size) - 1)
    return int(ids[position])


def _per_segment(
    segments: Sequence[Segment],
    rows: np.ndarray,
    alpha: np.ndarray,
    bonus_allowed: np.ndarray,
    variants: Sequence[str],
    stock_at: int,
    depth: int,
) -> list[dict[str, Any]]:
    """The same paired delta, one segment (one seed) at a time.

    Three seeds is too few to bootstrap over, but not too few to *look* at: a
    delta that flips sign between seeds is not a delta.
    """

    out: list[dict[str, Any]] = []
    for segment in segments:
        member = rows[(rows >= segment.start) & (rows < segment.stop)]
        if member.size == 0:
            continue
        stock = expected_tokens(
            reach_from_alpha(alpha[member, stock_at, :]), bonus_allowed[member]
        )
        entry: dict[str, Any] = {
            "segment": segment.index,
            "windows": int(member.size),
            "variants": [],
        }
        for position, name in enumerate(variants):
            tokens = expected_tokens(
                reach_from_alpha(alpha[member, position, :]), bonus_allowed[member]
            )
            entry["variants"].append(
                {
                    "variant": name,
                    "alpha": [
                        float(np.mean(alpha[member, position, d]))
                        for d in range(depth)
                    ],
                    "tokens_per_window": float(np.mean(tokens)),
                    "delta_tokens": (
                        None
                        if position == stock_at
                        else list(_mean_se(tokens - stock))
                    ),
                }
            )
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def budget(
    result: Mapping[str, Any],
    *,
    ms_per_window: float = BUDGET_MS_PER_WINDOW,
    chain_ms: float = BUDGET_CHAIN_MS,
    prefill_s: float = BUDGET_PREFILL_S,
) -> dict[str, Any]:
    """GPU seconds for one capture, from the log's own window and segment counts.

    A model, not a measurement.  Per window the replay pays the verify forward
    it teacher-forces (``ms_per_window`` -- the replay's forward is the same
    shape as the run's, so L Sec.3's retained frame is the right order) plus one
    draft chain per variant (``chain_ms``); per segment it pays one prompt
    prefill.  The marginal cost of *adding* a candidate is the chain term alone,
    which is the number that decides whether a variant is worth a capture.
    """

    windows = int(result.get("cycles_scored") or 0) or int(result["cycles"])
    seeds = max(1, len(result["segments"]))
    arms = max(1, len(result["variants"]))
    per_seed_windows = windows / seeds
    chain_s = per_seed_windows * arms * chain_ms / 1000.0
    forward_s = per_seed_windows * ms_per_window / 1000.0
    per_seed = prefill_s + forward_s + chain_s
    return {
        "windows": windows,
        "segments": seeds,
        "variants": arms,
        "assumptions": {
            "ms_per_window": float(ms_per_window),
            "chain_ms": float(chain_ms),
            "prefill_s": float(prefill_s),
        },
        "per_seed_s": float(per_seed),
        "total_s": float(per_seed * seeds),
        "marginal_chain_per_seed_s": float(per_seed_windows * chain_ms / 1000.0),
        "marginal_chain_total_s": float(windows * chain_ms / 1000.0),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pm(pair: Sequence[float], places: int = 4) -> str:
    mean, se = float(pair[0]), float(pair[1])
    text = f"{mean:.{places}f}"
    return text if not np.isfinite(se) else f"{text} +-{se:.{places}f}"


def report(
    result: Mapping[str, Any], *, budget_rows: Mapping[str, Any] | None = None
) -> str:
    depth = int(result["depth"])
    fidelity = result["fidelity"]
    lines = [
        f"layout {result['layout'] or '(unset)'}  mode {result['mode']}  "
        f"depth {depth}  windows {result['cycles']} "
        f"({result['cycles_scored']} scored, {result['cycles_skipped']} skipped)  "
        f"segments {len(result['segments'])}",
        "",
    ]

    if fidelity["pass"]:
        lines.append(
            "fidelity PASS: the re-drafted stock rows reproduce the logged q "
            f"rows (max TV {fidelity['max_tv']:.3e} <= {fidelity['tol']:.3e})."
        )
        matched = fidelity["accepted_matches"] == fidelity["accepted_total"]
        lines.append(
            "  realised accepts reproduce the log on "
            f"{fidelity['accepted_matches']}/{fidelity['accepted_total']} windows"
            + (
                "."
                if matched
                else " -- with rows this close they should all match, so a "
                "shortfall points at the uniform tape (a NaN-filled coin the "
                "log never drew) or at the accept accounting, not at the "
                "replay. Read the alpha columns, not the realised one."
            )
        )
    else:
        lines.append(
            f"fidelity FAIL: max TV {fidelity['max_tv']:.3e} > "
            f"{fidelity['tol']:.3e}. The replay's stock chain is NOT the chain "
            "the log recorded -- wrong hidden state, cache offset, shaping or "
            "id space. Every candidate number below is unreadable until this "
            "line says PASS."
        )
    lines.append("")

    lines.append(
        f"{'variant':<22} "
        + " ".join(f"{'a' + str(d + 1):>17}" for d in range(depth))
        + f" {'E[tok/win]':>17} {'realised':>15}"
    )
    for entry in result["variants"]:
        lines.append(
            f"{entry['variant']:<22} "
            + " ".join(f"{_pm(entry['alpha'][d]):>17}" for d in range(depth))
            + f" {_pm(entry['tokens_per_window']):>17}"
            + f" {_pm(entry['realised_accepted'], places=3):>15}"
        )

    lines.append("")
    lines.append(f"-- paired vs {STOCK_VARIANT}: same p rows, same uniform tape --")
    candidates = 0
    for entry in result["variants"]:
        if entry["paired"] is None:
            continue
        candidates += 1
        if not entry["armed"]:
            lines.append(
                f"{entry['variant']:<22} DID NOT ARM: its rows are bit-identical "
                "to stock on every scored window. The flag is read at import or "
                "at runtime construction, not per draft call -- capture it from "
                "a second process and score both files against this log."
            )
            continue
        paired = entry["paired"]
        mean, se = paired["delta_tokens"]
        lines.append(
            f"{entry['variant']:<22} d(tok/win) {mean:+.5f} +-{se:.5f} (iid) "
            f"+-{paired['delta_tokens_block_se']:.5f} (block)  "
            + "  ".join(
                f"da{d + 1} {paired['delta_alpha'][d][0]:+.5f}"
                f" +-{paired['delta_alpha'][d][1]:.5f}"
                for d in range(depth)
            )
        )
        lines.append(
            f"{'':<22} P(diverge) "
            + " ".join(f"d{d + 1} {entry['diverge'][d]:.3f}" for d in range(depth))
            + (
                "  [logged draft tape]"
                if result["draft_tape_used"]
                else "  [argmax proxy: this layout logged no draft tape]"
            )
        )
    if not candidates:
        lines.append("(no candidate variant in this capture -- stock only)")

    if len(result["per_segment"]) > 1 and candidates:
        lines.append("")
        lines.append(
            "-- per segment (one seed each); a delta that flips sign is not a delta --"
        )
        for entry in result["per_segment"]:
            for variant in entry["variants"]:
                if variant["delta_tokens"] is None:
                    continue
                mean, se = variant["delta_tokens"]
                lines.append(
                    f"seg {entry['segment']} n={entry['windows']:<5d} "
                    f"{variant['variant']:<22} d(tok/win) {mean:+.5f} +-{se:.5f}"
                )

    if budget_rows is not None:
        assumptions = budget_rows["assumptions"]
        lines.append("")
        lines.append(
            "-- capture budget (a MODEL, not a measurement): "
            f"{assumptions['ms_per_window']:.1f} ms/window verify forward, "
            f"{assumptions['chain_ms']:.1f} ms/draft chain, "
            f"{assumptions['prefill_s']:.1f} s/prefill --"
        )
        lines.append(
            f"{budget_rows['segments']} segment(s) x {budget_rows['variants']} "
            f"arm(s) over {budget_rows['windows']} windows: "
            f"{budget_rows['per_seed_s']:.1f} s/seed, "
            f"{budget_rows['total_s']:.1f} s total; one more candidate costs "
            f"{budget_rows['marginal_chain_total_s']:.1f} s."
        )

    lines.append("")
    if not fidelity["pass"]:
        lines.append("VERDICT WITHHELD: fix the replay until the fidelity line passes.")
    elif not candidates:
        lines.append(
            "No verdict: this capture has only the stock control arm. Pass "
            "--variant NAME=KEY=VALUE to put a candidate proposal next to it."
        )
    else:
        lines.append(
            "This is an acceptance verdict only. A candidate that wins alpha and "
            "costs milliseconds still loses; multiply E[tok/win] by the ledger's "
            "ms/window before deciding anything."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", help="path written by MTPLX_FABLE_K20_LOG")
    parser.add_argument(
        "--rows",
        default=None,
        help="an existing shadow-rows .npz to score (no GPU, no model load)",
    )
    parser.add_argument(
        "--capture-to",
        default=None,
        help="replay the trajectory through the model and write shadow rows "
        "here. THIS TOUCHES THE GPU -- run it under bench/laguna/run_guarded.py.",
    )
    parser.add_argument("--model", default=None, help="model path, for --capture-to")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="NAME=KEY=VALUE[,KEY=VALUE]; repeatable. The stock arm is always "
        "captured first and needs no spec.",
    )
    parser.add_argument(
        "--variant-module",
        default=None,
        help="dotted.module:factory returning ProposalVariant objects, for a "
        "proposal change an env var around a call cannot arm",
    )
    parser.add_argument(
        "--expect-segments",
        type=int,
        default=None,
        help="assert the reconstructed request count (e.g. 3 for a 3-seed log)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fidelity-tol", type=float, default=FIDELITY_TOL)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--block", type=int, default=BOOTSTRAP_BLOCK)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--budget",
        action="store_true",
        help="print the GPU-seconds estimate; on its own (no --rows, no "
        "--capture-to) it prints the estimate and exits without a capture",
    )
    parser.add_argument("--ms-per-window", type=float, default=BUDGET_MS_PER_WINDOW)
    parser.add_argument("--chain-ms", type=float, default=BUDGET_CHAIN_MS)
    parser.add_argument("--prefill-s", type=float, default=BUDGET_PREFILL_S)
    parser.add_argument("--json", default=None, help="also write the result as JSON")
    return parser


def resolve_variants(args: argparse.Namespace) -> list[ProposalVariant]:
    variants = [ProposalVariant(name=STOCK_VARIANT)]
    variants.extend(parse_variant(spec) for spec in args.variant)
    if args.variant_module:
        variants.extend(load_variant_module(args.variant_module))
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate variant names: {names}")
    return variants


def _production_prompt(runtime: Any) -> list[int]:
    """The pinned ``coding-16k-1k-xhigh-t1`` prompt the fable captures use.

    Imported here rather than at module scope: ``abba_driver`` is a driver, not
    a library, and pulling it in at import would drag its guard machinery into
    every pure scoring run.
    """

    from mtplx.sampling import SamplerConfig

    from scripts.fable.abba_driver import build_production_cell

    cell = build_production_cell(
        runtime, SamplerConfig, label="shadow-draft-replay", max_tokens=1
    )
    return list(cell["prompt_ids"])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = load_log(args.npz)
    try:
        segments = segment_windows(log)
    except TrajectoryGapError as error:
        print(
            f"FAIL: {args.npz} does not hold the whole committed stream. "
            f"{error}",
            file=sys.stderr,
        )
        return 1
    if args.expect_segments is not None and len(segments) != args.expect_segments:
        print(
            f"FAIL: reconstructed {len(segments)} request segments from "
            f"{args.npz}, expected {args.expect_segments}. The trajectory "
            "reconstruction disagrees with the run; do not replay it.",
            file=sys.stderr,
        )
        return 1

    if args.budget and not (args.rows or args.capture_to):
        estimate = budget(
            {
                "cycles": int(log["draft_tokens"].shape[0]),
                "cycles_scored": 0,
                "segments": [{"index": s.index} for s in segments],
                "variants": [{} for _ in resolve_variants(args)],
            },
            ms_per_window=args.ms_per_window,
            chain_ms=args.chain_ms,
            prefill_s=args.prefill_s,
        )
        print(json.dumps(estimate, indent=2))
        return 0

    if args.capture_to:
        if not args.model:
            print("FAIL: --capture-to needs --model", file=sys.stderr)
            return 1
        variants = resolve_variants(args)
        hooks = build_replay_hooks(
            model_path=args.model, prompt_builder=_production_prompt
        )
        shadow = replay_windows(
            log,
            hooks,
            variants,
            segments=segments,
            limit=args.limit,
            progress=lambda done, total: (
                print(f"[shadow-draft] {done}/{total} windows", flush=True)
                if done % 100 == 0
                else None
            ),
        )
        shadow.source = args.npz
        shadow.save(args.capture_to)
        print(f"[shadow-draft] wrote {args.capture_to}", flush=True)
        rows_path = args.capture_to
    else:
        rows_path = args.rows

    if not rows_path:
        print(
            "FAIL: nothing to score. Pass --rows <shadow.npz> to score an "
            "existing capture, or --capture-to <shadow.npz> --model <path> to "
            "make one (that one needs the GPU and the guard).",
            file=sys.stderr,
        )
        return 1

    shadow = ShadowRows.load(rows_path)
    result = score(
        log,
        shadow,
        segments=segments,
        fidelity_tol=args.fidelity_tol,
        bootstrap=args.bootstrap,
        block=args.block,
        bootstrap_seed=args.bootstrap_seed,
        limit=args.limit,
    )
    estimate = (
        budget(
            result,
            ms_per_window=args.ms_per_window,
            chain_ms=args.chain_ms,
            prefill_s=args.prefill_s,
        )
        if args.budget
        else None
    )
    print(report(result, budget_rows=estimate))
    if args.json:
        payload = dict(result)
        if estimate is not None:
            payload["budget"] = estimate
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    if not result["cycles_scored"]:
        print("\nFAIL: no window could be scored.", file=sys.stderr)
        return 1
    if not result["fidelity"]["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
