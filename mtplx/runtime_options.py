from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


KV_QUANT_MODES = ("off", "q8", "q4")

#: The one boolean vocabulary for MTPLX env flags.
#:
#: Every reader of a boolean ``MTPLX_*`` var should go through
#: :func:`env_bool` so a spelling means the same thing everywhere. Values
#: outside these sets raise rather than being silently read as "off" by one
#: reader and "on" by another — the failure mode catalogued in
#: docs/AUDIT_2026-07-18.md, where ``=enabled`` disabled a feature in the
#: server while enabling it in the generation loop.
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enable", "enabled"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disable", "disabled"})


def env_bool(
    name: str,
    *,
    default: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Parse a boolean ``MTPLX_*`` env var, or raise on an unknown spelling.

    Unset (and set-but-empty) yields ``default``. Anything that is neither
    a recognized true nor a recognized false value is a configuration
    error: guessing is what let one variable mean three things.
    """

    source = os.environ if env is None else env
    raw = source.get(name)
    if raw is None:
        return bool(default)
    token = str(raw).strip().lower()
    if not token:
        return bool(default)
    if token in ENV_TRUE_VALUES:
        return True
    if token in ENV_FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(ENV_TRUE_VALUES | ENV_FALSE_VALUES))
    raise ValueError(
        f"{name}={raw!r} is not a boolean; expected one of: {accepted}"
    )


#: Exact-preserving op diet for the compiled fixed-M4 verify graph.
#:
#: Read ONCE at import so the hot path never touches ``os.environ`` and so a
#: mid-run env change cannot make two traces of the same graph disagree. With
#: the flag off every gated site executes the pre-diet expression verbatim;
#: with it on the rewritten sites are value-identical by construction (see
#: tests/test_fable_opdiet.py, which proves each rewrite against its original
#: on random inputs).
_FABLE_OPDIET = env_bool("MTPLX_FABLE_OPDIET", default=False)

#: The independently selectable rewrites behind the master switch.
#:
#: ``bank``  QSA fixed pooled-bank conditional write (_extend_pooled_fixed)
#: ``rope``  half-width RoPE tables, shared per forward, split-half rotation
#: ``resid`` hyper-connection residual write fused into one kernel
#: ``k20``   eager K20 target/draft support (fused deterministic+ordered pair)
#:
#: Removing dispatches is not the same as removing GPU time: a rewrite can
#: trade contiguous vectorized kernels for broadcast/general ones and lose.
#: ``MTPLX_FABLE_OPDIET_ITEMS`` exists so a result can be attributed to ONE
#: item instead of the whole flag -- which is how the first ``bank`` spelling
#: was caught (2026-09-01: fewest dispatches of three, slowest of three;
#: scripts/fable/micro_opdiet.py).
FABLE_OPDIET_ITEMS = ("bank", "rope", "resid", "k20")


def parse_opdiet_items(
    raw: str | None,
    *,
    known: tuple[str, ...] = FABLE_OPDIET_ITEMS,
) -> frozenset[str]:
    """Parse ``MTPLX_FABLE_OPDIET_ITEMS``; unset/empty selects everything.

    An unknown name raises rather than being dropped: a typo that silently
    disables the item under test would make the A/B measure the wrong thing.
    """

    if raw is None:
        return frozenset(known)
    tokens = {token.strip().lower() for token in str(raw).split(",")}
    tokens.discard("")
    if not tokens:
        return frozenset(known)
    if tokens == {"all"}:
        return frozenset(known)
    unknown = sorted(tokens - set(known))
    if unknown:
        raise ValueError(
            f"MTPLX_FABLE_OPDIET_ITEMS={raw!r} has unknown item(s) "
            f"{', '.join(unknown)}; expected a comma list from: "
            f"{', '.join(known)}"
        )
    return frozenset(tokens)


_FABLE_OPDIET_SELECTED = parse_opdiet_items(
    os.environ.get("MTPLX_FABLE_OPDIET_ITEMS")
)


def fable_opdiet_enabled(item: str | None = None) -> bool:
    """True when the op diet is armed, and this item is selected.

    ``item=None`` answers only the master switch. Every gated call site names
    its item so ``MTPLX_FABLE_OPDIET_ITEMS`` can isolate one rewrite.
    """

    if not _FABLE_OPDIET:
        return False
    if item is None:
        return True
    if item not in FABLE_OPDIET_ITEMS:
        raise ValueError(f"unknown op-diet item {item!r}")
    return item in _FABLE_OPDIET_SELECTED


#: W70 -- fused glue inside the compiled fixed-M4 verify body.
#:
#: One flag, per-item selection, same shape as ``MTPLX_FABLE_OPDIET``: a
#: result must be attributable to ONE rewrite, because "fewer dispatches" and
#: "faster" are different claims (the op diet's first ``bank`` spelling issued
#: the fewest dispatches of three and was the slowest of three).
#:
#: ``qsa_rope``      the attention query/key rotation of a QSA layer -- the
#:                   RoPE table build plus two 5-dispatch rotations -- as ONE
#:                   ``mtplx/kernels/qwen4_m4_rope`` dispatch per layer.
#: ``qsa_rope_idx``  the indexer's query preparation (RMSNorm + partial RoPE)
#:                   through the SHIPPED ``qsa_indexer_prepare_queries_metal``,
#:                   which the fixed-M4 lane never called because
#:                   ``_prepare_queries`` gates on MTPLX_QSA_FUSED_INDEXER.
#:
#: Read ONCE at import, same reasoning as the flags above: the hot path must
#: not touch ``os.environ``, and two traces of the same compiled verify graph
#: must not disagree about which chain they contain.  Off by default.
#:
#: NOT AN ITEM, and the reason is structural rather than a matter of effort:
#: ``hc_triple`` (W69 §4, 194 nodes).  ``hc_norm -> hc_down -> hc_up`` cannot
#: become one dispatch.  ``hc_down`` produces ``mixv[R, 320]`` across 81
#: cooperating threadgroups and every one of ``hc_up``'s 320 threadgroups
#: reads the WHOLE vector; ``hc_norm``'s ``normed[R, 10240]`` is likewise read
#: in full by every ``hc_down`` threadgroup.  Those are grid-wide
#: read-after-write edges, and Metal has no grid-wide barrier inside a
#: dispatch.  The single-threadgroup spelling that would avoid them is the one
#: ``kernels/qwen4_m4_hyper_read`` already measured at 13.2 tok/s against 67.8.
FABLE_VERIFY_GLUE_ITEMS = ("qsa_rope", "qsa_rope_idx")

_FABLE_VERIFY_GLUE = env_bool("MTPLX_FABLE_VERIFY_GLUE", default=False)


def parse_verify_glue_items(
    raw: str | None,
    *,
    known: tuple[str, ...] = FABLE_VERIFY_GLUE_ITEMS,
) -> frozenset[str]:
    """Parse ``MTPLX_FABLE_VERIFY_GLUE_ITEMS``; unset/empty selects everything.

    An unknown name raises rather than being dropped: a typo that silently
    disabled the item under test would make the arm measure the control twice.
    """

    if raw is None:
        return frozenset(known)
    tokens = {token.strip().lower() for token in str(raw).split(",")}
    tokens.discard("")
    if not tokens:
        return frozenset(known)
    if tokens == {"all"}:
        return frozenset(known)
    unknown = sorted(tokens - set(known))
    if unknown:
        raise ValueError(
            f"MTPLX_FABLE_VERIFY_GLUE_ITEMS={raw!r} has unknown item(s) "
            f"{', '.join(unknown)}; expected a comma list from: "
            f"{', '.join(known)}"
        )
    return frozenset(tokens)


_FABLE_VERIFY_GLUE_SELECTED = parse_verify_glue_items(
    os.environ.get("MTPLX_FABLE_VERIFY_GLUE_ITEMS")
)


def fable_verify_glue_enabled(item: str | None = None) -> bool:
    """True when the verify-glue flag is armed, and this item is selected."""

    if not _FABLE_VERIFY_GLUE:
        return False
    if item is None:
        return True
    if item not in FABLE_VERIFY_GLUE_ITEMS:
        raise ValueError(f"unknown verify-glue item {item!r}")
    return item in _FABLE_VERIFY_GLUE_SELECTED


def reset_fable_verify_glue_cache(env: Mapping[str, str] | None = None) -> None:
    """Re-read the verify-glue gates from the environment.  Tests only.

    The hot path reads these once at import on purpose; this exists so a test
    can arm one item without a subprocess, and it is never called by the
    runtime.
    """

    global _FABLE_VERIFY_GLUE, _FABLE_VERIFY_GLUE_SELECTED
    source = os.environ if env is None else env
    _FABLE_VERIFY_GLUE = env_bool(
        "MTPLX_FABLE_VERIFY_GLUE", default=False, env=source
    )
    _FABLE_VERIFY_GLUE_SELECTED = parse_verify_glue_items(
        source.get("MTPLX_FABLE_VERIFY_GLUE_ITEMS")
    )


#: Verify-width fused hyper-connection read (mtplx/kernels/qwen4_m4_hyper_read).
#:
#: Read ONCE at import, same reasoning as ``MTPLX_FABLE_OPDIET``: the hot path
#: must not touch ``os.environ``, and two traces of the same compiled verify
#: graph must not disagree about which read they contain. Off by default.
#:
#: This flag is NOT a "try the kernel" switch. When it is armed and a
#: GatedResidual with 2..8 rows does not match the family contract (hc_count 4,
#: hidden 2560, unquantized mix weights whose dtype matches the activation,
#: down weight [320, 10240]), the read RAISES. A silent fallback to the eager
#: chain would hide the arming failure behind a performance mystery -- which is
#: exactly how MTPLX_FUSED_HC_V3 came to be armed-but-inert at M=4.
_FABLE_HC_M4 = env_bool("MTPLX_FABLE_HC_M4", default=False)


def fable_hc_m4_enabled() -> bool:
    """True when ``MTPLX_FABLE_HC_M4`` armed this process at import."""

    return _FABLE_HC_M4


#: Reduced-dispatch QSA indexer lane for the fixed-M4 verifier
#: (mtplx/kernels/qwen4_qsa_m4_indexer.py + the transposed-key gather).
#:
#: Read ONCE at import, same reasoning as the two flags above: the hot path
#: must not touch ``os.environ``, and two traces of the same compiled verify
#: graph must not disagree about which QSA chain they contain. Off by default.
#:
#: Eligibility is decided at CONSTRUCTION (graphbank installs the fixed QSA
#: cache) and RAISES on a mismatch -- it never silently reverts to the stock
#: chain, because a silently inert flag is how MTPLX_FUSED_HC_V3 came to be
#: armed-but-dead at M=4. The one deliberate non-error narrowing is verify
#: width: a fixed cache also serves the S=1 D3 route, which keeps the stock
#: chain because the fused kernels are wired for the 4-row shape.
_FABLE_QSA_M4 = env_bool("MTPLX_FABLE_QSA_M4", default=False)

#: Verify width the QSA M4 lane is wired for.
FABLE_QSA_M4_ROWS = 4


def fable_qsa_m4_enabled() -> bool:
    """True when ``MTPLX_FABLE_QSA_M4`` armed this process at import."""

    return _FABLE_QSA_M4


#: Transposed-key output for the fused QSA K/V gather.
#:
#: SEPARATE from MTPLX_FABLE_QSA_M4 on purpose, and off by default, because
#: the GPU microbench (2026-09-01, compiled lane, 12 QSA layers) falsified it
#: on BOTH axes while the other four rewrites passed on both:
#:
#:   prep   0.557 -> 0.199 ms (-64%)   0 differing elements
#:   bank   1.153 -> 0.251 ms (-78%)   0 differing elements
#:   score  0.500 -> 0.283 ms (-44%)   0 differing elements
#:   tokens 0.628 -> 0.205 ms (-67%)   0 differing elements
#:   gather 1.501 -> 1.657 ms (+10%)   104 differing elements, max abs 0.125
#:
#: So ``MTPLX_FABLE_QSA_M4=1`` alone is the bit-exact, uniformly faster set.
#: This flag keeps the transposed gather runnable as its own A/B arm (the
#: reason it still exists: see mtplx/kernels/qwen4_qsa_m4_fused_kv_gather.py
#: for why the layout changes the score GEMM's accumulation order and why the
#: tiled transpose is slower than the copy it removes).
_FABLE_QSA_M4_KT = env_bool("MTPLX_FABLE_QSA_M4_KT", default=False)


def fable_qsa_m4_kt_enabled() -> bool:
    """True when ``MTPLX_FABLE_QSA_M4_KT`` armed this process at import."""

    return _FABLE_QSA_M4_KT


#: Split-K (KV-split) native sparse-GQA attention for the DECODE geometries
#: (native_extensions/qsa_sparse_gqa, mtplx/kernels/qsa_sparse_decode.py).
#:
#: TWO flags, deliberately, because the two geometries are independently
#: gated and have very different expected value:
#:
#:   MTPLX_FABLE_QSA_SPARSE_DECODE  -- the M=4 fixed verify, all 12 QSA
#:       layers, once per verify cycle.  This is where the bytes are: the
#:       shipped lane materialises a [1, 2, 4, 2052, 256] gathered K/V pair
#:       per layer (16.8 MB written, then re-read by the score and P@V
#:       GEMMs), plus MLX's own 8.4 MB contiguous copy of the transposed key
#:       view.  The kernel reads the cache rows once and never writes them.
#:
#:   MTPLX_FABLE_QSA_SPARSE_DRAFT   -- the M=1 single-row path.  Today's
#:       retained-stack dispatch census shows ZERO QSA attention dispatches
#:       at M=1: the draft chain runs the MTP block, not the twelve QSA
#:       layers (the census's once-per-cycle counts are exactly 36 GDN and
#:       48 MoE layers, i.e. the full stack runs once per cycle, at M=4).
#:       So this flag exists for the non-speculative decode path and for a
#:       future draft that runs the full stack; it is NOT expected to move
#:       the 16K speculative ABBA, and claiming otherwise from a microbench
#:       would repeat W16 (an isolated -1.9 ms that was 0 end-to-end).
#:
#: Both off by default.  Both RAISE on a contract failure rather than
#: silently reverting -- a silently inert flag is how MTPLX_FUSED_HC_V3 came
#: to be armed-but-dead at M=4.  The one thing that does NOT raise is a
#: PARITY failure at install: this kernel is rounding-class, so a parity
#: miss is a numerical verdict, and the lane disables itself for the process
#: and reports the measured deltas.
_FABLE_QSA_SPARSE_DECODE = env_bool("MTPLX_FABLE_QSA_SPARSE_DECODE", default=False)
_FABLE_QSA_SPARSE_DRAFT = env_bool("MTPLX_FABLE_QSA_SPARSE_DRAFT", default=False)


def fable_qsa_sparse_decode_enabled() -> bool:
    """True when ``MTPLX_FABLE_QSA_SPARSE_DECODE`` armed this process."""

    return _FABLE_QSA_SPARSE_DECODE


def fable_qsa_sparse_draft_enabled() -> bool:
    """True when ``MTPLX_FABLE_QSA_SPARSE_DRAFT`` armed this process."""

    return _FABLE_QSA_SPARSE_DRAFT


def _parse_sparse_decode_tile(raw: str | None) -> tuple[int, int]:
    """``"BK:DC"`` -> the compiled tile pair; unset means the default."""

    if raw is None or not str(raw).strip():
        return (128, 32)
    token = str(raw).strip()
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"MTPLX_FABLE_QSA_SPARSE_DECODE_TILE={raw!r} must be 'BK:DC'"
        )
    try:
        tile = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise ValueError(
            f"MTPLX_FABLE_QSA_SPARSE_DECODE_TILE={raw!r} must be 'BK:DC'"
        ) from exc
    if tile not in FABLE_QSA_SPARSE_DECODE_TILES:
        accepted = ", ".join(f"{a}:{b}" for a, b in FABLE_QSA_SPARSE_DECODE_TILES)
        raise ValueError(
            f"MTPLX_FABLE_QSA_SPARSE_DECODE_TILE={raw!r} is not instantiated; "
            f"expected one of: {accepted}"
        )
    return tile


#: The (BK, DC) pairs the metallib instantiates.  Anything else raises rather
#: than falling back, so a typo in a sweep cannot quietly measure the default.
FABLE_QSA_SPARSE_DECODE_TILES = ((128, 32), (256, 32), (64, 64), (128, 64))
FABLE_QSA_SPARSE_DECODE_MAX_SPLITS = 64

_FABLE_QSA_SPARSE_DECODE_TILE = _parse_sparse_decode_tile(
    os.environ.get("MTPLX_FABLE_QSA_SPARSE_DECODE_TILE")
)


#: MEASURED default (2026-09-02, guarded micro, M=4, 16K, 12 layers).  The
#: kernel is occupancy-bound, and at the shipped tile (BK=128) there are 17
#: BK-tiles over the 2,051 selected keys, so 17 is the smallest split target
#: that reaches one tile per threadgroup -- a 4 x 2 x 17 = 136-threadgroup
#: grid on a 40-core M5 Max.  Everything below it leaves cores idle:
#:
#:     splits   n_splits   threadgroups   ms/layer   x baseline
#:          4          4             32      0.325         0.70
#:          8          6             48      0.210         1.08
#:         16          9             72      0.149         1.52
#:         17         17            136      0.094-0.099   2.3-2.4
#:
#: Larger values clamp to the same 17 at BK=128, so 17 is also the point past
#: which the knob stops doing anything -- which is why the first sweep's s17
#: and s32 rows are the SAME configuration measured twice, and their 5.3%
#: spread is the bench's noise floor rather than a result.
#:
#: The previous default of 8 was a placeholder, and it measured 2.2x slower.
FABLE_QSA_SPARSE_DECODE_DEFAULT_SPLITS = 17


def _parse_sparse_decode_splits(raw: str | None) -> int:
    """``MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS`` -- the KV-split target."""

    if raw is None or not str(raw).strip():
        return FABLE_QSA_SPARSE_DECODE_DEFAULT_SPLITS
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(
            f"MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS={raw!r} must be an integer"
        ) from exc
    if not 1 <= value <= FABLE_QSA_SPARSE_DECODE_MAX_SPLITS:
        raise ValueError(
            f"MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS={raw!r} must be in "
            f"[1, {FABLE_QSA_SPARSE_DECODE_MAX_SPLITS}]"
        )
    return value


_FABLE_QSA_SPARSE_DECODE_SPLITS = _parse_sparse_decode_splits(
    os.environ.get("MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS")
)


def fable_qsa_sparse_decode_tile() -> tuple[int, int]:
    """The armed ``(key_tile, dimension_tile)`` for the decode kernel."""

    return _FABLE_QSA_SPARSE_DECODE_TILE


def fable_qsa_sparse_decode_splits() -> int:
    """The armed KV-split target for the decode kernel."""

    return _FABLE_QSA_SPARSE_DECODE_SPLITS


@dataclass(frozen=True)
class ResolvedAPIKey:
    value: str | None
    source: str

    @property
    def required(self) -> bool:
        return bool(self.value)


def parser_option_names(parser: object, namespace: object = None) -> set[str]:
    """Every option name reachable in the parse context, without dashes.

    Walks the root parser plus whichever subparsers the parse actually
    descended into (using ``namespace`` to pick the branch), which is the
    same scope argparse resolves abbreviations against.
    """

    names: set[str] = set()
    seen: set[int] = set()
    pending = [parser]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for action in getattr(current, "_actions", ()):
            for option in getattr(action, "option_strings", ()) or ():
                names.add(str(option).lstrip("-"))
            choices = getattr(action, "choices", None)
            dest = getattr(action, "dest", None)
            if not isinstance(choices, dict) or not dest:
                continue
            # A subparsers action: follow only the branch that was taken.
            picked = getattr(namespace, str(dest), None) if namespace else None
            if picked is not None and picked in choices:
                pending.append(choices[picked])
            elif namespace is None:
                pending.extend(choices.values())
    return names


def canonicalize_flag_tokens(
    tokens: set[str],
    parser: object,
    namespace: object = None,
) -> set[str]:
    """Expand argparse abbreviations to the flag names they resolved to.

    ``--temp 0.9`` sets ``args.temperature`` but was recorded as ``temp``,
    so every ``"temperature" in cli_flags`` check read it as *not typed*
    and the config file happily overwrote the user's value. Resolving here
    (rather than setting ``allow_abbrev=False``) keeps abbreviations
    working while making the explicit-flag signal true.

    The raw token is kept alongside the expansion, so checks written
    against either spelling keep working. Ambiguous prefixes expand to
    nothing — argparse would have rejected the command anyway.
    """

    known = parser_option_names(parser, namespace)
    resolved = set(tokens)
    for token in tokens:
        if token in known:
            continue
        matches = {name for name in known if name.startswith(token)}
        if len(matches) == 1:
            resolved |= matches
    return resolved


def block_prefix_restore_enabled() -> bool:
    """The single parse of ``MTPLX_SESSION_BLOCK_PREFIX_RESTORE``.

    Default ON. Every reader — the decode loop, the engine session, the
    session bank's cold tier, and the server's settings view — goes through
    this, so one spelling cannot mean ON in one and OFF in another. Unset
    used to mean OFF in the cold tier and ON everywhere else, which
    silently disabled cold-tier block-prefix restore for library embedders
    (the CLI path masks it by force-setting "1"); and the server's
    allowlist-only read reported "off" for spellings the runtime honours as
    on, e.g. ``=enabled``.

    Lives here rather than in :mod:`mtplx.session_bank` because this module
    has no heavy imports: the server reads the setting on paths where the
    mlx-backed runtime may be unavailable.
    """

    return env_bool("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", default=True)


def normalize_paged_kv_quantization(value: object | None, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        return "off"
    raw = str(value).strip().lower().replace("-", "_")
    if raw in ("", "none", "false", "0", "disabled", "disable"):
        return "off"
    if raw in ("off", "q8", "q4"):
        return raw
    if raw in ("8", "8bit", "int8", "uint8", "q8_0"):
        return "q8"
    if raw in ("4", "4bit", "int4", "uint4", "q4_0"):
        return "q4"
    choices = ", ".join(KV_QUANT_MODES)
    raise ValueError(f"unsupported paged KV quantization mode {value!r}; expected one of: {choices}")


def paged_kv_quantization_env(mode: object | None) -> dict[str, str]:
    canonical = normalize_paged_kv_quantization(mode)
    return {
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": canonical,
        "MTPLX_PAGED_KV_QUANT": canonical,
    }


def apply_paged_kv_quantization_env(mode: object | None, env: dict[str, str] | None = None) -> str:
    canonical = normalize_paged_kv_quantization(mode)
    target = os.environ if env is None else env
    target.update(paged_kv_quantization_env(canonical))
    return canonical


def generate_api_key_file(api_key_file: str | os.PathLike[str]) -> str:
    """Create ``api_key_file`` holding a fresh random key and return the key.

    Server entrypoints call this when the user passed ``--api-key-file`` for a
    path that does not exist yet, so the recovery command our own non-localhost
    refusal prints is runnable as-is instead of dying on FileNotFoundError.
    Read-only consumers of key files (doctor, connect) must NOT call this — a
    missing file is a real error there. The file is created 0600.
    """
    import secrets

    path = Path(api_key_file).expanduser()
    key = "mtplx-" + secrets.token_hex(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return key


def resolve_api_key(
    *,
    explicit_api_key: str | None = None,
    api_key_file: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedAPIKey:
    explicit = _clean_secret(explicit_api_key)
    if explicit:
        return ResolvedAPIKey(explicit, "flag")

    if api_key_file:
        path = Path(api_key_file).expanduser()
        secret = _clean_secret(path.read_text(encoding="utf-8"))
        if not secret:
            raise ValueError(f"API key file is empty: {path}")
        return ResolvedAPIKey(secret, "file")

    source_env = os.environ if env is None else env
    api_key = _clean_secret(source_env.get("MTPLX_API_KEY"))
    if api_key:
        return ResolvedAPIKey(api_key, "env:MTPLX_API_KEY")

    legacy = _clean_secret(source_env.get("MTPLX_AUTH"))
    if legacy:
        return ResolvedAPIKey(legacy, "env:MTPLX_AUTH")

    return ResolvedAPIKey(None, "none")


def _clean_secret(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
