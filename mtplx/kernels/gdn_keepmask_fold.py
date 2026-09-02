"""Prefix-folded gated-delta step (MTPLX_FABLE_GDN_KEEPMASK_FOLD).

WHY A KERNEL AND NOT A CONCATENATE
-----------------------------------
The fold needs the next verify window's step kernel to run ``Tpre`` deferred
prefix rows before its own four rows, from one state read and one state write.
The obvious pure-MLX form is
``gated_delta_update(concat(q_pre, q), ..., mask=concat(mask_pre, ones))`` --
exact, but ``mx.concatenate`` copies every input, so five tensors x (ring
windows + 1) pieces x 35 foldable layers is 525 copy dispatches per cycle
against the ~145/cycle the eager replay costs today.  With the program's
measured dependent-launch bound (<= 0.4 us, HC_M4's 776-launch removal) that is
+0.15 ms against the -0.29 ms of state bytes the fold removes; at 2 us it is a
loss.  Taking the prefix as SEPARATE buffers removes all 525.

Both forms are provided.  ``folded_gated_delta_update`` is the pure-MLX
reference (used by the exactness probe and by the micro's parity arm);
``prefix_gated_delta_kernel`` is the kernel that makes the fold worth doing.

BIT-EXACTNESS
-------------
``mlx_lm.models.gated_delta``'s kernel holds the recurrent state in fp32
registers across all ``T`` iterations, loads it once from ``state_in`` and
stores it once to ``state_out``, and ``StT`` is fp32 on this model (the
verifier's state is ``float32``; see ``fable_gdn_keepmask_fold
.validate_state_contract``).  An fp32 store followed by an fp32 load is the
identity, so

    step(rows[k:],  step(rows[:k], S))  ==  step(rows, S)

bit for bit -- the T loop may be split or merged anywhere.  The kernel below
keeps the loop body character-identical to the stock body (the structural test
in ``tests/test_fable_gdn_keepmask_fold.py`` extracts the stock source from the
installed ``mlx_lm`` and compares) and changes only which buffer the pointers
for step ``t`` walk.  Masked steps take the stock ``else`` branch: they write
``y = 0`` and never touch the register state, so a padded prefix slot is an
exact no-op.

WHAT IS NOT CLAIMED
-------------------
That the fold is faster.  ``gated_delta_step`` is 6.29 MB of state traffic per
layer plus ``T`` iterations of 8 fp32 MACs and 2 ``simd_sum`` per thread; if it
is state-bound its wall time is flat in ``T`` and the fold is free, and if it
is T-bound the extra prefix rows cancel the saving.  The census records
dispatch counts, not per-kernel times, so it cannot answer that.
``scripts/fable/micro_gdn_keepmask_fold.py`` measures it under the flock.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import mlx.core as mx

from ..fable_gdn_keepmask_fold import (
    NUM_K_HEADS,
    NUM_V_HEADS,
    HEAD_DIM,
    VERIFY_WIDTH,
    prefix_mask_rows,
)

#: The stock loop body, transcribed verbatim from
#: ``mlx_lm/models/gated_delta.py::_make_gated_delta_kernel`` (has_mask=True,
#: vectorized=False).  ``{q_}``/``{k_}``/``{v_}``/``{g_}``/``{beta_}``/
#: ``{mask_}`` are the only substitutions, and each resolves to the same
#: pointer expression the stock kernel uses; the arithmetic lines are byte
#: identical.  ``tests/test_fable_gdn_keepmask_fold.py`` re-extracts the stock
#: body from the installed mlx_lm and fails if this drifts.
_STEP_BODY = """
          if ({mask_}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_}[hv_idx];
              kv_mem += state[i] * {k_}[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = ({v_}[dv_idx] - kv_mem) * {beta_}[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + {k_}[s_idx] * delta;
              out += state[i] * {q_}[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              {y_}[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            {y_}[dv_idx] = static_cast<InT>(0);
          }}
"""


def _prefix_source() -> str:
    """Metal source for ``Tpre`` prefix rows followed by ``T`` window rows.

    Two sequential copies of the stock body, one walking the prefix buffers and
    one walking the window buffers, between a single state load and a single
    state store.  The prefix rows carry their own mask (the ring's keep mask,
    padded); the window rows are unconditionally live, so their copy is emitted
    with ``mask_ = "true"`` -- exactly what the stock UNMASKED kernel compiles
    to, which is what the shipped verify runs today.
    """

    pre = _STEP_BODY.format(
        mask_="mask_pre[b_idx * Tpre + t]",
        g_="gp_",
        k_="kp_",
        v_="vp_",
        q_="qp_",
        beta_="betap_",
        y_="yp",
    )
    win = _STEP_BODY.format(
        mask_="true",
        g_="g_",
        k_="k_",
        v_="v_",
        q_="q_",
        beta_="beta_",
        y_="y",
    )
    return f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // prefix q, k: [B, Tpre, Hk, Dk]; window q, k: [B, T, Hk, Dk]
        auto qp_ = q_pre + b_idx * Tpre * Hk * Dk + hk_idx * Dk;
        auto kp_ = k_pre + b_idx * Tpre * Hk * Dk + hk_idx * Dk;
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // prefix v: [B, Tpre, Hv, Dv]; window v, y: [B, T, Hv, Dv]
        auto vp_ = v_pre + b_idx * Tpre * Hv * Dv + hv_idx * Dv;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto yp = y_pre + b_idx * Tpre * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        // g: [B, T, Hv]
        auto gp_ = g_pre + b_idx * Tpre * Hv;
        auto g_ = g + b_idx * T * Hv;
        auto betap_ = beta_pre + b_idx * Tpre * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < Tpre; ++t) {{
{pre}
          qp_ += Hk * Dk;
          kp_ += Hk * Dk;
          vp_ += Hv * Dv;
          yp += Hv * Dv;
          gp_ += Hv;
          betap_ += Hv;
        }}
        for (int t = 0; t < T; ++t) {{
{win}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """


_PREFIX_KERNEL = None


def prefix_gated_delta_kernel():
    """Lazily build (once) the prefix-folded step kernel, or ``None``."""

    global _PREFIX_KERNEL
    if _PREFIX_KERNEL is None:
        if not mx.metal.is_available():
            return None
        _PREFIX_KERNEL = mx.fast.metal_kernel(
            name="mtplx_gated_delta_step_prefix",
            input_names=[
                "q_pre",
                "k_pre",
                "v_pre",
                "g_pre",
                "beta_pre",
                "mask_pre",
                "Tpre",
                "q",
                "k",
                "v",
                "g",
                "beta",
                "state_in",
                "T",
            ],
            output_names=["y_pre", "y", "state_out"],
            source=_prefix_source(),
        )
    return _PREFIX_KERNEL


def prefix_gated_delta_update(
    q_pre,
    k_pre,
    v_pre,
    a_pre,
    b_pre,
    mask_pre,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
):
    """``Tpre`` masked prefix rows then ``T`` live rows, one state pass.

    Same ``(y, state)`` contract as ``gated_delta_update`` for the WINDOW rows;
    the prefix rows' ``y`` is produced and discarded (the verify already
    consumed it in the window that captured them).
    """

    from mlx_lm.models.gated_delta import compute_g

    kernel = prefix_gated_delta_kernel()
    if kernel is None:
        raise RuntimeError("prefix gated-delta kernel requires Metal")
    # One template instantiation covers BOTH halves, so a prefix in a
    # different dtype than the window would be read through the window's
    # `InT` -- wrong bytes, silently.  Cheap host check, no dispatch.
    if q_pre.dtype != q.dtype or k_pre.dtype != k.dtype:
        raise ValueError(
            f"prefix dtype {q_pre.dtype} does not match window dtype {q.dtype}"
        )
    beta_pre = mx.sigmoid(b_pre)
    beta = mx.sigmoid(b)
    g_pre = compute_g(A_log, a_pre, dt_bias)
    g = compute_g(A_log, a, dt_bias)
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    Tpre = k_pre.shape[1]
    outputs = kernel(
        inputs=[
            q_pre,
            k_pre,
            v_pre,
            g_pre,
            beta_pre,
            mask_pre,
            Tpre,
            q,
            k,
            v,
            g,
            beta,
            state,
            T,
        ],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, Tpre, Hv, Dv), (B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, q.dtype, state.dtype],
    )
    _y_pre, y, state_out = outputs
    return y, state_out


# --------------------------------------------------------------------------
# Pure-MLX reference fold (the concatenate form)
# --------------------------------------------------------------------------


def folded_gated_delta_update(
    prefix_rows_: Sequence[tuple[Any, Any, Any, Any, Any]],
    keeps: Sequence[int],
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    *,
    max_windows: int,
    pad: bool = True,
):
    """Stock-kernel fold: concatenate the ring's rows in front of the window.

    Exact, and the only form that needs no new Metal.  ``pad`` keeps the
    concatenated width at ``4 * max_windows + 4`` so a compiled graph traces
    once; the leading pad slots are masked off and are exact state no-ops.
    """

    from mlx_lm.models.gated_delta import gated_delta_update

    width = VERIFY_WIDTH * int(max_windows) if pad else VERIFY_WIDTH * len(keeps)
    pad_rows = width - VERIFY_WIDTH * len(keeps)
    if pad_rows < 0:
        raise ValueError("ring is deeper than max_windows")
    dtype = q.dtype
    pieces_q: list[Any] = []
    pieces_k: list[Any] = []
    pieces_v: list[Any] = []
    pieces_a: list[Any] = []
    pieces_b: list[Any] = []
    if pad_rows:
        pieces_q.append(mx.zeros((1, pad_rows, NUM_K_HEADS, HEAD_DIM), dtype=dtype))
        pieces_k.append(mx.zeros((1, pad_rows, NUM_K_HEADS, HEAD_DIM), dtype=dtype))
        pieces_v.append(mx.zeros((1, pad_rows, NUM_V_HEADS, HEAD_DIM), dtype=dtype))
        pieces_a.append(mx.zeros((1, pad_rows, NUM_V_HEADS), dtype=dtype))
        pieces_b.append(mx.zeros((1, pad_rows, NUM_V_HEADS), dtype=dtype))
    for row_q, row_k, row_v, row_a, row_b in prefix_rows_:
        pieces_q.append(row_q)
        pieces_k.append(row_k)
        pieces_v.append(row_v)
        pieces_a.append(row_a)
        pieces_b.append(row_b)
    pieces_q.append(q)
    pieces_k.append(k)
    pieces_v.append(v)
    pieces_a.append(a)
    pieces_b.append(b)

    mask_list = (
        prefix_mask_rows(keeps, max_windows=max_windows)
        if pad
        else [index < keep for keep in keeps for index in range(VERIFY_WIDTH)]
    )
    mask = mx.array([mask_list + [True] * q.shape[1]], dtype=mx.bool_)
    y, state_out = gated_delta_update(
        mx.concatenate(pieces_q, axis=1),
        mx.concatenate(pieces_k, axis=1),
        mx.concatenate(pieces_v, axis=1),
        mx.concatenate(pieces_a, axis=1),
        mx.concatenate(pieces_b, axis=1),
        A_log,
        dt_bias,
        state,
        mask,
        use_kernel=True,
    )
    return y[:, width:], state_out


def masked_replay_state(
    prefix_rows_: Sequence[tuple[Any, Any, Any, Any, Any]],
    keeps: Sequence[int],
    A_log,
    dt_bias,
    state,
):
    """The ring's committed state as ONE masked pass -- the pending leaf.

    This is what ``commit_verified_window`` would bind to ``cache[1]`` under
    the fold: an unevaluated array that any non-fold consumer can force, at
    exactly the cost of today's eager replay, and that the next compiled
    window recognises and replaces with a base + prefix pair.
    """

    from mlx_lm.models.gated_delta import gated_delta_update

    if not prefix_rows_:
        return state
    mask_list = [index < keep for keep in keeps for index in range(VERIFY_WIDTH)]
    if len(prefix_rows_) == 1:
        # A one-window ring is the common case; concatenating a single piece
        # would be five pure copies on a leaf that anything may force.
        rows = tuple(prefix_rows_[0])
    else:
        rows = tuple(
            mx.concatenate([row[index] for row in prefix_rows_], axis=1)
            for index in range(5)
        )
    _y, state_out = gated_delta_update(
        *rows,
        A_log,
        dt_bias,
        state,
        mx.array([mask_list], dtype=mx.bool_),
        use_kernel=True,
    )
    return state_out


# --------------------------------------------------------------------------
# Fixed-shape prefix buffers -- what the compiled graph is handed every window
# --------------------------------------------------------------------------
#
# The compiled physical-M4 verify traces ONCE, so the prefix it receives is one
# fixed shape on every window whatever the ring holds: ``[1, 4*W, ...]`` rows
# plus a ``[1, 4*W]`` bool mask.  A ring of depth ``d < W`` pads the FRONT with
# masked slots, and a ring of depth 0 is all pad -- an exact no-op on the state
# and, because the pad and the all-False mask are cached constants, zero
# dispatches.  Padding cannot reorder anything: a masked row takes the stock
# kernel's ``else`` branch, which writes ``y = 0`` and leaves the register
# state untouched, so pad slots commute with live ones.

_PAD_CACHE: dict[tuple[int, Any], tuple[Any, ...]] = {}
_MASK_CACHE: dict[tuple[tuple[int, ...], int], Any] = {}


def _pad_leaves(pad_rows: int, dtype) -> tuple[Any, ...]:
    """Cached all-zero ``(q, k, v, a, b)`` filler of ``pad_rows`` rows."""

    key = (int(pad_rows), dtype)
    cached = _PAD_CACHE.get(key)
    if cached is None:
        cached = (
            mx.zeros((1, pad_rows, NUM_K_HEADS, HEAD_DIM), dtype=dtype),
            mx.zeros((1, pad_rows, NUM_K_HEADS, HEAD_DIM), dtype=dtype),
            mx.zeros((1, pad_rows, NUM_V_HEADS, HEAD_DIM), dtype=dtype),
            mx.zeros((1, pad_rows, NUM_V_HEADS), dtype=dtype),
            mx.zeros((1, pad_rows, NUM_V_HEADS), dtype=dtype),
        )
        # Deliberately NOT evaluated here: the first window's pre-boundary
        # `mx.async_eval(*state_in)` materialises them with everything else,
        # and the cached array objects hold the buffers afterwards.  An
        # `mx.eval` here would put a host sync inside a decode cycle to build
        # a constant.
        _PAD_CACHE[key] = cached
    return cached


def prefix_mask_array(keeps: Sequence[int], *, max_windows: int):
    """Cached ``[1, 4*max_windows]`` bool prefix mask for one ring shape.

    There are at most ``3**W + ... + 1`` distinct rings (13 at W=2), so every
    mask a run can ever need is built once and then reused as a constant --
    no per-cycle host-to-device copy on the decode path.
    """

    key = (tuple(int(k) for k in keeps), int(max_windows))
    cached = _MASK_CACHE.get(key)
    if cached is None:
        cached = mx.array(
            [prefix_mask_rows(key[0], max_windows=key[1])], dtype=mx.bool_
        )
        _MASK_CACHE[key] = cached
    return cached


_EMPTY_CACHE: dict[tuple[int, Any, int], tuple[Any, ...]] = {}


def empty_prefix_leaves(*, max_windows: int, dtype, slot: int = 0):
    """The depth-0 prefix for ONE layer: all pad, all masked, all cached.

    ``slot`` is the layer's position in the fold plan, and the leaves are
    cached per slot rather than shared, so the 175 row inputs a depth-0 window
    hands the compiled graph are 175 DISTINCT arrays.  Handing one array to 35
    input positions would make the traced graph's input identity depend on the
    ring depth of the window that happened to trace it, which is exactly the
    kind of thing that works until the first depth-1 window.  The cost is 35
    copies of a ~163 kB constant, allocated once.
    """

    key = (VERIFY_WIDTH * int(max_windows), dtype, int(slot))
    cached = _EMPTY_CACHE.get(key)
    if cached is None:
        pad_rows = key[0]
        cached = (
            mx.zeros((1, pad_rows, NUM_K_HEADS, HEAD_DIM), dtype=dtype),
            mx.zeros((1, pad_rows, NUM_K_HEADS, HEAD_DIM), dtype=dtype),
            mx.zeros((1, pad_rows, NUM_V_HEADS, HEAD_DIM), dtype=dtype),
            mx.zeros((1, pad_rows, NUM_V_HEADS), dtype=dtype),
            mx.zeros((1, pad_rows, NUM_V_HEADS), dtype=dtype),
        )
        _EMPTY_CACHE[key] = cached
    return cached


def padded_prefix_leaves(
    ring_rows: Sequence[tuple[Any, Any, Any, Any, Any]],
    keeps: Sequence[int],
    *,
    max_windows: int,
    dtype,
) -> tuple[Any, ...]:
    """``(q, k, v, a, b)`` at ``[1, 4*max_windows, ...]`` for one layer's ring.

    A full ring of exactly one window (``max_windows == 1``) returns the
    captured rows THEMSELVES -- no pad, no concatenate, no dispatch.  Deeper
    rings pay one ``mx.concatenate`` per tensor per commit; the concatenates
    are lazy, so they ride the next window's pre-boundary ``async_eval``
    rather than costing a separate submission.
    """

    if len(ring_rows) != len(keeps):
        raise ValueError(
            f"ring has {len(ring_rows)} row groups for {len(keeps)} keeps"
        )
    width = VERIFY_WIDTH * int(max_windows)
    pad_rows = width - VERIFY_WIDTH * len(keeps)
    if pad_rows < 0:
        raise ValueError(
            f"ring of {len(keeps)} windows exceeds max_windows={max_windows}"
        )
    if not ring_rows:
        raise ValueError("an empty ring uses empty_prefix_leaves")
    if pad_rows == 0 and len(ring_rows) == 1:
        return tuple(ring_rows[0])
    pieces: list[list[Any]] = [[] for _ in range(5)]
    if pad_rows:
        pad = _pad_leaves(pad_rows, dtype)
        for index in range(5):
            pieces[index].append(pad[index])
    for row in ring_rows:
        for index in range(5):
            pieces[index].append(row[index])
    return tuple(mx.concatenate(piece, axis=1) for piece in pieces)


def reset_prefix_caches() -> None:
    """Test support: drop the cached pads and masks."""

    _PAD_CACHE.clear()
    _MASK_CACHE.clear()
    _EMPTY_CACHE.clear()


# --------------------------------------------------------------------------
# Install-time exactness probe (the ONE gate that disables instead of raising)
# --------------------------------------------------------------------------


_PROBE_CACHE: dict[tuple[int, int], tuple[bool, str]] = {}


def default_exactness_probe(*, max_windows: int = 2, seed: int = 0):
    """Compare the folded recurrence against the shipped two-pass one.

    Runs on the production geometry with one layer's worth of synthetic rows:
    two step dispatches for the reference, one for the candidate.  Returns
    ``(ok, detail)``; ``install_gdn_keepmask_fold`` DISABLES the lane on a
    False rather than raising, because a mismatch is a fact about this MLX
    build's kernel (does its fp32 state round-trip?) rather than a
    misconfiguration of this process.

    Memoised per ``(max_windows, seed)``: the answer is a property of the MLX
    build, and ``install_fixed_m4`` runs once per REQUEST -- a server would
    otherwise pay three dispatches and a host sync on every one.
    """

    key = (int(max_windows), int(seed))
    cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    result = _run_exactness_probe(max_windows=int(max_windows), seed=int(seed))
    _PROBE_CACHE[key] = result
    return result


def reset_exactness_probe_cache() -> None:
    """Test support: forget the memoised probe verdict."""

    _PROBE_CACHE.clear()


def _run_exactness_probe(*, max_windows: int, seed: int):
    from mlx_lm.models.gated_delta import gated_delta_update

    if not mx.metal.is_available():
        return False, "metal unavailable"
    state_key = mx.random.key(int(seed))
    keeps = (2,) if int(max_windows) >= 1 else ()

    def _rows(width: int, key):
        keys = mx.random.split(key, 5)
        return (
            mx.random.normal(
                (1, width, NUM_K_HEADS, HEAD_DIM), key=keys[0]
            ).astype(mx.bfloat16),
            mx.random.normal(
                (1, width, NUM_K_HEADS, HEAD_DIM), key=keys[1]
            ).astype(mx.bfloat16),
            mx.random.normal(
                (1, width, NUM_V_HEADS, HEAD_DIM), key=keys[2]
            ).astype(mx.bfloat16),
            mx.random.normal((1, width, NUM_V_HEADS), key=keys[3]).astype(
                mx.bfloat16
            ),
            mx.random.normal((1, width, NUM_V_HEADS), key=keys[4]).astype(
                mx.bfloat16
            ),
        )

    parts = mx.random.split(state_key, 4)
    A_log = mx.random.normal((NUM_V_HEADS,), key=parts[0]).astype(mx.float32)
    dt_bias = mx.random.normal((NUM_V_HEADS,), key=parts[1]).astype(mx.bfloat16)
    state = mx.random.normal(
        (1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM), key=parts[2]
    ).astype(mx.float32)
    ring = _rows(VERIFY_WIDTH, parts[3])
    window = _rows(VERIFY_WIDTH, parts[0])

    keep = keeps[0]
    _y, mid = gated_delta_update(
        *(tensor[:, :keep] for tensor in ring),
        A_log,
        dt_bias,
        state,
        None,
        use_kernel=True,
    )
    reference_y, reference_state = gated_delta_update(
        *window, A_log, dt_bias, mid, None, use_kernel=True
    )

    prefix = padded_prefix_leaves(
        [ring], keeps, max_windows=max_windows, dtype=mx.bfloat16
    )

    mask = prefix_mask_array(keeps, max_windows=max_windows)
    folded_y, folded_state = prefix_gated_delta_update(
        *prefix, mask, *window, A_log, dt_bias, state
    )
    state_diff = int(mx.sum(reference_state != folded_state).item())
    y_diff = int(mx.sum(reference_y != folded_y).item())
    if state_diff or y_diff:
        return False, (
            f"split/merged recurrence mismatch: state={state_diff} "
            f"y={y_diff} differing elements"
        )
    return True, "bit-exact on the production geometry"


__all__ = [
    "default_exactness_probe",
    "reset_exactness_probe_cache",
    "empty_prefix_leaves",
    "folded_gated_delta_update",
    "masked_replay_state",
    "padded_prefix_leaves",
    "prefix_gated_delta_kernel",
    "prefix_gated_delta_update",
    "prefix_mask_array",
    "reset_prefix_caches",
]
