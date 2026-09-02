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

    mask_list = [index < keep for keep in keeps for index in range(VERIFY_WIDTH)]
    _y, state_out = gated_delta_update(
        mx.concatenate([row[0] for row in prefix_rows_], axis=1),
        mx.concatenate([row[1] for row in prefix_rows_], axis=1),
        mx.concatenate([row[2] for row in prefix_rows_], axis=1),
        mx.concatenate([row[3] for row in prefix_rows_], axis=1),
        mx.concatenate([row[4] for row in prefix_rows_], axis=1),
        A_log,
        dt_bias,
        state,
        mx.array([mask_list], dtype=mx.bool_),
        use_kernel=True,
    )
    return state_out


__all__ = [
    "folded_gated_delta_update",
    "masked_replay_state",
    "prefix_gated_delta_kernel",
    "prefix_gated_delta_update",
]
