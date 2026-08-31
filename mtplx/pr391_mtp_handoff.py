"""Device-owned MTP history transition for the fixed PR391 D3 route."""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


def bind_pr391_mtp_device_replay(
    mtp_cache: Any,
    *,
    append_rows: Callable[[Any, Any], Any],
) -> Callable[[Any, Any, Any], tuple[Any, ...]]:
    """Bind the fixed-D3 authoritative history handoff to one QSA cache.

    The installed PR391 route reaches this handoff with three speculative MTP
    rows staged and therefore with ``offset == cycle_offset + 3``.  The first
    staged row (the primary token) is authoritative.  Rewinding the tensor
    offset by two consequently names ``cycle_offset + 1``, the start of the
    verifier-owned draft rows.

    The returned callable constructs the exact S=0/S=1/S=2/S=3 eager history
    transitions from that shared replay point, then uses ``accepted_count`` to
    select only their three-row K/V/raw deltas and single possible pooled-block
    delta.  One final slice update installs the selected state without running
    ``where`` across the full fixed-capacity banks. Width-specific construction
    is required: QSA pooling and the underlying matrix dispatch are
    S-dependent, so one fixed S=3 transition followed by an offset rewind is
    not exact.

    ``accepted_count`` is the verifier's uint32 device scalar.  Its 0..3 and
    fixed-shape contract are construction-time requirements of the installed
    route.  They are not revalidated in this measured callable.
    """

    if len(mtp_cache) != 1:
        raise ValueError("PR391 MTP replay requires exactly one cache entry")
    entry = mtp_cache[0]
    if not getattr(entry, "fixed_capacity", False):
        raise ValueError("PR391 MTP replay requires a fixed-capacity QSA cache")
    if getattr(entry, "ratio", None) != 4:
        raise ValueError("PR391 MTP replay requires the installed ratio-4 QSA cache")
    if len(tuple(entry.state_leaves)) != 5:
        raise ValueError("PR391 MTP replay requires the five-leaf QSA state")
    if len(entry.kv.cache) != 3 or len(entry.kv.rollback_state) != 3:
        raise ValueError("PR391 MTP replay requires tensor-offset KV ownership")
    if not callable(append_rows):
        raise TypeError("append_rows must be the prebound exact MTP history route")

    kv = entry.kv
    key_window_shape = list(kv.cache[0].shape)
    key_window_shape[2] = 3
    key_window_shape = tuple(key_window_shape)
    value_window_shape = list(kv.cache[1].shape)
    value_window_shape[2] = 3
    value_window_shape = tuple(value_window_shape)
    raw_window_shape = list(entry.aux[0].shape)
    raw_window_shape[1] = 3
    raw_window_shape = tuple(raw_window_shape)
    pooled_window_shape = list(entry.aux[1].shape)
    pooled_window_shape[1] = 1
    pooled_window_shape = tuple(pooled_window_shape)

    def install_state(state: tuple[Any, ...]) -> None:
        kv.cache[:] = state[:3]
        entry.aux[:] = state[3:]

    def replay(
        accepted_count: Any,
        authoritative_hidden: Any,
        draft_token_ids: Any,
    ) -> tuple[Any, ...]:
        # Preserve the post-D3 device dependency instead of rebuilding the
        # replay point from a host cycle offset.
        replay_offset = kv.offset - 2
        base_state = (
            kv.cache[0],
            kv.cache[1],
            replay_offset,
            entry.aux[0],
            entry.aux[1],
        )
        token_rows = draft_token_ids.reshape(1, 3).astype(mx.int32)
        candidates = [base_state]
        for width in range(1, 4):
            install_state(base_state)
            append_rows(
                authoritative_hidden[:, :width, :],
                token_rows[:, :width],
            )
            candidates.append(tuple(entry.state_leaves))

        device_width = accepted_count.reshape(-1)[0].astype(replay_offset.dtype)
        replay_block = replay_offset // entry.ratio

        def select_window(index, start, *, axis, shape):
            windows = [
                mx.slice(
                    candidate[index],
                    start,
                    axes=(axis,),
                    slice_size=shape,
                )
                for candidate in candidates
            ]
            selected = windows[0]
            for width in range(1, 4):
                selected = mx.where(
                    device_width == width,
                    windows[width],
                    selected,
                )
            return selected

        selected_state = (
            mx.slice_update(
                base_state[0],
                select_window(
                    0,
                    replay_offset,
                    axis=2,
                    shape=key_window_shape,
                ),
                replay_offset,
                axes=(2,),
            ),
            mx.slice_update(
                base_state[1],
                select_window(
                    1,
                    replay_offset,
                    axis=2,
                    shape=value_window_shape,
                ),
                replay_offset,
                axes=(2,),
            ),
            replay_offset + device_width,
            mx.slice_update(
                base_state[3],
                select_window(
                    3,
                    replay_offset,
                    axis=1,
                    shape=raw_window_shape,
                ),
                replay_offset,
                axes=(1,),
            ),
            mx.slice_update(
                base_state[4],
                select_window(
                    4,
                    replay_block,
                    axis=1,
                    shape=pooled_window_shape,
                ),
                replay_block,
                axes=(1,),
            ),
        )
        install_state(selected_state)

        # Candidate construction records rollback metadata for its final
        # physical write. It cannot describe the selected state and must not
        # be consumed by a later host trim.
        kv.rollback_state[:] = [None, None, None]
        return tuple(entry.state_leaves)

    return replay


def stage_pr391_mtp_authoritative_replay(
    mtp_cache: Any,
    *,
    accepted_count: int,
    authoritative_hidden: Any,
    draft_token_ids: Any,
    append_row: Callable[[Any, Any], Any],
) -> Any:
    """Select the exact production-width authoritative replay on device.

    The installed D3 route has already staged ``primary, d1, d2``. Its first
    row is authoritative, so the exact replay always rewinds the two remaining
    speculative rows. Keep that rewind dependent on the post-D3 tensor offset
    instead of materializing the offset on the host. Receive the already-
    decoded verifier width and perform the same one-shot S0/S1/S2/S3 history
    update as production. The installed family capture-commit route leaves a
    rejection correction as the next pending primary, so only accepted drafts
    enter this replay.
    """

    entry = mtp_cache[0]
    entry.trim(2)
    if accepted_count:
        append_row(
            authoritative_hidden[:, :accepted_count],
            draft_token_ids[:, :accepted_count].astype(mx.int32),
        )
    return tuple(entry.state_leaves)


__all__ = [
    "bind_pr391_mtp_device_replay",
    "stage_pr391_mtp_authoritative_replay",
]
