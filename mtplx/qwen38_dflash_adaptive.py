"""Qwen 3.8 challenge adaptive-depth policies for the DFlash2 block shape."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


_SUPPORTED_ROWS = (11, 15, 18, 24, 25, 26, 32)


@dataclass(frozen=True)
class _PolicyConfig:
    proposal_rows: tuple[int, ...]
    base_draft_cap: int
    deep_draft_cap: int
    head_step_cost_ratio: float
    streak_gate: int | None
    optimism_target: float
    confidence_positions: int


def _policy_config(proposal_rows: tuple[int, ...]) -> _PolicyConfig:
    rows = tuple(int(row) for row in proposal_rows)
    if not rows or rows[0] != 11:
        raise ValueError("DFlash adaptive proposal stack requires row 11")
    if tuple(sorted(set(rows))) != rows:
        raise ValueError("DFlash adaptive proposal rows must be unique and chronological")
    unknown = set(rows) - set(_SUPPORTED_ROWS)
    if unknown:
        raise ValueError(f"unsupported DFlash adaptive rows: {sorted(unknown)}")

    base_cap = 4
    deep_cap = 4
    head_cost = 0.20
    streak_gate = None
    optimism_target = 1.0
    confidence_positions = 0
    if 15 in rows:
        base_cap = deep_cap = 8
    if 18 in rows:
        base_cap = 4
        deep_cap = 8
        streak_gate = 3
        optimism_target = 0.95
    if 24 in rows:
        confidence_positions = 1
    if 25 in rows:
        streak_gate = 2
    if 26 in rows:
        base_cap = 5
        head_cost = 0.18
        streak_gate = 3
        confidence_positions = 2
    if 32 in rows:
        streak_gate = 2
    return _PolicyConfig(
        proposal_rows=rows,
        base_draft_cap=base_cap,
        deep_draft_cap=deep_cap,
        head_step_cost_ratio=head_cost,
        streak_gate=streak_gate,
        optimism_target=optimism_target,
        confidence_positions=confidence_positions,
    )


class Qwen38DFlashPositionEMAPolicy:
    """Map source draft depth 0..8 onto DFlash physical blocks 1..8."""

    reductions: int
    reduced_cycles: int
    min_seen: int | None

    def __init__(self, *, full_block_tokens: int, config: _PolicyConfig) -> None:
        if not 1 <= int(full_block_tokens) <= 8:
            raise ValueError("Qwen 3.8 DFlash blocks must be in the range 1..8")
        self.full_block_tokens = int(full_block_tokens)
        self.max_draft_depth = self.full_block_tokens - 1
        self.config = config
        self.position_accept_ema = [
            0.85 * (0.98**index) for index in range(self.max_draft_depth)
        ]
        self.full_accept_streak = 0
        self.current_draft_depth = 0
        self.cycles = 0
        self.serial_cycles = 0
        self.reduced_cycles = 0
        self.reductions = 0
        self.min_seen = None
        self.cycles_by_block: dict[int, int] = {}
        self._last_margins: tuple[float, ...] = ()
        self._recompute_depth()
        if self.block_limit() < self.full_block_tokens:
            self.reductions = 1

    def _active_cap(self) -> int:
        cap = self.config.base_draft_cap
        gate = self.config.streak_gate
        if gate is not None and self.full_accept_streak >= gate:
            cap = self.config.deep_draft_cap
        return min(int(cap), self.max_draft_depth)

    def _confidence(self, index: int) -> float | None:
        if index >= self.config.confidence_positions or index >= len(self._last_margins):
            return None
        divisor = 2.0 if index == 0 else 3.0
        return 1.0 / (1.0 + math.exp(-self._last_margins[index] / divisor))

    def _recompute_depth(self) -> int:
        reach = 1.0
        expected = 0.0
        depth = 0
        head_cost = float(self.config.head_step_cost_ratio)
        cap = self._active_cap()
        while depth < cap:
            probability = self.position_accept_ema[depth]
            confidence = self._confidence(depth)
            if confidence is not None:
                probability = min(probability, confidence)
            reach *= probability
            threshold = head_cost * (1.0 + expected) / (1.0 + depth * head_cost)
            if reach <= threshold:
                break
            expected += reach
            depth += 1
        self.current_draft_depth = depth
        return depth

    def block_limit(self) -> int:
        return max(1, min(self.full_block_tokens, 1 + self.current_draft_depth))

    def record(
        self,
        *,
        block_len: int,
        acceptance_len: int,
        cycle_cost_ns: int | None = None,
    ) -> None:
        del cycle_cost_ns
        block_len = max(1, min(int(block_len), self.full_block_tokens))
        attempted = block_len - 1
        accepted = max(0, min(int(acceptance_len), attempted))
        self.cycles += 1
        self.cycles_by_block[block_len] = self.cycles_by_block.get(block_len, 0) + 1
        if block_len < self.full_block_tokens:
            self.reduced_cycles += 1
            self.min_seen = block_len if self.min_seen is None else min(self.min_seen, block_len)
        if attempted == 0:
            self.serial_cycles += 1
            return

        alpha = 0.15
        for index in range(accepted):
            value = self.position_accept_ema[index]
            self.position_accept_ema[index] = value + alpha * (1.0 - value)
        if accepted < attempted:
            value = self.position_accept_ema[accepted]
            self.position_accept_ema[accepted] = value + alpha * (0.0 - value)
            self.full_accept_streak = 0
        else:
            self.full_accept_streak += 1
            if accepted < self.max_draft_depth:
                value = self.position_accept_ema[accepted]
                target = float(self.config.optimism_target)
                if value < target:
                    self.position_accept_ema[accepted] = value + alpha * (target - value)
        self._recompute_depth()

    def metrics(self) -> dict[str, Any]:
        return {
            "kind": "qwen38_position_ema",
            "proposal_rows": list(self.config.proposal_rows),
            "cycles": int(self.cycles),
            "serial_cycles": int(self.serial_cycles),
            "cycles_by_block": {
                str(block): int(cycles)
                for block, cycles in sorted(self.cycles_by_block.items())
            },
            "position_accept_ema": [float(value) for value in self.position_accept_ema],
            "full_accept_streak": int(self.full_accept_streak),
            "final_block_limit": int(self.block_limit()),
        }


def configure_qwen38_dflash_adaptive_policy(
    target_model: Any,
    *,
    active: bool,
    proposal_rows: tuple[int, ...],
) -> dict[str, Any]:
    """Install one chronological source-policy revision on a DFlash target."""

    attr = "_dflash_adaptive_block_policy_factory"
    if not active:
        if hasattr(target_model, attr):
            delattr(target_model, attr)
        return {"active": False, "proposal_rows": []}

    config = _policy_config(tuple(proposal_rows))

    def factory(
        *,
        full_block_tokens: int,
        verify_len_cap: int,
        prompt_len: int,
    ) -> Qwen38DFlashPositionEMAPolicy:
        del prompt_len
        if int(verify_len_cap) < int(full_block_tokens):
            raise ValueError("adaptive DFlash policy requires the complete verify block")
        return Qwen38DFlashPositionEMAPolicy(
            full_block_tokens=int(full_block_tokens),
            config=config,
        )

    setattr(target_model, attr, factory)
    return {
        "active": True,
        "proposal_rows": list(config.proposal_rows),
        "min_block_tokens": 1,
        "max_block_tokens": 8,
        "base_draft_cap": int(config.base_draft_cap),
        "deep_draft_cap": int(config.deep_draft_cap),
        "head_step_cost_ratio": float(config.head_step_cost_ratio),
        "streak_gate": config.streak_gate,
    }
