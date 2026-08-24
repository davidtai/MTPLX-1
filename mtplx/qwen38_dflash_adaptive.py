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

    def __init__(
        self,
        *,
        full_block_tokens: int,
        config: _PolicyConfig,
        cost_aligned_widths: bool = False,
    ) -> None:
        if not 1 <= int(full_block_tokens) <= 8:
            raise ValueError("Qwen 3.8 DFlash blocks must be in the range 1..8")
        self.full_block_tokens = int(full_block_tokens)
        self.max_draft_depth = self.full_block_tokens - 1
        self.config = config
        self.cost_aligned_widths = bool(cost_aligned_widths)
        self.wants_draft_top2 = bool(config.confidence_positions)
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
        self.commit_tokens_by_block: dict[int, int] = {}
        self.cost_ns_by_block: dict[int, int] = {}
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
        width = max(1, min(self.full_block_tokens, 1 + self.current_draft_depth))
        if not self.cost_aligned_widths or width not in (5, 7):
            return width
        promoted = width + 1
        if (
            self.cycles_by_block.get(width, 0) < 4
            or self.cycles_by_block.get(promoted, 0) < 4
        ):
            return width
        width_cost = self.cost_ns_by_block.get(width, 0)
        promoted_cost = self.cost_ns_by_block.get(promoted, 0)
        if width_cost <= 0 or promoted_cost <= 0:
            return width
        width_rate = self.commit_tokens_by_block.get(width, 0) / width_cost
        promoted_rate = self.commit_tokens_by_block.get(promoted, 0) / promoted_cost
        return promoted if promoted_rate > width_rate * 1.05 else width

    def record(
        self,
        *,
        block_len: int,
        acceptance_len: int,
        cycle_cost_ns: int | None = None,
        draft_top2_logprobs: tuple[tuple[float, ...], ...] = (),
    ) -> None:
        margins = []
        for row in draft_top2_logprobs[: self.config.confidence_positions]:
            if len(row) >= 2:
                margins.append(float(row[0]) - float(row[1]))
        self._last_margins = tuple(margins)
        block_len = max(1, min(int(block_len), self.full_block_tokens))
        attempted = block_len - 1
        accepted = max(0, min(int(acceptance_len), attempted))
        self.cycles += 1
        self.cycles_by_block[block_len] = self.cycles_by_block.get(block_len, 0) + 1
        self.commit_tokens_by_block[block_len] = (
            self.commit_tokens_by_block.get(block_len, 0) + 1 + accepted
        )
        if cycle_cost_ns is not None and int(cycle_cost_ns) > 0:
            self.cost_ns_by_block[block_len] = (
                self.cost_ns_by_block.get(block_len, 0) + int(cycle_cost_ns)
            )
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
        throughput = {
            str(block): (
                self.commit_tokens_by_block.get(block, 0)
                / (cost_ns / 1_000_000_000.0)
            )
            for block, cost_ns in sorted(self.cost_ns_by_block.items())
            if cost_ns > 0
        }
        promoted_widths = []
        if self.cost_aligned_widths:
            for width in (5, 7):
                promoted = width + 1
                if (
                    self.cycles_by_block.get(width, 0) >= 4
                    and self.cycles_by_block.get(promoted, 0) >= 4
                    and float(throughput.get(str(promoted), 0.0))
                    > float(throughput.get(str(width), 0.0)) * 1.05
                ):
                    promoted_widths.append(f"{width}->{promoted}")
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
            "last_draft_margins": [float(value) for value in self._last_margins],
            "cost_alignment": {
                "active": self.cost_aligned_widths,
                "minimum_samples": 4,
                "promotion_margin": 1.05,
                "promoted_widths": promoted_widths,
                "commit_tokens_by_block": {
                    str(block): int(tokens)
                    for block, tokens in sorted(self.commit_tokens_by_block.items())
                },
                "cost_ns_by_block": {
                    str(block): int(cost_ns)
                    for block, cost_ns in sorted(self.cost_ns_by_block.items())
                },
                "tokens_per_second_by_block": throughput,
            },
        }


def configure_qwen38_dflash_adaptive_policy(
    target_model: Any,
    *,
    active: bool,
    proposal_rows: tuple[int, ...],
    cost_aligned_widths: bool = False,
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
            cost_aligned_widths=cost_aligned_widths,
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
        "cost_aligned_widths": bool(cost_aligned_widths),
    }
