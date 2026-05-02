"""Execution-level Kelly shrinkage controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KellyShrinkageResult:
    raw_kelly_fraction: float
    effective_kelly_fraction: float
    kelly_multiplier: float
    kelly_cap_applied: bool
    total_trade_count: int
    already_shrunk: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_kelly_fraction": self.raw_kelly_fraction,
            "effective_kelly_fraction": self.effective_kelly_fraction,
            "kelly_multiplier": self.kelly_multiplier,
            "kelly_cap_applied": self.kelly_cap_applied,
            "total_trade_count": self.total_trade_count,
            "already_shrunk": self.already_shrunk,
        }


class KellyShrinkageController:
    """Applies trade-count shrinkage and a hard per-trade equity cap."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.max_equity_pct_per_trade = float(cfg.get("max_equity_pct_per_trade", 2.0))

    @staticmethod
    def multiplier_for_trades(total_trade_count: int) -> float:
        trades = int(max(0, total_trade_count))
        if trades < 30:
            return 0.3
        if trades < 50:
            return 0.5
        if trades < 100:
            return 0.7
        return 1.0

    @staticmethod
    def normalize_fraction(value: Any) -> float:
        try:
            raw = float(value)
        except Exception:
            return 0.0
        if raw > 1.0:
            raw = raw / 100.0
        return max(0.0, raw)

    def adjust(
        self,
        raw_kelly_fraction: Any,
        total_trade_count: int,
        *,
        already_effective: bool = False,
        existing_effective_fraction: Any | None = None,
    ) -> KellyShrinkageResult:
        raw = self.normalize_fraction(raw_kelly_fraction)
        trades = int(max(0, total_trade_count))
        multiplier = self.multiplier_for_trades(trades)
        cap = max(0.0, self.max_equity_pct_per_trade / 100.0)

        already_shrunk = bool(already_effective)
        if existing_effective_fraction is not None:
            existing = self.normalize_fraction(existing_effective_fraction)
            if existing > 0 and raw > 0 and existing <= raw:
                already_shrunk = True
        else:
            existing = 0.0

        candidate = existing if already_shrunk and existing > 0 else raw * multiplier
        effective = min(candidate, cap) if cap > 0 else candidate
        cap_applied = cap > 0 and candidate > cap

        result = KellyShrinkageResult(
            raw_kelly_fraction=round(raw, 6),
            effective_kelly_fraction=round(effective, 6),
            kelly_multiplier=float(multiplier),
            kelly_cap_applied=bool(cap_applied),
            total_trade_count=trades,
            already_shrunk=bool(already_shrunk),
        )
        logger.info(
            "[KELLY] raw=%.4f adjusted=%.4f trades=%d cap_applied=%s",
            result.raw_kelly_fraction,
            result.effective_kelly_fraction,
            trades,
            str(result.kelly_cap_applied).lower(),
        )
        if result.already_shrunk:
            logger.info("[KELLY] already_effective=true trades=%d", trades)
        return result

    @classmethod
    def from_position_sizing(cls, signal: dict[str, Any] | None, total_trade_count: int, config: dict[str, Any] | None = None) -> KellyShrinkageResult:
        payload = signal or {}
        sizing = payload.get("position_sizing") if isinstance(payload.get("position_sizing"), dict) else {}
        raw = sizing.get("raw_kelly", payload.get("raw_kelly_fraction", payload.get("kelly_fraction", payload.get("position_size_pct", 0.0))))
        existing = sizing.get("position_size_pct", payload.get("position_size_pct"))
        already_effective = any(k in sizing for k in ("quarter_kelly", "position_size_pct", "method"))
        return cls(config).adjust(
            raw,
            total_trade_count,
            already_effective=already_effective,
            existing_effective_fraction=existing,
        )
