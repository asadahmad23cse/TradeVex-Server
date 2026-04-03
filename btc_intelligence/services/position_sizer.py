from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KellyConfig:
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.02
    min_position_pct: float = 0.001
    drawdown_halt_pct: float = 0.08
    volatility_scalar: bool = True


class KellyPositionSizer:
    """
    Fractional Kelly position sizing with risk controls for live trading.
    """

    def __init__(self, config: KellyConfig | None = None) -> None:
        self.config = config or KellyConfig()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    def compute(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        confidence: float,
        drift_level: str,
        edge_decay: bool,
        current_drawdown_pct: float,
        volatility_regime: str,
        portfolio_value: float,
        size_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        p = self._clamp(float(win_rate), 0.0, 1.0)
        q = 1.0 - p
        avg_win = max(0.0, float(avg_win_pct))
        avg_loss = max(0.0, float(avg_loss_pct))
        conf = self._clamp(float(confidence), 0.0, 100.0)
        dd = max(0.0, float(current_drawdown_pct))
        drift = str(drift_level or "LOW").upper()
        vol_regime = str(volatility_regime or "NORMAL").upper()
        port_value = max(0.0, float(portfolio_value))
        external_size_multiplier = max(0.0, float(size_multiplier))

        kelly_full = 0.0
        if avg_win > 0.0 and avg_loss > 0.0:
            payoff_ratio = avg_win / avg_loss
            if payoff_ratio > 0.0:
                kelly_full = max(0.0, p - (q / payoff_ratio))

        kelly_fractional = max(0.0, float(kelly_full) * float(self.config.kelly_fraction))

        reduction = 1.0
        if drift == "MEDIUM":
            reduction *= 0.80
            reasons.append("drift_medium")
        elif drift == "HIGH":
            reduction *= 0.55
            reasons.append("drift_high")

        if bool(edge_decay):
            reduction *= 0.70
            reasons.append("edge_decay")

        if self.config.volatility_scalar and vol_regime == "EXPANSION":
            reduction *= 0.65
            reasons.append("volatility_expansion")

        if conf < 60.0:
            reduction *= 0.75
            reasons.append("low_confidence")

        if dd > 0.04:
            reduction *= 0.80
            reasons.append("drawdown_gt_4pct")
        if dd > 0.06:
            reduction *= 0.60
            reasons.append("drawdown_gt_6pct")

        halted = dd > float(self.config.drawdown_halt_pct)
        halt_reason = "drawdown_halt_triggered" if halted else None
        if halted:
            reasons.append("drawdown_halt")

        if external_size_multiplier != 1.0:
            reasons.append("external_size_multiplier")
        raw_position_pct = kelly_fractional * reduction * external_size_multiplier
        min_pct = min(float(self.config.min_position_pct), float(self.config.max_position_pct))
        max_pct = max(float(self.config.min_position_pct), float(self.config.max_position_pct))
        position_pct = self._clamp(raw_position_pct, min_pct, max_pct) if not halted else 0.0
        position_size_usd = position_pct * port_value

        return {
            "kelly_full": float(kelly_full),
            "kelly_fraction": float(kelly_fractional),
            "position_pct": float(position_pct),
            "position_size_usd": float(position_size_usd),
            "size_reduction_reason": reasons,
            "halted": bool(halted),
            "halt_reason": halt_reason,
        }
