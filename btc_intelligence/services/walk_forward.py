"""
Sharpe is computed on raw returns; weight delta determines applied_fraction tier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any


@dataclass
class WalkForwardConfig:
    train_window: int = 120
    test_window: int = 40
    min_improvement: float = 0.02
    max_degradation: float = 0.05


class WalkForwardValidator:
    """
    Walk-forward validator for adaptive strategy weight updates.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    @staticmethod
    def _safe_sharpe(returns: list[float]) -> float:
        if len(returns) < 2:
            return 0.0
        sd = stdev(returns)
        if sd <= 1e-9:
            return 0.0
        return float((mean(returns) / sd) * math.sqrt(len(returns)))

    def validate_weight_update(
        self,
        strategy: str,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        trade_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rows = list(trade_history or [])
        if len(rows) < 4:
            return {
                "approved": True,
                "recommendation": "CONDITIONAL",
                "old_sharpe": 0.0,
                "new_sharpe": 0.0,
                "improvement": 0.0,
                "applied_fraction": 0.5,
                "reason": "Insufficient walk-forward samples; apply partial update",
            }

        split_idx = int(len(rows) * 0.75)
        split_idx = max(1, min(len(rows) - 1, split_idx))
        train_rows = rows[:split_idx]
        test_rows = rows[split_idx:]

        if self.config.train_window > 0:
            train_rows = train_rows[-int(self.config.train_window) :]
        if self.config.test_window > 0:
            test_rows = test_rows[-int(self.config.test_window) :]

        if len(train_rows) < 2:
            return {
                "approved": True,
                "recommendation": "CONDITIONAL",
                "old_sharpe": 0.0,
                "new_sharpe": 0.0,
                "improvement": 0.0,
                "applied_fraction": 0.5,
                "reason": "Insufficient train window; apply partial update",
            }

        if len(test_rows) < 2:
            return {
                "approved": True,
                "recommendation": "CONDITIONAL",
                "old_sharpe": 0.0,
                "new_sharpe": 0.0,
                "improvement": 0.0,
                "applied_fraction": 0.5,
                "reason": "Insufficient test window for Sharpe; apply partial update",
            }

        returns = [float(t.get("pnl_pct", 0.0)) / 100.0 for t in test_rows]
        base_sharpe = self._safe_sharpe(returns)
        old_w = float(old_weights.get(strategy, 0.0))
        new_w = float(new_weights.get(strategy, old_w))
        weight_delta = abs(new_w - old_w)

        old_sharpe = float(base_sharpe)
        new_sharpe = float(base_sharpe)
        improvement = float(base_sharpe - 0.0)

        if base_sharpe < -float(self.config.max_degradation):
            recommendation = "REJECT"
            applied_fraction = 0.0
            approved = False
            reason = "Walk-forward reject: raw test Sharpe below degradation limit"
        elif base_sharpe > float(self.config.min_improvement):
            approved = True
            if weight_delta >= 0.02:
                recommendation = "APPROVE"
                applied_fraction = 1.0
                reason = "Walk-forward approve: raw Sharpe exceeds improvement threshold"
            else:
                recommendation = "CONDITIONAL"
                applied_fraction = 0.5
                reason = "Marginal weight delta; apply partial update"
        else:
            recommendation = "CONDITIONAL"
            applied_fraction = 0.5
            approved = True
            reason = "Walk-forward conditional: raw Sharpe in neutral band"

        return {
            "approved": bool(approved),
            "recommendation": recommendation,
            "old_sharpe": float(old_sharpe),
            "new_sharpe": float(new_sharpe),
            "improvement": float(improvement),
            "applied_fraction": float(applied_fraction),
            "reason": reason,
        }
