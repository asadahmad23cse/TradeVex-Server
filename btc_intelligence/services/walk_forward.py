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

        old_weight = float(old_weights.get(strategy, 0.0))
        new_weight = float(new_weights.get(strategy, old_weight))

        old_returns = [(float(t.get("pnl_pct", 0.0)) / 100.0) * old_weight for t in test_rows]
        new_returns = [(float(t.get("pnl_pct", 0.0)) / 100.0) * new_weight for t in test_rows]

        old_sharpe = self._safe_sharpe(old_returns)
        new_sharpe = self._safe_sharpe(new_returns)
        improvement = float(new_sharpe - old_sharpe)

        if new_sharpe < (old_sharpe - float(self.config.max_degradation)):
            recommendation = "REJECT"
            applied_fraction = 0.0
            approved = False
            reason = "Walk-forward reject: test Sharpe degraded beyond limit"
        elif new_sharpe > (old_sharpe + float(self.config.min_improvement)):
            recommendation = "APPROVE"
            applied_fraction = 1.0
            approved = True
            reason = "Walk-forward approve: test Sharpe improvement exceeds threshold"
        else:
            recommendation = "CONDITIONAL"
            applied_fraction = 0.5
            approved = True
            reason = "Walk-forward conditional: apply half delta on marginal change"

        return {
            "approved": bool(approved),
            "recommendation": recommendation,
            "old_sharpe": float(old_sharpe),
            "new_sharpe": float(new_sharpe),
            "improvement": float(improvement),
            "applied_fraction": float(applied_fraction),
            "reason": reason,
        }
