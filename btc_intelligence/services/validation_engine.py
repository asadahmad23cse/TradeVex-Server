from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationConfig:
    min_samples: int = 30
    min_expected_edge: float = 0.01
    max_calibration_error: float = 0.20
    max_brier_score: float = 0.30
    volatile_min_samples_multiplier: float = 1.25
    volatile_max_brier_multiplier: float = 0.90
    trending_min_samples_multiplier: float = 0.90
    trending_max_brier_multiplier: float = 1.05

    def thresholds_for_regime(self, regime: str | None = None) -> tuple[int, float]:
        regime_key = str(regime or "SIDEWAYS").upper()
        if regime_key == "VOLATILE":
            min_samples = int(math.ceil(self.min_samples * self.volatile_min_samples_multiplier))
            max_brier = self.max_brier_score * self.volatile_max_brier_multiplier
        elif regime_key == "TRENDING":
            min_samples = max(1, int(math.floor(self.min_samples * self.trending_min_samples_multiplier)))
            max_brier = self.max_brier_score * self.trending_max_brier_multiplier
        else:
            min_samples = self.min_samples
            max_brier = self.max_brier_score
        return min_samples, float(max(0.0, min(1.0, max_brier)))


class ValidationEngine:
    """
    Hard validation gate for adaptive modules.
    Rejects steps that are data-starved, likely to overfit, or add no robust edge.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate_step(
        self,
        step: str,
        payload: dict[str, Any] | None = None,
        regime: str | None = None,
    ) -> dict[str, Any]:
        data = payload or {}
        sample_size = int(max(0, data.get("sample_size", 0)))
        expected_edge = float(data.get("expected_edge", 0.0))
        robustness_gain = float(data.get("robustness_gain", 0.0))
        drawdown_delta = float(data.get("drawdown_delta", 0.0))
        overfit_risk = float(data.get("overfit_risk", 0.0))
        brier_score = float(data.get("brier_score", 0.25))
        calibration_error = float(data.get("calibration_error", 0.15))
        min_samples_threshold, max_brier_threshold = self.config.thresholds_for_regime(regime)

        checks = {
            "sufficient_data": sample_size >= min_samples_threshold,
            "has_edge": expected_edge >= self.config.min_expected_edge,
            "improves_robustness": robustness_gain > 0.0 or drawdown_delta <= 0.0,
            "overfitting_controlled": overfit_risk <= 0.45,
            "calibration_ok": (
                calibration_error <= self.config.max_calibration_error
                and brier_score <= max_brier_threshold
            ),
        }
        approved = all(checks.values())
        if approved:
            decision = "APPROVE"
            reason = f"{step}: robust and statistically acceptable"
        else:
            decision = "REJECT"
            failed = [k for k, v in checks.items() if not v]
            reason = f"{step}: rejected due to {', '.join(failed[:3])}"

        return {
            "step": step,
            "decision": decision,
            "approved": bool(approved),
            "reason": reason,
            "checks": checks,
            "sample_size": sample_size,
        }
