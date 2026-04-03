from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AggregatorConfig:
    min_sources_required: int = 3
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "order_flow": 0.28,
            "volume_profile": 0.22,
            "regime": 0.20,
            "momentum": 0.18,
            "volatility": 0.12,
        }
    )


class SignalAggregator:
    """
    Weighted evidence aggregator for multi-source directional signals.
    """

    def __init__(self, config: AggregatorConfig | None = None) -> None:
        self.config = config or AggregatorConfig()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    def aggregate(self, signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
        signal_rows = dict(signals or {})
        available_rows = {
            key: row for key, row in signal_rows.items() if bool((row or {}).get("available", False))
        }
        sources_available = len(available_rows)
        if sources_available < int(self.config.min_sources_required):
            return {
                "direction": "HOLD",
                "raw_score": 0.0,
                "confidence": 0.0,
                "agreement_ratio": 0.0,
                "sources_used": 0,
                "sources_available": sources_available,
                "source_contributions": {},
                "rejected": True,
                "reject_reason": "insufficient_sources",
            }

        weighted_sum = 0.0
        available_weight_sum = 0.0
        contributions_abs: dict[str, float] = {}
        source_dirs: dict[str, str] = {}
        for key, row in available_rows.items():
            weight = float(self.config.source_weights.get(key, 0.0))
            if weight <= 0.0:
                continue
            direction = str((row or {}).get("direction", "NEUTRAL")).upper()
            strength = self._clamp(float((row or {}).get("strength", 0.0)), 0.0, 1.0)
            sign = 1.0 if direction == "LONG" else -1.0 if direction == "SHORT" else 0.0
            contribution = sign * strength * weight
            weighted_sum += contribution
            available_weight_sum += weight
            contributions_abs[key] = contribution
            source_dirs[key] = direction

        sources_used = len(contributions_abs)
        if available_weight_sum <= 1e-9:
            return {
                "direction": "HOLD",
                "raw_score": 0.0,
                "confidence": 0.0,
                "agreement_ratio": 0.0,
                "sources_used": sources_used,
                "sources_available": sources_available,
                "source_contributions": {},
                "rejected": True,
                "reject_reason": "insufficient_sources",
            }

        raw_score = self._clamp(weighted_sum / available_weight_sum, -1.0, 1.0)
        majority_direction = "LONG" if raw_score > 0.0 else "SHORT" if raw_score < 0.0 else "NEUTRAL"
        if majority_direction in {"LONG", "SHORT"}:
            agree_count = sum(1 for d in source_dirs.values() if d == majority_direction)
            agreement_ratio = self._clamp(agree_count / max(sources_available, 1), 0.0, 1.0)
        else:
            agreement_ratio = 0.0

        confidence = self._clamp(abs(raw_score) * agreement_ratio * 100.0, 0.0, 100.0)
        if raw_score > 0.12:
            final_direction = "LONG"
        elif raw_score < -0.12:
            final_direction = "SHORT"
        else:
            final_direction = "HOLD"

        source_contributions = {
            key: round(float(val / available_weight_sum), 6) for key, val in contributions_abs.items()
        }
        return {
            "direction": final_direction,
            "raw_score": round(raw_score, 6),
            "confidence": round(confidence, 2),
            "agreement_ratio": round(agreement_ratio, 6),
            "sources_used": sources_used,
            "sources_available": sources_available,
            "source_contributions": source_contributions,
            "rejected": False,
            "reject_reason": None,
        }
