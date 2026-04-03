from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ProbabilityInput:
    momentum_score: float
    flow_score: float
    volatility_regime: str
    regime: str


class ProbabilityService:
    """Directional probability estimator from momentum + flow + volatility context."""

    def estimate(self, inp: ProbabilityInput) -> dict[str, Any]:
        momentum = float(np.clip(inp.momentum_score, -1.0, 1.0))
        flow = float(np.clip(inp.flow_score, -1.0, 1.0))
        vol = str(inp.volatility_regime or "").upper()
        regime = str(inp.regime or "").lower()

        regime_up_bias = 0.0
        regime_down_bias = 0.0
        if "bull" in regime or "breakout_up" in regime:
            regime_up_bias = 0.35
        elif "bear" in regime or "breakout_down" in regime:
            regime_down_bias = 0.35

        vol_sideways_bias = 0.0
        if vol in {"LOW", "COMPRESSION"}:
            vol_sideways_bias = 0.55
        elif vol in {"EXPANSION"}:
            vol_sideways_bias = -0.10
        elif vol in {"HIGH_VOL"}:
            vol_sideways_bias = 0.25

        up_logit = (0.90 * momentum) + (0.80 * flow) + regime_up_bias
        down_logit = (-0.90 * momentum) + (-0.80 * flow) + regime_down_bias
        side_logit = 0.35 + vol_sideways_bias - (0.60 * abs(momentum)) - (0.50 * abs(flow))

        logits = np.asarray([up_logit, down_logit, side_logit], dtype=float)
        logits = logits - float(np.max(logits))
        exps = np.exp(logits)
        probs = exps / max(float(np.sum(exps)), 1e-9)

        up_pct = float(probs[0] * 100.0)
        down_pct = float(probs[1] * 100.0)
        side_pct = float(probs[2] * 100.0)
        total = up_pct + down_pct + side_pct
        if total > 0:
            up_pct = up_pct / total * 100.0
            down_pct = down_pct / total * 100.0
            side_pct = side_pct / total * 100.0

        dominant = "UP" if up_pct >= down_pct and up_pct >= side_pct else "DOWN" if down_pct >= side_pct else "SIDEWAYS"

        return {
            "up_prob": round(up_pct, 2),
            "down_prob": round(down_pct, 2),
            "sideways_prob": round(side_pct, 2),
            "dominant_state": dominant,
            "model_inputs": {
                "momentum_score": round(momentum, 4),
                "flow_score": round(flow, 4),
                "volatility_regime": vol or "UNKNOWN",
                "regime": regime or "unknown",
            },
        }

