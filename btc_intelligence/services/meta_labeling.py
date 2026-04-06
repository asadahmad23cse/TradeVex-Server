from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REGIME_CONFIDENCE_GATES = {
    "TRENDING": 55.0,
    "BULLISH": 55.0,
    "BEARISH": 58.0,
    "RANGE": 68.0,
    "SQUEEZE": 72.0,
    "VOLATILE": 75.0,
    "SIDEWAYS": 68.0,
    "DEFAULT": 60.0,
}


def get_dynamic_confidence_gate(regime: str) -> float:
    regime_upper = str(regime or "").upper()
    for key, gate in REGIME_CONFIDENCE_GATES.items():
        if key == "DEFAULT":
            continue
        if key in regime_upper:
            return float(gate)
    return float(REGIME_CONFIDENCE_GATES["DEFAULT"])


@dataclass
class MetaLabelingConfig:
    min_confidence: float = 55.0
    min_probability: float = 0.56
    max_calibration_error: float = 0.22


class MetaLabelingEngine:
    """
    Second-stage trade filter:
    keep only high-conviction, calibrated, execution-worthy signals.
    """

    def __init__(self, config: MetaLabelingConfig | None = None) -> None:
        self.config = config or MetaLabelingConfig()

    def label_trade(
        self,
        *,
        base_decision: str,
        confidence: float,
        calibrated_prob: float,
        blockers: list[str] | None = None,
        calibration_error: float = 0.0,
        drift_level: str = "LOW",
        regime: str = "DEFAULT",
        edge_decay: bool = False,
    ) -> dict[str, Any]:
        decision = str(base_decision or "HOLD").upper()
        blockers_list = list(blockers or [])
        conf = float(max(0.0, min(100.0, confidence)))
        prob = float(max(0.0, min(1.0, calibrated_prob)))
        ece = float(max(0.0, calibration_error))
        drift = str(drift_level or "LOW").upper()
        confidence_gate = get_dynamic_confidence_gate(regime)

        reasons: list[str] = []
        allow = decision in {"LONG", "SHORT"}
        decay_override_granted = False
        if decision == "HOLD":
            reasons.append("Base decision HOLD")
            allow = False
        if blockers_list:
            reasons.append("Primary blockers active")
            allow = False
        if conf < confidence_gate:
            reasons.append(f"Confidence below threshold ({conf:.1f} < {confidence_gate:.1f})")
            allow = False
        if prob < self.config.min_probability:
            reasons.append(f"Calibrated probability too low ({prob:.2f})")
            allow = False
        if ece > self.config.max_calibration_error:
            reasons.append(f"Calibration instability (ECE {ece:.3f})")
            allow = False
        if drift == "HIGH":
            reasons.append("High data drift detected")
            allow = False
        if edge_decay:
            reasons.append("Edge decay safeguard active")
            decay_override_granted = bool(prob >= 0.62 and conf >= 65.0)
            if not decay_override_granted:
                reasons.append("Edge decay override denied: requires prob >= 0.62 and confidence >= 65.0")
                allow = False

        if allow:
            label = "ALLOW"
            final_decision = decision
            reason = "Meta-label approved high-quality setup"
        else:
            label = "REJECT"
            final_decision = "HOLD"
            reason = reasons[0] if reasons else "Meta-label rejected setup"

        return {
            "label": label,
            "allow_trade": bool(allow),
            "edge_decay_override": bool(edge_decay and allow and decay_override_granted),
            "final_decision": final_decision,
            "reason": reason,
            "reasons": reasons,
            "inputs": {
                "confidence": round(conf, 2),
                "calibrated_prob": round(prob, 6),
                "calibration_error": round(ece, 6),
                "drift_level": drift,
                "regime": str(regime or "DEFAULT").upper(),
                "confidence_gate": round(confidence_gate, 2),
                "edge_decay": bool(edge_decay),
            },
        }
