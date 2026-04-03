from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetaDecisionInput:
    base_decision: str
    base_confidence: float
    calibrated_probability: float
    strategy_used: str
    blockers: list[str]
    validation: dict[str, Any]
    meta_label: dict[str, Any]
    drift: dict[str, Any]
    edge_decay: dict[str, Any]
    execution_plan: dict[str, Any]
    adaptive_meta: dict[str, Any]
    drawdown_action: str = "NORMAL"


class MetaDecisionEngine:
    """
    Final decision layer:
    prioritizes survival and consistency over aggressiveness.
    """

    def evaluate(self, inp: MetaDecisionInput) -> dict[str, Any]:
        decision = str(inp.base_decision or "HOLD").upper()
        conf = float(max(0.0, min(100.0, inp.base_confidence)))
        prob = float(max(0.0, min(1.0, inp.calibrated_probability)))
        strategy = str(inp.strategy_used or "trend_following")

        blockers = list(inp.blockers or [])
        validation_ok = bool(inp.validation.get("approved", False))
        meta_allow = bool(inp.meta_label.get("allow_trade", False))
        drift_level = str(inp.drift.get("drift_level", "LOW")).upper()
        drift_conf_mul = float(inp.drift.get("confidence_multiplier", 1.0))
        edge_reduce = float(inp.edge_decay.get("reduce_risk_factor", 1.0))
        edge_decay = bool(inp.edge_decay.get("decay_detected", False))
        drawdown_action = str(inp.drawdown_action or "NORMAL").upper()

        final_decision = decision
        adjustments: list[str] = []

        if decision not in {"LONG", "SHORT"}:
            final_decision = "HOLD"
            adjustments.append("base_hold")
        if blockers:
            final_decision = "HOLD"
            adjustments.append("primary_blockers")
        if not validation_ok:
            final_decision = "HOLD"
            adjustments.append("validation_reject")
        if not meta_allow:
            final_decision = "HOLD"
            adjustments.append("meta_label_reject")
        if drift_level == "HIGH":
            final_decision = "HOLD"
            adjustments.append("high_drift_safety")
        if drawdown_action == "HALT":
            final_decision = "HOLD"
            adjustments.append("drawdown_circuit_breaker")

        adjusted_conf = conf * drift_conf_mul * edge_reduce
        if final_decision == "HOLD":
            adjusted_conf = min(adjusted_conf, 49.0)
        adjusted_conf = float(max(0.0, min(100.0, adjusted_conf)))

        risk_score = 35.0
        risk_score += 25.0 * float(inp.drift.get("drift_score", 0.0))
        risk_score += 18.0 if edge_decay else 0.0
        risk_score += 12.0 if bool(blockers) else 0.0
        risk_score += 10.0 if not meta_allow else 0.0
        risk_score += 12.0 if not validation_ok else 0.0
        risk_score = float(max(0.0, min(100.0, risk_score)))

        validation_status = "APPROVED" if validation_ok and meta_allow and final_decision in {"LONG", "SHORT"} else "REJECTED"
        if drawdown_action == "HALT":
            reason = "drawdown_circuit_breaker"
        else:
            reason = " | ".join(adjustments[:4]) if adjustments else "validated_trade"

        return {
            "decision": final_decision,
            "confidence": round(adjusted_conf, 2),
            "probability": round(prob * 100.0, 2),
            "strategy_used": strategy,
            "execution_plan": inp.execution_plan,
            "risk_score": round(risk_score, 2),
            "validation_status": validation_status,
            "adaptive_adjustments": adjustments,
            "reason": reason,
            "inputs": {
                "base_decision": decision,
                "base_confidence": round(conf, 2),
                "edge_decay": bool(edge_decay),
                "drift_level": drift_level,
                "drawdown_action": drawdown_action,
            },
            "adaptive_meta": inp.adaptive_meta,
        }
