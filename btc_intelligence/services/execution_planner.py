from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionPlanInput:
    current_price: float
    atr_pct: float
    volatility_regime: str
    liquidity_score: float
    direction: str


class ExecutionPlanner:
    """Execution planning with volatility-aware zone sizing and liquidity risk."""

    def plan(self, inp: ExecutionPlanInput) -> dict[str, Any]:
        px = max(float(inp.current_price), 0.0)
        if px <= 0:
            return {
                "entry_zone": [0.0, 0.0],
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "take_profit_2": 0.0,
                "slippage_risk": "HIGH",
                "expected_rr": 0.0,
                "reason": "Price unavailable",
            }

        direction = str(inp.direction or "HOLD").upper()
        if direction not in {"LONG", "SHORT"}:
            direction = "LONG"

        atr_pct = max(float(inp.atr_pct), 0.05)
        atr_value = px * (atr_pct / 100.0)
        vol = str(inp.volatility_regime or "").upper()
        liq = min(max(float(inp.liquidity_score), 0.0), 1.0)

        entry_buffer_base = {
            "LOW": 0.03,
            "NORMAL": 0.08,
            "EXPANSION": 0.15,
            "HIGH_VOL": 0.22,
        }.get(vol, 0.10)
        entry_buffer_pct = entry_buffer_base + ((1.0 - liq) * 0.08)
        half_zone = px * (entry_buffer_pct / 100.0)
        entry_min = px - half_zone
        entry_max = px + half_zone
        entry_ref = (entry_min + entry_max) / 2.0

        stop_mult = {
            "LOW": 1.1,
            "NORMAL": 1.5,
            "EXPANSION": 1.9,
            "HIGH_VOL": 2.2,
        }.get(vol, 1.5)
        tp1_mult = stop_mult * 1.2
        tp2_mult = stop_mult * 2.0

        if direction == "LONG":
            stop_loss = entry_ref - (stop_mult * atr_value)
            take_profit = entry_ref + (tp1_mult * atr_value)
            take_profit_2 = entry_ref + (tp2_mult * atr_value)
            risk = max(entry_ref - stop_loss, 1e-9)
            reward = max(take_profit - entry_ref, 0.0)
        else:
            stop_loss = entry_ref + (stop_mult * atr_value)
            take_profit = entry_ref - (tp1_mult * atr_value)
            take_profit_2 = entry_ref - (tp2_mult * atr_value)
            risk = max(stop_loss - entry_ref, 1e-9)
            reward = max(entry_ref - take_profit, 0.0)

        rr = reward / risk if risk > 0 else 0.0

        if liq < 0.35 or vol == "HIGH_VOL":
            slip = "HIGH"
        elif liq < 0.60 or vol == "EXPANSION":
            slip = "MEDIUM"
        else:
            slip = "LOW"

        return {
            "entry_zone": [round(entry_min, 2), round(entry_max, 2)],
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "take_profit_2": round(take_profit_2, 2),
            "slippage_risk": slip,
            "expected_rr": round(rr, 2),
            "atr_value": round(atr_value, 4),
            "entry_buffer_pct": round(entry_buffer_pct, 4),
            "direction": direction,
            "liquidity_score": round(liq, 4),
        }

    @staticmethod
    def merge_tail_risk(plan: dict[str, Any], kelly: dict[str, Any]) -> dict[str, Any]:
        """
        Attach Kelly / tail-risk bounds to the execution plan without removing
        existing keys (expected_rr, entry_zone, etc.).
        """
        out = dict(plan)
        if not isinstance(kelly, dict):
            return out
        pct = float(kelly.get("position_pct", 0.0))
        p = float(kelly.get("p", 0.0))
        b = float(kelly.get("b", 0.0))
        rk = float(kelly.get("raw_kelly", 0.0))
        rb = kelly.get("risk_budgets") if isinstance(kelly.get("risk_budgets"), dict) else {}
        out["position_size_pct"] = pct
        out["position_size_usd"] = float(kelly.get("position_size_usd", 0.0))
        out["p"] = p
        out["b"] = b
        out["raw_kelly"] = rk
        out["tail_risk_sizing"] = {
            "bounded_position_pct": pct,
            "p": p,
            "b": b,
            "raw_kelly": rk,
            "risk_budgets": dict(rb),
        }
        return out

