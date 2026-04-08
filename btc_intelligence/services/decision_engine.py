from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecisionEngineInput:
    regime: str
    cvd_slope: float
    obi_imbalance: float
    volatility_regime: str
    cost_score: float
    momentum_score: float
    aggression_buy_pct: float
    flow_score: float
    # Alpha barrier: require E[Ret]*P(win) > spread + slippage (all as fractions of notional).
    expected_return_frac: float | None = None
    win_prob: float | None = None
    spread_frac: float | None = None
    slippage_frac: float | None = None


class DecisionEngine:
    """Institutional decision validator for tradeability, direction, and explainability."""

    def evaluate(self, inp: DecisionEngineInput) -> dict[str, Any]:
        regime_raw = str(inp.regime or "").lower()
        vol_raw = str(inp.volatility_regime or "").upper()

        momentum = float(np.clip(inp.momentum_score, -1.0, 1.0))
        raw_flow = float(inp.flow_score)
        x = raw_flow
        if abs(x) > 1.0:
            x = x / 100.0
        x = float(np.clip(x, -20.0, 20.0))
        flow = float(np.clip(x, -1.0, 1.0))
        if abs(raw_flow - flow) > 1e-6:
            logger.debug("flow_score raw=%.3f clipped=%.3f", raw_flow, flow)
        cost = float(np.clip(inp.cost_score, -1.0, 1.0))
        obi = float(np.clip(inp.obi_imbalance, -1.0, 1.0))
        cvd_slope = float(inp.cvd_slope)
        aggression = float(np.clip((inp.aggression_buy_pct - 50.0) / 50.0, -1.0, 1.0))

        regime_dir = self._regime_direction(regime_raw)
        directional_bias = float(np.clip((0.65 * flow) + (0.35 * momentum), -1.0, 1.0))
        regime_alignment = directional_bias * regime_dir if regime_dir != 0 else 0.0
        regime_score = float(np.clip(regime_alignment, -1.0, 1.0))
        flow_score = float(np.clip((0.60 * flow) + (0.25 * obi) + (0.15 * aggression), -1.0, 1.0))
        momentum_score = float(momentum)
        cost_score = float(cost)
        vol_score = self._volatility_score(vol_raw)

        weights = {
            "regime": 0.22,
            "momentum": 0.20,
            "flow": 0.30,
            "cost": 0.16,
            "volatility": 0.12,
        }
        final_score = (
            (weights["regime"] * regime_score)
            + (weights["momentum"] * momentum_score)
            + (weights["flow"] * flow_score)
            + (weights["cost"] * cost_score)
            + (weights["volatility"] * vol_score)
        )
        final_score = float(np.clip(final_score, -1.0, 1.0))

        blockers: list[str] = []
        if vol_raw in {"LOW", "COMPRESSION"}:
            blockers.append("Volatility LOW: no trade until expansion")
        if vol_raw in {"HIGH_VOL", "PANIC"}:
            blockers.append("Extreme volatility: execution risk too high")
        if cost_score < -0.05:
            blockers.append("Cost gate negative: expected edge after costs <= 0")
        if regime_dir != 0 and directional_bias != 0 and np.sign(regime_dir) != np.sign(directional_bias):
            blockers.append("Regime conflict with flow/momentum direction")

        er = inp.expected_return_frac
        pw = inp.win_prob
        sf = inp.spread_frac
        slf = inp.slippage_frac
        if (
            er is not None
            and pw is not None
            and sf is not None
            and slf is not None
            and er >= 0.0
            and pw >= 0.0
        ):
            ev = float(er) * float(pw)
            cost_line = float(max(sf, 0.0)) + float(max(slf, 0.0))
            if ev <= cost_line + 1e-12:
                blockers.append(
                    f"Alpha barrier: E[Ret]*P(win)={ev:.6f} <= spread+slip={cost_line:.6f}"
                )

        decision = "HOLD"
        if not blockers:
            if final_score >= 0.12:
                decision = "LONG"
            elif final_score <= -0.12:
                decision = "SHORT"

        confidence = self._confidence(final_score, blockers)
        reason = self._build_reason(decision, final_score, blockers, vol_raw, regime_raw)
        triggers = self._build_trade_triggers(
            decision=decision,
            blockers=blockers,
            cvd_slope=cvd_slope,
            obi=obi,
            volatility_regime=vol_raw,
            cost_score=cost_score,
            directional_bias=directional_bias,
        )

        factor_contributions = [
            self._factor("regime", regime_score, weights["regime"]),
            self._factor("momentum", momentum_score, weights["momentum"]),
            self._factor("order_flow", flow_score, weights["flow"]),
            self._factor("cost", cost_score, weights["cost"]),
            self._factor("volatility", vol_score, weights["volatility"]),
        ]

        trade_verdict = {
            "regime_alignment": "HIGH" if (regime_dir == 0 or regime_alignment >= 0) else "LOW",
            "liquidity_quality": "STRONG" if abs(obi) >= 0.18 and abs(aggression) >= 0.10 else "WEAK",
            "volatility_state": "TRADEABLE" if vol_raw in {"NORMAL", "EXPANSION"} else "NOT_TRADEABLE",
            "final_verdict": "TRADE" if decision in {"LONG", "SHORT"} and not blockers else "AVOID",
        }

        return {
            "decision": decision,
            "confidence": round(confidence, 2),
            "reason": reason,
            "blockers": blockers,
            "trade_triggers": triggers,
            "decision_breakdown": {
                "regime_score": round(regime_score * 100.0, 2),
                "momentum_score": round(momentum_score * 100.0, 2),
                "flow_score": round(flow_score * 100.0, 2),
                "cost_score": round(cost_score * 100.0, 2),
                "final_score": round(final_score * 100.0, 2),
                "explanation": self._explain_breakdown(
                    regime_score=regime_score,
                    momentum_score=momentum_score,
                    flow_score=flow_score,
                    cost_score=cost_score,
                    final_score=final_score,
                ),
            },
            "factor_contributions": factor_contributions,
            "trade_verdict": trade_verdict,
        }

    @staticmethod
    def _regime_direction(regime: str) -> int:
        if any(x in regime for x in ("bear", "breakout_down", "panic")):
            return -1
        if any(x in regime for x in ("bull", "breakout_up")):
            return 1
        return 0

    @staticmethod
    def _volatility_score(volatility_regime: str) -> float:
        if volatility_regime in {"NORMAL"}:
            return 0.45
        if volatility_regime in {"EXPANSION"}:
            return 0.10
        if volatility_regime in {"LOW", "COMPRESSION"}:
            return -0.35
        if volatility_regime in {"HIGH_VOL", "PANIC"}:
            return -0.45
        return 0.0

    @staticmethod
    def _confidence(final_score: float, blockers: list[str]) -> float:
        base = (abs(final_score) * 100.0) + 18.0
        penalty = float(len(blockers) * 12.0)
        return float(np.clip(base - penalty, 0.0, 100.0))

    @staticmethod
    def _build_reason(
        decision: str,
        final_score: float,
        blockers: list[str],
        volatility_regime: str,
        regime: str,
    ) -> str:
        if blockers:
            return f"Blocked ({'; '.join(blockers[:2])})"
        if decision == "LONG":
            return f"Long bias confirmed by flow/momentum in {regime or 'current'} regime ({volatility_regime})"
        if decision == "SHORT":
            return f"Short bias confirmed by flow/momentum in {regime or 'current'} regime ({volatility_regime})"
        return f"No directional edge: final score {final_score * 100.0:.1f}"

    @staticmethod
    def _build_trade_triggers(
        decision: str,
        blockers: list[str],
        cvd_slope: float,
        obi: float,
        volatility_regime: str,
        cost_score: float,
        directional_bias: float,
    ) -> list[str]:
        triggers: list[str] = []
        target_long = decision == "LONG" or (decision == "HOLD" and directional_bias >= 0)
        target_short = decision == "SHORT" or (decision == "HOLD" and directional_bias < 0)

        if target_long:
            if cvd_slope <= 0:
                triggers.append("CVD must flip positive")
            if obi <= 0.30:
                triggers.append("OBI > 0.30")
        if target_short:
            if cvd_slope >= 0:
                triggers.append("CVD must flip negative")
            if obi >= -0.30:
                triggers.append("OBI < -0.30")

        if volatility_regime in {"LOW", "COMPRESSION"}:
            triggers.append("Volatility must expand")
        if volatility_regime in {"HIGH_VOL", "PANIC"}:
            triggers.append("Volatility must normalize")
        if cost_score <= 0:
            triggers.append("Net alpha after costs must turn positive")
        if any("Regime conflict" in b for b in blockers):
            triggers.append("Regime alignment required")

        if decision in {"LONG", "SHORT"} and not triggers:
            triggers.append("Execution window open")
        return triggers[:6]

    @staticmethod
    def _factor(name: str, value: float, weight: float) -> dict[str, Any]:
        contribution = float(value) * float(weight)
        return {
            "name": name,
            "value": round(float(value), 6),
            "weight": round(float(weight), 6),
            "contribution": round(float(contribution), 6),
        }

    @staticmethod
    def _explain_breakdown(
        regime_score: float,
        momentum_score: float,
        flow_score: float,
        cost_score: float,
        final_score: float,
    ) -> str:
        dominant = max(
            [
                ("regime", abs(regime_score)),
                ("momentum", abs(momentum_score)),
                ("flow", abs(flow_score)),
                ("cost", abs(cost_score)),
            ],
            key=lambda x: x[1],
        )[0]
        direction = "bullish" if final_score > 0 else "bearish" if final_score < 0 else "neutral"
        return f"{direction.capitalize()} tilt led by {dominant} contribution."

