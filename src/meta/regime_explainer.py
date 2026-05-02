"""Structured explanations for already-blocked regime gates."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RegimeBlockExplainer:
    """Attaches block_reason_detail without changing block decisions."""

    REGIME_BLOCKERS = {"regime_gate", "regime_filter"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def explain(self, signal_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        payload = signal_payload or {}
        blocked_by = str(payload.get("blocked_by") or "").strip().lower()
        checks = payload.get("validation_checks") if isinstance(payload.get("validation_checks"), dict) else {}
        regime_gate_failed = str(checks.get("regime_gate") or "").upper() == "FAIL" or checks.get("regime_gate") is False
        if blocked_by not in self.REGIME_BLOCKERS and not regime_gate_failed:
            return None

        action = str(payload.get("requested_signal") or payload.get("signal") or payload.get("direction") or "").upper()
        if action in {"BUY", "LONG"}:
            action = "LONG"
        elif action in {"SELL", "SHORT"}:
            action = "SHORT"
        regime = str(payload.get("regime") or "unknown")
        indicator = self._indicator(payload)
        factors = self._supporting_factors(payload)
        detail = {
            "regime": regime,
            "indicator": indicator,
            "supporting_factors": factors,
            "blocked_action": action or "UNKNOWN",
        }
        regime_label = self._regime_label(regime)
        indicator_label = "HTF EMA" if indicator == "HTF_EMA" else indicator
        logger.info(
            "[REGIME_BLOCK] %s blocked due to %s %s trend",
            detail["blocked_action"],
            regime_label,
            indicator_label,
        )
        return detail

    def attach(self, signal_payload: dict[str, Any]) -> dict[str, Any]:
        detail = self.explain(signal_payload)
        if detail is not None:
            signal_payload["block_reason_detail"] = detail
        return signal_payload

    @staticmethod
    def _indicator(payload: dict[str, Any]) -> str:
        reason = str(payload.get("reason") or "").lower()
        if "ema" in reason or "bullish trend" in reason or "bearish trend" in reason:
            return "HTF_EMA"
        if "multi" in reason or "mtf" in reason:
            return "MTF_BIAS"
        return "HTF_EMA"

    @staticmethod
    def _supporting_factors(payload: dict[str, Any]) -> list[str]:
        factors: list[str] = []
        if payload.get("funding_rate_z") is not None or payload.get("funding_rate_pct") is not None:
            factors.append("funding_rate")
        overview = payload.get("market_overview") if isinstance(payload.get("market_overview"), dict) else {}
        if overview.get("volatility") or overview.get("atr_pct") is not None:
            factors.append("volatility")
        if payload.get("mtf_bias") or payload.get("bias_4h") or payload.get("bias_1d"):
            factors.append("multi_timeframe")
        if payload.get("etf_flow") is not None:
            factors.append("etf_flow")
        return factors

    @staticmethod
    def _regime_label(regime: str) -> str:
        text = str(regime or "").upper()
        if "BULL" in text:
            return "bullish"
        if "BEAR" in text:
            return "bearish"
        return str(regime or "unknown").lower()
