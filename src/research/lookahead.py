"""
Lookahead-bias audit helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


FEATURE_MIN_LAGS = {
    "SMA_20": 20,
    "SMA_50": 50,
    "EMA_20": 20,
    "RSI": 14,
    "MACD": 26,
    "BB_Upper": 20,
    "ATR_14": 14,
    "Stoch_K": 14,
    "CCI": 20,
    "ROC_60d": 60,
    "Ichimoku_Kijun": 26,
    "Ichimoku_SpanB": 52,
    "OBV_Slope": 5,
    "CMF": 20,
    "Volume_Osc": 20,
    "ATR_Percentile": 90,
    "Rolling_Beta": 60,
}


@dataclass
class LookaheadAuditResult:
    passed: bool
    issues: list[str]


class LookaheadBiasAuditor:
    def audit(self, raw_df: pd.DataFrame, feature_df: pd.DataFrame) -> LookaheadAuditResult:
        issues: list[str] = []
        if raw_df.empty or feature_df.empty:
            return LookaheadAuditResult(passed=False, issues=["empty_input"])

        for feature, min_lag in FEATURE_MIN_LAGS.items():
            if feature not in feature_df.columns:
                continue
            first_valid = feature_df[feature].first_valid_index()
            if first_valid is None:
                issues.append(f"{feature}: never becomes valid")
                continue
            raw_position = raw_df.index.get_indexer([first_valid])[0]
            if raw_position < max(min_lag - 1, 0):
                issues.append(
                    f"{feature}: first valid row {raw_position} violates minimum lag {min_lag}"
                )

        if "Ichimoku_Chikou" in feature_df.columns:
            # Chikou should lag current close, not lead it.
            overlap = feature_df[["Ichimoku_Chikou", "Close"]].dropna().tail(5)
            if not overlap.empty and (overlap["Ichimoku_Chikou"] == overlap["Close"]).all():
                issues.append("Ichimoku_Chikou appears contemporaneous; verify lagging logic")

        return LookaheadAuditResult(passed=not issues, issues=issues)
