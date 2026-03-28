"""
Historical crisis stress tests.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)

CRISIS_WINDOWS = {
    "gfc_2008": ("2008-01-01", "2009-06-30"),
    "covid_2020": ("2020-02-01", "2020-09-30"),
    "rate_shock_2022": ("2022-01-01", "2022-12-31"),
    "taper_tantrum_2013": ("2013-05-01", "2013-12-31"),
}


class HistoricalStressTester:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path = config_path
        self.engine = BacktestEngine(config_path=config_path)

    def run(self, ticker: str, asset_class: str = "us_stock") -> pd.DataFrame:
        rows = []
        for label, (start, end) in CRISIS_WINDOWS.items():
            try:
                result = self.engine.run(ticker=ticker, start=start, end=end, asset_class=asset_class)
                summary = result.summary()
                rows.append(
                    {
                        "crisis": label,
                        "start": start,
                        "end": end,
                        "total_return_pct": summary.get("total_return_pct", 0.0),
                        "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
                        "sharpe_ratio": summary.get("sharpe_ratio", 0.0),
                        "survived": summary.get("max_drawdown_pct", 999.0) > -20.0,
                    }
                )
            except Exception as exc:
                logger.warning("Stress test failed for %s: %s", label, exc)
                rows.append(
                    {
                        "crisis": label,
                        "start": start,
                        "end": end,
                        "total_return_pct": 0.0,
                        "max_drawdown_pct": 0.0,
                        "sharpe_ratio": 0.0,
                        "survived": False,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def gate_result(stress_df: pd.DataFrame, required_survivals: int = 2) -> dict:
        survived = int(stress_df.get("survived", pd.Series(dtype=bool)).sum())
        return {
            "survived_crises": survived,
            "required_survivals": required_survivals,
            "passed": survived >= required_survivals,
        }
