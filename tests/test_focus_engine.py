import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.dashboard.focus_engine import FocusQuantEngine


def make_intraday_frame(start: datetime, periods: int = 220) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="5min", tz="UTC")
    base = np.linspace(100.0, 105.0, periods)
    close = base + np.sin(np.arange(periods) / 5.0)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.4,
            "Low": close - 0.4,
            "Close": close,
            "Volume": np.linspace(500_000, 900_000, periods),
        },
        index=idx,
    )


class FocusEngineTests(unittest.TestCase):
    def test_default_focus_assets_present(self) -> None:
        engine = FocusQuantEngine({})
        symbols = {a["symbol"] for a in engine.list_assets()}
        self.assertIn("XAUUSD", symbols)
        self.assertIn("XAGUSD", symbols)
        self.assertIn("BTCUSD", symbols)

    def test_trade_rejected_when_data_is_stale(self) -> None:
        engine = FocusQuantEngine({})
        stale_start = datetime.now(timezone.utc) - timedelta(days=2)
        stale_df = make_intraday_frame(stale_start)

        with patch.object(engine, "_fetch_ohlcv", return_value=stale_df), patch.object(
            engine,
            "_score_frame",
            return_value={
                "signal": "BUY",
                "strength": "STRONG",
                "confidence": 79.0,
                "alpha_score": 0.42,
                "factor_scores": {"F1": 0.9},
                "ic_weights": {"F1": 0.2},
                "atr_14": 1.1,
                "volatility_20": 0.2,
                "volume_ratio": 1.1,
            },
        ), patch.object(
            engine,
            "_horizon_scores",
            return_value=[
                {"interval": "5m", "period": "5d", "signal": "BUY", "confidence": 70.0, "alpha_score": 0.2},
                {"interval": "15m", "period": "5d", "signal": "BUY", "confidence": 68.0, "alpha_score": 0.18},
            ],
        ):
            trade = engine.get_focus_trade("XAUUSD", interval="5m", period="5d")

        self.assertFalse(trade["validated"])
        self.assertEqual(trade["validated_signal"], "HOLD")
        self.assertFalse(trade["validation"]["checks"]["fresh_data_ok"])

    def test_trade_validated_when_all_gates_pass(self) -> None:
        engine = FocusQuantEngine({})
        fresh_start = datetime.now(timezone.utc) - timedelta(hours=6)
        fresh_df = make_intraday_frame(fresh_start)

        with patch.object(engine, "_fetch_ohlcv", return_value=fresh_df), patch.object(
            engine,
            "_score_frame",
            return_value={
                "signal": "BUY",
                "strength": "STRONG",
                "confidence": 81.0,
                "alpha_score": 0.55,
                "factor_scores": {"F1": 1.1},
                "ic_weights": {"F1": 0.24},
                "atr_14": 0.9,
                "volatility_20": 0.18,
                "volume_ratio": 1.3,
            },
        ), patch.object(
            engine,
            "_horizon_scores",
            return_value=[
                {"interval": "5m", "period": "5d", "signal": "BUY", "confidence": 73.0, "alpha_score": 0.28},
                {"interval": "15m", "period": "5d", "signal": "BUY", "confidence": 75.0, "alpha_score": 0.25},
                {"interval": "1h", "period": "1mo", "signal": "SELL", "confidence": 60.0, "alpha_score": -0.10},
            ],
        ):
            trade = engine.get_focus_trade("XAGUSD", interval="5m", period="5d")

        self.assertTrue(trade["validation"]["checks"]["fresh_data_ok"])
        self.assertTrue(trade["validation"]["checks"]["consensus_ok"])
        self.assertTrue(trade["validated"])
        self.assertEqual(trade["validated_signal"], "BUY")


if __name__ == "__main__":
    unittest.main()
