import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features.engineer import FeatureEngineer
from src.risk.portfolio import PortfolioTracker
from src.signals.engine import SignalEngine, TradingSignal
from src.signals.store import SignalStore


def make_ohlcv(periods: int = 180) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="D")
    base = np.linspace(100.0, 130.0, periods)
    close = base + np.sin(np.arange(periods))
    frame = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1_000_000, 1_500_000, periods),
        },
        index=index,
    )
    return frame


class QuantPipelineTests(unittest.TestCase):
    def test_daily_feature_engineering_keeps_latest_bar(self) -> None:
        df = make_ohlcv()
        features = FeatureEngineer().compute_all_features(df, timeframe="daily")

        self.assertFalse(features.empty)
        self.assertEqual(features.index[-1], df.index[-1])
        self.assertTrue(features["Ichimoku_Chikou"].notna().iloc[-1])

    def test_portfolio_updates_equity_by_position_weight(self) -> None:
        portfolio = PortfolioTracker(initial_capital=100_000.0)
        signal = SimpleNamespace(
            signal_id="sig-1",
            asset="AAPL",
            asset_class="us_stock",
            signal="BUY",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=105.0,
            position_size_pct=2.0,
            slippage_cost_pct=0.0,
        )

        portfolio.open_position(signal)
        close_events = portfolio.update_prices({"AAPL": 110.0})

        self.assertEqual(len(close_events), 1)
        self.assertAlmostEqual(close_events[0]["pnl_pct"], 10.0, places=4)
        self.assertAlmostEqual(portfolio.equity, 100_200.0, places=4)

    def test_signal_store_buckets_are_regime_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SignalStore(Path(tmp_dir) / "signals.db")

            bull = TradingSignal(
                asset="AAPL",
                asset_class="us_stock",
                timeframe="intraday",
                signal="BUY",
                strength="STRONG",
                confidence=80.0,
                alpha_score=0.9,
                regime="BULL",
                entry_price=100.0,
                stop_loss=98.0,
                take_profit=106.0,
                position_size_pct=2.0,
                kelly_fraction=8.0,
                factor_scores={"F1": 1.0},
                ic_weights={"F1": 0.2},
                slippage_cost_pct=0.0001,
            )
            bear = TradingSignal(
                asset="AAPL",
                asset_class="us_stock",
                timeframe="intraday",
                signal="SELL",
                strength="STRONG",
                confidence=80.0,
                alpha_score=-0.9,
                regime="BEAR",
                entry_price=100.0,
                stop_loss=102.0,
                take_profit=94.0,
                position_size_pct=2.0,
                kelly_fraction=8.0,
                factor_scores={"F1": -1.0},
                ic_weights={"F1": 0.2},
                slippage_cost_pct=0.0001,
            )

            store.save_signal(bull)
            store.save_signal(bear)
            store.close_signal(bull.signal_id, "WIN", 106.0, 6.0)
            store.close_signal(bear.signal_id, "LOSS", 102.0, -2.0)

            bull_stats = store.get_bucket_stats("us_stock", "STRONG", "BULL", signal="BUY")
            bear_stats = store.get_bucket_stats("us_stock", "STRONG", "BEAR", signal="SELL")

            self.assertEqual(bull_stats["total"], 1)
            self.assertEqual(bear_stats["total"], 1)
            self.assertEqual(bull_stats["win_rate"], 1.0)
            self.assertEqual(bear_stats["win_rate"], 0.0)
            store.engine.dispose()

    def test_signal_engine_risk_pct_uses_portfolio_fraction(self) -> None:
        engine = SignalEngine()
        signal = engine.generate(
            asset="AAPL",
            asset_class="us_stock",
            timeframe="intraday",
            alpha_result={
                "signal": "BUY",
                "strength": "STRONG",
                "confidence": 82.0,
                "alpha_score": 0.7,
                "factor_scores": {"F1": 0.7},
                "ic_weights": {"F1": 0.2},
            },
            regime="BULL",
            regime_allows=True,
            hurst=0.6,
            entry_price=100.0,
            atr_14=1.0,
            kelly_fraction=9.0,
            position_size_pct=2.0,
        )

        assert signal is not None
        self.assertAlmostEqual(signal.risk_pct, 0.03, places=4)


if __name__ == "__main__":
    unittest.main()
