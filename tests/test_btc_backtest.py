import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.btc_backtest import (
    BTCHistoricalLoader,
    BTCFeatureGenerator,
    BTCWalkForwardBacktest,
    BTCBacktestReport,
)


def _make_klines_rows(start_ms: int, count: int, step_ms: int) -> list[list[object]]:
    rows: list[list[object]] = []
    base = 20_000.0
    for i in range(count):
        ts = start_ms + (i * step_ms)
        o = base + i
        h = o + 8.0
        l = o - 8.0
        c = o + 2.0
        v = 100.0 + i
        rows.append([ts, str(o), str(h), str(l), str(c), str(v)])
    return rows


def _make_market_df(periods: int = 420, freq: str = "1D") -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=periods, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    drift = np.linspace(0.0, 2000.0, periods)
    noise = rng.normal(0.0, 25.0, periods).cumsum()
    close = 20_000.0 + drift + noise
    open_ = close + rng.normal(0.0, 5.0, periods)
    high = np.maximum(open_, close) + rng.uniform(3.0, 15.0, periods)
    low = np.minimum(open_, close) - rng.uniform(3.0, 15.0, periods)
    volume = 1_000.0 + rng.uniform(0.0, 600.0, periods)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class BTCHistoricalLoaderTests(unittest.TestCase):
    def test_loader_returns_valid_ohlcv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loader = BTCHistoricalLoader(cache_dir=tmp)
            rows = _make_klines_rows(
                start_ms=int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000),
                count=30,
                step_ms=3_600_000,
            )
            loader._fetch_klines = lambda **_kwargs: rows  # type: ignore[method-assign]
            df = loader.fetch(symbol="BTCUSDT", interval="1h", start_date="2023-01-01", end_date="2023-01-03")
            self.assertFalse(df.empty)
            self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
            self.assertEqual(df.index.name, "timestamp")
            self.assertEqual(str(df.index.tz), "UTC")

    def test_cache_loads_on_second_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = _make_klines_rows(
                start_ms=int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000),
                count=40,
                step_ms=3_600_000,
            )
            loader_1 = BTCHistoricalLoader(cache_dir=tmp)
            loader_1._fetch_klines = lambda **_kwargs: rows  # type: ignore[method-assign]
            df_first = loader_1.fetch(symbol="BTCUSDT", interval="1h", start_date="2023-01-01", end_date="2023-01-03")
            self.assertGreater(len(df_first), 0)

            loader_2 = BTCHistoricalLoader(cache_dir=tmp)

            def _no_network(**_kwargs):
                raise AssertionError("Cache miss: network should not be called")

            loader_2._fetch_klines = _no_network  # type: ignore[method-assign]
            df_second = loader_2.fetch(symbol="BTCUSDT", interval="1h", start_date="2023-01-01", end_date="2023-01-03")
            self.assertEqual(len(df_first), len(df_second))


class BTCBacktestFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.market_df = _make_market_df(periods=420, freq="1D")
        cls.backtest = BTCWalkForwardBacktest(
            min_folds=6,
            alpha_threshold_grid=[0.20],
            ic_window_grid=[24],
            confidence_floor_grid=[60],
        )
        cls.results = cls.backtest.run(cls.market_df, symbol="BTCUSDT", interval="1d")

    def test_feature_generation_adds_crypto_proxy_columns(self) -> None:
        gen = BTCFeatureGenerator()
        feat = gen.generate(self.market_df.copy())
        self.assertIn("funding_rate_z", feat.columns)
        self.assertIn("oi_proxy", feat.columns)
        self.assertIn("etf_flow_proxy", feat.columns)

    def test_wfo_produces_minimum_folds(self) -> None:
        self.assertNotIn("error", self.results)
        folds = self.results.get("folds", [])
        self.assertGreaterEqual(len(folds), 6)
        self.assertTrue(all(int(f.get("test_days", 0)) == 30 for f in folds))

    def test_fold_metrics_keys_present(self) -> None:
        self.assertNotIn("error", self.results)
        fold0 = self.results["folds"][0]["metrics"]
        self.assertIn("sharpe", fold0)
        self.assertIn("win_rate", fold0)
        self.assertIn("profit_factor", fold0)
        self.assertIn("max_drawdown_pct", fold0)

    def test_report_saved_to_data_dir(self) -> None:
        self.assertNotIn("error", self.results)
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                yaml.safe_dump({"signal": {"alpha_score_threshold": 0.15, "ic_window": 60}}),
                encoding="utf-8",
            )
            report = BTCBacktestReport(output_dir=tmp, config_path=str(cfg_path))
            report.generate(dict(self.results))
            files = list(Path(tmp).glob("btc_backtest_report_*.json"))
            self.assertTrue(files)


if __name__ == "__main__":
    unittest.main()
