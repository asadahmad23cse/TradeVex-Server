import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.dashboard.btc_service as btc_service_module
from src.dashboard.btc_service import BitcoinMarketService


class BitcoinMarketServiceTests(unittest.TestCase):
    def test_klines_to_df_parses_rows(self) -> None:
        rows = [
            [1700000000000, "100.0", "105.0", "99.0", "103.0", "10.5", 1700000059999],
            [1700000060000, "103.0", "106.0", "102.5", "104.5", "9.2", 1700000119999],
        ]
        df = BitcoinMarketService._klines_to_df(rows)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(float(df["Close"].iloc[-1]), 104.5, places=6)
        self.assertIn("Open", df.columns)
        self.assertIn("Volume", df.columns)

    def test_history_payload_contains_metadata(self) -> None:
        svc = BitcoinMarketService({})
        df = BitcoinMarketService._klines_to_df(
            [[1700000000000, "100", "101", "99", "100.5", "1.0", 1700000059999]]
        )
        payload = svc._history_payload(df, interval="1d")
        self.assertEqual(payload["asset"], "BTCUSDT")
        self.assertEqual(payload["points"], 1)
        self.assertEqual(payload["interval"], "1d")
        self.assertEqual(len(payload["data"]), 1)

    def test_realtime_signal_falls_back_when_optional_features_missing(self) -> None:
        svc = BitcoinMarketService({})

        idx = pd.date_range("2026-01-01", periods=130, freq="5min", tz="UTC")
        close = np.linspace(60000.0, 61000.0, len(idx))
        recent = pd.DataFrame(
            {
                "Open": close - 10.0,
                "High": close + 20.0,
                "Low": close - 20.0,
                "Close": close,
                "Volume": np.full(len(idx), 100.0),
            },
            index=idx,
        )

        class _DQ:
            severe = False

            @staticmethod
            def to_dict() -> dict:
                return {"severe": False}

        svc.get_recent_frame = lambda interval="5m", limit=1200: recent  # type: ignore[method-assign]
        svc._anomaly.inspect_and_clean = lambda df, *_: (df, _DQ())  # type: ignore[method-assign]
        svc._engineer.compute_all_features = lambda df, timeframe="intraday": pd.DataFrame(  # type: ignore[method-assign]
            {
                "Close": df["Close"],
                "ATR_14": df["Close"] * 0.003,
                "Volatility_20": pd.Series(0.4, index=df.index),
                "Volume_Ratio": pd.Series(1.0, index=df.index),
            },
            index=df.index,
        )
        svc._alpha.score = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "signal": "BUY",
            "strength": "MODERATE",
            "confidence": 85.0,
            "alpha_score": 0.2,
            "factor_scores": {},
            "ic_weights": {},
        }
        svc._cost.net_alpha = lambda **_kwargs: (0.15, 0.01, True)  # type: ignore[method-assign]

        payload = svc.get_realtime_signal(interval="5m")
        self.assertIn(payload["signal"], {"LONG", "HOLD"})
        if payload["signal"] == "LONG":
            self.assertTrue(payload["validated"])
            self.assertGreater(payload["entry_price"], 0)
            self.assertGreater(payload["stop_loss"], 0)
            self.assertGreater(payload["take_profit"], payload["entry_price"])
        else:
            self.assertIsNone(payload["entry_price"])
            self.assertIsNotNone(payload["reason"])
        self.assertEqual(payload["alpha_score"], 20)
        if payload["signal"] == "LONG":
            self.assertIsNotNone(payload["tp1"])
            self.assertIsNotNone(payload["tp2"])
            self.assertIsNotNone(payload["tp3"])

    def test_hold_signal_returns_null_trade_levels(self) -> None:
        svc = BitcoinMarketService({})

        idx = pd.date_range("2026-01-01", periods=130, freq="5min", tz="UTC")
        close = np.linspace(60000.0, 61000.0, len(idx))
        recent = pd.DataFrame(
            {
                "Open": close - 10.0,
                "High": close + 20.0,
                "Low": close - 20.0,
                "Close": close,
                "Volume": np.full(len(idx), 100.0),
            },
            index=idx,
        )

        class _DQ:
            severe = False

            @staticmethod
            def to_dict() -> dict:
                return {"severe": False}

        svc.get_recent_frame = lambda interval="5m", limit=1200: recent  # type: ignore[method-assign]
        svc._anomaly.inspect_and_clean = lambda df, *_: (df, _DQ())  # type: ignore[method-assign]
        svc._engineer.compute_all_features = lambda df, timeframe="intraday": pd.DataFrame(  # type: ignore[method-assign]
            {"Close": df["Close"]},
            index=df.index,
        )
        svc._alpha.score = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "signal": "HOLD",
            "strength": "WEAK",
            "confidence": 50.0,
            "alpha_score": 0.01,
            "factor_scores": {},
            "ic_weights": {},
        }
        svc._cost.net_alpha = lambda **_kwargs: (0.0, 0.01, False)  # type: ignore[method-assign]

        payload = svc.get_realtime_signal(interval="5m")
        self.assertEqual(payload["signal"], "HOLD")
        self.assertIsNone(payload["entry_price"])
        self.assertIsNone(payload["stop_loss"])
        self.assertIsNone(payload["take_profit"])
        self.assertIsNotNone(payload["reason"])

    def test_signal_markers_contains_long_short(self) -> None:
        svc = BitcoinMarketService({})
        idx = pd.date_range("2024-01-01", periods=150, freq="D", tz="UTC")
        close = np.linspace(40000.0, 70000.0, len(idx))
        history_payload = {
            "data": [
                {
                    "time": int(ts.timestamp()),
                    "open": float(px - 100.0),
                    "high": float(px + 200.0),
                    "low": float(px - 200.0),
                    "close": float(px),
                    "volume": 100.0,
                }
                for ts, px in zip(idx, close)
            ]
        }

        feat = pd.DataFrame(
            {
                "Open": close - 100.0,
                "High": close + 200.0,
                "Low": close - 200.0,
                "Close": close,
                "Volume": np.full(len(idx), 100.0),
                "Returns": pd.Series(close).pct_change().fillna(0.0).to_numpy(),
            },
            index=idx,
        )

        class _DQ:
            severe = False

            @staticmethod
            def to_dict() -> dict:
                return {"severe": False}

        svc.get_all_time_history = lambda interval="1d": history_payload  # type: ignore[method-assign]
        svc._anomaly.inspect_and_clean = lambda df, *_: (df, _DQ())  # type: ignore[method-assign]
        svc._engineer.compute_all_features = lambda df, timeframe="daily": feat  # type: ignore[method-assign]

        z = pd.Series(np.linspace(-1.0, 1.0, len(idx)), index=idx)
        svc._alpha._factor1_momentum = lambda *_args, **_kwargs: z  # type: ignore[method-assign]
        svc._alpha._factor2_mean_reversion = lambda *_args, **_kwargs: z * 0  # type: ignore[method-assign]
        svc._alpha._factor3_volume = lambda *_args, **_kwargs: z * 0  # type: ignore[method-assign]
        svc._alpha._factor4_ml = lambda *_args, **_kwargs: z * 0  # type: ignore[method-assign]
        svc._alpha._factor5_volatility_squeeze = lambda *_args, **_kwargs: z * 0  # type: ignore[method-assign]
        svc._alpha._factor8_microstructure = lambda *_args, **_kwargs: z * 0  # type: ignore[method-assign]
        svc._alpha.alpha_threshold = 0.1
        svc._alpha.ic_window = 20

        with patch.object(btc_service_module, "_rolling_ic", return_value=pd.Series(0.5, index=idx)):
            markers = svc.get_signal_markers(interval="1d", limit=1000)

        self.assertTrue(markers)
        self.assertLessEqual(len(markers), 20)
        shapes = {m.get("shape") for m in markers}
        self.assertIn("arrowUp", shapes)
        self.assertIn("arrowDown", shapes)
        self.assertTrue(all(m.get("text", "") == "" for m in markers))


if __name__ == "__main__":
    unittest.main()
