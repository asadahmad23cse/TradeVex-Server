import unittest

import numpy as np
import pandas as pd

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
            {"Close": df["Close"]},
            index=df.index,
        )
        svc._alpha.score = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "signal": "BUY",
            "strength": "MODERATE",
            "confidence": 70.0,
            "alpha_score": 0.2,
            "factor_scores": {},
            "ic_weights": {},
        }
        svc._cost.net_alpha = lambda **_kwargs: (0.15, 0.01, True)  # type: ignore[method-assign]

        payload = svc.get_realtime_signal(interval="5m")
        self.assertEqual(payload["signal"], "BUY")
        self.assertTrue(payload["validated"])
        self.assertGreater(payload["entry_price"], 0)
        self.assertGreater(payload["stop_loss"], 0)
        self.assertGreater(payload["take_profit"], payload["entry_price"])


if __name__ == "__main__":
    unittest.main()
