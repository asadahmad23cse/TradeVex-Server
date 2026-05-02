import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.dashboard.altcoin_service import AltcoinMarketService


class AltcoinServiceScoringTests(unittest.TestCase):
    def _service_with_single_factor(self, latest_value: float) -> AltcoinMarketService:
        svc = AltcoinMarketService({"signal": {"ic_window": 10, "alpha_score_threshold": 0.15}})
        index = pd.date_range("2026-04-29", periods=24, freq="15min", tz="UTC")
        active = pd.Series(float(latest_value), index=index)
        empty = pd.Series(0.0, index=index)
        svc._alpha._factor1_momentum = lambda _df, _hurst=0.5: active
        svc._alpha._factor2_mean_reversion = lambda _df, _hurst=0.5: empty
        svc._alpha._factor3_volume = lambda _df: empty
        svc._alpha._factor4_ml = lambda _scores: empty
        svc._alpha._factor5_volatility_squeeze = lambda _df, momentum_factor=None: empty
        svc._alpha._factor8_microstructure = lambda _df: empty
        return svc

    def _frame(self) -> pd.DataFrame:
        index = pd.date_range("2026-04-29", periods=24, freq="15min", tz="UTC")
        return pd.DataFrame(
            {
                "Close": np.linspace(2200.0, 2250.0, len(index)),
                "Returns": np.linspace(-0.002, 0.002, len(index)),
                "ATR_14": np.full(len(index), 12.0),
            },
            index=index,
        )

    def test_score_frame_uses_latest_usable_ic_before_trailing_bar(self) -> None:
        svc = self._service_with_single_factor(1.0)

        def fake_rolling_ic(factor, _fwd_ret, _window):
            return pd.Series([0.0] * (len(factor) - 2) + [0.3, 0.0], index=factor.index)

        with patch("src.dashboard.altcoin_service._rolling_ic", fake_rolling_ic):
            score = svc._score_frame(self._frame())

        self.assertIsNotNone(score)
        assert score is not None
        self.assertGreater(float(score["alpha_score"]), 0.0)
        self.assertEqual(score["signal"], "BUY")
        self.assertAlmostEqual(float(score["ic_weights"]["F1"]), 0.3, places=4)

    def test_negative_alpha_has_directional_confidence_for_short(self) -> None:
        svc = self._service_with_single_factor(-1.0)

        def fake_rolling_ic(factor, _fwd_ret, _window):
            return pd.Series([0.0] * (len(factor) - 2) + [0.3, 0.0], index=factor.index)

        with patch("src.dashboard.altcoin_service._rolling_ic", fake_rolling_ic):
            score = svc._score_frame(self._frame())

        self.assertIsNotNone(score)
        assert score is not None
        self.assertLess(float(score["alpha_score"]), 0.0)
        self.assertEqual(score["signal"], "SELL")
        self.assertGreater(float(score["confidence"]), 50.0)


if __name__ == "__main__":
    unittest.main()
