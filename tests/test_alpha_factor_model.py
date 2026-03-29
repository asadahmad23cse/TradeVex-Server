import unittest
from unittest.mock import patch

import pandas as pd

import src.alpha.factor_model as factor_model_module
from src.alpha.factor_model import AlphaFactorModel


class AlphaFactorModelTests(unittest.TestCase):
    FACTOR_KEYS = (
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F21"
    )

    def setUp(self) -> None:
        factor_model_module._ic_weight_cache.clear()
        factor_model_module._ic_refresh_context.clear()

    def _frame(self, n: int = 120) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        close = pd.Series([100.0 + i * 0.1 for i in range(n)], index=idx)
        return pd.DataFrame(
            {
                "Close": close,
                "Returns": close.pct_change().fillna(0.0),
                "SMA_20": close.rolling(20).mean().bfill(),
                "BB_Std": pd.Series([1.0] * n, index=idx),
                "ATR_Percentile": pd.Series([50.0] * n, index=idx),
                "Keltner_Squeeze": pd.Series([0.0] * n, index=idx),
                "OFI": pd.Series([0.0] * n, index=idx),
            },
            index=idx,
        )

    def _score_fast_ic(self, model: AlphaFactorModel, df: pd.DataFrame, **kwargs) -> dict:
        with (
            patch(
                "src.alpha.factor_model._rolling_ic",
                side_effect=lambda factor, _fwd, _window: pd.Series(0.2, index=factor.index),
            ),
            patch("src.alpha.factor_model._bootstrap_ic_ci", return_value=(0.1, -0.1, 0.2)),
            patch("src.alpha.factor_model._turnover_penalty", return_value=(1.0, 0.9)),
            patch.object(model._meta_model, "adjust_weights", side_effect=lambda ics, **_x: dict(ics)),
        ):
            return model.score(df, ml_score=0.0, hurst=0.5, **kwargs)

    def test_score_uses_latest_usable_ic_not_trailing_zero(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        df = self._frame()
        ones = pd.Series([1.0] * len(df), index=df.index)

        with (
            patch.object(model, "_factor1_momentum", return_value=ones),
            patch.object(model, "_factor2_mean_reversion", return_value=ones),
            patch.object(model, "_factor3_volume", return_value=ones),
            patch.object(model, "_factor4_ml", return_value=ones),
            patch.object(model, "_factor5_volatility_squeeze", return_value=ones),
            patch.object(model, "_factor8_microstructure", return_value=ones),
            patch("src.alpha.factor_model._rolling_ic", return_value=pd.Series([0.6, 0.0], index=[0, 1])),
            patch("src.alpha.factor_model._bootstrap_ic_ci", return_value=(0.1, -0.2, 0.2)),
            patch("src.alpha.factor_model._turnover_penalty", return_value=(1.0, 0.9)),
        ):
            out = model.score(df, ml_score=0.0, hurst=0.5)

        self.assertEqual(out["signal"], "BUY")
        self.assertGreater(out["alpha_score"], 0.3)

    def test_f13_positive_funding_is_bearish(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        out = self._score_fast_ic(
            model,
            self._frame(180),
            asset="BTCUSDT_F13_POS",
            asset_class="crypto",
            funding_rate_z=2.0,
            oi_delta_1h=0.0,
            price_change_1h=0.0,
            etf_flow_z=0.0,
        )
        self.assertLess(float(out["factor_scores"]["F13"]), 0.0)

    def test_f13_negative_funding_is_bullish(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        out = self._score_fast_ic(
            model,
            self._frame(180),
            asset="BTCUSDT_F13_NEG",
            asset_class="crypto",
            funding_rate_z=-2.0,
            oi_delta_1h=0.0,
            price_change_1h=0.0,
            etf_flow_z=0.0,
        )
        self.assertGreater(float(out["factor_scores"]["F13"]), 0.0)

    def test_f14_oi_rising_with_price_up_is_bullish(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        out = self._score_fast_ic(
            model,
            self._frame(180),
            asset="BTCUSDT_F14_BULL",
            asset_class="crypto",
            funding_rate_z=0.0,
            oi_delta_1h=1.0,
            price_change_1h=0.5,
            etf_flow_z=0.0,
        )
        self.assertGreater(float(out["factor_scores"]["F14"]), 0.0)

    def test_f14_oi_rising_with_price_down_is_bearish(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        out = self._score_fast_ic(
            model,
            self._frame(180),
            asset="BTCUSDT_F14_BEAR",
            asset_class="crypto",
            funding_rate_z=0.0,
            oi_delta_1h=1.0,
            price_change_1h=-0.5,
            etf_flow_z=0.0,
        )
        self.assertLess(float(out["factor_scores"]["F14"]), 0.0)

    def test_f21_positive_etf_flow_is_bullish(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        out = self._score_fast_ic(
            model,
            self._frame(180),
            asset="BTCUSDT_F21_POS",
            asset_class="crypto",
            funding_rate_z=0.0,
            oi_delta_1h=0.0,
            price_change_1h=0.0,
            etf_flow_z=2.0,
        )
        self.assertGreater(float(out["factor_scores"]["F21"]), 0.0)

    def test_non_crypto_f13_f14_f21_are_zero(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        out = self._score_fast_ic(
            model,
            self._frame(180),
            asset="AAPL",
            asset_class="us_stock",
            funding_rate_z=2.0,
            oi_delta_1h=1.0,
            price_change_1h=0.5,
            etf_flow_z=2.0,
        )
        self.assertEqual(float(out["factor_scores"]["F13"]), 0.0)
        self.assertEqual(float(out["factor_scores"]["F14"]), 0.0)
        self.assertEqual(float(out["factor_scores"]["F21"]), 0.0)
        self.assertEqual(float(out["ic_weights"]["F13"]), 0.0)
        self.assertEqual(float(out["ic_weights"]["F14"]), 0.0)
        self.assertEqual(float(out["ic_weights"]["F21"]), 0.0)

    def test_f13_f14_f21_present_in_scores_and_weights(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        out = self._score_fast_ic(
            model,
            self._frame(180),
            asset="BTCUSDT_DICTS",
            asset_class="crypto",
            funding_rate_z=0.8,
            oi_delta_1h=0.7,
            price_change_1h=0.3,
            etf_flow_z=1.2,
        )
        for key in ("F13", "F14", "F21"):
            self.assertIn(key, out["factor_scores"])
            self.assertIn(key, out["ic_weights"])

    def test_alpha_score_changes_with_funding_input(self) -> None:
        model = AlphaFactorModel(alpha_threshold=0.3, ic_window=10)
        df = self._frame(180)
        common = {
            "asset": "BTCUSDT_ALPHA_DELTA",
            "asset_class": "crypto",
            "oi_delta_1h": 0.0,
            "price_change_1h": 0.0,
            "etf_flow_z": 0.0,
        }
        out_pos = self._score_fast_ic(model, df, funding_rate_z=2.0, **common)
        out_neg = self._score_fast_ic(model, df, funding_rate_z=-2.0, **common)
        self.assertNotEqual(float(out_pos["alpha_score"]), float(out_neg["alpha_score"]))


if __name__ == "__main__":
    unittest.main()
