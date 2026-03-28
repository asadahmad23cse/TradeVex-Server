import unittest
from unittest.mock import patch

import pandas as pd

from src.alpha.factor_model import AlphaFactorModel


class AlphaFactorModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
