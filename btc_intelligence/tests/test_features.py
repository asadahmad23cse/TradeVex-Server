import unittest

import numpy as np
import pandas as pd

from btc_intelligence.features.derivatives import compute_derivatives
from btc_intelligence.features.feature_vector import FEATURE_COLUMNS
from btc_intelligence.features.macro import compute_macro
from btc_intelligence.features.onchain import compute_onchain
from btc_intelligence.features.order_flow import compute_order_flow
from btc_intelligence.features.price_action import compute_price_action
from btc_intelligence.features.smc import compute_smc
from btc_intelligence.features.volatility import compute_volatility


def make_df(n: int, freq: str, start: float = 80000.0, drift: float = 20.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    base = start + np.arange(n) * drift + np.sin(np.arange(n) / 6.0) * 40
    return pd.DataFrame(
        {
            "Open": base - 10,
            "High": base + 30,
            "Low": base - 35,
            "Close": base,
            "Volume": np.linspace(100, 300, n),
        },
        index=idx,
    )


class FeatureTests(unittest.TestCase):
    def test_price_action(self) -> None:
        f = compute_price_action(make_df(220, "15min"), make_df(140, "1h"), make_df(80, "4h"))
        self.assertGreater(f.tf_15m.price, 0)
        self.assertIn(f.tf_15m.trend_bias, {"bullish", "bearish", "mixed"})

    def test_smc(self) -> None:
        smc = compute_smc(make_df(180, "15min"), trend_bias="bullish")
        self.assertIn(smc.bos_type, {"none", "bullish_continuation", "bearish_continuation"})

    def test_order_flow_enhanced(self) -> None:
        trades = []
        for i in range(500):
            price = 84000 + i * 0.2
            maker = i % 3 == 0
            qty = 0.02 + (i % 10) * 0.001
            trades.append({"price": price, "qty": qty, "maker": maker, "notional": price * qty, "time": i * 1000})

        depth = {
            "bids": [[84000 - i, 1.0 + i * 0.1] for i in range(10)],
            "asks": [[84001 + i, 1.2 + i * 0.1] for i in range(10)],
        }
        multi_exchange = {
            "bybit": {"cvd": [1, 2, 3, 4], "trades": [], "depth": {"bids": [], "asks": []}},
            "okx": {"cvd": [1, 2, 3, 5], "trades": [], "depth": {"bids": [], "asks": []}},
        }

        of = compute_order_flow(trades, depth, multi_exchange, make_df(220, "15min"))
        self.assertTrue(0 <= of.obi <= 1)
        self.assertIn(of.stacked_imbalance_direction, {"none", "bullish_3_levels", "bearish_3_levels"})
        self.assertIn(of.iceberg_direction, {"none", "bullish", "bearish"})

    def test_volatility(self) -> None:
        vol = compute_volatility(make_df(220, "15min"), make_df(130, "1h"), {"atm_iv": 0.6})
        self.assertGreaterEqual(vol.atr14, 0)
        self.assertIn(vol.vol_regime, {"compression", "normal", "expansion", "spike"})
        self.assertGreater(vol.iv_rv_ratio, 0)

    def test_derivatives(self) -> None:
        der = compute_derivatives(
            rest_data={"funding_rate": 0.0006, "open_interest": 1000},
            oi_hist=[
                {"open_interest": 900, "ts": 0},
                {"open_interest": 950, "ts": 1800},
                {"open_interest": 1000, "ts": 3700},
            ],
            funding_hist=[{"funding_rate": 0.0002, "ts": 0}, {"funding_rate": 0.0004, "ts": 8 * 3600}, {"funding_rate": 0.0006, "ts": 16 * 3600}],
            coinglass_data={"liquidation_heatmap_above": 85000, "liquidation_heatmap_below": 83000, "long_short_ratio": 0.62},
            deribit_data={"max_pain_level": 84000, "iv_skew": -0.02, "put_call_ratio": 0.9, "options_expiry_hours": 20},
            price_now=84000,
            price_1h_ago=83500,
        )
        self.assertIn(der.funding_sentiment, {"overleveraged_long", "overleveraged_short", "neutral"})
        self.assertIsInstance(der.max_pain_magnet, bool)

    def test_onchain_macro(self) -> None:
        on = compute_onchain(
            {"exchange_netflow": -250, "sopr": 0.98, "lth_supply_change": 1.2, "whale_wallet_count": 12},
            {"whale_exchange_deposits_1h": 2, "whale_exchange_withdrawals_1h": 5, "whale_net_flow": 250},
        )
        self.assertIn(on.exchange_netflow_bias, {"bullish", "bearish", "neutral"})

        macro = compute_macro({"fear_greed_score": 35, "dxy_bias": "risk_off", "vix_level": 20, "vix_regime": "NORMAL"}, [])
        self.assertIn(macro.session, {"asian", "london_open", "ny_london_overlap", "new_york", "dead"})

    def test_feature_column_count(self) -> None:
        self.assertGreaterEqual(len(FEATURE_COLUMNS), 51)


if __name__ == "__main__":
    unittest.main()
