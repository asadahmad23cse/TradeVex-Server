import unittest

from src.signals.engine import SignalEngine


class SignalEngineCostGateTests(unittest.TestCase):
    def test_blocks_trade_when_cost_exceeds_alpha(self) -> None:
        engine = SignalEngine()
        sig = engine.generate(
            asset="AAPL",
            asset_class="us_stock",
            timeframe="intraday",
            alpha_result={
                "signal": "BUY",
                "strength": "WEAK",
                "confidence": 55.0,
                "alpha_score": 0.0005,
                "factor_scores": {"F1": 0.1},
                "ic_weights": {"F1": 0.01},
            },
            regime="SIDEWAYS",
            regime_allows=True,
            hurst=0.5,
            entry_price=100.0,
            atr_14=1.2,
            kelly_fraction=1.0,
            position_size_pct=5.0,
            daily_vol=0.35,
            volume_ratio=0.3,
        )
        self.assertIsNone(sig)

    def test_keeps_trade_when_net_alpha_positive(self) -> None:
        engine = SignalEngine()
        sig = engine.generate(
            asset="AAPL",
            asset_class="us_stock",
            timeframe="intraday",
            alpha_result={
                "signal": "BUY",
                "strength": "STRONG",
                "confidence": 80.0,
                "alpha_score": 0.6,
                "factor_scores": {"F1": 0.9},
                "ic_weights": {"F1": 0.2},
            },
            regime="BULL",
            regime_allows=True,
            hurst=0.6,
            entry_price=100.0,
            atr_14=1.0,
            kelly_fraction=8.0,
            position_size_pct=2.0,
            daily_vol=0.2,
            volume_ratio=1.2,
        )
        assert sig is not None
        self.assertEqual(sig.signal, "BUY")
        self.assertGreater(sig.net_alpha_score, 0.0)
        self.assertGreaterEqual(sig.cost_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
