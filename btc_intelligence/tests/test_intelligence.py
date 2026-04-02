from __future__ import annotations

import unittest
from types import SimpleNamespace

from btc_intelligence.signals.execution import evaluate_execution, recommended_max_position_btc
from btc_intelligence.signals.intelligence import (
    execution_rejection_code,
    order_flow_decision_state,
    trade_based_volume_profile,
    volatility_tradeability,
)


def _mk_trade(ts_ms: int, price: float, qty: float) -> dict[str, float | int]:
    return {"time": ts_ms, "price": price, "qty": qty}


class TestIntelligence(unittest.TestCase):
    def test_order_flow_favor_long_state(self) -> None:
        state = SimpleNamespace(
            obi=0.62,
            cvd_slope=4.2,
            absorption_side="sell_absorption",
            absorption_strength=2.0,
        )
        out = order_flow_decision_state(state)
        self.assertEqual(out["decision_state"], "FAVOR_LONG")

    def test_order_flow_mixed_is_no_trade(self) -> None:
        state = SimpleNamespace(
            obi=0.51,
            cvd_slope=0.0,
            absorption_side="none",
            absorption_strength=0.0,
        )
        out = order_flow_decision_state(state)
        self.assertEqual(out["decision_state"], "NO_TRADE")

    def test_trade_volume_profile_insufficient_window(self) -> None:
        trades = [_mk_trade(1_000_000 + i * 1_000, 50_000.0, 0.01) for i in range(10)]
        out = trade_based_volume_profile(trades, window_minutes=45, n_bins=24)
        self.assertEqual(out["decision_state"], "NO_TRADE")
        self.assertEqual(out["reason"], "Insufficient trade window")

    def test_trade_volume_profile_respects_window(self) -> None:
        base = 2_000_000
        old = [_mk_trade(base + i * 1_000, 50_000.0, 0.01) for i in range(200)]
        latest = [_mk_trade(base + 3_600_000 + i * 1_000, 50_100.0, 0.02) for i in range(100)]
        out = trade_based_volume_profile(old + latest, window_minutes=30, n_bins=20)
        self.assertEqual(out["trade_count"], 100)
        self.assertEqual(out["window_minutes"], 30)
        self.assertGreater(out["poc"], 0)

    def test_volatility_mapping(self) -> None:
        low = volatility_tradeability(SimpleNamespace(vol_regime="compression"))
        normal = volatility_tradeability(SimpleNamespace(vol_regime="normal"))
        expansion = volatility_tradeability(SimpleNamespace(vol_regime="expansion"))
        self.assertEqual(low["tradeability"], "NO_TRADE")
        self.assertEqual(normal["tradeability"], "ALLOW")
        self.assertEqual(expansion["tradeability"], "CAUTION")

    def test_recommended_position_from_depth(self) -> None:
        deep_book = {"bids": [[50_000, 5.0]], "asks": [[50_010, 5.0], [50_020, 5.0]]}
        thin_book = {"bids": [[50_000, 0.1]], "asks": [[50_010, 0.1], [50_200, 0.2]]}

        deep = recommended_max_position_btc(deep_book, side="buy", slippage_limit_pct=0.03, max_qty_btc=5.0)
        thin = recommended_max_position_btc(thin_book, side="buy", slippage_limit_pct=0.03, max_qty_btc=5.0)
        self.assertGreaterEqual(deep, thin)
        self.assertGreater(deep, 0)

    def test_execution_rejects_when_volatility_no_trade(self) -> None:
        depth = {
            "bids": [[50_000, 2.0], [49_990, 2.0]],
            "asks": [[50_010, 2.0], [50_020, 2.0]],
        }
        trades = [_mk_trade(1_000_000 + i * 1_000, 50_000.0 + i * 0.1, 0.02) for i in range(60)]
        order_flow = SimpleNamespace(single_bar_delta_divergence=False, obi=0.7)

        out = evaluate_execution(
            depth=depth,
            agg_trades=trades,
            direction="LONG",
            order_flow=order_flow,
            market_state={"volatility_tradeability": {"tradeability": "NO_TRADE"}},
            fallback_tradeability="ALLOW",
        )
        self.assertFalse(out.accepted)
        self.assertEqual(out.reason, "Volatility regime too low for execution")
        self.assertEqual(execution_rejection_code(out.reason), "VOL_NO_TRADE")


if __name__ == "__main__":
    unittest.main()
