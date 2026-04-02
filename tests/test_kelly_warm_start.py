import tempfile
import unittest
from pathlib import Path

from src.risk.kelly_warm_start import BTCKellyWarmStart


def _trade(
    signal: str = "LONG",
    confidence: float = 72.0,
    regime: str = "BULLISH TREND",
    pnl_pct: float = 1.0,
    risk_reward: float = 2.0,
    status: str = "TP1_HIT",
) -> dict:
    return {
        "signal": signal,
        "confidence": confidence,
        "regime": regime,
        "pnl_pct": pnl_pct,
        "risk_reward": risk_reward,
        "status": status,
    }


class BTCKellyWarmStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.history_path = self.tmpdir / "signal_history.json"
        self.buckets_path = self.tmpdir / "kelly_buckets.json"
        self.history_path.write_text("[]", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _kelly(self) -> BTCKellyWarmStart:
        return BTCKellyWarmStart(
            buckets_file=str(self.buckets_path),
            history_path=str(self.history_path),
        )

    def test_empty_history_uses_bayesian_prior(self) -> None:
        kelly = self._kelly()
        result = kelly.compute_btc_position("LONG", 72, "BULLISH TREND")
        self.assertEqual(result["method"], "bayesian_prior")

    def test_15_trades_uses_bayesian_blend_with_half_weight(self) -> None:
        kelly = self._kelly()
        for idx in range(15):
            pnl = 1.2 if idx % 2 == 0 else -0.8
            kelly.update_bucket(_trade(pnl_pct=pnl))

        result = kelly.compute_btc_position("LONG", 72, "BULLISH TREND")
        self.assertEqual(result["method"], "bayesian_blend")
        self.assertAlmostEqual(float(result["smoothing_weight"]), 0.5, places=3)

    def test_30_plus_trades_uses_empirical_full_weight(self) -> None:
        kelly = self._kelly()
        for _ in range(31):
            kelly.update_bucket(_trade(pnl_pct=1.0))

        result = kelly.compute_btc_position("LONG", 72, "BULLISH TREND")
        self.assertEqual(result["method"], "empirical")
        self.assertAlmostEqual(float(result["smoothing_weight"]), 1.0, places=3)

    def test_high_confidence_gives_larger_size(self) -> None:
        kelly = self._kelly()
        high = kelly.compute_btc_position("LONG", 85, "BULLISH TREND")
        low = kelly.compute_btc_position("LONG", 60, "BULLISH TREND")
        self.assertGreater(float(high["position_size_pct"]), float(low["position_size_pct"]))

    def test_sideways_regime_smaller_than_bull(self) -> None:
        kelly = self._kelly()
        bull = kelly.compute_btc_position("LONG", 85, "BULLISH TREND")
        sideways = kelly.compute_btc_position("LONG", 85, "SIDEWAYS")
        self.assertLess(float(sideways["position_size_pct"]), float(bull["position_size_pct"]))

    def test_position_size_never_above_five_percent(self) -> None:
        kelly = self._kelly()
        for _ in range(40):
            kelly.update_bucket(_trade(confidence=85, pnl_pct=2.0, risk_reward=5.0))

        result = kelly.compute_btc_position("LONG", 95, "BULLISH TREND")
        self.assertLessEqual(float(result["position_size_pct"]), 5.0)

    def test_position_size_never_below_half_percent(self) -> None:
        kelly = self._kelly()
        for _ in range(40):
            kelly.update_bucket(_trade(confidence=85, pnl_pct=-1.0, risk_reward=1.0, status="SL_HIT"))

        result = kelly.compute_btc_position("LONG", 85, "BULLISH TREND")
        self.assertGreaterEqual(float(result["position_size_pct"]), 0.5)

    def test_update_bucket_updates_win_rate(self) -> None:
        kelly = self._kelly()
        kelly.update_bucket(_trade(pnl_pct=1.0))
        kelly.update_bucket(_trade(pnl_pct=-1.0, status="SL_HIT"))
        stats = kelly.buckets["LONG/MODERATE/BULL"]
        self.assertEqual(int(stats["total"]), 2)
        self.assertAlmostEqual(float(stats["win_rate"]), 0.5, places=4)

    def test_save_load_roundtrip_preserves_bucket_values(self) -> None:
        kelly = self._kelly()
        kelly.update_bucket(_trade(pnl_pct=1.0))
        kelly.update_bucket(_trade(pnl_pct=2.0))
        kelly.update_bucket(_trade(pnl_pct=-1.0, status="SL_HIT"))
        kelly.save_buckets()

        loaded = self._kelly()
        original = kelly.buckets["LONG/MODERATE/BULL"]
        restored = loaded.buckets["LONG/MODERATE/BULL"]

        self.assertEqual(int(restored["total"]), int(original["total"]))
        self.assertEqual(int(restored["wins"]), int(original["wins"]))
        self.assertEqual(int(restored["losses"]), int(original["losses"]))
        self.assertAlmostEqual(float(restored["win_rate"]), float(original["win_rate"]), places=6)
        self.assertAlmostEqual(float(restored["avg_rr"]), float(original["avg_rr"]), places=6)


if __name__ == "__main__":
    unittest.main()
