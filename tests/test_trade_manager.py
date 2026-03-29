import time
import unittest

from src.dashboard.trade_manager import TradeManager


def _base_long_record() -> dict:
    now = time.time()
    return {
        "id": 1,
        "status": "OPEN",
        "signal": "LONG",
        "entry": 100.0,
        "stop_loss": 99.0,
        "active_stop_loss": 99.0,
        "tp1": 102.0,
        "tp2": 104.0,
        "tp3": 108.0,
        "risk_reward": 2.0,
        "open_timestamp": now,
        "tp1_hit": False,
        "tp2_hit": False,
        "tp3_hit": False,
        "breakeven_activated": False,
        "trailing_active": False,
        "milestones": [],
    }


def _base_short_record() -> dict:
    now = time.time()
    return {
        "id": 2,
        "status": "OPEN",
        "signal": "SHORT",
        "entry": 100.0,
        "stop_loss": 101.0,
        "active_stop_loss": 101.0,
        "tp1": 98.0,
        "tp2": 96.0,
        "tp3": 92.0,
        "risk_reward": 2.0,
        "open_timestamp": now,
        "tp1_hit": False,
        "tp2_hit": False,
        "tp3_hit": False,
        "breakeven_activated": False,
        "trailing_active": False,
        "milestones": [],
    }


class TradeManagerTests(unittest.TestCase):
    def test_long_hits_tp1_sets_breakeven(self) -> None:
        tm = TradeManager()
        rec = _base_long_record()
        out = tm.manage(rec, current_price=102.5, current_alpha=10.0, current_signal="LONG")
        self.assertTrue(out["tp1_hit"])
        self.assertTrue(out["breakeven_activated"])
        self.assertEqual(out.get("milestone"), "BREAKEVEN_SET")
        self.assertAlmostEqual(float(out["active_stop_loss"]), 100.05, places=2)

    def test_long_hits_tp2_trailing_activates_and_only_moves_up(self) -> None:
        tm = TradeManager()
        rec = _base_long_record()
        rec["atr"] = 2.0
        out1 = tm.manage(rec, current_price=110.0, current_alpha=10.0, current_signal="LONG")
        first_stop = float(out1["active_stop_loss"])
        self.assertTrue(out1["trailing_active"])

        out2 = tm.manage(out1, current_price=109.0, current_alpha=10.0, current_signal="LONG")
        second_stop = float(out2["active_stop_loss"])
        self.assertGreaterEqual(second_stop, first_stop)

    def test_short_hits_tp2_trailing_only_moves_down(self) -> None:
        tm = TradeManager()
        rec = _base_short_record()
        rec["atr"] = 1.0
        out1 = tm.manage(rec, current_price=95.0, current_alpha=10.0, current_signal="SHORT")
        first_stop = float(out1["active_stop_loss"])
        self.assertTrue(out1["trailing_active"])

        out2 = tm.manage(out1, current_price=96.0, current_alpha=10.0, current_signal="SHORT")
        second_stop = float(out2["active_stop_loss"])
        self.assertLessEqual(second_stop, first_stop)

    def test_alpha_flip_one_opposite_reading_no_exit(self) -> None:
        tm = TradeManager()
        rec = _base_long_record()
        out = tm.manage(rec, current_price=101.0, current_alpha=80.0, current_signal="SHORT")
        self.assertEqual(out["status"], "OPEN")

    def test_alpha_flip_two_consecutive_readings_exit(self) -> None:
        tm = TradeManager()
        rec = _base_long_record()
        out1 = tm.manage(rec, current_price=101.0, current_alpha=80.0, current_signal="SHORT")
        self.assertEqual(out1["status"], "OPEN")
        out2 = tm.manage(out1, current_price=101.5, current_alpha=80.0, current_signal="SHORT")
        self.assertEqual(out2["status"], "ALPHA_FLIP_EXIT")
        self.assertIn("alpha flipped to SHORT", str(out2.get("exit_reason", "")))

    def test_expiry_after_12h_no_tp(self) -> None:
        tm = TradeManager()
        rec = _base_long_record()
        rec["open_timestamp"] = time.time() - (13 * 3600)
        out = tm.manage(rec, current_price=100.2, current_alpha=10.0, current_signal="LONG")
        self.assertEqual(out["status"], "EXPIRED")
        self.assertIn("12h", str(out.get("exit_reason", "")))

    def test_breakeven_hit_pnl_zero(self) -> None:
        tm = TradeManager()
        rec = _base_long_record()
        out1 = tm.manage(rec, current_price=102.2, current_alpha=10.0, current_signal="LONG")
        out2 = tm.manage(out1, current_price=100.0, current_alpha=10.0, current_signal="LONG")
        self.assertEqual(out2["status"], "BREAKEVEN_HIT")
        self.assertEqual(float(out2["pnl_pct"]), 0.0)

    def test_trail_stop_hit_positive_pnl(self) -> None:
        tm = TradeManager()
        rec = _base_long_record()
        rec["tp3"] = 120.0
        rec["atr"] = 2.0
        out1 = tm.manage(rec, current_price=110.0, current_alpha=10.0, current_signal="LONG")
        self.assertTrue(out1["trailing_active"])
        out2 = tm.manage(out1, current_price=106.0, current_alpha=10.0, current_signal="LONG")
        self.assertEqual(out2["status"], "TRAIL_STOP_HIT")
        self.assertGreater(float(out2["pnl_pct"]), 0.0)


if __name__ == "__main__":
    unittest.main()
