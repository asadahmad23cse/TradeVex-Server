import tempfile
import time
import unittest
from pathlib import Path

import src.data.signal_history as signal_history


class SignalHistoryGuardsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_history_file = signal_history.HISTORY_FILE
        signal_history.HISTORY_FILE = Path(self._tmp.name) / "signal_history.json"
        signal_history.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        signal_history.HISTORY_FILE = self._orig_history_file
        self._tmp.cleanup()

    def test_records_blocked_signal_event(self) -> None:
        signal_history.record_signal(
            {
                "as_of_utc": "2026-03-29T00:00:00+00:00",
                "signal": "WAIT",
                "requested_signal": "LONG",
                "validated": False,
                "blocked_by": "regime_gate",
                "reason": "regime_conflict: LONG blocked in BEARISH regime",
                "regime": "BEARISH TREND",
            }
        )
        history = signal_history.get_history(10)
        self.assertEqual(len(history), 1)
        rec = history[0]
        self.assertEqual(rec["status"], "BLOCKED")
        self.assertEqual(rec["result"], "BLOCKED")
        self.assertEqual(rec["signal"], "LONG")
        self.assertEqual(rec["blocked_by"], "regime_gate")

    def test_sl_pnl_uses_sl_distance_and_caps_loss(self) -> None:
        signal_history._save(
            [
                {
                    "id": 1,
                    "time": "2026-03-29T00:00:00+00:00",
                    "signal": "LONG",
                    "confidence": 90.0,
                    "alpha_score": 80,
                    "entry": 66929.84,
                    "stop_loss": 66699.58,
                    "tp1": 67206.15,
                    "tp2": 67390.35,
                    "tp3": 67620.61,
                    "risk_reward": 2.0,
                    "regime": "BULLISH TREND",
                    "reason": "",
                    "funding_rate": 0.0,
                    "status": "OPEN",
                    "result": "OPEN",
                    "exit_price": None,
                    "pnl_pct": None,
                    "closed_time": None,
                    "open_timestamp": time.time(),
                },
                {
                    "id": 2,
                    "time": "2026-03-29T00:05:00+00:00",
                    "signal": "LONG",
                    "confidence": 90.0,
                    "alpha_score": 80,
                    "entry": 100.0,
                    "stop_loss": 80.0,
                    "tp1": 130.0,
                    "tp2": 140.0,
                    "tp3": 150.0,
                    "risk_reward": 2.0,
                    "regime": "BULLISH TREND",
                    "reason": "",
                    "funding_rate": 0.0,
                    "status": "OPEN",
                    "result": "OPEN",
                    "exit_price": None,
                    "pnl_pct": None,
                    "closed_time": None,
                    "open_timestamp": time.time(),
                },
            ]
        )

        signal_history.check_open_signals(current_price=75.0)
        history = signal_history.get_history(10)
        rec_one = history[0]
        rec_two = history[1]

        self.assertEqual(rec_one["status"], "SL_HIT")
        self.assertEqual(rec_one["result"], "SL")
        self.assertAlmostEqual(float(rec_one["pnl_pct"]), -0.34, places=2)

        self.assertEqual(rec_two["status"], "SL_HIT")
        self.assertEqual(rec_two["result"], "SL")
        self.assertEqual(float(rec_two["pnl_pct"]), -2.0)

    def test_tp_pnl_uses_rr_formula(self) -> None:
        signal_history._save(
            [
                {
                    "id": 1,
                    "time": "2026-03-29T00:00:00+00:00",
                    "signal": "LONG",
                    "confidence": 88.0,
                    "alpha_score": 70,
                    "entry": 100.0,
                    "stop_loss": 99.66,  # 0.34% risk
                    "tp1": 101.0,
                    "tp2": 101.5,
                    "tp3": 102.0,
                    "risk_reward": 2.0,
                    "regime": "BULLISH TREND",
                    "reason": "",
                    "funding_rate": 0.0,
                    "status": "OPEN",
                    "result": "OPEN",
                    "exit_price": None,
                    "pnl_pct": None,
                    "closed_time": None,
                    "open_timestamp": time.time(),
                }
            ]
        )

        signal_history.check_open_signals(current_price=101.0)
        rec = signal_history.get_history(10)[0]
        self.assertEqual(rec["status"], "TP1_HIT")
        self.assertEqual(rec["result"], "TP1")
        self.assertAlmostEqual(float(rec["pnl_pct"]), 0.68, places=2)


if __name__ == "__main__":
    unittest.main()
