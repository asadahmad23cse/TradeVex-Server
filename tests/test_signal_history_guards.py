import tempfile
import time
import unittest
from pathlib import Path

import src.data.signal_history as signal_history
from src.risk.kelly_warm_start import BTCKellyWarmStart


class SignalHistoryGuardsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_history_file = signal_history.HISTORY_FILE
        self._orig_kelly_file = BTCKellyWarmStart.BUCKETS_FILE
        signal_history.HISTORY_FILE = Path(self._tmp.name) / "signal_history.json"
        signal_history.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        BTCKellyWarmStart.BUCKETS_FILE = str(Path(self._tmp.name) / "kelly_buckets.json")

    def tearDown(self) -> None:
        signal_history.HISTORY_FILE = self._orig_history_file
        BTCKellyWarmStart.BUCKETS_FILE = self._orig_kelly_file
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

    def test_record_closed_trade_adds_executed_signal(self) -> None:
        signal_history.record_closed_trade(
            {
                "trade_id": "PT-001",
                "ticker": "BTCUSDT",
                "signal": "SHORT",
                "entry_price": 75000.0,
                "exit_price": 74850.0,
                "stop_loss": 75200.0,
                "tp1": 74800.0,
                "risk_reward": 1.0,
                "pnl_pct": 0.2,
                "reason": "tp_hit",
                "opened_at": "2026-04-20T08:00:00+00:00",
                "closed_at": "2026-04-20T08:15:00+00:00",
                "source": "paper_trading_auto",
                "mode": "ALGO",
                "confidence": 71.5,
            }
        )

        history = signal_history.get_history(5)
        self.assertEqual(len(history), 1)
        rec = history[0]
        self.assertEqual(rec["status"], "CLOSED")
        self.assertEqual(rec["result"], "TP1")
        self.assertEqual(rec["trade_id"], "PT-001")
        self.assertEqual(rec["ticker"], "BTCUSDT")
        self.assertEqual(rec["mode"], "ALGO")
        self.assertAlmostEqual(float(rec["pnl_pct"]), 0.2, places=2)

        stats = signal_history.get_stats()
        self.assertEqual(int(stats["total"]), 1)

        kelly = BTCKellyWarmStart(
            buckets_file=BTCKellyWarmStart.BUCKETS_FILE,
            history_path=str(signal_history.HISTORY_FILE),
        )
        self.assertEqual(kelly.total_trades, 1)
        bucket = kelly.buckets.get("SHORT/MODERATE/SIDEWAYS")
        self.assertIsNotNone(bucket)
        self.assertEqual(bucket["wins"], 1)

        signal_history.record_closed_trade(
            {
                "trade_id": "PT-001",
                "ticker": "BTCUSDT",
                "signal": "SHORT",
                "entry_price": 75000.0,
                "exit_price": 74850.0,
                "stop_loss": 75200.0,
                "tp1": 74800.0,
                "risk_reward": 1.0,
                "pnl_pct": 0.2,
                "reason": "tp_hit",
                "opened_at": "2026-04-20T08:00:00+00:00",
                "closed_at": "2026-04-20T08:15:00+00:00",
                "source": "paper_trading_auto",
                "mode": "ALGO",
                "confidence": 71.5,
            }
        )
        kelly_after_duplicate = BTCKellyWarmStart(
            buckets_file=BTCKellyWarmStart.BUCKETS_FILE,
            history_path=str(signal_history.HISTORY_FILE),
        )
        self.assertEqual(kelly_after_duplicate.total_trades, 1)

    def test_record_signal_expires_stale_open_before_duplicate_guard(self) -> None:
        stale_ts = time.time() - (13 * 3600)
        signal_history._save(
            [
                {
                    "id": 1,
                    "time": "2026-03-29T00:00:00+00:00",
                    "signal": "LONG",
                    "confidence": 80.0,
                    "alpha_score": 70,
                    "entry": 100.0,
                    "stop_loss": 99.0,
                    "tp1": 102.0,
                    "risk_reward": 2.0,
                    "status": "OPEN",
                    "result": "OPEN",
                    "open_timestamp": stale_ts,
                    "closed_time": None,
                }
            ]
        )

        signal_history.record_signal(
            {
                "as_of_utc": "2026-03-29T13:30:00+00:00",
                "ticker": "BTCUSDT",
                "signal": "LONG",
                "validated": True,
                "confidence": 82.0,
                "alpha_score": 75,
                "entry_price": 101.0,
                "stop_loss": 100.0,
                "tp1": 103.0,
                "risk_reward": 2.0,
            }
        )

        history = signal_history.get_history(10)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["status"], "EXPIRED")
        self.assertEqual(history[0]["result"], "EXPIRED")
        self.assertEqual(history[1]["status"], "OPEN")


if __name__ == "__main__":
    unittest.main()
