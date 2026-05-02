import tempfile
import unittest
from pathlib import Path

import src.data.signal_history as signal_history
from src.paper_trading.paper_engine import PaperTradingEngine
from src.risk.kelly_warm_start import BTCKellyWarmStart


class PaperEngineSignalHistorySyncTests(unittest.TestCase):
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

    def test_auto_close_syncs_into_signal_history(self) -> None:
        engine = PaperTradingEngine(
            initial_capital=100000.0,
            data_file=Path(self._tmp.name) / "paper_state.json",
        )

        opened = engine.execute_trade(
            {
                "ticker": "BTCUSDT",
                "signal": "SHORT",
                "entry_price": 75000.0,
                "stop_loss": 75200.0,
                "take_profit": 74800.0,
                "confidence": 77.0,
                "regime": "BEARISH TREND",
            },
            mode="auto",
        )
        self.assertTrue(opened.get("success"), opened)

        closed = engine.close_position("BTCUSDT", 74800.0, "tp_hit")
        self.assertIsNotNone(closed)
        self.assertEqual(str(closed.get("mode")).lower(), "auto")

        history = signal_history.get_history(10)
        self.assertEqual(len(history), 1)
        rec = history[0]
        self.assertEqual(rec["status"], "CLOSED")
        self.assertEqual(rec["result"], "TP1")
        self.assertEqual(rec["trade_id"], closed["trade_id"])
        self.assertEqual(rec["ticker"], "BTCUSDT")
        self.assertEqual(rec["mode"], "ALGO")
        self.assertTrue(rec.get("kelly_bucket_updated"))

        kelly = BTCKellyWarmStart(
            buckets_file=BTCKellyWarmStart.BUCKETS_FILE,
            history_path=str(signal_history.HISTORY_FILE),
        )
        self.assertEqual(kelly.total_trades, 1)
        self.assertIn("SHORT/MODERATE/BEAR", kelly.buckets)


if __name__ == "__main__":
    unittest.main()
