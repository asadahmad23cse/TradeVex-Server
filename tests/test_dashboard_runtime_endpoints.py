import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.dashboard import api
import src.data.signal_history as signal_history
from src.paper_trading.paper_engine import PaperTradingEngine
from src.risk.kelly_warm_start import BTCKellyWarmStart


class _RunnerStub:
    def get_live_validation_report(self) -> dict:
        return {"graduation": {"stage_name": "Seed Capital"}, "performance": {"days": 12}}

    def get_latency_report(self) -> dict:
        return {"cycle_id": "intraday_20260327_120000", "total_ms": 123.4, "n_assets": 10}


class DashboardRuntimeEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        api._store = object()  # type: ignore[attr-defined]
        api.set_live_runner(_RunnerStub())

    def tearDown(self) -> None:
        api.set_live_runner(None)

    def test_live_validation_uses_runner_snapshot(self) -> None:
        payload = api.live_validation()
        self.assertEqual(payload["graduation"]["stage_name"], "Seed Capital")
        self.assertEqual(payload["performance"]["days"], 12)

    def test_latency_uses_runner_snapshot(self) -> None:
        payload = api.latency_report()
        self.assertEqual(payload["cycle_id"], "intraday_20260327_120000")
        self.assertEqual(payload["n_assets"], 10)

    def test_signal_history_backfills_sqlite_rows_when_json_is_short(self) -> None:
        orig_history_file = signal_history.HISTORY_FILE
        orig_dashboard_cfg = api._dashboard_cfg
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            signal_history.HISTORY_FILE = tmp_path / "signal_history.json"
            signal_history.HISTORY_FILE.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "ticker": "BTCUSDT",
                            "time": "2026-04-29T07:51:59+00:00",
                            "signal": "SHORT",
                            "status": "BLOCKED",
                            "result": "BLOCKED",
                            "confidence": 16.9,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            db_path = tmp_path / "signals.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE signals (
                        signal_id TEXT,
                        ticker TEXT,
                        timestamp TEXT,
                        signal TEXT,
                        confidence REAL,
                        entry_price REAL,
                        sl REAL,
                        tp1 REAL,
                        outcome TEXT,
                        pnl_pct REAL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO signals
                    (signal_id, ticker, timestamp, signal, confidence, entry_price, sl, tp1, outcome, pnl_pct)
                    VALUES ('BTC-070', 'BTCUSDT', '2026-04-07T08:21:01Z', 'LONG', 56.4, 68779.5, 68559.69, 68999.31, 'TP1', 0.325)
                    """
                )
                conn.commit()
            finally:
                conn.close()

            api._dashboard_cfg = {"database": {"url": str(db_path)}}
            try:
                rows = api._combined_signal_history_rows(limit=10, ticker="BTCUSDT")
                stats = api._signal_history_stats_from_rows(rows)
            finally:
                signal_history.HISTORY_FILE = orig_history_file
                api._dashboard_cfg = orig_dashboard_cfg

        outcomes = {str(row.get("outcome")).upper() for row in rows}
        self.assertIn("BLOCKED", outcomes)
        self.assertIn("TP1", outcomes)
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["blocked_total"], 1)

    def test_signal_history_includes_symbol_scoped_eth_paper_trades(self) -> None:
        orig_history_file = signal_history.HISTORY_FILE
        orig_dashboard_cfg = api._dashboard_cfg
        orig_kelly_file = BTCKellyWarmStart.BUCKETS_FILE
        with tempfile.TemporaryDirectory() as tmp:
            try:
                tmp_path = Path(tmp)
                signal_history.HISTORY_FILE = tmp_path / "signal_history.json"
                signal_history.HISTORY_FILE.write_text("[]", encoding="utf-8")
                BTCKellyWarmStart.BUCKETS_FILE = str(tmp_path / "kelly_buckets.json")
                api._dashboard_cfg = {"database": {"url": str(tmp_path / "missing.db")}}
                engine = PaperTradingEngine(initial_capital=100000.0, data_file=tmp_path / "paper.json")

                eth_open = engine.execute_trade(
                    {
                        "ticker": "ETHUSDT",
                        "signal": "LONG",
                        "entry_price": 2300.0,
                        "stop_loss": 2280.0,
                        "take_profit": 2350.0,
                        "confidence": 72.0,
                    },
                    mode="auto",
                )
                self.assertTrue(eth_open["success"])
                self.assertIsNotNone(engine.close_position("ETHUSDT", 2340.0, "tp_hit"))

                btc_open = engine.execute_trade(
                    {
                        "ticker": "BTCUSDT",
                        "signal": "SHORT",
                        "entry_price": 76000.0,
                        "stop_loss": 76500.0,
                        "take_profit": 75000.0,
                        "confidence": 70.0,
                    },
                    mode="auto",
                )
                self.assertTrue(btc_open["success"])
                self.assertIsNotNone(engine.close_position("BTCUSDT", 75500.0, "manual"))

                rows = api._combined_signal_history_rows(limit=20, ticker="ETHUSDT", paper_engine=engine)
                stats = api._signal_history_stats_from_rows(rows)
            finally:
                signal_history.HISTORY_FILE = orig_history_file
                api._dashboard_cfg = orig_dashboard_cfg
                BTCKellyWarmStart.BUCKETS_FILE = orig_kelly_file

        self.assertEqual({row["ticker"] for row in rows}, {"ETHUSDT"})
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["wins"], 1)


if __name__ == "__main__":
    unittest.main()
