import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.api.data_quality import DataAnomalyDetector
from src.execution.reconciliation import FillReconciler
from src.scheduler.watchdog import SchedulerHealthWatchdog
from src.signals.store import SignalStore


class OpsControlsTests(unittest.TestCase):
    def test_data_anomaly_detector_flags_stale_zero_volume_and_nan(self) -> None:
        idx = pd.date_range("2025-01-01", periods=6, freq="5min")
        df = pd.DataFrame(
            {
                "Open": [100, 100, 100, 100, None, 120],
                "High": [101, 101, 101, 101, None, 121],
                "Low": [99, 99, 99, 99, None, 119],
                "Close": [100, 100, 100, 100, None, 120],
                "Volume": [10, 10, 0, 0, 0, 20],
            },
            index=idx,
        )
        cleaned, report = DataAnomalyDetector().inspect_and_clean(df, "TEST", "us_stock", "intraday")
        self.assertFalse(cleaned.empty)
        self.assertIn("stale_price", report.issue_types)
        self.assertIn("zero_volume", report.issue_types)
        self.assertIn("nan_fill", report.issue_types)
        self.assertTrue(report.severe)

    def test_watchdog_marks_stale_jobs(self) -> None:
        wd = SchedulerHealthWatchdog({"enabled": True})
        wd.register_job("intraday", 5)
        wd.heartbeat("intraday", datetime.utcnow() - timedelta(minutes=9))
        stale = wd.check(datetime.utcnow())
        self.assertEqual(len(stale), 1)
        self.assertTrue(stale[0].stale)

    def test_fill_reconciler_detects_slippage_overrun_and_ghost_position(self) -> None:
        reconciler = FillReconciler(max_slippage_multiple=2.0)
        signal = type(
            "Signal",
            (),
            {"asset": "AAPL", "entry_price": 100.0, "slippage_cost_pct": 0.0001},
        )()
        receipt = type(
            "Receipt",
            (),
            {"status": "FILLED", "fill_price": 101.0, "fill_ratio": 1.0},
        )()
        events = reconciler.reconcile_receipt(signal, receipt)
        self.assertTrue(any(e.status == "OVERRUN" for e in events))

        pos_events = reconciler.reconcile_positions(
            internal_positions=[],
            broker_positions=[{"tradingsymbol": "AAPL", "quantity": 1}],
        )
        self.assertTrue(any(e.status == "GHOST_POSITION" for e in pos_events))

    def test_signal_store_persists_order_and_health_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SignalStore(Path(tmp_dir) / "signals.db")
            store.save_order(
                {
                    "order_id": "ord-1",
                    "signal_id": "sig-1",
                    "asset": "AAPL",
                    "side": "BUY",
                    "broker": "paper",
                    "state": "FILLED",
                    "requested_price": 100.0,
                    "fill_price": 100.1,
                    "requested_qty": 2.0,
                    "filled_qty": 2.0,
                    "slippage_pct": 0.1,
                    "expected_slippage_pct": 0.02,
                    "error": "",
                    "broker_payload": "{}",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
            store.save_system_health(
                {
                    "health_id": "h-1",
                    "component": "scheduler",
                    "status": "WARNING",
                    "message": "late heartbeat",
                    "details": "{}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            self.assertEqual(len(store.get_recent_orders()), 1)
            self.assertEqual(len(store.get_recent_system_health()), 1)
            store.engine.dispose()


if __name__ == "__main__":
    unittest.main()
