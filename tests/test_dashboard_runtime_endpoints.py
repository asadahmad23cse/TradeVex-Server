import unittest

from src.dashboard import api


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


if __name__ == "__main__":
    unittest.main()
