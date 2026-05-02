import unittest

from src.dashboard import api


class PaperExecutionMarketGuardTests(unittest.TestCase):
    def test_rejects_execution_plan_when_live_price_is_outside_entry_zone(self) -> None:
        msg = api._paper_execution_market_guard(
            {
                "signal": "LONG",
                "entry_price": 78337.30,
                "stop_loss": 78034.85,
                "take_profit": 78700.24,
            },
            76255.17,
            entry_zone_low=78252.90,
            entry_zone_high=78421.70,
            enforce_entry_zone=True,
        )

        self.assertIsNotNone(msg)
        self.assertIn("outside execution entry zone", str(msg))

    def test_rejects_long_when_live_price_already_crossed_stop(self) -> None:
        msg = api._paper_execution_market_guard(
            {
                "signal": "LONG",
                "entry_price": 105.0,
                "stop_loss": 100.0,
                "take_profit": 112.0,
            },
            99.5,
        )

        self.assertIsNotNone(msg)
        self.assertIn("LONG stop loss", str(msg))

    def test_alt_symbol_message_uses_asset_label(self) -> None:
        msg = api._paper_execution_market_guard(
            {
                "ticker": "ETHUSDT",
                "signal": "LONG",
                "entry_price": 2300.0,
                "stop_loss": 2250.0,
                "take_profit": 2400.0,
            },
            2249.5,
        )

        self.assertIsNotNone(msg)
        self.assertIn("Live ETH price", str(msg))
        self.assertNotIn("Live BTC price", str(msg))

    def test_allows_live_price_between_stop_and_take_profit(self) -> None:
        msg = api._paper_execution_market_guard(
            {
                "signal": "LONG",
                "entry_price": 105.0,
                "stop_loss": 100.0,
                "take_profit": 112.0,
            },
            104.0,
        )

        self.assertIsNone(msg)


if __name__ == "__main__":
    unittest.main()
