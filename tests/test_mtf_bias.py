import unittest

import numpy as np
import pandas as pd

from src.dashboard.mtf_bias import MTFBiasFilter


def _make_frame(kind: str, n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    if kind == "bullish":
        close = np.linspace(100.0, 160.0, n)
    elif kind == "bearish":
        close = np.linspace(160.0, 100.0, n)
    else:
        close = np.full(n, 120.0)

    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


class _DummySvc:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.calls: dict[str, int] = {}

    def get_recent_frame(self, interval: str = "15m", limit: int = 240) -> pd.DataFrame:
        self.calls[interval] = self.calls.get(interval, 0) + 1
        return self.frames.get(interval, pd.DataFrame())


class MTFBiasTests(unittest.TestCase):
    def test_long_4h_bearish_hard_block(self) -> None:
        svc = _DummySvc({"4h": _make_frame("bearish"), "1d": _make_frame("bullish")})
        flt = MTFBiasFilter(svc)
        out = flt.check_alignment("LONG", confidence=95.0)
        self.assertFalse(out["alignment_ok"])
        self.assertEqual(out["bias_4h"], "BEARISH")
        self.assertIn("4H structure bearish", str(out["block_reason"]))

    def test_short_4h_bullish_hard_block(self) -> None:
        svc = _DummySvc({"4h": _make_frame("bullish"), "1d": _make_frame("bearish")})
        flt = MTFBiasFilter(svc)
        out = flt.check_alignment("SHORT", confidence=95.0)
        self.assertFalse(out["alignment_ok"])
        self.assertEqual(out["bias_4h"], "BULLISH")
        self.assertIn("4H structure bullish", str(out["block_reason"]))

    def test_long_bullish_bullish_score_one(self) -> None:
        svc = _DummySvc({"4h": _make_frame("bullish"), "1d": _make_frame("bullish")})
        flt = MTFBiasFilter(svc)
        out = flt.check_alignment("LONG", confidence=90.0)
        self.assertTrue(out["alignment_ok"])
        self.assertAlmostEqual(float(out["alignment_score"]), 1.0, places=6)

    def test_long_neutral_neutral_score_point_four(self) -> None:
        svc = _DummySvc({"4h": _make_frame("neutral"), "1d": _make_frame("neutral")})
        flt = MTFBiasFilter(svc)
        out = flt.check_alignment("LONG", confidence=90.0)
        self.assertTrue(out["alignment_ok"])
        self.assertEqual(out["bias_4h"], "NEUTRAL")
        self.assertEqual(out["bias_1d"], "NEUTRAL")
        self.assertAlmostEqual(float(out["alignment_score"]), 0.4, places=6)

    def test_cache_second_call_uses_cached_data(self) -> None:
        svc = _DummySvc({"4h": _make_frame("bullish")})
        flt = MTFBiasFilter(svc)
        first = flt.get_bias("4h")
        second = flt.get_bias("4h")
        self.assertFalse(bool(first["cached"]))
        self.assertTrue(bool(second["cached"]))
        self.assertEqual(svc.calls.get("4h", 0), 1)

    def test_long_1d_bearish_confidence_75_soft_block(self) -> None:
        svc = _DummySvc({"4h": _make_frame("bullish"), "1d": _make_frame("bearish")})
        flt = MTFBiasFilter(svc)
        out = flt.check_alignment("LONG", confidence=75.0)
        self.assertFalse(out["alignment_ok"])
        self.assertIn("need 80%+ confidence", str(out["block_reason"]))

    def test_long_1d_bearish_confidence_85_allowed(self) -> None:
        svc = _DummySvc({"4h": _make_frame("bullish"), "1d": _make_frame("bearish")})
        flt = MTFBiasFilter(svc)
        out = flt.check_alignment("LONG", confidence=85.0)
        self.assertTrue(out["alignment_ok"])
        self.assertGreaterEqual(float(out["alignment_score"]), 0.0)


if __name__ == "__main__":
    unittest.main()
