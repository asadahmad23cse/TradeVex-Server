from __future__ import annotations

from btc_intelligence.signals.execution_adverse_selection import compute_adverse_selection


def _depth(bid: float, ask: float) -> dict:
    return {"bids": [[bid, 1.0]], "asks": [[ask, 1.0]]}


def _trades_linear(t0: int, prices: list[float]) -> list[dict]:
    """Ascending timestamps 100ms apart."""
    return [{"time": t0 + i * 100, "price": p, "qty": 0.01} for i, p in enumerate(prices)]


def test_insufficient_depth() -> None:
    r = compute_adverse_selection({}, [], "LONG")
    assert r.adverse_selection_flag is False
    assert r.execution_mode_recommendation == "LIMIT"
    assert r.reason == "depth_unavailable"


def test_insufficient_tape() -> None:
    mid = 100_000.0
    spread = mid * 0.0001
    d = _depth(mid - spread / 2, mid + spread / 2)
    r = compute_adverse_selection(d, [], "LONG", min_trades=5)
    assert r.adverse_selection_flag is False
    assert r.reason == "insufficient_tape"


def test_long_adverse_when_spread_wide_and_drift_down() -> None:
    """Wide book + tape drifting down => adverse for LONG."""
    mid = 100_000.0
    spread = mid * 0.0005
    depth = _depth(mid - spread / 2, mid + spread / 2)
    t0 = 1_700_000_000_000
    prices = [100_000.0] * 8 + [99_985.0] * 8
    trades = _trades_linear(t0, prices)
    r = compute_adverse_selection(
        depth,
        trades,
        "LONG",
        window_ms=50_000,
        drift_threshold_pct=0.01,
        spread_widen_ratio=1.05,
        min_trades=10,
        noise_mad_multiplier=0.5,
    )
    assert r.spread_widening is True
    assert r.mid_drift_pct < 0
    assert r.adverse_selection_flag is True
    assert r.execution_mode_recommendation == "HIDDEN_PASSIVE"


def test_short_adverse_when_spread_wide_and_drift_up() -> None:
    mid = 50_000.0
    spread = mid * 0.0006
    depth = _depth(mid - spread / 2, mid + spread / 2)
    t0 = 1_700_000_000_000
    prices = [50_000.0] * 8 + [50_020.0] * 8
    trades = _trades_linear(t0, prices)
    r = compute_adverse_selection(
        depth,
        trades,
        "SHORT",
        window_ms=50_000,
        drift_threshold_pct=0.01,
        spread_widen_ratio=1.05,
        min_trades=10,
        noise_mad_multiplier=0.5,
    )
    assert r.spread_widening is True
    assert r.mid_drift_pct > 0
    assert r.adverse_selection_flag is True
    assert r.execution_mode_recommendation == "HIDDEN_PASSIVE"


def test_calm_market_no_flag() -> None:
    mid = 100_000.0
    spread = mid * 0.00005
    depth = _depth(mid - spread / 2, mid + spread / 2)
    t0 = 1_700_000_000_000
    prices = [100_000.0 + i * 0.1 for i in range(20)]
    trades = _trades_linear(t0, prices)
    r = compute_adverse_selection(
        depth,
        trades,
        "LONG",
        window_ms=50_000,
        spread_widen_ratio=2.0,
        min_trades=10,
    )
    assert r.adverse_selection_flag is False
    assert r.execution_mode_recommendation == "LIMIT"
