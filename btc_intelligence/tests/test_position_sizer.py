from __future__ import annotations

import math

from btc_intelligence.services.position_sizer import (
    KellyConfig,
    KellyPositionSizer,
    book_microstructure_from_depth,
)


def test_book_microstructure_from_depth_empty():
    assert book_microstructure_from_depth({})["spread_bps"] == 0.0


def test_dynamic_R_wider_spread_reduces_b():
    s = KellyPositionSizer(
        KellyConfig(
            reference_spread_bps=4.0,
            reference_depth_notional_usd=1_000_000.0,
            execution_rr_weight=0.0,
        )
    )
    base = s.compute(
        win_rate=0.55,
        avg_win_pct=0.02,
        avg_loss_pct=0.01,
        confidence=80.0,
        drift_level="LOW",
        edge_decay=False,
        current_drawdown_pct=0.0,
        volatility_regime="NORMAL",
        portfolio_value=100_000.0,
        spread_bps=4.0,
        bid_notional_usd=2_000_000.0,
        ask_notional_usd=2_000_000.0,
        strategy_returns_pct=[],
        portfolio_heat_pct=0.0,
    )
    wide = s.compute(
        win_rate=0.55,
        avg_win_pct=0.02,
        avg_loss_pct=0.01,
        confidence=80.0,
        drift_level="LOW",
        edge_decay=False,
        current_drawdown_pct=0.0,
        volatility_regime="NORMAL",
        portfolio_value=100_000.0,
        spread_bps=16.0,
        bid_notional_usd=2_000_000.0,
        ask_notional_usd=2_000_000.0,
        strategy_returns_pct=[],
        portfolio_heat_pct=0.0,
    )
    assert wide["b"] < base["b"]


def test_cvar_and_stress_cap_bind_position():
    returns = [-3.0 - i * 0.1 for i in range(30)]
    s = KellyPositionSizer(KellyConfig(max_position_pct=0.05, cvar_equity_budget_pct=0.01, stress_equity_loss_cap_pct=0.02))
    out = s.compute(
        win_rate=0.52,
        avg_win_pct=0.02,
        avg_loss_pct=0.01,
        confidence=90.0,
        drift_level="LOW",
        edge_decay=False,
        current_drawdown_pct=0.0,
        volatility_regime="NORMAL",
        portfolio_value=100_000.0,
        strategy_returns_pct=returns,
        portfolio_heat_pct=1.0,
    )
    assert out["position_pct"] <= s.config.max_position_pct
    assert out["risk_budgets"]["cvar_95_pct"] is not None
    assert out["p"] == 0.52
    assert "raw_kelly" in out and not math.isnan(out["raw_kelly"])
    assert out["position_size_pct"] == out["position_pct"]


def test_execution_rr_blends_into_b():
    s = KellyPositionSizer(KellyConfig(execution_rr_weight=0.5))
    without = s.compute(
        win_rate=0.5,
        avg_win_pct=0.015,
        avg_loss_pct=0.01,
        confidence=70.0,
        drift_level="LOW",
        edge_decay=False,
        current_drawdown_pct=0.0,
        volatility_regime="NORMAL",
        portfolio_value=50_000.0,
        execution_rr=None,
        spread_bps=4.0,
        bid_notional_usd=3_000_000.0,
        ask_notional_usd=3_000_000.0,
    )
    with_rr = s.compute(
        win_rate=0.5,
        avg_win_pct=0.015,
        avg_loss_pct=0.01,
        confidence=70.0,
        drift_level="LOW",
        edge_decay=False,
        current_drawdown_pct=0.0,
        volatility_regime="NORMAL",
        portfolio_value=50_000.0,
        execution_rr=2.5,
        spread_bps=4.0,
        bid_notional_usd=3_000_000.0,
        ask_notional_usd=3_000_000.0,
    )
    assert abs(with_rr["b"] - without["b"]) > 1e-6


def test_merge_tail_risk_preserves_entry_zone():
    from btc_intelligence.services.execution_planner import ExecutionPlanner

    plan = {"entry_zone": [1.0, 2.0], "expected_rr": 1.5}
    kelly = {
        "position_pct": 0.01,
        "position_size_usd": 500.0,
        "p": 0.5,
        "b": 1.2,
        "raw_kelly": 0.08,
        "risk_budgets": {"cvar_95_pct": -2.0},
    }
    merged = ExecutionPlanner.merge_tail_risk(plan, kelly)
    assert merged["entry_zone"] == [1.0, 2.0]
    assert merged["position_size_pct"] == 0.01
    assert merged["p"] == 0.5
    assert merged["tail_risk_sizing"]["bounded_position_pct"] == 0.01
