from __future__ import annotations

from datetime import datetime, timezone

from btc_intelligence.services.walk_forward import (
    WalkForwardConfig,
    WalkForwardValidator,
    deflated_sharpe_ratio,
)


def test_deflated_sharpe_ratio_basic():
    r = [0.01, -0.005, 0.008, 0.002, -0.001, 0.004, 0.003, -0.002]
    out = deflated_sharpe_ratio(r, n_trials=5)
    assert "deflated_sharpe_ratio" in out
    assert "sharpe_observed" in out


def test_purged_kfold_populates_dsr():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(60):
        rows.append(
            {
                "pnl_pct": 0.5 if i % 3 else -0.3,
                "timestamp": (base.replace(hour=i % 24, minute=i % 60)).isoformat().replace("+00:00", "Z"),
                "regime": "X",
            }
        )
    v = WalkForwardValidator(
        WalkForwardConfig(n_splits=4, min_rows_for_pkf=24, train_window=500, test_window=500)
    )
    out = v.validate_weight_update(
        "trend_following",
        {"trend_following": 0.3},
        {"trend_following": 0.32},
        rows,
    )
    assert out.get("validation_mode") in {"purged_kfold", "legacy_split"}
    assert "deflated_sharpe_ratio" in out
    assert "mean_oos_sharpe" in out or out.get("new_sharpe") is not None
