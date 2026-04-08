from __future__ import annotations

import numpy as np

from btc_intelligence.features.feature_vector import FEATURE_COLUMNS
from btc_intelligence.services.ic_monitor import (
    ICFactorMonitor,
    apply_hibernation_mask,
    apply_hibernation_to_stack_edges,
)


def test_apply_hibernation_mask_zeros_columns() -> None:
    fm = {c: float(i % 3) for i, c in enumerate(FEATURE_COLUMNS)}
    vec = np.asarray([[fm[c] for c in FEATURE_COLUMNS]], dtype=float)
    hib = {"ema9_dist", "cvd_slope"}
    fm2, v2 = apply_hibernation_mask(fm, vec, hib)
    assert fm2["ema9_dist"] == 0.0
    assert fm2["cvd_slope"] == 0.0
    assert fm2["ema21_dist"] == fm["ema21_dist"]
    idx = FEATURE_COLUMNS.index("ema9_dist")
    assert v2[0, idx] == 0.0


def test_stack_edges_neutralized_when_hibernated() -> None:
    edges = {"cvd_confirming": 0.8, "macro_risk_on": 0.7}
    factors = list(edges.keys())
    adj, stacked = apply_hibernation_to_stack_edges(edges, factors, {"cvd_slope"})
    assert adj["cvd_confirming"] == 0.5
    assert adj["macro_risk_on"] == 0.7
    assert 0.0 <= stacked <= 1.0


def test_ic_monitor_recompute_runs() -> None:
    mon = ICFactorMonitor()
    base = {c: 0.0 for c in FEATURE_COLUMNS}
    rng = np.random.default_rng(42)
    for i in range(40):
        f = dict(base)
        f["ema9_dist"] = float(i) * 0.01
        f["cvd_slope"] = float(rng.normal(0, 1))
        y = f["ema9_dist"] * 0.5 + float(rng.normal(0, 0.01))
        mon._rows.append({"factors": f, "fwd_log_ret": float(y)})
    with mon._lock:
        mon._recompute_hibernation_locked()
    assert isinstance(mon.hibernated_factors, set)
    assert isinstance(mon.factor_ics, dict)
    assert len(mon.factor_ics) == len(FEATURE_COLUMNS)
