"""
Background fit of ElasticNet logistic meta-calibrator on trade history (not on hot path).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from btc_intelligence.config import settings
from btc_intelligence.services.adaptive_learning import FACTOR_KEYS, REGIMES, AdaptiveLearningEngine

logger = logging.getLogger(__name__)


def run_elasticnet_calibration_fit(engine: AdaptiveLearningEngine) -> None:
    if not bool(getattr(settings, "calibration_meta_fit_enabled", True)):
        return
    out_path = Path(str(getattr(settings, "elasticnet_calibrator_path", "btc_intelligence/logs/elasticnet_calibrator.json")))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for reg in REGIMES:
        for r in engine.trade_history.get(reg, []):
            rows.append(dict(r))

    if len(rows) < int(getattr(settings, "calibration_min_trades", 25)) + 5:
        return

    rows.sort(key=lambda r: str(r.get("timestamp", "")))
    n = len(rows)
    split = max(15, int(float(getattr(settings, "calibration_train_frac", 0.65)) * n))
    split = min(split, n - 5)
    train = rows[:split]

    X_list: list[list[float]] = []
    y_list: list[int] = []
    for r in train:
        p = float(r.get("raw_prob", 0.5))
        fac = r.get("factors", {})
        xrow = [p] + [float(fac.get(k, 0.0)) for k in FACTOR_KEYS]
        X_list.append(xrow)
        y_list.append(1 if float(r.get("pnl_pct", 0.0)) > 0 else 0)

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=int)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xn = (X - mean) / std

    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            C=1.0,
            max_iter=400,
            random_state=42,
        )
        clf.fit(Xn, y)
        coef = clf.coef_.ravel().tolist()
        intercept = float(clf.intercept_[0])
    except Exception as exc:
        logger.debug("ElasticNet calibrator fit skipped: %s", exc)
        return

    payload = {
        "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "feature_order": ["raw_prob"] + list(FACTOR_KEYS),
        "coef": coef,
        "intercept": intercept,
        "mean": mean.tolist(),
        "scale": std.tolist(),
        "train_samples": int(len(train)),
    }
    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("elasticnet_calibrator.json write failed: %s", exc)
