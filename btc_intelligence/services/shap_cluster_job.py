from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from btc_intelligence.config import settings
from btc_intelligence.features.feature_vector import FEATURE_COLUMNS
from btc_intelligence.models.inference import ModelInference
from btc_intelligence.services.ic_monitor import FACTOR_CLUSTER, cluster_weights_for_json, get_ic_monitor

logger = logging.getLogger(__name__)


def _build_xy_from_monitor(n_max: int = 512) -> tuple[np.ndarray, np.ndarray] | None:
    mon = get_ic_monitor()
    tail = mon.export_history_tail(n_max)
    if len(tail) < int(settings.ic_min_samples):
        return None
    X = np.asarray([[float(r["factors"].get(c, 0.0)) for c in FEATURE_COLUMNS] for r in tail], dtype=float)
    y = np.asarray([float(r["fwd_log_ret"]) for r in tail], dtype=float)
    if X.shape[0] != y.shape[0] or X.shape[0] < int(settings.ic_min_samples):
        return None
    return X, y


def run_shap_cluster_snapshot(model: ModelInference | None = None) -> None:
    """
    Background-only: compute clustered SHAP summary and write JSON. Never called from signal hot path.
    """
    out_path = Path(settings.shap_cluster_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "as_of_utc": now,
        "clusters": {"Momentum": [], "Flow": [], "Volatility": [], "Sentiment": []},
        "mean_abs_shap": {},
        "cluster_importance": {},
        "source": "fallback_ic_proxy",
    }

    for col in FEATURE_COLUMNS:
        bucket = FACTOR_CLUSTER.get(col, "Sentiment")
        payload["clusters"].setdefault(bucket, []).append(col)

    xy = _build_xy_from_monitor()
    mean_abs: dict[str, float] = {c: 0.0 for c in FEATURE_COLUMNS}

    if xy is not None:
        X, y = xy
        # Proxy: abs Spearman partial proxy via linear corr of rank — fast, no SHAP dep if model missing
        try:
            from scipy.stats import spearmanr

            for j, col in enumerate(FEATURE_COLUMNS):
                xs = X[:, j]
                if np.nanstd(xs) < 1e-12:
                    continue
                r, _ = spearmanr(xs, y, nan_policy="omit")
                if r is not None and not np.isnan(r):
                    mean_abs[col] = float(abs(r))
        except Exception as exc:
            logger.debug("shap_cluster_job IC proxy failed: %s", exc)

    lgbm = None
    if model is not None and getattr(model, "default_bundle", None) is not None:
        lgbm = model.default_bundle.lgbm

    if lgbm is not None and xy is not None:
        try:
            import shap

            X, y = xy
            n = min(X.shape[0], 300)
            bg = X[: max(32, n // 3)]
            explainer = shap.TreeExplainer(lgbm, data=bg)
            sv = explainer.shap_values(X[-n:])
            if isinstance(sv, list):
                sv = np.asarray(sv[1] if len(sv) > 1 else sv[0])
            arr = np.asarray(sv, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, len(FEATURE_COLUMNS))
            if arr.shape[1] == len(FEATURE_COLUMNS):
                for j, col in enumerate(FEATURE_COLUMNS):
                    mean_abs[col] = float(np.mean(np.abs(arr[:, j])))
                payload["source"] = "shap_tree_explainer"
        except Exception as exc:
            logger.debug("SHAP TreeExplainer skipped: %s", exc)

    payload["mean_abs_shap"] = {k: round(v, 6) for k, v in mean_abs.items()}
    merged = cluster_weights_for_json(mean_abs)
    payload["cluster_importance"] = {k: round(v, 6) for k, v in merged["cluster_importance"].items()}
    payload["hibernated_factors"] = sorted(get_ic_monitor().hibernated_factors)
    payload["factor_ics"] = {k: round(v, 6) for k, v in get_ic_monitor().factor_ics.items()}

    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("shap_clusters.json write failed: %s", exc)
