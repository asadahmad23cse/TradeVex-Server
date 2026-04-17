from __future__ import annotations

import json
import math
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

IC_CHECKPOINT_FILE = Path("btc_intelligence/logs/ic_monitor_checkpoint.json")

from btc_intelligence.config import settings
from btc_intelligence.features.feature_vector import FEATURE_COLUMNS, FeatureState


# Probability-stack edge keys -> FEATURE_COLUMNS they lean on (any hit hibernates edge weight to neutral).
EDGE_TO_FEATURES: dict[str, tuple[str, ...]] = {
    "mtf_3_alignment": ("mtf_alignment_score", "ema9_dist", "ema21_dist", "ema50_dist"),
    "ob_retest_with_fvg": ("ob_distance_pct", "fvg_present"),
    "cvd_confirming": ("cvd_slope", "obi"),
    "funding_against_crowd": ("funding_rate", "funding_roc"),
    "macro_risk_on": ("fng_score", "vix_level", "dxy_bias_encoded", "spx_intraday_direction"),
    "absorption_confirming": ("absorption_strength", "obi"),
    "london_session": ("session_encoded",),
    "liquidity_sweep_confirming": ("liquidity_sweep_recent",),
    "breakout_retest": ("bos_candles_ago", "choch_detected", "bb_position"),
    "stacked_imbalance_confirming": ("stacked_imbalance_direction",),
}

# Four thematic clusters for SHAP / reporting (each FEATURE_COLUMNS name appears once).
FACTOR_CLUSTER: dict[str, str] = {}
for _c in FEATURE_COLUMNS:
    FACTOR_CLUSTER[_c] = "Momentum"

_MOMENTUM = {
    "ema9_dist",
    "ema21_dist",
    "ema50_dist",
    "ema200_dist",
    "mtf_alignment_score",
    "bos_candles_ago",
    "choch_detected",
    "ob_distance_pct",
    "fvg_present",
    "liquidity_sweep_recent",
    "vwap_distance",
}
_FLOW = {
    "cvd_slope",
    "obi",
    "ofi_5bar",
    "whale_buy_ratio",
    "absorption_strength",
    "stacked_imbalance_direction",
    "single_bar_divergence",
    "iceberg_direction",
    "cross_exchange_cvd_divergence",
    "speed_of_tape_ratio",
    "funding_rate",
    "funding_roc",
    "oi_change_1h",
    "ls_ratio",
}
_VOL = {
    "atr_normalized",
    "bb_position",
    "liq_cluster_above_pct",
    "liq_cluster_below_pct",
    "max_pain_distance_pct",
    "iv_skew",
    "put_call_ratio",
    "iv_rv_ratio",
    "options_expiry_hours",
    "vol_regime_encoded",
}
_SENT = {
    "exchange_netflow_score",
    "sopr",
    "lth_supply_change",
    "whale_exchange_deposit_count",
    "whale_net_flow_score",
    "fng_score",
    "dxy_bias_encoded",
    "vix_level",
    "us_10y_yield_trend",
    "spx_btc_correlation",
    "spx_intraday_direction",
    "gold_btc_agreement",
    "eth_btc_ratio_trend",
    "btc_dominance_trend",
    "session_encoded",
    "regime_encoded",
}
for _name in _MOMENTUM:
    FACTOR_CLUSTER[_name] = "Momentum"
for _name in _FLOW:
    FACTOR_CLUSTER[_name] = "Flow"
for _name in _VOL:
    FACTOR_CLUSTER[_name] = "Volatility"
for _name in _SENT:
    FACTOR_CLUSTER[_name] = "Sentiment"

_ic_lock = threading.Lock()
_ic_singleton: Any = None


class ICFactorMonitor:
    """
    Rolling daily Spearman IC vs one-step forward log return; factors below threshold hibernate.
    """

    def __init__(self) -> None:
        self._rows: deque[dict[str, Any]] = deque(maxlen=max(int(settings.ic_rolling_days) + 10, 80))
        self._last_ic_date: str | None = None
        self._last_close: float | None = None
        self._last_factors: dict[str, float] | None = None
        self.hibernated_factors: set[str] = set()
        self.factor_ics: dict[str, float] = {}
        self._lock = threading.Lock()
        self._load_checkpoint()

    def maybe_record_daily(self, snapshot: dict[str, Any], state: FeatureState, regime: str) -> None:
        """Call from hot path; only writes history when UTC calendar day advances (1h candles)."""
        candles_1h = snapshot.get("candles", {}).get("1h", [])
        if not candles_1h:
            return
        last = candles_1h[-1]
        ts = int(last.get("open_time", 0)) / 1000.0
        if ts <= 0:
            return
        day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        close = float(last.get("close", 0.0))
        if close <= 0:
            return
        _, feature_map = state.to_vector(direction="LONG", regime=regime)

        with self._lock:
            if self._last_ic_date == day:
                self._last_factors = dict(feature_map)
                self._last_close = close
                return

            if self._last_ic_date is not None and self._last_factors is not None and self._last_close is not None:
                prev_c = float(self._last_close)
                if prev_c > 0:
                    fwd = math.log(close / prev_c)
                    self._rows.append({"factors": dict(self._last_factors), "fwd_log_ret": float(fwd)})

            self._last_ic_date = day
            self._last_factors = dict(feature_map)
            self._last_close = close
            self._recompute_hibernation_locked()

    def _recompute_hibernation_locked(self) -> None:
        window = int(settings.ic_rolling_days)
        min_samp = int(settings.ic_min_samples)
        thr = float(settings.ic_hibernation_threshold)
        rows = [r for r in self._rows if r.get("fwd_log_ret") is not None]
        if len(rows) < min_samp:
            self.hibernated_factors = set()
            self.factor_ics = {}
            return

        take = rows[-window:] if len(rows) > window else rows
        y = np.asarray([float(r["fwd_log_ret"]) for r in take], dtype=float)

        ics: dict[str, float] = {}
        hib: set[str] = set()
        for col in FEATURE_COLUMNS:
            x = np.asarray([float(r["factors"].get(col, 0.0)) for r in take], dtype=float)
            if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
                ics[col] = 0.0
                continue
            corr, _ = spearmanr(x, y, nan_policy="omit")
            ic = 0.0 if corr is None or (isinstance(corr, float) and np.isnan(corr)) else float(corr)
            ics[col] = ic
            if ic < thr:
                hib.add(col)

        self.factor_ics = ics
        self.hibernated_factors = hib
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        try:
            IC_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            rows_list = [dict(r) for r in self._rows]
            payload = {
                "hibernated_factors": sorted(self.hibernated_factors),
                "factor_ics": {k: round(v, 6) for k, v in self.factor_ics.items()},
                "last_ic_date": self._last_ic_date,
                "last_close": self._last_close,
                "rows": rows_list[-256:],
            }
            IC_CHECKPOINT_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("IC monitor checkpoint save failed: %s", exc)

    def _load_checkpoint(self) -> None:
        if not IC_CHECKPOINT_FILE.exists():
            return
        try:
            raw = json.loads(IC_CHECKPOINT_FILE.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            hib = raw.get("hibernated_factors", [])
            if isinstance(hib, list):
                self.hibernated_factors = set(str(f) for f in hib)
            ics = raw.get("factor_ics", {})
            if isinstance(ics, dict):
                self.factor_ics = {str(k): float(v) for k, v in ics.items()}
            self._last_ic_date = raw.get("last_ic_date")
            self._last_close = raw.get("last_close")
            rows = raw.get("rows", [])
            if isinstance(rows, list):
                for r in rows:
                    if isinstance(r, dict) and "factors" in r and "fwd_log_ret" in r:
                        self._rows.append(r)
            import logging
            logging.getLogger(__name__).info(
                "IC monitor: loaded checkpoint — %d rows, %d hibernated factors",
                len(self._rows), len(self.hibernated_factors),
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("IC monitor checkpoint load failed: %s", exc)

    def export_history_tail(self, n: int = 256) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._rows)[-n:]
        return [{"factors": r["factors"], "fwd_log_ret": r["fwd_log_ret"]} for r in rows]


def get_ic_monitor() -> ICFactorMonitor:
    global _ic_singleton
    with _ic_lock:
        if _ic_singleton is None:
            _ic_singleton = ICFactorMonitor()
        return _ic_singleton


def apply_hibernation_mask(
    feature_map: dict[str, float],
    vector: np.ndarray,
    hibernated: set[str],
) -> tuple[dict[str, float], np.ndarray]:
    if not hibernated:
        return feature_map, vector
    fm = dict(feature_map)
    for k in hibernated:
        if k in fm:
            fm[k] = 0.0
    arr = np.asarray([[float(fm.get(c, 0.0)) for c in FEATURE_COLUMNS]], dtype=float)
    return fm, arr


def apply_hibernation_to_stack_edges(
    independent_edges: dict[str, float],
    factors: list[str],
    hibernated: set[str],
) -> tuple[dict[str, float], float]:
    """Neutralize (0.5) edges that depend on any hibernated factor; recompute stacked probability."""
    edges = {f: float(independent_edges.get(f, 0.5)) for f in factors}
    for edge in list(edges.keys()):
        deps = EDGE_TO_FEATURES.get(edge, ())
        if deps and any(d in hibernated for d in deps):
            edges[edge] = 0.5
    product = 1.0
    for p in edges.values():
        product *= 1.0 - p * 0.7
    stacked = max(0.0, min(1.0, 1.0 - product))
    return edges, stacked


def cluster_weights_for_json(mean_abs_shap: dict[str, float]) -> dict[str, Any]:
    """Aggregate mean |SHAP| by thematic cluster."""
    out: dict[str, float] = {"Momentum": 0.0, "Flow": 0.0, "Volatility": 0.0, "Sentiment": 0.0}
    for feat, val in mean_abs_shap.items():
        bucket = FACTOR_CLUSTER.get(feat, "Sentiment")
        out[bucket] = out.get(bucket, 0.0) + float(val)
    return {"cluster_importance": out, "per_feature": mean_abs_shap}
