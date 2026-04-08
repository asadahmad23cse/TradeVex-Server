from __future__ import annotations

from bisect import bisect_right
from collections import deque
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


class DataDriftEngine:
    """
    Lightweight online drift detector for decision features.
    Detects distribution shift and returns safety adjustments.
    """

    _FEATURE_KEYS = ("momentum", "flow", "cost", "volatility", "probability")

    def __init__(
        self,
        baseline_window: int = 240,
        recent_window: int = 60,
        state_path: str | None = None,
        baseline_stats_path: str | None = None,
    ) -> None:
        self.baseline_window = max(120, int(baseline_window))
        self.recent_window = max(30, int(recent_window))
        self._history: deque[dict[str, float]] = deque(maxlen=self.baseline_window + self.recent_window + 80)
        resolved_state_path = state_path if state_path is not None else baseline_stats_path
        self._state_path = Path(resolved_state_path) if resolved_state_path else None
        self._baseline_stats: dict[str, dict[str, Any]] = {}
        self._load_baseline_stats()

    @staticmethod
    def _safe_stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        mu = mean(values)
        if len(values) >= 2:
            sd = stdev(values)
        else:
            sd = 1.0
        return float(mu), float(max(sd, 1e-6))

    @staticmethod
    def _build_bucket_edges(values: list[float], n_bins: int = 10) -> list[float]:
        if not values or n_bins < 1:
            return []
        vmin = float(min(values))
        vmax = float(max(values))
        if vmax <= vmin:
            eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
            low = vmin - eps
            high = vmin + eps
            step = (high - low) / float(n_bins)
            return [float(low + (step * i)) for i in range(n_bins + 1)]
        width = (vmax - vmin) / float(n_bins)
        edges = [float(vmin + (width * i)) for i in range(n_bins + 1)]
        edges[-1] = vmax
        return edges

    @staticmethod
    def _bucket_percentages(values: list[float], bucket_edges: list[float]) -> list[float]:
        n_bins = max(0, len(bucket_edges) - 1)
        if n_bins <= 0:
            return []
        counts = [0] * n_bins
        for value in values:
            idx = bisect_right(bucket_edges, float(value)) - 1
            idx = max(0, min(n_bins - 1, idx))
            counts[idx] += 1
        total = max(1, len(values))
        return [float(c) / float(total) for c in counts]

    @staticmethod
    def _compute_psi(baseline_vals: list[float], recent_vals: list[float], n_bins: int = 10) -> float:
        if not baseline_vals or not recent_vals or len(baseline_vals) < 1:
            return 0.0
        edges = DataDriftEngine._build_bucket_edges(baseline_vals, n_bins=n_bins)
        if len(edges) < 2:
            return 0.0
        expected_pct = DataDriftEngine._bucket_percentages(baseline_vals, edges)
        actual_pct = DataDriftEngine._bucket_percentages(recent_vals, edges)
        eps = 1e-6
        psi = 0.0
        for actual, expected in zip(actual_pct, expected_pct):
            a = max(float(actual), eps)
            e = max(float(expected), eps)
            psi += (a - e) * math.log(a / e)
        return float(max(0.0, psi))

    @staticmethod
    def _compute_psi_from_profile(
        recent_vals: list[float],
        bucket_edges: list[float],
        expected_pct: list[float],
    ) -> float:
        n_bins = max(0, len(bucket_edges) - 1)
        if not recent_vals or n_bins <= 0 or len(expected_pct) != n_bins:
            return 0.0
        actual_pct = DataDriftEngine._bucket_percentages(recent_vals, bucket_edges)
        eps = 1e-6
        psi = 0.0
        for actual, expected in zip(actual_pct, expected_pct):
            a = max(float(actual), eps)
            e = max(float(expected), eps)
            psi += (a - e) * math.log(a / e)
        return float(max(0.0, psi))

    def _load_baseline_stats(self) -> None:
        if not self._state_path or (not self._state_path.exists()):
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            features_payload = payload.get("features", {}) if isinstance(payload, dict) else {}
            if not isinstance(features_payload, dict):
                return
            restored: dict[str, dict[str, Any]] = {}
            for key in self._FEATURE_KEYS:
                row = features_payload.get(key, {})
                if not isinstance(row, dict):
                    continue
                mu = float(row.get("mean", 0.0))
                sd = float(max(float(row.get("std", 1.0)), 1e-6))
                edges_raw = row.get("psi_bucket_edges", [])
                expected_raw = row.get("psi_expected_pct", [])
                edges = [float(x) for x in edges_raw] if isinstance(edges_raw, list) else []
                expected = [float(x) for x in expected_raw] if isinstance(expected_raw, list) else []
                if len(edges) < 2 or len(expected) != (len(edges) - 1):
                    edges = []
                    expected = []
                restored[key] = {
                    "mean": mu,
                    "std": sd,
                    "psi_bucket_edges": edges,
                    "psi_expected_pct": expected,
                }
            if len(restored) == len(self._FEATURE_KEYS):
                self._baseline_stats = restored
        except Exception:
            return

    def _save_baseline_stats(self) -> None:
        if not self._state_path or not self._baseline_stats:
            return
        payload = {
            "baseline_window": self.baseline_window,
            "recent_window": self.recent_window,
            "features": self._baseline_stats,
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _build_baseline_stats(self, baseline_rows: list[dict[str, float]]) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for key in self._FEATURE_KEYS:
            b_vals = [float(x.get(key, 0.0)) for x in baseline_rows]
            b_mu, b_sd = self._safe_stats(b_vals)
            psi_edges: list[float] = []
            psi_expected_pct: list[float] = []
            if len(b_vals) >= 30:
                psi_edges = self._build_bucket_edges(b_vals, n_bins=10)
                psi_expected_pct = self._bucket_percentages(b_vals, psi_edges) if psi_edges else []
            stats[key] = {
                "mean": float(b_mu),
                "std": float(max(b_sd, 1e-6)),
                "psi_bucket_edges": psi_edges,
                "psi_expected_pct": psi_expected_pct,
            }
        return stats

    @staticmethod
    def _variance_shift_score(variance_ratio: float) -> float:
        ratio = max(1e-6, float(variance_ratio))
        if ratio >= 1.0:
            return float(min((ratio - 1.0) / 1.0, 1.0))
        return float(min((1.0 - ratio) / 0.6, 1.0))

    def update_and_detect(self, features: dict[str, Any]) -> dict[str, Any]:
        row = {
            "momentum": float(features.get("momentum", 0.0)),
            "flow": float(features.get("flow", 0.0)),
            "cost": float(features.get("cost", 0.0)),
            "volatility": float(features.get("volatility", 0.0)),
            "probability": float(features.get("probability", 0.5)),
        }
        self._history.append(row)
        values = list(self._history)
        has_recent_window = len(values) >= self.recent_window
        baseline_rows: list[dict[str, float]] = []
        if has_recent_window:
            baseline_rows = values[: -self.recent_window]
            if len(baseline_rows) > self.baseline_window:
                baseline_rows = baseline_rows[-self.baseline_window :]

        if len(baseline_rows) >= 40:
            self._baseline_stats = self._build_baseline_stats(baseline_rows)
            self._save_baseline_stats()

        if not self._baseline_stats:
            return {
                "drift_level": "LOW",
                "drift_score": 0.0,
                "mean_shift_score": 0.0,
                "variance_shift_score": 0.0,
                "psi_drift_score": 0.0,
                "confidence_multiplier": 1.0,
                "learning_multiplier": 1.0,
                "status": "warming_up",
                "feature_shifts": {},
                "variance_ratios": {},
                "psi_scores": {},
            }

        recent = values[-self.recent_window :] if has_recent_window else values
        if not recent:
            return {
                "drift_level": "LOW",
                "drift_score": 0.0,
                "mean_shift_score": 0.0,
                "variance_shift_score": 0.0,
                "psi_drift_score": 0.0,
                "confidence_multiplier": 1.0,
                "learning_multiplier": 1.0,
                "status": "warming_up",
                "feature_shifts": {},
                "variance_ratios": {},
                "psi_scores": {},
            }
        shifts: dict[str, float] = {}
        variance_ratios: dict[str, float] = {}
        variance_flags: dict[str, bool] = {}
        psi_scores: dict[str, float] = {}
        mean_shift_normalized: list[float] = []
        variance_shift_normalized: list[float] = []
        psi_shift_normalized: list[float] = []
        for key in row.keys():
            r_vals = [float(x.get(key, 0.0)) for x in recent]
            b_vals = [float(x.get(key, 0.0)) for x in baseline_rows]
            baseline_stats = self._baseline_stats.get(key, {})
            b_mu = float(baseline_stats.get("mean", 0.0))
            b_sd = float(max(float(baseline_stats.get("std", 1.0)), 1e-6))
            r_mu, r_sd = self._safe_stats(r_vals)
            z = abs(r_mu - b_mu) / b_sd
            ratio = r_sd / b_sd if b_sd > 1e-9 else 1.0
            shifts[key] = round(float(z), 6)
            variance_ratios[key] = round(float(ratio), 6)
            variance_flags[key] = bool(ratio > 2.0 or ratio < 0.4)
            mean_shift_normalized.append(min(z / 3.0, 1.0))
            variance_shift_normalized.append(self._variance_shift_score(ratio))
            psi_value: float | None = None
            if len(b_vals) >= 30:
                psi_value = self._compute_psi(b_vals, r_vals, n_bins=10)
            else:
                psi_edges = baseline_stats.get("psi_bucket_edges", [])
                psi_expected = baseline_stats.get("psi_expected_pct", [])
                if (
                    isinstance(psi_edges, list)
                    and isinstance(psi_expected, list)
                    and len(psi_edges) >= 2
                    and len(psi_expected) == (len(psi_edges) - 1)
                ):
                    psi_value = self._compute_psi_from_profile(r_vals, psi_edges, psi_expected)
            if psi_value is not None:
                psi_scores[key] = round(float(psi_value), 6)
                psi_shift_normalized.append(min(max(float(psi_value), 0.0) / 0.20, 1.0))

        mean_shift_score = float(sum(mean_shift_normalized) / max(len(mean_shift_normalized), 1))
        variance_shift_score = float(sum(variance_shift_normalized) / max(len(variance_shift_normalized), 1))
        psi_drift_score = float(sum(psi_shift_normalized) / len(psi_shift_normalized)) if psi_shift_normalized else 0.0
        use_psi = len(psi_shift_normalized) == len(row)
        if use_psi:
            drift_score = float((mean_shift_score * 0.35) + (variance_shift_score * 0.35) + (psi_drift_score * 0.30))
        else:
            drift_score = float((mean_shift_score + variance_shift_score) / 2.0)
        if drift_score >= 0.72:
            level = "HIGH"
            conf_mul = 0.72
            learn_mul = 0.60
        elif drift_score >= 0.40:
            level = "MEDIUM"
            conf_mul = 0.86
            learn_mul = 0.80
        else:
            level = "LOW"
            conf_mul = 1.0
            learn_mul = 1.0

        return {
            "drift_level": level,
            "drift_score": round(drift_score, 6),
            "mean_shift_score": round(mean_shift_score, 6),
            "variance_shift_score": round(variance_shift_score, 6),
            "psi_drift_score": round(psi_drift_score, 6),
            "confidence_multiplier": conf_mul,
            "learning_multiplier": learn_mul,
            "status": "ok",
            "feature_shifts": shifts,
            "variance_ratios": variance_ratios,
            "variance_drift_flags": variance_flags,
            "psi_scores": psi_scores,
        }
