from __future__ import annotations

from collections import deque
import json
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
        self._baseline_stats: dict[str, dict[str, float]] = {}
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

    def _load_baseline_stats(self) -> None:
        if not self._state_path or (not self._state_path.exists()):
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            features_payload = payload.get("features", {}) if isinstance(payload, dict) else {}
            if not isinstance(features_payload, dict):
                return
            restored: dict[str, dict[str, float]] = {}
            for key in self._FEATURE_KEYS:
                row = features_payload.get(key, {})
                if not isinstance(row, dict):
                    continue
                mu = float(row.get("mean", 0.0))
                sd = float(max(float(row.get("std", 1.0)), 1e-6))
                restored[key] = {"mean": mu, "std": sd}
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

    def _build_baseline_stats(self, baseline_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for key in self._FEATURE_KEYS:
            b_vals = [float(x.get(key, 0.0)) for x in baseline_rows]
            b_mu, b_sd = self._safe_stats(b_vals)
            stats[key] = {
                "mean": float(b_mu),
                "std": float(max(b_sd, 1e-6)),
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
                "confidence_multiplier": 1.0,
                "learning_multiplier": 1.0,
                "status": "warming_up",
                "feature_shifts": {},
                "variance_ratios": {},
            }

        recent = values[-self.recent_window :] if has_recent_window else values
        if not recent:
            return {
                "drift_level": "LOW",
                "drift_score": 0.0,
                "confidence_multiplier": 1.0,
                "learning_multiplier": 1.0,
                "status": "warming_up",
                "feature_shifts": {},
                "variance_ratios": {},
            }
        shifts: dict[str, float] = {}
        variance_ratios: dict[str, float] = {}
        variance_flags: dict[str, bool] = {}
        mean_shift_normalized: list[float] = []
        variance_shift_normalized: list[float] = []
        for key in row.keys():
            r_vals = [float(x.get(key, 0.0)) for x in recent]
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

        mean_shift_score = float(sum(mean_shift_normalized) / max(len(mean_shift_normalized), 1))
        variance_shift_score = float(sum(variance_shift_normalized) / max(len(variance_shift_normalized), 1))
        drift_score = float((mean_shift_score + variance_shift_score) / 2.0)
        if drift_score >= 0.66:
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
            "confidence_multiplier": conf_mul,
            "learning_multiplier": learn_mul,
            "status": "ok",
            "feature_shifts": shifts,
            "variance_ratios": variance_ratios,
            "variance_drift_flags": variance_flags,
        }
