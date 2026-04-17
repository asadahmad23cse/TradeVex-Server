from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from btc_intelligence.config import settings


logger = logging.getLogger(__name__)


REGIMES = ("bullish", "bearish", "sideways")
FACTOR_KEYS = ("regime", "momentum", "flow", "cost", "volatility")


@dataclass
class AdaptiveLearningConfig:
    learning_rate: float = 0.01
    min_trades_required: int = 20
    update_frequency: int = 15
    max_change_per_update: float = 0.05
    smoothing_alpha: float = 0.20
    train_window: int = 60
    test_window: int = 20
    cluster_loss_threshold: int = 3


class AdaptiveLearningEngine:
    """
    Slow adaptive layer with explicit anti-overfitting controls:
    - regime-isolated learning
    - update on batches only
    - walk-forward validation before apply
    - rollback on degradation
    """

    def __init__(self, state_path: str, config: AdaptiveLearningConfig | None = None) -> None:
        self.config = config or AdaptiveLearningConfig()
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.base_weights: dict[str, dict[str, float]] = {
            "bullish": {"regime": 0.22, "momentum": 0.20, "flow": 0.30, "cost": 0.16, "volatility": 0.12},
            "bearish": {"regime": 0.22, "momentum": 0.20, "flow": 0.30, "cost": 0.16, "volatility": 0.12},
            "sideways": {"regime": 0.18, "momentum": 0.20, "flow": 0.26, "cost": 0.20, "volatility": 0.16},
        }
        self.weights: dict[str, dict[str, float]] = {k: dict(v) for k, v in self.base_weights.items()}
        self.trade_history: dict[str, list[dict[str, Any]]] = {k: [] for k in REGIMES}
        self.loss_clusters: dict[str, float] = {}
        self.condition_counts: dict[str, int] = {}
        self.last_update_trade_count: dict[str, int] = {k: 0 for k in REGIMES}
        self.last_validation: dict[str, dict[str, Any]] = {}
        self._enet_cache: dict[str, Any] | None = None
        self._enet_mtime: float = 0.0
        self._calibration_brier_refit_requested: bool = False
        self._calibration_eval_cache: dict[str, tuple[float, int]] = {}  # regime -> (brier_oos, trade_count)
        self._platt_iso_cache: dict[str, tuple[Any, Any, int]] = {}  # regime -> (platt, iso, trade_count)
        self._platt_iso_dir: Path = self.state_path.parent / "platt_iso_models"
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            w = payload.get("weights", {})
            for regime in REGIMES:
                if isinstance(w.get(regime), dict):
                    self.weights[regime] = self._normalize_weights(w[regime], fallback=self.base_weights[regime])
            hist = payload.get("trade_history", {})
            for regime in REGIMES:
                rows = hist.get(regime, [])
                if isinstance(rows, list):
                    self.trade_history[regime] = rows[-600:]
            clusters = payload.get("loss_clusters", {})
            if isinstance(clusters, dict):
                self.loss_clusters = {str(k): float(v) for k, v in clusters.items()}
            condition_counts = payload.get("condition_counts", {})
            if isinstance(condition_counts, dict):
                self.condition_counts = {str(k): int(v) for k, v in condition_counts.items()}
            luc = payload.get("last_update_trade_count", {})
            if isinstance(luc, dict):
                for regime in REGIMES:
                    self.last_update_trade_count[regime] = int(luc.get(regime, 0))
            lv = payload.get("last_validation", {})
            if isinstance(lv, dict):
                self.last_validation = lv
            eval_cache = payload.get("calibration_eval_cache", {})
            if isinstance(eval_cache, dict):
                for regime in REGIMES:
                    row = eval_cache.get(regime)
                    if isinstance(row, dict):
                        self._calibration_eval_cache[regime] = (
                            float(row.get("brier_oos", 0.25)),
                            int(row.get("trade_count", 0)),
                        )
            self._load_platt_iso_models()
        except Exception as exc:
            logger.warning("AdaptiveLearningEngine load failed: %s", exc)

    def _save(self) -> None:
        payload = {
            "weights": self.weights,
            "trade_history": {k: v[-600:] for k, v in self.trade_history.items()},
            "loss_clusters": {k: round(float(v), 4) for k, v in self.loss_clusters.items()},
            "condition_counts": self.condition_counts,
            "last_update_trade_count": self.last_update_trade_count,
            "last_validation": self.last_validation,
            "calibration_eval_cache": {
                reg: {"brier_oos": round(float(v[0]), 6), "trade_count": int(v[1])}
                for reg, v in self._calibration_eval_cache.items()
            },
        }
        try:
            self.state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("AdaptiveLearningEngine save failed: %s", exc)

    @staticmethod
    def _normalize_regime_bucket(raw_regime: str) -> str:
        r = str(raw_regime or "").lower()
        if any(x in r for x in ("bear", "panic", "breakout_down")):
            return "bearish"
        if any(x in r for x in ("bull", "breakout_up")):
            return "bullish"
        return "sideways"

    def _normalize_weights(self, weights: dict[str, Any], fallback: dict[str, float]) -> dict[str, float]:
        out = {}
        for key in FACTOR_KEYS:
            try:
                out[key] = float(weights.get(key, fallback[key]))
            except Exception:
                out[key] = float(fallback[key])
            out[key] = min(max(out[key], 0.05), 0.60)
        total = sum(out.values())
        if total <= 0:
            return dict(fallback)
        return {k: v / total for k, v in out.items()}

    @staticmethod
    def _condition_key(regime: str, volatility_state: str, flow_state: str, direction: str) -> str:
        return f"{regime}|{str(volatility_state).upper()}|{str(flow_state).upper()}|{str(direction).upper()}"

    @staticmethod
    def _score_from_weights(weights: dict[str, float], factor_values: dict[str, float]) -> float:
        score = 0.0
        for key in FACTOR_KEYS:
            score += float(weights.get(key, 0.0)) * float(factor_values.get(key, 0.0))
        return float(max(-1.0, min(1.0, score)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        z = max(-30.0, min(30.0, x))
        return 1.0 / (1.0 + math.exp(-z))

    def _evaluate_weights(self, rows: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, float]:
        if not rows:
            return {"metric": 0.0, "win_rate": 0.0, "sharpe": 0.0, "brier": 1.0}
        outcomes: list[float] = []
        probs: list[float] = []
        rets: list[float] = []
        for row in rows:
            factors = row.get("factors", {})
            direction = str(row.get("direction", "HOLD")).upper()
            sign = 1.0 if direction == "LONG" else -1.0 if direction == "SHORT" else 0.0
            raw_score = self._score_from_weights(weights, factors) * sign
            p = self._sigmoid(raw_score * 3.0)
            probs.append(p)
            win = 1.0 if float(row.get("pnl_pct", 0.0)) > 0 else 0.0
            outcomes.append(win)
            rets.append(float(row.get("pnl_pct", 0.0)) / 100.0)

        win_rate = sum(outcomes) / max(len(outcomes), 1)
        brier = sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / max(len(outcomes), 1)
        if len(rets) > 1:
            mu = mean(rets)
            sd = pstdev(rets)
            sharpe = (mu / sd) * math.sqrt(len(rets)) if sd > 1e-9 else 0.0
        else:
            sharpe = 0.0
        metric = (0.65 * win_rate) + (0.25 * max(min(sharpe, 3.0), -3.0) / 3.0) - (0.10 * brier)
        return {"metric": float(metric), "win_rate": float(win_rate), "sharpe": float(sharpe), "brier": float(brier)}

    def _propose_weights(self, rows: list[dict[str, Any]], current: dict[str, float]) -> dict[str, float]:
        if not rows:
            return dict(current)
        signals = {k: 0.0 for k in FACTOR_KEYS}
        for row in rows:
            pnl = float(row.get("pnl_pct", 0.0))
            direction = str(row.get("direction", "HOLD")).upper()
            sign_dir = 1.0 if direction == "LONG" else -1.0 if direction == "SHORT" else 0.0
            # Positive signal when factor supported profitable directional outcome.
            outcome_signal = (1.0 if pnl > 0 else -1.0) * sign_dir
            factors = row.get("factors", {})
            for key in FACTOR_KEYS:
                signals[key] += float(factors.get(key, 0.0)) * outcome_signal

        n = max(len(rows), 1)
        proposed = {}
        for key in FACTOR_KEYS:
            avg_signal = signals[key] / n
            raw_delta = self.config.learning_rate * avg_signal
            max_delta = self.config.max_change_per_update * current[key]
            delta = max(-max_delta, min(max_delta, raw_delta))
            stepped = current[key] + delta
            # Smoothing: avoid abrupt changes even if validated.
            smoothed = ((1.0 - self.config.smoothing_alpha) * current[key]) + (self.config.smoothing_alpha * stepped)
            proposed[key] = min(max(smoothed, 0.05), 0.60)
        return self._normalize_weights(proposed, fallback=current)

    def _walk_forward_validate_and_apply(self, regime: str) -> None:
        rows = self.trade_history.get(regime, [])
        if len(rows) < self.config.min_trades_required:
            return

        since_last = len(rows) - int(self.last_update_trade_count.get(regime, 0))
        if since_last < self.config.update_frequency:
            return

        test_n = min(self.config.test_window, max(10, len(rows) // 4))
        train_n = min(self.config.train_window, len(rows) - test_n)
        if train_n < 20 or test_n < 10:
            return

        train_rows = rows[-(train_n + test_n) : -test_n]
        test_rows = rows[-test_n:]
        current_weights = self.weights[regime]
        candidate_weights = self._propose_weights(train_rows, current=current_weights)

        baseline = self._evaluate_weights(test_rows, current_weights)
        candidate = self._evaluate_weights(test_rows, candidate_weights)
        # Early phase (< 100 trades): relax improvement gates to allow faster learning
        # Mature phase (100+): standard strict gates to prevent overfitting
        n_total = len(rows)
        if n_total < 100:
            metric_gate = 0.005  # halved improvement threshold
            wr_gate = -0.04     # allow 4% win rate drop
            brier_gate = 0.02   # allow 2% brier degradation
        else:
            metric_gate = 0.01
            wr_gate = -0.02
            brier_gate = 0.01
        improved = (
            candidate["metric"] > (baseline["metric"] + metric_gate)
            and candidate["win_rate"] >= (baseline["win_rate"] + wr_gate)
            and candidate["brier"] <= (baseline["brier"] + brier_gate)
        )

        self.last_validation[regime] = {
            "baseline": baseline,
            "candidate": candidate,
            "improved": bool(improved),
            "samples": {"train": len(train_rows), "test": len(test_rows)},
        }

        if improved:
            self.weights[regime] = candidate_weights
            logger.info("Adaptive weights updated for %s regime via walk-forward validation", regime)
        else:
            logger.info("Adaptive update rejected for %s regime (no robust improvement)", regime)

        self.last_update_trade_count[regime] = len(rows)
        self._save()

    def record_trade(
        self,
        *,
        regime: str,
        volatility_state: str,
        flow_state: str,
        direction: str,
        pnl_pct: float,
        raw_prob: float,
        calibrated_prob: float,
        factor_values: dict[str, float],
        timestamp_utc: str,
    ) -> None:
        regime_bucket = self._normalize_regime_bucket(regime)
        row = {
            "timestamp": str(timestamp_utc),
            "regime": regime_bucket,
            "volatility_state": str(volatility_state).upper(),
            "flow_state": str(flow_state).upper(),
            "direction": str(direction).upper(),
            "pnl_pct": float(pnl_pct),
            "raw_prob": float(max(0.0, min(1.0, raw_prob))),
            "calibrated_prob": float(max(0.0, min(1.0, calibrated_prob))),
            "factors": {k: float(max(-1.0, min(1.0, factor_values.get(k, 0.0)))) for k in FACTOR_KEYS},
        }
        self.trade_history[regime_bucket].append(row)
        self.trade_history[regime_bucket] = self.trade_history[regime_bucket][-600:]

        cond_key = self._condition_key(regime_bucket, volatility_state, flow_state, direction)
        self.condition_counts[cond_key] = int(self.condition_counts.get(cond_key, 0)) + 1
        if float(pnl_pct) < 0:
            self.loss_clusters[cond_key] = float(self.loss_clusters.get(cond_key, 0.0)) + 1.0
        else:
            # Gradual decay on wins (0.25 per win instead of 1) to avoid
            # prematurely unblocking genuinely bad patterns.
            if cond_key in self.loss_clusters and self.loss_clusters[cond_key] > 0:
                self.loss_clusters[cond_key] = max(0.0, float(self.loss_clusters[cond_key]) - 0.25)

        self._walk_forward_validate_and_apply(regime_bucket)
        self._save()

    def _reliability_stats(self, regime: str) -> dict[str, Any]:
        rows = self.trade_history.get(regime, [])
        if len(rows) < 15:
            return {"brier": 0.25, "ece": 0.12, "bins": []}

        bins = [{"n": 0, "sum_prob": 0.0, "sum_outcome": 0.0} for _ in range(10)]
        sq_err = 0.0
        for row in rows:
            p = float(max(0.0, min(1.0, row.get("raw_prob", 0.5))))
            y = 1.0 if float(row.get("pnl_pct", 0.0)) > 0 else 0.0
            idx = min(9, max(0, int(p * 10)))
            bins[idx]["n"] += 1
            bins[idx]["sum_prob"] += p
            bins[idx]["sum_outcome"] += y
            sq_err += (p - y) ** 2

        total = len(rows)
        ece = 0.0
        bins_out = []
        for i, b in enumerate(bins):
            if b["n"] == 0:
                continue
            avg_p = b["sum_prob"] / b["n"]
            avg_y = b["sum_outcome"] / b["n"]
            gap = abs(avg_p - avg_y)
            ece += (b["n"] / total) * gap
            bins_out.append({"bin": i, "n": b["n"], "avg_prob": avg_p, "accuracy": avg_y, "gap": gap})

        brier = sq_err / max(total, 1)
        return {"brier": float(brier), "ece": float(ece), "bins": bins_out}

    def worst_regime_brier(self) -> float:
        """Max Brier across regime buckets (raw_prob vs win/loss), rolling window per regime."""
        mx = 0.0
        win = int(getattr(settings, "brier_watchdog_rolling_trades", 0) or 0)
        for reg in REGIMES:
            rows = self._sorted_rows(reg, self.trade_history)
            if win > 0:
                rows = rows[-win:]
            if not rows:
                continue
            sq_err = 0.0
            for row in rows:
                p = float(max(0.0, min(1.0, row.get("raw_prob", 0.5))))
                y = 1.0 if float(row.get("pnl_pct", 0.0)) > 0 else 0.0
                sq_err += (p - y) ** 2
            mx = max(mx, sq_err / len(rows))
        if mx > 0.25:
            self._calibration_brier_refit_requested = True
        else:
            self._calibration_brier_refit_requested = False
        return float(mx)

    def _get_enet_state(self) -> dict[str, Any] | None:
        if not bool(settings.calibration_use_elasticnet_meta):
            return None
        path = Path(str(settings.elasticnet_calibrator_path))
        if not path.is_file():
            return None
        try:
            mtime = float(path.stat().st_mtime)
            if self._enet_cache is not None and mtime == self._enet_mtime:
                return self._enet_cache
            self._enet_cache = json.loads(path.read_text(encoding="utf-8"))
            self._enet_mtime = mtime
            return self._enet_cache
        except Exception as exc:
            logger.debug("ElasticNet calibrator load failed: %s", exc)
            return None

    def _enet_pre_prob(self, raw_prob: float, factor_values: dict[str, float] | None, enet: dict[str, Any] | None) -> float:
        if not enet:
            return float(max(0.0, min(1.0, raw_prob)))
        order = list(enet.get("feature_order", ["raw_prob"] + list(FACTOR_KEYS)))
        mean = np.asarray(enet.get("mean", [0.0] * len(order)), dtype=float)
        scale = np.asarray(enet.get("scale", [1.0] * len(order)), dtype=float)
        coef = np.asarray(enet.get("coef", [0.0] * len(order)), dtype=float)
        intercept = float(enet.get("intercept", 0.0))
        fv = factor_values or {}
        vec = [float(max(0.0, min(1.0, raw_prob)))] + [float(max(-1.0, min(1.0, fv.get(k, 0.0)))) for k in FACTOR_KEYS]
        if len(vec) != len(order):
            return float(max(0.0, min(1.0, raw_prob)))
        x = (np.asarray(vec, dtype=float) - mean) / np.clip(scale, 1e-9, None)
        z = float(x @ coef + intercept)
        return float(max(0.0, min(1.0, self._sigmoid(z))))

    @staticmethod
    def _sorted_rows(regime_bucket: str, trade_history: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        rows = list(trade_history.get(regime_bucket, []))
        rows.sort(key=lambda r: str(r.get("timestamp", "")))
        return rows

    def _histogram_calibrate(self, regime_bucket: str, p: float) -> dict[str, Any]:
        stats = self._reliability_stats(regime_bucket)
        bins = stats.get("bins", [])
        observed = None
        for row in bins:
            lo = float(row["bin"]) / 10.0
            hi = lo + 0.1
            if p >= lo and (p < hi or row["bin"] == 9):
                observed = float(row["accuracy"])
                break
        if observed is None:
            calibrated = p
        else:
            calibrated = (0.6 * p) + (0.4 * observed)
        calibrated = float(max(0.0, min(1.0, calibrated)))
        return {
            "raw_prob": p,
            "calibrated_prob": calibrated,
            "brier_score": round(float(stats.get("brier", 0.25)), 6),
            "calibration_error": round(float(stats.get("ece", 0.12)), 6),
            "reliability_curve": stats.get("bins", []),
            "regime": regime_bucket,
        }

    def _fit_platt_isotonic(self, meta_inputs: list[float], labels: list[float]):
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return None, None

        X = np.asarray(meta_inputs, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=float)
        if X.shape[0] < 10 or len(np.unique(y)) < 2:
            return None, None

        try:
            platt = LogisticRegression(
                C=1e6,
                solver="lbfgs",
                max_iter=500,
                random_state=42,
            )
            platt.fit(X, y)
            z = platt.predict_proba(X)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(z, y)
            return platt, iso
        except Exception as exc:
            logger.debug("Platt/isotonic fit failed: %s", exc)
            return None, None

    def _platt_iso_path(self, regime_bucket: str) -> Path:
        return self._platt_iso_dir / f"{regime_bucket}.pkl"

    def _save_platt_iso_model(self, regime_bucket: str, platt: Any, iso: Any, trade_count: int) -> None:
        path = self._platt_iso_path(regime_bucket)
        payload = {"trade_count": int(trade_count), "platt": platt, "iso": iso}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(payload, f)
        except Exception as exc:
            logger.debug("Platt/isotonic model save failed for %s: %s", regime_bucket, exc)

    def _load_platt_iso_models(self) -> None:
        try:
            if not self._platt_iso_dir.exists():
                return
            for regime_bucket in REGIMES:
                path = self._platt_iso_path(regime_bucket)
                if not path.exists():
                    continue
                with path.open("rb") as f:
                    payload = pickle.load(f)
                if not isinstance(payload, dict):
                    continue
                platt = payload.get("platt")
                iso = payload.get("iso")
                trade_count = int(payload.get("trade_count", 0))
                if platt is None or iso is None or trade_count <= 0:
                    continue
                self._platt_iso_cache[regime_bucket] = (platt, iso, trade_count)
        except Exception as exc:
            logger.debug("Platt/isotonic model load failed: %s", exc)

    def _apply_platt_iso(self, p0: float, platt: Any, iso: Any) -> float:
        if platt is None or iso is None:
            return float(max(0.0, min(1.0, p0)))
        try:
            p1 = float(platt.predict_proba(np.asarray([[p0]], dtype=float))[0, 1])
            p2 = float(iso.predict([p1])[0])
            return float(max(0.0, min(1.0, p2)))
        except Exception:
            return float(max(0.0, min(1.0, p0)))

    def calibrate_probability(
        self, regime: str, raw_prob: float, factor_values: dict[str, float] | None = None
    ) -> dict[str, Any]:
        regime_bucket = self._normalize_regime_bucket(regime)
        p = float(max(0.0, min(1.0, raw_prob)))
        stats = self._reliability_stats(regime_bucket)
        rows = self._sorted_rows(regime_bucket, self.trade_history)
        min_tr_raw = int(getattr(settings, "calibration_min_trades", 0) or 0)
        if min_tr_raw == 0 or min_tr_raw > 50:
            min_tr = 30
        else:
            min_tr = min_tr_raw
        force_recal = bool(self._calibration_brier_refit_requested)
        min_tr_effective = min(min_tr, 15) if force_recal else min_tr

        if len(rows) < min_tr_effective:
            return self._histogram_calibrate(regime_bucket, p)

        enet = self._get_enet_state()
        frac = float(settings.calibration_train_frac)
        split = max(10, int(frac * len(rows)))
        split = min(split, len(rows) - max(5, min_tr_effective // 3))
        if split >= len(rows) - 2:
            split = max(10, len(rows) - 5)

        train_rows = rows[:split]
        val_rows = rows[split:]

        def _meta_inputs(batch: list[dict[str, Any]]) -> list[float]:
            return [
                self._enet_pre_prob(float(r.get("raw_prob", 0.5)), r.get("factors"), enet)
                for r in batch
            ]

        def _labels(batch: list[dict[str, Any]]) -> list[float]:
            return [1.0 if float(r.get("pnl_pct", 0.0)) > 0 else 0.0 for r in batch]

        cache_key = regime_bucket
        cached = self._platt_iso_cache.get(cache_key)
        cached_full_ok = bool(cached and cached[2] == len(rows))
        eval_cached = self._calibration_eval_cache.get(cache_key)
        brier_oos = float(stats.get("brier", 0.25))
        if eval_cached and eval_cached[1] == len(rows):
            brier_oos = float(eval_cached[0])
        elif cached_full_ok:
            # Model already loaded (potentially from disk checkpoint); avoid re-fitting
            # just for telemetry on every call.
            self._calibration_eval_cache[cache_key] = (float(brier_oos), len(rows))
            self._save()
        else:
            mi_train = _meta_inputs(train_rows)
            y_train = _labels(train_rows)
            platt_tr, iso_tr = self._fit_platt_isotonic(mi_train, y_train)
            if platt_tr is not None and iso_tr is not None and len(val_rows) >= 5:
                y_val = np.asarray(_labels(val_rows), dtype=float)
                preds = np.asarray(
                    [self._apply_platt_iso(x, platt_tr, iso_tr) for x in _meta_inputs(val_rows)],
                    dtype=float,
                )
                brier_oos = float(np.mean((preds - y_val) ** 2))
            self._calibration_eval_cache[cache_key] = (float(brier_oos), len(rows))
            self._save()

        mi_all = _meta_inputs(rows)
        y_all = _labels(rows)
        # Use cached Platt/Isotonic models if trade count unchanged
        if cached_full_ok:
            platt_full, iso_full = cached[0], cached[1]
        else:
            platt_full, iso_full = self._fit_platt_isotonic(mi_all, y_all)
            if platt_full is not None and iso_full is not None:
                self._platt_iso_cache[cache_key] = (platt_full, iso_full, len(rows))
                self._save_platt_iso_model(cache_key, platt_full, iso_full, len(rows))
        if platt_full is None or iso_full is None:
            out = self._histogram_calibrate(regime_bucket, p)
            out["brier_score"] = round(brier_oos, 6)
            return out

        p0_live = self._enet_pre_prob(p, factor_values, enet)
        calibrated = self._apply_platt_iso(p0_live, platt_full, iso_full)

        ece = float(stats.get("ece", 0.12))
        return {
            "raw_prob": p,
            "calibrated_prob": calibrated,
            "brier_score": round(brier_oos, 6),
            "calibration_error": round(ece, 6),
            "reliability_curve": stats.get("bins", []),
            "regime": regime_bucket,
        }

    def edge_decay_status(self, regime: str) -> dict[str, Any]:
        regime_bucket = self._normalize_regime_bucket(regime)
        rows = self.trade_history.get(regime_bucket, [])
        if len(rows) < 30:
            return {
                "decay_detected": False,
                "reduce_risk_factor": 1.0,
                "warning": "",
                "rolling_win_rate": 0.0,
                "rolling_sharpe": 0.0,
            }

        recent = rows[-20:]
        base = rows[-60:-20] if len(rows) >= 60 else rows[:-20]

        def _win_rate(block: list[dict[str, Any]]) -> float:
            if not block:
                return 0.0
            wins = sum(1 for r in block if float(r.get("pnl_pct", 0.0)) > 0)
            return wins / len(block)

        def _sharpe(block: list[dict[str, Any]]) -> float:
            vals = [float(r.get("pnl_pct", 0.0)) / 100.0 for r in block]
            if len(vals) < 2:
                return 0.0
            sd = pstdev(vals)
            if sd <= 1e-9:
                return 0.0
            return (mean(vals) / sd) * math.sqrt(len(vals))

        recent_wr = _win_rate(recent)
        base_wr = _win_rate(base)
        recent_sh = _sharpe(recent)
        base_sh = _sharpe(base)

        decay = (recent_wr < (base_wr - 0.12)) or (recent_sh < (base_sh - 0.60))
        reduce = 0.5 if decay else 1.0
        warn = (
            f"Edge decay in {regime_bucket}: rolling WR {recent_wr:.2f} vs {base_wr:.2f}, "
            f"Sharpe {recent_sh:.2f} vs {base_sh:.2f}"
            if decay
            else ""
        )
        return {
            "decay_detected": bool(decay),
            "reduce_risk_factor": float(reduce),
            "warning": warn,
            "rolling_win_rate": round(recent_wr, 4),
            "rolling_sharpe": round(recent_sh, 4),
        }

    def is_condition_blocked(
        self,
        *,
        regime: str,
        volatility_state: str,
        flow_state: str,
        direction: str,
    ) -> tuple[bool, str]:
        regime_bucket = self._normalize_regime_bucket(regime)
        key = self._condition_key(regime_bucket, volatility_state, flow_state, direction)
        losses = float(self.loss_clusters.get(key, 0.0))
        observations = int(self.condition_counts.get(key, 0))
        min_obs = max(int(self.config.cluster_loss_threshold) + 2, 5)
        if losses >= self.config.cluster_loss_threshold and observations >= min_obs:
            return True, f"Condition cluster blocked ({losses:.2f}/{observations} losses): {key}"
        return False, ""

    def meta_decision(
        self,
        *,
        regime: str,
        volatility_state: str,
        flow_state: str,
        base_decision: str,
        base_confidence: float,
        factor_values: dict[str, float],
        raw_prob: float,
        blockers: list[str] | None = None,
    ) -> dict[str, Any]:
        regime_bucket = self._normalize_regime_bucket(regime)
        decision = str(base_decision or "HOLD").upper()
        base_conf = float(max(0.0, min(100.0, base_confidence)))
        blockers_list = list(blockers or [])

        calibration = self.calibrate_probability(regime_bucket, raw_prob, factor_values=factor_values)
        calibrated_prob = float(calibration.get("calibrated_prob", raw_prob))
        weights = self.weights.get(regime_bucket, self.base_weights[regime_bucket])

        sign = 1.0 if decision == "LONG" else -1.0 if decision == "SHORT" else 0.0
        adaptive_score = self._score_from_weights(weights, factor_values) * sign
        blocked, block_reason = self.is_condition_blocked(
            regime=regime_bucket,
            volatility_state=volatility_state,
            flow_state=flow_state,
            direction=decision,
        )
        edge = self.edge_decay_status(regime_bucket)

        final_decision = decision
        reasons: list[str] = []
        if decision == "HOLD":
            reasons.append("Base decision is HOLD")
        if blockers_list:
            reasons.extend(blockers_list[:2])
            final_decision = "HOLD"
        if blocked:
            reasons.append(block_reason)
            final_decision = "HOLD"
        if edge.get("decay_detected", False) and decision in {"LONG", "SHORT"}:
            reasons.append("Edge decay active: stricter trade gate enabled")
            if calibrated_prob < 0.60 or adaptive_score < 0.10:
                final_decision = "HOLD"
        _platt = getattr(self, "_platt_full", None)
        _is_fitted = bool(_platt and getattr(_platt, "fitted", False))
        _prob_min = 0.53 if _is_fitted else 0.38
        if decision in {"LONG", "SHORT"} and calibrated_prob < _prob_min:
            reasons.append(f"Calibrated probability too low ({calibrated_prob:.2f})")
            final_decision = "HOLD"
        if decision in {"LONG", "SHORT"} and adaptive_score < 0.02:
            reasons.append(f"Adaptive score too weak ({adaptive_score:.3f})")
            final_decision = "HOLD"

        conf_adjust = base_conf * (0.85 + (0.30 * calibrated_prob))
        conf_adjust *= float(edge.get("reduce_risk_factor", 1.0))
        adjusted_confidence = max(0.0, min(100.0, conf_adjust))

        if final_decision == "HOLD":
            adjusted_confidence = min(adjusted_confidence, 49.0)

        return {
            "regime_bucket": regime_bucket,
            "base_decision": decision,
            "final_decision": final_decision,
            "base_confidence": round(base_conf, 2),
            "adjusted_confidence": round(adjusted_confidence, 2),
            "adaptive_score": round(float(adaptive_score), 6),
            "calibration": calibration,
            "weights": {k: round(float(v), 6) for k, v in weights.items()},
            "edge_decay": edge,
            "stability_priority": True,
            "blocked_by_filter": bool(blocked),
            "filter_reason": block_reason,
            "reasons": reasons,
            "validation": self.last_validation.get(regime_bucket, {}),
        }
