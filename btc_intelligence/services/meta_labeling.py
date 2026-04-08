from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REGIME_CONFIDENCE_GATES = {
    "TRENDING": 55.0,
    "BULLISH": 55.0,
    "BEARISH": 58.0,
    "RANGE": 68.0,
    "SQUEEZE": 72.0,
    "VOLATILE": 75.0,
    "SIDEWAYS": 68.0,
    "DEFAULT": 60.0,
}

RF_FEATURE_ORDER = (
    "hour_sin",
    "hour_cos",
    "hmm_mean_reverting",
    "hmm_trending",
    "hmm_liquidity_cascade",
    "vix_norm",
    "spread_frac",
)


def get_dynamic_confidence_gate(regime: str, utc_hour: int | None = None) -> float:
    regime_upper = str(regime or "").upper()
    base_gate: float | None = None
    for key, gate in REGIME_CONFIDENCE_GATES.items():
        if key == "DEFAULT":
            continue
        if key in regime_upper:
            base_gate = float(gate)
            break
    if base_gate is None:
        base_gate = float(REGIME_CONFIDENCE_GATES["DEFAULT"])
    if utc_hour is not None:
        h = int(utc_hour) % 24
        if 13 <= h <= 16:
            base_gate -= 5.0
        elif 7 <= h <= 10:
            base_gate -= 3.0
        elif 0 <= h <= 6:
            base_gate += 2.0
        base_gate = max(50.0, base_gate)
    logger.debug(
        "confidence gate: regime=%s hour=%d threshold=%.1f",
        regime,
        int(utc_hour) if utc_hour is not None else -1,
        float(base_gate),
    )
    return float(base_gate)


def build_rf_feature_vector(meta_features: dict[str, Any] | None) -> np.ndarray:
    """Fixed-order feature vector for the meta RF (single row)."""
    m = meta_features or {}
    hour_sin = float(m.get("hour_sin", 0.0))
    hour_cos = float(m.get("hour_cos", 1.0))
    mr = float(m.get("hmm_mean_reverting", m.get("mean_reverting", 1.0 / 3.0)))
    tr = float(m.get("hmm_trending", m.get("trending", 1.0 / 3.0)))
    lc = float(m.get("hmm_liquidity_cascade", m.get("liquidity_cascade", 1.0 / 3.0)))
    vix = float(m.get("vix_level", m.get("vix", 20.0)))
    vix_norm = float(np.clip(vix / 40.0, 0.0, 3.0))
    spread_pct = float(m.get("spread_pct", 0.0))
    spread_frac = float(np.clip(spread_pct / 100.0, 0.0, 0.05))
    vec = np.array(
        [[hour_sin, hour_cos, mr, tr, lc, vix_norm, spread_frac]],
        dtype=float,
    )
    return vec


def _try_import_sklearn_rf():
    try:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier, True
    except Exception as exc:
        logger.debug("sklearn RandomForest unavailable: %s", exc)
        return None, False


REGIME_CONFIDENCE_GATES = {
    "TRENDING": 55.0,
    "BULLISH": 55.0,
    "BEARISH": 58.0,
    "RANGE": 68.0,
    "SQUEEZE": 72.0,
    "VOLATILE": 75.0,
    "SIDEWAYS": 68.0,
    "DEFAULT": 60.0,
}


def get_dynamic_confidence_gate(regime: str) -> float:
    regime_upper = str(regime or "").upper()
    for key, gate in REGIME_CONFIDENCE_GATES.items():
        if key == "DEFAULT":
            continue
        if key in regime_upper:
            return float(gate)
    return float(REGIME_CONFIDENCE_GATES["DEFAULT"])


@dataclass
class MetaLabelingConfig:
    min_confidence: float = 55.0
    min_probability: float = 0.56
    max_calibration_error: float = 0.22
    # Random forest false-positive (bad trade) veto
    rf_fp_threshold: float = 0.55
    rf_min_train_samples: int = 60
    rf_refit_every: int = 15
    rf_max_buffer: int = 2500
    rf_training_path: str = "btc_intelligence/logs/meta_label_rf_training.jsonl"
    rf_n_estimators: int = 100
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 4
    rf_random_state: int = 42


class MetaLabelingEngine:
    """
    Second-stage trade filter:
    keep only high-conviction, calibrated, execution-worthy signals.
    Optional Random Forest vetoes likely false positives from trade metadata.
    """

    def __init__(self, config: MetaLabelingConfig | None = None) -> None:
        self.config = config or MetaLabelingConfig()
        self._rf_model: Any = None
        self._sklearn_ok = False
        self._samples_since_fit = 0
        _, ok = _try_import_sklearn_rf()
        self._sklearn_ok = bool(ok)
        self._load_training_and_fit()

    def _training_path(self) -> Path:
        return Path(str(self.config.rf_training_path))

    def _load_training_and_fit(self) -> None:
        path = self._training_path()
        if not path.is_file():
            return
        xs: list[list[float]] = []
        ys: list[int] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                feat = row.get("features") if isinstance(row.get("features"), dict) else {}
                vec = build_rf_feature_vector(feat).ravel().tolist()
                lab = int(row.get("label", 0))
                xs.append(vec)
                ys.append(1 if lab == 1 else 0)
        except Exception as exc:
            logger.warning("meta_label RF training load failed: %s", exc)
            return
        if len(xs) >= int(self.config.rf_min_train_samples) and len(set(ys)) >= 2:
            self._fit_rf(xs, ys)

    def _fit_rf(self, xs: list[list[float]], ys: list[int]) -> None:
        RF, ok = _try_import_sklearn_rf()
        if not ok or RF is None:
            self._sklearn_ok = False
            self._rf_model = None
            return
        try:
            X = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=int)
            clf = RF(
                n_estimators=int(self.config.rf_n_estimators),
                max_depth=int(self.config.rf_max_depth),
                min_samples_leaf=int(self.config.rf_min_samples_leaf),
                class_weight="balanced_subsample",
                random_state=int(self.config.rf_random_state),
                n_jobs=1,
            )
            clf.fit(X, y)
            self._rf_model = clf
            self._samples_since_fit = 0
            logger.info("Meta-label RF fit on %d rows (bad_frac=%.3f)", len(y), float(np.mean(y)))
        except Exception as exc:
            logger.debug("Meta-label RF fit failed: %s", exc)
            self._rf_model = None

    def record_closed_trade(self, *, features: dict[str, Any] | None, profitable: bool) -> None:
        """Append one labeled example (label=1 unprofitable / false positive, 0 = profitable)."""
        if not isinstance(features, dict) or not features:
            return
        label = 0 if profitable else 1
        path = self._training_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"features": features, "label": label}, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.debug("meta_label training append failed: %s", exc)
            return

        self._samples_since_fit += 1
        if self._samples_since_fit < int(self.config.rf_refit_every):
            return
        self._samples_since_fit = 0
        # Reload tail of journal for refit (bounded)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-int(self.config.rf_max_buffer) :]
            xs: list[list[float]] = []
            ys: list[int] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                feat = row.get("features") if isinstance(row.get("features"), dict) else {}
                vec = build_rf_feature_vector(feat).ravel().tolist()
                ys.append(1 if int(row.get("label", 0)) == 1 else 0)
                xs.append(vec)
            if len(xs) >= int(self.config.rf_min_train_samples) and len(set(ys)) >= 2:
                self._fit_rf(xs, ys)
        except Exception as exc:
            logger.debug("meta_label RF refit skipped: %s", exc)

    def _predict_fp_prob(self, meta_features: dict[str, Any] | None) -> float | None:
        if not self._sklearn_ok or self._rf_model is None or not meta_features:
            return None
        try:
            X = build_rf_feature_vector(meta_features)
            proba = self._rf_model.predict_proba(X)
            classes = list(getattr(self._rf_model, "classes_", [0, 1]))
            if 1 not in classes:
                return None
            idx = classes.index(1)
            return float(proba[0, idx])
        except Exception as exc:
            logger.debug("meta_label RF predict failed: %s", exc)
            return None

    def label_trade(
        self,
        *,
        base_decision: str,
        confidence: float,
        calibrated_prob: float,
        blockers: list[str] | None = None,
        calibration_error: float = 0.0,
        drift_level: str = "LOW",
        regime: str = "DEFAULT",
        edge_decay: bool = False,
        meta_features: dict[str, Any] | None = None,
        utc_hour: int | None = None,
        probability_service: Any | None = None,
    ) -> dict[str, Any]:
        decision = str(base_decision or "HOLD").upper()
        blockers_list = list(blockers or [])
        conf = float(max(0.0, min(100.0, confidence)))
        prob = float(max(0.0, min(1.0, calibrated_prob)))
        ece = float(max(0.0, calibration_error))
        drift = str(drift_level or "LOW").upper()
<<<<<<< HEAD
        gate_hour = int(utc_hour) if utc_hour is not None else datetime.now(timezone.utc).hour
        confidence_gate = get_dynamic_confidence_gate(regime, utc_hour=gate_hour)
=======
        confidence_gate = get_dynamic_confidence_gate(regime)
>>>>>>> origin/main

        reasons: list[str] = []
        allow = decision in {"LONG", "SHORT"}
        decay_override_granted = False
        if decision == "HOLD":
            reasons.append("Base decision HOLD")
            allow = False
        if blockers_list:
            reasons.append("Primary blockers active")
            allow = False
        if conf < confidence_gate:
            reasons.append(f"Confidence below threshold ({conf:.1f} < {confidence_gate:.1f})")
            allow = False
        prob_service = probability_service
        calibration_fitted = getattr(prob_service, "platt", None) if prob_service is not None else None
        is_fitted = bool(calibration_fitted and getattr(calibration_fitted, "fitted", False))
        effective_min_prob = self.config.min_probability if is_fitted else 0.38
        logger.debug(
            "calibration gate: fitted=%s prob=%.3f threshold=%.3f",
            is_fitted,
            prob,
            effective_min_prob,
        )
        if prob < effective_min_prob:
            reasons.append(f"Calibrated probability too low ({prob:.2f})")
            allow = False
        if is_fitted and ece > self.config.max_calibration_error:
            reasons.append(f"Calibration instability (ECE {ece:.3f})")
            allow = False
        if drift == "HIGH":
            reasons.append("High data drift detected")
            allow = False
        if edge_decay:
            reasons.append("Edge decay safeguard active")
            decay_override_granted = bool(prob >= 0.62 and conf >= 65.0)
            if not decay_override_granted:
                reasons.append("Edge decay override denied: requires prob >= 0.62 and confidence >= 65.0")
                allow = False

        rf_fp_prob: float | None = None
        rf_veto = False
        if allow and self._sklearn_ok:
            rf_fp_prob = self._predict_fp_prob(meta_features)
            thr = float(self.config.rf_fp_threshold)
            if rf_fp_prob is not None and rf_fp_prob > thr:
                allow = False
                rf_veto = True
                reasons.insert(
                    0,
                    f"RF meta-label veto (false-positive prob {rf_fp_prob:.3f} > {thr:.2f})",
                )

        if allow:
            label = "ALLOW"
            final_decision = decision
            reason = "Meta-label approved high-quality setup"
        else:
            label = "REJECT"
            final_decision = "HOLD"
            reason = reasons[0] if reasons else "Meta-label rejected setup"

        out_inputs = {
            "confidence": round(conf, 2),
            "calibrated_prob": round(prob, 6),
            "calibration_error": round(ece, 6),
            "drift_level": drift,
            "regime": str(regime or "DEFAULT").upper(),
            "confidence_gate": round(confidence_gate, 2),
            "edge_decay": bool(edge_decay),
            "rf_fp_probability": None if rf_fp_prob is None else round(float(rf_fp_prob), 6),
            "rf_veto_applied": bool(rf_veto),
            "rf_model_active": bool(self._rf_model is not None),
        }
        return {
            "label": label,
            "allow_trade": bool(allow),
            "edge_decay_override": bool(edge_decay and allow and decay_override_granted),
            "final_decision": final_decision,
            "reason": reason,
            "reasons": reasons,
<<<<<<< HEAD
            "inputs": out_inputs,
=======
            "inputs": {
                "confidence": round(conf, 2),
                "calibrated_prob": round(prob, 6),
                "calibration_error": round(ece, 6),
                "drift_level": drift,
                "regime": str(regime or "DEFAULT").upper(),
                "confidence_gate": round(confidence_gate, 2),
                "edge_decay": bool(edge_decay),
            },
>>>>>>> origin/main
        }
