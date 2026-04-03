from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ProbabilityInput:
    momentum_score: float
    flow_score: float
    volatility_regime: str
    regime: str


class PlattCalibrator:
    def __init__(self, state_path: str | None = None) -> None:
        self.A: float = -1.0
        self.B: float = 0.0
        self._state_path = Path(state_path) if state_path else None
        self._fitted: bool = False
        self._load()

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return float(1.0 / (1.0 + z))
        z = math.exp(x)
        return float(z / (1.0 + z))

    def _load(self) -> None:
        if not self._state_path or (not self._state_path.exists()):
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.A = float(payload.get("A", -1.0))
            self.B = float(payload.get("B", 0.0))
            self._fitted = bool(payload.get("fitted", True))
        except Exception:
            return

    def _save(self) -> None:
        if not self._state_path:
            return
        payload = {
            "A": float(self.A),
            "B": float(self.B),
            "fitted": bool(self._fitted),
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            return

    @property
    def fitted(self) -> bool:
        return bool(self._fitted)

    def fit(self, raw_scores: list[float], labels: list[int], min_samples: int = 30) -> bool:
        if len(raw_scores) < int(min_samples) or len(labels) < int(min_samples) or len(raw_scores) != len(labels):
            return False

        scores = [float(x) for x in raw_scores]
        ys = [1 if int(y) > 0 else 0 for y in labels]
        n = len(scores)
        lr = 0.01
        max_iters = 200
        convergence = 1e-6

        A = float(self.A)
        B = float(self.B)
        for _ in range(max_iters):
            grad_A = 0.0
            grad_B = 0.0
            for x_i, y_i in zip(scores, ys):
                p_i = self._sigmoid((A * x_i) + B)
                diff = p_i - float(y_i)
                grad_A += diff * x_i
                grad_B += diff
            grad_A /= float(n)
            grad_B /= float(n)

            next_A = A - (lr * grad_A)
            next_B = B - (lr * grad_B)
            delta = max(abs(next_A - A), abs(next_B - B))
            A = next_A
            B = next_B
            if delta < convergence:
                break

        self.A = float(A)
        self.B = float(B)
        self._fitted = True
        self._save()
        return True

    def predict(self, raw_score: float) -> float:
        p = self._sigmoid((float(self.A) * float(raw_score)) + float(self.B))
        return float(max(0.05, min(0.95, p)))

    @staticmethod
    def brier_score(probs: list[float], labels: list[int]) -> float:
        if not probs or not labels or len(probs) != len(labels):
            return 0.0
        total = 0.0
        n = len(probs)
        for p, y in zip(probs, labels):
            prob = float(max(0.0, min(1.0, p)))
            label = 1.0 if int(y) > 0 else 0.0
            total += (prob - label) ** 2
        return float(total / float(n))


class ProbabilityService:
    """Directional probability estimator from momentum + flow + volatility context."""

    def __init__(self, platt_state_path: str | None = "btc_intelligence/logs/platt_calibrator_state.json") -> None:
        self.platt = PlattCalibrator(state_path=platt_state_path)
        self._platt_raw_scores: list[float] = []
        self._platt_labels: list[int] = []
        self._new_labels_since_retrain: int = 0
        self._retrain_interval: int = 50
        self._max_training_rows: int = 4000

    def record_labeled_trade(self, raw_score: float, label: int) -> bool:
        self._platt_raw_scores.append(float(raw_score))
        self._platt_labels.append(1 if int(label) > 0 else 0)
        if len(self._platt_raw_scores) > self._max_training_rows:
            self._platt_raw_scores = self._platt_raw_scores[-self._max_training_rows :]
            self._platt_labels = self._platt_labels[-self._max_training_rows :]

        self._new_labels_since_retrain += 1
        if self._new_labels_since_retrain >= self._retrain_interval:
            trained = self.platt.fit(self._platt_raw_scores, self._platt_labels, min_samples=30)
            if trained:
                self._new_labels_since_retrain = 0
            return trained
        return False

    def estimate(self, inp: ProbabilityInput) -> dict[str, Any]:
        momentum = float(np.clip(inp.momentum_score, -1.0, 1.0))
        flow = float(np.clip(inp.flow_score, -1.0, 1.0))
        vol = str(inp.volatility_regime or "").upper()
        regime = str(inp.regime or "").lower()

        regime_up_bias = 0.0
        regime_down_bias = 0.0
        if "bull" in regime or "breakout_up" in regime:
            regime_up_bias = 0.35
        elif "bear" in regime or "breakout_down" in regime:
            regime_down_bias = 0.35

        vol_sideways_bias = 0.0
        if vol in {"LOW", "COMPRESSION"}:
            vol_sideways_bias = 0.55
        elif vol in {"EXPANSION"}:
            vol_sideways_bias = -0.10
        elif vol in {"HIGH_VOL"}:
            vol_sideways_bias = 0.25

        up_logit = (0.90 * momentum) + (0.80 * flow) + regime_up_bias
        down_logit = (-0.90 * momentum) + (-0.80 * flow) + regime_down_bias
        side_logit = 0.35 + vol_sideways_bias - (0.60 * abs(momentum)) - (0.50 * abs(flow))

        logits = np.asarray([up_logit, down_logit, side_logit], dtype=float)
        logits = logits - float(np.max(logits))
        exps = np.exp(logits)
        probs = exps / max(float(np.sum(exps)), 1e-9)

        up_pct = float(probs[0] * 100.0)
        down_pct = float(probs[1] * 100.0)
        side_pct = float(probs[2] * 100.0)
        total = up_pct + down_pct + side_pct
        if total > 0:
            up_pct = up_pct / total * 100.0
            down_pct = down_pct / total * 100.0
            side_pct = side_pct / total * 100.0

        dominant = "UP" if up_pct >= down_pct and up_pct >= side_pct else "DOWN" if down_pct >= side_pct else "SIDEWAYS"
        dominant_raw_score = float(up_logit if dominant == "UP" else down_logit if dominant == "DOWN" else side_logit)
        dominant_base_prob = float(max(up_pct, down_pct, side_pct) / 100.0)
        if self.platt.fitted:
            platt_probability = self.platt.predict(dominant_raw_score)
            platt_calibrated = True
        else:
            platt_probability = float(max(0.05, min(0.95, dominant_base_prob)))
            platt_calibrated = False

        return {
            "up_prob": round(up_pct, 2),
            "down_prob": round(down_pct, 2),
            "sideways_prob": round(side_pct, 2),
            "dominant_state": dominant,
            "platt_calibrated": bool(platt_calibrated),
            "platt_probability": round(float(platt_probability), 6),
            "model_inputs": {
                "momentum_score": round(momentum, 4),
                "flow_score": round(flow, 4),
                "volatility_regime": vol or "UNKNOWN",
                "regime": regime or "unknown",
            },
        }
