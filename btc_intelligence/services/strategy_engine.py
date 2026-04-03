from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .walk_forward import WalkForwardConfig, WalkForwardValidator


STRATEGIES = ("trend_following", "mean_reversion", "breakout", "volatility_based")


@dataclass
class StrategyEngineConfig:
    min_trades_required: int = 30
    update_frequency: int = 20
    learning_rate: float = 0.01
    max_change_per_update: float = 0.05


class StrategyEngine:
    """
    Regime-aware multi-strategy selector with independent strategy tracking.
    """

    def __init__(
        self,
        state_path: str,
        config: StrategyEngineConfig | None = None,
        walk_forward_validator: WalkForwardValidator | None = None,
    ) -> None:
        self.config = config or StrategyEngineConfig()
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights = {
            "trend_following": 0.30,
            "mean_reversion": 0.24,
            "breakout": 0.24,
            "volatility_based": 0.22,
        }
        self.history: dict[str, list[dict[str, Any]]] = {k: [] for k in STRATEGIES}
        self.last_update_count: dict[str, int] = {k: 0 for k in STRATEGIES}
        self.last_wf_validation: dict[str, dict[str, Any]] = {}
        self.walk_forward_validator = walk_forward_validator or WalkForwardValidator(WalkForwardConfig())
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("weights"), dict):
                for key in STRATEGIES:
                    if key in payload["weights"]:
                        self.weights[key] = float(payload["weights"][key])
                self._normalize_weights()
            if isinstance(payload.get("history"), dict):
                for key in STRATEGIES:
                    rows = payload["history"].get(key, [])
                    if isinstance(rows, list):
                        self.history[key] = rows[-800:]
            if isinstance(payload.get("last_update_count"), dict):
                for key in STRATEGIES:
                    self.last_update_count[key] = int(payload["last_update_count"].get(key, 0))
            if isinstance(payload.get("last_wf_validation"), dict):
                loaded_wf = payload.get("last_wf_validation", {})
                self.last_wf_validation = {
                    str(k): dict(v) for k, v in loaded_wf.items() if isinstance(v, dict)
                }
        except Exception:
            return

    def _save(self) -> None:
        payload = {
            "weights": self.weights,
            "history": {k: v[-800:] for k, v in self.history.items()},
            "last_update_count": self.last_update_count,
            "last_wf_validation": self.last_wf_validation,
        }
        try:
            self.state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            return

    def _normalize_weights(self) -> None:
        total = sum(max(0.05, float(v)) for v in self.weights.values())
        if total <= 0:
            total = float(len(STRATEGIES))
        for key in STRATEGIES:
            raw = max(0.05, float(self.weights.get(key, 0.1)))
            self.weights[key] = raw / total

    @staticmethod
    def _regime_bucket(regime: str) -> str:
        r = str(regime or "").upper()
        if "PANIC" in r or "HIGH_VOL" in r or "VOLAT" in r:
            return "VOLATILE"
        if "BULL" in r or "BEAR" in r or "TREND" in r or "BREAKOUT" in r:
            return "TREND"
        return "SIDEWAYS"

    @staticmethod
    def _base_by_regime(regime_bucket: str) -> str:
        if regime_bucket == "TREND":
            return "trend_following"
        if regime_bucket == "SIDEWAYS":
            return "mean_reversion"
        return "breakout"

    def select_strategy(
        self,
        *,
        regime: str,
        volatility_regime: str,
        momentum_score: float,
        flow_score: float,
        cost_score: float,
    ) -> dict[str, Any]:
        reg_bucket = self._regime_bucket(regime)
        vol = str(volatility_regime or "NORMAL").upper()
        m = float(max(-1.0, min(1.0, momentum_score)))
        f = float(max(-1.0, min(1.0, flow_score)))
        c = float(max(-1.0, min(1.0, cost_score)))
        vol_high = 1.0 if vol in {"EXPANSION", "HIGH_VOL", "PANIC"} else 0.0

        scores = {
            "trend_following": (0.52 * m) + (0.30 * f) + (0.18 * c),
            "mean_reversion": (0.48 * (-abs(m))) + (0.22 * c) + (0.30 * (1.0 - abs(f))),
            "breakout": (0.40 * abs(m)) + (0.30 * f) + (0.20 * vol_high) + (0.10 * c),
            "volatility_based": (0.55 * vol_high) + (0.20 * abs(m)) + (0.15 * f) + (0.10 * c),
        }
        weighted_scores = {
            k: float(scores[k]) * float(self.weights.get(k, 0.25))
            for k in STRATEGIES
        }

        selected = max(weighted_scores.items(), key=lambda kv: kv[1])[0]
        regime_default = self._base_by_regime(reg_bucket)
        if weighted_scores.get(selected, -1.0) < -0.15:
            selected = regime_default

        return {
            "strategy_used": selected,
            "strategy_scores": {k: round(float(v), 6) for k, v in weighted_scores.items()},
            "strategy_weights": {k: round(float(self.weights.get(k, 0.0)), 6) for k in STRATEGIES},
            "regime_bucket": reg_bucket,
            "regime_default": regime_default,
        }

    def record_trade(
        self,
        *,
        strategy: str,
        pnl_pct: float,
        regime: str,
        timestamp_utc: str,
    ) -> None:
        s = str(strategy or "").strip()
        if s not in STRATEGIES:
            return
        self.history[s].append(
            {
                "pnl_pct": float(pnl_pct),
                "regime": str(regime),
                "timestamp": str(timestamp_utc),
            }
        )
        self.history[s] = self.history[s][-800:]
        self._maybe_update_weights()
        self._save()

    def _strategy_quality(self, strategy: str) -> float:
        rows = self.history.get(strategy, [])
        if len(rows) < 10:
            return 0.0
        vals = [float(r.get("pnl_pct", 0.0)) / 100.0 for r in rows[-80:]]
        wr = sum(1 for x in vals if x > 0) / len(vals)
        sharpe_window = vals[-30:]
        n_sharpe = len(sharpe_window)
        if n_sharpe > 1:
            sd = stdev(sharpe_window)
            sharpe = ((mean(sharpe_window) / sd) * math.sqrt(n_sharpe)) if sd > 1e-9 else 0.0
        else:
            sharpe = 0.0
        return float((0.70 * wr) + (0.30 * max(min(sharpe, 3.0), -3.0) / 3.0))

    def _maybe_update_weights(self) -> None:
        total_trades = sum(len(v) for v in self.history.values())
        if total_trades < self.config.min_trades_required:
            return
        if total_trades % self.config.update_frequency != 0:
            return

        for key in STRATEGIES:
            prev_updates = int(self.last_update_count.get(key, 0))
            if len(self.history[key]) <= prev_updates:
                continue
            q = self._strategy_quality(key)
            delta_raw = self.config.learning_rate * q
            base = float(self.weights.get(key, 0.25))
            max_delta = self.config.max_change_per_update * base
            delta = max(-max_delta, min(max_delta, delta_raw))
            proposed_weight = max(0.05, base + delta)
            old_weights = dict(self.weights)
            new_weights = dict(self.weights)
            new_weights[key] = proposed_weight
            wf_result = self.walk_forward_validator.validate_weight_update(
                strategy=key,
                old_weights=old_weights,
                new_weights=new_weights,
                trade_history=self.history.get(key, []),
            )
            applied_fraction = float(wf_result.get("applied_fraction", 1.0))
            applied_fraction = max(0.0, min(1.0, applied_fraction))
            self.weights[key] = max(0.05, base + (delta * applied_fraction))
            self.last_wf_validation[key] = wf_result
            self.last_update_count[key] = len(self.history[key])
        self._normalize_weights()
