"""
Walk-forward validation with Purged K-Fold OOS Sharpe and Deflated Sharpe Ratio (DSR).

Purging removes training samples whose label window overlaps the test interval; the embargo
drops samples immediately after the test block to limit sequence leakage (state transitions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any


@dataclass
class WalkForwardConfig:
    train_window: int = 120
    test_window: int = 40
    min_improvement: float = 0.02
    max_degradation: float = 0.05
    # Purged K-Fold
    n_splits: int = 5
    label_horizon_seconds: int = 7200  # post-label / labeling overlap window (e.g. 2h)
    embargo_seconds: int = 7200  # embargo after test block end
    min_rows_for_pkf: int = 30
    # Multiple-testing adjustment for DSR
    dsr_n_trials: int = 8


def _parse_row_time(row: dict[str, Any]) -> datetime | None:
    for key in ("timestamp", "as_of_utc", "closed_at_utc"):
        v = row.get(key)
        if not v:
            continue
        s = str(v).strip()
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            continue
    return None


def _safe_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    sd = pstdev(returns)
    if sd <= 1e-12:
        return 0.0
    return float((mean(returns) / sd) * math.sqrt(len(returns)))


def _skew_and_pearson_kurtosis(returns: list[float]) -> tuple[float, float]:
    n = len(returns)
    if n < 4:
        return 0.0, 3.0
    mu = mean(returns)
    var = sum((x - mu) ** 2 for x in returns) / max(n - 1, 1)
    if var <= 1e-18:
        return 0.0, 3.0
    sd = math.sqrt(var)
    m3 = sum(((x - mu) / sd) ** 3 for x in returns) / n
    m4 = sum(((x - mu) / sd) ** 4 for x in returns) / n
    return float(m3), float(m4)


def variance_sharpe_ratio(sharpe: float, n: int, skewness: float, pearson_kurtosis: float) -> float:
    """Bailey & López de Prado (2015) variance of Sharpe ratio estimator (under non-null SR)."""
    if n < 4:
        return 1.0
    sr = float(sharpe)
    gamma = float(skewness)
    kappa = float(pearson_kurtosis)
    v = (1.0 - gamma * sr + ((kappa - 3.0) / 4.0) * sr * sr) / max(n - 1, 1)
    return float(max(v, 1e-18))


def deflated_sharpe_ratio(
    returns: list[float],
    *,
    n_trials: int,
) -> dict[str, float]:
    """
    Deflated Sharpe Ratio (conservative): (SR - E[max SR | null, M trials]) / σ_SR.
    Returns diagnostic fields for logging in validation / weekly reports.
    """
    r = [float(x) for x in returns]
    n = len(r)
    if n < 4:
        return {
            "sharpe_observed": 0.0,
            "sharpe_std_err": 0.0,
            "deflated_sharpe_ratio": 0.0,
            "probabilistic_sharpe_ratio": 0.0,
            "expected_max_sharpe_null": 0.0,
            "skew_returns": 0.0,
            "pearson_kurtosis_returns": 3.0,
        }
    sr = _safe_sharpe(r)
    sk, ku = _skew_and_pearson_kurtosis(r)
    var_sr = variance_sharpe_ratio(sr, n, sk, ku)
    sigma_sr = math.sqrt(var_sr)
    m = max(int(n_trials), 1)
    if m <= 1:
        e_max = 0.0
    else:
        e_max = sigma_sr * math.sqrt(2.0 * math.log(max(m, 2)))
    dsr = (sr - e_max) / sigma_sr if sigma_sr > 1e-12 else 0.0
    psr_denom = sigma_sr if sigma_sr > 1e-12 else 1e-12
    psr = _normal_cdf(sr / psr_denom)
    return {
        "sharpe_observed": round(float(sr), 6),
        "sharpe_std_err": round(float(sigma_sr), 6),
        "deflated_sharpe_ratio": round(float(dsr), 6),
        "probabilistic_sharpe_ratio": round(float(psr), 6),
        "expected_max_sharpe_null": round(float(e_max), 6),
        "skew_returns": round(float(sk), 6),
        "pearson_kurtosis_returns": round(float(ku), 6),
    }


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class WalkForwardValidator:
    """
    Walk-forward validator for adaptive strategy weight updates.
    Uses Purged K-Fold OOS Sharpe when enough timestamped rows exist; otherwise legacy split.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    @staticmethod
    def _safe_sharpe_inst(returns: list[float]) -> float:
        return _safe_sharpe(returns)

    def _legacy_validate(
        self,
        base_sharpe: float,
        weight_delta: float,
    ) -> dict[str, Any]:
        old_sharpe = float(base_sharpe)
        new_sharpe = float(base_sharpe)
        improvement = float(base_sharpe - 0.0)

        if base_sharpe < -float(self.config.max_degradation):
            recommendation = "REJECT"
            applied_fraction = 0.0
            approved = False
            reason = "Walk-forward reject: raw test Sharpe below degradation limit"
        elif base_sharpe > float(self.config.min_improvement):
            approved = True
            if weight_delta >= 0.02:
                recommendation = "APPROVE"
                applied_fraction = 1.0
                reason = "Walk-forward approve: raw Sharpe exceeds improvement threshold"
            else:
                recommendation = "CONDITIONAL"
                applied_fraction = 0.5
                reason = "Marginal weight delta; apply partial update"
        else:
            recommendation = "CONDITIONAL"
            applied_fraction = 0.5
            approved = True
            reason = "Walk-forward conditional: raw Sharpe in neutral band"

        return {
            "approved": bool(approved),
            "recommendation": recommendation,
            "old_sharpe": float(old_sharpe),
            "new_sharpe": float(new_sharpe),
            "improvement": float(improvement),
            "applied_fraction": float(applied_fraction),
            "reason": reason,
            "validation_mode": "legacy_split",
        }

    def _purged_kfold_oos_metrics(
        self,
        rows: list[dict[str, Any]],
        returns: list[float],
        times: list[datetime],
    ) -> dict[str, Any] | None:
        n = len(rows)
        k = max(2, int(self.config.n_splits))
        if n < max(int(self.config.min_rows_for_pkf), k * 2):
            return None

        label_td = timedelta(seconds=int(self.config.label_horizon_seconds))
        emb_td = timedelta(seconds=int(self.config.embargo_seconds))
        t0_list = times
        t1_list = [t + label_td for t in times]

        fold_sizes = [n // k] * k
        for i in range(n % k):
            fold_sizes[i] += 1
        boundaries: list[tuple[int, int]] = []
        start = 0
        for fs in fold_sizes:
            boundaries.append((start, start + fs))
            start += fs

        fold_sharpes: list[float] = []
        oos_returns: list[float] = []
        purge_drop_counts: list[int] = []

        for a, b in boundaries:
            test_idx = set(range(a, b))
            if len(test_idx) < 2:
                continue
            t0_test_min = min(t0_list[i] for i in test_idx)
            t1_test_max = max(t1_list[i] for i in test_idx)

            dropped = 0
            for j in range(n):
                if j in test_idx:
                    continue
                if t0_list[j] <= t1_test_max and t1_list[j] >= t0_test_min:
                    dropped += 1
                    continue
                if t0_list[j] > t1_test_max and t0_list[j] <= t1_test_max + emb_td:
                    dropped += 1
                    continue
            purge_drop_counts.append(dropped)

            test_returns = [returns[i] for i in sorted(test_idx)]
            fold_sharpes.append(_safe_sharpe(test_returns))
            oos_returns.extend(test_returns)

        if len(fold_sharpes) < 2:
            return None

        mean_fold = float(mean(fold_sharpes))
        dsr_payload = deflated_sharpe_ratio(oos_returns, n_trials=max(k, int(self.config.dsr_n_trials)))

        return {
            "mean_oos_sharpe": round(mean_fold, 6),
            "fold_sharpes": [round(x, 6) for x in fold_sharpes],
            "n_splits": k,
            "label_horizon_seconds": int(self.config.label_horizon_seconds),
            "embargo_seconds": int(self.config.embargo_seconds),
            "purge_drops_per_fold": purge_drop_counts,
            **dsr_payload,
        }

    def validate_weight_update(
        self,
        strategy: str,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        trade_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rows = list(trade_history or [])
        if len(rows) < 4:
            return {
                "approved": True,
                "recommendation": "CONDITIONAL",
                "old_sharpe": 0.0,
                "new_sharpe": 0.0,
                "improvement": 0.0,
                "applied_fraction": 0.5,
                "reason": "Insufficient walk-forward samples; apply partial update validation_mode=fallback",
                "validation_mode": "insufficient_samples",
                "deflated_sharpe_ratio": 0.0,
            }

        indexed: list[tuple[datetime, dict[str, Any], float]] = []
        for row in rows:
            t = _parse_row_time(row)
            if t is None:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            indexed.append((t, row, float(row.get("pnl_pct", 0.0)) / 100.0))

        if len(indexed) < max(8, int(self.config.min_rows_for_pkf) // 2):
            indexed = []
            base_t = datetime.now(timezone.utc)
            for i, row in enumerate(rows):
                indexed.append((base_t + timedelta(hours=i), row, float(row.get("pnl_pct", 0.0)) / 100.0))

        indexed.sort(key=lambda x: x[0])
        times = [x[0] for x in indexed]
        sorted_rows = [x[1] for x in indexed]
        returns = [x[2] for x in indexed]

        split_idx = int(len(sorted_rows) * 0.75)
        split_idx = max(1, min(len(sorted_rows) - 1, split_idx))
        train_rows = sorted_rows[:split_idx]
        test_rows = sorted_rows[split_idx:]

        if self.config.train_window > 0:
            train_rows = train_rows[-int(self.config.train_window) :]
        if self.config.test_window > 0:
            test_rows = test_rows[-int(self.config.test_window) :]

        if len(train_rows) < 2:
            return {
                "approved": True,
                "recommendation": "CONDITIONAL",
                "old_sharpe": 0.0,
                "new_sharpe": 0.0,
                "improvement": 0.0,
                "applied_fraction": 0.5,
                "reason": "Insufficient train window; apply partial update",
                "validation_mode": "legacy_train_short",
                "deflated_sharpe_ratio": 0.0,
            }

        if len(test_rows) < 2:
            return {
                "approved": True,
                "recommendation": "CONDITIONAL",
                "old_sharpe": 0.0,
                "new_sharpe": 0.0,
                "improvement": 0.0,
                "applied_fraction": 0.5,
                "reason": "Insufficient test window for Sharpe; apply partial update",
                "validation_mode": "legacy_test_short",
                "deflated_sharpe_ratio": 0.0,
            }

        test_returns_legacy = [float(t.get("pnl_pct", 0.0)) / 100.0 for t in test_rows]
        base_sharpe = _safe_sharpe(test_returns_legacy)
        old_w = float(old_weights.get(strategy, 0.0))
        new_w = float(new_weights.get(strategy, old_w))
        weight_delta = abs(new_w - old_w)

        pkf = self._purged_kfold_oos_metrics(sorted_rows, returns, times)
        if pkf is not None:
            base_sharpe = float(pkf.get("mean_oos_sharpe", base_sharpe))

        out = self._legacy_validate(base_sharpe, weight_delta)
        out["strategy"] = str(strategy)
        if pkf is not None:
            out["validation_mode"] = "purged_kfold"
            for key, val in pkf.items():
                out[key] = val
        else:
            tr = [float(t.get("pnl_pct", 0.0)) / 100.0 for t in test_rows]
            dsr_fallback = deflated_sharpe_ratio(tr, n_trials=max(2, int(self.config.dsr_n_trials)))
            for key, val in dsr_fallback.items():
                out.setdefault(key, val)
            out.setdefault("validation_mode", "legacy_split")

        out["improvement"] = float(out.get("new_sharpe", 0.0))
        return out
