"""
Step 4 — Overfitting Defense: Deflated Sharpe Ratio + PBO.

Two critical statistical tests that separate real alpha from noise:

1. Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014)
   Adjusts the Sharpe ratio for multiple testing bias.
   If you tested 100 parameter combos and picked the best one,
   the observed Sharpe is almost certainly inflated.

   DSR = Sharpe_obs - E[max(Sharpe)] under null
   Threshold: DSR > 0.0 means the strategy likely has real alpha.

2. Probability of Backtest Overfitting (PBO) — Bailey et al. (2015)
   Combinatorial approach:
     - Split N backtest trials into train/test subsets
     - For each split: pick best strategy in-sample, measure OOS
     - PBO = fraction of times in-sample best UNDERPERFORMS OOS median

   If PBO > 0.5 → strategy is overfit
   If PBO > 0.7 → strategy is garbage
   If PBO < 0.3 → likely has real edge

Usage:
    val = ResearchValidator()
    dsr = val.deflated_sharpe(sharpe_obs=1.2, n_trials=50, n_obs=252)
    pbo = val.compute_pbo(returns_matrix)  # (n_days, n_strategies)
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ResearchValidator:
    """
    Statistical validation layer for backtest results.
    Prevents deploying overfit strategies.
    """

    # ------------------------------------------------------------------
    # Deflated Sharpe Ratio
    # ------------------------------------------------------------------

    @staticmethod
    def deflated_sharpe(
        sharpe_obs: float,
        n_trials: int,
        n_obs: int = 252,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> dict:
        """
        Compute the Deflated Sharpe Ratio.

        Accounts for:
            - Multiple testing (n_trials strategies tried)
            - Non-normal returns (skewness, kurtosis)
            - Sample size (n_obs)

        Parameters
        ----------
        sharpe_obs : observed Sharpe ratio (annualised)
        n_trials   : number of strategies/parameter sets tested
        n_obs      : number of return observations
        skewness   : return distribution skewness (0 = normal)
        kurtosis   : return distribution kurtosis (3 = normal)

        Returns
        -------
        dict with: dsr, sharpe_threshold, sharpe_obs, is_significant, p_value
        """
        if n_trials <= 0 or n_obs <= 0:
            return {"dsr": 0.0, "is_significant": False, "error": "invalid inputs"}

        # Expected max Sharpe under null (Euler-Mascheroni adjusted)
        euler_m = 0.5772
        e_max_sharpe = np.sqrt(2 * np.log(n_trials)) - (
            (np.log(np.pi) + euler_m) / (2 * np.sqrt(2 * np.log(n_trials)))
        )

        # Variance of Sharpe estimator (Lo, 2002) with non-normality correction
        var_sharpe = (
            1.0
            + (skewness * sharpe_obs) / 2.0
            + ((kurtosis - 3) * sharpe_obs**2) / 4.0
        ) / n_obs

        se_sharpe = np.sqrt(max(var_sharpe, 1e-10))

        # DSR: how many SEs above the expected max
        dsr = (sharpe_obs - e_max_sharpe) / se_sharpe

        # P-value
        p_value = 1.0 - norm.cdf(dsr)

        result = {
            "dsr": round(float(dsr), 4),
            "sharpe_obs": round(sharpe_obs, 4),
            "sharpe_threshold": round(float(e_max_sharpe), 4),
            "n_trials": n_trials,
            "n_obs": n_obs,
            "p_value": round(float(p_value), 4),
            "is_significant": float(dsr) > 0.0,
        }

        logger.info(
            "DSR: observed Sharpe=%.3f, threshold=%.3f, DSR=%.3f, p=%.4f → %s",
            sharpe_obs, e_max_sharpe, dsr, p_value,
            "SIGNIFICANT ✅" if dsr > 0 else "OVERFIT ❌",
        )
        return result

    # ------------------------------------------------------------------
    # Probability of Backtest Overfitting (PBO)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pbo(
        returns_matrix: np.ndarray,
        n_partitions: int = 10,
    ) -> dict:
        """
        Compute PBO using CSCV (Combinatorially Symmetric Cross-Validation).

        Parameters
        ----------
        returns_matrix : np.ndarray of shape (n_days, n_strategies)
            Each column is a backtest return series for a different
            strategy/parameter set.
        n_partitions   : number of time blocks to partition data into

        Returns
        -------
        dict with: pbo, n_combos, logit_distribution, verdict
        """
        n_days, n_strategies = returns_matrix.shape

        if n_strategies < 2:
            return {"pbo": 0.0, "verdict": "INSUFFICIENT_STRATEGIES"}
        if n_days < n_partitions * 2:
            return {"pbo": 0.0, "verdict": "INSUFFICIENT_DATA"}

        # Split into n_partitions time blocks
        block_size = n_days // n_partitions
        blocks = [
            returns_matrix[i * block_size: (i+1) * block_size]
            for i in range(n_partitions)
        ]

        # Generate all combinations of n_partitions/2 blocks for train
        half = n_partitions // 2
        all_combos = list(combinations(range(n_partitions), half))

        # Cap at 100 combos for speed
        if len(all_combos) > 100:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(all_combos), 100, replace=False)
            all_combos = [all_combos[i] for i in idx]

        overfit_count = 0
        logits = []

        for train_blocks in all_combos:
            test_blocks = [i for i in range(n_partitions) if i not in train_blocks]

            # Aggregate train and test returns
            train_rets = np.vstack([blocks[i] for i in train_blocks])
            test_rets = np.vstack([blocks[i] for i in test_blocks])

            # Sharpe for each strategy
            train_sharpe = np.mean(train_rets, axis=0) / (np.std(train_rets, axis=0) + 1e-10)
            test_sharpe = np.mean(test_rets, axis=0) / (np.std(test_rets, axis=0) + 1e-10)

            # Best in-sample strategy
            best_is = np.argmax(train_sharpe)

            # Rank of best-IS in OOS
            oos_rank = np.sum(test_sharpe >= test_sharpe[best_is])
            relative_rank = oos_rank / n_strategies

            # Logit: log(rank / (1 - rank))
            rank_clipped = np.clip(relative_rank, 0.01, 0.99)
            logit = np.log(rank_clipped / (1 - rank_clipped))
            logits.append(logit)

            # Overfit if best-IS performs below median OOS
            if test_sharpe[best_is] < np.median(test_sharpe):
                overfit_count += 1

        pbo = overfit_count / max(len(all_combos), 1)

        if pbo > 0.7:
            verdict = "GARBAGE ❌ (PBO > 0.7)"
        elif pbo > 0.5:
            verdict = "LIKELY OVERFIT ⚠️ (PBO > 0.5)"
        elif pbo > 0.3:
            verdict = "MARGINAL 🟡 (PBO 0.3–0.5)"
        else:
            verdict = "LIKELY REAL EDGE ✅ (PBO < 0.3)"

        result = {
            "pbo": round(pbo, 4),
            "n_combos": len(all_combos),
            "n_strategies": n_strategies,
            "mean_logit": round(float(np.mean(logits)), 4),
            "verdict": verdict,
        }

        logger.info("PBO: %.3f (%d combos, %d strategies) → %s", pbo, len(all_combos), n_strategies, verdict)
        return result

    # ------------------------------------------------------------------
    # Combined Validation Report
    # ------------------------------------------------------------------

    def full_validation(
        self,
        sharpe_obs: float,
        n_trials: int,
        n_obs: int,
        returns_matrix: np.ndarray | None = None,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> dict:
        """
        Run full overfitting defense: DSR + PBO.

        Returns combined report with pass/fail verdict.
        """
        dsr_result = self.deflated_sharpe(sharpe_obs, n_trials, n_obs, skewness, kurtosis)

        pbo_result = {"pbo": 0.0, "verdict": "SKIPPED (no returns matrix)"}
        if returns_matrix is not None and returns_matrix.shape[1] >= 2:
            pbo_result = self.compute_pbo(returns_matrix)

        # Overall verdict
        dsr_pass = dsr_result.get("is_significant", False)
        pbo_pass = pbo_result.get("pbo", 1.0) < 0.5

        overall = "DEPLOY ✅" if (dsr_pass and pbo_pass) else "DO NOT DEPLOY ❌"

        return {
            "dsr": dsr_result,
            "pbo": pbo_result,
            "overall_verdict": overall,
            "dsr_pass": dsr_pass,
            "pbo_pass": pbo_pass,
        }


class CPCVValidator:
    """
    López de Prado-style CPCV validator for robust OOS diagnostics.
    """

    def __init__(self, n_splits: int = 6, n_test_splits: int = 2, embargo_pct: float = 0.01):
        self.n_splits = max(int(n_splits), 3)
        self.n_test_splits = max(1, int(n_test_splits))
        self.embargo_pct = float(np.clip(embargo_pct, 0.0, 0.25))

    @staticmethod
    def _annualized_sharpe(returns: np.ndarray) -> float:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 2:
            return 0.0
        sd = float(np.std(r))
        if sd <= 1e-12:
            return 0.0
        return float(np.mean(r) / sd * np.sqrt(252.0))

    def _purge_embargo(self, train_idx, test_idx, embargo_pct) -> np.ndarray:
        train = np.asarray(train_idx, dtype=int)
        test = np.asarray(test_idx, dtype=int)
        if train.size == 0 or test.size == 0:
            return train
        embargo = max(1, int(len(np.concatenate([train, test])) * float(embargo_pct)))
        t_min = int(test.min())
        t_max = int(test.max())
        keep = (train < (t_min - embargo)) | (train > (t_max + embargo))
        return train[keep]

    @staticmethod
    def _extract_returns(payload: object) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        if isinstance(payload, dict):
            is_ret = np.asarray(payload.get("is_returns", []), dtype=float)
            oos_ret = np.asarray(payload.get("oos_returns", []), dtype=float)
            curves = payload.get("equity_curve") or payload.get("oos_equity_curve") or []
            return is_ret, oos_ret, curves if isinstance(curves, list) else []
        arr = np.asarray(payload if payload is not None else [], dtype=float)
        return np.array([], dtype=float), arr, []

    def _estimate_pbo(self, oos_sharpes: list) -> float:
        if not oos_sharpes:
            return 1.0
        vals = np.asarray(oos_sharpes, dtype=float)
        return float(np.mean(vals < 0))

    def run_cpcv(self, feat_df: pd.DataFrame, strategy_fn: callable, n_jobs: int = 4) -> dict:
        _ = n_jobs  # reserved for future parallel implementation
        if feat_df is None or feat_df.empty or len(feat_df) < self.n_splits * 20:
            return {
                "oos_sharpe_mean": 0.0,
                "oos_sharpe_std": 0.0,
                "oos_sharpe_distribution": [],
                "pbo": 1.0,
                "dsr": -1.0,
                "is_sharpe": 0.0,
                "degradation_ratio": 0.0,
                "verdict": "FAIL",
                "n_combinations": 0,
                "equity_curves": [],
                "error": "insufficient_data",
            }

        n = len(feat_df)
        all_idx = np.arange(n)
        groups = [np.asarray(g, dtype=int) for g in np.array_split(all_idx, self.n_splits)]
        combos = list(combinations(range(self.n_splits), self.n_test_splits))

        oos_sharpes: list[float] = []
        is_sharpes: list[float] = []
        equity_curves: list = []

        for combo in combos:
            test_idx = np.concatenate([groups[i] for i in combo]) if combo else np.array([], dtype=int)
            train_idx = np.concatenate([groups[i] for i in range(self.n_splits) if i not in combo])
            train_idx = self._purge_embargo(train_idx, test_idx, self.embargo_pct)
            if train_idx.size < 30 or test_idx.size < 10:
                continue

            try:
                payload = strategy_fn(feat_df, train_idx, test_idx)
                is_ret, oos_ret, curves = self._extract_returns(payload)
                is_sr = self._annualized_sharpe(is_ret)
                oos_sr = self._annualized_sharpe(oos_ret)
                is_sharpes.append(is_sr)
                oos_sharpes.append(oos_sr)
                if curves:
                    equity_curves.append(curves)
                elif oos_ret.size:
                    eq = np.cumprod(1.0 + np.clip(oos_ret, -0.95, 10.0))
                    equity_curves.append([
                        {"time": str(feat_df.index[int(i)]), "value": float(v)}
                        for i, v in zip(test_idx[: len(eq)], eq)
                    ])
            except Exception as exc:
                logger.debug("CPCV combo failed: %s", exc)
                continue

        oos_mean = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        oos_std = float(np.std(oos_sharpes)) if oos_sharpes else 0.0
        is_mean = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        pbo = self._estimate_pbo(oos_sharpes)

        dsr_pack = ResearchValidator.deflated_sharpe(
            sharpe_obs=is_mean,
            n_trials=max(len(oos_sharpes), 1),
            n_obs=max(len(feat_df), 1),
        )
        dsr = float(dsr_pack.get("dsr", 0.0))
        degradation = float(oos_mean / is_mean) if abs(is_mean) > 1e-9 else 0.0

        if pbo < 0.3 and dsr > 0 and degradation > 0.5:
            verdict = "PASS"
        elif pbo <= 0.5 and dsr > -0.5 and degradation > 0.25:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        return {
            "oos_sharpe_mean": round(oos_mean, 4),
            "oos_sharpe_std": round(oos_std, 4),
            "oos_sharpe_distribution": [round(float(x), 4) for x in oos_sharpes],
            "pbo": round(float(pbo), 4),
            "dsr": round(float(dsr), 4),
            "is_sharpe": round(is_mean, 4),
            "degradation_ratio": round(degradation, 4),
            "verdict": verdict,
            "n_combinations": len(oos_sharpes),
            "equity_curves": equity_curves,
        }
