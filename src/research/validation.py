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
