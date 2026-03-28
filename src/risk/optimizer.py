"""
GAP 3 — Ledoit-Wolf Covariance + Hierarchical Risk Parity (HRP) Optimizer.

Replaces ad-hoc Kelly sizing at the portfolio level with a mathematically
rigorous allocation that minimises total portfolio variance.

Steps:
    1. Collect 60-day daily returns for all open/candidate assets.
    2. Estimate covariance matrix using Ledoit-Wolf shrinkage (sklearn).
       LW is much more stable than sample covariance for small N, large T.
    3. Run HRP:
       a. Cluster assets by Ward linkage on the correlation distance matrix.
       b. Recursively bisect the dendrogram, allocating risk budget inversely
          proportional to cluster variance (Marcos Lopez de Prado method).
    4. Return per-asset target weight in [0, max_position_pct].

HRP is preferred over Mean-Variance Optimization because:
    - No need to forecast expected returns (notoriously unstable)
    - Robust to estimation error in the covariance matrix
    - Naturally diversified, no corner solutions

Usage:
    opt = HRPOptimizer(config)
    weights = opt.compute(price_history_dict)
    # weights = {"AAPL": 0.04, "TSLA": 0.02, "RELIANCE": 0.03, ...}
"""

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

try:
    from sklearn.covariance import LedoitWolf
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("sklearn not installed — falling back to sample covariance.")


class HRPOptimizer:
    """
    Hierarchical Risk Parity portfolio optimizer.

    Parameters (from config.yaml `optimizer` section):
        lookback_days:      number of daily bars to use for cov estimation (default 60)
        max_position_pct:   hard cap per asset in percent (default 5.0)
        min_position_pct:   minimum allocation per asset in percent (default 0.5)
        shrinkage_method:   'ledoit_wolf' | 'sample' (default 'ledoit_wolf')
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.lookback = cfg.get("lookback_days", 60)
        self.max_pct = cfg.get("max_position_pct", 5.0) / 100.0
        self.min_pct = cfg.get("min_position_pct", 0.5) / 100.0
        self.method = cfg.get("shrinkage_method", "ledoit_wolf")

    def compute(self, price_history: dict[str, pd.Series]) -> dict[str, float]:
        """
        Compute HRP portfolio weights.

        Parameters
        ----------
        price_history : dict[asset → pd.Series of daily closes]
            Each series should have at least `lookback_days` bars.

        Returns
        -------
        dict[asset → weight as decimal (e.g. 0.04 = 4%)]
        """
        if len(price_history) < 2:
            logger.warning("HRP: need at least 2 assets. Returning equal weights.")
            return {a: self.max_pct for a in price_history}

        # Build returns matrix
        returns_df = pd.DataFrame({
            a: s.pct_change().dropna()
            for a, s in price_history.items()
        }).dropna()

        if len(returns_df) < 10:
            logger.warning("HRP: insufficient return history (%d rows). Equal weights.", len(returns_df))
            w = min(self.max_pct, 1.0 / len(price_history))
            return {a: w for a in price_history}

        # Use last lookback bars
        returns_df = returns_df.tail(self.lookback)
        assets = list(returns_df.columns)

        # Covariance estimation
        cov_matrix = self._estimate_cov(returns_df.values)
        cov_df = pd.DataFrame(cov_matrix, index=assets, columns=assets)
        corr_df = self._cov_to_corr(cov_df)

        # HRP
        weights = self._hrp(corr_df, cov_df)

        # Apply position limits
        for a in weights:
            weights[a] = float(np.clip(weights[a], self.min_pct, self.max_pct))

        # Normalise so sum ≤ 1.0 (never over-invest)
        total = sum(weights.values())
        if total > 1.0:
            weights = {a: w / total for a, w in weights.items()}

        logger.info(
            "HRP allocation (%d assets): %s",
            len(weights),
            {a: f"{w*100:.2f}%" for a, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)},
        )
        return weights

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_cov(self, returns: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix using Ledoit-Wolf shrinkage."""
        if _SKLEARN_AVAILABLE and self.method == "ledoit_wolf":
            lw = LedoitWolf().fit(returns)
            return lw.covariance_
        return np.cov(returns.T)

    @staticmethod
    def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
        """Convert covariance matrix to correlation matrix."""
        std = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

    def _hrp(self, corr: pd.DataFrame, cov: pd.DataFrame) -> dict[str, float]:
        """
        Hierarchical Risk Parity allocation.

        1. Build correlation-distance matrix: d = sqrt((1 - corr) / 2)
        2. Cluster with Ward linkage
        3. Recursive bisection: allocate risk budget ∝ 1/variance(cluster)
        """
        dist = np.sqrt((1 - corr.values) / 2.0)
        np.fill_diagonal(dist, 0.0)
        # Ensure symmetry and non-negative
        dist = (dist + dist.T) / 2.0
        dist = np.clip(dist, 0.0, 1.0)

        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="ward")

        assets = list(corr.index)
        n = len(assets)

        # Map asset name → index
        asset_idx = {a: i for i, a in enumerate(assets)}

        # Get sorted items from dendrogram
        sorted_items = self._get_sorted_items(link, n)
        sorted_assets = [assets[i] for i in sorted_items]

        # Recursive bisection
        weights = {a: 1.0 / n for a in assets}
        cluster_items = [sorted_assets]

        while len(cluster_items) > 0:
            cluster_items = [
                sub
                for item in cluster_items
                for sub in self._bisect(item)
                if len(sub) > 0
            ]
            for sub in cluster_items:
                if len(sub) <= 1:
                    continue
                left  = sub[: len(sub) // 2]
                right = sub[len(sub) // 2:]

                var_l = self._cluster_var(left, cov)
                var_r = self._cluster_var(right, cov)
                alpha = var_r / (var_l + var_r + 1e-12)  # left gets alpha, right gets 1-alpha

                for a in left:
                    weights[a] *= alpha
                for a in right:
                    weights[a] *= (1 - alpha)

        return weights

    @staticmethod
    def _bisect(items: list) -> list[list]:
        """Split a list into two halves."""
        mid = len(items) // 2
        if mid == 0:
            return [items]
        return [items[:mid], items[mid:]]

    @staticmethod
    def _cluster_var(cluster_assets: list[str], cov: pd.DataFrame) -> float:
        """Compute variance of an equally-weighted cluster portfolio."""
        if not cluster_assets:
            return 1e-12
        n = len(cluster_assets)
        w = np.array([1.0 / n] * n)
        sub_cov = cov.loc[cluster_assets, cluster_assets].values
        return float(w @ sub_cov @ w)

    @staticmethod
    def _get_sorted_items(link: np.ndarray, n: int) -> list[int]:
        """Return leaf indices in dendrogram order (left-to-right)."""
        tree = to_tree(link)

        def _recurse(node):
            if node.is_leaf():
                return [node.id]
            return _recurse(node.left) + _recurse(node.right)

        return _recurse(tree)
