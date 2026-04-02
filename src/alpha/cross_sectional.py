"""
GAP 1 — Cross-Sectional Alpha Model.

Ranks ALL assets in the watchlist simultaneously at each cycle by their
IC-weighted factor Z-scores. This is the industry-standard approach used
by AQR, Two Sigma, and Renaissance:

    "Don't ask if RELIANCE is bullish.
     Ask if RELIANCE is the MOST bullish stock right now."

Method:
    1. Compute raw factor Z-scores for each asset (from AlphaFactorModel)
    2. Rank-normalise per factor across the live universe →  [-1, +1]
    3. IC-weight the rank scores → cross-sectional alpha score per asset
    4. Blend with time-series score: final = 0.60 × CS + 0.40 × TS

The cross-sectional score is stronger for equity selection.
The time-series score is stronger for timing entry/exit.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

# Weight for cross-sectional vs time-series blend
CS_BLEND_WEIGHT = 0.60
TS_BLEND_WEIGHT = 0.40


class CrossSectionalAlpha:
    """
    Cross-sectional alpha computation across a universe of assets.

    Usage:
        cs = CrossSectionalAlpha()
        # Collect raw factor scores for each asset:
        cs.update("AAPL",    factor_scores={"F1": 0.5, "F2": -0.2, ...}, ic_weights={...})
        cs.update("TSLA",    factor_scores={...}, ic_weights={...})
        cs.update("NVDA",    factor_scores={...}, ic_weights={...})
        # Get cross-sectional score for one asset:
        cs_score = cs.get_score("AAPL")
        # Get blended score:
        final = cs.blend("AAPL", ts_alpha_score=0.35)
    """

    def __init__(
        self,
        factors: list[str] | None = None,
        cs_weight: float = CS_BLEND_WEIGHT,
        ts_weight: float = TS_BLEND_WEIGHT,
    ):
        self.factors = factors or ["F1", "F2", "F3", "F4", "F5"]
        self.cs_weight = cs_weight
        self.ts_weight = ts_weight
        # {asset: {"factor_scores": dict, "ic_weights": dict}}
        self._universe: dict[str, dict] = {}
        self._cs_scores: dict[str, float] = {}
        self._computed = False

    def update(
        self,
        asset: str,
        factor_scores: dict[str, float],
        ic_weights: dict[str, float],
    ) -> None:
        """Register or update an asset's factor scores in the universe."""
        self._universe[asset] = {
            "factor_scores": factor_scores,
            "ic_weights": ic_weights,
        }
        self._computed = False  # invalidate cached scores

    def compute_all(self) -> dict[str, float]:
        """
        Rank-normalise each factor across all assets and compute
        cross-sectional alpha scores for the entire universe.

        Returns
        -------
        dict[asset → cross-sectional alpha score in [-1, +1]]
        """
        assets = list(self._universe.keys())
        n = len(assets)

        if n < 2:
            # Need at least 2 assets for meaningful cross-sectional ranking
            self._cs_scores = {a: 0.0 for a in assets}
            self._computed = True
            return self._cs_scores

        cs_scores: dict[str, float] = {}

        # Matrix: assets × factors
        factor_matrix: dict[str, np.ndarray] = {}
        for f in self.factors:
            raw = np.array([
                self._universe[a]["factor_scores"].get(f, 0.0) for a in assets
            ])
            # Rank-normalise: percentile rank → [-1, +1]
            if raw.std() < 1e-9:
                factor_matrix[f] = np.zeros(n)
            else:
                ranks = rankdata(raw, method="average")   # 1..n
                factor_matrix[f] = (ranks - 1) / (n - 1) * 2 - 1  # → [-1, +1]

        # IC-weighted combination per asset
        for i, asset in enumerate(assets):
            ic_weights = self._universe[asset]["ic_weights"]
            # M1: floor calibrated to typical IC magnitude (0.02–0.06); 0.1 was too large
            denom = max(sum(abs(ic_weights.get(f, 0.0)) for f in self.factors), 0.01)

            cs_alpha = sum(
                ic_weights.get(f, 0.0) * factor_matrix[f][i]
                for f in self.factors
            ) / denom

            cs_scores[asset] = float(np.clip(cs_alpha, -1.0, 1.0))

        self._cs_scores = cs_scores
        self._computed = True

        logger.debug(
            "CrossSectional computed %d assets. Top: %s",
            n,
            sorted(cs_scores.items(), key=lambda x: x[1], reverse=True)[:3],
        )
        return cs_scores

    def get_score(self, asset: str) -> float:
        """Return pre-computed cross-sectional score for one asset."""
        if not self._computed:
            self.compute_all()
        return self._cs_scores.get(asset, 0.0)

    def blend(self, asset: str, ts_alpha_score: float) -> float:
        """
        Blend cross-sectional and time-series alpha scores.

        final = 0.60 × CS_score + 0.40 × TS_score
        Both CS and TS are in [-1, +1] after tanh-mapping.
        """
        cs_score = self.get_score(asset)
        blended = self.cs_weight * cs_score + self.ts_weight * np.tanh(ts_alpha_score)
        return float(np.clip(blended, -1.0, 1.0))

    def clear(self) -> None:
        """Reset the universe (call at start of each intraday cycle)."""
        self._universe.clear()
        self._cs_scores.clear()
        self._computed = False

    def universe_size(self) -> int:
        return len(self._universe)

    def get_rankings(self) -> list[tuple[str, float]]:
        """Return sorted list of (asset, cs_score) descending."""
        if not self._computed:
            self.compute_all()
        return sorted(self._cs_scores.items(), key=lambda x: x[1], reverse=True)
