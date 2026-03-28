"""
Step 1 — Expected Returns Module.

Core philosophy upgrade:
    ❌ OLD: Signal → Risk → Portfolio
    ✅ NEW: Expected Returns → Portfolio Optimization → Execution

Every alpha score is converted to an expected return forecast:
    E[R] = α_score × (confidence / 100) × regime_mult × vol_target / realised_vol

This expected return is then fed into the HRP optimizer as an
alpha-overlay, producing returns-first portfolio weights instead of
variance-only weights.

The key insight from AQR / BlackRock:
    "You don't trade signals. You trade expected returns."
    The optimizer should KNOW the returns forecast, not just
    minimise variance blindly.

Usage:
    er = ExpectedReturns(config)
    forecasts = er.compute(alpha_results, daily_vols)
    # forecasts = {"AAPL": 0.012, "TSLA": -0.008, ...}
    # Feed into optimizer: weights = optimizer.compute(prices, forecasts)
"""

import logging
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)

# Target annualised volatility (12% is typical for a diversified quant fund)
DEFAULT_VOL_TARGET = 0.12


class ExpectedReturns:
    """
    Convert alpha model output → annualised expected return per asset.

    Parameters (from config `expected_returns` section):
        vol_target:     annualised vol target (default 0.12 = 12%)
        shrinkage:      blend towards 0 to penalise uncertainty (default 0.3)
        horizon_days:   forecast horizon in trading days (default 5)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.vol_target = cfg.get("vol_target", DEFAULT_VOL_TARGET)
        self.shrinkage = cfg.get("shrinkage", 0.3)
        self.horizon_days = cfg.get("horizon_days", 5)

    def compute_single(
        self,
        alpha_score: float,
        confidence: float,
        daily_vol: float,
        regime: str = "SIDEWAYS",
    ) -> float:
        """
        Convert a single alpha result to an expected return.

        Expected return formula:
            raw = α × (confidence / 100)
            vol_scaled = raw × (vol_target / max(realised_vol, 0.01))
            regime_adj = vol_scaled × regime_mult
            shrunk = regime_adj × (1 - shrinkage) + 0 × shrinkage
            annualised = shrunk × √(252 / horizon_days)

        Parameters
        ----------
        alpha_score : float    [-1, +1]  from alpha model
        confidence  : float    [0, 100]  from alpha model
        daily_vol   : float    annualised vol (e.g. 0.25 = 25%)
        regime      : str      'BULL', 'BEAR', 'SIDEWAYS'

        Returns
        -------
        float : annualised expected return (e.g. 0.08 = 8% pa)
        """
        # Regime multiplier
        regime_mult = {"BULL": 1.2, "BEAR": 1.2, "SIDEWAYS": 0.6}.get(regime, 0.8)

        # Raw expected return
        raw = alpha_score * (confidence / 100.0)

        # Vol-scale: target vol relative to realised vol
        vol_ratio = self.vol_target / max(daily_vol, 0.01)
        vol_scaled = raw * min(vol_ratio, 3.0)  # cap at 3× to prevent extreme leverage

        # Regime adjustment
        regime_adj = vol_scaled * regime_mult

        # Bayesian shrinkage toward zero (penalise low-confidence forecasts)
        shrunk = regime_adj * (1.0 - self.shrinkage)

        # Annualise: scale from horizon_days to 252
        annualised = shrunk * np.sqrt(252.0 / self.horizon_days)

        return float(np.clip(annualised, -0.50, 0.50))  # cap at ±50% annualised

    def compute_universe(
        self,
        alpha_results: dict[str, dict],
        daily_vols: dict[str, float],
        regimes: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """
        Compute expected returns for all assets in the universe.

        Parameters
        ----------
        alpha_results : {asset: {"alpha_score": ..., "confidence": ..., ...}}
        daily_vols    : {asset: annualised_vol}
        regimes       : {asset: "BULL"/"BEAR"/"SIDEWAYS"} (optional)

        Returns
        -------
        dict[asset → expected annualised return]
        """
        regimes = regimes or {}
        forecasts = {}

        for asset, result in alpha_results.items():
            alpha = result.get("alpha_score", 0.0)
            conf = result.get("confidence", 50.0)
            vol = daily_vols.get(asset, 0.20)
            regime = regimes.get(asset, "SIDEWAYS")

            er = self.compute_single(alpha, conf, vol, regime)
            forecasts[asset] = er

        logger.debug(
            "Expected returns computed for %d assets. Top: %s",
            len(forecasts),
            sorted(forecasts.items(), key=lambda x: abs(x[1]), reverse=True)[:3],
        )
        return forecasts

    def blend_with_market(
        self,
        forecasts: dict[str, float],
        market_return: float = 0.08,
        blend_weight: float = 0.7,
    ) -> dict[str, float]:
        """
        Blend alpha-based forecasts with market equilibrium return
        (Black-Litterman inspired).

        final = blend_weight × alpha_forecast + (1 - blend_weight) × market_return
        """
        return {
            asset: blend_weight * er + (1 - blend_weight) * market_return
            for asset, er in forecasts.items()
        }
