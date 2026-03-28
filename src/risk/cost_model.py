"""
GAP 2 — Transaction Cost Model (Almgren-Chriss simplified).

Every signal is evaluated net of estimated transaction cost.
If net_alpha < cost, the signal is discarded before execution.

Cost = slippage_pct + market_impact_pct + half_spread

Market Impact (Almgren-Chriss simplified):
    impact = η × σ × √(Q / ADV)
    where:
        η    = market impact coefficient (default 0.6)
        σ    = daily return volatility
        Q    = trade size as fraction of capital
        ADV  = 30-day average daily volume turnover ratio (proxy)

Usage:
    cm = CostModel(config)
    cost = cm.estimate(entry_price, position_size_pct, daily_vol, volume_ratio)
    net_alpha = alpha_score - cost
    if net_alpha < 0: discard signal
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CostModel:
    """
    Estimates all-in transaction cost as a fraction of notional.

    Parameters (from config.yaml `cost_model` section):
        slippage:           per-asset-class base slippage (already in config)
        market_impact_coefficient:  η in Almgren-Chriss (default 0.6)
        half_spread:        assumed bid-ask half-spread per class (default per config)
    """

    # Default half-spreads (as fraction of price) if not in config
    HALF_SPREAD_DEFAULTS = {
        "indian_stock": 0.0002,   # ~0.02% for large-cap NSE
        "us_stock":     0.0001,   # ~0.01% for NYSE/NASDAQ large-cap
        "forex":        0.00005,  # Major forex pairs
    }

    # Default slippage if not in config
    SLIPPAGE_DEFAULTS = {
        "indian_stock": 0.0003,
        "us_stock":     0.0001,
        "forex":        0.00005,
    }

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.eta = cfg.get("market_impact_coefficient", 0.6)
        self._slippage_cfg = cfg.get("slippage", {})
        self._spread_cfg = cfg.get("half_spread", {})
        self._sideways_spread_mult = cfg.get("sideways_spread_multiplier", 1.5)
        self._low_liquidity_spread_mult = cfg.get("low_liquidity_spread_multiplier", 2.0)

    def estimate(
        self,
        asset_class: str,
        position_size_pct: float,    # as percent (e.g. 2.0 = 2%)
        daily_vol: float,            # annualised vol as decimal (e.g. 0.25 = 25%)
        volume_ratio: float = 1.0,   # current vol / 30d avg vol (proxy for ADV participation)
        regime: str = "SIDEWAYS",
        low_liquidity: bool = False,
    ) -> float:
        """
        Estimate total transaction cost as % of notional.

        Returns
        -------
        float : total cost percentage (e.g. 0.15 = 0.15% of trade value)
        """
        # 1. Slippage
        slippage = self._slippage_cfg.get(
            asset_class,
            self.SLIPPAGE_DEFAULTS.get(asset_class, 0.0003)
        )

        # 2. Half bid-ask spread
        half_spread = self._spread_cfg.get(
            asset_class,
            self.HALF_SPREAD_DEFAULTS.get(asset_class, 0.0002)
        )
        if regime == "SIDEWAYS":
            half_spread *= self._sideways_spread_mult
        if low_liquidity:
            half_spread *= self._low_liquidity_spread_mult

        # 3. Almgren-Chriss market impact
        # σ_daily = annualised vol / sqrt(252)
        sigma_daily = daily_vol / np.sqrt(252)
        # Q = position size fraction (e.g. 0.02 for 2%)
        q = position_size_pct / 100.0
        # Participation = q / volume_ratio (capped to avoid extreme values)
        participation = min(q / max(volume_ratio, 0.01), 0.5)
        market_impact = self.eta * sigma_daily * np.sqrt(participation)

        total_cost = slippage + half_spread + market_impact

        logger.debug(
            "CostModel [%s]: slippage=%.5f spread=%.5f impact=%.5f → total=%.5f (%.4f%%)",
            asset_class, slippage, half_spread, market_impact, total_cost, total_cost * 100
        )

        return round(total_cost * 100, 5)  # return as percent

    def net_alpha(
        self,
        alpha_score: float,
        asset_class: str,
        position_size_pct: float,
        daily_vol: float,
        volume_ratio: float = 1.0,
        regime: str = "SIDEWAYS",
        low_liquidity: bool = False,
    ) -> tuple[float, float, bool]:
        """
        Compute net alpha after transaction costs.

        Returns
        -------
        net_alpha_score : float  — alpha after cost deduction
        cost_pct       : float  — estimated cost %
        is_viable      : bool   — True if net alpha > 0
        """
        cost = self.estimate(
            asset_class,
            position_size_pct,
            daily_vol,
            volume_ratio,
            regime=regime,
            low_liquidity=low_liquidity,
        )
        # alpha_score in [-1, 1]; cost in percent scaled to same order
        # Map cost to alpha units: 1% cost ≈ 0.01 alpha deduction
        cost_in_alpha_units = cost / 100.0
        net = alpha_score - cost_in_alpha_units
        return round(net, 4), cost, net > 0

    def update_eta(self, eta: float) -> None:
        self.eta = max(float(eta), 0.0)
