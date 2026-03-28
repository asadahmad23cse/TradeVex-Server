"""
Scenario and Monte Carlo risk tools.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MonteCarloScenarioEngine:
    def __init__(self, n_paths: int = 1000, horizon_days: int = 5) -> None:
        self.n_paths = n_paths
        self.horizon_days = horizon_days

    def simulate_position_pnl(
        self,
        entry_price: float,
        position_size_pct: float,
        daily_vol: float,
        drift: float = 0.0,
    ) -> dict:
        if entry_price <= 0:
            return {"pnl_5pct": 0.0, "pnl_mean": 0.0, "pnl_std": 0.0}
        dt = 1 / 252
        rng = np.random.default_rng(42)
        shocks = rng.normal(size=(self.n_paths, self.horizon_days))
        log_paths = np.cumsum((drift - 0.5 * daily_vol ** 2) * dt + daily_vol * np.sqrt(dt) * shocks, axis=1)
        terminal_prices = entry_price * np.exp(log_paths[:, -1])
        pnl_pct = (terminal_prices - entry_price) / entry_price * position_size_pct
        return {
            "pnl_5pct": round(float(np.quantile(pnl_pct, 0.05)), 4),
            "pnl_mean": round(float(np.mean(pnl_pct)), 4),
            "pnl_std": round(float(np.std(pnl_pct)), 4),
        }
