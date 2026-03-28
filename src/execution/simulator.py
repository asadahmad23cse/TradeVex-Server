"""
Step 2 — Execution Simulator (Microstructure + Latency).

Provides REALISTIC execution modeling that separates retail from pro:

    1. Latency Simulation
       Real orders take 100ms–2000ms to reach exchange.
       Price moves during this window → random walk slippage.

    2. Market Impact (Almgren-Chriss with temporal decay)
       Permanent impact = γ × σ × (Q/ADV)^0.6
       Temporary impact = η × σ × (Q/ADV)^0.5 / T^0.5
       Total impact = permanent + temporary

    3. Partial Fills
       If order_size > available_liquidity → partial fill.
       Fill ratio = min(1, liquidity / order_size)
       Remaining is queued or cancelled.

    4. Order Queue Position
       Limit orders don't fill instantly.
       Queue position → fill probability based on volume.

    5. Geometric Brownian Motion (GBM) Tick Simulation
       Approximate intra-bar price path for realistic SL/TP checking.

Usage:
    sim = ExecutionSimulator(config)
    fill = sim.simulate_fill(price=100.0, volatility=0.25, order_size=50000, adv=5_000_000)
    # fill = SimulatedFill(fill_price=100.03, delay_ms=340, fill_ratio=1.0, ...)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulatedFill:
    """Result of a simulated execution."""
    original_price: float = 0.0
    fill_price: float = 0.0
    delay_ms: float = 0.0
    slippage_pct: float = 0.0
    market_impact_pct: float = 0.0
    fill_ratio: float = 1.0        # 1.0 = fully filled, < 1.0 = partial
    filled_quantity: float = 0.0
    unfilled_quantity: float = 0.0
    queue_position: float = 0.0
    fill_probability: float = 1.0
    is_filled: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_cost_pct(self) -> float:
        return self.slippage_pct + self.market_impact_pct


class ExecutionSimulator:
    """
    Realistic execution simulator with latency, impact, and partial fills.

    Parameters (from config `execution_sim` section):
        latency_min_ms:    minimum latency in ms (default 50)
        latency_max_ms:    maximum latency in ms (default 1500)
        permanent_gamma:   permanent impact coefficient (default 0.1)
        temporary_eta:     temporary impact coefficient (default 0.6)
        adv_default:       default avg daily volume in currency units (default 10M)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.latency_min = cfg.get("latency_min_ms", 50)
        self.latency_max = cfg.get("latency_max_ms", 1500)
        self.gamma = cfg.get("permanent_gamma", 0.1)     # permanent impact
        self.eta = cfg.get("temporary_eta", 0.6)          # temporary impact
        self.adv_default = cfg.get("adv_default", 10_000_000)

    def simulate_fill(
        self,
        price: float,
        volatility: float,
        order_size: float = 0.0,
        adv: float = 0.0,
        side: str = "BUY",
        order_type: str = "MARKET",
        limit_price: float = 0.0,
    ) -> SimulatedFill:
        """
        Simulate a realistic order fill.

        Parameters
        ----------
        price       : current mid price
        volatility  : annualised volatility (e.g. 0.25 = 25%)
        order_size  : notional order size in currency
        adv         : average daily volume in currency (0 = use default)
        side        : 'BUY' or 'SELL'
        order_type  : 'MARKET' or 'LIMIT'
        limit_price : limit price for LIMIT orders (ignored for MARKET)

        Returns
        -------
        SimulatedFill with all execution details
        """
        adv = adv or self.adv_default
        participation = order_size / max(adv, 1)

        # 1. Latency simulation
        delay_ms = np.random.uniform(self.latency_min, self.latency_max)

        # 2. Random walk slippage during latency window
        sigma_per_ms = volatility / np.sqrt(252 * 6.5 * 60 * 60 * 1000)
        random_walk = np.random.normal(0, sigma_per_ms * np.sqrt(delay_ms))
        latency_slip = abs(random_walk)

        # 3. Market impact (Almgren-Chriss)
        sigma_daily = volatility / np.sqrt(252)
        # Permanent impact (stays after execution)
        perm_impact = self.gamma * sigma_daily * (participation ** 0.6)
        # Temporary impact (reverts after execution)
        temp_impact = self.eta * sigma_daily * np.sqrt(participation)
        total_impact = perm_impact + temp_impact

        # 4. Partial fill check
        fill_ratio, queue_pos, fill_prob = self._partial_fill(
            order_size, adv, order_type, price, limit_price, side
        )

        # 5. Compute fill price
        direction = 1.0 if side == "BUY" else -1.0
        total_slip = latency_slip + total_impact
        fill_price = price * (1.0 + direction * total_slip)

        # Limit order constraint: if limit price is worse than fill, reject
        if order_type == "LIMIT" and limit_price > 0:
            if side == "BUY" and fill_price > limit_price:
                fill_ratio = 0.0  # wouldn't fill at this price
            elif side == "SELL" and fill_price < limit_price:
                fill_ratio = 0.0

        is_filled = fill_ratio > 0 and np.random.random() < fill_prob
        filled_qty = order_size * fill_ratio if is_filled else 0.0

        result = SimulatedFill(
            original_price=price,
            fill_price=round(fill_price, 6) if is_filled else 0.0,
            delay_ms=round(delay_ms, 1),
            slippage_pct=round(latency_slip * 100, 6),
            market_impact_pct=round(total_impact * 100, 6),
            fill_ratio=round(fill_ratio, 4) if is_filled else 0.0,
            filled_quantity=round(filled_qty, 2),
            unfilled_quantity=round(order_size - filled_qty, 2),
            queue_position=round(queue_pos, 4),
            fill_probability=round(fill_prob, 4),
            is_filled=is_filled,
        )

        logger.debug(
            "ExecSim %s %s: price=%.4f → fill=%.4f (delay=%dms slip=%.4f%% impact=%.4f%% fill_ratio=%.2f)",
            side, order_type, price, result.fill_price, delay_ms,
            result.slippage_pct, result.market_impact_pct, result.fill_ratio,
        )
        return result

    def _partial_fill(
        self,
        order_size: float,
        adv: float,
        order_type: str,
        price: float,
        limit_price: float,
        side: str,
    ) -> tuple[float, float, float]:
        """
        Compute fill ratio, queue position, and fill probability.

        Returns: (fill_ratio, queue_position, fill_probability)
        """
        # Participation rate
        participation = order_size / max(adv, 1)

        # Fill ratio: how much of the order can be filled
        fill_ratio = min(1.0, 1.0 / (1.0 + participation * 10))

        if order_type == "LIMIT":
            # Queue position: random 0-1 (how deep in the book)
            queue_pos = np.random.uniform(0.0, 1.0)
            # Fill probability depends on queue position and volume
            fill_prob = min(1.0, (1.0 - queue_pos) * (adv / max(order_size, 1)) * 0.1)
        else:
            # Market orders always fill (but maybe partially)
            queue_pos = 0.0
            fill_prob = 1.0

        return fill_ratio, queue_pos, fill_prob

    def simulate_gbm_path(
        self,
        price: float,
        volatility: float,
        n_ticks: int = 100,
        dt_seconds: float = 60.0,
    ) -> np.ndarray:
        """
        Simulate an intra-bar price path using Geometric Brownian Motion.

        Used for realistic SL/TP checking within a single bar.

        GBM: dS = μSdt + σSdW
        Discrete: S_{t+1} = S_t × exp((μ - σ²/2)dt + σ√dt × Z)

        Parameters
        ----------
        price       : starting price
        volatility  : annualised volatility
        n_ticks     : number of simulated ticks
        dt_seconds  : time between ticks in seconds

        Returns
        -------
        np.ndarray of shape (n_ticks,) with simulated prices
        """
        dt = dt_seconds / (252 * 6.5 * 3600)  # convert to annual fraction
        mu = 0.0  # drift = 0 for short horizon
        sigma = volatility

        Z = np.random.normal(0, 1, n_ticks)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_prices = np.cumsum(log_returns)
        prices = price * np.exp(log_prices)

        return prices

    def check_sl_tp_intrabar(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volatility: float,
        side: str = "BUY",
        n_ticks: int = 100,
    ) -> dict:
        """
        Simulate intra-bar price path and check if SL or TP is hit.

        Returns
        -------
        dict with keys: hit ('SL' | 'TP' | None), exit_price, tick_index
        """
        path = self.simulate_gbm_path(entry_price, volatility, n_ticks)

        for i, p in enumerate(path):
            if side == "BUY":
                if p <= stop_loss:
                    return {"hit": "SL", "exit_price": float(p), "tick_index": i}
                if p >= take_profit:
                    return {"hit": "TP", "exit_price": float(p), "tick_index": i}
            else:
                if p >= stop_loss:
                    return {"hit": "SL", "exit_price": float(p), "tick_index": i}
                if p <= take_profit:
                    return {"hit": "TP", "exit_price": float(p), "tick_index": i}

        return {"hit": None, "exit_price": float(path[-1]), "tick_index": n_ticks}
