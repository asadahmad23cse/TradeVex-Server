"""
Layer 6 — Kelly Criterion Position Sizing.

Uses empirical win probability from the SQLite signal history.
Cold-start rule: if < 30 closed trades in the bucket, use 2% flat.

f* = (p × b − q) / b
  p = empirical win rate from (asset_class, signal, strength) bucket
  q = 1 - p
  b = avg_win / avg_loss realized R:R  (falls back to signal R:R if < 30 trades)

Fractional Kelly: position = 0.25 × f*   (quarter-Kelly, Ed Thorp standard)
Hard cap: min(position, 0.05)            (never > 5% per trade)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Configuration
KELLY_FRACTION = 0.25       # quarter-Kelly multiplier
MAX_POSITION_PCT = 0.05     # 5% hard cap
COLD_START_PCT = 0.02       # 2% flat until 30 trades
MIN_TRADES_REQUIRED = 30    # minimum closed trades for empirical Kelly

# R:R ratios by strength (used as fallback b when < 30 trades)
FALLBACK_RR = {
    "STRONG": 3.0,
    "MODERATE": 2.5,
    "WEAK": 2.0,
}


class KellyCalculator:

    def __init__(
        self,
        kelly_fraction: float = KELLY_FRACTION,
        max_position_pct: float = MAX_POSITION_PCT,
        cold_start_pct: float = COLD_START_PCT,
        min_trades: int = MIN_TRADES_REQUIRED,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.cold_start_pct = cold_start_pct
        self.min_trades = min_trades

    def compute(
        self,
        asset_class: str,
        signal: str,           # 'BUY' | 'SELL'
        strength: str,         # 'STRONG' | 'MODERATE' | 'WEAK'
        regime: str,           # 'BULL' | 'BEAR' | 'SIDEWAYS'
        bucket_stats: dict,    # from SignalStore.get_bucket_stats()
    ) -> dict:
        """
        Compute position size.

        Returns
        -------
        dict with keys:
          kelly_fraction : float   — raw f* before capping
          position_size_pct : float — final capped position (as decimal 0-1)
          method : str             — 'empirical' | 'cold_start'
          p, b, q : floats         — Kelly inputs used
        """
        total = bucket_stats.get("total", 0)

        # Cold start: < 30 trades in bucket
        if total < self.min_trades:
            logger.debug(
                "Cold start for (%s, %s, %s): %d trades < %d required",
                asset_class, regime, strength, total, self.min_trades,
            )
            return {
                "kelly_fraction": self.cold_start_pct,
                "position_size_pct": self.cold_start_pct,
                "method": "cold_start",
                "p": 0.5,
                "q": 0.5,
                "b": FALLBACK_RR.get(strength, 2.0),
            }

        # Empirical Kelly
        p = float(bucket_stats["win_rate"] or 0.5)
        q = 1.0 - p

        avg_win = bucket_stats.get("avg_win") or 0.0
        avg_loss = bucket_stats.get("avg_loss") or 1.0

        if avg_loss == 0:
            b = FALLBACK_RR.get(strength, 2.0)
        else:
            b = avg_win / avg_loss

        # Kelly formula
        raw_f = (p * b - q) / b if b > 0 else 0.0
        raw_f = max(raw_f, 0.0)   # never go negative (never short more than 0)

        # Fractional Kelly
        fractional_f = self.kelly_fraction * raw_f

        # Hard cap
        position = min(fractional_f, self.max_position_pct)

        logger.debug(
            "Kelly (%s/%s/%s): p=%.2f q=%.2f b=%.2f → f*=%.4f → capped=%.4f",
            asset_class, regime, strength, p, q, b, raw_f, position,
        )

        return {
            "kelly_fraction": round(raw_f * 100, 2),    # percent
            "position_size_pct": round(position * 100, 2),  # percent
            "method": "empirical",
            "p": round(p, 4),
            "q": round(q, 4),
            "b": round(b, 4),
        }


class PortfolioRiskGuard:
    """
    Portfolio-level risk controls.
    Checks correlation limits, drawdown circuit breaker,
    max positions, class-level exposure caps, and cross-asset correlation.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.max_daily_dd_pct = cfg.get("max_daily_drawdown_pct", 5.0)
        self.max_positions = cfg.get("max_open_positions", 8)
        self.corr_threshold = cfg.get("correlation_threshold", 0.7)
        self.max_indian_pct = cfg.get("max_indian_exposure_pct", 40.0)
        self.max_us_pct = cfg.get("max_us_exposure_pct", 40.0)
        self.max_forex_pct = cfg.get("max_forex_exposure_pct", 20.0)
        self.var_limit_pct = cfg.get("var_limit_pct", 3.0)
        self.cvar_limit_pct = cfg.get("cvar_limit_pct", 4.0)
        self.bear_corr_threshold = cfg.get("bear_correlation_threshold", 0.5)

    def check_all(
        self,
        signal,                          # TradingSignal
        open_positions: list,            # list of open TradingSignal dicts
        daily_pnl_pct: float,
        portfolio_var_pct: float = 0.0,
        portfolio_cvar_pct: float = 0.0,
        regime: str = "SIDEWAYS",
        price_history: dict | None = None,  # {asset: pd.Series of recent closes}
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).

        price_history: optional dict mapping asset symbol → pd.Series of daily
        closing prices (at least 30 bars). Used for the correlation filter.
        If not provided, the correlation check is skipped.
        """
        # 1. Drawdown circuit breaker
        if daily_pnl_pct <= -self.max_daily_dd_pct:
            return False, f"Daily drawdown circuit breaker: {daily_pnl_pct:.2f}%"

        # 2. Max open positions
        if len(open_positions) >= self.max_positions:
            return False, f"Max open positions ({self.max_positions}) reached"

        if portfolio_var_pct > self.var_limit_pct:
            return False, (
                f"Portfolio VaR limit breached: {portfolio_var_pct:.2f}% > "
                f"{self.var_limit_pct:.2f}%"
            )
        if portfolio_cvar_pct > self.cvar_limit_pct:
            return False, (
                f"Portfolio CVaR limit breached: {portfolio_cvar_pct:.2f}% > "
                f"{self.cvar_limit_pct:.2f}%"
            )

        if any(p.get("asset") == signal.asset for p in open_positions):
            return False, f"Asset already has an open position: {signal.asset}"

        # 3. Class-level exposure
        class_total = sum(
            p.get("position_size_pct", 0)
            for p in open_positions
            if p.get("asset_class") == signal.asset_class
        )
        limit = {
            "indian_stock": self.max_indian_pct,
            "us_stock": self.max_us_pct,
            "forex": self.max_forex_pct,
        }.get(signal.asset_class, 100.0)

        if class_total + signal.position_size_pct > limit:
            return False, (
                f"Class exposure limit: {signal.asset_class} would reach "
                f"{class_total + signal.position_size_pct:.1f}% > {limit}%"
            )

        # 4. Cross-asset correlation filter (spec §8)
        # Block if any open position has rolling 30d |correlation| > 0.7
        if price_history and signal.asset in price_history:
            new_prices = price_history[signal.asset]
            corr_threshold = self.bear_corr_threshold if regime in {"BEAR", "HIGH_VOL_BEAR"} else self.corr_threshold
            for pos in open_positions:
                pos_asset = pos.get("asset", "")
                if pos_asset == signal.asset or pos_asset not in price_history:
                    continue
                pos_prices = price_history[pos_asset]
                # Align on common dates and take last 30 observations
                try:
                    aligned = pd.concat(
                        [new_prices.rename("new"), pos_prices.rename("pos")],
                        axis=1,
                    ).dropna().tail(30)
                    if len(aligned) >= 10:  # need at least 10 points
                        corr = float(aligned["new"].corr(aligned["pos"]))
                        if abs(corr) > corr_threshold:
                            return False, (
                                f"Correlation filter: {signal.asset} ↔ {pos_asset} "
                                f"corr={corr:.2f} exceeds threshold {corr_threshold}"
                            )
                except Exception:
                    pass  # skip correlation check on error

        return True, "OK"

    # ------------------------------------------------------------------
    # Step 8 — Factor Crowding Risk Detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_crowding(
        signal,
        open_positions: list,
        crowding_threshold: float = 0.6,
    ) -> tuple[float, str]:
        """
        Detect factor crowding: too many assets in the same direction.

        If > 60% of open positions share the same signal direction,
        the new signal's position size should be scaled down.

        Parameters
        ----------
        signal              : TradingSignal being evaluated
        open_positions      : list of open position dicts
        crowding_threshold  : fraction threshold (default 0.6 = 60%)

        Returns
        -------
        (scale_factor, reason) :
            scale_factor = 1.0 if no crowding, 0.5 if crowded
            reason = description string
        """
        if len(open_positions) < 3:
            return 1.0, "OK (not enough positions to detect crowding)"

        # Count same-direction positions in same asset class
        same_dir_class = sum(
            1 for p in open_positions
            if p.get("signal") == signal.signal
            and p.get("asset_class") == signal.asset_class
        )
        total_class = sum(
            1 for p in open_positions
            if p.get("asset_class") == signal.asset_class
        )
        total_class = max(total_class, 1)

        crowding_ratio = same_dir_class / total_class

        if crowding_ratio > crowding_threshold:
            return 0.5, (
                f"CROWDING: {crowding_ratio:.0%} of {signal.asset_class} positions "
                f"are {signal.signal} — scaling position 50%"
            )

        # Also check across all classes
        same_dir_all = sum(
            1 for p in open_positions
            if p.get("signal") == signal.signal
        )
        global_ratio = same_dir_all / max(len(open_positions), 1)

        if global_ratio > 0.75:
            return 0.5, (
                f"GLOBAL CROWDING: {global_ratio:.0%} of all positions "
                f"are {signal.signal} — scaling position 50%"
            )

        return 1.0, "OK"
