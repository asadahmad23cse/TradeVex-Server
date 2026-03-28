"""
Hurst Exponent via Rescaled Range (R/S) Analysis.

H < 0.45 → mean-reverting series  → weight Mean-Reversion factor higher
H > 0.55 → trending series        → weight Momentum factor higher
0.45 ≤ H ≤ 0.55 → random walk    → halve position size, lower conviction

This is computed on daily close prices (EOD loop only) and cached
for the full next trading session — R/S is too expensive to run every 5 min.
"""

import numpy as np
import pandas as pd


def _rs_single(series: np.ndarray) -> float:
    """
    Compute R/S statistic for a single sub-period.
    series: 1-D array of log returns.
    """
    n = len(series)
    if n < 4:
        return np.nan
    mean = np.mean(series)
    deviation = np.cumsum(series - mean)
    r = np.max(deviation) - np.min(deviation)
    s = np.std(series, ddof=1)
    if s == 0:
        return np.nan
    return r / s


def compute_hurst(prices: pd.Series, min_window: int = 10) -> float:
    """
    Estimate Hurst Exponent via R/S analysis on a price series.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices (at least 20 data points recommended).
    min_window : int
        Minimum sub-period length for R/S calculation.

    Returns
    -------
    float
        Hurst exponent H in [0, 1].
        Returns 0.5 (random walk default) if estimation fails.
    """
    prices = pd.Series(prices).dropna()
    n = len(prices)
    if n < 20:
        return 0.5

    log_returns = np.log(prices / prices.shift(1)).dropna().values

    # Build a range of sub-period lengths (logarithmically spaced)
    max_window = len(log_returns) // 2
    if max_window < min_window:
        return 0.5

    lags = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=20,
            dtype=int
        )
    )

    rs_values = []
    valid_lags = []

    for lag in lags:
        # Divide the return series into non-overlapping windows of length lag
        n_windows = len(log_returns) // lag
        if n_windows < 1:
            continue
        rs_list = []
        for i in range(n_windows):
            segment = log_returns[i * lag: (i + 1) * lag]
            rs = _rs_single(segment)
            if not np.isnan(rs):
                rs_list.append(rs)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            valid_lags.append(lag)

    if len(valid_lags) < 4:
        return 0.5

    # OLS regression: log(R/S) = H * log(n) + const
    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)

    # Remove any inf / nan
    mask = np.isfinite(log_lags) & np.isfinite(log_rs)
    if mask.sum() < 4:
        return 0.5

    coeffs = np.polyfit(log_lags[mask], log_rs[mask], 1)
    hurst = float(np.clip(coeffs[0], 0.0, 1.0))
    return hurst


def hurst_label(h: float) -> str:
    """Human-readable label for Hurst exponent."""
    if h < 0.45:
        return "mean-reverting"
    elif h > 0.55:
        return "trending"
    else:
        return "random-walk"


def hurst_factor_multiplier(h: float) -> dict:
    """
    Returns factor multipliers based on Hurst value.
    Used by the alpha factor model to adjust factor weights.
    """
    if h < 0.45:
        return {"momentum_mult": 0.75, "mean_rev_mult": 1.25, "size_mult": 1.0}
    elif h > 0.55:
        return {"momentum_mult": 1.25, "mean_rev_mult": 0.75, "size_mult": 1.0}
    else:
        return {"momentum_mult": 1.0, "mean_rev_mult": 1.0, "size_mult": 0.5}
