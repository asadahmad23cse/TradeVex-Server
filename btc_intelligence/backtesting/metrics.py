from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    std = float(arr.std())
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(252))


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd * 100.0


def win_rate(results: list[float]) -> float:
    if not results:
        return 0.0
    wins = sum(1 for x in results if x > 0)
    return wins / len(results) * 100.0
