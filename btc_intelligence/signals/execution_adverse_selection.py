from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


@dataclass(frozen=True)
class AdverseSelectionResult:
    """Fill-quality / adverse-selection diagnostics for execution."""

    adverse_selection_flag: bool
    execution_mode_recommendation: str
    spread_widening: bool
    mid_drift_pct: float
    spread_pct: float
    spread_baseline_proxy_pct: float
    noise_floor_pct: float
    reason: str


def _trade_times_prices(agg_trades: list[dict], window_ms: int) -> tuple[np.ndarray, np.ndarray]:
    if not agg_trades:
        return np.array([]), np.array([])
    latest_ts = int(agg_trades[-1].get("time", 0))
    if latest_ts <= 0:
        return np.array([]), np.array([])
    cutoff = latest_ts - int(window_ms)
    times: list[int] = []
    prices: list[float] = []
    for t in agg_trades:
        ts = int(t.get("time", 0))
        if ts < cutoff:
            continue
        p = float(t.get("price", 0.0))
        if p > 0:
            times.append(ts)
            prices.append(p)
    if not times:
        return np.array([]), np.array([])
    order = np.argsort(times)
    return np.asarray(times, dtype=np.int64)[order], np.asarray(prices, dtype=float)[order]


def _noise_floor_pct(prices: np.ndarray) -> float:
    if prices.size < 3:
        return 0.0
    r = np.diff(prices) / np.clip(prices[:-1], 1e-12, None) * 100.0
    if r.size == 0:
        return 0.0
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    return max(mad, 1e-6)


def _mid_drift_pct(prices: np.ndarray) -> float:
    """Signed drift from early-window to late-window microprice (tape proxy)."""
    n = int(prices.size)
    if n < 6:
        return 0.0
    k = max(2, n // 3)
    early = float(np.mean(prices[:k]))
    late = float(np.mean(prices[-k:]))
    if early <= 0:
        return 0.0
    return (late - early) / early * 100.0


def _spread_baseline_proxy_pct(prices: np.ndarray) -> float:
    """Proxy for typical micro range when the book was tight (tape-only)."""
    if prices.size < 5:
        return 0.0
    med = float(np.median(prices))
    if med <= 0:
        return 0.0
    lo = float(np.percentile(prices, 10))
    hi = float(np.percentile(prices, 90))
    return max((hi - lo) / med * 100.0, 1e-4)


def compute_adverse_selection(
    depth: dict[str, Any],
    agg_trades: list[dict],
    direction: str,
    *,
    window_ms: int | None = None,
    drift_threshold_pct: float | None = None,
    spread_widen_ratio: float | None = None,
    min_trades: int | None = None,
    noise_mad_multiplier: float | None = None,
) -> AdverseSelectionResult:
    """
    Detect adverse selection: spread widening vs tape + signed mid drift against intent.

    When True, execution should prefer hidden/passive posting over aggressive limits.
    """
    window_ms = int(window_ms if window_ms is not None else settings.adverse_selection_window_ms)
    drift_threshold_pct = float(
        drift_threshold_pct if drift_threshold_pct is not None else settings.adverse_mid_drift_threshold_pct
    )
    spread_widen_ratio = float(
        spread_widen_ratio if spread_widen_ratio is not None else settings.adverse_spread_widen_ratio
    )
    min_trades = int(min_trades if min_trades is not None else settings.adverse_min_trades)
    noise_mad_multiplier = float(
        noise_mad_multiplier if noise_mad_multiplier is not None else settings.adverse_noise_mad_multiplier
    )

    direction_u = str(direction or "LONG").upper()
    if direction_u not in {"LONG", "SHORT"}:
        direction_u = "LONG"

    bid, ask, mid = MarketDataBuffer.best_bid_ask(depth)
    if mid <= 0 or bid <= 0 or ask <= 0:
        return AdverseSelectionResult(
            adverse_selection_flag=False,
            execution_mode_recommendation="LIMIT",
            spread_widening=False,
            mid_drift_pct=0.0,
            spread_pct=999.0,
            spread_baseline_proxy_pct=0.0,
            noise_floor_pct=0.0,
            reason="depth_unavailable",
        )

    spread_pct = (ask - bid) / mid * 100.0

    _, prices = _trade_times_prices(agg_trades, window_ms)
    if prices.size < min_trades:
        return AdverseSelectionResult(
            adverse_selection_flag=False,
            execution_mode_recommendation="LIMIT",
            spread_widening=False,
            mid_drift_pct=0.0,
            spread_pct=round(float(spread_pct), 8),
            spread_baseline_proxy_pct=0.0,
            noise_floor_pct=0.0,
            reason="insufficient_tape",
        )

    base = _spread_baseline_proxy_pct(prices)
    spread_widening = spread_pct > spread_widen_ratio * max(base, settings.adverse_spread_baseline_floor_pct)

    drift = _mid_drift_pct(prices)
    noise = _noise_floor_pct(prices)
    floor = max(drift_threshold_pct, noise_mad_multiplier * noise)

    if direction_u == "LONG":
        drift_against = drift < -floor
    else:
        drift_against = drift > floor

    flag = bool(spread_widening and drift_against)
    mode = "HIDDEN_PASSIVE" if flag else "LIMIT"
    reason = "ok"
    if flag:
        reason = "spread_widening_and_drift_against_direction"
    elif spread_widening and not drift_against:
        reason = "spread_widening_only"
    elif drift_against and not spread_widening:
        reason = "drift_against_only"

    return AdverseSelectionResult(
        adverse_selection_flag=flag,
        execution_mode_recommendation=mode,
        spread_widening=spread_widening,
        mid_drift_pct=round(float(drift), 8),
        spread_pct=round(float(spread_pct), 8),
        spread_baseline_proxy_pct=round(float(base), 8),
        noise_floor_pct=round(float(noise), 8),
        reason=reason,
    )
