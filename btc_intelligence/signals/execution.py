from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from btc_intelligence.config import settings


@dataclass
class ExecutionCheck:
    spread_pct: float
    slippage_pct: float
    price_move_30s_pct: float
    quality: str
    accepted: bool
    trigger_ready: bool
    confidence_penalty: float
    reason: str


def evaluate_execution(
    depth: dict,
    agg_trades: list[dict],
    direction: str,
    order_flow: object | None = None,
    derivatives: object | None = None,
    market_state: dict[str, Any] | None = None,
    fallback_tradeability: str = 'ALLOW',
) -> ExecutionCheck:
    bids = depth.get('bids', [])
    asks = depth.get('asks', [])
    if not bids or not asks:
        return ExecutionCheck(999.0, 999.0, 999.0, 'poor', False, False, 0.0, 'Depth unavailable')

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
    spread_pct = ((best_ask - best_bid) / mid * 100.0) if mid > 0 else 999.0

    slippage_pct = estimate_slippage_pct(depth, side='buy' if direction == 'LONG' else 'sell', qty_btc=0.1)
    move_30s = _price_move_30s_pct(agg_trades)

    if spread_pct >= settings.spread_reject_pct:
        return ExecutionCheck(spread_pct, slippage_pct, move_30s, 'poor', False, False, 0.0, 'Spread >= 0.05%')
    if slippage_pct > settings.slippage_reject_pct:
        return ExecutionCheck(spread_pct, slippage_pct, move_30s, 'poor', False, False, 0.0, 'Slippage > 0.03%')
    if move_30s > 0.3:
        return ExecutionCheck(spread_pct, slippage_pct, move_30s, 'poor', False, False, 0.0, 'Price move 30s > 0.3%')

    quality = 'excellent' if spread_pct < 0.02 else 'good'

    trigger_ready = _micro_timing_trigger(agg_trades, direction, order_flow)
    if not trigger_ready:
        return ExecutionCheck(spread_pct, slippage_pct, move_30s, quality, False, False, 0.0, 'Micro-timing trigger not ready')

    penalty = 0.0
    if derivatives is not None:
        expiry_hours = float(getattr(derivatives, 'options_expiry_hours', 9999.0))
        max_pain_dist = abs(float(getattr(derivatives, 'max_pain_distance_pct', 99.0)))
        now = datetime.now(timezone.utc)
        if now.weekday() == 4 and max_pain_dist < 1.5:
            penalty += 20.0
        if now.weekday() == 4 and expiry_hours <= 4:
            return ExecutionCheck(spread_pct, slippage_pct, move_30s, 'poor', False, False, penalty, 'No signals 4h before monthly expiry window')

    tradeability = _resolve_volatility_tradeability(
        market_state=market_state,
        fallback_tradeability=fallback_tradeability,
    )
    if tradeability == 'NO_TRADE':
        return ExecutionCheck(
            spread_pct,
            slippage_pct,
            move_30s,
            quality,
            False,
            False,
            penalty,
            'Volatility regime too low for execution',
        )

    return ExecutionCheck(spread_pct, slippage_pct, move_30s, quality, True, True, penalty, 'Execution quality acceptable')


def estimate_slippage_pct(depth: dict, side: str, qty_btc: float = 0.1) -> float:
    levels = depth.get('asks', []) if side == 'buy' else depth.get('bids', [])
    if not levels:
        return 999.0

    best_bid = float(depth.get('bids', [[0, 0]])[0][0]) if depth.get('bids') else 0.0
    best_ask = float(depth.get('asks', [[0, 0]])[0][0]) if depth.get('asks') else 0.0
    mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
    if mid <= 0:
        return 999.0

    remaining = qty_btc
    total_cost = 0.0
    filled = 0.0

    for price, qty in levels:
        p = float(price)
        q = float(qty)
        take = min(remaining, q)
        total_cost += take * p
        filled += take
        remaining -= take
        if remaining <= 1e-9:
            break

    if filled <= 0:
        return 999.0

    avg_fill = total_cost / filled
    if side == 'buy':
        slip = (avg_fill - mid) / mid * 100.0
    else:
        slip = (mid - avg_fill) / mid * 100.0
    return max(0.0, slip)


def recommended_max_position_btc(
    depth: dict,
    side: str,
    slippage_limit_pct: float,
    min_qty_btc: float = 0.01,
    max_qty_btc: float = 10.0,
    steps: int = 18,
) -> float:
    """
    Estimate the maximum executable BTC size under a slippage limit.

    Uses a monotonic scan over log-spaced qty buckets for robustness
    against sparse order books.
    """
    if slippage_limit_pct <= 0:
        return 0.0
    lo = max(float(min_qty_btc), 1e-4)
    hi = max(float(max_qty_btc), lo)
    best = 0.0
    for i in range(max(int(steps), 2)):
        w = i / max(steps - 1, 1)
        qty = lo * ((hi / lo) ** w)
        slip = estimate_slippage_pct(depth, side=side, qty_btc=qty)
        if slip <= slippage_limit_pct:
            best = qty
        else:
            break
    return round(best, 4)


def _price_move_30s_pct(agg_trades: list[dict]) -> float:
    if len(agg_trades) < 2:
        return 0.0
    latest_ts = int(agg_trades[-1].get('time', 0))
    cutoff = latest_ts - 30_000
    window = [x for x in agg_trades if int(x.get('time', 0)) >= cutoff]
    if len(window) < 2:
        return 0.0
    p0 = float(window[0].get('price', 0.0))
    p1 = float(window[-1].get('price', 0.0))
    if p0 <= 0:
        return 0.0
    return abs((p1 - p0) / p0 * 100.0)


def _micro_timing_trigger(agg_trades: list[dict], direction: str, order_flow: object | None) -> bool:
    if len(agg_trades) < 30:
        return False
    latest_ts = int(agg_trades[-1].get('time', 0))

    recent = [t for t in agg_trades if latest_ts - 60_000 <= int(t.get('time', 0)) <= latest_ts]
    prior = [t for t in agg_trades if latest_ts - 300_000 <= int(t.get('time', 0)) < latest_ts - 60_000]

    recent_vol = sum(float(t.get('qty', 0.0)) for t in recent)
    prior_vol = sum(float(t.get('qty', 0.0)) for t in prior) / max(len(prior), 1)
    volume_spike = recent_vol > max(prior_vol * max(len(recent), 1) * 1.5, 1e-9)

    if order_flow is None:
        return volume_spike

    single_div = bool(getattr(order_flow, 'single_bar_delta_divergence', False))
    obi = float(getattr(order_flow, 'obi', 0.5))
    obi_flip = obi > 0.65 if direction == 'LONG' else obi < 0.35

    return volume_spike or (not single_div) or obi_flip


def _resolve_volatility_tradeability(
    market_state: dict[str, Any] | None,
    fallback_tradeability: str,
) -> str:
    if isinstance(market_state, dict):
        vol_payload = market_state.get('volatility_tradeability', {})
        if isinstance(vol_payload, dict):
            resolved = str(vol_payload.get('tradeability', '')).upper()
            if resolved in {'NO_TRADE', 'ALLOW', 'CAUTION'}:
                return resolved

    fallback = str(fallback_tradeability).upper()
    if fallback in {'NO_TRADE', 'ALLOW', 'CAUTION'}:
        return fallback
    return 'ALLOW'
