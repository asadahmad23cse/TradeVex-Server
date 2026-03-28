from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from btc_intelligence.features.feature_vector import FeatureState


@dataclass
class RegimeResult:
    regime: str
    score: int
    reason: str


def classify_regime(snapshot: dict, state: FeatureState) -> RegimeResult:
    tf4 = state.price_action.tf_4h
    tf15 = state.price_action.tf_15m

    panic_reason = _panic_condition(snapshot, state)
    if panic_reason:
        return RegimeResult('panic_liquidation', 6, panic_reason)

    bull_score = 0
    if tf4.price > tf4.ema50 > tf4.ema200:
        bull_score += 1
    if state.volatility.vol_regime in {'normal', 'expansion'}:
        bull_score += 1
    if state.order_flow.cvd_slope > 0:
        bull_score += 1
    if state.derivatives.oi_trend in {'rising_strong', 'stable'}:
        bull_score += 1
    if state.macro.spx_intraday_direction != 'down' and state.macro.vix_level < 30:
        bull_score += 1
    if state.price_action.alignment_long >= 2:
        bull_score += 1

    bear_score = 0
    if tf4.price < tf4.ema50 < tf4.ema200:
        bear_score += 1
    if state.order_flow.cvd_slope < 0:
        bear_score += 1
    if state.macro.vix_level > 25:
        bear_score += 1
    if state.price_action.alignment_short >= 2:
        bear_score += 1
    if state.derivatives.oi_trend in {'rising_strong', 'stable'}:
        bear_score += 1

    range_score = 0
    range_width_pct = ((tf15.swing_high_20 - tf15.swing_low_20) / max(tf15.price, 1e-9)) * 100.0
    if range_width_pct < 3.0:
        range_score += 1
    if state.volatility.vol_regime == 'compression':
        range_score += 1
    if 0.45 <= state.order_flow.obi <= 0.55:
        range_score += 1
    if 0.9 <= state.volatility.iv_rv_ratio <= 1.1:
        range_score += 1

    breakout_up = _breakout(snapshot, 'up', state)
    breakout_down = _breakout(snapshot, 'down', state)
    if breakout_up:
        return RegimeResult('breakout_up', 4, breakout_up)
    if breakout_down:
        return RegimeResult('breakout_down', 4, breakout_down)

    if bull_score >= 4:
        return RegimeResult('bullish_trend', bull_score, '4h uptrend + flow + macro support')
    if bear_score >= 4:
        return RegimeResult('bearish_trend', bear_score, '4h downtrend + flow + risk-off support')
    if range_score >= 3:
        return RegimeResult('sideways_range', range_score, 'Compression + neutral depth + fair options')

    if bull_score >= bear_score:
        return RegimeResult('bullish_trend', bull_score, 'Fallback bullish')
    return RegimeResult('bearish_trend', bear_score, 'Fallback bearish')


def _breakout(snapshot: dict, direction: str, state: FeatureState) -> str | None:
    df = snapshot.get('candles', {}).get('15m', [])
    if len(df) < 25:
        return None
    last = df[-1]
    prev = df[-21:-1]
    high20 = max(float(x['high']) for x in prev)
    low20 = min(float(x['low']) for x in prev)
    close = float(last['close'])
    vol = float(last['volume'])
    avg_vol = sum(float(x['volume']) for x in prev) / max(len(prev), 1)

    bb_released = state.volatility.bb_expansion

    if direction == 'up' and close > high20 and vol > 1.5 * avg_vol and state.derivatives.oi_change_1h_pct > 0 and bb_released:
        return 'Breakout up with volume+OI and BB release'
    if direction == 'down' and close < low20 and vol > 1.5 * avg_vol and state.derivatives.oi_change_1h_pct > 0 and bb_released:
        return 'Breakout down with volume+OI and BB release'
    return None


def _panic_condition(snapshot: dict, state: FeatureState) -> str | None:
    candles = snapshot.get('candles', {}).get('15m', [])
    if len(candles) < 2 or state.volatility.atr14 <= 0:
        return None

    c = candles[-1]
    move = abs(float(c['close']) - float(c['open']))
    move_atr = move / state.volatility.atr14

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=5)
    liq_5m = 0.0
    for row in snapshot.get('force_orders', []):
        ts = datetime.fromtimestamp(int(row.get('time', 0)) / 1000, tz=timezone.utc)
        if ts >= cutoff:
            liq_5m += float(row.get('notional', 0.0))

    if move_atr > 3 and liq_5m > 100_000_000 and abs(state.derivatives.funding_rate) > 0.001 and state.macro.vix_level > 35:
        return '3x ATR + force liquidations + funding extreme + VIX spike'
    return None
