from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class OrderBlock:
    direction: str
    low: float
    high: float
    midpoint: float
    created_at: str
    valid: bool


@dataclass
class FVG:
    direction: str
    low: float
    high: float
    created_at: str
    filled: bool


@dataclass
class SMCFeatures:
    bullish_obs: list[OrderBlock]
    bearish_obs: list[OrderBlock]
    fvgs: list[FVG]
    bos_type: str
    choch: bool
    liquidity_sweep_recent: bool


def compute_smc(df: pd.DataFrame, trend_bias: str) -> SMCFeatures:
    if len(df) < 50:
        return SMCFeatures([], [], [], 'none', False, False)

    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    open_ = df['Open'].astype(float)
    close = df['Close'].astype(float)
    volume = df['Volume'].astype(float)

    avg_vol = float(volume.tail(20).mean())
    bullish_obs: list[OrderBlock] = []
    bearish_obs: list[OrderBlock] = []
    fvgs: list[FVG] = []

    # Order block detection: last opposite candle before impulsive 3-candle move.
    for i in range(5, len(df) - 3):
        c0_o = float(open_.iloc[i])
        c0_c = float(close.iloc[i])
        c1 = float(close.iloc[i + 1]) > float(open_.iloc[i + 1])
        c2 = float(close.iloc[i + 2]) > float(open_.iloc[i + 2])
        c3 = float(close.iloc[i + 3]) > float(open_.iloc[i + 3])

        if c0_c < c0_o and c1 and c2 and c3 and float(volume.iloc[i + 1 : i + 4].mean()) > avg_vol * 1.3:
            lo = float(low.iloc[i])
            hi = float(high.iloc[i])
            bullish_obs.append(
                OrderBlock('bullish', lo, hi, (lo + hi) / 2.0, str(df.index[i]), True)
            )

        d1 = float(close.iloc[i + 1]) < float(open_.iloc[i + 1])
        d2 = float(close.iloc[i + 2]) < float(open_.iloc[i + 2])
        d3 = float(close.iloc[i + 3]) < float(open_.iloc[i + 3])
        if c0_c > c0_o and d1 and d2 and d3 and float(volume.iloc[i + 1 : i + 4].mean()) > avg_vol * 1.3:
            lo = float(low.iloc[i])
            hi = float(high.iloc[i])
            bearish_obs.append(
                OrderBlock('bearish', lo, hi, (lo + hi) / 2.0, str(df.index[i]), True)
            )

    # FVG detection from three-candle pattern.
    for i in range(1, len(df) - 1):
        p_high = float(high.iloc[i - 1])
        p_low = float(low.iloc[i - 1])
        n_high = float(high.iloc[i + 1])
        n_low = float(low.iloc[i + 1])
        px = float(close.iloc[i])

        bullish_gap = p_high < n_low
        bearish_gap = p_low > n_high
        if bullish_gap:
            width_pct = (n_low - p_high) / max(px, 1e-9) * 100.0
            if width_pct >= 0.05:
                fvgs.append(FVG('bullish', p_high, n_low, str(df.index[i]), False))
        if bearish_gap:
            width_pct = (p_low - n_high) / max(px, 1e-9) * 100.0
            if width_pct >= 0.05:
                fvgs.append(FVG('bearish', n_high, p_low, str(df.index[i]), False))

    # Mark OB validity and FVG fills with latest close.
    last_price = float(close.iloc[-1])
    for ob in bullish_obs:
        if last_price < ob.low:
            ob.valid = False
    for ob in bearish_obs:
        if last_price > ob.high:
            ob.valid = False

    for fvg in fvgs:
        if fvg.direction == 'bullish' and last_price <= fvg.low:
            fvg.filled = True
        if fvg.direction == 'bearish' and last_price >= fvg.high:
            fvg.filled = True

    # BOS on body close beyond recent swing.
    swing_high = float(high.tail(20).iloc[:-1].max())
    swing_low = float(low.tail(20).iloc[:-1].min())
    last_close = float(close.iloc[-1])
    bos_type = 'none'
    if last_close > swing_high:
        bos_type = 'bullish_continuation'
    elif last_close < swing_low:
        bos_type = 'bearish_continuation'

    choch = (trend_bias == 'bearish' and bos_type.startswith('bullish')) or (
        trend_bias == 'bullish' and bos_type.startswith('bearish')
    )

    # Liquidity sweep in last 3 candles.
    recent = df.tail(3)
    prior_swing_high = float(high.tail(23).head(20).max()) if len(df) >= 23 else swing_high
    prior_swing_low = float(low.tail(23).head(20).min()) if len(df) >= 23 else swing_low
    sweep = False
    for _, row in recent.iterrows():
        wick_high = float(row['High'])
        wick_low = float(row['Low'])
        body_close = float(row['Close'])
        if wick_low < prior_swing_low and body_close > prior_swing_low:
            sweep = True
        if wick_high > prior_swing_high and body_close < prior_swing_high:
            sweep = True

    return SMCFeatures(
        bullish_obs=[x for x in bullish_obs if x.valid][-10:],
        bearish_obs=[x for x in bearish_obs if x.valid][-10:],
        fvgs=[x for x in fvgs if not x.filled][-15:],
        bos_type=bos_type,
        choch=choch,
        liquidity_sweep_recent=sweep,
    )
