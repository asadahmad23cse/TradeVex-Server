from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OrderFlowFeatures:
    buy_volume: float
    sell_volume: float
    cvd: float
    cvd_slope: float
    cvd_trend: str
    cvd_divergence: bool
    divergence_threshold: float
    obi: float
    ofi_5bar: float
    whale_buy_count: int
    whale_sell_count: int
    whale_buy_ratio: float
    absorption_strength: float
    absorption_side: str
    stacked_imbalance_direction: str
    single_bar_delta_divergence: bool
    iceberg_direction: str
    speed_of_tape_ratio: float
    cross_exchange_cvd_divergence: bool
    cross_exchange_alignment: str


def compute_order_flow(
    agg_trades: list[dict],
    depth: dict[str, list[list[float]]],
    multi_exchange: dict,
    df_15m: pd.DataFrame,
) -> OrderFlowFeatures:
    window = agg_trades[-500:]
    buy_volume = sum(float(t.get('qty', 0.0)) for t in window if not bool(t.get('maker', False)))
    sell_volume = sum(float(t.get('qty', 0.0)) for t in window if bool(t.get('maker', False)))

    signed = np.asarray([
        float(t.get('qty', 0.0)) if not bool(t.get('maker', False)) else -float(t.get('qty', 0.0)) for t in window
    ], dtype=float)
    cvd_series = np.cumsum(signed) if len(signed) else np.asarray([0.0])
    cvd = float(cvd_series[-1])

    tail = cvd_series[-20:] if len(cvd_series) >= 20 else cvd_series
    x = np.arange(len(tail), dtype=float)
    cvd_slope = float(np.polyfit(x, tail, 1)[0]) if len(tail) > 1 else 0.0
    cvd_trend = 'rising' if cvd_slope > 0 else 'falling' if cvd_slope < 0 else 'flat'

    price_tail = np.asarray([float(t.get('price', 0.0)) for t in window[-200:]], dtype=float)
    price_slope = float(np.polyfit(np.arange(len(price_tail), dtype=float), price_tail, 1)[0]) if len(price_tail) > 1 else 0.0

    avg_abs_delta = float(np.mean(np.abs(np.diff(cvd_series[-100:])))) if len(cvd_series) > 2 else 0.0
    divergence_threshold = max(150.0, avg_abs_delta)
    cvd_diff = abs(float(tail[-1] - tail[0])) if len(tail) > 1 else 0.0
    cvd_divergence = ((price_slope > 0 and cvd_slope < 0) or (price_slope < 0 and cvd_slope > 0)) and cvd_diff >= divergence_threshold

    bids = depth.get('bids', [])[:10]
    asks = depth.get('asks', [])[:10]
    bid_depth = sum(float(px) * float(qty) for px, qty in bids)
    ask_depth = sum(float(px) * float(qty) for px, qty in asks)
    total_depth = bid_depth + ask_depth
    obi = (bid_depth / total_depth) if total_depth > 0 else 0.5

    chunks = np.array_split(window[-100:], 5) if window else []
    ofi_values = []
    for chunk in chunks:
        buy = sum(float(t.get('qty', 0.0)) for t in chunk if not bool(t.get('maker', False)))
        sell = sum(float(t.get('qty', 0.0)) for t in chunk if bool(t.get('maker', False)))
        ofi_values.append(buy - sell)
    ofi_5bar = float(sum(ofi_values)) if ofi_values else 0.0

    whales = [t for t in window if float(t.get('notional', 0.0)) >= 500_000]
    whale_buy = sum(1 for t in whales if not bool(t.get('maker', False)))
    whale_sell = sum(1 for t in whales if bool(t.get('maker', False)))
    whale_total = whale_buy + whale_sell
    whale_ratio = (whale_buy / whale_total) if whale_total > 0 else 0.5

    absorption_strength, absorption_side = _detect_absorption(window)
    stacked_imbalance_direction = _detect_stacked_imbalance(window)
    single_bar_delta_div = _single_bar_delta_divergence(window, df_15m)
    iceberg_direction = _detect_iceberg(window)
    speed_ratio = _speed_of_tape(window)
    cross_div, cross_align = _cross_exchange_cvd(multi_exchange, cvd_slope)

    return OrderFlowFeatures(
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        cvd=cvd,
        cvd_slope=cvd_slope,
        cvd_trend=cvd_trend,
        cvd_divergence=cvd_divergence,
        divergence_threshold=divergence_threshold,
        obi=obi,
        ofi_5bar=ofi_5bar,
        whale_buy_count=whale_buy,
        whale_sell_count=whale_sell,
        whale_buy_ratio=whale_ratio,
        absorption_strength=absorption_strength,
        absorption_side=absorption_side,
        stacked_imbalance_direction=stacked_imbalance_direction,
        single_bar_delta_divergence=single_bar_delta_div,
        iceberg_direction=iceberg_direction,
        speed_of_tape_ratio=speed_ratio,
        cross_exchange_cvd_divergence=cross_div,
        cross_exchange_alignment=cross_align,
    )


def _detect_absorption(trades: list[dict]) -> tuple[float, str]:
    if len(trades) < 40:
        return 0.0, 'none'
    window = trades[-180:]
    delta = sum(float(t.get('qty', 0.0)) if not bool(t.get('maker', False)) else -float(t.get('qty', 0.0)) for t in window)
    p0 = float(window[0].get('price', 0.0))
    p1 = float(window[-1].get('price', 0.0))
    if p0 <= 0:
        return 0.0, 'none'
    move = (p1 - p0) / p0 * 100.0
    denom = max(abs(move), 0.01)
    strength = abs(delta) / denom
    side = 'none'
    if delta > 0 and move <= 0:
        side = 'buy_absorption'
    elif delta < 0 and move >= 0:
        side = 'sell_absorption'
    return float(strength), side


def _detect_stacked_imbalance(trades: list[dict]) -> str:
    if not trades:
        return 'none'
    ladder: dict[float, list[float]] = {}
    for t in trades[-250:]:
        px = round(float(t.get('price', 0.0)), 1)
        if px <= 0:
            continue
        buy = float(t.get('qty', 0.0)) if not bool(t.get('maker', False)) else 0.0
        sell = float(t.get('qty', 0.0)) if bool(t.get('maker', False)) else 0.0
        if px not in ladder:
            ladder[px] = [0.0, 0.0]
        ladder[px][0] += buy
        ladder[px][1] += sell

    prices = sorted(ladder.keys())
    bullish_run = 0
    bearish_run = 0
    for px in prices:
        buy, sell = ladder[px]
        ratio = (buy / max(sell, 1e-9)) if sell > 0 else 10.0
        inv = (sell / max(buy, 1e-9)) if buy > 0 else 10.0
        if ratio >= 3.0:
            bullish_run += 1
        else:
            bullish_run = 0
        if inv >= 3.0:
            bearish_run += 1
        else:
            bearish_run = 0
        if bullish_run >= 3:
            return 'bullish_3_levels'
        if bearish_run >= 3:
            return 'bearish_3_levels'
    return 'none'


def _single_bar_delta_divergence(trades: list[dict], df_15m: pd.DataFrame) -> bool:
    if len(trades) < 50 or len(df_15m) < 25:
        return False
    last_c = df_15m.iloc[-1]
    end_ms = int(df_15m.index[-1].timestamp() * 1000)
    start_ms = end_ms - 15 * 60 * 1000
    candle_trades = [t for t in trades if start_ms <= int(t.get('time', 0)) <= end_ms]
    if not candle_trades:
        return False

    delta = sum(float(t.get('qty', 0.0)) if not bool(t.get('maker', False)) else -float(t.get('qty', 0.0)) for t in candle_trades)

    bar_deltas = []
    for i in range(1, 21):
        e = end_ms - (i - 1) * 15 * 60 * 1000
        s = e - 15 * 60 * 1000
        d = sum(float(t.get('qty', 0.0)) if not bool(t.get('maker', False)) else -float(t.get('qty', 0.0)) for t in trades if s <= int(t.get('time', 0)) <= e)
        bar_deltas.append(abs(d))
    avg_abs = float(np.mean(bar_deltas)) if bar_deltas else 0.0
    if avg_abs <= 0:
        return False

    bullish_candle = float(last_c['Close']) > float(last_c['Open'])
    bearish_candle = float(last_c['Close']) < float(last_c['Open'])
    significant = abs(delta) > 0.5 * avg_abs

    return significant and ((bullish_candle and delta < 0) or (bearish_candle and delta > 0))


def _detect_iceberg(trades: list[dict]) -> str:
    if len(trades) < 80:
        return 'none'
    by_key: dict[tuple[int, float, bool], int] = {}
    for t in trades[-300:]:
        second = int(int(t.get('time', 0)) / 1000)
        px = round(float(t.get('price', 0.0)), 1)
        maker = bool(t.get('maker', False))
        key = (second, px, maker)
        by_key[key] = by_key.get(key, 0) + 1
    top = max(by_key.items(), key=lambda kv: kv[1])
    if top[1] < 50:
        return 'none'
    return 'bearish' if top[0][2] else 'bullish'


def _speed_of_tape(trades: list[dict]) -> float:
    if len(trades) < 30:
        return 1.0
    latest = int(trades[-1].get('time', 0))
    counts = []
    for i in range(1, 21):
        e = latest - (i - 1) * 1000
        s = e - 1000
        c = sum(1 for t in trades if s <= int(t.get('time', 0)) <= e)
        counts.append(c)
    baseline = float(np.mean(counts[1:])) if len(counts) > 1 else 1.0
    baseline = max(baseline, 1.0)
    return counts[0] / baseline


def _cross_exchange_cvd(multi_exchange: dict, binance_cvd_slope: float) -> tuple[bool, str]:
    bybit = multi_exchange.get('bybit', {}).get('cvd', [])
    okx = multi_exchange.get('okx', {}).get('cvd', [])

    def slope(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        arr = np.asarray(vals[-20:], dtype=float)
        x = np.arange(len(arr), dtype=float)
        return float(np.polyfit(x, arr, 1)[0]) if len(arr) > 1 else 0.0

    s_bybit = slope(bybit)
    s_okx = slope(okx)
    avg = (s_bybit + s_okx) / 2.0

    div = (binance_cvd_slope > 0 and avg < 0) or (binance_cvd_slope < 0 and avg > 0)
    if abs(avg) < 1e-9 and abs(binance_cvd_slope) < 1e-9:
        align = 'neutral'
    elif div:
        align = 'divergent'
    else:
        align = 'aligned'
    return bool(div), align
