from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DerivativesFeatures:
    funding_rate: float
    funding_sentiment: str
    funding_roc: float
    open_interest: float
    oi_change_1h_pct: float
    oi_trend: str
    oi_price_divergence: str
    liq_cluster_above_pct: float
    liq_cluster_below_pct: float
    liq_magnet_bias: str
    long_short_ratio: float
    ls_extreme: bool
    max_pain_level: float
    max_pain_distance_pct: float
    max_pain_magnet: bool
    iv_skew: float
    put_call_ratio: float
    options_expiry_hours: float


def compute_derivatives(
    rest_data: dict,
    oi_hist: list[dict],
    funding_hist: list[dict],
    coinglass_data: dict,
    deribit_data: dict,
    price_now: float,
    price_1h_ago: float,
) -> DerivativesFeatures:
    funding = float(rest_data.get('funding_rate', 0.0))
    oi = float(rest_data.get('open_interest', 0.0))

    if funding > 0.0005:
        funding_sentiment = 'overleveraged_long'
    elif funding < -0.0005:
        funding_sentiment = 'overleveraged_short'
    else:
        funding_sentiment = 'neutral'

    funding_roc = _funding_roc(funding_hist)

    oi_change = _oi_change_1h(oi_hist)
    if oi_change > 2.0:
        oi_trend = 'rising_strong'
    elif oi_change < -2.0:
        oi_trend = 'falling'
    else:
        oi_trend = 'stable'

    price_change = ((price_now - price_1h_ago) / price_1h_ago * 100.0) if price_1h_ago > 0 else 0.0
    if oi_change > 0 and price_change < 0:
        oi_price_div = 'short_squeeze_building'
    elif oi_change > 0 and price_change > 0:
        oi_price_div = 'trend_continuation'
    elif oi_change < 0 and price_change < 0:
        oi_price_div = 'deleveraging'
    else:
        oi_price_div = 'neutral'

    liq_above = float(coinglass_data.get('liquidation_heatmap_above', 0.0))
    liq_below = float(coinglass_data.get('liquidation_heatmap_below', 0.0))
    above_pct = ((liq_above - price_now) / price_now * 100.0) if price_now > 0 else 0.0
    below_pct = ((price_now - liq_below) / price_now * 100.0) if price_now > 0 else 0.0

    if 0 < above_pct <= 1.5:
        liq_magnet = 'bullish'
    elif 0 < below_pct <= 1.5:
        liq_magnet = 'bearish'
    else:
        liq_magnet = 'neutral'

    ls_ratio = float(coinglass_data.get('long_short_ratio', 0.5))
    ls_extreme = ls_ratio > 0.7 or ls_ratio < 0.3

    max_pain = float(deribit_data.get('max_pain_level', 0.0))
    max_pain_dist = ((price_now - max_pain) / price_now * 100.0) if price_now > 0 and max_pain > 0 else 0.0
    expiry_hours = float(deribit_data.get('options_expiry_hours', 9999.0))

    return DerivativesFeatures(
        funding_rate=funding,
        funding_sentiment=funding_sentiment,
        funding_roc=funding_roc,
        open_interest=oi,
        oi_change_1h_pct=oi_change,
        oi_trend=oi_trend,
        oi_price_divergence=oi_price_div,
        liq_cluster_above_pct=above_pct,
        liq_cluster_below_pct=below_pct,
        liq_magnet_bias=liq_magnet,
        long_short_ratio=ls_ratio,
        ls_extreme=ls_extreme,
        max_pain_level=max_pain,
        max_pain_distance_pct=max_pain_dist,
        max_pain_magnet=(abs(max_pain_dist) < 2.0 and expiry_hours <= 48),
        iv_skew=float(deribit_data.get('iv_skew', 0.0)),
        put_call_ratio=float(deribit_data.get('put_call_ratio', 1.0)),
        options_expiry_hours=expiry_hours,
    )


def _oi_change_1h(oi_hist: list[dict]) -> float:
    if len(oi_hist) < 3:
        return 0.0
    latest = float(oi_hist[-1].get('open_interest', 0.0))
    latest_ts = float(oi_hist[-1].get('ts', 0.0))
    target_ts = latest_ts - 3600

    nearest = oi_hist[0]
    min_dist = abs(float(nearest.get('ts', 0.0)) - target_ts)
    for row in oi_hist:
        dist = abs(float(row.get('ts', 0.0)) - target_ts)
        if dist < min_dist:
            nearest = row
            min_dist = dist

    base = float(nearest.get('open_interest', 0.0))
    if base == 0:
        return 0.0
    return (latest - base) / base * 100.0


def _funding_roc(hist: list[dict]) -> float:
    if len(hist) < 3:
        return 0.0
    current = float(hist[-1].get('funding_rate', 0.0))
    last = float(hist[0].get('funding_rate', 0.0))
    target_ts = float(hist[-1].get('ts', 0.0)) - 8 * 3600
    min_dist = 1e18
    for row in hist:
        dist = abs(float(row.get('ts', 0.0)) - target_ts)
        if dist < min_dist:
            last = float(row.get('funding_rate', 0.0))
            min_dist = dist
    denom = abs(last) if abs(last) > 1e-9 else 1e-9
    return (current - last) / denom
