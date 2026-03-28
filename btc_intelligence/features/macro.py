from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from btc_intelligence.config import settings


@dataclass
class MacroFeatures:
    session: str
    session_weight: float
    dxy_bias: str
    fear_greed_score: int
    fear_greed_zone: str
    blocking_news: bool
    vix_level: float
    vix_regime: str
    us10y_yield_trend: str
    spx_intraday_direction: str
    spx_btc_correlation: float
    gold_direction: str
    eth_btc_ratio_trend: str
    btc_dominance_trend: str


def compute_macro(macro_data: dict, news_items: list[dict]) -> MacroFeatures:
    now = datetime.now(timezone.utc)
    hour = now.hour
    if 0 <= hour < 8:
        session = 'asian'
        weight = 0.7
    elif 8 <= hour < 13:
        session = 'london_open'
        weight = 1.1
    elif 13 <= hour < 16:
        session = 'ny_london_overlap'
        weight = 1.25
    elif 16 <= hour < 21:
        session = 'new_york'
        weight = 1.1
    else:
        session = 'dead'
        weight = 0.3

    dxy = str(macro_data.get('dxy_bias', 'neutral')).lower()
    fng = int(macro_data.get('fear_greed_score', 50))
    if fng < 25:
        zone = 'extreme_fear'
    elif fng < 45:
        zone = 'fear'
    elif fng <= 55:
        zone = 'neutral'
    elif fng <= 75:
        zone = 'greed'
    else:
        zone = 'extreme_greed'

    blocking_news = _has_blocking_news(news_items)

    vix_level = float(macro_data.get('vix_level', 20.0))
    vix_regime = str(macro_data.get('vix_regime', 'NORMAL'))

    return MacroFeatures(
        session=session,
        session_weight=weight,
        dxy_bias=dxy,
        fear_greed_score=fng,
        fear_greed_zone=zone,
        blocking_news=blocking_news,
        vix_level=vix_level,
        vix_regime=vix_regime,
        us10y_yield_trend=str(macro_data.get('us10y_yield_trend', 'flat')),
        spx_intraday_direction=str(macro_data.get('spx_intraday_direction', 'flat')),
        spx_btc_correlation=float(macro_data.get('spx_btc_correlation', 0.5)),
        gold_direction=str(macro_data.get('gold_direction', 'flat')),
        eth_btc_ratio_trend=str(macro_data.get('eth_btc_ratio_trend', 'flat')),
        btc_dominance_trend=str(macro_data.get('btc_dominance_trend', 'flat')),
    )


def _has_blocking_news(news_items: list[dict]) -> bool:
    if not news_items:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    for row in news_items:
        title = str(row.get('title', '')).lower()
        ts = str(row.get('published_at', ''))
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            continue
        if dt < cutoff:
            continue
        if row.get('importance') == 'high':
            return True
        if any(k in title for k in settings.blocking_news_keywords):
            return True
    return False
