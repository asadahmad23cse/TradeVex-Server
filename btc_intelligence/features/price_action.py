from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


TrendBias = Literal['bullish', 'bearish', 'mixed']


@dataclass
class TimeframeStructure:
    timeframe: str
    price: float
    ema9: float
    ema21: float
    ema50: float
    ema200: float
    trend_bias: TrendBias
    swing_high_20: float
    swing_low_20: float
    higher_high: bool
    higher_low: bool
    lower_high: bool
    lower_low: bool


@dataclass
class PriceActionFeatures:
    tf_15m: TimeframeStructure
    tf_1h: TimeframeStructure
    tf_4h: TimeframeStructure
    alignment_long: int
    alignment_short: int


def _build_tf(df: pd.DataFrame, timeframe: str) -> TimeframeStructure:
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

    price = float(close.iloc[-1])
    if price > ema21 > ema50 > ema200:
        trend_bias: TrendBias = 'bullish'
    elif price < ema21 < ema50 < ema200:
        trend_bias = 'bearish'
    else:
        trend_bias = 'mixed'

    swing_high = float(high.tail(20).max())
    swing_low = float(low.tail(20).min())

    higher_high = len(high) >= 3 and float(high.iloc[-1]) > float(high.iloc[-2]) > float(high.iloc[-3])
    higher_low = len(low) >= 3 and float(low.iloc[-1]) > float(low.iloc[-2]) > float(low.iloc[-3])
    lower_high = len(high) >= 3 and float(high.iloc[-1]) < float(high.iloc[-2]) < float(high.iloc[-3])
    lower_low = len(low) >= 3 and float(low.iloc[-1]) < float(low.iloc[-2]) < float(low.iloc[-3])

    return TimeframeStructure(
        timeframe=timeframe,
        price=price,
        ema9=ema9,
        ema21=ema21,
        ema50=ema50,
        ema200=ema200,
        trend_bias=trend_bias,
        swing_high_20=swing_high,
        swing_low_20=swing_low,
        higher_high=higher_high,
        higher_low=higher_low,
        lower_high=lower_high,
        lower_low=lower_low,
    )


def compute_price_action(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> PriceActionFeatures:
    tf15 = _build_tf(df_15m, '15m')
    tf1h = _build_tf(df_1h, '1h')
    tf4h = _build_tf(df_4h, '4h')

    long_alignment = sum(1 for tf in (tf15, tf1h, tf4h) if tf.trend_bias == 'bullish')
    short_alignment = sum(1 for tf in (tf15, tf1h, tf4h) if tf.trend_bias == 'bearish')

    return PriceActionFeatures(
        tf_15m=tf15,
        tf_1h=tf1h,
        tf_4h=tf4h,
        alignment_long=long_alignment,
        alignment_short=short_alignment,
    )
