from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolatilityFeatures:
    atr14: float
    atr_multiplier_pct: float
    normal_atr: float
    vol_regime: str
    bb_squeeze: bool
    bb_expansion: bool
    bb_position: float
    vwap: float
    vwap_distance_pct: float
    vwap_bias: str
    realized_vol: float
    iv_rv_ratio: float


def compute_volatility(df_15m: pd.DataFrame, df_1h: pd.DataFrame, deribit_data: dict | None = None) -> VolatilityFeatures:
    if len(df_15m) < 120 or len(df_1h) < 100:
        return VolatilityFeatures(0.0, 0.0, 0.0, 'compression', False, False, 0.0, 0.0, 0.0, 'below', 0.0, 1.0)

    high = df_15m['High'].astype(float)
    low = df_15m['Low'].astype(float)
    close = df_15m['Close'].astype(float)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    atr14 = float(atr.iloc[-1])
    normal_atr = float(atr.tail(100).mean())
    price = float(close.iloc[-1])
    atr_multiplier = (atr14 / price * 100.0) if price > 0 else 0.0

    ratio = (atr14 / normal_atr) if normal_atr > 0 else 0.0
    if ratio > 2.5:
        vol_regime = 'spike'
    elif ratio > 1.5:
        vol_regime = 'expansion'
    elif ratio < 0.7:
        vol_regime = 'compression'
    else:
        vol_regime = 'normal'

    c1h = df_1h['Close'].astype(float)
    ma = c1h.rolling(20).mean()
    std = c1h.rolling(20).std()
    upper = ma + 2.0 * std
    lower = ma - 2.0 * std
    width = (upper - lower) / ma.replace(0, np.nan)
    width_series = width.dropna().tail(100)

    if len(width_series) >= 20:
        p20 = float(np.percentile(width_series, 20))
        p80 = float(np.percentile(width_series, 80))
        bb_squeeze = float(width.iloc[-1]) < p20
        bb_expansion = float(width.iloc[-1]) > p80
    else:
        bb_squeeze = False
        bb_expansion = False

    last_upper = float(upper.iloc[-1])
    last_lower = float(lower.iloc[-1])
    if last_upper == last_lower:
        bb_pos = 0.0
    else:
        bb_pos = ((float(c1h.iloc[-1]) - last_lower) / (last_upper - last_lower)) * 2 - 1
        bb_pos = float(max(-1.0, min(1.0, bb_pos)))

    work = df_15m.copy()
    work['date'] = work.index.date
    work['pv'] = work['Close'] * work['Volume']
    work['cum_pv'] = work.groupby('date')['pv'].cumsum()
    work['cum_vol'] = work.groupby('date')['Volume'].cumsum()
    vwap = float((work['cum_pv'] / work['cum_vol'].replace(0, np.nan)).iloc[-1])
    vwap_dist = ((price - vwap) / vwap * 100.0) if vwap > 0 else 0.0
    vwap_bias = 'above' if price > vwap else 'below'

    log_ret = np.log(close / close.shift(1)).dropna()
    rv = float(log_ret.std() * np.sqrt(96)) if len(log_ret) > 5 else 0.0

    atm_iv = float((deribit_data or {}).get('atm_iv', 0.0))
    iv_rv_ratio = (atm_iv / rv) if rv > 1e-9 and atm_iv > 0 else 1.0

    return VolatilityFeatures(
        atr14=atr14,
        atr_multiplier_pct=atr_multiplier,
        normal_atr=normal_atr,
        vol_regime=vol_regime,
        bb_squeeze=bb_squeeze,
        bb_expansion=bb_expansion,
        bb_position=bb_pos,
        vwap=vwap,
        vwap_distance_pct=vwap_dist,
        vwap_bias=vwap_bias,
        realized_vol=rv,
        iv_rv_ratio=iv_rv_ratio,
    )
