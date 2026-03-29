"""Binance BTC futures context: funding, OI deltas, and liquidation-proxy events."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

FUTURES_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"
BTC_SYMBOL = "BTCUSDT"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _get_json(base: str, path: str, params: dict[str, Any] | None = None) -> Any:
    resp = requests.get(f"{base}{path}", params=params or {}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_funding_rate() -> dict[str, Any]:
    """Fetch current BTC funding rate from Binance Futures."""
    try:
        data = _get_json(FUTURES_BASE, "/fapi/v1/premiumIndex", {"symbol": BTC_SYMBOL})
        rate = _safe_float(data.get("lastFundingRate"))
        mark_price = _safe_float(data.get("markPrice"))
        return {
            "funding_rate": rate,
            "funding_rate_pct": round(rate * 100.0, 4),
            "mark_price": mark_price,
            "next_funding_time": data.get("nextFundingTime"),
        }
    except Exception as exc:
        logger.warning("Futures API failed (possibly geo-blocked): %s", exc)
        try:
            data = _get_json(SPOT_BASE, "/api/v3/ticker/price", {"symbol": BTC_SYMBOL})
            return {
                "funding_rate": 0.0,
                "funding_rate_pct": 0.0,
                "mark_price": _safe_float(data.get("price")),
                "note": "futures_api_unavailable",
            }
        except Exception:
            return {"funding_rate": 0.0, "funding_rate_pct": 0.0, "mark_price": 0.0}


def get_funding_history(limit: int = 60) -> list[dict[str, Any]]:
    try:
        payload = _get_json(
            FUTURES_BASE,
            "/fapi/v1/fundingRate",
            {"symbol": BTC_SYMBOL, "limit": int(max(10, min(limit, 1000)))},
        )
        if not isinstance(payload, list):
            return []
        return payload
    except Exception as exc:
        logger.warning("Funding history fetch failed: %s", exc)
        return []


def get_open_interest() -> dict[str, Any]:
    """Fetch current BTC open interest from Binance Futures."""
    try:
        data = _get_json(FUTURES_BASE, "/fapi/v1/openInterest", {"symbol": BTC_SYMBOL})
        oi = _safe_float(data.get("openInterest"))
        return {"open_interest_btc": oi}
    except Exception as exc:
        logger.warning("Open interest fetch failed: %s", exc)
        return {"open_interest_btc": 0.0}


def get_open_interest_history(period: str = "5m", limit: int = 60) -> list[dict[str, Any]]:
    try:
        payload = _get_json(
            FUTURES_BASE,
            "/futures/data/openInterestHist",
            {
                "symbol": BTC_SYMBOL,
                "period": period,
                "limit": int(max(10, min(limit, 500))),
            },
        )
        if not isinstance(payload, list):
            return []
        return payload
    except Exception as exc:
        logger.warning("Open interest history fetch failed: %s", exc)
        return []


def get_liquidation_snapshot(limit: int = 40) -> dict[str, Any]:
    """
    Public liquidation proxy.

    Binance does not expose force-order history without signed user data, so this
    estimates cascade conditions using 5m futures candles plus concurrent OI drops.
    """
    try:
        bars = _get_json(
            FUTURES_BASE,
            "/fapi/v1/klines",
            {"symbol": BTC_SYMBOL, "interval": "5m", "limit": int(max(20, min(limit, 200)))},
        )
        if not isinstance(bars, list) or len(bars) < 12:
            return {
                "liquidation_event": False,
                "liquidation_bias": "NONE",
                "liquidation_score": 0.0,
                "price_move_pct": 0.0,
                "volume_z": 0.0,
            }

        closes = np.array([_safe_float(r[4]) for r in bars], dtype=float)
        opens = np.array([_safe_float(r[1]) for r in bars], dtype=float)
        highs = np.array([_safe_float(r[2]) for r in bars], dtype=float)
        lows = np.array([_safe_float(r[3]) for r in bars], dtype=float)
        volumes = np.array([_safe_float(r[5]) for r in bars], dtype=float)
        body = np.abs(closes - opens)
        span = np.maximum(highs - lows, 1e-9)
        returns = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        last_ret = float(returns[-1]) if len(returns) else 0.0

        volume_window = volumes[:-1] if len(volumes) > 6 else volumes
        volume_mu = float(np.mean(volume_window)) if len(volume_window) else 0.0
        volume_sd = float(np.std(volume_window)) if len(volume_window) else 0.0
        volume_z = (float(volumes[-1]) - volume_mu) / max(volume_sd, 1e-9)
        wick_ratio = float((span[-1] - body[-1]) / max(body[-1], 1e-9))

        oi_hist = get_open_interest_history(period="5m", limit=max(limit, 24))
        oi_values = np.array([_safe_float(row.get("sumOpenInterest")) for row in oi_hist], dtype=float)
        oi_delta_pct = 0.0
        if len(oi_values) >= 2 and oi_values[-2] > 0:
            oi_delta_pct = ((float(oi_values[-1]) - float(oi_values[-2])) / float(oi_values[-2])) * 100.0

        event = False
        bias = "NONE"
        if last_ret <= -0.008 and volume_z >= 2.0 and oi_delta_pct <= -0.75:
            event = True
            bias = "LONG_FLUSH"
        elif last_ret >= 0.008 and volume_z >= 2.0 and oi_delta_pct <= -0.75:
            event = True
            bias = "SHORT_SQUEEZE"

        score = abs(last_ret) * 100.0 + max(volume_z, 0.0) + max(-oi_delta_pct, 0.0)
        if wick_ratio > 1.5:
            score += 0.5

        return {
            "liquidation_event": bool(event),
            "liquidation_bias": bias,
            "liquidation_score": round(float(score), 3),
            "price_move_pct": round(last_ret * 100.0, 3),
            "volume_z": round(float(volume_z), 3),
            "oi_delta_pct_5m": round(float(oi_delta_pct), 3),
            "wick_ratio": round(float(wick_ratio), 3),
        }
    except Exception as exc:
        logger.warning("Liquidation proxy fetch failed: %s", exc)
        return {
            "liquidation_event": False,
            "liquidation_bias": "NONE",
            "liquidation_score": 0.0,
            "price_move_pct": 0.0,
            "volume_z": 0.0,
        }


def _zscore(latest: float, values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 5:
        return 0.0
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd <= 1e-9:
        return 0.0
    return float((latest - mu) / sd)


def _oi_delta_pct(values: list[float], lookback: int) -> float:
    if len(values) <= lookback:
        return 0.0
    base = float(values[-(lookback + 1)])
    latest = float(values[-1])
    if base <= 0:
        return 0.0
    return ((latest - base) / base) * 100.0


def get_futures_sentiment() -> dict[str, Any]:
    """Combined BTC futures context used by live signal gating."""
    funding = get_funding_rate()
    oi = get_open_interest()
    funding_hist = get_funding_history(limit=60)
    oi_hist = get_open_interest_history(period="5m", limit=60)
    liq = get_liquidation_snapshot(limit=40)

    current_rate = float(funding.get("funding_rate", 0.0))
    funding_series = [_safe_float(row.get("fundingRate")) for row in funding_hist]
    funding_z = _zscore(current_rate, funding_series)

    oi_series = [_safe_float(row.get("sumOpenInterest")) for row in oi_hist]
    current_oi = float(oi.get("open_interest_btc", 0.0) or (oi_series[-1] if oi_series else 0.0))
    if not oi_series and current_oi > 0:
        oi_series = [current_oi]
    if oi_series and current_oi > 0 and abs(current_oi - oi_series[-1]) > 1e-9:
        oi_series.append(current_oi)

    oi_delta_1h = _oi_delta_pct(oi_series, lookback=12)
    oi_delta_4h = _oi_delta_pct(oi_series, lookback=48)

    funding_rate_pct = float(funding.get("funding_rate_pct", 0.0))
    if funding_z > 2.0 or funding_rate_pct > 0.05:
        funding_sentiment = "OVERLEVERAGED_LONG"
    elif funding_z < -2.0 or funding_rate_pct < -0.05:
        funding_sentiment = "OVERLEVERAGED_SHORT"
    elif abs(funding_z) < 0.5 and -0.02 <= funding_rate_pct <= 0.02:
        funding_sentiment = "NEUTRAL"
    else:
        funding_sentiment = "MILD_LONG" if funding_rate_pct > 0 else "MILD_SHORT"

    derivatives_score = float(np.clip((-funding_z * 0.6) + (oi_delta_1h / 4.0), -3.0, 3.0))

    return {
        **funding,
        **oi,
        **liq,
        "funding_rate_z": round(float(funding_z), 3),
        "oi_delta_pct_1h": round(float(oi_delta_1h), 3),
        "oi_delta_pct_4h": round(float(oi_delta_4h), 3),
        "funding_sentiment": funding_sentiment,
        "derivatives_score": round(derivatives_score, 3),
        "data_points": {
            "funding_samples": len(funding_series),
            "oi_samples": len(oi_series),
        },
    }
