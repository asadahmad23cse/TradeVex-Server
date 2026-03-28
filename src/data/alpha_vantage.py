"""Alpha Vantage helpers for US stock quote/candles/signal data."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

try:
    from dotenv import load_dotenv  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency path
    load_dotenv = None  # type: ignore[assignment]

from src.alpha.factor_model import AlphaFactorModel
from src.features.engineer import FeatureEngineer

if load_dotenv is not None:
    load_dotenv()

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
BASE_URL = "https://www.alphavantage.co/query"
TIMEOUT_SECONDS = 10
DAILY_LIMIT = 23

_cache: dict[str, dict[str, Any]] = {}
_request_day_utc = ""
_request_count = 0

_INTRADAY_MAP = {
    "5m": "5min",
    "5min": "5min",
    "15m": "15min",
    "15min": "15min",
    "1h": "60min",
    "60min": "60min",
}


def _day_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _sync_counter_day() -> None:
    global _request_day_utc, _request_count
    day = _day_utc()
    if _request_day_utc != day:
        _request_day_utc = day
        _request_count = 0


def _cache_get(key: str, ttl: int) -> Any | None:
    item = _cache.get(key)
    if not item:
        return None
    if (time.time() - float(item.get("ts", 0.0))) <= ttl:
        return item.get("data")
    return None


def _cache_set(key: str, data: Any) -> None:
    _cache[key] = {"ts": time.time(), "data": data}


def _cached_or_default(key: str, default: Any) -> Any:
    entry = _cache.get(key)
    if entry and "data" in entry:
        return entry["data"]
    return default


def _remaining_requests() -> int:
    _sync_counter_day()
    return max(DAILY_LIMIT - _request_count, 0)


def get_status() -> dict[str, int]:
    """Expose Alpha Vantage budget status."""
    return {"remaining_requests": _remaining_requests(), "daily_limit": DAILY_LIMIT}


def _is_rate_limited(payload: Any) -> bool:
    return isinstance(payload, dict) and ("Note" in payload or "Information" in payload)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _fetch(params: dict[str, str]) -> tuple[dict[str, Any] | None, bool]:
    """
    Returns: (payload, was_rate_limited)
    `was_rate_limited=True` means caller should return cached fallback.
    """
    global _request_count
    _sync_counter_day()

    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("ALPHA_VANTAGE_API_KEY missing in environment")
        return None, False

    if _request_count >= DAILY_LIMIT:
        logger.warning("Alpha Vantage daily budget exhausted (%s)", DAILY_LIMIT)
        return None, True

    request_params = dict(params)
    request_params["apikey"] = ALPHA_VANTAGE_API_KEY

    try:
        _request_count += 1
        resp = requests.get(BASE_URL, params=request_params, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
        if _is_rate_limited(payload):
            logger.warning("Alpha Vantage rate limited: %s", payload)
            return None, True
        if not isinstance(payload, dict):
            logger.warning("Alpha Vantage payload is not a dict")
            return None, False
        return payload, False
    except Exception as e:
        logger.warning("Alpha Vantage fetch failed: %s", e)
        return None, False


def _to_unix(ts: str) -> int | None:
    try:
        dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return int(dt.timestamp())
    except Exception:
        return None


def _parse_candles(series: dict[str, Any]) -> list[dict[str, Any]]:
    candles: list[dict[str, Any]] = []
    for t, vals in series.items():
        if not isinstance(vals, dict):
            continue
        unix = _to_unix(str(t))
        if unix is None:
            continue
        candles.append(
            {
                "time": unix,
                "open": _safe_float(vals.get("1. open")),
                "high": _safe_float(vals.get("2. high")),
                "low": _safe_float(vals.get("3. low")),
                "close": _safe_float(vals.get("4. close")),
                "volume": _safe_int(vals.get("5. volume")),
            }
        )
    candles.sort(key=lambda x: int(x["time"]))
    return candles


def _candles_to_df(candles: list[dict[str, Any]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles).copy()
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    if df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            df[col] = 0.0
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def get_quote(ticker: str) -> dict[str, Any]:
    """GLOBAL_QUOTE with 60s cache."""
    symbol = (ticker or "").upper().strip()
    key = f"quote::{symbol}"
    try:
        cached = _cache_get(key, ttl=60)
        if cached is not None:
            return cached

        payload, limited = _fetch({"function": "GLOBAL_QUOTE", "symbol": symbol})
        if payload is None:
            return _cached_or_default(key, {} if not limited else [])

        q = payload.get("Global Quote", {})
        if not isinstance(q, dict) or not q:
            return _cached_or_default(key, {})

        out = {
            "price": _safe_float(q.get("05. price")),
            "change": _safe_float(q.get("09. change")),
            "change_pct": _safe_float(str(q.get("10. change percent", "0")).replace("%", "")),
            "volume": _safe_int(q.get("06. volume")),
            "high": _safe_float(q.get("03. high")),
            "low": _safe_float(q.get("04. low")),
        }
        _cache_set(key, out)
        return out
    except Exception as e:
        logger.warning("get_quote failed for %s: %s", symbol, e)
        return _cached_or_default(key, {})


def get_intraday_candles(ticker: str, interval: str = "5min") -> list[dict[str, Any]]:
    """TIME_SERIES_INTRADAY with 60s cache."""
    symbol = (ticker or "").upper().strip()
    av_interval = _INTRADAY_MAP.get(str(interval).lower(), "5min")
    key = f"intraday::{symbol}::{av_interval}"
    try:
        cached = _cache_get(key, ttl=60)
        if cached is not None:
            return cached

        payload, limited = _fetch(
            {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": av_interval,
            }
        )
        if payload is None:
            return _cached_or_default(key, [] if limited else [])

        ts_key = f"Time Series ({av_interval})"
        series = payload.get(ts_key, {})
        if not isinstance(series, dict) or not series:
            return _cached_or_default(key, [])

        candles = _parse_candles(series)
        _cache_set(key, candles)
        return candles
    except Exception as e:
        logger.warning("get_intraday_candles failed for %s: %s", symbol, e)
        return _cached_or_default(key, [])


def get_daily_candles(ticker: str) -> list[dict[str, Any]]:
    """TIME_SERIES_DAILY with 300s cache."""
    symbol = (ticker or "").upper().strip()
    key = f"daily::{symbol}"
    try:
        cached = _cache_get(key, ttl=300)
        if cached is not None:
            return cached

        payload, limited = _fetch({"function": "TIME_SERIES_DAILY", "symbol": symbol})
        if payload is None:
            return _cached_or_default(key, [] if limited else [])

        series = payload.get("Time Series (Daily)", {})
        if not isinstance(series, dict) or not series:
            return _cached_or_default(key, [])

        candles = _parse_candles(series)
        _cache_set(key, candles)
        return candles
    except Exception as e:
        logger.warning("get_daily_candles failed for %s: %s", symbol, e)
        return _cached_or_default(key, [])


def get_us_stock_signal(ticker: str) -> dict[str, Any]:
    """Run existing AlphaFactorModel on AV daily candles. Cache TTL = 120s."""
    symbol = (ticker or "").upper().strip()
    key = f"signal::{symbol}"
    try:
        cached = _cache_get(key, ttl=120)
        if cached is not None:
            return cached

        candles = get_daily_candles(symbol)
        if not candles or len(candles) < 30:
            return _cached_or_default(key, {})

        df = _candles_to_df(candles)
        if df.empty or len(df) < 30:
            return _cached_or_default(key, {})

        eng = FeatureEngineer()
        feat_df = eng.compute_all_features(df, timeframe="daily")
        if feat_df.empty:
            return _cached_or_default(key, {})

        model = AlphaFactorModel(alpha_threshold=0.01)
        result = model.score(feat_df, asset=symbol, asset_class="us_stock")

        entry_price = float(feat_df["Close"].iloc[-1])
        atr = float(feat_df.get("ATR_14", feat_df["Close"] * 0.02).iloc[-1])
        sig = str(result.get("signal", "HOLD")).upper()
        stop_loss = entry_price - 2 * atr if sig == "BUY" else entry_price + 2 * atr
        take_profit = entry_price + 3 * atr if sig == "BUY" else entry_price - 3 * atr

        out = {
            "ticker": symbol,
            "asset": symbol,
            "signal": sig,
            "strength": result.get("strength", "WEAK"),
            "confidence": _safe_float(result.get("confidence", 50)),
            "alpha_score": _safe_float(result.get("alpha_score", 0)),
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "factor_scores": result.get("factor_scores", {}),
            "ic_weights": result.get("ic_weights", {}),
            "live": True,
            "provider": "alpha_vantage",
        }
        _cache_set(key, out)
        return out
    except Exception as e:
        logger.warning("get_us_stock_signal failed for %s: %s", symbol, e)
        return _cached_or_default(key, {})
