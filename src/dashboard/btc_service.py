"""
Bitcoin market data + signal service backed by Binance public APIs.

Provides:
  - all-time historical candles via paginated /api/v3/klines
  - recent candles for real-time quant signal generation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.alpha.factor_model import AlphaFactorModel
from src.api.data_quality import DataAnomalyDetector
from src.api.rate_limiter import TTLCache
from src.features.engineer import FeatureEngineer
from src.risk.cost_model import CostModel

logger = logging.getLogger(__name__)

BINANCE_REST = "https://api.binance.com"
BTC_SYMBOL = "BTCUSDT"
BINANCE_BTC_EARLIEST_MS = 1502942400000  # 2017-08-17T04:00:00Z approx listing start

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


class BitcoinMarketService:
    def __init__(self, cfg: dict | None = None):
        config = cfg or {}
        signal_cfg = config.get("signal", {}) or {}

        self._session = requests.Session()
        self._history_cache = TTLCache()
        self._recent_cache = TTLCache()
        self._signal_cache = TTLCache()

        self._engineer = FeatureEngineer()
        self._alpha = AlphaFactorModel(
            alpha_threshold=float(signal_cfg.get("alpha_score_threshold", 0.15)),
            ic_window=int(signal_cfg.get("ic_window", 60)),
        )
        self._anomaly = DataAnomalyDetector()
        self._cost = CostModel(config.get("cost_model", {}) or {})

    def get_all_time_history(self, interval: str = "1d") -> dict[str, Any]:
        interval = interval if interval in INTERVAL_TO_MS else "1d"
        cache_key = f"btc_all_{interval}"
        cached = self._history_cache.get(cache_key)
        if cached is not None:
            return cached

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = BINANCE_BTC_EARLIEST_MS
        all_rows: list[list[Any]] = []

        while True:
            rows = self._fetch_klines(interval=interval, start_time_ms=start_ms, limit=1000)
            if not rows:
                break
            all_rows.extend(rows)

            last_open_ms = int(rows[-1][0])
            step = INTERVAL_TO_MS[interval]
            next_start = last_open_ms + step
            if next_start <= start_ms:
                break
            start_ms = next_start
            if len(rows) < 1000 or start_ms >= now_ms:
                break

        df = self._klines_to_df(all_rows)
        payload = self._history_payload(df, interval=interval)
        # daily history barely changes, cache longer
        ttl = 3600 if interval == "1d" else 600
        self._history_cache.set(cache_key, payload, ttl_seconds=ttl)
        return payload

    def get_recent_frame(self, interval: str = "5m", limit: int = 1200) -> pd.DataFrame:
        interval = interval if interval in INTERVAL_TO_MS else "5m"
        cache_key = f"btc_recent_{interval}_{limit}"
        cached = self._recent_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = self._fetch_klines(interval=interval, limit=limit)
        df = self._klines_to_df(rows)
        self._recent_cache.set(cache_key, df, ttl_seconds=8)
        return df

    def get_realtime_signal(self, interval: str = "5m") -> dict[str, Any]:
        interval = interval if interval in INTERVAL_TO_MS else "5m"
        cache_key = f"btc_signal_{interval}"
        cached = self._signal_cache.get(cache_key)
        if cached is not None:
            return cached

        df = self.get_recent_frame(interval=interval, limit=1200)
        if df.empty or len(df) < 120:
            payload = {
                "asset": "BTCUSDT",
                "signal": "HOLD",
                "validated_signal": "HOLD",
                "validated": False,
                "reason": "Insufficient Binance candles",
                "as_of_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._signal_cache.set(cache_key, payload, ttl_seconds=5)
            return payload

        timeframe = "daily" if interval in {"1d", "1w", "1M"} else "intraday"
        clean_df, dq = self._anomaly.inspect_and_clean(df, "BTCUSDT", "crypto", timeframe)
        feat = self._engineer.compute_all_features(clean_df, timeframe=timeframe)

        if feat.empty or len(feat) < self._alpha.ic_window + 5:
            payload = {
                "asset": "BTCUSDT",
                "signal": "HOLD",
                "validated_signal": "HOLD",
                "validated": False,
                "reason": "Feature window not ready",
                "data_quality": dq.to_dict(),
                "as_of_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._signal_cache.set(cache_key, payload, ttl_seconds=5)
            return payload

        alpha = self._alpha.score(feat, ml_score=0.0, hurst=0.5)
        entry = float(feat["Close"].iloc[-1])
        atr = self._last_feature_or_default(feat, "ATR_14", entry * 0.02)
        daily_vol = self._last_feature_or_default(feat, "Volatility_20", 0.5)
        volume_ratio = self._last_feature_or_default(feat, "Volume_Ratio", 1.0)
        net_alpha, cost_pct, viable = self._cost.net_alpha(
            alpha_score=float(alpha["alpha_score"]),
            asset_class="forex",
            position_size_pct=1.0,
            daily_vol=daily_vol,
            volume_ratio=max(volume_ratio, 0.1),
            regime="SIDEWAYS",
            low_liquidity=volume_ratio < 0.8,
        )

        strength = str(alpha["strength"])
        sl_mult = 1.5 if strength == "STRONG" else 2.0 if strength == "MODERATE" else 2.5
        rr = 3.0 if strength == "STRONG" else 2.5 if strength == "MODERATE" else 2.0
        raw_signal = str(alpha["signal"])

        if raw_signal == "BUY":
            stop = entry - sl_mult * atr
            take = entry + (entry - stop) * rr
        elif raw_signal == "SELL":
            stop = entry + sl_mult * atr
            take = entry - (stop - entry) * rr
        else:
            stop = entry
            take = entry

        checks = {
            "data_quality_ok": not dq.severe,
            "confidence_ok": float(alpha["confidence"]) >= 55.0,
            "cost_ok": bool(viable),
        }
        validated = raw_signal in {"BUY", "SELL"} and all(checks.values())

        payload = {
            "asset": "BTCUSDT",
            "interval": interval,
            "signal": raw_signal,
            "validated_signal": raw_signal if validated else "HOLD",
            "validated": validated,
            "strength": strength,
            "confidence": round(float(alpha["confidence"]), 2),
            "alpha_score": round(float(alpha["alpha_score"]), 4),
            "net_alpha_score": round(float(net_alpha), 4),
            "cost_pct": round(float(cost_pct), 5),
            "entry_price": round(entry, 2),
            "stop_loss": round(float(stop), 2),
            "take_profit": round(float(take), 2),
            "factor_scores": alpha.get("factor_scores", {}),
            "ic_weights": alpha.get("ic_weights", {}),
            "validation_checks": checks,
            "data_quality": dq.to_dict(),
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "algo": "quant_alpha_factor_model_v1",
        }
        self._signal_cache.set(cache_key, payload, ttl_seconds=5)
        return payload

    @staticmethod
    def _last_feature_or_default(feat: pd.DataFrame, column: str, default: float) -> float:
        """
        Return the latest value from a feature column with robust numeric fallback.
        """
        if column in feat.columns and not feat.empty:
            value = feat[column].iloc[-1]
            try:
                value_f = float(value)
                if np.isfinite(value_f):
                    return value_f
            except Exception:
                pass
        return float(default)

    def _fetch_klines(
        self,
        interval: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        params: dict[str, Any] = {
            "symbol": BTC_SYMBOL,
            "interval": interval,
            "limit": min(max(limit, 1), 1000),
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        url = f"{BINANCE_REST}/api/v3/klines"
        try:
            resp = self._session.get(url, params=params, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Binance klines fetch failed: %s", exc)
            return []

    @staticmethod
    def _klines_to_df(rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        records = []
        for r in rows:
            try:
                open_ms = int(r[0])
                records.append(
                    {
                        "time": pd.to_datetime(open_ms, unit="ms", utc=True),
                        "Open": float(r[1]),
                        "High": float(r[2]),
                        "Low": float(r[3]),
                        "Close": float(r[4]),
                        "Volume": float(r[5]),
                    }
                )
            except Exception:
                continue
        if not records:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(records).drop_duplicates(subset=["time"]).set_index("time").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]

    @staticmethod
    def _history_payload(df: pd.DataFrame, interval: str) -> dict[str, Any]:
        if df.empty:
            return {"asset": "BTCUSDT", "interval": interval, "data": [], "error": "No history"}
        data = []
        for ts, row in df.iterrows():
            t = pd.Timestamp(ts).to_pydatetime()
            data.append(
                {
                    "time": int(t.timestamp()),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": round(float(row["Volume"]), 6),
                }
            )
        return {
            "asset": "BTCUSDT",
            "interval": interval,
            "points": len(data),
            "start_utc": pd.Timestamp(df.index[0]).isoformat(),
            "end_utc": pd.Timestamp(df.index[-1]).isoformat(),
            "data": data,
        }
