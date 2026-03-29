"""Multi-timeframe bias validation for BTC signals."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


class MTFBiasFilter:
    def __init__(self, btc_service):
        self._svc = btc_service
        self._cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._cache_ttl = 300

    @staticmethod
    def _empty_bias(timeframe: str, cached: bool = False) -> dict[str, Any]:
        return {
            "timeframe": timeframe,
            "bias": "NEUTRAL",
            "ema21": 0.0,
            "ema50": 0.0,
            "price": 0.0,
            "structure": "mixed",
            "cached": cached,
        }

    def _get_frame(self, timeframe: str) -> tuple[pd.DataFrame, bool]:
        now = time.time()
        cached_item = self._cache.get(timeframe)
        if cached_item is not None:
            cached_df, cached_ts = cached_item
            if (now - cached_ts) < self._cache_ttl:
                return cached_df.copy(), True

        df = self._svc.get_recent_frame(interval=timeframe, limit=240)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        self._cache[timeframe] = (df.copy(), now)
        return df, False

    def get_bias(self, timeframe: str) -> dict:
        """
        Fetch candles for given timeframe, compute bias.
        """
        df, cached = self._get_frame(timeframe)
        if df.empty or "Close" not in df.columns or len(df) < 55:
            return self._empty_bias(timeframe, cached=cached)

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty or len(close) < 55:
            return self._empty_bias(timeframe, cached=cached)

        ema21_series = close.ewm(span=21).mean()
        ema50_series = close.ewm(span=50).mean()

        price = float(close.iloc[-1])
        ema21 = float(ema21_series.iloc[-1])
        ema50 = float(ema50_series.iloc[-1])
        slope_pos = bool(ema21_series.iloc[-1] > ema21_series.iloc[-2] > ema21_series.iloc[-3])
        slope_neg = bool(ema21_series.iloc[-1] < ema21_series.iloc[-2] < ema21_series.iloc[-3])

        if price > ema21 > ema50 and slope_pos:
            bias = "BULLISH"
            structure = "price > EMA21 > EMA50"
        elif price < ema21 < ema50 and slope_neg:
            bias = "BEARISH"
            structure = "price < EMA21 < EMA50"
        else:
            bias = "NEUTRAL"
            structure = "mixed structure"

        return {
            "timeframe": timeframe,
            "bias": bias,
            "ema21": round(float(ema21), 4),
            "ema50": round(float(ema50), 4),
            "price": round(float(price), 4),
            "structure": structure,
            "cached": cached,
        }

    @staticmethod
    def _long_alignment_score(bias_4h: str, bias_1d: str) -> float:
        if bias_4h == "BULLISH" and bias_1d == "BULLISH":
            return 1.0
        if bias_4h == "BULLISH" and bias_1d == "NEUTRAL":
            return 0.75
        if bias_4h == "NEUTRAL" and bias_1d == "BULLISH":
            return 0.6
        if bias_4h == "NEUTRAL" and bias_1d == "NEUTRAL":
            return 0.4
        if bias_4h == "BULLISH" and bias_1d == "BEARISH":
            return 0.5
        if bias_4h == "NEUTRAL" and bias_1d == "BEARISH":
            return 0.4
        return 0.0

    @classmethod
    def _short_alignment_score(cls, bias_4h: str, bias_1d: str) -> float:
        # Mirror long map
        mirror_4h = "BULLISH" if bias_4h == "BEARISH" else "BEARISH" if bias_4h == "BULLISH" else "NEUTRAL"
        mirror_1d = "BULLISH" if bias_1d == "BEARISH" else "BEARISH" if bias_1d == "BULLISH" else "NEUTRAL"
        return cls._long_alignment_score(mirror_4h, mirror_1d)

    def check_alignment(self, signal: str, entry_timeframe: str = "15m", confidence: float | None = None) -> dict:
        """
        Check if signal aligns with 4H and 1D bias.
        """
        signal_u = str(signal or "").upper()
        if signal_u not in {"LONG", "SHORT"}:
            bias_4h_row = self.get_bias("4h")
            bias_1d_row = self.get_bias("1d")
            return {
                "alignment_ok": True,
                "bias_4h": str(bias_4h_row.get("bias", "NEUTRAL")),
                "bias_1d": str(bias_1d_row.get("bias", "NEUTRAL")),
                "block_reason": None,
                "alignment_score": 1.0,
                "entry_timeframe": entry_timeframe,
                "raw_4h": bias_4h_row,
                "raw_1d": bias_1d_row,
            }

        conf = float(confidence or 0.0)
        bias_4h_row = self.get_bias("4h")
        bias_1d_row = self.get_bias("1d")
        bias_4h = str(bias_4h_row.get("bias", "NEUTRAL"))
        bias_1d = str(bias_1d_row.get("bias", "NEUTRAL"))

        alignment_ok = True
        block_reason: str | None = None
        if signal_u == "LONG":
            if bias_4h == "BEARISH":
                alignment_ok = False
                block_reason = "4H structure bearish: price < EMA21 < EMA50"
            elif bias_1d == "BEARISH" and conf < 80.0:
                alignment_ok = False
                block_reason = "1D bearish, need 80%+ confidence for counter-trend LONG"
            score = self._long_alignment_score(bias_4h, bias_1d)
        else:
            if bias_4h == "BULLISH":
                alignment_ok = False
                block_reason = "4H structure bullish: price > EMA21 > EMA50"
            elif bias_1d == "BULLISH" and conf < 80.0:
                alignment_ok = False
                block_reason = "1D bullish, need 80%+ confidence for counter-trend SHORT"
            score = self._short_alignment_score(bias_4h, bias_1d)

        return {
            "alignment_ok": alignment_ok,
            "bias_4h": bias_4h,
            "bias_1d": bias_1d,
            "block_reason": block_reason,
            "alignment_score": float(np.clip(score, 0.0, 1.0)),
            "entry_timeframe": entry_timeframe,
            "raw_4h": bias_4h_row,
            "raw_1d": bias_1d_row,
        }
