"""
Bitcoin market data + signal service backed by Binance public APIs.

Provides:
  - all-time historical candles via paginated /api/v3/klines
  - recent candles for real-time quant signal generation
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.alpha.factor_model import AlphaFactorModel, _rolling_ic
from src.api.data_quality import DataAnomalyDetector
from src.api.rate_limiter import TTLCache
from src.data.signal_history import check_open_signals, record_signal
from src.features.engineer import FeatureEngineer
from src.risk.cost_model import CostModel

logger = logging.getLogger(__name__)
_last_signal_time: dict[str, float] = {}  # {"LONG": timestamp, "SHORT": timestamp}
COOLDOWN_SECONDS = 4 * 3600  # 4 hours between same-direction signals

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
        self._marker_cache = TTLCache()

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
        ttl = 3600 if interval == "1d" else 600
        self._history_cache.set(cache_key, payload, ttl_seconds=ttl)
        return payload

    def get_recent_candles(self, interval: str = "15m", limit: int = 200) -> dict[str, Any]:
        interval = interval if interval in INTERVAL_TO_MS else "15m"
        limit = int(max(50, min(limit, 1000)))
        df = self.get_recent_frame(interval=interval, limit=limit)
        return self._history_payload(df.tail(limit), interval=interval)

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
        raw_alpha = float(alpha.get("alpha_score", 0.0))
        confidence = float(alpha.get("confidence", 50.0))
        model_signal = str(alpha.get("signal", "HOLD")).upper()
        signal = "LONG" if model_signal == "BUY" else "SHORT" if model_signal == "SELL" else "HOLD"
        reason = ""

        # --- SESSION FILTER: block signals during low-edge hours ---
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        is_dead_session = 21 <= hour or hour < 1   # 21:00-01:00 UTC
        is_asia_low_vol = 1 <= hour < 7            # deep Asia, low vol

        if signal in {"LONG", "SHORT"} and is_dead_session:
            signal = "HOLD"
            reason = f"HOLD: Dead session ({hour:02d}:00 UTC) — no signals generated"

        if signal in {"LONG", "SHORT"} and is_asia_low_vol and confidence < 75:
            signal = "HOLD"
            reason = f"HOLD: Asia session low confidence ({confidence:.0f}% < 75% required)"

        # --- TREND ALIGNMENT FILTER: don't trade against the trend ---
        if signal in {"LONG", "SHORT"} and "Close" in feat.columns and len(feat) > 50:
            close = float(feat["Close"].iloc[-1])
            ema_21 = float(feat["Close"].ewm(span=21).mean().iloc[-1])
            ema_50 = float(feat["Close"].ewm(span=50).mean().iloc[-1])

            trend_bullish = close > ema_21 > ema_50
            trend_bearish = close < ema_21 < ema_50

            if signal == "LONG" and trend_bearish:
                signal = "HOLD"
                reason = "HOLD: LONG signal rejected — price below EMA21 < EMA50 (bearish structure)"
            elif signal == "SHORT" and trend_bullish:
                signal = "HOLD"
                reason = "HOLD: SHORT signal rejected — price above EMA21 > EMA50 (bullish structure)"

        # --- FUNDING RATE FILTER (proven edge) ---
        futures = {
            "funding_rate_pct": 0,
            "funding_sentiment": "NEUTRAL",
            "open_interest_btc": 0,
            "mark_price": 0,
        }
        funding_confirms = False
        try:
            from src.data.futures_data import get_futures_sentiment
            futures = get_futures_sentiment()
            logger.info(
                "Futures data: funding=%.4f%%, OI=%.1f",
                futures.get("funding_rate_pct", 0),
                futures.get("open_interest_btc", 0),
            )
        except Exception as e:
            logger.warning("Futures data unavailable (may be geo-blocked): %s", e)

        funding_rate = futures.get("funding_rate_pct", 0)
        funding_sentiment = futures.get("funding_sentiment", "NEUTRAL")

        # Block LONG when market is overleveraged long (funding > +0.05%)
        if signal == "LONG" and funding_sentiment == "OVERLEVERAGED_LONG":
            signal = "HOLD"
            reason = f"HOLD: LONG blocked â€” funding rate {funding_rate:+.4f}% (overleveraged longs, squeeze risk)"

        # Block SHORT when market is overleveraged short (funding < -0.05%)
        if signal == "SHORT" and funding_sentiment == "OVERLEVERAGED_SHORT":
            signal = "HOLD"
            reason = f"HOLD: SHORT blocked â€” funding rate {funding_rate:+.4f}% (overleveraged shorts, squeeze risk)"

        # BONUS: When funding is extreme AGAINST your signal = extra confidence
        # (shorts paying = bullish for longs, longs paying = bearish for shorts)
        if signal == "LONG" and funding_rate < -0.03:
            funding_confirms = True  # shorts are paying â€” bullish
        if signal == "SHORT" and funding_rate > 0.03:
            funding_confirms = True  # longs are paying â€” bearish

        # --- COOLDOWN FILTER ---
        if signal in {"LONG", "SHORT"}:
            cooldown_signal = signal
            last_ts = _last_signal_time.get(cooldown_signal, 0.0)
            elapsed = _time.time() - last_ts
            if elapsed < COOLDOWN_SECONDS:
                remaining_h = (COOLDOWN_SECONDS - elapsed) / 3600.0
                signal = "HOLD"
                reason = f"HOLD: {cooldown_signal} cooldown active — {remaining_h:.1f}h remaining"

        entry = float(feat["Close"].iloc[-1])
        atr = self._last_feature_or_default(feat, "ATR_14", entry * 0.02)
        daily_vol = self._last_feature_or_default(feat, "Volatility_20", 0.5)
        volume_ratio = self._last_feature_or_default(feat, "Volume_Ratio", 1.0)
        net_alpha, cost_pct, viable = self._cost.net_alpha(
            alpha_score=raw_alpha,
            asset_class="forex",
            position_size_pct=1.0,
            daily_vol=daily_vol,
            volume_ratio=max(volume_ratio, 0.1),
            regime="SIDEWAYS",
            low_liquidity=volume_ratio < 0.8,
        )

        strength = str(alpha.get("strength", "WEAK"))
        sl_mult = 1.5 if strength == "STRONG" else 2.0 if strength == "MODERATE" else 2.5

        stop: float | None = None
        tp1: float | None = None
        tp2: float | None = None
        tp3: float | None = None
        take: float | None = None
        entry_zone_low: float | None = None
        entry_zone_high: float | None = None
        sl_pct: float | None = None
        tp1_pct: float | None = None
        tp2_pct: float | None = None
        tp3_pct: float | None = None
        risk_reward: float | None = None
        position_size_pct: float | None = None

        if signal == "LONG":
            stop = entry - sl_mult * atr
        elif signal == "SHORT":
            stop = entry + sl_mult * atr

        if signal in {"LONG", "SHORT"} and stop is not None:
            sl_dist = abs(entry - stop)
            atr_ratio = atr / (entry * 0.01) if entry > 0 else 1.0
            if atr_ratio < 0.3:
                tp1_mult, tp2_mult, tp3_mult = 1.2, 2.0, 3.0
            elif atr_ratio > 0.8:
                tp1_mult, tp2_mult, tp3_mult = 2.0, 3.5, 6.0
            else:
                tp1_mult, tp2_mult, tp3_mult = 1.5, 2.5, 4.0
            if signal == "LONG":
                tp1 = entry + sl_dist * tp1_mult
                tp2 = entry + sl_dist * tp2_mult
                tp3 = entry + sl_dist * tp3_mult
                entry_zone_low = entry - sl_dist * 0.25
                entry_zone_high = entry + sl_dist * 0.25
            else:
                tp1 = entry - sl_dist * tp1_mult
                tp2 = entry - sl_dist * tp2_mult
                tp3 = entry - sl_dist * tp3_mult
                entry_zone_low = entry - sl_dist * 0.25
                entry_zone_high = entry + sl_dist * 0.25
            take = tp2
            sl_pct = (sl_dist / entry) * 100 if entry > 0 else None
            tp1_pct = ((tp1 - entry) / entry) * 100 if entry > 0 and tp1 is not None else None
            tp2_pct = ((tp2 - entry) / entry) * 100 if entry > 0 and tp2 is not None else None
            tp3_pct = ((tp3 - entry) / entry) * 100 if entry > 0 and tp3 is not None else None
            risk_reward = abs((tp2 - entry) / (entry - stop)) if (tp2 is not None and entry != stop) else None

        regime = self._infer_regime(feat)
        validated = False

        # --- VOLATILITY SPIKE FILTER ---
        if signal in {"LONG", "SHORT"} and regime == "VOLATILITY SPIKE":
            signal = "HOLD"
            reason = "HOLD: Volatility spike detected (ATR > 2.5x normal) — no signals"
            stop = None
            take = None
            tp1 = None
            tp2 = None
            tp3 = None
            entry_zone_low = None
            entry_zone_high = None
            sl_pct = None
            tp1_pct = None
            tp2_pct = None
            tp3_pct = None
            risk_reward = None
            validated = False

        checks = {
            "data_quality_ok": not dq.severe,
            "confidence_ok": confidence >= 62.0,
            "cost_ok": bool(viable),
        }

        if signal in {"LONG", "SHORT"} and sl_pct is not None:
            checks["sl_distance_ok"] = 0.3 <= sl_pct <= 2.5
            if not checks["sl_distance_ok"]:
                reason = f"SL distance {sl_pct:.2f}% outside 0.3-2.5% range"
                signal = "HOLD"
                stop = None
                take = None
                tp1 = None
                tp2 = None
                tp3 = None
                entry_zone_low = None
                entry_zone_high = None
                sl_pct = None
                tp1_pct = None
                tp2_pct = None
                tp3_pct = None
                risk_reward = None

        validated = signal in {"LONG", "SHORT"} and all(checks.values())
        if validated:
            position_size_pct = round(float(np.clip((confidence - 50.0) / 10.0, 0.5, 5.0)), 2)
            if signal in {"LONG", "SHORT"}:
                _last_signal_time[signal] = _time.time()

        session_name = self._session_name(datetime.now(timezone.utc))

        if signal == "HOLD":
            if not reason:
                failed = []
                if not checks.get("confidence_ok", True):
                    failed.append(f"confidence {confidence:.0f}% < threshold")
                if not checks.get("cost_ok", True):
                    failed.append(f"net alpha ({net_alpha:.2f}) below cost")
                if not checks.get("data_quality_ok", True):
                    failed.append("data quality issue")
                if confidence >= 60 and not checks.get("cost_ok", True):
                    failed.append("alpha score too low to cover trading costs")
                reason = f"HOLD: {', '.join(failed)}" if failed else "Alpha below signal threshold"
            entry_out = None
            stop_out = None
            take_out = None
        else:
            entry_out = round(entry, 2)
            stop_out = round(float(stop), 2) if stop is not None else None
            take_out = round(float(take), 2) if take is not None else None
            if not reason:
                if validated:
                    factor_names = {
                        "F1": "Momentum",
                        "F2": "MeanRev",
                        "F3": "Volume",
                        "F4": "ML",
                        "F5": "VolSqueeze",
                        "F8": "Microstructure",
                    }
                    parts = [regime, session_name, f"Conf:{confidence:.0f}%", strength]
                    fs = alpha.get("factor_scores", {})
                    for fk, fv in sorted(fs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
                        name = factor_names.get(fk, fk)
                        parts.append(f"{name}({fv:+.1f})")
                    reason = " | ".join(parts)
                else:
                    failed = []
                    if not checks.get("confidence_ok", True):
                        failed.append(f"confidence {confidence:.0f}% < threshold")
                    if not checks.get("cost_ok", True):
                        failed.append(f"net alpha {net_alpha:.4f} below cost gate")
                    if not checks.get("data_quality_ok", True):
                        failed.append("data quality severe")
                    if not checks.get("sl_distance_ok", True):
                        failed.append("SL distance outside 0.3-2.5% range")
                    reason = f"Signal {signal} blocked: {', '.join(failed)}" if failed else "All gates passed but unvalidated"

        fear_greed = self._fear_greed_index(feat)

        payload = {
            "asset": "BTCUSDT",
            "interval": interval,
            "signal": signal,
            "validated_signal": signal if validated else "HOLD",
            "validated": validated,
            "strength": strength,
            "confidence": round(confidence, 2),
            "alpha_score": int(np.clip(round(raw_alpha * 100), 0, 100)),
            "alpha_score_raw": round(raw_alpha, 4),
            "net_alpha_score": int(np.clip(round(float(net_alpha) * 100), 0, 100)),
            "net_alpha_score_raw": round(float(net_alpha), 4),
            "cost_pct": round(float(cost_pct), 5),
            "entry_price": entry_out,
            "entry_zone_low": round(float(entry_zone_low), 2) if entry_zone_low is not None else None,
            "entry_zone_high": round(float(entry_zone_high), 2) if entry_zone_high is not None else None,
            "stop_loss": stop_out,
            "take_profit": take_out,
            "tp1": round(float(tp1), 2) if tp1 is not None else None,
            "tp2": round(float(tp2), 2) if tp2 is not None else None,
            "tp3": round(float(tp3), 2) if tp3 is not None else None,
            "sl_pct": round(float(sl_pct), 2) if sl_pct is not None else None,
            "tp1_pct": round(float(tp1_pct), 2) if tp1_pct is not None else None,
            "tp2_pct": round(float(tp2_pct), 2) if tp2_pct is not None else None,
            "tp3_pct": round(float(tp3_pct), 2) if tp3_pct is not None else None,
            "risk_reward": round(float(risk_reward), 2) if risk_reward is not None else None,
            "position_size_pct": position_size_pct,
            "factor_scores": alpha.get("factor_scores", {}),
            "ic_weights": alpha.get("ic_weights", {}),
            "validation_checks": checks,
            "data_quality": dq.to_dict(),
            "reason": reason,
            "regime": regime,
            "session": session_name,
            "fear_greed": fear_greed,
            "funding_rate_pct": futures.get("funding_rate_pct", 0),
            "funding_sentiment": futures.get("funding_sentiment", "UNKNOWN"),
            "open_interest_btc": futures.get("open_interest_btc", 0),
            "mark_price": futures.get("mark_price", 0),
            "funding_confirms_signal": funding_confirms if signal in {"LONG", "SHORT"} else None,
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "algo": "quant_alpha_factor_model_v1",
        }
        # Track signal outcomes
        check_open_signals(entry)  # check existing open signals against current price
        record_signal(payload)  # record new signal if validated LONG/SHORT
        self._signal_cache.set(cache_key, payload, ttl_seconds=5)
        return payload

    def get_signal_markers(self, interval: str = "1d", limit: int = 1000) -> list[dict[str, Any]]:
        """Compute historical LONG/SHORT markers for chart overlays."""
        interval = interval if interval in INTERVAL_TO_MS else "1d"
        limit = int(max(120, min(limit, 4000)))
        cache_key = f"btc_markers_{interval}_{limit}"
        cached = self._marker_cache.get(cache_key)
        if cached is not None:
            return cached

        if interval == "1d":
            history = self.get_all_time_history(interval="1d")
            df = self._history_to_df(history)
        else:
            df = self.get_recent_frame(interval=interval, limit=limit)

        if df.empty or len(df) < self._alpha.ic_window + 20:
            self._marker_cache.set(cache_key, [], ttl_seconds=60)
            return []

        timeframe = "daily" if interval in {"1d", "1w", "1M"} else "intraday"
        clean_df, _ = self._anomaly.inspect_and_clean(df, "BTCUSDT", "crypto", timeframe)
        feat = self._engineer.compute_all_features(clean_df, timeframe=timeframe)
        if feat.empty or len(feat) < self._alpha.ic_window + 20:
            self._marker_cache.set(cache_key, [], ttl_seconds=60)
            return []

        markers = self._build_markers_from_features(feat)
        self._marker_cache.set(cache_key, markers, ttl_seconds=180)
        return markers

    @staticmethod
    def _last_feature_or_default(feat: pd.DataFrame, column: str, default: float) -> float:
        if column in feat.columns and not feat.empty:
            value = feat[column].iloc[-1]
            try:
                value_f = float(value)
                if np.isfinite(value_f):
                    return value_f
            except Exception:
                pass
        return float(default)

    @staticmethod
    def _session_name(dt_utc: datetime) -> str:
        hour = dt_utc.hour
        if 13 <= hour < 16:
            return "London/NY Overlap"
        if 8 <= hour < 13:
            return "London Session"
        if 16 <= hour < 21:
            return "New York Session"
        if 0 <= hour < 8:
            return "Asia Session"
        return "Dead Session"

    @staticmethod
    def _infer_regime(feat: pd.DataFrame) -> str:
        if feat.empty or "Close" not in feat.columns or len(feat) < 50:
            return "RANGE"

        close = float(feat["Close"].iloc[-1])
        ema_21 = float(feat["Close"].ewm(span=21).mean().iloc[-1])
        ema_50 = float(feat["Close"].ewm(span=50).mean().iloc[-1])

        # ATR-based volatility regime
        atr_col = feat.get("ATR_14")
        current_atr = None
        avg_atr = None
        if atr_col is not None and len(atr_col) > 50:
            current_atr = float(atr_col.iloc[-1])
            avg_atr = float(atr_col.tail(50).mean())
            if avg_atr > 0 and current_atr > 2.5 * avg_atr:
                return "VOLATILITY SPIKE"

        # Trend detection using EMA structure
        if close > ema_21 > ema_50:
            return "BULLISH TREND"
        if close < ema_21 < ema_50:
            return "BEARISH TREND"

        # Check for range (price between EMAs, low ATR)
        if atr_col is not None and current_atr is not None and avg_atr is not None:
            if avg_atr > 0 and current_atr < 0.7 * avg_atr:
                return "RANGE (SQUEEZE)"

        return "RANGE"

    @staticmethod
    def _fear_greed_index(feat: pd.DataFrame) -> int:
        if feat.empty:
            return 50
        if "RSI_14" in feat.columns:
            try:
                rsi = float(feat["RSI_14"].iloc[-1])
            except Exception:
                rsi = 50.0
        else:
            rsi = 50.0
        return int(np.clip(round(rsi), 0, 100))

    @staticmethod
    def _history_to_df(payload: dict[str, Any]) -> pd.DataFrame:
        rows = payload.get("data", [])
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        records = []
        for row in rows:
            try:
                ts = pd.to_datetime(int(row["time"]), unit="s", utc=True)
                records.append(
                    {
                        "time": ts,
                        "Open": float(row["open"]),
                        "High": float(row["high"]),
                        "Low": float(row["low"]),
                        "Close": float(row["close"]),
                        "Volume": float(row.get("volume", 0.0)),
                    }
                )
            except Exception:
                continue
        if not records:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(records).drop_duplicates(subset=["time"]).set_index("time").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def _build_markers_from_features(self, feat: pd.DataFrame) -> list[dict[str, Any]]:
        fwd_ret = feat["Returns"].shift(-1)
        ml_series = pd.Series(0.0, index=feat.index)
        f1 = self._alpha._factor1_momentum(feat, 0.5)
        f2 = self._alpha._factor2_mean_reversion(feat, 0.5)
        f3 = self._alpha._factor3_volume(feat)
        f4 = self._alpha._factor4_ml(ml_series)
        f5 = self._alpha._factor5_volatility_squeeze(feat, momentum_factor=f1)
        f8 = self._alpha._factor8_microstructure(feat)
        factors = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F8": f8}

        ic_map: dict[str, pd.Series] = {}
        for name, factor in factors.items():
            ic_series = _rolling_ic(factor, fwd_ret, self._alpha.ic_window)
            if len(ic_series) > 1:
                ic_series = ic_series.copy()
                ic_series.iloc[-1] = 0.0
            ic_map[name] = ic_series.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        denom = sum(abs(v) for v in ic_map.values())
        denom = denom.where(denom > 0.1, 0.1)
        alpha_series = sum(ic_map[k] * factors[k] for k in factors) / denom
        alpha_series = alpha_series.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        threshold = float(self._alpha.alpha_threshold)
        signal_series = pd.Series("HOLD", index=alpha_series.index, dtype="object")
        signal_series.loc[alpha_series > threshold] = "LONG"
        signal_series.loc[alpha_series < -threshold] = "SHORT"

        markers: list[dict[str, Any]] = []
        current_pos = 0
        for ts, sig in signal_series.items():
            sig_u = str(sig).upper()
            if sig_u == "HOLD":
                current_pos = 0
                continue
            conf = float(50.0 + 50.0 * np.tanh(float(alpha_series.loc[ts])))
            entry_px = float(feat["Close"].loc[ts]) if "Close" in feat.columns else None
            if sig_u == "LONG" and current_pos != 1:
                markers.append(
                    {
                        "time": int(pd.Timestamp(ts).timestamp()),
                        "position": "belowBar",
                        "color": "#00c853",
                        "shape": "arrowUp",
                        "text": "",
                        "signal": "LONG",
                        "confidence": round(conf, 2),
                        "entry": round(entry_px, 2) if entry_px is not None else None,
                    }
                )
                current_pos = 1
            elif sig_u == "SHORT" and current_pos != -1:
                markers.append(
                    {
                        "time": int(pd.Timestamp(ts).timestamp()),
                        "position": "aboveBar",
                        "color": "#ff1744",
                        "shape": "arrowDown",
                        "text": "",
                        "signal": "SHORT",
                        "confidence": round(conf, 2),
                        "entry": round(entry_px, 2) if entry_px is not None else None,
                    }
                )
                current_pos = -1

        return markers[-20:]

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
