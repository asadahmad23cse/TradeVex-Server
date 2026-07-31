"""
Bitcoin market data + signal service backed by Binance public APIs.

Provides:
  - all-time historical candles via paginated /api/v3/klines
  - recent candles for real-time quant signal generation
"""

from __future__ import annotations

import logging
import time as _time
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import urllib3

from src.alpha.factor_model import AlphaFactorModel, _rolling_ic
from src.api.data_quality import DataAnomalyDetector
from src.api.rate_limiter import TTLCache
from src.data import futures_data as futures_data_module
from src.data.etf_flow import get_etf_flow_provider
from src.data.fear_greed import get_fear_greed_provider
from src.data.futures_data import get_futures_sentiment
from src.dashboard.mtf_bias import MTFBiasFilter
from src.data.signal_history import check_open_signals, record_signal
from src.features.engineer import FeatureEngineer
from src.meta.calibration_freshness import CalibrationFreshnessGuard
from src.meta.config import load_meta_controls_config, module_enabled
from src.meta.data_confidence import DataConfidenceEngine
from src.meta.regime_explainer import RegimeBlockExplainer
from src.risk.cost_model import CostModel
from src.risk.kelly_warm_start import BTCKellyWarmStart

logger = logging.getLogger(__name__)
_last_signal_time: dict[str, float] = {}  # {"LONG": ts, "SHORT": ts, "__ANY__": ts}
DEFAULT_COOLDOWN_MINUTES = {
    "same_direction": 45,
    "any_signal": 10,
}
# Pure directional trend regimes.
BEARISH_REGIMES = {"BEARISH TREND", "HIGH_VOL_BEAR", "BEAR"}
BULLISH_REGIMES = {"BULLISH TREND", "HIGH_VOL_BULL", "BULL"}

# Contrarian reversal regimes:
#   CAPITULATION favors LONG (extreme negative funding in bearish structure)
#   DISTRIBUTION favors SHORT (extreme positive funding in bullish structure)
LONG_CONTRARIAN_REGIMES = {"CAPITULATION"}
SHORT_CONTRARIAN_REGIMES = {"DISTRIBUTION"}

LONG_FAVORED_REGIMES = BULLISH_REGIMES | LONG_CONTRARIAN_REGIMES
SHORT_FAVORED_REGIMES = BEARISH_REGIMES | SHORT_CONTRARIAN_REGIMES
BASE_CONFIDENCE_THRESHOLD = 0.65

# Multiplier applied to BASE_CONFIDENCE_THRESHOLD per regime.
# WITH-TREND signals get an additional 0.85× discount (regime confirms direction).
# COUNTER-TREND signals get an additional 1.10× penalty (trading against regime).
REGIME_CONFIDENCE_MULTIPLIER = {
    "BEARISH TREND": 1.25,
    "HIGH_VOL_BEAR": 1.20,
    "HIGH_VOL_BULL": 1.20,
    "BULLISH TREND": 1.00,
    "SIDEWAYS": 1.15,
    # Capitulation / Distribution: contrarian setups where sentiment extremes
    # support a reversal — allow with-trend signals at a LOWER bar than normal.
    "CAPITULATION": 0.85,   # extreme-fear LONG: lower bar for LONG entry
    "DISTRIBUTION": 0.85,   # extreme-greed SHORT: lower bar for SHORT entry
}
SHORT_SETUP_SL_PCT = 0.0035
SHORT_SETUP_TP1_PCT = 0.0070
SHORT_SETUP_TP2_PCT = 0.0105
SHORT_SETUP_TP3_PCT = 0.0140
OI_EXTREME_NEG_DELTA_PCT = -2.0

BINANCE_REST = "https://api.binance.com"
BTC_SYMBOL = "BTCUSDT"
BINANCE_BTC_EARLIEST_MS = 1502942400000  # 2017-08-17T04:00:00Z approx listing start
BINANCE_HTTP_TIMEOUT_SECONDS = 6

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
        self._etf_flow = get_etf_flow_provider()
        self._fear_greed = get_fear_greed_provider()
        self._mtf_bias = MTFBiasFilter(self)
        self._kelly = BTCKellyWarmStart()
        self._config = config
        self._cooldown_minutes = {
            "same_direction": max(
                0.0,
                float(signal_cfg.get("cooldown_same_direction_min", DEFAULT_COOLDOWN_MINUTES["same_direction"])),
            ),
            "any_signal": max(
                0.0,
                float(signal_cfg.get("cooldown_any_signal_min", DEFAULT_COOLDOWN_MINUTES["any_signal"])),
            ),
        }

    def _attach_meta_controls(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach optional meta-control metadata without mutating raw signal fields."""
        try:
            meta_cfg = load_meta_controls_config(self._config)
            if not bool(meta_cfg.get("enabled", False)):
                return payload

            shadow_mode = bool(meta_cfg.get("shadow_mode", True))
            enforce_execution_gates = bool(meta_cfg.get("enforce_execution_gates", False))
            meta_out = payload.get("meta_controls") if isinstance(payload.get("meta_controls"), dict) else {}
            meta_out = dict(meta_out)

            if module_enabled(meta_cfg, "data_confidence"):
                feed_status = DataConfidenceEngine.status_from_signal_payload(payload)
                dc_result = DataConfidenceEngine(
                    meta_cfg.get("data_confidence", {}) or {},
                    shadow_mode=shadow_mode,
                    enforce_execution_gates=enforce_execution_gates,
                ).evaluate(feed_status)
                dc_payload = dc_result.to_dict()
                meta_out["data_confidence"] = dc_payload
                payload["data_confidence_score"] = dc_payload["data_confidence_score"]
                payload["data_confidence_degraded"] = dc_payload["degraded"]
                payload["data_confidence_blocked"] = dc_payload["blocked"]

            if module_enabled(meta_cfg, "calibration_freshness"):
                cal_result = CalibrationFreshnessGuard(
                    meta_cfg.get("calibration_freshness", {}) or {},
                    shadow_mode=shadow_mode,
                    enforce_execution_gates=enforce_execution_gates,
                ).evaluate(
                    payload.get("last_calibration_timestamp"),
                    checkpoint_path=(meta_cfg.get("calibration_freshness") or {}).get(
                        "checkpoint_path",
                        CalibrationFreshnessGuard.DEFAULT_CHECKPOINT_PATH,
                    ),
                )
                cal_payload = cal_result.to_dict()
                meta_out["calibration_freshness"] = cal_payload
                payload["calibration_status"] = cal_payload["calibration_status"]
                payload["calibration_warning"] = cal_payload["calibration_warning"]

            if module_enabled(meta_cfg, "regime_explainer"):
                detail = RegimeBlockExplainer(meta_cfg.get("regime_explainer", {}) or {}).explain(payload)
                if detail is not None:
                    payload["block_reason_detail"] = detail
                    meta_out["regime_block"] = detail

            if meta_out:
                payload["meta_controls"] = meta_out
        except Exception as exc:
            logger.debug("Meta-control attachment skipped: %s", exc)
        return payload

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
        if df.empty:
            df = self._fetch_yfinance_recent_frame(interval=interval, limit=limit)
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
            payload = self._attach_meta_controls(payload)
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
            payload = self._attach_meta_controls(payload)
            self._signal_cache.set(cache_key, payload, ttl_seconds=5)
            return payload

        price_change_1h = 0.0
        price_change_24h = 0.0
        bars_1h = self._bars_for_hours(interval, 1)
        bars_24h = self._bars_for_hours(interval, 24)
        if "Close" in feat.columns and len(feat) > max(bars_1h, 2):
            current = float(feat["Close"].iloc[-1])
            price_1h_ago = float(feat["Close"].iloc[-(bars_1h + 1)]) if len(feat) > bars_1h else current
            price_change_1h = round(((current - price_1h_ago) / price_1h_ago) * 100, 2)
            if len(feat) > bars_24h:
                price_24h_ago = float(feat["Close"].iloc[-(bars_24h + 1)])
                price_change_24h = round(((current - price_24h_ago) / price_24h_ago) * 100, 2)

        futures = {
            "funding_rate_pct": 0.0,
            "funding_rate_z": 0.0,
            "funding_sentiment": "NEUTRAL",
            "open_interest_btc": 0.0,
            "oi_delta_pct_1h": 0.0,
            "oi_delta_pct_4h": 0.0,
            "liquidation_event": False,
            "liquidation_bias": "NONE",
            "liquidation_score": 0.0,
            "mark_price": 0.0,
        }
        try:
            futures = futures_data_module.get_futures_sentiment()
            logger.info(
                "Futures data: funding=%.4f%%, OI=%.1f",
                futures.get("funding_rate_pct", 0.0),
                futures.get("open_interest_btc", 0.0),
            )
        except Exception as exc:
            logger.warning("Futures data unavailable (may be geo-blocked): %s", exc)

        funding_rate = float(futures.get("funding_rate_pct", 0.0))
        funding_rate_z = float(futures.get("funding_rate_z", 0.0))
        funding_sentiment = str(futures.get("funding_sentiment", "NEUTRAL"))
        oi_delta_1h = float(futures.get("oi_delta_pct_1h", 0.0))
        oi_delta_4h = float(futures.get("oi_delta_pct_4h", 0.0))
        liquidation_event = bool(futures.get("liquidation_event", False))
        liquidation_bias = str(futures.get("liquidation_bias", "NONE"))
        etf_flow = self._etf_flow_snapshot()
        etf_flow_z = float(etf_flow.get("flow_z", 0.0))

        alpha = self._alpha.compute(
            feat,
            ml_score=0.0,
            hurst=0.5,
            asset_class="crypto",
            funding_rate_z=funding_rate_z,
            oi_delta_1h=oi_delta_1h,
            price_change_1h=price_change_1h,
            etf_flow_z=etf_flow_z,
        )
        raw_alpha = float(alpha.get("alpha_score", 0.0))
        confidence = float(alpha.get("confidence", 50.0))
        model_signal = str(alpha.get("signal", "HOLD")).upper()
        signal = "LONG" if model_signal == "BUY" else "SHORT" if model_signal == "SELL" else "HOLD"
        requested_signal = signal if signal in {"LONG", "SHORT"} else None
        reason = ""
        blocked_by: str | None = None
        regime = self._infer_regime(feat, funding_rate_z=funding_rate_z, funding_rate_pct=funding_rate)

        # Direction-aware regime threshold:
        #   WITH-TREND  (SHORT in BEARISH / LONG in BULLISH / LONG in CAPITULATION
        #                / SHORT in DISTRIBUTION) → base threshold, no penalty.
        #   COUNTER-TREND (LONG in BEARISH / SHORT in BULLISH) → full multiplier;
        #                need high conviction to trade against the regime.
        #   NEUTRAL regimes (SIDEWAYS, RANGE) → moderate multiplier as before.
        _base_mult = REGIME_CONFIDENCE_MULTIPLIER.get(regime, 1.0)
        _with_trend = (
            (signal == "LONG" and regime in LONG_FAVORED_REGIMES)
            or (signal == "SHORT" and regime in SHORT_FAVORED_REGIMES)
        )
        _counter_trend = (
            (signal == "LONG" and regime in SHORT_FAVORED_REGIMES)
            or (signal == "SHORT" and regime in LONG_FAVORED_REGIMES)
        )
        if _with_trend:
            # Regime already confirms direction → lower the bar by 15%
            adjusted_threshold = BASE_CONFIDENCE_THRESHOLD * _base_mult * 0.85 * 100.0
        elif _counter_trend:
            # Extra 10% penalty on top of regime multiplier for counter-trend signals
            adjusted_threshold = BASE_CONFIDENCE_THRESHOLD * _base_mult * 1.10 * 100.0
        else:
            adjusted_threshold = BASE_CONFIDENCE_THRESHOLD * _base_mult * 100.0
        computed_score = round(float(np.clip((abs(raw_alpha) * 100.0 * 0.55) + (confidence * 0.45), 0, 100)), 2)

        entry = float(feat["Close"].iloc[-1])
        close = float(feat["Close"].iloc[-1])
        ema_21 = float(feat["Close"].ewm(span=21).mean().iloc[-1]) if len(feat) > 21 else close
        ema_50 = float(feat["Close"].ewm(span=50).mean().iloc[-1]) if len(feat) > 50 else ema_21
        trend_bullish = close > ema_21 > ema_50
        trend_bearish = close < ema_21 < ema_50
        vol_ratio = self._last_feature_or_default(feat, "Volume_Ratio", 1.0)
        obv_slope = self._last_feature_or_default(feat, "OBV_Slope", 0.0)
        cmf = self._last_feature_or_default(feat, "CMF", 0.0)
        rsi = self._last_feature_or_default(feat, "RSI_14", 50.0)

        smart_money_state = "FALLING" if obv_slope < 0 else "RISING"
        short_setup_candidate = (
            regime in {"BEARISH TREND", "HIGH_VOL_BEAR"}
            and trend_bearish
            and confidence >= adjusted_threshold
            and smart_money_state == "FALLING"
            and rsi < 50
        )
        short_setup_active = False
        if short_setup_candidate:
            signal = "SHORT"
            requested_signal = "SHORT"
            short_setup_active = True
            reason = f"Bearish structure confirmed: price < EMA21 < EMA50, Smart Money falling, regime {regime}"
            blocked_by = None

        # 1) Regime gate
        if signal in {"LONG", "SHORT"}:
            if signal == "LONG" and regime in SHORT_FAVORED_REGIMES:
                signal = "WAIT"
                reason = f"regime_conflict: LONG blocked in {regime} regime"
                blocked_by = "regime_gate"
            elif signal == "SHORT" and regime in LONG_FAVORED_REGIMES:
                signal = "WAIT"
                reason = f"regime_conflict: SHORT blocked in {regime} regime"
                blocked_by = "regime_gate"

        # 2) Regime confidence gate
        if signal in {"LONG", "SHORT"} and confidence < adjusted_threshold:
            signal = "WAIT"
            reason = f"low_confidence_for_regime: {confidence:.1f}% < {adjusted_threshold:.1f}% ({regime})"
            blocked_by = blocked_by or "regime_confidence_gate"

        mtf_result = {
            "alignment_ok": True,
            "bias_4h": "NEUTRAL",
            "bias_1d": "NEUTRAL",
            "block_reason": None,
            "alignment_score": 1.0,
        }
        if signal in {"LONG", "SHORT"}:
            mtf_result = self._mtf_bias.check_alignment(signal, entry_timeframe=interval, confidence=confidence)
            if not bool(mtf_result.get("alignment_ok", True)):
                signal = "WAIT"
                reason = str(mtf_result.get("block_reason") or "MTF alignment blocked")
                blocked_by = blocked_by or "mtf_bias"
            else:
                confidence = float(np.clip(confidence * float(mtf_result.get("alignment_score", 1.0)), 0.0, 100.0))

        now_utc = datetime.now(timezone.utc)
        session_name = self._session_name(now_utc)
        hour = now_utc.hour

        fear_greed_snapshot = self._fear_greed_snapshot(feat)
        fear_greed = int(fear_greed_snapshot.get("value", self._fear_greed_index(feat)))
        fear_greed_z = float(fear_greed_snapshot.get("z_score", 0.0))
        etf_source = str(etf_flow.get("source", "")).lower()
        fear_source = str(fear_greed_snapshot.get("source", "")).lower()
        halving = self._halving_phase_context(now_utc.date())
        halving_multiplier = float(halving.get("confidence_multiplier", 1.0))
        mtf = {
            "alignment_ok": bool(mtf_result.get("alignment_ok", True)),
            "biases": {
                "4h": str(mtf_result.get("bias_4h", "NEUTRAL")),
                "1d": str(mtf_result.get("bias_1d", "NEUTRAL")),
            },
            "alignment_score": float(mtf_result.get("alignment_score", 1.0)),
            "block_reason": mtf_result.get("block_reason"),
            "entry_timeframe": interval,
        }

        etf_multiplier = 1.0
        if signal in {"LONG", "SHORT"}:
            try:
                etf_multiplier = float(self._etf_flow.get_conviction_boost(signal, funding_rate_z))
            except Exception:
                etf_multiplier = 1.0
            confidence = float(np.clip(confidence * etf_multiplier * halving_multiplier, 0.0, 100.0))

        # NOTE: F13/F14/F21 are now formal IC-weighted factors in alpha_score.
        # These gates remain as hard stops for extreme values only.
        funding_confirms = False
        if signal == "LONG" and (funding_rate < -0.03 or funding_rate_z < -0.5):
            funding_confirms = True
        if signal == "SHORT" and (funding_rate > 0.03 or funding_rate_z > 0.5):
            funding_confirms = True

        etf_ok = True
        fear_greed_ok = True
        liquidation_ok = True
        mtf_ok = bool(mtf.get("alignment_ok", True))
        oi_ok = True

        # 3) ETF flow filter
        etf_filter_enabled = etf_source not in {"fallback", "stale_cache", "", "unknown"}
        if signal == "LONG" and etf_filter_enabled and etf_flow_z < -1.0:
            signal = "WAIT"
            reason = f"HOLD: ETF outflow regime ({etf_flow_z:+.2f}z) is hostile to longs"
            blocked_by = blocked_by or "etf_flow_filter"
            etf_ok = False
        elif signal == "SHORT" and etf_filter_enabled and etf_flow_z > 1.0:
            signal = "WAIT"
            reason = f"HOLD: ETF inflow regime ({etf_flow_z:+.2f}z) is hostile to shorts"
            blocked_by = blocked_by or "etf_flow_filter"
            etf_ok = False

        # 4) Fear & Greed filter
        fear_filter_enabled = fear_source in {"alternative_me", "live"}
        fear_threshold = 2.5
        if signal == "LONG" and fear_filter_enabled and fear_greed_z > fear_threshold:
            signal = "WAIT"
            reason = f"HOLD: LONG blocked by stretched greed ({fear_greed_z:+.2f}z)"
            blocked_by = blocked_by or "fear_greed_filter"
            fear_greed_ok = False
        elif signal == "SHORT" and fear_filter_enabled and fear_greed_z < -fear_threshold:
            signal = "WAIT"
            reason = f"HOLD: SHORT blocked by washed-out fear ({fear_greed_z:+.2f}z)"
            blocked_by = blocked_by or "fear_greed_filter"
            fear_greed_ok = False

        # 5) Liquidation filter
        if signal == "LONG" and liquidation_bias == "SHORT_SQUEEZE":
            signal = "WAIT"
            reason = "HOLD: LONG blocked because a short squeeze likely already fired"
            blocked_by = blocked_by or "liquidation_filter"
            liquidation_ok = False
        elif signal == "SHORT" and liquidation_bias == "LONG_FLUSH":
            signal = "WAIT"
            reason = "HOLD: SHORT blocked because a long flush likely already fired"
            blocked_by = blocked_by or "liquidation_filter"
            liquidation_ok = False

        # 6) Multi-timeframe hierarchy filter
        if signal in {"LONG", "SHORT"} and not mtf_ok:
            signal = "WAIT"
            reason = "HOLD: Multi-timeframe bias misaligned (1D/4H hierarchy)"
            blocked_by = blocked_by or "mtf_filter"

        # 7) Session/timing filters
        is_dead_session = 21 <= hour or hour < 1
        is_asia_low_vol = 1 <= hour < 7
        if signal in {"LONG", "SHORT"} and is_dead_session:
            signal = "HOLD"
            reason = f"HOLD: Dead session ({hour:02d}:00 UTC) - no signals generated"
        if signal in {"LONG", "SHORT"} and is_asia_low_vol and confidence < 75:
            signal = "HOLD"
            reason = f"HOLD: Asia session low confidence ({confidence:.0f}% < 75% required)"

        if signal in {"LONG", "SHORT"} and len(feat) > 50:
            if signal == "LONG" and trend_bearish and regime not in LONG_CONTRARIAN_REGIMES:
                signal = "HOLD"
                reason = "HOLD: LONG signal rejected - price below EMA21 < EMA50 (bearish structure)"
            elif signal == "SHORT" and trend_bullish and regime not in SHORT_CONTRARIAN_REGIMES:
                signal = "HOLD"
                reason = "HOLD: SHORT signal rejected - price above EMA21 > EMA50 (bullish structure)"

        if signal == "LONG" and funding_sentiment == "OVERLEVERAGED_LONG" and funding_rate >= 0.03:
            signal = "HOLD"
            reason = f"HOLD: LONG blocked - funding rate {funding_rate:+.4f}% (overleveraged longs, squeeze risk)"
        if signal == "SHORT" and funding_sentiment == "OVERLEVERAGED_SHORT" and funding_rate <= -0.03:
            signal = "HOLD"
            reason = f"HOLD: SHORT blocked - funding rate {funding_rate:+.4f}% (overleveraged shorts, squeeze risk)"

        if signal in {"LONG", "SHORT"}:
            oi_ok = oi_delta_1h > OI_EXTREME_NEG_DELTA_PCT

        # 8) Cooldown (must run last)
        if signal in {"LONG", "SHORT"}:
            now_ts = _time.time()
            signal_for_cooldown = signal
            elapsed_same = now_ts - _last_signal_time.get(signal, 0.0)
            elapsed_any = now_ts - _last_signal_time.get("__ANY__", 0.0)

            same_direction_cd = float(self._cooldown_minutes.get("same_direction", DEFAULT_COOLDOWN_MINUTES["same_direction"]))
            any_signal_cd = float(self._cooldown_minutes.get("any_signal", DEFAULT_COOLDOWN_MINUTES["any_signal"]))

            if _last_signal_time.get(signal) and elapsed_same < same_direction_cd * 60:
                signal = "WAIT"
                reason = f"cooldown: same direction {signal_for_cooldown} fired {elapsed_same / 60.0:.1f} min ago"
                blocked_by = blocked_by or "cooldown_same_direction"
            elif _last_signal_time.get("__ANY__") and elapsed_any < any_signal_cd * 60:
                signal = "WAIT"
                reason = f"cooldown: signal fired {elapsed_any / 60.0:.1f} min ago"
                blocked_by = blocked_by or "cooldown_any_signal"

        atr = self._last_feature_or_default(feat, "ATR_14", entry * 0.02)
        daily_vol = self._last_feature_or_default(feat, "Volatility_20", 0.5)
        volume_ratio = vol_ratio
        # BTC FIX: pass the actual representative order size to the Almgren-Chriss
        # cost model instead of 1.0 (100%).  Market impact scales with √(order_size),
        # so using 1.0 inflated cost ~10× vs the real 1-2% Kelly size and caused
        # every signal to fail the cost gate.  Use 0.02 (2%) as a conservative
        # upper-bound estimate that matches cold-start Kelly sizing.
        _btc_position_size_for_cost = 0.02
        net_alpha, cost_pct, viable = self._cost.net_alpha(
            alpha_score=raw_alpha,
            asset_class="crypto",
            position_size_pct=_btc_position_size_for_cost,
            daily_vol=daily_vol,
            volume_ratio=max(volume_ratio, 0.1),
            regime=regime,
            low_liquidity=volume_ratio < 0.8,
        )

        strength = str(alpha.get("strength", "WEAK"))
        sl_mult = 2.5 if strength == "STRONG" else 3.0 if strength == "MODERATE" else 3.5

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
            if short_setup_active:
                stop = entry * (1.0 + SHORT_SETUP_SL_PCT)
                tp1 = entry * (1.0 - SHORT_SETUP_TP1_PCT)
                tp2 = entry * (1.0 - SHORT_SETUP_TP2_PCT)
                tp3 = entry * (1.0 - SHORT_SETUP_TP3_PCT)
                entry_zone_low = entry * 0.999
                entry_zone_high = entry * 1.001
                take = tp1
                sl_dist = abs(entry - stop)
                sl_pct = (sl_dist / entry) * 100 if entry > 0 else None
                tp1_pct = ((tp1 - entry) / entry) * 100 if entry > 0 else None
                tp2_pct = ((tp2 - entry) / entry) * 100 if entry > 0 else None
                tp3_pct = ((tp3 - entry) / entry) * 100 if entry > 0 else None
                risk_reward = 2.0
            else:
                stop = entry + sl_mult * atr

        if signal == "LONG" and stop is not None:
            min_distance = entry * 0.003
            if (entry - stop) < min_distance:
                stop = entry - min_distance

        if signal == "SHORT" and stop is not None and not short_setup_active:
            min_distance = entry * 0.003
            if (stop - entry) < min_distance:
                stop = entry + min_distance

        if signal in {"LONG", "SHORT"} and stop is not None and not short_setup_active:
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

        validated = False

        if signal in {"LONG", "SHORT"} and regime == "VOLATILITY SPIKE":
            signal = "HOLD"
            reason = "HOLD: Volatility spike detected (ATR > 2.5x normal) - no signals"
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

        sl_debug_stop = float(stop) if stop is not None else float("nan")
        sl_debug_dist = (abs(entry - sl_debug_stop) / entry) * 100 if entry > 0 and np.isfinite(sl_debug_stop) else float("nan")
        logger.info("SL DEBUG: entry=%.2f stop=%.2f dist=%.4f%% signal=%s", entry, sl_debug_stop, sl_debug_dist, signal)

        checks = {
            "data_quality_ok": not dq.severe,
            "confidence_ok": confidence >= adjusted_threshold,
            "cost_ok": bool(viable),
            "mtf_ok": mtf_ok,
            "etf_ok": etf_ok,
            "liquidation_ok": liquidation_ok,
            "fear_greed_ok": fear_greed_ok,
            "oi_ok": oi_ok,
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
        sizing_signal = signal if signal in {"LONG", "SHORT"} else (requested_signal if requested_signal in {"LONG", "SHORT"} else "LONG")
        kelly_result = self._kelly.compute_btc_position(
            signal=sizing_signal,
            confidence=confidence,
            regime=regime,
        )
        if validated:
            position_size_pct = float(kelly_result["position_size_pct"])
            _last_signal_time[signal] = _time.time()
            _last_signal_time["__ANY__"] = _last_signal_time[signal]

        if signal not in {"LONG", "SHORT"}:
            if not reason:
                failed = []
                if not checks.get("confidence_ok", True):
                    failed.append(f"confidence {confidence:.1f}% < regime threshold {adjusted_threshold:.1f}%")
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
                    reason = (
                        f"{regime} | {session_name} | Conf:{confidence:.0f}% | {strength} "
                        f"| F13(funding:{funding_rate_z:+.1f}z) "
                        f"| F14(OI:{oi_delta_1h:+.1f}%) "
                        f"| F21(ETF:{etf_flow_z:+.1f}z)"
                    )
                else:
                    failed = []
                    if not checks.get("confidence_ok", True):
                        failed.append(f"confidence {confidence:.1f}% < regime threshold {adjusted_threshold:.1f}%")
                    if not checks.get("cost_ok", True):
                        failed.append(f"net alpha {net_alpha:.4f} below cost gate")
                    if not checks.get("data_quality_ok", True):
                        failed.append("data quality severe")
                    if not checks.get("mtf_ok", True):
                        failed.append("multi-timeframe misalignment")
                    if not checks.get("etf_ok", True):
                        failed.append("ETF flow conflict")
                    if not checks.get("fear_greed_ok", True):
                        failed.append("fear/greed conflict")
                    if not checks.get("liquidation_ok", True):
                        failed.append("liquidation conflict")
                    if not checks.get("oi_ok", True):
                        failed.append("open interest not supportive")
                    if not checks.get("sl_distance_ok", True):
                        failed.append("SL distance outside 0.3-2.5% range")
                    reason = f"Signal {signal} blocked: {', '.join(failed)}" if failed else "All gates passed but unvalidated"

        atr_pct = round(atr / entry * 100, 3) if entry > 0 else 0

        market_context = {
            "futures": futures,
            "etf_flow": etf_flow,
            "fear_greed_snapshot": fear_greed_snapshot,
            "multi_timeframe": mtf,
            "halving": halving,
            "price_change_1h": price_change_1h,
            "price_change_24h": price_change_24h,
            "atr_pct": atr_pct,
        }

        payload = {
            "asset": "BTCUSDT",
            "asset_class": "crypto",
            "interval": interval,
            "signal": signal,
            "validated_signal": signal if validated else "HOLD",
            "validated": validated,
            "validated_label": signal if validated else "NO TRADE",
            "direction": "long" if signal == "LONG" else "short" if signal == "SHORT" else "flat",
            "requested_signal": requested_signal,
            "blocked_by": blocked_by,
            "strength": strength,
            "signal_strength": computed_score,
            "confidence": round(confidence, 2),
            "ai_confidence": round(confidence, 2),
            "adjusted_confidence_threshold": round(adjusted_threshold, 2),
            "alpha_score": int(np.clip(round(abs(raw_alpha) * 100), 0, 100)),
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
            "position_sizing": {
                "position_size_pct": kelly_result["position_size_pct"],
                "position_size_usd": kelly_result["position_size_usd"],
                "method": kelly_result["method"],
                "raw_kelly": kelly_result["raw_kelly"],
                "quarter_kelly": kelly_result["quarter_kelly"],
                "bucket_key": kelly_result["bucket_key"],
                "trades_in_bucket": kelly_result["trades_in_bucket"],
                "smoothing_weight": kelly_result["smoothing_weight"],
                "p": kelly_result["p"],
                "b": kelly_result["b"],
            },
            "entry": entry_out,
            "sl": stop_out,
            "rr": round(float(risk_reward), 2) if risk_reward is not None else None,
            "factor_scores": alpha.get("factor_scores", {}),
            "ic_weights": alpha.get("ic_weights", {}),
            "validation_checks": checks,
            "data_quality": dq.to_dict(),
            "reason": reason,
            "regime": regime,
            "session": session_name,
            "fear_greed": fear_greed,
            "fear_greed_z": round(fear_greed_z, 3),
            "bias_4h": str(mtf_result.get("bias_4h", "NEUTRAL")),
            "bias_1d": str(mtf_result.get("bias_1d", "NEUTRAL")),
            "mtf_bias": {
                "bias_4h": str(mtf_result.get("bias_4h", "NEUTRAL")),
                "bias_1d": str(mtf_result.get("bias_1d", "NEUTRAL")),
                "alignment_score": round(float(mtf_result.get("alignment_score", 1.0)), 3),
                "alignment_ok": bool(mtf_result.get("alignment_ok", True)),
            },
            "order_flow": {
                "volume_ratio": round(vol_ratio, 2),
                "volume_trend": "HIGH" if vol_ratio > 1.5 else "LOW" if vol_ratio < 0.7 else "NORMAL",
                "obv_slope": round(obv_slope, 4),
                "obv_trend": "RISING" if obv_slope > 0 else "FALLING",
                "cmf": round(cmf, 4),
                "cmf_signal": "ACCUMULATION" if cmf > 0.05 else "DISTRIBUTION" if cmf < -0.05 else "NEUTRAL",
                "rsi": round(rsi, 1),
                "rsi_zone": "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL",
            },
            "market_overview": {
                "price_change_1h": price_change_1h,
                "price_change_24h": price_change_24h,
                "regime": regime,
                "bias": (
                    "BEARISH"
                    if regime in {"BEARISH TREND", "HIGH_VOL_BEAR"}
                    else "BULLISH"
                    if regime in {"BULLISH TREND", "HIGH_VOL_BULL"}
                    else "NEUTRAL"
                ),
                "session": session_name,
                "fear_greed": fear_greed,
                "fear_greed_z": round(fear_greed_z, 3),
                "fear_greed_label": str(fear_greed_snapshot.get("label", "Neutral")),
                "atr_pct": atr_pct,
                "volatility": "HIGH" if atr_pct > 0.8 else "LOW" if atr_pct < 0.3 else "NORMAL",
                "halving_phase": halving.get("phase"),
                "halving_bias": halving.get("bias"),
            },
            "funding_rate_pct": futures.get("funding_rate_pct", 0.0),
            "funding_rate_z": round(funding_rate_z, 3),
            "funding_sentiment": futures.get("funding_sentiment", "UNKNOWN"),
            "open_interest_btc": futures.get("open_interest_btc", 0.0),
            "oi_delta_pct_1h": round(oi_delta_1h, 3),
            "oi_delta_pct_4h": round(oi_delta_4h, 3),
            "liquidation_event": liquidation_event,
            "liquidation_bias": liquidation_bias,
            "liquidation_score": round(float(futures.get("liquidation_score", 0.0)), 3),
            "mark_price": futures.get("mark_price", 0.0),
            "funding_confirms_signal": funding_confirms if signal in {"LONG", "SHORT"} else None,
            "etf_flow": etf_flow,
            "market_context": market_context,
            "on_chain_sqs": self._compute_on_chain_signal_quality(
                signal=signal if signal in {"LONG", "SHORT"} else (requested_signal or "LONG"),
                regime=regime,
                funding_rate_z=funding_rate_z,
                oi_delta_1h=oi_delta_1h,
                fear_greed=fear_greed,
                fear_greed_z=fear_greed_z,
                etf_flow_z=etf_flow_z,
                mtf_alignment_score=float(mtf_result.get("alignment_score", 0.5)),
                rsi=rsi,
                cmf=cmf,
            ),
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "algo": "quant_alpha_factor_model_v1",
        }
        payload = self._attach_meta_controls(payload)
        # Track signal outcomes
        check_open_signals(
            entry,
            market_signal=signal,
            validated_signal=(signal if validated else "HOLD"),
            alpha_score=raw_alpha,
        )
        # [ADDITIVE] Only record signal if it's actionable (BLOCKED or first OPEN)
        # record_signal() itself has duplicate prevention, but skip HOLD signals entirely
        try:
            _should_record = (
                str(payload.get("signal", "")).upper() in ("LONG", "SHORT")
                or str(payload.get("blocked_by", "")).strip()
                or str(payload.get("result", "")).upper() == "BLOCKED"
            )
            if _should_record:
                record_signal(payload)
        except Exception:
            pass
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
    def _bars_for_hours(interval: str, hours: int) -> int:
        step_ms = INTERVAL_TO_MS.get(interval, INTERVAL_TO_MS["5m"])
        return max(int(round((hours * 3_600_000) / step_ms)), 1)

    @staticmethod
    def _compute_on_chain_signal_quality(
        signal: str,
        regime: str,
        funding_rate_z: float,
        oi_delta_1h: float,
        fear_greed: int,
        fear_greed_z: float,
        etf_flow_z: float,
        mtf_alignment_score: float,
        rsi: float,
        cmf: float,
    ) -> dict[str, Any]:
        """
        BTC-specific Signal Quality Score (SQS) — 0 to 100.

        Four components:
          1. On-chain / derivatives alignment  (35 pts)
             Funding rate, OI delta, and ETF flow all supporting the signal.
          2. Sentiment alignment               (25 pts)
             Fear & Greed reading consistent with the signal direction.
          3. Regime + MTF alignment            (25 pts)
             EMA regime structure + multi-timeframe bias score.
          4. Technical momentum                (15 pts)
             RSI and CMF pointing in signal direction.
        """
        s = signal.upper()
        score = 0.0
        detail: dict[str, float] = {}

        # 1 — On-chain / derivatives (35 pts)
        # Funding: negative z supports LONG (shorts paying), positive supports SHORT
        fund_pts = 0.0
        if s == "LONG":
            fund_pts = float(np.clip(((-funding_rate_z + 1.0) / 2.5) * 12, 0, 12))
        elif s == "SHORT":
            fund_pts = float(np.clip(((funding_rate_z + 1.0) / 2.5) * 12, 0, 12))

        # OI delta: rising OI + rising price confirms LONG; falling OI confirms SHORT
        oi_pts = 0.0
        if s == "LONG" and oi_delta_1h > 0:
            oi_pts = float(np.clip((oi_delta_1h / 3.0) * 11, 0, 11))
        elif s == "SHORT" and oi_delta_1h < 0:
            oi_pts = float(np.clip((abs(oi_delta_1h) / 3.0) * 11, 0, 11))

        # ETF flow: inflows support LONG, outflows support SHORT
        etf_pts = 0.0
        if s == "LONG" and etf_flow_z > 0:
            etf_pts = float(np.clip((etf_flow_z / 2.0) * 12, 0, 12))
        elif s == "SHORT" and etf_flow_z < 0:
            etf_pts = float(np.clip((abs(etf_flow_z) / 2.0) * 12, 0, 12))

        onchain_score = fund_pts + oi_pts + etf_pts
        detail["onchain_pts"] = round(onchain_score, 1)

        # 2 — Sentiment (25 pts)
        sent_pts = 0.0
        if s == "LONG":
            # Extreme fear (< 25) is a LONG setup; greed (> 75) is hostile
            if fear_greed < 25:
                sent_pts = float(np.clip(((25 - fear_greed) / 25) * 25, 0, 25))
            elif fear_greed < 50:
                sent_pts = float(np.clip(((50 - fear_greed) / 50) * 12, 0, 12))
        elif s == "SHORT":
            # Extreme greed (> 75) is a SHORT setup; fear (< 25) is hostile
            if fear_greed > 75:
                sent_pts = float(np.clip(((fear_greed - 75) / 25) * 25, 0, 25))
            elif fear_greed > 50:
                sent_pts = float(np.clip(((fear_greed - 50) / 50) * 12, 0, 12))
        detail["sentiment_pts"] = round(sent_pts, 1)

        # 3 — Regime + MTF (25 pts)
        regime_pts = 0.0
        with_trend = (
            (s == "SHORT" and regime in {"BEARISH TREND", "HIGH_VOL_BEAR", "BEAR"})
            or (s == "LONG" and regime in {"BULLISH TREND", "HIGH_VOL_BULL", "BULL"})
            or (s == "LONG" and regime == "CAPITULATION")
            or (s == "SHORT" and regime == "DISTRIBUTION")
        )
        if with_trend:
            regime_pts = 15.0
        elif regime in {"SIDEWAYS", "RANGE", "RANGE (SQUEEZE)"}:
            regime_pts = 8.0
        mtf_pts = float(np.clip(mtf_alignment_score * 10.0, 0, 10))
        detail["regime_pts"] = round(regime_pts + mtf_pts, 1)

        # 4 — Technical momentum (15 pts)
        rsi_pts = 0.0
        if s == "LONG" and rsi < 50:
            rsi_pts = float(np.clip(((50 - rsi) / 50) * 8, 0, 8))
        elif s == "SHORT" and rsi > 50:
            rsi_pts = float(np.clip(((rsi - 50) / 50) * 8, 0, 8))

        cmf_pts = 0.0
        if s == "LONG" and cmf > 0:
            cmf_pts = float(np.clip((cmf / 0.15) * 7, 0, 7))
        elif s == "SHORT" and cmf < 0:
            cmf_pts = float(np.clip((abs(cmf) / 0.15) * 7, 0, 7))
        detail["momentum_pts"] = round(rsi_pts + cmf_pts, 1)

        total = onchain_score + sent_pts + regime_pts + mtf_pts + rsi_pts + cmf_pts
        return {
            "score": int(np.clip(round(total), 0, 100)),
            "detail": detail,
            "grade": (
                "A" if total >= 75 else
                "B" if total >= 55 else
                "C" if total >= 35 else
                "D"
            ),
        }

    def _fear_greed_snapshot(self, feat: pd.DataFrame) -> dict[str, Any]:
        try:
            snap = self._fear_greed.get_snapshot()
            if isinstance(snap, dict) and snap:
                return snap
        except Exception as exc:
            logger.debug("Fear & Greed fetch failed: %s", exc)
        return {
            "value": self._fear_greed_index(feat),
            "label": "Fallback",
            "z_score": 0.0,
            "source": "rsi_fallback",
            "as_of": None,
        }

    def _etf_flow_snapshot(self) -> dict[str, Any]:
        try:
            latest = self._etf_flow.get_latest_row()
            return latest if isinstance(latest, dict) else {}
        except Exception as exc:
            logger.debug("ETF flow fetch failed: %s", exc)
            return {"date": None, "total_usd_m": 0.0, "flow_z": 0.0, "flow_label": "NEUTRAL", "factor_score": 0.0}

    def _multi_timeframe_context(self, interval: str, signal: str) -> dict[str, Any]:
        requirements = {
            "5m": ["15m", "4h", "1d"],
            "15m": ["4h", "1d"],
            "1h": ["4h", "1d"],
            "4h": ["1d"],
        }
        tfs = requirements.get(interval, [])
        biases: dict[str, str] = {}
        ordered_tfs = [interval, *[tf for tf in tfs if tf != interval]]
        for tf in ordered_tfs:
            try:
                df = self.get_recent_frame(interval=tf, limit=400)
                if df.empty:
                    biases[tf] = "NEUTRAL"
                    continue
                timeframe = "daily" if tf in {"1d", "1w", "1M"} else "intraday"
                clean_df, _ = self._anomaly.inspect_and_clean(df, BTC_SYMBOL, "crypto", timeframe)
                tf_feat = self._engineer.compute_all_features(clean_df, timeframe=timeframe)
                biases[tf] = self._bias_from_features(tf_feat)
            except Exception:
                biases[tf] = "NEUTRAL"

        desired = signal if signal in {"LONG", "SHORT"} else None
        alignment_ok = True
        aligned = 0
        blocking_timeframes: list[str] = []
        for tf, bias in biases.items():
            if tf == interval or desired is None:
                continue
            if bias == desired:
                aligned += 1
            elif bias not in {"NEUTRAL", desired}:
                alignment_ok = False
                blocking_timeframes.append(tf)

        higher_biases = {tf: bias for tf, bias in biases.items() if tf != interval}
        if desired is not None and higher_biases:
            alignment_ok = alignment_ok and aligned >= max(1, len(higher_biases) - 1)
        return {
            "biases": biases,
            "current_tf": interval,
            "current_bias": biases.get(interval, "NEUTRAL"),
            "alignment_ok": alignment_ok,
            "aligned_count": aligned,
            "required": list(tfs),
            "blocking_timeframes": blocking_timeframes,
        }

    @classmethod
    def _bias_from_features(cls, feat: pd.DataFrame) -> str:
        if feat.empty or "Close" not in feat.columns:
            return "NEUTRAL"
        close = cls._last_feature_or_default(feat, "Close", 0.0)
        ema_21_default = float(feat["Close"].ewm(span=21).mean().iloc[-1]) if len(feat) > 21 else close
        ema_55_default = float(feat["Close"].ewm(span=55).mean().iloc[-1]) if len(feat) > 55 else ema_21_default
        ema_21 = cls._last_feature_or_default(feat, "EMA_21", ema_21_default)
        ema_55 = cls._last_feature_or_default(feat, "EMA_55", cls._last_feature_or_default(feat, "EMA_50", ema_55_default))
        if close > ema_21 > ema_55:
            return "LONG"
        if close < ema_21 < ema_55:
            return "SHORT"
        return "NEUTRAL"

    @staticmethod
    def _halving_phase_context(today: date) -> dict[str, Any]:
        halving_date = date(2024, 4, 19)
        days_since = (today - halving_date).days
        if days_since < 0:
            phase = "pre_halving"
            bias = "LONG"
            mult = 1.02
        elif days_since < 180:
            phase = "post_halving_accumulation"
            bias = "LONG"
            mult = 1.05
        elif days_since < 330:
            phase = "post_halving_expansion"
            bias = "LONG"
            mult = 1.03
        elif days_since < 540:
            phase = "distribution"
            bias = "SHORT"
            mult = 0.94
        else:
            phase = "late_cycle"
            bias = "NEUTRAL"
            mult = 0.98
        return {
            "phase": phase,
            "bias": bias,
            "days_since_halving": days_since,
            "confidence_multiplier": mult,
        }

    @staticmethod
    def _infer_regime(feat: pd.DataFrame, funding_rate_z: float = 0.0, funding_rate_pct: float = 0.0) -> str:
        """
        Infer BTC market regime from price structure + on-chain sentiment.

        Regimes (priority order):
          VOLATILITY SPIKE   — ATR > 2.5× 50-bar avg → no trades
          CAPITULATION       — bearish structure + extreme negative funding
                               (overleveraged shorts funding longs → reversal setup)
          DISTRIBUTION       — bullish structure + extreme positive funding
                               (overleveraged longs funding shorts → reversal setup)
          BULLISH TREND      — price > EMA21 > EMA50
          BEARISH TREND      — price < EMA21 < EMA50
          RANGE (SQUEEZE)    — compressed ATR between EMAs
          RANGE              — mixed / transitioning structure
        """
        if feat.empty or "Close" not in feat.columns or len(feat) < 50:
            return "RANGE"

        close = float(feat["Close"].iloc[-1])
        ema_21 = float(feat["Close"].ewm(span=21).mean().iloc[-1])
        ema_50 = float(feat["Close"].ewm(span=50).mean().iloc[-1])

        # ATR-based volatility regime (highest priority — overrides everything)
        atr_col = feat.get("ATR_14")
        current_atr: float | None = None
        avg_atr: float | None = None
        if atr_col is not None and len(atr_col) > 50:
            current_atr = float(atr_col.iloc[-1])
            avg_atr = float(atr_col.tail(50).mean())
            if avg_atr > 0 and current_atr > 2.5 * avg_atr:
                return "VOLATILITY SPIKE"

        bearish_structure = close < ema_21 < ema_50
        bullish_structure = close > ema_21 > ema_50

        extreme_negative_funding = funding_rate_z < -1.5 and funding_rate_pct <= -0.03
        extreme_positive_funding = funding_rate_z > 1.5 and funding_rate_pct >= 0.03

        # CAPITULATION: bearish EMA structure + extremely negative funding
        # Funding rate Z < -1.5 means shorts are paying longs a premium →
        # overleveraged short positioning historically precedes sharp reversals.
        if bearish_structure and extreme_negative_funding:
            return "CAPITULATION"

        # DISTRIBUTION: bullish EMA structure + extremely positive funding
        # Funding rate Z > 1.5 means longs are paying shorts a premium →
        # overleveraged long positioning historically precedes sharp sell-offs.
        if bullish_structure and extreme_positive_funding:
            return "DISTRIBUTION"

        if bullish_structure:
            return "BULLISH TREND"
        if bearish_structure:
            return "BEARISH TREND"

        # Range-bound: price between EMAs
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
            try:
                resp = self._session.get(url, params=params, timeout=BINANCE_HTTP_TIMEOUT_SECONDS)
            except requests.exceptions.SSLError:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                resp = self._session.get(
                    url,
                    params=params,
                    timeout=BINANCE_HTTP_TIMEOUT_SECONDS,
                    verify=False,
                )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Binance klines fetch failed: %s", exc)
            return []

    @staticmethod
    def _fetch_yfinance_recent_frame(interval: str, limit: int = 1000) -> pd.DataFrame:
        yf_interval_map = {
            "1m": "1m",
            "3m": "5m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "2h": "60m",
            "4h": "60m",
            "6h": "60m",
            "8h": "60m",
            "12h": "60m",
            "1d": "1d",
            "3d": "1d",
            "1w": "1wk",
            "1M": "1mo",
        }
        yf_interval = yf_interval_map.get(interval, "15m")
        if yf_interval == "1m":
            period = "7d"
        elif yf_interval in {"5m", "15m", "30m", "60m"}:
            period = "60d"
        elif yf_interval == "1d":
            period = "2y"
        else:
            period = "10y"
        try:
            import yfinance as yf  # type: ignore[import]

            raw = yf.download(
                "BTC-USD",
                interval=yf_interval,
                period=period,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if raw is None or raw.empty:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            keep = raw.rename(columns={c: str(c).title() for c in raw.columns})
            keep = keep[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in keep.columns]].copy()
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in keep.columns:
                    keep[col] = 0.0
                keep[col] = pd.to_numeric(keep[col], errors="coerce")
            keep = keep.dropna(subset=["Open", "High", "Low", "Close"]).tail(int(max(1, limit)))
            if keep.index.tz is None:
                keep.index = keep.index.tz_localize("UTC")
            else:
                keep.index = keep.index.tz_convert("UTC")
            return keep[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as exc:
            logger.warning("yfinance BTC fallback failed: %s", exc)
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

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



