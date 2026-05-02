"""
ETH/SOL market data + deep signal service.

Design goals:
  - Keep BTC service untouched.
  - Reuse the same quant factor architecture style (IC-weighted factors + gate pipeline).
  - Use spot-only market data (no futures/leverage execution logic).
  - Enrich with chain context from Alchemy (ETH) / Helius (SOL) and DEX context (Jupiter for SOL).
"""

from __future__ import annotations

import logging
import os
import statistics
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import urllib3

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

from src.alpha.factor_model import AlphaFactorModel, _latest_usable_ic, _rolling_ic
from src.api.data_quality import DataAnomalyDetector
from src.api.rate_limiter import TTLCache
from src.features.engineer import FeatureEngineer
from src.risk.cost_model import CostModel

logger = logging.getLogger(__name__)

BINANCE_SPOT_REST = "https://api.binance.com"
SUPPORTED_SYMBOLS = {"ETHUSDT", "SOLUSDT"}

REGIME_CONFIDENCE_MULTIPLIER = {
    "BEARISH TREND": 1.20,
    "BULLISH TREND": 1.00,
    "SIDEWAYS": 1.10,
    "RANGE": 1.08,
    "VOLATILITY SPIKE": 1.25,
}
BASE_CONFIDENCE_THRESHOLD = 65.0
BEARISH_REGIMES = {"BEARISH TREND"}
BULLISH_REGIMES = {"BULLISH TREND"}

INTERVAL_TO_BINANCE = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class AltcoinMarketService:
    def __init__(self, cfg: dict | None = None):
        if load_dotenv is not None:
            load_dotenv()

        self._config = cfg or {}
        signal_cfg = (self._config.get("signal") or {})

        self._session = requests.Session()
        self._ohlcv_cache = TTLCache()
        self._signal_cache = TTLCache()

        self._engineer = FeatureEngineer()
        self._anomaly = DataAnomalyDetector()
        self._alpha = AlphaFactorModel(
            alpha_threshold=float(signal_cfg.get("alpha_score_threshold", 0.15)),
            ic_window=int(signal_cfg.get("ic_window", 60)),
        )
        self._cost = CostModel((self._config.get("cost_model") or {}))

        self._binance_api_key = os.getenv("BINANCE_API_KEY", "").strip()
        self._alchemy_api_key = os.getenv("ALCHEMY_API_KEY", "").strip()
        self._alchemy_eth_rpc_url = os.getenv("ALCHEMY_ETH_RPC_URL", "").strip()
        self._helius_api_key = os.getenv("HELIUS_API_KEY", "").strip()
        self._helius_sol_rpc_url = os.getenv("HELIUS_SOL_RPC_URL", "").strip()
        self._jupiter_api_key = os.getenv("JUPITER_API_KEY", "").strip()

    def get_recent_candles(self, symbol: str = "ETHUSDT", interval: str = "15m", limit: int = 300) -> dict[str, Any]:
        sym = self._normalize_symbol(symbol)
        if sym not in SUPPORTED_SYMBOLS:
            return {"symbol": sym, "data": [], "error": "Unsupported symbol"}

        tf = self._normalize_interval(interval)
        bars = max(80, min(int(limit), 1200))
        frame = self._fetch_ohlcv(sym, interval=tf, limit=bars)
        if frame.empty:
            return {"symbol": sym, "interval": tf, "data": [], "error": "No candles"}

        rows: list[dict[str, Any]] = []
        for idx, row in frame.tail(bars).iterrows():
            rows.append(
                {
                    "time": int(pd.Timestamp(idx).timestamp()),
                    "open": round(float(row["Open"]), 6),
                    "high": round(float(row["High"]), 6),
                    "low": round(float(row["Low"]), 6),
                    "close": round(float(row["Close"]), 6),
                    "volume": float(row["Volume"]),
                }
            )
        return {
            "symbol": sym,
            "interval": tf,
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "data": rows,
        }

    def get_realtime_signal(self, symbol: str = "ETHUSDT", interval: str = "15m") -> dict[str, Any]:
        sym = self._normalize_symbol(symbol)
        if sym not in SUPPORTED_SYMBOLS:
            return self._hold_payload(sym, interval, "Unsupported symbol")

        tf = self._normalize_interval(interval)
        cache_key = f"alt_sig:{sym}:{tf}"
        cached = self._signal_cache.get(cache_key)
        if cached is not None:
            return cached

        frame = self._fetch_ohlcv(sym, interval=tf, limit=1200)
        if frame.empty or len(frame) < 160:
            payload = self._hold_payload(sym, tf, "Insufficient spot candles")
            self._signal_cache.set(cache_key, payload, ttl_seconds=6)
            return payload

        timeframe = "daily" if tf == "1d" else "intraday"
        clean, dq = self._anomaly.inspect_and_clean(frame, sym, "crypto", timeframe)
        feat = self._engineer.compute_all_features(clean, timeframe=timeframe)
        score = self._score_frame(feat)
        if score is None:
            payload = self._hold_payload(sym, tf, "Feature window not ready")
            payload["data_quality"] = dq.to_dict()
            self._signal_cache.set(cache_key, payload, ttl_seconds=6)
            return payload

        close = float(feat["Close"].iloc[-1])
        ema21 = float(feat["Close"].ewm(span=21).mean().iloc[-1]) if len(feat) > 21 else close
        ema50 = float(feat["Close"].ewm(span=50).mean().iloc[-1]) if len(feat) > 50 else ema21
        price_1h = self._lookback_price(feat, tf, hours=1, fallback=close)
        price_24h = self._lookback_price(feat, tf, hours=24, fallback=close)
        price_change_1h = ((close - price_1h) / max(price_1h, 1e-9)) * 100.0
        price_change_24h = ((close - price_24h) / max(price_24h, 1e-9)) * 100.0

        atr = float(score["atr_14"])
        atr_pct = (atr / max(close, 1e-9)) * 100.0
        rsi = self._last_feature_or_default(feat, "RSI_14", 50.0)
        cmf = self._last_feature_or_default(feat, "CMF", 0.0)
        obv_slope = self._last_feature_or_default(feat, "OBV_Slope", 0.0)
        volume_ratio = self._last_feature_or_default(feat, "Volume_Ratio", 1.0)
        volatility_20 = self._last_feature_or_default(feat, "Volatility_20", 0.2)

        regime = self._infer_regime(close=close, ema21=ema21, ema50=ema50, atr_pct=atr_pct, rsi=rsi)

        model_signal = str(score["signal"])
        signal = "LONG" if model_signal == "BUY" else "SHORT" if model_signal == "SELL" else "HOLD"
        confidence = float(score["confidence"])
        adjusted_threshold = self._regime_threshold(signal, regime)

        alpha_score = float(score["alpha_score"])
        signal_edge = abs(alpha_score) if signal in {"LONG", "SHORT"} else 0.0
        net_edge, cost_pct, cost_ok = self._cost.net_alpha(
            alpha_score=signal_edge,
            asset_class="crypto",
            position_size_pct=1.0,
            daily_vol=max(float(volatility_20), 0.05),
            volume_ratio=max(float(volume_ratio), 0.05),
            regime=regime,
            low_liquidity=volume_ratio < 0.8,
        )
        net_alpha = net_edge if alpha_score >= 0 else -net_edge

        onchain = self._onchain_snapshot(sym)
        liquidity = self._liquidity_snapshot(sym)
        mtf = self._multi_timeframe_alignment(sym, signal)

        checks = {
            "data_quality": not bool(dq.severe),
            "confidence_gate": confidence >= adjusted_threshold,
            "cost_gate": bool(cost_ok),
            "regime_gate": self._regime_allows(signal, regime),
            "mtf_alignment": bool(mtf.get("aligned", False)),
            "onchain_gate": bool(onchain.get("gate_pass", True)),
            "liquidity_gate": bool(liquidity.get("gate_pass", True)),
        }

        validated = bool(signal in {"LONG", "SHORT"} and all(checks.values()))
        validated_signal = signal if validated else "HOLD"
        strength = self._strength_from_confidence(confidence)

        entry_price: float | None = close if validated else None
        stop_loss: float | None = None
        tp1: float | None = None
        tp2: float | None = None
        tp3: float | None = None
        risk_reward: float | None = None
        if validated and entry_price is not None:
            stop_loss, tp1, tp2, tp3, risk_reward = self._risk_targets(validated_signal, entry_price, atr)

        failed = [k for k, ok in checks.items() if not ok]
        if validated:
            reason = (
                f"{sym} {validated_signal} validated | {regime} | conf {confidence:.1f}% | "
                f"net edge {net_edge:+.4f}"
            )
        elif signal in {"LONG", "SHORT"}:
            reason = "Signal blocked: " + ", ".join(failed) if failed else "Signal blocked by validation"
        else:
            reason = f"HOLD: alpha below threshold in {regime}"

        validation_score = round(sum(1 for ok in checks.values() if ok) / max(len(checks), 1), 4)

        context = {
            "price_change_1h": round(price_change_1h, 3),
            "price_change_24h": round(price_change_24h, 3),
            "volume_ratio": round(float(volume_ratio), 4),
            "volatility_20": round(float(volatility_20), 4),
            "atr_pct": round(float(atr_pct), 3),
            "rsi": round(float(rsi), 2),
            "cmf": round(float(cmf), 4),
            "obv_slope": round(float(obv_slope), 4),
            "smart_money_state": "RISING" if obv_slope >= 0 else "FALLING",
            "onchain": onchain,
            "liquidity": liquidity,
            "mtf": mtf,
        }

        position_sizing = self._position_sizing(
            confidence=confidence,
            checks=checks,
            volatility_20=float(volatility_20),
            signal=validated_signal,
        )

        payload: dict[str, Any] = {
            "asset": sym,
            "symbol": sym,
            "interval": tf,
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "signal": signal,
            "validated_signal": validated_signal,
            "validated": validated,
            "reason": reason,
            "regime": regime,
            "strength": strength,
            "confidence": round(confidence, 2),
            "ai_confidence": round(confidence, 2),
            "adjusted_confidence_threshold": round(adjusted_threshold, 2),
            "alpha_score": round(float(score["alpha_score"]), 4),
            "net_alpha_score": round(float(net_edge), 4),
            "signed_net_alpha_score": round(float(net_alpha), 4),
            "entry_price": round(float(entry_price), 4) if entry_price is not None else None,
            "entry": round(float(entry_price), 4) if entry_price is not None else None,
            "stop_loss": round(float(stop_loss), 4) if stop_loss is not None else None,
            "take_profit": round(float(tp2), 4) if tp2 is not None else None,
            "tp1": round(float(tp1), 4) if tp1 is not None else None,
            "tp2": round(float(tp2), 4) if tp2 is not None else None,
            "tp3": round(float(tp3), 4) if tp3 is not None else None,
            "risk_reward": round(float(risk_reward), 4) if risk_reward is not None else None,
            "factor_scores": score["factor_scores"],
            "ic_weights": score["ic_weights"],
            "gate_status": checks,
            "market_context": context,
            "validation": {
                "score": validation_score,
                "checks": checks,
                "cost_pct": round(float(cost_pct), 5),
                "net_alpha_score": round(float(net_edge), 4),
                "signed_net_alpha_score": round(float(net_alpha), 4),
            },
            "pipeline": {
                "raw_signal": signal,
                "regime_gate": "PASS" if checks["regime_gate"] else "FAIL",
                "confidence_val": round(confidence, 2),
                "confidence_threshold": round(adjusted_threshold, 2),
                "confidence_gate": "PASS" if checks["confidence_gate"] else "FAIL",
                "cost_gate_val": round(float(net_edge), 4),
                "cost_gate": "PASS" if checks["cost_gate"] else "FAIL",
                "output": validated_signal,
            },
            "position_sizing": position_sizing,
            "algo": "quant_alpha_factor_model_v1",
        }
        payload["data_quality"] = dq.to_dict()
        self._signal_cache.set(cache_key, payload, ttl_seconds=6)
        return payload

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "ETHUSDT").strip().upper()

    def _normalize_interval(self, interval: str) -> str:
        iv = str(interval or "15m").strip().lower()
        if iv in INTERVAL_TO_BINANCE:
            return iv
        return "15m"

    def _fetch_ohlcv(self, symbol: str, interval: str, limit: int = 1200) -> pd.DataFrame:
        cache_key = f"alt_ohlcv:{symbol}:{interval}:{int(limit)}"
        cached = self._ohlcv_cache.get(cache_key)
        if cached is not None:
            return cached

        headers: dict[str, str] = {}
        if self._binance_api_key:
            headers["X-MBX-APIKEY"] = self._binance_api_key

        try:
            url = f"{BINANCE_SPOT_REST}/api/v3/klines"
            try:
                res = self._session.get(
                    url,
                    params={"symbol": symbol, "interval": interval, "limit": int(limit)},
                    timeout=8,
                    headers=headers or None,
                )
            except requests.exceptions.SSLError:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                res = self._session.get(
                    url,
                    params={"symbol": symbol, "interval": interval, "limit": int(limit)},
                    timeout=8,
                    headers=headers or None,
                    verify=False,
                )
            res.raise_for_status()
            rows = res.json()
            if not isinstance(rows, list):
                return pd.DataFrame()
            frame = self._klines_to_df(rows)
            ttl = 6 if interval in {"5m", "15m"} else 15
            self._ohlcv_cache.set(cache_key, frame, ttl_seconds=ttl)
            return frame
        except Exception as exc:
            logger.warning("Altcoin OHLCV fetch failed for %s (%s): %s", symbol, interval, exc)
            return pd.DataFrame()

    @staticmethod
    def _klines_to_df(rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        records: list[dict[str, Any]] = []
        for row in rows:
            try:
                ts_ms = int(row[0])
                records.append(
                    {
                        "time": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                        "Open": float(row[1]),
                        "High": float(row[2]),
                        "Low": float(row[3]),
                        "Close": float(row[4]),
                        "Volume": float(row[5]),
                    },
                )
            except Exception:
                continue
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records).set_index("time").sort_index()
        return frame

    def _score_frame(self, feat: pd.DataFrame) -> dict[str, Any] | None:
        if feat.empty or len(feat) < self._alpha.ic_window + 5:
            return None
        if "Returns" not in feat.columns:
            return None

        fwd_ret = feat["Returns"].shift(-1)
        ml_series = pd.Series(0.0, index=feat.index)

        f1 = self._alpha._factor1_momentum(feat, 0.5)
        f2 = self._alpha._factor2_mean_reversion(feat, 0.5)
        f3 = self._alpha._factor3_volume(feat)
        f4 = self._alpha._factor4_ml(ml_series)
        f5 = self._alpha._factor5_volatility_squeeze(feat, momentum_factor=f1)
        f8 = self._alpha._factor8_microstructure(feat)
        factors = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F8": f8}

        ic_map: dict[str, float] = {}
        latest_map: dict[str, float] = {}
        for name, factor in factors.items():
            ic_series = _rolling_ic(factor, fwd_ret, self._alpha.ic_window)
            ic = _latest_usable_ic(ic_series)
            if not np.isfinite(ic):
                ic = 0.0
            lv = float(factor.iloc[-1]) if len(factor) else 0.0
            if not np.isfinite(lv):
                lv = 0.0
            ic_map[name] = ic
            latest_map[name] = lv

        denom = max(sum(abs(v) for v in ic_map.values()), 0.1)
        alpha_score = float(sum(ic_map[k] * latest_map[k] for k in ic_map) / denom)
        confidence = float(np.clip(50.0 + 50.0 * np.tanh(abs(alpha_score)), 0.0, 100.0))

        if alpha_score > self._alpha.alpha_threshold:
            signal = "BUY"
        elif alpha_score < -self._alpha.alpha_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        atr_14 = self._last_feature_or_default(feat, "ATR_14", 0.0)
        return {
            "signal": signal,
            "alpha_score": alpha_score,
            "confidence": confidence,
            "factor_scores": {k: round(v, 4) for k, v in latest_map.items()},
            "ic_weights": {k: round(v, 4) for k, v in ic_map.items()},
            "atr_14": max(float(atr_14), 0.0),
        }

    @staticmethod
    def _last_feature_or_default(feat: pd.DataFrame, column: str, default: float) -> float:
        try:
            if column not in feat.columns or feat.empty:
                return float(default)
            val = float(feat[column].iloc[-1])
            if not np.isfinite(val):
                return float(default)
            return val
        except Exception:
            return float(default)

    @staticmethod
    def _lookback_price(feat: pd.DataFrame, interval: str, hours: int, fallback: float) -> float:
        bars_per_hour = {"5m": 12, "15m": 4, "1h": 1, "4h": 0.25, "1d": 1 / 24}.get(interval, 4)
        lookback = int(max(1, round(hours * bars_per_hour)))
        if "Close" not in feat.columns or len(feat) <= lookback:
            return fallback
        try:
            return float(feat["Close"].iloc[-(lookback + 1)])
        except Exception:
            return fallback

    @staticmethod
    def _strength_from_confidence(confidence: float) -> str:
        if confidence >= 75:
            return "STRONG"
        if confidence >= 60:
            return "MODERATE"
        return "WEAK"

    @staticmethod
    def _infer_regime(close: float, ema21: float, ema50: float, atr_pct: float, rsi: float) -> str:
        if atr_pct >= 2.5:
            return "VOLATILITY SPIKE"
        if close > ema21 > ema50 and rsi >= 55:
            return "BULLISH TREND"
        if close < ema21 < ema50 and rsi <= 45:
            return "BEARISH TREND"
        if atr_pct < 0.7:
            return "SIDEWAYS"
        return "RANGE"

    @staticmethod
    def _regime_allows(signal: str, regime: str) -> bool:
        if signal == "LONG" and regime in BEARISH_REGIMES:
            return False
        if signal == "SHORT" and regime in BULLISH_REGIMES:
            return False
        return True

    def _regime_threshold(self, signal: str, regime: str) -> float:
        mult = float(REGIME_CONFIDENCE_MULTIPLIER.get(regime, 1.0))
        with_trend = (
            (signal == "LONG" and regime in BULLISH_REGIMES)
            or (signal == "SHORT" and regime in BEARISH_REGIMES)
        )
        counter_trend = (
            (signal == "LONG" and regime in BEARISH_REGIMES)
            or (signal == "SHORT" and regime in BULLISH_REGIMES)
        )
        if with_trend:
            return BASE_CONFIDENCE_THRESHOLD * mult
        if counter_trend:
            return BASE_CONFIDENCE_THRESHOLD * mult * 1.10
        return BASE_CONFIDENCE_THRESHOLD * mult

    @staticmethod
    def _risk_targets(signal: str, entry: float, atr: float) -> tuple[float, float, float, float, float]:
        sl_mult = 1.8
        rr1, rr2, rr3 = 1.2, 2.0, 2.8
        dist = max(atr * sl_mult, entry * 0.004)
        if signal == "LONG":
            stop = entry - dist
            tp1 = entry + dist * rr1
            tp2 = entry + dist * rr2
            tp3 = entry + dist * rr3
        else:
            stop = entry + dist
            tp1 = entry - dist * rr1
            tp2 = entry - dist * rr2
            tp3 = entry - dist * rr3
        rr = abs((tp2 - entry) / max(abs(entry - stop), 1e-9))
        return stop, tp1, tp2, tp3, rr

    def _multi_timeframe_alignment(self, symbol: str, signal: str) -> dict[str, Any]:
        if signal not in {"LONG", "SHORT"}:
            return {"aligned": True, "score": 1.0, "details": {"1h": "NEUTRAL", "4h": "NEUTRAL"}}

        details: dict[str, str] = {}
        aligned = 0
        total = 0
        for tf in ("1h", "4h"):
            frame = self._fetch_ohlcv(symbol, interval=tf, limit=280)
            if frame.empty or len(frame) < 80:
                details[tf] = "UNAVAILABLE"
                continue
            total += 1
            close = float(frame["Close"].iloc[-1])
            ema21 = float(frame["Close"].ewm(span=21).mean().iloc[-1])
            ema50 = float(frame["Close"].ewm(span=50).mean().iloc[-1])
            trend = "BULLISH" if close > ema21 > ema50 else "BEARISH" if close < ema21 < ema50 else "SIDEWAYS"
            details[tf] = trend
            if (signal == "LONG" and trend == "BULLISH") or (signal == "SHORT" and trend == "BEARISH"):
                aligned += 1

        if total == 0:
            return {"aligned": True, "score": 1.0, "details": details}
        score = aligned / total
        return {"aligned": score >= 0.5, "score": round(score, 3), "details": details}

    def _position_sizing(self, confidence: float, checks: dict[str, bool], volatility_20: float, signal: str) -> dict[str, Any]:
        if signal not in {"LONG", "SHORT"}:
            return {"size_pct": 0.0, "method": "hold", "bucket": "NO_TRADE"}
        pass_ratio = sum(1 for v in checks.values() if v) / max(len(checks), 1)
        base = 0.8 + (confidence / 100.0) * 1.4
        vol_penalty = 0.8 if volatility_20 > 0.6 else 1.0
        size_pct = max(0.25, min(3.0, base * pass_ratio * vol_penalty))
        bucket = "HIGH" if size_pct >= 2.0 else "MEDIUM" if size_pct >= 1.0 else "LOW"
        return {
            "size_pct": round(float(size_pct), 3),
            "method": "confidence_x_gate_ratio",
            "bucket": bucket,
            "volatility_penalty": round(float(vol_penalty), 3),
            "pass_ratio": round(float(pass_ratio), 3),
        }

    def _onchain_snapshot(self, symbol: str) -> dict[str, Any]:
        if symbol == "ETHUSDT":
            return self._eth_onchain_snapshot()
        return self._sol_onchain_snapshot()

    def _eth_onchain_snapshot(self) -> dict[str, Any]:
        url = self._alchemy_eth_rpc_url
        if not url and self._alchemy_api_key:
            url = f"https://eth-mainnet.g.alchemy.com/v2/{self._alchemy_api_key}"
        if not url:
            return {"provider": "alchemy", "available": False, "gate_pass": True, "reason": "ALCHEMY key missing"}

        block_num_hex = self._rpc_call(url, "eth_blockNumber", [])
        gas_hex = self._rpc_call(url, "eth_gasPrice", [])
        latest_block = self._rpc_call(url, "eth_getBlockByNumber", ["latest", False])

        block_num = int(block_num_hex, 16) if isinstance(block_num_hex, str) and block_num_hex.startswith("0x") else 0
        gas_wei = int(gas_hex, 16) if isinstance(gas_hex, str) and gas_hex.startswith("0x") else 0
        base_fee_hex = latest_block.get("baseFeePerGas") if isinstance(latest_block, dict) else None
        base_fee_wei = int(base_fee_hex, 16) if isinstance(base_fee_hex, str) and base_fee_hex.startswith("0x") else 0

        gas_gwei = gas_wei / 1e9 if gas_wei > 0 else 0.0
        base_fee_gwei = base_fee_wei / 1e9 if base_fee_wei > 0 else 0.0
        # Higher gas/base fee indicates elevated chain congestion.
        congestion = max(gas_gwei, base_fee_gwei)
        activity_score = float(np.clip((congestion / 120.0) * 100.0, 0.0, 100.0))
        gate_pass = congestion < 250.0 if congestion > 0 else True

        return {
            "provider": "alchemy",
            "available": block_num > 0,
            "block_height": block_num,
            "gas_gwei": round(gas_gwei, 3),
            "base_fee_gwei": round(base_fee_gwei, 3),
            "activity_score": round(activity_score, 2),
            "gate_pass": gate_pass,
        }

    def _sol_onchain_snapshot(self) -> dict[str, Any]:
        url = self._helius_sol_rpc_url
        if not url and self._helius_api_key:
            url = f"https://mainnet.helius-rpc.com/?api-key={self._helius_api_key}"
        if not url:
            return {"provider": "helius", "available": False, "gate_pass": True, "reason": "HELIUS key missing"}

        slot_val = self._rpc_call(url, "getSlot", [])
        fees_arr = self._rpc_call(url, "getRecentPrioritizationFees", [])
        latest_blockhash = self._rpc_call(url, "getLatestBlockhash", [{"commitment": "processed"}])

        try:
            slot = int(slot_val) if slot_val is not None else 0
        except Exception:
            slot = 0

        fees: list[float] = []
        if isinstance(fees_arr, list):
            for row in fees_arr:
                try:
                    fees.append(float((row or {}).get("prioritizationFee", 0.0)))
                except Exception:
                    continue
        median_fee = statistics.median(fees) if fees else 0.0
        gate_pass = median_fee < 25000 if median_fee > 0 else True

        return {
            "provider": "helius",
            "available": slot > 0,
            "slot": slot,
            "priority_fee_lamports": round(float(median_fee), 2),
            "latest_blockhash_available": bool(
                isinstance(latest_blockhash, dict) and ((latest_blockhash.get("value") or {}).get("blockhash"))
            ),
            "activity_score": round(float(np.clip((median_fee / 25000.0) * 100.0, 0.0, 100.0)), 2),
            "gate_pass": gate_pass,
        }

    def _liquidity_snapshot(self, symbol: str) -> dict[str, Any]:
        # Binance spot depth is common baseline for both ETH and SOL.
        depth = self._binance_depth_snapshot(symbol)
        if symbol == "SOLUSDT":
            dex = self._jupiter_liquidity_snapshot()
            gate_pass = bool(depth.get("gate_pass", True)) and bool(dex.get("gate_pass", True))
            return {"venue": "binance+jupiter", "binance": depth, "jupiter": dex, "gate_pass": gate_pass}
        return {"venue": "binance", "binance": depth, "gate_pass": bool(depth.get("gate_pass", True))}

    def _binance_depth_snapshot(self, symbol: str) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self._binance_api_key:
            headers["X-MBX-APIKEY"] = self._binance_api_key
        try:
            res = self._session.get(
                f"{BINANCE_SPOT_REST}/api/v3/depth",
                params={"symbol": symbol, "limit": 100},
                timeout=8,
                headers=headers or None,
            )
            res.raise_for_status()
            payload = res.json()
            bids = payload.get("bids", []) if isinstance(payload, dict) else []
            asks = payload.get("asks", []) if isinstance(payload, dict) else []
            bid_notional = sum(float(p) * float(q) for p, q in bids[:30]) if bids else 0.0
            ask_notional = sum(float(p) * float(q) for p, q in asks[:30]) if asks else 0.0
            total_notional = bid_notional + ask_notional
            imbalance = ((bid_notional - ask_notional) / total_notional) if total_notional > 0 else 0.0
            # Very small top-book notional indicates weak liquidity.
            gate_pass = total_notional >= 800000.0
            score = float(np.clip((total_notional / 4_000_000.0) * 100.0, 0.0, 100.0))
            return {
                "available": True,
                "top30_notional_usd": round(total_notional, 2),
                "book_imbalance": round(imbalance, 4),
                "liquidity_score": round(score, 2),
                "gate_pass": gate_pass,
            }
        except Exception as exc:
            logger.debug("Binance depth fetch failed for %s: %s", symbol, exc)
            return {"available": False, "gate_pass": True, "reason": "depth unavailable"}

    def _jupiter_liquidity_snapshot(self) -> dict[str, Any]:
        # Public quote path first; API key optional for stricter plans.
        headers: dict[str, str] = {}
        if self._jupiter_api_key:
            headers["x-api-key"] = self._jupiter_api_key
        params = {"inputMint": SOL_MINT, "outputMint": USDC_MINT, "amount": "100000000", "slippageBps": "30"}
        urls = [
            "https://lite-api.jup.ag/swap/v1/quote",
            "https://quote-api.jup.ag/v6/quote",
        ]
        for url in urls:
            try:
                res = self._session.get(url, params=params, timeout=8, headers=headers or None)
                if not res.ok:
                    continue
                data = res.json()
                out_amount = float(data.get("outAmount", 0.0) or 0.0)
                price_impact_pct = float(data.get("priceImpactPct", 0.0) or 0.0) * 100.0
                # outAmount is in USDC 6 decimals for output mint.
                usdc_out = out_amount / 1_000_000.0 if out_amount > 0 else 0.0
                sol_in = 0.1  # 100000000 lamports
                px = usdc_out / max(sol_in, 1e-9)
                gate_pass = price_impact_pct <= 1.0 if price_impact_pct > 0 else True
                liq_score = float(np.clip(100.0 - (price_impact_pct * 35.0), 0.0, 100.0))
                return {
                    "available": True,
                    "route_price_usdc": round(px, 4) if px > 0 else 0.0,
                    "price_impact_pct": round(price_impact_pct, 4),
                    "liquidity_score": round(liq_score, 2),
                    "gate_pass": gate_pass,
                }
            except Exception:
                continue
        return {"available": False, "gate_pass": True, "reason": "jupiter unavailable"}

    def _rpc_call(self, url: str, method: str, params: list[Any]) -> Any:
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        try:
            res = self._session.post(url, json=body, timeout=8)
            if not res.ok:
                return None
            data = res.json()
            if isinstance(data, dict):
                return data.get("result")
            return None
        except Exception:
            return None

    def _hold_payload(self, symbol: str, interval: str, reason: str) -> dict[str, Any]:
        return {
            "asset": symbol,
            "symbol": symbol,
            "interval": interval,
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "signal": "HOLD",
            "validated_signal": "HOLD",
            "validated": False,
            "confidence": 0.0,
            "alpha_score": 0.0,
            "net_alpha_score": 0.0,
            "regime": "SIDEWAYS",
            "reason": reason,
            "gate_status": {
                "data_quality": False,
                "confidence_gate": False,
                "cost_gate": False,
                "regime_gate": False,
                "mtf_alignment": False,
                "onchain_gate": False,
                "liquidity_gate": False,
            },
            "validation": {"score": 0.0, "checks": {}},
        }
