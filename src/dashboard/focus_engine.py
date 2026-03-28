"""
Focused real-time quant engine for Gold, Silver, and Bitcoin.

This module powers a dedicated dashboard that surfaces:
  - live OHLCV candles
  - fast factor-based AI trade signals
  - multi-horizon validation gates before a trade is marked actionable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    yf = None  # type: ignore[assignment]
    _YF_AVAILABLE = False

from src.alpha.factor_model import AlphaFactorModel, _rolling_ic
from src.api.data_quality import DataAnomalyDetector
from src.api.rate_limiter import TTLCache
from src.features.engineer import FeatureEngineer
from src.risk.cost_model import CostModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FocusAsset:
    symbol: str
    ticker: str
    name: str
    market: str
    quote_currency: str = "USD"


DEFAULT_FOCUS_ASSETS = [
    FocusAsset(symbol="XAUUSD", ticker="GC=F", name="Gold", market="metals"),
    FocusAsset(symbol="XAGUSD", ticker="SI=F", name="Silver", market="metals"),
    FocusAsset(symbol="BTCUSD", ticker="BTC-USD", name="Bitcoin", market="crypto"),
]


INTERVAL_DEFAULTS: dict[str, dict[str, int | str]] = {
    "1m": {"period": "1d", "freshness_sec": 10 * 60},
    "5m": {"period": "5d", "freshness_sec": 15 * 60},
    "15m": {"period": "5d", "freshness_sec": 30 * 60},
    "1h": {"period": "1mo", "freshness_sec": 3 * 60 * 60},
    "1d": {"period": "6mo", "freshness_sec": 72 * 60 * 60},
}

DEFAULT_HORIZONS: list[tuple[str, str]] = [
    ("5m", "5d"),
    ("15m", "5d"),
    ("1h", "1mo"),
]


def _to_utc(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").to_pydatetime()
    return ts.tz_convert("UTC").to_pydatetime()


class FocusQuantEngine:
    """Low-latency scoring and validation for focus assets."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        signal_cfg = cfg.get("signal", {}) or {}

        self._engineer = FeatureEngineer()
        self._anomaly = DataAnomalyDetector(
            stale_bars=3,
            spike_sigma=5.0,
            min_spike_lookback=20,
        )
        self._model = AlphaFactorModel(
            alpha_threshold=float(signal_cfg.get("alpha_score_threshold", 0.15)),
            ic_window=int(signal_cfg.get("ic_window", 60)),
        )
        self._cost_model = CostModel(cfg.get("cost_model", {}) or {})
        self._ohlcv_cache = TTLCache()
        self._trade_cache = TTLCache()

        self._assets = self._load_assets(cfg)
        self._horizons = self._load_horizons(cfg)

    def _load_assets(self, cfg: dict) -> list[FocusAsset]:
        entries = cfg.get("focus_assets", []) or []
        assets: list[FocusAsset] = []
        for item in entries:
            symbol = str(item.get("symbol", "")).strip().upper()
            ticker = str(item.get("yf_ticker", "")).strip()
            name = str(item.get("name", symbol)).strip() or symbol
            market = str(item.get("market", "crypto")).strip().lower()
            quote = str(item.get("quote_currency", "USD")).strip().upper() or "USD"
            if symbol and ticker:
                assets.append(
                    FocusAsset(
                        symbol=symbol,
                        ticker=ticker,
                        name=name,
                        market=market,
                        quote_currency=quote,
                    )
                )
        return assets if assets else list(DEFAULT_FOCUS_ASSETS)

    def _load_horizons(self, cfg: dict) -> list[tuple[str, str]]:
        raw = (cfg.get("focus_validation", {}) or {}).get("horizons", []) or []
        horizons: list[tuple[str, str]] = []
        for item in raw:
            interval = str(item.get("interval", "")).strip()
            period = str(item.get("period", "")).strip()
            if interval and period:
                horizons.append((interval, period))
        return horizons if horizons else list(DEFAULT_HORIZONS)

    def list_assets(self) -> list[dict[str, str]]:
        return [
            {
                "symbol": a.symbol,
                "yf_ticker": a.ticker,
                "name": a.name,
                "market": a.market,
                "quote_currency": a.quote_currency,
            }
            for a in self._assets
        ]

    def resolve_asset(self, symbol_or_ticker: str) -> FocusAsset | None:
        key = (symbol_or_ticker or "").strip().upper()
        if not key:
            return None
        for asset in self._assets:
            if key in {asset.symbol.upper(), asset.ticker.upper()}:
                return asset
        return None

    def get_chart_data(
        self,
        symbol_or_ticker: str,
        interval: str = "5m",
        period: str | None = None,
    ) -> dict[str, Any]:
        asset = self.resolve_asset(symbol_or_ticker)
        if asset is None:
            return {"symbol": symbol_or_ticker, "data": [], "error": "Unknown asset"}
        df = self._fetch_ohlcv(asset.ticker, interval=interval, period=period)
        if df.empty:
            return {"symbol": asset.symbol, "ticker": asset.ticker, "data": [], "error": "No data"}
        records = []
        for idx, row in df.iterrows():
            ts = _to_utc(pd.Timestamp(idx))
            records.append(
                {
                    "time": int(ts.timestamp()),
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                    "volume": int(float(row["Volume"])),
                }
            )
        return {
            "symbol": asset.symbol,
            "ticker": asset.ticker,
            "name": asset.name,
            "interval": interval,
            "period": period or self._default_period(interval),
            "data": records,
            "last_bar_utc": _to_utc(pd.Timestamp(df.index[-1])).isoformat(),
        }

    def get_focus_trades(self, interval: str = "5m") -> list[dict[str, Any]]:
        trades = [self.get_focus_trade(asset.symbol, interval=interval) for asset in self._assets]
        return sorted(trades, key=lambda x: x.get("symbol", ""))

    def get_focus_trade(
        self,
        symbol_or_ticker: str,
        interval: str = "5m",
        period: str | None = None,
    ) -> dict[str, Any]:
        asset = self.resolve_asset(symbol_or_ticker)
        if asset is None:
            return {
                "symbol": symbol_or_ticker,
                "signal": "HOLD",
                "validated_signal": "HOLD",
                "validated": False,
                "reason": "Unknown focus asset",
            }

        cache_key = f"focus_trade:{asset.symbol}:{interval}:{period or self._default_period(interval)}"
        cached = self._trade_cache.get(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_ohlcv(asset.ticker, interval=interval, period=period)
        if df.empty or len(df) < 120:
            result = self._hold_payload(asset, "Insufficient OHLCV bars", interval, period)
            self._trade_cache.set(cache_key, result, ttl_seconds=6)
            return result

        cleaned, report = self._anomaly.inspect_and_clean(
            df,
            asset=asset.symbol,
            asset_class=asset.market,
            timeframe="intraday" if interval != "1d" else "daily",
        )
        score = self._score_frame(cleaned, interval)
        if score is None:
            result = self._hold_payload(asset, "Feature computation failed", interval, period)
            self._trade_cache.set(cache_key, result, ttl_seconds=6)
            return result

        entry = float(cleaned["Close"].iloc[-1])
        atr = float(score["atr_14"])
        sl_mult = 2.0 if score["strength"] != "STRONG" else 1.6
        rr = 2.5 if score["strength"] != "STRONG" else 3.0
        if score["signal"] == "SELL":
            stop = entry + sl_mult * atr
            take = entry - sl_mult * atr * rr
        else:
            stop = entry - sl_mult * atr
            take = entry + sl_mult * atr * rr

        horizons = self._horizon_scores(asset)
        primary_signal = str(score["signal"])
        directional_horizons = [h for h in horizons if h["signal"] in {"BUY", "SELL"}]
        if primary_signal in {"BUY", "SELL"} and directional_horizons:
            agree = sum(1 for h in directional_horizons if h["signal"] == primary_signal)
            consensus_ratio = agree / len(directional_horizons)
        else:
            consensus_ratio = 0.0

        last_bar_utc = _to_utc(pd.Timestamp(cleaned.index[-1]))
        freshness_sec = max((datetime.now(timezone.utc) - last_bar_utc).total_seconds(), 0.0)
        freshness_limit = int(INTERVAL_DEFAULTS.get(interval, INTERVAL_DEFAULTS["5m"])["freshness_sec"])

        market_vol = float(score["volatility_20"])
        volume_ratio = float(score["volume_ratio"])
        cost_asset_class = "forex" if asset.market in {"metals", "crypto"} else "us_stock"
        net_alpha, cost_pct, cost_ok = self._cost_model.net_alpha(
            alpha_score=float(score["alpha_score"]),
            asset_class=cost_asset_class,
            position_size_pct=1.0,
            daily_vol=market_vol if market_vol > 0 else 0.2,
            volume_ratio=max(volume_ratio, 0.1),
            regime="SIDEWAYS",
            low_liquidity=volume_ratio < 0.8,
        )

        checks = {
            "data_quality_ok": not report.severe,
            "fresh_data_ok": freshness_sec <= freshness_limit,
            "confidence_ok": float(score["confidence"]) >= 55.0,
            "consensus_ok": consensus_ratio >= 0.5 if directional_horizons else False,
            "cost_ok": cost_ok,
        }
        validated = bool(primary_signal in {"BUY", "SELL"} and all(checks.values()))
        validated_signal = primary_signal if validated else "HOLD"
        validation_score = round(sum(1 for ok in checks.values() if ok) / len(checks), 4)

        result = {
            "symbol": asset.symbol,
            "ticker": asset.ticker,
            "name": asset.name,
            "market": asset.market,
            "quote_currency": asset.quote_currency,
            "interval": interval,
            "period": period or self._default_period(interval),
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "last_bar_utc": last_bar_utc.isoformat(),
            "freshness_sec": round(freshness_sec, 1),
            "signal": primary_signal,
            "validated_signal": validated_signal,
            "validated": validated,
            "strength": score["strength"],
            "confidence": round(float(score["confidence"]), 2),
            "alpha_score": round(float(score["alpha_score"]), 4),
            "entry_price": round(entry, 4),
            "stop_loss": round(stop, 4),
            "take_profit": round(take, 4),
            "risk_reward": round(abs((take - entry) / max(abs(entry - stop), 1e-9)), 4),
            "atr_14": round(atr, 4),
            "volatility_20": round(market_vol, 4),
            "volume_ratio": round(volume_ratio, 4),
            "factor_scores": score["factor_scores"],
            "ic_weights": score["ic_weights"],
            "validation": {
                "score": validation_score,
                "checks": checks,
                "consensus_ratio": round(consensus_ratio, 4),
                "horizons": horizons,
                "data_quality": report.to_dict(),
                "cost_pct": round(float(cost_pct), 5),
                "net_alpha_score": round(float(net_alpha), 4),
            },
        }
        self._trade_cache.set(cache_key, result, ttl_seconds=6)
        return result

    def _hold_payload(
        self,
        asset: FocusAsset,
        reason: str,
        interval: str,
        period: str | None,
    ) -> dict[str, Any]:
        return {
            "symbol": asset.symbol,
            "ticker": asset.ticker,
            "name": asset.name,
            "market": asset.market,
            "quote_currency": asset.quote_currency,
            "interval": interval,
            "period": period or self._default_period(interval),
            "signal": "HOLD",
            "validated_signal": "HOLD",
            "validated": False,
            "reason": reason,
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "validation": {
                "score": 0.0,
                "checks": {
                    "data_quality_ok": False,
                    "fresh_data_ok": False,
                    "confidence_ok": False,
                    "consensus_ok": False,
                    "cost_ok": False,
                },
                "horizons": [],
            },
        }

    def _score_frame(self, df: pd.DataFrame, interval: str) -> dict[str, Any] | None:
        timeframe = "daily" if interval == "1d" else "intraday"
        feat = self._engineer.compute_all_features(df, timeframe=timeframe)
        if feat.empty:
            return None
        if len(feat) < self._model.ic_window + 5:
            return None

        fwd_ret = feat["Returns"].shift(-1)
        ml_series = pd.Series(0.0, index=feat.index)
        f1 = self._model._factor1_momentum(feat, hurst=0.5)
        f2 = self._model._factor2_mean_reversion(feat, hurst=0.5)
        f3 = self._model._factor3_volume(feat)
        f4 = self._model._factor4_ml(ml_series)
        f5 = self._model._factor5_volatility_squeeze(feat, momentum_factor=f1)
        f8 = self._model._factor8_microstructure(feat)
        factors = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F8": f8}

        ics: dict[str, float] = {}
        latest: dict[str, float] = {}
        for name, series in factors.items():
            ic_series = _rolling_ic(series, fwd_ret, self._model.ic_window)
            ic = float(ic_series.iloc[-1]) if not ic_series.empty else 0.0
            if not np.isfinite(ic):
                ic = 0.0
            latest_val = float(series.iloc[-1]) if len(series) else 0.0
            if not np.isfinite(latest_val):
                latest_val = 0.0
            ics[name] = ic
            latest[name] = latest_val

        denom = max(sum(abs(v) for v in ics.values()), 0.1)
        alpha_score = float(sum(ics[k] * latest[k] for k in ics) / denom)
        confidence = float(np.clip(50.0 + 50.0 * np.tanh(alpha_score), 0.0, 100.0))

        if alpha_score > self._model.alpha_threshold:
            signal = "BUY"
        elif alpha_score < -self._model.alpha_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        if confidence > 75:
            strength = "STRONG"
        elif confidence > 50:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        atr_14 = float(feat.get("ATR_14", pd.Series([0.0], index=feat.index)).iloc[-1] or 0.0)
        vol_20 = float(feat.get("Volatility_20", pd.Series([0.2], index=feat.index)).iloc[-1] or 0.2)
        vol_ratio = float(feat.get("Volume_Ratio", pd.Series([1.0], index=feat.index)).iloc[-1] or 1.0)

        return {
            "signal": signal,
            "strength": strength,
            "confidence": round(confidence, 2),
            "alpha_score": round(alpha_score, 4),
            "factor_scores": {k: round(v, 4) for k, v in latest.items()},
            "ic_weights": {k: round(v, 4) for k, v in ics.items()},
            "atr_14": max(atr_14, 0.0),
            "volatility_20": max(vol_20, 0.0),
            "volume_ratio": max(vol_ratio, 0.0),
        }

    def _horizon_scores(self, asset: FocusAsset) -> list[dict[str, Any]]:
        horizon_scores: list[dict[str, Any]] = []
        for interval, period in self._horizons:
            try:
                frame = self._fetch_ohlcv(asset.ticker, interval=interval, period=period)
                if frame.empty or len(frame) < 120:
                    continue
                score = self._score_frame(frame, interval)
                if score is None:
                    continue
                horizon_scores.append(
                    {
                        "interval": interval,
                        "period": period,
                        "signal": score["signal"],
                        "confidence": score["confidence"],
                        "alpha_score": score["alpha_score"],
                    }
                )
            except Exception as exc:
                logger.debug("Focus horizon score failed for %s (%s/%s): %s", asset.symbol, interval, period, exc)
        return horizon_scores

    def _default_period(self, interval: str) -> str:
        return str(INTERVAL_DEFAULTS.get(interval, INTERVAL_DEFAULTS["5m"])["period"])

    def _fetch_ohlcv(self, ticker: str, interval: str, period: str | None = None) -> pd.DataFrame:
        use_period = period or self._default_period(interval)
        cache_key = f"focus_ohlcv:{ticker}:{interval}:{use_period}"
        cached = self._ohlcv_cache.get(cache_key)
        if cached is not None:
            return cached

        ttl = 6 if interval in {"1m", "5m", "15m"} else 20
        if not _YF_AVAILABLE:
            logger.warning("yfinance is not installed; focus engine cannot fetch %s", ticker)
            return pd.DataFrame()
        try:
            raw = yf.download(
                ticker,
                interval=interval,
                period=use_period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frame = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            frame = frame.astype(float).dropna(subset=["Open", "High", "Low", "Close"])
            frame.index = pd.to_datetime(frame.index)
            self._ohlcv_cache.set(cache_key, frame, ttl_seconds=ttl)
            return frame
        except Exception as exc:
            logger.warning("Focus data fetch failed for %s (%s/%s): %s", ticker, interval, use_period, exc)
            return pd.DataFrame()
