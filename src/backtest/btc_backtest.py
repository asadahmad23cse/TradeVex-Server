"""BTC-specific walk-forward backtesting with Binance historical data."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

from src.alpha.factor_model import AlphaFactorModel
from src.features.engineer import FeatureEngineer

try:
    from src.risk.kelly_warm_start import BTCKellyWarmStart
except Exception:  # pragma: no cover - optional at runtime
    BTCKellyWarmStart = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

BINANCE_REST = "https://api.binance.com"
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
}


def _parse_utc(date_text: str | None) -> pd.Timestamp:
    if date_text is None:
        return pd.Timestamp.now(tz="UTC")
    ts = pd.Timestamp(date_text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        dd = (value / peak) - 1.0 if peak > 0 else 0.0
        worst = min(worst, dd)
    return float(worst)


class BTCHistoricalLoader:
    """Loads historical BTC candles from Binance with 24h cache."""

    def __init__(self, cache_dir: str = "data", cache_ttl_hours: int = 24, session: requests.Session | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._session = session or requests.Session()

    def _cache_path(self, symbol: str, interval: str) -> Path:
        safe_symbol = "".join(ch for ch in symbol.upper() if ch.isalnum() or ch in {"_", "-"})
        safe_interval = "".join(ch for ch in interval if ch.isalnum())
        return self.cache_dir / f"btc_history_{safe_symbol}_{safe_interval}.parquet"

    def _cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return datetime.now(timezone.utc) - modified < self.cache_ttl

    @staticmethod
    def _read_cached_frame(path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_pickle(path)

    @staticmethod
    def _write_cached_frame(df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_parquet(path)
        except Exception:
            df.to_pickle(path)

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(max(int(limit), 1), 1000),
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        url = f"{BINANCE_REST}/api/v3/klines"
        try:
            response = self._session.get(url, params=params, timeout=12)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Binance klines fetch failed for %s %s: %s", symbol, interval, exc)
            return []

    @staticmethod
    def _rows_to_df(rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        records: list[dict[str, float | pd.Timestamp]] = []
        for row in rows:
            try:
                records.append(
                    {
                        "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                    }
                )
            except Exception:
                continue
        if not records:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out = pd.DataFrame(records).drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
        out.index.name = "timestamp"
        return out[["open", "high", "low", "close", "volume"]]

    def fetch(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = "2023-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        interval = interval if interval in INTERVAL_TO_MS else "1h"
        cache_path = self._cache_path(symbol, interval)

        if self._cache_fresh(cache_path):
            df = self._read_cached_frame(cache_path)
        else:
            start_ts = _parse_utc(start_date)
            end_ts = _parse_utc(end_date)
            step_ms = INTERVAL_TO_MS[interval]
            start_ms = int(start_ts.timestamp() * 1000)
            end_ms = int(end_ts.timestamp() * 1000)

            rows: list[list[Any]] = []
            while start_ms < end_ms:
                batch = self._fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    limit=1000,
                )
                if not batch:
                    break
                rows.extend(batch)
                last_open_ms = int(batch[-1][0])
                next_start = last_open_ms + step_ms
                if next_start <= start_ms:
                    break
                start_ms = next_start
                if len(batch) < 1000:
                    break

            df = self._rows_to_df(rows)
            if len(df) > 1:
                # Last candle is still forming in exchange stream.
                df = df.iloc[:-1]
            self._write_cached_frame(df, cache_path)

        if df.empty:
            logger.info("Loaded 0 candles for %s %s", symbol, interval)
            return df

        start_ts = _parse_utc(start_date)
        end_ts = _parse_utc(end_date)
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        logger.info("Loaded %d candles for %s %s", len(df), symbol, interval)
        return df


class BTCFeatureGenerator:
    """Generates base + BTC proxy features for historical backtesting."""

    def __init__(self) -> None:
        self._engineer = FeatureEngineer()

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std().replace(0.0, np.nan)
        z = (series - mu) / sigma
        return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df.copy()

        work = df.copy().sort_index()
        rename_to_fe = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        fe_in = work.rename(columns=rename_to_fe)
        feat = self._engineer.compute_all_features(fe_in, timeframe="intraday", ticker="BTCUSDT")
        if feat.empty:
            feat = fe_in.copy()

        out = feat.copy()
        out["open"] = out.get("Open", np.nan)
        out["high"] = out.get("High", np.nan)
        out["low"] = out.get("Low", np.nan)
        out["close"] = out.get("Close", np.nan)
        out["volume"] = out.get("Volume", np.nan)

        ret = out["close"].pct_change()
        out["funding_rate_z"] = self._rolling_zscore(ret, window=24)
        out["oi_proxy"] = self._rolling_zscore(out["volume"].pct_change(4), window=24)
        out["etf_flow_proxy"] = (out["volume"].rolling(5).mean() / out["volume"].rolling(30).mean()) - 1.0

        out["funding_rate_z_simulated"] = True
        out["oi_proxy_simulated"] = True
        out["etf_flow_proxy_simulated"] = True
        out["crypto_proxy_note"] = "SIMULATED_FROM_OHLCV"

        out["etf_flow_proxy"] = out["etf_flow_proxy"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out = out.dropna(subset=["open", "high", "low", "close", "volume"])
        return out


class BTCWalkForwardBacktest:
    """Walk-forward BTC backtest with parameter optimization per fold."""

    DEFAULT_ALPHA_THRESHOLDS = [0.15, 0.20, 0.25]
    DEFAULT_IC_WINDOWS = [24, 72, 168]
    DEFAULT_CONFIDENCE_FLOORS = [65, 70]

    def __init__(
        self,
        train_window_days: int = 90,
        test_window_days: int = 30,
        step_days: int = 30,
        min_folds: int = 6,
        alpha_threshold_grid: list[float] | None = None,
        ic_window_grid: list[int] | None = None,
        confidence_floor_grid: list[int] | None = None,
        fee_pct: float = 0.05,
        slippage_pct: float = 0.01,
        config_path: str = "config.yaml",
    ) -> None:
        self.train_window_days = int(train_window_days)
        self.test_window_days = int(test_window_days)
        self.step_days = int(step_days)
        self.min_folds = int(min_folds)
        self.alpha_threshold_grid = list(alpha_threshold_grid or self.DEFAULT_ALPHA_THRESHOLDS)
        self.ic_window_grid = list(ic_window_grid or self.DEFAULT_IC_WINDOWS)
        self.confidence_floor_grid = list(confidence_floor_grid or self.DEFAULT_CONFIDENCE_FLOORS)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.config_path = config_path
        self._features = BTCFeatureGenerator()
        self._kelly = None
        if BTCKellyWarmStart is not None:
            try:
                self._kelly = BTCKellyWarmStart()
            except Exception:
                self._kelly = None

    @staticmethod
    def _bars_per_day(index: pd.DatetimeIndex) -> int:
        if len(index) < 2:
            return 24
        deltas = pd.Series(index[1:]).reset_index(drop=True) - pd.Series(index[:-1]).reset_index(drop=True)
        sec = float(deltas.dt.total_seconds().median())
        if not np.isfinite(sec) or sec <= 0:
            return 24
        return max(int(round(86_400 / sec)), 1)

    @staticmethod
    def _infer_regime(window: pd.DataFrame) -> str:
        if window.empty:
            return "SIDEWAYS"
        close = float(window["close"].iloc[-1])
        ema21 = float(window["close"].ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = float(window["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        if close > ema21 > ema50:
            return "BULLISH TREND"
        if close < ema21 < ema50:
            return "BEARISH TREND"
        return "SIDEWAYS"

    def _signal_from_window(
        self,
        model: AlphaFactorModel,
        window: pd.DataFrame,
        confidence_floor: int,
    ) -> dict[str, Any]:
        row = window.iloc[-1]
        ret = float(row.get("Returns", 0.0))
        alpha = model.compute(
            window,
            asset="BTCUSDT",
            asset_class="crypto",
            funding_rate_z=float(row.get("funding_rate_z", 0.0)),
            oi_delta_1h=float(row.get("oi_proxy", 0.0)),
            price_change_1h=ret * 100.0,
            etf_flow_z=float(row.get("etf_flow_proxy", 0.0)),
        )
        raw_signal = str(alpha.get("signal", "HOLD")).upper()
        signal = "LONG" if raw_signal == "BUY" else "SHORT" if raw_signal == "SELL" else "HOLD"
        confidence = float(alpha.get("confidence", 50.0))
        regime = self._infer_regime(window)

        if signal == "LONG" and regime.startswith("BEARISH"):
            signal = "HOLD"
        if signal == "SHORT" and regime.startswith("BULLISH"):
            signal = "HOLD"
        if signal in {"LONG", "SHORT"} and confidence < float(confidence_floor):
            signal = "HOLD"

        return {
            "signal": signal,
            "confidence": confidence,
            "regime": regime,
            "alpha_score": float(alpha.get("alpha_score", 0.0)),
        }

    def _entry_stop_and_targets(self, side: str, history: pd.DataFrame, entry_price: float) -> dict[str, float]:
        atr = float(history["ATR_14"].iloc[-1]) if "ATR_14" in history.columns else entry_price * 0.01
        atr = max(atr, entry_price * 0.002)
        lookback = history.tail(24)

        if side == "LONG":
            swing = float(lookback["low"].min())
            dist = entry_price - swing if swing > 0 else 0.0
            if dist <= 0 or (dist / entry_price) > 0.03:
                dist = atr * 1.5
            stop = entry_price - dist
            tp1 = entry_price + (dist * 2.0)
            tp2 = entry_price + (dist * 3.0)
            tp3 = entry_price + (dist * 4.0)
        else:
            swing = float(lookback["high"].max())
            dist = swing - entry_price if swing > 0 else 0.0
            if dist <= 0 or (dist / entry_price) > 0.03:
                dist = atr * 1.5
            stop = entry_price + dist
            tp1 = entry_price - (dist * 2.0)
            tp2 = entry_price - (dist * 3.0)
            tp3 = entry_price - (dist * 4.0)

        dist = max(dist, entry_price * 0.003)
        return {
            "stop_loss": float(stop),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "sl_dist": float(dist),
            "trail_gap": float(max(atr * 1.5, entry_price * 0.006)),
        }

    @staticmethod
    def _directional_pnl_pct(side: str, entry: float, exit_price: float) -> float:
        raw = ((exit_price - entry) / entry) * 100.0 if entry > 0 else 0.0
        return raw if side == "LONG" else -raw

    def _finalize_trade(
        self,
        trade: dict[str, Any],
        exit_price: float,
        reason: str,
        timestamp: pd.Timestamp,
    ) -> dict[str, Any]:
        gross = self._directional_pnl_pct(str(trade["side"]), float(trade["entry_price"]), float(exit_price))
        net = gross - self.fee_pct - self.slippage_pct
        out = dict(trade)
        out["exit_time"] = timestamp.isoformat()
        out["exit_price"] = float(exit_price)
        out["exit_reason"] = reason
        out["gross_pnl_pct"] = float(gross)
        out["net_pnl_pct"] = float(net)
        out["win"] = bool(net > 0.0)
        return out

    def _simulate_range(
        self,
        combined: pd.DataFrame,
        active_start: int,
        active_end: int,
        alpha_threshold: float,
        ic_window: int,
        confidence_floor: int,
        bars_per_day: int,
    ) -> dict[str, Any]:
        model = AlphaFactorModel(alpha_threshold=float(alpha_threshold), ic_window=int(ic_window))
        warmup = max(int(ic_window) + 20, 120)
        equity = [1.0]
        active_bar_returns: list[float] = []
        closed_trades: list[dict[str, Any]] = []
        open_trade: dict[str, Any] | None = None
        active_start_eval = max(active_start, warmup)

        for i in range(active_start_eval, len(combined) - 1):
            in_active = active_start <= i < active_end
            if in_active:
                active_bar_returns.append(0.0)

            history = combined.iloc[: i + 1]
            row = combined.iloc[i]
            next_row = combined.iloc[i + 1]
            signal_payload = self._signal_from_window(model, history, confidence_floor=confidence_floor)
            signal_now = str(signal_payload["signal"])
            ts = pd.Timestamp(combined.index[i])

            if open_trade is not None:
                side = str(open_trade["side"])
                high = float(row["high"])
                low = float(row["low"])
                close = float(row["close"])

                if side == "LONG":
                    if not open_trade["tp1_hit"] and high >= float(open_trade["tp1"]):
                        open_trade["tp1_hit"] = True
                        open_trade["active_stop_loss"] = float(open_trade["entry_price"]) * 1.0005
                        open_trade["breakeven_activated"] = True
                    if not open_trade["tp2_hit"] and high >= float(open_trade["tp2"]):
                        open_trade["tp2_hit"] = True
                        open_trade["trailing_active"] = True
                    if not open_trade["tp3_hit"] and high >= float(open_trade["tp3"]):
                        closed = self._finalize_trade(open_trade, float(open_trade["tp3"]), "TP3_HIT", ts)
                        closed_trades.append(closed)
                        ret = (closed["position_size_pct"] / 100.0) * (closed["net_pnl_pct"] / 100.0)
                        equity.append(equity[-1] * (1.0 + ret))
                        if in_active and active_bar_returns:
                            active_bar_returns[-1] += ret
                        open_trade = None
                        continue
                    if open_trade.get("trailing_active"):
                        tr_stop = close - float(open_trade["trail_gap"])
                        open_trade["active_stop_loss"] = max(float(open_trade["active_stop_loss"]), tr_stop)
                    if low <= float(open_trade["active_stop_loss"]):
                        if open_trade.get("trailing_active"):
                            reason = "TRAIL_STOP_HIT"
                        elif open_trade.get("breakeven_activated"):
                            reason = "BREAKEVEN_HIT"
                        else:
                            reason = "SL_HIT"
                        closed = self._finalize_trade(open_trade, float(open_trade["active_stop_loss"]), reason, ts)
                        closed_trades.append(closed)
                        ret = (closed["position_size_pct"] / 100.0) * (closed["net_pnl_pct"] / 100.0)
                        equity.append(equity[-1] * (1.0 + ret))
                        if in_active and active_bar_returns:
                            active_bar_returns[-1] += ret
                        open_trade = None
                        continue
                else:
                    if not open_trade["tp1_hit"] and low <= float(open_trade["tp1"]):
                        open_trade["tp1_hit"] = True
                        open_trade["active_stop_loss"] = float(open_trade["entry_price"]) * 0.9995
                        open_trade["breakeven_activated"] = True
                    if not open_trade["tp2_hit"] and low <= float(open_trade["tp2"]):
                        open_trade["tp2_hit"] = True
                        open_trade["trailing_active"] = True
                    if not open_trade["tp3_hit"] and low <= float(open_trade["tp3"]):
                        closed = self._finalize_trade(open_trade, float(open_trade["tp3"]), "TP3_HIT", ts)
                        closed_trades.append(closed)
                        ret = (closed["position_size_pct"] / 100.0) * (closed["net_pnl_pct"] / 100.0)
                        equity.append(equity[-1] * (1.0 + ret))
                        if in_active and active_bar_returns:
                            active_bar_returns[-1] += ret
                        open_trade = None
                        continue
                    if open_trade.get("trailing_active"):
                        tr_stop = close + float(open_trade["trail_gap"])
                        open_trade["active_stop_loss"] = min(float(open_trade["active_stop_loss"]), tr_stop)
                    if high >= float(open_trade["active_stop_loss"]):
                        if open_trade.get("trailing_active"):
                            reason = "TRAIL_STOP_HIT"
                        elif open_trade.get("breakeven_activated"):
                            reason = "BREAKEVEN_HIT"
                        else:
                            reason = "SL_HIT"
                        closed = self._finalize_trade(open_trade, float(open_trade["active_stop_loss"]), reason, ts)
                        closed_trades.append(closed)
                        ret = (closed["position_size_pct"] / 100.0) * (closed["net_pnl_pct"] / 100.0)
                        equity.append(equity[-1] * (1.0 + ret))
                        if in_active and active_bar_returns:
                            active_bar_returns[-1] += ret
                        open_trade = None
                        continue

                if (open_trade is not None) and signal_now in {"LONG", "SHORT"}:
                    opposite = (side == "LONG" and signal_now == "SHORT") or (side == "SHORT" and signal_now == "LONG")
                    if opposite:
                        open_trade["flip_count"] = int(open_trade.get("flip_count", 0)) + 1
                    else:
                        open_trade["flip_count"] = 0
                    if int(open_trade.get("flip_count", 0)) >= 2:
                        closed = self._finalize_trade(open_trade, close, "ALPHA_FLIP_EXIT", ts)
                        closed_trades.append(closed)
                        ret = (closed["position_size_pct"] / 100.0) * (closed["net_pnl_pct"] / 100.0)
                        equity.append(equity[-1] * (1.0 + ret))
                        if in_active and active_bar_returns:
                            active_bar_returns[-1] += ret
                        open_trade = None
                        continue

            if open_trade is None and in_active and signal_now in {"LONG", "SHORT"}:
                entry_price = float(next_row["open"])
                levels = self._entry_stop_and_targets(signal_now, history, entry_price)
                if self._kelly is not None and getattr(self._kelly, "total_trades", 0) > 0:
                    kelly = self._kelly.compute_btc_position(
                        signal=signal_now,
                        confidence=float(signal_payload["confidence"]),
                        regime=str(signal_payload["regime"]),
                    )
                    pos_size_pct = float(kelly.get("position_size_pct", 2.0))
                else:
                    pos_size_pct = 2.0
                open_trade = {
                    "entry_time": pd.Timestamp(combined.index[i + 1]).isoformat(),
                    "entry_price": entry_price,
                    "side": signal_now,
                    "confidence": float(signal_payload["confidence"]),
                    "regime": str(signal_payload["regime"]),
                    "position_size_pct": float(pos_size_pct),
                    "stop_loss": levels["stop_loss"],
                    "active_stop_loss": levels["stop_loss"],
                    "tp1": levels["tp1"],
                    "tp2": levels["tp2"],
                    "tp3": levels["tp3"],
                    "trail_gap": levels["trail_gap"],
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "tp3_hit": False,
                    "breakeven_activated": False,
                    "trailing_active": False,
                    "flip_count": 0,
                }

            if open_trade is not None and i >= active_end:
                close_px = float(row["close"])
                closed = self._finalize_trade(open_trade, close_px, "WINDOW_END_EXIT", ts)
                closed_trades.append(closed)
                ret = (closed["position_size_pct"] / 100.0) * (closed["net_pnl_pct"] / 100.0)
                equity.append(equity[-1] * (1.0 + ret))
                if in_active and active_bar_returns:
                    active_bar_returns[-1] += ret
                open_trade = None

        total = len(closed_trades)
        wins = [t for t in closed_trades if float(t["net_pnl_pct"]) > 0]
        losses = [t for t in closed_trades if float(t["net_pnl_pct"]) <= 0]
        avg_win = float(np.mean([float(t["net_pnl_pct"]) for t in wins])) if wins else 0.0
        avg_loss = float(abs(np.mean([float(t["net_pnl_pct"]) for t in losses]))) if losses else 0.0
        gross_profit = float(sum(max(float(t["net_pnl_pct"]), 0.0) for t in closed_trades))
        gross_loss = float(abs(sum(min(float(t["net_pnl_pct"]), 0.0) for t in closed_trades)))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        ret_arr = np.array(active_bar_returns, dtype=float)
        annual_factor = float(max(bars_per_day * 365, 1))
        if len(ret_arr) > 1 and np.std(ret_arr) > 1e-12:
            sharpe = float(np.mean(ret_arr) / np.std(ret_arr) * math.sqrt(annual_factor))
            info_ratio = float(np.mean(ret_arr) / np.std(ret_arr) * math.sqrt(annual_factor))
        else:
            sharpe = 0.0
            info_ratio = 0.0

        max_dd = _max_drawdown(equity)
        periods = max(len(active_bar_returns), 1)
        annual_return = (equity[-1] ** (annual_factor / periods)) - 1.0 if equity else 0.0
        calmar = (annual_return / abs(max_dd)) if max_dd < 0 else 0.0

        return {
            "trades": closed_trades,
            "total_trades": int(total),
            "win_rate": float((len(wins) / total) * 100.0) if total > 0 else 0.0,
            "avg_win_pct": float(avg_win),
            "avg_loss_pct": float(avg_loss),
            "profit_factor": float(profit_factor),
            "sharpe": float(sharpe),
            "max_drawdown_pct": float(abs(max_dd) * 100.0),
            "calmar": float(calmar),
            "information_ratio": float(info_ratio),
            "gross_profit_pct": float(gross_profit),
            "gross_loss_pct": float(gross_loss),
            "equity_end": float(equity[-1]) if equity else 1.0,
        }

    def _optimize_fold(self, train_df: pd.DataFrame, bars_per_day: int) -> tuple[dict[str, Any], dict[str, Any]]:
        best_params: dict[str, Any] | None = None
        best_metrics: dict[str, Any] | None = None
        best_score = -1e18
        active_start = max(int(0.2 * len(train_df)), 120)
        total = len(self.alpha_threshold_grid) * len(self.ic_window_grid) * len(self.confidence_floor_grid)
        combo_idx = 0
        for alpha_threshold in self.alpha_threshold_grid:
            for ic_window in self.ic_window_grid:
                for conf_floor in self.confidence_floor_grid:
                    combo_idx += 1
                    print(
                        f"Testing combo {combo_idx}/{total}: "
                        f"alpha={alpha_threshold} ic={ic_window} conf={conf_floor}"
                    )
                    metrics = self._simulate_range(
                        combined=train_df,
                        active_start=active_start,
                        active_end=len(train_df) - 1,
                        alpha_threshold=float(alpha_threshold),
                        ic_window=int(ic_window),
                        confidence_floor=int(conf_floor),
                        bars_per_day=bars_per_day,
                    )
                    score = float(metrics["sharpe"]) + (0.25 * float(metrics["profit_factor"]))
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "alpha_threshold": float(alpha_threshold),
                            "ic_window": int(ic_window),
                            "confidence_floor": int(conf_floor),
                        }
                        best_metrics = metrics

        return best_params or {}, best_metrics or {}

    @staticmethod
    def _parameter_stability(folds: list[dict[str, Any]]) -> dict[str, float]:
        if not folds:
            return {
                "std_alpha_threshold": 0.0,
                "std_ic_window": 0.0,
                "std_confidence_floor": 0.0,
                "parameter_stability_score": 1.0,
            }
        alpha_vals = [float(f["best_params"]["alpha_threshold"]) for f in folds]
        ic_vals = [float(f["best_params"]["ic_window"]) for f in folds]
        conf_vals = [float(f["best_params"]["confidence_floor"]) for f in folds]
        std_alpha = float(np.std(alpha_vals))
        std_ic = float(np.std(ic_vals))
        std_conf = float(np.std(conf_vals))
        normalized = np.mean([std_alpha / 0.20, std_ic / 144.0, std_conf / 15.0])
        stability_score = float(np.clip(1.0 - normalized, 0.0, 1.0))
        return {
            "std_alpha_threshold": std_alpha,
            "std_ic_window": std_ic,
            "std_confidence_floor": std_conf,
            "parameter_stability_score": stability_score,
        }

    def run(self, df: pd.DataFrame, symbol: str = "BTCUSDT", interval: str = "1h") -> dict[str, Any]:
        features = self._features.generate(df)
        if features.empty:
            return {"error": "No features generated", "passed": False}

        bars_per_day = self._bars_per_day(pd.DatetimeIndex(features.index))
        train_bars = self.train_window_days * bars_per_day
        test_bars = self.test_window_days * bars_per_day
        step_bars = self.step_days * bars_per_day

        fold_starts: list[int] = []
        cursor = 0
        while cursor + train_bars + test_bars <= len(features):
            fold_starts.append(cursor)
            cursor += step_bars

        if len(fold_starts) < self.min_folds:
            return {
                "error": f"Insufficient data for {self.min_folds} folds",
                "available_folds": len(fold_starts),
                "passed": False,
            }
        # Bound runtime by evaluating only the requested number of walk-forward folds.
        fold_starts = fold_starts[: self.min_folds]

        fold_results: list[dict[str, Any]] = []
        for fold_idx, start_idx in enumerate(fold_starts, start=1):
            train = features.iloc[start_idx: start_idx + train_bars].copy()
            test = features.iloc[start_idx + train_bars: start_idx + train_bars + test_bars].copy()
            best_params, _ = self._optimize_fold(train, bars_per_day=bars_per_day)
            combined = pd.concat([train, test], axis=0)
            metrics = self._simulate_range(
                combined=combined,
                active_start=len(train),
                active_end=len(combined) - 1,
                alpha_threshold=float(best_params["alpha_threshold"]),
                ic_window=int(best_params["ic_window"]),
                confidence_floor=int(best_params["confidence_floor"]),
                bars_per_day=bars_per_day,
            )
            fold_results.append(
                {
                    "fold": fold_idx,
                    "train_start": str(train.index[0]),
                    "train_end": str(train.index[-1]),
                    "test_start": str(test.index[0]),
                    "test_end": str(test.index[-1]),
                    "test_days": int(self.test_window_days),
                    "best_params": best_params,
                    "metrics": metrics,
                }
            )

        total_trades = int(sum(int(f["metrics"]["total_trades"]) for f in fold_results))
        total_gross_profit = float(sum(float(f["metrics"]["gross_profit_pct"]) for f in fold_results))
        total_gross_loss = float(sum(float(f["metrics"]["gross_loss_pct"]) for f in fold_results))
        aggregate_pf = (total_gross_profit / total_gross_loss) if total_gross_loss > 0 else 0.0
        mean_sharpe = float(np.mean([float(f["metrics"]["sharpe"]) for f in fold_results]))
        mean_win_rate = float(np.mean([float(f["metrics"]["win_rate"]) for f in fold_results]))
        max_dd = float(max(float(f["metrics"]["max_drawdown_pct"]) for f in fold_results))
        best_fold = max(fold_results, key=lambda f: float(f["metrics"]["sharpe"]))
        worst_fold = min(fold_results, key=lambda f: float(f["metrics"]["sharpe"]))

        param_counter = Counter(
            (
                float(f["best_params"]["alpha_threshold"]),
                int(f["best_params"]["ic_window"]),
                int(f["best_params"]["confidence_floor"]),
            )
            for f in fold_results
        )
        mode_params = param_counter.most_common(1)[0][0]
        best_params = {
            "alpha_threshold": mode_params[0],
            "ic_window": mode_params[1],
            "confidence_floor": mode_params[2],
        }

        stability = self._parameter_stability(fold_results)
        passed = all(
            [
                mean_sharpe > 0.5,
                mean_win_rate > 48.0,
                aggregate_pf > 1.2,
                max_dd < 25.0,
                total_trades >= 30,
            ]
        )

        return {
            "symbol": symbol,
            "interval": interval,
            "period_start": str(features.index[0]),
            "period_end": str(features.index[-1]),
            "train_window_days": self.train_window_days,
            "test_window_days": self.test_window_days,
            "step_days": self.step_days,
            "total_folds": len(fold_results),
            "folds": fold_results,
            "aggregate": {
                "mean_oos_sharpe": mean_sharpe,
                "mean_oos_win_rate": mean_win_rate,
                "profit_factor": aggregate_pf,
                "max_drawdown_pct": max_dd,
                "total_trades": total_trades,
                "best_fold": int(best_fold["fold"]),
                "worst_fold": int(worst_fold["fold"]),
                **stability,
            },
            "best_params": best_params,
            "pass_criteria": {
                "oos_sharpe_gt_0_5": bool(mean_sharpe > 0.5),
                "win_rate_gt_48": bool(mean_win_rate > 48.0),
                "profit_factor_gt_1_2": bool(aggregate_pf > 1.2),
                "max_drawdown_lt_25": bool(max_dd < 25.0),
                "trades_gte_30": bool(total_trades >= 30),
            },
            "passed": bool(passed),
            "config_path": self.config_path,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }


class BTCBacktestReport:
    """Persists and prints BTC walk-forward backtest reports."""

    def __init__(self, output_dir: str = "data", config_path: str = "config.yaml") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)

    def _save_report(self, results: dict[str, Any]) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"btc_backtest_report_{ts}.json"
        path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        return path

    def _update_config(self, best_params: dict[str, Any]) -> None:
        if not self.config_path.exists():
            return
        try:
            cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            cfg.setdefault("signal", {})
            cfg["signal"]["alpha_score_threshold"] = float(best_params.get("alpha_threshold", 0.20))
            cfg["signal"]["ic_window"] = int(best_params.get("ic_window", 72))
            cfg.setdefault("btc", {})
            cfg["btc"]["confidence_floor"] = int(best_params.get("confidence_floor", 65))
            self.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to update config with BTC best params: %s", exc)

    def generate(self, results: dict) -> None:
        report_path = self._save_report(results)
        best_params = results.get("best_params", {})
        agg = results.get("aggregate", {})
        status = "PASSED" if bool(results.get("passed", False)) else "FAILED"
        status_icon = "OK" if status == "PASSED" else "NO"

        line = "+" + "-" * 38 + "+"
        print("\n" + line)
        print("|   BTC Walk-Forward Backtest Report   |")
        print(line)
        print(f"| Period:      {str(results.get('period_start', '--'))[:10]} -> {str(results.get('period_end', '--'))[:10]} |")
        print(f"| Interval:    {results.get('interval', '--'):<24}|")
        print(f"| Total Folds: {int(results.get('total_folds', 0)):<24}|")
        print(f"| Total Trades: {int(agg.get('total_trades', 0)):<23}|")
        print(line)
        print("| PERFORMANCE                          |")
        print(f"| Win Rate:     {float(agg.get('mean_oos_win_rate', 0.0)):>6.1f}%              |")
        print(f"| Profit Factor: {float(agg.get('profit_factor', 0.0)):>6.2f}              |")
        print(f"| OOS Sharpe:   {float(agg.get('mean_oos_sharpe', 0.0)):>6.2f}              |")
        print(f"| Max Drawdown: -{float(agg.get('max_drawdown_pct', 0.0)):>5.1f}%             |")
        print(line)
        print("| BEST PARAMS                          |")
        print(f"| alpha_threshold: {float(best_params.get('alpha_threshold', 0.0)):<12.2f} |")
        print(f"| ic_window:       {int(best_params.get('ic_window', 0)):<12d} |")
        print(f"| confidence_floor:{int(best_params.get('confidence_floor', 0)):<12d} |")
        print(line)
        print(f"| STATUS: [{status_icon}] {status:<22}|")
        print(line)
        print(f"Saved report -> {report_path}")

        if best_params:
            self._update_config(best_params)
