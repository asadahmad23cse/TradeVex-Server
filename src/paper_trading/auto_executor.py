"""Auto executor for paper trading signals."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
import urllib3

from src.paper_trading.paper_engine import PaperTradingEngine

logger = logging.getLogger(__name__)
SUPPORTED_CRYPTO_TICKERS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


class AutoExecutor:
    """
    Watches for new signals and auto-executes in paper mode.
    Runs as background thread when auto mode enabled.
    """

    def __init__(self, paper_engine: PaperTradingEngine, notifier=None):
        self.engine = paper_engine
        self.notifier = notifier
        self._running = False
        self._last_signals: dict[str, float] = {}
        self._pending_signals: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.min_confidence = 65
        self.min_sqs = 60

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        while self._running:
            try:
                self._check_and_execute()
            except Exception as e:
                logger.warning("AutoExecutor cycle failed: %s", e)
            for _ in range(60):
                if not self._running:
                    break
                time.sleep(1)

    def stop(self) -> None:
        self._running = False

    def _check_and_execute(self) -> None:
        tickers = self._watchlist_tickers()
        # 1) update open positions with latest prices
        prices: dict[str, float] = {}
        for pos in self.engine.get_open_positions():
            ticker = str(pos.get("ticker", "")).upper()
            px = self._fetch_last_price(ticker)
            if px > 0:
                prices[ticker] = px
        closed = self.engine.update_prices(prices)
        for c in closed:
            self._notify("Paper auto close", json.dumps(c, ensure_ascii=False))

        # 2) check new signals
        for ticker in tickers:
            if not self._market_is_open(ticker):
                continue
            sig = self._fetch_signal(ticker)
            if not sig:
                continue
            self._try_execute(ticker, sig)

        # 3) remove expired pending signals
        now = time.time()
        with self._lock:
            self._pending_signals = [x for x in self._pending_signals if self._to_float(x.get("expires_ts"), 0.0) > now]

    def _watchlist_tickers(self) -> list[str]:
        try:
            cfg = {}
            cfg_path = Path("config.yaml")
            if cfg_path.exists():
                import yaml

                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            watch = cfg.get("watchlist", {}) if isinstance(cfg, dict) else {}
            out: list[str] = []
            for item in watch.get("indian_stocks", []):
                symbol = str(item.get("yf_ticker") or item.get("symbol") or "").upper().strip()
                if symbol:
                    out.append(symbol)
            for item in watch.get("us_stocks", []):
                symbol = str(item.get("yf_ticker") or item.get("symbol") or "").upper().strip()
                if symbol:
                    out.append(symbol)
            for item in watch.get("crypto", []):
                symbol = str(item.get("binance_ticker") or item.get("symbol") or "").upper().strip()
                if symbol in SUPPORTED_CRYPTO_TICKERS:
                    out.append(symbol)
            if not out:
                out = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "MSFT", "BTCUSDT"]
            seen = set()
            dedup = []
            for t in out:
                if t not in seen:
                    seen.add(t)
                    dedup.append(t)
            return dedup
        except Exception:
            return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "MSFT", "BTCUSDT"]

    def _fetch_signal(self, ticker: str) -> dict | None:
        t = str(ticker).upper().strip()
        try:
            if t in {"BTC", "BTCUSDT"}:
                from src.dashboard.btc_service import BitcoinMarketService

                svc = BitcoinMarketService({})
                raw = svc.get_realtime_signal(interval="15m")
                sig = str(raw.get("validated_signal") or raw.get("signal") or "HOLD").upper()
                return {
                    "ticker": "BTCUSDT",
                    "signal": sig,
                    "entry_price": raw.get("entry") or raw.get("entry_price"),
                    "stop_loss": raw.get("stop_loss"),
                    "take_profit": raw.get("take_profit") or raw.get("tp2") or raw.get("tp1"),
                    "confidence": raw.get("confidence", 0),
                    "alpha_score": raw.get("alpha_score", raw.get("confidence", 0)),
                    "regime": raw.get("regime"),
                    "asset_class": "crypto",
                    "strength": raw.get("strength", ""),
                    "sqs": raw.get("sqs"),
                    "position_sizing": raw.get("position_sizing"),
                    "meta_controls": raw.get("meta_controls"),
                    "last_calibration_timestamp": raw.get("last_calibration_timestamp"),
                }

            if t in {"ETH", "ETHUSDT", "SOL", "SOLUSDT"}:
                from src.dashboard.altcoin_service import AltcoinMarketService

                sym = {"ETH": "ETHUSDT", "SOL": "SOLUSDT"}.get(t, t)
                svc = AltcoinMarketService({})
                raw = svc.get_realtime_signal(symbol=sym, interval="15m")
                sig = str(raw.get("validated_signal") or raw.get("signal") or "HOLD").upper()
                validation = raw.get("validation") if isinstance(raw.get("validation"), dict) else {}
                validation_score = self._to_float(validation.get("score"), 0.0) * 100.0
                return {
                    "ticker": sym,
                    "signal": sig,
                    "entry_price": raw.get("entry") or raw.get("entry_price"),
                    "stop_loss": raw.get("stop_loss"),
                    "take_profit": raw.get("take_profit") or raw.get("tp2") or raw.get("tp1"),
                    "confidence": raw.get("confidence", 0),
                    "alpha_score": raw.get("alpha_score", raw.get("confidence", 0)),
                    "regime": raw.get("regime"),
                    "asset_class": "crypto",
                    "strength": raw.get("strength", ""),
                    "sqs": validation_score or raw.get("sqs") or raw.get("confidence", 0),
                    "position_sizing": raw.get("position_sizing"),
                    "meta_controls": raw.get("meta_controls"),
                    "last_calibration_timestamp": raw.get("last_calibration_timestamp"),
                }

            if t.endswith("USDT"):
                return None

            from src.dashboard.api import get_stock_signal

            raw = get_stock_signal(t)
            if not isinstance(raw, dict):
                return None
            raw["ticker"] = t
            raw["asset_class"] = raw.get("asset_class") or self._asset_class(t)
            return raw
        except Exception as e:
            logger.debug("Signal fetch failed for %s: %s", t, e)
            return None

    def _fetch_last_price(self, ticker: str) -> float:
        t = str(ticker).upper().strip()
        try:
            if t in {"BTC", "BTCUSDT"}:
                r = self._public_binance_get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": "BTCUSDT"},
                    timeout=8,
                )
                r.raise_for_status()
                return self._to_float(r.json().get("price"), 0.0)
            if t in {"ETH", "ETHUSDT", "SOL", "SOLUSDT"}:
                sym = {"ETH": "ETHUSDT", "SOL": "SOLUSDT"}.get(t, t)
                r = self._public_binance_get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": sym},
                    timeout=8,
                )
                r.raise_for_status()
                return self._to_float(r.json().get("price"), 0.0)

            import yfinance as yf

            yf_t = t if t.endswith((".NS", ".BO")) or t.startswith("^") or t.endswith("=X") else t
            h = yf.Ticker(yf_t).history(period="1d", interval="1m")
            if not h.empty:
                return self._to_float(h["Close"].iloc[-1], 0.0)
        except Exception:
            return 0.0
        return 0.0

    @staticmethod
    def _public_binance_get(url: str, **kwargs: Any) -> requests.Response:
        try:
            return requests.get(url, **kwargs)
        except requests.exceptions.SSLError:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return requests.get(url, verify=False, **kwargs)

    def _is_duplicate(self, ticker: str, direction: str) -> bool:
        key = f"{ticker}:{direction}"
        ts = self._last_signals.get(key, 0.0)
        return (time.time() - ts) < 4 * 3600

    def _record_signal_ts(self, ticker: str, direction: str) -> None:
        self._last_signals[f"{ticker}:{direction}"] = time.time()

    def _try_execute(self, ticker: str, signal: dict) -> dict | None:
        t = str(signal.get("ticker") or ticker or "").upper().strip()
        direction = self._normalize_direction(signal.get("signal"))
        if direction not in {"LONG", "SHORT"}:
            return None
        if not self._market_is_open(t):
            return None

        confidence = self._to_float(signal.get("confidence"), 0.0)
        sqs = self._to_float(signal.get("sqs"), 100.0)
        if confidence < self.min_confidence:
            return None
        if sqs < self.min_sqs:
            return None
        if self._is_duplicate(t, direction):
            return None

        payload = {
            "ticker": t,
            "signal": direction,
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "confidence": confidence,
            "alpha_score": signal.get("alpha_score", confidence),
            "regime": signal.get("regime"),
            "asset_class": signal.get("asset_class") or self._asset_class(t),
            "strength": signal.get("strength", ""),
            "position_sizing": signal.get("position_sizing"),
            "meta_controls": signal.get("meta_controls"),
            "last_calibration_timestamp": signal.get("last_calibration_timestamp"),
        }

        if str(self.engine.mode).lower() == "auto":
            result = self.engine.execute_trade(payload, mode="auto")
            if result.get("success"):
                self._record_signal_ts(t, direction)
                self._notify(
                    "Paper auto execute",
                    f"{direction} {t} @ {payload.get('entry_price')} conf={confidence:.1f}",
                )
            return result

        with self._lock:
            exists = any(x.get("ticker") == t and x.get("signal") == direction for x in self._pending_signals)
            if exists:
                return None
            pending = {
                "trade_id": self.engine._generate_trade_id(),  # intentionally reused id format
                "ticker": t,
                "signal": direction,
                "confidence": confidence,
                "alpha_score": payload.get("alpha_score", confidence),
                "regime": payload.get("regime"),
                "entry_price": self._to_float(payload.get("entry_price"), 0.0),
                "stop_loss": self._to_float(payload.get("stop_loss"), 0.0),
                "take_profit": self._to_float(payload.get("take_profit"), 0.0),
                "asset_class": payload.get("asset_class"),
                "strength": payload.get("strength"),
                "position_sizing": payload.get("position_sizing"),
                "meta_controls": payload.get("meta_controls"),
                "last_calibration_timestamp": payload.get("last_calibration_timestamp"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_ts": time.time() + 60 * 60,
            }
            self._pending_signals.append(pending)
        return {"success": True, "queued": True, "ticker": t, "trade_id": pending["trade_id"]}

    def get_pending_signals(self) -> list:
        now = time.time()
        with self._lock:
            self._pending_signals = [x for x in self._pending_signals if self._to_float(x.get("expires_ts"), 0.0) > now]
            out = list(self._pending_signals)
        out.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        for row in out:
            row["expires_at"] = datetime.fromtimestamp(self._to_float(row.get("expires_ts"), 0.0), tz=timezone.utc).isoformat()
        return out

    def approve_signal(self, ticker: str, trade_id: str) -> dict:
        t = str(ticker or "").upper().strip()
        tid = str(trade_id or "").strip()
        with self._lock:
            idx = -1
            row = None
            for i, p in enumerate(self._pending_signals):
                if str(p.get("ticker")) == t and str(p.get("trade_id")) == tid:
                    idx = i
                    row = p
                    break
            if row is None:
                return {"success": False, "message": "Pending signal not found"}
            payload = {
                "ticker": row.get("ticker"),
                "signal": row.get("signal"),
                "entry_price": row.get("entry_price"),
                "stop_loss": row.get("stop_loss"),
                "take_profit": row.get("take_profit"),
                "confidence": row.get("confidence"),
                "alpha_score": row.get("alpha_score"),
                "regime": row.get("regime"),
                "asset_class": row.get("asset_class"),
                "strength": row.get("strength"),
                "position_sizing": row.get("position_sizing"),
                "meta_controls": row.get("meta_controls"),
                "last_calibration_timestamp": row.get("last_calibration_timestamp"),
            }
            result = self.engine.execute_trade(payload, mode="manual")
            if result.get("success"):
                self._pending_signals.pop(idx)
                self._record_signal_ts(str(row.get("ticker")), str(row.get("signal")))
            return result

    def reject_signal(self, ticker: str, trade_id: str) -> None:
        t = str(ticker or "").upper().strip()
        tid = str(trade_id or "").strip()
        with self._lock:
            self._pending_signals = [
                p for p in self._pending_signals if not (str(p.get("ticker")) == t and str(p.get("trade_id")) == tid)
            ]

    def _notify(self, subject: str, message: str) -> None:
        try:
            if self.notifier and hasattr(self.notifier, "notify"):
                self.notifier.notify(subject, message, severity="INFO")
        except Exception:
            pass

    @staticmethod
    def _asset_class(ticker: str) -> str:
        t = str(ticker).upper()
        if t.endswith((".NS", ".BO")):
            return "indian_stock"
        if t in {"BTC", "ETH", "SOL"}:
            t = f"{t}USDT"
        if t in SUPPORTED_CRYPTO_TICKERS:
            return "crypto"
        return "us_stock"

    @staticmethod
    def _market_is_open(ticker: str, now_utc: datetime | None = None) -> bool:
        t = str(ticker or "").upper().strip()
        if t in {"BTC", "ETH", "SOL"}:
            t = f"{t}USDT"
        if t in SUPPORTED_CRYPTO_TICKERS:
            return True
        now = now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if t.endswith((".NS", ".BO")):
            local = now.astimezone(ZoneInfo("Asia/Kolkata"))
            if local.weekday() >= 5:
                return False
            minutes = local.hour * 60 + local.minute
            return (9 * 60 + 15) <= minutes <= (15 * 60 + 30)
        local = now.astimezone(ZoneInfo("America/New_York"))
        if local.weekday() >= 5:
            return False
        minutes = local.hour * 60 + local.minute
        return (9 * 60 + 30) <= minutes <= (16 * 60)

    @staticmethod
    def _normalize_direction(signal: Any) -> str:
        s = str(signal or "").upper().strip()
        if s in {"BUY", "LONG"}:
            return "LONG"
        if s in {"SELL", "SHORT"}:
            return "SHORT"
        return "HOLD"

    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default
