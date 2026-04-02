"""
WebhookReceiver — processes incoming webhook trading signals.

Security: HMAC-SHA256 token validation before any order routing.
Safety:   process() returns safe dicts; does not raise to callers.
Thread:  Stateless — dependencies injected via __init__.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from src.signals.engine import SLIPPAGE, TradingSignal

logger = logging.getLogger("webhook.receiver")

_MAX_KELLY_FRACTION = 0.05
_MIN_KELLY_FRACTION = 0.01
_DEFAULT_SL_PCT = 0.03
_DEFAULT_TP_PCT = 0.06
_VALID_ACTIONS = frozenset({"BUY", "SELL", "CLOSE"})


def _norm_asset_key(symbol: str) -> str:
    u = (symbol or "").upper().strip()
    for suf in (".NS", ".BO", "=X"):
        if u.endswith(suf):
            u = u[: -len(suf)]
    return u


class WebhookReceiver:
    """
    Processes incoming webhook signals end-to-end.
    Uses broker.execute(TradingSignal); does not use SignalEngine.
    """

    def __init__(
        self,
        secret_token: str,
        broker: Any,
        portfolio_tracker: Any,
        kelly_calculator: Any,
        store: Any,
        notifier: Any,
        max_kelly: float = _MAX_KELLY_FRACTION,
    ) -> None:
        self._secret = (secret_token or "").strip()
        self._broker = broker
        self._portfolio = portfolio_tracker
        self._kelly = kelly_calculator
        self._store = store
        self._notifier = notifier
        self._max_kelly = float(max_kelly)
        self._min_kelly = float(_MIN_KELLY_FRACTION)

    def validate_token(self, received: str) -> bool:
        try:
            if not self._secret:
                return False
            expected = hmac.new(
                self._secret.encode("utf-8"),
                msg=b"tradevex-webhook",
                digestmod=hashlib.sha256,
            ).hexdigest()
            rec = str(received).strip()
            if len(rec) != len(expected):
                return False
            return hmac.compare_digest(expected, rec)
        except Exception as exc:
            logger.warning("Token validation error: %s", exc)
            return False

    def parse_payload(self, raw: dict) -> Tuple[Optional[dict], str]:
        try:
            ticker = str(raw.get("ticker", "")).strip().upper()
            if not ticker:
                return None, "missing_ticker"

            action = str(raw.get("action", "")).strip().upper()
            if action not in _VALID_ACTIONS:
                return None, f"invalid_action '{action}' — must be BUY/SELL/CLOSE"

            price_raw = raw.get("price")
            price = float(price_raw) if price_raw is not None else None

            qty_raw = raw.get("quantity")
            quantity = float(qty_raw) if qty_raw is not None else None

            sl_raw = raw.get("sl")
            sl = float(sl_raw) if sl_raw is not None else None

            tp_raw = raw.get("tp")
            tp = float(tp_raw) if tp_raw is not None else None

            source = str(raw.get("source", "custom")).strip() or "custom"

            return {
                "ticker": ticker,
                "action": action,
                "price": price,
                "quantity": quantity,
                "sl": sl,
                "tp": tp,
                "source": source,
            }, ""
        except Exception as exc:
            return None, f"parse_error: {exc}"

    def check_duplicate(self, ticker: str) -> bool:
        try:
            want = _norm_asset_key(ticker)
            rows = self._portfolio.get_open_positions_list()
            for row in rows:
                if _norm_asset_key(str(row.get("asset", ""))) == want:
                    return True
            return False
        except Exception as exc:
            logger.warning(
                "Duplicate check failed for %s: %s — allowing order",
                ticker,
                exc,
            )
            return False

    def _infer_asset_class(self, ticker: str) -> str:
        t = ticker.upper()
        if t.endswith(".NS") or t.endswith(".BO"):
            return "indian_stock"
        if "USD" in t and ("=" in t or len(t) >= 6):
            return "forex"
        if t.endswith("USDT") or t.endswith("USD") and "BTC" in t:
            return "crypto"
        return "us_stock"

    def _slippage_for(self, asset_class: str, asset: str) -> float:
        if asset_class == "forex":
            key = "forex_major" if asset in {"EURUSD", "GBPUSD"} else "forex_minor"
            return float(SLIPPAGE.get(key, SLIPPAGE["forex"]))
        return float(SLIPPAGE.get(asset_class, SLIPPAGE["us_stock"]))

    def compute_auto_size(self, ticker: str, signal_str: str) -> float:
        """
        Returns position_size_pct as percent of portfolio (e.g. 2.0 = 2%).
        """
        try:
            ac = self._infer_asset_class(ticker)
            bucket = self._store.get_bucket_stats(
                ac,
                "MODERATE",
                "SIDEWAYS",
                signal=signal_str,
            )
            kr = self._kelly.compute(ac, signal_str, "MODERATE", "SIDEWAYS", bucket)
            p = float(kr.get("position_size_pct", self._min_kelly))
            if kr.get("method") == "cold_start":
                p *= 100.0
            p = min(p, self._max_kelly * 100.0)
            p = max(p, self._min_kelly * 100.0)
            return float(p)
        except Exception as exc:
            logger.warning(
                "Auto-size failed for %s: %s — using %.0f%% fallback",
                ticker,
                exc,
                self._min_kelly * 100.0,
            )
            return float(self._min_kelly * 100.0)

    def _manual_quantity_to_pct(
        self,
        quantity: float,
        price: Optional[float],
        ticker: str,
    ) -> float:
        try:
            eq = float(
                getattr(self._portfolio, "mark_to_market_equity", 0.0)
                or getattr(self._portfolio, "initial_capital", 100_000.0)
            )
            eq = max(eq, 1.0)
            if quantity <= 0.25:
                pct = quantity * 100.0
            elif price and quantity >= 1.0 and quantity == int(quantity) and quantity > 25:
                pct = (quantity * float(price)) / eq * 100.0
            else:
                pct = float(quantity)
            pct = min(pct, self._max_kelly * 100.0)
            pct = max(pct, self._min_kelly * 100.0)
            return float(pct)
        except Exception:
            return float(self._min_kelly * 100.0)

    def compute_auto_sltp(
        self,
        ticker: str,
        price: Optional[float],
        action: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        try:
            if price is None or price <= 0:
                return None, None
            act = str(action).upper()
            if act in ("BUY", "LONG"):
                sl = round(float(price) * (1.0 - _DEFAULT_SL_PCT), 4)
                tp = round(float(price) * (1.0 + _DEFAULT_TP_PCT), 4)
            else:
                sl = round(float(price) * (1.0 + _DEFAULT_SL_PCT), 4)
                tp = round(float(price) * (1.0 - _DEFAULT_TP_PCT), 4)
            return sl, tp
        except Exception as exc:
            logger.warning("Auto SL/TP failed for %s: %s", ticker, exc)
            return None, None

    def _build_signal(
        self,
        ticker: str,
        signal_str: str,
        price: float,
        sl: float,
        tp: float,
        position_size_pct: float,
    ) -> TradingSignal:
        ac = self._infer_asset_class(ticker)
        slip = self._slippage_for(ac, ticker)
        risk_pct = (
            abs(price - sl) / max(price, 1e-12) * position_size_pct
            if price > 0
            else 0.0
        )
        return TradingSignal(
            signal_id=str(uuid.uuid4()),
            asset=ticker,
            asset_class=ac,
            timeframe="webhook",
            signal=signal_str,
            strength="MODERATE",
            confidence=60.0,
            alpha_score=0.0,
            regime="SIDEWAYS",
            entry_price=round(float(price), 6),
            stop_loss=round(float(sl), 6),
            take_profit=round(float(tp), 6),
            risk_pct=round(float(risk_pct), 4),
            kelly_fraction=round(float(position_size_pct), 4),
            position_size_pct=round(float(position_size_pct), 4),
            slippage_cost_pct=slip,
        )

    def _receipt_order_id(self, receipt: Any, fallback: str) -> str:
        if isinstance(receipt, dict):
            return str(receipt.get("order_id", fallback))
        return str(getattr(receipt, "order_id", fallback))

    def _receipt_ok(self, receipt: Any) -> bool:
        st = (
            str(receipt.get("status", ""))
            if isinstance(receipt, dict)
            else str(getattr(receipt, "status", ""))
        ).upper()
        return st in {"FILLED", "PARTIAL", "PLACED"}

    def process(self, raw_payload: dict) -> dict:
        _ts = datetime.now(timezone.utc).isoformat()

        received_token = raw_payload.get("secret", "")
        if not self.validate_token(str(received_token)):
            logger.warning("Webhook rejected: invalid token")
            return {"status": "rejected", "reason": "invalid_token", "timestamp": _ts}

        parsed, parse_error = self.parse_payload(raw_payload)
        if parsed is None:
            logger.warning("Webhook rejected: %s", parse_error)
            return {
                "status": "rejected",
                "reason": "invalid_payload",
                "detail": parse_error,
                "timestamp": _ts,
            }

        ticker = parsed["ticker"]
        action = parsed["action"]
        price = parsed["price"]
        quantity = parsed["quantity"]
        sl = parsed["sl"]
        tp = parsed["tp"]
        source = parsed["source"]

        if action == "CLOSE":
            return self._process_close(ticker, price, source, _ts)

        if self.check_duplicate(ticker):
            logger.info("Webhook rejected: open position exists for %s", ticker)
            return {
                "status": "rejected",
                "reason": "duplicate_position",
                "ticker": ticker,
                "timestamp": _ts,
            }

        if price is None or price <= 0:
            return {
                "status": "rejected",
                "reason": "invalid_payload",
                "detail": "missing_or_invalid_price",
                "timestamp": _ts,
            }

        signal_str = "BUY" if action == "BUY" else "SELL"

        auto_sized = quantity is None
        if auto_sized:
            position_size_pct = self.compute_auto_size(ticker, signal_str)
        else:
            position_size_pct = self._manual_quantity_to_pct(
                float(quantity), float(price), ticker
            )

        auto_sltp = sl is None or tp is None
        if sl is None or tp is None:
            _sl, _tp = self.compute_auto_sltp(ticker, price, signal_str)
            if sl is None:
                sl = _sl
            if tp is None:
                tp = _tp

        if sl is None or tp is None:
            return {
                "status": "rejected",
                "reason": "invalid_payload",
                "detail": "could_not_compute_sl_tp",
                "ticker": ticker,
                "timestamp": _ts,
            }

        sig = self._build_signal(ticker, signal_str, float(price), float(sl), float(tp), position_size_pct)

        try:
            receipt = self._broker.execute(sig)
        except Exception as exc:
            logger.error("Webhook broker execution failed for %s: %s", ticker, exc)
            return {
                "status": "rejected",
                "reason": "broker_error",
                "detail": str(exc),
                "ticker": ticker,
                "timestamp": _ts,
            }

        if not self._receipt_ok(receipt):
            err = (
                receipt.get("error", "rejected")
                if isinstance(receipt, dict)
                else getattr(receipt, "error", "rejected")
            )
            return {
                "status": "rejected",
                "reason": "broker_error",
                "detail": str(err),
                "ticker": ticker,
                "timestamp": _ts,
            }

        order_id = self._receipt_order_id(receipt, f"WH_{_ts}")
        fill_px = float(
            receipt.get("fill_price", price)
            if isinstance(receipt, dict)
            else getattr(receipt, "fill_price", price)
        )
        sig.execution_price = fill_px

        response = {
            "status": "success",
            "order_id": order_id,
            "ticker": ticker,
            "action": action,
            "sized_qty": position_size_pct,
            "sl": sl,
            "tp": tp,
            "source": source,
            "auto_sized": auto_sized,
            "auto_sltp": auto_sltp,
            "timestamp": _ts,
        }

        self._finalize_success(sig, receipt, response, source, _ts)
        logger.info(
            "Webhook SUCCESS: %s %s position_pct=%.4f order_id=%s",
            action,
            ticker,
            position_size_pct,
            order_id,
        )
        return response

    def _process_close(self, ticker: str, price: Optional[float], source: str, _ts: str) -> dict:
        try:
            rows = self._portfolio.get_open_positions_list()
            target = None
            for row in rows:
                if _norm_asset_key(str(row.get("asset", ""))) == _norm_asset_key(ticker):
                    target = row
                    break
            if target is None:
                return {
                    "status": "rejected",
                    "reason": "no_open_position",
                    "ticker": ticker,
                    "timestamp": _ts,
                }
            sig_id = str(target["signal_id"])
            exit_px = float(price) if price and price > 0 else float(target.get("current_price") or target.get("entry_price") or 0.0)
            if exit_px <= 0:
                return {
                    "status": "rejected",
                    "reason": "invalid_payload",
                    "detail": "missing_price_for_close",
                    "ticker": ticker,
                    "timestamp": _ts,
                }
            receipt = self._broker.close_position(sig_id, exit_px)
            order_id = self._receipt_order_id(receipt, f"WH_CLOSE_{_ts}")
            try:
                if hasattr(self._portfolio, "close_position_at_market"):
                    self._portfolio.close_position_at_market(sig_id, exit_px)
            except Exception as exc:
                logger.warning("Portfolio close sync failed (non-critical): %s", exc)

            response = {
                "status": "success",
                "order_id": order_id,
                "ticker": ticker,
                "action": "CLOSE",
                "sized_qty": 0.0,
                "sl": None,
                "tp": None,
                "source": source,
                "auto_sized": False,
                "auto_sltp": False,
                "timestamp": _ts,
            }
            self._finalize_success(None, receipt, response, source, _ts, close_mode=True)
            return response
        except Exception as exc:
            logger.error("Webhook CLOSE failed for %s: %s", ticker, exc)
            return {
                "status": "rejected",
                "reason": "broker_error",
                "detail": str(exc),
                "ticker": ticker,
                "timestamp": _ts,
            }

    def _finalize_success(
        self,
        sig: Optional[TradingSignal],
        receipt: Any,
        response: dict,
        source: str,
        _ts: str,
        close_mode: bool = False,
    ) -> None:
        st = str(response.get("action", ""))
        ticker = str(response.get("ticker", ""))
        order_id = str(response.get("order_id", ""))
        try:
            self._store.save_webhook_event(
                {
                    "timestamp": _ts,
                    "source": source,
                    "ticker": ticker,
                    "action": st,
                    "status": "success",
                    "reason": None,
                    "order_id": order_id,
                    "sized_qty": float(response.get("sized_qty") or 0.0),
                    "sl_used": response.get("sl"),
                    "tp_used": response.get("tp"),
                    "auto_sized": 1 if response.get("auto_sized") else 0,
                    "auto_sltp": 1 if response.get("auto_sltp") else 0,
                }
            )
        except Exception as exc:
            logger.warning("Webhook store failed (non-critical): %s", exc)

        if sig is not None and not close_mode:
            try:
                st_r = (
                    str(receipt.get("status", "")).upper()
                    if isinstance(receipt, dict)
                    else str(getattr(receipt, "status", "")).upper()
                )
                if st_r == "FILLED":
                    self._portfolio.open_position(sig)
            except Exception as exc:
                logger.warning("Portfolio open sync failed (non-critical): %s", exc)
            try:
                self._store.save_signal(sig)
            except Exception as exc:
                logger.warning("save_signal after webhook failed (non-critical): %s", exc)

        try:
            qty_disp = float(response.get("sized_qty") or 0.0)
            _msg = (
                f"Webhook {st} {ticker}\n"
                f"Order: {order_id}\n"
                f"Position %: {qty_disp:.4f}\n"
                f"SL: {response.get('sl')} | TP: {response.get('tp')}\n"
                f"Source: {source}"
            )
            if hasattr(self._notifier, "notify"):
                self._notifier.notify("Webhook Order", _msg, severity="INFO")
        except Exception as exc:
            logger.warning("Webhook notification failed (non-critical): %s", exc)
