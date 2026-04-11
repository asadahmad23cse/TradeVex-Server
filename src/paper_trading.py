"""Lightweight paper-trading and auto-execution support for the dashboard."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILE = Path("data/paper_trading_state.json")
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_state(capital: float = 100_000.0) -> dict[str, Any]:
    initial = round(float(capital), 2)
    return {
        "initial_capital": initial,
        "cash": initial,
        "mode": "manual",
        "open_positions": [],
        "closed_trades": [],
        "equity_curve": [],
    }


class PaperTradingEngine:
    def __init__(self, capital: float = 100_000.0) -> None:
        self._lock = threading.RLock()
        self._state = _default_state(capital)
        self._load()

    def _load(self) -> None:
        if not STATE_FILE.exists():
            self._save()
            return
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                state = _default_state(float(raw.get("initial_capital", 100_000.0)))
                state.update(raw)
                state["open_positions"] = list(state.get("open_positions") or [])
                state["closed_trades"] = list(state.get("closed_trades") or [])
                state["equity_curve"] = list(state.get("equity_curve") or [])
                self._state = state
        except Exception as exc:
            logger.warning("Paper trading state load failed, resetting: %s", exc)
            self._state = _default_state()
            self._save()

    def _save(self) -> None:
        STATE_FILE.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_signal(value: Any) -> str:
        txt = str(value or "").strip().upper()
        if txt in {"BUY", "LONG"}:
            return "LONG"
        if txt in {"SELL", "SHORT"}:
            return "SHORT"
        return "HOLD"

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._state["mode"] = "auto" if str(mode).lower() == "auto" else "manual"
            self._save()

    def reset(self, capital: float = 100_000.0) -> None:
        with self._lock:
            self._state = _default_state(capital)
            self._save()

    def get_open_positions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(pos) for pos in self._state.get("open_positions", [])]

    def get_closed_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._state.get("closed_trades", []))
            return rows[-max(1, int(limit)) :]

    def _compute_metrics(self) -> dict[str, Any]:
        closed = list(self._state.get("closed_trades", []))
        open_positions = list(self._state.get("open_positions", []))
        realized = round(sum(float(t.get("pnl", 0.0) or 0.0) for t in closed), 2)
        unrealized = round(sum(float(p.get("unrealized_pnl", 0.0) or 0.0) for p in open_positions), 2)
        initial = float(self._state.get("initial_capital", 100_000.0) or 100_000.0)
        cash = float(self._state.get("cash", initial) or initial)
        reserved = sum(float(p.get("capital_allocated", 0.0) or 0.0) for p in open_positions)
        portfolio_value = round(cash + reserved + unrealized, 2)
        total_pnl = round(portfolio_value - initial, 2)
        total_pnl_pct = round((total_pnl / initial) * 100.0, 2) if initial > 0 else 0.0
        wins = [t for t in closed if float(t.get("pnl", 0.0) or 0.0) > 0]
        losses = [t for t in closed if float(t.get("pnl", 0.0) or 0.0) <= 0]
        win_rate = round((len(wins) / len(closed)) * 100.0, 1) if closed else 0.0
        avg_win_pct = round(sum(float(t.get("pnl_pct", 0.0) or 0.0) for t in wins) / len(wins), 2) if wins else 0.0
        avg_loss_pct = round(sum(float(t.get("pnl_pct", 0.0) or 0.0) for t in losses) / len(losses), 2) if losses else 0.0
        gross_profit = sum(max(float(t.get("pnl", 0.0) or 0.0), 0.0) for t in closed)
        gross_loss = abs(sum(min(float(t.get("pnl", 0.0) or 0.0), 0.0) for t in closed))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        best_trade_pct = max((float(t.get("pnl_pct", 0.0) or 0.0) for t in closed), default=0.0)
        worst_trade_pct = min((float(t.get("pnl_pct", 0.0) or 0.0) for t in closed), default=0.0)
        peak = initial
        max_drawdown = 0.0
        curve = self._state.get("equity_curve", []) or []
        for point in curve:
            equity = float(point.get("equity", initial) or initial)
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, ((peak - equity) / peak) * 100.0)
        return {
            "capital": round(cash, 2),
            "initial_capital": round(initial, 2),
            "portfolio_value": portfolio_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "total_trades": len(closed),
            "open_positions": len(open_positions),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "sharpe_ratio": 0.0,
            "max_drawdown": round(max_drawdown, 2),
            "profit_factor": profit_factor,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "best_trade_pct": round(best_trade_pct, 2),
            "worst_trade_pct": round(worst_trade_pct, 2),
            "mode": str(self._state.get("mode", "manual")),
        }

    def get_portfolio_metrics(self) -> dict[str, Any]:
        with self._lock:
            return self._compute_metrics()

    def execute_trade(self, payload: dict[str, Any], mode: str = "manual") -> dict[str, Any]:
        with self._lock:
            ticker = str(payload.get("ticker") or payload.get("asset") or "").strip().upper()
            # Accept both legacy `signal` and UI `direction` payload keys.
            signal = self._normalize_signal(payload.get("signal", payload.get("direction")))
            entry_price = float(payload.get("entry_price") or 0.0)
            stop_loss = float(payload.get("stop_loss") or 0.0)
            take_profit = float(payload.get("take_profit") or payload.get("tp1") or 0.0)
            if not ticker:
                return {"success": False, "error": "Ticker is required"}
            if signal not in {"LONG", "SHORT"}:
                return {"success": False, "error": "Signal must be LONG/SHORT"}
            if entry_price <= 0:
                return {"success": False, "error": "Entry price must be positive"}
            if any(str(pos.get("ticker", "")).upper() == ticker for pos in self._state.get("open_positions", [])):
                return {"success": False, "error": f"Position already open for {ticker}"}

            cash = float(self._state.get("cash", 0.0) or 0.0)
            confidence = float(payload.get("confidence", 60.0) or 60.0)
            requested_pct = float(payload.get("position_size_pct") or max(1.0, min((confidence - 40.0) / 10.0, 5.0)))
            requested_pct = max(0.5, min(requested_pct, 10.0))
            capital_allocated = round(min(cash * (requested_pct / 100.0), cash), 2)
            if capital_allocated <= 0:
                return {"success": False, "error": "No cash available"}
            quantity = round(capital_allocated / entry_price, 6)
            trade_id = str(payload.get("trade_id") or uuid.uuid4().hex[:12])
            position = {
                "trade_id": trade_id,
                "ticker": ticker,
                "direction": signal,
                "entry_price": round(entry_price, 2),
                "current_price": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2) if stop_loss > 0 else None,
                "take_profit": round(take_profit, 2) if take_profit > 0 else None,
                "confidence": round(confidence, 2),
                "asset_class": str(payload.get("asset_class", "crypto")),
                "quantity": quantity,
                "capital_allocated": capital_allocated,
                "position_size_pct": round(requested_pct, 2),
                "source_mode": str(mode),
                "opened_at": str(payload.get("as_of_utc") or _now_iso()),
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
            }
            self._state["cash"] = round(cash - capital_allocated, 2)
            self._state.setdefault("open_positions", []).append(position)
            self._state.setdefault("equity_curve", []).append({"time": _now_iso(), "equity": self._compute_metrics()["portfolio_value"]})
            self._save()
            return {
                "success": True,
                "trade_id": trade_id,
                "ticker": ticker,
                "direction": signal,
                "entry_price": position["entry_price"],
                "quantity": quantity,
                "stop_loss": position["stop_loss"],
                "take_profit": position["take_profit"],
                "capital_allocated": capital_allocated,
                "position_size_pct": position["position_size_pct"],
                "mode": self._state.get("mode", "manual"),
            }

    def close_position(self, ticker: str, exit_price: float, reason: str = "manual") -> dict[str, Any] | None:
        with self._lock:
            symbol = str(ticker or "").strip().upper()
            if exit_price <= 0:
                return None
            positions = list(self._state.get("open_positions", []))
            match = next((p for p in positions if str(p.get("ticker", "")).upper() == symbol), None)
            if match is None:
                return None
            positions.remove(match)
            self._state["open_positions"] = positions
            entry_price = float(match.get("entry_price", 0.0) or 0.0)
            quantity = float(match.get("quantity", 0.0) or 0.0)
            direction = str(match.get("direction", "LONG"))
            pnl = (exit_price - entry_price) * quantity if direction == "LONG" else (entry_price - exit_price) * quantity
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0 if direction == "LONG" else ((entry_price - exit_price) / entry_price) * 100.0
            capital_allocated = float(match.get("capital_allocated", 0.0) or 0.0)
            self._state["cash"] = round(float(self._state.get("cash", 0.0) or 0.0) + capital_allocated + pnl, 2)
            closed = {
                **match,
                "exit_price": round(float(exit_price), 2),
                "closed_at": _now_iso(),
                "pnl": round(float(pnl), 2),
                "pnl_pct": round(float(pnl_pct), 2),
                "reason": str(reason or "manual"),
                "result": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT",
            }
            self._state.setdefault("closed_trades", []).append(closed)
            self._state.setdefault("equity_curve", []).append({"time": _now_iso(), "equity": self._compute_metrics()["portfolio_value"]})
            self._save()
            return closed


class AutoExecutor:
    def __init__(self, engine: PaperTradingEngine) -> None:
        self._engine = engine
        self._running = False
        self._lock = threading.RLock()
        self._pending: list[dict[str, Any]] = []
        self._seen_keys: dict[str, float] = {}

    @staticmethod
    def _signal_key(payload: dict[str, Any]) -> str:
        ticker = str(payload.get("ticker") or payload.get("asset") or "").upper()
        signal = str(payload.get("signal") or "").upper()
        entry = round(float(payload.get("entry_price") or 0.0), 2)
        stop = round(float(payload.get("stop_loss") or 0.0), 2)
        take = round(float(payload.get("take_profit") or payload.get("tp1") or 0.0), 2)
        return f"{ticker}|{signal}|{entry}|{stop}|{take}"

    def start(self) -> None:
        self._running = True
        while self._running:
            time.sleep(2.0)

    def stop(self) -> None:
        self._running = False

    def get_pending_signals(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._pending]

    def submit_signal(self, payload: dict[str, Any]) -> dict[str, Any]:
        signal = PaperTradingEngine._normalize_signal(payload.get("signal"))
        if signal not in {"LONG", "SHORT"}:
            return {"success": False, "error": "Only directional signals can be submitted"}
        key = self._signal_key(payload)
        with self._lock:
            now = time.time()
            self._seen_keys = {k: ts for k, ts in self._seen_keys.items() if now - ts < 1800}
            ticker = str(payload.get("ticker") or payload.get("asset") or "").upper()
            if any(str(pos.get("ticker", "")).upper() == ticker for pos in self._engine.get_open_positions()):
                return {"success": False, "duplicate": True, "reason": "position already open"}
            if any(str(row.get("ticker", "")).upper() == ticker and str(row.get("signal", "")).upper() == signal for row in self._pending):
                return {"success": False, "duplicate": True, "reason": "signal already pending"}
            if key in self._seen_keys:
                return {"success": False, "duplicate": True}
            self._seen_keys[key] = now
            mode = str(self._engine._state.get("mode", "manual"))
            enriched = dict(payload)
            enriched["trade_id"] = enriched.get("trade_id") or uuid.uuid4().hex[:12]
            enriched["ticker"] = enriched.get("ticker") or enriched.get("asset") or "BTCUSDT"
            if mode == "auto":
                return self._engine.execute_trade(enriched, mode="auto")
            pending = {
                "trade_id": enriched["trade_id"],
                "ticker": str(enriched["ticker"]).upper(),
                "signal": signal,
                "confidence": round(float(enriched.get("confidence", 0.0) or 0.0), 2),
                "entry_price": round(float(enriched.get("entry_price", 0.0) or 0.0), 2),
                "stop_loss": round(float(enriched.get("stop_loss", 0.0) or 0.0), 2),
                "take_profit": round(float(enriched.get("take_profit") or enriched.get("tp1") or 0.0), 2),
                "asset_class": enriched.get("asset_class", "crypto"),
                "position_size_pct": enriched.get("position_size_pct"),
                "created_at": _now_iso(),
                "payload": enriched,
            }
            self._pending.append(pending)
            return {"success": True, "queued": True, "trade_id": pending["trade_id"], "mode": mode}

    def approve_signal(self, ticker: str, trade_id: str) -> dict[str, Any]:
        with self._lock:
            match = next(
                (
                    row
                    for row in self._pending
                    if str(row.get("ticker", "")).upper() == str(ticker).upper()
                    and str(row.get("trade_id")) == str(trade_id)
                ),
                None,
            )
            if match is None:
                return {"success": False, "error": "Pending signal not found"}
            self._pending.remove(match)
            return self._engine.execute_trade(dict(match.get("payload") or {}), mode="manual-approved")

    def reject_signal(self, ticker: str, trade_id: str) -> None:
        with self._lock:
            self._pending = [
                row
                for row in self._pending
                if not (
                    str(row.get("ticker", "")).upper() == str(ticker).upper()
                    and str(row.get("trade_id")) == str(trade_id)
                )
            ]


_PAPER_ENGINE: PaperTradingEngine | None = None
_AUTO_EXECUTOR: AutoExecutor | None = None


def get_paper_engine() -> PaperTradingEngine:
    global _PAPER_ENGINE
    if _PAPER_ENGINE is None:
        _PAPER_ENGINE = PaperTradingEngine()
    return _PAPER_ENGINE


def get_auto_executor() -> AutoExecutor:
    global _AUTO_EXECUTOR
    if _AUTO_EXECUTOR is None:
        _AUTO_EXECUTOR = AutoExecutor(get_paper_engine())
    return _AUTO_EXECUTOR
