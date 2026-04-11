"""Paper trading engine for multi-asset virtual execution."""

from __future__ import annotations

import json
import math
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PaperTradingEngine:
    """
    Virtual paper trading engine.
    Persists to: data/paper_trading.json
    Supports: Manual + Auto execution modes
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.data_file = Path("data/paper_trading.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state = self._load()
        if not self._state:
            self._state = self._fresh_state(initial_capital=initial_capital)
            self._save()

    @property
    def mode(self) -> str:
        return str(self._state.get("mode", "manual"))

    def _fresh_state(self, initial_capital: float) -> dict[str, Any]:
        return {
            "capital": float(initial_capital),
            "initial_capital": float(initial_capital),
            "positions": {},
            "closed_trades": [],
            "mode": "manual",
            "auto_enabled": False,
            "created_at": time.time(),
            "stats": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "peak_equity": float(initial_capital),
                "max_drawdown": 0.0,
            },
        }

    def execute_trade(self, signal: dict, mode: str = "manual") -> dict:
        with self._lock:
            ticker = str(signal.get("ticker") or signal.get("asset") or "").strip().upper()
            # Accept both legacy `signal` and UI `direction` payload keys.
            direction = self._normalize_direction(signal.get("signal", signal.get("direction")))
            if not ticker:
                return {"success": False, "message": "Ticker missing"}
            if direction not in {"LONG", "SHORT"}:
                return {"success": False, "message": "Signal is HOLD"}
            if ticker in self._state["positions"]:
                return {"success": False, "message": f"Position already open for {ticker}"}

            entry_price = self._to_float(signal.get("entry_price"), 0.0)
            stop_loss = self._to_float(signal.get("stop_loss"), 0.0)
            take_profit = self._to_float(signal.get("take_profit"), 0.0)
            if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
                return {"success": False, "message": "Invalid entry/SL/TP prices"}

            capital = self._to_float(self._state.get("capital"), 0.0)
            risk_amount = capital * 0.02
            risk_dist = abs(entry_price - stop_loss)
            if risk_dist <= 0:
                return {"success": False, "message": "Invalid risk distance"}

            position_size = risk_amount / risk_dist
            max_value = capital * 0.10
            max_qty = max_value / entry_price if entry_price > 0 else 0.0
            quantity = min(position_size, max_qty)
            if quantity <= 0:
                return {"success": False, "message": "Calculated quantity is zero"}

            value = quantity * entry_price
            if value > capital:
                return {"success": False, "message": "Insufficient capital"}

            trade_id = self._generate_trade_id()
            now = datetime.now(timezone.utc).isoformat()

            self._state["capital"] = capital - value
            self._state["positions"][ticker] = {
                "trade_id": trade_id,
                "ticker": ticker,
                "direction": direction,
                "entry_price": entry_price,
                "current_price": entry_price,
                "quantity": quantity,
                "value": value,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "opened_at": now,
                "opened_ts": time.time(),
                "asset_class": str(signal.get("asset_class") or "unknown"),
                "confidence": self._to_float(signal.get("confidence"), 0.0),
                "strength": str(signal.get("strength") or ""),
                "mode": mode,
            }
            self._save()
            return {
                "success": True,
                "trade_id": trade_id,
                "ticker": ticker,
                "direction": direction,
                "entry_price": entry_price,
                "quantity": quantity,
                "value": value,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timestamp": now,
                "message": f"Paper trade executed for {ticker}",
            }

    def close_position(self, ticker: str, exit_price: float, reason: str = "manual") -> dict | None:
        with self._lock:
            ticker = str(ticker or "").upper().strip()
            pos = self._state["positions"].get(ticker)
            if not pos:
                return None

            entry = self._to_float(pos.get("entry_price"), 0.0)
            qty = self._to_float(pos.get("quantity"), 0.0)
            direction = str(pos.get("direction", "LONG")).upper()
            if entry <= 0 or qty <= 0 or exit_price <= 0:
                return None

            pnl = (exit_price - entry) * qty if direction == "LONG" else (entry - exit_price) * qty
            entry_value = entry * qty
            pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0.0
            held_hours = max(0.0, (time.time() - self._to_float(pos.get("opened_ts"), time.time())) / 3600.0)

            closed = {
                "trade_id": pos.get("trade_id"),
                "ticker": ticker,
                "entry_price": entry,
                "exit_price": float(exit_price),
                "quantity": qty,
                "pnl": float(round(pnl, 4)),
                "pnl_pct": float(round(pnl_pct, 4)),
                "direction": direction,
                "reason": reason,
                "opened_at": pos.get("opened_at"),
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "held_hours": float(round(held_hours, 4)),
                "asset_class": pos.get("asset_class", "unknown"),
                "confidence": self._to_float(pos.get("confidence"), 0.0),
            }
            self._state["closed_trades"].append(closed)
            self._state["capital"] = self._to_float(self._state.get("capital"), 0.0) + entry_value + pnl
            self._state["positions"].pop(ticker, None)

            stats = self._state.setdefault("stats", {})
            stats["total_trades"] = int(stats.get("total_trades", 0)) + 1
            stats["wins"] = int(stats.get("wins", 0)) + (1 if pnl > 0 else 0)
            stats["losses"] = int(stats.get("losses", 0)) + (1 if pnl <= 0 else 0)
            stats["total_pnl"] = self._to_float(stats.get("total_pnl"), 0.0) + pnl

            portfolio_value = self._compute_portfolio_value()
            peak = max(self._to_float(stats.get("peak_equity"), portfolio_value), portfolio_value)
            stats["peak_equity"] = peak
            drawdown = ((portfolio_value - peak) / peak * 100.0) if peak > 0 else 0.0
            stats["max_drawdown"] = min(self._to_float(stats.get("max_drawdown"), 0.0), drawdown)

            self._save()
            return {
                "ticker": ticker,
                "entry_price": entry,
                "exit_price": float(exit_price),
                "quantity": qty,
                "pnl": float(round(pnl, 4)),
                "pnl_pct": float(round(pnl_pct, 4)),
                "direction": direction,
                "reason": reason,
                "held_hours": float(round(held_hours, 4)),
            }

    def check_sl_tp(self, ticker: str, current_price: float) -> dict | None:
        with self._lock:
            pos = self._state["positions"].get(str(ticker).upper().strip())
            if not pos:
                return None
            direction = str(pos.get("direction", "LONG")).upper()
            sl = self._to_float(pos.get("stop_loss"), 0.0)
            tp = self._to_float(pos.get("take_profit"), 0.0)
            px = self._to_float(current_price, 0.0)
            if px <= 0:
                return None
            if direction == "LONG":
                if sl > 0 and px <= sl:
                    return self.close_position(str(ticker), sl, "sl_hit")
                if tp > 0 and px >= tp:
                    return self.close_position(str(ticker), tp, "tp_hit")
            else:
                if sl > 0 and px >= sl:
                    return self.close_position(str(ticker), sl, "sl_hit")
                if tp > 0 and px <= tp:
                    return self.close_position(str(ticker), tp, "tp_hit")
            return None

    def update_prices(self, prices: dict) -> list:
        closes: list[dict] = []
        with self._lock:
            dirty = False
            for ticker in list(self._state["positions"].keys()):
                px = self._to_float(prices.get(ticker), 0.0)
                if px <= 0:
                    continue
                pos = self._state["positions"].get(ticker)
                if not pos:
                    continue
                entry = self._to_float(pos.get("entry_price"), 0.0)
                qty = self._to_float(pos.get("quantity"), 0.0)
                direction = str(pos.get("direction", "LONG")).upper()
                pnl = (px - entry) * qty if direction == "LONG" else (entry - px) * qty
                base = entry * qty
                pos["current_price"] = px
                pos["unrealized_pnl"] = float(round(pnl, 4))
                pos["unrealized_pnl_pct"] = float(round((pnl / base * 100.0) if base > 0 else 0.0, 4))
                dirty = True
                closed = self.check_sl_tp(ticker, px)
                if closed:
                    closes.append(closed)

            if dirty:
                self._save()
        return closes

    def get_portfolio_metrics(self) -> dict:
        with self._lock:
            initial = self._to_float(self._state.get("initial_capital"), 0.0)
            capital = self._to_float(self._state.get("capital"), 0.0)
            positions = self.get_open_positions()
            invested = sum(self._to_float(p.get("entry_price"), 0.0) * self._to_float(p.get("quantity"), 0.0) for p in positions)
            unrealized = sum(self._to_float(p.get("unrealized_pnl"), 0.0) for p in positions)
            realized = sum(self._to_float(t.get("pnl"), 0.0) for t in self._state.get("closed_trades", []))
            portfolio_value = capital + invested + unrealized
            total_pnl = portfolio_value - initial
            total_pnl_pct = (total_pnl / initial * 100.0) if initial > 0 else 0.0

            closed = self._state.get("closed_trades", [])
            wins = [t for t in closed if self._to_float(t.get("pnl"), 0.0) > 0]
            losses = [t for t in closed if self._to_float(t.get("pnl"), 0.0) <= 0]
            gross_profit = sum(self._to_float(t.get("pnl"), 0.0) for t in wins)
            gross_loss = abs(sum(self._to_float(t.get("pnl"), 0.0) for t in losses))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

            avg_win = sum(self._to_float(t.get("pnl_pct"), 0.0) for t in wins) / len(wins) if wins else 0.0
            avg_loss = sum(self._to_float(t.get("pnl_pct"), 0.0) for t in losses) / len(losses) if losses else 0.0
            best_trade = max((self._to_float(t.get("pnl_pct"), 0.0) for t in closed), default=0.0)
            worst_trade = min((self._to_float(t.get("pnl_pct"), 0.0) for t in closed), default=0.0)
            win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
            stats = self._state.get("stats", {})

            return {
                "capital": float(round(capital, 4)),
                "initial_capital": float(round(initial, 4)),
                "portfolio_value": float(round(portfolio_value, 4)),
                "total_pnl": float(round(total_pnl, 4)),
                "total_pnl_pct": float(round(total_pnl_pct, 4)),
                "unrealized_pnl": float(round(unrealized, 4)),
                "realized_pnl": float(round(realized, 4)),
                "total_trades": int(len(closed)),
                "open_positions": int(len(positions)),
                "wins": int(len(wins)),
                "losses": int(len(losses)),
                "win_rate": float(round(win_rate, 4)),
                "sharpe_ratio": float(round(self._compute_sharpe(), 4)),
                "max_drawdown": float(round(self._to_float(stats.get("max_drawdown"), 0.0), 4)),
                "profit_factor": float(round(profit_factor, 4)),
                "avg_win_pct": float(round(avg_win, 4)),
                "avg_loss_pct": float(round(avg_loss, 4)),
                "best_trade_pct": float(round(best_trade, 4)),
                "worst_trade_pct": float(round(worst_trade, 4)),
                "mode": str(self._state.get("mode", "manual")),
            }

    def get_open_positions(self) -> list:
        with self._lock:
            out: list[dict[str, Any]] = []
            now = time.time()
            for ticker, p in self._state.get("positions", {}).items():
                opened_ts = self._to_float(p.get("opened_ts"), now)
                out.append(
                    {
                        "ticker": ticker,
                        "direction": str(p.get("direction", "LONG")),
                        "entry_price": self._to_float(p.get("entry_price"), 0.0),
                        "current_price": self._to_float(p.get("current_price"), self._to_float(p.get("entry_price"), 0.0)),
                        "quantity": self._to_float(p.get("quantity"), 0.0),
                        "value": self._to_float(p.get("entry_price"), 0.0) * self._to_float(p.get("quantity"), 0.0),
                        "unrealized_pnl": self._to_float(p.get("unrealized_pnl"), 0.0),
                        "unrealized_pnl_pct": self._to_float(p.get("unrealized_pnl_pct"), 0.0),
                        "stop_loss": self._to_float(p.get("stop_loss"), 0.0),
                        "take_profit": self._to_float(p.get("take_profit"), 0.0),
                        "opened_at": str(p.get("opened_at", "")),
                        "held_hours": round(max(0.0, (now - opened_ts) / 3600.0), 4),
                        "asset_class": str(p.get("asset_class", "unknown")),
                        "confidence": self._to_float(p.get("confidence"), 0.0),
                        "trade_id": str(p.get("trade_id", "")),
                    }
                )
            out.sort(key=lambda x: str(x.get("opened_at", "")), reverse=True)
            return out

    def get_closed_trades(self, limit: int = 50) -> list:
        with self._lock:
            trades = list(self._state.get("closed_trades", []))
            trades.sort(key=lambda x: str(x.get("closed_at", "")), reverse=True)
            return trades[: max(1, int(limit))]

    def set_mode(self, mode: str) -> None:
        with self._lock:
            m = str(mode or "manual").lower()
            m = "auto" if m == "auto" else "manual"
            self._state["mode"] = m
            self._state["auto_enabled"] = m == "auto"
            self._save()

    def reset(self, new_capital: float | None = None) -> None:
        with self._lock:
            if self.data_file.exists():
                backup = self.data_file.parent / f"paper_trading_backup_{int(time.time())}.json"
                try:
                    backup.write_text(self.data_file.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass
            initial = float(new_capital) if new_capital is not None else self._to_float(self._state.get("initial_capital"), 100000.0)
            self._state = self._fresh_state(initial_capital=initial)
            self._save()

    def get_daily_returns(self) -> list:
        with self._lock:
            closed = list(self._state.get("closed_trades", []))
            if not closed:
                return []
            per_day: dict[str, float] = {}
            for tr in closed:
                ts = str(tr.get("closed_at") or tr.get("closed_time") or "")
                day = ts[:10] if len(ts) >= 10 else datetime.utcnow().date().isoformat()
                per_day[day] = per_day.get(day, 0.0) + self._to_float(tr.get("pnl"), 0.0)
            equity = self._to_float(self._state.get("initial_capital"), 0.0)
            out = []
            for day in sorted(per_day.keys()):
                equity += per_day[day]
                out.append({"date": day, "equity": float(round(equity, 4))})
            return out

    def _compute_sharpe(self) -> float:
        series = self.get_daily_returns()
        if len(series) < 10:
            return 0.0
        rets: list[float] = []
        prev = self._to_float(self._state.get("initial_capital"), 0.0)
        for row in series:
            eq = self._to_float(row.get("equity"), prev)
            if prev > 0:
                rets.append((eq - prev) / prev)
            prev = eq
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((x - mean) ** 2 for x in rets) / max(1, len(rets) - 1)
        std = math.sqrt(max(var, 1e-12))
        return (mean / std) * math.sqrt(252.0)

    def _load(self) -> dict:
        if not self.data_file.exists():
            return {}
        try:
            return json.loads(self.data_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        tmp = self.data_file.with_suffix(".json.tmp")
        payload = json.dumps(self._state, indent=2, sort_keys=False)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.data_file)

    def _generate_trade_id(self) -> str:
        return f"PT-{int(time.time())}-{random.randint(1000, 9999)}"

    def _compute_portfolio_value(self) -> float:
        positions = self._state.get("positions", {})
        capital = self._to_float(self._state.get("capital"), 0.0)
        invested = 0.0
        unrealized = 0.0
        for p in positions.values():
            entry = self._to_float(p.get("entry_price"), 0.0)
            qty = self._to_float(p.get("quantity"), 0.0)
            invested += entry * qty
            unrealized += self._to_float(p.get("unrealized_pnl"), 0.0)
        return capital + invested + unrealized

    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _normalize_direction(signal: Any) -> str:
        s = str(signal or "").upper().strip()
        if s in {"BUY", "LONG"}:
            return "LONG"
        if s in {"SELL", "SHORT"}:
            return "SHORT"
        return "HOLD"

