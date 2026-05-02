"""Paper trading engine for multi-asset virtual execution."""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.meta.calibration_freshness import CalibrationFreshnessGuard
from src.meta.config import enforcement_active, load_meta_controls_config, module_enabled
from src.meta.data_confidence import DataConfidenceEngine
from src.meta.kelly_shrinkage import KellyShrinkageController

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Virtual paper trading engine.
    Persists to: data/paper_trading.json
    Supports: Manual + Auto execution modes
    """

    def __init__(self, initial_capital: float = 100000.0, data_file: str | Path | None = None):
        self.data_file = Path(data_file) if data_file is not None else Path("data/paper_trading.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state = self._load()
        if not self._state:
            self._state = self._fresh_state(initial_capital=initial_capital)
            self._save()

    def _total_trade_count(self) -> int:
        stats = self._state.get("stats") if isinstance(self._state.get("stats"), dict) else {}
        try:
            total = int(stats.get("total_trades", 0) or 0)
        except Exception:
            total = 0
        closed = self._state.get("closed_trades")
        if isinstance(closed, list):
            total = max(total, len(closed))
        return max(0, total)

    def _meta_execution_review(self, signal: dict[str, Any]) -> dict[str, Any]:
        try:
            meta_cfg = load_meta_controls_config()
            if not bool(meta_cfg.get("enabled", False)):
                return {"enabled": False}

            shadow_mode = bool(meta_cfg.get("shadow_mode", True))
            enforce_execution_gates = bool(meta_cfg.get("enforce_execution_gates", False))
            enforced = enforcement_active(meta_cfg)
            total_trades = self._total_trade_count()
            out: dict[str, Any] = {
                "enabled": True,
                "shadow_mode": shadow_mode,
                "enforced": enforced,
                "execution_position_multiplier": 1.0,
                "execution_confidence_multiplier": 1.0,
                "block_new_execution": False,
            }

            if module_enabled(meta_cfg, "data_confidence"):
                embedded = signal.get("meta_controls") if isinstance(signal.get("meta_controls"), dict) else {}
                embedded_dc = embedded.get("data_confidence") if isinstance(embedded.get("data_confidence"), dict) else {}
                feed_status = embedded_dc.get("feed_status") if isinstance(embedded_dc.get("feed_status"), dict) else None
                dc_result = DataConfidenceEngine(
                    meta_cfg.get("data_confidence", {}) or {},
                    shadow_mode=shadow_mode,
                    enforce_execution_gates=enforce_execution_gates,
                ).evaluate(feed_status or DataConfidenceEngine.status_from_signal_payload(signal))
                dc_payload = dc_result.to_dict()
                out["data_confidence"] = dc_payload
                out["execution_position_multiplier"] = min(
                    float(out["execution_position_multiplier"]),
                    float(dc_payload.get("execution_position_multiplier", 1.0)),
                )
                out["block_new_execution"] = bool(out["block_new_execution"] or dc_payload.get("execution_block_new_trades", False))

            if module_enabled(meta_cfg, "calibration_freshness"):
                cal_result = CalibrationFreshnessGuard(
                    meta_cfg.get("calibration_freshness", {}) or {},
                    shadow_mode=shadow_mode,
                    enforce_execution_gates=enforce_execution_gates,
                ).evaluate(
                    signal.get("last_calibration_timestamp"),
                    checkpoint_path=(meta_cfg.get("calibration_freshness") or {}).get(
                        "checkpoint_path",
                        CalibrationFreshnessGuard.DEFAULT_CHECKPOINT_PATH,
                    ),
                )
                cal_payload = cal_result.to_dict()
                out["calibration_freshness"] = cal_payload
                out["execution_confidence_multiplier"] = min(
                    float(out["execution_confidence_multiplier"]),
                    float(cal_payload.get("execution_confidence_multiplier", 1.0)),
                )
                out["execution_position_multiplier"] = min(
                    float(out["execution_position_multiplier"]),
                    float(cal_payload.get("execution_position_multiplier", 1.0)),
                )

            if module_enabled(meta_cfg, "kelly_shrinkage"):
                kelly_result = KellyShrinkageController.from_position_sizing(
                    signal,
                    total_trade_count=total_trades,
                    config=meta_cfg.get("kelly_shrinkage", {}) or {},
                )
                out["kelly_shrinkage"] = kelly_result.to_dict()

            return out
        except Exception as exc:
            logger.debug("Paper meta-control review skipped: %s", exc)
            return {"enabled": False, "error": str(exc)}

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
            if direction == "LONG" and not (stop_loss < entry_price < take_profit):
                return {
                    "success": False,
                    "message": "Invalid LONG setup: Stop Loss must be below entry and Take Profit above entry",
                }
            if direction == "SHORT" and not (stop_loss > entry_price > take_profit):
                return {
                    "success": False,
                    "message": "Invalid SHORT setup: Stop Loss must be above entry and Take Profit below entry",
                }

            meta_controls = self._meta_execution_review(signal)
            if bool(meta_controls.get("block_new_execution", False)):
                return {
                    "success": False,
                    "message": "Blocked by data confidence gate",
                    "blocked_by": "data_confidence",
                    "meta_controls": meta_controls,
                }

            capital = self._to_float(self._state.get("capital"), 0.0)
            risk_amount = capital * 0.02
            max_value = capital * 0.10
            risk_dist = abs(entry_price - stop_loss)
            if risk_dist <= 0:
                return {"success": False, "message": "Invalid risk distance"}

            sizing_mode_raw = str(signal.get("sizing_mode") or "auto").strip().lower()
            custom_usd = self._to_float(signal.get("capital_to_use"), 0.0)
            custom_pct = self._to_float(signal.get("capital_pct"), 0.0)

            if sizing_mode_raw in {"usd", "fixed", "custom_usd"}:
                if custom_usd <= 0:
                    return {"success": False, "message": "Invalid custom capital amount"}
                if custom_usd > capital:
                    return {"success": False, "message": "Requested capital exceeds available paper capital"}
                quantity = custom_usd / entry_price
                sizing_mode = "user-usd"
            elif sizing_mode_raw in {"pct", "percent", "percentage", "custom_pct"}:
                if custom_pct <= 0:
                    return {"success": False, "message": "Invalid custom capital percentage"}
                if custom_pct > 100:
                    return {"success": False, "message": "Capital percentage cannot exceed 100%"}
                value_target = capital * (custom_pct / 100.0)
                quantity = value_target / entry_price
                sizing_mode = "user-pct"
            else:
                position_size = risk_amount / risk_dist
                max_qty = max_value / entry_price if entry_price > 0 else 0.0
                quantity = min(position_size, max_qty)
                sizing_mode = "auto-risk" if position_size <= max_qty else "auto-cap"
            if bool(meta_controls.get("enforced", False)):
                position_multiplier = self._to_float(meta_controls.get("execution_position_multiplier"), 1.0)
                if 0.0 < position_multiplier < 1.0:
                    quantity *= position_multiplier
                    sizing_mode = f"{sizing_mode}+meta"
                kelly_meta = meta_controls.get("kelly_shrinkage") if isinstance(meta_controls.get("kelly_shrinkage"), dict) else {}
                effective_kelly = self._to_float(kelly_meta.get("effective_kelly_fraction"), 0.0)
                if effective_kelly > 0:
                    kelly_qty_cap = (capital * effective_kelly) / entry_price
                    if kelly_qty_cap > 0 and quantity > kelly_qty_cap:
                        quantity = kelly_qty_cap
                        sizing_mode = f"{sizing_mode}+kelly"
            if quantity <= 0:
                return {"success": False, "message": "Calculated quantity is zero"}

            value = quantity * entry_price
            if value > capital:
                return {"success": False, "message": "Insufficient capital"}
            capital_used_pct = (value / capital * 100.0) if capital > 0 else 0.0
            risk_at_sl = quantity * risk_dist
            risk_pct_at_sl = (risk_at_sl / capital * 100.0) if capital > 0 else 0.0

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
                "alpha_score": self._to_float(signal.get("alpha_score"), self._to_float(signal.get("confidence"), 0.0)),
                "regime": str(signal.get("regime") or ""),
                "strength": str(signal.get("strength") or ""),
                "mode": mode,
                "sizing_mode": sizing_mode,
                "meta_controls": meta_controls,
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
                "capital_used_pct": capital_used_pct,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_at_sl": risk_at_sl,
                "risk_pct_at_sl": risk_pct_at_sl,
                "sizing_mode": sizing_mode,
                "timestamp": now,
                "message": f"Paper trade executed for {ticker}",
                "meta_controls": meta_controls,
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
                "alpha_score": self._to_float(pos.get("alpha_score"), self._to_float(pos.get("confidence"), 0.0)),
                "regime": str(pos.get("regime", "")),
                "stop_loss": self._to_float(pos.get("stop_loss"), 0.0),
                "take_profit": self._to_float(pos.get("take_profit"), 0.0),
                "mode": str(pos.get("mode", "manual")),
                "sizing_mode": str(pos.get("sizing_mode", "auto")),
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
            if str(pos.get("mode", "manual")).lower() == "auto":
                self._record_auto_closed_trade_signal(pos=pos, closed=closed)
            return {
                "trade_id": str(pos.get("trade_id", "")),
                "ticker": ticker,
                "entry_price": entry,
                "exit_price": float(exit_price),
                "quantity": qty,
                "pnl": float(round(pnl, 4)),
                "pnl_pct": float(round(pnl_pct, 4)),
                "direction": direction,
                "reason": reason,
                "held_hours": float(round(held_hours, 4)),
                "mode": str(pos.get("mode", "manual")),
                "regime": str(pos.get("regime", "")),
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
                        "mode": str(p.get("mode", "manual")),
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

    def _record_auto_closed_trade_signal(self, pos: dict[str, Any], closed: dict[str, Any]) -> None:
        try:
            from src.data.signal_history import record_closed_trade

            entry = self._to_float(closed.get("entry_price"), 0.0)
            stop = self._to_float(pos.get("stop_loss"), self._to_float(closed.get("stop_loss"), 0.0))
            tp1 = self._to_float(pos.get("take_profit"), self._to_float(closed.get("take_profit"), 0.0))
            risk = abs(entry - stop)
            reward = abs(tp1 - entry)
            rr_ratio = (reward / risk) if risk > 0 and reward > 0 else 0.0

            record_closed_trade(
                {
                    "trade_id": str(closed.get("trade_id", "")),
                    "ticker": str(closed.get("ticker") or pos.get("ticker") or "BTCUSDT"),
                    "signal": str(closed.get("direction") or pos.get("direction") or "HOLD"),
                    "confidence": self._to_float(pos.get("confidence"), 0.0),
                    "alpha_score": self._to_float(pos.get("alpha_score"), self._to_float(pos.get("confidence"), 0.0)),
                    "entry_price": entry,
                    "exit_price": self._to_float(closed.get("exit_price"), 0.0),
                    "stop_loss": stop,
                    "tp1": tp1,
                    "risk_reward": rr_ratio,
                    "pnl_pct": self._to_float(closed.get("pnl_pct"), 0.0),
                    "reason": str(closed.get("reason", "")),
                    "regime": str(pos.get("regime", "")),
                    "opened_at": pos.get("opened_at"),
                    "closed_at": closed.get("closed_at"),
                    "source": "paper_trading_auto",
                    "mode": "ALGO",
                }
            )
        except Exception as exc:
            logger.debug("paper auto trade sync to signal_history skipped: %s", exc)

