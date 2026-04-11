"""Tracks signal outcomes for paper trading and win rate monitoring."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.dashboard.trade_manager import TradeManager

logger = logging.getLogger(__name__)

HISTORY_FILE = Path("data/signal_history.json")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
MAX_LOSS_CAP_PCT = -2.0
_trade_manager = TradeManager()


def _load() -> list[dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def _save(history: list[dict[str, Any]]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _loss_cap(pnl_pct: float) -> float:
    return max(float(pnl_pct), MAX_LOSS_CAP_PCT)


def _directional_pnl_pct(entry: float, exit_price: float, signal: str) -> float:
    if entry <= 0:
        return 0.0
    raw = ((exit_price - entry) / entry) * 100
    return raw if signal == "LONG" else -raw


def _rr_pnl_pct(record: dict[str, Any], result: str, fallback_exit: float | None = None) -> float:
    entry = _safe_float(record.get("entry"), 0.0)
    signal = str(record.get("signal", "LONG")).upper()
    sl = _safe_float(record.get("stop_loss"), 0.0)
    rr_ratio = _safe_float(record.get("risk_reward"), 2.0)
    if rr_ratio <= 0:
        rr_ratio = 2.0
    sl_distance_pct = abs(entry - sl) / entry if entry > 0 and sl > 0 else 0.0

    if result == "SL":
        if sl_distance_pct > 0:
            return _loss_cap(-(sl_distance_pct * 100))
        if fallback_exit is not None:
            return _loss_cap(_directional_pnl_pct(entry, fallback_exit, signal))
        return 0.0

    if result == "TP1":
        if sl_distance_pct > 0:
            if signal == "SHORT":
                return sl_distance_pct * 2.0 * 100
            return sl_distance_pct * rr_ratio * 100
        if fallback_exit is not None:
            return _directional_pnl_pct(entry, fallback_exit, signal)
        return 0.0

    if result == "TP2":
        if sl_distance_pct > 0:
            if signal == "SHORT":
                return sl_distance_pct * 3.0 * 100
            return sl_distance_pct * (rr_ratio + 0.5) * 100
        if fallback_exit is not None:
            return _directional_pnl_pct(entry, fallback_exit, signal)
        return 0.0

    if result == "TP3":
        if sl_distance_pct > 0:
            if signal == "SHORT":
                return sl_distance_pct * 4.0 * 100
            return sl_distance_pct * (rr_ratio + 1.0) * 100
        if fallback_exit is not None:
            return _directional_pnl_pct(entry, fallback_exit, signal)
        return 0.0

    if fallback_exit is not None:
        pnl_pct = _directional_pnl_pct(entry, fallback_exit, signal)
        return _loss_cap(pnl_pct) if pnl_pct < 0 else pnl_pct
    return 0.0


def _status_to_result(status: str) -> str:
    status_u = str(status).upper()
    if status_u == "SL_HIT":
        return "SL"
    if status_u == "TP1_HIT":
        return "TP1"
    if status_u == "TP2_HIT":
        return "TP2"
    if status_u == "TP3_HIT":
        return "TP3"
    if status_u == "BLOCKED":
        return "BLOCKED"
    if status_u == "OPEN":
        return "OPEN"
    return status_u


def _close_record(rec: dict[str, Any], status: str, exit_price: float) -> None:
    result = _status_to_result(status)
    rec["status"] = status
    rec["result"] = result
    rec["exit_price"] = round(float(exit_price), 2)
    rec["pnl_pct"] = round(_rr_pnl_pct(rec, result, fallback_exit=float(exit_price)), 2)
    rec["closed_time"] = time.time()


def _is_today_utc(value: object) -> bool:
    if value is None:
        return False
    today = datetime.now(timezone.utc).date()
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            return dt.date() == today
        text = str(value).strip()
        if not text:
            return False
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.date() == today
    except Exception:
        return False


def record_signal(signal_data: dict[str, Any]) -> None:
    """Save a new OPEN trade signal or BLOCKED signal event."""
    signal_side = str(signal_data.get("signal", "")).upper()
    requested_signal = str(signal_data.get("requested_signal", "")).upper()
    blocked_by_raw = signal_data.get("blocked_by")
    blocked_by = "" if blocked_by_raw is None else str(blocked_by_raw).strip()
    is_blocked = bool(blocked_by) or str(signal_data.get("result", "")).upper() == "BLOCKED"
    is_trade_signal = signal_side in ("LONG", "SHORT") and bool(signal_data.get("validated"))

    if is_blocked and signal_side not in ("LONG", "SHORT") and requested_signal in {"LONG", "SHORT"}:
        signal_side = requested_signal

    if not is_trade_signal and not is_blocked:
        return

    history = _load()
    now_ts = time.time()

    # [ADDITIVE] Prevent duplicate OPEN records — if there's already an OPEN
    # record for the same direction, skip creating another one.
    # This prevents signal_history.json from inflating with duplicates every 15s.
    try:
        if not is_blocked:
            for _existing in reversed(history):
                if str(_existing.get("status", "")).upper() == "OPEN":
                    _existing_dir = str(_existing.get("signal", "")).upper()
                    if _existing_dir == signal_side:
                        # Same direction OPEN already exists — skip
                        return
                    else:
                        # Different direction — close the old one first
                        _existing["status"] = "CLOSED"
                        _existing["result"] = "FLIPPED"
                        _existing["closed_time"] = now_ts
                    break
    except Exception:
        pass

    status = "BLOCKED" if is_blocked else "OPEN"
    result = "BLOCKED" if is_blocked else "OPEN"
    direction = "long" if signal_side == "LONG" else "short" if signal_side == "SHORT" else "flat"
    signal_type = "BLOCKED" if status == "BLOCKED" else signal_side

    trade_management_raw = signal_data.get("trade_management") if isinstance(signal_data.get("trade_management"), dict) else {}
    trail_gap = _safe_float(trade_management_raw.get("trail_gap"), 0.0)
    entry = _safe_float(signal_data.get("entry_price"), 0.0)
    stop_loss = _safe_float(signal_data.get("stop_loss"), 0.0)
    if trail_gap <= 0 and entry > 0 and stop_loss > 0:
        trail_gap = max(abs(entry - stop_loss) * 0.75, entry * 0.0035)

    trade_management = {
        "breakeven_after_tp1": bool(trade_management_raw.get("breakeven_after_tp1", True)),
        "trail_after_tp2": bool(trade_management_raw.get("trail_after_tp2", True)),
        "alpha_flip_exit": bool(trade_management_raw.get("alpha_flip_exit", True)),
        "trail_gap": round(float(trail_gap), 2),
    }

    record = {
        "id": len(history) + 1,
        "time": signal_data.get("as_of_utc"),
        "signal": signal_side or "HOLD",
        "confidence": signal_data.get("confidence", 0),
        "alpha_score": signal_data.get("alpha_score", 0),
        "entry": signal_data.get("entry_price"),
        "stop_loss": signal_data.get("stop_loss"),
        "active_stop_loss": signal_data.get("stop_loss"),
        "tp1": signal_data.get("tp1"),
        "tp2": signal_data.get("tp2"),
        "tp3": signal_data.get("tp3"),
        "risk_reward": signal_data.get("risk_reward"),
        "direction": direction,
        "type": signal_type,
        "regime": signal_data.get("regime"),
        "reason": signal_data.get("reason", ""),
        "blocked_by": blocked_by or None,
        "funding_rate": signal_data.get("funding_rate_pct", 0),
        "market_context": signal_data.get("market_context", {}),
        "trade_management": trade_management,
        "tp1_hit": False,
        "tp2_hit": False,
        "tp3_hit": False,
        "highest_price": signal_data.get("entry_price"),
        "lowest_price": signal_data.get("entry_price"),
        "milestones": [],
        "status": status,
        "result": result,
        "exit_price": None,
        "pnl_pct": None,
        "closed_time": now_ts if status == "BLOCKED" else None,
        "open_timestamp": now_ts,
    }
    history.append(record)
    _save(history)
    logger.info("Signal recorded: #%d %s status=%s", record["id"], record["signal"], record["status"])


def _check_open_signal_legacy(rec: dict[str, Any], current_price: float) -> bool:
    """Legacy close-at-first-target behavior for old/manual records."""
    entry = _safe_float(rec.get("entry"), 0.0)
    if entry <= 0:
        return False

    sl = _safe_float(rec.get("stop_loss"), 0.0)
    tp1 = _safe_float(rec.get("tp1"), 0.0)
    tp2 = _safe_float(rec.get("tp2"), 0.0)
    tp3 = _safe_float(rec.get("tp3"), 0.0)
    sig = str(rec.get("signal", "LONG")).upper()

    if sl and ((sig == "LONG" and current_price <= sl) or (sig == "SHORT" and current_price >= sl)):
        _close_record(rec, "SL_HIT", sl)
        return True

    if tp3 and ((sig == "LONG" and current_price >= tp3) or (sig == "SHORT" and current_price <= tp3)):
        _close_record(rec, "TP3_HIT", tp3)
        return True
    if tp2 and ((sig == "LONG" and current_price >= tp2) or (sig == "SHORT" and current_price <= tp2)):
        _close_record(rec, "TP2_HIT", tp2)
        return True
    if tp1 and ((sig == "LONG" and current_price >= tp1) or (sig == "SHORT" and current_price <= tp1)):
        _close_record(rec, "TP1_HIT", tp1)
        return True

    return False


def check_open_signals(
    current_price: float,
    market_signal: str | None = None,
    validated_signal: str | None = None,
    alpha_score: float | None = None,
) -> None:
    """Check open signals and apply legacy or managed post-entry logic."""
    history = _load()
    changed = False
    live_signal = str(validated_signal or market_signal or "").upper()
    _kelly_instance = None

    # [ADDITIVE] Close stale OPEN records older than 12 hours
    # and ensure only ONE OPEN record exists at a time
    try:
        _now_ts = time.time()
        _open_count = 0
        for _rec in history:
            if str(_rec.get("status", "")).upper() != "OPEN":
                continue
            _open_ts = float(_rec.get("open_timestamp", 0) or 0)
            _age_hours = (_now_ts - _open_ts) / 3600.0 if _open_ts > 0 else 999
            # Close records older than 12 hours
            if _age_hours > 12:
                _rec["status"] = "EXPIRED"
                _rec["result"] = "EXPIRED"
                _rec["closed_time"] = _now_ts
                changed = True
                continue
            _open_count += 1
            # Only allow 1 OPEN at a time — close extras (oldest first)
            if _open_count > 1:
                _rec["status"] = "CLOSED"
                _rec["result"] = "SUPERSEDED"
                _rec["closed_time"] = _now_ts
                changed = True
    except Exception:
        pass

    for rec in history:
        if str(rec.get("status", "")).upper() != "OPEN":
            continue

        entry = _safe_float(rec.get("entry"), 0.0)
        if entry <= 0:
            continue

        before = dict(rec)
        updated = _trade_manager.manage(
            record=rec,
            current_price=float(current_price),
            current_alpha=_safe_float(alpha_score, 0.0),
            current_signal=live_signal,
        )
        rec.update(updated)

        # Keep legacy close behavior for old records not using trade_manager metadata.
        managed_mode = isinstance(rec.get("trade_management"), dict) and bool(rec.get("trade_management"))
        if str(rec.get("status", "")).upper() == "OPEN" and not managed_mode:
            if _check_open_signal_legacy(rec, float(current_price)):
                changed = True

        if rec != before:
            changed = True

        before_status = str(before.get("status", "")).upper()
        after_status = str(rec.get("status", "")).upper()
        if before_status == "OPEN" and after_status not in {"OPEN", "BLOCKED"}:
            if _kelly_instance is None:
                from src.risk.kelly_warm_start import BTCKellyWarmStart

                _kelly_instance = BTCKellyWarmStart()
            _kelly_instance.update_bucket(dict(rec))

    if changed:
        _save(history)


def get_history(limit: int = 50) -> list[dict[str, Any]]:
    return _load()[-limit:]


def get_stats() -> dict[str, Any]:
    history = _load()
    blocked = [h for h in history if str(h.get("status", "")).upper() == "BLOCKED"]
    closed = [h for h in history if str(h.get("status", "")).upper() not in {"OPEN", "BLOCKED"}]
    today_directional = [
        h
        for h in history
        if _is_today_utc(h.get("time"))
        and str(h.get("signal", "")).upper() in {"LONG", "SHORT"}
        and str(h.get("status", "")).upper() != "BLOCKED"
    ]
    closed_long = [h for h in closed if str(h.get("signal", "")).upper() == "LONG"]
    closed_short = [h for h in closed if str(h.get("signal", "")).upper() == "SHORT"]
    long_wins = [h for h in closed_long if _safe_float(h.get("pnl_pct"), 0.0) > 0]
    short_wins = [h for h in closed_short if _safe_float(h.get("pnl_pct"), 0.0) > 0]
    long_pnl = sum(_safe_float(h.get("pnl_pct"), 0.0) for h in closed_long)
    short_pnl = sum(_safe_float(h.get("pnl_pct"), 0.0) for h in closed_short)

    if not closed:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "open_signals": len([h for h in history if str(h.get("status", "")).upper() == "OPEN"]),
            "blocked_total": len(blocked),
            "blocked_today": len([h for h in blocked if _is_today_utc(h.get("time"))]),
            "long_total": 0,
            "short_total": 0,
            "long_signals_today": len([h for h in today_directional if str(h.get("signal", "")).upper() == "LONG"]),
            "short_signals_today": len([h for h in today_directional if str(h.get("signal", "")).upper() == "SHORT"]),
            "long_win_rate": 0,
            "short_win_rate": 0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
        }

    wins = [h for h in closed if _safe_float(h.get("pnl_pct"), 0.0) > 0]
    losses = [h for h in closed if _safe_float(h.get("pnl_pct"), 0.0) <= 0]
    total_pnl = sum(_safe_float(h.get("pnl_pct"), 0.0) for h in closed)

    return {
        "total": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(closed), 2) if closed else 0,
        "open_signals": len([h for h in history if str(h.get("status", "")).upper() == "OPEN"]),
        "blocked_total": len(blocked),
        "blocked_today": len([h for h in blocked if _is_today_utc(h.get("time"))]),
        "long_total": len(closed_long),
        "short_total": len(closed_short),
        "long_signals_today": len([h for h in today_directional if str(h.get("signal", "")).upper() == "LONG"]),
        "short_signals_today": len([h for h in today_directional if str(h.get("signal", "")).upper() == "SHORT"]),
        "long_win_rate": round(len(long_wins) / len(closed_long) * 100, 1) if closed_long else 0,
        "short_win_rate": round(len(short_wins) / len(closed_short) * 100, 1) if closed_short else 0,
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
    }
