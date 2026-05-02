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

ROOT_DIR = Path(__file__).resolve().parents[2]
HISTORY_FILE = ROOT_DIR / "data" / "signal_history.json"
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
MAX_LOSS_CAP_PCT = -2.0
BLOCKED_DEDUP_WINDOW_SEC = 75.0
OPEN_SIGNAL_EXPIRY_HOURS = 12.0
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


def _parse_epoch_seconds(value: object) -> float:
    try:
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1e12:
                ts /= 1000.0
            return ts if ts > 0 else 0.0
        text = str(value or "").strip()
        if not text:
            return 0.0
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def _expire_stale_open_records(history: list[dict[str, Any]], now_ts: float | None = None) -> bool:
    """Expire stale OPEN signal rows so they cannot block fresh signal recording."""
    changed = False
    now = float(now_ts if now_ts is not None else time.time())
    expiry_sec = OPEN_SIGNAL_EXPIRY_HOURS * 3600.0
    for rec in history:
        if str(rec.get("status", "")).upper() != "OPEN":
            continue
        opened_ts = _parse_epoch_seconds(rec.get("open_timestamp") or rec.get("time"))
        age_sec = (now - opened_ts) if opened_ts > 0 else expiry_sec + 1.0
        if age_sec > expiry_sec:
            rec["status"] = "EXPIRED"
            rec["result"] = "EXPIRED"
            rec["closed_time"] = now
            changed = True
    return changed


def _normalize_ticker(value: object, default: str = "BTCUSDT") -> str:
    text = str(value or "").strip().upper().replace("-", "")
    return text or default


def _result_from_close_reason(reason: str, pnl_pct: float) -> str:
    reason_l = str(reason or "").strip().lower()
    if reason_l in {"tp", "tp_hit", "tp1", "tp1_hit"}:
        return "TP1"
    if reason_l in {"tp2", "tp2_hit"}:
        return "TP2"
    if reason_l in {"tp3", "tp3_hit"}:
        return "TP3"
    if reason_l in {"sl", "stop", "sl_hit", "stop_loss"}:
        return "SL"
    if pnl_pct > 0:
        return "WIN"
    if pnl_pct < 0:
        return "LOSS"
    return "FLAT"


def _update_kelly_bucket_once(record: dict[str, Any], kelly_instance: Any | None = None) -> bool:
    """Feed a closed trade into warm-start Kelly exactly once."""
    if record.get("kelly_bucket_updated"):
        return False
    if str(record.get("status", "")).upper() in {"OPEN", "BLOCKED"}:
        return False
    if str(record.get("signal", "")).upper() not in {"LONG", "SHORT"}:
        return False
    try:
        if kelly_instance is None:
            from src.risk.kelly_warm_start import BTCKellyWarmStart

            kelly_instance = BTCKellyWarmStart(history_path=str(HISTORY_FILE))
        kelly_instance.update_bucket(dict(record))
        record["kelly_bucket_updated"] = True
        return True
    except Exception as exc:
        logger.debug("Kelly bucket update skipped: %s", exc)
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
    expired_open = _expire_stale_open_records(history, now_ts)

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
                        expired_open = True
                    break
    except Exception:
        pass

    # Avoid flooding history with repeated blocked records on every poll cycle.
    # Keep the latest timestamp/reason for an identical blocked state instead.
    try:
        if is_blocked and history:
            for _existing in reversed(history):
                if str(_existing.get("status", "")).upper() != "BLOCKED":
                    break
                same_dir = str(_existing.get("signal", "")).upper() == (signal_side or "HOLD")
                same_blocker = str(_existing.get("blocked_by") or "").strip() == blocked_by
                last_ts = _safe_float(_existing.get("open_timestamp") or _existing.get("closed_time"), 0.0)
                if same_dir and same_blocker and last_ts > 0 and (now_ts - last_ts) <= BLOCKED_DEDUP_WINDOW_SEC:
                    _existing["time"] = signal_data.get("as_of_utc") or _existing.get("time")
                    _existing["confidence"] = signal_data.get("confidence", _existing.get("confidence", 0))
                    _existing["alpha_score"] = signal_data.get("alpha_score", _existing.get("alpha_score", 0))
                    _existing["reason"] = signal_data.get("reason", _existing.get("reason", ""))
                    _existing["closed_time"] = now_ts
                    _existing["open_timestamp"] = now_ts
                    _save(history)
                    return
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
        "ticker": _normalize_ticker(signal_data.get("ticker"), "BTCUSDT"),
        "trade_id": str(signal_data.get("trade_id", "") or ""),
        "source": str(signal_data.get("source", "") or ""),
        "mode": str(signal_data.get("mode", "") or ""),
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


def record_closed_trade(trade_data: dict[str, Any]) -> None:
    """Append a CLOSED trade row directly into signal history."""
    signal_side = str(trade_data.get("signal") or trade_data.get("direction") or "").upper()
    if signal_side not in {"LONG", "SHORT"}:
        return

    history = _load()
    trade_id = str(trade_data.get("trade_id", "") or "").strip()
    ticker = _normalize_ticker(trade_data.get("ticker"), "BTCUSDT")
    source = str(trade_data.get("source", "") or "").strip()
    mode = str(trade_data.get("mode", "") or "").strip()

    entry = _safe_float(trade_data.get("entry_price") or trade_data.get("entry"), 0.0)
    exit_price = _safe_float(trade_data.get("exit_price"), 0.0)
    stop_loss = _safe_float(trade_data.get("stop_loss") or trade_data.get("sl"), 0.0)
    tp1 = _safe_float(trade_data.get("tp1") or trade_data.get("take_profit"), 0.0)
    tp2 = _safe_float(trade_data.get("tp2"), tp1)
    tp3 = _safe_float(trade_data.get("tp3"), tp2)
    risk_reward = _safe_float(trade_data.get("risk_reward"), 0.0)
    confidence = _safe_float(trade_data.get("confidence"), 0.0)
    alpha_score = _safe_float(trade_data.get("alpha_score"), confidence)

    pnl_pct = trade_data.get("pnl_pct")
    if pnl_pct is None:
        pnl_pct = _directional_pnl_pct(entry, exit_price, signal_side) if entry > 0 and exit_price > 0 else 0.0
    pnl_pct = round(_safe_float(pnl_pct, 0.0), 2)

    opened_ts = _parse_epoch_seconds(trade_data.get("opened_at") or trade_data.get("entry_time"))
    closed_ts = _parse_epoch_seconds(trade_data.get("closed_at") or trade_data.get("exit_time"))
    now_ts = time.time()
    if opened_ts <= 0:
        opened_ts = now_ts
    if closed_ts <= 0:
        closed_ts = now_ts
    if closed_ts < opened_ts:
        closed_ts = opened_ts

    result = _result_from_close_reason(str(trade_data.get("reason", "")), pnl_pct)

    if trade_id:
        for existing in history:
            if str(existing.get("trade_id", "")).strip() != trade_id:
                continue
            if str(existing.get("ticker", "")).strip().upper().replace("-", "") != ticker:
                continue
            existing.update(
                {
                    "status": "CLOSED",
                    "result": result,
                    "exit_price": round(exit_price, 2) if exit_price > 0 else existing.get("exit_price"),
                    "pnl_pct": pnl_pct,
                    "closed_time": closed_ts,
                    "reason": str(trade_data.get("reason", existing.get("reason", ""))),
                    "source": source or str(existing.get("source", "")),
                    "mode": mode or str(existing.get("mode", "")),
                }
            )
            _update_kelly_bucket_once(existing)
            _save(history)
            return

    direction = "long" if signal_side == "LONG" else "short"
    record = {
        "id": len(history) + 1,
        "ticker": ticker,
        "trade_id": trade_id,
        "source": source,
        "mode": mode,
        "time": trade_data.get("opened_at") or trade_data.get("entry_time"),
        "signal": signal_side,
        "confidence": confidence,
        "alpha_score": alpha_score,
        "entry": round(entry, 2) if entry > 0 else None,
        "stop_loss": round(stop_loss, 2) if stop_loss > 0 else None,
        "active_stop_loss": round(stop_loss, 2) if stop_loss > 0 else None,
        "tp1": round(tp1, 2) if tp1 > 0 else None,
        "tp2": round(tp2, 2) if tp2 > 0 else None,
        "tp3": round(tp3, 2) if tp3 > 0 else None,
        "risk_reward": round(risk_reward, 3) if risk_reward > 0 else None,
        "direction": direction,
        "type": signal_side,
        "regime": trade_data.get("regime"),
        "reason": str(trade_data.get("reason", "")),
        "blocked_by": None,
        "funding_rate": 0.0,
        "market_context": {},
        "trade_management": {},
        "tp1_hit": result == "TP1",
        "tp2_hit": result == "TP2",
        "tp3_hit": result == "TP3",
        "highest_price": None,
        "lowest_price": None,
        "milestones": [],
        "status": "CLOSED",
        "result": result,
        "exit_price": round(exit_price, 2) if exit_price > 0 else None,
        "pnl_pct": pnl_pct,
        "closed_time": closed_ts,
        "open_timestamp": opened_ts,
        "duration_seconds": max(0, int(round(closed_ts - opened_ts))),
    }
    _update_kelly_bucket_once(record)
    history.append(record)
    _save(history)


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
    changed = _expire_stale_open_records(history)
    live_signal = str(validated_signal or market_signal or "").upper()
    _kelly_instance = None

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

                _kelly_instance = BTCKellyWarmStart(history_path=str(HISTORY_FILE))
            if _update_kelly_bucket_once(rec, _kelly_instance):
                changed = True

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
