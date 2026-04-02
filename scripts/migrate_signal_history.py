"""One-time migration for BTC signal history schema and P&L recalculation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_PATH = Path("data/signal_history.json")
MAX_LOSS_CAP_PCT = -2.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_result(record: dict) -> str:
    result = str(record.get("result", "")).upper()
    if result in {"SL", "TP1", "TP2", "TP3", "EXPIRED", "BLOCKED"}:
        return result
    status = str(record.get("status", "")).upper()
    status_map = {
        "SL_HIT": "SL",
        "TP1_HIT": "TP1",
        "TP2_HIT": "TP2",
        "TP3_HIT": "TP3",
        "EXPIRED": "EXPIRED",
        "BLOCKED": "BLOCKED",
    }
    return status_map.get(status, "OPEN")


def _directional_pnl(entry: float, exit_price: float, signal: str) -> float:
    if entry <= 0:
        return 0.0
    base = ((exit_price - entry) / entry) * 100.0
    return base if signal == "LONG" else -base


def _capped_loss(pnl_pct: float) -> float:
    return max(float(pnl_pct), MAX_LOSS_CAP_PCT)


def _recompute_pnl(record: dict) -> float | None:
    signal = str(record.get("signal", "")).upper()
    if signal not in {"LONG", "SHORT"}:
        return None

    result = _normalize_result(record)
    if result in {"OPEN", "BLOCKED"}:
        return None

    entry = _safe_float(record.get("entry"), 0.0)
    sl_price = _safe_float(record.get("stop_loss"), 0.0)
    if entry <= 0 or sl_price <= 0:
        exit_price = _safe_float(record.get("exit_price"), 0.0)
        if exit_price > 0:
            pnl_fallback = _directional_pnl(entry, exit_price, signal)
            return round(_capped_loss(pnl_fallback) if pnl_fallback < 0 else pnl_fallback, 2)
        return None

    sl_distance_pct = abs(sl_price - entry) / entry
    rr_ratio = _safe_float(record.get("risk_reward"), 2.0)
    if rr_ratio <= 0:
        rr_ratio = 2.0

    if result == "SL":
        return round(_capped_loss(-(sl_distance_pct * 100.0)), 2)
    if result == "TP1":
        if signal == "SHORT":
            return round(sl_distance_pct * 2.0 * 100.0, 2)
        return round(sl_distance_pct * rr_ratio * 100.0, 2)
    if result == "TP2":
        if signal == "SHORT":
            return round(sl_distance_pct * 3.0 * 100.0, 2)
        return round(sl_distance_pct * (rr_ratio + 0.5) * 100.0, 2)
    if result == "TP3":
        if signal == "SHORT":
            return round(sl_distance_pct * 4.0 * 100.0, 2)
        return round(sl_distance_pct * (rr_ratio + 1.0) * 100.0, 2)

    exit_price = _safe_float(record.get("exit_price"), 0.0)
    if exit_price > 0:
        pnl_fallback = _directional_pnl(entry, exit_price, signal)
        return round(_capped_loss(pnl_fallback) if pnl_fallback < 0 else pnl_fallback, 2)
    return None


def main() -> int:
    if not HISTORY_PATH.exists():
        print(f"No history file found at {HISTORY_PATH}")
        return 1

    data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("History file format is invalid (expected JSON array).")
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = HISTORY_PATH.parent / f"signal_history_backup_{ts}.json"
    backup_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    migrated = 0
    for rec in data:
        if not isinstance(rec, dict):
            continue
        signal = str(rec.get("signal", "")).upper()
        status = str(rec.get("status", "")).upper()

        if signal == "LONG":
            rec["direction"] = "long"
            rec["type"] = "BLOCKED" if status == "BLOCKED" else "LONG"
        elif signal == "SHORT":
            rec["direction"] = "short"
            rec["type"] = "BLOCKED" if status == "BLOCKED" else "SHORT"
        else:
            rec["direction"] = "flat"
            rec["type"] = "BLOCKED" if status == "BLOCKED" else "UNKNOWN"

        result = _normalize_result(rec)
        rec["result"] = result

        pnl = _recompute_pnl(rec)
        rec["pnl_pct"] = pnl
        migrated += 1

    HISTORY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Migrated {migrated} records, backup saved to {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
