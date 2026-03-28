"""Tracks signal outcomes for paper trading and win rate monitoring."""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_FILE = Path("data/signal_history.json")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def _save(history: list[dict]):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def record_signal(signal_data: dict):
    """Save a new signal when it fires (LONG/SHORT only, not HOLD)."""
    if signal_data.get("signal") not in ("LONG", "SHORT"):
        return
    if not signal_data.get("validated"):
        return

    history = _load()
    # Block duplicate: skip if same direction + same entry within 4 hours
    if history:
        last = history[-1]
        if (
            last["signal"] == signal_data["signal"]
            and last.get("entry") == signal_data.get("entry_price")
            and time.time() - last.get("open_timestamp", 0) < 4 * 3600
        ):
            return

    record = {
        "id": len(history) + 1,
        "time": signal_data.get("as_of_utc"),
        "signal": signal_data["signal"],
        "confidence": signal_data.get("confidence", 0),
        "alpha_score": signal_data.get("alpha_score", 0),
        "entry": signal_data.get("entry_price"),
        "stop_loss": signal_data.get("stop_loss"),
        "tp1": signal_data.get("tp1"),
        "tp2": signal_data.get("tp2"),
        "tp3": signal_data.get("tp3"),
        "risk_reward": signal_data.get("risk_reward"),
        "regime": signal_data.get("regime"),
        "reason": signal_data.get("reason", ""),
        "funding_rate": signal_data.get("funding_rate_pct", 0),
        # Outcome tracking (updated later by price checker)
        "status": "OPEN",  # OPEN -> TP1_HIT / TP2_HIT / TP3_HIT / SL_HIT / EXPIRED
        "exit_price": None,
        "pnl_pct": None,
        "closed_time": None,
        "open_timestamp": time.time(),
    }
    history.append(record)
    _save(history)
    logger.info("Signal recorded: #%d %s @ %.2f", record["id"], record["signal"], record["entry"])


def check_open_signals(current_price: float):
    """Check all OPEN signals against current price for TP/SL hits."""
    history = _load()
    changed = False

    for rec in history:
        if rec["status"] != "OPEN":
            continue
        if rec["entry"] is None:
            continue

        entry = rec["entry"]
        sl = rec.get("stop_loss")
        tp1 = rec.get("tp1")
        tp2 = rec.get("tp2")
        tp3 = rec.get("tp3")
        sig = rec["signal"]

        # Check SL hit
        if sl and ((sig == "LONG" and current_price <= sl) or (sig == "SHORT" and current_price >= sl)):
            rec["status"] = "SL_HIT"
            rec["exit_price"] = current_price
            rec["pnl_pct"] = round(((current_price - entry) / entry) * 100 * (1 if sig == "LONG" else -1), 2)
            rec["closed_time"] = time.time()
            changed = True
            continue

        # Check TP3 first (best outcome)
        if tp3 and ((sig == "LONG" and current_price >= tp3) or (sig == "SHORT" and current_price <= tp3)):
            rec["status"] = "TP3_HIT"
            rec["exit_price"] = tp3
            rec["pnl_pct"] = round(((tp3 - entry) / entry) * 100 * (1 if sig == "LONG" else -1), 2)
            rec["closed_time"] = time.time()
            changed = True
        elif tp2 and ((sig == "LONG" and current_price >= tp2) or (sig == "SHORT" and current_price <= tp2)):
            rec["status"] = "TP2_HIT"
            rec["exit_price"] = tp2
            rec["pnl_pct"] = round(((tp2 - entry) / entry) * 100 * (1 if sig == "LONG" else -1), 2)
            rec["closed_time"] = time.time()
            changed = True
        elif tp1 and ((sig == "LONG" and current_price >= tp1) or (sig == "SHORT" and current_price <= tp1)):
            rec["status"] = "TP1_HIT"
            rec["exit_price"] = tp1
            rec["pnl_pct"] = round(((tp1 - entry) / entry) * 100 * (1 if sig == "LONG" else -1), 2)
            rec["closed_time"] = time.time()
            changed = True

        # Expire after 48 candles (12 hours on 15m)
        if rec["status"] == "OPEN" and time.time() - rec["open_timestamp"] > 12 * 3600:
            rec["status"] = "EXPIRED"
            rec["exit_price"] = current_price
            rec["pnl_pct"] = round(((current_price - entry) / entry) * 100 * (1 if sig == "LONG" else -1), 2)
            rec["closed_time"] = time.time()
            changed = True

    if changed:
        _save(history)


def get_history(limit: int = 50) -> list[dict]:
    return _load()[-limit:]


def get_stats() -> dict:
    history = _load()
    closed = [h for h in history if h["status"] != "OPEN"]
    if not closed:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}

    wins = [h for h in closed if h.get("pnl_pct", 0) > 0]
    losses = [h for h in closed if h.get("pnl_pct", 0) <= 0]
    total_pnl = sum(h.get("pnl_pct", 0) for h in closed)

    return {
        "total": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(closed), 2) if closed else 0,
        "open_signals": len([h for h in history if h["status"] == "OPEN"]),
    }
