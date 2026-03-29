"""Post-entry trade management for BTC signal records."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

MAX_LOSS_CAP_PCT = -2.0


class TradeManager:
    """Manages OPEN trade records across breakeven, trailing, and exits."""

    def __init__(self) -> None:
        self._flip_count: dict[str, int] = {}

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _is_opposite(signal: str, current_signal: str) -> bool:
        s = str(signal or "").upper()
        c = str(current_signal or "").upper()
        return (s == "LONG" and c == "SHORT") or (s == "SHORT" and c == "LONG")

    @staticmethod
    def _is_stop_hit(signal: str, current_price: float, stop_price: float) -> bool:
        s = str(signal).upper()
        if s == "LONG":
            return current_price <= stop_price
        return current_price >= stop_price

    @staticmethod
    def _directional_pct(entry: float, exit_price: float, signal: str) -> float:
        if entry <= 0:
            return 0.0
        raw = ((exit_price - entry) / entry) * 100.0
        return raw if str(signal).upper() == "LONG" else -raw

    @staticmethod
    def _status_result(status: str) -> str:
        status_u = str(status).upper()
        mapping = {
            "SL_HIT": "SL",
            "TP1_HIT": "TP1",
            "TP2_HIT": "TP2",
            "TP3_HIT": "TP3",
            "BREAKEVEN_HIT": "BREAKEVEN",
            "TRAIL_STOP_HIT": "TRAIL_STOP",
            "ALPHA_FLIP_EXIT": "ALPHA_FLIP_EXIT",
            "EXPIRED": "EXPIRED",
            "OPEN": "OPEN",
        }
        return mapping.get(status_u, status_u)

    @classmethod
    def _compute_pnl_pct(cls, rec: dict[str, Any], status: str, exit_price: float) -> float:
        entry = cls._safe_float(rec.get("entry"), 0.0)
        stop = cls._safe_float(rec.get("stop_loss"), 0.0)
        rr = cls._safe_float(rec.get("risk_reward"), 2.0)
        if rr <= 0:
            rr = 2.0

        sl_distance = abs(entry - stop) / entry if entry > 0 and stop > 0 else 0.0
        status_u = str(status).upper()
        if status_u == "SL_HIT":
            pnl = -(sl_distance * 100.0) if sl_distance > 0 else cls._directional_pct(entry, exit_price, rec.get("signal", "LONG"))
            return max(float(pnl), MAX_LOSS_CAP_PCT)
        if status_u == "BREAKEVEN_HIT":
            return 0.0
        if status_u == "TRAIL_STOP_HIT":
            return cls._directional_pct(entry, exit_price, rec.get("signal", "LONG"))
        if status_u == "ALPHA_FLIP_EXIT":
            return cls._directional_pct(entry, exit_price, rec.get("signal", "LONG"))
        if status_u == "EXPIRED":
            return cls._directional_pct(entry, exit_price, rec.get("signal", "LONG"))
        if status_u == "TP1_HIT":
            return sl_distance * rr * 100.0
        if status_u == "TP2_HIT":
            return sl_distance * (rr + 0.5) * 100.0
        if status_u == "TP3_HIT":
            return sl_distance * (rr + 1.0) * 100.0
        return cls._directional_pct(entry, exit_price, rec.get("signal", "LONG"))

    def _close(self, rec: dict[str, Any], status: str, exit_price: float, exit_reason: str | None = None) -> None:
        rec["status"] = str(status).upper()
        rec["result"] = self._status_result(status)
        rec["exit_price"] = round(float(exit_price), 2)
        rec["pnl_pct"] = round(self._compute_pnl_pct(rec, status, float(exit_price)), 2)
        rec["closed_time"] = time.time()
        if exit_reason:
            rec["exit_reason"] = str(exit_reason)

    def manage(self, record: dict, current_price: float, current_alpha: float, current_signal: str) -> dict:
        """
        Takes one OPEN signal record + current market state.
        Returns updated record with action taken.
        """
        rec = dict(record)
        if str(rec.get("status", "")).upper() != "OPEN":
            return rec

        signal = str(rec.get("signal", "LONG")).upper()
        if signal not in {"LONG", "SHORT"}:
            return rec

        entry = self._safe_float(rec.get("entry"), 0.0)
        if entry <= 0:
            return rec

        rec_id = str(rec.get("id", ""))
        now_ts = time.time()
        tp1 = self._safe_float(rec.get("tp1"), 0.0)
        tp2 = self._safe_float(rec.get("tp2"), 0.0)
        tp3 = self._safe_float(rec.get("tp3"), 0.0)
        current_price_f = float(current_price)

        active_stop = self._safe_float(rec.get("active_stop_loss"), self._safe_float(rec.get("stop_loss"), 0.0))
        rec.setdefault("milestones", [])
        rec.setdefault("tp1_hit", False)
        rec.setdefault("tp2_hit", False)
        rec.setdefault("tp3_hit", False)
        rec.setdefault("breakeven_activated", False)
        rec.setdefault("trailing_active", False)
        rec["highest_price"] = max(self._safe_float(rec.get("highest_price"), entry), current_price_f)
        rec["lowest_price"] = min(self._safe_float(rec.get("lowest_price"), entry), current_price_f)

        # Track TP milestones first.
        if not bool(rec.get("tp1_hit")) and tp1 > 0:
            if (signal == "LONG" and current_price_f >= tp1) or (signal == "SHORT" and current_price_f <= tp1):
                rec["tp1_hit"] = True
                rec["milestones"].append({"event": "TP1_HIT", "price": round(tp1, 2), "time": now_ts})
        if not bool(rec.get("tp2_hit")) and tp2 > 0:
            if (signal == "LONG" and current_price_f >= tp2) or (signal == "SHORT" and current_price_f <= tp2):
                rec["tp2_hit"] = True
                rec["milestones"].append({"event": "TP2_HIT", "price": round(tp2, 2), "time": now_ts})
        if not bool(rec.get("tp3_hit")) and tp3 > 0:
            if (signal == "LONG" and current_price_f >= tp3) or (signal == "SHORT" and current_price_f <= tp3):
                rec["tp3_hit"] = True
                rec["milestones"].append({"event": "TP3_HIT", "price": round(tp3, 2), "time": now_ts})

        # RULE 1 — Breakeven after TP1.
        if bool(rec.get("tp1_hit")) and not bool(rec.get("breakeven_activated")):
            if signal == "LONG":
                active_stop = entry + (entry * 0.0005)
            else:
                active_stop = entry - (entry * 0.0005)
            rec["active_stop_loss"] = round(active_stop, 2)
            rec["breakeven_activated"] = True
            rec["milestone"] = "BREAKEVEN_SET"
            logger.info("Breakeven set at %.2f for signal id=%s", active_stop, rec_id)

        # RULE 2 — Trailing stop after TP2.
        if bool(rec.get("tp2_hit")):
            atr = self._safe_float(rec.get("atr"), 0.0)
            trail_gap = (atr * 1.5) if atr > 0 else (entry * 0.006)
            if signal == "LONG":
                new_stop = current_price_f - trail_gap
                active_stop = max(active_stop, new_stop) if active_stop > 0 else new_stop
            else:
                new_stop = current_price_f + trail_gap
                active_stop = min(active_stop, new_stop) if active_stop > 0 else new_stop
            rec["active_stop_loss"] = round(active_stop, 2)
            rec["trailing_active"] = True
            logger.info("Trail stop updated to %.2f for signal id=%s", active_stop, rec_id)

        # RULE 3 — Alpha flip early exit.
        if self._is_opposite(signal, current_signal) and self._safe_float(current_alpha, 0.0) > 70.0:
            flip_count = self._flip_count.get(rec_id, 0) + 1
            self._flip_count[rec_id] = flip_count
            if flip_count >= 2:
                self._close(
                    rec,
                    "ALPHA_FLIP_EXIT",
                    current_price_f,
                    exit_reason=f"alpha flipped to {str(current_signal).upper()}",
                )
                logger.info("Alpha flip exit at %.2f for signal id=%s", current_price_f, rec_id)
                self._flip_count.pop(rec_id, None)
                return rec
        else:
            self._flip_count[rec_id] = 0

        # RULE 4 — Time-based expiry.
        open_ts = self._safe_float(rec.get("open_timestamp"), 0.0)
        no_tp_hit = not bool(rec.get("tp1_hit")) and not bool(rec.get("tp2_hit")) and not bool(rec.get("tp3_hit"))
        if open_ts > 0 and (now_ts - open_ts) > (12 * 3600) and no_tp_hit:
            self._close(rec, "EXPIRED", current_price_f, exit_reason="12h time limit, no TP reached")
            self._flip_count.pop(rec_id, None)
            return rec

        # RULE 5 — Active stop loss check.
        if active_stop > 0 and self._is_stop_hit(signal, current_price_f, active_stop):
            if bool(rec.get("trailing_active")):
                status = "TRAIL_STOP_HIT"
            elif bool(rec.get("breakeven_activated")):
                status = "BREAKEVEN_HIT"
            else:
                status = "SL_HIT"
            self._close(rec, status, active_stop)
            self._flip_count.pop(rec_id, None)
            return rec

        return rec
