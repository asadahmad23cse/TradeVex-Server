from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from btc_intelligence.services.meta_labeling import get_dynamic_confidence_gate

MAX_HOLD_SECONDS = 2 * 3600        # 2 hour hard cap per trade
SESSION_FORCE_EXIT_UTC = [7, 13]   # Force exit at London open (07:00) and NY open (13:00) UTC
TP_PARTIAL_RULES = [
    {"pct": 0.40, "at": "tp1"},
    {"pct": 0.35, "at": "tp2"},
]


@dataclass
class PaperPosition:
    direction: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    size_btc: float


class PaperTradeBook:
    def __init__(self) -> None:
        self.open_position: PaperPosition | None = None
        self.closed: list[dict[str, Any]] = []
        self._open: dict[str, dict[str, Any]] = {}
        self._open_trade_id: str | None = None
        self._next_trade_id = 0

    def _next_id(self) -> str:
        self._next_trade_id += 1
        return f"PAPER-{self._next_trade_id:06d}"

    @staticmethod
    def _move_pct(direction: str, entry: float, price: float) -> float:
        if entry <= 0:
            return 0.0
        if str(direction).upper() == "SHORT":
            return ((entry - price) / entry) * 100.0
        return ((price - entry) / entry) * 100.0

    def _force_close(self, tid: str, exit_price: float, now_utc: datetime, reason: str) -> dict[str, Any] | None:
        trade = self._open.get(tid)
        if not trade:
            return None
        pos: PaperPosition = trade["position"]
        direction = str(trade.get("direction", pos.direction)).upper()
        entry = float(trade.get("entry_price", pos.entry))
        filled_pct = float(trade.get("filled_pct", 0.0))
        partial_pnl = float(trade.get("partial_pnl", 0.0))
        remaining_pct = max(0.0, 1.0 - min(1.0, filled_pct))
        move_pct = self._move_pct(direction, entry, float(exit_price))
        realized_pnl_pct = float(partial_pnl + (remaining_pct * move_pct))
        notional_usd = entry * pos.size_btc
        pnl = (realized_pnl_pct / 100.0) * notional_usd
        if direction == "SHORT":
            effective_exit = entry * (1.0 - (realized_pnl_pct / 100.0))
        else:
            effective_exit = entry * (1.0 + (realized_pnl_pct / 100.0))
        row = {
            'trade_id': tid,
            'direction': direction,
            'entry': entry,
            'exit': float(effective_exit),
            'market_exit': float(exit_price),
            'size_btc': pos.size_btc,
            'pnl_usd': pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'filled_pct': min(1.0, filled_pct),
            'remaining_pct': remaining_pct,
            'partial_pnl_pct': partial_pnl,
            'reason': reason,
            'opened_at_utc': trade["opened_at_utc"].isoformat(),
            'closed_at_utc': now_utc.isoformat(),
        }
        self.closed.append(row)
        self._open.pop(tid, None)
        if self._open_trade_id == tid:
            self._open_trade_id = None
            self.open_position = None
        return row

    def open(
        self,
        direction: str,
        entry: float,
        stop: float,
        tp1: float,
        tp2: float,
        tp3: float,
        size_btc: float,
        confidence: float = 0.0,
        regime: str = "DEFAULT",
    ) -> bool:
        paper_min_confidence = get_dynamic_confidence_gate(regime)
        conf = float(confidence if confidence is not None else 0.0)
        if conf < paper_min_confidence:
            return False
        if self.open_position is not None:
            return False
        direction_clean = str(direction or "LONG").upper()
        self.open_position = PaperPosition(direction_clean, entry, stop, tp1, tp2, tp3, size_btc)
        now_utc = datetime.now(timezone.utc)
        trade_id = self._next_id()
        self._open_trade_id = trade_id
        self._open[trade_id] = {
            "position": self.open_position,
            "opened_at_utc": now_utc,
            "direction": direction_clean,
            "entry_price": float(entry),
            "sl": float(stop),
            "tp1": float(tp1),
            "filled_pct": 0.0,
            "partial_pnl": 0.0,
        }
        return True

    def close(self, exit_price: float, reason: str) -> dict[str, Any] | None:
        tid = self._open_trade_id
        if tid is None:
            return None
        return self._force_close(
            tid=tid,
            exit_price=float(exit_price),
            now_utc=datetime.now(timezone.utc),
            reason=reason,
        )

    def check_forced_exits(self, current_price: float, now_utc: datetime) -> list[str]:
        """Force-close stale/session-boundary positions."""
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        closed: list[str] = []
        for tid, trade in list(self._open.items()):
            age = (now_utc - trade["opened_at_utc"]).total_seconds()
            hour = now_utc.hour

            # Rule 1: Hard max hold
            if age >= MAX_HOLD_SECONDS:
                self._force_close(tid, current_price, now_utc, reason="MAX_HOLD_TIMEOUT")
                closed.append(tid)
                continue

            # Rule 2: Session boundary exit
            minute = now_utc.minute
            if hour in SESSION_FORCE_EXIT_UTC and minute < 5 and age > 900:
                self._force_close(tid, current_price, now_utc, reason="SESSION_BOUNDARY_EXIT")
                closed.append(tid)
        return closed

    def check_partial_exits(self, tid: str, current_price: float, now_utc: datetime):
        trade = self._open.get(tid)
        if not trade:
            return

        filled_pct = float(trade.get("filled_pct", 0.0))
        entry = float(trade.get("entry_price", 0.0))
        tp1 = float(trade.get("tp1", entry))
        direction = str(trade.get("direction", "LONG")).upper()
        if entry <= 0:
            return

        tp1_dist = abs(tp1 - entry)
        tp2 = tp1 + (tp1_dist * 0.5) if direction == "LONG" else tp1 - (tp1_dist * 0.5)
        trade["tp2_dynamic"] = float(tp2)

        move_pct = self._move_pct(direction, entry, float(current_price))
        partial_pnl = float(trade.get("partial_pnl", 0.0))

        # TP1 hit -> take 40%, move SL to breakeven.
        if filled_pct < TP_PARTIAL_RULES[0]["pct"]:
            hit = (direction == "LONG" and current_price >= tp1) or (direction == "SHORT" and current_price <= tp1)
            if hit:
                trade["filled_pct"] = float(TP_PARTIAL_RULES[0]["pct"])
                trade["sl"] = float(entry)
                trade["partial_pnl"] = partial_pnl + (TP_PARTIAL_RULES[0]["pct"] * move_pct)

        # TP2 hit -> take 35%, move SL to TP1 (trail remaining 25%).
        elif filled_pct < (TP_PARTIAL_RULES[0]["pct"] + TP_PARTIAL_RULES[1]["pct"]):
            hit = (direction == "LONG" and current_price >= tp2) or (direction == "SHORT" and current_price <= tp2)
            if hit:
                trade["filled_pct"] = float(TP_PARTIAL_RULES[0]["pct"] + TP_PARTIAL_RULES[1]["pct"])
                trade["partial_pnl"] = partial_pnl + (TP_PARTIAL_RULES[1]["pct"] * move_pct)
                trade["sl"] = float(tp1)

    def mark_to_market(self, price: float) -> dict[str, Any] | None:
        now_utc = datetime.now(timezone.utc)
        forced = self.check_forced_exits(current_price=float(price), now_utc=now_utc)
        if forced:
            return self.closed[-1] if self.closed else None

        pos = self.open_position
        if pos is None:
            return None
        tid = self._open_trade_id
        if tid is None:
            return None
        self.check_partial_exits(tid=tid, current_price=float(price), now_utc=now_utc)
        trade = self._open.get(tid)
        if not trade:
            return None

        direction = str(trade.get("direction", pos.direction)).upper()
        sl = float(trade.get("sl", pos.stop))
        entry = float(trade.get("entry_price", pos.entry))
        filled_pct = float(trade.get("filled_pct", 0.0))

        if direction == 'LONG':
            if price <= sl:
                if filled_pct >= 0.75:
                    reason = 'trailing_stop'
                elif filled_pct >= 0.40 and sl >= entry:
                    reason = 'breakeven_stop'
                else:
                    reason = 'stop'
                return self._force_close(tid=tid, exit_price=float(sl), now_utc=now_utc, reason=reason)
            if price >= pos.tp3:
                return self._force_close(tid=tid, exit_price=float(pos.tp3), now_utc=now_utc, reason='tp3')
        else:
            if price >= sl:
                if filled_pct >= 0.75:
                    reason = 'trailing_stop'
                elif filled_pct >= 0.40 and sl <= entry:
                    reason = 'breakeven_stop'
                else:
                    reason = 'stop'
                return self._force_close(tid=tid, exit_price=float(sl), now_utc=now_utc, reason=reason)
            if price <= pos.tp3:
                return self._force_close(tid=tid, exit_price=float(pos.tp3), now_utc=now_utc, reason='tp3')
        return None
