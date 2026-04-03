from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    ) -> bool:
        PAPER_MIN_CONFIDENCE = 55.0
        conf = float(confidence if confidence is not None else 0.0)
        if conf < PAPER_MIN_CONFIDENCE:
            return False
        if self.open_position is not None:
            return False
        self.open_position = PaperPosition(direction, entry, stop, tp1, tp2, tp3, size_btc)
        return True

    def close(self, exit_price: float, reason: str) -> dict[str, Any] | None:
        pos = self.open_position
        if pos is None:
            return None
        pnl = (exit_price - pos.entry) * pos.size_btc if pos.direction == 'LONG' else (pos.entry - exit_price) * pos.size_btc
        row = {
            'direction': pos.direction,
            'entry': pos.entry,
            'exit': exit_price,
            'size_btc': pos.size_btc,
            'pnl_usd': pnl,
            'reason': reason,
        }
        self.closed.append(row)
        self.open_position = None
        return row

    def mark_to_market(self, price: float) -> dict[str, Any] | None:
        pos = self.open_position
        if pos is None:
            return None
        if pos.direction == 'LONG':
            if price <= pos.stop:
                return self.close(pos.stop, 'stop')
            if price >= pos.tp3:
                return self.close(pos.tp3, 'tp3')
            if price >= pos.tp2:
                return self.close(pos.tp2, 'tp2')
            if price >= pos.tp1:
                return self.close(pos.tp1, 'tp1')
        else:
            if price >= pos.stop:
                return self.close(pos.stop, 'stop')
            if price <= pos.tp3:
                return self.close(pos.tp3, 'tp3')
            if price <= pos.tp2:
                return self.close(pos.tp2, 'tp2')
            if price <= pos.tp1:
                return self.close(pos.tp1, 'tp1')
        return None
