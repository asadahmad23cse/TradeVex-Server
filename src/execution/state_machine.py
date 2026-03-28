"""
Order state machine for live/paper execution tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

VALID_TRANSITIONS = {
    "PENDING": {"SUBMITTED", "REJECTED", "EXPIRED", "CANCELLED", "FILLED", "PARTIAL"},
    "SUBMITTED": {"PARTIAL", "FILLED", "REJECTED", "CANCELLED", "EXPIRED"},
    "PARTIAL": {"FILLED", "CANCELLED", "EXPIRED", "REJECTED"},
    "FILLED": set(),
    "REJECTED": set(),
    "CANCELLED": set(),
    "EXPIRED": set(),
}


@dataclass
class OrderStateRecord:
    order_id: str
    signal_id: str
    asset: str
    side: str
    broker: str
    state: str = "PENDING"
    requested_price: float = 0.0
    fill_price: float = 0.0
    requested_qty: float = 0.0
    filled_qty: float = 0.0
    error: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def transition(self, new_state: str) -> None:
        current = self.state.upper()
        target = new_state.upper()
        if target not in VALID_TRANSITIONS.get(current, set()) and current != target:
            raise ValueError(f"Invalid order state transition: {current} -> {target}")
        self.state = target
        self.updated_at = datetime.utcnow()
