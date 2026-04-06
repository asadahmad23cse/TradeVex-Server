from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TradeThrottleConfig:
    max_trades_per_hour: int = 4
    min_cooldown_seconds: int = 180
    chop_cooldown_seconds: int = 600
    loss_streak_cooldown: int = 900


class TradeThrottle:
    def __init__(self, config: TradeThrottleConfig = TradeThrottleConfig()):
        self.cfg = config
        self._trade_times: deque[datetime] = deque(maxlen=20)
        self._last_trade_utc: datetime | None = None
        self._consecutive_losses: int = 0

    def record_trade(self, pnl_pct: float, now: datetime) -> None:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        self._trade_times.append(now)
        self._last_trade_utc = now
        if pnl_pct < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def can_trade(self, regime: str, now: datetime) -> tuple[bool, str]:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if self._last_trade_utc:
            elapsed = (now - self._last_trade_utc).total_seconds()

            if self._consecutive_losses >= 3 and elapsed < self.cfg.loss_streak_cooldown:
                return False, f"loss_streak_cooldown ({self._consecutive_losses} losses)"

            is_chop = any(x in regime.upper() for x in ["RANGE", "SQUEEZE", "SIDEWAYS"])
            cooldown = self.cfg.chop_cooldown_seconds if is_chop else self.cfg.min_cooldown_seconds
            if elapsed < cooldown:
                return False, f"cooldown ({int(cooldown - elapsed)}s remaining)"

        hour_ago = now.timestamp() - 3600
        recent = sum(1 for t in self._trade_times if t.timestamp() > hour_ago)
        if recent >= self.cfg.max_trades_per_hour:
            return False, f"hourly_cap ({recent}/{self.cfg.max_trades_per_hour} trades)"

        return True, "ok"
