from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DrawdownConfig:
    soft_limit_pct: float = 0.04
    hard_limit_pct: float = 0.08
    daily_loss_limit_pct: float = 0.03
    recovery_trades: int = 5
    peak_update_min_trades: int = 3


class DrawdownController:
    """
    Real-time drawdown circuit breaker with persisted state.
    """

    def __init__(self, state_path: str, config: DrawdownConfig | None = None) -> None:
        self.config = config or DrawdownConfig()
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.equity_peak: float = 0.0
        self.current_equity: float = 0.0
        self.daily_start_equity: float = 0.0
        self.daily_date: str = ""
        self.halted: bool = False
        self.halt_reason: str | None = None
        self.trades_since_halt: int = 0
        self._updates_count: int = 0
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.equity_peak = float(payload.get("equity_peak", 0.0))
            self.current_equity = float(payload.get("current_equity", 0.0))
            self.daily_start_equity = float(payload.get("daily_start_equity", 0.0))
            self.daily_date = str(payload.get("daily_date", ""))
            self.halted = bool(payload.get("halted", False))
            raw_reason = payload.get("halt_reason")
            self.halt_reason = str(raw_reason) if raw_reason is not None else None
            self.trades_since_halt = int(max(0, payload.get("trades_since_halt", 0)))
            self._updates_count = int(max(0, payload.get("updates_count", 0)))
        except Exception:
            return

    def _save(self) -> None:
        payload = {
            "equity_peak": float(self.equity_peak),
            "current_equity": float(self.current_equity),
            "daily_start_equity": float(self.daily_start_equity),
            "daily_date": self.daily_date,
            "halted": bool(self.halted),
            "halt_reason": self.halt_reason,
            "trades_since_halt": int(self.trades_since_halt),
            "updates_count": int(self._updates_count),
        }
        try:
            self.state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            return

    @staticmethod
    def _day_from_timestamp(timestamp_utc: str) -> str:
        ts = str(timestamp_utc or "").strip()
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.date().isoformat()
            except Exception:
                pass
        return datetime.now(timezone.utc).date().isoformat()

    def _current_drawdown_pct(self) -> float:
        if self.equity_peak <= 0.0:
            return 0.0
        return float(max(0.0, (self.equity_peak - self.current_equity) / self.equity_peak))

    def _daily_loss_pct(self) -> float:
        if self.daily_start_equity <= 0.0:
            return 0.0
        return float(max(0.0, (self.daily_start_equity - self.current_equity) / self.daily_start_equity))

    def get_status(self) -> dict[str, Any]:
        drawdown = self._current_drawdown_pct()
        daily_loss = self._daily_loss_pct()
        if self.halted:
            action = "HALT"
            multiplier = 0.0
        elif drawdown > float(self.config.soft_limit_pct):
            action = "REDUCE_SIZE"
            multiplier = 0.5
        else:
            action = "NORMAL"
            multiplier = 1.0
        return {
            "current_drawdown_pct": float(drawdown),
            "daily_loss_pct": float(daily_loss),
            "halted": bool(self.halted),
            "halt_reason": self.halt_reason,
            "action": action,
            "size_multiplier": float(multiplier),
            "trades_since_halt": int(self.trades_since_halt),
            "equity_peak": float(self.equity_peak),
        }

    def update(self, current_equity: float, timestamp_utc: str) -> dict[str, Any]:
        self._updates_count += 1
        self.current_equity = float(current_equity)
        day = self._day_from_timestamp(timestamp_utc)

        if not self.daily_date or self.daily_date != day:
            self.daily_date = day
            self.daily_start_equity = self.current_equity
            if self.halted and self.halt_reason == "daily_loss_limit":
                self.halted = False
                self.halt_reason = None
                self.trades_since_halt = 0

        if self.equity_peak <= 0.0:
            self.equity_peak = self.current_equity
        if (
            self.current_equity > self.equity_peak
            and self._updates_count >= int(max(1, self.config.peak_update_min_trades))
        ):
            self.equity_peak = self.current_equity

        current_drawdown = self._current_drawdown_pct()
        daily_loss = self._daily_loss_pct()

        if self.halted:
            self.trades_since_halt += 1
            can_resume = (
                self.trades_since_halt >= int(max(1, self.config.recovery_trades))
                and current_drawdown < float(self.config.soft_limit_pct)
            )
            if can_resume:
                self.halted = False
                self.halt_reason = None
                self.trades_since_halt = 0
        else:
            if daily_loss > float(self.config.daily_loss_limit_pct):
                self.halted = True
                self.halt_reason = "daily_loss_limit"
                self.trades_since_halt = 0
            elif current_drawdown > float(self.config.hard_limit_pct):
                self.halted = True
                self.halt_reason = "hard_drawdown_limit"
                self.trades_since_halt = 0

        self._save()
        return self.get_status()
