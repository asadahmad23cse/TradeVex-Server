"""Expiry tracking utilities for NSE F&O symbols."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


def _weekday_from_name(name: str) -> int:
    mapping = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    return mapping.get(name.lower().strip(), 3)


def _next_weekday(start: date, target_weekday: int) -> date:
    days_ahead = (target_weekday - start.weekday()) % 7
    return start + timedelta(days=days_ahead)


def _last_weekday_of_month(year: int, month: int, target_weekday: int) -> date:
    # Start from first day of next month, then step backwards.
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != target_weekday:
        d -= timedelta(days=1)
    return d


@dataclass
class ExpiryTracker:
    """Tracks weekly/monthly expiry conventions for NSE index derivatives."""

    NIFTY_EXPIRY: str = "thursday"
    BANKNIFTY_EXPIRY: str = "wednesday"
    FINNIFTY_EXPIRY: str = "tuesday"

    def _target_weekday(self, symbol: str) -> int:
        sym = (symbol or "").upper().strip()
        if "BANK" in sym:
            return _weekday_from_name(self.BANKNIFTY_EXPIRY)
        if "FIN" in sym:
            return _weekday_from_name(self.FINNIFTY_EXPIRY)
        return _weekday_from_name(self.NIFTY_EXPIRY)

    def get_next_expiry(self, symbol: str) -> date:
        today = date.today()
        wd = self._target_weekday(symbol)
        next_expiry = _next_weekday(today, wd)
        return next_expiry

    def get_days_to_expiry(self, symbol: str) -> int:
        return max((self.get_next_expiry(symbol) - date.today()).days, 0)

    def is_expiry_week(self, symbol: str) -> bool:
        return self.get_days_to_expiry(symbol) <= 5

    def is_expiry_day(self, symbol: str) -> bool:
        return self.get_days_to_expiry(symbol) == 0

    def get_monthly_expiry(self, month: int, year: int) -> date:
        # Monthly expiry (Nifty convention): last Thursday of month.
        return _last_weekday_of_month(year, month, _weekday_from_name(self.NIFTY_EXPIRY))
