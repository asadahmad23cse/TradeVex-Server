from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def validate_candle(candle: dict[str, Any], max_spike_pct: float = 5.0) -> bool:
    open_px = float(candle.get('open', 0.0))
    high = float(candle.get('high', 0.0))
    low = float(candle.get('low', 0.0))
    close = float(candle.get('close', 0.0))
    volume = float(candle.get('volume', 0.0))

    if min(open_px, high, low, close) <= 0 or volume <= 0:
        return False

    if open_px > 0 and abs(close - open_px) / open_px * 100.0 > max_spike_pct:
        return False

    if not (low <= min(open_px, close) and high >= max(open_px, close)):
        return False

    return True


def in_range(value: float, low: float, high: float) -> bool:
    return low <= value <= high
