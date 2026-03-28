"""
Token-bucket rate limiter for Alpha Vantage API.
Enforces: 5 requests/minute + 500 requests/day hard limit.

Also provides a TTL in-memory cache so repeated calls for the same
ticker within the TTL window don't consume API quota.
"""

import time
import threading
from datetime import date


class AVRateLimiter:
    """Thread-safe token-bucket limiter for Alpha Vantage."""

    def __init__(self, calls_per_minute: int = 5, daily_limit: int = 500):
        self.calls_per_minute = calls_per_minute
        self.daily_limit = daily_limit
        self._lock = threading.Lock()
        self._tokens: float = float(calls_per_minute)
        self._last_refill = time.monotonic()
        self._daily_count: int = 0
        self._last_reset_date = date.today()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self.calls_per_minute / 60.0)
        self._tokens = min(float(self.calls_per_minute), self._tokens + new_tokens)
        self._last_refill = now

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._last_reset_date:
            self._daily_count = 0
            self._last_reset_date = today

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """
        Block until a token is available, then consume one.
        Raises RuntimeError if the daily limit is exhausted.
        """
        with self._lock:
            self._reset_daily_if_needed()
            if self._daily_count >= self.daily_limit:
                raise RuntimeError(
                    f"Alpha Vantage daily limit of {self.daily_limit} requests reached. "
                    "Resets at midnight."
                )
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._daily_count += 1
                    return
                # Wait until the next token arrives
                wait_sec = (1.0 - self._tokens) / (self.calls_per_minute / 60.0)
                time.sleep(min(wait_sec, 1.0))

    @property
    def daily_used(self) -> int:
        with self._lock:
            self._reset_daily_if_needed()
            return self._daily_count

    @property
    def daily_remaining(self) -> int:
        with self._lock:
            self._reset_daily_if_needed()
            return self.daily_limit - self._daily_count


class TTLCache:
    """
    Simple in-memory TTL cache.
    Keys are strings (e.g. "AAPL_5m"), values are any object.
    """

    def __init__(self) -> None:
        self._store: dict = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        """Return cached value or None if missing / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.monotonic() < expiry:
                return value
            del self._store[key]
            return None

    def get_stale(self, key: str):
        """Return cached value even if expired, along with age in seconds."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expiry = entry
            age_seconds = max(time.monotonic() - expiry, 0.0)
            return value, age_seconds

    def set(self, key: str, value, ttl_seconds: int) -> None:
        """Store a value with a TTL."""
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl_seconds)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
