"""
NSEOptionsChain — fetches and parses NSE option chain data.

Safety contract:
- All public methods return None on any failure
- Never raises exceptions to caller
- Respects 5-minute TTL cache (max 1 request/symbol/5min)
- NSE session established with standard browser headers
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from src.api.rate_limiter import TTLCache

logger = logging.getLogger("api.options_chain")

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
    "X-Requested-With": "XMLHttpRequest",
}

_CACHE_TTL_SEC = 300
_SESSION_TO = 10
_INDEX_SYMS = frozenset(
    {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}
)


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        if v != v or abs(v) == float("inf"):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class NSEOptionsChain:
    """
    Fetches NSE option chain for indices and equities.
    Uses TTLCache (thread-safe) for chain payloads per symbol.
    """

    def __init__(self, cache_ttl_seconds: int = _CACHE_TTL_SEC) -> None:
        self._session: Optional[requests.Session] = None
        self._cache = TTLCache()
        self._cache_ttl = int(cache_ttl_seconds)

    def _get_session(self) -> Optional[requests.Session]:
        if self._session is not None:
            return self._session
        try:
            sess = requests.Session()
            sess.headers.update(_NSE_HEADERS)
            sess.get("https://www.nseindia.com/", timeout=_SESSION_TO)
            self._session = sess
            return self._session
        except Exception as exc:
            logger.warning("NSE session creation failed: %s", exc)
            return None

    def _invalidate_session(self) -> None:
        self._session = None

    @staticmethod
    def _nse_symbol(ticker: str) -> str:
        sym = (ticker or "").upper().strip()
        for suffix in (".NS", ".BO", ".BSE"):
            if sym.endswith(suffix):
                sym = sym[: -len(suffix)]
                break
        return sym

    @staticmethod
    def _is_index(symbol: str) -> bool:
        return symbol in _INDEX_SYMS

    def fetch_chain(self, ticker: str) -> Optional[dict]:
        symbol = self._nse_symbol(ticker)
        if not symbol:
            return None

        cached = self._cache.get(symbol)
        if cached is not None:
            return cached  # type: ignore[return-value]

        sess = self._get_session()
        if sess is None:
            return None

        try:
            if self._is_index(symbol):
                url = (
                    "https://www.nseindia.com/api/"
                    f"option-chain-indices?symbol={symbol}"
                )
            else:
                url = (
                    "https://www.nseindia.com/api/"
                    f"option-chain-equities?symbol={symbol}"
                )

            resp = sess.get(url, timeout=_SESSION_TO)

            if resp.status_code in (401, 403):
                logger.warning(
                    "NSE session expired for %s (HTTP %d) — invalidating session",
                    symbol,
                    resp.status_code,
                )
                self._invalidate_session()
                return None

            if resp.status_code != 200:
                logger.warning(
                    "NSE chain fetch failed for %s: HTTP %d",
                    symbol,
                    resp.status_code,
                )
                return None

            raw = resp.json()
            parsed = self._parse_chain(raw, symbol)
            if parsed is not None:
                self._cache.set(symbol, parsed, self._cache_ttl)
            return parsed

        except requests.exceptions.Timeout:
            logger.warning("NSE chain timeout for %s", symbol)
            return None
        except Exception as exc:
            logger.warning("NSE chain fetch error for %s: %s", symbol, exc)
            self._invalidate_session()
            return None

    def _parse_chain(self, raw: dict, symbol: str) -> Optional[dict]:
        try:
            records = raw.get("records") or {}
            underlying = _safe_float(records.get("underlyingValue"), 0.0)
            if underlying <= 0:
                underlying = _safe_float(
                    (raw.get("filtered") or {}).get("underlyingValue"), 0.0
                )

            expiry_dates = list(records.get("expiryDates") or [])
            data_rows = records.get("data") or []

            strikes: dict = {}
            for row in data_rows:
                strike = row.get("strikePrice")
                if strike is None:
                    continue
                strike_i = _safe_int(strike, -1)
                if strike_i < 0:
                    continue
                strikes[strike_i] = {}

                for opt_type in ("CE", "PE"):
                    opt = row.get(opt_type) or {}
                    if not opt:
                        continue
                    strikes[strike_i][opt_type] = {
                        "iv": _safe_float(opt.get("impliedVolatility"), 0.0),
                        "oi": _safe_int(opt.get("openInterest"), 0),
                        "volume": _safe_int(opt.get("totalTradedVolume"), 0),
                        "ltp": _safe_float(opt.get("lastPrice"), 0.0),
                    }

            if not strikes or underlying <= 0:
                logger.warning("NSE chain empty for %s", symbol)
                return None

            return {
                "symbol": symbol,
                "underlying_price": underlying,
                "expiry_dates": expiry_dates,
                "strikes": strikes,
            }

        except Exception as exc:
            logger.warning("NSE chain parse error for %s: %s", symbol, exc)
            return None

    def get_nearest_expiry_chain(self, ticker: str) -> Optional[dict]:
        return self.fetch_chain(ticker)
