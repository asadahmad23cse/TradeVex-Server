"""
Layer 1 — Data Ingestion.

Priority routing:
  Indian stocks (intraday): NSE Scraper → yfinance .NS → stale cache
  Indian stocks (daily):    yfinance .NS
  US stocks (intraday):     yfinance
  US stocks (daily):        yfinance
  Forex (intraday):         yfinance =X tickers
  Forex (EOD):              Alpha Vantage FX_DAILY (uses AV budget)

Alpha Vantage budget: reserved for Forex EOD only (~5 req/day).
All intraday uses yfinance (free, no hard cap).
"""

import logging
import os
import time

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from src.api.rate_limiter import AVRateLimiter, TTLCache

load_dotenv()
logger = logging.getLogger(__name__)

# Shared singletons
_rate_limiter = AVRateLimiter(calls_per_minute=4, daily_limit=20, min_interval_seconds=15.0)
_cache = TTLCache()

# TTL constants (seconds)
TTL_INTRADAY = 240   # 4 minutes
TTL_DAILY = 3600     # 1 hour
TTL_FOREX = 240      # 4 minutes per spec


# ---------------------------------------------------------------------------
# NSE Scraper
# ---------------------------------------------------------------------------

class NSEScraper:
    """
    Scrapes NSE India public endpoints.
    Session is refreshed automatically on 401 / 403.
    """

    BASE = "https://www.nseindia.com"

    def __init__(self):
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "accept-language": "en-US,en;q=0.9",
            "accept-encoding": "gzip, deflate, br",
        }
        self._session = requests.Session()
        self._session.headers.update(self._headers)
        self._init_session()

    def _init_session(self) -> None:
        try:
            self._session.get(self.BASE, timeout=10)
        except Exception as exc:
            logger.warning("NSE session init failed: %s", exc)

    def _refresh_session(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(self._headers)
        self._init_session()

    def get_live_quote(self, symbol: str) -> dict | None:
        """
        Return {'price': float, 'change': float, 'pChange': float} or None.
        Auto-refreshes session on 401/403, retries once.
        """
        url = f"{self.BASE}/api/quote-equity?symbol={symbol}"
        for attempt in range(2):
            try:
                r = self._session.get(url, timeout=8)
                if r.status_code == 200:
                    data = r.json()
                    return {
                        "price": data["priceInfo"]["lastPrice"],
                        "change": data["priceInfo"]["change"],
                        "pChange": data["priceInfo"]["pChange"],
                    }
                if r.status_code in (401, 403) and attempt == 0:
                    logger.info("NSE 401/403 — refreshing session")
                    self._refresh_session()
                    continue
                logger.warning("NSE quote %s → HTTP %s", symbol, r.status_code)
                return None
            except Exception as exc:
                logger.warning("NSE quote %s error: %s", symbol, exc)
                return None
        return None

    def get_index_data(self) -> dict | None:
        url = f"{self.BASE}/api/allIndices"
        try:
            r = self._session.get(url, timeout=8)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Market Data Connector
# ---------------------------------------------------------------------------

class MarketDataConnector:
    """
    Unified data connector for all asset classes.
    All methods return a pd.DataFrame with columns:
      Open, High, Low, Close, Volume (float64)
    indexed by pd.DatetimeIndex.
    """

    def __init__(self):
        self.av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.nse = NSEScraper()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_intraday(
        self,
        yf_ticker: str,
        interval: str = "5m",
        period: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV via yfinance (all asset classes).
        Results cached for TTL_INTRADAY seconds.
        """
        cache_key = f"intraday_{yf_ticker}_{interval}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_yfinance(yf_ticker, period=period, interval=interval)
        if not df.empty:
            _cache.set(cache_key, df, TTL_INTRADAY)
        return df

    def get_daily(
        self,
        yf_ticker: str,
        period: str = "2y",
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV via yfinance.
        Results cached for TTL_DAILY seconds.
        """
        cache_key = f"daily_{yf_ticker}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_yfinance(yf_ticker, period=period, interval="1d")
        if not df.empty:
            _cache.set(cache_key, df, TTL_DAILY)
        return df

    def get_forex_eod(self, from_symbol: str, to_symbol: str) -> pd.DataFrame:
        """
        Fetch Forex daily OHLCV via Alpha Vantage FX_DAILY.
        Consumes 1 AV request. Falls back to yfinance on failure.
        """
        cache_key = f"forex_eod_{from_symbol}_{to_symbol}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_av_forex(from_symbol, to_symbol)
        if df.empty:
            # Fallback: yfinance
            yf_ticker = f"{from_symbol}{to_symbol}=X"
            logger.warning("AV Forex failed, falling back to yfinance: %s", yf_ticker)
            df = self._fetch_yfinance(yf_ticker, period="2y", interval="1d")

        if not df.empty:
            _cache.set(cache_key, df, TTL_FOREX)
        return df

    def get_nse_live(self, symbol: str, yf_ticker: str) -> float | None:
        """
        Get latest price for an NSE stock.
        Fallback chain: NSE Scraper → yfinance → stale cache → None
        """
        # 1. NSE Scraper
        quote = self.nse.get_live_quote(symbol)
        if quote is not None:
            return float(quote["price"])

        # 2. yfinance (last 1-day 5-min bar)
        logger.warning("NSE scraper failed for %s — trying yfinance", symbol)
        df = self.get_intraday(yf_ticker, interval="5m", period="1d")
        if not df.empty:
            return float(df["Close"].iloc[-1])

        # 3. Stale cache (already returned above if TTL not expired)
        stale_entry = _cache.get_stale(f"intraday_{yf_ticker}_5m")
        if stale_entry is not None:
            stale, age_seconds = stale_entry
            if age_seconds <= 600:
                logger.warning("Serving stale cache for %s (age %.0fs)", symbol, age_seconds)
                return float(stale["Close"].iloc[-1])

        # 4. Suppress
        logger.error("All data sources failed for %s — suppressing signal", symbol)
        return None

    # Legacy compatibility
    def get_stock_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        return self._fetch_yfinance(ticker, period=period, interval=interval)

    # ------------------------------------------------------------------
    # Private fetch helpers
    # ------------------------------------------------------------------

    def _fetch_yfinance(
        self,
        ticker: str,
        period: str,
        interval: str,
        asset_info: dict | None = None,
    ) -> pd.DataFrame:
        try:
            logger.debug("yfinance fetch: %s period=%s interval=%s", ticker, period, interval)
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            info = asset_info or {}
            if raw.empty and info.get("alt_ticker"):
                alt = str(info["alt_ticker"])
                logger.warning("Primary ticker %s failed, trying alt %s", ticker, alt)
                raw = yf.download(
                    alt,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            if raw.empty:
                raise ValueError(f"yfinance returned empty data for {ticker}")

            # Flatten MultiIndex columns (happens when auto_adjust=True on some versions)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.astype(float).dropna(subset=["Close"])
            df.index = pd.to_datetime(df.index)
            return df

        except Exception as exc:
            logger.error("yfinance error for %s: %s", ticker, exc)
            return pd.DataFrame()

    def _fetch_av_forex(self, from_symbol: str, to_symbol: str) -> pd.DataFrame:
        if not self.av_api_key:
            logger.warning("No AV API key — skipping AV Forex fetch")
            return pd.DataFrame()

        try:
            _rate_limiter.acquire()
            url = (
                "https://www.alphavantage.co/query"
                f"?function=FX_DAILY"
                f"&from_symbol={from_symbol}"
                f"&to_symbol={to_symbol}"
                f"&outputsize=full"
                f"&apikey={self.av_api_key}"
            )
            r = requests.get(url, timeout=15)
            data = r.json()

            ts = data.get("Time Series FX (Daily)", {})
            if not ts:
                logger.warning("AV FX_DAILY empty response for %s/%s", from_symbol, to_symbol)
                return pd.DataFrame()

            df = pd.DataFrame.from_dict(ts, orient="index")
            df.index = pd.to_datetime(df.index)
            df.columns = ["Open", "High", "Low", "Close"]
            df = df.astype(float).sort_index()
            df["Volume"] = 0.0  # Forex has no volume from AV free tier
            return df[["Open", "High", "Low", "Close", "Volume"]]

        except RuntimeError as exc:
            # Daily limit hit
            logger.error("AV rate limit: %s", exc)
            return pd.DataFrame()
        except Exception as exc:
            logger.error("AV Forex error %s/%s: %s", from_symbol, to_symbol, exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Breeze (ICICI) — template for future use
# ---------------------------------------------------------------------------

class BreezeConnector:
    """Template for ICICIdirect Breeze API Integration (Phase 2)."""

    def __init__(self, api_key: str, secret_key: str, session_token: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.session_token = session_token

    def get_historical_data(self, stock_code, exchange_code, from_date, to_date, interval):
        raise NotImplementedError("Breeze integration is Phase 2 scope.")
