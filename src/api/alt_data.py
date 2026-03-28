"""
GAP 6 — Alternative Data Layer (Free Sources).

Provides India-specific alpha signals unavailable in price data:

    1. FII/DII Cash Flow (NSE website — free, daily)
       FII net buy > 0 → institutional accumulation signal
       DII net buy > 0 → domestic institution support

    2. NSE F&O Option Chain → Put/Call Ratio (PCR)
       PCR > 1.3 → contrarian BUY (too many puts, market is oversold)
       PCR < 0.7 → contrarian SELL (too many calls, market is overbought)

    3. Google Trends (via pytrends — free, weekly)
       Rising search volume for a stock/sector = retail interest surge
       Used as a contrarian or momentum signal depending on context

    4. Computed F6 factor:
       F6 = 0.5 × FII_Flow_Z + 0.3 × PCR_Signal + 0.2 × Trends_Z
       Where each component is Z-score normalised over 20-day history

Usage:
    alt = AltDataProvider()
    f6 = alt.get_f6_score("NSE_BROAD")  # for Indian stocks
    # Returns float in [-1, +1]; pass as ml_score override or extra factor
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Optional pytrends ────────────────────────────────────────────────
try:
    from pytrends.request import TrendReq
    _PYTRENDS = True
except ImportError:
    _PYTRENDS = False
    logger.info("pytrends not installed — Google Trends F6 component will be 0.0")


class AltDataProvider:
    """
    Fetches free alternative data for Indian and global markets.

    Data is cached in-memory with TTL to avoid redundant HTTP calls.
    """

    NSE_FII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
    NSE_OI_URL  = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

    NSE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
    }

    def __init__(self, cache_ttl_seconds: int = 3600):
        self._cache: dict = {}
        self._ttl = cache_ttl_seconds
        self._session = requests.Session()
        self._session.headers.update(self.NSE_HEADERS)
        # Warm up session cookie
        try:
            self._session.get("https://www.nseindia.com", timeout=5)
        except Exception:
            pass

        # History buffers for Z-score normalisation
        self._fii_history: list[float] = []
        self._pcr_history: list[float] = []
        self._trends_history: list[float] = []

    # ------------------------------------------------------------------
    # FII / DII Data
    # ------------------------------------------------------------------

    def get_fii_dii(self) -> dict:
        """
        Fetch today's FII and DII cash flow data from NSE.

        Returns
        -------
        dict with keys: fii_net_crore, dii_net_crore, date
        """
        cached = self._from_cache("fii_dii")
        if cached:
            return cached

        try:
            resp = self._session.get(self.NSE_FII_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # NSE returns a list of entries; last entry is today
            latest = data[-1] if isinstance(data, list) and data else {}
            result = {
                "fii_net_crore": float(latest.get("netVal", 0)),
                "dii_net_crore": float(latest.get("diiNetVal", 0) if "diiNetVal" in latest else 0),
                "date": latest.get("date", datetime.utcnow().strftime("%d-%b-%Y")),
                "source": "nse_website",
            }
            self._to_cache("fii_dii", result)
            return result
        except Exception as exc:
            logger.warning("FII/DII fetch failed: %s — returning 0", exc)
            return {"fii_net_crore": 0.0, "dii_net_crore": 0.0, "date": None, "source": "fallback"}

    # ------------------------------------------------------------------
    # NSE Option Chain → Put/Call Ratio
    # ------------------------------------------------------------------

    def get_pcr(self) -> float:
        """
        Compute NIFTY Put/Call Ratio from NSE option chain.

        PCR = total put OI / total call OI
        """
        cached = self._from_cache("pcr")
        if cached is not None:
            return cached

        try:
            resp = self._session.get(self.NSE_OI_URL, timeout=15)
            resp.raise_for_status()
            chain = resp.json().get("records", {}).get("data", [])

            total_put_oi  = sum(r.get("PE", {}).get("openInterest", 0) for r in chain if r.get("PE"))
            total_call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in chain if r.get("CE"))

            pcr = total_put_oi / max(total_call_oi, 1)
            pcr = float(np.clip(pcr, 0.1, 5.0))
            self._to_cache("pcr", pcr)
            return pcr
        except Exception as exc:
            logger.warning("PCR fetch failed: %s — returning 1.0 (neutral)", exc)
            return 1.0

    def pcr_signal(self, pcr: float) -> float:
        """
        Convert PCR to a directional signal [-1, +1].
        PCR > 1.3 → contrarian BUY (+1)
        PCR < 0.7 → contrarian SELL (-1)
        Between 0.7–1.3 → linear interpolation
        """
        if pcr >= 1.3:
            return min((pcr - 1.3) / 0.5, 1.0)       # BUY signal
        elif pcr <= 0.7:
            return -min((0.7 - pcr) / 0.3, 1.0)      # SELL signal
        else:
            return (pcr - 1.0) / 0.3                  # linear, near 0

    # ------------------------------------------------------------------
    # Google Trends
    # ------------------------------------------------------------------

    def get_trends_score(self, keywords: list[str], geo: str = "IN") -> float:
        """
        Get normalised Google Trends interest score for keywords.
        Returns float in [0, 100] (raw pytrends interest_over_time).
        Returns 50.0 if pytrends not available.
        """
        if not _PYTRENDS:
            return 50.0

        cache_key = f"trends_{'_'.join(keywords[:2])}"
        cached = self._from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            pt = TrendReq(hl="en-US", tz=330)
            pt.build_payload(keywords[:5], timeframe="today 3-m", geo=geo)
            df = pt.interest_over_time()
            if df.empty:
                return 50.0
            latest = float(df[keywords[0]].iloc[-1])
            self._to_cache(cache_key, latest)
            return latest
        except Exception as exc:
            logger.debug("Google Trends failed: %s", exc)
            return 50.0

    # ------------------------------------------------------------------
    # F6 Composite Alt-Data Factor
    # ------------------------------------------------------------------

    def get_f6_score(
        self,
        asset_class: str = "indian_stock",
        trend_keywords: list[str] | None = None,
    ) -> float:
        """
        Compute the F6 composite alternative data factor.

        F6 = 0.5 × FII_Flow_Z + 0.3 × PCR_Signal + 0.2 × Trends_Z
        All components normalised to [-1, +1].

        Only meaningful for indian_stock asset class.
        Returns 0.0 for US stocks and Forex (FII/PCR not applicable).
        """
        if asset_class != "indian_stock":
            return 0.0

        # FII flow component
        fii_data = self.get_fii_dii()
        fii_net = fii_data["fii_net_crore"]
        self._fii_history.append(fii_net)
        if len(self._fii_history) > 20:
            self._fii_history.pop(0)

        if len(self._fii_history) >= 5:
            mu = np.mean(self._fii_history)
            sigma = np.std(self._fii_history) or 1.0
            fii_z = float(np.clip((fii_net - mu) / sigma, -3, 3) / 3.0)
        else:
            fii_z = np.sign(fii_net) * 0.5

        # PCR component (contrarian)
        pcr = self.get_pcr()
        pcr_sig = self.pcr_signal(pcr)

        # Google Trends component
        keywords = trend_keywords or ["NSE", "Nifty", "Indian stocks"]
        raw_trends = self.get_trends_score(keywords)
        self._trends_history.append(raw_trends)
        if len(self._trends_history) > 20:
            self._trends_history.pop(0)

        if len(self._trends_history) >= 5:
            t_mu = np.mean(self._trends_history)
            t_sigma = np.std(self._trends_history) or 1.0
            trends_z = float(np.clip((raw_trends - t_mu) / t_sigma, -3, 3) / 3.0)
        else:
            trends_z = 0.0

        f6 = 0.5 * fii_z + 0.3 * pcr_sig + 0.2 * trends_z
        f6 = float(np.clip(f6, -1.0, 1.0))

        logger.debug(
            "F6 [%s]: fii_z=%.3f pcr_sig=%.3f trends_z=%.3f → F6=%.4f",
            asset_class, fii_z, pcr_sig, trends_z, f6
        )
        return f6

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _from_cache(self, key: str):
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self._ttl:
                return value
        return None

    def _to_cache(self, key: str, value) -> None:
        self._cache[key] = (value, time.time())
