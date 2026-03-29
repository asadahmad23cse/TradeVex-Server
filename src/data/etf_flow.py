"""
Bitcoin Spot ETF Daily Flow Factor
====================================
Source: Farside Investors (farside.co.uk/bitcoin-etf-flow-all-data-table/)
Fallback: cached JSON on disk so the system never goes dark.

Why this matters
----------------
Since January 2024, Bitcoin ETF net flows from BlackRock (IBIT), Fidelity (FBTC),
and the other 9 US spot ETFs have become the dominant institutional price driver.
A single day of strong ETF inflow can add $500M–$1B of buying pressure.

Factor signal logic
-------------------
  F21 = Z-score of 5-day rolling ETF net flow sum vs 30-day rolling window.
  +ve Z  → institutional accumulation  → BUY bias
  -ve Z  → institutional distribution  → SELL bias

Highest-conviction combo (from empirical observation 2024–2026):
  F21 (ETF flow) > +1.5 AND F13 (funding rate) < -0.5
  → Institutional buying while retail is net short = strong LONG setup.

Data format (Farside HTML table)
----------------------------------
  Date | IBIT | FBTC | BITB | ARKB | BTCO | EZBC | BRRR | HODL | BTCW | GBTC | Total
  Each cell: USD millions (positive = inflow, negative = outflow, blank = 0)

Integration
-----------
  from src.data.etf_flow import ETFFlowProvider
  etf = ETFFlowProvider()
  row = etf.get_latest()          # dict: {date, total_usd_m, flow_z, ...}
  score = etf.get_factor_score()  # float in [-1, 1] for alpha model
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FARSIDE_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data-table/"

# On-disk cache path (relative to project root)
_HERE = Path(__file__).parent
_CACHE_PATH = _HERE.parent.parent / "data" / "etf_flow_cache.json"

# How long before re-fetching from the web (seconds)
_CACHE_TTL_SECONDS = 3600 * 4  # refresh every 4 hours (data is updated once daily)

# Rolling window for Z-score computation
_ZSCORE_WINDOW = 30     # 30 trading days
_ROLLING_SUM_DAYS = 5   # 5-day cumulative flow for the factor signal


# ──────────────────────────────────────────────────────────────────────────────
# ETF ticker list (order matches Farside table columns)
# ──────────────────────────────────────────────────────────────────────────────

ETF_TICKERS = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "GBTC", "BTC"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_cell(val: str) -> float:
    """Convert a Farside cell ('123.4', '-45.6', '-', '') → float USD millions."""
    v = str(val).strip().replace(",", "")
    if not v or v in {"-", "—", "–", "n/a", "N/A"}:
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def _zscore_series(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Rolling Z-score; NaN-safe."""
    mu = series.rolling(window, min_periods=5).mean()
    sigma = series.rolling(window, min_periods=5).std().replace(0, np.nan)
    z = (series - mu) / sigma
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Main provider class
# ──────────────────────────────────────────────────────────────────────────────

class ETFFlowProvider:
    """
    Fetches and caches Bitcoin ETF daily flow data from Farside Investors.

    Usage
    -----
    etf = ETFFlowProvider()
    score = etf.get_factor_score()   # float in [-1, 1]  — plug into AlphaFactorModel
    df    = etf.get_dataframe()      # full daily history as DataFrame
    latest = etf.get_latest_row()    # most recent trading day dict
    """

    def __init__(
        self,
        cache_path: str | None = None,
        cache_ttl: int = _CACHE_TTL_SECONDS,
        failure_retry_seconds: int = 900,
    ):
        self._cache_path = Path(cache_path) if cache_path else _CACHE_PATH
        self._cache_ttl = int(cache_ttl)
        self._failure_retry_seconds = int(max(60, failure_retry_seconds))
        self._df: pd.DataFrame | None = None
        self._fetched_at: float = 0.0
        self._last_failure_at: float = 0.0
        self._last_status: str = "INIT"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_factor_score(self) -> float:
        """
        Return a single float in [-1, 1] for use as factor F21.

        Score = clip(5-day rolling sum Z-score / 3, -1, 1)
        Dividing by 3 maps Z~3 → 1.0 so extreme flows saturate the factor
        without completely dominating the composite alpha.
        """
        df = self._ensure_df()
        if df.empty or "flow_z" not in df.columns:
            return 0.0
        latest_z = float(df["flow_z"].iloc[-1])
        return float(np.clip(latest_z / 3.0, -1.0, 1.0))

    def get_latest_row(self) -> dict[str, Any]:
        """Return the most recent day's data as a plain dict."""
        df = self._ensure_df()
        if df.empty:
            return {"date": None, "total_usd_m": 0.0, "flow_z": 0.0,
                    "flow_5d_sum": 0.0, "factor_score": 0.0,
                    "flow_label": "UNAVAILABLE", "source": self._last_status.lower()}
        row = df.iloc[-1].to_dict()
        row["factor_score"] = self.get_factor_score()
        row["source"] = str(row.get("source", self._last_status.lower()))
        return row

    def get_dataframe(self) -> pd.DataFrame:
        """Return the full daily DataFrame with all engineered columns."""
        return self._ensure_df().copy()

    def get_regime_signal(self) -> str:
        """
        Return a qualitative institutional regime signal.
        'STRONG_ACCUMULATION' | 'ACCUMULATION' | 'NEUTRAL' | 'DISTRIBUTION' | 'STRONG_DISTRIBUTION'
        """
        score = self.get_factor_score()
        if score > 0.5:
            return "STRONG_ACCUMULATION"
        if score > 0.15:
            return "ACCUMULATION"
        if score < -0.5:
            return "STRONG_DISTRIBUTION"
        if score < -0.15:
            return "DISTRIBUTION"
        return "NEUTRAL"

    def get_conviction_boost(self, signal: str, funding_rate_z: float = 0.0) -> float:
        """
        Compute confidence multiplier based on ETF flow + funding rate combo.

        Highest conviction setup:
            ETF flow Z > +1.5 AND funding rate Z < -0.5
            → Institutions buying while retail is net short
            → Return 1.20 (20% confidence boost)

        Lowest conviction:
            ETF flow Z < -1.0 AND signal is BUY
            → Institutions selling into your long
            → Return 0.70 (30% confidence reduction)
        """
        df = self._ensure_df()
        if df.empty:
            return 1.0

        flow_z = float(df["flow_z"].iloc[-1])
        sig = (signal or "").upper()

        if sig in {"BUY", "LONG"}:
            if flow_z > 1.5 and funding_rate_z < -0.5:
                return 1.20   # best case: institutions buying + retail short
            if flow_z > 1.0:
                return 1.10   # institutions buying
            if flow_z < -1.0:
                return 0.70   # institutions selling into your long
            if flow_z < -0.5:
                return 0.85
        elif sig in {"SELL", "SHORT"}:
            if flow_z < -1.5 and funding_rate_z > 0.5:
                return 1.20   # institutions selling + retail long
            if flow_z < -1.0:
                return 1.10
            if flow_z > 1.0:
                return 0.70   # institutions buying against your short
            if flow_z > 0.5:
                return 0.85

        return 1.0

    # ------------------------------------------------------------------
    # Internal: data loading
    # ------------------------------------------------------------------

    def _ensure_df(self) -> pd.DataFrame:
        """Load from cache or web, return engineered DataFrame."""
        now = time.time()
        if self._df is not None and (now - self._fetched_at) < self._cache_ttl:
            return self._df

        # Try loading on-disk cache
        disk_df = self._load_disk_cache()
        if disk_df is not None and not disk_df.empty:
            age = now - float(disk_df.attrs.get("fetched_at", 0))
            if age < self._cache_ttl:
                self._df = self._engineer(disk_df)
                self._last_status = "DISK_CACHE"
                self._fetched_at = now
                return self._df

        if self._last_failure_at and (now - self._last_failure_at) < self._failure_retry_seconds and self._df is not None:
            return self._df

        # Try live web fetch
        try:
            raw_df = self._fetch_farside()
            if raw_df is not None and not raw_df.empty:
                self._save_disk_cache(raw_df)
                self._df = self._engineer(raw_df)
                self._last_status = "LIVE"
                self._fetched_at = now
                logger.info("ETFFlowProvider: fetched %d rows from Farside", len(raw_df))
                return self._df
        except Exception as exc:
            logger.warning("ETFFlowProvider: live fetch failed: %s", exc)
            self._last_failure_at = now

        # Fall back to whatever is on disk even if stale
        if disk_df is not None and not disk_df.empty:
            logger.warning("ETFFlowProvider: using stale disk cache (%d rows)", len(disk_df))
            self._df = self._engineer(disk_df)
            self._last_status = "STALE_CACHE"
            self._fetched_at = now
            return self._df

        logger.warning("ETFFlowProvider: no live or cached ETF data, falling back to neutral flow")
        self._df = self._engineer(self._neutral_fallback_df())
        self._last_status = "FALLBACK"
        self._fetched_at = now
        return self._df

    # ------------------------------------------------------------------
    # Web scraping
    # ------------------------------------------------------------------

    def _fetch_farside(self) -> pd.DataFrame | None:
        """
        Scrape Farside Investors Bitcoin ETF flow table.
        Returns raw DataFrame with columns: date, IBIT, FBTC, ..., Total
        """
        if not _REQUESTS_OK:
            raise ImportError("requests not installed — pip install requests")
        if not _BS4_OK:
            raise ImportError("beautifulsoup4 not installed — pip install beautifulsoup4")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://farside.co.uk/",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        resp = requests.get(FARSIDE_URL, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table is None:
            raise ValueError("No table found on Farside page")

        rows_data: list[dict] = []
        headers_row: list[str] = []

        for i, tr in enumerate(table.find_all("tr")):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if not cells:
                continue
            if i == 0 or not headers_row:
                # Header row — normalise
                headers_row = [c.upper().strip() for c in cells]
                # Ensure 'DATE' is first
                if "DATE" not in headers_row and headers_row:
                    headers_row[0] = "DATE"
                continue

            if len(cells) < 2:
                continue

            row: dict[str, Any] = {}
            for j, val in enumerate(cells):
                col = headers_row[j] if j < len(headers_row) else f"COL_{j}"
                row[col] = val

            # Parse date
            date_str = str(row.get("DATE", "")).strip()
            if not date_str or date_str.lower() in {"date", ""}:
                continue
            try:
                parsed_date = pd.to_datetime(date_str, dayfirst=True)
            except Exception:
                continue

            # Parse Total — try common column names
            total_val = None
            for col_name in ["TOTAL", "TOTAL FLOW", "NET FLOW", "SUM"]:
                if col_name in row:
                    total_val = _parse_cell(row[col_name])
                    break
            if total_val is None:
                # Fall back: sum all numeric columns except date
                numeric_vals = []
                for k, v in row.items():
                    if k == "DATE":
                        continue
                    parsed = _parse_cell(v)
                    numeric_vals.append(parsed)
                total_val = sum(numeric_vals)

            rows_data.append({
                "date": parsed_date,
                "total_usd_m": float(total_val),
                **{
                    ticker: _parse_cell(row.get(ticker, "0"))
                    for ticker in ETF_TICKERS
                    if ticker != "TOTAL" and ticker in row
                },
            })

        if not rows_data:
            raise ValueError("Farside table parsed but no data rows found")

        df = pd.DataFrame(rows_data)
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df["source"] = "live"
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Z-score, rolling sum, and categorical flow columns."""
        if df.empty or "total_usd_m" not in df.columns:
            return df

        df = df.copy().sort_values("date").reset_index(drop=True)
        flow = df["total_usd_m"].fillna(0.0)

        # 5-day rolling sum — captures multi-day institutional momentum
        df["flow_5d_sum"] = flow.rolling(_ROLLING_SUM_DAYS, min_periods=1).sum()

        # Z-score the 5-day sum (30-day window)
        df["flow_z"] = _zscore_series(df["flow_5d_sum"], window=_ZSCORE_WINDOW)

        # Daily Z-score for the raw daily flow (secondary signal)
        df["flow_daily_z"] = _zscore_series(flow, window=_ZSCORE_WINDOW)

        # Cumulative flow to track total institutional exposure
        df["flow_cumulative_usd_m"] = flow.cumsum()

        # Categorical label
        df["flow_label"] = df["flow_z"].apply(_classify_flow)
        if "source" in df.columns:
            df.loc[df["source"].astype(str).str.lower().eq("fallback"), "flow_label"] = "UNAVAILABLE"

        return df

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _load_disk_cache(self) -> pd.DataFrame | None:
        """Load cached DataFrame from disk."""
        try:
            if not self._cache_path.exists():
                return None
            with open(self._cache_path) as f:
                payload = json.load(f)
            df = pd.DataFrame(payload.get("rows", []))
            if df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            df.attrs["fetched_at"] = float(payload.get("fetched_at", 0))
            return df
        except Exception as exc:
            logger.debug("ETFFlowProvider disk cache load failed: %s", exc)
            return None

    def _save_disk_cache(self, df: pd.DataFrame) -> None:
        """Persist DataFrame to disk as JSON."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            rows = df.copy()
            rows["date"] = rows["date"].astype(str)
            payload = {
                "fetched_at": time.time(),
                "rows": rows.to_dict(orient="records"),
            }
            with open(self._cache_path, "w") as f:
                json.dump(payload, f)
        except Exception as exc:
            logger.warning("ETFFlowProvider disk cache save failed: %s", exc)

    def _neutral_fallback_df(self) -> pd.DataFrame:
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        return pd.DataFrame(
            [
                {
                    "date": today,
                    "total_usd_m": 0.0,
                    "flow_5d_sum": 0.0,
                    "flow_z": 0.0,
                    "flow_daily_z": 0.0,
                    "flow_cumulative_usd_m": 0.0,
                    "flow_label": "UNAVAILABLE",
                    "source": "fallback",
                }
            ]
        )


def _classify_flow(z: float) -> str:
    if z > 1.5:
        return "STRONG_INFLOW"
    if z > 0.5:
        return "INFLOW"
    if z < -1.5:
        return "STRONG_OUTFLOW"
    if z < -0.5:
        return "OUTFLOW"
    return "NEUTRAL"


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton (shared across all callers in a process)
# ──────────────────────────────────────────────────────────────────────────────

_provider: ETFFlowProvider | None = None


def get_etf_flow_provider() -> ETFFlowProvider:
    global _provider
    if _provider is None:
        _provider = ETFFlowProvider()
    return _provider
