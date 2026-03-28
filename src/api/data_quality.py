"""
Data quality controls for live and research pipelines.

This module guards the system before features are computed:
  - stale prices (no change for N+ bars)
  - return spikes above a rolling sigma threshold
  - zero volume bars
  - NaN-filled OHLCV
  - optional secondary-source daily close reconciliation
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class DataQualityReport:
    asset: str
    asset_class: str
    timeframe: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stale_price_rows: int = 0
    spike_rows: int = 0
    zero_volume_rows: int = 0
    nan_fill_rows: int = 0
    severe: bool = False
    issue_types: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "asset_class": self.asset_class,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "stale_price_rows": self.stale_price_rows,
            "spike_rows": self.spike_rows,
            "zero_volume_rows": self.zero_volume_rows,
            "nan_fill_rows": self.nan_fill_rows,
            "severe": self.severe,
            "issue_types": list(self.issue_types),
            "notes": dict(self.notes),
        }


class DataAnomalyDetector:
    """Flags suspicious prints and optionally sanitises the incoming frame."""

    def __init__(
        self,
        stale_bars: int = 3,
        spike_sigma: float = 5.0,
        min_spike_lookback: int = 20,
    ) -> None:
        self.stale_bars = stale_bars
        self.spike_sigma = spike_sigma
        self.min_spike_lookback = min_spike_lookback

    def inspect_and_clean(
        self,
        df: pd.DataFrame,
        asset: str,
        asset_class: str,
        timeframe: str,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        report = DataQualityReport(asset=asset, asset_class=asset_class, timeframe=timeframe)
        if df.empty:
            report.severe = True
            report.issue_types.append("empty_frame")
            return df, report

        cleaned = df.copy()
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

        nan_rows = cleaned[OHLCV_COLUMNS].isna().any(axis=1) if all(
            c in cleaned.columns for c in OHLCV_COLUMNS
        ) else pd.Series(False, index=cleaned.index)
        report.nan_fill_rows = int(nan_rows.sum())
        if report.nan_fill_rows:
            report.issue_types.append("nan_fill")
            cleaned[OHLCV_COLUMNS] = cleaned[OHLCV_COLUMNS].ffill().bfill()
            cleaned["DQ_NaNFill"] = nan_rows.astype(int)
        else:
            cleaned["DQ_NaNFill"] = 0

        if "Close" in cleaned.columns:
            unchanged = cleaned["Close"].diff().abs().fillna(1.0) < 1e-12
            stale_mask = unchanged.rolling(self.stale_bars).sum().fillna(0) >= self.stale_bars
            report.stale_price_rows = int(stale_mask.sum())
            cleaned["DQ_StalePrice"] = stale_mask.astype(int)
            if report.stale_price_rows:
                report.issue_types.append("stale_price")
        else:
            cleaned["DQ_StalePrice"] = 0

        if "Returns" in cleaned.columns:
            returns = cleaned["Returns"]
        elif "Close" in cleaned.columns:
            returns = cleaned["Close"].pct_change()
        else:
            returns = pd.Series(0.0, index=cleaned.index)
        rolling_sigma = returns.rolling(self.min_spike_lookback).std().replace(0, np.nan)
        spike_mask = (returns.abs() > (self.spike_sigma * rolling_sigma)).fillna(False)
        report.spike_rows = int(spike_mask.sum())
        cleaned["DQ_Spike"] = spike_mask.astype(int)
        if report.spike_rows:
            report.issue_types.append("spike")
            spike_idx = spike_mask[spike_mask].index
            if len(spike_idx) > 0 and "Close" in cleaned.columns:
                # Replace obvious bad ticks with a local median rather than emitting false signals.
                cleaned.loc[spike_idx, "Close"] = (
                    cleaned["Close"].rolling(5, min_periods=1).median().reindex(spike_idx)
                )

        if "Volume" in cleaned.columns:
            zero_volume = cleaned["Volume"].fillna(0) <= 0
            report.zero_volume_rows = int(zero_volume.sum())
            cleaned["DQ_ZeroVolume"] = zero_volume.astype(int)
            if report.zero_volume_rows:
                report.issue_types.append("zero_volume")
        else:
            cleaned["DQ_ZeroVolume"] = 0

        recent_window = cleaned.tail(max(self.stale_bars, 5))
        report.severe = bool(
            recent_window[["DQ_NaNFill", "DQ_StalePrice", "DQ_Spike", "DQ_ZeroVolume"]].any().any()
        )
        report.notes["rows"] = len(cleaned)
        report.notes["recent_flags"] = int(
            recent_window[["DQ_NaNFill", "DQ_StalePrice", "DQ_Spike", "DQ_ZeroVolume"]].sum().sum()
        )
        return cleaned, report


@dataclass
class PriceValidationResult:
    source: str
    available: bool
    primary_close: float
    secondary_close: float | None = None
    mismatch_pct: float | None = None
    flagged: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "available": self.available,
            "primary_close": self.primary_close,
            "secondary_close": self.secondary_close,
            "mismatch_pct": self.mismatch_pct,
            "flagged": self.flagged,
            "message": self.message,
        }


class SecondaryPriceValidator:
    """
    Optional daily-close cross validation against a secondary provider.

    Supports:
      - NSE bhavcopy for Indian cash equities
      - Polygon or Alpaca for US equities when API keys are configured
      - Alpha Vantage daily fallback if configured
    """

    def __init__(
        self,
        polygon_api_key: str = "",
        alpaca_api_key: str = "",
        alpaca_secret_key: str = "",
        alpha_vantage_key: str = "",
        mismatch_threshold_pct: float = 0.5,
    ) -> None:
        self.polygon_api_key = polygon_api_key
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.alpha_vantage_key = alpha_vantage_key
        self.mismatch_threshold_pct = mismatch_threshold_pct
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
                )
            }
        )

    def validate_daily_close(
        self,
        asset_info: dict,
        asset_class: str,
        primary_df: pd.DataFrame,
    ) -> PriceValidationResult:
        if primary_df.empty or "Close" not in primary_df.columns:
            return PriceValidationResult(
                source="unavailable",
                available=False,
                primary_close=0.0,
                message="Primary close unavailable",
            )

        primary_close = float(primary_df["Close"].iloc[-1])
        if asset_class == "indian_stock":
            result = self._validate_indian(asset_info, primary_df.index[-1], primary_close)
        elif asset_class == "us_stock":
            result = self._validate_us(asset_info, primary_df.index[-1], primary_close)
        else:
            return PriceValidationResult(
                source="not_applicable",
                available=False,
                primary_close=primary_close,
                message="Secondary daily validation skipped for this asset class",
            )

        if result.available and result.secondary_close:
            mismatch = abs(primary_close - result.secondary_close) / max(abs(result.secondary_close), 1e-9) * 100
            result.mismatch_pct = round(mismatch, 4)
            result.flagged = mismatch > self.mismatch_threshold_pct
            if result.flagged and not result.message:
                result.message = (
                    f"Secondary close mismatch {mismatch:.3f}% > {self.mismatch_threshold_pct:.3f}%"
                )
        return result

    def _validate_indian(
        self,
        asset_info: dict,
        target_dt: pd.Timestamp,
        primary_close: float,
    ) -> PriceValidationResult:
        symbol = asset_info.get("symbol", "")
        try:
            dt = target_dt.to_pydatetime()
            url = (
                "https://archives.nseindia.com/content/historical/EQUITIES/"
                f"{dt.year}/{dt.strftime('%b').upper()}/cm{dt.strftime('%d%b%Y').upper()}bhav.csv.zip"
            )
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_name = zf.namelist()[0]
            frame = pd.read_csv(zf.open(csv_name))
            row = frame.loc[frame["SYMBOL"] == symbol]
            if row.empty:
                raise ValueError(f"{symbol} not present in bhavcopy")
            close = float(row.iloc[0]["CLOSE"])
            return PriceValidationResult(
                source="nse_bhavcopy",
                available=True,
                primary_close=primary_close,
                secondary_close=close,
            )
        except Exception as exc:
            return PriceValidationResult(
                source="nse_bhavcopy",
                available=False,
                primary_close=primary_close,
                message=f"NSE bhavcopy unavailable: {exc}",
            )

    def _validate_us(
        self,
        asset_info: dict,
        target_dt: pd.Timestamp,
        primary_close: float,
    ) -> PriceValidationResult:
        symbol = asset_info.get("yf_ticker") or asset_info.get("symbol", "")
        if self.polygon_api_key:
            try:
                date_str = target_dt.strftime("%Y-%m-%d")
                url = (
                    f"https://api.polygon.io/v1/open-close/{symbol}/{date_str}"
                    f"?adjusted=true&apiKey={self.polygon_api_key}"
                )
                resp = self._session.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                return PriceValidationResult(
                    source="polygon",
                    available=True,
                    primary_close=primary_close,
                    secondary_close=float(data["close"]),
                )
            except Exception as exc:
                logger.debug("Polygon validation failed for %s: %s", symbol, exc)

        if self.alpaca_api_key and self.alpaca_secret_key:
            try:
                date_str = target_dt.strftime("%Y-%m-%d")
                resp = self._session.get(
                    f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                    params={"timeframe": "1Day", "start": date_str, "end": date_str, "limit": 1},
                    headers={
                        "APCA-API-KEY-ID": self.alpaca_api_key,
                        "APCA-API-SECRET-KEY": self.alpaca_secret_key,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                bars = resp.json().get("bars", [])
                if bars:
                    return PriceValidationResult(
                        source="alpaca",
                        available=True,
                        primary_close=primary_close,
                        secondary_close=float(bars[0]["c"]),
                    )
            except Exception as exc:
                logger.debug("Alpaca validation failed for %s: %s", symbol, exc)

        if self.alpha_vantage_key:
            try:
                resp = self._session.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "TIME_SERIES_DAILY_ADJUSTED",
                        "symbol": symbol,
                        "apikey": self.alpha_vantage_key,
                        "outputsize": "compact",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                ts = resp.json().get("Time Series (Daily)", {})
                if ts:
                    close = float(ts[max(ts.keys())]["4. close"])
                    return PriceValidationResult(
                        source="alpha_vantage",
                        available=True,
                        primary_close=primary_close,
                        secondary_close=close,
                    )
            except Exception as exc:
                logger.debug("Alpha Vantage validation failed for %s: %s", symbol, exc)

        return PriceValidationResult(
            source="secondary_unavailable",
            available=False,
            primary_close=primary_close,
            message="No US secondary price source configured",
        )
