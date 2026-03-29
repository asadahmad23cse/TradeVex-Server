"""
Gap 1 — Order Book / Microstructure Alpha (F8).

Since real L2 order book data requires expensive exchange feeds, this module
builds a **synthetic LOB engine** that estimates order flow features from
standard OHLCV + volume data.  The techniques are taken from published
market-microstructure literature:

  ①  Order Flow Imbalance (OFI)   — Cont, Kukanov & Stoikov (2014)
  ②  VPIN                         — Easley, López de Prado & O'Hara (2012)
  ③  Kyle's Lambda                — Kyle (1985)  — price impact per flow unit
  ④  Trade Arrival Rate           — Poisson intensity of trade direction change
  ⑤  Micro-Price                  — volume-weighted fair value

These features become factor F8 in AlphaFactorModel and are also available
as standalone columns from FeatureEngineer.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Synthetic Order Book
# ------------------------------------------------------------------

@dataclass
class LOBSnapshot:
    """Single synthetic limit-order-book snapshot."""
    mid_price: float
    spread: float
    bid_prices: np.ndarray    # shape (n_levels,)
    ask_prices: np.ndarray
    bid_volumes: np.ndarray
    ask_volumes: np.ndarray
    imbalance: float          # (bid_vol - ask_vol) / total_vol


class SyntheticOrderBook:
    """
    Reconstruct plausible L2 snapshots from OHLCV bars.

    Method
    ------
    1. Estimate mid-price as (High + Low) / 2
    2. Estimate spread from Corwin–Schultz estimator (already in engineer.py)
       — fallback to 2 × ATR/100
    3. Build depth using power-law volume decay from best bid/ask.
    """

    def __init__(self, n_levels: int = 5, decay_alpha: float = 1.5):
        self.n_levels = n_levels
        self.decay_alpha = decay_alpha

    def build_snapshot(
        self,
        high: float,
        low: float,
        close: float,
        volume: float,
        spread_estimate: float = 0.0,
        atr: float = 0.0,
    ) -> LOBSnapshot:
        mid = (high + low) / 2.0
        spread = spread_estimate if spread_estimate > 0 else max(atr * 0.02, mid * 0.0002)
        half = spread / 2.0

        # Build price ladders
        tick = max(spread / self.n_levels, mid * 0.00005)
        bid_prices = np.array([mid - half - i * tick for i in range(self.n_levels)])
        ask_prices = np.array([mid + half + i * tick for i in range(self.n_levels)])

        # Volume distribution: power-law decay from BBO
        weights = np.array([1.0 / (i + 1) ** self.decay_alpha for i in range(self.n_levels)])
        weights /= weights.sum()

        # Distribute volume with slight buy/sell asymmetry from close position
        close_position = (close - low) / max(high - low, 1e-9)  # 0=bearish, 1=bullish
        buy_frac = 0.3 + 0.4 * close_position  # [0.3, 0.7]
        sell_frac = 1.0 - buy_frac

        bid_volumes = weights * volume * buy_frac
        ask_volumes = weights * volume * sell_frac

        total = bid_volumes.sum() + ask_volumes.sum()
        imbalance = (bid_volumes.sum() - ask_volumes.sum()) / max(total, 1e-9)

        return LOBSnapshot(
            mid_price=mid,
            spread=spread,
            bid_prices=bid_prices,
            ask_prices=ask_prices,
            bid_volumes=bid_volumes,
            ask_volumes=ask_volumes,
            imbalance=imbalance,
        )

    def build_series(self, df: pd.DataFrame) -> list[LOBSnapshot]:
        """Build LOB snapshots for all bars in a DataFrame."""
        snapshots = []
        spread_col = "CS_Spread" if "CS_Spread" in df.columns else None
        atr_col = "ATR_14" if "ATR_14" in df.columns else None

        for i in range(len(df)):
            row = df.iloc[i]
            spread = float(row[spread_col]) if spread_col and not np.isnan(row[spread_col]) else 0.0
            atr = float(row[atr_col]) if atr_col and not np.isnan(row[atr_col]) else 0.0
            snapshots.append(self.build_snapshot(
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0)),
                spread_estimate=spread,
                atr=atr,
            ))
        return snapshots


# ------------------------------------------------------------------
# Order Flow Analyser — produces alpha features from LOB snapshots
# ------------------------------------------------------------------

class OrderFlowAnalyser:
    """
    Compute microstructure alpha features from OHLCV data.

    Features produced:
        OFI              — Order Flow Imbalance (Cont et al., 2014)
        VPIN             — Volume-synchronised probability of informed trading
        Kyle_Lambda      — price sensitivity to order flow (Kyle, 1985)
        Trade_Arrival    — Poisson arrival rate of direction changes
        Micro_Price      — volume-weighted fair price offset vs close
    """

    def __init__(self, vpin_buckets: int = 20, lambda_window: int = 20):
        self.vpin_buckets = vpin_buckets
        self.lambda_window = lambda_window
        self._lob = SyntheticOrderBook()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructure columns to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain OHLCV columns.

        Returns
        -------
        pd.DataFrame with new columns: OFI, VPIN, Kyle_Lambda,
            Trade_Arrival, Micro_Price_Offset
        """
        df = df.copy()

        if len(df) < 10:
            for col in (
                "OFI",
                "VPIN",
                "Kyle_Lambda",
                "Trade_Arrival",
                "Micro_Price_Offset",
                "CVD",
                "CVD_Z",
                "Micro_Price_Trend",
                "Flow_Imbalance_Shock",
            ):
                df[col] = 0.0
            return df

        # --- 1. Order Flow Imbalance (OFI) ---
        # Tick-rule signed volume: if close > open → buy, else sell
        price_delta = df["Close"].diff().fillna(df["Close"] - df["Open"])
        candle_bias = np.sign(df["Close"] - df["Open"])
        signed_direction = np.sign(price_delta).replace(0.0, np.nan).fillna(candle_bias).replace(0.0, 1.0)
        signed_volume = (df["Volume"] * signed_direction).astype(float).values

        # OFI = Δ(signed cumulative volume) normalised
        ofi = pd.Series(signed_volume, index=df.index).rolling(5, min_periods=1).sum().values
        vol_sum = df["Volume"].rolling(20, min_periods=1).sum().values
        ofi = ofi / np.maximum(vol_sum, 1e-9)
        df["OFI"] = ofi

        # --- 2. VPIN (Volume-synchronised PIN) ---
        vpin = self._compute_vpin(df, signed_volume)
        df["VPIN"] = vpin

        # --- 3. Kyle's Lambda ---
        # λ = Cov(ΔP, signed_vol) / Var(signed_vol)  over rolling window
        returns = df["Close"].pct_change().fillna(0).values
        kyle_lambda = np.zeros(len(df))
        w = self.lambda_window
        for i in range(w, len(df)):
            sv_win = signed_volume[i - w : i]
            ret_win = returns[i - w : i]
            var_sv = np.var(sv_win)
            if var_sv > 1e-12:
                kyle_lambda[i] = np.cov(ret_win, sv_win)[0, 1] / var_sv
        df["Kyle_Lambda"] = kyle_lambda

        # --- 4. Trade Arrival Rate ---
        # Direction change frequency over rolling window
        direction = np.sign(signed_volume)
        changes = np.zeros(len(df))
        changes[1:] = (direction[1:] != direction[:-1]).astype(float)
        arrival_rate = pd.Series(changes).rolling(20, min_periods=1).mean().values
        df["Trade_Arrival"] = arrival_rate

        # --- 5. Micro-Price Offset ---
        # micro_price = (bid_vol * ask_price + ask_vol * bid_price) / (bid_vol + ask_vol)
        # offset = (micro_price - close) / close
        snapshots = self._lob.build_series(df)
        micro_offset = np.zeros(len(df))
        for i, snap in enumerate(snapshots):
            bv = snap.bid_volumes[0]
            av = snap.ask_volumes[0]
            bp = snap.bid_prices[0]
            ap = snap.ask_prices[0]
            total = bv + av
            if total > 0:
                micro = (bv * ap + av * bp) / total
                micro_offset[i] = (micro - df["Close"].iloc[i]) / max(df["Close"].iloc[i], 1e-9)
        df["Micro_Price_Offset"] = micro_offset

        cvd_series = pd.Series(np.cumsum(signed_volume), index=df.index)
        cvd_std = cvd_series.rolling(20, min_periods=5).std().replace(0.0, np.nan)
        df["CVD"] = cvd_series
        df["CVD_Z"] = (
            (cvd_series - cvd_series.rolling(20, min_periods=5).mean()) / cvd_std
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["Micro_Price_Trend"] = pd.Series(micro_offset, index=df.index).rolling(5, min_periods=1).mean().diff().fillna(0.0)
        signed_std = pd.Series(signed_volume, index=df.index).rolling(20, min_periods=5).std().replace(0.0, np.nan)
        df["Flow_Imbalance_Shock"] = (
            pd.Series(signed_volume, index=df.index) / signed_std
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return df

    def _compute_vpin(self, df: pd.DataFrame, signed_volume: np.ndarray) -> np.ndarray:
        """
        VPIN: partition total volume into equal-sized buckets,
        compute |buy_vol – sell_vol| / bucket_vol per bucket.
        """
        vpin = np.zeros(len(df))
        total_volume = df["Volume"].values.astype(float)
        cum_vol = np.cumsum(total_volume)

        if cum_vol[-1] < 1:
            return vpin

        bucket_size = cum_vol[-1] / max(self.vpin_buckets, 1)
        if bucket_size < 1:
            return vpin

        buy_vol = np.where(signed_volume > 0, signed_volume, 0)
        sell_vol = np.where(signed_volume < 0, -signed_volume, 0)

        bucket_start = 0
        bucket_buy = 0.0
        bucket_sell = 0.0
        bucket_count = 0
        running_vpin = 0.0

        for i in range(len(df)):
            bucket_buy += buy_vol[i]
            bucket_sell += sell_vol[i]
            bucket_total = bucket_buy + bucket_sell

            if bucket_total >= bucket_size:
                ratio = abs(bucket_buy - bucket_sell) / max(bucket_total, 1e-9)
                bucket_count += 1
                # Exponential moving average of bucket ratios
                alpha = 2.0 / (min(bucket_count, self.vpin_buckets) + 1)
                running_vpin = alpha * ratio + (1 - alpha) * running_vpin
                bucket_buy = 0.0
                bucket_sell = 0.0

            vpin[i] = running_vpin

        return vpin

    def composite_score(self, df: pd.DataFrame) -> float:
        """
        Aggregate microstructure features into a single [-1, +1] score.

        Positive = informed buying pressure, Negative = informed selling.
        """
        if len(df) < 5:
            return 0.0

        last = df.iloc[-1]
        ofi = float(last.get("OFI", 0))
        vpin = float(last.get("VPIN", 0))
        micro = float(last.get("Micro_Price_Offset", 0))
        arrival = float(last.get("Trade_Arrival", 0))

        # Weighted combination
        raw = (
            0.35 * np.tanh(ofi * 10)
            + 0.25 * np.tanh(micro * 500)
            + 0.20 * (vpin - 0.5) * 2      # centred around 0.5
            + 0.20 * np.tanh((arrival - 0.3) * 5)
        )
        return float(np.clip(raw, -1.0, 1.0))
