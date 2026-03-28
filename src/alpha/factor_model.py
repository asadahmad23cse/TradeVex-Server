"""
Layer 3 — 8-Factor IC-Weighted Alpha Model.

Factors (all pre-signed: + = bullish, - = bearish):
  F1: Momentum        — IC-weighted multi-period ROC
  F2: Mean Rev        — negated Z-score vs SMA_20 (Hurst-amplified)
  F3: Volume          — OBV slope + CMF + Volume Oscillator
  F4: ML Alpha        — LSTM directional score (MC Dropout weighted)
  F5: Vol/Squeeze     — Keltner Squeeze + ATR percentile
  F6: Alt Data        — FII/DII + PCR + Google Trends (Indian only)
  F7: Ensemble ML     — XGBoost + LightGBM + Ridge meta-learner
  F8: Microstructure  — OFI + VPIN + Kyle Lambda + Micro-Price

Combination: alpha_score = Σ(IC_i × Z_i) / max(Σ|IC_i|, 0.1)
Confidence:  50 + 50 * tanh(alpha_score)   → maps to (0, 100)
Signal:      alpha_score > 0.30 → BUY
             alpha_score < -0.30 → SELL
             else → HOLD
"""

import logging

import numpy as np
import pandas as pd

from src.alpha.orderbook import OrderFlowAnalyser
from src.features.hurst import hurst_factor_multiplier
from src.utils.math_utils import bootstrap_confidence_interval

logger = logging.getLogger(__name__)

_orderflow = OrderFlowAnalyser()

ALPHA_THRESHOLD = 0.30
IC_WINDOW = 60
TURNOVER_AUTOCORR_THRESHOLD = 0.30


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Z-score normalisation."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std().replace(0, np.nan)
    z = (series - mu) / sigma
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _rolling_ic(factor: pd.Series, fwd_ret: pd.Series, window: int = IC_WINDOW) -> pd.Series:
    """
    Rolling Information Coefficient: Pearson corr(factor_t, fwd_return_t+1).
    """
    combined = pd.concat([factor, fwd_ret], axis=1).dropna()
    if len(combined) < window:
        return pd.Series(0.0, index=factor.index)

    combined.columns = ["f", "r"]
    ic = combined["f"].rolling(window).corr(combined["r"])
    return ic.reindex(factor.index).fillna(0.0)


def _bootstrap_ic_ci(factor: pd.Series, fwd_ret: pd.Series, window: int = 126) -> tuple[float, float, float]:
    combined = pd.concat([factor.rename("f"), fwd_ret.rename("r")], axis=1).dropna().tail(window)
    if len(combined) < 10:
        return 0.0, 0.0, 0.0

    def _corr_stat(sample_idx: np.ndarray) -> float:
        sample_idx = np.asarray(sample_idx, dtype=int)
        sampled = combined.iloc[sample_idx]
        lhs = sampled["f"].values
        rhs = sampled["r"].values
        if len(lhs) < 2 or len(rhs) < 2 or np.std(lhs) == 0 or np.std(rhs) == 0:
            return 0.0
        return float(np.corrcoef(lhs, rhs)[0, 1])

    return bootstrap_confidence_interval(
        np.arange(len(combined)),
        n_bootstrap=1000,
        confidence=0.95,
        statistic=_corr_stat,
    )


def _turnover_penalty(factor: pd.Series, lookback: int = 60) -> tuple[float, float]:
    recent = pd.Series(factor).dropna().tail(lookback)
    if len(recent) < 5:
        return 1.0, 1.0
    autocorr = float(recent.autocorr(lag=1))
    if not np.isfinite(autocorr):
        return 1.0, 1.0
    if autocorr >= TURNOVER_AUTOCORR_THRESHOLD:
        return 1.0, autocorr
    penalty = float(np.clip(max(autocorr, 0.0) / TURNOVER_AUTOCORR_THRESHOLD, 0.25, 1.0))
    return penalty, autocorr


class AlphaFactorModel:
    """
    Computes the 8-factor IC-weighted alpha score for a single asset.
    Requires a fully-featured DataFrame from FeatureEngineer.
    """

    def __init__(
        self,
        alpha_threshold: float = ALPHA_THRESHOLD,
        ic_window: int = IC_WINDOW,
    ):
        self.alpha_threshold = alpha_threshold
        self.ic_window = ic_window

    # ------------------------------------------------------------------
    # Individual factors
    # ------------------------------------------------------------------

    def _factor1_momentum(self, df: pd.DataFrame, hurst: float = 0.5) -> pd.Series:
        """
        Multi-period ROC momentum.
        F1 = Z(0.4×ROC_5d + 0.3×ROC_20d + 0.2×ROC_60d + 0.1×ROC_1d)
        Hurst > 0.55 → multiply by 1.25
        """
        roc1 = df.get("ROC_1d", df["Close"].pct_change(1) * 100)
        roc5 = df.get("ROC_5d", df["Close"].pct_change(5) * 100)
        roc20 = df.get("ROC_20d", df["Close"].pct_change(20) * 100)
        roc60 = df.get("ROC_60d", df["Close"].pct_change(60) * 100)

        raw = 0.1 * roc1 + 0.4 * roc5 + 0.3 * roc20 + 0.2 * roc60
        f1 = _zscore(raw, self.ic_window)

        mults = hurst_factor_multiplier(hurst)
        return f1 * mults["momentum_mult"]

    def _factor2_mean_reversion(self, df: pd.DataFrame, hurst: float = 0.5) -> pd.Series:
        """
        Negated Z-score of Close vs SMA_20.
        Price far above mean → negative (expect reversion down).
        Hurst < 0.45 → multiply by 1.25
        """
        if "Alpha_MeanReversion" in df.columns:
            raw = df["Alpha_MeanReversion"]
        else:
            bb_std = df["BB_Std"].replace(0, np.nan) if "BB_Std" in df.columns else 1.0
            raw = (df["Close"] - df["SMA_20"]) / bb_std

        f2 = -_zscore(raw, self.ic_window)   # negate: high z-score → bearish F2

        mults = hurst_factor_multiplier(hurst)
        return f2 * mults["mean_rev_mult"]

    def _factor3_volume(self, df: pd.DataFrame) -> pd.Series:
        """
        F3 = Z(0.5×OBV_Slope + 0.3×CMF + 0.2×Volume_Osc)
        Rising OBV + rising price = institutional accumulation.
        """
        obv_slope = df.get("OBV_Slope", pd.Series(0.0, index=df.index))
        cmf = df.get("CMF", pd.Series(0.0, index=df.index))
        vol_osc = df.get("Volume_Osc", pd.Series(0.0, index=df.index))

        # Normalise each component before combining
        obv_z = _zscore(obv_slope.fillna(0), self.ic_window)
        cmf_z = _zscore(cmf.fillna(0), self.ic_window)
        vosc_z = _zscore(vol_osc.fillna(0), self.ic_window)

        raw = 0.5 * obv_z + 0.3 * cmf_z + 0.2 * vosc_z
        return _zscore(raw, self.ic_window)

    def _factor4_ml(self, ml_scores: pd.Series) -> pd.Series:
        """
        ML Alpha from LSTM directional scores (MC Dropout weighted).
        Already in [-1, +1], just Z-score normalise over history.
        """
        return _zscore(ml_scores.fillna(0), self.ic_window)

    def _factor5_volatility_squeeze(
        self,
        df: pd.DataFrame,
        momentum_factor: pd.Series | None = None,
    ) -> pd.Series:
        """
        F5:  When squeeze inactive → -percentile (low vol = higher confidence baseline)
             When squeeze fires   → sign(F1_latest) × (1 - percentile) breakout signal
        """
        atr_pct = df.get("ATR_Percentile", pd.Series(50.0, index=df.index)) / 100.0
        squeeze = df.get("Keltner_Squeeze", pd.Series(0.0, index=df.index))

        # Base: penalise high-vol environments
        f5 = -atr_pct

        # Squeeze boost: align with short-term momentum direction
        if momentum_factor is not None:
            momentum_dir = np.sign(momentum_factor.fillna(0))
            f5 = f5.where(squeeze == 0, (1 - atr_pct) * momentum_dir)

        return _zscore(f5.fillna(0), self.ic_window)

    def _factor8_microstructure(self, df: pd.DataFrame) -> pd.Series:
        """
        F8: Microstructure alpha — OFI + VPIN + Kyle Lambda + Micro-Price.
        Requires OFI/VPIN/Kyle_Lambda/Trade_Arrival/Micro_Price_Offset columns
        from FeatureEngineer.add_orderflow_features().
        """
        if "OFI" not in df.columns:
            return pd.Series(0.0, index=df.index)

        ofi_z = _zscore(df["OFI"].fillna(0), self.ic_window)
        vpin_z = _zscore(df.get("VPIN", pd.Series(0.0, index=df.index)).fillna(0), self.ic_window)
        micro_z = _zscore(df.get("Micro_Price_Offset", pd.Series(0.0, index=df.index)).fillna(0), self.ic_window)
        kyle_z = _zscore(df.get("Kyle_Lambda", pd.Series(0.0, index=df.index)).fillna(0), self.ic_window)

        raw = 0.35 * ofi_z + 0.25 * micro_z + 0.25 * vpin_z + 0.15 * kyle_z
        return _zscore(raw, self.ic_window)

    # ------------------------------------------------------------------
    # IC computation
    # ------------------------------------------------------------------

    def compute_ic(
        self,
        df: pd.DataFrame,
        ml_scores: pd.Series | None = None,
        hurst: float = 0.5,
    ) -> dict[str, float]:
        """
        Compute latest rolling IC for each factor.
        Returns dict {'F1': ic_value, ..., 'F8': ic_value}
        """
        fwd_ret = df["Returns"].shift(-1)   # t+1 return

        f1 = self._factor1_momentum(df, hurst)
        f2 = self._factor2_mean_reversion(df, hurst)
        f3 = self._factor3_volume(df)
        f4 = self._factor4_ml(ml_scores if ml_scores is not None else pd.Series(0.0, index=df.index))
        f5 = self._factor5_volatility_squeeze(df, momentum_factor=f1)
        f8 = self._factor8_microstructure(df)

        ics = {}
        for name, factor in [("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4), ("F5", f5), ("F8", f8)]:
            ic_series = _rolling_ic(factor, fwd_ret, self.ic_window)
            ics[name] = float(ic_series.iloc[-1]) if len(ic_series) > 0 else 0.0

        return ics

    # ------------------------------------------------------------------
    # Score computation (single latest bar)
    # ------------------------------------------------------------------

    def score(
        self,
        df: pd.DataFrame,
        ml_score: float = 0.0,
        hurst: float = 0.5,
    ) -> dict:
        """
        Compute alpha score for the most recent bar.

        Returns
        -------
        dict with keys:
          alpha_score, confidence, signal, strength,
          factor_scores {F1..F5}, ic_weights {F1..F5}
        """
        if len(df) < self.ic_window + 10:
            return self._hold_result("Insufficient data")

        fwd_ret = df["Returns"].shift(-1)

        # Build ML score series (constant backfill for history)
        ml_series = pd.Series(0.0, index=df.index)
        ml_series.iloc[-1] = ml_score

        f1 = self._factor1_momentum(df, hurst)
        f2 = self._factor2_mean_reversion(df, hurst)
        f3 = self._factor3_volume(df)
        f4 = self._factor4_ml(ml_series)
        f5 = self._factor5_volatility_squeeze(df, momentum_factor=f1)
        f8 = self._factor8_microstructure(df)

        factors = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F8": f8}

        # Compute rolling IC for each factor
        ics: dict[str, float] = {}
        factor_meta: dict[str, dict] = {}
        for name, factor in factors.items():
            ic_series = _rolling_ic(factor, fwd_ret, self.ic_window)
            latest_ic = float(ic_series.iloc[-1]) if not ic_series.empty else 0.0
            ci_center, ci_low, ci_high = _bootstrap_ic_ci(factor, fwd_ret)
            penalty, autocorr = _turnover_penalty(factor)
            effective_ic = latest_ic * penalty
            if ci_low < -0.03:
                effective_ic = 0.0
            ics[name] = effective_ic
            factor_meta[name] = {
                "ic_latest": round(latest_ic, 4),
                "ic_ci_center": round(ci_center, 4),
                "ic_ci_low": round(ci_low, 4),
                "ic_ci_high": round(ci_high, 4),
                "lag1_autocorr": round(autocorr, 4),
                "turnover_penalty": round(penalty, 4),
                "zero_weighted": ci_low < -0.03,
            }

        # Latest factor scores (last bar)
        latest: dict[str, float] = {
            name: float(f.iloc[-1]) if not f.empty else 0.0
            for name, f in factors.items()
        }

        # IC-weighted combination
        denom = max(sum(abs(v) for v in ics.values()), 0.1)
        alpha_score = sum(ics[k] * latest[k] for k in ics) / denom

        # Map to confidence via tanh
        confidence = 50.0 + 50.0 * float(np.tanh(alpha_score))
        confidence = float(np.clip(confidence, 0.0, 100.0))

        # Signal
        if alpha_score > self.alpha_threshold:
            signal = "BUY"
        elif alpha_score < -self.alpha_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Strength
        if confidence > 75:
            strength = "STRONG"
        elif confidence > 50:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return {
            "alpha_score": round(alpha_score, 4),
            "confidence": round(confidence, 2),
            "signal": signal,
            "strength": strength,
            "factor_scores": {k: round(v, 4) for k, v in latest.items()},
            "ic_weights": {k: round(v, 4) for k, v in ics.items()},
            "factor_meta": factor_meta,
        }

    @staticmethod
    def _hold_result(reason: str) -> dict:
        _zeros = {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0, "F5": 0.0, "F8": 0.0}
        return {
            "alpha_score": 0.0,
            "confidence": 50.0,
            "signal": "HOLD",
            "strength": "WEAK",
            "factor_scores": dict(_zeros),
            "ic_weights": dict(_zeros),
            "factor_meta": {},
            "reason": reason,
        }
