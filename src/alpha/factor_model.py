"""
Layer 3 - 12-Factor IC-Weighted Alpha Model.

Factors (signed: + bullish, - bearish):
  F1  Momentum
  F2  Mean Reversion
  F3  Volume Flow
  F4  ML Alpha
  F5  Volatility / Squeeze
  F6  Alternative Data Composite (India)
  F7  Ensemble Proxy
  F8  Microstructure
  F9  Options Flow (PCR, India)
  F10 FII/DII Net Flow (India)
  F11 India VIX Regime Factor (India)
  F12 Expiry Week Calendar Factor (India)

IC weighting:
  Composite IC = 0.2 * IC_10 + 0.5 * IC_21 + 0.3 * IC_63
  If composite IC < 0 -> weight = 0
  If composite IC > 0.08 -> weight *= 1.2

Confidence:
  Platt-style calibration:
    calibrated = 1 / (1 + exp(-5 * alpha_score))
    confidence = calibrated * 100
  Then regime and VIX penalties/boosts are applied.
"""

from __future__ import annotations

import logging
import threading
import time
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.api.alt_data import AltDataProvider
from src.alpha.orderbook import OrderFlowAnalyser
from src.features.hurst import hurst_factor_multiplier
from src.utils.math_utils import bootstrap_confidence_interval

try:
    import yfinance as yf  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency for VIX enrichment
    yf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_orderflow = OrderFlowAnalyser()
_alt_data = AltDataProvider(cache_ttl_seconds=300)

ALPHA_THRESHOLD = 0.45
IC_WINDOW = 60
TURNOVER_AUTOCORR_THRESHOLD = 0.30
IC_WINDOWS = (10, 21, 63)
IC_BLEND_WEIGHTS = (0.2, 0.5, 0.3)
VIX_CACHE_TTL_SEC = 300
IC_CACHE_TTL_SEC = 300

_vix_cache: dict[str, float] = {"value": 18.0, "ts": 0.0}
_ic_weight_cache: dict[str, dict[str, object]] = {}
_ic_refresh_context: dict[str, dict[str, object]] = {}
_ic_cache_lock = threading.Lock()
_ic_refresh_started = False
_model_cache: dict[str, "AlphaFactorModel"] = {}
_model_cache_lock = threading.Lock()


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Z-score normalization."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std().replace(0, np.nan)
    z = (series - mu) / sigma
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _rolling_ic(factor: pd.Series, fwd_ret: pd.Series, window: int = IC_WINDOW) -> pd.Series:
    """Rolling Information Coefficient corr(factor_t, fwd_return_t+1)."""
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


def _latest_usable_ic(ic_series: pd.Series) -> float:
    """Use last actionable IC value (ignore trailing non-actionable bar)."""
    if ic_series.empty:
        return 0.0
    usable = ic_series.iloc[:-1] if len(ic_series) > 1 else ic_series
    if usable.empty:
        return 0.0
    val = float(usable.iloc[-1])
    return val if np.isfinite(val) else 0.0


def _is_indian_asset(asset: str | None = None, asset_class: str | None = None) -> bool:
    ac = (asset_class or "").strip().lower()
    if ac == "indian_stock":
        return True
    a = (asset or "").strip().upper()
    if a.endswith(".NS") or a.endswith(".BO"):
        return True
    if a in {"NIFTY", "BANKNIFTY", "SENSEX", "^NSEI", "^NSEBANK", "^BSESN", "^INDIAVIX"}:
        return True
    return False


def _last_thursday_of_month(d: date) -> date:
    last_day = monthrange(d.year, d.month)[1]
    end = date(d.year, d.month, last_day)
    # Thursday = 3 (Mon=0)
    shift = (end.weekday() - 3) % 7
    return end - timedelta(days=shift)


def _in_expiry_window(d: date, lookback_days: int = 5) -> bool:
    expiry = _last_thursday_of_month(d)
    days_to_expiry = (expiry - d).days
    return 0 <= days_to_expiry <= lookback_days


def _compute_effective_ic_bundle(
    factors: dict[str, pd.Series],
    fwd_ret: pd.Series,
) -> tuple[dict[str, float], dict[str, dict[str, float | bool]]]:
    """
    Compute effective IC weights + metadata bundle for all factors.
    This is shared by request-time scoring and background refresh.
    """
    ics: dict[str, float] = {}
    factor_meta: dict[str, dict[str, float | bool]] = {}
    for name, factor in factors.items():
        ic_vals: list[float] = []
        for w in IC_WINDOWS:
            ic_series = _rolling_ic(factor, fwd_ret, w)
            ic_vals.append(_latest_usable_ic(ic_series))
        composite_ic = float(
            IC_BLEND_WEIGHTS[0] * ic_vals[0]
            + IC_BLEND_WEIGHTS[1] * ic_vals[1]
            + IC_BLEND_WEIGHTS[2] * ic_vals[2]
        )

        ci_center, ci_low, ci_high = _bootstrap_ic_ci(factor, fwd_ret)
        penalty, autocorr = _turnover_penalty(factor)

        effective_ic = composite_ic
        if composite_ic < 0.0:
            effective_ic = 0.0
        if composite_ic > 0.08:
            effective_ic *= 1.2
        effective_ic *= penalty
        if ci_high < 0.0:
            effective_ic = 0.0

        ics[name] = float(effective_ic)
        factor_meta[name] = {
            "ic_10": round(float(ic_vals[0]), 4),
            "ic_21": round(float(ic_vals[1]), 4),
            "ic_63": round(float(ic_vals[2]), 4),
            "ic_composite": round(float(composite_ic), 4),
            "ic_ci_center": round(float(ci_center), 4),
            "ic_ci_low": round(float(ci_low), 4),
            "ic_ci_high": round(float(ci_high), 4),
            "lag1_autocorr": round(float(autocorr), 4),
            "turnover_penalty": round(float(penalty), 4),
            "ic_boosted": bool(composite_ic > 0.08),
            "zero_weighted": bool((composite_ic < 0.0) or (ci_high < 0.0)),
        }
    return ics, factor_meta


def _refresh_ic_cache_worker() -> None:
    while True:
        time.sleep(IC_CACHE_TTL_SEC)
        with _ic_cache_lock:
            contexts = list(_ic_refresh_context.items())
        if not contexts:
            continue
        for key, ctx in contexts:
            factors = ctx.get("factors")
            fwd_ret = ctx.get("fwd_ret")
            if not isinstance(factors, dict) or not isinstance(fwd_ret, pd.Series):
                continue
            try:
                ics, factor_meta = _compute_effective_ic_bundle(factors, fwd_ret)
                with _ic_cache_lock:
                    _ic_weight_cache[key] = {
                        "ts": time.time(),
                        "ics": ics,
                        "factor_meta": factor_meta,
                    }
            except Exception as exc:
                logger.debug("IC background refresh failed for %s: %s", key, exc)


class AlphaFactorModel:
    """Computes the IC-weighted alpha score for a single asset."""

    def __init__(
        self,
        alpha_threshold: float = ALPHA_THRESHOLD,
        ic_window: int = IC_WINDOW,
    ):
        self.alpha_threshold = alpha_threshold
        self.ic_window = ic_window
        global _ic_refresh_started
        if not _ic_refresh_started:
            with _ic_cache_lock:
                if not _ic_refresh_started:
                    th = threading.Thread(
                        target=_refresh_ic_cache_worker,
                        name="alpha-ic-refresh",
                        daemon=True,
                    )
                    th.start()
                    _ic_refresh_started = True

    @staticmethod
    def _ic_cache_key(
        asset: str | None,
        asset_class: str | None,
        n_rows: int,
        last_ts: object,
    ) -> str:
        return f"{(asset or 'UNKNOWN').upper()}:{(asset_class or 'unknown').lower()}:{n_rows}:{last_ts}"

    def _get_cached_ic_bundle(
        self,
        cache_key: str,
        factors: dict[str, pd.Series],
        fwd_ret: pd.Series,
    ) -> tuple[dict[str, float], dict[str, dict[str, float | bool]]]:
        now = time.time()
        with _ic_cache_lock:
            entry = _ic_weight_cache.get(cache_key)
            if entry and (now - float(entry.get("ts", 0.0)) < IC_CACHE_TTL_SEC):
                ics = entry.get("ics")
                meta = entry.get("factor_meta")
                if isinstance(ics, dict) and isinstance(meta, dict):
                    return ics, meta  # type: ignore[return-value]

        ics, factor_meta = _compute_effective_ic_bundle(factors, fwd_ret)
        with _ic_cache_lock:
            _ic_weight_cache[cache_key] = {"ts": now, "ics": ics, "factor_meta": factor_meta}
            # store small rolling context for async refresh
            _ic_refresh_context[cache_key] = {
                "factors": {k: v.tail(700).copy() for k, v in factors.items()},
                "fwd_ret": fwd_ret.tail(700).copy(),
            }
        return ics, factor_meta

    # ------------------------------------------------------------------
    # Individual factors
    # ------------------------------------------------------------------

    def _factor1_momentum(self, df: pd.DataFrame, hurst: float = 0.5) -> pd.Series:
        """
        Multi-period ROC momentum.
        F1 = Z(0.4*ROC_5d + 0.3*ROC_20d + 0.2*ROC_60d + 0.1*ROC_1d)
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
        """Negated Z-score of close vs SMA_20."""
        if "Alpha_MeanReversion" in df.columns:
            raw = df["Alpha_MeanReversion"]
        else:
            bb_std = df["BB_Std"].replace(0, np.nan) if "BB_Std" in df.columns else 1.0
            raw = (df["Close"] - df["SMA_20"]) / bb_std

        f2 = -_zscore(raw, self.ic_window)
        mults = hurst_factor_multiplier(hurst)
        return f2 * mults["mean_rev_mult"]

    def _factor3_volume(self, df: pd.DataFrame) -> pd.Series:
        """F3 = Z(0.5*OBV_Slope + 0.3*CMF + 0.2*Volume_Osc)."""
        obv_slope = df.get("OBV_Slope", pd.Series(0.0, index=df.index))
        cmf = df.get("CMF", pd.Series(0.0, index=df.index))
        vol_osc = df.get("Volume_Osc", pd.Series(0.0, index=df.index))

        obv_z = _zscore(obv_slope.fillna(0), self.ic_window)
        cmf_z = _zscore(cmf.fillna(0), self.ic_window)
        vosc_z = _zscore(vol_osc.fillna(0), self.ic_window)

        raw = 0.5 * obv_z + 0.3 * cmf_z + 0.2 * vosc_z
        return _zscore(raw, self.ic_window)

    def _factor4_ml(self, ml_scores: pd.Series) -> pd.Series:
        """ML alpha score normalized to rolling Z-space."""
        return _zscore(ml_scores.fillna(0), self.ic_window)

    def _factor5_volatility_squeeze(
        self,
        df: pd.DataFrame,
        momentum_factor: pd.Series | None = None,
    ) -> pd.Series:
        """Volatility/squeeze state factor."""
        atr_pct = df.get("ATR_Percentile", pd.Series(50.0, index=df.index)) / 100.0
        squeeze = df.get("Keltner_Squeeze", pd.Series(0.0, index=df.index))

        f5 = -atr_pct
        if momentum_factor is not None:
            momentum_dir = np.sign(momentum_factor.fillna(0))
            f5 = f5.where(squeeze == 0, (1 - atr_pct) * momentum_dir)

        return _zscore(f5.fillna(0), self.ic_window)

    def _factor6_alt_data(self, df: pd.DataFrame, asset: str | None, asset_class: str | None) -> pd.Series:
        """India alternative data composite (from AltDataProvider)."""
        if not _is_indian_asset(asset, asset_class):
            return pd.Series(0.0, index=df.index)
        try:
            val = float(_alt_data.get_f6_score(asset_class="indian_stock"))
        except Exception as exc:
            logger.warning("F6 alt data fetch failed: %s", exc)
            val = 0.0
        return pd.Series(float(np.clip(val, -1.0, 1.0)), index=df.index)

    def _factor7_ensemble(self, df: pd.DataFrame) -> pd.Series:
        """Ensemble proxy factor from available engineered columns."""
        if "Ensemble_Score" in df.columns:
            raw = pd.Series(df["Ensemble_Score"], index=df.index)
        elif "Alpha_Momentum" in df.columns:
            raw = pd.Series(df["Alpha_Momentum"], index=df.index)
        else:
            raw = pd.Series(0.0, index=df.index)
        return _zscore(raw.fillna(0.0), self.ic_window)

    def _factor8_microstructure(self, df: pd.DataFrame) -> pd.Series:
        """Microstructure alpha from OFI + VPIN + Kyle + micro price offset."""
        if "OFI" not in df.columns:
            return pd.Series(0.0, index=df.index)

        ofi_z = _zscore(df["OFI"].fillna(0), self.ic_window)
        vpin_z = _zscore(df.get("VPIN", pd.Series(0.0, index=df.index)).fillna(0), self.ic_window)
        micro_z = _zscore(df.get("Micro_Price_Offset", pd.Series(0.0, index=df.index)).fillna(0), self.ic_window)
        kyle_z = _zscore(df.get("Kyle_Lambda", pd.Series(0.0, index=df.index)).fillna(0), self.ic_window)

        raw = 0.35 * ofi_z + 0.25 * micro_z + 0.25 * vpin_z + 0.15 * kyle_z
        return _zscore(raw, self.ic_window)

    def _factor9_options_flow(self, df: pd.DataFrame, asset: str | None, asset_class: str | None) -> pd.Series:
        """India options flow factor from PCR with explicit thresholds."""
        if not _is_indian_asset(asset, asset_class):
            return pd.Series(0.0, index=df.index)
        try:
            pcr = float(_alt_data.get_pcr())
        except Exception as exc:
            logger.warning("F9 PCR fetch failed: %s", exc)
            pcr = 1.0

        if pcr > 1.3:
            score = -1.0
        elif pcr < 0.7:
            score = 1.0
        else:
            score = 0.0
        return pd.Series(score, index=df.index)

    def _factor10_fii_dii_flow(self, df: pd.DataFrame, asset: str | None, asset_class: str | None) -> pd.Series:
        """India FII net flow factor, scaled linearly over [-500, +500] crore."""
        if not _is_indian_asset(asset, asset_class):
            return pd.Series(0.0, index=df.index)
        try:
            fii = float(_alt_data.get_fii_dii().get("fii_net_crore", 0.0))
        except Exception as exc:
            logger.warning("F10 FII flow fetch failed: %s", exc)
            fii = 0.0
        score = float(np.clip(fii / 500.0, -1.0, 1.0))
        return pd.Series(score, index=df.index)

    def _india_vix_value(self, asset: str | None, asset_class: str | None) -> float:
        """Fetch and cache India VIX level."""
        if not _is_indian_asset(asset, asset_class):
            return 18.0

        now = time.time()
        if now - _vix_cache["ts"] < VIX_CACHE_TTL_SEC:
            return float(_vix_cache["value"])

        if yf is None:
            return float(_vix_cache["value"])

        try:
            vix_df = yf.download("^INDIAVIX", period="7d", interval="1d", progress=False)
            if not vix_df.empty:
                if hasattr(vix_df.columns, "levels"):
                    vix_df.columns = vix_df.columns.get_level_values(0)
                val = float(vix_df["Close"].iloc[-1])
                if np.isfinite(val) and val > 0:
                    _vix_cache["value"] = val
                    _vix_cache["ts"] = now
                    return val
        except Exception as exc:
            logger.warning("INDIAVIX fetch failed: %s", exc)
        return float(_vix_cache["value"])

    def _factor11_vix_regime(self, df: pd.DataFrame, asset: str | None, asset_class: str | None) -> tuple[pd.Series, float]:
        """India VIX factor plus multiplier for final alpha."""
        vix = self._india_vix_value(asset, asset_class)
        if vix < 13:
            score = 0.5
            mult = 1.10
        elif vix < 18:
            score = 0.0
            mult = 1.00
        elif vix <= 22:
            score = -0.5
            mult = 0.80
        else:
            score = -1.0
            mult = 0.60
        return pd.Series(score, index=df.index), mult

    def _factor12_expiry_calendar(self, df: pd.DataFrame, asset: str | None, asset_class: str | None) -> tuple[pd.Series, bool]:
        """Expiry-week calendar factor and boolean flag."""
        if not _is_indian_asset(asset, asset_class):
            return pd.Series(0.0, index=df.index), False
        now_ist = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).date()
        in_window = _in_expiry_window(now_ist, lookback_days=5)
        return pd.Series(1.0 if in_window else 0.0, index=df.index), in_window

    # ------------------------------------------------------------------
    # IC computation helpers
    # ------------------------------------------------------------------

    def _composite_ic(self, factor: pd.Series, fwd_ret: pd.Series) -> tuple[float, dict[str, float]]:
        """Compute weighted IC blend over short/medium/long windows."""
        ic_vals: list[float] = []
        for w in IC_WINDOWS:
            ic_series = _rolling_ic(factor, fwd_ret, w)
            ic_vals.append(_latest_usable_ic(ic_series))
        composite = float(
            IC_BLEND_WEIGHTS[0] * ic_vals[0]
            + IC_BLEND_WEIGHTS[1] * ic_vals[1]
            + IC_BLEND_WEIGHTS[2] * ic_vals[2]
        )
        return composite, {
            "ic_10": round(float(ic_vals[0]), 4),
            "ic_21": round(float(ic_vals[1]), 4),
            "ic_63": round(float(ic_vals[2]), 4),
            "ic_composite": round(float(composite), 4),
        }

    def _calibrated_confidence(self, alpha_score: float, regime: str, vix_value: float) -> float:
        """Platt-style calibrated confidence with regime and VIX adjustments."""
        calibrated = 1.0 / (1.0 + float(np.exp(-5.0 * alpha_score)))
        confidence = calibrated * 100.0

        ru = (regime or "SIDEWAYS").upper()
        if ru == "HIGH_VOL_BEAR":
            confidence *= 0.5
        elif ru == "BEAR":
            confidence *= 0.7
        elif ru == "BULL":
            confidence *= 1.1

        if vix_value > 22:
            confidence *= 0.6
        elif 18 <= vix_value <= 22:
            confidence *= 0.8

        return float(np.clip(confidence, 0.0, 100.0))

    def _regime_for_confidence(self, df: pd.DataFrame, vix_value: float, regime: str | None) -> str:
        """Infer/normalize regime label used by confidence penalties."""
        if regime:
            return str(regime).upper()

        close = float(df["Close"].iloc[-1])
        sma20 = float(df.get("SMA_20", df["Close"].rolling(20).mean()).iloc[-1])
        roc5 = float(df.get("ROC_5d", df["Close"].pct_change(5) * 100).iloc[-1])
        inferred = "SIDEWAYS"
        if close > sma20 and roc5 >= 0:
            inferred = "BULL"
        elif close < sma20 and roc5 < 0:
            inferred = "BEAR"
        if inferred == "BEAR" and vix_value > 22:
            return "HIGH_VOL_BEAR"
        return inferred

    # ------------------------------------------------------------------
    # Public IC API
    # ------------------------------------------------------------------

    def compute_ic(
        self,
        df: pd.DataFrame,
        ml_scores: pd.Series | None = None,
        hurst: float = 0.5,
        asset: str | None = None,
        asset_class: str | None = None,
    ) -> dict[str, float]:
        """Compute latest composite IC for each factor."""
        fwd_ret = df["Returns"].shift(-1)

        f1 = self._factor1_momentum(df, hurst)
        f2 = self._factor2_mean_reversion(df, hurst)
        f3 = self._factor3_volume(df)
        f4 = self._factor4_ml(ml_scores if ml_scores is not None else pd.Series(0.0, index=df.index))
        f5 = self._factor5_volatility_squeeze(df, momentum_factor=f1)
        f6 = self._factor6_alt_data(df, asset, asset_class)
        f7 = self._factor7_ensemble(df)
        f8 = self._factor8_microstructure(df)
        f9 = self._factor9_options_flow(df, asset, asset_class)
        f10 = self._factor10_fii_dii_flow(df, asset, asset_class)
        f11, _ = self._factor11_vix_regime(df, asset, asset_class)
        f12, in_expiry = self._factor12_expiry_calendar(df, asset, asset_class)
        if in_expiry:
            f1 = f1 * 0.7
            f2 = f2 * 1.5

        factors = {
            "F1": f1,
            "F2": f2,
            "F3": f3,
            "F4": f4,
            "F5": f5,
            "F6": f6,
            "F7": f7,
            "F8": f8,
            "F9": f9,
            "F10": f10,
            "F11": f11,
            "F12": f12,
        }

        out: dict[str, float] = {}
        for name, factor in factors.items():
            composite_ic, _ = self._composite_ic(factor, fwd_ret)
            out[name] = composite_ic
        return out

    # ------------------------------------------------------------------
    # Score computation (single latest bar)
    # ------------------------------------------------------------------

    def score(
        self,
        df: pd.DataFrame,
        ml_score: float = 0.0,
        hurst: float = 0.5,
        asset: str | None = None,
        asset_class: str | None = None,
        regime: str | None = None,
    ) -> dict:
        """
        Compute alpha score for the most recent bar.

        Returns keys:
          alpha_score, confidence, signal, strength,
          factor_scores {F1..F12}, ic_weights {F1..F12}, factor_meta.
        """
        min_required = max(self.ic_window + 10, 80)
        if len(df) < min_required:
            return self._hold_result("Insufficient data")

        fwd_ret = df["Returns"].shift(-1)

        ml_series = pd.Series(0.0, index=df.index)
        ml_series.iloc[-1] = ml_score

        f1 = self._factor1_momentum(df, hurst)
        f2 = self._factor2_mean_reversion(df, hurst)
        f3 = self._factor3_volume(df)
        f4 = self._factor4_ml(ml_series)
        f5 = self._factor5_volatility_squeeze(df, momentum_factor=f1)
        f6 = self._factor6_alt_data(df, asset, asset_class)
        f7 = self._factor7_ensemble(df)
        f8 = self._factor8_microstructure(df)
        f9 = self._factor9_options_flow(df, asset, asset_class)
        f10 = self._factor10_fii_dii_flow(df, asset, asset_class)
        f11, vix_mult = self._factor11_vix_regime(df, asset, asset_class)
        f12, in_expiry = self._factor12_expiry_calendar(df, asset, asset_class)

        if in_expiry:
            # Expiry-week bias: higher mean-reversion weight, lower momentum weight.
            f1 = f1 * 0.7
            f2 = f2 * 1.5

        factors = {
            "F1": f1,
            "F2": f2,
            "F3": f3,
            "F4": f4,
            "F5": f5,
            "F6": f6,
            "F7": f7,
            "F8": f8,
            "F9": f9,
            "F10": f10,
            "F11": f11,
            "F12": f12,
        }

        cache_key = self._ic_cache_key(
            asset=asset,
            asset_class=asset_class,
            n_rows=len(df),
            last_ts=df.index[-1] if len(df.index) else "na",
        )
        ics, factor_meta = self._get_cached_ic_bundle(cache_key, factors, fwd_ret)

        latest: dict[str, float] = {
            name: float(series.iloc[-1]) if not series.empty else 0.0
            for name, series in factors.items()
        }

        denom = max(sum(abs(v) for v in ics.values()), 0.1)
        alpha_score = sum(ics[k] * latest[k] for k in ics) / denom
        alpha_score *= vix_mult

        if alpha_score > self.alpha_threshold:
            signal = "BUY"
        elif alpha_score < -self.alpha_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        vix_value = self._india_vix_value(asset, asset_class)
        regime_for_conf = self._regime_for_confidence(df, vix_value=vix_value, regime=regime)
        confidence = self._calibrated_confidence(alpha_score, regime_for_conf, vix_value)

        if confidence > 75:
            strength = "STRONG"
        elif confidence > 50:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return {
            "alpha_score": round(float(alpha_score), 4),
            "confidence": round(float(confidence), 2),
            "signal": signal,
            "strength": strength,
            "factor_scores": {k: round(float(v), 4) for k, v in latest.items()},
            "ic_weights": {k: round(float(v), 4) for k, v in ics.items()},
            "factor_meta": factor_meta,
            "regime_for_confidence": regime_for_conf,
            "india_vix": round(float(vix_value), 3),
            "expiry_week": bool(in_expiry),
        }

    @staticmethod
    def _hold_result(reason: str) -> dict:
        _zeros = {
            "F1": 0.0,
            "F2": 0.0,
            "F3": 0.0,
            "F4": 0.0,
            "F5": 0.0,
            "F6": 0.0,
            "F7": 0.0,
            "F8": 0.0,
            "F9": 0.0,
            "F10": 0.0,
            "F11": 0.0,
            "F12": 0.0,
        }
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


def get_cached_alpha_model(
    asset_class: str = "default",
    alpha_threshold: float = ALPHA_THRESHOLD,
    ic_window: int = IC_WINDOW,
) -> AlphaFactorModel:
    """
    Reuse model instances per asset class for lower request-time overhead.
    """
    key = f"{(asset_class or 'default').lower()}:{alpha_threshold:.4f}:{ic_window}"
    with _model_cache_lock:
        model = _model_cache.get(key)
        if model is None:
            model = AlphaFactorModel(alpha_threshold=alpha_threshold, ic_window=ic_window)
            _model_cache[key] = model
    return model
