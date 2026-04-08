from __future__ import annotations

import logging
from typing import Any

import numpy as np

from btc_intelligence.config import settings

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover
    GaussianHMM = None  # type: ignore[misc, assignment]


def candles_to_hmm_observations(candles: list[dict[str, Any]]) -> np.ndarray | None:
    """Build (T-1, 2) observation matrix: log return, normalized range."""
    need = max(int(settings.hmm_min_bars), 10)
    if len(candles) < need:
        return None
    take = min(len(candles), int(settings.hmm_training_bars))
    rows = candles[-take:]
    closes = np.array([float(c["close"]) for c in rows], dtype=float)
    highs = np.array([float(c["high"]) for c in rows], dtype=float)
    lows = np.array([float(c["low"]) for c in rows], dtype=float)
    closes = np.clip(closes, 1e-9, None)
    rets = np.diff(np.log(closes))
    c1 = closes[1:]
    ranges = (highs[1:] - lows[1:]) / np.clip(c1, 1e-9, None)
    x = np.column_stack([rets, ranges])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] < max(20, need // 2):
        return None
    return x


def _semantic_indices(means: np.ndarray) -> tuple[int, int, int]:
    """
    Map HMM state indices to (mean_reverting, trending, liquidity_cascade).
    mean_reverting: smallest |return mean| + range mean (compression).
    liquidity_cascade: largest |return| + range (stress).
    trending: remainder.
    """
    n = means.shape[0]
    if n != 3:
        return 0, 1, 2
    abs_ret = np.abs(means[:, 0])
    rng = np.clip(means[:, 1], 1e-9, None)
    mr_score = abs_ret + rng
    cas_score = abs_ret + 2.0 * rng
    idx_mr = int(np.argmin(mr_score))
    idx_cas = int(np.argmax(cas_score))
    if idx_cas == idx_mr:
        idx_cas = (idx_cas + 1) % 3
    remaining = {0, 1, 2} - {idx_mr, idx_cas}
    idx_tr = int(remaining.pop()) if remaining else (3 - idx_mr - idx_cas) % 3
    return idx_mr, idx_tr, idx_cas


def posterior_state_probs(obs: np.ndarray) -> dict[str, float] | None:
    """Fit 3-state Gaussian HMM and return named posterior at last timestep."""
    if GaussianHMM is None:
        logger.warning("hmmlearn not available; HMM regime fallback")
        return None
    if obs is None or obs.shape[0] < 20:
        return None
    try:
        model = GaussianHMM(
            n_components=int(settings.hmm_n_states),
            covariance_type="diag",
            n_iter=int(settings.hmm_max_iter),
            random_state=int(settings.hmm_random_state),
            init_params="stmc",
            params="stmc",
        )
        with np.errstate(all="ignore"):
            model.fit(obs)
            post = model.predict_proba(obs)
        last = post[-1]
        idx_mr, idx_tr, idx_cas = _semantic_indices(np.asarray(model.means_, dtype=float))
        return {
            "mean_reverting": float(last[idx_mr]),
            "trending": float(last[idx_tr]),
            "liquidity_cascade": float(last[idx_cas]),
        }
    except Exception as exc:
        logger.debug("HMM fit/predict failed: %s", exc)
        return None


def hmm_regime_label(
    state_probs: dict[str, float],
    feature_state: Any,
    recent_log_ret_sum: float,
) -> tuple[str, int, str]:
    """
    Map HMM posteriors + price context to legacy regime strings for SignalEngine.
    Returns (regime, score, reason).
    """
    pmr = float(state_probs.get("mean_reverting", 0.0))
    ptr = float(state_probs.get("trending", 0.0))
    plc = float(state_probs.get("liquidity_cascade", 0.0))
    top = max(pmr, ptr, plc)
    score = int(max(1, min(99, round(100.0 * top))))

    if plc >= float(settings.hmm_cascade_prob_threshold):
        return (
            "panic_liquidation",
            min(6, max(score // 15, 4)),
            f"HMM cascade dominance p={plc:.2f} (liquidity stress)",
        )
    if pmr >= ptr and pmr >= plc:
        return (
            "sideways_range",
            max(1, score // 18),
            f"HMM mean-reversion p={pmr:.2f} (compression / MR)",
        )

    tf4 = feature_state.price_action.tf_4h
    bullish = float(tf4.price) >= float(tf4.ema50)
    if recent_log_ret_sum >= 0 and bullish:
        reg = "bullish_trend"
        reason = f"HMM trending p={ptr:.2f} + bullish 4h structure"
    elif recent_log_ret_sum <= 0 and not bullish:
        reg = "bearish_trend"
        reason = f"HMM trending p={ptr:.2f} + bearish 4h structure"
    elif recent_log_ret_sum >= 0:
        reg = "bullish_trend"
        reason = f"HMM trending p={ptr:.2f} + upside impulse"
    else:
        reg = "bearish_trend"
        reason = f"HMM trending p={ptr:.2f} + downside impulse"
    return reg, max(2, score // 16), reason
