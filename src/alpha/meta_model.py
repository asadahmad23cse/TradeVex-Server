"""
Step 7 — Meta-Strategy Layer (Regime-Adaptive Factor Weighting).

Instead of fixed IC weights, the meta-model dynamically adjusts
factor emphasis based on the current market regime:

    TRENDING market (Hurst > 0.55, regime = BULL/BEAR):
        → Momentum weight ↑ 1.5×
        → Mean reversion weight ↓ 0.5×
        → ML ensemble weight ↑ 1.3×

    MEAN-REVERTING market (Hurst < 0.45, regime = SIDEWAYS):
        → Momentum weight ↓ 0.5×
        → Mean reversion weight ↑ 1.5×
        → Vol/Squeeze weight ↑ 1.3×

    VOLATILE/CRISIS (VIX-Z > 2.0, drawdown active):
        → All weights ↓ 0.5× (reduce exposure)
        → Volume weight ↑ 1.5× (flow signals dominate in panic)

This is adaptive alpha — the next-level edge.

Usage:
    meta = MetaModel()
    adjusted_ics = meta.adjust_weights(ic_weights, regime, hurst, vix_z, is_crisis)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Default factor names mapping
FACTOR_NAMES = {
    "F1": "Momentum",
    "F2": "MeanReversion",
    "F3": "Volume",
    "F4": "ML_LSTM",
    "F5": "VolSqueeze",
    "F6": "AltData",
    "F7": "Ensemble",
    "F8": "Microstructure",
    "F9": "OptionsPCR",
    "F10": "FIIDIIFlow",
    "F11": "IndiaVIX",
    "F12": "ExpiryCalendar",
}

# Regime-specific multipliers (can be tuned via config)
REGIME_WEIGHTS = {
    "TRENDING": {
        "F1": 1.5,   # Momentum up
        "F2": 0.5,   # Mean reversion down
        "F3": 1.0,
        "F4": 1.3,   # ML ensemble up
        "F5": 0.8,
        "F6": 1.0,
        "F7": 1.3,   # Ensemble up
        "F8": 0.9,
        "F9": 0.8,
        "F10": 1.1,
        "F11": 0.9,
        "F12": 0.8,
    },
    "MEAN_REVERT": {
        "F1": 0.5,   # Momentum down
        "F2": 1.5,   # Mean reversion up
        "F3": 1.0,
        "F4": 0.8,
        "F5": 1.3,   # Vol/squeeze up
        "F6": 1.0,
        "F7": 0.8,
        "F8": 0.8,
        "F9": 1.1,
        "F10": 0.9,
        "F11": 1.1,
        "F12": 1.3,
    },
    "CRISIS": {
        "F1": 0.3,   # Reduce all
        "F2": 0.3,
        "F3": 1.5,   # Volume dominant in panic
        "F4": 0.5,
        "F5": 0.3,
        "F6": 1.2,
        "F7": 0.5,
        "F8": 2.0,   # Microstructure dominates during dislocation
        "F9": 1.5,
        "F10": 1.7,
        "F11": 1.6,
        "F12": 1.2,
    },
    "NEUTRAL": {
        "F1": 1.0,
        "F2": 1.0,
        "F3": 1.0,
        "F4": 1.0,
        "F5": 1.0,
        "F6": 1.0,
        "F7": 1.0,
        "F8": 1.0,
        "F9": 1.0,
        "F10": 1.0,
        "F11": 1.0,
        "F12": 1.0,
    },
}


class MetaModel:
    """
    Regime-adaptive factor weighting meta-layer.

    Sits on top of the AlphaFactorModel and dynamically adjusts IC weights
    based on the current market microstate.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._custom_weights = cfg.get("regime_weights", REGIME_WEIGHTS)
        self._momentum_smoothing = cfg.get("momentum_smoothing", 0.7)

    def classify_microstate(
        self,
        regime: str,
        hurst: float,
        vix_z: float = 0.0,
        drawdown_active: bool = False,
    ) -> str:
        """
        Classify the current market into a meta-state for factor weighting.

        Returns: 'TRENDING' | 'MEAN_REVERT' | 'CRISIS' | 'NEUTRAL'
        """
        # Crisis override: VIX spike or active drawdown circuit breaker
        if vix_z > 2.0 or drawdown_active:
            return "CRISIS"

        # Hurst-based classification
        if hurst > 0.55 and regime in ("BULL", "BEAR", "HIGH_VOL_BULL", "HIGH_VOL_BEAR"):
            return "TRENDING"
        elif hurst < 0.45 and regime == "SIDEWAYS":
            return "MEAN_REVERT"
        else:
            return "NEUTRAL"

    def adjust_weights(
        self,
        ic_weights: dict[str, float],
        regime: str = "SIDEWAYS",
        hurst: float = 0.5,
        vix_z: float = 0.0,
        drawdown_active: bool = False,
    ) -> dict[str, float]:
        """
        Apply regime-adaptive multipliers to IC weights.

        Parameters
        ----------
        ic_weights       : {F1: 0.12, F2: 0.08, ...} from AlphaFactorModel
        regime           : 'BULL', 'BEAR', 'SIDEWAYS' from HMM
        hurst            : Hurst exponent [0, 1]
        vix_z            : VIX Z-score (0 = normal, >2 = crisis)
        drawdown_active  : True if drawdown circuit breaker is active

        Returns
        -------
        dict[factor → adjusted IC weight]
        """
        state = self.classify_microstate(regime, hurst, vix_z, drawdown_active)
        multipliers = self._custom_weights.get(state, REGIME_WEIGHTS["NEUTRAL"])

        adjusted = {}
        for factor, ic in ic_weights.items():
            mult = multipliers.get(factor, 1.0)
            adjusted[factor] = ic * mult

        logger.debug(
            "MetaModel [%s] hurst=%.2f vix_z=%.1f: %s → %s",
            state, hurst, vix_z,
            {k: f"{v:.3f}" for k, v in ic_weights.items()},
            {k: f"{v:.3f}" for k, v in adjusted.items()},
        )
        return adjusted

    def get_position_scale(
        self,
        regime: str,
        hurst: float,
        vix_z: float = 0.0,
        drawdown_active: bool = False,
    ) -> float:
        """
        Get an overall position scale multiplier based on meta-state.

        TRENDING  → 1.2× (higher conviction)
        NEUTRAL   → 1.0× (standard)
        MEAN_REVERT → 0.8× (lower conviction, more noise)
        CRISIS    → 0.4× (capital preservation)
        """
        state = self.classify_microstate(regime, hurst, vix_z, drawdown_active)
        scales = {
            "TRENDING": 1.2,
            "NEUTRAL": 1.0,
            "MEAN_REVERT": 0.8,
            "CRISIS": 0.4,
        }
        return scales.get(state, 1.0)
