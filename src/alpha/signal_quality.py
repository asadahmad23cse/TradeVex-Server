"""Signal Quality Score (SQS) computation for alpha signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalQualityResult:
    sqs: float
    passes: bool
    components: dict[str, float]
    threshold: float


class SignalQualityScorer:
    """
    Composite SQS (0-100):
      - Factor agreement: 30%
      - IC quality: 25%
      - Regime alignment: 25%
      - Volatility environment: 20%
    """

    def __init__(self, min_sqs: float = 55.0):
        self.min_sqs = float(min_sqs)

    @staticmethod
    def _normalize_signal(signal: str) -> str:
        s = (signal or "HOLD").upper()
        if s in {"BUY", "LONG"}:
            return "LONG"
        if s in {"SELL", "SHORT"}:
            return "SHORT"
        return "HOLD"

    @staticmethod
    def factor_agreement_score(factor_scores: dict, signal: str) -> float:
        """Agreement based on 12-factor directional consensus."""
        norm_sig = SignalQualityScorer._normalize_signal(signal)
        if norm_sig == "HOLD":
            return 0.0

        total = 12
        agree = 0
        for _, value in (factor_scores or {}).items():
            v = float(value)
            if norm_sig == "LONG" and v > 0:
                agree += 1
            elif norm_sig == "SHORT" and v < 0:
                agree += 1

        if agree < 6:
            return 0.0
        return float(np.clip((agree / total) * 100.0, 0.0, 100.0))

    @staticmethod
    def ic_quality_score(ic_weights: dict) -> float:
        """IC quality from top absolute IC factors."""
        vals = [abs(float(v)) for v in (ic_weights or {}).values()]
        if not vals:
            return 10.0
        vals.sort(reverse=True)
        top = vals[:3]
        mean_ic = float(np.mean(top))

        if mean_ic > 0.08:
            return 100.0
        if mean_ic >= 0.05:
            return 70.0
        if mean_ic >= 0.02:
            return 40.0
        return 10.0

    @staticmethod
    def regime_alignment_score(signal: str, regime: str) -> float:
        """Signal alignment with market regime."""
        norm_sig = SignalQualityScorer._normalize_signal(signal)
        reg = (regime or "SIDEWAYS").upper()

        if norm_sig == "HOLD":
            return 50.0

        if norm_sig == "LONG":
            if reg in {"BULL", "HIGH_VOL_BULL"}:
                return 100.0
            if reg == "SIDEWAYS":
                return 50.0
            return 0.0

        if reg in {"BEAR", "HIGH_VOL_BEAR"}:
            return 100.0
        if reg == "SIDEWAYS":
            return 50.0
        return 0.0

    @staticmethod
    def volatility_environment_score(atr_percentile: float) -> float:
        """Sweet-spot volatility score from ATR percentile."""
        p = float(np.clip(atr_percentile, 0.0, 100.0))
        if 20.0 <= p <= 60.0:
            return 100.0
        if p < 20.0 or p > 80.0:
            return 40.0
        return 70.0

    def score(
        self,
        signal: str,
        factor_scores: dict,
        ic_weights: dict,
        regime: str,
        atr_percentile: float,
    ) -> SignalQualityResult:
        fac = self.factor_agreement_score(factor_scores, signal)
        icq = self.ic_quality_score(ic_weights)
        reg = self.regime_alignment_score(signal, regime)
        vol = self.volatility_environment_score(atr_percentile)

        sqs = float(
            0.30 * fac
            + 0.25 * icq
            + 0.25 * reg
            + 0.20 * vol
        )
        sqs = float(np.clip(sqs, 0.0, 100.0))
        return SignalQualityResult(
            sqs=round(sqs, 2),
            passes=sqs >= self.min_sqs,
            components={
                "factor_agreement": round(fac, 2),
                "ic_quality": round(icq, 2),
                "regime_alignment": round(reg, 2),
                "volatility_environment": round(vol, 2),
            },
            threshold=self.min_sqs,
        )
