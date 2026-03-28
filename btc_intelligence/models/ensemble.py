from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnsembleOutput:
    confidence: float
    validated: str
    confidence_lgbm: float
    confidence_xgb: float
    confidence_lstm: float
    inference_ms: float


def blend_confidence(
    direction: str,
    p_lgbm: float,
    p_xgb: float,
    p_lstm: float,
    inference_ms: float,
) -> EnsembleOutput:
    raw = 0.40 * p_lgbm + 0.35 * p_xgb + 0.25 * p_lstm
    confidence = max(0.0, min(100.0, raw * 100.0))

    # Agreement check from primary and validator models.
    lgbm_dir = direction if p_lgbm >= 0.55 else ('SHORT' if direction == 'LONG' else 'LONG')
    xgb_dir = direction if p_xgb >= 0.55 else ('SHORT' if direction == 'LONG' else 'LONG')
    validated = direction if lgbm_dir == xgb_dir else 'HOLD'

    return EnsembleOutput(
        confidence=confidence,
        validated=validated,
        confidence_lgbm=p_lgbm,
        confidence_xgb=p_xgb,
        confidence_lstm=p_lstm,
        inference_ms=inference_ms,
    )
