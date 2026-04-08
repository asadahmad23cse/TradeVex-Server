from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from btc_intelligence.config import settings
from btc_intelligence.models.ensemble import EnsembleOutput, blend_confidence
from btc_intelligence.utils.validator import clamp


logger = logging.getLogger(__name__)


@dataclass
class RegimeModelBundle:
    lgbm: Any | None
    xgb: Any | None
    lstm: Any | None
    version: str


class ModelInference:
    def __init__(self) -> None:
        self.bundles: dict[str, RegimeModelBundle] = {}
        self.default_bundle = RegimeModelBundle(None, None, None, 'rule_fallback_v2')
        self.version = self.default_bundle.version
        self._load()

    def _load(self) -> None:
        root = Path(settings.model_dir)
        self.default_bundle = self._load_bundle(root, fallback_version='ml_default_v2')

        regime_dirs = {
            'bullish_trend': root / 'bullish_trend',
            'bearish_trend': root / 'bearish_trend',
            'sideways_range': root / 'sideways_range',
            'breakout': root / 'breakout',
        }
        for regime, path in regime_dirs.items():
            if path.exists():
                self.bundles[regime] = self._load_bundle(path, fallback_version=f'ml_{regime}_v2')

        if self.default_bundle.lgbm is not None or self.default_bundle.xgb is not None:
            self.version = self.default_bundle.version
        else:
            # Artifacts live under regime subdirs; HOLD / early-exit paths never call infer().
            for key in ('sideways_range', 'bullish_trend', 'bearish_trend', 'breakout'):
                b = self.bundles.get(key)
                if b is not None and b.lgbm is not None and b.xgb is not None:
                    self.version = b.version
                    break

    def _load_bundle(self, path: Path, fallback_version: str) -> RegimeModelBundle:
        lgbm = None
        xgb = None
        lstm = None
        version = fallback_version

        lgbm_path = path / 'lightgbm.pkl'
        xgb_path = path / 'xgboost.pkl'
        lstm_path = path / 'attention_lstm.h5'
        legacy_lstm = path / 'lstm.h5'
        meta_path = path / 'metadata.json'

        try:
            if lgbm_path.exists():
                lgbm = joblib.load(lgbm_path)
            if xgb_path.exists():
                xgb = joblib.load(xgb_path)

            chosen_lstm = lstm_path if lstm_path.exists() else legacy_lstm
            if chosen_lstm.exists():
                try:
                    from tensorflow.keras.models import load_model

                    lstm = load_model(chosen_lstm)
                except Exception as exc:
                    logger.warning('LSTM model load failed (%s): %s', chosen_lstm, exc)
                    lstm = None

            if meta_path.exists():
                version = json.loads(meta_path.read_text(encoding='utf-8')).get('version', version)
        except Exception as exc:
            logger.warning('Model bundle load failed (%s): %s', path, exc)
            lgbm = None
            xgb = None
            lstm = None
            version = 'rule_fallback_v2'

        return RegimeModelBundle(lgbm=lgbm, xgb=xgb, lstm=lstm, version=version)

    def infer(
        self,
        direction: str,
        vector: np.ndarray,
        feature_map: dict[str, float],
        sequence_24: np.ndarray | None = None,
        regime: str = 'sideways_range',
    ) -> EnsembleOutput:
        t0 = time.perf_counter()

        key = self._regime_key(regime)
        bundle = self.bundles.get(key, self.default_bundle)
        self.version = bundle.version

        if bundle.lgbm is None or bundle.xgb is None:
            score = self._rule_score(direction, feature_map)
            elapsed = (time.perf_counter() - t0) * 1000
            return EnsembleOutput(
                confidence=score,
                validated='RULE_BASED',
                confidence_lgbm=score / 100.0,
                confidence_xgb=score / 100.0,
                confidence_lstm=score / 100.0,
                inference_ms=elapsed,
            )

        p_lgbm = self._predict_prob(bundle.lgbm, vector, direction)
        p_xgb = self._predict_prob(bundle.xgb, vector, direction)

        if bundle.lstm is not None and sequence_24 is not None and sequence_24.shape[0] >= 24:
            p_lstm = float(bundle.lstm.predict(sequence_24[-24:][None, ...], verbose=0)[0][0])
            p_lstm = self._direction_adjust(p_lstm, direction)
        else:
            p_lstm = (p_lgbm + p_xgb) / 2.0

        elapsed = (time.perf_counter() - t0) * 1000
        return blend_confidence(direction=direction, p_lgbm=p_lgbm, p_xgb=p_xgb, p_lstm=p_lstm, inference_ms=elapsed)

    @staticmethod
    def _regime_key(regime: str) -> str:
        if regime in {'breakout_up', 'breakout_down'}:
            return 'breakout'
        if regime in {'bullish_trend', 'bearish_trend', 'sideways_range'}:
            return regime
        return 'sideways_range'

    def _predict_prob(self, model: Any, vector: np.ndarray, direction: str) -> float:
        p = float(model.predict_proba(vector)[0][1])
        return self._direction_adjust(p, direction)

    @staticmethod
    def _direction_adjust(prob: float, direction: str) -> float:
        prob = clamp(prob, 0.0, 1.0)
        if direction == 'LONG':
            return prob
        return 1.0 - prob

    @staticmethod
    def _rule_score(direction: str, f: dict[str, float]) -> float:
        score = 0.0
        mtf = f.get('mtf_alignment_score', 0.0)
        obi = f.get('obi', 0.5)
        ofi = f.get('ofi_5bar', 0.0)
        netflow = f.get('exchange_netflow_score', 0.0)
        vol = f.get('vol_regime_encoded', 1.0)
        absorption = f.get('absorption_strength', 0.0)

        score += 20 if mtf >= 3 else 12 if mtf >= 2 else 5
        if direction == 'LONG':
            score += 16 if obi > 0.6 else 10 if obi > 0.55 else 4
            score += 12 if ofi > 0 else 4
            score += 10 if netflow > 0 else 4
        else:
            score += 16 if obi < 0.4 else 10 if obi < 0.45 else 4
            score += 12 if ofi < 0 else 4
            score += 10 if netflow < 0 else 4

        score += 10 if vol in {1.0, 2.0} else 0
        score += 10 if f.get('fvg_present', 0.0) > 0 else 0
        score += 10 if absorption > 3.0 else 4 if absorption > 1.0 else 0
        return clamp(score, 0.0, 100.0)
