from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from btc_intelligence.models.inference import ModelInference
from btc_intelligence.models.train_attention_lstm import train_attention_lstm
from btc_intelligence.models.train_lgbm import train_lgbm
from btc_intelligence.models.train_xgb import train_xgb


class ModelRetrainer:
    def __init__(self, inference_model: ModelInference):
        self.inference_model = inference_model

    async def run(self, data_path: str, out_dir: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._run_sync, data_path, out_dir)

    def _run_sync(self, data_path: str, out_dir: str) -> dict[str, Any]:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {'started': True, 'data_path': data_path, 'out_dir': str(root), 'regimes': {}}
        for regime in ['bullish_trend', 'bearish_trend', 'sideways_range', 'breakout']:
            regime_dir = root / regime
            regime_dir.mkdir(parents=True, exist_ok=True)
            m1 = train_lgbm(data_path, str(regime_dir), regime=regime)
            m2 = train_xgb(data_path, str(regime_dir), regime=regime)
            m3 = train_attention_lstm(data_path, str(regime_dir), regime=regime)
            results['regimes'][regime] = {'lightgbm': m1, 'xgboost': m2, 'attention_lstm': m3}

        results['version'] = f"v2_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        (root / 'metadata.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

        # Hot reload inference bundles after retraining.
        self.inference_model._load()  # noqa: SLF001 - deliberate internal refresh
        return results
