from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from btc_intelligence.features.feature_vector import FEATURE_COLUMNS


def _filter_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    if 'regime' not in df.columns:
        return df
    if regime == 'breakout':
        return df[df['regime'].isin(['breakout_up', 'breakout_down'])]
    if regime in {'bullish_trend', 'bearish_trend', 'sideways_range'}:
        return df[df['regime'] == regime]
    return df


def _recency_weights(df: pd.DataFrame, lambda_decay: float = 0.003) -> np.ndarray:
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    elif 'as_of_utc' in df.columns:
        ts = pd.to_datetime(df['as_of_utc'], errors='coerce', utc=True)
    else:
        days_ago = np.arange(len(df))[::-1] / 96.0
        return np.exp(-lambda_decay * days_ago)

    latest = ts.max()
    days_ago = (latest - ts).dt.total_seconds().fillna(0.0) / 86400.0
    return np.exp(-lambda_decay * days_ago.to_numpy(dtype=float))


def train_xgb(csv_path: str, out_dir: str, regime: str = 'all') -> dict:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('xgboost is required for train_xgb.py') from exc

    df = pd.read_csv(csv_path).dropna(subset=FEATURE_COLUMNS + ['was_profitable'])
    df = _filter_regime(df, regime)
    if len(df) < 300:
        return {'rows': len(df), 'walk_forward_auc_xgb': 0.0, 'trained': False, 'regime': regime}

    x = df[FEATURE_COLUMNS].astype(float).values
    y = df['was_profitable'].astype(int).values
    w = _recency_weights(df)

    model = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for tr, te in tscv.split(x):
        model.fit(x[tr], y[tr], sample_weight=w[tr])
        p = model.predict_proba(x[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))

    model.fit(x, y, sample_weight=w)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / 'xgboost.pkl')

    return {
        'rows': len(df),
        'walk_forward_auc_xgb': float(np.mean(aucs)) if aucs else 0.0,
        'trained': True,
        'regime': regime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='btc_intelligence/models/artifacts')
    parser.add_argument('--regime', default='all')
    args = parser.parse_args()

    metrics = train_xgb(args.data, args.out, regime=args.regime)
    meta_path = Path(args.out) / 'metadata.json'
    meta = {'version': f'v2_xgb_{args.regime}', **metrics}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        existing.update(meta)
        meta = existing
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
