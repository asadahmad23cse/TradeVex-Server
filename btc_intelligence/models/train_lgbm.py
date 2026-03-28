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


def train_lgbm(csv_path: str, out_dir: str, regime: str = 'all') -> dict:
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('lightgbm is required for train_lgbm.py') from exc

    df = pd.read_csv(csv_path).dropna(subset=FEATURE_COLUMNS + ['was_profitable'])
    df = _filter_regime(df, regime)
    if len(df) < 300:
        return {'rows': len(df), 'walk_forward_auc_lgbm': 0.0, 'trained': False, 'regime': regime}

    x = df[FEATURE_COLUMNS].astype(float).values
    y = df['was_profitable'].astype(int).values
    w = _recency_weights(df)

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
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
    joblib.dump(model, out / 'lightgbm.pkl')

    return {
        'rows': len(df),
        'walk_forward_auc_lgbm': float(np.mean(aucs)) if aucs else 0.0,
        'trained': True,
        'regime': regime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='btc_intelligence/models/artifacts')
    parser.add_argument('--regime', default='all')
    args = parser.parse_args()

    metrics = train_lgbm(args.data, args.out, regime=args.regime)
    meta_path = Path(args.out) / 'metadata.json'
    meta = {'version': f'v2_lgbm_{args.regime}', **metrics}
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
