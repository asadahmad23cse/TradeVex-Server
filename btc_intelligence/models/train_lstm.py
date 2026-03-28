from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from btc_intelligence.features.feature_vector import FEATURE_COLUMNS


def _to_sequences(x: np.ndarray, y: np.ndarray, seq_len: int = 24) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(seq_len, len(x)):
        xs.append(x[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def train_lstm(csv_path: str, out_dir: str) -> dict:
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('tensorflow is required for train_lstm.py') from exc

    df = pd.read_csv(csv_path).dropna(subset=FEATURE_COLUMNS + ['was_profitable'])
    x_raw = df[FEATURE_COLUMNS].astype(float).values
    y_raw = df['was_profitable'].astype(int).values

    x, y = _to_sequences(x_raw, y_raw, seq_len=24)
    if len(x) < 200:
        raise RuntimeError('Not enough rows for LSTM sequence training')

    tscv = TimeSeriesSplit(n_splits=4)
    acc_scores = []

    for tr, te in tscv.split(x):
        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=(x.shape[1], x.shape[2])),
                LSTM(64),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid'),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x[tr], y[tr], epochs=8, batch_size=64, verbose=0)
        _, acc = model.evaluate(x[te], y[te], verbose=0)
        acc_scores.append(float(acc))

    final_model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(x.shape[1], x.shape[2])),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    )
    final_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    final_model.fit(x, y, epochs=10, batch_size=64, verbose=0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_model.save(out / 'lstm.h5')

    return {'rows': len(df), 'walk_forward_acc_lstm': float(np.mean(acc_scores))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='btc_intelligence/models/artifacts')
    args = parser.parse_args()

    metrics = train_lstm(args.data, args.out)
    meta_path = Path(args.out) / 'metadata.json'
    meta = {'version': 'btc_intel_advanced_v1', **metrics}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        existing.update(meta)
        meta = existing
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
