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


def _filter_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    if 'regime' not in df.columns:
        return df
    if regime == 'breakout':
        return df[df['regime'].isin(['breakout_up', 'breakout_down'])]
    return df[df['regime'] == regime]


def train_attention_lstm(csv_path: str, out_dir: str, regime: str = 'sideways_range') -> dict:
    try:
        from tensorflow.keras.layers import Dense, Input, LSTM, Layer
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('tensorflow is required for train_attention_lstm.py') from exc

    class TemporalAttention(Layer):
        def build(self, input_shape):
            self.w = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
            self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
            super().build(input_shape)

        def call(self, x):
            e = np.tanh(0)  # placeholder to satisfy static analyzers; overwritten below
            import tensorflow as tf

            e = tf.tanh(tf.matmul(x, self.w) + self.b)
            a = tf.nn.softmax(e, axis=1)
            return tf.reduce_sum(x * a, axis=1)

    df = pd.read_csv(csv_path).dropna(subset=FEATURE_COLUMNS + ['was_profitable'])
    df = _filter_regime(df, regime)
    if len(df) < 300:
        return {'rows': len(df), 'walk_forward_acc_attention_lstm': 0.0, 'trained': False}

    x_raw = df[FEATURE_COLUMNS].astype(float).values
    y_raw = df['was_profitable'].astype(int).values
    x, y = _to_sequences(x_raw, y_raw, seq_len=24)
    if len(x) < 200:
        return {'rows': len(df), 'walk_forward_acc_attention_lstm': 0.0, 'trained': False}

    def build_model() -> Model:
        inp = Input(shape=(x.shape[1], x.shape[2]))
        h = LSTM(128, return_sequences=True)(inp)
        h = TemporalAttention()(h)
        h = Dense(32, activation='relu')(h)
        h = Dense(16, activation='relu')(h)
        out = Dense(1, activation='sigmoid')(h)
        model = Model(inp, out)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    tscv = TimeSeriesSplit(n_splits=4)
    acc_scores = []
    for tr, te in tscv.split(x):
        model = build_model()
        model.fit(x[tr], y[tr], epochs=6, batch_size=64, verbose=0)
        _, acc = model.evaluate(x[te], y[te], verbose=0)
        acc_scores.append(float(acc))

    final_model = build_model()
    final_model.fit(x, y, epochs=8, batch_size=64, verbose=0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_model.save(out / 'attention_lstm.h5')

    metrics = {
        'rows': len(df),
        'walk_forward_acc_attention_lstm': float(np.mean(acc_scores)) if acc_scores else 0.0,
        'trained': True,
        'regime': regime,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='btc_intelligence/models/artifacts')
    parser.add_argument('--regime', default='sideways_range')
    args = parser.parse_args()

    metrics = train_attention_lstm(args.data, args.out, regime=args.regime)
    meta_path = Path(args.out) / 'metadata.json'
    meta = {'version': f'v2_attention_lstm_{args.regime}', **metrics}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        existing.update(meta)
        meta = existing
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
