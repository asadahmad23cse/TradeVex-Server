"""
Gap 2 — Temporal Attention Model (Transformer).

Lightweight 2-layer multi-head self-attention for time-series directional
prediction.  Replaces / supplements LSTM as a deep-learning alpha component.

Architecture
------------
Input:  (batch, seq_len, n_features)  →  e.g. (1, 60, 30)
    ↓ Linear projection → d_model
    ↓ Sinusoidal positional encoding
    ↓ TransformerEncoderLayer × 2  (causal mask, dropout)
    ↓ Global average pooling over time axis
    ↓ Dense → 1  (tanh activation → [-1, +1])

Training uses the same forward-return target as the ensemble model.
MC Dropout at inference provides uncertainty.

Falls back gracefully if TensorFlow is not installed — returns 0.0.
"""

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TensorFlow / Keras is optional
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _TF = True
except ImportError:
    _TF = False
    logger.info("TensorFlow not installed — TemporalAttentionModel disabled")


# Shared feature list (same as ensemble for compatibility)
ATTENTION_FEATURES = [
    "RSI", "MACD", "MACD_Hist", "BB_Width",
    "ATR_14", "Volatility_20", "Returns", "Log_Returns",
    "ROC_1d", "ROC_5d", "ROC_20d", "ROC_60d",
    "OBV_Slope", "CMF", "Volume_Osc", "Volume_Ratio",
    "ATR_Percentile", "Keltner_Squeeze",
    "Stoch_K", "Stoch_D", "Williams_R", "CCI",
    "OFI", "VPIN", "Kyle_Lambda", "Trade_Arrival", "Micro_Price_Offset",
]


def _build_transformer(
    seq_len: int,
    n_features: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.15,
):
    """Build a causal Transformer encoder for time-series classification."""
    if not _TF:
        return None

    inp = keras.Input(shape=(seq_len, n_features))

    # Linear projection to d_model
    x = layers.Dense(d_model)(inp)

    # Sinusoidal positional encoding
    positions = tf.range(seq_len, dtype=tf.float32)
    div = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(np.log(10000.0) / d_model))
    pe_sin = tf.sin(positions[:, tf.newaxis] * div[tf.newaxis, :])
    pe_cos = tf.cos(positions[:, tf.newaxis] * div[tf.newaxis, :])
    pe = tf.reshape(tf.stack([pe_sin, pe_cos], axis=-1), [seq_len, d_model])
    x = x + pe  # broadcast over batch

    # Causal mask: each position can only attend to positions ≤ itself
    causal_mask = tf.linalg.band_part(
        tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0
    )
    # Keras MultiHeadAttention expects True = attend, False = mask
    # attention_mask shape: (seq_len, seq_len)

    for _ in range(n_layers):
        # Multi-head self-attention with causal mask
        attn_out = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout,
        )(x, x, attention_mask=causal_mask)
        attn_out = layers.Dropout(dropout)(attn_out)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

        # Feed-forward
        ff = layers.Dense(ff_dim, activation="gelu")(x)
        ff = layers.Dense(d_model)(ff)
        ff = layers.Dropout(dropout)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    # Global average pooling over time
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(1, activation="tanh")(x)

    model = keras.Model(inp, x)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


class TemporalAttentionModel:
    """
    Transformer-based directional predictor.

    Interface is compatible with TrendPredictionModel (LSTM) so the
    live_runner can swap or blend them.

    Parameters (from config `attention_model` section)
    ----------
    seq_len      : lookback window in bars (default 60)
    d_model      : embedding dimension (default 64)
    n_heads      : attention heads (default 4)
    n_layers     : Transformer encoder layers (default 2)
    epochs       : training epochs (default 30)
    retrain_days : re-train every N days (default 63)
    mc_samples   : MC Dropout forward passes at inference (default 10)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.seq_len = cfg.get("seq_len", 60)
        self.d_model = cfg.get("d_model", 64)
        self.n_heads = cfg.get("n_heads", 4)
        self.n_layers = cfg.get("n_layers", 2)
        self.ff_dim = cfg.get("ff_dim", 128)
        self.dropout = cfg.get("dropout", 0.15)
        self.epochs = cfg.get("epochs", 30)
        self.batch_size = cfg.get("batch_size", 32)
        self.retrain_days = cfg.get("retrain_days", 63)
        self.mc_samples = cfg.get("mc_samples", 10)
        self.forward_horizon = cfg.get("forward_horizon", 5)
        self.min_oot_accuracy = cfg.get("min_oot_accuracy", 0.52)

        self._model = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._trained = False
        self._last_train_date: Optional[date] = None
        self._feature_cols: list[str] = []
        self._oot_accuracy: float = 0.0

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained and _TF

    def needs_retrain(self) -> bool:
        if not _TF:
            return False
        if not self._trained or self._last_train_date is None:
            return True
        return (date.today() - self._last_train_date).days >= self.retrain_days

    def train(self, df: pd.DataFrame, oot_days: int = 60) -> bool:  # I4: was 10 days
        """Train on feature DataFrame. Returns True on success."""
        if not _TF:
            logger.warning("TensorFlow unavailable — skipping Attention training")
            return False

        available = [c for c in ATTENTION_FEATURES if c in df.columns]
        if len(available) < 8:
            logger.warning("Attention: too few features (%d). Skipping.", len(available))
            return False

        self._feature_cols = available
        df = df.copy()
        df["_fwd_ret"] = df["Close"].pct_change(self.forward_horizon).shift(-self.forward_horizon)
        df = df.dropna(subset=available + ["_fwd_ret"])

        if len(df) < self.seq_len + 50:
            logger.warning("Attention: insufficient data (%d).", len(df))
            return False

        # Split train / OOT
        if len(df) > self.seq_len + oot_days + 30:
            train_df = df.iloc[:-oot_days]
            oot_df = df.iloc[-oot_days - self.seq_len:]
        else:
            train_df = df
            oot_df = pd.DataFrame()

        # Build sequences
        X_train, y_train = self._make_sequences(train_df, available)
        if X_train is None or len(X_train) < 30:
            return False

        # Fit scaler
        flat = X_train.reshape(-1, len(available))
        self._scaler_mean = flat.mean(axis=0)
        self._scaler_std = flat.std(axis=0) + 1e-8
        X_train = (X_train - self._scaler_mean) / self._scaler_std

        # Build & train model
        self._model = _build_transformer(
            seq_len=self.seq_len,
            n_features=len(available),
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        )
        if self._model is None:
            return False

        self._model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

        self._trained = True
        self._last_train_date = date.today()

        # OOT validation
        if not oot_df.empty:
            X_oot, y_oot = self._make_sequences(oot_df, available)
            if X_oot is not None and len(X_oot) > 0:
                X_oot = (X_oot - self._scaler_mean) / self._scaler_std
                preds = self._model.predict(X_oot, verbose=0).flatten()
                self._oot_accuracy = float((np.sign(preds) == np.sign(y_oot)).mean())
                logger.info("Attention OOT accuracy: %.4f", self._oot_accuracy)

                if self._oot_accuracy < self.min_oot_accuracy:
                    logger.warning(
                        "Attention OOT accuracy %.4f < %.4f — model will be down-weighted",
                        self._oot_accuracy, self.min_oot_accuracy,
                    )

        logger.info(
            "TemporalAttention trained on %d sequences, %d features, %d epochs",
            len(X_train), len(available), self.epochs,
        )
        return True

    def directional_score(self, df: pd.DataFrame) -> float:
        """
        Predict directional score with MC Dropout.

        Returns
        -------
        float in [-1, +1]
        """
        if not self.is_trained or self._model is None:
            return 0.0

        available = [c for c in self._feature_cols if c in df.columns]
        if len(available) < 8 or len(df) < self.seq_len:
            return 0.0

        X = df[available].iloc[-self.seq_len:].values.reshape(1, self.seq_len, len(available))
        X = (X - self._scaler_mean) / self._scaler_std

        # MC Dropout: run multiple forward passes with dropout active
        preds = []
        for _ in range(self.mc_samples):
            pred = self._model(X, training=True).numpy().flatten()[0]  # training=True keeps dropout
            preds.append(pred)

        mean_pred = float(np.mean(preds))
        uncertainty = float(np.std(preds))

        # Down-weight if uncertainty is high
        confidence_penalty = max(0.5, 1.0 - uncertainty * 2)
        score = float(np.tanh(mean_pred * 50)) * confidence_penalty

        return round(np.clip(score, -1.0, 1.0), 4)

    @property
    def oot_accuracy(self) -> float:
        return self._oot_accuracy

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _make_sequences(self, df: pd.DataFrame, cols: list[str]):
        """Create (X, y) sliding windows."""
        data = df[cols].values
        target = df["_fwd_ret"].values

        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i - self.seq_len : i])
            y.append(target[i])

        if not X:
            return None, None
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
