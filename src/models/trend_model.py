"""
LSTM Trend Prediction Model with Monte Carlo Dropout.

Key change from v1: predict() now uses model(X, training=True) called
N_SAMPLES times to get mean prediction + uncertainty.
model.predict() disables dropout at inference — that defeats MC Dropout.

Usage:
    model = TrendPredictionModel()
    model.train(df, epochs=20)
    mean_price, uncertainty = model.predict_with_uncertainty(last_seq_df)
    direction_score = model.directional_score(last_seq_df, current_price)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

MC_SAMPLES = 20   # Number of stochastic forward passes for MC Dropout


class TrendPredictionModel:

    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.model: Sequential | None = None
        self.scaler = MinMaxScaler()
        self._last_train_date: pd.Timestamp | None = None
        self._validation_report: dict = {}
        self.active_weight: float = 1.0

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_sequences(self, data: np.ndarray):
        """Convert scaled array into (X, y) sequences."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i: i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])  # Close is column 0
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------

    def build_model(self, input_shape: tuple) -> Sequential:
        """LSTM with Dropout layers that stay active during inference (MC Dropout)."""
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),          # stays ON during training=True forward pass
            LSTM(units=64, return_sequences=True),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=32, activation="relu"),
            Dropout(0.1),
            Dense(units=1),        # next-price prediction
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        self.model = model
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list | None = None,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: int = 0,
        oot_days: int = 60,  # I4: 10 days gave only 2 non-overlapping 5-day observations
    ) -> None:
        """
        Train on a DataFrame. First column used as target (Close).
        feature_cols: list of column names to use as input features.
                      If None, uses all numeric columns.
        """
        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

        data = df[feature_cols].values.astype(np.float32)
        scaled = self.scaler.fit_transform(data)
        X_all, y_all = self._prepare_sequences(scaled)

        if len(X_all) <= oot_days:
            X_train, y_train = X_all, y_all
            X_val, y_val = np.empty((0,)), np.empty((0,))
        else:
            X_train, y_train = X_all[:-oot_days], y_all[:-oot_days]
            X_val, y_val = X_all[-oot_days:], y_all[-oot_days:]

        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))

        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self._last_train_date = pd.Timestamp.utcnow()
        self._validation_report = self._compute_oot_validation(X_val, y_val, feature_cols)
        self.active_weight = 0.0 if self._validation_report.get("oot_ic", 0.0) < 0 else 1.0

    # ------------------------------------------------------------------
    # Inference — Monte Carlo Dropout
    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        last_sequence_df: pd.DataFrame,
        n_samples: int = MC_SAMPLES,
    ) -> tuple[float, float]:
        """
        MC Dropout inference: run model(X, training=True) N times.

        Returns
        -------
        mean_price : float   — mean predicted next price
        uncertainty : float  — std of predictions (normalised 0-1)
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        scaled_seq = self.scaler.transform(last_sequence_df.values.astype(np.float32))
        X = tf.constant(scaled_seq[np.newaxis, :, :], dtype=tf.float32)  # (1, seq, feat)

        preds_scaled = np.array(
            [self.model(X, training=True).numpy()[0, 0] for _ in range(n_samples)]
        )

        # Inverse-transform: reconstruct full-width row for scaler
        n_features = scaled_seq.shape[1]
        dummy = np.zeros((len(preds_scaled), n_features), dtype=np.float32)
        dummy[:, 0] = preds_scaled
        preds = self.scaler.inverse_transform(dummy)[:, 0]

        mean_price = float(np.mean(preds))
        # Normalise uncertainty by the mean price magnitude
        raw_std = float(np.std(preds))
        uncertainty = raw_std / (abs(mean_price) + 1e-9)
        return mean_price, uncertainty

    def directional_score(
        self,
        last_sequence_df: pd.DataFrame,
        current_price: float,
        n_samples: int = MC_SAMPLES,
    ) -> float:
        """
        Returns a signed score in [-1, +1]:
          +1 = confident bullish LSTM prediction
          -1 = confident bearish LSTM prediction
           0 = high uncertainty

        Score = sign(mean_pred - current_price) * confidence_weight
        confidence_weight = 1 / (1 + uncertainty)
        """
        mean_pred, uncertainty = self.predict_with_uncertainty(last_sequence_df, n_samples)
        direction = np.sign(mean_pred - current_price)
        confidence_weight = 1.0 / (1.0 + uncertainty)
        return float(direction * confidence_weight * self.active_weight)

    # Legacy predict (kept for WFO validator compatibility)
    def predict(self, last_sequence_df: pd.DataFrame) -> float:
        mean_price, _ = self.predict_with_uncertainty(last_sequence_df, n_samples=1)
        return mean_price

    def needs_retrain(self, drop_threshold: float = 0.52, retrain_days: int = 63) -> bool:
        if self.model is None or self._last_train_date is None:
            return True
        if (pd.Timestamp.utcnow() - self._last_train_date).days >= retrain_days:
            return True
        return self._validation_report.get("oot_accuracy", 1.0) < drop_threshold

    @property
    def validation_report(self) -> dict:
        return dict(self._validation_report)

    def _compute_oot_validation(self, X_val, y_val, feature_cols: list) -> dict:
        if self.model is None or len(np.atleast_1d(y_val)) == 0:
            return {"available": False}
        preds_scaled = self.model.predict(X_val, verbose=0).reshape(-1)
        dummy_pred = np.zeros((len(preds_scaled), len(feature_cols)), dtype=np.float32)
        dummy_true = np.zeros((len(y_val), len(feature_cols)), dtype=np.float32)
        dummy_pred[:, 0] = preds_scaled
        dummy_true[:, 0] = y_val
        preds = self.scaler.inverse_transform(dummy_pred)[:, 0]
        true_vals = self.scaler.inverse_transform(dummy_true)[:, 0]
        pred_ret = np.diff(np.r_[true_vals[0], preds]) / np.maximum(np.r_[true_vals[0], preds[:-1]], 1e-9)
        true_ret = np.diff(np.r_[true_vals[0], true_vals]) / np.maximum(np.r_[true_vals[0], true_vals[:-1]], 1e-9)
        accuracy = float((np.sign(pred_ret) == np.sign(true_ret)).mean()) if len(true_ret) else 0.0
        ic = float(pearsonr(pred_ret, true_ret)[0]) if len(true_ret) > 1 and np.std(pred_ret) > 0 and np.std(true_ret) > 0 else 0.0
        return {
            "available": True,
            "oot_accuracy": round(accuracy, 4),
            "oot_ic": round(ic, 4),
            "active": ic >= 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    @staticmethod
    def load(path: str) -> "TrendPredictionModel":
        m = TrendPredictionModel()
        m.model = tf.keras.models.load_model(path)
        return m
