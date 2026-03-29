"""
Gap 2 Fix — ML Ensemble Model (XGBoost + LightGBM + Ridge Stacking).

Replaces the basic LSTM-only F4 with a multi-model ensemble:

    Layer 1 (Base learners):
        - XGBoost (gradient-boosted trees — best on tabular data)
        - LightGBM (histogram-based, handles missing values natively)
        - Ridge Regression (linear baseline, prevents ensemble overfitting)

    Layer 2 (Meta-learner):
        - Blending via Ridge on out-of-fold predictions from Layer 1
        - Final output: directional score in [-1, +1]

Features consumed:
    All 30+ indicators from FeatureEngineer, plus:
    - Lag features (Close_Lag_1..5, Returns_Lag_1..5)
    - Rolling stats (skew, kurtosis of returns)
    - Factor scores from alpha model (F1..F5) as meta-features

Training:
    Walk-forward: train on expanding window, predict OOS on next 21 days.
    Retrained every 21 days during EOD cycle.

Usage:
    model = EnsembleModel()
    model.train(feature_df, forward_returns)
    score = model.predict(latest_features)  # → float in [-1, +1]
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports — degrade gracefully if not installed
try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False
    logger.warning("xgboost not installed — ensemble F7 will use Ridge only")

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False
    logger.warning("lightgbm not installed — ensemble F7 will skip LightGBM")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False


# Feature columns that the ensemble expects from FeatureEngineer output
ENSEMBLE_FEATURES = [
    "RSI", "MACD", "MACD_Hist", "BB_Width",
    "ATR_14", "Volatility_20", "Returns", "Log_Returns",
    "ROC_1d", "ROC_5d", "ROC_20d", "ROC_60d",
    "OBV_Slope", "CMF", "Volume_Osc", "Volume_Ratio",
    "ATR_Percentile", "Keltner_Squeeze",
    "Stoch_K", "Stoch_D", "Williams_R", "CCI",
    "Alpha_MeanReversion", "Alpha_Momentum", "Alpha_VolumeIntensity",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3",
    "Returns_Lag_1", "Returns_Lag_2", "Returns_Lag_3",
]


class EnsembleModel:
    """
    Multi-model gradient boosting ensemble for directional prediction.

    Parameters (from config `ensemble` section):
        retrain_days:  how often to retrain (default 21)
        n_estimators:  trees per base model (default 200)
        max_depth:     tree depth (default 5)
        min_train:     minimum training samples (default 252)
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.retrain_days = cfg.get("retrain_days", 21)
        self.n_estimators = cfg.get("n_estimators", 200)
        self.max_depth = cfg.get("max_depth", 5)
        self.min_train = cfg.get("min_train_rows", 252)
        self.cv_splits = cfg.get("cv_splits", 5)
        self.output_gain = float(cfg.get("output_gain", 1.25))

        self._xgb_model = None
        self._lgb_model = None
        self._ridge_model = None
        self._meta_model = None
        self._scaler = StandardScaler()
        self._trained = False
        self._last_train_date: date | None = None
        self._feature_cols: list[str] = []
        self._model_weights = {"xgb": 1.0, "lgb": 1.0, "ridge": 1.0}
        self._validation_report: dict = {}
        self._top_features: list[dict] = []
        self._target_scale = 0.01
        self._meta_walkforward_report: dict = {}

    def needs_retrain(self) -> bool:
        if not self._trained or self._last_train_date is None:
            return True
        return (date.today() - self._last_train_date).days >= self.retrain_days

    def _resolve_time_series_splits(self, n_samples: int, desired: int | None = None) -> int:
        target = int(desired or self.cv_splits)
        target = max(target, 2)
        if n_samples <= 3:
            return 2
        max_by_sample_count = max(2, min(target, n_samples // 40))
        return min(max_by_sample_count, n_samples - 1)

    def train(
        self,
        df: pd.DataFrame,
        forward_horizon: int = 5,
        oot_days: int = 21,
        refit_meta_walkforward: bool = True,
    ) -> bool:
        """
        Train the ensemble on a feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Fully enriched DataFrame from FeatureEngineer with all indicators.
        forward_horizon : int
            Number of bars ahead for the target variable (default 5 = 1 week).

        Returns
        -------
        bool : True if training succeeded
        """
        # Prepare features
        available = [c for c in ENSEMBLE_FEATURES if c in df.columns]
        if len(available) < 10:
            logger.warning("Ensemble: too few features (%d). Skipping train.", len(available))
            return False

        self._feature_cols = available

        # Target: sign of forward return (classification → regression)
        df = df.copy()
        future_close = df["Close"].shift(-forward_horizon)
        df["_fwd_ret"] = np.log(future_close / df["Close"])
        df = df.dropna(subset=available + ["_fwd_ret"])

        if len(df) < self.min_train:
            logger.warning("Ensemble: insufficient data (%d < %d)", len(df), self.min_train)
            return False

        train_df = df.iloc[:-oot_days].copy() if len(df) > (self.min_train + oot_days) else df.copy()
        oot_df = df.iloc[-oot_days:].copy() if len(df) > oot_days else pd.DataFrame()
        X = train_df[available].values
        y = train_df["_fwd_ret"].values
        finite_abs_y = np.abs(y[np.isfinite(y)])
        self._target_scale = float(
            max(
                np.nanpercentile(finite_abs_y, 90) if finite_abs_y.size else 0.0,
                float(np.nanstd(y)) if len(y) > 1 else 0.0,
                1e-4,
            )
        )

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Time-series cross-validation for stacking
        tscv = TimeSeriesSplit(n_splits=self._resolve_time_series_splits(len(X_scaled)))
        oof_preds = np.zeros((len(X_scaled), 3))  # 3 base models

        for _, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train = y[train_idx]

            # XGBoost
            if _XGB:
                xgb_m = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                )
                xgb_m.fit(X_train, y_train)
                oof_preds[val_idx, 0] = xgb_m.predict(X_val)

            # LightGBM
            if _LGB:
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                )
                lgb_m.fit(X_train, y_train)
                oof_preds[val_idx, 1] = lgb_m.predict(X_val)

            # Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            oof_preds[val_idx, 2] = ridge.predict(X_val)

        # Train final base models on all data
        if _XGB:
            self._xgb_model = xgb.XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )
            self._xgb_model.fit(X_scaled, y)

        if _LGB:
            self._lgb_model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )
            self._lgb_model.fit(X_scaled, y)

        self._ridge_model = Ridge(alpha=1.0)
        self._ridge_model.fit(X_scaled, y)

        # Meta-learner: Ridge on OOF predictions
        valid_mask = oof_preds.any(axis=1)
        if valid_mask.sum() > 10:
            self._meta_model = Ridge(alpha=1.0)
            self._meta_model.fit(oof_preds[valid_mask], y[valid_mask])
        else:
            self._meta_model = None

        if refit_meta_walkforward and valid_mask.sum() > 25:
            self._meta_model = self._walkforward_meta_refit(oof_preds[valid_mask], y[valid_mask])

        self._trained = True
        self._last_train_date = date.today()
        self._validation_report = self._compute_oot_validation(oot_df)
        if self._meta_walkforward_report:
            self._validation_report["meta_walkforward"] = dict(self._meta_walkforward_report)
        self._model_weights = self._derive_model_weights(self._validation_report)
        self._top_features = self._compute_feature_drift_summary(train_df[available], y)
        logger.info(
            "Ensemble trained on %d samples, %d features. XGB=%s LGB=%s",
            len(X_scaled), len(available), _XGB, _LGB,
        )
        return True

    def predict(self, df: pd.DataFrame) -> float:
        """
        Predict directional score for the latest bar.

        Returns
        -------
        float in [-1, +1]:  positive = bullish, negative = bearish
        """
        series = self.predict_series(df)
        if series.empty:
            return 0.0
        return round(float(series.iloc[-1]), 4)

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Score each fully-populated row for downstream factor integration."""
        if not self._trained:
            return pd.Series(0.0, index=df.index)

        available = [c for c in self._feature_cols if c in df.columns]
        if len(available) < 10:
            return pd.Series(0.0, index=df.index)

        feature_df = df[available].replace([np.inf, -np.inf], np.nan).dropna()
        if feature_df.empty:
            return pd.Series(0.0, index=df.index)

        X_scaled = self._scaler.transform(feature_df.values)
        raw_scores = self._blend_raw_scores(self._base_prediction_matrix(X_scaled))
        score_series = pd.Series(0.0, index=df.index)
        score_series.loc[feature_df.index] = np.round(self._squash_scores(raw_scores), 4)
        return score_series

    def directional_score(self, df: pd.DataFrame) -> float:
        """Alias for predict() — compatible with LSTM model interface."""
        return self.predict(df)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def validation_report(self) -> dict:
        return dict(self._validation_report)

    @property
    def top_features(self) -> list[dict]:
        return list(self._top_features)

    def _base_prediction_matrix(self, X_scaled: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(X_scaled), 3), dtype=float)
        if _XGB and self._xgb_model is not None:
            preds[:, 0] = self._xgb_model.predict(X_scaled)
        if _LGB and self._lgb_model is not None:
            preds[:, 1] = self._lgb_model.predict(X_scaled)
        if self._ridge_model is not None:
            preds[:, 2] = self._ridge_model.predict(X_scaled)

        preds[:, 0] *= self._model_weights.get("xgb", 1.0)
        preds[:, 1] *= self._model_weights.get("lgb", 1.0)
        preds[:, 2] *= self._model_weights.get("ridge", 1.0)
        return preds

    def _blend_raw_scores(self, preds: np.ndarray) -> np.ndarray:
        if preds.size == 0:
            return np.array([], dtype=float)
        if self._meta_model is not None:
            return np.asarray(self._meta_model.predict(preds), dtype=float)

        raw_scores = np.zeros(len(preds), dtype=float)
        for idx, row in enumerate(preds):
            active = row[row != 0.0]
            raw_scores[idx] = float(np.mean(active)) if active.size else 0.0
        return raw_scores

    def _squash_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        scale = max(float(self._target_scale), 1e-4)
        normalized = np.asarray(raw_scores, dtype=float) / scale
        return np.tanh(normalized * self.output_gain)

    def _walkforward_meta_refit(self, oof_preds: np.ndarray, y: np.ndarray) -> Ridge:
        meta = Ridge(alpha=1.0)
        if len(oof_preds) < 30:
            self._meta_walkforward_report = {"available": False, "message": "Insufficient OOF rows"}
            meta.fit(oof_preds, y)
            return meta

        alphas = (0.3, 1.0, 3.0, 10.0)
        n_splits = self._resolve_time_series_splits(len(oof_preds), desired=max(3, self.cv_splits - 1))
        candidates: list[dict] = []
        best_alpha = 1.0
        best_score = float("-inf")

        for alpha in alphas:
            fold_ics: list[float] = []
            fold_accs: list[float] = []
            for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(oof_preds):
                fitted = Ridge(alpha=alpha)
                fitted.fit(oof_preds[train_idx], y[train_idx])
                val_pred = fitted.predict(oof_preds[val_idx])
                val_y = y[val_idx]
                ic = float(pearsonr(val_pred, val_y)[0]) if len(val_y) > 1 and np.std(val_pred) > 0 and np.std(val_y) > 0 else 0.0
                acc = float((np.sign(val_pred) == np.sign(val_y)).mean()) if len(val_y) else 0.0
                fold_ics.append(ic)
                fold_accs.append(acc)

            mean_ic = float(np.mean(fold_ics)) if fold_ics else 0.0
            mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
            candidates.append(
                {
                    "alpha": alpha,
                    "mean_ic": round(mean_ic, 4),
                    "mean_accuracy": round(mean_acc, 4),
                }
            )
            score = mean_ic + max(mean_acc - 0.5, 0.0)
            if score > best_score:
                best_score = score
                best_alpha = alpha

        self._meta_walkforward_report = {
            "available": True,
            "n_splits": n_splits,
            "best_alpha": best_alpha,
            "candidates": candidates,
        }
        meta = Ridge(alpha=best_alpha)
        meta.fit(oof_preds, y)
        return meta

    def _compute_oot_validation(self, oot_df: pd.DataFrame) -> dict:
        if oot_df.empty or not self._trained:
            return {"available": False, "message": "OOT validation unavailable"}

        X_oot = self._scaler.transform(oot_df[self._feature_cols].values)
        y_oot = oot_df["_fwd_ret"].values
        preds = {}
        if _XGB and self._xgb_model is not None:
            preds["xgb"] = self._xgb_model.predict(X_oot)
        if _LGB and self._lgb_model is not None:
            preds["lgb"] = self._lgb_model.predict(X_oot)
        if self._ridge_model is not None:
            preds["ridge"] = self._ridge_model.predict(X_oot)

        report = {"available": True, "models": {}, "ensemble_weight_zeroed": []}
        for name, pred in preds.items():
            acc = float((np.sign(pred) == np.sign(y_oot)).mean()) if len(y_oot) else 0.0
            ic = float(pearsonr(pred, y_oot)[0]) if len(y_oot) > 1 and np.std(pred) > 0 and np.std(y_oot) > 0 else 0.0
            report["models"][name] = {
                "oot_accuracy": round(acc, 4),
                "oot_ic": round(ic, 4),
                "active": ic >= 0,
            }
            if ic < 0:
                report["ensemble_weight_zeroed"].append(name)
        return report

    def _derive_model_weights(self, report: dict) -> dict:
        weights = {"xgb": 1.0 if _XGB else 0.0, "lgb": 1.0 if _LGB else 0.0, "ridge": 1.0}
        for model, metrics in report.get("models", {}).items():
            if metrics.get("oot_ic", 0.0) < 0:
                weights[model] = 0.0
        if sum(weights.values()) <= 0:
            weights["ridge"] = 1.0
        return weights

    def _compute_feature_drift_summary(self, X_df: pd.DataFrame, y: np.ndarray) -> list[dict]:
        summary: list[dict] = []
        if X_df.empty:
            return summary

        importance = None
        model = self._xgb_model or self._lgb_model
        if _SHAP and model is not None:
            try:
                explainer = shap.Explainer(model)
                sample = X_df.tail(min(100, len(X_df)))
                shap_values = explainer(sample)
                importance = np.abs(shap_values.values).mean(axis=0)
            except Exception:
                importance = None
        if importance is None and model is not None and hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        if importance is None:
            importance = np.abs(np.corrcoef(X_df.values.T, y)[-1, :-1])
        order = np.argsort(importance)[::-1][:5]
        for rank, idx in enumerate(order, start=1):
            summary.append(
                {
                    "rank": rank,
                    "feature": X_df.columns[idx],
                    "importance": round(float(importance[idx]), 6),
                }
            )
        return summary
