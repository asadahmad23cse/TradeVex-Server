"""
Gap 3 — Bayesian Hyperparameter Tuning (Optuna).

Replaces grid-search in WFO and ensemble training with Tree-structured
Parzen Estimator (TPE) search.  Persists studies to sqlite so they can
be resumed or inspected later.

Two tuners:
    OptunaEnsembleTuner  — optimises XGB/LGB/Ridge hyperparams
    OptunaAlphaTuner     — optimises alpha_threshold + ic_window for WFO

Usage:
    tuner = OptunaEnsembleTuner(df)
    best = tuner.run(n_trials=50)
    model.train(df, hyperparams=best)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    _OPTUNA = True
except ImportError:
    _OPTUNA = False
    logger.info("optuna not installed — Bayesian tuning disabled, falling back to defaults")

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


# ------------------------------------------------------------------
# Ensemble Hyperparameter Tuner
# ------------------------------------------------------------------

class OptunaEnsembleTuner:
    """
    Bayesian optimisation of ensemble hyperparameters.

    Objective: maximise walk-forward OOS information coefficient (IC).
    Search space covers XGBoost, LightGBM, and Ridge parameters.
    """

    ENSEMBLE_FEATURES = [
        "RSI", "MACD", "MACD_Hist", "BB_Width",
        "ATR_14", "Volatility_20", "Returns", "Log_Returns",
        "ROC_1d", "ROC_5d", "ROC_20d", "ROC_60d",
        "OBV_Slope", "CMF", "Volume_Osc", "Volume_Ratio",
        "ATR_Percentile", "Keltner_Squeeze",
        "Stoch_K", "Stoch_D", "Williams_R", "CCI",
        "Alpha_MeanReversion", "Alpha_Momentum", "Alpha_VolumeIntensity",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        forward_horizon: int = 5,
        study_name: str = "ensemble_tuning",
        db_path: str = "data/optuna_study.db",
    ):
        self.df = df.copy()
        self.forward_horizon = forward_horizon
        self.study_name = study_name
        self.db_path = db_path

        # Prepare features + target
        self._available = [c for c in self.ENSEMBLE_FEATURES if c in df.columns]
        self.df["_fwd_ret"] = self.df["Close"].pct_change(forward_horizon).shift(-forward_horizon)
        self.df = self.df.dropna(subset=self._available + ["_fwd_ret"])

    def run(self, n_trials: int = 50, timeout_sec: int = 600) -> dict:
        """
        Run Bayesian optimisation.

        Returns dict of best hyperparameters.
        """
        if not _OPTUNA:
            logger.warning("Optuna not available — returning default hyperparams")
            return self._defaults()

        if len(self.df) < 252:
            logger.warning("Insufficient data for Optuna tuning (%d rows)", len(self.df))
            return self._defaults()

        storage = f"sqlite:///{self.db_path}"

        try:
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_warmup_steps=5),
                storage=storage,
                load_if_exists=True,
            )

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(self._objective, n_trials=n_trials, timeout=timeout_sec)

            best = study.best_params
            best["best_ic"] = study.best_value
            logger.info("Optuna best IC=%.4f params=%s", study.best_value, best)
            return best

        except Exception as e:
            logger.error("Optuna tuning failed: %s — using defaults", e)
            return self._defaults()

    def _objective(self, trial) -> float:
        """Single trial objective: walk-forward OOS IC."""
        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        n_est = trial.suggest_int("n_estimators", 50, 500, step=50)
        depth = trial.suggest_int("max_depth", 3, 8)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        ridge_alpha = trial.suggest_float("ridge_alpha", 0.01, 10.0, log=True)

        X = self.df[self._available].values
        y = self.df["_fwd_ret"].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=3)
        ics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            preds = np.zeros(len(val_idx))
            n_models = 0

            # XGBoost
            if _XGB:
                m = xgb.XGBRegressor(
                    n_estimators=n_est, max_depth=depth,
                    learning_rate=lr, subsample=subsample,
                    colsample_bytree=colsample,
                    random_state=42, verbosity=0,
                )
                m.fit(X_tr, y_tr)
                preds += m.predict(X_val)
                n_models += 1

            # LightGBM
            if _LGB:
                m = lgb.LGBMRegressor(
                    n_estimators=n_est, max_depth=depth,
                    learning_rate=lr, subsample=subsample,
                    colsample_bytree=colsample,
                    random_state=42, verbose=-1,
                )
                m.fit(X_tr, y_tr)
                preds += m.predict(X_val)
                n_models += 1

            # Ridge
            m = Ridge(alpha=ridge_alpha)
            m.fit(X_tr, y_tr)
            preds += m.predict(X_val)
            n_models += 1

            preds /= n_models

            if np.std(preds) > 1e-8 and np.std(y_val) > 1e-8:
                ic, _ = pearsonr(preds, y_val)
                ics.append(ic)

            # Optuna pruning: report intermediate results
            trial.report(np.mean(ics) if ics else 0.0, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(ics) if ics else 0.0

    @staticmethod
    def _defaults() -> dict:
        return {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "ridge_alpha": 1.0,
        }


# ------------------------------------------------------------------
# Alpha / WFO Parameter Tuner
# ------------------------------------------------------------------

class OptunaAlphaTuner:
    """
    Bayesian optimisation of alpha_threshold and ic_window for WFO.

    Replaces the grid search in WFOValidator with TPE search.
    """

    def __init__(
        self,
        wfo_evaluator,
        study_name: str = "alpha_tuning",
        db_path: str = "data/optuna_study.db",
    ):
        """
        Parameters
        ----------
        wfo_evaluator : callable(alpha_threshold, ic_window) -> float (OOS IR)
        """
        self.evaluator = wfo_evaluator
        self.study_name = study_name
        self.db_path = db_path

    def run(self, n_trials: int = 30, timeout_sec: int = 300) -> dict:
        if not _OPTUNA:
            return {"alpha_threshold": 0.30, "ic_window": 60}

        storage = f"sqlite:///{self.db_path}"

        try:
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                sampler=TPESampler(seed=42),
                storage=storage,
                load_if_exists=True,
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(self._objective, n_trials=n_trials, timeout=timeout_sec)

            best = study.best_params
            best["best_ir"] = study.best_value
            logger.info("Alpha tuning best IR=%.4f params=%s", study.best_value, best)
            return best
        except Exception as e:
            logger.error("Alpha tuning failed: %s", e)
            return {"alpha_threshold": 0.30, "ic_window": 60}

    def _objective(self, trial) -> float:
        at = trial.suggest_float("alpha_threshold", 0.15, 0.50, step=0.05)
        ic_w = trial.suggest_int("ic_window", 20, 120, step=10)
        try:
            ir = self.evaluator(at, ic_w)
            return float(ir) if np.isfinite(ir) else 0.0
        except Exception:
            return 0.0
