"""
Layer — Walk-Forward Optimization (WFO) Validator.

Validates the 5-Factor Alpha Model parameters using an expanding window:
  - Minimum 252-day training window
  - 21-day out-of-sample (OOS) step
  - Parameter grid: alpha_score_threshold × ic_window
  - Acceptance criterion: OOS Information Ratio (IC_mean / IC_std) > 0.3

Per spec §5 (WFO Validation of Alpha Model Parameters).

Usage:
    python main.py --mode backtest --ticker AAPL --train_days 252
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.api.connectors import MarketDataConnector
from src.features.engineer import FeatureEngineer
from src.alpha.factor_model import AlphaFactorModel
from src.research.validation import CPCVValidator
from src.backtest.regime_analysis import RegimePerformanceTracker

logger = logging.getLogger(__name__)


class WFOValidator:
    """
    Walk-Forward Optimizer for the 5-factor alpha model.

    Expands the training window from min_train_days onward, steps by
    oos_step_days, and returns the best parameter set that achieves
    out-of-sample IR > ir_threshold.
    """

    # Default WFO config (overridden by config.yaml `wfo` section)
    DEFAULT_ALPHA_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    DEFAULT_IC_WINDOWS = [30, 45, 60, 75, 90]
    DEFAULT_MIN_TRAIN = 252
    DEFAULT_OOS_STEP = 21
    DEFAULT_IR_THRESHOLD = 0.3

    def __init__(
        self,
        ticker: str,
        train_window: int = DEFAULT_MIN_TRAIN,
        wfo_config: dict | None = None,
    ):
        self.ticker = ticker
        self.min_train = train_window
        cfg = wfo_config or {}
        self.alpha_thresholds = cfg.get("alpha_thresholds", self.DEFAULT_ALPHA_THRESHOLDS)
        self.ic_windows = cfg.get("ic_windows", self.DEFAULT_IC_WINDOWS)
        self.oos_step = cfg.get("oos_step_days", self.DEFAULT_OOS_STEP)
        self.ir_threshold = cfg.get("ir_threshold", self.DEFAULT_IR_THRESHOLD)

        self.connector = MarketDataConnector()
        self.engineer = FeatureEngineer()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_validation(self, **kwargs) -> dict:
        """
        Execute WFO grid search over alpha_score_threshold × ic_window.

        Returns
        -------
        dict with keys:
          best_alpha_threshold, best_ic_window, best_ir, all_results, passed
        """
        print(f"\n{'='*60}")
        print(f"  Walk-Forward Optimization — {self.ticker}")
        print(f"  Min train: {self.min_train}d | OOS step: {self.oos_step}d")
        print(f"  Alpha thresholds: {self.alpha_thresholds}")
        print(f"  IC windows: {self.ic_windows}")
        print(f"  IR threshold (accept): {self.ir_threshold}")
        print(f"{'='*60}\n")

        # Fetch daily data (need at least min_train + several OOS steps)
        print(f"Fetching daily data for {self.ticker}...")
        df_raw = self.connector.get_daily(self.ticker, period="5y")
        if df_raw.empty:
            print(f"[ERROR] No data returned for {self.ticker}")
            return {"passed": False, "reason": "no_data"}

        # Engineer full set of features
        print("Engineering features...")
        df = self.engineer.compute_all_features(df_raw, timeframe="daily")
        if df.empty or len(df) < self.min_train + self.oos_step:
            print(f"[ERROR] Insufficient data: {len(df)} rows after engineering")
            return {"passed": False, "reason": "insufficient_data"}

        print(f"Data ready: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}\n")

        # Grid search
        all_results = []
        for alpha_thr in self.alpha_thresholds:
            for ic_win in self.ic_windows:
                ir = self._run_wfo(df, alpha_thr, ic_win)
                status = "✓ PASS" if ir >= self.ir_threshold else "✗ FAIL"
                print(f"  alpha_thr={alpha_thr:.2f} | ic_window={ic_win:3d}d | OOS IR={ir:+.3f}  {status}")
                all_results.append({
                    "alpha_threshold": alpha_thr,
                    "ic_window": ic_win,
                    "oos_ir": round(ir, 4),
                    "passed": ir >= self.ir_threshold,
                })

        # Select best
        passing = [r for r in all_results if r["passed"]]
        if passing:
            best = max(passing, key=lambda r: r["oos_ir"])
        else:
            best = max(all_results, key=lambda r: r["oos_ir"])

        print(f"\n{'='*60}")
        print(f"  Best params: alpha_threshold={best['alpha_threshold']:.2f} | ic_window={best['ic_window']}d | OOS IR={best['oos_ir']:+.3f}")
        if passing:
            print(f"  ✓ {len(passing)} / {len(all_results)} parameter sets passed IR > {self.ir_threshold}")
        else:
            print(f"  ✗ NO parameter sets passed IR > {self.ir_threshold} — system not ready for live trading")
        print(f"{'='*60}\n")

        result_payload = {
            "best_alpha_threshold": best["alpha_threshold"],
            "best_ic_window": best["ic_window"],
            "best_ir": best["oos_ir"],
            "all_results": all_results,
            "passed": len(passing) > 0,
            "ticker": self.ticker,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            from src.alpha.regime import RegimeDetector

            _rd = RegimeDetector()
            _tn = min(252, len(df))
            if _tn >= 20:
                _rd.train(df.iloc[:_tn])
            self._wfo_regime_detector = _rd
        except Exception as exc:
            logger.warning("WFO regime detector train failed: %s", exc)
            self._wfo_regime_detector = None

        try:
            self._collect_regime_folds = True
            self._wfo_fold_regime_accumulator = []
            _ = self._run_wfo(df, best["alpha_threshold"], best["ic_window"])
        except Exception as exc:
            logger.warning("WFO regime fold collection failed: %s", exc)
        finally:
            self._collect_regime_folds = False

        try:
            _acc = getattr(self, "_wfo_fold_regime_accumulator", [])
            result_payload["regime_analysis_by_fold"] = list(_acc)
            _all_fold_regime = [x for x in _acc if isinstance(x, dict)]
            result_payload["regime_analysis"] = RegimePerformanceTracker().aggregate_folds(
                _all_fold_regime
            )
        except Exception as e:
            logger.warning("WFO regime aggregation failed: %s", e)
            result_payload["regime_analysis"] = {}

        return result_payload

    # ------------------------------------------------------------------
    # Single WFO run for one parameter set
    # ------------------------------------------------------------------

    def _run_wfo(self, df: pd.DataFrame, alpha_threshold: float, ic_window: int) -> float:
        """
        Expanding-window WFO for one (alpha_threshold, ic_window) pair.

        Returns
        -------
        float : out-of-sample Information Ratio = mean(OOS IC) / std(OOS IC)
                Returns 0.0 if fewer than 2 OOS steps are available.
        """
        model = AlphaFactorModel(alpha_threshold=alpha_threshold, ic_window=ic_window)
        fwd_returns = df["Returns"].shift(-1)  # forward 1-day return

        oos_ics: list[float] = []
        n = len(df)

        # Expanding window: train on [0:train_end], test on [train_end : train_end+oos_step]
        train_end = self.min_train
        while train_end + self.oos_step <= n:
            train_df = df.iloc[:train_end]
            oos_df = df.iloc[train_end: train_end + self.oos_step]
            oos_fwd = fwd_returns.iloc[train_end: train_end + self.oos_step]
            fold_trades: list[dict] = []

            if len(train_df) < ic_window + 10:
                train_end += self.oos_step
                continue

            try:
                # Score each OOS bar individually (using train history + OOS bar)
                for i in range(len(oos_df)):
                    # Use a window ending at this OOS bar
                    window_end = train_end + i + 1
                    window_df = df.iloc[max(0, window_end - ic_window - 10): window_end]
                    if len(window_df) < ic_window + 5:
                        continue

                    result = model.score(window_df, ml_score=0.0, hurst=0.5)
                    alpha_score = result.get("alpha_score", 0.0)
                    actual_fwd = float(oos_fwd.iloc[i]) if i < len(oos_fwd) else np.nan

                    if getattr(self, "_collect_regime_folds", False):
                        try:
                            regime_guess = "SIDEWAYS"
                            det = getattr(self, "_wfo_regime_detector", None)
                            if det is not None and getattr(det, "_trained", False):
                                try:
                                    win = df.iloc[max(0, window_end - 60): window_end]
                                    if len(win) >= 20:
                                        regime_guess, _ = det.predict(win.tail(60))
                                except Exception:
                                    regime_guess = "SIDEWAYS"
                            if not np.isnan(actual_fwd):
                                fold_trades.append({
                                    "regime_at_entry": str(regime_guess),
                                    "net_pnl_pct": float(actual_fwd) * 100.0,
                                    "outcome": "WIN" if actual_fwd > 0 else "LOSS",
                                    "hold_bars": 1.0,
                                })
                        except Exception:
                            pass

                    if not np.isnan(actual_fwd) and not np.isnan(alpha_score):
                        oos_ics.append(alpha_score * np.sign(actual_fwd))

            except Exception as exc:
                logger.debug("WFO step error (alpha=%.2f, ic=%d): %s", alpha_threshold, ic_window, exc)

            if getattr(self, "_collect_regime_folds", False):
                try:
                    _fold_regime_stats = RegimePerformanceTracker().compute_regime_stats(fold_trades)
                    self._wfo_fold_regime_accumulator.append(_fold_regime_stats)
                except Exception as e:
                    logger.warning("Fold regime analysis failed: %s", e)
                    self._wfo_fold_regime_accumulator.append({})

            train_end += self.oos_step

        if len(oos_ics) < 2:
            return 0.0

        ic_array = np.array(oos_ics)
        ic_mean = float(np.mean(ic_array))
        ic_std = float(np.std(ic_array))
        return ic_mean / ic_std if ic_std > 1e-9 else 0.0

    def run_with_cpcv(self, ticker: str | None = None) -> dict:
        """Run both WFO and CPCV, return a combined deployment verdict."""
        tk = ticker or self.ticker
        df_raw = self.connector.get_daily(tk, period="5y")
        if df_raw.empty:
            return {"ticker": tk, "combined_verdict": "FAIL", "error": "no_data"}

        feat_df = self.engineer.compute_all_features(df_raw, timeframe="daily", ticker=tk)
        if feat_df.empty or len(feat_df) < self.min_train + self.oos_step:
            return {"ticker": tk, "combined_verdict": "FAIL", "error": "insufficient_data"}

        base_alpha = float(self.alpha_thresholds[0]) if self.alpha_thresholds else 0.30
        base_ic_window = int(self.ic_windows[0]) if self.ic_windows else 60
        model = AlphaFactorModel(alpha_threshold=base_alpha, ic_window=base_ic_window)
        close = feat_df["Close"].astype(float)
        rets = close.pct_change().fillna(0.0).values

        def _strategy_fn(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
            def _build_slice_returns(indices: np.ndarray) -> np.ndarray:
                sig = np.zeros(len(frame), dtype=float)
                for i in indices:
                    if i < 80:
                        continue
                    window = frame.iloc[max(0, i - 200): i + 1]
                    out = model.score(window, asset=tk, asset_class=("indian_stock" if tk.upper().endswith((".NS", ".BO")) else "us_stock"))
                    raw = str(out.get("signal", "HOLD")).upper()
                    sig[i] = 1.0 if raw == "BUY" else 0.0
                pos = np.roll(sig, 1)
                pos[0] = 0.0
                r = rets * pos
                return r[indices]

            return {
                "is_returns": _build_slice_returns(train_idx),
                "oos_returns": _build_slice_returns(test_idx),
            }

        cpcv = CPCVValidator()
        cpcv_result = cpcv.run_cpcv(feat_df, _strategy_fn)
        wfo_result = self.run_validation()

        cpcv_pass = str(cpcv_result.get("verdict", "FAIL")).upper() == "PASS"
        wfo_pass = bool(wfo_result.get("passed", False))
        if cpcv_pass and wfo_pass:
            combined = "PASS"
        elif wfo_pass or str(cpcv_result.get("verdict", "FAIL")).upper() == "WARN":
            combined = "WARN"
        else:
            combined = "FAIL"

        return {
            "ticker": tk,
            "wfo": wfo_result,
            "cpcv": cpcv_result,
            "combined_verdict": combined,
            "timestamp": datetime.utcnow().isoformat(),
        }
