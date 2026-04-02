"""
Layer 4 — Regime Detection via Hidden Markov Model.

Uses GaussianHMM (hmmlearn) with 5 hidden states and 7 emission features:
  1. Daily log-return          — captures drift direction
  2. Realised volatility (20d) — captures volatility regime
  3. Volume ratio              — volume / 20d avg (panic vs accumulation)
  4. ATR percentile rank (90d) — cross-asset vol rank
  5. VIX level (normalised)   — fear gauge
  6. DXY momentum (5d ROC)    — dollar strength (risk-off proxy)
  7. Yield spread (10Y-2Y)    — recession predictor

States are auto-labeled after training by ranking on mean log-return:
  lowest mean return  → BEAR
  middle mean return  → SIDEWAYS
  highest mean return → BULL

Signal filter:
  BULL      → only BUY signals pass
  BEAR      → only SELL signals pass
  SIDEWAYS  → only STRONG signals pass; position size halved

Persistence:
  Model is saved to data/hmm_{asset_class}.pkl after training
  and loaded at startup to avoid cold-start.
"""

import logging
import pickle
import threading
import time as _time
from pathlib import Path

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import yfinance as yf  # type: ignore[import]
from hmmlearn.hmm import GaussianHMM  # type: ignore[import]

logger = logging.getLogger(__name__)

STATES = ["BULL", "BEAR", "SIDEWAYS"]
FIVE_STATES = ["HIGH_VOL_BULL", "BULL", "SIDEWAYS", "BEAR", "HIGH_VOL_BEAR"]
HMM_N_ITER = 300  # increased from 200 for better convergence with 7 features
HMM_PERSIST_DIR = Path("data")

# I2: Module-level cache for macro series (VIX/DXY/TNX/IRX).
# Each asset calling predict() previously triggered 4 yfinance downloads.
# With 31 assets that was 124 downloads per cycle; now it is at most 4.
_MACRO_CACHE: dict[str, tuple[pd.Series, float]] = {}  # ticker → (series, fetch_ts)
_MACRO_CACHE_LOCK = threading.Lock()
MACRO_CACHE_TTL_SEC = 300  # re-fetch at most once every 5 minutes


class RegimeDetector:

    def __init__(self, n_states: int = 5, n_iter: int = HMM_N_ITER):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model: GaussianHMM | None = None
        # Maps HMM state index → human label
        self.state_labels: dict[int, str] = {}
        self._trained = False
        self._online_updates = 0

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build (N, 7) emission matrix from OHLCV + macro features.
        Features 1-4: same as before.
        Features 5-7: VIX, DXY momentum, yield spread (fetched via yfinance).
        """
        n = len(df)
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
        vol_20 = log_ret.rolling(20).std().fillna(log_ret.std())
        vol_avg20 = df["Volume"].rolling(20).mean().replace(0, np.nan)
        vol_ratio = (df["Volume"] / vol_avg20).fillna(1.0)

        def _pct_rank(x: np.ndarray) -> float:
            return float((x[-1] >= x).mean())

        if "ATR_14" in df.columns:
            atr_pct = df["ATR_14"].rolling(90).apply(_pct_rank, raw=True).fillna(0.5)
        else:
            atr_pct = pd.Series(0.5, index=df.index)

        # --- Macro features (GAP 8) ---
        # Try to fetch macro series aligned to df.index; fallback to 0 if unavailable
        vix_series   = self._fetch_macro("^VIX", df.index, col="Close")
        dxy_series   = self._fetch_macro("DX-Y.NYB", df.index, col="Close")
        t10_series   = self._fetch_macro("^TNX", df.index, col="Close")   # 10Y yield
        t2_series    = self._fetch_macro("^IRX", df.index, col="Close")   # 2Y yield (proxy)

        # Normalise VIX: z-score over 50-day window
        vix_z = ((vix_series - vix_series.rolling(50).mean()) /
                 vix_series.rolling(50).std().replace(0, 1)).fillna(0.0)

        # DXY momentum: 5-day ROC
        dxy_mom = dxy_series.pct_change(5).fillna(0.0)

        # Yield spread: 10Y - 2Y; positive = normal, negative = inverted (recession)
        yield_spread = (t10_series - t2_series).fillna(0.0)
        yield_z = ((yield_spread - yield_spread.rolling(50).mean()) /
                   yield_spread.rolling(50).std().replace(0, 1)).fillna(0.0)

        features = np.column_stack([
            log_ret.values,
            vol_20.values,
            vol_ratio.values,
            atr_pct.values,
            vix_z.values,
            dxy_mom.values,
            yield_z.values,
        ]).astype(np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    @staticmethod
    def _fetch_macro(ticker: str, target_index: pd.DatetimeIndex, col: str = "Close") -> pd.Series:
        """
        Fetch a macro series from yfinance and reindex/forward-fill to target_index.
        Returns a zero-filled series on failure.

        I2: Results are cached at module level for MACRO_CACHE_TTL_SEC (300 s) so that
        repeated calls across all assets in the same intraday cycle hit the cache
        instead of issuing a new HTTP request every time.
        """
        now = _time.monotonic()
        # Check cache under lock to avoid thundering-herd on first call
        with _MACRO_CACHE_LOCK:
            entry = _MACRO_CACHE.get(ticker)
            if entry is not None:
                cached_series, cached_ts = entry
                if now - cached_ts < MACRO_CACHE_TTL_SEC:
                    raw_series = cached_series
                else:
                    raw_series = None
            else:
                raw_series = None

        if raw_series is None:
            try:
                start = target_index[0] - pd.Timedelta(days=100)
                end   = target_index[-1] + pd.Timedelta(days=1)
                downloaded = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                raw_series = downloaded[col].dropna()
                with _MACRO_CACHE_LOCK:
                    _MACRO_CACHE[ticker] = (raw_series, now)
            except Exception:
                return pd.Series(0.0, index=target_index)

        try:
            return raw_series.reindex(target_index, method="ffill").fillna(0.0)
        except Exception:
            return pd.Series(0.0, index=target_index)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, asset_class: str = "unknown") -> "RegimeDetector":
        """
        Fit HMM on a rolling window of daily data (typically 252 days).
        Persists trained model to data/hmm_{asset_class}.pkl.
        """
        features = self._build_features(df)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=HMM_N_ITER,
            random_state=42,
            verbose=False,
        )
        self.model.fit(features)

        mean_returns = self.model.means_[:, 0]
        self.state_labels = self._assign_state_labels()

        self._trained = True
        logger.info(
            "HMM trained (7-feat). State labels: %s | mean returns: %s",
            self.state_labels,
            {i: f"{mean_returns[i]:.5f}" for i in range(self.n_states)},
        )

        # Persist model
        if asset_class != "unknown":
            try:
                HMM_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
                self.save(HMM_PERSIST_DIR / f"hmm_{asset_class}.pkl")
            except Exception as exc:
                logger.warning("HMM persist failed: %s", exc)

        return self

    def _assign_state_labels(self) -> dict[int, str]:
        if self.model is None:
            return {}
        mean_returns = self.model.means_[:, 0]
        vol_feature = self.model.means_[:, 1] if self.model.means_.shape[1] > 1 else np.zeros(self.n_states)
        sorted_idx = np.argsort(mean_returns)
        n = self.n_states

        # C6: handle any n_states value instead of hard-coding exactly-5 unpacking
        if n == 1:
            return {int(sorted_idx[0]): "SIDEWAYS"}

        if n == 2:
            return {int(sorted_idx[0]): "BEAR", int(sorted_idx[1]): "BULL"}

        if n == 3:
            return {
                int(sorted_idx[0]): "BEAR",
                int(sorted_idx[1]): "SIDEWAYS",
                int(sorted_idx[2]): "BULL",
            }

        if n == 4:
            bear, low_mid, high_mid, bull = [int(i) for i in sorted_idx]
            return {
                bear: "BEAR",
                low_mid: "SIDEWAYS",
                high_mid: "BULL",
                bull: "HIGH_VOL_BULL",
            }

        # n >= 5: assign primary labels using extreme and median-return states,
        # then map remaining states to nearest label by mean-return proximity.
        labels: dict[int, str] = {}
        lowest = int(sorted_idx[0])
        second_low = int(sorted_idx[1])
        middle = int(sorted_idx[n // 2])
        second_high = int(sorted_idx[-2])
        highest = int(sorted_idx[-1])
        labels[middle] = "SIDEWAYS"
        bull_candidates = [second_high, highest]
        bear_candidates = [lowest, second_low]
        bull_vols = sorted(bull_candidates, key=lambda i: vol_feature[i])
        bear_vols = sorted(bear_candidates, key=lambda i: vol_feature[i])
        labels[bull_vols[0]] = "BULL"
        labels[bull_vols[-1]] = "HIGH_VOL_BULL"
        labels[bear_vols[-1]] = "HIGH_VOL_BEAR"
        labels[bear_vols[0]] = "BEAR"

        anchor_returns = {
            "HIGH_VOL_BEAR": float(mean_returns[bear_vols[-1]]),
            "BEAR": float(mean_returns[bear_vols[0]]),
            "SIDEWAYS": float(mean_returns[middle]),
            "BULL": float(mean_returns[bull_vols[0]]),
            "HIGH_VOL_BULL": float(mean_returns[bull_vols[-1]]),
        }
        labeled_states = set(labels.keys())
        for idx in sorted_idx:
            state_idx = int(idx)
            if state_idx in labeled_states:
                continue
            state_ret = float(mean_returns[state_idx])
            nearest_label = min(
                anchor_returns.keys(),
                key=lambda label: abs(state_ret - anchor_returns[label]),
            )
            labels[state_idx] = nearest_label
        return labels

    @classmethod
    def load_or_create(cls, asset_class: str) -> "RegimeDetector":
        """Load persisted model if it exists, else return untrained instance."""
        path = HMM_PERSIST_DIR / f"hmm_{asset_class}.pkl"
        inst = cls()
        if path.exists():
            try:
                inst.load(path)
                logger.info("HMM model loaded from %s", path)
            except Exception as exc:
                logger.warning("HMM load failed: %s — will retrain", exc)
        return inst

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> tuple[str, dict[str, float]]:
        """
        Predict the current market regime.

        Returns
        -------
        label : str
            'BULL' | 'BEAR' | 'SIDEWAYS'
        probs : dict[str, float]
            Probability of each state for the most recent bar.
        """
        if not self._trained or self.model is None:
            raise RuntimeError("RegimeDetector not trained. Call train() first.")

        features = self._build_features(df)
        hidden_states = self.model.predict(features)
        state_probs = self.model.predict_proba(features)

        current_state = int(hidden_states[-1])
        current_label = self.state_labels.get(current_state, "SIDEWAYS")
        probs = {
            self.state_labels.get(i, str(i)): float(state_probs[-1, i])
            for i in range(self.n_states)
        }
        return current_label, probs

    def allows_signal(self, regime: str, signal: str, strength: str) -> bool:
        """
        Gate logic: returns True if the signal is allowed in the current regime.
        """
        if regime == "BULL":
            return signal == "BUY"
        if regime == "HIGH_VOL_BULL":
            return signal == "BUY" and strength in {"STRONG", "MODERATE"}
        if regime == "BEAR":
            return signal == "SELL"
        if regime == "HIGH_VOL_BEAR":
            return signal == "SELL" and strength in {"STRONG", "MODERATE"}
        # SIDEWAYS — only STRONG signals pass
        return strength == "STRONG"

    def sideways_size_multiplier(self, regime: str) -> float:
        """Adjust size by regime volatility."""
        if regime == "SIDEWAYS":
            return 0.5
        if regime in {"HIGH_VOL_BULL", "HIGH_VOL_BEAR"}:
            return 0.7
        return 1.0

    def online_update(self, df: pd.DataFrame, retrain_window: int = 252) -> bool:
        """
        Approximate incremental update by re-fitting on the most recent window.

        hmmlearn does not expose partial_fit. Re-fitting a rolling window every few
        bars is the safest stable approximation in the current stack.
        """
        if len(df) < max(60, self.n_states * 20):
            return False
        try:
            self.train(df.tail(retrain_window))
            self._online_updates += 1
            return True
        except Exception as exc:
            logger.warning("Online regime update failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "labels": self.state_labels}, f)
        logger.info("RegimeDetector saved to %s", path)

    def load(self, path: str | Path) -> "RegimeDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.state_labels = data["labels"]
        if self.model is not None:
            self.n_states = int(self.model.n_components)
        self._trained = True
        logger.info("RegimeDetector loaded from %s", path)
        return self
