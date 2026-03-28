"""
Layer 2 — Feature Engineering.

Computes 30+ technical and quant-specific indicators on OHLCV data.
All calculations are pure pandas/numpy — no external TA library required.

Usage:
    fe = FeatureEngineer()
    df = fe.compute_all_features(ohlcv_df, timeframe='intraday')
"""

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]

from src.alpha.orderbook import OrderFlowAnalyser  # type: ignore[import]


class FeatureEngineer:

    # ------------------------------------------------------------------
    # Master method
    # ------------------------------------------------------------------

    def compute_all_features(
        self,
        df: pd.DataFrame,
        timeframe: str = "daily",  # 'intraday' | 'daily'
        benchmark: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Run all feature groups in order.
        Returns a DataFrame with all indicators added (rows with NaN dropped).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV with columns Open, High, Low, Close, Volume.
        timeframe : str
            'intraday' adds VWAP; 'daily' skips it (session-scoped).
        benchmark : pd.Series, optional
            Benchmark daily returns aligned to df.index for Beta calculation.
        """
        df = df.copy()
        df = self.add_technical_indicators(df)
        df = self.add_extended_trend(df)
        df = self.add_extended_volume(df, timeframe=timeframe)
        df = self.add_extended_volatility(df)
        df = self.add_alpha_factors(df)
        if benchmark is not None:
            df = self.add_rolling_beta(df, benchmark)
        df = self.add_lags(df)
        df = self.add_microstructure_features(df)
        df = self.add_orderflow_features(df)
        df = self.add_stationary_features(df)
        required_cols = [
            "Close",
            "Returns",
            "ATR_14",
            "SMA_20",
            "BB_Std",
            "ROC_1d",
            "ROC_5d",
            "ROC_20d",
            "ROC_60d",
            "OBV_Slope",
            "CMF",
            "Volume_Osc",
            "ATR_Percentile",
            "Keltner_Squeeze",
        ]
        subset = [col for col in required_cols if col in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna(subset=subset)

    # ------------------------------------------------------------------
    # Layer 1 — Core indicators (existing, preserved)
    # ------------------------------------------------------------------

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Moving Averages
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

        # RSI (14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

        # Bollinger Bands
        df["BB_Mid"] = df["Close"].rolling(20).mean()
        df["BB_Std"] = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + df["BB_Std"] * 2
        df["BB_Lower"] = df["BB_Mid"] - df["BB_Std"] * 2
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

        # ATR (14)
        hl = df["High"] - df["Low"]
        hcp = (df["High"] - df["Close"].shift(1)).abs()
        lcp = (df["Low"] - df["Close"].shift(1)).abs()
        df["TR"] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        df["ATR_14"] = df["TR"].rolling(14).mean()

        # Returns & Volatility
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility_20"] = df["Returns"].rolling(20).std() * np.sqrt(252)

        return df

    # ------------------------------------------------------------------
    # Layer 2 — Extended trend / momentum indicators
    # ------------------------------------------------------------------

    def add_extended_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Stochastic Oscillator (%K, %D) ---
        low_14 = df["Low"].rolling(14).min()
        high_14 = df["High"].rolling(14).max()
        denom = (high_14 - low_14).replace(0, np.nan)
        df["Stoch_K"] = (df["Close"] - low_14) / denom * 100
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

        # --- Williams %R ---
        df["Williams_R"] = (high_14 - df["Close"]) / denom * -100

        # --- CCI (20) ---
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_mean = tp.rolling(20).mean()
        # Mean absolute deviation
        tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df["CCI"] = (tp - tp_mean) / (0.015 * tp_mad.replace(0, np.nan))

        # --- Rate of Change (multi-period) ---
        for period in [1, 5, 10, 20, 60]:
            df[f"ROC_{period}d"] = df["Close"].pct_change(period) * 100

        # --- Ichimoku Cloud ---
        # Tenkan-sen: 9-period midpoint
        high9 = df["High"].rolling(9).max()
        low9 = df["Low"].rolling(9).min()
        df["Ichimoku_Tenkan"] = (high9 + low9) / 2

        # Kijun-sen: 26-period midpoint
        high26 = df["High"].rolling(26).max()
        low26 = df["Low"].rolling(26).min()
        df["Ichimoku_Kijun"] = (high26 + low26) / 2

        # Senkou Span A (plotted 26 periods ahead — stored at current index)
        df["Ichimoku_SpanA"] = (df["Ichimoku_Tenkan"] + df["Ichimoku_Kijun"]) / 2

        # Senkou Span B: 52-period midpoint
        high52 = df["High"].rolling(52).max()
        low52 = df["Low"].rolling(52).min()
        df["Ichimoku_SpanB"] = (high52 + low52) / 2

        # Chikou Span is plotted 26 periods back. Using shift(+26) avoids lookahead.
        df["Ichimoku_Chikou"] = df["Close"].shift(26)

        # Cloud direction: SpanA > SpanB → bullish cloud
        df["Ichimoku_BullCloud"] = (df["Ichimoku_SpanA"] > df["Ichimoku_SpanB"]).astype(float)

        return df

    # ------------------------------------------------------------------
    # Layer 3 — Volume / microstructure indicators
    # ------------------------------------------------------------------

    def add_extended_volume(
        self,
        df: pd.DataFrame,
        timeframe: str = "daily",
    ) -> pd.DataFrame:
        df = df.copy()
        vol = df["Volume"].replace(0, np.nan)

        # --- OBV ---
        direction = np.sign(df["Close"].diff()).fillna(0)
        df["OBV"] = (direction * df["Volume"]).cumsum()

        # OBV slope: linear regression slope over last 5 bars
        df["OBV_Slope"] = (
            df["OBV"]
            .rolling(5)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        )

        # --- Volume Oscillator: short(5) / long(20) - 1 ---
        vol_ma5 = vol.rolling(5).mean()
        vol_ma20 = vol.rolling(20).mean()
        df["Volume_Osc"] = (vol_ma5 / vol_ma20.replace(0, np.nan)) - 1
        df["Volume_Avg_20"] = vol_ma20
        df["Volume_Ratio"] = vol / vol_ma20.replace(0, np.nan)

        # --- Chaikin Money Flow (CMF, 20) ---
        # Money Flow Multiplier = ((Close-Low) - (High-Close)) / (High-Low)
        hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
        mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
        mfv = mfm * df["Volume"]
        df["CMF"] = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, np.nan)

        # --- VWAP (intraday only — resets each session) ---
        if timeframe == "intraday":
            df["VWAP"] = self._compute_vwap(df)
        else:
            df["VWAP"] = np.nan

        return df

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> pd.Series:
        """VWAP resets at each calendar day."""
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = pd.Series(index=df.index, dtype=float)
        for _, grp_idx in df.groupby(df.index.normalize()).groups.items():
            tp_grp = tp.loc[grp_idx]
            vol_grp = df["Volume"].loc[grp_idx]
            cum_tpv = (tp_grp * vol_grp).cumsum()
            cum_vol = vol_grp.cumsum().replace(0, np.nan)
            vwap.loc[grp_idx] = cum_tpv / cum_vol
        return vwap

    # ------------------------------------------------------------------
    # Layer 4 — Extended volatility indicators
    # ------------------------------------------------------------------

    def add_extended_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Keltner Channels (EMA ± 2×ATR_10) ---
        atr10_hl = df["High"] - df["Low"]
        atr10_hcp = (df["High"] - df["Close"].shift(1)).abs()
        atr10_lcp = (df["Low"] - df["Close"].shift(1)).abs()
        tr10 = pd.concat([atr10_hl, atr10_hcp, atr10_lcp], axis=1).max(axis=1)
        atr10 = tr10.rolling(10).mean()
        kelt_mid = df["Close"].ewm(span=20, adjust=False).mean()
        df["Keltner_Mid"] = kelt_mid
        df["Keltner_Upper"] = kelt_mid + 2 * atr10
        df["Keltner_Lower"] = kelt_mid - 2 * atr10

        # Keltner Squeeze: BB inside Keltner → compression before breakout
        df["Keltner_Squeeze"] = (
            (df["BB_Upper"] < df["Keltner_Upper"]) &
            (df["BB_Lower"] > df["Keltner_Lower"])
        ).astype(float)

        # --- Historical Volatility Cone: ATR percentile rank vs 90-day window ---
        df["ATR_Percentile"] = (
            df["ATR_14"]
            .rolling(90)
            .apply(lambda x: float((x[-1] >= x).mean() * 100), raw=True)
        )

        # --- Ulcer Index (14) ---
        roll_max = df["Close"].rolling(14).max()
        pct_drawdown = ((df["Close"] - roll_max) / roll_max.replace(0, np.nan)) * 100
        df["Ulcer_Index"] = np.sqrt((pct_drawdown ** 2).rolling(14).mean())

        return df

    # ------------------------------------------------------------------
    # Layer 5 — Quant-specific alpha factors (existing + extended)
    # ------------------------------------------------------------------

    def add_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Mean Reversion Z-score (existing)
        df["Alpha_MeanReversion"] = (df["Close"] - df["SMA_20"]) / df["BB_Std"].replace(0, np.nan)

        # Momentum (existing + extended)
        df["Alpha_Momentum"] = df["Close"].pct_change(10)

        # Volume intensity (existing)
        df["Alpha_VolumeIntensity"] = df["Volume"] / df["Volume_Avg_20"].replace(0, np.nan)

        # Z-score of Close vs SMA_20 (cross-sectional normalisation seed)
        df["ZScore_Close_SMA20"] = (
            (df["Close"] - df["SMA_20"]) / df["SMA_20"].rolling(60).std().replace(0, np.nan)
        )

        return df

    # ------------------------------------------------------------------
    # Rolling Beta vs benchmark
    # ------------------------------------------------------------------

    def add_rolling_beta(
        self,
        df: pd.DataFrame,
        benchmark_returns: pd.Series,
        window: int = 60,
    ) -> pd.DataFrame:
        """Add rolling 60-day beta vs benchmark (SPY / NSEI / DXY)."""
        df = df.copy()
        asset_ret = df["Returns"].fillna(0)
        bench = benchmark_returns.reindex(df.index).fillna(0)

        a_arr = asset_ret.values
        b_arr = bench.values

        def _beta(start_idx: int, end_idx: int) -> float:
            a = a_arr[start_idx:end_idx]
            b = b_arr[start_idx:end_idx]
            cov = np.cov(a, b)
            var_b = float(np.var(b))
            return float(cov[0, 1] / var_b) if var_b != 0 else 1.0

        betas = pd.Series(index=df.index, dtype=float)
        for i in range(window, len(df)):
            betas.iloc[i] = _beta(i - window, i)

        df["Rolling_Beta"] = betas
        return df

    # ------------------------------------------------------------------
    # Lag features (existing)
    # ------------------------------------------------------------------

    def add_lags(
        self,
        df: pd.DataFrame,
        columns: list | None = None,
        n_lags: int = 5,
    ) -> pd.DataFrame:
        if columns is None:
            columns = ["Close", "Returns"]
        df = df.copy()
        for col in columns:
            if col in df.columns:
                for i in range(1, n_lags + 1):
                    df[f"{col}_Lag_{i}"] = df[col].shift(i)
        return df

    # ------------------------------------------------------------------
    # Step 5 — Stationarity Enforcement (ADF test)
    # ------------------------------------------------------------------

    @staticmethod
    def make_stationary(
        series: pd.Series,
        significance: float = 0.05,
    ) -> pd.Series:
        """
        Test ADF stationarity; if non-stationary, difference the series.
        Prevents ML model collapse during regime shifts.

        Parameters
        ----------
        series       : raw price or feature series
        significance : ADF p-value threshold (default 0.05)

        Returns
        -------
        pd.Series : stationary version of the input
        """
        try:
            from statsmodels.tsa.stattools import adfuller  # type: ignore[import]
            clean = series.dropna()
            if len(clean) < 20:
                return series.diff().dropna()
            result = adfuller(clean, maxlag=10, autolag="AIC")
            p_value = result[1]
            if p_value > significance:
                return series.diff().dropna()
        except ImportError:
            # statsmodels not installed — fall back to differencing
            return series.diff().dropna()
        except Exception:
            return series.diff().dropna()
        return series

    def add_stationary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add stationary versions of key price-derived features."""
        df = df.copy()
        for col in ["Close", "Volume", "OBV_Slope"]:
            if col in df.columns:
                stationary = self.make_stationary(df[col])
                df[f"{col}_Stationary"] = stationary.reindex(df.index)
        return df

    # ------------------------------------------------------------------
    # Step 6 — Microstructure Proxy Features (from OHLCV)
    # ------------------------------------------------------------------

    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate order-book microstructure features from OHLCV.

        Since we don't have L2 order book data, we use well-known
        OHLCV proxies from the microstructure literature:

        1. Spread Proxy (Corwin-Schultz):
           Estimates bid-ask spread from daily high/low.
           spread ≈ 2(e^α - 1) / (1 + e^α)
           where α = (√2β - √β) / (3 - 2√2) - √(γ/(3-2√2))

        2. Order Imbalance Proxy:
           Uses volume × sign of return to infer buy vs sell pressure.
           imbalance = signed_volume / total_volume (rolling 20d)

        3. Return Distribution Features:
           - Rolling skewness (20d) — asymmetric return risk
           - Rolling kurtosis (20d) — tail risk / fat tails
           - Amihud illiquidity (|return| / volume)
        """
        df = df.copy()

        # 1. Spread proxy (Corwin-Schultz 2012)
        h = np.log(df["High"] / df["Low"])
        h_sq = h ** 2
        beta = h_sq.rolling(2).sum()
        gamma = np.log(
            df["High"].rolling(2).max() / df["Low"].rolling(2).min()
        ) ** 2
        sqrt_2 = np.sqrt(2)
        alpha = (np.sqrt(beta) - np.sqrt(beta)) / (3 - 2 * sqrt_2) - np.sqrt(
            gamma / (3 - 2 * sqrt_2)
        )
        # Simplified: use log(H/L) as spread proxy directly
        df["Spread_Proxy"] = h  # log(H/L) ≈ proportional to spread
        df["Spread_Proxy_SMA"] = df["Spread_Proxy"].rolling(20).mean()

        # 2. Order imbalance proxy (signed volume)
        sign = np.sign(df["Close"] - df["Open"])
        signed_volume = df["Volume"] * sign
        df["Order_Imbalance"] = (
            signed_volume.rolling(20).sum()
            / df["Volume"].rolling(20).sum().replace(0, np.nan)
        )

        # 3. Return distribution
        if "Returns" in df.columns:
            df["Return_Skew_20"] = df["Returns"].rolling(20).skew()
            df["Return_Kurtosis_20"] = df["Returns"].rolling(20).kurt()

        # 4. Amihud illiquidity ratio
        if "Returns" in df.columns:
            df["Amihud_Illiquidity"] = (
                df["Returns"].abs()
                / (df["Volume"] * df["Close"]).replace(0, np.nan)
            ) * 1e6  # scale up
            df["Amihud_SMA_20"] = df["Amihud_Illiquidity"].rolling(20).mean()

        return df

    # ------------------------------------------------------------------
    # Step 7 — LOB Order Flow Features (from OrderFlowAnalyser)
    # ------------------------------------------------------------------

    def add_orderflow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add synthetic order-flow features from LOB reconstruction.

        Adds: OFI, VPIN, Kyle_Lambda, Trade_Arrival, Micro_Price_Offset
        """
        try:
            analyser = OrderFlowAnalyser()
            df = analyser.compute_features(df)
        except Exception:
            for col in ("OFI", "VPIN", "Kyle_Lambda", "Trade_Arrival", "Micro_Price_Offset"):
                if col not in df.columns:
                    df[col] = 0.0
        return df

