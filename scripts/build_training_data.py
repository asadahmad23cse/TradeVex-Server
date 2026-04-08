"""
BTC ML Model Training Pipeline
================================
Ek command se sab kuch ho jaata hai:

    python scripts/build_training_data.py

Yeh script:
  1. Binance se 2 years ka 15m BTCUSDT data fetch karta hai (free, no API key)
  2. Saare 42 ML features calculate karta hai
  3. was_profitable labels generate karta hai
  4. LightGBM + XGBoost models train karta hai (4 regimes ke liye)
  5. LSTM bhi train karta hai (--skip-lstm se skip kar sakte hain)
  6. Models artifacts folder mein save karta hai
  7. Verify karta hai ki sab sahi hua

Usage:
    python scripts/build_training_data.py
    python scripts/build_training_data.py --skip-lstm     # LSTM skip (fast mode)
    python scripts/build_training_data.py --years 1       # 1 year data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --- Project root ko path mein add karo ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# STEP 1: Binance se data fetch karo
# ============================================================

def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "15m", years: int = 2) -> pd.DataFrame:
    """Binance Futures se historical candles fetch karo. Free, no API key needed."""
    print(f"\n[1/5] Binance se {years} years ka {interval} data fetch ho raha hai...")

    url = "https://fapi.binance.com/fapi/v1/klines"
    limit = 1500
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - (years * 365 * 24 * 60 * 60 * 1000)

    all_rows: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        try:
            resp = requests.get(url, params={
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": limit,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Fetch error, retry kar raha hun... ({e})")
            time.sleep(3)
            continue

        if not data:
            break

        all_rows.extend(data)
        last_open_ms = int(data[-1][0])

        if len(data) < limit:
            break

        current_start = last_open_ms + 1
        fetched_days = (last_open_ms - start_ms) / (1000 * 86400)
        total_days = years * 365
        print(f"  Progress: {fetched_days:.0f}/{total_days} days ({len(all_rows):,} candles)...", end="\r")
        time.sleep(0.15)

    if not all_rows:
        raise RuntimeError("Data fetch fail hua. Internet check karo.")

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"\n  Fetched: {len(df):,} candles  ({df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()})")
    return df


# ============================================================
# STEP 2: Features calculate karo
# ============================================================

def add_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_bollinger(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return mid - 2 * std, mid, mid + 2 * std


def add_vwap_rolling(df: pd.DataFrame, window: int = 96) -> pd.Series:
    """Rolling VWAP (96 bars = 24 saat 15m timeframe pe)"""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    return tp_vol.rolling(window).sum() / df["volume"].rolling(window).sum()


def detect_swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> tuple[pd.Series, pd.Series]:
    """Local swing highs aur lows detect karo."""
    highs = df["high"].rolling(lookback * 2 + 1, center=True).max() == df["high"]
    lows = df["low"].rolling(lookback * 2 + 1, center=True).min() == df["low"]
    return highs, lows


def detect_bos_choch(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Break of Structure aur Change of Character detect karo."""
    n = len(df)
    bos_candles_ago = np.full(n, 50.0)
    choch = np.zeros(n)

    recent_high = df["high"].rolling(20).max().values
    recent_low = df["low"].rolling(20).min().values

    for i in range(20, n):
        # BOS: price breaks above recent high (bullish) or below recent low (bearish)
        if df["close"].iloc[i] > recent_high[i - 1]:
            bos_candles_ago[i] = 1.0
        elif df["close"].iloc[i] < recent_low[i - 1]:
            bos_candles_ago[i] = 1.0
        else:
            bos_candles_ago[i] = min(bos_candles_ago[i - 1] + 1, 50)

        # CHOCH: direction reversal after trend — simplified
        prev_trend = 1 if df["close"].iloc[i - 5] > df["close"].iloc[i - 10] else -1
        curr_trend = 1 if df["close"].iloc[i] > df["close"].iloc[i - 5] else -1
        choch[i] = 1.0 if prev_trend != curr_trend and bos_candles_ago[i] <= 3 else 0.0

    return pd.Series(bos_candles_ago, index=df.index), pd.Series(choch, index=df.index)


def detect_fvg(df: pd.DataFrame) -> pd.Series:
    """Fair Value Gap detect karo (3-candle pattern)."""
    n = len(df)
    fvg = np.zeros(n)
    for i in range(2, n):
        # Bullish FVG: candle[i-2].high < candle[i].low
        if df["high"].iloc[i - 2] < df["low"].iloc[i]:
            fvg[i] = 1.0
        # Bearish FVG: candle[i-2].low > candle[i].high
        elif df["low"].iloc[i - 2] > df["high"].iloc[i]:
            fvg[i] = 1.0
    return pd.Series(fvg, index=df.index)


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Liquidity sweep detect karo: price ne recent high/low ko briefly cross kiya."""
    sweep = np.zeros(len(df))
    for i in range(lookback, len(df)):
        window_high = df["high"].iloc[i - lookback:i].max()
        window_low = df["low"].iloc[i - lookback:i].min()
        # Price wick ne level cross kiya lekin close nahi kiya
        if df["high"].iloc[i] > window_high and df["close"].iloc[i] < window_high:
            sweep[i] = 1.0
        elif df["low"].iloc[i] < window_low and df["close"].iloc[i] > window_low:
            sweep[i] = 1.0
    return pd.Series(sweep, index=df.index)


def detect_ob_distance(df: pd.DataFrame) -> pd.Series:
    """Nearest Order Block distance (swing high/low se approximate karo)."""
    n = len(df)
    ob_dist = np.full(n, 1.0)
    rolling_high = df["high"].rolling(20).max().values
    rolling_low = df["low"].rolling(20).min().values

    for i in range(20, n):
        price = df["close"].iloc[i]
        dist_high = abs(price - rolling_high[i]) / price * 100.0
        dist_low = abs(price - rolling_low[i]) / price * 100.0
        ob_dist[i] = min(dist_high, dist_low)

    return pd.Series(ob_dist, index=df.index)


def compute_mtf_alignment(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe alignment score (0-5).
    15m data se simulate karo: short/medium/long EMAs ka agreement check karo.
    """
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    ema200 = df["close"].ewm(span=200, adjust=False).mean()
    # 1h simulate: 4 bars ka EMA
    ema_1h = df["close"].ewm(span=4 * 9, adjust=False).mean()

    score = (
        (df["close"] > ema9).astype(int) +
        (df["close"] > ema21).astype(int) +
        (ema9 > ema21).astype(int) +
        (ema21 > ema50).astype(int) +
        (df["close"] > ema_1h).astype(int)
    )
    # SHORT direction ke liye reverse — label_generator direction decide karta hai
    return score.astype(float)


def compute_regime(df: pd.DataFrame) -> pd.Series:
    """Regime assign karo price action se."""
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    atr = add_atr(df, 14)
    close = df["close"]

    # Trend strength: ema9 aur ema50 ka separation
    trend_strength = (ema9 - ema50).abs() / close * 100.0

    regimes = []
    for i in range(len(df)):
        if i < 50:
            regimes.append("sideways_range")
            continue

        ts = trend_strength.iloc[i]
        e9, e21, e50 = ema9.iloc[i], ema21.iloc[i], ema50.iloc[i]
        c = close.iloc[i]

        # Volatility check for breakout
        atr_pct = atr.iloc[i] / c * 100.0 if c > 0 else 0

        if ts > 1.5 and e9 > e21 > e50:
            regimes.append("bullish_trend")
        elif ts > 1.5 and e9 < e21 < e50:
            regimes.append("bearish_trend")
        elif atr_pct > 0.8 and abs(c - e50) / c * 100.0 > 1.0:
            # Big move relative to recent range
            regimes.append("breakout_up" if c > e50 else "breakout_down")
        else:
            regimes.append("sideways_range")

    return pd.Series(regimes, index=df.index)


def regime_to_encoded(regime_series: pd.Series) -> pd.Series:
    mapping = {
        "sideways_range": 0.0,
        "bullish_trend": 1.0,
        "bearish_trend": 2.0,
        "breakout_up": 3.0,
        "breakout_down": 3.0,
    }
    return regime_series.map(mapping).fillna(0.0)


def session_encoded(ts_series: pd.Series) -> pd.Series:
    """UTC hour se session encode karo."""
    hour = ts_series.dt.hour
    conditions = [
        hour < 6,
        (hour >= 6) & (hour < 9),
        (hour >= 9) & (hour < 13),
        (hour >= 13) & (hour < 17),
    ]
    choices = [1.0, 2.0, 3.0, 4.0]
    return pd.Series(np.select(conditions, choices, default=0.0), index=ts_series.index)


def vol_regime_encoded(atr_pct: pd.Series) -> pd.Series:
    """ATR % se volatility regime encode karo."""
    conditions = [
        atr_pct < 0.3,
        (atr_pct >= 0.3) & (atr_pct < 0.7),
        atr_pct >= 0.7,
    ]
    choices = [0.0, 1.0, 2.0]
    return pd.Series(np.select(conditions, choices, default=1.0), index=atr_pct.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Saare 42 ML features calculate karo."""
    print("\n[2/5] Features calculate ho rahe hain...")

    close = df["close"]
    price = close

    # EMAs
    ema9 = add_ema(df, 9)
    ema21 = add_ema(df, 21)
    ema50 = add_ema(df, 50)
    ema200 = add_ema(df, 200)

    # ATR
    atr14 = add_atr(df, 14)
    atr_pct = atr14 / price * 100.0

    # Bollinger Bands
    bb_lower, bb_mid, bb_upper = add_bollinger(df, 20)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    bb_pos = ((price - bb_lower) / bb_range).clip(0.0, 1.0)

    # VWAP
    vwap = add_vwap_rolling(df, 96)
    vwap_dist = ((price - vwap) / price * 100.0).fillna(0.0)

    # Price / EMA distances
    df["ema9_dist"] = ((price - ema9) / price * 100.0).fillna(0.0)
    df["ema21_dist"] = ((price - ema21) / price * 100.0).fillna(0.0)
    df["ema50_dist"] = ((price - ema50) / price * 100.0).fillna(0.0)
    df["ema200_dist"] = ((price - ema200) / price * 100.0).fillna(0.0)
    df["atr_normalized"] = atr_pct.fillna(0.0)
    df["bb_position"] = bb_pos.fillna(0.5)
    df["vwap_distance"] = vwap_dist
    df["atr14"] = atr14.fillna(0.0)  # label_generator ke liye

    # SMC / Structure
    df["mtf_alignment_score"] = compute_mtf_alignment(df)
    bos_ago, choch = detect_bos_choch(df)
    df["bos_candles_ago"] = bos_ago
    df["choch_detected"] = choch
    df["ob_distance_pct"] = detect_ob_distance(df)
    df["fvg_present"] = detect_fvg(df)
    df["liquidity_sweep_recent"] = detect_liquidity_sweep(df, 20)

    # Order flow (live data se aata hai — realistic defaults use karo)
    # CVD slope: price momentum se approximate karo
    df["cvd_slope"] = ((close - close.shift(5)) / close.shift(5)).fillna(0.0).clip(-0.02, 0.02)
    # OBI: volume asymmetry se approximate karo
    buy_vol = df["taker_buy_base"]
    total_vol = df["volume"].replace(0, np.nan)
    df["obi"] = (buy_vol / total_vol).fillna(0.5).clip(0.0, 1.0)
    # OFI: candle direction se approximate karo
    df["ofi_5bar"] = ((close - close.shift(1)).rolling(5).sum() / atr14).fillna(0.0).clip(-3.0, 3.0)

    # Defaults for live-only features
    df["whale_buy_ratio"] = 0.5
    df["absorption_strength"] = 0.0
    # Stacked imbalance: OBI consistency se
    obi_consistent = (df["obi"] > 0.6).rolling(3).sum()
    df["stacked_imbalance_direction"] = np.where(obi_consistent >= 3, 1.0,
                                        np.where((df["obi"] < 0.4).rolling(3).sum() >= 3, -1.0, 0.0))
    df["single_bar_divergence"] = 0.0
    df["iceberg_direction"] = 0.0
    df["cross_exchange_cvd_divergence"] = 0.0
    df["speed_of_tape_ratio"] = 1.0

    # Derivatives (reasonable defaults)
    df["funding_rate"] = 0.0001
    df["funding_roc"] = 0.0
    df["oi_change_1h"] = 0.0
    df["ls_ratio"] = 0.5
    df["liq_cluster_above_pct"] = 0.5
    df["liq_cluster_below_pct"] = 0.5
    df["max_pain_distance_pct"] = 0.0
    df["iv_skew"] = 0.0

    # Options
    df["put_call_ratio"] = 0.7
    df["iv_rv_ratio"] = 1.0
    df["options_expiry_hours"] = 168.0

    # On-chain
    df["exchange_netflow_score"] = 0.0
    df["sopr"] = 1.0
    df["lth_supply_change"] = 0.0
    df["whale_exchange_deposit_count"] = 0.0
    df["whale_net_flow_score"] = 0.0

    # Macro
    df["fng_score"] = 50.0
    df["dxy_bias_encoded"] = 0.0
    df["vix_level"] = 20.0
    df["us_10y_yield_trend"] = 0.0
    df["spx_btc_correlation"] = 0.5
    df["spx_intraday_direction"] = 0.0
    df["gold_btc_agreement"] = 0.0

    # Crypto correlations
    df["eth_btc_ratio_trend"] = 0.0
    df["btc_dominance_trend"] = 0.0

    # Context encodings
    df["session_encoded"] = session_encoded(df["timestamp"])
    df["vol_regime_encoded"] = vol_regime_encoded(atr_pct)

    # Regime
    df["regime"] = compute_regime(df)
    df["regime_encoded"] = regime_to_encoded(df["regime"])

    print(f"  Features ready. Rows: {len(df):,}")
    regime_counts = df["regime"].value_counts()
    for r, c in regime_counts.items():
        print(f"    {r}: {c:,} rows")

    return df


# ============================================================
# STEP 3: Labels generate karo
# ============================================================

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """was_profitable label lagao: 1=profitable, 0=loss"""
    print("\n[3/5] Labels generate ho rahe hain (thoda time lagega)...")
    from btc_intelligence.models.label_generator import generate_labels
    df_labeled = generate_labels(df, horizon=48)
    profitable = df_labeled["was_profitable"].sum()
    total = len(df_labeled)
    print(f"  Labels done. Profitable: {profitable:,}/{total:,} ({profitable/total*100:.1f}%)")
    return df_labeled


# ============================================================
# STEP 4: Models train karo
# ============================================================

def train_all_models(csv_path: str, out_dir: str, skip_lstm: bool = False) -> dict:
    """Sab regimes ke liye LightGBM + XGBoost + LSTM train karo."""
    print("\n[4/5] Models training shuru ho rahi hai...")

    from btc_intelligence.models.train_lgbm import train_lgbm
    from btc_intelligence.models.train_xgb import train_xgb

    root = Path(out_dir)
    results: dict = {}
    regimes = ["bullish_trend", "bearish_trend", "sideways_range", "breakout"]

    for regime in regimes:
        print(f"\n  --- {regime.upper()} ---")
        regime_dir = root / regime
        regime_dir.mkdir(parents=True, exist_ok=True)

        r1 = train_lgbm(csv_path, str(regime_dir), regime=regime)
        status = "TRAINED" if r1.get("trained") else f"SKIP (rows={r1.get('rows',0)} < 300)"
        print(f"  LightGBM: {status}  AUC={r1.get('walk_forward_auc_lgbm', 0):.3f}")

        r2 = train_xgb(csv_path, str(regime_dir), regime=regime)
        status = "TRAINED" if r2.get("trained") else f"SKIP (rows={r2.get('rows',0)} < 300)"
        print(f"  XGBoost:  {status}  AUC={r2.get('walk_forward_auc_xgb', 0):.3f}")

        r3 = {"trained": False, "skipped": True}
        if not skip_lstm:
            try:
                from btc_intelligence.models.train_attention_lstm import train_attention_lstm
                print(f"  LSTM: training...", end=" ", flush=True)
                r3 = train_attention_lstm(csv_path, str(regime_dir), regime=regime)
                status = "TRAINED" if r3.get("trained") else f"SKIP (rows={r3.get('rows',0)} < 300)"
                print(f"{status}")
            except Exception as e:
                print(f"SKIP ({e})")
                r3 = {"trained": False, "error": str(e)}
        else:
            print(f"  LSTM: SKIPPED (--skip-lstm flag)")

        # Metadata save karo
        meta = {
            "version": f"ml_{regime}_v2",
            "regime": regime,
            "lightgbm": r1,
            "xgboost": r2,
            "lstm": r3,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        (regime_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        results[regime] = meta

    # Root metadata
    root_meta = {
        "version": f"ml_ensemble_v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "regimes": results,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "metadata.json").write_text(
        json.dumps(root_meta, indent=2), encoding="utf-8"
    )
    print(f"\n  Models saved to: {root}")
    return results


# ============================================================
# STEP 5: Verify karo
# ============================================================

def verify_results(out_dir: str) -> None:
    """Check karo ki sab models ban gaye hain."""
    print("\n[5/5] Verification...")
    root = Path(out_dir)
    all_ok = True

    for regime in ["bullish_trend", "bearish_trend", "sideways_range", "breakout"]:
        lgbm = root / regime / "lightgbm.pkl"
        xgb = root / regime / "xgboost.pkl"
        meta = root / regime / "metadata.json"

        lgbm_ok = lgbm.exists()
        xgb_ok = xgb.exists()
        meta_ok = meta.exists()

        status = "OK" if (lgbm_ok and xgb_ok) else "PARTIAL"
        print(f"  {regime}: LightGBM={'YES' if lgbm_ok else 'NO'}  XGBoost={'YES' if xgb_ok else 'NO'}  [{status}]")
        if not (lgbm_ok and xgb_ok):
            all_ok = False

    if all_ok:
        print("\n  ALL MODELS READY!")
        print("\n  Ab server restart karo aur check karo:")
        print("    curl http://127.0.0.1:9000/signal | python -m json.tool | grep model_version")
        print("    Expected: \"model_version\": \"ml_sideways_range_v2\"")
    else:
        print("\n  Kuch models train nahi hue. Data rows check karo (minimum 300 per regime chahiye).")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="BTC ML Training Pipeline")
    parser.add_argument("--years", type=int, default=2, help="Kitne saal ka data (default: 2)")
    parser.add_argument("--skip-lstm", action="store_true", help="LSTM skip karo (fast mode, ~10 min)")
    parser.add_argument("--data-only", action="store_true", help="Sirf data fetch karo, training mat karo")
    parser.add_argument("--out-dir", default="btc_intelligence/models/artifacts", help="Models save karne ki jagah")
    parser.add_argument("--csv-path", default="data/training_data_15m.csv", help="Training CSV path")
    args = parser.parse_args()

    start_time = time.time()
    csv_path = ROOT / args.csv_path
    out_dir = ROOT / args.out_dir

    print("=" * 60)
    print("  BTC ML Training Pipeline")
    print("=" * 60)
    print(f"  Data:      {args.years} years of 15m BTCUSDT")
    print(f"  CSV:       {csv_path}")
    print(f"  Models:    {out_dir}")
    print(f"  LSTM:      {'SKIP' if args.skip_lstm else 'YES (slow, ~15 min)'}")
    print("=" * 60)

    # Step 1–3: Agar CSV pehle se hai to reuse (fetch/features/labels skip — fast path)
    if csv_path.exists():
        print(f"\n[1/5]-[3/5] Existing CSV reuse: {csv_path}")
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        print(f"  Loaded: {len(df):,} rows")
    else:
        print("\n[1/5] Binance se data fetch...")
        df = fetch_binance_klines(years=args.years)
        df = build_features(df)
        df = add_labels(df)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\n  CSV saved: {csv_path}  ({len(df):,} rows)")

    if args.data_only:
        print("\n  --data-only flag set, training skip.")
        return

    # Step 4: Train
    train_all_models(str(csv_path), str(out_dir), skip_lstm=args.skip_lstm)

    # Step 5: Verify
    verify_results(str(out_dir))

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Total time: {elapsed:.1f} minutes")
    print("\n  Server restart karo taake models load ho jayein!")
    print("=" * 60)


if __name__ == "__main__":
    main()
