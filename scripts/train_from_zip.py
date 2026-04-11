"""
Historical ZIP se ML Model Training
=====================================
Binance se download kiye hue 1h OHLCV ZIP files se models train karo.

Usage:
    python scripts/train_from_zip.py
    python scripts/train_from_zip.py --zip-dir data/historical
    python scripts/train_from_zip.py --skip-lstm
    python scripts/train_from_zip.py --data-only   # Sirf CSV banao, training mat karo

Steps:
    1. data/historical/ se saare ZIPs extract karta hai
    2. Sab CSVs merge karta hai (sorted by timestamp)
    3. 42 ML features calculate karta hai (1h timeframe ke liye optimized)
    4. was_profitable labels generate karta hai
    5. LightGBM + XGBoost train karta hai (4 regimes)
    6. Meta-labeling Random Forest ke liye training data bhi banata hai
    7. Models save karta hai → btc_intelligence/models/artifacts/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Project root add karo
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────
# STEP 1: ZIP Extract + Merge
# ─────────────────────────────────────────────

BINANCE_COLS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

def load_zips(zip_dir: Path) -> pd.DataFrame:
    """data/historical/ mein se saare BTCUSDT ZIPs extract karo aur merge karo."""
    print(f"\n[1/5] ZIP files load ho rahi hain: {zip_dir}")

    zip_files = sorted(zip_dir.glob("BTCUSDT-1h-*.zip"))
    if not zip_files:
        # Non-1h files bhi try karo
        zip_files = sorted(zip_dir.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError(
            f"Koi ZIP nahi mili: {zip_dir}\n"
            "Pehle data/historical/ mein Binance ZIP files rakh do."
        )

    print(f"  {len(zip_files)} ZIP files mili:")
    for z in zip_files:
        print(f"    {z.name}")

    frames: list[pd.DataFrame] = []

    for zpath in zip_files:
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    print(f"  WARNING: {zpath.name} mein CSV nahi mili, skip.")
                    continue
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f, header=None, names=BINANCE_COLS)
                    frames.append(df)
                    print(f"  {zpath.name}: {len(df):,} rows")
        except Exception as exc:
            print(f"  ERROR {zpath.name}: {exc} — skip")

    if not frames:
        raise RuntimeError("Koi bhi ZIP properly load nahi hua!")

    merged = pd.concat(frames, ignore_index=True)

    # Timestamp fix karo (Binance files can be in s/ms/us/ns)
    ts_numeric = pd.to_numeric(merged["timestamp"], errors="coerce")
    ts_sample = ts_numeric.dropna()
    if ts_sample.empty:
        raise RuntimeError("Timestamp column parse nahi ho rahi.")

    median_ts = float(ts_sample.median())
    if median_ts >= 1e17:
        ts_unit = "ns"
    elif median_ts >= 1e14:
        ts_unit = "us"
    elif median_ts >= 1e11:
        ts_unit = "ms"
    else:
        ts_unit = "s"

    merged["timestamp"] = pd.to_datetime(
        ts_numeric,
        unit=ts_unit,
        utc=True,
        errors="coerce",
    )
    print(f"  Timestamp unit detected: {ts_unit}")

    # Numeric columns
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "trades", "taker_buy_base", "taker_buy_quote"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Sort + deduplicate
    merged = (
        merged
        .dropna(subset=["timestamp", "close"])
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"\n  Total rows merged: {len(merged):,}")
    print(f"  Date range: {merged['timestamp'].iloc[0].date()} → {merged['timestamp'].iloc[-1].date()}")
    return merged


# ─────────────────────────────────────────────
# STEP 2: Feature Engineering (1h optimized)
# ─────────────────────────────────────────────

def add_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_bollinger(df: pd.DataFrame, period: int = 20):
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return mid - 2 * std, mid, mid + 2 * std


def add_vwap_rolling(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """1h data ke liye 24-bar VWAP = 24 hours"""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    return tp_vol.rolling(window).sum() / df["volume"].rolling(window).sum()


def detect_bos_choch(df: pd.DataFrame):
    n = len(df)
    bos_candles_ago = np.full(n, 50.0)
    choch = np.zeros(n)
    recent_high = df["high"].rolling(20).max().values
    recent_low = df["low"].rolling(20).min().values
    for i in range(20, n):
        if df["close"].iloc[i] > recent_high[i - 1]:
            bos_candles_ago[i] = 1.0
        elif df["close"].iloc[i] < recent_low[i - 1]:
            bos_candles_ago[i] = 1.0
        else:
            bos_candles_ago[i] = min(bos_candles_ago[i - 1] + 1, 50)
        prev_trend = 1 if df["close"].iloc[i - 5] > df["close"].iloc[i - 10] else -1
        curr_trend = 1 if df["close"].iloc[i] > df["close"].iloc[i - 5] else -1
        choch[i] = 1.0 if prev_trend != curr_trend and bos_candles_ago[i] <= 3 else 0.0
    return (pd.Series(bos_candles_ago, index=df.index),
            pd.Series(choch, index=df.index))


def detect_fvg(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    fvg = np.zeros(n)
    for i in range(2, n):
        if df["high"].iloc[i - 2] < df["low"].iloc[i]:
            fvg[i] = 1.0
        elif df["low"].iloc[i - 2] > df["high"].iloc[i]:
            fvg[i] = 1.0
    return pd.Series(fvg, index=df.index)


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    sweep = np.zeros(len(df))
    for i in range(lookback, len(df)):
        window_high = df["high"].iloc[i - lookback:i].max()
        window_low = df["low"].iloc[i - lookback:i].min()
        if df["high"].iloc[i] > window_high and df["close"].iloc[i] < window_high:
            sweep[i] = 1.0
        elif df["low"].iloc[i] < window_low and df["close"].iloc[i] > window_low:
            sweep[i] = 1.0
    return pd.Series(sweep, index=df.index)


def detect_ob_distance(df: pd.DataFrame) -> pd.Series:
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
    """1h data: 4 bars = 4h, yani multi-timeframe alignment"""
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    ema_4h = df["close"].ewm(span=4 * 9, adjust=False).mean()  # 4h approximate
    ema_daily = df["close"].ewm(span=24, adjust=False).mean()  # Daily approximate

    score = (
        (df["close"] > ema9).astype(int) +
        (df["close"] > ema21).astype(int) +
        (ema9 > ema21).astype(int) +
        (ema21 > ema50).astype(int) +
        (df["close"] > ema_4h).astype(int)
    )
    return score.astype(float)


def compute_regime(df: pd.DataFrame) -> pd.Series:
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    atr = add_atr(df, 14)
    close = df["close"]
    trend_strength = (ema9 - ema50).abs() / close * 100.0

    regimes = []
    for i in range(len(df)):
        if i < 50:
            regimes.append("sideways_range")
            continue
        ts = trend_strength.iloc[i]
        e9, e21, e50 = ema9.iloc[i], ema21.iloc[i], ema50.iloc[i]
        c = close.iloc[i]
        atr_pct = atr.iloc[i] / c * 100.0 if c > 0 else 0

        if ts > 1.5 and e9 > e21 > e50:
            regimes.append("bullish_trend")
        elif ts > 1.5 and e9 < e21 < e50:
            regimes.append("bearish_trend")
        elif atr_pct > 0.8 and abs(c - e50) / c * 100.0 > 1.0:
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
    conditions = [
        atr_pct < 0.3,
        (atr_pct >= 0.3) & (atr_pct < 0.7),
        atr_pct >= 0.7,
    ]
    choices = [0.0, 1.0, 2.0]
    return pd.Series(np.select(conditions, choices, default=1.0), index=atr_pct.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Features calculate ho rahe hain (1h timeframe)...")

    close = df["close"]
    price = close

    ema9 = add_ema(df, 9)
    ema21 = add_ema(df, 21)
    ema50 = add_ema(df, 50)
    ema200 = add_ema(df, 200)

    atr14 = add_atr(df, 14)
    atr_pct = atr14 / price * 100.0

    bb_lower, bb_mid, bb_upper = add_bollinger(df, 20)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    bb_pos = ((price - bb_lower) / bb_range).clip(0.0, 1.0)

    vwap = add_vwap_rolling(df, 24)  # 24h rolling VWAP for 1h data
    vwap_dist = ((price - vwap) / price * 100.0).fillna(0.0)

    df["ema9_dist"] = ((price - ema9) / price * 100.0).fillna(0.0)
    df["ema21_dist"] = ((price - ema21) / price * 100.0).fillna(0.0)
    df["ema50_dist"] = ((price - ema50) / price * 100.0).fillna(0.0)
    df["ema200_dist"] = ((price - ema200) / price * 100.0).fillna(0.0)
    df["atr_normalized"] = atr_pct.fillna(0.0)
    df["bb_position"] = bb_pos.fillna(0.5)
    df["vwap_distance"] = vwap_dist
    df["atr14"] = atr14.fillna(0.0)

    # SMC features
    df["mtf_alignment_score"] = compute_mtf_alignment(df)
    bos_ago, choch = detect_bos_choch(df)
    df["bos_candles_ago"] = bos_ago
    df["choch_detected"] = choch
    df["ob_distance_pct"] = detect_ob_distance(df)
    df["fvg_present"] = detect_fvg(df)
    df["liquidity_sweep_recent"] = detect_liquidity_sweep(df, 20)

    # Order flow (taker_buy_base Binance mein hota hai — real data)
    df["cvd_slope"] = ((close - close.shift(5)) / close.shift(5)).fillna(0.0).clip(-0.02, 0.02)
    buy_vol = df["taker_buy_base"]
    total_vol = df["volume"].replace(0, np.nan)
    df["obi"] = (buy_vol / total_vol).fillna(0.5).clip(0.0, 1.0)
    df["ofi_5bar"] = ((close - close.shift(1)).rolling(5).sum() / atr14).fillna(0.0).clip(-3.0, 3.0)

    # Realistic defaults for live-only features
    df["whale_buy_ratio"] = 0.5
    df["absorption_strength"] = 0.0
    obi_consistent = (df["obi"] > 0.6).rolling(3).sum()
    df["stacked_imbalance_direction"] = np.where(
        obi_consistent >= 3, 1.0,
        np.where((df["obi"] < 0.4).rolling(3).sum() >= 3, -1.0, 0.0)
    )
    df["single_bar_divergence"] = 0.0
    df["iceberg_direction"] = 0.0
    df["cross_exchange_cvd_divergence"] = 0.0
    df["speed_of_tape_ratio"] = 1.0

    # Derivatives defaults
    df["funding_rate"] = 0.0001
    df["funding_roc"] = 0.0
    df["oi_change_1h"] = 0.0
    df["ls_ratio"] = 0.5
    df["liq_cluster_above_pct"] = 0.5
    df["liq_cluster_below_pct"] = 0.5
    df["max_pain_distance_pct"] = 0.0
    df["iv_skew"] = 0.0

    # Options defaults
    df["put_call_ratio"] = 0.7
    df["iv_rv_ratio"] = 1.0
    df["options_expiry_hours"] = 168.0

    # On-chain defaults
    df["exchange_netflow_score"] = 0.0
    df["sopr"] = 1.0
    df["lth_supply_change"] = 0.0
    df["whale_exchange_deposit_count"] = 0.0
    df["whale_net_flow_score"] = 0.0

    # Macro defaults
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

    # Encodings
    df["session_encoded"] = session_encoded(df["timestamp"])
    df["vol_regime_encoded"] = vol_regime_encoded(atr_pct)
    df["regime"] = compute_regime(df)
    df["regime_encoded"] = regime_to_encoded(df["regime"])

    print(f"  Features ready. Total rows: {len(df):,}")
    for r, c in df["regime"].value_counts().items():
        print(f"    {r}: {c:,} rows")

    return df


# ─────────────────────────────────────────────
# STEP 3: Labels
# ─────────────────────────────────────────────

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    1h data ke liye horizon=24 (24 hours forward-looking).
    ATR-based SL/TP: 1.5x ATR.
    """
    print("\n[3/5] Labels generate ho rahe hain (24-bar horizon)...")
    from btc_intelligence.models.label_generator import generate_labels
    df_labeled = generate_labels(df, horizon=24)
    profitable = df_labeled["was_profitable"].sum()
    total = len(df_labeled)
    print(f"  Labels done. Profitable: {profitable:,}/{total:,} ({profitable/total*100:.1f}%)")
    return df_labeled


# ─────────────────────────────────────────────
# STEP 4: Meta-labeling RF training data
# ─────────────────────────────────────────────

def generate_meta_labeling_data(df: pd.DataFrame) -> None:
    """
    Labeled trades se meta-labeling Random Forest ke liye training data banao.
    Ye data btc_intelligence/logs/meta_label_rf_training.jsonl mein save hoga.
    """
    print("\n[4a] Meta-labeling RF training data generate ho raha hai...")

    log_path = ROOT / "btc_intelligence" / "logs" / "meta_label_rf_training.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Only labeled rows use karo
    labeled = df.dropna(subset=["was_profitable"]).copy()

    # Hour sin/cos
    hour = labeled["timestamp"].dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # HMM proxy (regime se approximate karo)
    def hmm_probs(regime_str: str):
        mapping = {
            "sideways_range": (0.6, 0.2, 0.2),
            "bullish_trend":  (0.1, 0.8, 0.1),
            "bearish_trend":  (0.1, 0.8, 0.1),
            "breakout_up":    (0.1, 0.3, 0.6),
            "breakout_down":  (0.1, 0.3, 0.6),
        }
        return mapping.get(str(regime_str), (1/3, 1/3, 1/3))

    written = 0
    with log_path.open("w", encoding="utf-8") as f:
        for idx in range(len(labeled)):
            row = labeled.iloc[idx]
            mr, tr, lc = hmm_probs(row.get("regime", "sideways_range"))
            features = {
                "hour_sin": float(hour_sin.iloc[idx]),
                "hour_cos": float(hour_cos.iloc[idx]),
                "hmm_mean_reverting": mr,
                "hmm_trending": tr,
                "hmm_liquidity_cascade": lc,
                "vix_level": float(row.get("vix_level", 20.0)),
                "spread_pct": 0.01,  # Default spread
            }
            label = 0 if int(row["was_profitable"]) == 1 else 1  # 0=good, 1=bad/false-positive
            f.write(json.dumps({"features": features, "label": label}) + "\n")
            written += 1

    print(f"  Meta-label training data: {written:,} rows → {log_path}")


# ─────────────────────────────────────────────
# STEP 5: Train Models
# ─────────────────────────────────────────────

def train_all_models(csv_path: str, out_dir: str, skip_lstm: bool = False) -> dict:
    print("\n[4/5] Models training shuru ho rahi hai...")

    from btc_intelligence.models.train_lgbm import train_lgbm
    from btc_intelligence.models.train_xgb import train_xgb

    root = Path(out_dir)
    results: dict = {}
    regimes = ["bullish_trend", "bearish_trend", "sideways_range", "breakout"]

    for regime in regimes:
        print(f"\n  ─── {regime.upper()} ───")
        regime_dir = root / regime
        regime_dir.mkdir(parents=True, exist_ok=True)

        # LightGBM
        try:
            r1 = train_lgbm(csv_path, str(regime_dir), regime=regime)
            status = "✓ TRAINED" if r1.get("trained") else f"✗ SKIP (rows={r1.get('rows',0)} < 300)"
            auc = r1.get("walk_forward_auc_lgbm", 0)
            print(f"  LightGBM: {status}  AUC={auc:.3f}")
        except Exception as e:
            r1 = {"trained": False, "error": str(e)}
            print(f"  LightGBM: ERROR — {e}")

        # XGBoost
        try:
            r2 = train_xgb(csv_path, str(regime_dir), regime=regime)
            status = "✓ TRAINED" if r2.get("trained") else f"✗ SKIP (rows={r2.get('rows',0)} < 300)"
            auc = r2.get("walk_forward_auc_xgb", 0)
            print(f"  XGBoost:  {status}  AUC={auc:.3f}")
        except Exception as e:
            r2 = {"trained": False, "error": str(e)}
            print(f"  XGBoost: ERROR — {e}")

        # LSTM (optional, slow)
        r3 = {"trained": False, "skipped": True}
        if not skip_lstm:
            try:
                from btc_intelligence.models.train_attention_lstm import train_attention_lstm
                print(f"  LSTM: training...", end=" ", flush=True)
                r3 = train_attention_lstm(csv_path, str(regime_dir), regime=regime)
                status = "✓ TRAINED" if r3.get("trained") else f"✗ SKIP"
                print(status)
            except Exception as e:
                print(f"SKIP ({e})")
                r3 = {"trained": False, "error": str(e)}
        else:
            print(f"  LSTM: SKIPPED (--skip-lstm)")

        meta = {
            "version": f"ml_{regime}_v2_1h",
            "regime": regime,
            "timeframe": "1h",
            "lightgbm": r1,
            "xgboost": r2,
            "lstm": r3,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        (regime_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        results[regime] = meta

    root_meta = {
        "version": f"ml_ensemble_1h_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timeframe": "1h",
        "regimes": list(results.keys()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "metadata.json").write_text(
        json.dumps(root_meta, indent=2), encoding="utf-8"
    )
    print(f"\n  Models saved: {root}")
    return results


# ─────────────────────────────────────────────
# STEP 6: Verify
# ─────────────────────────────────────────────

def verify(out_dir: str) -> None:
    print("\n[5/5] Verification...")
    root = Path(out_dir)
    all_ok = True

    for regime in ["bullish_trend", "bearish_trend", "sideways_range", "breakout"]:
        lgbm = root / regime / "lightgbm.pkl"
        xgb  = root / regime / "xgboost.pkl"
        lgbm_ok = lgbm.exists()
        xgb_ok  = xgb.exists()
        status = "✓ OK" if (lgbm_ok and xgb_ok) else "⚠ PARTIAL"
        print(f"  {regime:<20} LightGBM={'YES' if lgbm_ok else 'NO ':3}  XGBoost={'YES' if xgb_ok else 'NO ':3}  [{status}]")
        if not (lgbm_ok and xgb_ok):
            all_ok = False

    # Meta-labeling RF check
    meta_log = ROOT / "btc_intelligence" / "logs" / "meta_label_rf_training.jsonl"
    if meta_log.exists():
        lines = meta_log.read_text(encoding="utf-8").splitlines()
        print(f"\n  Meta-label RF training data: {len(lines):,} rows ✓")
    else:
        print(f"\n  Meta-label RF data: NOT FOUND")

    if all_ok:
        print("\n" + "="*50)
        print("  ALL MODELS READY! ✓")
        print("="*50)
        print("\n  Ab server restart karo:")
        print("  > Dono servers band karo (Ctrl+C)")
        print("  > start_all.bat run karo")
        print("\n  Model verify karo:")
        print("  curl http://127.0.0.1:9000/signal | python -m json.tool")
    else:
        print("\n  ⚠ Kuch models skip hue (data kam tha). Zyada ZIP files se try karo.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Historical ZIP → ML Model Training")
    parser.add_argument("--zip-dir", default="data/historical",
                        help="ZIP files ka folder (default: data/historical)")
    parser.add_argument("--csv-path", default="data/training_data_1h.csv",
                        help="Output CSV path")
    parser.add_argument("--out-dir", default="btc_intelligence/models/artifacts",
                        help="Models save karne ki jagah")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="LSTM skip karo (fast mode, ~5-10 min)")
    parser.add_argument("--data-only", action="store_true",
                        help="Sirf CSV banao, training mat karo")
    args = parser.parse_args()

    zip_dir  = ROOT / args.zip_dir
    csv_path = ROOT / args.csv_path
    out_dir  = ROOT / args.out_dir

    start = time.time()

    print("=" * 55)
    print("  BTC ML Training Pipeline (Historical 1h Data)")
    print("=" * 55)
    print(f"  ZIP dir:  {zip_dir}")
    print(f"  CSV:      {csv_path}")
    print(f"  Models:   {out_dir}")
    print(f"  LSTM:     {'SKIP' if args.skip_lstm else 'YES (slow ~15 min)'}")
    print("=" * 55)

    # Step 1-3: CSV already hai to reuse karo
    if csv_path.exists():
        print(f"\n[1-3] Existing CSV reuse: {csv_path}")
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        print(f"  Loaded: {len(df):,} rows")
    else:
        # Fresh se banao
        df = load_zips(zip_dir)
        df = build_features(df)
        df = add_labels(df)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\n  CSV saved: {csv_path}  ({len(df):,} rows)")

    # Meta-labeling data
    if "was_profitable" in df.columns:
        generate_meta_labeling_data(df)

    if args.data_only:
        print("\n  --data-only mode. Training skip.")
        elapsed = (time.time() - start) / 60
        print(f"  Time: {elapsed:.1f} min")
        return

    # Step 4: Train
    train_all_models(str(csv_path), str(out_dir), skip_lstm=args.skip_lstm)

    # Step 5: Verify
    verify(str(out_dir))

    elapsed = (time.time() - start) / 60
    print(f"\n  Total time: {elapsed:.1f} minutes")
    print("  Server restart karo taaki models load hon!")
    print("=" * 55)


if __name__ == "__main__":
    main()
