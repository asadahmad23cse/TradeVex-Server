from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from btc_intelligence.features.feature_vector import FeatureState
from btc_intelligence.regime.hmm_regime import (
    candles_to_hmm_observations,
    hmm_regime_label,
    posterior_state_probs,
)


@dataclass
class RegimeResult:
    regime: str
    score: int
    reason: str
    state_probs: dict[str, float] = field(default_factory=dict)


def classify_regime(snapshot: dict, state: FeatureState) -> RegimeResult:
    panic_reason = _panic_condition(snapshot, state)
    if panic_reason:
        return RegimeResult(
            "panic_liquidation",
            6,
            panic_reason,
            state_probs={"mean_reverting": 0.0, "trending": 0.0, "liquidity_cascade": 1.0},
        )

    breakout_up = _breakout(snapshot, "up", state)
    if breakout_up:
        return RegimeResult(
            "breakout_up",
            4,
            breakout_up,
            state_probs={"mean_reverting": 0.15, "trending": 0.80, "liquidity_cascade": 0.05},
        )
    breakout_down = _breakout(snapshot, "down", state)
    if breakout_down:
        return RegimeResult(
            "breakout_down",
            4,
            breakout_down,
            state_probs={"mean_reverting": 0.15, "trending": 0.80, "liquidity_cascade": 0.05},
        )

    candles_15m = snapshot.get("candles", {}).get("15m", [])
    obs = candles_to_hmm_observations(candles_15m)
    probs = posterior_state_probs(obs) if obs is not None else None

    if probs is None:
        return RegimeResult(
            "sideways_range",
            2,
            "HMM unavailable or insufficient bars; defaulting to range",
            state_probs={"mean_reverting": 0.34, "trending": 0.33, "liquidity_cascade": 0.33},
        )

    recent_ret = _recent_log_ret_sum(candles_15m, n_bars=12)
    regime, score, reason = hmm_regime_label(probs, state, recent_ret)
    return RegimeResult(regime, score, reason, state_probs=dict(probs))


def _recent_log_ret_sum(candles: list, n_bars: int = 12) -> float:
    if len(candles) < n_bars + 1:
        return 0.0
    rows = candles[-(n_bars + 1) :]
    closes = np.array([float(c["close"]) for c in rows], dtype=float)
    closes = np.clip(closes, 1e-9, None)
    return float(np.sum(np.diff(np.log(closes))))


def _breakout(snapshot: dict, direction: str, state: FeatureState) -> str | None:
    df = snapshot.get("candles", {}).get("15m", [])
    if len(df) < 25:
        return None
    last = df[-1]
    prev = df[-21:-1]
    high20 = max(float(x["high"]) for x in prev)
    low20 = min(float(x["low"]) for x in prev)
    close = float(last["close"])
    vol = float(last["volume"])
    avg_vol = sum(float(x["volume"]) for x in prev) / max(len(prev), 1)

    bb_released = state.volatility.bb_expansion

    if direction == "up" and close > high20 and vol > 1.5 * avg_vol and state.derivatives.oi_change_1h_pct > 0 and bb_released:
        return "Breakout up with volume+OI and BB release"
    if direction == "down" and close < low20 and vol > 1.5 * avg_vol and state.derivatives.oi_change_1h_pct > 0 and bb_released:
        return "Breakout down with volume+OI and BB release"
    return None


def _panic_condition(snapshot: dict, state: FeatureState) -> str | None:
    candles = snapshot.get("candles", {}).get("15m", [])
    if len(candles) < 2 or state.volatility.atr14 <= 0:
        return None

    c = candles[-1]
    move = abs(float(c["close"]) - float(c["open"]))
    move_atr = move / state.volatility.atr14

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=5)
    liq_5m = 0.0
    for row in snapshot.get("force_orders", []):
        ts = datetime.fromtimestamp(int(row.get("time", 0)) / 1000, tz=timezone.utc)
        if ts >= cutoff:
            liq_5m += float(row.get("notional", 0.0))

    if move_atr > 3 and liq_5m > 100_000_000 and abs(state.derivatives.funding_rate) > 0.001 and state.macro.vix_level > 35:
        return "3x ATR + force liquidations + funding extreme + VIX spike"
    return None
