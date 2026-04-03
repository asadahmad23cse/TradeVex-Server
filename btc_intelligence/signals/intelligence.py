from __future__ import annotations

from typing import Any

import numpy as np


def order_flow_decision_state(order_flow: Any) -> dict[str, Any]:
    """
    Map order-flow metrics to a clear decision state.
    States: FAVOR_LONG / FAVOR_SHORT / NO_TRADE
    """
    obi = float(getattr(order_flow, "obi", 0.5))
    cvd_slope = float(getattr(order_flow, "cvd_slope", 0.0))
    absorption_side = str(getattr(order_flow, "absorption_side", "none"))
    absorption_strength = float(getattr(order_flow, "absorption_strength", 0.0))

    long_score = 0
    short_score = 0
    reasons_long: list[str] = []
    reasons_short: list[str] = []

    if obi >= 0.56:
        long_score += 1
        reasons_long.append("OBI bid-dominant")
    elif obi <= 0.44:
        short_score += 1
        reasons_short.append("OBI ask-dominant")

    if cvd_slope > 0:
        long_score += 1
        reasons_long.append("CVD rising")
    elif cvd_slope < 0:
        short_score += 1
        reasons_short.append("CVD falling")

    if absorption_strength >= 1.0:
        if absorption_side == "sell_absorption":
            long_score += 1
            reasons_long.append("Sell absorption")
        elif absorption_side == "buy_absorption":
            short_score += 1
            reasons_short.append("Buy absorption")

    diff = long_score - short_score
    if max(long_score, short_score) < 2 or abs(diff) < 1:
        decision = "NO_TRADE"
        reason = "Flow mixed / weak edge"
    elif diff > 0:
        decision = "FAVOR_LONG"
        reason = "; ".join(reasons_long[:3]) or "Flow favors buyers"
    else:
        decision = "FAVOR_SHORT"
        reason = "; ".join(reasons_short[:3]) or "Flow favors sellers"

    return {
        "decision_state": decision,
        "reason": reason,
        "obi": round(obi, 4),
        "cvd_slope": round(cvd_slope, 6),
        "absorption_side": absorption_side,
        "absorption_strength": round(absorption_strength, 4),
        "stacked_imbalance_direction": str(getattr(order_flow, "stacked_imbalance_direction", "none")),
        "single_bar_delta_divergence": bool(getattr(order_flow, "single_bar_delta_divergence", False)),
        "cross_exchange_alignment": str(getattr(order_flow, "cross_exchange_alignment", "neutral")),
    }


def trade_based_volume_profile(
    agg_trades: list[dict[str, Any]],
    window_minutes: int = 45,
    n_bins: int = 24,
) -> dict[str, Any]:
    """Sliding-window, trade-based volume profile built from agg trades."""
    if not agg_trades:
        return {
            "decision_state": "NO_TRADE",
            "reason": "No trade data",
            "window_minutes": int(window_minutes),
            "bins": [],
            "poc": 0.0,
            "hvn": [],
            "lvn": [],
            "price_to_poc_pct": 0.0,
        }

    latest_ms = int(agg_trades[-1].get("time", 0))
    cutoff_ms = latest_ms - int(max(window_minutes, 1) * 60_000)
    rows = [x for x in agg_trades if int(x.get("time", 0)) >= cutoff_ms]
    if len(rows) < 20:
        return {
            "decision_state": "NO_TRADE",
            "reason": "Insufficient trade window",
            "window_minutes": int(window_minutes),
            "bins": [],
            "poc": 0.0,
            "hvn": [],
            "lvn": [],
            "price_to_poc_pct": 0.0,
        }

    prices = np.asarray([float(x.get("price", 0.0)) for x in rows], dtype=float)
    qtys = np.asarray([float(x.get("qty", 0.0)) for x in rows], dtype=float)
    valid = (prices > 0) & (qtys > 0)
    prices = prices[valid]
    qtys = qtys[valid]
    if prices.size < 20:
        return {
            "decision_state": "NO_TRADE",
            "reason": "Invalid trade samples",
            "window_minutes": int(window_minutes),
            "bins": [],
            "poc": 0.0,
            "hvn": [],
            "lvn": [],
            "price_to_poc_pct": 0.0,
        }

    p_min = float(np.min(prices))
    p_max = float(np.max(prices))
    if p_max <= p_min:
        p_max = p_min + 1e-6

    bins = max(int(n_bins), 8)
    hist, edges = np.histogram(prices, bins=bins, range=(p_min, p_max), weights=qtys)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(np.sum(hist)) if float(np.sum(hist)) > 0 else 1.0

    poc_idx = int(np.argmax(hist))
    poc = float(centers[poc_idx])
    last_price = float(prices[-1])
    price_to_poc_pct = ((last_price - poc) / last_price * 100.0) if last_price > 0 else 0.0

    ranked = np.argsort(hist)
    hvn_idx = ranked[-3:][::-1]
    lvn_idx = ranked[:3]

    hvn = [round(float(centers[i]), 2) for i in hvn_idx if hist[i] > 0]
    lvn = [round(float(centers[i]), 2) for i in lvn_idx if hist[i] > 0]

    volume_above = float(np.sum(hist[centers > last_price]))
    volume_below = float(np.sum(hist[centers < last_price]))
    if abs(price_to_poc_pct) < 0.05:
        decision = "NO_TRADE"
        reason = "Price at fair value (POC)"
    elif volume_below > volume_above * 1.2:
        decision = "FAVOR_LONG"
        reason = "Demand concentrated below price"
    elif volume_above > volume_below * 1.2:
        decision = "FAVOR_SHORT"
        reason = "Supply concentrated above price"
    else:
        decision = "NO_TRADE"
        reason = "Balanced profile"

    profile_bins = []
    for i in range(len(hist)):
        profile_bins.append(
            {
                "low": round(float(edges[i]), 2),
                "high": round(float(edges[i + 1]), 2),
                "center": round(float(centers[i]), 2),
                "volume": round(float(hist[i]), 6),
                "relative": round(float(hist[i] / total), 6),
            }
        )

    return {
        "decision_state": decision,
        "reason": reason,
        "window_minutes": int(window_minutes),
        "trade_count": int(prices.size),
        "poc": round(poc, 2),
        "hvn": hvn,
        "lvn": lvn,
        "price_to_poc_pct": round(price_to_poc_pct, 4),
        "bins": profile_bins,
    }


def volatility_tradeability(volatility: Any) -> dict[str, Any]:
    regime = str(getattr(volatility, "vol_regime", "normal")).lower()
    if regime == "compression":
        mapped_regime = "LOW"
        tradeability = "NO_TRADE"
        reason = "Compressed range; poor payoff asymmetry"
    elif regime == "normal":
        mapped_regime = "NORMAL"
        tradeability = "ALLOW"
        reason = "Volatility in tradable regime"
    else:
        mapped_regime = "EXPANSION"
        tradeability = "CAUTION"
        reason = "Elevated volatility; reduce size and demand cleaner entries"

    return {
        "volatility_regime": mapped_regime,
        "tradeability": tradeability,
        "reason": reason,
        "atr14": round(float(getattr(volatility, "atr14", 0.0)), 6),
        "atr_multiplier_pct": round(float(getattr(volatility, "atr_multiplier_pct", 0.0)), 6),
        "bb_squeeze": bool(getattr(volatility, "bb_squeeze", False)),
        "bb_expansion": bool(getattr(volatility, "bb_expansion", False)),
        "raw_regime": str(getattr(volatility, "vol_regime", "normal")),
    }


def execution_rejection_code(reason: str) -> str:
    r = reason.lower()
    if "volatility regime too low for execution" in r:
        return "VOL_NO_TRADE"
    if "depth unavailable" in r:
        return "DEPTH_UNAVAILABLE"
    if "spread" in r:
        return "SPREAD_REJECT"
    if "slippage" in r:
        return "SLIPPAGE_REJECT"
    if "price move" in r:
        return "FAST_MOVE_REJECT"
    if "trigger" in r:
        return "TRIGGER_NOT_READY"
    if "expiry" in r:
        return "EXPIRY_WINDOW_REJECT"
    return "UNKNOWN_REJECT"


def combined_btc_signal(
    orderflow_payload: dict[str, Any],
    volatility_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a compact executable BTC signal from flow + volatility gate.
    """
    decision = str(orderflow_payload.get("decision_state", "NO_TRADE")).upper()
    tradeability = str(volatility_payload.get("tradeability", "NO_TRADE")).upper()
    obi = float(orderflow_payload.get("obi", 0.0))

    if decision == "FAVOR_LONG" and tradeability == "ALLOW":
        signal = "LONG"
        confidence = max(0.0, min(100.0, obi * 100.0))
    elif decision == "FAVOR_SHORT" and tradeability == "ALLOW":
        signal = "SHORT"
        confidence = max(0.0, min(100.0, abs(obi) * 100.0))
    else:
        signal = "HOLD"
        confidence = 0.0

    return {
        "ticker": "BTCUSDT",
        "signal": signal,
        "confidence": round(confidence, 2),
        "orderflow_decision": decision,
        "tradeability": tradeability,
        "obi": round(obi, 4),
    }
