from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OrderFlowSnapshot:
    cvd_value: float
    cvd_slope: float
    cvd_trend: str
    obi_imbalance: float
    aggression_buy_pct: float
    aggression_sell_pct: float
    absorption_levels: list[float]
    flow_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cvd_value": round(self.cvd_value, 4),
            "cvd_slope": round(self.cvd_slope, 8),
            "cvd_trend": self.cvd_trend,
            "obi_imbalance": round(self.obi_imbalance, 6),
            "aggression_buy_pct": round(self.aggression_buy_pct, 2),
            "aggression_sell_pct": round(self.aggression_sell_pct, 2),
            "absorption_levels": [round(float(x), 2) for x in self.absorption_levels],
            "flow_score": round(self.flow_score, 6),
        }


class OrderFlowService:
    """Microstructure-grade order-flow metrics from live trades + depth."""

    def __init__(self, max_trade_window: int = 1200, depth_levels: int = 15) -> None:
        self.max_trade_window = max(200, int(max_trade_window))
        self.depth_levels = max(5, int(depth_levels))

    def analyze(self, trades: list[dict[str, Any]], depth: dict[str, Any]) -> OrderFlowSnapshot:
        window = list(trades[-self.max_trade_window :]) if trades else []

        buy_qty = 0.0
        sell_qty = 0.0
        signed: list[float] = []
        for row in window:
            qty = float(row.get("qty", 0.0) or 0.0)
            if qty <= 0:
                continue
            is_maker_sell = bool(row.get("maker", False))
            if is_maker_sell:
                sell_qty += qty
                signed.append(-qty)
            else:
                buy_qty += qty
                signed.append(qty)

        total_qty = buy_qty + sell_qty
        aggression_buy = (buy_qty / total_qty * 100.0) if total_qty > 0 else 50.0
        aggression_sell = 100.0 - aggression_buy

        signed_arr = np.asarray(signed, dtype=float) if signed else np.asarray([0.0], dtype=float)
        cvd_series = np.cumsum(signed_arr)
        cvd_value = float(cvd_series[-1])
        tail = cvd_series[- min(120, len(cvd_series)) :]
        if len(tail) > 1:
            x = np.arange(len(tail), dtype=float)
            cvd_slope = float(np.polyfit(x, tail, 1)[0])
        else:
            cvd_slope = 0.0
        slope_eps = max(float(np.std(tail)) * 0.01, 1e-6)
        if cvd_slope > slope_eps:
            cvd_trend = "RISING"
        elif cvd_slope < -slope_eps:
            cvd_trend = "FALLING"
        else:
            cvd_trend = "FLAT"

        bids = depth.get("bids", [])[: self.depth_levels] if isinstance(depth, dict) else []
        asks = depth.get("asks", [])[: self.depth_levels] if isinstance(depth, dict) else []
        bid_weighted = 0.0
        ask_weighted = 0.0
        for i, level in enumerate(bids):
            try:
                price = float(level[0])
                qty = float(level[1])
                w = 1.0 / (1.0 + (i * 0.25))
                bid_weighted += price * qty * w
            except Exception:
                continue
        for i, level in enumerate(asks):
            try:
                price = float(level[0])
                qty = float(level[1])
                w = 1.0 / (1.0 + (i * 0.25))
                ask_weighted += price * qty * w
            except Exception:
                continue
        depth_total = bid_weighted + ask_weighted
        if depth_total > 0:
            obi_imbalance = (bid_weighted - ask_weighted) / depth_total
        else:
            obi_imbalance = 0.0

        absorption_levels = self._absorption_levels(depth)

        slope_scale = max(float(np.mean(np.abs(np.diff(tail)))) if len(tail) > 2 else 1.0, 1e-6)
        cvd_score = float(np.tanh(cvd_slope / slope_scale))
        aggression_score = ((aggression_buy - 50.0) / 50.0)
        flow_score = float(np.clip((0.55 * obi_imbalance) + (0.35 * cvd_score) + (0.10 * aggression_score), -1.0, 1.0))

        return OrderFlowSnapshot(
            cvd_value=cvd_value,
            cvd_slope=cvd_slope,
            cvd_trend=cvd_trend,
            obi_imbalance=obi_imbalance,
            aggression_buy_pct=aggression_buy,
            aggression_sell_pct=aggression_sell,
            absorption_levels=absorption_levels,
            flow_score=flow_score,
        )

    def _absorption_levels(self, depth: dict[str, Any]) -> list[float]:
        if not isinstance(depth, dict):
            return []
        levels: list[tuple[float, float]] = []
        for side in ("bids", "asks"):
            for level in depth.get(side, [])[: self.depth_levels]:
                try:
                    price = float(level[0])
                    qty = float(level[1])
                    if price > 0 and qty > 0:
                        levels.append((price, qty))
                except Exception:
                    continue
        if not levels:
            return []

        sizes = np.asarray([q for _, q in levels], dtype=float)
        threshold = float(np.percentile(sizes, 85))
        selected = [px for px, qty in levels if qty >= threshold]
        selected = sorted(set(float(x) for x in selected))
        if len(selected) > 6:
            selected = selected[:6]
        return selected

