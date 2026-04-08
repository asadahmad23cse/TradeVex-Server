"""
Anti-crowding gate: Herfindahl-Hirschman Index (HHI) on aggressive flow by price cluster,
plus one-sided imbalance. High concentration → defer execution (stop-hunt / liquidity sweep risk).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from btc_intelligence.config import settings


@dataclass
class AntiCrowdingState:
    """Flow concentration diagnostic; crowding_score_0_100 maps joint HHI + imbalance to [0,100]."""

    hhi: float
    aggressive_buy_share: float
    aggressive_sell_share: float
    dominant_side: str
    crowding_score_0_100: float
    n_trades_used: int
    n_price_bins: int
    trigger_delay: bool
    reason: str


def _trade_time_ms(t: dict[str, Any]) -> int:
    for k in ("T", "time", "E"):
        v = t.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return 0


def _trade_price_qty(t: dict[str, Any]) -> tuple[float, float]:
    p = float(t.get("p", t.get("price", 0.0)) or 0.0)
    q = float(t.get("q", t.get("qty", 0.0)) or 0.0)
    return p, q


def _is_buyer_maker(t: dict[str, Any]) -> bool:
    m = t.get("m", t.get("is_buyer_maker", True))
    return bool(m)


def compute_flow_hhi_and_imbalance(
    agg_trades: list[dict[str, Any]],
    *,
    max_trades: int = 500,
    price_tick_bps: float = 2.0,
    min_trades: int = 25,
) -> AntiCrowdingState:
    """
    HHI over share of *aggressive* notional by rounded price bucket; imbalance = max side share.
    crowding_score_0_100 blends normalized HHI and imbalance (100 = extremely crowded / one-sided).
    """
    if not agg_trades or len(agg_trades) < min_trades:
        return AntiCrowdingState(
            hhi=0.0,
            aggressive_buy_share=0.5,
            aggressive_sell_share=0.5,
            dominant_side="NONE",
            crowding_score_0_100=0.0,
            n_trades_used=0,
            n_price_bins=0,
            trigger_delay=False,
            reason="Insufficient recent trades for crowding scan",
        )

    window = list(agg_trades)[-max_trades:]
    mids = [_trade_price_qty(t)[0] for t in window if _trade_price_qty(t)[0] > 0]
    mid_ref = float(sum(mids) / len(mids)) if mids else 0.0
    if mid_ref <= 0:
        return AntiCrowdingState(
            hhi=0.0,
            aggressive_buy_share=0.5,
            aggressive_sell_share=0.5,
            dominant_side="NONE",
            crowding_score_0_100=0.0,
            n_trades_used=len(window),
            n_price_bins=0,
            trigger_delay=False,
            reason="No valid mid for price bins",
        )

    tick = max(mid_ref * price_tick_bps / 10_000.0, mid_ref * 1e-6)

    aggr_buy_notional = 0.0
    aggr_sell_notional = 0.0
    bin_buy: dict[int, float] = defaultdict(float)
    bin_sell: dict[int, float] = defaultdict(float)

    for t in window:
        p, q = _trade_price_qty(t)
        if p <= 0 or q <= 0:
            continue
        notional = p * q
        maker = _is_buyer_maker(t)
        bkey = int(round(p / tick))
        if maker:
            aggr_sell_notional += notional
            bin_sell[bkey] += notional
        else:
            aggr_buy_notional += notional
            bin_buy[bkey] += notional

    total_aggr = aggr_buy_notional + aggr_sell_notional
    if total_aggr <= 1e-12:
        return AntiCrowdingState(
            hhi=0.0,
            aggressive_buy_share=0.5,
            aggressive_sell_share=0.5,
            dominant_side="NONE",
            crowding_score_0_100=0.0,
            n_trades_used=len(window),
            n_price_bins=0,
            trigger_delay=False,
            reason="No aggressive notional in window",
        )

    combined_bins: dict[int, float] = defaultdict(float)
    for k, v in bin_buy.items():
        combined_bins[k] += v
    for k, v in bin_sell.items():
        combined_bins[k] += v
    shares = [v / total_aggr for v in combined_bins.values() if v > 0]
    n_bins = len(shares)
    hhi = float(sum(s * s for s in shares)) if shares else 0.0

    buy_sh = aggr_buy_notional / total_aggr
    sell_sh = aggr_sell_notional / total_aggr
    imb = max(buy_sh, sell_sh)
    hhi_norm = float((hhi - (1.0 / max(n_bins, 1))) / max(1.0 - 1.0 / max(n_bins, 1), 1e-9)) if n_bins > 1 else hhi
    hhi_norm = max(0.0, min(1.0, hhi_norm))
    score = float(min(100.0, 100.0 * (0.55 * hhi_norm + 0.45 * max(0.0, imb - 0.5) * 2.0)))

    dominant = "BUY" if buy_sh >= sell_sh else "SELL"
    thr_hhi = float(getattr(settings, "crowd_hhi_threshold", 0.38))
    thr_imb = float(getattr(settings, "crowd_flow_imbalance_min", 0.68))
    thr_score = float(getattr(settings, "crowd_score_trigger", 72.0))

    trigger = bool(hhi >= thr_hhi and imb >= thr_imb and score >= thr_score)
    reason = (
        f"Anti-crowd: HHI={hhi:.3f} imb={imb:.3f} score={score:.1f}/100 side={dominant}"
        if trigger
        else "Flow dispersion acceptable"
    )

    return AntiCrowdingState(
        hhi=round(hhi, 6),
        aggressive_buy_share=round(buy_sh, 6),
        aggressive_sell_share=round(sell_sh, 6),
        dominant_side=dominant,
        crowding_score_0_100=round(score, 2),
        n_trades_used=len(window),
        n_price_bins=n_bins,
        trigger_delay=trigger,
        reason=reason,
    )


def evaluate_anti_crowding_gate(
    agg_trades: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> AntiCrowdingState:
    now = now or datetime.now(timezone.utc)
    _ = now
    return compute_flow_hhi_and_imbalance(
        agg_trades,
        max_trades=int(getattr(settings, "crowd_max_trades", 500)),
        price_tick_bps=float(getattr(settings, "crowd_price_tick_bps", 2.0)),
        min_trades=int(getattr(settings, "crowd_min_trades", 25)),
    )
