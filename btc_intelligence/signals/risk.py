from __future__ import annotations

import logging
from dataclasses import dataclass

from btc_intelligence.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RiskPlan:
    entry_zone: tuple[float, float]
    entry_mid: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk_reward_tp2: float
    position_size_pct: float
    position_size_btc: float
    adjusted_risk_pct: float
    accepted: bool
    reason: str


def build_risk_plan(
    direction: str,
    entry_low: float,
    entry_high: float,
    structure_stop: float,
    atr14: float,
    portfolio_usdt: float,
    vol_regime: str,
    monitoring_stats: dict,
) -> RiskPlan:
    entry_mid = (entry_low + entry_high) / 2.0
    if entry_mid <= 0 or atr14 <= 0:
        return _reject(entry_low, entry_high, 'Invalid entry or ATR')

    heat = float(monitoring_stats.get('portfolio_heat_pct', 0.0))
    if heat >= settings.max_portfolio_heat_pct:
        return _reject(entry_low, entry_high, 'Portfolio heat limit reached')

    atr_stop = entry_mid - 1.5 * atr14 if direction == 'LONG' else entry_mid + 1.5 * atr14

    if direction == 'LONG':
        stop_loss = max(structure_stop, atr_stop)
        sl_distance = entry_mid - stop_loss
    else:
        stop_loss = min(structure_stop, atr_stop)
        sl_distance = stop_loss - entry_mid

    if sl_distance <= 0:
        return _reject(entry_low, entry_high, 'Stop distance <= 0')

    sl_pct = sl_distance / entry_mid * 100.0
    if sl_pct < 0.3 or sl_pct > 2.5:
        return _reject(entry_low, entry_high, 'SL distance outside 0.3%-2.5%')

    # Adaptive Kelly-inspired scaling.
    base_risk = settings.risk_per_trade_pct / 100.0
    recent_wr = float(monitoring_stats.get('recent_win_rate', 0.55))
    recent_rr = max(float(monitoring_stats.get('avg_rr', 2.0)), 1.0)
    kelly_fraction = recent_wr - ((1 - recent_wr) / recent_rr)
    adjusted = base_risk * min(max(kelly_fraction * 2.0, 0.5), 1.5)

    dd = float(monitoring_stats.get('current_drawdown_pct', 0.0))
    if dd > 10:
        adjusted *= 0.5
    elif dd > 5:
        adjusted *= 0.75

    adjusted = max(0.005, min(0.015, adjusted))

    # Volatility-scaled TP multipliers.
    if vol_regime == 'compression':
        tp_mult = (0.9, 2.2, 3.5)  # TP1 closer, TP2/3 wider
    elif vol_regime == 'expansion':
        tp_mult = (1.5, 4.0, 7.0)  # TP1 closer, TP2/3 wider
    else:
        tp_mult = (1.1, 3.0, 5.0)  # TP1 closer, TP2/3 wider

    tp1 = entry_mid + sl_distance * tp_mult[0] if direction == 'LONG' else entry_mid - sl_distance * tp_mult[0]
    tp2 = entry_mid + sl_distance * tp_mult[1] if direction == 'LONG' else entry_mid - sl_distance * tp_mult[1]
    tp3 = entry_mid + sl_distance * tp_mult[2] if direction == 'LONG' else entry_mid - sl_distance * tp_mult[2]
    logger.debug(
        'tp_levels tp1=%.2f tp2=%.2f tp3=%.2f vol=%s',
        tp1,
        tp2,
        tp3,
        vol_regime,
    )

    rr_tp2 = abs(tp2 - entry_mid) / sl_distance
    if rr_tp2 < 2.0:
        return _reject(entry_low, entry_high, 'RR to TP2 < 2.0')

    risk_budget = portfolio_usdt * adjusted
    position_size_usd = risk_budget / (sl_distance / entry_mid)
    position_size_btc = position_size_usd / entry_mid

    return RiskPlan(
        entry_zone=(entry_low, entry_high),
        entry_mid=entry_mid,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        risk_reward_tp2=rr_tp2,
        position_size_pct=adjusted * 100.0,
        position_size_btc=position_size_btc,
        adjusted_risk_pct=adjusted * 100.0,
        accepted=True,
        reason='Risk accepted',
    )


def _reject(entry_low: float, entry_high: float, reason: str) -> RiskPlan:
    entry_mid = (entry_low + entry_high) / 2.0
    return RiskPlan(
        entry_zone=(entry_low, entry_high),
        entry_mid=entry_mid,
        stop_loss=0.0,
        tp1=0.0,
        tp2=0.0,
        tp3=0.0,
        risk_reward_tp2=0.0,
        position_size_pct=0.0,
        position_size_btc=0.0,
        adjusted_risk_pct=0.0,
        accepted=False,
        reason=reason,
    )
