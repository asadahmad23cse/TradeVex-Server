from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from btc_intelligence.config import settings
from btc_intelligence.features.feature_vector import FeatureState
from btc_intelligence.signals.execution import ExecutionCheck
from btc_intelligence.signals.risk import RiskPlan


@dataclass
class NoTradeDecision:
    allow: bool
    reason: str
    cooldown_until: str


def apply_no_trade_filters(
    state: FeatureState,
    execution: ExecutionCheck,
    risk: RiskPlan,
    confidence: float,
    stacked_probability: float,
    net_alpha: float,
    direction: str,
    last_signal_times: dict[str, datetime],
    validated: str,
    auto_pause: bool,
) -> NoTradeDecision:
    now = datetime.now(timezone.utc)

    if state.volatility.vol_regime == 'spike':
        return _block('vol_regime = SPIKE', now)
    if execution.spread_pct > settings.spread_reject_pct:
        return _block('spread > 0.05%', now)
    if state.alignment_score(direction) < 2:
        return _block('MTF alignment score < 2', now)
    if risk.risk_reward_tp2 < 1.5:
        return _block('Risk-reward < 1.5', now)

    last = last_signal_times.get(direction)
    if last is not None and (now - last) < timedelta(hours=settings.signal_cooldown_hours):
        until = last + timedelta(hours=settings.signal_cooldown_hours)
        return NoTradeDecision(
            False,
            'Signal cooldown active',
            until.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        )

    if state.macro.blocking_news:
        return _block('News event within 30 minutes', now)
    if state.macro.session == 'dead':
        return _block('Dead session (21:00-00:00 UTC)', now)
    if state.macro.vix_level > 40:
        return _block('VIX > 40 (extreme panic)', now)
    if net_alpha < settings.min_net_alpha:
        return _block('Net Alpha < 0', now)
    if confidence < settings.min_confidence:
        return _block('Confidence < 60', now)
    if stacked_probability * 100.0 < 55.0:
        return _block('Stacked probability below 55%', now)
    if validated not in {direction, 'RULE_BASED'}:
        return _block('Validated direction mismatch', now)
    if auto_pause:
        return _block('System is in auto-pause mode', now)

    return NoTradeDecision(True, 'Passed all no-trade filters', '')


def _block(reason: str, now: datetime) -> NoTradeDecision:
    until = now + timedelta(minutes=15)
    return NoTradeDecision(False, reason, until.replace(microsecond=0).isoformat().replace('+00:00', 'Z'))
