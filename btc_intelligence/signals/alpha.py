from __future__ import annotations

from dataclasses import dataclass

from btc_intelligence.features.feature_vector import FeatureState
from btc_intelligence.utils.validator import clamp


@dataclass
class AlphaResult:
    alpha_score: float
    net_alpha: float


def compute_alpha_score(
    state: FeatureState,
    direction: str,
    regime: str,
    slippage_pct: float,
) -> AlphaResult:
    score = 0.0

    align = state.alignment_score(direction)
    if align == 3:
        score += 20
    elif align == 2:
        score += 12

    # Order flow block (20)
    obi = state.order_flow.obi
    if (direction == 'LONG' and obi > 0.65) or (direction == 'SHORT' and obi < 0.35):
        score += 10
    elif (direction == 'LONG' and obi > 0.58) or (direction == 'SHORT' and obi < 0.42):
        score += 7

    if (direction == 'LONG' and state.order_flow.cvd_slope > 0) or (direction == 'SHORT' and state.order_flow.cvd_slope < 0):
        score += 5
    if not state.order_flow.single_bar_delta_divergence:
        score += 5

    # Absorption/microstructure (15)
    if state.order_flow.absorption_side in {'buy_absorption', 'sell_absorption'}:
        if direction == 'LONG' and state.order_flow.absorption_side == 'sell_absorption':
            score += 8
        elif direction == 'SHORT' and state.order_flow.absorption_side == 'buy_absorption':
            score += 8
    if state.order_flow.stacked_imbalance_direction == 'bullish_3_levels' and direction == 'LONG':
        score += 4
    if state.order_flow.stacked_imbalance_direction == 'bearish_3_levels' and direction == 'SHORT':
        score += 4
    if state.order_flow.iceberg_direction == 'bullish' and direction == 'LONG':
        score += 3
    if state.order_flow.iceberg_direction == 'bearish' and direction == 'SHORT':
        score += 3

    # SMC confluence (15)
    obs = state.smc.bullish_obs if direction == 'LONG' else state.smc.bearish_obs
    if obs:
        score += 7
    if any(x.direction == ('bullish' if direction == 'LONG' else 'bearish') for x in state.smc.fvgs):
        score += 4
    if state.smc.liquidity_sweep_recent:
        score += 4

    # Regime match (10)
    if (direction == 'LONG' and regime in {'bullish_trend', 'breakout_up'}) or (
        direction == 'SHORT' and regime in {'bearish_trend', 'breakout_down'}
    ):
        score += 10
    elif regime == 'sideways_range':
        score += 5

    # Derivatives (10)
    if (direction == 'LONG' and state.derivatives.funding_rate < 0) or (direction == 'SHORT' and state.derivatives.funding_rate > 0):
        score += 5
    if state.derivatives.oi_trend in {'rising_strong', 'stable'}:
        score += 3
    if not state.derivatives.max_pain_magnet:
        score += 2

    # On-chain (5)
    if (direction == 'LONG' and state.onchain.exchange_netflow_bias == 'bullish') or (
        direction == 'SHORT' and state.onchain.exchange_netflow_bias == 'bearish'
    ):
        score += 3
    if (direction == 'LONG' and state.onchain.lth_behavior == 'accumulating') or (
        direction == 'SHORT' and state.onchain.lth_behavior == 'distributing'
    ):
        score += 2

    # Macro (5)
    if state.macro.vix_regime in {'LOW_VOL', 'NORMAL'}:
        score += 2
    if (state.macro.spx_intraday_direction == 'up' and direction == 'LONG') or (
        state.macro.spx_intraday_direction == 'down' and direction == 'SHORT'
    ):
        score += 2
    if state.macro.gold_direction in {'up', 'down'}:
        score += 1

    alpha_score = clamp(score, 0.0, 100.0)

    binance_taker_fee = 0.05
    total_cost_pct = binance_taker_fee * 2 + slippage_pct
    net_alpha = alpha_score - (total_cost_pct * 100.0)

    return AlphaResult(alpha_score=alpha_score, net_alpha=net_alpha)
