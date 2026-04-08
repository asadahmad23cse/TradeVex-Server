from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from btc_intelligence.config import settings
from btc_intelligence.features.feature_vector import FeatureState
from btc_intelligence.models.inference import ModelInference
from btc_intelligence.regime.classifier import RegimeResult
from btc_intelligence.signals.alpha import compute_alpha_score
from btc_intelligence.signals.anti_crowding import evaluate_anti_crowding_gate
from btc_intelligence.signals.execution import evaluate_execution
from btc_intelligence.signals.no_trade_filter import apply_no_trade_filters
from btc_intelligence.services.ic_monitor import (
    apply_hibernation_mask,
    apply_hibernation_to_stack_edges,
    get_ic_monitor,
)
from btc_intelligence.signals.probability_stacker import ProbabilityStacker, StackResult
from btc_intelligence.signals.risk import build_risk_plan
from btc_intelligence.utils.validator import utc_now_iso


@dataclass
class StrategyCandidate:
    direction: str
    strategy: str
    algo: str
    reason: str
    entry_low: float
    entry_high: float
    structure_stop: float
    factors: list[str]


class SignalEngine:
    def __init__(self, model: ModelInference, stacker: ProbabilityStacker):
        self.model = model
        self.stacker = stacker
        self.last_signal_times: dict[str, datetime] = {}
        self._crowd_suppress_until: datetime | None = None

    def build(
        self,
        snapshot: dict[str, Any],
        state: FeatureState,
        regime: RegimeResult,
        sequence_vectors: list[np.ndarray],
        monitoring_stats: dict[str, Any],
        auto_pause: bool,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        if bool(getattr(settings, "crowd_gate_enabled", True)):
            if self._crowd_suppress_until is not None and now >= self._crowd_suppress_until:
                self._crowd_suppress_until = None
            if self._crowd_suppress_until is not None:
                cu = self._crowd_suppress_until.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                return self._hold(
                    "Anti-crowding delay: waiting for post-sweep liquidity",
                    state,
                    regime.regime,
                    monitoring_stats,
                    cooldown_until=cu,
                )
            crowd = evaluate_anti_crowding_gate(snapshot.get("agg_trades", []), now=now)
            if crowd.trigger_delay:
                delay_ms = int(getattr(settings, "crowd_delay_ms", 30000))
                self._crowd_suppress_until = now + timedelta(milliseconds=delay_ms)
                cu = self._crowd_suppress_until.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                ac = {
                    "hhi": crowd.hhi,
                    "crowding_score_0_100": crowd.crowding_score_0_100,
                    "dominant_side": crowd.dominant_side,
                    "aggressive_buy_share": crowd.aggressive_buy_share,
                    "delay_ms": delay_ms,
                }
                return self._hold(
                    crowd.reason,
                    state,
                    regime.regime,
                    monitoring_stats,
                    cooldown_until=cu,
                    anti_crowding=ac,
                )

        candidate = self._pick_strategy(state, regime.regime)
        if candidate is None:
            return self._hold('No strategy confluence', state, regime.regime, monitoring_stats)

        execution = evaluate_execution(
            snapshot.get('depth', {}),
            snapshot.get('agg_trades', []),
            candidate.direction,
            order_flow=state.order_flow,
            derivatives=state.derivatives,
        )
        if not execution.accepted:
            return self._hold(execution.reason, state, regime.regime, monitoring_stats)

        risk = build_risk_plan(
            direction=candidate.direction,
            entry_low=candidate.entry_low,
            entry_high=candidate.entry_high,
            structure_stop=candidate.structure_stop,
            atr14=state.volatility.atr14,
            portfolio_usdt=settings.portfolio_usdt,
            vol_regime=state.volatility.vol_regime,
            monitoring_stats=monitoring_stats,
        )
        if not risk.accepted:
            return self._hold(risk.reason, state, regime.regime, monitoring_stats)

        vector, feature_map = state.to_vector(direction=candidate.direction, regime=regime.regime)
        hibernated = get_ic_monitor().hibernated_factors
        feature_map, vector = apply_hibernation_mask(feature_map, vector, hibernated)

        seq = np.vstack(sequence_vectors[-24:]) if len(sequence_vectors) >= 24 else None
        ensemble = self.model.infer(candidate.direction, vector, feature_map, sequence_24=seq, regime=regime.regime)

        alpha = compute_alpha_score(state, candidate.direction, regime.regime, execution.slippage_pct)
        stack_raw = self.stacker.stack(candidate.factors)
        adj_edges, stacked_p = apply_hibernation_to_stack_edges(
            stack_raw.independent_edges, candidate.factors, hibernated
        )
        stack = StackResult(stacked_probability=stacked_p, independent_edges=adj_edges)

        confidence = float(ensemble.confidence)
        if abs(confidence / 100.0 - stack.stacked_probability) > 0.20:
            confidence -= 10.0
        confidence -= execution.confidence_penalty
        confidence = max(0.0, min(100.0, confidence))

        decision = apply_no_trade_filters(
            state=state,
            execution=execution,
            risk=risk,
            confidence=confidence,
            stacked_probability=stack.stacked_probability,
            net_alpha=alpha.net_alpha,
            direction=candidate.direction,
            last_signal_times=self.last_signal_times,
            validated=ensemble.validated,
            auto_pause=auto_pause,
        )
        if not decision.allow:
            return self._hold(decision.reason, state, regime.regime, monitoring_stats, cooldown_until=decision.cooldown_until)

        cooldown_until = now + timedelta(hours=settings.signal_cooldown_hours)
        stale_after = now + timedelta(minutes=settings.signal_stale_minutes)
        self.last_signal_times[candidate.direction] = now

        payload = {
            'signal': candidate.direction,
            'validated': ensemble.validated,
            'confidence': int(round(confidence)),
            'stacked_probability': int(round(stack.stacked_probability * 100.0)),
            'alpha_score': int(round(alpha.alpha_score)),
            'net_alpha': int(round(alpha.net_alpha)),
            'entry_zone': [int(round(risk.entry_zone[0])), int(round(risk.entry_zone[1]))],
            'stop_loss': int(round(risk.stop_loss)),
            'take_profit': {
                'TP1': int(round(risk.tp1)),
                'TP2': int(round(risk.tp2)),
                'TP3': int(round(risk.tp3)),
            },
            'risk_reward': round(risk.risk_reward_tp2, 2),
            'position_size_pct': round(risk.position_size_pct, 2),
            'position_size_btc': round(risk.position_size_btc, 4),
            'market_regime': regime.regime,
            'session': state.macro.session,
            'execution_quality': execution.quality,
            'mtf': {
                '15m': state.price_action.tf_15m.trend_bias,
                '1h': state.price_action.tf_1h.trend_bias,
                '4h': state.price_action.tf_4h.trend_bias,
                'alignment_score': state.alignment_score(candidate.direction),
            },
            'smc': {
                'ob_level': int(round(self._nearest_ob_level(state, candidate.direction))),
                'fvg_present': any(
                    x.direction == ('bullish' if candidate.direction == 'LONG' else 'bearish') for x in state.smc.fvgs
                ),
                'bos_type': state.smc.bos_type,
                'liq_sweep_recent': state.smc.liquidity_sweep_recent,
                'choch': state.smc.choch,
            },
            'order_flow': {
                'cvd_slope': 'rising' if state.order_flow.cvd_slope > 0 else 'falling',
                'obi': round(state.order_flow.obi, 3),
                'whale_pressure': self._whale_pressure(state),
                'absorption': state.order_flow.absorption_side,
                'stacked_imbalance': state.order_flow.stacked_imbalance_direction,
                'single_bar_divergence': state.order_flow.single_bar_delta_divergence,
                'cross_exchange_cvd': state.order_flow.cross_exchange_alignment,
                'speed_of_tape': 'fast' if state.order_flow.speed_of_tape_ratio > 3 else 'normal',
            },
            'derivatives': {
                'funding_rate': round(state.derivatives.funding_rate, 6),
                'funding_momentum': self._funding_momentum_label(state.derivatives.funding_roc),
                'oi_change_1h': f"{state.derivatives.oi_change_1h_pct:+.2f}%",
                'liq_magnet': state.derivatives.liq_magnet_bias,
                'ls_ratio': f"{int(state.derivatives.long_short_ratio * 100)}/{int((1 - state.derivatives.long_short_ratio) * 100)}",
                'max_pain': int(round(state.derivatives.max_pain_level)),
                'iv_skew': round(state.derivatives.iv_skew, 4),
                'put_call_ratio': round(state.derivatives.put_call_ratio, 4),
            },
            'on_chain': {
                'exchange_netflow': 'negative' if state.onchain.exchange_netflow < 0 else 'positive',
                'sopr': round(state.onchain.sopr, 3),
                'lth_behavior': state.onchain.lth_behavior,
                'whale_deposits_1h': state.onchain.whale_exchange_deposits_1h,
                'whale_withdrawals_1h': state.onchain.whale_exchange_withdrawals_1h,
            },
            'macro': {
                'fear_greed': state.macro.fear_greed_score,
                'dxy_bias': state.macro.dxy_bias,
                'vix': round(state.macro.vix_level, 2),
                'vix_regime': state.macro.vix_regime,
                'spx_direction': state.macro.spx_intraday_direction,
                'spx_btc_correlation': round(state.macro.spx_btc_correlation, 3),
                'us_10y_trend': state.macro.us10y_yield_trend,
                'gold_direction': state.macro.gold_direction,
                'eth_btc_trend': state.macro.eth_btc_ratio_trend,
                'btc_dominance_trend': state.macro.btc_dominance_trend,
            },
            'algo': candidate.algo,
            'strategy': candidate.strategy,
            'reason': candidate.reason,
            'factors_present': candidate.factors,
            'independent_edges': {k: round(v, 4) for k, v in stack.independent_edges.items()},
            'as_of_utc': utc_now_iso(),
            'cooldown_until': cooldown_until.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'next_check_utc': (now + timedelta(minutes=15)).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'stale_after_utc': stale_after.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'model_version': self.model.version,
            'system_status': {
                'recent_win_rate': round(float(monitoring_stats.get('recent_win_rate', 0.0)), 4),
                'current_drawdown_pct': round(float(monitoring_stats.get('current_drawdown_pct', 0.0)), 2),
                'portfolio_heat_pct': round(float(monitoring_stats.get('portfolio_heat_pct', 0.0)), 2),
                'auto_pause': bool(auto_pause),
            },
            'model': {
                'confidence_lgbm': round(ensemble.confidence_lgbm, 4),
                'confidence_xgb': round(ensemble.confidence_xgb, 4),
                'confidence_lstm': round(ensemble.confidence_lstm, 4),
                'inference_ms': round(ensemble.inference_ms, 2),
            },
            'feature_state': feature_map,
            'hibernated_factors': sorted(hibernated),
            'factor_ic_snapshot': {k: round(v, 6) for k, v in get_ic_monitor().factor_ics.items()},
        }
        return payload

    def _pick_strategy(self, state: FeatureState, regime: str) -> StrategyCandidate | None:
        for picker in [
            self._s1_smc_trend,
            self._s2_liquidity_sweep_reversal,
            self._s3_range_rejection,
            self._s4_breakout_confirmation,
            self._s5_absorption_reversal,
        ]:
            candidate = picker(state, regime)
            if candidate is not None:
                return candidate
        return None

    def _s1_smc_trend(self, state: FeatureState, regime: str) -> StrategyCandidate | None:
        if regime not in {'bullish_trend', 'bearish_trend'}:
            return None

        direction = 'LONG' if regime == 'bullish_trend' else 'SHORT'
        if state.alignment_score(direction) < 3:
            return None

        obs = state.smc.bullish_obs if direction == 'LONG' else state.smc.bearish_obs
        if not obs:
            return None
        ob = obs[-1]

        near_ob = abs(state.price_action.tf_15m.price - ob.midpoint) / max(state.price_action.tf_15m.price, 1e-9) * 100.0 <= 0.3
        fvg_ok = any(x.direction == ('bullish' if direction == 'LONG' else 'bearish') for x in state.smc.fvgs)
        obi_ok = state.order_flow.obi > 0.58 if direction == 'LONG' else state.order_flow.obi < 0.42
        funding_ok = not (
            direction == 'LONG' and state.derivatives.funding_sentiment == 'overleveraged_long'
        ) and not (
            direction == 'SHORT' and state.derivatives.funding_sentiment == 'overleveraged_short'
        )
        vix_ok = state.macro.vix_regime != 'PANIC'

        if not all([
            near_ob,
            fvg_ok,
            not state.order_flow.cvd_divergence,
            obi_ok,
            not state.smc.liquidity_sweep_recent,
            state.volatility.vol_regime in {'normal', 'expansion'},
            funding_ok,
            not state.order_flow.single_bar_delta_divergence,
            vix_ok,
        ]):
            return None

        entry_mid = ob.midpoint
        entry_low = entry_mid * 0.999
        entry_high = entry_mid * 1.001
        structure_stop = ob.low - 0.5 * state.volatility.atr14 if direction == 'LONG' else ob.high + 0.5 * state.volatility.atr14

        factors = [
            'mtf_3_alignment',
            'ob_retest_with_fvg',
            'cvd_confirming',
            'funding_against_crowd',
            'macro_risk_on',
        ]
        if state.order_flow.absorption_side in {'sell_absorption', 'buy_absorption'}:
            factors.append('absorption_confirming')
        if state.macro.session in {'london_open', 'ny_london_overlap', 'new_york'}:
            factors.append('london_session')

        return StrategyCandidate(
            direction=direction,
            strategy='S1',
            algo='SMC_Trend_Continuation_v3',
            reason='3/3 MTF + OB retest + FVG confluence + OBI/flow alignment',
            entry_low=entry_low,
            entry_high=entry_high,
            structure_stop=structure_stop,
            factors=factors,
        )

    def _s2_liquidity_sweep_reversal(self, state: FeatureState, regime: str) -> StrategyCandidate | None:
        if regime == 'panic_liquidation' or state.macro.session in {'asian', 'dead'}:
            return None
        if not state.smc.liquidity_sweep_recent or not state.smc.choch:
            return None

        direction = 'LONG' if state.price_action.tf_15m.trend_bias == 'bearish' else 'SHORT'
        fvg_ok = any(x.direction == ('bullish' if direction == 'LONG' else 'bearish') for x in state.smc.fvgs)
        obi_flip = state.order_flow.obi > 0.60 if direction == 'LONG' else state.order_flow.obi < 0.40
        absorption_ok = state.order_flow.absorption_strength > 1.0

        if not (fvg_ok and obi_flip and state.order_flow.cvd_divergence and absorption_ok):
            return None

        px = state.price_action.tf_15m.price
        entry_low = px * 0.999
        entry_high = px * 1.001
        sweep_extreme = state.price_action.tf_15m.swing_low_20 if direction == 'LONG' else state.price_action.tf_15m.swing_high_20
        structure_stop = sweep_extreme - 0.3 * state.volatility.atr14 if direction == 'LONG' else sweep_extreme + 0.3 * state.volatility.atr14

        return StrategyCandidate(
            direction=direction,
            strategy='S2',
            algo='Liquidity_Sweep_Reversal_v3',
            reason='Liquidity sweep + CHoCH + OBI flip + calibrated CVD divergence + absorption',
            entry_low=entry_low,
            entry_high=entry_high,
            structure_stop=structure_stop,
            factors=['liquidity_sweep_confirming', 'cvd_confirming', 'absorption_confirming'],
        )

    def _s3_range_rejection(self, state: FeatureState, regime: str) -> StrategyCandidate | None:
        if regime != 'sideways_range':
            return None

        px = state.price_action.tf_15m.price
        support = state.price_action.tf_15m.swing_low_20
        resistance = state.price_action.tf_15m.swing_high_20
        near_support = abs(px - support) / max(px, 1e-9) * 100.0 < 0.2
        near_resistance = abs(px - resistance) / max(px, 1e-9) * 100.0 < 0.2

        if near_support and state.order_flow.obi > 0.65 and state.smc.bos_type == 'none' and state.order_flow.stacked_imbalance_direction == 'bullish_3_levels':
            entry_low = support * 0.999
            entry_high = support * 1.001
            return StrategyCandidate(
                'LONG',
                'S3',
                'Range_Boundary_Rejection_v3',
                'Range support rejection + stacked imbalance + OBI support',
                entry_low,
                entry_high,
                support - 0.5 * state.volatility.atr14,
                ['stacked_imbalance_confirming'],
            )

        if near_resistance and state.order_flow.obi < 0.35 and state.smc.bos_type == 'none' and state.order_flow.stacked_imbalance_direction == 'bearish_3_levels':
            entry_low = resistance * 0.999
            entry_high = resistance * 1.001
            return StrategyCandidate(
                'SHORT',
                'S3',
                'Range_Boundary_Rejection_v3',
                'Range resistance rejection + stacked imbalance + OBI resistance',
                entry_low,
                entry_high,
                resistance + 0.5 * state.volatility.atr14,
                ['stacked_imbalance_confirming'],
            )

        return None

    def _s4_breakout_confirmation(self, state: FeatureState, regime: str) -> StrategyCandidate | None:
        if regime not in {'breakout_up', 'breakout_down'}:
            return None

        direction = 'LONG' if regime == 'breakout_up' else 'SHORT'
        cvd_ok = state.order_flow.cvd_slope > 0 if direction == 'LONG' else state.order_flow.cvd_slope < 0
        liq_ok = state.derivatives.liq_magnet_bias == ('bullish' if direction == 'LONG' else 'bearish')
        iv_ok = state.volatility.iv_rv_ratio > 1.0
        if not (cvd_ok and liq_ok and iv_ok and state.derivatives.oi_change_1h_pct > 0):
            return None

        level = state.price_action.tf_15m.swing_high_20 if direction == 'LONG' else state.price_action.tf_15m.swing_low_20
        px = state.price_action.tf_15m.price
        if abs(px - level) / max(px, 1e-9) * 100.0 > 0.25:
            return None

        entry_low = level * 0.999
        entry_high = level * 1.001
        structure_stop = level - 0.5 * state.volatility.atr14 if direction == 'LONG' else level + 0.5 * state.volatility.atr14
        return StrategyCandidate(
            direction=direction,
            strategy='S4',
            algo='Breakout_Confirmation_v3',
            reason='Breakout retest + OI rise + CVD confirm + liq magnet align + IV expansion',
            entry_low=entry_low,
            entry_high=entry_high,
            structure_stop=structure_stop,
            factors=['breakout_retest', 'cvd_confirming'],
        )

    def _s5_absorption_reversal(self, state: FeatureState, regime: str) -> StrategyCandidate | None:
        if regime == 'panic_liquidation' or state.macro.session in {'asian', 'dead'}:
            return None

        if state.order_flow.absorption_strength <= 3.0:
            return None
        if state.order_flow.stacked_imbalance_direction == 'none':
            return None
        if not state.order_flow.single_bar_delta_divergence:
            return None

        if state.order_flow.absorption_side == 'sell_absorption':
            direction = 'LONG'
        elif state.order_flow.absorption_side == 'buy_absorption':
            direction = 'SHORT'
        else:
            return None

        near_ob = self._nearest_ob_distance_pct(state, direction) <= 0.35
        funding_against = (direction == 'LONG' and state.derivatives.funding_rate < 0) or (
            direction == 'SHORT' and state.derivatives.funding_rate > 0
        )

        if not near_ob or not funding_against or state.order_flow.cross_exchange_cvd_divergence:
            return None

        px = state.price_action.tf_15m.price
        entry_low = px * 0.999
        entry_high = px * 1.001
        extreme = state.price_action.tf_15m.swing_low_20 if direction == 'LONG' else state.price_action.tf_15m.swing_high_20
        structure_stop = extreme - 0.3 * state.volatility.atr14 if direction == 'LONG' else extreme + 0.3 * state.volatility.atr14

        return StrategyCandidate(
            direction=direction,
            strategy='S5',
            algo='Absorption_Reversal_v2',
            reason='Strong absorption + stacked imbalance + single-bar divergence + OB proximity',
            entry_low=entry_low,
            entry_high=entry_high,
            structure_stop=structure_stop,
            factors=['absorption_confirming', 'stacked_imbalance_confirming'],
        )

    def _hold(
        self,
        reason: str,
        state: FeatureState,
        regime: str,
        monitoring_stats: dict[str, Any],
        cooldown_until: str = '',
        *,
        anti_crowding: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        stale_after = now + timedelta(minutes=settings.signal_stale_minutes)
        payload: dict[str, Any] = {
            'signal': 'HOLD',
            'validated': 'HOLD',
            'confidence': 0,
            'stacked_probability': 0,
            'alpha_score': 0,
            'net_alpha': 0,
            'entry_zone': [0, 0],
            'stop_loss': 0,
            'take_profit': {'TP1': 0, 'TP2': 0, 'TP3': 0},
            'risk_reward': 0,
            'position_size_pct': 1.0,
            'position_size_btc': 0.0,
            'market_regime': regime,
            'session': state.macro.session,
            'execution_quality': 'poor',
            'mtf': {
                '15m': state.price_action.tf_15m.trend_bias,
                '1h': state.price_action.tf_1h.trend_bias,
                '4h': state.price_action.tf_4h.trend_bias,
                'alignment_score': max(state.alignment_score('LONG'), state.alignment_score('SHORT')),
            },
            'smc': {
                'ob_level': 0,
                'fvg_present': False,
                'bos_type': state.smc.bos_type,
                'liq_sweep_recent': state.smc.liquidity_sweep_recent,
                'choch': state.smc.choch,
            },
            'order_flow': {
                'cvd_slope': 'rising' if state.order_flow.cvd_slope > 0 else 'falling',
                'obi': round(state.order_flow.obi, 3),
                'whale_pressure': self._whale_pressure(state),
                'absorption': state.order_flow.absorption_side,
                'stacked_imbalance': state.order_flow.stacked_imbalance_direction,
                'single_bar_divergence': state.order_flow.single_bar_delta_divergence,
                'cross_exchange_cvd': state.order_flow.cross_exchange_alignment,
                'speed_of_tape': 'fast' if state.order_flow.speed_of_tape_ratio > 3 else 'normal',
            },
            'derivatives': {
                'funding_rate': round(state.derivatives.funding_rate, 6),
                'funding_momentum': self._funding_momentum_label(state.derivatives.funding_roc),
                'oi_change_1h': f"{state.derivatives.oi_change_1h_pct:+.2f}%",
                'liq_magnet': state.derivatives.liq_magnet_bias,
                'ls_ratio': f"{int(state.derivatives.long_short_ratio * 100)}/{int((1 - state.derivatives.long_short_ratio) * 100)}",
                'max_pain': int(round(state.derivatives.max_pain_level)),
                'iv_skew': round(state.derivatives.iv_skew, 4),
                'put_call_ratio': round(state.derivatives.put_call_ratio, 4),
            },
            'on_chain': {
                'exchange_netflow': 'negative' if state.onchain.exchange_netflow < 0 else 'positive',
                'sopr': round(state.onchain.sopr, 3),
                'lth_behavior': state.onchain.lth_behavior,
                'whale_deposits_1h': state.onchain.whale_exchange_deposits_1h,
                'whale_withdrawals_1h': state.onchain.whale_exchange_withdrawals_1h,
            },
            'macro': {
                'fear_greed': state.macro.fear_greed_score,
                'dxy_bias': state.macro.dxy_bias,
                'vix': round(state.macro.vix_level, 2),
                'vix_regime': state.macro.vix_regime,
                'spx_direction': state.macro.spx_intraday_direction,
                'spx_btc_correlation': round(state.macro.spx_btc_correlation, 3),
                'us_10y_trend': state.macro.us10y_yield_trend,
                'gold_direction': state.macro.gold_direction,
                'eth_btc_trend': state.macro.eth_btc_ratio_trend,
                'btc_dominance_trend': state.macro.btc_dominance_trend,
            },
            'algo': 'NO_TRADE',
            'strategy': 'NONE',
            'reason': reason,
            'factors_present': [],
            'independent_edges': {},
            'as_of_utc': utc_now_iso(),
            'cooldown_until': cooldown_until,
            'next_check_utc': (now + timedelta(minutes=15)).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'stale_after_utc': stale_after.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'model_version': self.model.version,
            'system_status': {
                'recent_win_rate': round(float(monitoring_stats.get('recent_win_rate', 0.0)), 4),
                'current_drawdown_pct': round(float(monitoring_stats.get('current_drawdown_pct', 0.0)), 2),
                'portfolio_heat_pct': round(float(monitoring_stats.get('portfolio_heat_pct', 0.0)), 2),
                'auto_pause': bool(monitoring_stats.get('auto_pause', False)),
            },
            'feature_state': {
                'ema9': state.price_action.tf_15m.ema9,
                'ema21': state.price_action.tf_15m.ema21,
                'ema50': state.price_action.tf_15m.ema50,
                'ema200': state.price_action.tf_15m.ema200,
                'atr14': state.volatility.atr14,
                'obi': state.order_flow.obi,
                'cvd_slope': state.order_flow.cvd_slope,
                'funding_rate': state.derivatives.funding_rate,
                'oi_change_1h': state.derivatives.oi_change_1h_pct,
                'fear_greed': state.macro.fear_greed_score,
            },
        }
        if anti_crowding:
            payload["anti_crowding"] = anti_crowding
        return payload

    @staticmethod
    def _nearest_ob_level(state: FeatureState, direction: str) -> float:
        obs = state.smc.bullish_obs if direction == 'LONG' else state.smc.bearish_obs
        if not obs:
            return 0.0
        return float(obs[-1].midpoint)

    @staticmethod
    def _nearest_ob_distance_pct(state: FeatureState, direction: str) -> float:
        obs = state.smc.bullish_obs if direction == 'LONG' else state.smc.bearish_obs
        price = max(state.price_action.tf_15m.price, 1e-9)
        if not obs:
            return 999.0
        return min(abs(price - x.midpoint) / price * 100.0 for x in obs)

    @staticmethod
    def _funding_momentum_label(funding_roc: float) -> str:
        if funding_roc > 0.25:
            return 'increasing_positive'
        if funding_roc < -0.25:
            return 'decreasing_negative'
        return 'stable'

    @staticmethod
    def _whale_pressure(state: FeatureState) -> str:
        if state.order_flow.whale_buy_ratio > 0.55:
            return 'buy'
        if state.order_flow.whale_buy_ratio < 0.45:
            return 'sell'
        return 'neutral'
