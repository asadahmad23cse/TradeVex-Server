from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from btc_intelligence.features.derivatives import DerivativesFeatures, compute_derivatives
from btc_intelligence.features.macro import MacroFeatures, compute_macro
from btc_intelligence.features.onchain import OnChainFeatures, compute_onchain
from btc_intelligence.features.order_flow import OrderFlowFeatures, compute_order_flow
from btc_intelligence.features.price_action import PriceActionFeatures, compute_price_action
from btc_intelligence.features.smc import SMCFeatures, compute_smc
from btc_intelligence.features.volatility import VolatilityFeatures, compute_volatility
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


FEATURE_COLUMNS = [
    # Price features
    'ema9_dist', 'ema21_dist', 'ema50_dist', 'ema200_dist', 'atr_normalized', 'bb_position', 'vwap_distance',
    # Structure
    'mtf_alignment_score', 'bos_candles_ago', 'choch_detected', 'ob_distance_pct', 'fvg_present', 'liquidity_sweep_recent',
    # Order flow enhanced
    'cvd_slope', 'obi', 'ofi_5bar', 'whale_buy_ratio',
    'absorption_strength', 'stacked_imbalance_direction', 'single_bar_divergence', 'iceberg_direction',
    'cross_exchange_cvd_divergence', 'speed_of_tape_ratio',
    # Derivatives enhanced
    'funding_rate', 'funding_roc', 'oi_change_1h', 'ls_ratio', 'liq_cluster_above_pct', 'liq_cluster_below_pct',
    'max_pain_distance_pct', 'iv_skew',
    # Options
    'put_call_ratio', 'iv_rv_ratio', 'options_expiry_hours',
    # On-chain enhanced
    'exchange_netflow_score', 'sopr', 'lth_supply_change', 'whale_exchange_deposit_count', 'whale_net_flow_score',
    # Macro enhanced
    'fng_score', 'dxy_bias_encoded', 'vix_level', 'us_10y_yield_trend', 'spx_btc_correlation',
    'spx_intraday_direction', 'gold_btc_agreement',
    # Crypto correlations
    'eth_btc_ratio_trend', 'btc_dominance_trend',
    # Context
    'session_encoded', 'vol_regime_encoded', 'regime_encoded',
]


@dataclass
class FeatureState:
    price_action: PriceActionFeatures
    smc: SMCFeatures
    order_flow: OrderFlowFeatures
    volatility: VolatilityFeatures
    derivatives: DerivativesFeatures
    onchain: OnChainFeatures
    macro: MacroFeatures

    def alignment_score(self, direction: str) -> int:
        return self.price_action.alignment_long if direction == 'LONG' else self.price_action.alignment_short

    def nearest_ob_distance_pct(self, direction: str) -> float:
        price = self.price_action.tf_15m.price
        obs = self.smc.bullish_obs if direction == 'LONG' else self.smc.bearish_obs
        if not obs or price <= 0:
            return 999.0
        d = min(abs(price - x.midpoint) / price * 100.0 for x in obs)
        return float(d)

    def to_vector(self, direction: str, regime: str) -> tuple[np.ndarray, dict[str, float]]:
        tf = self.price_action.tf_15m
        price = max(tf.price, 1e-9)

        ob_dist = self.nearest_ob_distance_pct(direction)
        fvg_present = int(any(x.direction == ('bullish' if direction == 'LONG' else 'bearish') for x in self.smc.fvgs))
        bos_ago = 1 if self.smc.bos_type != 'none' else 50

        enc_dir = {'up': 1.0, 'down': -1.0, 'flat': 0.0}
        dxy_enc = {'risk_on': -1.0, 'neutral': 0.0, 'risk_off': 1.0}.get(self.macro.dxy_bias, 0.0)

        session_enc = {
            'asian': 1,
            'london_open': 2,
            'ny_london_overlap': 3,
            'new_york': 4,
            'dead': 0,
        }.get(self.macro.session, 0)
        vol_enc = {'compression': 0, 'normal': 1, 'expansion': 2, 'spike': 3}.get(self.volatility.vol_regime, 1)
        regime_enc = {
            'bullish_trend': 1,
            'bearish_trend': -1,
            'sideways_range': 0,
            'breakout_up': 2,
            'breakout_down': -2,
            'panic_liquidation': 3,
        }.get(regime, 0)

        stacked_enc = {
            'none': 0.0,
            'bullish_3_levels': 1.0,
            'bearish_3_levels': -1.0,
        }.get(self.order_flow.stacked_imbalance_direction, 0.0)

        iceberg_enc = {'none': 0.0, 'bullish': 1.0, 'bearish': -1.0}.get(self.order_flow.iceberg_direction, 0.0)

        gold_btc_agree = 1.0 if self.macro.gold_direction == self.macro.spx_intraday_direction and self.macro.gold_direction != 'flat' else 0.0

        mapping = {
            'ema9_dist': (tf.price - tf.ema9) / price,
            'ema21_dist': (tf.price - tf.ema21) / price,
            'ema50_dist': (tf.price - tf.ema50) / price,
            'ema200_dist': (tf.price - tf.ema200) / price,
            'atr_normalized': self.volatility.atr14 / price,
            'bb_position': self.volatility.bb_position,
            'vwap_distance': self.volatility.vwap_distance_pct / 100.0,

            'mtf_alignment_score': float(self.alignment_score(direction)),
            'bos_candles_ago': float(bos_ago),
            'choch_detected': float(int(self.smc.choch)),
            'ob_distance_pct': ob_dist,
            'fvg_present': float(fvg_present),
            'liquidity_sweep_recent': float(int(self.smc.liquidity_sweep_recent)),

            'cvd_slope': self.order_flow.cvd_slope,
            'obi': self.order_flow.obi,
            'ofi_5bar': self.order_flow.ofi_5bar,
            'whale_buy_ratio': self.order_flow.whale_buy_ratio,
            'absorption_strength': self.order_flow.absorption_strength,
            'stacked_imbalance_direction': stacked_enc,
            'single_bar_divergence': float(int(self.order_flow.single_bar_delta_divergence)),
            'iceberg_direction': iceberg_enc,
            'cross_exchange_cvd_divergence': float(int(self.order_flow.cross_exchange_cvd_divergence)),
            'speed_of_tape_ratio': self.order_flow.speed_of_tape_ratio,

            'funding_rate': self.derivatives.funding_rate,
            'funding_roc': self.derivatives.funding_roc,
            'oi_change_1h': self.derivatives.oi_change_1h_pct,
            'ls_ratio': self.derivatives.long_short_ratio,
            'liq_cluster_above_pct': self.derivatives.liq_cluster_above_pct,
            'liq_cluster_below_pct': self.derivatives.liq_cluster_below_pct,
            'max_pain_distance_pct': self.derivatives.max_pain_distance_pct,
            'iv_skew': self.derivatives.iv_skew,

            'put_call_ratio': self.derivatives.put_call_ratio,
            'iv_rv_ratio': self.volatility.iv_rv_ratio,
            'options_expiry_hours': self.derivatives.options_expiry_hours,

            'exchange_netflow_score': self.onchain.netflow_score,
            'sopr': self.onchain.sopr,
            'lth_supply_change': self.onchain.lth_supply_change,
            'whale_exchange_deposit_count': float(self.onchain.whale_exchange_deposits_1h),
            'whale_net_flow_score': self.onchain.whale_net_flow_score,

            'fng_score': float(self.macro.fear_greed_score),
            'dxy_bias_encoded': dxy_enc,
            'vix_level': self.macro.vix_level,
            'us_10y_yield_trend': enc_dir.get(self.macro.us10y_yield_trend, 0.0),
            'spx_btc_correlation': self.macro.spx_btc_correlation,
            'spx_intraday_direction': enc_dir.get(self.macro.spx_intraday_direction, 0.0),
            'gold_btc_agreement': gold_btc_agree,

            'eth_btc_ratio_trend': enc_dir.get(self.macro.eth_btc_ratio_trend, 0.0),
            'btc_dominance_trend': enc_dir.get(self.macro.btc_dominance_trend, 0.0),

            'session_encoded': float(session_enc),
            'vol_regime_encoded': float(vol_enc),
            'regime_encoded': float(regime_enc),
        }

        vector = np.asarray([[float(mapping[col]) for col in FEATURE_COLUMNS]], dtype=float)
        return vector, mapping


def build_feature_state(snapshot: dict[str, Any]) -> FeatureState:
    df_15m = MarketDataBuffer.candles_to_df(snapshot, '15m')
    df_1h = MarketDataBuffer.candles_to_df(snapshot, '1h')
    df_4h = MarketDataBuffer.candles_to_df(snapshot, '4h')

    price_action = compute_price_action(df_15m, df_1h, df_4h)
    smc = compute_smc(df_15m, trend_bias=price_action.tf_15m.trend_bias)

    order_flow = compute_order_flow(
        agg_trades=snapshot.get('agg_trades', []),
        depth=snapshot.get('depth', {}),
        multi_exchange=snapshot.get('multi_exchange', {}),
        df_15m=df_15m,
    )

    volatility = compute_volatility(df_15m, df_1h, deribit_data=snapshot.get('deribit', {}))

    close_1h = df_1h['Close'].astype(float)
    price_1h_ago = float(close_1h.iloc[-2]) if len(close_1h) >= 2 else float(close_1h.iloc[-1])
    derivatives = compute_derivatives(
        rest_data=snapshot.get('binance_rest', {}),
        oi_hist=snapshot.get('open_interest_hist', []),
        funding_hist=snapshot.get('funding_hist', []),
        coinglass_data=snapshot.get('coinglass', {}),
        deribit_data=snapshot.get('deribit', {}),
        price_now=price_action.tf_15m.price,
        price_1h_ago=price_1h_ago,
    )

    onchain = compute_onchain(snapshot.get('glassnode', {}), snapshot.get('whale_tracker', {}))
    macro = compute_macro(snapshot.get('macro', {}), snapshot.get('news', []))

    return FeatureState(
        price_action=price_action,
        smc=smc,
        order_flow=order_flow,
        volatility=volatility,
        derivatives=derivatives,
        onchain=onchain,
        macro=macro,
    )
