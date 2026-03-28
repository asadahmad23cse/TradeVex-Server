from __future__ import annotations

from dataclasses import dataclass

from btc_intelligence.utils.validator import clamp


@dataclass
class OnChainFeatures:
    exchange_netflow: float
    exchange_netflow_bias: str
    netflow_score: float
    sopr: float
    sopr_state: str
    lth_supply_change: float
    lth_behavior: str
    whale_wallet_count: float
    whale_exchange_deposits_1h: int
    whale_exchange_withdrawals_1h: int
    whale_net_flow: float
    whale_net_flow_score: float


def compute_onchain(glassnode_data: dict, whale_tracker_data: dict) -> OnChainFeatures:
    netflow = float(glassnode_data.get('exchange_netflow', 0.0))
    sopr = float(glassnode_data.get('sopr', 1.0))
    lth = float(glassnode_data.get('lth_supply_change', 0.0))
    whales = float(glassnode_data.get('whale_wallet_count', 0.0))

    deposits = int(whale_tracker_data.get('whale_exchange_deposits_1h', 0))
    withdrawals = int(whale_tracker_data.get('whale_exchange_withdrawals_1h', 0))
    whale_net = float(whale_tracker_data.get('whale_net_flow', 0.0))

    netflow_bias = 'bullish' if netflow < 0 else 'bearish' if netflow > 0 else 'neutral'
    netflow_score = clamp(-netflow / 1000.0, -1.0, 1.0)

    if sopr > 1.0:
        sopr_state = 'above_1'
    elif sopr < 1.0:
        sopr_state = 'below_1'
    else:
        sopr_state = 'neutral'

    lth_behavior = 'accumulating' if lth > 0 else 'distributing' if lth < 0 else 'flat'
    whale_score = clamp(whale_net / 500.0, -1.0, 1.0)

    return OnChainFeatures(
        exchange_netflow=netflow,
        exchange_netflow_bias=netflow_bias,
        netflow_score=netflow_score,
        sopr=sopr,
        sopr_state=sopr_state,
        lth_supply_change=lth,
        lth_behavior=lth_behavior,
        whale_wallet_count=whales,
        whale_exchange_deposits_1h=deposits,
        whale_exchange_withdrawals_1h=withdrawals,
        whale_net_flow=whale_net,
        whale_net_flow_score=whale_score,
    )
