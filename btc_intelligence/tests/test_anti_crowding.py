from __future__ import annotations

from btc_intelligence.signals.anti_crowding import compute_flow_hhi_and_imbalance


def synthetic_trades_one_sided(n: int = 120) -> list[dict]:
    # All aggressive buys at nearly same price → high HHI + imbalance
    mid = 100_000.0
    out: list[dict] = []
    for i in range(n):
        out.append({"p": str(mid + (i % 3) * 0.5), "q": "0.01", "T": 1_700_000_000_000 + i * 1000, "m": False})
    return out


def test_crowding_triggers_on_concentrated_aggressive_flow():
    st = compute_flow_hhi_and_imbalance(synthetic_trades_one_sided(80), min_trades=20)
    assert st.aggressive_buy_share > 0.95
    assert st.hhi > 0.3
    assert st.crowding_score_0_100 > 50
