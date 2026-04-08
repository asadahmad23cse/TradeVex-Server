from __future__ import annotations

from btc_intelligence.regime.hmm_regime import (
    candles_to_hmm_observations,
    hmm_regime_label,
    posterior_state_probs,
)
from btc_intelligence.regime.classifier import RegimeResult


def _candles_close_only(n: int, start: float, drift: float) -> list[dict]:
    out: list[dict] = []
    p = start
    base_t = 1_700_000_000_000
    for i in range(n):
        h = p * 1.001
        l = p * 0.999
        o = p
        c = p * (1.0 + drift)
        out.append(
            {
                "open_time": base_t + i * 900_000,
                "open": o,
                "high": max(h, c),
                "low": min(l, c),
                "close": c,
                "volume": 100.0,
            }
        )
        p = c
    return out


def test_candles_to_observations_shape() -> None:
    c = _candles_close_only(120, 50_000.0, 0.0001)
    obs = candles_to_hmm_observations(c)
    assert obs is not None
    assert obs.ndim == 2
    assert obs.shape[1] == 2


def test_posterior_probs_sums_to_one() -> None:
    c = _candles_close_only(180, 50_000.0, 0.0002)
    obs = candles_to_hmm_observations(c)
    probs = posterior_state_probs(obs)
    if probs is None:
        return  # hmmlearn optional failure environments
    s = sum(probs.values())
    assert abs(s - 1.0) < 0.01


def test_hmm_regime_label_trending_bull() -> None:
    class _PA:
        class _TF:
            price = 51_000.0
            ema50 = 50_000.0

        tf_4h = _TF()

    st = type("FS", (), {"price_action": _PA()})()
    probs = {"mean_reverting": 0.2, "trending": 0.65, "liquidity_cascade": 0.15}
    reg, score, reason = hmm_regime_label(probs, st, recent_log_ret_sum=0.01)
    assert reg == "bullish_trend"
    assert score >= 1
    assert "HMM" in reason


def test_regime_result_has_state_probs() -> None:
    r = RegimeResult("sideways_range", 3, "x", state_probs={"mean_reverting": 1.0})
    assert r.state_probs["mean_reverting"] == 1.0
