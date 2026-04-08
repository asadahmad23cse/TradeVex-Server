from __future__ import annotations

from btc_intelligence.services.meta_labeling import MetaLabelingEngine, build_rf_feature_vector


def test_build_rf_feature_vector_shape():
    v = build_rf_feature_vector({"hour_sin": 0.1, "spread_pct": 0.02})
    assert v.shape == (1, 7)


def test_meta_label_trade_accepts_meta_features():
    eng = MetaLabelingEngine()
    meta = {
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "hmm_mean_reverting": 0.33,
        "hmm_trending": 0.34,
        "hmm_liquidity_cascade": 0.33,
        "vix_level": 22.0,
        "spread_pct": 0.03,
    }
    out = eng.label_trade(
        base_decision="LONG",
        confidence=80.0,
        calibrated_prob=0.62,
        blockers=[],
        calibration_error=0.05,
        drift_level="LOW",
        regime="BULLISH",
        edge_decay=False,
        meta_features=meta,
    )
    assert "allow_trade" in out
    assert "rf_model_active" in out["inputs"]


def test_alpha_barrier_decision_engine_blocks():
    from btc_intelligence.services.decision_engine import DecisionEngine, DecisionEngineInput

    de = DecisionEngine()
    out = de.evaluate(
        DecisionEngineInput(
            regime="bullish_trend",
            cvd_slope=0.1,
            obi_imbalance=0.2,
            volatility_regime="NORMAL",
            cost_score=0.1,
            momentum_score=0.2,
            aggression_buy_pct=55.0,
            flow_score=0.3,
            expected_return_frac=0.0001,
            win_prob=0.5,
            spread_frac=0.0005,
            slippage_frac=0.0005,
        )
    )
    assert any("Alpha barrier" in b for b in out.get("blockers", []))


def test_validation_alpha_barrier_rejects():
    from btc_intelligence.services.validation_engine import ValidationEngine

    v = ValidationEngine()
    payload = {
        "sample_size": 40,
        "expected_edge": 0.05,
        "robustness_gain": 0.01,
        "drawdown_delta": -0.02,
        "overfit_risk": 0.1,
        "brier_score": 0.2,
        "calibration_error": 0.05,
        "expected_return_frac": 0.0002,
        "win_prob": 0.5,
        "spread_frac": 0.0004,
        "slippage_frac": 0.0004,
    }
    r = v.validate_step("meta_decision", payload)
    assert r["checks"].get("alpha_barrier") is False
    assert r["approved"] is False
