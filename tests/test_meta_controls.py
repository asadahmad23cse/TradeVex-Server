from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.dashboard.btc_service import BitcoinMarketService
from src.meta.alert_filter import AlertNoiseFilter
from src.meta.calibration_freshness import CalibrationFreshnessGuard
from src.meta.config import MetaControlConfigLoader
from src.meta.data_confidence import DataConfidenceEngine
from src.meta.kelly_shrinkage import KellyShrinkageController
from src.meta.regime_explainer import RegimeBlockExplainer
from src.paper_trading.paper_engine import PaperTradingEngine


def _reset_meta_config_cache() -> None:
    MetaControlConfigLoader._cached = None
    MetaControlConfigLoader._cached_path = None
    MetaControlConfigLoader._cached_mtime = None
    MetaControlConfigLoader._cached_loaded_at = 0.0


def test_data_confidence_scores_and_blocks() -> None:
    engine = DataConfidenceEngine(
        {"block_threshold": 0.5, "degrade_threshold": 0.7},
        shadow_mode=False,
        enforce_execution_gates=True,
    )

    result = engine.evaluate(
        {
            "binance": "working",
            "coinglass": "fallback",
            "etf": "fallback",
            "fred": "fallback",
            "news": "fallback",
        }
    )

    assert result.data_confidence_score == 0.7
    assert not result.degraded
    assert not result.execution_block_new_trades

    blocked = engine.evaluate({feed: "failed" for feed in DataConfidenceEngine.WEIGHTS})
    assert blocked.blocked
    assert blocked.execution_block_new_trades


def test_kelly_shrinkage_caps_and_detects_existing_effective() -> None:
    controller = KellyShrinkageController({"max_equity_pct_per_trade": 2.0})

    capped = controller.adjust(0.08, 17)
    assert capped.kelly_multiplier == 0.3
    assert capped.effective_kelly_fraction == 0.02
    assert capped.kelly_cap_applied

    existing = controller.adjust(0.08, 17, already_effective=True, existing_effective_fraction=0.015)
    assert existing.already_shrunk
    assert existing.effective_kelly_fraction == 0.015


def test_calibration_freshness_shadow_and_enforce_modes() -> None:
    now = datetime(2026, 5, 2, tzinfo=timezone.utc)
    stale_ts = now - timedelta(days=18)
    critical_ts = now - timedelta(days=31)

    shadow = CalibrationFreshnessGuard({"stale_days": 14, "critical_days": 30}, shadow_mode=True, enforce_execution_gates=True)
    stale = shadow.evaluate(stale_ts, current_time=now)
    assert stale.status == "stale"
    assert stale.calibration_warning
    assert stale.execution_confidence_multiplier == 1.0

    enforce = CalibrationFreshnessGuard({"stale_days": 14, "critical_days": 30}, shadow_mode=False, enforce_execution_gates=True)
    critical = enforce.evaluate(critical_ts, current_time=now)
    assert critical.status == "critical"
    assert critical.execution_position_multiplier == 0.5


def test_alert_filter_shadow_and_enforce() -> None:
    alert = {"asset": "INFY", "severity": "CRITICAL", "type": "EOD"}

    shadow = AlertNoiseFilter({"active_assets": ["BTC"]}, shadow_mode=True, enforce=True).filter_alert(alert)
    assert shadow.downgraded
    assert shadow.filtered_level == "INFO"
    assert shadow.alert["severity"] == "CRITICAL"

    enforced = AlertNoiseFilter({"active_assets": ["BTC"]}, shadow_mode=False, enforce=True).filter_alert(alert)
    assert enforced.alert["severity"] == "INFO"

    btc = AlertNoiseFilter({"active_assets": ["BTC"]}, shadow_mode=False, enforce=True).filter_alert(
        {"asset": "BTCUSDT", "severity": "CRITICAL", "type": "BTC"}
    )
    assert not btc.downgraded
    assert btc.alert["severity"] == "CRITICAL"


def test_regime_block_explainer_attaches_only_for_existing_regime_blocks() -> None:
    payload = {
        "requested_signal": "SHORT",
        "blocked_by": "regime_gate",
        "regime": "BULLISH TREND",
        "reason": "regime_conflict: SHORT blocked in BULLISH TREND regime",
        "funding_rate_z": 0.4,
        "market_overview": {"volatility": "NORMAL"},
    }

    detail = RegimeBlockExplainer().explain(payload)
    assert detail == {
        "regime": "BULLISH TREND",
        "indicator": "HTF_EMA",
        "supporting_factors": ["funding_rate", "volatility"],
        "blocked_action": "SHORT",
    }
    assert RegimeBlockExplainer().explain({"blocked_by": "cost_gate"}) is None


def test_btc_meta_attachment_preserves_existing_signal_fields() -> None:
    service = BitcoinMarketService(
        {
            "meta_controls": {
                "enabled": True,
                "shadow_mode": True,
                "enforce_execution_gates": False,
            }
        }
    )
    payload = {
        "asset": "BTCUSDT",
        "signal": "WAIT",
        "validated_signal": "HOLD",
        "confidence": 73.7,
        "alpha_score": 81,
        "regime": "BULLISH TREND",
        "requested_signal": "SHORT",
        "blocked_by": "regime_gate",
        "reason": "regime_conflict: SHORT blocked in BULLISH TREND regime",
        "entry_price": 64000.0,
        "market_context": {
            "futures": {"mark_price": 64000.0, "funding_rate_pct": 0.01},
            "etf_flow": {"source": "fallback"},
        },
    }
    original = {key: payload[key] for key in ("signal", "validated_signal", "confidence", "alpha_score", "regime")}

    out = service._attach_meta_controls(payload)

    assert {key: out[key] for key in original} == original
    assert "meta_controls" in out
    assert "data_confidence_score" in out
    assert out["block_reason_detail"]["blocked_action"] == "SHORT"


def test_paper_engine_shadow_mode_does_not_change_trade_behavior(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(
        """
meta_controls:
  enabled: true
  shadow_mode: true
  enforce_execution_gates: false
  data_confidence: { enabled: true }
  kelly_shrinkage: { enabled: true, max_equity_pct_per_trade: 2.0 }
  calibration_freshness: { enabled: true, stale_days: 14, critical_days: 30 }
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_meta_config_cache()
    engine = PaperTradingEngine(initial_capital=100000.0, data_file=tmp_path / "paper.json")

    result = engine.execute_trade(
        {
            "ticker": "BTCUSDT",
            "signal": "LONG",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "confidence": 80.0,
        },
        mode="auto",
    )

    assert result["success"]
    assert result["quantity"] == 100.0
    assert result["value"] == 10000.0
    assert result["risk_pct_at_sl"] == 0.5
    assert result["meta_controls"]["shadow_mode"] is True


def test_enforce_mode_blocks_execution_only_without_mutating_signal(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(
        """
meta_controls:
  enabled: true
  shadow_mode: false
  enforce_execution_gates: true
  data_confidence:
    enabled: true
    block_threshold: 0.5
    degrade_threshold: 0.7
  kelly_shrinkage: { enabled: false }
  calibration_freshness: { enabled: false }
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_meta_config_cache()
    engine = PaperTradingEngine(initial_capital=100000.0, data_file=tmp_path / "paper.json")
    signal = {
        "ticker": "BTCUSDT",
        "signal": "LONG",
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "confidence": 80.0,
        "meta_controls": {
            "data_confidence": {
                "feed_status": {
                    "binance": "failed",
                    "coinglass": "failed",
                    "etf": "failed",
                    "fred": "failed",
                    "news": "failed",
                }
            }
        },
    }

    result = engine.execute_trade(signal, mode="auto")

    assert result["success"] is False
    assert result["blocked_by"] == "data_confidence"
    assert signal["signal"] == "LONG"
    assert signal["confidence"] == 80.0
