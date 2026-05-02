from __future__ import annotations

import json
from pathlib import Path

import src.paper_trading as paper_trading


def _legacy_state() -> dict:
    return {
        "capital": 100500.0,
        "initial_capital": 100000.0,
        "positions": {},
        "closed_trades": [
            {
                "trade_id": "PT-LEGACY-1",
                "ticker": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 70000.0,
                "exit_price": 70500.0,
                "quantity": 0.1,
                "pnl": 50.0,
                "pnl_pct": 0.7142,
                "reason": "manual",
                "opened_at": "2026-04-01T00:00:00+00:00",
                "closed_at": "2026-04-01T01:00:00+00:00",
                "held_hours": 1.0,
                "asset_class": "crypto",
                "confidence": 70.0,
                "mode": "auto",
            }
        ],
        "mode": "manual",
        "auto_enabled": False,
        "created_at": 0.0,
        "stats": {
            "total_trades": 1,
            "wins": 1,
            "losses": 0,
            "total_pnl": 50.0,
            "peak_equity": 100500.0,
            "max_drawdown": 0.0,
        },
    }


def setup_function() -> None:
    paper_trading._paper_engines_by_user.clear()
    paper_trading._auto_executors_by_user.clear()


def test_local_default_user_reads_legacy_paper_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    legacy_file = Path("data") / "paper_trading.json"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text(json.dumps(_legacy_state()), encoding="utf-8")

    engine = paper_trading.get_user_paper_engine("local-default")

    assert engine.data_file == legacy_file
    trades = engine.get_closed_trades(10)
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "PT-LEGACY-1"


def test_named_user_still_uses_isolated_user_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    legacy_file = Path("data") / "paper_trading.json"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text(json.dumps(_legacy_state()), encoding="utf-8")

    engine = paper_trading.get_user_paper_engine("alice@example.com")

    assert "paper_trading_users" in str(engine.data_file).replace("\\", "/")
    assert engine.data_file != legacy_file


def test_local_default_prefers_user_scoped_file_when_it_has_data(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    legacy_file = Path("data") / "paper_trading.json"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text(
        json.dumps(
            {
                "capital": 100000.0,
                "initial_capital": 100000.0,
                "positions": {},
                "closed_trades": [],
                "mode": "manual",
                "auto_enabled": False,
                "created_at": 0.0,
                "stats": {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "peak_equity": 100000.0, "max_drawdown": 0.0},
            }
        ),
        encoding="utf-8",
    )

    user_file = Path("data") / "paper_trading_users" / "local-default.json"
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text(json.dumps(_legacy_state()), encoding="utf-8")

    engine = paper_trading.get_user_paper_engine("local-default")

    assert engine.data_file == user_file
    trades = engine.get_closed_trades(10)
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "PT-LEGACY-1"
