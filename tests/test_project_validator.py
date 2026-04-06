from types import SimpleNamespace

from src.research.project_validator import compute_trade_accuracy, infer_asset_class


def _trade(signal: str, outcome: str, net_pnl_pct: float) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="AAPL",
        signal=signal,
        net_pnl_pct=net_pnl_pct,
        outcome=outcome,
    )


def test_compute_trade_accuracy_directional_and_side_rates() -> None:
    trades = [
        _trade("BUY", "WIN", 1.5),
        _trade("BUY", "LOSS", -0.6),
        _trade("SELL", "WIN", 1.0),
        _trade("SELL", "LOSS", -0.4),
    ]
    metrics = compute_trade_accuracy(trades)

    assert metrics["closed_trades"] == 4
    assert metrics["wins"] == 2
    assert metrics["losses"] == 2
    assert metrics["signal_accuracy_pct"] == 50.0
    assert metrics["buy_win_rate_pct"] == 50.0
    assert metrics["sell_win_rate_pct"] == 50.0
    assert metrics["profit_factor"] == 2.5


def test_infer_asset_class() -> None:
    assert infer_asset_class("RELIANCE.NS") == "indian_stock"
    assert infer_asset_class("EURUSD") == "forex"
    assert infer_asset_class("AAPL") == "us_stock"
