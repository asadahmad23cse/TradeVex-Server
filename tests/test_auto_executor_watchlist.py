from pathlib import Path
from datetime import datetime, timezone

from src.paper_trading.auto_executor import AutoExecutor
from src.paper_trading.paper_engine import PaperTradingEngine


def test_watchlist_prefers_tradeable_tickers(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        """
watchlist:
  indian_stocks:
    - { symbol: INFY, yf_ticker: INFY.NS, asset_class: indian_stock }
  us_stocks:
    - { symbol: AAPL, yf_ticker: AAPL, asset_class: us_stock }
  crypto:
    - { symbol: BTC, binance_ticker: BTCUSDT, asset_class: crypto }
    - { symbol: XRP, binance_ticker: XRPUSDT, asset_class: crypto }
    - { symbol: BNB, binance_ticker: BNBUSDT, asset_class: crypto }
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    engine = PaperTradingEngine(data_file=tmp_path / "paper.json")

    tickers = AutoExecutor(engine)._watchlist_tickers()

    assert tickers == ["INFY.NS", "AAPL", "BTCUSDT"]


def test_market_hours_guard_keeps_crypto_24_7_and_blocks_weekend_stocks():
    saturday = datetime(2026, 5, 2, 15, 30, tzinfo=timezone.utc)

    assert AutoExecutor._market_is_open("BTCUSDT", saturday)
    assert not AutoExecutor._market_is_open("RELIANCE.NS", saturday)
    assert not AutoExecutor._market_is_open("AAPL", saturday)
