"""
QuantTrader — Main Entry Point.

Modes:
  live       — Full live system (scheduler + dashboard server)
  dashboard  — Dashboard server only (no data fetching)
  signals    — Terminal-only signal printing (no dashboard)
  backtest   — Run Walk-Forward Optimization on a ticker
  btc_backtest — Run BTC walk-forward backtest on Binance history
"""

import argparse
import logging
import sys
import threading

import uvicorn  # type: ignore[import]
import yaml  # type: ignore[import]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def run_live(config_path: str) -> None:
    """Start the full live trading system + dashboard."""
    from src.dashboard.api import app, init_dashboard, push_broadcast_threadsafe, set_live_runner  # type: ignore[import]
    from src.scheduler.live_runner import LiveRunner  # type: ignore[import]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    runner = LiveRunner(config_path=config_path)
    init_dashboard(runner.store, runner.portfolio, cfg)
    set_live_runner(runner)
    runner.set_broadcast(push_broadcast_threadsafe)

    # Start scheduler in background thread
    sched_thread = threading.Thread(target=runner.start, daemon=True)
    sched_thread.start()

    host = cfg["dashboard"]["host"]
    port = cfg["dashboard"]["port"]
    logger.info("Dashboard: http://%s:%d", host, port)

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
        )
    finally:
        # Ensure scheduler/resources are flushed when the API process exits.
        try:
            runner.stop()
        except Exception as exc:
            logger.warning("Live runner shutdown during exit failed: %s", exc)


def run_dashboard(config_path: str) -> None:
    """Dashboard only — no live data fetching."""
    from src.dashboard.api import app, init_dashboard, set_live_runner  # type: ignore[import]
    from src.signals.store import SignalStore  # type: ignore[import]
    from src.risk.portfolio import PortfolioTracker  # type: ignore[import]
    import yaml  # type: ignore[import]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    db_url = cfg.get("database", {}).get("url", "data/signals.db")
    store = SignalStore(db_url)
    portfolio = PortfolioTracker(
        initial_capital=float(cfg["portfolio"]["initial_capital"])
    )
    init_dashboard(store, portfolio, cfg)
    set_live_runner(None)

    host = cfg["dashboard"]["host"]
    port = cfg["dashboard"]["port"]
    logger.info("Dashboard (read-only): http://%s:%d", host, port)

    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_signals(config_path: str) -> None:
    """Terminal-only mode: prints signals to stdout without a dashboard."""
    from src.scheduler.live_runner import LiveRunner  # type: ignore[import]
    runner = LiveRunner(config_path=config_path)
    logger.info("Running in terminal-signals mode. Ctrl+C to stop.")
    runner.start()


def run_backtest(ticker: str, train_days: int, config_path: str = "config.yaml") -> None:
    """Run Walk-Forward Optimization with 5-factor alpha model."""
    from src.validator import WFOValidator  # type: ignore[import]
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        wfo_cfg = cfg.get("wfo", {})
    except Exception:
        wfo_cfg = {}
    validator = WFOValidator(ticker=ticker, train_window=train_days, wfo_config=wfo_cfg)
    result = validator.run_validation()
    if result.get("passed"):
        logger.info(
            "WFO PASSED — best params: alpha_threshold=%.2f, ic_window=%d, OOS IR=%.3f",
            result["best_alpha_threshold"], result["best_ic_window"], result["best_ir"]
        )
    else:
        logger.warning(
            "WFO FAILED — no parameter set achieved IR > 0.3. Best OOS IR=%.3f",
            result.get("best_ir", 0.0)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuantTrader — Institutional-Grade Signal System"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "dashboard", "signals", "backtest", "capacity", "btc_backtest"],
        default="live",
        help="Execution mode (default: live)",
    )
    parser.add_argument(
        "--engine",
        choices=["wfo", "full"],
        default="wfo",
        help="Backtest engine: 'wfo' = Walk-Forward IC/IR validation, 'full' = event-driven PnL backtest (default: wfo)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--ticker",
        default="RELIANCE.NS",
        help="Ticker for backtest mode",
    )
    parser.add_argument(
        "--train_days",
        type=int,
        default=200,
        help="Training window in days for backtest mode",
    )
    args = parser.parse_args()

    logger.info("=== QuantTrader | Mode: %s ===", args.mode.upper())

    if args.mode == "live":
        run_live(args.config)
    elif args.mode == "dashboard":
        run_dashboard(args.config)
    elif args.mode == "signals":
        run_signals(args.config)
    elif args.mode == "backtest":
        if args.engine == "full":
            from src.backtest.engine import BacktestEngine  # type: ignore[import]
            engine = BacktestEngine(config_path=args.config)
            engine.run(ticker=args.ticker)
        else:
            run_backtest(args.ticker, args.train_days, args.config)
    elif args.mode == "capacity":
        from src.backtest.capacity import CapacitySimulator  # type: ignore[import]
        sim = CapacitySimulator(config_path=args.config)
        if args.ticker.upper() == "ALL":
            print(sim.run_watchlist())
        else:
            sim.run(ticker=args.ticker)
    elif args.mode == "btc_backtest":
        from src.backtest.btc_backtest import (  # type: ignore[import]
            BTCHistoricalLoader,
            BTCWalkForwardBacktest,
            BTCBacktestReport,
        )

        loader = BTCHistoricalLoader()
        df = loader.fetch(
            symbol=args.ticker or "BTCUSDT",
            interval="1h",
        )
        backtest = BTCWalkForwardBacktest()
        results = backtest.run(df, symbol=args.ticker or "BTCUSDT", interval="1h")
        report = BTCBacktestReport(config_path=args.config)
        report.generate(results)


if __name__ == "__main__":
    main()
