import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.btc_backtest import (
    BTCHistoricalLoader,
    BTCWalkForwardBacktest,
    BTCBacktestReport,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--folds", type=int, default=6)
    args = parser.parse_args()

    print(f"Loading {args.symbol} {args.interval} data...")
    loader = BTCHistoricalLoader()
    df = loader.fetch(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
    )
    print(f"Loaded {len(df)} candles")

    print("Running walk-forward backtest...")
    backtest = BTCWalkForwardBacktest(min_folds=args.folds)
    results = backtest.run(df, symbol=args.symbol, interval=args.interval)

    print("Generating report...")
    report = BTCBacktestReport()
    report.generate(results)


if __name__ == "__main__":
    main()
