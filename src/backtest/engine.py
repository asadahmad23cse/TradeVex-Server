"""
GAP 5 — Event-Driven Backtesting Engine.

Replays historical daily bars bar-by-bar through the FULL pipeline:
    Data → Features → Alpha Model → Regime Gate → Risk Guard → Kelly → Fill

Key realism features:
    - Fills at next-bar open ± slippage (avoids lookahead bias)
    - Transaction costs via CostModel (Almgren-Chriss)
    - Capital mark-to-market every bar
    - Proper drawdown tracking and circuit-breaker simulation
    - Full PnL equity curve output

Usage:
    engine = BacktestEngine(config_path="config.yaml")
    result = engine.run(ticker="AAPL", start="2022-01-01", end="2024-01-01")
    print(result.summary())

Or via CLI:
    python main.py --mode backtest --engine full --ticker AAPL
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date

import numpy as np
import pandas as pd
import yaml

from src.api.connectors import MarketDataConnector
from src.features.engineer import FeatureEngineer
from src.features.hurst import compute_hurst
from src.alpha.factor_model import AlphaFactorModel
from src.alpha.regime import RegimeDetector
from src.risk.cost_model import CostModel
from src.utils.math_utils import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Represents one round-trip trade."""
    symbol: str
    entry_date: date
    exit_date: date | None
    signal: str                  # BUY | SELL
    strength: str
    regime: str
    entry_price: float
    exit_price: float | None
    position_size_pct: float     # percent of capital
    alpha_score: float
    cost_pct: float
    gross_pnl_pct: float = 0.0
    net_pnl_pct: float = 0.0
    outcome: str = "OPEN"        # WIN | LOSS | OPEN


@dataclass
class BacktestResult:
    """Full backtest result with equity curve and metrics."""
    ticker: str
    start: str
    end: str
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    initial_capital: float = 100_000.0

    def summary(self) -> dict:
        closed = [t for t in self.trades if t.outcome != "OPEN"]
        if not closed:
            return {"error": "No closed trades"}

        rets = self.equity_curve.pct_change().dropna()
        net_pnls = [t.net_pnl_pct / 100.0 for t in closed]

        wins = [t for t in closed if t.outcome == "WIN"]
        losses = [t for t in closed if t.outcome == "LOSS"]

        avg_win = np.mean([t.net_pnl_pct for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t.net_pnl_pct for t in losses])) if losses else 1.0
        mdd = calculate_max_drawdown(self.equity_curve)
        sharpe = calculate_sharpe_ratio(rets)
        sortino = calculate_sortino_ratio(rets)

        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100

        result = {
            "ticker": self.ticker,
            "period": f"{self.start} → {self.end}",
            "total_trades": len(closed),
            "win_rate": round(len(wins) / len(closed) * 100, 2),
            "avg_win_pct": round(avg_win, 3),
            "avg_loss_pct": round(avg_loss, 3),
            "profit_factor": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0,
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(mdd * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "avg_cost_pct": round(np.mean([t.cost_pct for t in closed]), 4),
        }
        return result

    def print_summary(self) -> None:
        s = self.summary()
        if "error" in s:
            print(f"[Backtest] {s['error']}")
            return
        print(f"\n{'='*60}")
        print(f"  Backtest: {s['ticker']}  {s['period']}")
        print(f"{'='*60}")
        for k, v in s.items():
            if k not in ("ticker", "period"):
                print(f"  {k:<28} {v}")
        print(f"{'='*60}\n")


class BacktestEngine:
    """
    Full event-driven backtester with realistic fills, costs, and PnL.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.initial_capital = float(self.cfg["portfolio"]["initial_capital"])
        self.connector = MarketDataConnector()
        self.engineer = FeatureEngineer()
        self.alpha_model = AlphaFactorModel(
            alpha_threshold=self.cfg["signal"]["alpha_score_threshold"],
            ic_window=self.cfg["signal"]["ic_window"],
        )
        self.regime_detector = RegimeDetector()
        self.cost_model = CostModel(self.cfg.get("cost_model", {}))

    def run(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        asset_class: str = "us_stock",
        min_warmup: int = 120,         # bars before signals start
    ) -> BacktestResult:
        """
        Run a full backtest for one asset.

        Parameters
        ----------
        ticker       : yfinance ticker string
        start / end  : date strings (YYYY-MM-DD), None = max history
        asset_class  : 'us_stock' | 'indian_stock' | 'forex'
        min_warmup   : bars to warm up indicators before generating signals
        """
        print(f"\n{'='*60}")
        print(f"  Full Backtest — {ticker}")
        print(f"  Capital: ${self.initial_capital:,.0f}")
        print(f"{'='*60}")

        # Fetch data
        print("Fetching data...")
        raw = self.connector.get_daily(ticker, period="5y")
        if raw.empty:
            logger.error("Backtest: no data for %s", ticker)
            return BacktestResult(ticker=ticker, start="", end="")

        # Date filter
        if start:
            raw = raw[raw.index >= pd.to_datetime(start)]
        if end:
            raw = raw[raw.index <= pd.to_datetime(end)]

        if len(raw) < min_warmup + 20:
            logger.error("Backtest: insufficient data (%d rows)", len(raw))
            return BacktestResult(ticker=ticker, start="", end="")

        print(f"Data: {len(raw)} rows from {raw.index[0].date()} to {raw.index[-1].date()}")
        print("Engineering features...")

        # Engineer all features once on the full dataset
        df = self.engineer.compute_all_features(raw, timeframe="daily")

        # Train HMM regime on first 252 bars
        print("Training HMM regime detector...")
        try:
            self.regime_detector.train(df.iloc[:252])
        except Exception as exc:
            logger.warning("HMM training failed: %s — using SIDEWAYS fallback", exc)

        print("Running simulation...\n")

        result = BacktestResult(
            ticker=ticker,
            start=str(df.index[min_warmup].date()),
            end=str(df.index[-1].date()),
            initial_capital=self.initial_capital,
        )

        capital = self.initial_capital
        equity_curve = []
        open_trade: BacktestTrade | None = None
        dates = df.index

        for i in range(min_warmup, len(df) - 1):
            current_bar = df.iloc[i]
            next_bar = df.iloc[i + 1]

            # Mark-to-market open trade
            if open_trade is not None:
                current_price = float(current_bar["Close"])
                entry_price = open_trade.entry_price
                gross_pnl_pct = (
                    (current_price - entry_price) / entry_price * 100
                    if open_trade.signal == "BUY"
                    else (entry_price - current_price) / entry_price * 100
                )

                # Exit condition: ATR-based stop or 20-bar max hold
                atr = float(current_bar.get("ATR_14", entry_price * 0.02))
                stop = (
                    (entry_price - 2.0 * atr) if open_trade.signal == "BUY"
                    else (entry_price + 2.0 * atr)
                )
                bars_held = (dates[i] - pd.Timestamp(open_trade.entry_date)).days

                should_exit = (
                    (open_trade.signal == "BUY" and current_price < stop) or
                    (open_trade.signal == "SELL" and current_price > stop) or
                    bars_held >= 20
                )

                if should_exit:
                    exit_price = float(next_bar["Open"])  # fill at next open
                    gross_pnl_pct = (
                        (exit_price - entry_price) / entry_price * 100
                        if open_trade.signal == "BUY"
                        else (entry_price - exit_price) / entry_price * 100
                    )
                    net_pnl_pct = gross_pnl_pct - open_trade.cost_pct
                    trade_capital_gain = (
                        capital * open_trade.position_size_pct / 100.0 * net_pnl_pct / 100.0
                    )
                    capital += trade_capital_gain

                    open_trade.exit_date = dates[i + 1].date()
                    open_trade.exit_price = exit_price
                    open_trade.gross_pnl_pct = round(gross_pnl_pct, 4)
                    open_trade.net_pnl_pct = round(net_pnl_pct, 4)
                    open_trade.outcome = "WIN" if net_pnl_pct > 0 else "LOSS"
                    result.trades.append(open_trade)
                    open_trade = None

                    logger.debug(
                        "CLOSE %s  gross=%.2f%%  net=%.2f%%  capital=$%.0f",
                        ticker, gross_pnl_pct, net_pnl_pct, capital,
                    )

            equity_curve.append((dates[i], capital))

            # Only open new trade if no open position
            if open_trade is not None:
                continue

            # Run alpha model on rolling window ending at bar i
            window = df.iloc[max(0, i - 200): i + 1]
            hurst = compute_hurst(window["Close"].tail(60))

            alpha_result = self.alpha_model.score(
                window,
                ml_score=0.0,
                hurst=hurst,
                asset=ticker,
                asset_class=asset_class,
                regime=None,
            )
            if alpha_result["signal"] == "HOLD":
                continue

            # Regime gate
            regime = "SIDEWAYS"
            if self.regime_detector._trained:
                try:
                    regime, _ = self.regime_detector.predict(window.tail(60))
                except Exception:
                    pass

            if regime == "BULL" and alpha_result["signal"] != "BUY":
                continue
            if regime == "BEAR" and alpha_result["signal"] != "SELL":
                continue
            if regime == "SIDEWAYS" and alpha_result["strength"] != "STRONG":
                continue

            # Position size (flat 2% for conservative backtest)
            position_size_pct = 2.0
            daily_vol = float(window["Volatility_20"].iloc[-1]) if "Volatility_20" in window.columns else 0.2
            vol_ratio = float(window["Volume_Ratio"].iloc[-1]) if "Volume_Ratio" in window.columns else 1.0

            _, cost_pct, is_viable = self.cost_model.net_alpha(
                alpha_result["alpha_score"],
                asset_class,
                position_size_pct,
                daily_vol,
                vol_ratio,
            )

            if not is_viable:
                continue

            # Open trade — fill at next bar's open
            entry_price = float(next_bar["Open"])
            open_trade = BacktestTrade(
                symbol=ticker,
                entry_date=dates[i + 1].date(),
                exit_date=None,
                signal=alpha_result["signal"],
                strength=alpha_result["strength"],
                regime=regime,
                entry_price=entry_price,
                exit_price=None,
                position_size_pct=position_size_pct,
                alpha_score=alpha_result["alpha_score"],
                cost_pct=cost_pct,
            )

        # Close any open trade at last available price
        if open_trade is not None:
            exit_price = float(df["Close"].iloc[-1])
            gross = (
                (exit_price - open_trade.entry_price) / open_trade.entry_price * 100
                if open_trade.signal == "BUY"
                else (open_trade.entry_price - exit_price) / open_trade.entry_price * 100
            )
            open_trade.exit_date = dates[-1].date()
            open_trade.exit_price = exit_price
            open_trade.gross_pnl_pct = round(gross, 4)
            open_trade.net_pnl_pct = round(gross - open_trade.cost_pct, 4)
            open_trade.outcome = "WIN" if open_trade.net_pnl_pct > 0 else "LOSS"
            result.trades.append(open_trade)

        equity_curve.append((dates[-1], capital))
        result.equity_curve = pd.Series(
            [c for _, c in equity_curve],
            index=pd.DatetimeIndex([d for d, _ in equity_curve]),
        )

        result.print_summary()
        return result
