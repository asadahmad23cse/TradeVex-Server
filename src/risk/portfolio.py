"""
Layer 7 — Portfolio Tracker.

Tracks open positions, mark-to-market PnL, and computes all
institutional risk metrics: Sharpe, Sortino, Calmar, VaR, CVaR,
Win Rate, Profit Factor.

Transaction costs are applied on every PnL calculation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date

import numpy as np
import pandas as pd

from src.utils.math_utils import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_profit_factor,
    calculate_annualized_return,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    signal_id: str
    asset: str
    asset_class: str
    signal: str          # 'BUY' | 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    slippage_cost_pct: float
    opened_at: datetime = field(default_factory=datetime.utcnow)
    current_price: float = 0.0

    @property
    def gross_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.signal == "BUY":
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100

    @property
    def net_pnl_pct(self) -> float:
        """PnL after transaction costs (entry + exit slippage)."""
        return self.gross_pnl_pct - (2 * self.slippage_cost_pct * 100)

    @property
    def weighted_gross_pnl_pct(self) -> float:
        return self.gross_pnl_pct * (self.position_size_pct / 100)

    @property
    def weighted_net_pnl_pct(self) -> float:
        return self.net_pnl_pct * (self.position_size_pct / 100)

    def should_close(self) -> str | None:
        """Returns 'WIN' | 'LOSS' | None based on SL/TP levels."""
        if self.current_price <= 0:
            return None
        if self.signal == "BUY":
            if self.current_price >= self.take_profit:
                return "WIN"
            if self.current_price <= self.stop_loss:
                return "LOSS"
        else:  # SELL
            if self.current_price <= self.take_profit:
                return "WIN"
            if self.current_price >= self.stop_loss:
                return "LOSS"
        return None


class PortfolioTracker:
    """
    Tracks the full demo portfolio state and computes performance metrics.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.open_positions: dict[str, Position] = {}   # signal_id → Position
        self.daily_returns: list[float] = []
        self.equity_curve: list[float] = [initial_capital]
        self._today_start_equity = initial_capital
        self._last_reset_date = date.today()
        self._closed_pnl_history: list[float] = []     # net PnL % of closed trades

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(self, signal) -> None:
        """Register a new open position from a TradingSignal."""
        pos = Position(
            signal_id=signal.signal_id,
            asset=signal.asset,
            asset_class=signal.asset_class,
            signal=signal.signal,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size_pct=signal.position_size_pct,
            slippage_cost_pct=signal.slippage_cost_pct,
            current_price=signal.entry_price,
        )
        self.open_positions[signal.signal_id] = pos
        self._append_equity_snapshot()
        logger.info("Position opened: %s %s @ %.4f", signal.signal, signal.asset, signal.entry_price)

    def close_position_at_market(self, signal_id: str, exit_price: float) -> bool:
        """
        Force-close an open position at exit_price (e.g. webhook CLOSE).
        Updates equity like an immediate SL/TP hit; returns False if not found.
        """
        try:
            self._sync_day_boundary()
            pos = self.open_positions.get(signal_id)
            if pos is None:
                return False
            pos.current_price = float(exit_price)
            trade_pnl = pos.net_pnl_pct
            self._record_closed(trade_pnl)
            self._update_equity(pos.weighted_net_pnl_pct)
            del self.open_positions[signal_id]
            logger.info(
                "Position market-closed: %s %s @ %.4f | PnL=%.2f%%",
                pos.signal,
                pos.asset,
                float(exit_price),
                trade_pnl,
            )
            self._append_equity_snapshot()
            return True
        except Exception as exc:
            logger.warning("close_position_at_market failed for %s: %s", signal_id, exc)
            return False

    def update_prices(self, price_map: dict[str, float]) -> list[dict]:
        """
        Update current prices for all open positions.
        Returns list of auto-close events {signal_id, outcome, pnl_pct, close_price}.

        price_map: {asset_symbol: current_price}
        """
        self._sync_day_boundary()
        close_events = []
        for sig_id, pos in list(self.open_positions.items()):
            if pos.asset in price_map:
                pos.current_price = price_map[pos.asset]
                outcome = pos.should_close()
                if outcome:
                    trade_pnl = pos.net_pnl_pct
                    close_events.append({
                        "signal_id": sig_id,
                        "outcome": outcome,
                        "close_price": pos.current_price,
                        "pnl_pct": round(trade_pnl, 4),
                    })
                    self._record_closed(trade_pnl)
                    self._update_equity(pos.weighted_net_pnl_pct)
                    del self.open_positions[sig_id]
                    logger.info(
                        "Position closed: %s %s | outcome=%s | PnL=%.2f%%",
                        pos.signal, pos.asset, outcome, trade_pnl,
                    )
        self._append_equity_snapshot()
        return close_events

    def _record_closed(self, pnl_pct: float) -> None:
        self._closed_pnl_history.append(pnl_pct)

    def _append_equity_snapshot(self) -> None:
        mtm = round(self.mark_to_market_equity, 6)
        if not self.equity_curve or abs(self.equity_curve[-1] - mtm) > 1e-9:
            self.equity_curve.append(mtm)

    def _sync_day_boundary(self) -> None:
        today = date.today()
        if today != self._last_reset_date:
            daily_ret = (
                self.mark_to_market_equity - self._today_start_equity
            ) / max(self._today_start_equity, 1e-9)
            self.daily_returns.append(daily_ret)
            self._today_start_equity = self.mark_to_market_equity
            self._last_reset_date = today
            self._append_equity_snapshot()

    def _update_equity(self, weighted_pnl_pct: float) -> None:
        self.equity *= (1 + weighted_pnl_pct / 100)

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    @property
    def daily_pnl_pct(self) -> float:
        """Today's PnL as % of start-of-day equity."""
        self._sync_day_boundary()
        return (
            (self.mark_to_market_equity - self._today_start_equity)
            / max(self._today_start_equity, 1)
            * 100
        )

    @property
    def total_pnl_pct(self) -> float:
        return (self.mark_to_market_equity - self.initial_capital) / self.initial_capital * 100

    @property
    def unrealized_pnl_pct(self) -> float:
        return float(sum(pos.weighted_net_pnl_pct for pos in self.open_positions.values()))

    @property
    def mark_to_market_equity(self) -> float:
        return self.equity * (1 + self.unrealized_pnl_pct / 100)

    @property
    def current_drawdown_pct(self) -> float:
        curve = pd.Series(self.equity_curve + [self.mark_to_market_equity]).dropna()
        if curve.empty:
            return 0.0
        running_peak = curve.cummax()
        drawdown = (curve - running_peak) / running_peak
        return float(drawdown.iloc[-1] * 100)

    def get_metrics(self) -> dict:
        """Compute and return all performance metrics."""
        self._sync_day_boundary()
        rets = pd.Series(self.daily_returns)
        closed = pd.Series(self._closed_pnl_history)

        metrics = {
            "equity": round(self.mark_to_market_equity, 2),
            "equity_curve": [round(v, 2) for v in self.equity_curve + [self.mark_to_market_equity]],
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "total_return_pct": round(self.total_pnl_pct, 4),
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "open_positions": len(self.open_positions),
            "closed_trades": len(closed),
            "sharpe": 0.0,
            "sharpe_ratio": 0.0,
            "sortino": 0.0,
            "sortino_ratio": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "current_drawdown_pct": round(self.current_drawdown_pct, 4),
            "var_95": 0.0,
            "cvar_95": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

        if len(rets) >= 5:
            metrics["sharpe"] = round(calculate_sharpe_ratio(rets), 4)
            metrics["sortino"] = round(calculate_sortino_ratio(rets), 4)
            metrics["calmar"] = round(calculate_calmar_ratio(rets), 4)
            # Build equity series for drawdown
            equity_curve = pd.Series(self.initial_capital) * (1 + rets).cumprod()
            metrics["max_drawdown"] = round(calculate_max_drawdown(equity_curve) * 100, 4)
            metrics["max_drawdown_pct"] = metrics["max_drawdown"]
            metrics["var_95"] = round(calculate_var(rets) * 100, 4)
            metrics["cvar_95"] = round(calculate_cvar(rets) * 100, 4)
            metrics["sharpe_ratio"] = metrics["sharpe"]
            metrics["sortino_ratio"] = metrics["sortino"]

        if metrics["max_drawdown_pct"] == 0.0 and len(metrics["equity_curve"]) >= 2:
            curve = pd.Series(metrics["equity_curve"])
            metrics["max_drawdown_pct"] = round(calculate_max_drawdown(curve) * 100, 4)
            metrics["max_drawdown"] = metrics["max_drawdown_pct"]

        if len(closed) >= 1:
            wins = int((closed > 0).sum())
            metrics["win_rate"] = round(wins / len(closed) * 100, 2)
            metrics["profit_factor"] = round(calculate_profit_factor(closed / 100), 4)

        return metrics

    def get_snapshot_dict(self) -> dict:
        """Return a dict suitable for SignalStore.save_snapshot()."""
        m = self.get_metrics()
        return {
            "snapshot_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "total_pnl_pct": m["total_pnl_pct"],
            "sharpe": m["sharpe_ratio"],
            "sortino": m["sortino_ratio"],
            "calmar": m["calmar"],
            "max_drawdown": m["max_drawdown_pct"],
            "open_positions": m["open_positions"],
            "daily_var_95": m["var_95"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
        }

    def get_open_positions_list(self) -> list[dict]:
        return [
            {
                "signal_id": pos.signal_id,
                "asset": pos.asset,
                "asset_class": pos.asset_class,
                "signal": pos.signal,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "position_size_pct": pos.position_size_pct,
                "gross_pnl_pct": round(pos.gross_pnl_pct, 4),
                "net_pnl_pct": round(pos.net_pnl_pct, 4),
                "weighted_gross_pnl_pct": round(pos.weighted_gross_pnl_pct, 4),
                "weighted_net_pnl_pct": round(pos.weighted_net_pnl_pct, 4),
                "opened_at": pos.opened_at.isoformat(),
            }
            for pos in self.open_positions.values()
        ]
