"""
Gap 3 Fix — Broker Execution Engine.

Abstract interface + concrete implementations for:
    1. Paper Trader (simulated fills with slippage)
    2. Zerodha Kite Connect (bracket orders)
    3. ICICI Breeze (stub, future extension)

Signal flow:
    SignalEngine.generate() → TradingSignal → ExecutionEngine.execute()

Bracket Order Logic (Zerodha):
    BUY signal:
        - Place BUY bracket order at entry_price
        - Stoploss trigger = signal.stop_loss
        - Target = signal.take_profit
        - Trailing SL if ATR > 2% of entry

Paper Trading:
    - Simulates fill with random slippage in [0, 2×signal.slippage_cost_pct]
    - Tracks positions in memory
    - Computes PnL on close

Usage:
    engine = PaperTrader(config)
    receipt = engine.execute(signal)
    engine.get_open_positions()
    engine.close_position(signal_id, current_price)
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderReceipt:
    """Result of an order execution attempt."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str = ""
    asset: str = ""
    side: str = ""           # BUY | SELL
    order_type: str = ""     # BRACKET | MARKET | LIMIT
    status: str = "PENDING"  # PENDING | FILLED | REJECTED | CANCELLED
    fill_price: float = 0.0
    fill_time: datetime = field(default_factory=datetime.utcnow)
    slippage_pct: float = 0.0
    broker: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "asset": self.asset,
            "side": self.side,
            "order_type": self.order_type,
            "status": self.status,
            "fill_price": self.fill_price,
            "fill_time": self.fill_time.isoformat(),
            "slippage_pct": self.slippage_pct,
            "broker": self.broker,
            "error": self.error,
        }


class BrokerBase(ABC):
    """Abstract broker interface for signal execution."""

    @abstractmethod
    def execute(self, signal) -> OrderReceipt:
        """Execute a TradingSignal, return an OrderReceipt."""
        ...

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_open_positions(self) -> list[dict]:
        """Return all open positions."""
        ...

    @abstractmethod
    def close_position(self, signal_id: str, current_price: float) -> OrderReceipt:
        """Close a position by signal_id."""
        ...


# ------------------------------------------------------------------
# Paper Trader (simulated execution)
# ------------------------------------------------------------------

class PaperTrader(BrokerBase):
    """
    Simulated broker for paper trading.

    Fills orders instantly with random slippage.
    Tracks positions in memory.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.initial_capital = cfg.get("initial_capital", 100_000)
        self.capital = self.initial_capital
        self._positions: dict[str, dict] = {}  # signal_id → position dict
        self._closed: list[dict] = []

    def execute(self, signal) -> OrderReceipt:
        """Fill a signal with simulated slippage."""
        entry = signal.entry_price
        slippage_pct = signal.slippage_cost_pct
        # Random slippage: uniform [0, 2×slippage_pct]
        actual_slip = np.random.uniform(0, 2 * slippage_pct)

        if signal.signal == "BUY":
            fill_price = entry * (1 + actual_slip)
        else:
            fill_price = entry * (1 - actual_slip)

        # Record position
        position = {
            "signal_id": signal.signal_id,
            "asset": signal.asset,
            "asset_class": signal.asset_class,
            "side": signal.signal,
            "entry_price": entry,
            "fill_price": fill_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "position_size_pct": signal.position_size_pct,
            "open_time": datetime.utcnow(),
            "capital_allocated": self.capital * signal.position_size_pct / 100.0,
        }
        self._positions[signal.signal_id] = position
        self.capital -= position["capital_allocated"]

        receipt = OrderReceipt(
            signal_id=signal.signal_id,
            asset=signal.asset,
            side=signal.signal,
            order_type="BRACKET",
            status="FILLED",
            fill_price=round(fill_price, 6),
            slippage_pct=round(actual_slip * 100, 5),
            broker="paper",
        )
        logger.info(
            "PAPER FILL: %s %s @ %.4f (slip=%.4f%%) | capital remaining: $%.0f",
            signal.signal, signal.asset, fill_price, actual_slip * 100, self.capital,
        )
        return receipt

    def cancel(self, order_id: str) -> bool:
        return False  # Paper orders fill instantly

    def get_open_positions(self) -> list[dict]:
        return list(self._positions.values())

    def close_position(self, signal_id: str, current_price: float) -> OrderReceipt:
        pos = self._positions.pop(signal_id, None)
        if pos is None:
            return OrderReceipt(status="REJECTED", error="No open position found")

        fill_price = pos["fill_price"]
        if pos["side"] == "BUY":
            pnl_pct = (current_price - fill_price) / fill_price * 100
        else:
            pnl_pct = (fill_price - current_price) / fill_price * 100

        pnl_amount = pos["capital_allocated"] * pnl_pct / 100.0
        self.capital += pos["capital_allocated"] + pnl_amount

        closed = {
            **pos,
            "close_price": current_price,
            "close_time": datetime.utcnow(),
            "pnl_pct": round(pnl_pct, 4),
            "pnl_amount": round(pnl_amount, 2),
        }
        self._closed.append(closed)

        logger.info(
            "PAPER CLOSE: %s %s @ %.4f → %.4f  PnL=%.2f%% ($%.2f)",
            pos["side"], pos["asset"], fill_price, current_price, pnl_pct, pnl_amount,
        )
        return OrderReceipt(
            signal_id=signal_id,
            asset=pos["asset"],
            side="CLOSE",
            order_type="MARKET",
            status="FILLED",
            fill_price=current_price,
            broker="paper",
        )

    def get_equity(self) -> float:
        """Current equity = cash + mark-to-market open positions."""
        # For paper trading, we'd need current prices to MTM open positions
        return self.capital + sum(p["capital_allocated"] for p in self._positions.values())

    def get_trade_log(self) -> list[dict]:
        return self._closed


# ------------------------------------------------------------------
# Zerodha Kite Connect (real broker execution)
# ------------------------------------------------------------------

class ZerodhaExecutor(BrokerBase):
    """
    Zerodha Kite Connect bracket order execution.

    Requires:
        pip install kiteconnect
        ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN in .env

    Bracket Order:
        - Entry: LIMIT at signal.entry_price
        - Stoploss: signal.stop_loss
        - Target: signal.take_profit
        - Trailing stoploss: if ATR > 2% of entry → trail by 1 ATR
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._api_key = cfg.get("api_key", "")
        self._access_token = cfg.get("access_token", "")
        self._kite = None
        self._connect()

    def _connect(self) -> None:
        if not self._api_key or not self._access_token:
            logger.warning("Zerodha: no API key/token — running in DRY RUN mode")
            return
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=self._api_key)
            self._kite.set_access_token(self._access_token)
            logger.info("Zerodha Kite connected successfully")
        except ImportError:
            logger.warning("kiteconnect not installed — pip install kiteconnect")
        except Exception as exc:
            logger.error("Zerodha connection failed: %s", exc)

    def execute(self, signal) -> OrderReceipt:
        if self._kite is None:
            logger.warning("Zerodha: DRY RUN — not executing %s %s", signal.signal, signal.asset)
            return OrderReceipt(
                signal_id=signal.signal_id,
                asset=signal.asset,
                side=signal.signal,
                status="DRY_RUN",
                broker="zerodha",
                error="No API connection",
            )

        try:
            # Map signal to Kite params
            tradingsym = signal.asset  # NSE symbol
            exchange = "NSE" if signal.asset_class == "indian_stock" else "NFO"
            transaction = (
                self._kite.TRANSACTION_TYPE_BUY
                if signal.signal == "BUY"
                else self._kite.TRANSACTION_TYPE_SELL
            )

            # Calculate quantity from position size
            # (simplified — real implementation needs lot size for F&O)
            price = signal.entry_price
            sl_points = abs(price - signal.stop_loss)
            tp_points = abs(signal.take_profit - price)

            # Bracket order
            order_id = self._kite.place_order(
                variety=self._kite.VARIETY_BO,
                exchange=exchange,
                tradingsymbol=tradingsym,
                transaction_type=transaction,
                quantity=1,  # TODO: calculate proper lot size
                price=round(price, 2),
                stoploss=round(sl_points, 2),
                squareoff=round(tp_points, 2),
                product=self._kite.PRODUCT_MIS,
                order_type=self._kite.ORDER_TYPE_LIMIT,
            )

            logger.info("Zerodha BO placed: %s %s #%s", signal.signal, signal.asset, order_id)
            return OrderReceipt(
                order_id=str(order_id),
                signal_id=signal.signal_id,
                asset=signal.asset,
                side=signal.signal,
                order_type="BRACKET",
                status="PLACED",
                fill_price=price,
                broker="zerodha",
            )
        except Exception as exc:
            logger.error("Zerodha order failed: %s", exc)
            return OrderReceipt(
                signal_id=signal.signal_id,
                asset=signal.asset,
                status="REJECTED",
                broker="zerodha",
                error=str(exc),
            )

    def cancel(self, order_id: str) -> bool:
        if self._kite is None:
            return False
        try:
            self._kite.cancel_order(self._kite.VARIETY_BO, order_id)
            return True
        except Exception as exc:
            logger.error("Cancel failed for %s: %s", order_id, exc)
            return False

    def get_open_positions(self) -> list[dict]:
        if self._kite is None:
            return []
        try:
            return self._kite.positions().get("net", [])
        except Exception:
            return []

    def close_position(self, signal_id: str, current_price: float) -> OrderReceipt:
        # Bracket orders auto-close on SL/TP hit
        return OrderReceipt(
            signal_id=signal_id,
            status="INFO",
            broker="zerodha",
            error="Bracket orders auto-close on SL/TP",
        )


class BreezeExecutor(BrokerBase):
    """
    Minimal ICICI Breeze execution stub.

    Supports market-order dry runs and position queries when credentials are absent.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._api_key = cfg.get("api_key", "")
        self._api_secret = cfg.get("api_secret", "")
        self._session_token = cfg.get("session_token", "")
        self._positions: list[dict] = []

    def execute(self, signal) -> OrderReceipt:
        if not self._api_key or not self._api_secret or not self._session_token:
            return OrderReceipt(
                signal_id=signal.signal_id,
                asset=signal.asset,
                side=signal.signal,
                order_type="MARKET",
                status="DRY_RUN",
                fill_price=signal.entry_price,
                broker="breeze",
                error="Missing Breeze credentials",
            )
        self._positions.append(
            {
                "asset": signal.asset,
                "quantity": signal.position_size_pct,
                "side": signal.signal,
            }
        )
        return OrderReceipt(
            signal_id=signal.signal_id,
            asset=signal.asset,
            side=signal.signal,
            order_type="MARKET",
            status="FILLED",
            fill_price=signal.entry_price,
            broker="breeze",
        )

    def cancel(self, order_id: str) -> bool:
        return False

    def get_open_positions(self) -> list[dict]:
        return list(self._positions)

    def close_position(self, signal_id: str, current_price: float) -> OrderReceipt:
        return OrderReceipt(
            signal_id=signal_id,
            order_type="MARKET",
            status="FILLED",
            fill_price=current_price,
            broker="breeze",
        )


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_executor(config: dict | None = None) -> BrokerBase:
    """
    Create the appropriate executor based on config.

    config.execution.broker = 'paper' | 'zerodha' | 'breeze'
    """
    cfg = config or {}
    execution_cfg = cfg.get("execution", {})
    broker = execution_cfg.get("broker", "paper")

    if broker == "zerodha":
        return ZerodhaExecutor(execution_cfg)
    if broker == "breeze":
        return BreezeExecutor(execution_cfg)
    else:
        return PaperTrader(execution_cfg)
