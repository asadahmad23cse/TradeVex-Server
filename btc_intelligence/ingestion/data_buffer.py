from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from btc_intelligence.state import RedisStateStore


@dataclass
class LatencyTracker:
    values_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, value_ms: float) -> None:
        if value_ms >= 0:
            self.values_ms.append(value_ms)

    def stats(self) -> dict[str, float]:
        if not self.values_ms:
            return {"avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        arr = np.asarray(self.values_ms, dtype=float)
        return {
            "avg_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(np.max(arr)),
        }


class MarketDataBuffer:
    def __init__(
        self,
        candles_max: dict[str, int],
        trades_max: int,
        signal_history_size: int,
        redis_store: RedisStateStore | None = None,
    ) -> None:
        self.lock = asyncio.Lock()
        self.redis = redis_store
        self._candles_max = dict(candles_max)
        self._trades_max = int(trades_max)
        self._signal_history_size = int(signal_history_size)
        self._force_orders_max = 1500
        self._open_interest_hist_max = 800
        self._funding_hist_max = 300
        self._cross_exchange_cvd_max = 500

        self.candles: dict[str, deque[dict[str, Any]]] = {
            tf: deque(maxlen=max_len) for tf, max_len in candles_max.items()
        }
        self.agg_trades: deque[dict[str, Any]] = deque(maxlen=trades_max)
        self.force_orders: deque[dict[str, Any]] = deque(maxlen=self._force_orders_max)
        self.depth: dict[str, list[list[float]]] = {"bids": [], "asks": []}

        self.multi_exchange: dict[str, dict[str, Any]] = {
            "bybit": {"trades": deque(maxlen=trades_max), "depth": {"bids": [], "asks": []}, "cvd": deque(maxlen=self._cross_exchange_cvd_max)},
            "okx": {"trades": deque(maxlen=trades_max), "depth": {"bids": [], "asks": []}, "cvd": deque(maxlen=self._cross_exchange_cvd_max)},
        }

        self.binance_rest: dict[str, Any] = {
            "funding_rate": 0.0,
            "mark_price": 0.0,
            "open_interest": 0.0,
            "open_interest_prev": 0.0,
            "ticker_24h": {},
        }
        self.open_interest_hist: deque[dict[str, float]] = deque(maxlen=self._open_interest_hist_max)
        self.funding_hist: deque[dict[str, float]] = deque(maxlen=self._funding_hist_max)

        self.coinglass: dict[str, Any] = {}
        self.glassnode: dict[str, Any] = {}
        self.deribit: dict[str, Any] = {}
        self.whale_tracker: dict[str, Any] = {}

        self.news: list[dict[str, Any]] = []
        self.macro: dict[str, Any] = {}

        self.latest_signal: dict[str, Any] = {}
        self.signal_history: deque[dict[str, Any]] = deque(maxlen=signal_history_size)

        self.monitoring_stats: dict[str, Any] = {}
        self.edge_stats: dict[str, Any] = {}
        self.volatility_tradeability: dict[str, Any] = {}

        self.last_ws_update_utc: str = ""
        self.last_15m_close_time: int = 0
        self.last_feature_eval_close_time: int = 0
        self.last_model_metrics: dict[str, Any] = {}
        self.latency = LatencyTracker()

    def _redis_enabled(self) -> bool:
        return bool(self.redis is not None and self.redis.connected)

    def _memory_snapshot(self) -> dict[str, Any]:
        return {
            "candles": {k: list(v) for k, v in self.candles.items()},
            "agg_trades": list(self.agg_trades),
            "force_orders": list(self.force_orders),
            "depth": {"bids": list(self.depth.get("bids", [])), "asks": list(self.depth.get("asks", []))},
            "multi_exchange": {
                k: {
                    "trades": list(v["trades"]),
                    "depth": {"bids": list(v["depth"].get("bids", [])), "asks": list(v["depth"].get("asks", []))},
                    "cvd": list(v["cvd"]),
                }
                for k, v in self.multi_exchange.items()
            },
            "binance_rest": dict(self.binance_rest),
            "open_interest_hist": list(self.open_interest_hist),
            "funding_hist": list(self.funding_hist),
            "coinglass": dict(self.coinglass),
            "glassnode": dict(self.glassnode),
            "deribit": dict(self.deribit),
            "whale_tracker": dict(self.whale_tracker),
            "news": list(self.news),
            "macro": dict(self.macro),
            "latest_signal": dict(self.latest_signal),
            "signal_history": list(self.signal_history),
            "monitoring_stats": dict(self.monitoring_stats),
            "edge_stats": dict(self.edge_stats),
            "volatility_tradeability": dict(self.volatility_tradeability),
            "last_ws_update_utc": self.last_ws_update_utc,
            "last_15m_close_time": self.last_15m_close_time,
            "last_feature_eval_close_time": self.last_feature_eval_close_time,
            "latency": self.latency.stats(),
            "model_metrics": dict(self.last_model_metrics),
        }

    async def add_kline(self, timeframe: str, candle: dict[str, Any]) -> None:
        async with self.lock:
            bucket = self.candles.get(timeframe)
            if bucket is None:
                return
            if bucket and int(bucket[-1]["open_time"]) == int(candle["open_time"]):
                bucket[-1] = candle
            else:
                bucket.append(candle)

            if timeframe == "15m" and candle.get("is_closed"):
                self.last_15m_close_time = int(candle["close_time"])

            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.upsert_latest_json(
                    f"candles:{timeframe}",
                    candle,
                    max_len=self._candles_max.get(timeframe, 500),
                    id_field="open_time",
                )
                if timeframe == "15m" and candle.get("is_closed"):
                    await self.redis.set_json("last_15m_close_time", self.last_15m_close_time)

    async def seed_klines(self, timeframe: str, candles: list[dict[str, Any]]) -> None:
        async with self.lock:
            bucket = self.candles.get(timeframe)
            if bucket is None:
                return
            bucket.clear()
            ordered = sorted(candles, key=lambda x: int(x.get("open_time", 0)))
            for row in ordered:
                bucket.append(row)

            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.clear_key(f"candles:{timeframe}")
                # Push oldest first so final list order in Redis is newest->oldest.
                for row in ordered:
                    await self.redis.lpush_json(
                        f"candles:{timeframe}",
                        row,
                        max_len=self._candles_max.get(timeframe, 500),
                    )

    async def add_agg_trade(self, trade: dict[str, Any]) -> None:
        async with self.lock:
            self.agg_trades.append(trade)
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.lpush_json("agg_trades", trade, max_len=self._trades_max)

    async def add_force_order(self, row: dict[str, Any]) -> None:
        async with self.lock:
            self.force_orders.append(row)
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.lpush_json("force_orders", row, max_len=self._force_orders_max)

    async def set_depth(self, bids: list[list[float]], asks: list[list[float]]) -> None:
        async with self.lock:
            self.depth = {"bids": bids, "asks": asks}
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("depth", self.depth)

    async def set_multi_exchange(
        self,
        exchange: str,
        trades: list[dict[str, Any]],
        depth: dict[str, list[list[float]]],
        cvd_value: float,
    ) -> None:
        async with self.lock:
            slot = self.multi_exchange.get(exchange)
            if slot is None:
                return
            q: deque = slot["trades"]
            for row in trades:
                q.append(row)
            slot["depth"] = depth
            slot["cvd"].append(cvd_value)

            if self._redis_enabled():
                assert self.redis is not None
                for row in trades:
                    await self.redis.lpush_json(
                        f"multi_exchange:{exchange}:trades",
                        row,
                        max_len=self._trades_max,
                    )
                await self.redis.set_json(f"multi_exchange:{exchange}:depth", depth)
                await self.redis.lpush_json(
                    f"multi_exchange:{exchange}:cvd",
                    {"value": float(cvd_value)},
                    max_len=self._cross_exchange_cvd_max,
                )

    async def set_rest_metrics(
        self,
        funding_rate: float,
        mark_price: float,
        open_interest: float,
        ticker_24h: dict[str, Any],
    ) -> None:
        async with self.lock:
            prev = float(self.binance_rest.get("open_interest", 0.0))
            self.binance_rest["open_interest_prev"] = prev
            self.binance_rest["open_interest"] = open_interest
            self.binance_rest["funding_rate"] = funding_rate
            self.binance_rest["mark_price"] = mark_price
            self.binance_rest["ticker_24h"] = ticker_24h

            ts = float(pd.Timestamp.utcnow().timestamp())
            oi_row = {"open_interest": open_interest, "ts": ts}
            fr_row = {"funding_rate": funding_rate, "ts": ts}
            self.open_interest_hist.append(oi_row)
            self.funding_hist.append(fr_row)

            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("binance_rest", self.binance_rest)
                await self.redis.lpush_json("open_interest_hist", oi_row, max_len=self._open_interest_hist_max)
                await self.redis.lpush_json("funding_hist", fr_row, max_len=self._funding_hist_max)

    async def set_coinglass(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.coinglass = payload
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("coinglass", payload)

    async def set_glassnode(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.glassnode = payload
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("glassnode", payload)

    async def set_deribit(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.deribit = payload
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("deribit", payload)

    async def set_whale_tracker(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.whale_tracker = payload
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("whale_tracker", payload)

    async def set_news(self, items: list[dict[str, Any]]) -> None:
        async with self.lock:
            self.news = items
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("news", items)

    async def set_macro(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.macro.update(payload)
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("macro", self.macro)

    async def set_monitoring(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.monitoring_stats = payload
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("monitoring_stats", payload)

    async def set_edge_stats(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.edge_stats = payload
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("edge_stats", payload)

    async def set_volatility_tradeability(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.volatility_tradeability = payload
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("volatility_tradeability", payload)

    async def set_ws_update(self, iso_ts: str) -> None:
        async with self.lock:
            self.last_ws_update_utc = iso_ts
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("last_ws_update_utc", iso_ts)
                await self.redis.set_json("latency", self.latency.stats())

    async def set_latest_signal(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.latest_signal = payload
            self.signal_history.appendleft(payload)
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("latest_signal", payload)
                await self.redis.lpush_json("signal_history", payload, max_len=self._signal_history_size)

    async def set_model_metrics(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.last_model_metrics = payload
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("model_metrics", payload)

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            if not self._redis_enabled():
                return self._memory_snapshot()
            assert self.redis is not None

            # Redis is the canonical state source when enabled.
            candles = {}
            for tf in self._candles_max:
                candles[tf] = await self.redis.lrange_json(
                    f"candles:{tf}",
                    0,
                    self._candles_max[tf] - 1,
                    reverse=True,
                )

            multi_exchange: dict[str, Any] = {}
            for ex in ("bybit", "okx"):
                ex_trades = await self.redis.lrange_json(
                    f"multi_exchange:{ex}:trades",
                    0,
                    self._trades_max - 1,
                    reverse=True,
                )
                ex_depth = await self.redis.get_json(
                    f"multi_exchange:{ex}:depth",
                    {"bids": [], "asks": []},
                )
                ex_cvd_rows = await self.redis.lrange_json(
                    f"multi_exchange:{ex}:cvd",
                    0,
                    self._cross_exchange_cvd_max - 1,
                    reverse=True,
                )
                multi_exchange[ex] = {
                    "trades": ex_trades,
                    "depth": ex_depth,
                    "cvd": [float(x.get("value", 0.0)) for x in ex_cvd_rows if isinstance(x, dict)],
                }

            return {
                "candles": candles,
                "agg_trades": await self.redis.lrange_json("agg_trades", 0, self._trades_max - 1, reverse=True),
                "force_orders": await self.redis.lrange_json("force_orders", 0, self._force_orders_max - 1, reverse=True),
                "depth": await self.redis.get_json("depth", {"bids": [], "asks": []}),
                "multi_exchange": multi_exchange,
                "binance_rest": await self.redis.get_json("binance_rest", dict(self.binance_rest)),
                "open_interest_hist": await self.redis.lrange_json("open_interest_hist", 0, self._open_interest_hist_max - 1, reverse=True),
                "funding_hist": await self.redis.lrange_json("funding_hist", 0, self._funding_hist_max - 1, reverse=True),
                "coinglass": await self.redis.get_json("coinglass", dict(self.coinglass)),
                "glassnode": await self.redis.get_json("glassnode", dict(self.glassnode)),
                "deribit": await self.redis.get_json("deribit", dict(self.deribit)),
                "whale_tracker": await self.redis.get_json("whale_tracker", dict(self.whale_tracker)),
                "news": await self.redis.get_json("news", list(self.news)),
                "macro": await self.redis.get_json("macro", dict(self.macro)),
                "latest_signal": await self.redis.get_json("latest_signal", dict(self.latest_signal)),
                "signal_history": await self.redis.lrange_json("signal_history", 0, self._signal_history_size - 1, reverse=False),
                "monitoring_stats": await self.redis.get_json("monitoring_stats", dict(self.monitoring_stats)),
                "edge_stats": await self.redis.get_json("edge_stats", dict(self.edge_stats)),
                "volatility_tradeability": await self.redis.get_json("volatility_tradeability", dict(self.volatility_tradeability)),
                "last_ws_update_utc": await self.redis.get_json("last_ws_update_utc", self.last_ws_update_utc),
                "last_15m_close_time": int(await self.redis.get_json("last_15m_close_time", self.last_15m_close_time)),
                "last_feature_eval_close_time": int(await self.redis.get_json("last_feature_eval_close_time", self.last_feature_eval_close_time)),
                "latency": await self.redis.get_json("latency", self.latency.stats()),
                "model_metrics": await self.redis.get_json("model_metrics", dict(self.last_model_metrics)),
            }

    async def mark_feature_eval(self, close_time: int) -> None:
        async with self.lock:
            self.last_feature_eval_close_time = close_time
            if self._redis_enabled():
                assert self.redis is not None
                await self.redis.set_json("last_feature_eval_close_time", int(close_time))

    @staticmethod
    def candles_to_df(snapshot: dict[str, Any], timeframe: str) -> pd.DataFrame:
        rows = snapshot.get("candles", {}).get(timeframe, [])
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("time").sort_index()
        return df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )[["Open", "High", "Low", "Close", "Volume"]]

    @staticmethod
    def best_bid_ask(depth: dict[str, list[list[float]]]) -> tuple[float, float, float]:
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])
        if not bids or not asks:
            return 0.0, 0.0, 0.0
        bid = float(bids[0][0])
        ask = float(asks[0][0])
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        return bid, ask, mid
