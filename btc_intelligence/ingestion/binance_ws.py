from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import websockets

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer
from btc_intelligence.utils.validator import utc_now_iso, validate_candle


logger = logging.getLogger(__name__)


class BinanceWebSocketManager:
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        await self._bootstrap_klines()
        self._running = True
        self._task = asyncio.create_task(self._run(), name='binance_ws_manager')

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _bootstrap_klines(self) -> None:
        tf_limits = {'1m': 500, '5m': 300, '15m': 200, '1h': 100, '4h': 50}
        async with httpx.AsyncClient(timeout=20.0) as client:
            for tf, limit in tf_limits.items():
                try:
                    resp = await client.get(
                        f'{settings.binance_rest_base}/fapi/v1/klines',
                        params={'symbol': 'BTCUSDT', 'interval': tf, 'limit': limit},
                    )
                    resp.raise_for_status()
                    rows = resp.json()
                    candles: list[dict[str, Any]] = []
                    for row in rows:
                        candle = {
                            'open_time': int(row[0]),
                            'open': float(row[1]),
                            'high': float(row[2]),
                            'low': float(row[3]),
                            'close': float(row[4]),
                            'volume': float(row[5]),
                            'close_time': int(row[6]),
                            'is_closed': True,
                        }
                        if validate_candle(candle, settings.max_price_spike_pct):
                            candles.append(candle)
                    await self.buffer.seed_klines(tf, candles)
                    logger.info('Seeded %s candles for %s', len(candles), tf)
                except Exception as exc:
                    logger.warning('Failed to seed timeframe %s: %s', tf, exc)

    async def _run(self) -> None:
        retries = 0
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(settings.binance_ws_url, ping_interval=20, ping_timeout=20) as ws:
                    logger.info('Connected to Binance futures stream')
                    retries = 0
                    backoff = 1.0
                    async for raw in ws:
                        recv_ms = int(time.time() * 1000)
                        msg = json.loads(raw)
                        await self._handle_message(msg, recv_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retries += 1
                logger.warning('Binance WS disconnected (retry %s): %s', retries, exc)
                if retries > settings.max_ws_retries:
                    retries = 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 20.0)

    async def _handle_message(self, msg: dict[str, Any], recv_ms: int) -> None:
        stream = str(msg.get('stream', ''))
        data = msg.get('data', {})
        event_ms = int(data.get('E', recv_ms))
        self.buffer.latency.add(recv_ms - event_ms)
        await self.buffer.set_ws_update(utc_now_iso())

        if '@kline_' in stream:
            k = data.get('k', {})
            timeframe = str(k.get('i', ''))
            candle = {
                'open_time': int(k.get('t', 0)),
                'open': float(k.get('o', 0.0)),
                'high': float(k.get('h', 0.0)),
                'low': float(k.get('l', 0.0)),
                'close': float(k.get('c', 0.0)),
                'volume': float(k.get('v', 0.0)),
                'close_time': int(k.get('T', 0)),
                'is_closed': bool(k.get('x', False)),
            }
            if validate_candle(candle, settings.max_price_spike_pct):
                await self.buffer.add_kline(timeframe, candle)
            return

        if stream.endswith('@aggTrade'):
            price = float(data.get('p', 0.0))
            qty = float(data.get('q', 0.0))
            if price <= 0 or qty <= 0:
                return
            trade = {
                'time': int(data.get('T', event_ms)),
                'price': price,
                'qty': qty,
                'maker': bool(data.get('m', False)),
                'notional': price * qty,
            }
            await self.buffer.add_agg_trade(trade)
            return

        if '@depth20' in stream:
            bids = [[float(p), float(q)] for p, q in data.get('b', [])]
            asks = [[float(p), float(q)] for p, q in data.get('a', [])]
            if bids and asks:
                await self.buffer.set_depth(bids=bids, asks=asks)
            return

        if stream.endswith('@forceOrder'):
            order = data.get('o', {})
            side = str(order.get('S', ''))
            price = float(order.get('ap', 0.0))
            qty = float(order.get('q', 0.0))
            if price <= 0 or qty <= 0:
                return
            forced = {
                'time': int(data.get('E', event_ms)),
                'side': side,
                'price': price,
                'qty': qty,
                'notional': price * qty,
            }
            await self.buffer.add_force_order(forced)
