from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


logger = logging.getLogger(__name__)


class MultiExchangeWebSocketManager:
    """Collect public trade/depth from Bybit + OKX to measure cross-exchange flow disagreement."""

    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._bybit_loop(), name='bybit_ws_loop'),
            asyncio.create_task(self._okx_loop(), name='okx_ws_loop'),
        ]

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _bybit_loop(self) -> None:
        retry = 1.0
        while self._running:
            try:
                async with websockets.connect(settings.bybit_ws_url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps({'op': 'subscribe', 'args': ['publicTrade.BTCUSDT', 'orderbook.50.BTCUSDT']}))
                    retry = 1.0
                    trades_batch: list[dict[str, Any]] = []
                    depth = {'bids': [], 'asks': []}
                    cvd_val = 0.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        topic = str(msg.get('topic', ''))
                        data = msg.get('data', [])

                        if topic.startswith('publicTrade') and isinstance(data, list):
                            for row in data:
                                px = float(row.get('p', 0.0))
                                qty = float(row.get('v', 0.0))
                                side = str(row.get('S', '')).lower()
                                if px <= 0 or qty <= 0:
                                    continue
                                is_buy = side == 'buy'
                                cvd_val += qty if is_buy else -qty
                                trades_batch.append(
                                    {
                                        'time': int(row.get('T', 0)),
                                        'price': px,
                                        'qty': qty,
                                        'maker': not is_buy,
                                        'notional': px * qty,
                                    }
                                )

                        if topic.startswith('orderbook') and isinstance(data, dict):
                            bids = [[float(p), float(q)] for p, q in data.get('b', [])[:20]]
                            asks = [[float(p), float(q)] for p, q in data.get('a', [])[:20]]
                            if bids and asks:
                                depth = {'bids': bids, 'asks': asks}

                        if trades_batch:
                            await self.buffer.set_multi_exchange('bybit', trades_batch, depth, cvd_val)
                            trades_batch = []
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('Bybit WS error: %s', exc)
                await asyncio.sleep(retry)
                retry = min(retry * 2, 30.0)

    async def _okx_loop(self) -> None:
        retry = 1.0
        while self._running:
            try:
                async with websockets.connect(settings.okx_ws_url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                'op': 'subscribe',
                                'args': [
                                    {'channel': 'trades', 'instId': 'BTC-USDT-SWAP'},
                                    {'channel': 'books5', 'instId': 'BTC-USDT-SWAP'},
                                ],
                            }
                        )
                    )
                    retry = 1.0
                    trades_batch: list[dict[str, Any]] = []
                    depth = {'bids': [], 'asks': []}
                    cvd_val = 0.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        arg = msg.get('arg', {})
                        channel = str(arg.get('channel', ''))
                        data = msg.get('data', [])
                        if not isinstance(data, list) or not data:
                            continue

                        if channel == 'trades':
                            for row in data:
                                px = float(row.get('px', 0.0))
                                qty = float(row.get('sz', 0.0))
                                side = str(row.get('side', '')).lower()
                                if px <= 0 or qty <= 0:
                                    continue
                                is_buy = side == 'buy'
                                cvd_val += qty if is_buy else -qty
                                trades_batch.append(
                                    {
                                        'time': int(row.get('ts', 0)),
                                        'price': px,
                                        'qty': qty,
                                        'maker': not is_buy,
                                        'notional': px * qty,
                                    }
                                )

                        if channel == 'books5':
                            row = data[0]
                            bids = [[float(p), float(q)] for p, q, *_ in row.get('bids', [])[:20]]
                            asks = [[float(p), float(q)] for p, q, *_ in row.get('asks', [])[:20]]
                            if bids and asks:
                                depth = {'bids': bids, 'asks': asks}

                        if trades_batch:
                            await self.buffer.set_multi_exchange('okx', trades_batch, depth, cvd_val)
                            trades_batch = []
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('OKX WS error: %s', exc)
                await asyncio.sleep(retry)
                retry = min(retry * 2, 30.0)
