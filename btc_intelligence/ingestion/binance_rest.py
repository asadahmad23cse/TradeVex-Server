from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


logger = logging.getLogger(__name__)


class BinanceRestPoller:
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name='binance_rest_poller')

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            while self._running:
                try:
                    premium = await self._get_premium_index(client)
                    oi = await self._get_open_interest(client)
                    ticker = await self._get_ticker_24h(client)
                    await self.buffer.set_rest_metrics(
                        funding_rate=float(premium.get('lastFundingRate', 0.0)),
                        mark_price=float(premium.get('markPrice', 0.0)),
                        open_interest=float(oi.get('openInterest', 0.0)),
                        ticker_24h=ticker,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning('Binance REST polling failed: %s', exc)
                await asyncio.sleep(settings.binance_rest_poll_sec)

    async def _get_premium_index(self, client: httpx.AsyncClient) -> dict[str, Any]:
        url = f'{settings.binance_rest_base}/fapi/v1/premiumIndex'
        resp = await client.get(url, params={'symbol': 'BTCUSDT'})
        resp.raise_for_status()
        return resp.json()

    async def _get_open_interest(self, client: httpx.AsyncClient) -> dict[str, Any]:
        url = f'{settings.binance_rest_base}/fapi/v1/openInterest'
        resp = await client.get(url, params={'symbol': 'BTCUSDT'})
        resp.raise_for_status()
        return resp.json()

    async def _get_ticker_24h(self, client: httpx.AsyncClient) -> dict[str, Any]:
        url = f'{settings.binance_rest_base}/fapi/v1/ticker/24hr'
        resp = await client.get(url, params={'symbol': 'BTCUSDT'})
        resp.raise_for_status()
        return resp.json()
