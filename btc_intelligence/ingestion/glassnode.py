from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


logger = logging.getLogger(__name__)


class GlassnodePoller:
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name='glassnode_poller')

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=20.0) as client:
            while self._running:
                try:
                    payload = await self._fetch(client)
                    await self.buffer.set_glassnode(payload)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning('Glassnode polling failed: %s', exc)
                await asyncio.sleep(settings.glassnode_poll_sec)

    async def _fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not settings.glassnode_api_key:
            return {
                'exchange_netflow': 0.0,
                'sopr': 1.0,
                'lth_supply_change': 0.0,
                'whale_wallet_count': 0.0,
                'source': 'fallback_no_api_key',
            }

        params = {'a': 'BTC', 'api_key': settings.glassnode_api_key}
        netflow = await self._get_latest(client, '/transactions/transfers_volume_exchanges_net', params)
        sopr = await self._get_latest(client, '/indicators/sopr_adjusted', params)
        lth = await self._get_latest(client, '/supply/current_lth_supply', params)
        whales = await self._get_latest(client, '/addresses/count', {**params, 'min_balance': 1000})

        return {
            'exchange_netflow': netflow,
            'sopr': sopr,
            'lth_supply_change': lth,
            'whale_wallet_count': whales,
            'source': 'glassnode',
        }

    async def _get_latest(self, client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> float:
        url = f'{settings.glassnode_base}{path}'
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list) and rows:
            row = rows[-1]
            return float(row.get('v', 0.0))
        if isinstance(rows, dict):
            return float(rows.get('v', 0.0))
        return 0.0
