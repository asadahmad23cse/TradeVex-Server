from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


logger = logging.getLogger(__name__)


class CoinglassPoller:
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name='coinglass_poller')

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        headers = {'CG-API-KEY': settings.coinglass_api_key} if settings.coinglass_api_key else {}
        async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
            while self._running:
                try:
                    payload = await self._fetch(client)
                    await self.buffer.set_coinglass(payload)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning('Coinglass polling failed: %s', exc)
                await asyncio.sleep(settings.coinglass_poll_sec)

    async def _fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not settings.coinglass_api_key:
            return {
                'liquidation_heatmap_above': 0.0,
                'liquidation_heatmap_below': 0.0,
                'long_short_ratio': 0.5,
                'oi_change_exchanges_pct': 0.0,
                'source': 'fallback_no_api_key',
            }

        ls_resp = await client.get(f'{settings.coinglass_base}/futures/longShortChart', params={'symbol': 'BTC'})
        heat_resp = await client.get(f'{settings.coinglass_base}/futures/liquidationHeatMap', params={'symbol': 'BTC'})
        oi_resp = await client.get(f'{settings.coinglass_base}/futures/openInterest/aggregated', params={'symbol': 'BTC'})

        ls_resp.raise_for_status()
        heat_resp.raise_for_status()
        oi_resp.raise_for_status()

        ls_json = ls_resp.json()
        heat_json = heat_resp.json()
        oi_json = oi_resp.json()

        ls_ratio = self._extract_ls_ratio(ls_json)
        heat_above, heat_below = self._extract_heatmap(heat_json)
        oi_change = self._extract_oi_change(oi_json)

        return {
            'liquidation_heatmap_above': heat_above,
            'liquidation_heatmap_below': heat_below,
            'long_short_ratio': ls_ratio,
            'oi_change_exchanges_pct': oi_change,
            'source': 'coinglass',
        }

    @staticmethod
    def _extract_ls_ratio(payload: dict[str, Any]) -> float:
        rows = payload.get('data', []) if isinstance(payload.get('data', []), list) else []
        if not rows:
            return 0.5
        last = rows[-1]
        for key in ('longShortRatio', 'lsRatio', 'ratio'):
            if key in last:
                value = float(last[key])
                return value / (1.0 + value) if value > 1 else max(0.0, min(1.0, value))
        return 0.5

    @staticmethod
    def _extract_heatmap(payload: dict[str, Any]) -> tuple[float, float]:
        data = payload.get('data', {})
        above = float(data.get('nearestShortLiq', 0.0) or data.get('above', 0.0) or 0.0)
        below = float(data.get('nearestLongLiq', 0.0) or data.get('below', 0.0) or 0.0)
        return above, below

    @staticmethod
    def _extract_oi_change(payload: dict[str, Any]) -> float:
        rows = payload.get('data', []) if isinstance(payload.get('data', []), list) else []
        if len(rows) < 2:
            return 0.0
        a = float(rows[-2].get('openInterest', rows[-2].get('value', 0.0)))
        b = float(rows[-1].get('openInterest', rows[-1].get('value', 0.0)))
        return ((b - a) / a * 100.0) if a != 0 else 0.0
