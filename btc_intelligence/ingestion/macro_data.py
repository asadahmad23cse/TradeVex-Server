from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import httpx

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


logger = logging.getLogger(__name__)


class MacroDataPoller:
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name='macro_data_poller')

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
                    await self.buffer.set_macro(payload)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning('Macro polling failed: %s', exc)
                await asyncio.sleep(settings.macro_poll_sec)

    async def _fetch(self, client: httpx.AsyncClient) -> dict:
        fng = await self._fear_greed(client)
        dxy = await self._fred_series(client, 'DTWEXBGS', 60)
        vix = await self._fred_series(client, 'VIXCLS', 60)
        us10y = await self._fred_series(client, 'DGS10', 60)
        spx = await self._fred_series(client, 'SP500', 60)
        gold = await self._fred_series(client, 'GOLDAMGBD228NLBM', 60)

        dxy_bias = self._trend_label(dxy)
        spx_dir = self._trend_label(spx)
        gold_dir = self._trend_label(gold)
        yield_trend = self._trend_label(us10y)

        vix_level = float(vix[-1]) if vix else 20.0
        if vix_level < 15:
            vix_regime = 'LOW_VOL'
        elif vix_level <= 25:
            vix_regime = 'NORMAL'
        elif vix_level <= 35:
            vix_regime = 'HIGH_VOL'
        else:
            vix_regime = 'PANIC'

        spx_btc_corr = 0.5  # static fallback; live rolling corr computed in AppRuntime
        spx_daily_closes = [float(x) for x in spx if x > 0]

        return {
            'fear_greed_score': fng,
            'dxy_bias': 'risk_on' if dxy_bias == 'down' else 'risk_off' if dxy_bias == 'up' else 'neutral',
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'us10y_yield_trend': yield_trend,
            'spx_intraday_direction': spx_dir,
            'spx_btc_correlation': spx_btc_corr,
            'spx_daily_closes': spx_daily_closes,
            'gold_direction': gold_dir,
            'eth_btc_ratio_trend': 'flat',
            'btc_dominance_trend': 'flat',
            'source': 'macro_poll',
        }

    async def _fear_greed(self, client: httpx.AsyncClient) -> int:
        try:
            resp = await client.get(settings.fng_api, params={'limit': 1})
            resp.raise_for_status()
            rows = resp.json().get('data', [])
            if rows:
                return int(rows[0].get('value', 50))
        except Exception:
            return 50
        return 50

    async def _fred_series(self, client: httpx.AsyncClient, series_id: str, days: int) -> list[float]:
        if not settings.fred_api_key:
            return []
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        resp = await client.get(
            settings.fred_api,
            params={
                'series_id': series_id,
                'api_key': settings.fred_api_key,
                'file_type': 'json',
                'observation_start': start.isoformat(),
                'observation_end': end.isoformat(),
            },
        )
        resp.raise_for_status()
        obs = resp.json().get('observations', [])
        vals = []
        for row in obs:
            v = row.get('value')
            if v in (None, '.'):
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        return vals

    @staticmethod
    def _trend_label(values: list[float]) -> str:
        if len(values) < 10:
            return 'flat'
        recent = sum(values[-5:]) / 5.0
        prior = sum(values[-10:-5]) / 5.0
        if recent > prior:
            return 'up'
        if recent < prior:
            return 'down'
        return 'flat'
