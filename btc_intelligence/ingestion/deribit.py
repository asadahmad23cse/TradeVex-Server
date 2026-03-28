from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


logger = logging.getLogger(__name__)


class DeribitPoller:
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name='deribit_poller')

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
                    await self.buffer.set_deribit(payload)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning('Deribit polling failed: %s', exc)
                await asyncio.sleep(settings.deribit_poll_sec)

    async def _fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        # Public endpoints: no auth needed for top-level options stats.
        summary_resp = await client.get(f'{settings.deribit_base}/public/get_book_summary_by_currency', params={'currency': 'BTC', 'kind': 'option'})
        summary_resp.raise_for_status()
        rows = summary_resp.json().get('result', [])

        index_resp = await client.get(f'{settings.deribit_base}/public/get_index_price', params={'index_name': 'btc_usd'})
        index_resp.raise_for_status()
        index_price = float(index_resp.json().get('result', {}).get('index_price', 0.0))

        max_pain = self._compute_max_pain(rows)
        put_call = self._put_call_ratio(rows)
        iv_skew = self._iv_skew(rows)
        atm_iv = self._atm_iv(rows, index_price)
        expiry_hours = self._nearest_expiry_hours(rows)

        return {
            'max_pain_level': max_pain,
            'put_call_ratio': put_call,
            'iv_skew': iv_skew,
            'atm_iv': atm_iv,
            'options_expiry_hours': expiry_hours,
            'source': 'deribit',
        }

    @staticmethod
    def _compute_max_pain(rows: list[dict[str, Any]]) -> float:
        strikes: dict[float, float] = {}
        for r in rows:
            ins = str(r.get('instrument_name', ''))
            parts = ins.split('-')
            if len(parts) < 4:
                continue
            try:
                strike = float(parts[2])
                oi = float(r.get('open_interest', 0.0))
            except Exception:
                continue
            strikes[strike] = strikes.get(strike, 0.0) + oi
        if not strikes:
            return 0.0
        return float(max(strikes.items(), key=lambda x: x[1])[0])

    @staticmethod
    def _put_call_ratio(rows: list[dict[str, Any]]) -> float:
        put_oi = 0.0
        call_oi = 0.0
        for r in rows:
            ins = str(r.get('instrument_name', ''))
            oi = float(r.get('open_interest', 0.0))
            if ins.endswith('-P'):
                put_oi += oi
            elif ins.endswith('-C'):
                call_oi += oi
        if call_oi <= 0:
            return 1.0
        return put_oi / call_oi

    @staticmethod
    def _iv_skew(rows: list[dict[str, Any]]) -> float:
        puts = [float(r.get('mark_iv', 0.0)) for r in rows if str(r.get('instrument_name', '')).endswith('-P')]
        calls = [float(r.get('mark_iv', 0.0)) for r in rows if str(r.get('instrument_name', '')).endswith('-C')]
        if not puts or not calls:
            return 0.0
        return (sum(puts) / len(puts) - sum(calls) / len(calls)) / 100.0

    @staticmethod
    def _atm_iv(rows: list[dict[str, Any]], index_price: float) -> float:
        nearest = None
        min_dist = 10e9
        for r in rows:
            ins = str(r.get('instrument_name', ''))
            parts = ins.split('-')
            if len(parts) < 4:
                continue
            try:
                strike = float(parts[2])
                dist = abs(strike - index_price)
                if dist < min_dist:
                    min_dist = dist
                    nearest = float(r.get('mark_iv', 0.0)) / 100.0
            except Exception:
                continue
        return float(nearest or 0.0)

    @staticmethod
    def _nearest_expiry_hours(rows: list[dict[str, Any]]) -> float:
        now = datetime.now(timezone.utc)
        min_hours = 9999.0
        for r in rows:
            ins = str(r.get('instrument_name', ''))
            parts = ins.split('-')
            if len(parts) < 4:
                continue
            expiry = parts[1]
            try:
                dt = datetime.strptime(expiry, '%d%b%y').replace(tzinfo=timezone.utc)
                hours = (dt - now).total_seconds() / 3600.0
                if 0 <= hours < min_hours:
                    min_hours = hours
            except Exception:
                continue
        return min_hours if min_hours < 9999 else 9999.0
