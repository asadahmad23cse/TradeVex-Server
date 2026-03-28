from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


logger = logging.getLogger(__name__)


class WhaleTrackerPoller:
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name='whale_tracker_poller')

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
                    await self.buffer.set_whale_tracker(payload)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning('Whale tracker polling failed: %s', exc)
                await asyncio.sleep(settings.whale_poll_sec)

    async def _fetch(self, client: httpx.AsyncClient) -> dict:
        if not settings.arkham_api_key:
            return {
                'whale_exchange_deposits_1h': 0,
                'whale_exchange_withdrawals_1h': 0,
                'whale_net_flow': 0.0,
                'whale_alert_score': 0.0,
                'source': 'fallback_no_api_key',
            }

        now = int(datetime.now(timezone.utc).timestamp())
        start = now - 3600
        resp = await client.get(
            settings.whale_alert_base,
            params={
                'api_key': settings.arkham_api_key,
                'start': start,
                'end': now,
                'min_value': 100_000_000,  # roughly 100 BTC notional threshold in USD scale
                'currency': 'btc',
            },
        )
        resp.raise_for_status()
        txs = resp.json().get('transactions', [])

        deposits = 0
        withdrawals = 0
        net_flow_btc = 0.0
        for tx in txs:
            tx_type = str(tx.get('transaction_type', '')).lower()
            amount = float(tx.get('amount', 0.0))
            if amount < 100:
                continue
            if 'deposit' in tx_type:
                deposits += 1
                net_flow_btc -= amount
            if 'withdrawal' in tx_type:
                withdrawals += 1
                net_flow_btc += amount

        score = max(-1.0, min(1.0, net_flow_btc / 500.0))
        return {
            'whale_exchange_deposits_1h': deposits,
            'whale_exchange_withdrawals_1h': withdrawals,
            'whale_net_flow': net_flow_btc,
            'whale_alert_score': score,
            'source': 'whale_alert',
        }
