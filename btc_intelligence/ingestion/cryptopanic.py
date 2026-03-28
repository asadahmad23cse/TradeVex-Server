from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from btc_intelligence.config import settings
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer


logger = logging.getLogger(__name__)


class CryptoPanicPoller:
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name='cryptopanic_poller')

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            while self._running:
                try:
                    news = await self._fetch_news(client)
                    await self.buffer.set_news(news)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning('CryptoPanic polling failed: %s', exc)
                await asyncio.sleep(settings.cryptopanic_poll_sec)

    async def _fetch_news(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        if not settings.cryptopanic_api_key:
            return []

        params = {
            'auth_token': settings.cryptopanic_api_key,
            'currencies': 'BTC',
            'kind': 'news',
            'filter': 'hot',
        }
        resp = await client.get(settings.cryptopanic_base, params=params)
        resp.raise_for_status()
        rows = resp.json().get('results', [])

        parsed: list[dict[str, Any]] = []
        for row in rows[:40]:
            title = str(row.get('title', ''))
            published_at = str(row.get('published_at', ''))
            sentiment = self._score_votes(row.get('votes', {}))
            is_blocking = any(k in title.lower() for k in settings.blocking_news_keywords)
            parsed.append(
                {
                    'title': title,
                    'published_at': published_at,
                    'sentiment_score': sentiment,
                    'importance': 'high' if is_blocking else 'normal',
                    'url': row.get('url', ''),
                }
            )
        return parsed

    @staticmethod
    def _score_votes(votes: dict[str, Any]) -> float:
        positive = float(votes.get('positive', 0) or 0)
        negative = float(votes.get('negative', 0) or 0)
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total

    @staticmethod
    def has_blocking_news(news_items: list[dict[str, Any]], minutes: int = 30) -> bool:
        if not news_items:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        for row in news_items:
            if row.get('importance') != 'high':
                continue
            ts = str(row.get('published_at', ''))
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except Exception:
                continue
            if dt >= cutoff:
                return True
        return False
