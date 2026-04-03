from __future__ import annotations

import json
import logging
from typing import Any

from redis.asyncio import Redis  # type: ignore[import]


logger = logging.getLogger(__name__)


class RedisStateStore:
    """Lightweight JSON/list state persistence for runtime market state."""

    def __init__(self, redis_url: str, key_prefix: str = "btc") -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix.strip() or "btc"
        self._client: Redis | None = None
        self.connected = False

    def _key(self, name: str) -> str:
        return f"{self.key_prefix}:{name}"

    async def connect(self) -> bool:
        if self._client is not None and self.connected:
            return True
        try:
            self._client = Redis.from_url(self.redis_url, decode_responses=True)
            pong = await self._client.ping()
            self.connected = bool(pong)
            logger.info("Redis state store connected: %s", self.redis_url)
        except Exception as exc:
            self.connected = False
            self._client = None
            logger.warning("Redis state store unavailable; using in-memory fallback: %s", exc)
        return self.connected

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.close()
        except Exception:
            pass
        finally:
            self._client = None
            self.connected = False

    async def set_json(self, name: str, payload: Any, ttl_seconds: int | None = None) -> None:
        if not self.connected or self._client is None:
            return
        try:
            key = self._key(name)
            data = json.dumps(payload, ensure_ascii=True)
            if ttl_seconds is None:
                await self._client.set(key, data)
            else:
                await self._client.set(key, data, ex=int(ttl_seconds))
        except Exception as exc:
            logger.debug("Redis set_json failed for %s: %s", name, exc)

    async def get_json(self, name: str, default: Any) -> Any:
        if not self.connected or self._client is None:
            return default
        try:
            raw = await self._client.get(self._key(name))
            if not raw:
                return default
            return json.loads(raw)
        except Exception as exc:
            logger.debug("Redis get_json failed for %s: %s", name, exc)
            return default

    async def lpush_json(self, name: str, payload: Any, max_len: int) -> None:
        if not self.connected or self._client is None:
            return
        try:
            key = self._key(name)
            data = json.dumps(payload, ensure_ascii=True)
            async with self._client.pipeline(transaction=False) as pipe:
                await pipe.lpush(key, data)
                await pipe.ltrim(key, 0, max(0, int(max_len) - 1))
                await pipe.execute()
        except Exception as exc:
            logger.debug("Redis lpush_json failed for %s: %s", name, exc)

    async def clear_key(self, name: str) -> None:
        if not self.connected or self._client is None:
            return
        try:
            await self._client.delete(self._key(name))
        except Exception as exc:
            logger.debug("Redis clear_key failed for %s: %s", name, exc)

    async def upsert_latest_json(
        self,
        name: str,
        payload: Any,
        max_len: int,
        id_field: str = "open_time",
    ) -> None:
        """
        Upsert most-recent element in a list keyed by a stable id field.

        If the latest element has the same id, replace index 0.
        Otherwise prepend and trim.
        """
        if not self.connected or self._client is None:
            return
        try:
            key = self._key(name)
            encoded = json.dumps(payload, ensure_ascii=True)
            new_id = None
            if isinstance(payload, dict):
                new_id = payload.get(id_field)

            if new_id is None:
                await self.lpush_json(name, payload, max_len=max_len)
                return

            head_raw = await self._client.lindex(key, 0)
            if head_raw:
                try:
                    head_obj = json.loads(head_raw)
                except Exception:
                    head_obj = None
                if isinstance(head_obj, dict) and head_obj.get(id_field) == new_id:
                    await self._client.lset(key, 0, encoded)
                    return

            async with self._client.pipeline(transaction=False) as pipe:
                await pipe.lpush(key, encoded)
                await pipe.ltrim(key, 0, max(0, int(max_len) - 1))
                await pipe.execute()
        except Exception as exc:
            logger.debug("Redis upsert_latest_json failed for %s: %s", name, exc)

    async def lrange_json(self, name: str, start: int, end: int, reverse: bool = False) -> list[Any]:
        if not self.connected or self._client is None:
            return []
        try:
            raw_rows = await self._client.lrange(self._key(name), int(start), int(end))
            rows = [json.loads(x) for x in raw_rows]
            if reverse:
                rows.reverse()
            return rows
        except Exception as exc:
            logger.debug("Redis lrange_json failed for %s: %s", name, exc)
            return []
