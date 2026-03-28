from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class LiveWebSocketHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, initial_payload: dict[str, Any] | None = None) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        if initial_payload is not None:
            await ws.send_json(initial_payload)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        msg = json.dumps(payload)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {'subscribers': len(self._clients)}
