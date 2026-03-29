"""Bitcoin Fear & Greed provider with API/cache fallback."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.alternative.me/fng/?limit=60&format=json"
TTL_SECONDS = 3600
_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "fear_greed_cache.json"


class FearGreedProvider:
    def __init__(self, cache_path: Path | None = None, ttl_seconds: int = TTL_SECONDS):
        self._cache_path = cache_path or _CACHE_PATH
        self._ttl_seconds = int(ttl_seconds)
        self._cached: dict[str, Any] | None = None
        self._cached_at = 0.0

    def get_snapshot(self) -> dict[str, Any]:
        now = time.time()
        if self._cached is not None and (now - self._cached_at) < self._ttl_seconds:
            return dict(self._cached)

        payload = self._load_disk_cache()
        if payload is not None and (now - float(payload.get("fetched_at", 0.0))) < self._ttl_seconds:
            self._cached = self._build_snapshot(payload)
            self._cached_at = now
            return dict(self._cached)

        try:
            raw = self._fetch()
            self._save_disk_cache(raw)
            self._cached = self._build_snapshot(raw)
            self._cached_at = now
            return dict(self._cached)
        except Exception as exc:
            logger.warning("FearGreedProvider live fetch failed: %s", exc)

        if payload is not None:
            self._cached = self._build_snapshot(payload)
            self._cached_at = now
            return dict(self._cached)

        return {
            "value": 50,
            "label": "Neutral",
            "z_score": 0.0,
            "source": "fallback",
            "as_of": None,
        }

    def _fetch(self) -> dict[str, Any]:
        resp = requests.get(API_URL, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Unexpected fear & greed payload")
        return {"fetched_at": time.time(), "data": payload.get("data", [])}

    def _build_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows = payload.get("data", [])
        values = []
        for row in rows:
            try:
                values.append(float(row.get("value", 50)))
            except Exception:
                continue
        if not values:
            return {
                "value": 50,
                "label": "Neutral",
                "z_score": 0.0,
                "source": "fallback",
                "as_of": None,
            }

        latest = float(values[0])
        arr = np.asarray(list(reversed(values[:30])), dtype=float)
        mu = float(np.mean(arr)) if arr.size else latest
        sd = float(np.std(arr)) if arr.size else 0.0
        z_score = (latest - mu) / max(sd, 1e-9) if sd > 0 else 0.0
        latest_row = rows[0] if rows else {}

        return {
            "value": int(round(latest)),
            "label": str(latest_row.get("value_classification", "Neutral")),
            "z_score": round(float(z_score), 3),
            "source": "alternative_me",
            "as_of": latest_row.get("timestamp"),
        }

    def _load_disk_cache(self) -> dict[str, Any] | None:
        try:
            if not self._cache_path.exists():
                return None
            return json.loads(self._cache_path.read_text())
        except Exception:
            return None

    def _save_disk_cache(self, payload: dict[str, Any]) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(payload))
        except Exception as exc:
            logger.debug("FearGreedProvider cache save failed: %s", exc)


_provider: FearGreedProvider | None = None


def get_fear_greed_provider() -> FearGreedProvider:
    global _provider
    if _provider is None:
        _provider = FearGreedProvider()
    return _provider
