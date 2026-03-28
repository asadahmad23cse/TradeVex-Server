"""Fetch latest BTC news from CryptoPanic (free, no API key for public feed)."""

import logging
import time

import requests

logger = logging.getLogger(__name__)

_cache = {"data": [], "ts": 0}
CACHE_TTL = 120  # refresh every 2 min


def get_btc_news(limit: int = 8) -> list[dict]:
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL and _cache["data"]:
        return _cache["data"][:limit]

    try:
        resp = requests.get(
            "https://cryptopanic.com/api/free/v1/posts/",
            params={"currencies": "BTC", "kind": "news", "public": "true"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        news = []
        for item in results[:limit]:
            votes = item.get("votes", {})
            pos = votes.get("positive", 0)
            neg = votes.get("negative", 0)
            sentiment = "🟢" if pos > neg + 1 else "🔴" if neg > pos + 1 else "⚪"

            news.append(
                {
                    "title": item.get("title", ""),
                    "source": item.get("source", {}).get("title", ""),
                    "url": item.get("url", ""),
                    "published": item.get("published_at", ""),
                    "sentiment": sentiment,
                }
            )

        _cache["data"] = news
        _cache["ts"] = now
        return news[:limit]

    except Exception as e:
        logger.warning("CryptoPanic fetch failed: %s", e)
        return _cache["data"][:limit] if _cache["data"] else []
