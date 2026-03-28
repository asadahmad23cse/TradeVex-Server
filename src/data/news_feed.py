"""Fetch latest BTC news from CryptoPanic (free, no API key for public feed)."""

import logging
import time

import requests

logger = logging.getLogger(__name__)

_cache = {"data": [], "ts": 0}
CACHE_TTL = 120  # refresh every 2 min


def _fetch_cryptocompare(limit: int) -> list[dict]:
    news: list[dict] = []
    try:
        alt_resp = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/?categories=BTC&lang=EN",
            timeout=10,
        )
        alt_resp.raise_for_status()
        alt_data = alt_resp.json().get("Data", [])
        for item in alt_data[:limit]:
            news.append(
                {
                    "title": item.get("title", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "published": item.get("published_on", ""),
                    "sentiment": "⚪",
                }
            )
    except Exception:
        pass
    return news


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

        # If CryptoPanic returns nothing, try free alternative:
        if not news:
            news = _fetch_cryptocompare(limit)

        _cache["data"] = news
        _cache["ts"] = now
        return news[:limit]

    except Exception as e:
        logger.warning("CryptoPanic fetch failed: %s", e)
        news = _fetch_cryptocompare(limit)
        if news:
            _cache["data"] = news
            _cache["ts"] = now
            return news[:limit]
        return _cache["data"][:limit] if _cache["data"] else []
