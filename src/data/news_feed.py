"""RSS news fetcher for crypto, stocks, and forex assets.

Provides a stable interface for dashboard endpoints with:
- per-asset in-memory TTL cache (5 minutes)
- keyword filtering by asset
- simple keyword-based sentiment classification
- robust error handling (never raises to callers)
"""

from __future__ import annotations

import html
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree as ET

import requests
try:
    import feedparser
except Exception:  # pragma: no cover - environment-dependent
    feedparser = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CACHE_TTL = 300  # 5 minutes
HTTP_TIMEOUT = 5

_cache: dict[str, dict[str, Any]] = {}

POSITIVE_KEYWORDS = [
    "surge",
    "rally",
    "gain",
    "bullish",
    "high",
    "rise",
    "up",
    "record",
    "buy",
    "long",
]
NEGATIVE_KEYWORDS = [
    "crash",
    "drop",
    "fall",
    "bearish",
    "low",
    "down",
    "sell",
    "short",
    "ban",
    "hack",
    "fear",
]

ASSET_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "ETH": ["ethereum", "eth", "ether"],
    "XAUUSD": ["gold", "xau", "precious metal"],
    "EURUSD": ["euro", "eur", "eurusd", "ecb"],
    "USDINR": ["rupee", "inr", "india", "rbi"],
    "USDJPY": ["yen", "jpy", "japan", "boj"],
    "GBPUSD": ["pound", "gbp", "uk", "boe"],
}

# Free RSS feeds
CRYPTO_RSS_SOURCES = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/feed",
]
US_STOCK_RSS_SOURCES = [
    "https://feeds.reuters.com/reuters/businessNews",
]
INDIA_STOCK_RSS_SOURCES = [
    "https://economictimes.indiatimes.com/markets/rss.cms",
    "https://www.moneycontrol.com/rss/business.xml",
]
FOREX_RSS_SOURCES = [
    "https://www.fxstreet.com/rss/news",
    "https://www.investing.com/rss/news_95.rss",
]

_TAG_RE = re.compile(r"<[^>]+>")


def _cache_key(symbol: str, asset_class: str, limit: int) -> str:
    return f"{asset_class.lower()}::{symbol.upper()}::{int(limit)}"


def _strip_html(text: str) -> str:
    if not text:
        return ""
    clean = _TAG_RE.sub(" ", text)
    clean = html.unescape(clean)
    return " ".join(clean.split())


def _entry_get(entry: Any, field: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(field, default)
    return getattr(entry, field, default)


def _extract_published(entry: Any) -> str:
    # Prefer parsed timestamps when available
    try:
        published_parsed = _entry_get(entry, "published_parsed", None)
        if published_parsed:
            dt = datetime(
                published_parsed.tm_year,
                published_parsed.tm_mon,
                published_parsed.tm_mday,
                published_parsed.tm_hour,
                published_parsed.tm_min,
                published_parsed.tm_sec,
                tzinfo=timezone.utc,
            )
            return dt.isoformat()
    except Exception:
        pass

    for field in ("published", "updated"):
        value = _entry_get(entry, field, None)
        if value:
            return str(value)
    return datetime.now(timezone.utc).isoformat()


def _sentiment_for_text(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    pos = sum(1 for w in POSITIVE_KEYWORDS if w in text)
    neg = sum(1 for w in NEGATIVE_KEYWORDS if w in text)
    if pos > neg:
        return "POSITIVE"
    if neg > pos:
        return "NEGATIVE"
    return "NEUTRAL"


def _keywords_for_asset(symbol: str, asset_class: str) -> list[str]:
    sym = symbol.upper().strip()
    if asset_class.lower() == "stock":
        base = sym.split(".")[0]
        # For stocks, ticker itself is the primary keyword.
        return [sym.lower(), base.lower()]
    return ASSET_KEYWORDS.get(sym, [sym.lower()])


def _matches_asset(text: str, keywords: list[str]) -> bool:
    text_l = text.lower()
    return any(k.lower() in text_l for k in keywords)


def _parse_feed_entries(url: str) -> list[Any]:
    if feedparser is None:
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            entries: list[dict[str, Any]] = []
            for item in root.findall(".//item")[:20]:
                entries.append(
                    {
                        "title": item.findtext("title", default=""),
                        "link": item.findtext("link", default=""),
                        "summary": item.findtext("description", default=""),
                        "description": item.findtext("description", default=""),
                        "published": item.findtext("pubDate", default=""),
                        "author": item.findtext("source", default="Unknown"),
                    }
                )
            return entries
        except Exception as exc:
            logger.warning("RSS XML fallback failed for %s: %s", url, exc)
            return []
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        entries = getattr(parsed, "entries", []) or []
        return list(entries)
    except Exception as e:
        logger.warning("RSS fetch failed for %s: %s", url, e)
        return []


def _normalize_items(
    entries: list[Any],
    asset: str,
    keywords: list[str],
    limit: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    for entry in entries:
        title = _strip_html(str(_entry_get(entry, "title", "") or ""))
        link = str(_entry_get(entry, "link", "") or "")
        summary = _strip_html(
            str(
                _entry_get(entry, "summary", "")
                or _entry_get(entry, "description", "")
                or ""
            )
        )
        source = ""
        source_obj = _entry_get(entry, "source", None)
        if isinstance(source_obj, dict):
            source = str(source_obj.get("title", "") or "")
        elif hasattr(source_obj, "title"):
            source = str(getattr(source_obj, "title", "") or "")

        if not source:
            source = str(_entry_get(entry, "author", "") or "Unknown")

        text_blob = f"{title} {summary}"
        if keywords and not _matches_asset(text_blob, keywords):
            continue

        dedupe_key = (link or title).strip().lower()
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        out.append(
            {
                "title": title or "Untitled",
                "url": link,
                "source": source,
                "published": _extract_published(entry),
                "summary": summary[:280] if summary else "",
                "sentiment": _sentiment_for_text(title, summary),
                "asset": asset.upper(),
            }
        )
        if len(out) >= limit:
            break
    return out


def _fetch_news(
    *,
    symbol: str,
    asset_class: str,
    limit: int,
    source_urls: list[str],
) -> list[dict[str, str]]:
    symbol_u = symbol.upper().strip()
    safe_limit = max(1, int(limit))
    key = _cache_key(symbol_u, asset_class, safe_limit)
    now = time.time()

    cached = _cache.get(key)
    if cached and (now - float(cached.get("fetched_at", 0))) < CACHE_TTL:
        return list(cached.get("data", []))[:safe_limit]

    try:
        keywords = _keywords_for_asset(symbol_u, asset_class)
        collected: list[dict[str, str]] = []
        for url in source_urls:
            entries = _parse_feed_entries(url)
            if not entries:
                continue
            items = _normalize_items(entries, symbol_u, keywords, safe_limit)
            if items:
                collected.extend(items)
            if len(collected) >= safe_limit:
                break

        # Deduplicate one more time across sources.
        deduped: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for item in collected:
            u = item.get("url", "").strip().lower() or item.get("title", "").strip().lower()
            if not u or u in seen_urls:
                continue
            seen_urls.add(u)
            deduped.append(item)
            if len(deduped) >= safe_limit:
                break

        _cache[key] = {"data": deduped, "fetched_at": now}
        return deduped
    except Exception as e:
        logger.warning("News fetch failed for %s/%s: %s", asset_class, symbol_u, e)
        return []


def get_crypto_news(symbol: str = "BTC", limit: int = 8) -> list[dict]:
    try:
        return _fetch_news(
            symbol=symbol,
            asset_class="crypto",
            limit=limit,
            source_urls=CRYPTO_RSS_SOURCES,
        )
    except Exception as e:
        logger.warning("get_crypto_news failed for %s: %s", symbol, e)
        return []


def get_btc_news(limit: int = 8) -> list[dict]:
    try:
        return get_crypto_news(symbol="BTC", limit=limit)
    except Exception as e:
        logger.warning("get_btc_news failed: %s", e)
        return []


def get_stock_news(ticker: str = "AAPL", limit: int = 8) -> list[dict]:
    sym = (ticker or "AAPL").upper().strip()
    yahoo_url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
    )
    sources = [yahoo_url]

    # Route to Indian or US generic business feeds.
    if sym.endswith(".NS") or sym.endswith(".BO"):
        sources.extend(INDIA_STOCK_RSS_SOURCES)
    else:
        sources.extend(US_STOCK_RSS_SOURCES)

    try:
        return _fetch_news(
            symbol=sym,
            asset_class="stock",
            limit=limit,
            source_urls=sources,
        )
    except Exception as e:
        logger.warning("get_stock_news failed for %s: %s", sym, e)
        return []


def get_forex_news(pair: str = "EURUSD", limit: int = 8) -> list[dict]:
    try:
        return _fetch_news(
            symbol=(pair or "EURUSD").upper().strip(),
            asset_class="forex",
            limit=limit,
            source_urls=FOREX_RSS_SOURCES,
        )
    except Exception as e:
        logger.warning("get_forex_news failed for %s: %s", pair, e)
        return []


def get_news_for_asset(symbol: str, asset_class: str, limit: int = 8) -> list[dict]:
    cls = (asset_class or "crypto").lower().strip()
    sym = (symbol or "BTC").upper().strip()
    safe_limit = max(1, int(limit))
    try:
        if cls == "crypto":
            return get_crypto_news(sym, safe_limit)
        if cls == "stock":
            return get_stock_news(sym, safe_limit)
        if cls == "forex":
            return get_forex_news(sym, safe_limit)
        logger.warning("Unknown asset_class '%s' for symbol '%s'", cls, sym)
        return []
    except Exception as e:
        logger.warning("get_news_for_asset failed for %s/%s: %s", cls, sym, e)
        return []
