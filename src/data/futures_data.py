"""Binance Futures public data: funding rate + open interest."""

import logging

import requests

logger = logging.getLogger(__name__)

FUTURES_BASE = "https://fapi.binance.com"


def get_funding_rate() -> dict:
    """Fetch current BTC funding rate from Binance Futures."""
    try:
        resp = requests.get(
            f"{FUTURES_BASE}/fapi/v1/premiumIndex",
            params={"symbol": "BTCUSDT"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = float(data.get("lastFundingRate", 0))
        mark_price = float(data.get("markPrice", 0))
        return {
            "funding_rate": rate,
            "funding_rate_pct": round(rate * 100, 4),
            "mark_price": mark_price,
            "next_funding_time": data.get("nextFundingTime"),
        }
    except Exception as e:
        logger.warning("Funding rate fetch failed: %s", e)
        return {"funding_rate": 0, "funding_rate_pct": 0, "mark_price": 0}


def get_open_interest() -> dict:
    """Fetch current BTC open interest from Binance Futures."""
    try:
        resp = requests.get(
            f"{FUTURES_BASE}/fapi/v1/openInterest",
            params={"symbol": "BTCUSDT"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        oi = float(data.get("openInterest", 0))
        return {"open_interest_btc": oi}
    except Exception as e:
        logger.warning("Open interest fetch failed: %s", e)
        return {"open_interest_btc": 0}


def get_futures_sentiment() -> dict:
    """Combined futures sentiment analysis."""
    funding = get_funding_rate()
    oi = get_open_interest()
    rate = funding["funding_rate_pct"]

    # Funding rate sentiment
    if rate > 0.05:
        funding_sentiment = "OVERLEVERAGED_LONG"
    elif rate < -0.05:
        funding_sentiment = "OVERLEVERAGED_SHORT"
    elif -0.02 <= rate <= 0.02:
        funding_sentiment = "NEUTRAL"
    else:
        funding_sentiment = "MILD_LONG" if rate > 0 else "MILD_SHORT"

    return {
        **funding,
        **oi,
        "funding_sentiment": funding_sentiment,
    }
