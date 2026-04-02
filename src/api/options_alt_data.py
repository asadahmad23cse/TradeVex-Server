"""
OptionsAltDataProvider — NSE chain fetch + F15/F16/F17 for factor_model.

Disabled by default (options_intelligence.enabled: false).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.alpha.options_intelligence import OptionsIntelligence
from src.api.options_chain import NSEOptionsChain

logger = logging.getLogger("api.options_alt_data")

_CACHE_TTL = 300
_INDIAN_SUFFIXES = (".NS", ".BO")
_INDEX_TICKERS = frozenset(
    {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}
)

_ZERO_RESULT: dict[str, Any] = {
    "F15_iv_skew": 0.0,
    "F16_max_pain": 0.0,
    "F17_gex": 0.0,
    "max_pain_level": None,
    "available": False,
}


class OptionsAltDataProvider:
    """Shared chain fetcher; per-ticker OptionsIntelligence for z-history."""

    def __init__(self, config: dict | None) -> None:
        cfg = config or {}
        oi = cfg.get("options_intelligence") or {}
        self._enabled = bool(oi.get("enabled", False))
        self._log_only = bool(oi.get("log_only_mode", False))
        ttl = int(oi.get("cache_ttl_minutes", 5)) * 60
        self._result_ttl = max(60, ttl)
        self._chain_fetcher: NSEOptionsChain | None = None
        self._analyzers: dict[str, OptionsIntelligence] = {}
        self._result_cache: dict[str, tuple[float, dict[str, Any]]] = {}

        if not self._enabled:
            logger.info("OptionsAltDataProvider: disabled in config")
            return

        self._chain_fetcher = NSEOptionsChain(cache_ttl_seconds=self._result_ttl)
        logger.info(
            "OptionsAltDataProvider: enabled (log_only=%s)",
            self._log_only,
        )

    def get_options_factors(self, ticker: str) -> dict[str, Any]:
        try:
            if not self._enabled or self._chain_fetcher is None:
                return dict(_ZERO_RESULT)

            if not self.is_indian_ticker(ticker):
                return dict(_ZERO_RESULT)

            tkey = (ticker or "").strip().upper()
            now = time.time()
            cached = self._result_cache.get(tkey)
            if cached is not None:
                ts, result = cached
                if (now - ts) < self._result_ttl:
                    return result

            chain = self._chain_fetcher.get_nearest_expiry_chain(ticker)

            if tkey not in self._analyzers:
                self._analyzers[tkey] = OptionsIntelligence()
            analyzer = self._analyzers[tkey]

            underlying = 0.0
            if chain is not None:
                underlying = float(chain.get("underlying_price") or 0.0)

            factors = analyzer.get_all_factors(chain, underlying, days_to_expiry=7)

            result = {
                "F15_iv_skew": float(factors["F15_iv_skew"]),
                "F16_max_pain": float(factors["F16_max_pain"]),
                "F17_gex": float(factors["F17_gex"]),
                "max_pain_level": factors["max_pain_level"],
                "available": bool(factors["computation_ok"]),
            }

            if self._log_only:
                logger.info(
                    "[OPTIONS LOG-ONLY] %s → IV_skew=%.3f MaxPain=%.3f GEX=%.3f pain_level=%s",
                    ticker,
                    result["F15_iv_skew"],
                    result["F16_max_pain"],
                    result["F17_gex"],
                    result["max_pain_level"],
                )
                cached_result = dict(_ZERO_RESULT)
                cached_result["available"] = result["available"]
                self._result_cache[tkey] = (now, cached_result)
                return cached_result

            self._result_cache[tkey] = (now, result)
            return result

        except Exception as exc:
            logger.warning("get_options_factors failed for %s: %s", ticker, exc)
            return dict(_ZERO_RESULT)

    @staticmethod
    def is_indian_ticker(ticker: str) -> bool:
        t = (ticker or "").strip().upper()
        if t in _INDEX_TICKERS:
            return True
        return any(t.endswith(s) for s in _INDIAN_SUFFIXES)

    def get_all_tickers_snapshot(self, tickers: list) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for t in tickers:
            if t and self.is_indian_ticker(str(t)):
                out[str(t)] = self.get_options_factors(str(t))
        return out
