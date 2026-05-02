"""Global data confidence scoring without network calls."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfidenceResult:
    data_confidence_score: float
    degraded: bool
    blocked: bool
    feed_status: dict[str, str]
    execution_position_multiplier: float
    execution_block_new_trades: bool
    shadow_mode: bool
    enforced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_confidence_score": self.data_confidence_score,
            "score": self.data_confidence_score,
            "degraded": self.degraded,
            "blocked": self.blocked,
            "feed_status": dict(self.feed_status),
            "execution_position_multiplier": self.execution_position_multiplier,
            "execution_block_new_trades": self.execution_block_new_trades,
            "shadow_mode": self.shadow_mode,
            "enforced": self.enforced,
        }


class DataConfidenceEngine:
    """Scores latest known feed state and emits optional execution gates."""

    WEIGHTS = {
        "binance": 0.40,
        "coinglass": 0.20,
        "etf": 0.15,
        "fred": 0.10,
        "news": 0.15,
    }
    STATUS_FACTORS = {
        "working": 1.0,
        "ok": 1.0,
        "live": 1.0,
        "fallback": 0.5,
        "placeholder": 0.5,
        "stale_cache": 0.5,
        "unknown": 0.5,
        "failed": 0.0,
        "error": 0.0,
        "down": 0.0,
        "unavailable": 0.0,
    }

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        shadow_mode: bool = True,
        enforce_execution_gates: bool = False,
    ) -> None:
        cfg = config or {}
        self.block_threshold = float(cfg.get("block_threshold", 0.5))
        self.degrade_threshold = float(cfg.get("degrade_threshold", 0.7))
        self.cache_ttl_seconds = float(cfg.get("cache_ttl_seconds", 60))
        self.shadow_mode = bool(shadow_mode)
        self.enforce_execution_gates = bool(enforce_execution_gates)
        self._lock = threading.RLock()
        self._feed_status: dict[str, str] = {feed: "fallback" for feed in self.WEIGHTS}
        self._updated_at = 0.0

    @classmethod
    def normalize_status(cls, value: Any) -> str:
        text = str(value or "fallback").strip().lower()
        if text in cls.STATUS_FACTORS:
            return text
        if text in {"healthy", "connected", "available"}:
            return "working"
        if text in {"degraded", "cached", "stale"}:
            return "fallback"
        if text in {"missing", "none", "disabled"}:
            return "failed"
        return "fallback"

    def update_feed_status(self, feed_status: dict[str, Any]) -> None:
        with self._lock:
            for feed, status in (feed_status or {}).items():
                key = str(feed or "").strip().lower()
                if key in self.WEIGHTS:
                    self._feed_status[key] = self.normalize_status(status)
            self._updated_at = time.monotonic()

    def evaluate(self, feed_status: dict[str, Any] | None = None) -> DataConfidenceResult:
        if feed_status is not None:
            self.update_feed_status(feed_status)

        with self._lock:
            snapshot = dict(self._feed_status)
            if self._updated_at and (time.monotonic() - self._updated_at) > self.cache_ttl_seconds:
                snapshot = {feed: "fallback" for feed in self.WEIGHTS}

        score = 0.0
        for feed, weight in self.WEIGHTS.items():
            status = self.normalize_status(snapshot.get(feed, "fallback"))
            snapshot[feed] = status
            score += float(weight) * float(self.STATUS_FACTORS.get(status, 0.5))
        score = max(0.0, min(1.0, score))

        degraded = score < self.degrade_threshold
        blocked = score < self.block_threshold
        enforced = self.enforce_execution_gates and not self.shadow_mode
        result = DataConfidenceResult(
            data_confidence_score=round(score, 4),
            degraded=bool(degraded),
            blocked=bool(blocked),
            feed_status=snapshot,
            execution_position_multiplier=0.5 if degraded else 1.0,
            execution_block_new_trades=bool(enforced and blocked),
            shadow_mode=self.shadow_mode,
            enforced=bool(enforced),
        )
        logger.info(
            "[DATA_CONFIDENCE] score=%.2f degraded=%s blocked=%s",
            result.data_confidence_score,
            str(result.degraded).lower(),
            str(result.blocked).lower(),
        )
        return result

    @classmethod
    def status_from_signal_payload(cls, payload: dict[str, Any] | None) -> dict[str, str]:
        data = payload or {}
        market_context = data.get("market_context") if isinstance(data.get("market_context"), dict) else {}
        futures = market_context.get("futures") if isinstance(market_context.get("futures"), dict) else {}
        etf_flow = data.get("etf_flow") if isinstance(data.get("etf_flow"), dict) else market_context.get("etf_flow", {})
        if not isinstance(etf_flow, dict):
            etf_flow = {}

        dq = data.get("data_quality") if isinstance(data.get("data_quality"), dict) else {}
        reason = str(data.get("reason") or "").lower()
        has_price = any(
            cls._safe_float(data.get(k), 0.0) > 0
            for k in ("entry_price", "entry", "mark_price", "current_price", "price")
        )
        has_futures_price = cls._safe_float(futures.get("mark_price"), 0.0) > 0

        if bool(dq.get("severe", False)) or "insufficient binance" in reason:
            binance_status = "failed"
        elif has_price or has_futures_price:
            binance_status = "working"
        else:
            binance_status = "fallback"

        futures_source = str(futures.get("source") or data.get("coinglass_source") or "").lower()
        if futures_source in {"coinglass", "live", "working"}:
            coinglass_status = "working"
        elif any(cls._safe_float(futures.get(k), 0.0) != 0.0 for k in ("funding_rate_pct", "open_interest_btc", "liquidation_score")):
            coinglass_status = "fallback"
        else:
            coinglass_status = "fallback"

        etf_source = str(etf_flow.get("source") or "").lower()
        if etf_source in {"fallback", "stale_cache", "unknown", ""}:
            etf_status = "fallback"
        elif etf_source in {"failed", "error", "unavailable"}:
            etf_status = "failed"
        else:
            etf_status = "working"

        fred_ctx = market_context.get("fred") if isinstance(market_context.get("fred"), dict) else {}
        news_ctx = market_context.get("news") if isinstance(market_context.get("news"), dict) else {}
        fred_status = cls.normalize_status(fred_ctx.get("status", "fallback") if isinstance(fred_ctx, dict) else "fallback")
        news_status = cls.normalize_status(news_ctx.get("status", data.get("news_status", "fallback")) if isinstance(news_ctx, dict) else data.get("news_status", "fallback"))

        return {
            "binance": binance_status,
            "coinglass": coinglass_status,
            "etf": etf_status,
            "fred": fred_status,
            "news": news_status,
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)
