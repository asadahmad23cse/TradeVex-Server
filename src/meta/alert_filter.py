"""Alert noise filtering for monitoring notifications."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertFilterResult:
    alert: dict[str, Any]
    downgraded: bool
    asset: str
    original_level: str
    filtered_level: str
    shadow_mode: bool
    enforced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert": dict(self.alert),
            "downgraded": self.downgraded,
            "asset": self.asset,
            "original_level": self.original_level,
            "filtered_level": self.filtered_level,
            "shadow_mode": self.shadow_mode,
            "enforced": self.enforced,
        }


class AlertNoiseFilter:
    """Downgrades non-active-asset CRITICAL/WARNING alerts to INFO."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        shadow_mode: bool = True,
        enforce: bool = False,
    ) -> None:
        cfg = config or {}
        self.active_assets = {self._normalize_asset(x) for x in cfg.get("active_assets", ["BTC"])}
        self.shadow_mode = bool(shadow_mode)
        self.enforce = bool(enforce)

    def filter_alert(self, alert: dict[str, Any] | None) -> AlertFilterResult:
        source = dict(alert or {})
        asset = self._extract_asset(source)
        original_level = self._extract_level(source)
        should_downgrade = bool(asset and asset not in self.active_assets and original_level in {"CRITICAL", "WARNING"})
        filtered_level = "INFO" if should_downgrade else original_level
        enforced = self.enforce and not self.shadow_mode
        out = dict(source)
        if should_downgrade and enforced:
            self._set_level(out, filtered_level)

        if should_downgrade:
            logger.info(
                "[ALERT_FILTER] downgraded asset=%s type=%s",
                asset,
                str(source.get("type") or source.get("component") or source.get("event_type") or "alert"),
            )

        return AlertFilterResult(
            alert=out,
            downgraded=should_downgrade,
            asset=asset,
            original_level=original_level,
            filtered_level=filtered_level,
            shadow_mode=self.shadow_mode,
            enforced=bool(enforced),
        )

    @classmethod
    def _extract_asset(cls, alert: dict[str, Any]) -> str:
        for key in ("asset", "ticker", "symbol"):
            val = alert.get(key)
            if val:
                return cls._normalize_asset(val)
        details = alert.get("details")
        if isinstance(details, dict):
            for key in ("asset", "ticker", "symbol"):
                val = details.get(key)
                if val:
                    return cls._normalize_asset(val)
        return ""

    @staticmethod
    def _extract_level(alert: dict[str, Any]) -> str:
        for key in ("severity", "level", "status"):
            val = alert.get(key)
            if val:
                text = str(val).strip().upper()
                for level in ("CRITICAL", "WARNING", "INFO"):
                    if text.startswith(level):
                        return level
        return "INFO"

    @staticmethod
    def _set_level(alert: dict[str, Any], level: str) -> None:
        for key in ("severity", "level"):
            if key in alert:
                alert[key] = level
                return
        status = alert.get("status")
        if status:
            text = str(status)
            upper = text.upper()
            for prefix in ("CRITICAL", "WARNING", "INFO"):
                if upper.startswith(prefix):
                    alert["status"] = level + text[len(prefix):]
                    return
        alert["severity"] = level

    @staticmethod
    def _normalize_asset(value: Any) -> str:
        text = str(value or "").strip().upper()
        if text.endswith("USDT"):
            text = text[:-4]
        if text.endswith("-USD"):
            text = text[:-4]
        return text
