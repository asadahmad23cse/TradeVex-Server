"""Calibration freshness guard for execution metadata and gates."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationFreshnessResult:
    age_days: int | None
    status: str
    calibration_warning: bool
    execution_confidence_multiplier: float
    execution_position_multiplier: float
    shadow_mode: bool
    enforced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "age_days": self.age_days,
            "status": self.status,
            "calibration_status": self.status,
            "calibration_warning": self.calibration_warning,
            "execution_confidence_multiplier": self.execution_confidence_multiplier,
            "execution_position_multiplier": self.execution_position_multiplier,
            "shadow_mode": self.shadow_mode,
            "enforced": self.enforced,
        }


class CalibrationFreshnessGuard:
    """Detects stale calibration without mutating raw confidence."""

    DEFAULT_CHECKPOINT_PATH = Path("btc_intelligence/data/platt_buffer_checkpoint.json")

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        shadow_mode: bool = True,
        enforce_execution_gates: bool = False,
    ) -> None:
        cfg = config or {}
        self.stale_days = int(cfg.get("stale_days", 14))
        self.critical_days = int(cfg.get("critical_days", 30))
        self.shadow_mode = bool(shadow_mode)
        self.enforce_execution_gates = bool(enforce_execution_gates)

    def evaluate(
        self,
        last_calibration_timestamp: Any | None = None,
        current_time: Any | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> CalibrationFreshnessResult:
        ts = last_calibration_timestamp
        if ts is None:
            ts = self._timestamp_from_checkpoint(checkpoint_path or self.DEFAULT_CHECKPOINT_PATH)
        last_dt = self._parse_time(ts)
        now_dt = self._parse_time(current_time) or datetime.now(timezone.utc)

        if last_dt is None:
            result = CalibrationFreshnessResult(
                age_days=None,
                status="unknown",
                calibration_warning=True,
                execution_confidence_multiplier=1.0,
                execution_position_multiplier=1.0,
                shadow_mode=self.shadow_mode,
                enforced=False,
            )
            logger.info("[CALIBRATION] age=unknown status=unknown")
            return result

        age_days = max(0, int((now_dt - last_dt).total_seconds() // 86400))
        if age_days > self.critical_days:
            status = "critical"
        elif age_days > self.stale_days:
            status = "stale"
        else:
            status = "fresh"

        enforced = self.enforce_execution_gates and not self.shadow_mode
        confidence_multiplier = 0.8 if enforced and status == "stale" else 1.0
        position_multiplier = 0.5 if enforced and status == "critical" else 1.0

        result = CalibrationFreshnessResult(
            age_days=age_days,
            status=status,
            calibration_warning=status in {"stale", "critical"},
            execution_confidence_multiplier=float(confidence_multiplier),
            execution_position_multiplier=float(position_multiplier),
            shadow_mode=self.shadow_mode,
            enforced=bool(enforced),
        )
        logger.info("[CALIBRATION] age=%s status=%s", age_days, status)
        return result

    @classmethod
    def _timestamp_from_checkpoint(cls, checkpoint_path: str | Path) -> Any | None:
        path = Path(checkpoint_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        for key in ("last_calibration_timestamp", "saved_at", "updated_at", "created_at"):
            if payload.get(key):
                return payload.get(key)
        return None

    @staticmethod
    def _parse_time(value: Any | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            text = str(value).strip()
            if not text:
                return None
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
            except Exception:
                try:
                    dt = datetime.fromtimestamp(float(text), tz=timezone.utc)
                except Exception:
                    return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
