"""Config helpers for optional meta-control layers."""

from __future__ import annotations

import copy
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_META_CONTROLS: dict[str, Any] = {
    "enabled": True,
    "shadow_mode": True,
    "enforce_execution_gates": False,
    "data_confidence": {
        "enabled": True,
        "block_threshold": 0.5,
        "degrade_threshold": 0.7,
        "cache_ttl_seconds": 60,
    },
    "kelly_shrinkage": {
        "enabled": True,
        "max_equity_pct_per_trade": 2.0,
    },
    "calibration_freshness": {
        "enabled": True,
        "stale_days": 14,
        "critical_days": 30,
    },
    "alert_filter": {
        "enabled": True,
        "active_assets": ["BTC"],
    },
    "regime_explainer": {
        "enabled": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _disabled_config() -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_META_CONTROLS)
    cfg["enabled"] = False
    cfg["shadow_mode"] = True
    cfg["enforce_execution_gates"] = False
    return cfg


class MetaControlConfigLoader:
    """Small TTL-cached loader so execution paths do not repeatedly read disk."""

    _lock = threading.RLock()
    _cached_path: str | None = None
    _cached_mtime: float | None = None
    _cached_loaded_at: float = 0.0
    _cached: dict[str, Any] | None = None
    _ttl_seconds: float = 2.0

    @classmethod
    def load(cls, root_config: dict[str, Any] | None = None, path: str | Path = "config.yaml") -> dict[str, Any]:
        if isinstance(root_config, dict) and "meta_controls" in root_config:
            section = root_config.get("meta_controls") or {}
            if not isinstance(section, dict):
                return _disabled_config()
            return _deep_merge(DEFAULT_META_CONTROLS, section)

        cfg_path = Path(path)
        try:
            mtime = cfg_path.stat().st_mtime if cfg_path.exists() else None
        except Exception:
            mtime = None

        now = time.monotonic()
        with cls._lock:
            if (
                cls._cached is not None
                and cls._cached_path == str(cfg_path)
                and cls._cached_mtime == mtime
                and now - cls._cached_loaded_at <= cls._ttl_seconds
            ):
                return copy.deepcopy(cls._cached)

            loaded = _disabled_config()
            if cfg_path.exists():
                try:
                    import yaml  # type: ignore[import]

                    root = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                    if isinstance(root, dict) and isinstance(root.get("meta_controls"), dict):
                        loaded = _deep_merge(DEFAULT_META_CONTROLS, root["meta_controls"])
                except Exception as exc:
                    logger.debug("Meta-control config load skipped: %s", exc)

            cls._cached = copy.deepcopy(loaded)
            cls._cached_path = str(cfg_path)
            cls._cached_mtime = mtime
            cls._cached_loaded_at = now
            return loaded


def load_meta_controls_config(root_config: dict[str, Any] | None = None, path: str | Path = "config.yaml") -> dict[str, Any]:
    return MetaControlConfigLoader.load(root_config=root_config, path=path)


def meta_controls_active(meta_cfg: dict[str, Any] | None) -> bool:
    return bool(isinstance(meta_cfg, dict) and meta_cfg.get("enabled", False))


def module_enabled(meta_cfg: dict[str, Any] | None, name: str) -> bool:
    if not meta_controls_active(meta_cfg):
        return False
    section = (meta_cfg or {}).get(name) or {}
    return bool(isinstance(section, dict) and section.get("enabled", True))


def enforcement_active(meta_cfg: dict[str, Any] | None) -> bool:
    if not meta_controls_active(meta_cfg):
        return False
    return bool((meta_cfg or {}).get("enforce_execution_gates", False)) and not bool((meta_cfg or {}).get("shadow_mode", True))
