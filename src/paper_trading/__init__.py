"""Paper trading singletons."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .auto_executor import AutoExecutor
from .paper_engine import PaperTradingEngine

_paper_engine: PaperTradingEngine | None = None
_auto_executor: AutoExecutor | None = None
_paper_engines_by_user: dict[str, PaperTradingEngine] = {}
_auto_executors_by_user: dict[str, AutoExecutor] = {}
_LEGACY_PAPER_FILE = Path("data") / "paper_trading.json"
_LEGACY_USER_KEYS = {"local-default", "default", "anonymous"}


def _safe_user_key(user_id: str) -> str:
    raw = str(user_id or "").strip()
    if not raw:
        return "anonymous"
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", raw)
    return safe[:120]


def _paper_state_path_for_user(user_id: str) -> Path:
    key = _safe_user_key(user_id)
    user_scoped_path = Path("data") / "paper_trading_users" / f"{key}.json"
    if key not in _LEGACY_USER_KEYS:
        return user_scoped_path

    # Backward-compatibility path chooser for anonymous/local users:
    # - Prefer whichever state file already has trades/open positions.
    # - Fall back to the existing file when both are empty.
    # This keeps older single-file installs working without hiding newer per-user data.
    if _paper_state_has_data(user_scoped_path):
        return user_scoped_path
    if _paper_state_has_data(_LEGACY_PAPER_FILE):
        return _LEGACY_PAPER_FILE
    if user_scoped_path.exists():
        return user_scoped_path
    return _LEGACY_PAPER_FILE


def _paper_state_has_data(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    closed = payload.get("closed_trades")
    if isinstance(closed, list) and len(closed) > 0:
        return True
    positions = payload.get("positions")
    if isinstance(positions, dict) and len(positions) > 0:
        return True
    stats = payload.get("stats")
    if isinstance(stats, dict):
        try:
            if int(stats.get("total_trades", 0) or 0) > 0:
                return True
        except Exception:
            return False
    return False


def get_paper_engine() -> PaperTradingEngine:
    global _paper_engine
    if _paper_engine is None:
        _paper_engine = PaperTradingEngine(initial_capital=100000.0)
    return _paper_engine


def get_auto_executor() -> AutoExecutor:
    global _auto_executor
    if _auto_executor is None:
        from src.utils.notifiers import NotificationManager

        notifier = NotificationManager({})
        _auto_executor = AutoExecutor(get_paper_engine(), notifier)
    return _auto_executor


def get_user_paper_engine(user_id: str, initial_capital: float = 100000.0) -> PaperTradingEngine:
    key = _safe_user_key(user_id)
    engine = _paper_engines_by_user.get(key)
    if engine is None:
        engine = PaperTradingEngine(
            initial_capital=float(initial_capital),
            data_file=_paper_state_path_for_user(key),
        )
        _paper_engines_by_user[key] = engine
    return engine


def get_user_auto_executor(user_id: str) -> AutoExecutor:
    key = _safe_user_key(user_id)
    executor = _auto_executors_by_user.get(key)
    if executor is None:
        from src.utils.notifiers import NotificationManager

        notifier = NotificationManager({})
        executor = AutoExecutor(get_user_paper_engine(key), notifier)
        _auto_executors_by_user[key] = executor
    return executor


__all__ = [
    "PaperTradingEngine",
    "AutoExecutor",
    "get_paper_engine",
    "get_auto_executor",
    "get_user_paper_engine",
    "get_user_auto_executor",
]
