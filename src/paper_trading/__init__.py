"""Paper trading singletons."""

from __future__ import annotations

import re
from pathlib import Path

from .auto_executor import AutoExecutor
from .paper_engine import PaperTradingEngine

_paper_engine: PaperTradingEngine | None = None
_auto_executor: AutoExecutor | None = None
_paper_engines_by_user: dict[str, PaperTradingEngine] = {}
_auto_executors_by_user: dict[str, AutoExecutor] = {}


def _safe_user_key(user_id: str) -> str:
    raw = str(user_id or "").strip()
    if not raw:
        return "anonymous"
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", raw)
    return safe[:120]


def _paper_state_path_for_user(user_id: str) -> Path:
    key = _safe_user_key(user_id)
    return Path("data") / "paper_trading_users" / f"{key}.json"


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
