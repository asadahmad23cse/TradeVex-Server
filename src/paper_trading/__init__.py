"""Paper trading singletons."""

from __future__ import annotations

from .auto_executor import AutoExecutor
from .paper_engine import PaperTradingEngine

_paper_engine: PaperTradingEngine | None = None
_auto_executor: AutoExecutor | None = None


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


__all__ = ["PaperTradingEngine", "AutoExecutor", "get_paper_engine", "get_auto_executor"]

