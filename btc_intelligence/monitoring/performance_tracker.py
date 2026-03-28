from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class MonitoringStats:
    recent_win_rate: float
    current_drawdown_pct: float
    portfolio_heat_pct: float
    auto_pause: bool
    loss_streak: int
    trades_count: int
    avg_rr: float
    avg_mae: float
    avg_mfe: float


class PerformanceTracker:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[float] = [1.0]
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
            self.trades = payload.get('trades', [])
            self.equity_curve = payload.get('equity_curve', [1.0])
        except Exception:
            self.trades = []
            self.equity_curve = [1.0]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps({'trades': self.trades[-3000:], 'equity_curve': self.equity_curve[-3000:]}, indent=2),
            encoding='utf-8',
        )

    def record_trade(self, trade: dict[str, Any]) -> None:
        row = dict(trade)
        row.setdefault('as_of_utc', datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'))
        self.trades.append(row)
        pnl = float(row.get('pnl_usd', 0.0))
        base = self.equity_curve[-1]
        self.equity_curve.append(base + pnl)
        self._save()

    def stats(self, portfolio_heat_pct: float = 0.0) -> MonitoringStats:
        recent = self.trades[-30:]
        wins = sum(1 for t in recent if float(t.get('pnl_usd', 0.0)) > 0)
        win_rate = wins / max(len(recent), 1) if recent else 0.0

        peak = max(self.equity_curve) if self.equity_curve else 1.0
        current = self.equity_curve[-1] if self.equity_curve else 1.0
        dd = ((peak - current) / peak * 100.0) if peak > 0 else 0.0

        streak = 0
        for t in reversed(self.trades):
            if float(t.get('pnl_usd', 0.0)) < 0:
                streak += 1
            else:
                break

        rr_vals = [float(t.get('rr_achieved', 0.0)) for t in self.trades[-50:] if float(t.get('rr_achieved', 0.0)) != 0]
        mae_vals = [float(t.get('mae', 0.0)) for t in self.trades[-50:] if 'mae' in t]
        mfe_vals = [float(t.get('mfe', 0.0)) for t in self.trades[-50:] if 'mfe' in t]

        return MonitoringStats(
            recent_win_rate=win_rate,
            current_drawdown_pct=dd,
            portfolio_heat_pct=portfolio_heat_pct,
            auto_pause=False,
            loss_streak=streak,
            trades_count=len(self.trades),
            avg_rr=(sum(rr_vals) / len(rr_vals)) if rr_vals else 0.0,
            avg_mae=(sum(mae_vals) / len(mae_vals)) if mae_vals else 0.0,
            avg_mfe=(sum(mfe_vals) / len(mfe_vals)) if mfe_vals else 0.0,
        )
