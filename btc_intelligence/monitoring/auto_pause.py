from __future__ import annotations

from dataclasses import dataclass

from btc_intelligence.config import settings
from btc_intelligence.monitoring.performance_tracker import MonitoringStats


@dataclass
class AutoPauseDecision:
    paused: bool
    reason: str


class AutoPauseManager:
    def evaluate(self, stats: MonitoringStats) -> AutoPauseDecision:
        if stats.current_drawdown_pct > settings.drawdown_pause_pct:
            return AutoPauseDecision(True, 'Drawdown > configured pause threshold')
        if stats.recent_win_rate > 0 and stats.recent_win_rate < settings.auto_pause_winrate_floor and stats.trades_count >= 30:
            return AutoPauseDecision(True, 'Recent win rate below floor')
        if stats.loss_streak >= settings.auto_pause_loss_streak:
            return AutoPauseDecision(True, 'Consecutive loss streak reached')
        return AutoPauseDecision(False, 'System active')
