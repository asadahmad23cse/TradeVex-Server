from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_weekly_report(stats: dict[str, Any], top_edges: dict[str, float]) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    edges = sorted(top_edges.items(), key=lambda x: x[1], reverse=True)[:6]
    edge_lines = '\n'.join(f'- {k}: {v:.2%}' for k, v in edges) if edges else '- no edge data'

    return (
        f'Weekly BTC Intelligence Report ({now})\n'
        f"Win rate: {float(stats.get('recent_win_rate', 0.0)):.2%}\n"
        f"Drawdown: {float(stats.get('current_drawdown_pct', 0.0)):.2f}%\n"
        f"Loss streak: {int(stats.get('loss_streak', 0))}\n"
        f"Portfolio heat: {float(stats.get('portfolio_heat_pct', 0.0)):.2f}%\n"
        'Top independent edges:\n'
        f'{edge_lines}'
    )
