from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_EDGE_PRIORS = {
    'mtf_3_alignment': 0.58,
    'ob_retest_with_fvg': 0.55,
    'absorption_confirming': 0.62,
    'cvd_confirming': 0.56,
    'funding_against_crowd': 0.58,
    'london_session': 0.57,
    'macro_risk_on': 0.53,
    'stacked_imbalance_confirming': 0.60,
    'liquidity_sweep_confirming': 0.59,
    'breakout_retest': 0.57,
}


@dataclass
class StackResult:
    stacked_probability: float
    independent_edges: dict[str, float]


class ProbabilityStacker:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stats: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.stats = json.loads(self.path.read_text(encoding='utf-8'))
        except Exception:
            self.stats = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.stats, indent=2), encoding='utf-8')

    def record_outcome(self, factors: list[str], was_win: bool, as_of_utc: str) -> None:
        for f in factors:
            self.stats.setdefault(f, []).append({'ts': as_of_utc, 'win': bool(was_win)})
            # keep rolling 90 days only
            cutoff = datetime.now(timezone.utc) - timedelta(days=90)
            fresh = []
            for row in self.stats[f][-1500:]:
                ts = str(row.get('ts', ''))
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except Exception:
                    dt = datetime.now(timezone.utc)
                if dt >= cutoff:
                    fresh.append(row)
            self.stats[f] = fresh
        self._save()

    def current_edges(self) -> dict[str, float]:
        edges = dict(DEFAULT_EDGE_PRIORS)
        for f, rows in self.stats.items():
            if len(rows) < 10:
                continue
            wins = sum(1 for r in rows if bool(r.get('win')))
            wr = wins / max(len(rows), 1)
            # keep only factors with positive edge in rolling window.
            if wr >= 0.5:
                edges[f] = wr
            elif f in edges:
                edges.pop(f, None)
        return edges

    def stack(self, factors: list[str]) -> StackResult:
        edges = self.current_edges()
        active = {f: float(edges.get(f, DEFAULT_EDGE_PRIORS.get(f, 0.5))) for f in factors}
        product = 1.0
        for p in active.values():
            product *= (1.0 - p * 0.7)
        stacked = 1.0 - product
        return StackResult(stacked_probability=max(0.0, min(1.0, stacked)), independent_edges=active)
