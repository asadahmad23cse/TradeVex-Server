"""
GAP 4 — Factor IC Decay Monitor.

Tracks each factor's rolling Information Coefficient at three horizons:
    21-day  (short-term, tactical)
    63-day  (medium-term, strategic)
    126-day (long-term, structural)

Auto-decay rule:
    If IC_21d < -0.05 AND IC_63d < 0.0 → factor weight decays to 0.
    Weight recovers linearly over 10 days once IC_21d turns positive again.

This prevents paying Kelly-sized positions on dead alpha signals.

Usage:
    dm = FactorDecayMonitor()
    dm.update("F1", ic_21d=0.12, ic_63d=0.08, ic_126d=0.05)
    dm.update("F2", ic_21d=-0.08, ic_63d=-0.02, ic_126d=0.04)
    weights = dm.get_weights()  # {"F1": 1.0, "F2": 0.0, ...}

    # Call dm.update() each EOD cycle.
    # Pass weights into AlphaFactorModel.score() to modulate IC weights.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Thresholds for declaring factor decay
DECAY_IC_21D_THRESHOLD = -0.05   # IC_21d must be below this
DECAY_IC_63D_THRESHOLD = 0.0     # IC_63d must be below this (sustained)
RECOVERY_DAYS = 10               # bars until full weight recovery after IC turns positive


@dataclass
class FactorState:
    factor_name: str
    ic_21d: float = 0.0
    ic_63d: float = 0.0
    ic_126d: float = 0.0
    is_decayed: bool = False
    recovery_bar: int = 0            # count of consecutive positive IC bars since re-start
    last_updated: date = field(default_factory=date.today)
    weight: float = 1.0             # current weight [0.0, 1.0]
    decay_history: list = field(default_factory=list)


class FactorDecayMonitor:
    """
    Monitors IC at 3 horizons and adjusts factor weights.
    """

    def __init__(
        self,
        decay_ic_21d: float = DECAY_IC_21D_THRESHOLD,
        decay_ic_63d: float = DECAY_IC_63D_THRESHOLD,
        recovery_days: int = RECOVERY_DAYS,
    ):
        self.decay_ic_21d = decay_ic_21d
        self.decay_ic_63d = decay_ic_63d
        self.recovery_days = recovery_days
        self._states: dict[str, FactorState] = {}

    def update(
        self,
        factor: str,
        ic_21d: float,
        ic_63d: float = 0.0,
        ic_126d: float = 0.0,
    ) -> float:
        """
        Update IC values for a factor and recompute its weight.

        Returns
        -------
        float : new weight for this factor in [0.0, 1.0]
        """
        if factor not in self._states:
            self._states[factor] = FactorState(factor_name=factor)

        state = self._states[factor]
        state.ic_21d = ic_21d
        state.ic_63d = ic_63d
        state.ic_126d = ic_126d
        state.last_updated = date.today()

        # Decay detection
        if ic_21d < self.decay_ic_21d and ic_63d < self.decay_ic_63d:
            if not state.is_decayed:
                logger.warning(
                    "Factor decay detected: %s — IC_21d=%.4f IC_63d=%.4f → weight → 0",
                    factor, ic_21d, ic_63d
                )
                state.decay_history.append({
                    "date": state.last_updated.isoformat(),
                    "ic_21d": ic_21d,
                    "ic_63d": ic_63d,
                    "event": "DECAY_START",
                })
            state.is_decayed = True
            state.recovery_bar = 0
            state.weight = 0.0

        elif state.is_decayed:
            # Recovery phase: IC_21d has turned positive
            if ic_21d > 0.0:
                state.recovery_bar += 1
                # Linear recovery over `recovery_days` bars
                state.weight = min(state.recovery_bar / self.recovery_days, 1.0)
                if state.weight >= 1.0:
                    state.is_decayed = False
                    logger.info("Factor %s fully recovered after %d bars.", factor, state.recovery_bar)
                    state.decay_history.append({
                        "date": state.last_updated.isoformat(),
                        "ic_21d": ic_21d,
                        "event": "DECAY_END",
                    })
            else:
                state.weight = 0.0  # still decayed
        else:
            state.weight = 1.0

        return state.weight

    def get_weights(self) -> dict[str, float]:
        """Return current weight for all tracked factors."""
        return {f: s.weight for f, s in self._states.items()}

    def get_weight(self, factor: str) -> float:
        """Return weight for a single factor (1.0 if never tracked)."""
        return self._states.get(factor, FactorState(factor)).weight

    def get_status(self) -> list[dict]:
        """Return full status for all factors (for dashboard /api/factors)."""
        return [
            {
                "factor": s.factor_name,
                "ic_21d": round(s.ic_21d, 4),
                "ic_63d": round(s.ic_63d, 4),
                "ic_126d": round(s.ic_126d, 4),
                "weight": round(s.weight, 4),
                "is_decayed": s.is_decayed,
                "recovery_bar": s.recovery_bar,
                "last_updated": s.last_updated.isoformat(),
            }
            for s in self._states.values()
        ]

    def apply_weights(
        self,
        ic_weights: dict[str, float],
        *,
        options_coldstart_bars: int | None = None,
        bars_seen_by_ticker: dict[str, int] | None = None,
        ticker_key: str | None = None,
    ) -> dict[str, float]:
        """
        Multiply IC weights by decay weights.

        F15/F16/F17: while bars_seen <= options_coldstart_bars for this ticker,
        skip decay (treat decay multiplier as 1.0) so IC_21 gating does not
        zero them during the options cold-start window.
        """
        tk = (ticker_key or "").strip().upper()
        bars = 999_999
        if bars_seen_by_ticker is not None and tk:
            bars = int(bars_seen_by_ticker.get(tk, 999_999))
        out: dict[str, float] = {}
        for f, ic in ic_weights.items():
            if (
                f in ("F15", "F16", "F17")
                and options_coldstart_bars is not None
                and bars <= int(options_coldstart_bars)
            ):
                out[f] = float(ic)
                continue
            w = self.get_weight(f)
            out[f] = float(ic) * float(w)
        return out
