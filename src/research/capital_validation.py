"""
Gap 5 — Live Capital Validation Framework.

Statistical framework for graduating from paper trading to live capital
deployment.  Implements:

  ①  PaperToLiveGraduator — multi-stage ramp with statistical gates
  ②  LivePerformanceTracker — running metrics vs paper baseline

Logic loosely based on:
    - De Prado "Advances in Financial Machine Learning" Ch. 14
    - Prop-desk risk frameworks (10% → 25% → 50% → 100% ramp)
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Graduation Framework
# ------------------------------------------------------------------

@dataclass
class GraduationStage:
    """One ramp stage in paper → live transition."""
    name: str
    capital_pct: float          # e.g. 0.10 = 10% of full capital
    min_days: int               # minimum days at this stage
    min_sharpe: float           # minimum rolling Sharpe to advance
    max_drawdown_pct: float     # maximum drawdown allowed


DEFAULT_STAGES = [
    GraduationStage("Paper Validation",  0.0,    30, 1.0,  10.0),
    GraduationStage("Seed Capital",      0.10,   14, 0.8,   5.0),
    GraduationStage("Quarter Capital",   0.25,   21, 0.6,   7.0),
    GraduationStage("Half Capital",      0.50,   21, 0.5,   8.0),
    GraduationStage("Full Capital",      1.00,    0, 0.0,  10.0),
]


class PaperToLiveGraduator:
    """
    Manages the transition from paper trading to full live deployment.

    Process
    -------
    1. Paper must pass statistical tests first (Sharpe, KS-test)
    2. Then capital ramps through stages: 10% → 25% → 50% → 100%
    3. At each stage, performance must meet minimums before advancing
    4. Divergence circuit breaker: if live Sharpe deviates from paper by > 2σ,
       the system drops back one stage

    Usage
    -----
    grad = PaperToLiveGraduator()
    grad.record_daily_return(0.005, is_live=True)
    report = grad.evaluate()
    if report["can_advance"]:
        grad.advance()
    """

    def __init__(
        self,
        stages: list[GraduationStage] | None = None,
        divergence_sigma: float = 2.0,
        min_paper_days: int = 30,
        min_paper_sharpe: float = 1.0,
    ):
        self.stages = stages or list(DEFAULT_STAGES)
        self.divergence_sigma = divergence_sigma
        self.min_paper_days = min_paper_days
        self.min_paper_sharpe = min_paper_sharpe

        self._current_stage_idx: int = 0
        self._stage_start_date: date = date.today()
        self._paper_returns: list[float] = []
        self._live_returns: list[float] = []
        self._daily_log: list[dict] = []

    @property
    def current_stage(self) -> GraduationStage:
        return self.stages[self._current_stage_idx]

    @property
    def capital_fraction(self) -> float:
        """Current fraction of full capital to deploy."""
        return self.current_stage.capital_pct

    @property
    def is_paper_phase(self) -> bool:
        return self._current_stage_idx == 0

    @property
    def is_fully_deployed(self) -> bool:
        return self._current_stage_idx >= len(self.stages) - 1

    def record_daily_return(self, daily_return: float, is_live: bool = False):
        """Record one day's return."""
        entry = {
            "date": str(date.today()),
            "return": daily_return,
            "stage": self.current_stage.name,
            "is_live": is_live,
        }
        self._daily_log.append(entry)

        if is_live:
            self._live_returns.append(daily_return)
        else:
            self._paper_returns.append(daily_return)

    def evaluate(self) -> dict:
        """
        Evaluate whether the system should advance, hold, or retreat.

        Returns a dict with:
            stage, capital_pct, can_advance, should_retreat, metrics, reasons
        """
        stage = self.current_stage
        days_in_stage = (date.today() - self._stage_start_date).days

        result = {
            "stage_name": stage.name,
            "stage_index": self._current_stage_idx,
            "capital_pct": stage.capital_pct,
            "days_in_stage": days_in_stage,
            "can_advance": False,
            "should_retreat": False,
            "reasons": [],
            "metrics": {},
        }

        # Paper phase evaluation
        if self.is_paper_phase:
            return self._evaluate_paper_phase(result)

        # Live phase evaluation
        return self._evaluate_live_phase(result, days_in_stage)

    def advance(self) -> bool:
        """Move to next stage. Returns True if advancement happened."""
        if self._current_stage_idx >= len(self.stages) - 1:
            return False
        self._current_stage_idx += 1
        self._stage_start_date = date.today()
        logger.info(
            "Graduated to stage '%s' (%.0f%% capital)",
            self.current_stage.name, self.current_stage.capital_pct * 100,
        )
        return True

    def retreat(self) -> bool:
        """Drop back one stage due to performance divergence."""
        if self._current_stage_idx <= 0:
            return False
        old_stage = self.current_stage.name
        self._current_stage_idx -= 1
        self._stage_start_date = date.today()
        logger.warning(
            "RETREATED from '%s' to '%s' (%.0f%% capital)",
            old_stage, self.current_stage.name,
            self.current_stage.capital_pct * 100,
        )
        return True

    def to_dict(self) -> dict:
        """Full state for API / persistence."""
        report = self.evaluate()
        report["paper_days"] = len(self._paper_returns)
        report["live_days"] = len(self._live_returns)
        report["total_stages"] = len(self.stages)
        report["stages"] = [
            {
                "name": s.name,
                "capital_pct": s.capital_pct,
                "min_days": s.min_days,
                "min_sharpe": s.min_sharpe,
                "max_drawdown_pct": s.max_drawdown_pct,
                "status": "completed" if i < self._current_stage_idx
                          else "active" if i == self._current_stage_idx
                          else "pending",
            }
            for i, s in enumerate(self.stages)
        ]
        return report

    def _evaluate_paper_phase(self, result: dict) -> dict:
        returns = np.array(self._paper_returns)
        n = len(returns)

        result["metrics"]["paper_days"] = n
        result["metrics"]["paper_sharpe"] = 0.0

        if n < self.min_paper_days:
            result["reasons"].append(
                f"Need {self.min_paper_days - n} more paper trading days"
            )
            return result

        sharpe = self._compute_sharpe(returns)
        result["metrics"]["paper_sharpe"] = round(sharpe, 3)

        if sharpe < self.min_paper_sharpe:
            result["reasons"].append(
                f"Paper Sharpe {sharpe:.3f} < {self.min_paper_sharpe:.1f}"
            )
            return result

        # Statistical test: returns significantly > 0
        if n >= 20:
            t_stat, p_val = stats.ttest_1samp(returns, 0)
            result["metrics"]["t_stat"] = round(float(t_stat), 3)
            result["metrics"]["p_value"] = round(float(p_val), 4)
            if p_val > 0.10 or t_stat <= 0:
                result["reasons"].append(
                    f"Returns not statistically significant (p={p_val:.4f})"
                )
                return result

        result["can_advance"] = True
        result["reasons"].append("Paper validation passed — ready for seed capital")
        return result

    def _evaluate_live_phase(self, result: dict, days_in_stage: int) -> dict:
        stage = self.current_stage
        returns = np.array(self._live_returns[-max(days_in_stage, 5):])

        if len(returns) < 3:
            result["reasons"].append("Accumulating live data...")
            return result

        sharpe = self._compute_sharpe(returns)
        drawdown = self._compute_drawdown(returns)

        result["metrics"]["live_sharpe"] = round(sharpe, 3)
        result["metrics"]["live_drawdown_pct"] = round(drawdown * 100, 2)
        result["metrics"]["days_remaining"] = max(0, stage.min_days - days_in_stage)

        # 1. Check drawdown
        if drawdown * 100 > stage.max_drawdown_pct:
            result["should_retreat"] = True
            result["reasons"].append(
                f"Drawdown {drawdown*100:.1f}% > {stage.max_drawdown_pct}% limit"
            )
            return result

        # 2. Check paper vs live divergence
        if len(self._paper_returns) > 20 and len(self._live_returns) > 10:
            divergence = self._check_divergence()
            result["metrics"]["divergence"] = divergence
            if divergence is not None and divergence.get("divergent"):
                result["should_retreat"] = True
                result["reasons"].append(
                    f"Live-paper divergence detected (KS p={divergence.get('ks_pvalue', 0):.4f})"
                )
                return result

        # 3. Can advance?
        if days_in_stage >= stage.min_days and sharpe >= stage.min_sharpe:
            result["can_advance"] = True
            result["reasons"].append(
                f"Stage requirements met (Sharpe={sharpe:.2f}, {days_in_stage}d)"
            )
        else:
            if days_in_stage < stage.min_days:
                result["reasons"].append(
                    f"Need {stage.min_days - days_in_stage} more days"
                )
            if sharpe < stage.min_sharpe:
                result["reasons"].append(
                    f"Sharpe {sharpe:.2f} < {stage.min_sharpe:.1f}"
                )

        return result

    def _check_divergence(self) -> Optional[dict]:
        """KS test: paper vs live return distributions."""
        paper = np.array(self._paper_returns[-60:])
        live = np.array(self._live_returns[-30:])

        if len(paper) < 10 or len(live) < 5:
            return None

        ks_stat, ks_p = stats.ks_2samp(paper, live)
        paper_sharpe = self._compute_sharpe(paper)
        live_sharpe = self._compute_sharpe(live)
        sharpe_diff = abs(paper_sharpe - live_sharpe)
        paper_std = np.std(paper) * np.sqrt(252) if len(paper) > 1 else 1.0

        return {
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "paper_sharpe": round(paper_sharpe, 3),
            "live_sharpe": round(live_sharpe, 3),
            "sharpe_diff": round(sharpe_diff, 3),
            "divergent": ks_p < 0.05 and sharpe_diff > self.divergence_sigma * paper_std,
        }

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, risk_free: float = 0.05) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / 252
        std = np.std(excess)
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    @staticmethod
    def _compute_drawdown(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / np.maximum(peak, 1e-9)
        return float(np.max(dd))


# ------------------------------------------------------------------
# Live Performance Tracker
# ------------------------------------------------------------------

class LivePerformanceTracker:
    """
    Tracks running live performance metrics for dashboard display.

    Maintains:
        - Running Sharpe ratio
        - Running drawdown
        - Fill quality (actual vs expected slippage)
        - Comparison vs paper baseline
    """

    def __init__(self):
        self._returns: list[float] = []
        self._fill_quality: list[float] = []  # actual_slip / expected_slip
        self._dates: list[str] = []

    def record(self, daily_return: float, fill_quality: float = 1.0):
        """Record one day's live metrics."""
        self._returns.append(daily_return)
        self._fill_quality.append(fill_quality)
        self._dates.append(str(date.today()))

    def metrics(self) -> dict:
        """Return current running metrics."""
        returns = np.array(self._returns)
        if len(returns) < 2:
            return {
                "days": len(returns),
                "sharpe": 0.0,
                "cumulative_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_fill_quality": 1.0,
                "win_rate": 0.0,
            }

        cum = np.prod(1 + returns) - 1
        peak = np.maximum.accumulate(np.cumprod(1 + returns))
        dd = (peak - np.cumprod(1 + returns)) / np.maximum(peak, 1e-9)
        sharpe = PaperToLiveGraduator._compute_sharpe(returns)
        win_rate = float((returns > 0).mean())
        avg_fq = float(np.mean(self._fill_quality)) if self._fill_quality else 1.0

        return {
            "days": len(returns),
            "sharpe": round(sharpe, 3),
            "cumulative_return_pct": round(cum * 100, 2),
            "max_drawdown_pct": round(float(np.max(dd)) * 100, 2),
            "avg_fill_quality": round(avg_fq, 3),
            "win_rate": round(win_rate, 3),
            "annualised_return_pct": round(
                (np.prod(1 + returns) ** (252 / max(len(returns), 1)) - 1) * 100, 2
            ),
        }
