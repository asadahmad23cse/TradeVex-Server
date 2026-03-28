from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CalibrationCheck:
    calibrated: bool
    reason: str
    confidence_gap: float


class AutoCorrector:
    """Simple self-calibration checks for confidence reliability and MAE stress."""

    def check(self, trades: list[dict[str, Any]]) -> CalibrationCheck:
        if len(trades) < 20:
            return CalibrationCheck(True, 'Not enough trades for calibration check', 0.0)

        bins: dict[str, list[int]] = {
            '60_70': [],
            '70_80': [],
            '80_90': [],
            '90_100': [],
        }
        for t in trades[-120:]:
            conf = float(t.get('confidence', 0.0))
            win = 1 if float(t.get('pnl_usd', 0.0)) > 0 else 0
            if 60 <= conf < 70:
                bins['60_70'].append(win)
            elif 70 <= conf < 80:
                bins['70_80'].append(win)
            elif 80 <= conf < 90:
                bins['80_90'].append(win)
            elif conf >= 90:
                bins['90_100'].append(win)

        gaps = []
        for key, rows in bins.items():
            if len(rows) < 5:
                continue
            lo, hi = key.split('_')
            predicted = (float(lo) + float(hi)) / 2.0 / 100.0
            actual = sum(rows) / len(rows)
            gaps.append(abs(predicted - actual))

        gap = max(gaps) if gaps else 0.0
        if gap > 0.2:
            return CalibrationCheck(False, 'Confidence calibration drift > 20%', gap)
        return CalibrationCheck(True, 'Calibration healthy', gap)
