"""
Factor confluence scoring — standalone; does not import the signal pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConfluenceScorer:
    """
    Measures how many factors agree with signal direction.
    NEVER modifies signals — only scores and annotates.
    """

    GRADE_CONFIG = {
        "A": {"min_pct": 75, "kelly_multiplier": 1.00},
        "B": {"min_pct": 55, "kelly_multiplier": 0.60},
        "C": {"min_pct": 40, "kelly_multiplier": 0.35},
        "D": {"min_pct": 0, "kelly_multiplier": 0.00},
    }

    def _contribution(self, entry: Any) -> float:
        try:
            if isinstance(entry, dict):
                return float(entry.get("contribution", 0.0))
            return float(entry)
        except Exception:
            return 0.0

    def score(self, factor_breakdown: dict, direction: str) -> dict:
        """
        factor_breakdown: factor_name → contribution float OR dict with 'contribution'.
        direction: "LONG" or "SHORT"
        """
        try:
            d = (direction or "").strip().upper()
            if d in {"BUY"}:
                d = "LONG"
            if d in {"SELL"}:
                d = "SHORT"
            if d not in {"LONG", "SHORT"}:
                d = "LONG"

            if not factor_breakdown:
                return {
                    "confluence_pct": 0.0,
                    "aligned_factors": 0,
                    "drag_factors": 0,
                    "total_factors": 0,
                    "grade": "UNKNOWN",
                    "kelly_multiplier": 1.0,
                    "fire_signal": True,
                    "top_aligned": [],
                    "top_drags": [],
                    "computation_ok": True,
                }

            items: list[tuple[str, float]] = []
            for name, raw in factor_breakdown.items():
                c = self._contribution(raw)
                items.append((str(name), c))

            total_factors = len(items)
            aligned: list[tuple[str, float]] = []
            drags: list[tuple[str, float]] = []
            for name, c in items:
                if d == "LONG":
                    if c > 0:
                        aligned.append((name, c))
                    elif c < 0:
                        drags.append((name, abs(c)))
                else:
                    if c < 0:
                        aligned.append((name, abs(c)))
                    elif c > 0:
                        drags.append((name, c))

            aligned_factors = len(aligned)
            drag_factors = len(drags)
            confluence_pct = (
                100.0 * aligned_factors / total_factors if total_factors else 0.0
            )

            grade = "D"
            kelly_mult = 0.0
            for g in ("A", "B", "C", "D"):
                cfg = self.GRADE_CONFIG[g]
                if confluence_pct >= cfg["min_pct"]:
                    grade = g
                    kelly_mult = float(cfg["kelly_multiplier"])
                    break

            aligned.sort(key=lambda x: -abs(x[1]))
            drags.sort(key=lambda x: -abs(x[1]))
            top_aligned = [n for n, _ in aligned[:3]]
            top_drags = [n for n, _ in drags[:2]]

            fire_signal = grade != "D"

            return {
                "confluence_pct": round(float(confluence_pct), 2),
                "aligned_factors": int(aligned_factors),
                "drag_factors": int(drag_factors),
                "total_factors": int(total_factors),
                "grade": grade,
                "kelly_multiplier": float(kelly_mult),
                "fire_signal": bool(fire_signal),
                "top_aligned": top_aligned,
                "top_drags": top_drags,
                "computation_ok": True,
            }
        except Exception as exc:
            logger.warning("ConfluenceScorer.score failed: %s", exc)
            return {
                "confluence_pct": 0.0,
                "aligned_factors": 0,
                "drag_factors": 0,
                "total_factors": 0,
                "grade": "UNKNOWN",
                "kelly_multiplier": 1.0,
                "fire_signal": True,
                "top_aligned": [],
                "top_drags": [],
                "computation_ok": False,
            }
