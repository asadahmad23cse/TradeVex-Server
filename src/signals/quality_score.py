"""
Contextual Signal Quality Score (SQS) — standalone scoring only.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SignalQualityScorer:
    """
    Computes Signal Quality Score (SQS) 0-100.
    Higher = better quality signal.
    NEVER modifies signals — scoring only.
    """

    WEIGHTS = {
        "alpha": 25,
        "confluence": 20,
        "regime": 20,
        "mtf": 15,
        "ic_recency": 10,
        "liquidity": 10,
    }

    THRESHOLDS = {
        "PREMIUM": {"min": 75, "size_multiplier": 1.00},
        "STANDARD": {"min": 55, "size_multiplier": 1.00},
        "WEAK": {"min": 40, "size_multiplier": 0.50},
        "SUPPRESSED": {"min": 0, "size_multiplier": 0.00},
    }

    @staticmethod
    def _confluence_subscore(grade: str | None) -> float:
        g = (grade or "UNKNOWN").strip().upper()
        # Map grade to 0-100 within the confluence component
        m = {"A": 100.0, "B": 75.0, "C": 50.0, "D": 0.0, "UNKNOWN": 50.0}
        return float(m.get(g, 50.0))

    def compute_sqs(self, context: dict) -> dict:
        try:
            ctx = context or {}
            components: dict[str, float] = {}
            skipped: list[str] = []
            weighted_sum = 0.0
            weight_total = 0.0

            if ctx.get("alpha_score") is not None:
                try:
                    a = abs(float(ctx["alpha_score"]))
                    sub = float(min(1.0, max(0.0, a / 0.8)) * 100.0)
                    w = self.WEIGHTS["alpha"]
                    weighted_sum += w * sub
                    weight_total += w
                    components["alpha"] = round(sub, 2)
                except Exception:
                    skipped.append("alpha")
            else:
                skipped.append("alpha")

            if ctx.get("confluence_grade") is not None:
                try:
                    sub = self._confluence_subscore(str(ctx["confluence_grade"]))
                    w = self.WEIGHTS["confluence"]
                    weighted_sum += w * sub
                    weight_total += w
                    components["confluence"] = round(sub, 2)
                except Exception:
                    skipped.append("confluence")
            else:
                skipped.append("confluence")

            if ctx.get("regime_confidence") is not None:
                try:
                    r = float(ctx["regime_confidence"])
                    r = min(1.0, max(0.0, r))
                    sub = r * 100.0
                    w = self.WEIGHTS["regime"]
                    weighted_sum += w * sub
                    weight_total += w
                    components["regime"] = round(sub, 2)
                except Exception:
                    skipped.append("regime")
            else:
                skipped.append("regime")

            if ctx.get("mtf_aligned") is not None:
                try:
                    sub = 100.0 if bool(ctx["mtf_aligned"]) else 0.0
                    w = self.WEIGHTS["mtf"]
                    weighted_sum += w * sub
                    weight_total += w
                    components["mtf"] = round(sub, 2)
                except Exception:
                    skipped.append("mtf")
            else:
                skipped.append("mtf")

            if ctx.get("factor_ic_recency") is not None:
                try:
                    icr = float(ctx["factor_ic_recency"])
                    icr = min(1.0, max(0.0, icr))
                    sub = icr * 100.0
                    w = self.WEIGHTS["ic_recency"]
                    weighted_sum += w * sub
                    weight_total += w
                    components["ic_recency"] = round(sub, 2)
                except Exception:
                    skipped.append("ic_recency")
            else:
                skipped.append("ic_recency")

            if ctx.get("liquidity_score") is not None:
                try:
                    liq = float(ctx["liquidity_score"])
                    liq = min(1.0, max(0.0, liq))
                    sub = liq * 100.0
                    w = self.WEIGHTS["liquidity"]
                    weighted_sum += w * sub
                    weight_total += w
                    components["liquidity"] = round(sub, 2)
                except Exception:
                    skipped.append("liquidity")
            else:
                skipped.append("liquidity")

            if weight_total <= 0:
                return {
                    "sqs": 50,
                    "grade": "STANDARD",
                    "fire": True,
                    "size_multiplier": 1.0,
                    "components": {},
                    "skipped_components": skipped,
                    "computation_ok": True,
                    "suppression_reason": None,
                }

            sqs = int(round(weighted_sum / weight_total))
            sqs = max(0, min(100, sqs))

            if sqs >= self.THRESHOLDS["PREMIUM"]["min"]:
                gname = "PREMIUM"
                sm = self.THRESHOLDS["PREMIUM"]["size_multiplier"]
            elif sqs >= self.THRESHOLDS["STANDARD"]["min"]:
                gname = "STANDARD"
                sm = self.THRESHOLDS["STANDARD"]["size_multiplier"]
            elif sqs >= self.THRESHOLDS["WEAK"]["min"]:
                gname = "WEAK"
                sm = self.THRESHOLDS["WEAK"]["size_multiplier"]
            else:
                gname = "SUPPRESSED"
                sm = self.THRESHOLDS["SUPPRESSED"]["size_multiplier"]

            fire = sqs >= self.THRESHOLDS["WEAK"]["min"]
            reason = None
            if not fire:
                reason = f"SQS {sqs} below minimum {self.THRESHOLDS['WEAK']['min']}"

            return {
                "sqs": sqs,
                "grade": gname,
                "fire": fire,
                "size_multiplier": float(sm),
                "components": components,
                "skipped_components": skipped,
                "computation_ok": True,
                "suppression_reason": reason,
            }
        except Exception as exc:
            logger.warning("compute_sqs failed: %s", exc)
            return {
                "sqs": 50,
                "grade": "STANDARD",
                "fire": True,
                "size_multiplier": 1.0,
                "components": {},
                "skipped_components": [],
                "computation_ok": False,
                "suppression_reason": None,
            }
