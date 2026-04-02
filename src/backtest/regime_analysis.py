"""
Regime-stratified performance analytics (standalone; no signal pipeline imports).

Consumes completed trade dicts with optional regime_at_entry; never mutates inputs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RegimePerformanceTracker:
    """
    Purely analytical — never modifies signals or config.
    Receives completed trade list, tags by regime, computes stats.
    """

    VALID_REGIMES = [
        "BULL",
        "HIGH_VOL_BULL",
        "SIDEWAYS",
        "BEAR",
        "HIGH_VOL_BEAR",
    ]
    MIN_TRADES_FOR_STATS = 5

    def tag_trade(self, trade: dict, regime_at_entry: str) -> dict:
        """
        Add regime_at_entry field to trade dict.
        Returns copy of trade — never mutates original.
        Safe default: if regime invalid, tag as UNKNOWN.
        """
        try:
            out = dict(trade)
            r = str(regime_at_entry).strip().upper() if regime_at_entry is not None else ""
            out["regime_at_entry"] = r if r in self.VALID_REGIMES else "UNKNOWN"
            return out
        except Exception:
            return {"regime_at_entry": "UNKNOWN"}

    def _empty_skeleton(self) -> dict[str, Any]:
        return {
            "by_regime": {},
            "optimal_regimes": [],
            "weak_regimes": [],
            "insufficient_data_regimes": [],
            "overall_alpha_score": 0.0,
            "summary": "",
        }

    def _recommendation(self, trade_count: int, win_rate: float, avg_rr: float) -> str:
        try:
            if trade_count < self.MIN_TRADES_FOR_STATS:
                return "INSUFFICIENT_DATA"
            wr = float(win_rate)
            rr = float(avg_rr)
            if wr >= 0.55 and rr >= 1.5:
                return "KEEP"
            if wr >= 0.45 and rr >= 1.0:
                return "REDUCE_SIZE"
            return "REVIEW"
        except Exception:
            return "REVIEW"

    def _avg_rr_from_pnls(self, pnls: list[float], outcomes: list[str]) -> float:
        try:
            wins = [pnls[i] for i, o in enumerate(outcomes) if o == "WIN"]
            losses = [abs(pnls[i]) for i, o in enumerate(outcomes) if o == "LOSS"]
            aw = float(np.mean(wins)) if wins else 0.0
            al = float(np.mean(losses)) if losses else 0.0
            if al < 1e-9:
                return float(aw) if aw > 0 else 0.0
            return aw / al
        except Exception:
            return 0.0

    def _one_regime_block(
        self,
        regime: str,
        pnls: list[float],
        outcomes: list[str],
        hold_bars: list[float],
    ) -> dict[str, Any]:
        n = len(pnls)
        wins = sum(1 for o in outcomes if o == "WIN")
        win_rate = wins / n if n else 0.0
        avg_rr = self._avg_rr_from_pnls(pnls, outcomes)
        avg_pnl_pct = float(np.mean(pnls)) if n else 0.0
        total_pnl_pct = float(np.sum(pnls)) if n else 0.0

        rets = np.array([p / 100.0 for p in pnls], dtype=float)
        sharpe = 0.0
        if n >= self.MIN_TRADES_FOR_STATS:
            try:
                std = float(np.std(rets))
                if std > 1e-12:
                    sharpe = float(np.mean(rets) / std * np.sqrt(252))
            except Exception:
                sharpe = 0.0

        max_dd = 0.0
        if n > 0:
            try:
                eq = np.cumsum(pnls)
                peak = np.maximum.accumulate(eq)
                trough = eq
                dd = (trough - peak) / np.maximum(np.abs(peak), 1e-9)
                max_dd = float(dd.min()) if len(dd) else 0.0
            except Exception:
                max_dd = 0.0

        avg_hold = float(np.mean(hold_bars)) if hold_bars else 0.0
        best_t = float(max(pnls)) if pnls else 0.0
        worst_t = float(min(pnls)) if pnls else 0.0
        rec = self._recommendation(n, win_rate, avg_rr)

        return {
            "trade_count": n,
            "win_rate": win_rate,
            "avg_rr": avg_rr,
            "avg_pnl_pct": avg_pnl_pct,
            "total_pnl_pct": total_pnl_pct,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "avg_hold_bars": avg_hold,
            "best_trade_pct": best_t,
            "worst_trade_pct": worst_t,
            "recommendation": rec,
        }

    def compute_regime_stats(self, trades: list) -> dict[str, Any]:
        """
        Input: list of trades, each having 'regime_at_entry' field (or tag first).
        """
        try:
            if not trades:
                return self._empty_skeleton()

            by_bucket: dict[str, dict[str, list]] = {}
            for t in trades:
                try:
                    if not isinstance(t, dict):
                        continue
                    raw = t.get("regime_at_entry", t.get("regime", "UNKNOWN"))
                    reg = str(raw).strip().upper() if raw is not None else "UNKNOWN"
                    if reg not in self.VALID_REGIMES and reg != "UNKNOWN":
                        reg = "UNKNOWN"
                    pnl = t.get("net_pnl_pct", t.get("pnl_pct", 0.0))
                    pnl_f = float(pnl) if pnl is not None else 0.0
                    oc = str(t.get("outcome", t.get("result", "LOSS"))).upper()
                    if oc not in ("WIN", "LOSS"):
                        oc = "WIN" if pnl_f > 0 else "LOSS"
                    hb = float(t.get("hold_bars", t.get("hold_days", 1)) or 1.0)
                    by_bucket.setdefault(reg, {"pnl": [], "out": [], "hold": []})
                    by_bucket[reg]["pnl"].append(pnl_f)
                    by_bucket[reg]["out"].append(oc)
                    by_bucket[reg]["hold"].append(hb)
                except Exception:
                    continue

            by_regime: dict[str, Any] = {}
            for reg in self.VALID_REGIMES:
                try:
                    if reg in by_bucket:
                        b = by_bucket[reg]
                        by_regime[reg] = self._one_regime_block(
                            reg, b["pnl"], b["out"], b["hold"]
                        )
                    else:
                        by_regime[reg] = {
                            "trade_count": 0,
                            "win_rate": 0.0,
                            "avg_rr": 0.0,
                            "avg_pnl_pct": 0.0,
                            "total_pnl_pct": 0.0,
                            "sharpe": 0.0,
                            "max_drawdown": 0.0,
                            "avg_hold_bars": 0.0,
                            "best_trade_pct": 0.0,
                            "worst_trade_pct": 0.0,
                            "recommendation": "INSUFFICIENT_DATA",
                        }
                except Exception:
                    by_regime[reg] = {"recommendation": "COMPUTATION_ERROR"}

            if "UNKNOWN" in by_bucket:
                try:
                    b = by_bucket["UNKNOWN"]
                    by_regime["UNKNOWN"] = self._one_regime_block(
                        "UNKNOWN", b["pnl"], b["out"], b["hold"]
                    )
                except Exception:
                    by_regime["UNKNOWN"] = {"recommendation": "COMPUTATION_ERROR"}

            optimal = [r for r, s in by_regime.items() if s.get("recommendation") == "KEEP"]
            weak = [r for r, s in by_regime.items() if s.get("recommendation") == "REVIEW"]
            insuf = [
                r
                for r, s in by_regime.items()
                if s.get("recommendation") == "INSUFFICIENT_DATA"
            ]

            total_n = sum(
                int(by_regime[r].get("trade_count", 0))
                for r in by_regime
                if isinstance(by_regime[r], dict)
            )
            overall = 0.0
            if total_n > 0:
                try:
                    acc = 0.0
                    for r, s in by_regime.items():
                        if not isinstance(s, dict):
                            continue
                        n_i = int(s.get("trade_count", 0))
                        if n_i <= 0:
                            continue
                        acc += float(s.get("win_rate", 0.0)) * n_i
                    overall = acc / total_n
                except Exception:
                    overall = 0.0

            parts = [
                f"{r}: {by_regime[r].get('recommendation', '?')} ({by_regime[r].get('trade_count', 0)} trades)"
                for r in self.VALID_REGIMES
                if r in by_regime and isinstance(by_regime[r], dict)
            ]
            summary = (
                f"Weighted win rate {overall:.1%}; KEEP in {optimal or 'none'}; "
                f"review {weak or 'none'}."
            )
            if parts:
                summary = summary + " " + "; ".join(parts[:5])

            return {
                "by_regime": by_regime,
                "optimal_regimes": optimal,
                "weak_regimes": weak,
                "insufficient_data_regimes": insuf,
                "overall_alpha_score": overall,
                "summary": summary[:2000],
            }
        except Exception as exc:
            logger.warning("compute_regime_stats failed: %s", exc)
            return self._empty_skeleton()

    def aggregate_folds(self, fold_results: list) -> dict[str, Any]:
        """
        Takes list of per-fold regime_stats dicts from WFO.
        Combines them weighted by trade_count per fold.
        """
        try:
            if not fold_results:
                return self._empty_skeleton()

            regime_totals: dict[str, dict[str, float]] = {}
            regime_weight: dict[str, float] = {}

            for fold in fold_results:
                try:
                    if not isinstance(fold, dict):
                        continue
                    br = fold.get("by_regime", {})
                    if not isinstance(br, dict):
                        continue
                    for reg, stats in br.items():
                        if not isinstance(stats, dict):
                            continue
                        n = float(stats.get("trade_count", 0) or 0)
                        if n <= 0:
                            continue
                        regime_weight[reg] = regime_weight.get(reg, 0.0) + n
                        if reg not in regime_totals:
                            regime_totals[reg] = {
                                "win_rate": 0.0,
                                "avg_rr": 0.0,
                                "avg_pnl_pct": 0.0,
                                "total_pnl_pct": 0.0,
                                "sharpe": 0.0,
                                "max_drawdown_wsum": 0.0,
                                "avg_hold_bars": 0.0,
                                "best_trade_pct": -1e18,
                                "worst_trade_pct": 1e18,
                            }
                        acc = regime_totals[reg]
                        acc["win_rate"] += float(stats.get("win_rate", 0.0)) * n
                        acc["avg_rr"] += float(stats.get("avg_rr", 0.0)) * n
                        acc["avg_pnl_pct"] += float(stats.get("avg_pnl_pct", 0.0)) * n
                        acc["total_pnl_pct"] += float(stats.get("total_pnl_pct", 0.0))
                        acc["sharpe"] += float(stats.get("sharpe", 0.0)) * n
                        acc["max_drawdown_wsum"] += (
                            float(stats.get("max_drawdown", 0.0)) * n
                        )
                        acc["avg_hold_bars"] += float(stats.get("avg_hold_bars", 0.0)) * n
                        acc["best_trade_pct"] = max(
                            acc["best_trade_pct"],
                            float(stats.get("best_trade_pct", 0.0)),
                        )
                        acc["worst_trade_pct"] = min(
                            acc["worst_trade_pct"],
                            float(stats.get("worst_trade_pct", 0.0)),
                        )
                except Exception:
                    continue

            for reg in list(regime_totals.keys()):
                acc = regime_totals[reg]
                w = regime_weight.get(reg, 0.0)
                if w <= 0:
                    continue
                acc["win_rate"] /= w
                acc["avg_rr"] /= w
                acc["avg_pnl_pct"] /= w
                acc["sharpe"] /= w
                acc["avg_hold_bars"] /= w
                acc["max_drawdown"] = acc["max_drawdown_wsum"] / w
                acc.pop("max_drawdown_wsum", None)
                if acc["best_trade_pct"] < -1e17:
                    acc["best_trade_pct"] = 0.0
                if acc["worst_trade_pct"] > 1e17:
                    acc["worst_trade_pct"] = 0.0

            by_regime: dict[str, Any] = {}
            for reg in self.VALID_REGIMES:
                try:
                    w = regime_weight.get(reg, 0.0)
                    if w <= 0:
                        by_regime[reg] = {
                            "trade_count": 0,
                            "win_rate": 0.0,
                            "avg_rr": 0.0,
                            "avg_pnl_pct": 0.0,
                            "total_pnl_pct": 0.0,
                            "sharpe": 0.0,
                            "max_drawdown": 0.0,
                            "avg_hold_bars": 0.0,
                            "best_trade_pct": 0.0,
                            "worst_trade_pct": 0.0,
                            "recommendation": "INSUFFICIENT_DATA",
                        }
                        continue
                    acc = regime_totals[reg]
                    n = int(w)
                    rec = self._recommendation(
                        n, acc["win_rate"], acc["avg_rr"]
                    )
                    by_regime[reg] = {
                        "trade_count": n,
                        "win_rate": acc["win_rate"],
                        "avg_rr": acc["avg_rr"],
                        "avg_pnl_pct": acc["avg_pnl_pct"],
                        "total_pnl_pct": acc["total_pnl_pct"],
                        "sharpe": acc["sharpe"],
                        "max_drawdown": acc["max_drawdown"],
                        "avg_hold_bars": acc["avg_hold_bars"],
                        "best_trade_pct": acc["best_trade_pct"],
                        "worst_trade_pct": acc["worst_trade_pct"],
                        "recommendation": rec,
                    }
                except Exception:
                    by_regime[reg] = {"recommendation": "COMPUTATION_ERROR"}

            if "UNKNOWN" in regime_weight and regime_weight.get("UNKNOWN", 0) > 0:
                try:
                    reg = "UNKNOWN"
                    w = regime_weight[reg]
                    acc = regime_totals[reg]
                    n = int(w)
                    rec = self._recommendation(n, acc["win_rate"], acc["avg_rr"])
                    by_regime["UNKNOWN"] = {
                        "trade_count": n,
                        "win_rate": acc["win_rate"],
                        "avg_rr": acc["avg_rr"],
                        "avg_pnl_pct": acc["avg_pnl_pct"],
                        "total_pnl_pct": acc["total_pnl_pct"],
                        "sharpe": acc["sharpe"],
                        "max_drawdown": acc["max_drawdown"],
                        "avg_hold_bars": acc["avg_hold_bars"],
                        "best_trade_pct": acc["best_trade_pct"],
                        "worst_trade_pct": acc["worst_trade_pct"],
                        "recommendation": rec,
                    }
                except Exception:
                    by_regime["UNKNOWN"] = {"recommendation": "COMPUTATION_ERROR"}

            optimal = [r for r, s in by_regime.items() if s.get("recommendation") == "KEEP"]
            weak = [r for r, s in by_regime.items() if s.get("recommendation") == "REVIEW"]
            insuf = [
                r
                for r, s in by_regime.items()
                if s.get("recommendation") == "INSUFFICIENT_DATA"
            ]

            total_n = sum(int(by_regime[r].get("trade_count", 0)) for r in by_regime if isinstance(by_regime[r], dict))
            overall = 0.0
            if total_n > 0:
                try:
                    overall = sum(
                        float(by_regime[r].get("win_rate", 0.0))
                        * int(by_regime[r].get("trade_count", 0))
                        for r in by_regime
                        if isinstance(by_regime[r], dict)
                    ) / total_n
                except Exception:
                    overall = 0.0

            summary = (
                f"Aggregated {len(fold_results)} folds; weighted win rate {overall:.1%}; "
                f"KEEP: {optimal or 'none'}."
            )
            return {
                "by_regime": by_regime,
                "optimal_regimes": optimal,
                "weak_regimes": weak,
                "insufficient_data_regimes": insuf,
                "overall_alpha_score": overall,
                "summary": summary[:2000],
            }
        except Exception as exc:
            logger.warning("aggregate_folds failed: %s", exc)
            return self._empty_skeleton()
