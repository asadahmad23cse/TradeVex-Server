from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import pstdev
from typing import Any


@dataclass
class KellyConfig:
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.02
    min_position_pct: float = 0.001
    drawdown_halt_pct: float = 0.08
    volatility_scalar: bool = True
    # Microstructure: dynamic payoff ratio R vs reference book.
    reference_spread_bps: float = 4.0
    reference_depth_notional_usd: float = 1_500_000.0
    execution_rr_weight: float = 0.42
    micro_r_clip: tuple[float, float] = (0.28, 6.0)
    # Tail risk: empirical CVaR (expected shortfall, left tail) on strategy pnl_% .
    cvar_alpha: float = 0.05  # 95% CVaR
    cvar_min_samples: int = 25
    cvar_equity_budget_pct: float = 0.012
    # Aggregate stress: (heat_frac + position_frac) * sigma_mult * sigma <= cap.
    stress_sigma_multiplier: float = 2.0
    stress_equity_loss_cap_pct: float = 0.015
    stress_min_samples: int = 20
    sigma_floor_pct: float = 0.08  # floor on σ (% pts) to avoid exploding caps


def book_microstructure_from_depth(depth: dict[str, Any] | None) -> dict[str, float]:
    """Bid/ask spread (bps) and shallow notionals for dynamic R."""
    if not isinstance(depth, dict):
        return {"spread_bps": 0.0, "bid_notional_usd": 0.0, "ask_notional_usd": 0.0}
    bids = depth.get("bids", []) or []
    asks = depth.get("asks", []) or []
    if not bids or not asks:
        return {"spread_bps": 0.0, "bid_notional_usd": 0.0, "ask_notional_usd": 0.0}
    try:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        spread_bps = ((best_ask - best_bid) / mid * 10_000.0) if mid > 0 else 100.0
        bid_notional = sum(float(px) * float(qty) for px, qty in bids[:12])
        ask_notional = sum(float(px) * float(qty) for px, qty in asks[:12])
        return {
            "spread_bps": float(max(spread_bps, 0.0)),
            "bid_notional_usd": float(max(bid_notional, 0.0)),
            "ask_notional_usd": float(max(ask_notional, 0.0)),
        }
    except Exception:
        return {"spread_bps": 0.0, "bid_notional_usd": 0.0, "ask_notional_usd": 0.0}


def _empirical_cvar_alpha(returns_pct: list[float], alpha: float) -> float | None:
    if len(returns_pct) < 5 or alpha <= 0.0 or alpha >= 1.0:
        return None
    x = sorted(returns_pct)
    k = max(1, int(math.ceil(alpha * len(x))))
    tail = x[:k]
    return float(sum(tail) / len(tail))


def _return_sigma_pct(returns_pct: list[float]) -> float | None:
    if len(returns_pct) < 2:
        return None
    try:
        return float(pstdev(returns_pct))
    except Exception:
        return None


class KellyPositionSizer:
    """
    Fractional Kelly position sizing with risk controls for live trading.
    Payoff ratio R blends historical win/loss, execution RR, and book microstructure.
    """

    def __init__(self, config: KellyConfig | None = None) -> None:
        self.config = config or KellyConfig()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    def _dynamic_payoff_ratio(
        self,
        *,
        avg_win_pct: float,
        avg_loss_pct: float,
        execution_rr: float | None,
        spread_bps: float,
        bid_notional_usd: float,
        ask_notional_usd: float,
    ) -> tuple[float, dict[str, float]]:
        avg_win = max(0.0, float(avg_win_pct))
        avg_loss = max(0.0, float(avg_loss_pct))
        R_hist = avg_win / avg_loss if avg_loss > 1e-12 else max(avg_win / 1e-6, 0.5)

        ref_s = max(float(self.config.reference_spread_bps), 0.25)
        s = max(float(spread_bps), 0.05)
        rel_spread = s / ref_s
        spread_factor = 1.0 / math.sqrt(rel_spread)

        ref_d = max(float(self.config.reference_depth_notional_usd), 1.0)
        min_side = max(0.0, min(float(bid_notional_usd), float(ask_notional_usd)))
        depth_ratio = min(1.0, min_side / ref_d)
        depth_factor = 0.52 + 0.48 * math.sqrt(depth_ratio)
        micro_mult = self._clamp(spread_factor * depth_factor, 0.35, 1.12)

        w = self._clamp(float(self.config.execution_rr_weight), 0.0, 1.0)
        Rex = float(execution_rr) if execution_rr is not None and execution_rr > 0 else None
        if Rex is not None:
            Rex = self._clamp(Rex, 0.35, min(R_hist * 3.0, 8.0))
            R_core = (1.0 - w) * R_hist + w * Rex
        else:
            R_core = R_hist

        lo, hi = self.config.micro_r_clip
        R_eff = self._clamp(R_core * micro_mult, lo, hi)
        diag = {
            "R_hist": float(R_hist),
            "R_execution_rr": float(Rex) if Rex is not None else None,
            "micro_mult": float(micro_mult),
            "spread_factor": float(spread_factor),
            "depth_factor": float(depth_factor),
        }
        return float(R_eff), diag

    def compute(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        confidence: float,
        drift_level: str,
        edge_decay: bool,
        current_drawdown_pct: float,
        volatility_regime: str,
        portfolio_value: float,
        size_multiplier: float = 1.0,
        *,
        execution_rr: float | None = None,
        depth: dict[str, Any] | None = None,
        spread_bps: float | None = None,
        bid_notional_usd: float | None = None,
        ask_notional_usd: float | None = None,
        strategy_returns_pct: list[float] | None = None,
        portfolio_heat_pct: float = 0.0,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        p = self._clamp(float(win_rate), 0.0, 1.0)
        q = 1.0 - p
        conf = self._clamp(float(confidence), 0.0, 100.0)
        dd = max(0.0, float(current_drawdown_pct))
        drift = str(drift_level or "LOW").upper()
        vol_regime = str(volatility_regime or "NORMAL").upper()
        port_value = max(0.0, float(portfolio_value))
        external_size_multiplier = max(0.0, float(size_multiplier))

        book = book_microstructure_from_depth(depth)
        sb = float(spread_bps if spread_bps is not None else book["spread_bps"])
        bn = float(bid_notional_usd if bid_notional_usd is not None else book["bid_notional_usd"])
        an = float(ask_notional_usd if ask_notional_usd is not None else book["ask_notional_usd"])

        b, rdiag = self._dynamic_payoff_ratio(
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            execution_rr=execution_rr,
            spread_bps=sb,
            bid_notional_usd=bn,
            ask_notional_usd=an,
        )

        raw_kelly = 0.0
        if b > 1e-9:
            raw_kelly = max(0.0, p - (q / b))
        kelly_full = float(raw_kelly)

        kelly_fractional = max(0.0, float(kelly_full) * float(self.config.kelly_fraction))

        reduction = 1.0
        if drift == "MEDIUM":
            reduction *= 0.80
            reasons.append("drift_medium")
        elif drift == "HIGH":
            reduction *= 0.55
            reasons.append("drift_high")

        if bool(edge_decay):
            reduction *= 0.70
            reasons.append("edge_decay")

        if self.config.volatility_scalar and vol_regime == "EXPANSION":
            reduction *= 0.65
            reasons.append("volatility_expansion")

        if conf < 60.0:
            reduction *= 0.75
            reasons.append("low_confidence")

        if dd > 0.06:
            reduction *= 0.50
            reasons.append("drawdown_gt_6pct")
        elif dd > 0.04:
            reduction *= 0.80
            reasons.append("drawdown_gt_4pct")

        halted = dd > float(self.config.drawdown_halt_pct)
        halt_reason = "drawdown_halt_triggered" if halted else None
        if halted:
            reasons.append("drawdown_halt")

        if external_size_multiplier != 1.0:
            reasons.append("external_size_multiplier")

        returns = [float(x) for x in (strategy_returns_pct or []) if isinstance(x, (int, float))]

        cvar_95_pct: float | None = None
        cvar_cap_pct: float | None = None
        if len(returns) >= int(self.config.cvar_min_samples):
            cvar_95_pct = _empirical_cvar_alpha(returns, float(self.config.cvar_alpha))
            if cvar_95_pct is not None and cvar_95_pct < 0.0:
                loss_mag = abs(cvar_95_pct) / 100.0
                bud = max(float(self.config.cvar_equity_budget_pct), 1e-6)
                cvar_cap_pct = bud / max(loss_mag, 1e-9)
                reasons.append("cvar_tail_cap")

        sigma_pct = _return_sigma_pct(returns) if len(returns) >= int(self.config.stress_min_samples) else None
        stress_cap_pct: float | None = None
        heat_frac = max(0.0, float(portfolio_heat_pct) / 100.0)
        cap_loss = float(self.config.stress_equity_loss_cap_pct)
        sig_mult = float(self.config.stress_sigma_multiplier)
        if sigma_pct is not None:
            sigma_eff_pct = max(float(sigma_pct), float(self.config.sigma_floor_pct))
            sigma_frac = sigma_eff_pct / 100.0
            room = (cap_loss / max(sig_mult * sigma_frac, 1e-9)) - heat_frac
            stress_cap_pct = max(0.0, float(room))
            reasons.append("stress_sigma_cap")

        raw_position_pct = kelly_fractional * reduction * external_size_multiplier
        min_pct = min(float(self.config.min_position_pct), float(self.config.max_position_pct))
        max_pct = max(float(self.config.min_position_pct), float(self.config.max_position_pct))
        position_pct = self._clamp(raw_position_pct, min_pct, max_pct) if not halted else 0.0

        pre_tail = float(position_pct)
        if cvar_cap_pct is not None:
            position_pct = min(position_pct, float(cvar_cap_pct))
            if position_pct < pre_tail - 1e-12:
                reasons.append("cvar_95_binding")
        pre_stress = float(position_pct)
        if stress_cap_pct is not None:
            position_pct = min(position_pct, float(stress_cap_pct))
            if position_pct < pre_stress - 1e-12:
                reasons.append("stress_2sigma_binding")

        position_pct = self._clamp(position_pct, 0.0, max_pct)
        position_size_usd = position_pct * port_value

        risk_budgets = {
            "cvar_alpha": float(self.config.cvar_alpha),
            "cvar_95_pct": None if cvar_95_pct is None else round(float(cvar_95_pct), 6),
            "cvar_equity_budget_pct": float(self.config.cvar_equity_budget_pct),
            "cvar_capped_position_pct": None if cvar_cap_pct is None else round(float(cvar_cap_pct), 6),
            "strategy_sigma_pct": None if sigma_pct is None else round(float(sigma_pct), 6),
            "stress_2sigma_equity_loss_cap_pct": float(cap_loss),
            "stress_sigma_multiplier": float(sig_mult),
            "stress_capped_position_pct": None if stress_cap_pct is None else round(float(stress_cap_pct), 6),
            "portfolio_heat_pct": round(float(portfolio_heat_pct), 6),
            "portfolio_heat_frac": round(float(heat_frac), 6),
            "dynamic_R": round(float(b), 6),
            "payoff_ratio_diagnostics": {
                k: round(v, 6)
                if isinstance(v, float)
                else v
                for k, v in rdiag.items()
            },
            "spread_bps_observed": round(float(sb), 4),
        }

        return {
            "kelly_full": float(kelly_full),
            "kelly_fraction": float(kelly_fractional),
            "position_pct": float(position_pct),
            "position_size_usd": float(position_size_usd),
            "position_size_pct": float(position_pct),
            "p": float(p),
            "b": float(b),
            "raw_kelly": float(raw_kelly),
            "size_reduction_reason": reasons,
            "halted": bool(halted),
            "halt_reason": halt_reason,
            "risk_budgets": risk_budgets,
        }
