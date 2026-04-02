"""
Layer 5 â€” Signal Engine.

Takes the alpha score, regime, ATR, and entry price, then
produces a fully-specified TradingSignal dataclass with:
  - BUY / SELL / HOLD
  - ATR-based Stop Loss
  - R:R-based Take Profit
  - Almgren-Chriss transaction cost estimate
  - Net alpha check (discard if cost > gross alpha)
  - Kelly position size (passed in from risk layer)
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.risk.cost_model import CostModel  # type: ignore[import]

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Signal dataclass
# ------------------------------------------------------------------

@dataclass
class TradingSignal:
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    asset: str = ""
    asset_class: str = ""       # 'indian_stock' | 'us_stock' | 'forex'
    timeframe: str = ""         # 'intraday' | 'swing'

    signal: str = "HOLD"        # 'BUY' | 'SELL' | 'HOLD'
    strength: str = "WEAK"      # 'STRONG' | 'MODERATE' | 'WEAK'
    confidence: float = 50.0    # 0-100
    alpha_score: float = 0.0

    regime: str = "SIDEWAYS"    # 'BULL' | 'BEAR' | 'SIDEWAYS'
    hurst_exponent: float = 0.5

    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_pct: float = 0.0       # % of portfolio at risk on this trade

    kelly_fraction: float = 0.0
    position_size_pct: float = 0.0

    factor_scores: dict = field(default_factory=dict)
    ic_weights: dict = field(default_factory=dict)

    slippage_cost_pct: float = 0.0
    cost_pct: float = 0.0              # total estimated cost (slippage + impact + spread)
    net_alpha_score: float = 0.0       # alpha_score - cost (in alpha units)
    cs_alpha_score: float = 0.0        # cross-sectional blended score
    execution_price: float = 0.0       # actual fill price (0 = not yet filled)
    implementation_shortfall_pct: float = 0.0  # (exec - entry) / entry * 100

    factor_breakdown: dict = field(default_factory=dict)
    confluence_grade: str | None = None
    confluence_pct: float | None = None
    sqs: int | None = None
    sqs_grade: str | None = None
    size_multiplier_used: float | None = None
    top_aligned_factors: list = field(default_factory=list)
    top_drag_factors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "asset_class": self.asset_class,
            "timeframe": self.timeframe,
            "signal": self.signal,
            "strength": self.strength,
            "confidence": self.confidence,
            "alpha_score": self.alpha_score,
            "regime": self.regime,
            "hurst_exponent": self.hurst_exponent,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_pct": self.risk_pct,
            "kelly_fraction": self.kelly_fraction,
            "position_size_pct": self.position_size_pct,
            "factor_scores": json.dumps(self.factor_scores),
            "ic_weights": json.dumps(self.ic_weights),
            "slippage_cost_pct": self.slippage_cost_pct,
            "cost_pct": self.cost_pct,
            "net_alpha_score": self.net_alpha_score,
            "cs_alpha_score": self.cs_alpha_score,
            "execution_price": self.execution_price,
            "implementation_shortfall_pct": self.implementation_shortfall_pct,
            "factor_breakdown": json.dumps(self.factor_breakdown) if self.factor_breakdown else "{}",
            "confluence_grade": self.confluence_grade,
            "confluence_pct": self.confluence_pct,
            "sqs": self.sqs,
            "sqs_grade": self.sqs_grade,
            "size_multiplier_used": self.size_multiplier_used,
            "top_aligned_factors": json.dumps(self.top_aligned_factors)
            if self.top_aligned_factors
            else "[]",
            "top_drag_factors": json.dumps(self.top_drag_factors)
            if self.top_drag_factors
            else "[]",
        }


# ------------------------------------------------------------------
# ATR multipliers by strength
# ------------------------------------------------------------------

ATR_MULTIPLIERS = {
    "STRONG": 1.5,
    "MODERATE": 2.0,
    "WEAK": 2.5,
}

RR_RATIOS = {
    "STRONG": 3.0,
    "MODERATE": 2.5,
    "WEAK": 2.0,
}

SLIPPAGE = {
    "indian_stock": 0.0003,
    "us_stock": 0.0001,
    "forex_major": 0.00005,
    "forex_minor": 0.0001,
    "forex": 0.0001,
}


# ------------------------------------------------------------------
# Signal Engine
# ------------------------------------------------------------------

class SignalEngine:
    """
    Converts alpha model output â†’ fully specified TradingSignal.
    Applies net-alpha gate: discards signals where transaction cost > gross alpha.
    """

    def __init__(
        self,
        cost_config: dict | None = None,
        min_sqs: float = 55.0,
        dedup_hours: float = 4.0,
        min_holding_candles: int = 2,
    ):
        self._cost_model = CostModel(cost_config or {})
        self._min_sqs = float(min_sqs)
        self._dedup_seconds = float(dedup_hours) * 3600.0
        self._min_holding_candles = int(min_holding_candles)
        self._last_same_direction_ts: dict[tuple[str, str], datetime] = {}
        self._last_direction_state: dict[str, dict[str, object]] = {}
        self._bar_counters: dict[str, int] = {}

    @staticmethod
    def _normalize_direction(signal: str) -> str:
        s = (signal or "").upper()
        if s in {"BUY", "LONG"}:
            return "LONG"
        if s in {"SELL", "SHORT"}:
            return "SHORT"
        return "HOLD"

    def generate(
        self,
        asset: str,
        asset_class: str,
        timeframe: str,
        alpha_result: dict,
        regime: str,
        regime_allows: bool,
        hurst: float,
        entry_price: float,
        atr_14: float,
        kelly_fraction: float,
        position_size_pct: float,
        regime_size_mult: float = 1.0,
        daily_vol: float = 0.20,
        volume_ratio: float = 1.0,
        cs_alpha_score: float = 0.0,
    ) -> TradingSignal | None:
        """
        Build a TradingSignal.
        Returns None for HOLD signals or regime-blocked signals.

        Parameters
        ----------
        regime_allows : bool   â€” output of RegimeDetector.allows_signal()
        regime_size_mult : float â€” 0.5 for SIDEWAYS, 1.0 otherwise
        """
        signal_str = alpha_result["signal"]
        direction = self._normalize_direction(signal_str)
        strength = alpha_result["strength"]
        confidence = alpha_result["confidence"]
        alpha_score = alpha_result["alpha_score"]
        now_utc = datetime.now(timezone.utc)
        state_key = f"{asset}:{timeframe}"

        # HOLD or regime gate
        if signal_str == "HOLD" or not regime_allows:
            return None

        # Same-direction deduplication window
        last_same = self._last_same_direction_ts.get((state_key, direction))
        if last_same is not None:
            elapsed = (now_utc - last_same).total_seconds()
            if elapsed < self._dedup_seconds:
                return None

        # Minimum holding period before accepting reversal
        bar_index = int(self._bar_counters.get(state_key, 0)) + 1
        self._bar_counters[state_key] = bar_index
        st = self._last_direction_state.get(state_key)
        if st:
            prev_dir = str(st.get("direction", "HOLD"))
            prev_bar = int(st.get("bar_index", 0))
            if direction in {"LONG", "SHORT"} and prev_dir in {"LONG", "SHORT"} and direction != prev_dir:
                if (bar_index - prev_bar) < self._min_holding_candles:
                    return None

        # Net-alpha gate (GAP 2): discard if transaction cost exceeds gross alpha
        # Uses base Kelly x regime only; explainability multiplier applied after pass.
        base_size = position_size_pct * regime_size_mult
        low_liquidity = volume_ratio < 0.8
        net_alpha, cost_pct, is_viable = self._cost_model.net_alpha(
            alpha_score,
            asset_class,
            base_size,
            daily_vol,
            volume_ratio,
            regime=regime,
            low_liquidity=low_liquidity,
        )
        if not is_viable:
            return None  # cost eats the alpha â€” don't fire the signal

        # === Confluence + SQS (additive annotation layer) ===
        _confluence_result: dict = {
            "grade": "UNKNOWN",
            "kelly_multiplier": 1.0,
            "fire_signal": True,
            "computation_ok": False,
        }
        _sqs_result: dict = {
            "sqs": 50,
            "grade": "STANDARD",
            "fire": True,
            "size_multiplier": 1.0,
            "computation_ok": False,
        }
        _combined_multiplier = 1.0
        factor_breakdown = alpha_result.get("factor_breakdown") or {}

        try:
            from src.alpha.confluence import ConfluenceScorer
            from src.signals.quality_score import SignalQualityScorer as ContextualSQS

            _confluence_result = ConfluenceScorer().score(factor_breakdown, direction)

            if _confluence_result.get("grade") == "D":
                logger.info("Signal suppressed: confluence grade D for %s", asset)
                return None

            _sqs_context = {
                "alpha_score": alpha_score,
                "confluence_grade": _confluence_result.get("grade"),
                "regime_confidence": alpha_result.get("regime_confidence"),
                "mtf_aligned": alpha_result.get("mtf_aligned"),
                "factor_ic_recency": alpha_result.get("factor_ic_recency"),
                "liquidity_score": alpha_result.get("liquidity_score"),
            }
            _sqs_context = {k: v for k, v in _sqs_context.items() if v is not None}

            _sqs_result = ContextualSQS().compute_sqs(_sqs_context)

            if not _sqs_result.get("fire", True):
                logger.info(
                    "Signal suppressed by SQS=%d for %s: %s",
                    _sqs_result.get("sqs", 0),
                    asset,
                    _sqs_result.get("suppression_reason", ""),
                )
                return None

            _raw_multiplier = (
                float(_confluence_result.get("kelly_multiplier", 1.0))
                * float(_sqs_result.get("size_multiplier", 1.0))
            )
            _combined_multiplier = max(_raw_multiplier, 0.25)

        except Exception as e:
            logger.warning(
                "Confluence/SQS failed for %s (using defaults): %s",
                asset,
                e,
            )
            _combined_multiplier = 1.0
        # === End Confluence + SQS ===

        final_size = base_size * _combined_multiplier

        # ATR-based Stop Loss
        atr_mult = ATR_MULTIPLIERS.get(strength, 2.0)
        rr = RR_RATIOS.get(strength, 2.0)

        if signal_str == "BUY":
            stop_loss = entry_price - atr_mult * atr_14
            take_profit = entry_price + abs(entry_price - stop_loss) * rr
        else:  # SELL
            stop_loss = entry_price + atr_mult * atr_14
            take_profit = entry_price - abs(entry_price - stop_loss) * rr

        risk_pct = abs(entry_price - stop_loss) / entry_price * final_size

        # Slippage
        slip_key = asset_class
        if asset_class == "forex":
            slip_key = "forex_major" if asset in {"EURUSD", "GBPUSD"} else "forex_minor"
        slippage = SLIPPAGE.get(slip_key, SLIPPAGE["us_stock"])

        out = TradingSignal(
            asset=asset,
            asset_class=asset_class,
            timeframe=timeframe,
            signal=signal_str,
            strength=strength,
            confidence=confidence,
            alpha_score=alpha_score,
            regime=regime,
            hurst_exponent=hurst,
            entry_price=round(entry_price, 6),  # type: ignore[call-overload]
            stop_loss=round(stop_loss, 6),  # type: ignore[call-overload]
            take_profit=round(take_profit, 6),  # type: ignore[call-overload]
            risk_pct=round(risk_pct, 4),  # type: ignore[call-overload]
            kelly_fraction=round(kelly_fraction, 4),  # type: ignore[call-overload]
            position_size_pct=round(final_size, 4),  # type: ignore[call-overload]
            factor_scores=alpha_result.get("factor_scores", {}),
            ic_weights=alpha_result.get("ic_weights", {}),
            slippage_cost_pct=slippage,
            cost_pct=round(cost_pct, 5),  # type: ignore[call-overload]
            net_alpha_score=round(net_alpha, 4),  # type: ignore[call-overload]
            cs_alpha_score=round(cs_alpha_score, 4),  # type: ignore[call-overload]
            factor_breakdown=dict(factor_breakdown) if factor_breakdown else {},
            confluence_grade=_confluence_result.get("grade", "UNKNOWN"),
            confluence_pct=_confluence_result.get("confluence_pct"),
            sqs=_sqs_result.get("sqs"),
            sqs_grade=_sqs_result.get("grade", "UNKNOWN"),
            size_multiplier_used=round(float(_combined_multiplier), 6),
            top_aligned_factors=list(_confluence_result.get("top_aligned", [])),
            top_drag_factors=list(_confluence_result.get("top_drags", [])),
        )
        self._last_same_direction_ts[(state_key, direction)] = now_utc
        self._last_direction_state[state_key] = {"direction": direction, "bar_index": bar_index}
        return out

    def format_cli(self, sig: TradingSignal) -> str:
        """Return a formatted CLI string for the signal."""
        bar_len = int(sig.confidence / 10)
        bar = "â–ˆ" * bar_len + "â–‘" * (10 - bar_len)
        direction = "â–²" if sig.signal == "BUY" else "â–¼"
        sl_pct = (sig.stop_loss - sig.entry_price) / sig.entry_price * 100
        tp_pct = (sig.take_profit - sig.entry_price) / sig.entry_price * 100

        factor_line = "  ".join(
            f"{k}:{'+' if v >= 0 else ''}{v:.2f}"
            for k, v in sig.factor_scores.items()
        )

        return (
            f"\nâ•”{'â•'*58}â•—\n"
            f"â•‘  SIGNAL FIRED â€” {sig.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC{' ' * 5}â•‘\n"
            f"â•‘  Asset     : {sig.asset} ({sig.asset_class}){' ' * (38 - len(sig.asset) - len(sig.asset_class))}â•‘\n"
            f"â•‘  Timeframe : {sig.timeframe.upper()}{' ' * (45 - len(sig.timeframe))}â•‘\n"
            f"â•‘  Signal    : {bar} {direction} {sig.signal}  [{sig.strength}]{' ' * (17 - len(sig.strength))}â•‘\n"
            f"â•‘  Confidence: {sig.confidence:.0f}/100  |  Alpha Score: {sig.alpha_score:+.3f}{' ' * 10}â•‘\n"
            f"â•‘  Regime    : {sig.regime}  |  Hurst: {sig.hurst_exponent:.2f}{' ' * 20}â•‘\n"
            f"â•‘  {'â”€'*54}  â•‘\n"
            f"â•‘  Entry     : {sig.entry_price:.4f}{' ' * 40}â•‘\n"
            f"â•‘  Stop Loss : {sig.stop_loss:.4f}  ({sl_pct:+.2f}%)  [{ATR_MULTIPLIERS.get(sig.strength, 2.0)}Ã— ATR]{' ' * 12}â•‘\n"
            f"â•‘  Target    : {sig.take_profit:.4f}  ({tp_pct:+.2f}%)  [{RR_RATIOS.get(sig.strength, 2.0):.1f}:1 R:R]{' ' * 10}â•‘\n"
            f"â•‘  {'â”€'*54}  â•‘\n"
            f"â•‘  Position  : {sig.position_size_pct:.1f}%  [Kelly raw: {sig.kelly_fraction:.1f}%]{' ' * 22}â•‘\n"
            f"â•‘  Slippage  : {sig.slippage_cost_pct*100:.3f}%{' ' * 43}â•‘\n"
            f"â•‘  {'â”€'*54}  â•‘\n"
            f"â•‘  Factors   : {factor_line[:44]}{' ' * max(0, 44-len(factor_line))}  â•‘\n"  # type: ignore[index]
            f"â•š{'â•'*58}â•\n"
        )

