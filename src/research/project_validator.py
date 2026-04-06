"""
Project-level validation runner for model robustness and trading accuracy.

This module combines:
1) WFO + CPCV robustness checks (from WFOValidator)
2) Full event-driven backtest performance
3) Trade-level directional accuracy diagnostics
4) Buy-and-hold benchmark comparison
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def infer_asset_class(ticker: str) -> str:
    """Infer asset class from ticker format."""
    t = str(ticker or "").upper()
    if t.endswith((".NS", ".BO")):
        return "indian_stock"
    if t.endswith("=X") or t in {"EURUSD", "GBPUSD", "USDINR", "USDJPY", "XAUUSD"}:
        return "forex"
    return "us_stock"


def compute_trade_accuracy(trades: list[Any]) -> dict[str, Any]:
    """Compute directional hit-rate and side-wise precision from closed trades."""
    closed = [t for t in trades if getattr(t, "outcome", "OPEN") in {"WIN", "LOSS"}]
    if not closed:
        return {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "signal_accuracy_pct": 0.0,
            "buy_win_rate_pct": 0.0,
            "sell_win_rate_pct": 0.0,
            "expectancy_pct": 0.0,
            "profit_factor": 0.0,
        }

    wins = [t for t in closed if t.outcome == "WIN"]
    losses = [t for t in closed if t.outcome == "LOSS"]
    buys = [t for t in closed if str(t.signal).upper() == "BUY"]
    sells = [t for t in closed if str(t.signal).upper() == "SELL"]

    buy_wins = [t for t in buys if t.outcome == "WIN"]
    sell_wins = [t for t in sells if t.outcome == "WIN"]

    win_pnls = [float(t.net_pnl_pct) for t in wins]
    loss_pnls = [abs(float(t.net_pnl_pct)) for t in losses]

    gross_win = float(np.sum(win_pnls)) if win_pnls else 0.0
    gross_loss = float(np.sum(loss_pnls)) if loss_pnls else 0.0

    return {
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "signal_accuracy_pct": round(len(wins) / len(closed) * 100.0, 2),
        "buy_win_rate_pct": round(len(buy_wins) / len(buys) * 100.0, 2) if buys else 0.0,
        "sell_win_rate_pct": round(len(sell_wins) / len(sells) * 100.0, 2) if sells else 0.0,
        "expectancy_pct": round(float(np.mean([float(t.net_pnl_pct) for t in closed])), 4),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 1e-12 else 0.0,
    }


def compute_buy_hold_return(
    connector: Any,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> float | None:
    """Compute simple buy-and-hold return (%) over the requested period."""
    try:
        raw = connector.get_daily(ticker, period="5y")
        if raw.empty:
            return None
        if start:
            raw = raw[raw.index >= start]
        if end:
            raw = raw[raw.index <= end]
        if len(raw) < 2:
            return None
        first = float(raw["Close"].iloc[0])
        last = float(raw["Close"].iloc[-1])
        if first <= 0:
            return None
        return round((last / first - 1.0) * 100.0, 2)
    except Exception:
        return None


class ProjectValidator:
    """Runs combined validation and builds a single project accuracy report."""

    def __init__(self, config_path: str = "config.yaml", train_days: int = 252):
        from src.api.connectors import MarketDataConnector

        self.config_path = config_path
        self.train_days = int(train_days)
        self.connector = MarketDataConnector()

    def _load_wfo_config(self) -> dict[str, Any]:
        try:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f) or {}
            return dict(cfg.get("wfo", {}) or {})
        except Exception:
            return {}

    @staticmethod
    def _overall_verdict(
        combined_verdict: str,
        signal_accuracy_pct: float,
        sharpe_ratio: float,
        excess_return_pct: float | None,
    ) -> str:
        cv = str(combined_verdict or "FAIL").upper()
        if cv == "PASS" and signal_accuracy_pct >= 52.0 and sharpe_ratio > 0.5:
            if excess_return_pct is None or excess_return_pct > 0.0:
                return "READY_FOR_PAPER_TRADING"
        if cv in {"PASS", "WARN"} and signal_accuracy_pct >= 50.0 and sharpe_ratio >= 0.0:
            return "PROMISING_NEEDS_TUNING"
        return "NOT_READY"

    def run(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        asset_class: str | None = None,
        include_cpcv: bool = True,
    ) -> dict[str, Any]:
        from src.backtest.engine import BacktestEngine
        from src.validator import WFOValidator

        wfo_cfg = self._load_wfo_config()
        wfo_validator = WFOValidator(
            ticker=ticker,
            train_window=self.train_days,
            wfo_config=wfo_cfg,
        )
        if include_cpcv:
            robustness = wfo_validator.run_with_cpcv(ticker=ticker)
        else:
            wfo_result = wfo_validator.run_validation()
            robustness = {
                "ticker": ticker,
                "wfo": wfo_result,
                "cpcv": {
                    "verdict": "SKIPPED",
                    "oos_sharpe_mean": 0.0,
                    "pbo": 1.0,
                    "dsr": 0.0,
                },
                "combined_verdict": "PASS" if bool(wfo_result.get("passed", False)) else "FAIL",
                "timestamp": datetime.utcnow().isoformat(),
            }

        bt = BacktestEngine(config_path=self.config_path)
        resolved_asset_class = asset_class or infer_asset_class(ticker)
        bt_result = bt.run(
            ticker=ticker,
            start=start,
            end=end,
            asset_class=resolved_asset_class,
        )
        bt_summary = bt_result.summary()
        trade_accuracy = compute_trade_accuracy(bt_result.trades)

        buy_hold = compute_buy_hold_return(
            self.connector,
            ticker=ticker,
            start=start or bt_result.start or None,
            end=end or bt_result.end or None,
        )

        total_return = float(bt_summary.get("total_return_pct", 0.0)) if isinstance(bt_summary, dict) else 0.0
        sharpe = float(bt_summary.get("sharpe_ratio", 0.0)) if isinstance(bt_summary, dict) else 0.0
        excess = round(total_return - buy_hold, 2) if buy_hold is not None else None
        combined_verdict = str(robustness.get("combined_verdict", "FAIL"))

        report = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "asset_class": resolved_asset_class,
            "robustness": {
                "combined_verdict": combined_verdict,
                "wfo_passed": bool((robustness.get("wfo") or {}).get("passed", False)),
                "wfo_best_ir": float((robustness.get("wfo") or {}).get("best_ir", 0.0)),
                "cpcv_verdict": str((robustness.get("cpcv") or {}).get("verdict", "FAIL")),
                "cpcv_oos_sharpe_mean": float((robustness.get("cpcv") or {}).get("oos_sharpe_mean", 0.0)),
                "cpcv_pbo": float((robustness.get("cpcv") or {}).get("pbo", 1.0)),
                "cpcv_dsr": float((robustness.get("cpcv") or {}).get("dsr", 0.0)),
            },
            "backtest": bt_summary,
            "trade_accuracy": trade_accuracy,
            "benchmark": {
                "buy_hold_return_pct": buy_hold,
                "strategy_minus_buy_hold_pct": excess,
            },
        }

        report["overall_verdict"] = self._overall_verdict(
            combined_verdict=combined_verdict,
            signal_accuracy_pct=float(trade_accuracy.get("signal_accuracy_pct", 0.0)),
            sharpe_ratio=sharpe,
            excess_return_pct=excess,
        )
        return report

    @staticmethod
    def save_report(report: dict[str, Any], output_path: str) -> str:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return str(out)

    @staticmethod
    def print_report(report: dict[str, Any]) -> None:
        r = report.get("robustness", {}) or {}
        b = report.get("backtest", {}) or {}
        a = report.get("trade_accuracy", {}) or {}
        bm = report.get("benchmark", {}) or {}

        print(f"\n{'='*72}")
        print(f"PROJECT VALIDATION REPORT - {report.get('ticker', 'UNKNOWN')}")
        print(f"{'='*72}")
        print(f"Overall verdict: {report.get('overall_verdict', 'NOT_READY')}")
        print()
        print("Robustness checks")
        print(f"  Combined verdict      : {r.get('combined_verdict', 'FAIL')}")
        print(f"  WFO passed            : {r.get('wfo_passed', False)}")
        print(f"  WFO best IR           : {r.get('wfo_best_ir', 0.0):.4f}")
        print(f"  CPCV verdict          : {r.get('cpcv_verdict', 'FAIL')}")
        print(f"  CPCV OOS Sharpe mean  : {r.get('cpcv_oos_sharpe_mean', 0.0):.4f}")
        print(f"  CPCV PBO              : {r.get('cpcv_pbo', 1.0):.4f}")
        print(f"  CPCV DSR              : {r.get('cpcv_dsr', 0.0):.4f}")
        print()
        print("Trading performance")
        if b.get("error"):
            print(f"  Backtest error        : {b.get('error')}")
        else:
            print(f"  Total return (%)      : {b.get('total_return_pct', 0.0)}")
            print(f"  Sharpe ratio          : {b.get('sharpe_ratio', 0.0)}")
            print(f"  Max drawdown (%)      : {b.get('max_drawdown_pct', 0.0)}")
            print(f"  Profit factor         : {b.get('profit_factor', 0.0)}")
            print(f"  Total trades          : {b.get('total_trades', 0)}")
            print(f"  Win rate (%)          : {b.get('win_rate', 0.0)}")
        print()
        print("Signal accuracy")
        print(f"  Directional accuracy (%) : {a.get('signal_accuracy_pct', 0.0)}")
        print(f"  Buy win rate (%)         : {a.get('buy_win_rate_pct', 0.0)}")
        print(f"  Sell win rate (%)        : {a.get('sell_win_rate_pct', 0.0)}")
        print(f"  Expectancy (%)           : {a.get('expectancy_pct', 0.0)}")
        print()
        print("Benchmark")
        print(f"  Buy & hold return (%)    : {bm.get('buy_hold_return_pct')}")
        print(f"  Strategy - buy & hold (%) : {bm.get('strategy_minus_buy_hold_pct')}")
        print(f"{'='*72}\n")
