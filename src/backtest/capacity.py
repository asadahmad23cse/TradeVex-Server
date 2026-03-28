"""
Gap 4 Fix — Capacity Decay Simulator.

Tests how alpha degrades as AUM (Assets Under Management) scales up.

Key question: "Does your 2% alpha survive at ₹10 Cr?"

Method:
    For each AUM level, simulate the backtest with increasing market impact:
        - At ₹10L:   impact ≈ 0 (your trades are invisible)
        - At ₹1Cr:   impact starts eating alpha
        - At ₹10Cr:  impact may consume all alpha
        - At ₹100Cr: likely unfeasible for most signals

Market Impact Formula (Almgren-Chriss):
    impact = η × σ × √(Q / ADV)

    As Q grows (because AUM grows), impact grows as √AUM.
    α_net(AUM) = α_gross - impact(AUM)
    Capacity = AUM where α_net drops below 0.

Output:
    Table showing Sharpe at each AUM level.
    Estimated capacity: max AUM where Sharpe > 0.5.

Usage:
    sim = CapacitySimulator(config_path="config.yaml")
    sim.run(ticker="AAPL", aum_levels=[10_00_000, 1_00_00_000, 10_00_00_000])
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)

# Default AUM levels in INR
DEFAULT_AUM_LEVELS = [
    10_00_000,       # ₹10L
    50_00_000,       # ₹50L
    1_00_00_000,     # ₹1Cr
    5_00_00_000,     # ₹5Cr
    10_00_00_000,    # ₹10Cr
    50_00_00_000,    # ₹50Cr
    100_00_00_000,   # ₹100Cr
]


class CapacitySimulator:
    """
    Test how alpha decays with increasing AUM.

    At each AUM level:
        1. Scale position sizes proportionally
        2. Increase market impact via √(AUM / base_AUM)
        3. Run backtest with adjusted costs
        4. Record Sharpe, return, MDD
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.base_capital = float(self.cfg["portfolio"]["initial_capital"])

    def run(
        self,
        ticker: str,
        asset_class: str = "indian_stock",
        aum_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Run capacity decay analysis.

        Parameters
        ----------
        ticker      : yfinance ticker
        asset_class : 'indian_stock' | 'us_stock' | 'forex'
        aum_levels  : list of AUM values to test (in INR)

        Returns
        -------
        pd.DataFrame with columns: AUM, Sharpe, Return_pct, MDD_pct, Avg_Cost_pct
        """
        levels = aum_levels or DEFAULT_AUM_LEVELS
        results = []

        print(f"\n{'='*70}")
        print(f"  Capacity Decay Analysis — {ticker}")
        print(f"  Base Capital: ₹{self.base_capital:,.0f}")
        print(f"{'='*70}")
        print(f"  {'AUM':>15}  {'Sharpe':>8}  {'Return %':>10}  {'MDD %':>8}  {'Avg Cost %':>12}")
        print(f"  {'-'*15}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*12}")

        for aum in levels:
            # Scale factor
            scale = aum / self.base_capital

            # Adjust market impact coefficient proportionally to √scale
            adjusted_cfg = {**self.cfg}
            cost_cfg = dict(adjusted_cfg.get("cost_model", {}))
            base_eta = cost_cfg.get("market_impact_coefficient", 0.6)
            cost_cfg["market_impact_coefficient"] = base_eta * np.sqrt(scale)
            adjusted_cfg["cost_model"] = cost_cfg

            # Set capital
            adjusted_cfg["portfolio"]["initial_capital"] = float(aum)

            # Write temp config
            tmp_path = Path(tempfile.gettempdir()) / f"cap_test_{aum}.yaml"
            with open(tmp_path, "w") as f:
                yaml.dump(adjusted_cfg, f)

            try:
                engine = BacktestEngine(config_path=str(tmp_path))
                result = engine.run(ticker=ticker, asset_class=asset_class)
                summary = result.summary()

                row = {
                    "AUM": aum,
                    "Sharpe": summary.get("sharpe_ratio", 0.0),
                    "Return_pct": summary.get("total_return_pct", 0.0),
                    "MDD_pct": summary.get("max_drawdown_pct", 0.0),
                    "Avg_Cost_pct": summary.get("avg_cost_pct", 0.0),
                    "Trades": summary.get("total_trades", 0),
                    "Win_Rate": summary.get("win_rate", 0.0),
                }
            except Exception as exc:
                logger.warning("Capacity test at AUM=%.0f failed: %s", aum, exc)
                row = {
                    "AUM": aum, "Sharpe": 0.0, "Return_pct": 0.0,
                    "MDD_pct": 0.0, "Avg_Cost_pct": 0.0, "Trades": 0, "Win_Rate": 0.0,
                }
            finally:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

            results.append(row)
            print(
                f"  ₹{aum:>13,.0f}  {row['Sharpe']:>8.3f}  "
                f"{row['Return_pct']:>9.2f}%  {row['MDD_pct']:>7.2f}%  "
                f"{row['Avg_Cost_pct']:>11.4f}%"
            )

        df = pd.DataFrame(results)

        # Report capacity
        viable = df[df["Sharpe"] > 0.5]
        if not viable.empty:
            max_cap = viable["AUM"].max()
            print(f"\n  ✅ Estimated capacity: ₹{max_cap:,.0f} (Sharpe > 0.5)")
        else:
            print(f"\n  ⚠️  Strategy doesn't achieve Sharpe > 0.5 at any tested AUM level")

        print(f"{'='*70}\n")
        return df

    def run_watchlist(
        self,
        asset_class: str = "indian_stock",
        aum_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Approximate portfolio capacity by running the configured watchlist and
        aggregating the per-ticker summary statistics at each AUM bucket.
        """
        levels = aum_levels or DEFAULT_AUM_LEVELS
        key = {
            "indian_stock": "indian_stocks",
            "us_stock": "us_stocks",
            "forex": "forex",
        }[asset_class]
        assets = self.cfg.get("watchlist", {}).get(key, [])
        rows = []

        for aum in levels:
            sharpe_vals = []
            ret_vals = []
            mdd_vals = []
            cost_vals = []
            for asset in assets:
                ticker = asset.get("yf_ticker") or asset.get("symbol")
                if not ticker:
                    continue
                try:
                    single = self.run(ticker=ticker, asset_class=asset_class, aum_levels=[aum])
                    if single.empty:
                        continue
                    sharpe_vals.append(float(single.iloc[0]["Sharpe"]))
                    ret_vals.append(float(single.iloc[0]["Return_pct"]))
                    mdd_vals.append(float(single.iloc[0]["MDD_pct"]))
                    cost_vals.append(float(single.iloc[0]["Avg_Cost_pct"]))
                except Exception as exc:
                    logger.warning("Watchlist capacity failed for %s @ %.0f: %s", ticker, aum, exc)
            if sharpe_vals:
                rows.append(
                    {
                        "AUM": aum,
                        "Assets": len(sharpe_vals),
                        "Sharpe": round(float(np.mean(sharpe_vals)), 4),
                        "Return_pct": round(float(np.mean(ret_vals)), 4),
                        "MDD_pct": round(float(np.mean(mdd_vals)), 4),
                        "Avg_Cost_pct": round(float(np.mean(cost_vals)), 4),
                    }
                )
        return pd.DataFrame(rows)
