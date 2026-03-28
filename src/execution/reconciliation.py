"""
Execution and position reconciliation utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    scope: str
    asset: str = ""
    status: str = "OK"
    severity: str = "INFO"
    message: str = ""
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "asset": self.asset,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "details": dict(self.details),
            "timestamp": self.timestamp,
        }


class FillReconciler:
    def __init__(self, max_slippage_multiple: float = 2.0) -> None:
        self.max_slippage_multiple = max_slippage_multiple

    def reconcile_receipt(self, signal, receipt) -> list[ReconciliationResult]:
        results: list[ReconciliationResult] = []
        if receipt is None:
            return results

        status = str(getattr(receipt, "status", "UNKNOWN")).upper()
        if status in {"REJECTED", "CANCELLED", "EXPIRED"}:
            results.append(
                ReconciliationResult(
                    scope="order_fill",
                    asset=signal.asset,
                    status=status,
                    severity="CRITICAL",
                    message=f"Order {status.lower()} for {signal.asset}",
                )
            )
            return results

        requested = float(signal.entry_price or 0.0)
        filled = float(getattr(receipt, "fill_price", 0.0) or 0.0)
        if requested > 0 and filled > 0:
            actual_slippage_pct = abs(filled - requested) / requested * 100
            expected_slippage_pct = float(signal.slippage_cost_pct or 0.0) * 100
            overrun_limit = expected_slippage_pct * self.max_slippage_multiple
            if expected_slippage_pct > 0 and actual_slippage_pct > overrun_limit:
                results.append(
                    ReconciliationResult(
                        scope="slippage",
                        asset=signal.asset,
                        status="OVERRUN",
                        severity="CRITICAL",
                        message=(
                            f"Slippage overrun for {signal.asset}: "
                            f"{actual_slippage_pct:.4f}% > {overrun_limit:.4f}%"
                        ),
                        details={
                            "expected_slippage_pct": round(expected_slippage_pct, 6),
                            "actual_slippage_pct": round(actual_slippage_pct, 6),
                            "fill_price": filled,
                            "requested_price": requested,
                        },
                    )
                )

        fill_ratio = float(getattr(receipt, "fill_ratio", 1.0) or 1.0)
        if fill_ratio < 0.999:
            results.append(
                ReconciliationResult(
                    scope="partial_fill",
                    asset=signal.asset,
                    status="PARTIAL",
                    severity="CRITICAL",
                    message=f"Partial fill detected for {signal.asset}",
                    details={"fill_ratio": round(fill_ratio, 4)},
                )
            )
        return results

    def reconcile_positions(
        self,
        internal_positions: list[dict],
        broker_positions: list[dict],
    ) -> list[ReconciliationResult]:
        results: list[ReconciliationResult] = []
        internal_assets = {p.get("asset", ""): p for p in internal_positions if p.get("asset")}
        broker_assets = {}
        for pos in broker_positions:
            asset = pos.get("asset") or pos.get("tradingsymbol") or pos.get("symbol") or ""
            qty = (
                pos.get("quantity")
                or pos.get("qty")
                or pos.get("net_quantity")
                or pos.get("filled_quantity")
                or 0
            )
            broker_assets[asset] = {"raw": pos, "qty": float(qty or 0)}

        missing_internal = sorted(set(broker_assets) - set(internal_assets))
        for asset in missing_internal:
            results.append(
                ReconciliationResult(
                    scope="position_reconciliation",
                    asset=asset,
                    status="GHOST_POSITION",
                    severity="CRITICAL",
                    message=f"Broker reports {asset} but internal tracker does not",
                    details=broker_assets[asset],
                )
            )

        missing_broker = sorted(set(internal_assets) - set(broker_assets))
        for asset in missing_broker:
            results.append(
                ReconciliationResult(
                    scope="position_reconciliation",
                    asset=asset,
                    status="MISSING_BROKER_POSITION",
                    severity="CRITICAL",
                    message=f"Internal tracker shows {asset} but broker does not",
                    details=internal_assets[asset],
                )
            )

        for asset in sorted(set(internal_assets) & set(broker_assets)):
            internal_qty = float(internal_assets[asset].get("position_size_pct", 0.0))
            broker_qty = float(broker_assets[asset].get("qty", 0.0))
            if broker_qty and internal_qty and np.sign(broker_qty) != np.sign(internal_qty):
                results.append(
                    ReconciliationResult(
                        scope="position_reconciliation",
                        asset=asset,
                        status="SIDE_MISMATCH",
                        severity="CRITICAL",
                        message=f"Side mismatch detected for {asset}",
                        details={"internal_qty": internal_qty, "broker_qty": broker_qty},
                    )
                )
        return results


class ImpactCalibrator:
    """Re-fit the market impact coefficient from realised fills."""

    def __init__(self, min_samples: int = 30) -> None:
        self.min_samples = min_samples

    def calibrate_eta(self, observations: list[dict], default_eta: float = 0.6) -> dict:
        usable = []
        for row in observations:
            try:
                target = float(row["actual_slippage_pct"]) - float(row.get("spread_pct", 0.0))
                feature = float(row["sigma"]) * np.sqrt(
                    float(row["participation"]) / max(float(row["adv_ratio"]), 1e-9)
                )
                if np.isfinite(target) and np.isfinite(feature) and feature > 0:
                    usable.append((feature, target))
            except Exception:
                continue

        if len(usable) < self.min_samples:
            return {
                "eta": default_eta,
                "fitted": False,
                "n_samples": len(usable),
                "message": "Insufficient live fills for eta calibration",
            }

        x = np.array([u[0] for u in usable])
        y = np.array([u[1] for u in usable])
        eta = float(np.maximum(np.dot(x, y) / np.dot(x, x), 0.0))
        return {
            "eta": round(eta, 6),
            "fitted": True,
            "n_samples": len(usable),
            "message": "Eta fitted from realised slippage observations",
        }
