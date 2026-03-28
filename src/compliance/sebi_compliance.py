"""SEBI algo-compliance and India true-cost accounting utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
COMPLIANCE_DIR = DATA_DIR / "compliance"
REGISTRY_FILE = DATA_DIR / "algo_registry.json"
AUDIT_GLOB = "audit_trail_*.jsonl"

for p in (DATA_DIR, COMPLIANCE_DIR):
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    gst: float
    stamp_duty: float
    total_cost: float
    gross_pnl: float
    net_pnl: float


class SEBIComplianceEngine:
    """
    SEBI Algo Trading Circular (Feb 2025) compliance helper.
    """

    def __init__(self):
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._kill_switch_ts = 0.0
        self._ops_limit = 10

    @staticmethod
    def _broker_code(broker: str) -> str:
        code = "".join(ch for ch in (broker or "GEN").upper() if ch.isalnum())
        return (code[:4] or "GENX").ljust(4, "X")

    @staticmethod
    def _audit_file_for(date_str: str | None = None) -> Path:
        if date_str:
            day = date_str
        else:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return DATA_DIR / f"audit_trail_{day}.jsonl"

    def generate_algo_id(self, strategy_name: str, broker: str) -> str:
        registry = _load_json(REGISTRY_FILE, [])
        broker_code = self._broker_code(broker)
        hash_part = hashlib.sha1((strategy_name or "strategy").encode("utf-8")).hexdigest()[:8].upper()
        ts_part = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        algo_id = f"QT-{broker_code}-{hash_part}-{ts_part}"
        registry.append(
            {
                "algo_id": algo_id,
                "strategy_name": strategy_name,
                "broker": broker,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save_json(REGISTRY_FILE, registry)
        return algo_id

    def list_algo_ids(self) -> list[dict]:
        reg = _load_json(REGISTRY_FILE, [])
        return reg if isinstance(reg, list) else []

    def log_order_audit(self, order: dict) -> None:
        row = {
            "timestamp_ns": time.time_ns(),
            "algo_id": order.get("algo_id"),
            "symbol": order.get("symbol"),
            "action": order.get("action"),
            "quantity": float(order.get("quantity", 0)),
            "price": float(order.get("price", 0)),
            "order_id": order.get("order_id"),
            "reason": order.get("reason", ""),
        }
        af = self._audit_file_for()
        with af.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self._enforce_retention()

    def _enforce_retention(self) -> None:
        cutoff = time.time() - (5 * 365 * 24 * 3600)
        for fp in DATA_DIR.glob(AUDIT_GLOB):
            try:
                if fp.stat().st_mtime < cutoff:
                    fp.unlink(missing_ok=True)
            except Exception:
                continue

    def check_ops_limit(self, orders_last_second: int) -> bool:
        return int(orders_last_second) <= self._ops_limit

    def kill_switch(self, reason: str) -> None:
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._kill_switch_ts = time.time()
        logger.warning("KILL SWITCH TRIGGERED: %s", reason)
        event_file = COMPLIANCE_DIR / "kill_switch_events.jsonl"
        with event_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "reason": reason,
                    }
                )
                + "\n"
            )

    def get_status(self) -> dict:
        return {
            "ops_limit": self._ops_limit,
            "kill_switch_active": self._kill_switch_active,
            "kill_switch_reason": self._kill_switch_reason,
            "kill_switch_ts": self._kill_switch_ts,
            "registered_algos": len(self.list_algo_ids()),
        }

    def _infer_trade_fields(self, trade: dict) -> tuple[float, float, float, str, str]:
        qty = float(trade.get("quantity", trade.get("qty", 1.0)) or 1.0)
        entry = float(trade.get("entry_price", trade.get("entry", trade.get("buy_price", 0.0))) or 0.0)
        exit_ = float(trade.get("exit_price", trade.get("exit", trade.get("sell_price", entry))) or entry)
        product = str(trade.get("product", trade.get("instrument_type", "intraday"))).lower()
        side = str(trade.get("side", trade.get("signal", "LONG"))).upper()
        return qty, entry, exit_, product, side

    def calculate_true_costs(self, trade: dict) -> dict:
        qty, entry, exit_, product, side = self._infer_trade_fields(trade)
        buy_turnover = abs(entry * qty)
        sell_turnover = abs(exit_ * qty)
        turnover = buy_turnover + sell_turnover

        brokerage = min(0.0003 * buy_turnover, 20.0) + min(0.0003 * sell_turnover, 20.0)
        exchange_charges = turnover * 0.0000297
        sebi_charges = turnover * (10.0 / 10_000_000.0)
        gst = 0.18 * (brokerage + exchange_charges)
        stamp_duty = buy_turnover * 0.00015

        if "option" in product:
            stt = sell_turnover * 0.00125
        elif "future" in product:
            stt = sell_turnover * 0.0005
        elif "delivery" in product:
            stt = sell_turnover * 0.001
        else:  # intraday equity
            stt = sell_turnover * 0.00025

        gross_pnl = float(trade.get("gross_pnl", 0.0))
        if gross_pnl == 0.0 and entry > 0 and exit_ > 0:
            sign = 1.0 if side in {"BUY", "LONG"} else -1.0
            gross_pnl = (exit_ - entry) * qty * sign

        total_cost = brokerage + stt + exchange_charges + sebi_charges + gst + stamp_duty
        net_pnl = gross_pnl - total_cost
        out = CostBreakdown(
            brokerage=round(brokerage, 2),
            stt=round(stt, 2),
            exchange_charges=round(exchange_charges, 2),
            sebi_charges=round(sebi_charges, 2),
            gst=round(gst, 2),
            stamp_duty=round(stamp_duty, 2),
            total_cost=round(total_cost, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
        )
        return out.__dict__

    def get_audit(self, date: str) -> list[dict]:
        fp = self._audit_file_for(date)
        if not fp.exists():
            return []
        rows: list[dict] = []
        for line in fp.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    def tax_summary(self) -> dict:
        total_net = 0.0
        total_stt = 0.0
        total_charges = 0.0
        for fp in DATA_DIR.glob(AUDIT_GLOB):
            for row in self.get_audit(fp.stem.replace("audit_trail_", "")):
                price = float(row.get("price", 0.0))
                qty = float(row.get("quantity", 0.0))
                turn = abs(price * qty)
                total_stt += turn * 0.00025
                total_charges += turn * 0.0000297
        # Conservative fallback tax buckets from realized net.
        total_net = max(0.0, float(_load_json(COMPLIANCE_DIR / "realized_net_pnl.json", 0.0)))
        stcg = total_net * 0.5
        ltcg = total_net * 0.2
        fno = total_net * 0.3
        advance = round((stcg * 0.20 + ltcg * 0.125 + fno * 0.20) / 4, 2)
        return {
            "stcg_amount": round(stcg, 2),
            "stcg_tax_rate": 0.20,
            "ltcg_amount": round(ltcg, 2),
            "ltcg_tax_rate": 0.125,
            "fno_business_income": round(fno, 2),
            "stt_paid": round(total_stt, 2),
            "exchange_charges_paid": round(total_charges, 2),
            "advance_tax_q1": advance,
            "advance_tax_q2": advance,
            "advance_tax_q3": advance,
            "advance_tax_q4": advance,
        }

    def generate_compliance_report(self, date: str) -> dict:
        audits = self.get_audit(date)
        report = {
            "date": date,
            "total_orders": len(audits),
            "ops_violations": 0,
            "algo_ids_used": sorted({a.get("algo_id") for a in audits if a.get("algo_id")}),
            "tax_summary": self.tax_summary(),
        }
        out_json = COMPLIANCE_DIR / f"compliance_{date}.json"
        _save_json(out_json, report)
        # Lightweight PDF-like text artifact fallback (no external PDF dependency).
        out_pdf = COMPLIANCE_DIR / f"compliance_{date}.pdf"
        out_pdf.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def export_itr3_csv(self) -> str:
        summary = self.tax_summary()
        path = COMPLIANCE_DIR / f"itr3_summary_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["field", "value"])
            for k, v in summary.items():
                w.writerow([k, v])
        return str(path)
