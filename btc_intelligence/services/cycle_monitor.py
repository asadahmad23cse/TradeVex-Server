from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CycleMonitorConfig:
    log_dir: str = "btc_intelligence/logs/monitor"
    max_log_lines: int = 10000
    signal_reject_alert_streak: int = 10
    drift_alert_threshold: float = 0.66
    drawdown_alert_pct: float = 0.03
    missing_source_alert: bool = True


@dataclass
class CycleMonitor:
    config: CycleMonitorConfig = field(default_factory=CycleMonitorConfig)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._consecutive_rejects: int = 0
        self._last_drift_alert: str = ""
        self._last_dd_alert: str = ""
        self._cycle_count: int = 0
        self._reject_reasons_tally: dict[str, int] = {}
        self._alerts_by_type: dict[str, int] = {}
        self._alerts_by_severity: dict[str, int] = {"INFO": 0, "WARN": 0, "CRITICAL": 0}

    @staticmethod
    def _minute_key(timestamp_utc: str) -> str:
        ts = str(timestamp_utc or "").strip()
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M")
            except Exception:
                pass
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")

    @staticmethod
    def _day_key(timestamp_utc: str) -> str:
        ts = str(timestamp_utc or "").strip()
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _extract_rejection_reasons(
        validation: dict[str, Any],
        meta_label: dict[str, Any],
        adaptive_adjustments: list[str],
    ) -> list[str]:
        reasons: list[str] = []
        for reason in list(meta_label.get("reasons", [])) if isinstance(meta_label, dict) else []:
            text = str(reason).strip()
            if text:
                reasons.append(text)
        checks = validation.get("checks", {}) if isinstance(validation, dict) else {}
        if isinstance(checks, dict):
            for key, passed in checks.items():
                if not bool(passed):
                    reasons.append(f"validation:{str(key)}")
        for adj in list(adaptive_adjustments or []):
            text = str(adj).strip()
            if text:
                reasons.append(f"adjustment:{text}")
        return reasons

    @staticmethod
    def _top_n(counter: dict[str, int], n: int = 5) -> dict[str, int]:
        ranked = sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        return {str(k): int(v) for k, v in ranked[:n]}

    @staticmethod
    def _next_rotated_path(log_dir: Path, day_key: str, minute_key: str) -> Path:
        base = log_dir / f"cycles_{day_key}_{minute_key}.jsonl"
        if not base.exists():
            return base
        idx = 1
        while True:
            cand = log_dir / f"cycles_{day_key}_{minute_key}_{idx}.jsonl"
            if not cand.exists():
                return cand
            idx += 1

    def _write_cycle_log(self, timestamp_utc: str, cycle_record: dict[str, Any]) -> None:
        day_key = self._day_key(timestamp_utc)
        minute_key = self._minute_key(timestamp_utc).replace(":", "-")
        base_path = self.log_dir / f"cycles_{day_key}.jsonl"

        if base_path.exists():
            line_count = 0
            with base_path.open("r", encoding="utf-8") as fh:
                for _ in fh:
                    line_count += 1
            if line_count >= int(self.config.max_log_lines):
                rotated = self._next_rotated_path(self.log_dir, day_key, minute_key[-5:])
                base_path.replace(rotated)

        with base_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(cycle_record, ensure_ascii=True) + "\n")

    def record_cycle(
        self,
        timestamp_utc: str,
        final_decision: str,
        confidence: float,
        validation: dict[str, Any],
        meta_label: dict[str, Any],
        drift: dict[str, Any],
        drawdown_status: dict[str, Any],
        signal_aggregation: dict[str, Any],
        walk_forward: dict[str, Any],
        kelly_sizing: dict[str, Any],
        strategy_used: str,
        adaptive_adjustments: list[str],
    ) -> dict[str, Any]:
        self._cycle_count += 1
        decision = str(final_decision or "HOLD").upper()
        if decision == "HOLD":
            self._consecutive_rejects += 1
        else:
            self._consecutive_rejects = 0

        rejection_reasons = self._extract_rejection_reasons(validation, meta_label, adaptive_adjustments)
        for reason in rejection_reasons:
            self._reject_reasons_tally[reason] = int(self._reject_reasons_tally.get(reason, 0)) + 1

        minute_key = self._minute_key(timestamp_utc)
        alerts: list[dict[str, str]] = []

        def add_alert(alert_type: str, severity: str, message: str) -> None:
            alert = {
                "type": alert_type,
                "severity": severity,
                "message": message,
                "timestamp": str(timestamp_utc),
            }
            alerts.append(alert)
            self._alerts_by_type[alert_type] = int(self._alerts_by_type.get(alert_type, 0)) + 1
            sev = str(severity).upper()
            self._alerts_by_severity[sev] = int(self._alerts_by_severity.get(sev, 0)) + 1

        if self._consecutive_rejects >= int(self.config.signal_reject_alert_streak):
            add_alert(
                "consecutive_rejects",
                "WARN",
                f"{self._consecutive_rejects} signals rejected in a row",
            )

        drift_score = float(drift.get("drift_score", 0.0)) if isinstance(drift, dict) else 0.0
        if drift_score >= float(self.config.drift_alert_threshold) and minute_key != self._last_drift_alert:
            add_alert(
                "high_drift",
                "CRITICAL",
                f"Drift score {drift_score:.3f} - system in caution mode",
            )
            self._last_drift_alert = minute_key

        drawdown_pct = float(drawdown_status.get("current_drawdown_pct", 0.0)) if isinstance(drawdown_status, dict) else 0.0
        halted = bool(drawdown_status.get("halted", False)) if isinstance(drawdown_status, dict) else False
        halt_reason = str(drawdown_status.get("halt_reason", "")) if isinstance(drawdown_status, dict) else ""
        if drawdown_pct >= float(self.config.drawdown_alert_pct) and not halted and minute_key != self._last_dd_alert:
            add_alert(
                "drawdown_warning",
                "WARN",
                f"Drawdown {drawdown_pct:.2%} approaching hard limit",
            )
            self._last_dd_alert = minute_key

        if halted:
            add_alert(
                "drawdown_halt",
                "CRITICAL",
                f"Circuit breaker ACTIVE - {halt_reason or 'halted'}",
            )

        missing_sources = list(signal_aggregation.get("missing_sources", [])) if isinstance(signal_aggregation, dict) else []
        if bool(self.config.missing_source_alert) and len(missing_sources) > 0:
            add_alert(
                "missing_sources",
                "WARN",
                f"Missing signal sources: {missing_sources}",
            )

        wf_recommendations: dict[str, str] = {}
        if isinstance(walk_forward, dict):
            for key, payload in walk_forward.items():
                if isinstance(payload, dict):
                    rec = str(payload.get("recommendation", ""))
                    if rec:
                        wf_recommendations[str(key)] = rec
                        if rec == "REJECT":
                            add_alert(
                                "walk_forward_reject",
                                "WARN",
                                f"WalkForward rejected weight update for {key}",
                            )

        if bool(kelly_sizing.get("halted", False)) if isinstance(kelly_sizing, dict) else False:
            add_alert(
                "kelly_halted",
                "CRITICAL",
                f"Kelly sizer halted: {kelly_sizing.get('halt_reason', 'unknown')}",
            )

        cycle_record: dict[str, Any] = {
            "cycle": int(self._cycle_count),
            "timestamp": str(timestamp_utc),
            "decision": decision,
            "confidence": float(confidence),
            "strategy": str(strategy_used),
            "consecutive_rejects": int(self._consecutive_rejects),
            "rejection_reasons": rejection_reasons,
            "drift_score": float(drift_score),
            "drift_level": str(drift.get("drift_level", "LOW")) if isinstance(drift, dict) else "LOW",
            "variance_drift_flags": dict(drift.get("variance_drift_flags", {})) if isinstance(drift, dict) else {},
            "psi_drift_score": float(drift.get("psi_drift_score", 0.0)) if isinstance(drift, dict) else 0.0,
            "drawdown_pct": float(drawdown_pct),
            "drawdown_action": str(drawdown_status.get("action", "NORMAL")) if isinstance(drawdown_status, dict) else "NORMAL",
            "kelly_position_pct": float(kelly_sizing.get("position_pct", 0.0)) if isinstance(kelly_sizing, dict) else 0.0,
            "kelly_halted": bool(kelly_sizing.get("halted", False)) if isinstance(kelly_sizing, dict) else False,
            "sources_missing": missing_sources,
            "wf_recommendations": wf_recommendations,
            "alerts": alerts,
            "reject_reasons_tally": self._top_n(self._reject_reasons_tally, n=5),
        }

        try:
            self._write_cycle_log(timestamp_utc=timestamp_utc, cycle_record=cycle_record)
        except Exception:
            pass
        return cycle_record

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_cycles": int(self._cycle_count),
            "consecutive_rejects": int(self._consecutive_rejects),
            "top_reject_reasons": self._top_n(self._reject_reasons_tally, n=5),
            "total_alerts_by_type": {str(k): int(v) for k, v in self._alerts_by_type.items()},
            "total_alerts_by_severity": {
                "INFO": int(self._alerts_by_severity.get("INFO", 0)),
                "WARN": int(self._alerts_by_severity.get("WARN", 0)),
                "CRITICAL": int(self._alerts_by_severity.get("CRITICAL", 0)),
            },
        }

