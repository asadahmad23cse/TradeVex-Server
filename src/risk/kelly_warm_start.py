"""BTC Kelly sizing with warm-started empirical buckets and Bayesian smoothing."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

logger = logging.getLogger(__name__)


class BTCKellyWarmStart:
    """Computes BTC position size from bucketed empirical trade outcomes."""

    PRIOR_WIN_RATE = 0.45
    PRIOR_AVG_RR = 2.0
    BUCKETS_FILE = "data/kelly_buckets.json"

    REGIME_MAP = {
        "BULLISH TREND": "BULL",
        "HIGH_VOL_BULL": "BULL",
        "BEARISH TREND": "BEAR",
        "HIGH_VOL_BEAR": "BEAR",
        "SIDEWAYS": "SIDEWAYS",
    }

    REGIME_MULTIPLIERS = {
        "BULL": 1.0,
        "SIDEWAYS": 0.6,
        "BEAR": 0.8,
    }

    def __init__(self, buckets_file: str | None = None, history_path: str = "data/signal_history.json") -> None:
        self._buckets_file = Path(buckets_file or self.BUCKETS_FILE)
        self._history_path = Path(history_path)
        self.buckets: dict[str, dict[str, Any]] = {}
        self._bucket_records: dict[str, list[dict[str, float]]] = {}
        self.total_trades = 0
        self._load_buckets()

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _strength_from_confidence(confidence: float) -> str:
        if confidence >= 80.0:
            return "STRONG"
        if confidence >= 65.0:
            return "MODERATE"
        return "WEAK"

    @classmethod
    def _regime_category(cls, regime: str) -> str:
        return cls.REGIME_MAP.get(str(regime or "").upper(), "SIDEWAYS")

    @staticmethod
    def _bucket_id(bucket_key: tuple[str, str, str]) -> str:
        return "/".join(bucket_key)

    @classmethod
    def _build_bucket_key(cls, signal: str, confidence: float, regime: str) -> tuple[str, str, str]:
        signal_direction = "SHORT" if str(signal or "").upper() == "SHORT" else "LONG"
        strength = cls._strength_from_confidence(float(confidence))
        regime_category = cls._regime_category(regime)
        return (signal_direction, strength, regime_category)

    @staticmethod
    def _method_from_total(total: int) -> str:
        if total >= 30:
            return "empirical"
        if total >= 10:
            return "bayesian_blend"
        return "bayesian_prior"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _default_bucket_stats(cls) -> dict[str, Any]:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "avg_rr": cls.PRIOR_AVG_RR,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "last_updated": cls._now_iso(),
            "smoothing_weight": 0.0,
        }

    def _record_to_sample(self, record: dict[str, Any]) -> dict[str, float] | None:
        status = str(record.get("status", "")).upper()
        if status in {"OPEN", "BLOCKED"}:
            return None

        signal = str(record.get("signal", "")).upper()
        if signal not in {"LONG", "SHORT"}:
            return None

        pnl_pct = self._safe_float(record.get("pnl_pct"), 0.0)
        rr = self._safe_float(record.get("risk_reward"), self.PRIOR_AVG_RR)
        if rr <= 0:
            rr = self.PRIOR_AVG_RR

        return {
            "pnl_pct": float(pnl_pct),
            "rr": float(rr),
        }

    def _recompute_bucket_stats(self, bucket_id: str) -> None:
        trades = self._bucket_records.get(bucket_id, [])
        if not trades:
            self.buckets[bucket_id] = self._default_bucket_stats()
            return

        pnl_values = [self._safe_float(t.get("pnl_pct"), 0.0) for t in trades]
        rr_values = [max(self._safe_float(t.get("rr"), self.PRIOR_AVG_RR), 0.01) for t in trades]
        wins = [p for p in pnl_values if p > 0]
        losses = [abs(p) for p in pnl_values if p <= 0]

        total = len(pnl_values)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total) if total > 0 else 0.0
        avg_win_pct = mean(wins) if wins else 0.0
        avg_loss_pct = mean(losses) if losses else 0.0
        avg_rr = mean(rr_values) if rr_values else self.PRIOR_AVG_RR
        expectancy = (win_rate * avg_win_pct) - ((1.0 - win_rate) * avg_loss_pct)
        sigma = pstdev(pnl_values) if len(pnl_values) > 1 else 0.0
        sharpe = (mean(pnl_values) / sigma) if sigma > 0 else 0.0
        smoothing_weight = min(total / 30.0, 1.0)

        self.buckets[bucket_id] = {
            "total": int(total),
            "wins": int(win_count),
            "losses": int(loss_count),
            "win_rate": float(win_rate),
            "avg_win_pct": float(avg_win_pct),
            "avg_loss_pct": float(avg_loss_pct),
            "avg_rr": float(avg_rr),
            "expectancy": float(expectancy),
            "sharpe": float(sharpe),
            "last_updated": self._now_iso(),
            "smoothing_weight": float(smoothing_weight),
        }

    def _refresh_totals(self) -> None:
        self.total_trades = sum(int(stats.get("total", 0)) for stats in self.buckets.values())

    def compute_btc_position(
        self,
        signal: str,
        confidence: float,
        regime: str,
        base_capital: float = 1000.0,
    ) -> dict[str, Any]:
        bucket_key = self._build_bucket_key(signal, confidence, regime)
        bucket_id = self._bucket_id(bucket_key)
        stats = dict(self.buckets.get(bucket_id, self._default_bucket_stats()))
        total = int(stats.get("total", 0))

        empirical_win_rate = float(stats.get("win_rate", 0.0))
        empirical_rr = float(stats.get("avg_rr", self.PRIOR_AVG_RR))
        if empirical_rr <= 0:
            empirical_rr = self.PRIOR_AVG_RR

        weight_empirical = min(total / 30.0, 1.0)
        weight_prior = 1.0 - weight_empirical
        smoothed_win_rate = (weight_empirical * empirical_win_rate) + (weight_prior * self.PRIOR_WIN_RATE)
        smoothed_rr = (weight_empirical * empirical_rr) + (weight_prior * self.PRIOR_AVG_RR)
        smoothed_rr = max(smoothed_rr, 0.01)

        p = smoothed_win_rate
        b = smoothed_rr
        q = 1.0 - p

        raw_kelly = max((p * b - q) / b, 0.0)
        quarter_kelly = raw_kelly * 0.25

        regime_category = bucket_key[2]
        regime_mult = self.REGIME_MULTIPLIERS.get(regime_category, 0.8)

        conf_scale = min((float(confidence) - 65.0) / 35.0, 1.0)
        conf_scale = max(conf_scale, 0.3)

        final_size = quarter_kelly * regime_mult * conf_scale
        final_size = max(final_size, 0.005)
        final_size = min(final_size, 0.05)

        return {
            "position_size_pct": round(final_size * 100.0, 2),
            "position_size_usd": round(float(base_capital) * final_size, 2),
            "raw_kelly": round(raw_kelly * 100.0, 2),
            "quarter_kelly": round(quarter_kelly * 100.0, 2),
            "method": self._method_from_total(total),
            "bucket_key": str(bucket_key),
            "bucket_stats": stats,
            "regime_mult": regime_mult,
            "conf_scale": round(conf_scale, 3),
            "smoothing_weight": round(weight_empirical, 3),
            "p": round(p, 4),
            "b": round(b, 4),
            "trades_in_bucket": total,
        }

    def _load_from_history_impl(self, history_file: Path, verbose: bool) -> None:
        trades: list[dict[str, Any]] = []
        if history_file.exists():
            try:
                raw = json.loads(history_file.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    trades = [r for r in raw if isinstance(r, dict)]
            except Exception as exc:
                logger.warning("Failed to read trade history from %s: %s", history_file, exc)

        self._bucket_records = {}
        self.buckets = {}
        loaded = 0
        for rec in trades:
            sample = self._record_to_sample(rec)
            if sample is None:
                continue
            bucket_key = self._build_bucket_key(
                signal=str(rec.get("signal", "LONG")),
                confidence=self._safe_float(rec.get("confidence"), 0.0),
                regime=str(rec.get("regime", "SIDEWAYS")),
            )
            bucket_id = self._bucket_id(bucket_key)
            self._bucket_records.setdefault(bucket_id, []).append(sample)
            loaded += 1

        for bucket_id in list(self._bucket_records.keys()):
            self._recompute_bucket_stats(bucket_id)
        self._refresh_totals()

        if verbose:
            print(f"Kelly warm-start: {loaded} trades loaded into {len(self.buckets)} buckets")
            for key in sorted(self.buckets.keys()):
                stats = self.buckets[key]
                total = int(stats.get("total", 0))
                method = self._method_from_total(total)
                print(
                    f"Bucket {key}: {total} trades | WR={stats.get('win_rate', 0.0):.0%} | "
                    f"RR={stats.get('avg_rr', 0.0):.1f} | E={stats.get('expectancy', 0.0):+.3f}% | "
                    f"Method={method}"
                )

    def load_from_history(self, history_path: str = "data/signal_history.json") -> None:
        self._load_from_history_impl(Path(history_path), verbose=True)

    def update_bucket(self, trade_record: dict) -> None:
        if not isinstance(trade_record, dict):
            return

        sample = self._record_to_sample(trade_record)
        if sample is None:
            return

        bucket_key = self._build_bucket_key(
            signal=str(trade_record.get("signal", "LONG")),
            confidence=self._safe_float(trade_record.get("confidence"), 0.0),
            regime=str(trade_record.get("regime", "SIDEWAYS")),
        )
        bucket_id = self._bucket_id(bucket_key)
        self._bucket_records.setdefault(bucket_id, []).append(sample)
        self._recompute_bucket_stats(bucket_id)
        self._refresh_totals()
        self.save_buckets()

    def save_buckets(self) -> None:
        self._buckets_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_trades": int(self.total_trades),
            "saved_at": self._now_iso(),
            "buckets": self.buckets,
            "bucket_records": self._bucket_records,
        }
        self._buckets_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _load_buckets(self) -> None:
        if self._buckets_file.exists():
            try:
                raw = json.loads(self._buckets_file.read_text(encoding="utf-8"))
                buckets = raw.get("buckets", {}) if isinstance(raw, dict) else {}
                bucket_records = raw.get("bucket_records", {}) if isinstance(raw, dict) else {}
                if isinstance(buckets, dict):
                    self.buckets = {str(k): v for k, v in buckets.items() if isinstance(v, dict)}
                else:
                    self.buckets = {}
                if isinstance(bucket_records, dict):
                    parsed: dict[str, list[dict[str, float]]] = {}
                    for key, items in bucket_records.items():
                        if not isinstance(items, list):
                            continue
                        parsed[str(key)] = [item for item in items if isinstance(item, dict)]
                    self._bucket_records = parsed
                else:
                    self._bucket_records = {}

                # If records are missing (older file format), keep stats and continue.
                if self._bucket_records:
                    for bucket_id in list(self._bucket_records.keys()):
                        self._recompute_bucket_stats(bucket_id)
                self._refresh_totals()
                return
            except Exception as exc:
                logger.warning("Failed to load Kelly buckets from %s: %s", self._buckets_file, exc)

        self._load_from_history_impl(self._history_path, verbose=False)
        self.save_buckets()
