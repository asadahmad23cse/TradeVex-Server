"""
Layer 8 â€” Live Scheduler.

Two APScheduler jobs:
  1. intraday_job  â€” every 5 minutes during market hours
  2. eod_job       â€” triggered after each market close

Market hours (UTC offsets used so no pytz dependency for basic checks):
  NSE:   09:15â€“15:30 IST  (UTC+5:30)  â†’ 03:45â€“10:00 UTC
  NYSE:  09:30â€“16:00 EST  (UTC-5)     â†’ 14:30â€“21:00 UTC
  Forex: 24/5 (always open on weekdays)
"""

import json
import logging
import time as time_module
import uuid
from datetime import datetime, timezone, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.api.connectors import MarketDataConnector
from src.api.data_quality import DataAnomalyDetector, SecondaryPriceValidator
from src.validator import WFOValidator
from src.features.engineer import FeatureEngineer
from src.features.hurst import compute_hurst
from src.alpha.factor_model import AlphaFactorModel
from src.alpha.regime import RegimeDetector
from src.execution.broker import create_executor
from src.execution.reconciliation import FillReconciler, ImpactCalibrator
from src.execution.state_machine import OrderStateRecord
from src.models.ensemble_model import EnsembleModel
from src.research.lookahead import LookaheadBiasAuditor
from src.research.stress import HistoricalStressTester
from src.research.capital_validation import PaperToLiveGraduator, LivePerformanceTracker
from src.execution.async_executor import AsyncExecutionPipeline
from src.signals.engine import SignalEngine
from src.signals.store import SignalStore
from src.risk.kelly import KellyCalculator, PortfolioRiskGuard
from src.risk.portfolio import PortfolioTracker
from src.scheduler.watchdog import SchedulerHealthWatchdog
from src.utils.alerts import fire_all_alerts, print_status
from src.utils.notifiers import NotificationManager

logger = logging.getLogger(__name__)

try:
    from src.models.trend_model import TrendPredictionModel
    _TF_AVAILABLE = True
except ImportError:
    TrendPredictionModel = None  # type: ignore[assignment,misc]
    _TF_AVAILABLE = False
    logger.warning("tensorflow not installed â€” LSTM factor (F4) will score 0.0")

try:
    from src.models.attention_model import TemporalAttentionModel
    _ATTN_AVAILABLE = True
except ImportError:
    TemporalAttentionModel = None  # type: ignore[assignment,misc]
    _ATTN_AVAILABLE = False

IST = ZoneInfo("Asia/Kolkata")
EST = ZoneInfo("America/New_York")


def _is_market_open(asset_class: str) -> bool:
    """Check if the relevant market is currently open."""
    now_utc = datetime.now(timezone.utc)
    weekday = now_utc.weekday()  # 0=Mon, 6=Sun

    if weekday >= 5:  # Saturday / Sunday
        if asset_class in ("indian_stock", "us_stock"):
            return False
        # Forex: closed Sun before 17:00 EST
        if weekday == 6:
            now_est = now_utc.astimezone(EST)
            return now_est.time() >= time(17, 0)

    if asset_class == "indian_stock":
        now_ist = now_utc.astimezone(IST)
        t = now_ist.time()
        return time(9, 15) <= t <= time(15, 30)

    if asset_class == "us_stock":
        now_est = now_utc.astimezone(EST)
        t = now_est.time()
        return time(9, 30) <= t <= time(16, 0)

    # Forex: open weekdays 24h (already handled weekends above)
    return True


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _plus_minutes(base: time, minutes: int) -> time:
    dt = datetime.combine(datetime.utcnow().date(), base) + timedelta(minutes=minutes)
    return dt.time()


class LiveRunner:
    """
    Orchestrates the full live signal pipeline via APScheduler.

    Usage:
        runner = LiveRunner("config.yaml")
        runner.start()
        # blocks until KeyboardInterrupt
    """

    def __init__(self, config_path: str = "config.yaml", db_path: str = "data/signals.db"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.connector = MarketDataConnector()
        self.engineer = FeatureEngineer()
        self.alpha_model = AlphaFactorModel(
            alpha_threshold=self.cfg["signal"]["alpha_score_threshold"],
            ic_window=self.cfg["signal"]["ic_window"],
        )
        self.regime_detector = RegimeDetector()
        self.signal_engine = SignalEngine(
            self.cfg.get("cost_model", {}),
            min_sqs=float(self.cfg.get("signal", {}).get("min_sqs", 55)),
            dedup_hours=float(self.cfg.get("signal", {}).get("dedup_hours", 4)),
            min_holding_candles=int(self.cfg.get("signal", {}).get("min_holding_candles", 2)),
        )
        db_url = self.cfg.get("database", {}).get("url", db_path)
        self.store = SignalStore(db_url)
        self.kelly = KellyCalculator(
            kelly_fraction=self.cfg["risk"]["kelly_fraction"],
            max_position_pct=self.cfg["risk"]["max_position_size_pct"] / 100,
            cold_start_pct=self.cfg["risk"]["cold_start_position_pct"] / 100,
        )
        self.risk_guard = PortfolioRiskGuard(self.cfg["risk"])
        self.portfolio = PortfolioTracker(
            initial_capital=float(self.cfg["portfolio"]["initial_capital"])
        )
        self.executor = create_executor(self.cfg)
        self.reconciler = FillReconciler(
            max_slippage_multiple=self.cfg.get("execution", {}).get("max_slippage_multiple", 2.0)
        )
        self.impact_calibrator = ImpactCalibrator(
            min_samples=self.cfg.get("execution", {}).get("eta_calibration_min_samples", 30)
        )
        self.notifier = NotificationManager(self.cfg.get("notifications", {}))
        self.watchdog = SchedulerHealthWatchdog(self.cfg.get("watchdog", {}))
        self.data_anomaly_detector = DataAnomalyDetector(
            stale_bars=self.cfg.get("data_quality", {}).get("stale_bars", 3),
            spike_sigma=self.cfg.get("data_quality", {}).get("spike_sigma", 5.0),
            min_spike_lookback=self.cfg.get("data_quality", {}).get("spike_lookback", 20),
        )
        secondary_cfg = self.cfg.get("secondary_validation", {})
        self.price_validator = SecondaryPriceValidator(
            polygon_api_key=secondary_cfg.get("polygon_api_key", ""),
            alpaca_api_key=secondary_cfg.get("alpaca_api_key", ""),
            alpaca_secret_key=secondary_cfg.get("alpaca_secret_key", ""),
            alpha_vantage_key=secondary_cfg.get("alpha_vantage_key", ""),
            mismatch_threshold_pct=secondary_cfg.get("mismatch_threshold_pct", 0.5),
        )
        self.lookahead_auditor = LookaheadBiasAuditor()
        self.stress_tester = HistoricalStressTester(config_path=config_path)
        self.lstm_model = TrendPredictionModel() if _TF_AVAILABLE else None
        self.attention_model = TemporalAttentionModel(
            self.cfg.get("attention_model", {})
        ) if _ATTN_AVAILABLE and TemporalAttentionModel is not None else None
        self.ensemble_models: dict[str, EnsembleModel] = {}
        self.async_pipeline = AsyncExecutionPipeline(
            max_workers=self.cfg.get("async", {}).get("max_workers", 4),
            feature_cache_ttl=self.cfg.get("data", {}).get("ttl_intraday_sec", 240),
        )
        self.graduator = PaperToLiveGraduator()
        self.live_tracker = LivePerformanceTracker()

        # Cache Hurst exponents (asset â†’ float), populated by EOD job
        self._hurst_cache: dict[str, float] = {}
        # Cache HMM models (asset_class â†’ RegimeDetector), populated by EOD job
        self._regime_cache: dict[str, RegimeDetector] = {
            asset_class: RegimeDetector.load_or_create(asset_class)
            for asset_class in ("indian_stock", "us_stock", "forex")
        }
        # Last HMM retrain date per asset_class (for 7-day gate)
        self._last_hmm_retrain: dict[str, datetime] = {}
        # Last WFO run date per asset_class (for 30-day gate)
        self._last_wfo_run: dict[str, datetime] = {}
        # Last stress-test run date per asset_class
        self._last_stress_test: dict[str, datetime] = {}
        self._stress_gate: dict[str, bool] = {
            "indian_stock": True,
            "us_stock": True,
            "forex": True,
        }
        # Price history for correlation filter {asset: pd.Series of daily closes}
        self._price_history: dict[str, pd.Series] = {}
        self._impact_observations: list[dict] = []
        self._intraday_cycles = 0
        self._consumed_daily_returns = 0
        self._last_latency_report: dict = {
            "cycle_id": "",
            "data_fetch_ms": 0.0,
            "feature_compute_ms": 0.0,
            "alpha_score_ms": 0.0,
            "risk_check_ms": 0.0,
            "execution_ms": 0.0,
            "total_ms": 0.0,
            "n_assets": 0,
            "errors": [],
        }
        self._last_live_validation: dict = {
            "graduation": self.graduator.to_dict(),
            "performance": self.live_tracker.metrics(),
            "as_of_utc": datetime.utcnow().isoformat(),
            "mode": "paper" if self.cfg.get("execution", {}).get("broker", "paper") == "paper" else "live",
        }

        self._regime_trained = False
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._ws_broadcast = None   # set by dashboard if WebSocket is running
        self.watchdog.register_job("intraday", self.cfg.get("watchdog", {}).get("intraday_expected_min", 7))
        self.watchdog.register_job("eod_indian_stock", self.cfg.get("watchdog", {}).get("eod_expected_min", 1440))
        self.watchdog.register_job("eod_us_stock", self.cfg.get("watchdog", {}).get("eod_expected_min", 1440))
        self.watchdog.register_job("eod_forex", self.cfg.get("watchdog", {}).get("eod_expected_min", 1440))
        self.watchdog.register_job("watchdog", self.cfg.get("watchdog", {}).get("watchdog_expected_min", 2))

    # ------------------------------------------------------------------
    # Scheduler setup
    # ------------------------------------------------------------------

    def set_broadcast(self, fn) -> None:
        """Inject a WebSocket broadcast function from the dashboard."""
        self._ws_broadcast = fn

    def get_latency_report(self) -> dict:
        """Expose latest intraday latency metrics for the dashboard API."""
        return dict(self._last_latency_report)

    def get_live_validation_report(self) -> dict:
        """Expose latest capital-validation report for the dashboard API."""
        self._refresh_live_validation()
        return dict(self._last_live_validation)

    def _estimate_fill_quality(self, lookback: int = 30) -> float:
        """
        Estimate fill-quality ratio from recent fills.
        <1 means better than expected slippage, >1 worse.
        """
        obs = self._impact_observations[-lookback:]
        if not obs:
            return 1.0
        ratios: list[float] = []
        for row in obs:
            expected = float(row.get("sigma", 0.0))
            actual = float(row.get("actual_slippage_pct", 0.0))
            if expected > 1e-9:
                ratios.append(actual / expected)
        if not ratios:
            return 1.0
        return float(sum(ratios) / len(ratios))

    def _refresh_live_validation(self) -> None:
        """
        Sync graduator/tracker with realised daily returns and update cached report.
        """
        daily_returns = list(getattr(self.portfolio, "daily_returns", []))
        broker_mode = str(self.cfg.get("execution", {}).get("broker", "paper")).lower()
        is_live = broker_mode not in {"paper", "sim", "simulator"}

        while self._consumed_daily_returns < len(daily_returns):
            ret = float(daily_returns[self._consumed_daily_returns])
            self.graduator.record_daily_return(ret, is_live=is_live)
            self.live_tracker.record(ret, fill_quality=self._estimate_fill_quality())
            self._consumed_daily_returns += 1

        report = self.graduator.evaluate()
        if report.get("should_retreat"):
            self.graduator.retreat()
        elif report.get("can_advance"):
            self.graduator.advance()

        self._last_live_validation = {
            "graduation": self.graduator.to_dict(),
            "performance": self.live_tracker.metrics(),
            "as_of_utc": datetime.utcnow().isoformat(),
            "mode": "live" if is_live else "paper",
        }

    def _log_health(self, component: str, status: str, message: str, details: dict | None = None) -> None:
        event = {
            "health_id": str(uuid.uuid4()),
            "component": component,
            "status": status,
            "message": message,
            "details": json.dumps(details or {}),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.store.save_system_health(event)
        if status.upper() in {"CRITICAL", "WARNING"}:
            self.notifier.notify(f"{component}: {status}", message, severity=status)

    def _save_data_quality(self, report) -> None:
        if not report.issue_types:
            return
        self.store.save_data_quality_event(
            {
                "event_id": str(uuid.uuid4()),
                "asset": report.asset,
                "asset_class": report.asset_class,
                "timeframe": report.timeframe,
                "severity": "CRITICAL" if report.severe else "WARNING",
                "issue_types": json.dumps(report.issue_types),
                "details": json.dumps(report.to_dict()),
                "timestamp": report.timestamp,
            }
        )
        if report.severe:
            self.notifier.notify(
                f"Data anomaly: {report.asset}",
                f"{report.issue_types} on {report.asset} ({report.timeframe})",
                severity="CRITICAL",
            )

    def _save_model_validation(self, model_name: str, asset_class: str, metrics: dict, top_features: list | None = None) -> None:
        self.store.save_model_validation(
            {
                "validation_id": str(uuid.uuid4()),
                "model_name": model_name,
                "asset_class": asset_class,
                "metrics": json.dumps(metrics),
                "top_features": json.dumps(top_features or []),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def _save_reconciliation(self, event) -> None:
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        self.store.save_reconciliation_event(
            {
                "event_id": str(uuid.uuid4()),
                "scope": data.get("scope", "reconciliation"),
                "asset": data.get("asset", ""),
                "status": data.get("status", "UNKNOWN"),
                "severity": data.get("severity", "INFO"),
                "message": data.get("message", ""),
                "details": json.dumps(data.get("details", {})),
                "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
            }
        )
        if data.get("severity", "INFO").upper() == "CRITICAL":
            self.notifier.notify(
                f"Reconciliation: {data.get('status', 'UNKNOWN')}",
                data.get("message", ""),
                severity="CRITICAL",
            )

    def _record_order(self, signal, receipt) -> OrderStateRecord:
        state = str(getattr(receipt, "status", "PENDING")).upper()
        if state == "PLACED":
            state = "SUBMITTED"
        record = OrderStateRecord(
            order_id=str(getattr(receipt, "order_id", str(uuid.uuid4()))),
            signal_id=signal.signal_id,
            asset=signal.asset,
            side=signal.signal,
            broker=str(getattr(receipt, "broker", "paper")),
            state="PENDING",
            requested_price=signal.entry_price,
            fill_price=float(getattr(receipt, "fill_price", 0.0) or 0.0),
            requested_qty=float(signal.position_size_pct),
            filled_qty=float(signal.position_size_pct),
            error=str(getattr(receipt, "error", "") or ""),
        )
        record.transition(state if state in {"SUBMITTED", "PARTIAL", "FILLED", "REJECTED", "EXPIRED", "CANCELLED"} else "SUBMITTED")
        self.store.save_order(
            {
                "order_id": record.order_id,
                "signal_id": record.signal_id,
                "asset": record.asset,
                "side": record.side,
                "broker": record.broker,
                "state": record.state,
                "requested_price": record.requested_price,
                "fill_price": record.fill_price,
                "requested_qty": record.requested_qty,
                "filled_qty": record.filled_qty,
                "slippage_pct": getattr(receipt, "slippage_pct", 0.0),
                "expected_slippage_pct": signal.slippage_cost_pct * 100,
                "error": record.error,
                "broker_payload": json.dumps(receipt.to_dict() if hasattr(receipt, "to_dict") else {}),
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
            }
        )
        self.store.save_order_event(
            {
                "event_id": str(uuid.uuid4()),
                "order_id": record.order_id,
                "old_state": "PENDING",
                "new_state": record.state,
                "event_type": "execution",
                "details": json.dumps(receipt.to_dict() if hasattr(receipt, "to_dict") else {}),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return record

    def _reconcile_positions(self, scope: str) -> None:
        try:
            internal = self.portfolio.get_open_positions_list()
            broker_positions = self.executor.get_open_positions()
            for result in self.reconciler.reconcile_positions(internal, broker_positions):
                result.scope = scope
                self._save_reconciliation(result)
        except Exception as exc:
            self._log_health("reconciliation", "WARNING", f"Position reconciliation failed: {exc}")

    def _run_watchdog(self) -> None:
        self.watchdog.heartbeat("watchdog")
        for job in self.watchdog.check():
            if self.watchdog.due_for_alert(job):
                self._log_health("scheduler", "CRITICAL", job.message, {"job": job.name})
                self.watchdog.mark_alerted(job)

    def _run_stress_if_due(self, asset_class: str) -> bool:
        if asset_class == "forex":
            return True

        cadence_days = self.cfg.get("data", {}).get("stress_test_run_days", 30)
        last = self._last_stress_test.get(asset_class)
        now = datetime.utcnow()
        if last is not None and (now - last).days < cadence_days:
            return self._stress_gate.get(asset_class, True)

        ticker = self._benchmark_ticker(asset_class)
        try:
            stress_df = self.stress_tester.run(ticker=ticker, asset_class=asset_class)
            gate = HistoricalStressTester.gate_result(stress_df, required_survivals=2)
            self._stress_gate[asset_class] = bool(gate["passed"])
            self._last_stress_test[asset_class] = now
            self._log_health(
                "stress_test",
                "INFO" if gate["passed"] else "CRITICAL",
                (
                    f"Stress gate {'passed' if gate['passed'] else 'failed'} for {asset_class} "
                    f"using {ticker}: survived {gate['survived_crises']}/{len(stress_df)} crises"
                ),
                {
                    "asset_class": asset_class,
                    "ticker": ticker,
                    "gate": gate,
                    "windows": stress_df.to_dict(orient="records"),
                },
            )
        except Exception as exc:
            self._log_health("stress_test", "WARNING", f"Stress test failed for {asset_class}: {exc}")
        return self._stress_gate.get(asset_class, True)

    def _combined_ml_score(self, symbol: str, asset_class: str, df: pd.DataFrame, entry_price: float) -> float:
        scores: list[float] = []
        _lstm = self.lstm_model
        if _lstm is not None and _lstm.model is not None and len(df) >= _lstm.sequence_length:
            seq_df = df[["Close"]].tail(_lstm.sequence_length)
            scores.append(_lstm.directional_score(seq_df, entry_price))

        ensemble = self.ensemble_models.get(symbol)
        if ensemble is not None and ensemble.is_trained:
            if "Ensemble_Score" in df.columns:
                scores.append(float(df["Ensemble_Score"].iloc[-1]))
            else:
                scores.append(ensemble.directional_score(df))

        # Temporal Attention Transformer
        _attn = self.attention_model
        if _attn is not None and _attn.is_trained:
            attn_score = _attn.directional_score(df)
            if attn_score != 0.0:
                scores.append(attn_score)

        return float(sum(scores) / len(scores)) if scores else 0.0

    def _attach_ensemble_score(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        ensemble = self.ensemble_models.get(symbol)
        if ensemble is None or not ensemble.is_trained:
            return df
        try:
            enriched = df.copy()
            enriched["Ensemble_Score"] = ensemble.predict_series(df)
            return enriched
        except Exception as exc:
            logger.warning("Ensemble score injection failed for %s: %s", symbol, exc)
            return df

    def _execute_and_track(self, sig) -> bool:
        receipt = self.executor.execute(sig)
        order_record = self._record_order(sig, receipt)

        sig.execution_price = float(getattr(receipt, "fill_price", 0.0) or 0.0)
        if sig.execution_price > 0:
            direction = 1.0 if sig.signal == "BUY" else -1.0
            sig.implementation_shortfall_pct = round(
                direction * (sig.execution_price - sig.entry_price) / max(sig.entry_price, 1e-9) * 100,
                6,
            )

        for event in self.reconciler.reconcile_receipt(sig, receipt):
            self._save_reconciliation(event)

        if order_record.state not in {"FILLED", "PARTIAL", "SUBMITTED"}:
            return False

        self.store.save_signal(sig)
        if order_record.state in {"FILLED", "PARTIAL"}:
            self.portfolio.open_position(sig)
        fire_all_alerts(sig)
        if self._ws_broadcast:
            self._ws_broadcast(sig.to_dict())

        self._impact_observations.append(
            {
                "actual_slippage_pct": abs(sig.implementation_shortfall_pct),
                "spread_pct": sig.cost_pct / 2,
                "sigma": max(sig.slippage_cost_pct, 1e-6),
                "participation": max(sig.position_size_pct / 100, 1e-6),
                "adv_ratio": 1.0,
            }
        )
        if len(self._impact_observations) >= self.cfg.get("execution", {}).get("eta_calibration_min_samples", 30):
            fit = self.impact_calibrator.calibrate_eta(
                self._impact_observations,
                default_eta=self.signal_engine._cost_model.eta,
            )
            if fit.get("fitted"):
                self.signal_engine._cost_model.update_eta(fit["eta"])
                self._log_health("cost_model", "INFO", fit["message"], fit)
        self._reconcile_positions("post_order")
        return True

    def _benchmark_ticker(self, asset_class: str) -> str:
        return {
            "indian_stock": self.cfg["benchmarks"]["nifty"],
            "us_stock": self.cfg["benchmarks"]["sp500"],
            "forex": self.cfg["benchmarks"]["dxy"],
        }[asset_class]

    def _benchmark_returns(self, asset_class: str, timeframe: str) -> pd.Series | None:
        ticker = self._benchmark_ticker(asset_class)
        if timeframe == "intraday":
            df = self.connector.get_intraday(
                ticker,
                interval=self.cfg["data"]["intraday_interval"],
                period=self.cfg["data"]["intraday_lookback"],
            )
        else:
            df = self.connector.get_daily(
                ticker,
                period=self.cfg["data"]["daily_lookback"],
            )
        if df.empty:
            return None
        return df["Close"].pct_change()

    def _regime_input_df(self, asset_class: str, timeframe: str) -> pd.DataFrame:
        ticker = self._benchmark_ticker(asset_class)
        if timeframe == "intraday":
            df = self.connector.get_intraday(
                ticker,
                interval=self.cfg["data"]["intraday_interval"],
                period=self.cfg["data"]["intraday_lookback"],
            )
            return self.engineer.compute_all_features(
                df,
                timeframe="intraday",
                benchmark=self._benchmark_returns(asset_class, timeframe),
            )
        df = self.connector.get_daily(ticker, period=self.cfg["data"]["daily_lookback"])
        return self.engineer.compute_all_features(
            df,
            timeframe="daily",
            benchmark=self._benchmark_returns(asset_class, timeframe),
        )

    def start(self) -> None:
        # Intraday: every 5 minutes
        self._scheduler.add_job(
            self._intraday_cycle,
            IntervalTrigger(minutes=5),
            id="intraday",
            max_instances=1,
            misfire_grace_time=60,
        )

        nse_close = _plus_minutes(_parse_hhmm(self.cfg["market_hours"]["nse"]["close"]), 5)
        self._scheduler.add_job(
            lambda: self._eod_cycle("indian_stock"),
            CronTrigger(
                hour=nse_close.hour,
                minute=nse_close.minute,
                day_of_week="mon-fri",
                timezone=self.cfg["market_hours"]["nse"]["timezone"],
            ),
            id="eod_nse",
        )

        nyse_close = _plus_minutes(_parse_hhmm(self.cfg["market_hours"]["nyse"]["close"]), 5)
        self._scheduler.add_job(
            lambda: self._eod_cycle("us_stock"),
            CronTrigger(
                hour=nyse_close.hour,
                minute=nyse_close.minute,
                day_of_week="mon-fri",
                timezone=self.cfg["market_hours"]["nyse"]["timezone"],
            ),
            id="eod_nyse",
        )

        # EOD Forex: 22:00 UTC (after London/NY overlap)
        self._scheduler.add_job(
            lambda: self._eod_cycle("forex"),
            CronTrigger(hour=22, minute=0, day_of_week="mon-fri", timezone="UTC"),
            id="eod_forex",
        )

        self._scheduler.add_job(
            self._run_watchdog,
            IntervalTrigger(minutes=1),
            id="watchdog",
            max_instances=1,
            misfire_grace_time=30,
        )

        # Run an immediate intraday cycle so signals appear right away
        import threading
        threading.Thread(
            target=self._safe_immediate_cycle,
            daemon=True,
        ).start()

        self._scheduler.start()
        print_status("LiveRunner started. Press Ctrl+C to stop.", "OK")

        try:
            import time
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        print_status("LiveRunner stopped.", "WARN")

    def _safe_immediate_cycle(self) -> None:
        """Run a single intraday cycle immediately so signals appear at startup."""
        import time as _time
        _time.sleep(3)  # wait for FastAPI to bind
        try:
            logger.info("Running immediate intraday cycle for real-time signalsâ€¦")
            self._intraday_cycle()
            logger.info("Immediate intraday cycle complete â€” signals are now live.")
        except Exception as exc:
            logger.warning("Immediate cycle skipped: %s", exc)

    # ------------------------------------------------------------------
    # Intraday cycle
    # ------------------------------------------------------------------

    def _intraday_cycle(self) -> None:
        logger.info("=== Intraday cycle: %s UTC ===", datetime.utcnow().strftime("%H:%M:%S"))
        self._intraday_cycles += 1
        cycle_start_ms = time_module.perf_counter() * 1000.0
        timing_totals = {
            "data_fetch_ms": 0.0,
            "feature_compute_ms": 0.0,
            "alpha_score_ms": 0.0,
            "risk_check_ms": 0.0,
            "execution_ms": 0.0,
        }
        cycle_errors: list[str] = []
        n_assets_timed = 0

        all_assets = (
            [(a, "indian_stock") for a in self.cfg["watchlist"]["indian_stocks"]] +
            [(a, "us_stock") for a in self.cfg["watchlist"]["us_stocks"]] +
            [(a, "forex") for a in self.cfg["watchlist"]["forex"]]
        )

        price_map: dict[str, float] = {}
        benchmark_returns = {
            "indian_stock": self._benchmark_returns("indian_stock", "intraday"),
            "us_stock": self._benchmark_returns("us_stock", "intraday"),
            "forex": self._benchmark_returns("forex", "intraday"),
        }
        regime_state: dict[str, tuple[str, dict]] = {}
        for asset_class in ("indian_stock", "us_stock", "forex"):
            regime_det = self._regime_cache.get(asset_class)
            if regime_det is None or not regime_det._trained:
                regime_state[asset_class] = ("SIDEWAYS", {})
                continue
            try:
                regime_df = self._regime_input_df(asset_class, "intraday")
                if regime_df.empty:
                    regime_state[asset_class] = ("SIDEWAYS", {})
                else:
                    if self._intraday_cycles % 5 == 0:
                        regime_det.online_update(
                            regime_df,
                            retrain_window=self.cfg["data"]["hmm_train_window"],
                        )
                    regime_state[asset_class] = regime_det.predict(regime_df)
            except Exception as exc:
                logger.warning("Regime inference fallback for %s: %s", asset_class, exc)
                regime_state[asset_class] = ("SIDEWAYS", {})
        portfolio_metrics = self.portfolio.get_metrics()

        for asset_info, asset_class in all_assets:
            if not _is_market_open(asset_class):
                continue

            symbol = asset_info["symbol"]
            yf_ticker = asset_info["yf_ticker"]

            try:
                fetch_start_ms = time_module.perf_counter() * 1000.0
                df = self.connector.get_intraday(
                    yf_ticker,
                    interval=self.cfg["data"]["intraday_interval"],
                    period=self.cfg["data"]["intraday_lookback"],
                )
                timing_totals["data_fetch_ms"] += (time_module.perf_counter() * 1000.0 - fetch_start_ms)
                if df.empty or len(df) < 30:
                    continue

                feature_start_ms = time_module.perf_counter() * 1000.0
                df, dq_report = self.data_anomaly_detector.inspect_and_clean(
                    df, symbol, asset_class, "intraday"
                )
                self._save_data_quality(dq_report)
                if dq_report.severe:
                    logger.warning("Suppressing %s due to severe intraday data anomaly", symbol)
                    continue

                # Feature engineering (skip Hurst; use cached value)
                df = self.engineer.compute_all_features(
                    df,
                    timeframe="intraday",
                    benchmark=benchmark_returns.get(asset_class),
                )
                timing_totals["feature_compute_ms"] += (time_module.perf_counter() * 1000.0 - feature_start_ms)
                if df.empty:
                    continue
                n_assets_timed += 1

                entry_price = float(df["Close"].iloc[-1])
                if asset_class == "indian_stock":
                    live_price = self.connector.get_nse_live(symbol, yf_ticker)
                    if live_price is not None:
                        entry_price = float(live_price)
                price_map[symbol] = entry_price
                atr_14 = float(df["ATR_14"].iloc[-1])
                hurst = self._hurst_cache.get(symbol, 0.5)
                daily_vol = float(df.get("Volatility_20", pd.Series([0.2], index=df.index)).iloc[-1] or 0.2)
                volume_ratio = float(df.get("Volume_Ratio", pd.Series([1.0], index=df.index)).iloc[-1] or 1.0)

                regime_det = self._regime_cache.get(asset_class)
                regime, regime_probs = regime_state.get(asset_class, ("SIDEWAYS", {}))
                df = self._attach_ensemble_score(symbol, df)

                alpha_start_ms = time_module.perf_counter() * 1000.0
                ml_score = self._combined_ml_score(symbol, asset_class, df, entry_price)
                alpha_result = self.alpha_model.score(
                    df,
                    ml_score=ml_score,
                    hurst=hurst,
                    asset=symbol,
                    asset_class=asset_class,
                    regime=regime,
                )
                alpha_result["atr_percentile"] = float(
                    df.get("ATR_Percentile", pd.Series([50.0], index=df.index)).iloc[-1]
                )
                timing_totals["alpha_score_ms"] += (time_module.perf_counter() * 1000.0 - alpha_start_ms)

                if alpha_result["signal"] == "HOLD":
                    continue

                risk_start_ms = time_module.perf_counter() * 1000.0
                allows = regime_det.allows_signal(
                    regime, alpha_result["signal"], alpha_result["strength"]
                ) if regime_det and regime_det._trained else True
                size_mult = (
                    regime_det.sideways_size_multiplier(regime)
                    if regime_det and regime_det._trained
                    else 1.0
                )

                bucket = self.store.get_bucket_stats(
                    asset_class,
                    alpha_result["strength"],
                    regime,
                    signal=alpha_result["signal"],
                )
                kelly_result = self.kelly.compute(
                    asset_class,
                    alpha_result["signal"],
                    alpha_result["strength"],
                    regime,
                    bucket,
                )

                sig = self.signal_engine.generate(
                    asset=symbol,
                    asset_class=asset_class,
                    timeframe="intraday",
                    alpha_result=alpha_result,
                    regime=regime,
                    regime_allows=allows,
                    hurst=hurst,
                    entry_price=entry_price,
                    atr_14=atr_14,
                    kelly_fraction=kelly_result["kelly_fraction"],
                    position_size_pct=kelly_result["position_size_pct"],
                    regime_size_mult=size_mult,
                    daily_vol=daily_vol,
                    volume_ratio=volume_ratio,
                )
                if sig is None:
                    continue

                open_pos = self.store.get_open_signals()
                allowed, reason = self.risk_guard.check_all(
                    sig,
                    open_pos,
                    self.portfolio.daily_pnl_pct,
                    portfolio_var_pct=portfolio_metrics.get("var_95", 0.0),
                    portfolio_cvar_pct=portfolio_metrics.get("cvar_95", 0.0),
                    regime=regime,
                    price_history=self._price_history,
                )
                if not allowed:
                    logger.info("Signal blocked for %s: %s", symbol, reason)
                    continue

                timing_totals["risk_check_ms"] += (time_module.perf_counter() * 1000.0 - risk_start_ms)
                exec_start_ms = time_module.perf_counter() * 1000.0
                self._execute_and_track(sig)
                timing_totals["execution_ms"] += (time_module.perf_counter() * 1000.0 - exec_start_ms)

            except Exception as exc:
                cycle_errors.append(f"{symbol}: {exc}")
                logger.error("Intraday cycle error for %s: %s", symbol, exc, exc_info=True)

        close_events = self.portfolio.update_prices(price_map)
        for evt in close_events:
            self.store.close_signal(**evt)

        snap = self.portfolio.get_snapshot_dict()
        self.store.save_snapshot(snap)
        self._reconcile_positions("intraday_cycle")
        self.watchdog.heartbeat("intraday")
        self._refresh_live_validation()

        n = max(n_assets_timed, 1)
        self._last_latency_report = {
            "cycle_id": datetime.utcnow().strftime("intraday_%Y%m%d_%H%M%S"),
            "data_fetch_ms": round(timing_totals["data_fetch_ms"] / n, 1),
            "feature_compute_ms": round(timing_totals["feature_compute_ms"] / n, 1),
            "alpha_score_ms": round(timing_totals["alpha_score_ms"] / n, 1),
            "risk_check_ms": round(timing_totals["risk_check_ms"] / n, 1),
            "execution_ms": round(timing_totals["execution_ms"] / n, 1),
            "total_ms": round(time_module.perf_counter() * 1000.0 - cycle_start_ms, 1),
            "n_assets": n_assets_timed,
            "errors": cycle_errors[-20:],
        }

    # ------------------------------------------------------------------
    # EOD cycle
    # ------------------------------------------------------------------

    def _eod_cycle(self, asset_class: str) -> None:
        logger.info("=== EOD cycle (%s): %s UTC ===", asset_class, datetime.utcnow().strftime("%H:%M"))

        assets = self.cfg["watchlist"].get(
            "indian_stocks" if asset_class == "indian_stock"
            else "us_stocks" if asset_class == "us_stock"
            else "forex",
            []
        )

        all_dfs = []
        benchmark_returns = self._benchmark_returns(asset_class, "daily")
        for asset_info in assets:
            symbol = asset_info["symbol"]
            yf_ticker = asset_info["yf_ticker"]

            try:
                if asset_class == "forex":
                    df = self.connector.get_forex_eod(
                        asset_info.get("av_from", "EUR"),
                        asset_info.get("av_to", "USD"),
                    )
                    if df.empty:
                        df = self.connector.get_daily(yf_ticker)
                else:
                    df = self.connector.get_daily(yf_ticker)

                if df.empty or len(df) < 60:
                    continue

                dq_df, dq_report = self.data_anomaly_detector.inspect_and_clean(
                    df, symbol, asset_class, "daily"
                )
                self._save_data_quality(dq_report)
                if dq_report.severe:
                    logger.warning("Suppressing %s due to severe daily data anomaly", symbol)
                    continue

                validation = self.price_validator.validate_daily_close(asset_info, asset_class, dq_df)
                if validation.available and validation.flagged:
                    self.store.save_data_quality_event(
                        {
                            "event_id": str(uuid.uuid4()),
                            "asset": symbol,
                            "asset_class": asset_class,
                            "timeframe": "daily",
                            "severity": "CRITICAL",
                            "issue_types": json.dumps(["secondary_price_mismatch"]),
                            "details": json.dumps(validation.to_dict()),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    logger.warning("Suppressing %s due to secondary source mismatch", symbol)
                    continue
                df = dq_df

                df = self.engineer.compute_all_features(
                    df,
                    timeframe="daily",
                    benchmark=benchmark_returns,
                )
                if df.empty:
                    continue

                audit = self.lookahead_auditor.audit(dq_df, df)
                if not audit.passed:
                    self._log_health(
                        "lookahead_audit",
                        "CRITICAL",
                        f"Lookahead audit failed for {symbol}",
                        {"asset": symbol, "issues": audit.issues},
                    )
                    continue

                # Update price history for correlation filter (last 30 daily closes)
                self._price_history[symbol] = df["Close"].tail(30)

                # Compute & cache Hurst exponent
                hurst = compute_hurst(df["Close"].tail(self.cfg["data"]["hurst_window"]))
                self._hurst_cache[symbol] = hurst
                logger.info("Hurst(%s) = %.3f", symbol, hurst)

                all_dfs.append((symbol, df))

            except Exception as exc:
                logger.error("EOD data error for %s: %s", symbol, exc)

        # Retrain HMM if enough data
        if all_dfs:
            regime_df = self._regime_input_df(asset_class, "daily")
            if not regime_df.empty:
                self._train_regime(asset_class, regime_df)

        regime_det = self._regime_cache.get(asset_class)
        regime = "SIDEWAYS"
        if regime_det and regime_det._trained:
            try:
                regime_df = self._regime_input_df(asset_class, "daily")
                if not regime_df.empty:
                    regime, _ = regime_det.predict(regime_df)
            except Exception as exc:
                logger.warning("EOD regime fallback for %s: %s", asset_class, exc)

        for symbol, df in all_dfs:
            ensemble = self.ensemble_models.get(symbol)
            if ensemble is None:
                ensemble = EnsembleModel(self.cfg.get("ensemble", {}))
                self.ensemble_models[symbol] = ensemble
            if ensemble.needs_retrain():
                try:
                    if ensemble.train(
                        df,
                        forward_horizon=self.cfg.get("ensemble", {}).get("forward_horizon", 5),
                        oot_days=self.cfg.get("ml_validation", {}).get("oot_days", 21),
                        refit_meta_walkforward=True,
                    ):
                        self._save_model_validation(
                            "ensemble",
                            asset_class,
                            ensemble.validation_report,
                            ensemble.top_features,
                        )
                except Exception as exc:
                    self._log_health("ensemble", "WARNING", f"Ensemble retrain failed for {symbol}: {exc}")

        portfolio_metrics = self.portfolio.get_metrics()
        stress_gate_passed = self._run_stress_if_due(asset_class)

        # Generate swing signals
        for symbol, df in all_dfs:
            try:
                if not stress_gate_passed:
                    logger.warning("Skipping %s swing signals because stress gate is closed", asset_class)
                    break
                entry_price = float(df["Close"].iloc[-1])
                atr_14 = float(df["ATR_14"].iloc[-1])
                hurst = self._hurst_cache.get(symbol, 0.5)
                daily_vol = float(df.get("Volatility_20", pd.Series([0.2], index=df.index)).iloc[-1] or 0.2)
                volume_ratio = float(df.get("Volume_Ratio", pd.Series([1.0], index=df.index)).iloc[-1] or 1.0)

                _lstm = self.lstm_model
                if _lstm is not None and len(df) >= max(_lstm.sequence_length + 30, 80) and _lstm.needs_retrain(
                    drop_threshold=self.cfg.get("ml_validation", {}).get("lstm_min_oot_accuracy", 0.52),
                    retrain_days=self.cfg.get("ml_validation", {}).get("lstm_retrain_days", 63),
                ):
                    try:
                        _lstm.train(df[["Close"]], epochs=5, verbose=0, oot_days=self.cfg.get("ml_validation", {}).get("oot_days", 21))
                        self._save_model_validation("lstm", asset_class, _lstm.validation_report, [{"feature": "Close", "importance": 1.0, "rank": 1}])
                    except Exception as exc:
                        self._log_health("lstm", "WARNING", f"LSTM retrain failed for {symbol}: {exc}")

                df = self._attach_ensemble_score(symbol, df)
                ml_score = self._combined_ml_score(symbol, asset_class, df, entry_price)

                alpha_result = self.alpha_model.score(
                    df,
                    ml_score=ml_score,
                    hurst=hurst,
                    asset=symbol,
                    asset_class=asset_class,
                    regime=regime,
                )
                alpha_result["atr_percentile"] = float(
                    df.get("ATR_Percentile", pd.Series([50.0], index=df.index)).iloc[-1]
                )
                if alpha_result["signal"] == "HOLD":
                    continue

                allows = regime_det.allows_signal(
                    regime, alpha_result["signal"], alpha_result["strength"]
                ) if regime_det and regime_det._trained else True
                size_mult = (regime_det.sideways_size_multiplier(regime)
                             if regime_det and regime_det._trained else 1.0)

                bucket = self.store.get_bucket_stats(
                    asset_class,
                    alpha_result["strength"],
                    regime,
                    signal=alpha_result["signal"],
                )
                kelly_result = self.kelly.compute(
                    asset_class,
                    alpha_result["signal"],
                    alpha_result["strength"],
                    regime,
                    bucket,
                )

                sig = self.signal_engine.generate(
                    asset=symbol,
                    asset_class=asset_class,
                    timeframe="swing",
                    alpha_result=alpha_result,
                    regime=regime,
                    regime_allows=allows,
                    hurst=hurst,
                    entry_price=entry_price,
                    atr_14=atr_14,
                    kelly_fraction=kelly_result["kelly_fraction"],
                    position_size_pct=kelly_result["position_size_pct"],
                    regime_size_mult=size_mult,
                    daily_vol=daily_vol,
                    volume_ratio=volume_ratio,
                )
                if sig:
                    open_pos = self.store.get_open_signals()
                    allowed, reason = self.risk_guard.check_all(
                        sig,
                        open_pos,
                        self.portfolio.daily_pnl_pct,
                        portfolio_var_pct=portfolio_metrics.get("var_95", 0.0),
                        portfolio_cvar_pct=portfolio_metrics.get("cvar_95", 0.0),
                        regime=regime,
                        price_history=self._price_history,
                    )
                    if not allowed:
                        logger.info("Swing signal blocked for %s: %s", symbol, reason)
                        continue
                    self._execute_and_track(sig)

            except Exception as exc:
                logger.error("EOD signal error for %s: %s", symbol, exc)

        self._run_wfo_if_due(asset_class, assets)
        self._reconcile_positions(f"eod_{asset_class}")
        self.watchdog.heartbeat(f"eod_{asset_class}")
        self._refresh_live_validation()

    def _run_wfo_if_due(self, asset_class: str, assets: list) -> None:
        """
        Run WFO parameter validation if 30+ days since last run for this asset_class.
        Per spec Â§10 EOD loop step 5.
        """
        wfo_days = self.cfg["data"].get("wfo_run_days", 30)
        last = self._last_wfo_run.get(asset_class)
        now = datetime.utcnow()
        if last is not None and (now - last).days < wfo_days:
            logger.debug(
                "WFO skipped for %s â€” last run %d days ago (threshold=%d)",
                asset_class, (now - last).days, wfo_days,
            )
            return
        if not assets:
            return
        # Pick representative ticker (first asset in class)
        rep_ticker = assets[0].get("yf_ticker", "")
        if not rep_ticker:
            return
        try:
            wfo_cfg = self.cfg.get("wfo", {})
            validator = WFOValidator(
                ticker=rep_ticker,
                train_window=wfo_cfg.get("min_train_days", 252),
                wfo_config=wfo_cfg,
            )
            result = validator.run_validation()
            self._last_wfo_run[asset_class] = now
            if result.get("passed"):
                logger.info(
                    "WFO PASSED for %s (%s) â€” best: alpha_thr=%.2f, ic_win=%d, OOS IR=%.3f",
                    asset_class, rep_ticker,
                    result["best_alpha_threshold"], result["best_ic_window"], result["best_ir"],
                )
            else:
                logger.warning(
                    "WFO FAILED for %s (%s) â€” best OOS IR=%.3f (threshold=%.1f)",
                    asset_class, rep_ticker,
                    result.get("best_ir", 0.0), validator.ir_threshold,
                )
        except Exception as exc:
            logger.error("WFO run failed for %s: %s", asset_class, exc)

    def _train_regime(self, asset_class: str, df: pd.DataFrame) -> None:
        """Retrain HMM only if 7+ days have elapsed since last retrain (spec Â§EOD)."""
        retrain_days = self.cfg["data"].get("hmm_retrain_days", 7)
        last = self._last_hmm_retrain.get(asset_class)
        now = datetime.utcnow()
        if last is not None and (now - last).days < retrain_days:
            logger.debug(
                "HMM retrain skipped for %s â€” last retrain %d days ago (threshold=%d)",
                asset_class, (now - last).days, retrain_days,
            )
            return
        try:
            det = self._regime_cache.get(
                asset_class,
                RegimeDetector(n_states=self.cfg.get("regime", {}).get("n_states", 5)),
            )
            det.train(
                df.tail(self.cfg["data"]["hmm_train_window"]),
                asset_class=asset_class,
            )
            self._regime_cache[asset_class] = det
            self._last_hmm_retrain[asset_class] = now
            logger.info("HMM retrained for %s", asset_class)
        except Exception as exc:
            logger.error("HMM training failed for %s: %s", asset_class, exc)

