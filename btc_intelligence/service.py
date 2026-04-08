from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time as time_module
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from btc_intelligence.api.websocket import LiveWebSocketHub
from btc_intelligence.backtesting.paper_trade import PaperTradeBook
from btc_intelligence.config import settings
from btc_intelligence.features.feature_vector import build_feature_state
from btc_intelligence.ingestion.binance_rest import BinanceRestPoller
from btc_intelligence.ingestion.binance_ws import BinanceWebSocketManager
from btc_intelligence.ingestion.coinglass import CoinglassPoller
from btc_intelligence.ingestion.cryptopanic import CryptoPanicPoller
from btc_intelligence.ingestion.data_buffer import MarketDataBuffer
from btc_intelligence.ingestion.deribit import DeribitPoller
from btc_intelligence.ingestion.glassnode import GlassnodePoller
from btc_intelligence.ingestion.macro_data import MacroDataPoller
from btc_intelligence.ingestion.multi_exchange_ws import MultiExchangeWebSocketManager
from btc_intelligence.ingestion.whale_tracker import WhaleTrackerPoller
from btc_intelligence.models.inference import ModelInference
from btc_intelligence.models.retrainer import ModelRetrainer
from btc_intelligence.monitoring.auto_correct import AutoCorrector
from btc_intelligence.monitoring.auto_pause import AutoPauseManager
from btc_intelligence.monitoring.performance_tracker import PerformanceTracker
from btc_intelligence.regime.classifier import classify_regime
from btc_intelligence.signals.engine import SignalEngine
from btc_intelligence.signals.execution import evaluate_execution, recommended_max_position_btc
from btc_intelligence.signals.execution_adverse_selection import compute_adverse_selection
from btc_intelligence.signals.intelligence import (
    combined_btc_signal,
    execution_rejection_code,
    order_flow_decision_state,
    trade_based_volume_profile,
    volatility_tradeability,
)
from btc_intelligence.signals.probability_stacker import ProbabilityStacker
from btc_intelligence.services.ic_monitor import apply_hibernation_mask, get_ic_monitor
from btc_intelligence.services.calibration_meta_job import run_elasticnet_calibration_fit
from btc_intelligence.services.shap_cluster_job import run_shap_cluster_snapshot
from btc_intelligence.services import (
    AggregatorConfig,
    AdaptiveLearningConfig,
    AdaptiveLearningEngine,
    CycleMonitor,
    CycleMonitorConfig,
    DataDriftEngine,
    DecisionEngine,
    DecisionEngineInput,
    DrawdownConfig,
    DrawdownController,
    ExecutionPlanInput,
    ExecutionPlanner,
    KellyConfig,
    KellyPositionSizer,
    MetaDecisionEngine,
    MetaDecisionInput,
    MetaLabelingConfig,
    MetaLabelingEngine,
    OrderFlowService,
    ProbabilityInput,
    ProbabilityService,
    SignalAggregator,
    StrategyEngine,
    StrategyEngineConfig,
    TradeThrottle,
    TradeThrottleConfig,
    ValidationConfig,
    ValidationEngine,
    WalkForwardConfig,
    WalkForwardValidator,
)
from btc_intelligence.state import RedisStateStore
from btc_intelligence.utils.notifier import TelegramNotifier


logger = logging.getLogger(__name__)
ABSOLUTE_MIN_CONFIDENCE = 25.0
MAX_OPEN_SIGNALS = 1
MIN_HOLD_SECONDS = 300
MIN_PRICE_MOVE_PCT = 0.05
MAX_HOLD_SECONDS = 14400
MTF_ALIGNMENT_REQUIRED = True


class AppRuntime:
    def __init__(self) -> None:
        self.redis_state = RedisStateStore(
            redis_url=settings.redis_url,
            key_prefix=settings.redis_key_prefix,
        )
        candles_max = {
            '1m': settings.candles_1m_max,
            '5m': settings.candles_5m_max,
            '15m': settings.candles_15m_max,
            '1h': settings.candles_1h_max,
            '4h': settings.candles_4h_max,
        }
        self.buffer = MarketDataBuffer(
            candles_max,
            settings.trades_max,
            settings.signal_history_size,
            redis_store=(self.redis_state if settings.redis_state_enabled else None),
        )

        self.ws_manager = BinanceWebSocketManager(self.buffer)
        self.rest_poller = BinanceRestPoller(self.buffer)
        self.multi_exchange = MultiExchangeWebSocketManager(self.buffer)
        self.coinglass = CoinglassPoller(self.buffer)
        self.glassnode = GlassnodePoller(self.buffer)
        self.deribit = DeribitPoller(self.buffer)
        self.whale_tracker = WhaleTrackerPoller(self.buffer)
        self.macro_data = MacroDataPoller(self.buffer)
        self.cryptopanic = CryptoPanicPoller(self.buffer)

        self.model = ModelInference()
        self.stacker = ProbabilityStacker(settings.edge_store_path)
        self.engine = SignalEngine(self.model, self.stacker)
        self.order_flow_service = OrderFlowService()
        self.decision_engine = DecisionEngine()
        self.probability_service = ProbabilityService()
        self.execution_planner = ExecutionPlanner()
        self.validation_engine = ValidationEngine(ValidationConfig())
        self.meta_labeling_engine = MetaLabelingEngine(MetaLabelingConfig())
        try:
            self.walk_forward_validator = WalkForwardValidator(WalkForwardConfig())
        except Exception:
            self.walk_forward_validator = WalkForwardValidator()
        self.strategy_engine = StrategyEngine(
            state_path="btc_intelligence/logs/strategy_engine_state.json",
            config=StrategyEngineConfig(
                min_trades_required=30,
                update_frequency=20,
                learning_rate=0.01,
            ),
            walk_forward_validator=self.walk_forward_validator,
        )
        self.data_drift_engine = DataDriftEngine(
            baseline_window=240,
            recent_window=60,
            state_path="btc_intelligence/logs/data_drift_baseline.json",
        )
        try:
            self.signal_aggregator = SignalAggregator(AggregatorConfig())
        except Exception:
            self.signal_aggregator = SignalAggregator()
        try:
            self.kelly_position_sizer = KellyPositionSizer(KellyConfig())
        except Exception:
            self.kelly_position_sizer = KellyPositionSizer()
        try:
            self.drawdown_controller = DrawdownController(
                state_path="btc_intelligence/logs/drawdown_state.json",
                config=DrawdownConfig(),
            )
        except Exception:
            self.drawdown_controller = DrawdownController(
                state_path="btc_intelligence/logs/drawdown_state.json",
            )
        try:
            self.cycle_monitor = CycleMonitor(
                CycleMonitorConfig(log_dir="btc_intelligence/logs/monitor")
            )
        except Exception:
            self.cycle_monitor = None
        self.meta_decision_engine = MetaDecisionEngine()
        self.adaptive_learning = AdaptiveLearningEngine(
            state_path="btc_intelligence/logs/adaptive_learning_state.json",
            config=AdaptiveLearningConfig(
                learning_rate=0.01,
                min_trades_required=30,
                update_frequency=20,
            ),
        )
        self.paper = PaperTradeBook()
        self.trade_throttle = TradeThrottle(TradeThrottleConfig())
        self.hub = LiveWebSocketHub()
        self.notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)

        self.performance_tracker = PerformanceTracker(settings.monitoring_log_path)
        self.auto_pause_manager = AutoPauseManager()
        self.auto_corrector = AutoCorrector()
        self.retrainer = ModelRetrainer(self.model)

        self._signal_task: asyncio.Task | None = None
        self._shap_cluster_task: asyncio.Task | None = None
        self._running = False
        self.feature_seq: deque[np.ndarray] = deque(maxlen=200)
        self._open_trade_context: dict[str, Any] = {}
        self._shared_signals_db = Path('data/signals.db')
        self._shared_signals_db.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_seed_count: int = 0
        self._last_saved_btc_signal = 'HOLD'
        self._last_open_signal_id: str | None = None
        self._open_signal_state: dict[str, Any] | None = None
        self._last_open_pnl_update_ts: float = 0.0
        self._last_btc_signal_time: float = 0.0
        self._last_btc_signal_direction: str | None = None
        self._latest_intelligence_bundle: dict[str, Any] = {}
        self._brier_observation_mode: bool = False

        self.signal_log_path = Path(settings.signal_log_path)
        self.signal_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_shared_db_schema()
        self._last_btc_signal_time = self._get_last_signal_time_from_db()
        self._last_btc_signal_direction = self._get_last_signal_direction_from_db()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if settings.redis_state_enabled:
            await self.redis_state.connect()
            await self._maybe_warm_start_redis_signal_from_sqlite()

        await self._seed_calibration_from_sqlite()

        await self.ws_manager.start()
        await self.rest_poller.start()
        await self.multi_exchange.start()
        await self.coinglass.start()
        await self.glassnode.start()
        await self.deribit.start()
        await self.whale_tracker.start()
        await self.macro_data.start()
        await self.cryptopanic.start()

        self._signal_task = asyncio.create_task(self._signal_loop(), name='signal_loop')
        self._shap_cluster_task = asyncio.create_task(self._shap_cluster_background_loop(), name='shap_clusters')
        logger.info('Runtime started')

    async def stop(self) -> None:
        self._running = False
        if self._signal_task:
            self._signal_task.cancel()
            await asyncio.gather(self._signal_task, return_exceptions=True)
        if self._shap_cluster_task:
            self._shap_cluster_task.cancel()
            await asyncio.gather(self._shap_cluster_task, return_exceptions=True)

        await self.ws_manager.stop()
        await self.rest_poller.stop()
        await self.multi_exchange.stop()
        await self.coinglass.stop()
        await self.glassnode.stop()
        await self.deribit.stop()
        await self.whale_tracker.stop()
        await self.macro_data.stop()
        await self.cryptopanic.stop()
        await self.redis_state.close()

        self._persist_shutdown_checkpoints()

        logger.info('Runtime stopped')

    def _persist_shutdown_checkpoints(self) -> None:
        data_root = Path(settings.data_path)
        data_root.mkdir(parents=True, exist_ok=True)

        pos = self.paper.open_position
        if pos is not None:
            opened_at: str | None = None
            tid = self.paper._open_trade_id
            if tid and tid in self.paper._open:
                oa = self.paper._open[tid].get("opened_at_utc")
                if isinstance(oa, datetime):
                    opened_at = oa.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                elif oa is not None:
                    opened_at = str(oa)
            checkpoint = {
                "direction": pos.direction,
                "entry": float(pos.entry),
                "stop": float(pos.stop),
                "tp1": float(pos.tp1),
                "tp2": float(pos.tp2),
                "tp3": float(pos.tp3),
                "size_btc": float(pos.size_btc),
                "opened_at": opened_at,
            }
            try:
                out = data_root / "open_position_checkpoint.json"
                out.write_text(json.dumps(checkpoint, ensure_ascii=True, indent=2), encoding="utf-8")
                logger.info("shutdown: saved open position checkpoint")
            except Exception as exc:
                logger.debug("open position checkpoint write failed: %s", exc)

        self.probability_service.flush_buffer_to_disk(data_root / "platt_buffer_checkpoint.json")

    async def _shap_cluster_background_loop(self) -> None:
        await asyncio.sleep(30)
        while self._running:
            try:
                await asyncio.to_thread(run_shap_cluster_snapshot, self.model)
                await asyncio.to_thread(run_elasticnet_calibration_fit, self.adaptive_learning)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug('SHAP cluster background task: %s', exc)
            try:
                await asyncio.sleep(max(60, int(settings.shap_cluster_interval_sec)))
            except asyncio.CancelledError:
                break

    async def _signal_loop(self) -> None:
        while self._running:
            try:
                snapshot = await self.buffer.snapshot()
                monitoring = self._monitoring_dict()
                pause = self.auto_pause_manager.evaluate(self.performance_tracker.stats(portfolio_heat_pct=monitoring['portfolio_heat_pct']))
                brier_pause = False
                brier_reason = ""
                try:
                    wb = self.adaptive_learning.worst_regime_brier()
                    monitoring['calibration_brier_max'] = round(float(wb), 6)
                    thr = float(settings.brier_watchdog_threshold)
                    if wb > thr:
                        brier_pause = True
                        brier_reason = f"brier_watchdog:{wb:.4f}>{thr}"
                        self._brier_observation_mode = True
                    else:
                        self._brier_observation_mode = False
                except Exception as exc:
                    logger.debug("Brier watchdog skipped: %s", exc)
                    self._brier_observation_mode = False

                monitoring['auto_pause'] = bool(pause.paused or brier_pause)
                reasons = [pause.reason] if pause.paused else []
                if brier_pause:
                    reasons.append(brier_reason)
                    monitoring['brier_observation_mode'] = True
                else:
                    monitoring['brier_observation_mode'] = False
                monitoring['auto_pause_reason'] = "; ".join([r for r in reasons if r]) or (pause.reason if pause.paused else "")
                now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
                try:
                    current_equity = float(self.performance_tracker.equity_curve[-1]) if self.performance_tracker.equity_curve else 0.0
                    drawdown_status = self.drawdown_controller.update(current_equity=current_equity, timestamp_utc=now_iso)
                except Exception as exc:
                    logger.debug('Drawdown controller update fallback: %s', exc)
                    fallback_dd = max(float(monitoring.get('current_drawdown_pct', 0.0)) / 100.0, 0.0)
                    fallback_action = "HALT" if fallback_dd >= 0.08 else "REDUCE_SIZE" if fallback_dd >= 0.04 else "NORMAL"
                    drawdown_status = {
                        "current_drawdown_pct": fallback_dd,
                        "daily_loss_pct": 0.0,
                        "halted": bool(fallback_action == "HALT"),
                        "halt_reason": "fallback_monitoring_drawdown" if fallback_action == "HALT" else None,
                        "action": fallback_action,
                        "size_multiplier": 0.0 if fallback_action == "HALT" else 0.5 if fallback_action == "REDUCE_SIZE" else 1.0,
                        "trades_since_halt": 0,
                        "equity_peak": float(current_equity if 'current_equity' in locals() else 0.0),
                    }
                monitoring['drawdown_status'] = drawdown_status
                monitoring['drawdown_action'] = str(drawdown_status.get('action', 'NORMAL')).upper()
                monitoring['drawdown_size_multiplier'] = float(drawdown_status.get('size_multiplier', 1.0))
                monitoring['current_equity'] = float(current_equity if 'current_equity' in locals() else 0.0)
                monitoring['current_drawdown_ratio'] = float(drawdown_status.get('current_drawdown_pct', 0.0))
                await self.buffer.set_monitoring(monitoring)
                await self.buffer.set_edge_stats(self.stacker.current_edges())
                snapshot['monitoring_stats'] = monitoring
                current_price = self._extract_current_price(snapshot)
                try:
                    forced_closed = self.paper.check_forced_exits(
                        current_price=current_price,
                        now_utc=datetime.now(timezone.utc),
                    )
                    if forced_closed:
                        logger.info('Forced exits closed trades: %s', ', '.join(forced_closed))
                        await asyncio.to_thread(self._refresh_btc_open_signals, current_price)
                except Exception as exc:
                    logger.debug('Forced exit check fallback: %s', exc)

                if str(drawdown_status.get('action', 'NORMAL')).upper() == 'HALT':
                    logger.info('Drawdown circuit breaker active; skipping signal generation cycle')
                    await asyncio.to_thread(self._refresh_btc_open_signals, current_price)
                    await asyncio.sleep(settings.feature_eval_interval_sec)
                    continue

                if not self._has_required_candles(snapshot):
                    await asyncio.sleep(settings.feature_eval_interval_sec)
                    continue

                last_close = int(snapshot.get('last_15m_close_time', 0))
                if last_close <= int(snapshot.get('last_feature_eval_close_time', 0)):
                    with np.errstate(invalid='ignore', divide='ignore'):
                        self._mark_to_market(snapshot)
                        state = build_feature_state(snapshot)
                        regime = classify_regime(snapshot, state)
<<<<<<< HEAD
                        get_ic_monitor().maybe_record_daily(snapshot, state, regime.regime)
=======
>>>>>>> origin/main
                    await self._publish_intelligence_state(
                        snapshot,
                        state,
                        regime_name=regime.regime,
                        signal_payload=snapshot.get('latest_signal', {}),
                        regime_state_probs=regime.state_probs,
                    )
                    await asyncio.sleep(2)
                    continue

                with np.errstate(invalid='ignore', divide='ignore'):
                    state = build_feature_state(snapshot)
                    vol_payload = volatility_tradeability(state.volatility)
                    await self.buffer.set_volatility_tradeability(vol_payload)

                    regime = classify_regime(snapshot, state)
<<<<<<< HEAD
                    get_ic_monitor().maybe_record_daily(snapshot, state, regime.regime)
=======
>>>>>>> origin/main
                    payload = self.engine.build(
                        snapshot=snapshot,
                        state=state,
                        regime=regime,
                        sequence_vectors=list(self.feature_seq),
                        monitoring_stats=monitoring,
                        auto_pause=bool(monitoring['auto_pause']),
                    )

<<<<<<< HEAD
                    vec, fm_long = state.to_vector(direction='LONG', regime=regime.regime)
                    _, vec_masked = apply_hibernation_mask(fm_long, vec, get_ic_monitor().hibernated_factors)
                    self.feature_seq.append(vec_masked.squeeze(0))
=======
                    vec, _ = state.to_vector(direction='LONG', regime=regime.regime)
                    self.feature_seq.append(vec.squeeze(0))
>>>>>>> origin/main

                await self.buffer.mark_feature_eval(last_close)
                intelligence_payload = await self._publish_intelligence_state(
                    snapshot,
                    state,
                    regime_name=regime.regime,
                    signal_payload=payload,
                    regime_state_probs=regime.state_probs,
                )
                ws_payload = dict(payload)
                ws_payload["raw_confidence"] = float(payload.get("confidence", 0.0))
                if isinstance(intelligence_payload, dict):
                    meta_output = intelligence_payload.get("meta_output", {})
                    if not isinstance(meta_output, dict):
                        meta_output = {}
                    ws_payload["meta_confidence"] = float(meta_output.get("confidence", ws_payload["raw_confidence"]))
                    ws_payload["meta_decision"] = str(meta_output.get("decision", "HOLD")).upper()
                    ws_payload["validation_status"] = str(meta_output.get("validation_status", "REJECTED")).upper()
                else:
                    ws_payload["meta_confidence"] = float(ws_payload["raw_confidence"])
                    ws_payload["meta_decision"] = str(ws_payload.get("signal", "HOLD")).upper()
                    ws_payload["validation_status"] = "REJECTED"

                await self.buffer.set_latest_signal(ws_payload)
                await self._append_signal_log(ws_payload)
                await self.hub.broadcast(ws_payload)

                self._handle_paper_trade(ws_payload, snapshot)

                if ws_payload.get('signal') in {'LONG', 'SHORT'}:
                    await self.notifier.send(self._format_telegram(ws_payload))

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception('Signal loop error: %s', exc)
            await asyncio.sleep(settings.feature_eval_interval_sec)

    def _has_required_candles(self, snapshot: dict[str, Any]) -> bool:
        c = snapshot.get('candles', {})
        return (
            len(c.get('15m', [])) >= 120
            and len(c.get('1h', [])) >= 100
            and len(c.get('4h', [])) >= 50
            and len(snapshot.get('agg_trades', [])) >= 50
        )

    async def _append_signal_log(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        with self.signal_log_path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def _ensure_shared_db_schema(self) -> None:
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS signals (
                        ticker TEXT,
                        signal TEXT,
                        confidence REAL,
                        quality_score REAL,
                        outcome TEXT,
                        pnl_pct REAL,
                        mfe_pct REAL,
                        mae_pct REAL,
                        exit_price REAL,
                        duration_seconds INT,
                        sl REAL,
                        tp1 REAL,
                        tp2 REAL,
                        tp3 REAL,
                        rr_ratio REAL,
                        size_multiplier REAL,
                        signal_note TEXT,
                        entry_price REAL,
                        timestamp TEXT
                    )
                    """
                )
                for ddl in (
                    "ALTER TABLE signals ADD COLUMN ticker TEXT",
                    "ALTER TABLE signals ADD COLUMN confidence REAL",
                    "ALTER TABLE signals ADD COLUMN quality_score REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN outcome TEXT",
                    "ALTER TABLE signals ADD COLUMN pnl_pct REAL",
                    "ALTER TABLE signals ADD COLUMN result TEXT",
                    "ALTER TABLE signals ADD COLUMN size_multiplier REAL",
                    "ALTER TABLE signals ADD COLUMN signal_note TEXT",
                    "ALTER TABLE signals ADD COLUMN entry_price REAL",
                    "ALTER TABLE signals ADD COLUMN sl REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN tp1 REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN tp2 REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN tp3 REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN rr_ratio REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN mfe_pct REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN mae_pct REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN exit_price REAL DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN duration_seconds INT DEFAULT 0",
                    "ALTER TABLE signals ADD COLUMN raw_score REAL",
                ):
                    try:
                        cur.execute(ddl)
                    except Exception:
                        pass
                conn.commit()
        except Exception as exc:
            logger.warning('Shared signals DB schema init failed: %s', exc)

    def _read_signal_columns(self, cur: sqlite3.Cursor) -> set[str]:
        cur.execute("PRAGMA table_info(signals)")
        return {str(row[1]) for row in cur.fetchall()}

    def _get_last_signal_time_from_db(self) -> float:
        """Read last BTC signal timestamp from DB."""
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT timestamp
                    FROM signals
                    WHERE ticker = 'BTCUSDT'
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if not row or not row[0]:
                    return 0.0
                raw_ts = str(row[0]).strip()
                try:
                    if raw_ts.endswith("Z"):
                        dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromisoformat(raw_ts)
                except Exception:
                    try:
                        dt = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        return 0.0
                return float(dt.timestamp())
        except Exception:
            return 0.0

    def _get_last_signal_direction_from_db(self) -> str | None:
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT signal
                    FROM signals
                    WHERE ticker = 'BTCUSDT'
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if not row or row[0] is None:
                    return None
                side = str(row[0]).upper().strip()
                return side if side else None
        except Exception:
            return None

    @staticmethod
    def _compute_signal_quality(
        confidence: float,
        obi: float,
        cvd_slope: float,
        vol_regime: str,
        mtf_bias: str,
    ) -> float:
        score = 0.0
        score += min(float(confidence) * 0.4, 40.0)
        score += min(abs(float(obi)) * 20.0, 20.0)
        if float(cvd_slope) > 0:
            score += 20.0

        regime = str(vol_regime or "").upper()
        if regime == 'NORMAL':
            score += 10.0
        elif regime == 'EXPANSION':
            score += 5.0

        bias = str(mtf_bias or "").upper().strip()
        if bias and bias != 'UNKNOWN':
            score += 10.0

        return round(score, 1)

    @staticmethod
    def _extract_current_price(snapshot: dict[str, Any]) -> float:
        candles_1m = snapshot.get('candles', {}).get('1m', [])
        if candles_1m:
            try:
                return float(candles_1m[-1].get('close', 0.0))
            except Exception:
                pass
        try:
            return float(snapshot.get('binance_rest', {}).get('mark_price', 0.0))
        except Exception:
            return 0.0

    def _check_mtf_alignment(self, signal: str, mtf_bias: dict[str, Any]) -> tuple[bool, str]:
        """Require 4h and 1h trend to align with proposed trade direction."""
        if not MTF_ALIGNMENT_REQUIRED:
            return True, "mtf_check_disabled"

        signal_upper = str(signal or "").upper()
        bias_4h = str(mtf_bias.get("bias_4h", "NEUTRAL")).upper()
        bias_1h = str(mtf_bias.get("bias_1h", "NEUTRAL")).upper()

        if signal_upper == "LONG":
            if "BEARISH" in bias_4h:
                return False, f"counter_trend_long_blocked_4h_{bias_4h}"
            if "BEARISH" in bias_1h and "BEARISH" in bias_4h:
                return False, f"counter_trend_long_blocked_1h_{bias_1h}"
        elif signal_upper == "SHORT":
            if "BULLISH" in bias_4h:
                return False, f"counter_trend_short_blocked_4h_{bias_4h}"

        return True, "mtf_aligned"

    @staticmethod
    def _calc_signal_pnl_pct(signal: str, entry_price: float, current_price: float) -> float:
        if entry_price <= 0 or current_price <= 0:
            return 0.0
        if str(signal).upper() == 'SHORT':
            return ((entry_price - current_price) / entry_price) * 100.0
        return ((current_price - entry_price) / entry_price) * 100.0

    @staticmethod
    def _normalize_regime_label(regime_name: str) -> str:
        r = str(regime_name or "").strip().lower()
        if not r:
            return "SIDEWAYS"
        mapping = {
            "bullish_trend": "BULLISH",
            "bearish_trend": "BEARISH",
            "sideways_range": "SIDEWAYS",
            "breakout_up": "BULLISH_BREAKOUT",
            "breakout_down": "BEARISH_BREAKOUT",
            "panic_liquidation": "PANIC",
        }
        return mapping.get(r, r.upper())

    @staticmethod
    def _regime_bucket(regime_name: str) -> str:
        r = str(regime_name or "").upper()
        if "BEAR" in r or "PANIC" in r:
            return "bearish"
        if "BULL" in r:
            return "bullish"
        return "sideways"

    @staticmethod
    def _sanitize_orderflow_flow_score(orderflow_payload: dict[str, Any]) -> None:
        """
        DecisionEngine expects flow_score in ~[-1, 1]. Leaks of decision_breakdown-style
        values (~±100, e.g. -34.4) must be scaled back. Cap intermediate magnitude [-20, 20].
        """
        raw_in = float(orderflow_payload.get("flow_score", 0.0))
        x = raw_in
        if abs(x) > 1.0:
            x = x / 100.0
        x = float(np.clip(x, -20.0, 20.0))
        out = float(np.clip(x, -1.0, 1.0))
        if abs(raw_in - out) > 1e-6:
            logger.debug("flow_score raw=%.3f clipped=%.3f", raw_in, out)
        orderflow_payload["flow_score"] = out

    @staticmethod
    def _derive_momentum_score(state) -> float:
        tf = state.price_action.tf_15m
        price = max(float(tf.price), 1e-9)
        ema21_dist = (float(tf.price) - float(tf.ema21)) / price
        ema50_dist = (float(tf.price) - float(tf.ema50)) / price
        tf_alignment = (float(state.price_action.alignment_long) - float(state.price_action.alignment_short)) / 3.0
        raw = (0.45 * np.tanh(ema21_dist * 180.0)) + (0.35 * np.tanh(ema50_dist * 120.0)) + (0.20 * tf_alignment)
        return float(np.clip(raw, -1.0, 1.0))

    @staticmethod
    def _derive_cost_score(execution_payload: dict[str, Any]) -> float:
        spread = max(float(execution_payload.get("spread_pct", 0.0)), 0.0)
        slippage = max(float(execution_payload.get("slippage_pct", 0.0)), 0.0)
        spread_ratio = min(spread / max(settings.spread_reject_pct, 1e-6), 3.0)
        slippage_ratio = min(slippage / max(settings.slippage_reject_pct, 1e-6), 3.0)
        penalty = (0.55 * spread_ratio) + (0.45 * slippage_ratio)
        score = 1.0 - penalty
        if not bool(execution_payload.get("accepted", False)):
            score -= 0.25
        return float(np.clip(score, -1.0, 1.0))

    @staticmethod
    def _derive_liquidity_score(depth: dict[str, Any]) -> float:
        bids = depth.get("bids", []) if isinstance(depth, dict) else []
        asks = depth.get("asks", []) if isinstance(depth, dict) else []
        if not bids or not asks:
            return 0.0
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
            spread_pct = ((best_ask - best_bid) / mid * 100.0) if mid > 0 else 10.0
            bid_notional = sum(float(px) * float(qty) for px, qty in bids[:12])
            ask_notional = sum(float(px) * float(qty) for px, qty in asks[:12])
            depth_notional = bid_notional + ask_notional
            spread_score = max(0.0, 1.0 - (spread_pct / max(settings.spread_reject_pct, 1e-6)))
            depth_score = min(depth_notional / 5_000_000.0, 1.0)
            return float(np.clip((0.55 * spread_score) + (0.45 * depth_score), 0.0, 1.0))
        except Exception:
            return 0.0

    def _alpha_barrier_fracs(
        self,
        monitoring_stats: dict[str, Any],
        execution_payload: dict[str, Any],
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """
        E[Ret] and P(win) vs spread/slippage, all as fractions except P(win) in [0,1].
        """
        try:
            spread_frac = float(execution_payload.get("spread_pct", 0.0)) / 100.0
            slip_frac = float(execution_payload.get("slippage_pct", 0.0)) / 100.0
            win_prob = float(monitoring_stats.get("recent_win_rate", 0.0))
            win_prob = float(max(0.0, min(1.0, win_prob)))
            rows: list[Any] = []
            for block in self.strategy_engine.history.values():
                if isinstance(block, list):
                    rows.extend(block[-80:])
            pnls = [float(r.get("pnl_pct", 0.0)) for r in rows if isinstance(r, dict)]
            wins = [x for x in pnls if x > 0.0]
            if wins:
                e_ret_pct = float(sum(wins) / len(wins))
            elif pnls:
                e_ret_pct = float(sum(abs(x) for x in pnls) / max(len(pnls), 1)) * 0.30
            else:
                e_ret_pct = 0.35
            e_ret_frac = max(0.0, e_ret_pct / 100.0)
            if not pnls:
                win_prob = max(win_prob, 0.48)
            return e_ret_frac, win_prob, spread_frac, slip_frac
        except Exception as exc:
            logger.debug("alpha barrier fracs skipped: %s", exc)
            return None, None, None, None

    def _collect_meta_rf_features(
        self,
        state,
        regime_state_probs: dict[str, float] | None,
        execution_payload: dict[str, Any],
    ) -> dict[str, float]:
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60.0
        hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
        probs = regime_state_probs if isinstance(regime_state_probs, dict) else {}
        return {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "hmm_mean_reverting": float(probs.get("mean_reverting", 1.0 / 3.0)),
            "hmm_trending": float(probs.get("trending", 1.0 / 3.0)),
            "hmm_liquidity_cascade": float(probs.get("liquidity_cascade", 1.0 / 3.0)),
            "vix_level": float(state.macro.vix_level),
            "spread_pct": float(execution_payload.get("spread_pct", 0.0)),
        }

    def _collect_meta_rf_features_snapshot(self, snapshot: dict[str, Any], direction: str) -> dict[str, float]:
        state = build_feature_state(snapshot)
        bundle = self._latest_intelligence_bundle if isinstance(self._latest_intelligence_bundle, dict) else {}
        probs = bundle.get("regime_state_probs") if isinstance(bundle.get("regime_state_probs"), dict) else {}
        ex = self._build_execution_payload(snapshot, state, direction=str(direction).upper())
        return self._collect_meta_rf_features(state, probs, ex)

    @staticmethod
    def _btc_daily_closes_from_1h(snapshot: dict[str, Any]) -> list[float]:
        candles = snapshot.get("candles", {}).get("1h", [])
        if not candles or len(candles) < 24:
            return []
        by_day: dict[str, float] = {}
        for c in candles:
            ts = int(c.get("open_time", 0)) / 1000.0
            if ts <= 0:
                continue
            day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            by_day[day] = float(c["close"])
        days = sorted(by_day.keys())
        return [by_day[d] for d in days[-(int(settings.macro_corr_lookback_days) + 5) :]]

    @staticmethod
    def _rolling_btc_spx_correlation(snapshot: dict[str, Any]) -> float:
        macro = snapshot.get("macro", {}) if isinstance(snapshot.get("macro", {}), dict) else {}
        spx = macro.get("spx_daily_closes", [])
        if not isinstance(spx, list) or len(spx) < int(settings.macro_corr_min_samples):
            return float(macro.get("spx_btc_correlation", 0.0) or 0.0)
        btc = AppRuntime._btc_daily_closes_from_1h(snapshot)
        if len(btc) < int(settings.macro_corr_min_samples):
            return float(macro.get("spx_btc_correlation", 0.0) or 0.0)
        max_days = int(settings.macro_corr_lookback_days) + 1
        sb = np.asarray(spx[-max_days:], dtype=float)
        bb = np.asarray(btc[-max_days:], dtype=float)
        if sb.size < 2 or bb.size < 2:
            return float(macro.get("spx_btc_correlation", 0.0) or 0.0)
        m = int(min(sb.size, bb.size))
        sb = sb[-m:]
        bb = bb[-m:]
        if np.any(sb <= 0) or np.any(bb <= 0):
            return 0.0
        lr_s = np.diff(np.log(sb))
        lr_b = np.diff(np.log(bb))
        m2 = min(lr_s.size, lr_b.size)
        if m2 < int(settings.macro_corr_min_samples):
            return float(macro.get("spx_btc_correlation", 0.0) or 0.0)
        lr_s = lr_s[-m2:]
        lr_b = lr_b[-m2:]
        c = float(np.corrcoef(lr_s, lr_b)[0, 1])
        if np.isnan(c):
            return float(macro.get("spx_btc_correlation", 0.0) or 0.0)
        return c

    def _apply_macro_correlation_kelly_cap(
        self,
        kelly_sizing: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(kelly_sizing) if isinstance(kelly_sizing, dict) else {}
        reasons = list(out.get("size_reduction_reason", [])) if isinstance(out.get("size_reduction_reason"), list) else []
        corr = self._rolling_btc_spx_correlation(snapshot)
        out["btc_spx_correlation_rolling"] = round(corr, 6)
        out["macro_corr_gate_applied"] = False
        thr = float(settings.macro_corr_threshold)
        mult = float(settings.macro_corr_kelly_multiplier)
        if corr > thr and mult > 0.0 and mult < 1.0 and not bool(out.get("halted", False)):
            pct = float(out.get("position_pct", 0.0)) * mult
            usd = float(out.get("position_size_usd", 0.0)) * mult
            out["position_pct"] = pct
            out["position_size_pct"] = pct
            out["position_size_usd"] = usd
            out["macro_corr_gate_applied"] = True
            out["macro_corr_kelly_multiplier"] = mult
            reasons.append("macro_high_btc_spx_corr")
        out["size_reduction_reason"] = reasons
        return out

    def _compute_intelligence_bundle(
        self,
        snapshot: dict[str, Any],
        state,
        regime_name: str,
        signal_payload: dict[str, Any] | None = None,
        regime_state_probs: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        current_price = self._extract_current_price(snapshot)

        flow_metrics = self.order_flow_service.analyze(
            trades=list(snapshot.get("agg_trades", [])),
            depth=snapshot.get("depth", {}),
        ).to_dict()
        orderflow_payload = order_flow_decision_state(state.order_flow)
        orderflow_payload.update(
            {
                "cvd_trend": flow_metrics.get("cvd_trend", "FLAT"),
                "cvd_value": flow_metrics.get("cvd_value", 0.0),
                "cvd_slope": flow_metrics.get("cvd_slope", orderflow_payload.get("cvd_slope", 0.0)),
                "obi_imbalance": flow_metrics.get("obi_imbalance", 0.0),
                "obi": round(0.5 + (float(flow_metrics.get("obi_imbalance", 0.0)) / 2.0), 4),
                "aggression_buy_pct": flow_metrics.get("aggression_buy_pct", 50.0),
                "aggression_sell_pct": flow_metrics.get("aggression_sell_pct", 50.0),
                "absorption_levels": flow_metrics.get("absorption_levels", []),
                "flow_score": flow_metrics.get("flow_score", 0.0),
            }
        )
        self._sanitize_orderflow_flow_score(orderflow_payload)

        vol_payload = volatility_tradeability(state.volatility)
        vol_payload["regime"] = vol_payload.get("volatility_regime", "NORMAL")
        monitoring_stats = snapshot.get("monitoring_stats", {}) if isinstance(snapshot.get("monitoring_stats", {}), dict) else {}
        drawdown_status = monitoring_stats.get("drawdown_status", {}) if isinstance(monitoring_stats.get("drawdown_status", {}), dict) else {}
        volume_profile_payload = trade_based_volume_profile(
            agg_trades=snapshot.get('agg_trades', []),
            window_minutes=45,
            n_bins=24,
        )

        preferred_direction = "LONG"
        if str(orderflow_payload.get("decision_state", "")).upper() == "FAVOR_SHORT":
            preferred_direction = "SHORT"

        execution_payload = self._build_execution_payload(snapshot=snapshot, state=state, direction=preferred_direction)

        momentum_score = self._derive_momentum_score(state)
        cost_score = self._derive_cost_score(execution_payload)
        normalized_regime = self._normalize_regime_label(regime_name)
        er_b, pw_b, sf_b, sl_b = self._alpha_barrier_fracs(monitoring_stats, execution_payload)

        decision_output = self.decision_engine.evaluate(
            DecisionEngineInput(
                regime=normalized_regime,
                cvd_slope=float(orderflow_payload.get("cvd_slope", 0.0)),
                obi_imbalance=float(orderflow_payload.get("obi_imbalance", 0.0)),
                volatility_regime=str(vol_payload.get("volatility_regime", "NORMAL")),
                cost_score=cost_score,
                momentum_score=momentum_score,
                aggression_buy_pct=float(orderflow_payload.get("aggression_buy_pct", 50.0)),
                flow_score=float(orderflow_payload.get("flow_score", 0.0)),
                expected_return_frac=er_b,
                win_prob=pw_b,
                spread_frac=sf_b,
                slippage_frac=sl_b,
            )
        )
        decision_breakdown = dict(decision_output.get("decision_breakdown", {}))
        vol_for_factor = str(vol_payload.get("volatility_regime", "NORMAL")).upper()
        if vol_for_factor == "NORMAL":
            vol_factor_score = 0.45
        elif vol_for_factor == "EXPANSION":
            vol_factor_score = 0.10
        elif vol_for_factor in {"LOW", "COMPRESSION"}:
            vol_factor_score = -0.35
        elif vol_for_factor in {"HIGH_VOL", "PANIC"}:
            vol_factor_score = -0.45
        else:
            vol_factor_score = 0.0

        factor_values = {
            "regime": float(decision_breakdown.get("regime_score", 0.0)) / 100.0,
            "momentum": float(decision_breakdown.get("momentum_score", 0.0)) / 100.0,
            "flow": float(decision_breakdown.get("flow_score", 0.0)) / 100.0,
            "cost": float(decision_breakdown.get("cost_score", 0.0)) / 100.0,
            "volatility": vol_factor_score,
        }

        decision_payload = {
            **decision_output,
            "regime": normalized_regime,
            "momentum_score": round(momentum_score, 6),
            "cost_score": round(cost_score, 6),
            "as_of_utc": now_iso,
        }
        signal_aggregation_payload: dict[str, Any] = {
            "direction": str(decision_payload.get("decision", "HOLD")).upper(),
            "raw_score": 0.0,
            "confidence": float(decision_payload.get("confidence", 0.0)),
            "agreement_ratio": 0.0,
            "sources_used": 0,
            "sources_available": 0,
            "source_contributions": {},
            "rejected": True,
            "reject_reason": "fallback_decision_engine",
        }
        try:
            regime_upper = str(normalized_regime or "").upper()
            if "BULL" in regime_upper:
                regime_dir, regime_strength = "LONG", 0.70
            elif "BEAR" in regime_upper or "PANIC" in regime_upper:
                regime_dir, regime_strength = "SHORT", 0.70
            else:
                regime_dir, regime_strength = "NEUTRAL", 0.0

            flow_decision = str(orderflow_payload.get("decision_state", "NO_TRADE")).upper()
            flow_dir = "LONG" if flow_decision == "FAVOR_LONG" else "SHORT" if flow_decision == "FAVOR_SHORT" else "NEUTRAL"
            flow_strength = float(np.clip(abs(float(orderflow_payload.get("obi_imbalance", 0.0))), 0.0, 1.0))

            vp_decision = str(volume_profile_payload.get("decision_state", "NO_TRADE")).upper()
            vp_dir = "LONG" if vp_decision == "FAVOR_LONG" else "SHORT" if vp_decision == "FAVOR_SHORT" else "NEUTRAL"
            vp_strength = float(np.clip(abs(float(volume_profile_payload.get("price_to_poc_pct", 0.0))) / 0.5, 0.0, 1.0))

            momentum_dir = "LONG" if momentum_score > 0 else "SHORT" if momentum_score < 0 else "NEUTRAL"
            momentum_strength = float(np.clip(abs(momentum_score), 0.0, 1.0))

            vol_tradeability = str(vol_payload.get("tradeability", "ALLOW")).upper()
            vol_dir = preferred_direction if vol_tradeability in {"ALLOW", "CAUTION"} else "NEUTRAL"
            vol_strength = 0.5 if vol_tradeability == "ALLOW" else 0.35 if vol_tradeability == "CAUTION" else 0.0

            signal_aggregation_payload = self.signal_aggregator.aggregate(
                {
                    "order_flow": {"direction": flow_dir, "strength": flow_strength, "available": True},
                    "volume_profile": {"direction": vp_dir, "strength": vp_strength, "available": True},
                    "regime": {"direction": regime_dir, "strength": regime_strength, "available": True},
                    "momentum": {"direction": momentum_dir, "strength": momentum_strength, "available": True},
                    "volatility": {"direction": vol_dir, "strength": vol_strength, "available": True},
                }
            )
        except Exception as exc:
            logger.debug("Signal aggregation fallback to decision engine: %s", exc)

        probability_payload = self.probability_service.estimate(
            ProbabilityInput(
                momentum_score=momentum_score,
                flow_score=float(orderflow_payload.get("flow_score", 0.0)),
                volatility_regime=str(vol_payload.get("volatility_regime", "NORMAL")),
                regime=normalized_regime,
            )
        )
        dominant_state = str(
            probability_payload.get("dominant_state")
            or probability_payload.get("dominant")
            or "SIDEWAYS"
        ).upper()
        probability_payload["dominant_state"] = dominant_state
        probability_payload["dominant"] = dominant_state
        calibration_score = probability_payload.get("calibration_score")
        if calibration_score is None:
            platt_probability = probability_payload.get("platt_probability")
            if platt_probability is not None:
                calibration_score = float(platt_probability) * 100.0
            else:
                calibration_score = float(
                    max(
                        probability_payload.get("up_prob", 0.0),
                        probability_payload.get("down_prob", 0.0),
                        probability_payload.get("sideways_prob", 0.0),
                    )
                )
        probability_payload["calibration_score"] = round(float(calibration_score), 2)
        probability_payload["as_of_utc"] = now_iso

        base_decision = str(decision_payload.get("decision", "HOLD")).upper()
        _de_confidence = float(decision_payload.get("confidence", 0.0))
        base_confidence = _de_confidence
        if not bool(signal_aggregation_payload.get("rejected", True)):
            agg_dir = str(signal_aggregation_payload.get("direction", base_decision)).upper()
            if agg_dir in {"LONG", "SHORT", "HOLD"}:
                base_decision = agg_dir
            _agg_conf = float(signal_aggregation_payload.get("confidence", base_confidence))
            # Signal aggregation may give near-zero confidence when sources disagree.
            # Use decision engine as floor to avoid killing confidence on weak agreement.
            base_confidence = _agg_conf if _agg_conf >= _de_confidence * 0.7 else _de_confidence
        up_prob = float(probability_payload.get("up_prob", 0.0)) / 100.0
        down_prob = float(probability_payload.get("down_prob", 0.0)) / 100.0
        raw_prob = up_prob if base_decision == "LONG" else down_prob if base_decision == "SHORT" else max(up_prob, down_prob)

        adaptive_meta = self.adaptive_learning.meta_decision(
            regime=normalized_regime,
            volatility_state=str(vol_payload.get("volatility_regime", "NORMAL")),
            flow_state=str(orderflow_payload.get("cvd_trend", orderflow_payload.get("decision_state", "FLAT"))),
            base_decision=base_decision,
            base_confidence=base_confidence,
            factor_values=factor_values,
            raw_prob=raw_prob,
            blockers=list(decision_payload.get("blockers", [])),
        )

        _cal_block = adaptive_meta.get("calibration") if isinstance(adaptive_meta.get("calibration"), dict) else {}
        _cal_val = _cal_block.get("calibrated_prob")
        if _cal_val is None:
            calibrated_prob = float(raw_prob)
        else:
            calibrated_prob = float(_cal_val)
        if calibrated_prob == 0.0:
            pp = probability_payload.get("platt_probability")
            if pp is not None:
                calibrated_prob = float(pp)
            else:
                calibrated_prob = float(raw_prob)
        strategy_payload = self.strategy_engine.select_strategy(
            regime=normalized_regime,
            volatility_regime=str(vol_payload.get("volatility_regime", "NORMAL")),
            momentum_score=momentum_score,
            flow_score=float(orderflow_payload.get("flow_score", 0.0)),
            cost_score=cost_score,
        )
        drift_payload = self.data_drift_engine.update_and_detect(
            {
                "momentum": momentum_score,
                "flow": float(orderflow_payload.get("flow_score", 0.0)),
                "cost": cost_score,
                "volatility": float(vol_payload.get("atr_pct", 0.0)),
                "probability": calibrated_prob,
            }
        )

        liquidity_score = self._derive_liquidity_score(snapshot.get("depth", {}))
        planned_direction = str(adaptive_meta.get("final_decision", base_decision)).upper()
        if planned_direction not in {"LONG", "SHORT"}:
            planned_direction = preferred_direction
        execution_plan_payload = self.execution_planner.plan(
            ExecutionPlanInput(
                current_price=float(current_price),
                atr_pct=float(vol_payload.get("atr_pct", 0.0)),
                volatility_regime=str(vol_payload.get("volatility_regime", "NORMAL")),
                liquidity_score=liquidity_score,
                direction=planned_direction,
            )
        )
        execution_plan_payload["as_of_utc"] = now_iso

        validation_meta = adaptive_meta.get("validation", {}) if isinstance(adaptive_meta.get("validation", {}), dict) else {}
        val_samples = validation_meta.get("samples", {}) if isinstance(validation_meta.get("samples", {}), dict) else {}
        baseline_metric = float(validation_meta.get("baseline", {}).get("metric", 0.0)) if isinstance(validation_meta.get("baseline", {}), dict) else 0.0
        candidate_metric = float(validation_meta.get("candidate", {}).get("metric", baseline_metric)) if isinstance(validation_meta.get("candidate", {}), dict) else baseline_metric
        expected_edge = abs(float(decision_breakdown.get("final_score", 0.0))) / 100.0
        er_val, _pw_m, sf_val, sl_val = self._alpha_barrier_fracs(monitoring_stats, execution_payload)
        validation_payload = self.validation_engine.validate_step(
            "meta_decision",
            {
                "sample_size": int(val_samples.get("train", 0)) + int(val_samples.get("test", 0)) or len(self.paper.closed) or self._sqlite_seed_count,
                "expected_edge": expected_edge,
                "robustness_gain": candidate_metric - baseline_metric,
                "drawdown_delta": 0.05 if bool(adaptive_meta.get("edge_decay", {}).get("decay_detected", False)) else -0.01,
                "overfit_risk": float(adaptive_meta.get("calibration", {}).get("calibration_error", 0.0)) * 1.4,
                "brier_score": float(adaptive_meta.get("calibration", {}).get("brier_score", 0.25)),
                "calibration_error": float(adaptive_meta.get("calibration", {}).get("calibration_error", 0.15)),
                "expected_return_frac": er_val,
                "win_prob": float(calibrated_prob),
                "spread_frac": sf_val,
                "slippage_frac": sl_val,
            },
        )
        meta_rf_features = self._collect_meta_rf_features(state, regime_state_probs, execution_payload)
        meta_label_payload = self.meta_labeling_engine.label_trade(
            base_decision=str(adaptive_meta.get("final_decision", base_decision)),
            confidence=float(adaptive_meta.get("adjusted_confidence", decision_payload.get("confidence", 0.0))),
            calibrated_prob=calibrated_prob,
            blockers=list(decision_payload.get("blockers", [])),
            calibration_error=float(adaptive_meta.get("calibration", {}).get("calibration_error", 0.0)),
            drift_level=str(drift_payload.get("drift_level", "LOW")),
            regime=normalized_regime,
            edge_decay=bool(adaptive_meta.get("edge_decay", {}).get("decay_detected", False)),
            meta_features=meta_rf_features,
            utc_hour=datetime.utcnow().hour,
            probability_service=self.probability_service,
        )
        drawdown_action = str(drawdown_status.get("action", monitoring_stats.get("drawdown_action", ""))).upper()
        if drawdown_action not in {"NORMAL", "REDUCE_SIZE", "HALT"}:
            current_dd_ratio = float(drawdown_status.get("current_drawdown_pct", monitoring_stats.get("current_drawdown_ratio", 0.0)))
            if current_dd_ratio <= 0.0:
                current_dd_ratio = max(float(monitoring_stats.get("current_drawdown_pct", 0.0)) / 100.0, 0.0)
            if current_dd_ratio >= 0.08:
                drawdown_action = "HALT"
            elif current_dd_ratio >= 0.04:
                drawdown_action = "REDUCE_SIZE"
            else:
                drawdown_action = "NORMAL"
        meta_output = self.meta_decision_engine.evaluate(
            MetaDecisionInput(
                base_decision=base_decision,
                base_confidence=base_confidence,
                calibrated_probability=calibrated_prob,
                strategy_used=str(strategy_payload.get("strategy_used", "trend_following")),
                blockers=list(decision_payload.get("blockers", [])),
                validation=validation_payload,
                meta_label=meta_label_payload,
                drift=drift_payload,
                edge_decay=adaptive_meta.get("edge_decay", {}),
                execution_plan=execution_plan_payload,
                adaptive_meta=adaptive_meta,
                drawdown_action=drawdown_action,
            )
        )
        kelly_sizing_payload: dict[str, Any] = {
            "kelly_full": 0.0,
            "kelly_fraction": 0.0,
            "position_pct": 0.0,
            "position_size_pct": 0.0,
            "position_size_usd": 0.0,
            "p": 0.0,
            "b": 0.0,
            "raw_kelly": 0.0,
            "size_reduction_reason": ["fallback"],
            "halted": bool(drawdown_action == "HALT"),
            "halt_reason": "drawdown_circuit_breaker" if drawdown_action == "HALT" else None,
            "risk_budgets": {},
        }
        try:
            all_strategy_rows = []
            for rows in self.strategy_engine.history.values():
                if isinstance(rows, list):
                    all_strategy_rows.extend(rows[-80:])
            pnl_vals = [float(r.get("pnl_pct", 0.0)) for r in all_strategy_rows if isinstance(r, dict)]
            wins = [x for x in pnl_vals if x > 0.0]
            losses = [abs(x) for x in pnl_vals if x < 0.0]
            win_rate = float(len(wins) / max(len(pnl_vals), 1)) if pnl_vals else float(monitoring_stats.get("recent_win_rate", 0.0))
            avg_win_pct = float((sum(wins) / len(wins)) / 100.0) if wins else 0.015
            avg_loss_pct = float((sum(losses) / len(losses)) / 100.0) if losses else 0.010

            drawdown_ratio = float(drawdown_status.get("current_drawdown_pct", monitoring_stats.get("current_drawdown_ratio", 0.0)))
            if drawdown_ratio <= 0.0:
                drawdown_ratio = max(float(monitoring_stats.get("current_drawdown_pct", 0.0)) / 100.0, 0.0)
            drawdown_size_multiplier = float(drawdown_status.get("size_multiplier", monitoring_stats.get("drawdown_size_multiplier", 1.0)))
            portfolio_value = float(monitoring_stats.get("current_equity", 0.0))
            if portfolio_value <= 0.0:
                portfolio_value = float(self.performance_tracker.equity_curve[-1]) if self.performance_tracker.equity_curve else 0.0

            ex_rr = float(execution_plan_payload.get("expected_rr", 0.0) or 0.0)
            kelly_sizing_payload = self.kelly_position_sizer.compute(
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                confidence=float(meta_output.get("confidence", base_confidence)),
                drift_level=str(drift_payload.get("drift_level", "LOW")),
                edge_decay=bool(adaptive_meta.get("edge_decay", {}).get("decay_detected", False)),
                current_drawdown_pct=drawdown_ratio,
                volatility_regime=str(vol_payload.get("volatility_regime", "NORMAL")),
                portfolio_value=portfolio_value,
                size_multiplier=drawdown_size_multiplier,
                execution_rr=ex_rr if ex_rr > 0 else None,
                depth=snapshot.get("depth", {}),
                strategy_returns_pct=pnl_vals,
                portfolio_heat_pct=float(monitoring_stats.get("portfolio_heat_pct", 0.0)),
            )
        except Exception as exc:
            logger.debug("Kelly sizing fallback applied: %s", exc)

        kelly_sizing_payload = self._apply_macro_correlation_kelly_cap(kelly_sizing_payload, snapshot)

        execution_plan_payload["kelly_sizing"] = kelly_sizing_payload
        execution_plan_payload["drawdown_size_multiplier"] = float(drawdown_status.get("size_multiplier", 1.0))
        execution_plan_payload = self.execution_planner.merge_tail_risk(execution_plan_payload, kelly_sizing_payload)

        decision_payload["base_decision"] = base_decision
        decision_payload["decision"] = str(meta_output.get("decision", adaptive_meta.get("final_decision", base_decision)))
        decision_payload["confidence"] = float(meta_output.get("confidence", adaptive_meta.get("adjusted_confidence", 0.0)))
        decision_payload["probability"] = float(meta_output.get("probability", calibrated_prob * 100.0))
        decision_payload["strategy_used"] = str(meta_output.get("strategy_used", strategy_payload.get("strategy_used", "trend_following")))
        decision_payload["risk_score"] = float(meta_output.get("risk_score", 0.0))
        decision_payload["validation_status"] = str(meta_output.get("validation_status", "REJECTED"))
        decision_payload["adaptive_adjustments"] = list(meta_output.get("adaptive_adjustments", []))
        decision_payload["adaptive_reasons"] = list(adaptive_meta.get("reasons", []))
        decision_payload["calibrated_probability"] = round(calibrated_prob * 100.0, 2)
        decision_payload["adaptive_score"] = float(adaptive_meta.get("adaptive_score", 0.0))
        decision_payload["edge_decay"] = adaptive_meta.get("edge_decay", {})

        verdict_payload = dict(decision_payload.get("trade_verdict", {}))
        verdict_payload["final_verdict"] = "TRADE" if decision_payload["decision"] in {"LONG", "SHORT"} else "AVOID"
        decision_payload["trade_verdict"] = verdict_payload

        if signal_payload is not None and isinstance(signal_payload, dict):
            net_alpha = float(signal_payload.get("net_alpha", signal_payload.get("net_alpha_score", 0.0)) or 0.0)
        else:
            net_alpha = 0.0

        aggregate_payload = {
            "decision_engine": decision_payload,
            "order_flow": orderflow_payload,
            "decision_breakdown": decision_payload.get("decision_breakdown", {}),
            "probability": probability_payload,
            "execution_gate": execution_payload,
            "execution_plan": execution_plan_payload,
            "trade_verdict": decision_payload.get("trade_verdict", {}),
            "factor_contributions": decision_payload.get("factor_contributions", []),
            "trade_triggers": decision_payload.get("trade_triggers", []),
            "meta_decision": adaptive_meta,
            "meta_labeling": meta_label_payload,
            "validation_engine": validation_payload,
            "strategy_selection": strategy_payload,
            "data_drift": drift_payload,
            "signal_aggregation": signal_aggregation_payload,
            "kelly_sizing": kelly_sizing_payload,
            "drawdown_status": drawdown_status,
            "meta_output": {
                "decision": meta_output.get("decision", "HOLD"),
                "confidence": meta_output.get("confidence", 0.0),
                "probability": meta_output.get("probability", 0.0),
                "strategy_used": meta_output.get("strategy_used", strategy_payload.get("strategy_used", "trend_following")),
                "execution_plan": execution_plan_payload,
                "risk_score": meta_output.get("risk_score", 100.0),
                "validation_status": meta_output.get("validation_status", "REJECTED"),
                "adaptive_adjustments": meta_output.get("adaptive_adjustments", []),
            },
            "adaptive_learning": {
                "regime_bucket": self._regime_bucket(normalized_regime),
                "weights": adaptive_meta.get("weights", {}),
                "validation": adaptive_meta.get("validation", {}),
                "calibration": adaptive_meta.get("calibration", {}),
                "edge_decay": adaptive_meta.get("edge_decay", {}),
            },
            "regime": normalized_regime,
            "regime_state_probs": dict(regime_state_probs or {}),
            "momentum_score": round(momentum_score, 6),
            "cost_score": round(cost_score, 6),
            "net_alpha": round(net_alpha, 6),
            "as_of_utc": now_iso,
        }
        aggregate_payload["monitoring_summary"] = {}
        try:
            if self.cycle_monitor is not None:
                walk_forward_payload = self.strategy_engine.last_wf_validation
                if not isinstance(walk_forward_payload, dict):
                    walk_forward_payload = {}
                cycle_record = self.cycle_monitor.record_cycle(
                    timestamp_utc=now_iso,
                    final_decision=str(meta_output.get("decision", "HOLD")),
                    confidence=float(meta_output.get("confidence", 0.0)),
                    validation=validation_payload if isinstance(validation_payload, dict) else {},
                    meta_label=meta_label_payload if isinstance(meta_label_payload, dict) else {},
                    drift=drift_payload if isinstance(drift_payload, dict) else {},
                    drawdown_status=drawdown_status if isinstance(drawdown_status, dict) else {},
                    signal_aggregation=signal_aggregation_payload if isinstance(signal_aggregation_payload, dict) else {},
                    walk_forward=walk_forward_payload,
                    kelly_sizing=kelly_sizing_payload if isinstance(kelly_sizing_payload, dict) else {},
                    strategy_used=str(meta_output.get("strategy_used", strategy_payload.get("strategy_used", "trend_following"))),
                    adaptive_adjustments=list(meta_output.get("adaptive_adjustments", [])),
                )
                aggregate_payload["monitoring_summary"] = self.cycle_monitor.get_summary()
                for alert in list(cycle_record.get("alerts", [])):
                    if str(alert.get("severity", "")).upper() == "CRITICAL":
                        logger.warning(
                            "Cycle monitor critical alert [%s]: %s",
                            str(alert.get("type", "unknown")),
                            str(alert.get("message", "")),
                        )
        except Exception:
            pass
        return aggregate_payload

    def _next_btc_signal_id(self, cur: sqlite3.Cursor) -> str:
        cur.execute("SELECT COUNT(*) FROM signals WHERE ticker = 'BTCUSDT'")
        count = int((cur.fetchone() or [0])[0] or 0)
        return f"BTC-{count + 1:03d}"

    def _insert_btc_signal_row(
        self,
        signal: str,
        confidence: float,
        quality_score: float,
        size_multiplier: float,
        signal_note: str,
        entry_price: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        rr_ratio: float,
    ) -> dict[str, Any] | None:
        """
        Insert BTC signal into shared data/signals.db.
        Uses minimal schema if present, otherwise terminal rich schema.
        """
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cols = self._read_signal_columns(cur)

                is_minimal_schema = (
                    {'ticker', 'signal', 'confidence', 'outcome', 'pnl_pct', 'timestamp'}.issubset(cols)
                    and 'signal_id' not in cols
                    and 'asset' not in cols
                )
                if is_minimal_schema:
                    cur.execute(
                        """
                        INSERT INTO signals (
                            ticker, signal, confidence, quality_score, outcome, pnl_pct,
                            entry_price, sl, tp1, tp2, tp3, rr_ratio,
                            size_multiplier, signal_note, timestamp
                        )
                        VALUES ('BTCUSDT', ?, ?, ?, 'OPEN', 0.0, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                        """,
                        (
                            str(signal).upper(),
                            float(confidence),
                            float(quality_score),
                            float(entry_price),
                            float(sl),
                            float(tp1),
                            float(tp2),
                            float(tp3),
                            float(rr_ratio),
                            float(size_multiplier),
                            str(signal_note),
                        ),
                    )
                    conn.commit()
                    row_id = cur.lastrowid
                    return {
                        "signal_ref": str(row_id) if row_id is not None else "",
                        "is_minimal_schema": True,
                    }

                signal_id = self._next_btc_signal_id(cur)
                now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
                cur.execute(
                    """
                    INSERT INTO signals (
                        signal_id, ticker, timestamp, asset, asset_class, timeframe,
                        signal, strength, confidence, quality_score, alpha_score, regime,
                        entry_price, stop_loss, take_profit, sl, tp1, tp2, tp3, rr_ratio, position_size_pct,
                        kelly_fraction, hurst_exponent, factor_scores, ic_weights,
                        slippage_cost_pct, cost_pct, net_alpha_score, cs_alpha_score,
                        execution_price, implementation_shortfall_pct,
                        outcome, result, pnl_pct, size_multiplier, signal_note
                    )
                    VALUES (?, 'BTCUSDT', ?, 'BTCUSDT', 'crypto', 'intraday',
                            ?, 'MEDIUM', ?, ?, 0.0, 'SIDEWAYS',
                            ?, ?, ?, ?, ?, ?, ?, ?, 0.0,
                            0.0, NULL, '{}', '{}',
                            0.0, 0.0, 0.0, 0.0,
                            ?, 0.0,
                            'OPEN', 'OPEN', 0.0, ?, ?)
                    """,
                    (
                        signal_id,
                        now_iso,
                        str(signal).upper(),
                        float(confidence),
                        float(quality_score),
                        float(entry_price),
                        float(sl),
                        float(tp1),
                        float(sl),
                        float(tp1),
                        float(tp2),
                        float(tp3),
                        float(rr_ratio),
                        float(entry_price),
                        float(size_multiplier),
                        str(signal_note),
                    ),
                )
                conn.commit()
                return {"signal_ref": signal_id, "is_minimal_schema": False}
        except Exception as exc:
            logger.error('Failed to insert BTC signal row: %s', exc, exc_info=True)
            return None

    def _close_btc_signal_row(self, signal_ref: str | None, is_minimal_schema: bool, pnl_pct: float, outcome: str) -> None:
        if not signal_ref:
            return
        try:
            out_val = str(outcome).upper()
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                if is_minimal_schema:
                    cur.execute(
                        "UPDATE signals SET outcome = ?, pnl_pct = ? WHERE rowid = ?",
                        (out_val, float(pnl_pct), int(signal_ref)),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE signals
                        SET outcome = ?, result = ?, pnl_pct = ?
                        WHERE signal_id = ?
                        """,
                        (out_val, out_val, float(pnl_pct), str(signal_ref)),
                    )
                conn.commit()
        except Exception as exc:
            logger.error('Failed to close BTC signal row: %s', exc, exc_info=True)

    def _update_open_btc_signal_pnl(self, signal_ref: str | None, is_minimal_schema: bool, pnl_pct: float) -> None:
        if not signal_ref:
            return
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                if is_minimal_schema:
                    cur.execute(
                        "UPDATE signals SET pnl_pct = ? WHERE rowid = ?",
                        (float(pnl_pct), int(signal_ref)),
                    )
                else:
                    cur.execute(
                        "UPDATE signals SET pnl_pct = ? WHERE signal_id = ?",
                        (float(pnl_pct), str(signal_ref)),
                    )
                conn.commit()
        except Exception as exc:
            logger.error('Failed to update BTC open signal pnl: %s', exc, exc_info=True)

    def _update_mfe_mae(
        self,
        signal_id: str,
        current_price: float,
        entry_price: float,
        signal_direction: str,
    ) -> None:
        """
        Track MFE/MAE for open BTC trades and keep live exit/duration fields updated.
        """
        if not signal_id or entry_price <= 0 or current_price <= 0:
            return

        side = str(signal_direction).upper()
        if side == "LONG":
            favorable = current_price - entry_price
            adverse = entry_price - current_price
        else:
            favorable = entry_price - current_price
            adverse = current_price - entry_price

        favorable_pct = (favorable / entry_price) * 100.0
        adverse_pct = (adverse / entry_price) * 100.0

        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cols = self._read_signal_columns(cur)
                is_minimal_schema = 'signal_id' not in cols and 'asset' not in cols
                if is_minimal_schema:
                    cur.execute(
                        """
                        UPDATE signals
                        SET
                            mfe_pct = MAX(COALESCE(mfe_pct, 0), ?),
                            mae_pct = MAX(COALESCE(mae_pct, 0), ?),
                            exit_price = ?,
                            duration_seconds = (strftime('%s','now') - strftime('%s', timestamp))
                        WHERE rowid = ? AND outcome = 'OPEN'
                        """,
                        (
                            float(favorable_pct),
                            float(adverse_pct),
                            float(current_price),
                            int(signal_id),
                        ),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE signals
                        SET
                            mfe_pct = MAX(COALESCE(mfe_pct, 0), ?),
                            mae_pct = MAX(COALESCE(mae_pct, 0), ?),
                            exit_price = ?,
                            duration_seconds = (strftime('%s','now') - strftime('%s', timestamp))
                        WHERE signal_id = ? AND outcome = 'OPEN'
                        """,
                        (
                            float(favorable_pct),
                            float(adverse_pct),
                            float(current_price),
                            str(signal_id),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.error('Failed to update BTC MFE/MAE: %s', exc, exc_info=True)

    def _count_open_btc_signals(self) -> int:
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM signals WHERE outcome = 'OPEN' AND ticker = 'BTCUSDT'"
                )
                row = cur.fetchone()
                return int((row or [0])[0] or 0)
        except Exception as exc:
            logger.error('Failed to count BTC open signals: %s', exc, exc_info=True)
            return 0

    def _close_all_open_btc_signals(self, current_price: float, outcome: str = 'CLOSED') -> int:
        if current_price <= 0:
            return 0
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cols = self._read_signal_columns(cur)
                signal_ref_expr = "signal_id" if 'signal_id' in cols else "CAST(rowid AS TEXT)"
                cur.execute(
                    f"""
                    SELECT rowid, COALESCE(signal, 'HOLD'), COALESCE(entry_price, 0.0), {signal_ref_expr}
                    FROM signals
                    WHERE ticker = 'BTCUSDT' AND outcome = 'OPEN'
                    """
                )
                rows = cur.fetchall()
                if not rows:
                    return 0

                out = str(outcome).upper()
                for rowid, signal, entry_price, signal_ref in rows:
                    pnl_pct = self._calc_signal_pnl_pct(str(signal), float(entry_price or 0.0), current_price)
                    set_parts = ["outcome = ?", "pnl_pct = ?"]
                    params: list[Any] = [out, float(pnl_pct)]
                    if 'result' in cols:
                        set_parts.append("result = ?")
                        params.append(out)
                    if 'exit_price' in cols:
                        set_parts.append("exit_price = ?")
                        params.append(float(current_price))
                    if 'duration_seconds' in cols:
                        set_parts.append(
                            "duration_seconds = CAST((strftime('%s','now') - strftime('%s', COALESCE(timestamp, datetime('now')))) AS INTEGER)"
                        )
                    params.append(int(rowid))
                    cur.execute(
                        f"UPDATE signals SET {', '.join(set_parts)} WHERE rowid = ?",
                        params,
                    )

                    state_ref = str(self._open_signal_state.get('signal_ref', '')) if self._open_signal_state else ''
                    if state_ref and state_ref in {str(rowid), str(signal_ref)}:
                        learn_state = dict(self._open_signal_state or {})
                        try:
                            self.adaptive_learning.record_trade(
                                regime=str(learn_state.get('regime', 'sideways')),
                                volatility_state=str(learn_state.get('volatility_state', 'NORMAL')),
                                flow_state=str(learn_state.get('flow_state', 'FLAT')),
                                direction=str(learn_state.get('signal', signal)),
                                pnl_pct=float(pnl_pct),
                                raw_prob=float(learn_state.get('raw_prob', 0.5)),
                                calibrated_prob=float(learn_state.get('calibrated_prob', learn_state.get('raw_prob', 0.5))),
                                factor_values=learn_state.get('factor_values', {}),
                                timestamp_utc=str(datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')),
                            )
                        except Exception as exc:
                            logger.debug('Adaptive learning trade record skipped: %s', exc)
                        try:
                            self.strategy_engine.record_trade(
                                strategy=str(learn_state.get('strategy_used', 'trend_following')),
                                pnl_pct=float(pnl_pct),
                                regime=str(learn_state.get('regime', 'SIDEWAYS')),
                                timestamp_utc=str(datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')),
                            )
                        except Exception as exc:
                            logger.debug('Strategy performance record skipped: %s', exc)
                        self._open_signal_state = None
                        self._last_open_signal_id = None
                        self._last_saved_btc_signal = 'HOLD'

                conn.commit()
                return len(rows)
        except Exception as exc:
            logger.error('Failed to close BTC open signals: %s', exc, exc_info=True)
            return 0

    @staticmethod
    def _parse_signal_timestamp(raw_ts: Any) -> datetime | None:
        if raw_ts is None:
            return None
        raw = str(raw_ts).strip()
        if not raw:
            return None
        try:
            if raw.endswith('Z'):
                dt = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(raw)
        except Exception:
            try:
                dt = datetime.strptime(raw, '%Y-%m-%d %H:%M:%S')
            except Exception:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _close_signal(
        self,
        signal_id: str,
        outcome: str,
        current_price: float,
        entry_price: float,
        direction: str,
        hold_secs: float,
    ) -> None:
        side = str(direction).upper()
        if side == 'LONG':
            pnl = ((float(current_price) - float(entry_price)) / float(entry_price)) * 100.0
        else:
            pnl = ((float(entry_price) - float(current_price)) / float(entry_price)) * 100.0

        out_val = str(outcome).upper()
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cols = self._read_signal_columns(cur)
                set_parts: list[str] = []
                params: list[Any] = []
                if 'outcome' in cols:
                    set_parts.append("outcome = ?")
                    params.append(out_val)
                if 'result' in cols:
                    set_parts.append("result = ?")
                    params.append(out_val)
                if 'pnl_pct' in cols:
                    set_parts.append("pnl_pct = ?")
                    params.append(round(float(pnl), 4))
                if 'exit_price' in cols:
                    set_parts.append("exit_price = ?")
                    params.append(float(current_price))
                if 'duration_seconds' in cols:
                    set_parts.append("duration_seconds = ?")
                    params.append(int(max(0.0, float(hold_secs))))

                if not set_parts:
                    return

                if 'signal_id' in cols:
                    cur.execute(
                        f"UPDATE signals SET {', '.join(set_parts)} WHERE signal_id = ?",
                        [*params, str(signal_id)],
                    )
                else:
                    cur.execute(
                        f"UPDATE signals SET {', '.join(set_parts)} WHERE rowid = ?",
                        [*params, int(float(signal_id))],
                    )
                conn.commit()

            if self._open_signal_state is not None:
                state_ref = str(self._open_signal_state.get('signal_ref', ''))
                if state_ref and state_ref == str(signal_id):
                    learn_state = dict(self._open_signal_state)
                    try:
                        self.adaptive_learning.record_trade(
                            regime=str(learn_state.get('regime', 'sideways')),
                            volatility_state=str(learn_state.get('volatility_state', 'NORMAL')),
                            flow_state=str(learn_state.get('flow_state', 'FLAT')),
                            direction=str(learn_state.get('signal', side)),
                            pnl_pct=float(pnl),
                            raw_prob=float(learn_state.get('raw_prob', 0.5)),
                            calibrated_prob=float(learn_state.get('calibrated_prob', learn_state.get('raw_prob', 0.5))),
                            factor_values=learn_state.get('factor_values', {}),
                            timestamp_utc=str(datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')),
                        )
                    except Exception as exc:
                        logger.debug('Adaptive learning trade record skipped: %s', exc)
                    try:
                        self.strategy_engine.record_trade(
                            strategy=str(learn_state.get('strategy_used', 'trend_following')),
                            pnl_pct=float(pnl),
                            regime=str(learn_state.get('regime', 'SIDEWAYS')),
                            timestamp_utc=str(datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')),
                        )
                    except Exception as exc:
                        logger.debug('Strategy performance record skipped: %s', exc)
                    self._open_signal_state = None
                    self._last_open_signal_id = None
                    self._last_saved_btc_signal = 'HOLD'

            logger.info(
                "Signal closed: %s outcome=%s pnl=%.3f%% duration=%.0fs",
                signal_id,
                out_val,
                pnl,
                float(hold_secs),
            )
        except Exception as exc:
            logger.error('Failed to close signal %s: %s', signal_id, exc, exc_info=True)

    def _refresh_btc_open_signals(self, current_price: float) -> None:
        if current_price <= 0:
            return
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                cur = conn.cursor()
                cols = self._read_signal_columns(cur)

                if 'outcome' in cols and 'result' in cols:
                    open_expr = "UPPER(COALESCE(outcome, result, '')) = 'OPEN'"
                elif 'outcome' in cols:
                    open_expr = "UPPER(COALESCE(outcome, '')) = 'OPEN'"
                elif 'result' in cols:
                    open_expr = "UPPER(COALESCE(result, '')) = 'OPEN'"
                else:
                    return

                if 'ticker' in cols:
                    ticker_expr = "ticker = 'BTCUSDT'"
                elif 'asset' in cols:
                    ticker_expr = "asset = 'BTCUSDT'"
                else:
                    ticker_expr = "1 = 1"

                signal_id_expr = "signal_id" if 'signal_id' in cols else "CAST(rowid AS TEXT)"
                entry_expr = "COALESCE(entry_price, 0.0)" if 'entry_price' in cols else "0.0"
                if 'sl' in cols and 'stop_loss' in cols:
                    sl_expr = "COALESCE(sl, stop_loss, 0.0)"
                elif 'sl' in cols:
                    sl_expr = "COALESCE(sl, 0.0)"
                elif 'stop_loss' in cols:
                    sl_expr = "COALESCE(stop_loss, 0.0)"
                else:
                    sl_expr = "0.0"
                if 'tp1' in cols and 'take_profit' in cols:
                    tp1_expr = "COALESCE(tp1, take_profit, 0.0)"
                elif 'tp1' in cols:
                    tp1_expr = "COALESCE(tp1, 0.0)"
                elif 'take_profit' in cols:
                    tp1_expr = "COALESCE(take_profit, 0.0)"
                else:
                    tp1_expr = "0.0"
                ts_expr = "COALESCE(timestamp, datetime('now'))"

                cur.execute(
                    f"""
                    SELECT
                        rowid,
                        {signal_id_expr} AS signal_ref,
                        COALESCE(signal, 'HOLD') AS signal,
                        {entry_expr} AS entry_price,
                        {sl_expr} AS sl,
                        {tp1_expr} AS tp1,
                        {ts_expr} AS opened_ts
                    FROM signals
                    WHERE {ticker_expr} AND {open_expr}
                    """
                )
                rows = cur.fetchall()
                if not rows:
                    return

                now_utc = datetime.now(timezone.utc)
                for rowid, signal_ref, signal, entry_price, sl, tp1, opened_ts in rows:
                    sid = str(signal_ref if signal_ref is not None else rowid)
                    side = str(signal or 'HOLD').upper()
                    if side not in {'LONG', 'SHORT'}:
                        continue
                    entry = float(entry_price or 0.0)
                    if entry <= 0:
                        continue

                    open_dt = self._parse_signal_timestamp(opened_ts)
                    if open_dt is None:
                        continue
                    hold_secs = max(0.0, (now_utc - open_dt).total_seconds())

                    self._update_mfe_mae(
                        signal_id=sid,
                        current_price=float(current_price),
                        entry_price=float(entry),
                        signal_direction=side,
                    )

                    if hold_secs > MAX_HOLD_SECONDS:
                        logger.info(
                            "%s: Max hold %ss exceeded, closing at market",
                            sid,
                            MAX_HOLD_SECONDS,
                        )
                        self._close_signal(sid, 'TIMEOUT', float(current_price), float(entry), side, hold_secs)
                        continue

                    if hold_secs < MIN_HOLD_SECONDS:
                        logger.debug(
                            "%s: hold=%.0fs < %ss minimum, skip SL/TP check",
                            sid,
                            hold_secs,
                            MIN_HOLD_SECONDS,
                        )
                        continue

                    move_pct = abs((float(current_price) - float(entry)) / float(entry)) * 100.0
                    if move_pct < MIN_PRICE_MOVE_PCT:
                        logger.debug(
                            "%s: move=%.3f%% < %.2f%% minimum, skip",
                            sid,
                            move_pct,
                            MIN_PRICE_MOVE_PCT,
                        )
                        continue

                    sl_val = float(sl or 0.0)
                    tp1_val = float(tp1 or 0.0)
                    outcome: str | None = None
                    if side == 'LONG':
                        if sl_val > 0 and current_price <= sl_val:
                            outcome = 'SL'
                        elif tp1_val > 0 and current_price >= tp1_val:
                            outcome = 'TP1'
                    else:
                        if sl_val > 0 and current_price >= sl_val:
                            outcome = 'SL'
                        elif tp1_val > 0 and current_price <= tp1_val:
                            outcome = 'TP1'

                    if outcome is not None:
                        self._close_signal(sid, outcome, float(current_price), float(entry), side, hold_secs)
                        continue

                    pnl = self._calc_signal_pnl_pct(side, float(entry), float(current_price))
                    set_parts: list[str] = []
                    params: list[Any] = []
                    if 'pnl_pct' in cols:
                        set_parts.append("pnl_pct = ?")
                        params.append(round(float(pnl), 4))
                    if 'exit_price' in cols:
                        set_parts.append("exit_price = ?")
                        params.append(float(current_price))
                    if 'duration_seconds' in cols:
                        set_parts.append("duration_seconds = ?")
                        params.append(int(hold_secs))
                    if 'mfe_pct' in cols:
                        set_parts.append("mfe_pct = MAX(COALESCE(mfe_pct, 0), ?)")
                        params.append(max(float(pnl), 0.0))
                    if 'mae_pct' in cols:
                        set_parts.append("mae_pct = MAX(COALESCE(mae_pct, 0), ?)")
                        params.append(max(-float(pnl), 0.0))

                    if set_parts:
                        cur.execute(
                            f"UPDATE signals SET {', '.join(set_parts)} WHERE rowid = ?",
                            [*params, int(rowid)],
                        )

                conn.commit()
        except Exception as exc:
            logger.error('Failed to refresh BTC open signals: %s', exc, exc_info=True)

    async def _read_mtf_bias(self) -> str:
        if not (settings.redis_state_enabled and self.redis_state.connected):
            return ""
        bias = ""
        payload = await self.redis_state.get_json('mtf_bias', default=None)
        if isinstance(payload, dict):
            bias = str(
                payload.get('bias')
                or payload.get('bias_4h')
                or payload.get('mtf_bias')
                or payload.get('direction')
                or ""
            )
        elif isinstance(payload, str):
            bias = payload

        if not bias and self.redis_state._client is not None:
            try:
                raw = await self.redis_state._client.get(self.redis_state._key('mtf_bias'))
                if raw:
                    raw_s = str(raw).strip()
                    if raw_s.startswith('{') and raw_s.endswith('}'):
                        try:
                            obj = json.loads(raw_s)
                            if isinstance(obj, dict):
                                bias = str(
                                    obj.get('bias')
                                    or obj.get('bias_4h')
                                    or obj.get('mtf_bias')
                                    or obj.get('direction')
                                    or ""
                                )
                        except Exception:
                            bias = ""
                    else:
                        bias = raw_s
            except Exception:
                bias = ""

        normalized = str(bias).replace('"', '').upper().strip()
        if normalized in {"BULLISH", "BEARISH"}:
            return normalized
        return ""

    async def _persist_btc_signal_from_redis(self, current_price: float) -> None:
        if not (settings.redis_state_enabled and self.redis_state.connected and self.redis_state._client is not None):
            return

        try:
            raw = await self.redis_state._client.get(self.redis_state._key('signal'))
        except Exception:
            raw = None
        if not raw:
            return

        try:
            redis_payload = json.loads(raw)
        except Exception:
            return
        if not isinstance(redis_payload, dict):
            return

        signal = str(redis_payload.get('signal', 'HOLD')).upper()
        confidence = float(redis_payload.get('confidence', 0.0))
        if confidence < ABSOLUTE_MIN_CONFIDENCE:
            logger.debug(
                'persist blocked: raw_conf=%.1f < %.1f',
                confidence,
                ABSOLUTE_MIN_CONFIDENCE,
            )
            return

        meta_output_payload = (
            redis_payload.get('meta_output', {})
            if isinstance(redis_payload.get('meta_output', {}), dict)
            else {}
        )
        meta_decision = str(
            redis_payload.get('meta_decision')
            or meta_output_payload.get('decision')
            or redis_payload.get('decision')
            or 'HOLD'
        ).upper()
        meta_conf = float(
            redis_payload.get('meta_confidence')
            or redis_payload.get('adjusted_confidence')
            or meta_output_payload.get('confidence')
            or confidence
        )
        _hour = datetime.utcnow().hour
        META_MIN_CONFIDENCE = 50.0 if 13 <= _hour <= 16 else 55.0
        if meta_decision not in {'LONG', 'SHORT'}:
            logger.warning(
                'Signal blocked by meta gate: decision=%s conf=%.1f',
                meta_decision,
                meta_conf,
            )
            logger.debug('persist blocked: meta_decision=%s', meta_decision)
            return
        if meta_conf < META_MIN_CONFIDENCE:
            logger.warning(
                'Signal blocked by meta gate: decision=%s conf=%.1f',
                meta_decision,
                meta_conf,
            )
            logger.debug(
                'persist blocked: meta_conf=%.1f < %.1f',
                meta_conf,
                META_MIN_CONFIDENCE,
            )
            return

        # Raw signal payload must still carry executable side.
        if signal not in {'LONG', 'SHORT'}:
            logger.debug('persist blocked: non-executable signal=%s', signal)
            return

        obi = float(redis_payload.get('obi', 0.0))
        cvd_slope = float(redis_payload.get('cvd_slope', 0.0))
        vol_regime = str(redis_payload.get('volatility_regime', redis_payload.get('regime', ''))).upper()
        mtf_bias = str(redis_payload.get('mtf_bias_4h', redis_payload.get('mtf_bias', ''))).upper()
        quality_score = self._compute_signal_quality(
            confidence=confidence,
            obi=obi,
            cvd_slope=cvd_slope,
            vol_regime=vol_regime,
            mtf_bias=mtf_bias,
        )
        if quality_score < 30.0:
            logger.debug('Low quality signal skipped: score=%.1f', quality_score)
            return

        # Gate 3/4: DB-backed last-signal timestamp and 300s minimum throttle
        db_last_time = await asyncio.to_thread(self._get_last_signal_time_from_db)
        if db_last_time > self._last_btc_signal_time:
            self._last_btc_signal_time = float(db_last_time)
        last_time = float(self._last_btc_signal_time)
        now_ts = float(time_module.time())
        elapsed = now_ts - last_time if last_time > 0 else float('inf')
        if elapsed < MIN_HOLD_SECONDS:
            logger.debug(
                'Throttled: only %.0fs since last signal. Need %ss.',
                elapsed,
                MIN_HOLD_SECONDS,
            )
            return

        logger.info(
            'Signal allowed: %s conf=%.1f%% q=%.1f elapsed=%.0fs',
            signal,
            confidence,
            quality_score,
            0.0 if elapsed == float('inf') else elapsed,
        )

        size_multiplier = float(redis_payload.get('size_multiplier', 1.0))
        signal_note = str(redis_payload.get('signal_note', ''))
        entry_price = float(redis_payload.get('entry_price', 0.0))
        if entry_price <= 0:
            entry_price = float(current_price)
        sl = float(redis_payload.get('sl', 0.0))
        tp1 = float(redis_payload.get('tp1', 0.0))
        tp2 = float(redis_payload.get('tp2', 0.0))
        tp3 = float(redis_payload.get('tp3', 0.0))
        rr_ratio = float(redis_payload.get('rr_ratio', 1.0))

        # Close existing open signals first (single-position policy).
        await asyncio.to_thread(
            self._close_all_open_btc_signals,
            float(entry_price if entry_price > 0 else current_price),
            'CLOSED',
        )
        self._open_signal_state = None
        self._last_open_signal_id = None

        insert_meta = await asyncio.to_thread(
            self._insert_btc_signal_row,
            signal,
            confidence,
            quality_score,
            size_multiplier,
            signal_note,
            entry_price,
            sl,
            tp1,
            tp2,
            tp3,
            rr_ratio,
        )
        if insert_meta:
            decision_block = redis_payload.get('decision_engine', {}) if isinstance(redis_payload.get('decision_engine', {}), dict) else {}
            prob_block = redis_payload.get('probability', {}) if isinstance(redis_payload.get('probability', {}), dict) else {}
            meta_block = redis_payload.get('meta_decision_details', {}) if isinstance(redis_payload.get('meta_decision_details', {}), dict) else {}
            meta_output = redis_payload.get('meta_output', {}) if isinstance(redis_payload.get('meta_output', {}), dict) else {}
            strategy_block = redis_payload.get('strategy_selection', {}) if isinstance(redis_payload.get('strategy_selection', {}), dict) else {}
            breakdown_block = decision_block.get('decision_breakdown', {}) if isinstance(decision_block.get('decision_breakdown', {}), dict) else {}
            raw_prob = float(prob_block.get('up_prob', 0.0)) / 100.0 if signal == 'LONG' else float(prob_block.get('down_prob', 0.0)) / 100.0
            factor_values = {
                "regime": float(breakdown_block.get("regime_score", 0.0)) / 100.0,
                "momentum": float(breakdown_block.get("momentum_score", 0.0)) / 100.0,
                "flow": float(breakdown_block.get("flow_score", 0.0)) / 100.0,
                "cost": float(breakdown_block.get("cost_score", 0.0)) / 100.0,
                "volatility": 0.0,
            }
            self._open_signal_state = {
                'signal_ref': str(insert_meta.get('signal_ref', '')),
                'is_minimal_schema': bool(insert_meta.get('is_minimal_schema', False)),
                'signal': signal,
                'entry_price': float(entry_price),
                'sl': float(sl),
                'tp1': float(tp1),
                'tp2': float(tp2),
                'opened_at_ts': float(time_module.time()),
                'opened_at_iso': datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
                'regime': str(decision_block.get('regime', redis_payload.get('volatility_regime', 'SIDEWAYS'))),
                'volatility_state': str(redis_payload.get('volatility_regime', 'NORMAL')),
                'flow_state': str(redis_payload.get('orderflow_decision', redis_payload.get('decision_state', 'FLAT'))),
                'raw_prob': float(max(0.0, min(1.0, raw_prob))),
                'calibrated_prob': float(max(0.0, min(1.0, meta_block.get('calibration', {}).get('calibrated_prob', raw_prob) if isinstance(meta_block.get('calibration', {}), dict) else raw_prob))),
                'factor_values': factor_values,
                'strategy_used': str(meta_output.get('strategy_used', strategy_block.get('strategy_used', 'trend_following'))),
            }
            self._last_open_signal_id = str(insert_meta.get('signal_ref', ''))
            self._last_saved_btc_signal = signal
            self._last_btc_signal_time = now_ts
            self._last_btc_signal_direction = signal
            logger.info('Saved: %s %s', str(insert_meta.get('signal_ref', '')), signal)
            if settings.redis_state_enabled and self.redis_state.connected:
                await self.redis_state.set_json('signal:persistent', redis_payload, ttl_seconds=3600)

    async def _seed_calibration_from_sqlite(self) -> None:
        await asyncio.to_thread(self._seed_calibration_from_sqlite_sync)

    def _seed_calibration_from_sqlite_sync(self) -> None:
        if len(self.probability_service._platt_raw_scores) > 30:
            self._sqlite_seed_count = len(self.probability_service._platt_raw_scores)
            return
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cols = self._read_signal_columns(cur)
                has_raw = "raw_score" in cols
                has_quality = "quality_score" in cols
                select_parts = ["outcome", "pnl_pct"]
                if has_raw:
                    select_parts.append("raw_score")
                if has_quality:
                    select_parts.append("quality_score")
                sel = ", ".join(select_parts)
                cur.execute(
                    f"""
                    SELECT {sel}
                    FROM signals
                    WHERE (ticker = 'BTCUSDT' OR ticker IS NULL OR ticker = '')
                      AND outcome IS NOT NULL
                      AND UPPER(TRIM(COALESCE(outcome, ''))) NOT IN ('OPEN', 'INTERRUPTED')
                    ORDER BY rowid DESC
                    LIMIT 200
                    """
                )
                rows = list(cur.fetchall())
        except Exception as exc:
            logger.warning("startup: calibration SQLite seed failed (continuing): %s", exc)
            return

        if not rows:
            logger.warning("startup: calibration seed skipped — no closed BTC rows in SQLite")
            return

        for row in reversed(rows):
            od = row["outcome"]
            if od is None or str(od).strip() == "":
                continue
            ou = str(od).strip().upper()
            if ou in {"OPEN", "INTERRUPTED"}:
                continue
            pnl_raw = row["pnl_pct"]
            try:
                pnl_pct = float(pnl_raw) if pnl_raw is not None else 0.0
            except (TypeError, ValueError):
                pnl_pct = 0.0
            label = 1 if pnl_pct > 0 else 0

            raw_score_val = 0.0
            try:
                if has_raw and row["raw_score"] is not None:
                    raw_score_val = float(row["raw_score"])
                elif has_quality and row["quality_score"] is not None:
                    qs = float(row["quality_score"])
                    raw_score_val = (qs / 50.0) - 1.0
                else:
                    raw_score_val = 0.0
            except (TypeError, ValueError, KeyError):
                raw_score_val = 0.0

            self.probability_service.record_labeled_trade(raw_score_val, label, defer_retrain=True)

        nbuf = len(self.probability_service._platt_raw_scores)
        self._sqlite_seed_count = nbuf
        if nbuf >= 30:
            trained = self.probability_service.platt.fit(
                self.probability_service._platt_raw_scores,
                self.probability_service._platt_labels,
                min_samples=30,
            )
            if trained:
                self.probability_service._new_labels_since_retrain = 0
                logger.info("startup: seeded calibration from SQLite n=%d trades", nbuf)
                try:
                    self.probability_service.flush_buffer_to_disk(
                        Path(settings.data_path) / "platt_buffer_checkpoint.json",
                    )
                except Exception:
                    pass
            else:
                logger.warning(
                    "startup: loaded %d calibration rows from SQLite but Platt fit failed",
                    nbuf,
                )
        else:
            logger.warning(
                "startup: calibration seed loaded n=%d rows (< 30); Platt still uncalibrated",
                nbuf,
            )

    async def _maybe_warm_start_redis_signal_from_sqlite(self) -> None:
        if not (settings.redis_state_enabled and self.redis_state.connected and self.redis_state._client is not None):
            return
        try:
            raw = await self.redis_state._client.get(self.redis_state._key('signal'))
        except Exception:
            raw = None
        if raw:
            return
        loaded = await asyncio.to_thread(self._fetch_latest_btc_signal_for_warm_start)
        if not loaded:
            return
        row_id, payload = loaded
        await self.redis_state.set_json('signal', payload, ttl_seconds=10)
        logger.info('startup: warm-started Redis from SQLite signal id=%d', row_id)

    def _signal_payload_from_db_row(self, d: dict[str, Any], *, freshen_as_of: bool = False) -> dict[str, Any]:
        ep = float(d.get('entry_price') or 0.0)
        sig = str(d.get('signal') or 'HOLD').upper()
        sl = float(d.get('sl') or d.get('stop_loss') or 0.0)
        tp1 = float(d.get('tp1') or 0.0)
        tp2 = float(d.get('tp2') or 0.0)
        tp3 = float(d.get('tp3') or 0.0)
        ts_raw = d.get('timestamp')
        persisted_ts: str | None = None
        as_of: str
        if ts_raw is not None and str(ts_raw).strip():
            try:
                s = str(ts_raw).strip().replace(' ', 'T')
                dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                persisted_ts = dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
            except Exception:
                persisted_ts = None
        if freshen_as_of:
            as_of = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        elif persisted_ts is not None:
            as_of = persisted_ts
        else:
            as_of = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        pad = 0.0005 if ep > 0 else 0.0
        entry_zone = [round(ep * (1.0 - pad), 2), round(ep * (1.0 + pad), 2)] if ep > 0 else [0.0, 0.0]
        conf = float(d.get('confidence') or 0.0)
        out: dict[str, Any] = {
            'signal': sig,
            'confidence': conf,
            'validated': True,
            'meta_decision': sig if sig in {'LONG', 'SHORT'} else 'HOLD',
            'meta_confidence': conf,
            'entry_price': ep,
            'entry_zone': entry_zone,
            'stop_loss': sl,
            'sl': sl,
            'take_profit': {'TP1': tp1, 'TP2': tp2, 'TP3': tp3},
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'rr_ratio': float(d.get('rr_ratio') or 0.0),
            'signal_note': str(d.get('signal_note') or ''),
            'size_multiplier': float(d.get('size_multiplier') or 1.0),
            'quality_score': float(d.get('quality_score') or 0.0),
            'as_of_utc': as_of,
            'outcome': str(d.get('outcome') or ''),
            'stale': False,
            'source': 'sqlite_warm_start',
        }
        if persisted_ts is not None:
            out['persisted_timestamp_utc'] = persisted_ts
        return out

    def _fetch_latest_btc_signal_for_warm_start(self) -> tuple[int, dict[str, Any]] | None:
        try:
            with sqlite3.connect(self._shared_signals_db) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT rowid AS _sqlite_rowid, *
                    FROM signals
                    WHERE ticker = 'BTCUSDT' OR ticker IS NULL OR ticker = ''
                    ORDER BY rowid DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row is None:
                    return None
                keys = row.keys()
                d = {str(k): row[k] for k in keys}
                rid = int(d.pop('_sqlite_rowid', 0) or 0)
                payload = self._signal_payload_from_db_row(d, freshen_as_of=True)
                return (rid, payload)
        except Exception as exc:
            logger.debug('warm-start SQLite fetch failed: %s', exc)
            return None

    async def _maybe_update_open_signal_pnl(self, current_price: float) -> None:
        if self._open_signal_state is None:
            return
        now_ts = float(time_module.time())
        if (now_ts - self._last_open_pnl_update_ts) < 300.0:
            return
        open_signal = str(self._open_signal_state.get('signal', 'HOLD')).upper()
        entry_price = float(self._open_signal_state.get('entry_price', 0.0))
        pnl_pct = self._calc_signal_pnl_pct(open_signal, entry_price, current_price)
        await asyncio.to_thread(
            self._update_open_btc_signal_pnl,
            str(self._open_signal_state.get('signal_ref', '')),
            bool(self._open_signal_state.get('is_minimal_schema', False)),
            pnl_pct,
        )
        self._last_open_pnl_update_ts = now_ts

    async def _publish_intelligence_state(
        self,
        snapshot: dict[str, Any],
        state,
        regime_name: str = "",
        signal_payload: dict[str, Any] | None = None,
        regime_state_probs: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        intelligence_bundle = self._compute_intelligence_bundle(
            snapshot=snapshot,
            state=state,
            regime_name=regime_name,
            signal_payload=signal_payload,
            regime_state_probs=regime_state_probs,
        )
        self._latest_intelligence_bundle = intelligence_bundle

        orderflow_payload = dict(intelligence_bundle.get("order_flow", {}))
        orderflow_payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

        vol_payload = volatility_tradeability(state.volatility)
        vol_payload.update(
            {
                "regime": vol_payload.get("volatility_regime", "NORMAL"),
                "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            }
        )

        volume_payload = trade_based_volume_profile(
            agg_trades=snapshot.get('agg_trades', []),
            window_minutes=45,
            n_bins=24,
        )
        volume_payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

        execution_payload = dict(intelligence_bundle.get("execution_gate", {}))

        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('orderflow', orderflow_payload, ttl_seconds=10)
            await self.redis_state.set_json('volatility', vol_payload, ttl_seconds=10)
            await self.redis_state.set_json('volprofile', volume_payload, ttl_seconds=10)
            await self.redis_state.set_json('execution', execution_payload, ttl_seconds=10)
            await self.redis_state.set_json('decision', intelligence_bundle.get("decision_engine", {}), ttl_seconds=10)
            await self.redis_state.set_json('probability', intelligence_bundle.get("probability", {}), ttl_seconds=10)
            await self.redis_state.set_json('execution_plan', intelligence_bundle.get("execution_plan", {}), ttl_seconds=10)
            await self.redis_state.set_json('intelligence', intelligence_bundle, ttl_seconds=10)

        current_price = self._extract_current_price(snapshot)
        mtf_bias = await self._read_mtf_bias()
        combined_payload = combined_btc_signal(
            orderflow_payload,
            vol_payload,
            current_price=current_price,
            mtf_bias=mtf_bias,
        )
        combined_payload["decision_engine"] = intelligence_bundle.get("decision_engine", {})
        combined_payload["trade_verdict"] = intelligence_bundle.get("trade_verdict", {})
        combined_payload["probability"] = intelligence_bundle.get("probability", {})
        combined_payload["execution_plan"] = intelligence_bundle.get("execution_plan", {})
        combined_payload["trade_triggers"] = intelligence_bundle.get("trade_triggers", [])
        combined_payload["meta_decision_details"] = intelligence_bundle.get("meta_decision", {})
        combined_payload["meta_labeling"] = intelligence_bundle.get("meta_labeling", {})
        combined_payload["validation_engine"] = intelligence_bundle.get("validation_engine", {})
        combined_payload["strategy_selection"] = intelligence_bundle.get("strategy_selection", {})
        combined_payload["data_drift"] = intelligence_bundle.get("data_drift", {})
        combined_payload["signal_aggregation"] = intelligence_bundle.get("signal_aggregation", {})
        combined_payload["kelly_sizing"] = intelligence_bundle.get("kelly_sizing", {})
        combined_payload["drawdown_status"] = intelligence_bundle.get("drawdown_status", {})
        combined_payload["meta_output"] = intelligence_bundle.get("meta_output", {})
        combined_payload["adaptive_learning"] = intelligence_bundle.get("adaptive_learning", {})
        combined_payload["orderflow"] = orderflow_payload
        combined_payload["flow_decision"] = str(
            orderflow_payload.get("decision_state", "NO_TRADE")
        ).upper()
        combined_payload["obi"] = float(
            orderflow_payload.get("obi", 0.5)
        )
        combined_payload["raw_confidence"] = float(combined_payload.get("confidence", 0.0))
        # --- UI fields missing from combined_btc_signal() ---
        _now_utc_h = datetime.now(timezone.utc).hour
        if _now_utc_h < 6:
            combined_payload["session"] = "asian"
        elif _now_utc_h < 9:
            combined_payload["session"] = "london_open"
        elif _now_utc_h < 13:
            combined_payload["session"] = "london"
        elif _now_utc_h < 17:
            combined_payload["session"] = "new_york"
        else:
            combined_payload["session"] = "after_hours"
        combined_payload["market_regime"] = str(
            intelligence_bundle.get("regime") or "SIDEWAYS"
        ).upper()
        _meta_final = str(intelligence_bundle.get("meta_output", {}).get("decision", "HOLD")).upper()
        combined_payload["algo"] = "SIGNAL" if _meta_final in {"LONG", "SHORT"} else "NO_TRADE"
        _sa = intelligence_bundle.get("signal_aggregation", {})
        _alpha = _sa.get("raw_score") or _sa.get("confidence")
        if _alpha is not None:
            try:
                _af = float(_alpha)
                combined_payload["alpha_score"] = abs(_af) * 100.0 if abs(_af) <= 1.0 else abs(_af)
            except (TypeError, ValueError):
                pass
        # ATR and mark_price for UI panels
        combined_payload["atr_value"] = float(vol_payload.get("atr_pct", 0.0))
        if not combined_payload.get("mark_price"):
            _snap_mp = float(snapshot.get("binance_rest", {}).get("mark_price", 0.0))
            combined_payload["mark_price"] = _snap_mp if _snap_mp > 100 else float(current_price)
        _oi = snapshot.get("binance_rest", {}).get("open_interest", 0.0)
        if _oi:
            combined_payload["open_interest_btc"] = float(_oi)
        try:
            kelly_payload = intelligence_bundle.get("kelly_sizing", {})
            if isinstance(kelly_payload, dict) and kelly_payload:
                position_pct = float(kelly_payload.get("position_pct", 0.0))
                position_usd = float(kelly_payload.get("position_size_usd", 0.0))
                combined_payload["position_size_pct"] = position_pct
                combined_payload["position_size_usd"] = position_usd
                max_pct = float(self.kelly_position_sizer.config.max_position_pct)
                if max_pct > 0:
                    combined_payload["size_multiplier"] = float(np.clip(position_pct / max_pct, 0.0, 1.0))
        except Exception as exc:
            logger.debug("Kelly payload publish fallback: %s", exc)

        meta_output_block = combined_payload.get("meta_output", {})
        if not isinstance(meta_output_block, dict):
            meta_output_block = {}
        meta_details = combined_payload.get("meta_decision_details", {})
        if not isinstance(meta_details, dict):
            meta_details = {}
        combined_payload["meta_decision"] = str(meta_output_block.get("decision", "HOLD")).upper()
        combined_payload["meta_confidence"] = float(meta_output_block.get("confidence", 0.0))
        combined_payload["validation_status"] = str(meta_output_block.get("validation_status", "REJECTED")).upper()

        meta_final = str(meta_output_block.get("decision", "")).upper()
        if not meta_final:
            meta_final = str(meta_details.get("final_decision", "")).upper()
        if meta_final == "HOLD":
            combined_payload["signal"] = "HOLD"
            combined_payload["confidence"] = min(float(combined_payload.get("confidence", 0.0)), 49.0)
            meta_reasons = meta_output_block.get("adaptive_adjustments", [])
            if not meta_reasons:
                meta_reasons = meta_details.get("reasons", [])
            if isinstance(meta_reasons, list) and meta_reasons:
                combined_payload["signal_note"] = f"meta_hold:{str(meta_reasons[0])[:80]}"
            else:
                combined_payload["signal_note"] = "meta_hold:stability_gate"

        combined_payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('signal', combined_payload, ttl_seconds=10)
        await self._persist_btc_signal_from_redis(current_price=current_price)
        await asyncio.to_thread(self._refresh_btc_open_signals, current_price)
        return combined_payload

    def _observation_paper_trade(self) -> bool:
        """Brier watchdog forces observation: behave like paper even if settings.paper_trade is False."""
        return bool(settings.paper_trade or self._brier_observation_mode)

    def _handle_paper_trade(self, payload: dict[str, Any], snapshot: dict[str, Any]) -> None:
        if not self._observation_paper_trade():
            return

        signal = str(payload.get('signal', 'HOLD'))
        if signal in {'LONG', 'SHORT'} and self.paper.open_position is None:
            now_utc = datetime.now(timezone.utc)
            mtf_context = (
                payload.get('mtf_bias')
                if isinstance(payload.get('mtf_bias'), dict)
                else {}
            )
            if not mtf_context and isinstance(self._latest_intelligence_bundle, dict):
                mtf_context = (
                    self._latest_intelligence_bundle.get('mtf_bias')
                    if isinstance(self._latest_intelligence_bundle.get('mtf_bias'), dict)
                    else {}
                )
            bias_4h = str(
                payload.get('mtf_bias_4h')
                or mtf_context.get('bias_4h')
                or self._latest_intelligence_bundle.get('mtf_bias_4h', 'NEUTRAL')
            ).upper()
            bias_1h = str(
                payload.get('mtf_bias_1h')
                or mtf_context.get('bias_1h')
                or self._latest_intelligence_bundle.get('mtf_bias_1h', 'NEUTRAL')
            ).upper()
            mtf_ok, mtf_reason = self._check_mtf_alignment(
                signal=signal,
                mtf_bias={"bias_4h": bias_4h, "bias_1h": bias_1h},
            )
            if not mtf_ok:
                logger.info(
                    'Paper trade blocked by MTF alignment gate: signal=%s bias_4h=%s bias_1h=%s reason=%s',
                    signal,
                    bias_4h,
                    bias_1h,
                    mtf_reason,
                )
            else:
                regime_label = str(
                    payload.get('regime')
                    or payload.get('market_regime')
                    or payload.get('regime_name')
                    or self._latest_intelligence_bundle.get('regime', 'SIDEWAYS')
                )
                can_open, throttle_reason = self.trade_throttle.can_trade(regime=regime_label, now=now_utc)
                if not can_open:
                    logger.info(
                        'Paper trade blocked by throttle: signal=%s regime=%s reason=%s',
                        signal,
                        regime_label,
                        throttle_reason,
                    )
                else:
                    entry_zone = payload.get('entry_zone', [0, 0])
                    entry = (float(entry_zone[0]) + float(entry_zone[1])) / 2.0
                    meta_confidence = float(payload.get('meta_confidence', 0.0))
                    if meta_confidence <= 0.0:
                        meta_output = (
                            self._latest_intelligence_bundle.get('meta_output', {})
                            if isinstance(self._latest_intelligence_bundle, dict)
                            else {}
                        )
                        if isinstance(meta_output, dict):
                            meta_confidence = float(meta_output.get('confidence', payload.get('confidence', 0.0)))
                        else:
                            meta_confidence = float(payload.get('confidence', 0.0))
                    opened = self.paper.open(
                        direction=signal,
                        entry=entry,
                        stop=float(payload.get('stop_loss', 0.0)),
                        tp1=float(payload.get('take_profit', {}).get('TP1', 0.0)),
                        tp2=float(payload.get('take_profit', {}).get('TP2', 0.0)),
                        tp3=float(payload.get('take_profit', {}).get('TP3', 0.0)),
                        size_btc=float(payload.get('position_size_btc', 0.0)),
                        confidence=meta_confidence,
                        regime=regime_label,
                    )
                    if opened:
                        self._open_trade_context = {
                            'factors': list(payload.get('factors_present', [])),
                            'confidence': float(meta_confidence),
                            'strategy': str(payload.get('strategy', 'NONE')),
<<<<<<< HEAD
                            'meta_rf_features': self._collect_meta_rf_features_snapshot(snapshot, signal),
=======
>>>>>>> origin/main
                        }

        closed = self._mark_to_market(snapshot)
        if closed:
            rr_achieved = self._rr_from_reason(str(closed.get('reason', 'manual')))
            direction = str(closed.get('direction', 'LONG')).upper()
            entry = float(closed.get('entry', 0.0))
            exit_price = float(closed.get('exit', 0.0))
            if entry > 0:
                if direction == 'SHORT':
                    pnl_pct = ((entry - exit_price) / entry) * 100.0
                else:
                    pnl_pct = ((exit_price - entry) / entry) * 100.0
            else:
                pnl_pct = 0.0

            reason = str(closed.get('reason', 'timeout')).lower()
            if reason.startswith('tp'):
                outcome = 'TP'
            elif reason in {'stop', 'sl'}:
                outcome = 'SL'
            else:
                outcome = 'TIMEOUT'
            if self._open_signal_state is not None:
                self._close_btc_signal_row(
                    signal_ref=str(self._open_signal_state.get('signal_ref', '')),
                    is_minimal_schema=bool(self._open_signal_state.get('is_minimal_schema', False)),
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                )
            self._open_signal_state = None
            self._last_open_signal_id = None
            self._last_saved_btc_signal = 'HOLD'

            row = {
                **closed,
                'rr_achieved': rr_achieved,
                'mae': 0.0,
                'mfe': max(rr_achieved, 0.0),
                'confidence': float(self._open_trade_context.get('confidence', 0.0)),
                'strategy': str(self._open_trade_context.get('strategy', 'NONE')),
            }
            self.performance_tracker.record_trade(row)
            rf_feat = self._open_trade_context.get('meta_rf_features') if isinstance(self._open_trade_context, dict) else None
            if isinstance(rf_feat, dict) and rf_feat:
                try:
                    self.meta_labeling_engine.record_closed_trade(
                        features=rf_feat,
                        profitable=float(closed.get('pnl_usd', 0.0)) > 0.0,
                    )
                except Exception as exc:
                    logger.debug('meta RF training append skipped: %s', exc)
            factors = list(self._open_trade_context.get('factors', []))
            if factors:
                self.stacker.record_outcome(
                    factors=factors,
                    was_win=float(closed.get('pnl_usd', 0.0)) > 0,
                    as_of_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
                )
            self._open_trade_context = {}

    def _mark_to_market(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        if self.paper.open_position is None:
            return None
        candles = snapshot.get('candles', {}).get('1m', [])
        if not candles:
            return None
        last_price = float(candles[-1].get('close', 0.0))
        closed = self.paper.mark_to_market(last_price)
        if closed:
            direction = str(closed.get('direction', 'LONG')).upper()
            entry = float(closed.get('entry', 0.0))
            exit_price = float(closed.get('exit', 0.0))
            if entry > 0:
                pnl_pct = ((entry - exit_price) / entry) * 100.0 if direction == 'SHORT' else ((exit_price - entry) / entry) * 100.0
            else:
                pnl_pct = 0.0
            closed_at = str(closed.get('closed_at_utc', '')).strip()
            try:
                closed_now = datetime.fromisoformat(closed_at.replace('Z', '+00:00')) if closed_at else datetime.now(timezone.utc)
            except Exception:
                closed_now = datetime.now(timezone.utc)
            self.trade_throttle.record_trade(pnl_pct=pnl_pct, now=closed_now)
        return closed

    @staticmethod
    def _rr_from_reason(reason: str) -> float:
        # Nominal RR aligned with risk.py TP multipliers (~0.9–1.5 / ~3–4 / ~5–7 by regime).
        if reason == 'tp1':
            return 1.2
        if reason == 'tp2':
            return 3.5
        if reason == 'tp3':
            return 6.0
        if reason == 'stop':
            return -1.0
        return 0.0

    @staticmethod
    def _format_telegram(payload: dict[str, Any]) -> str:
        return (
            f"BTC Signal: {payload.get('signal')}\n"
            f"Validated: {payload.get('validated')}\n"
            f"Confidence: {payload.get('confidence')}%\n"
            f"Stacked Prob: {payload.get('stacked_probability')}%\n"
            f"Entry: {payload.get('entry_zone')}\n"
            f"SL: {payload.get('stop_loss')}\n"
            f"TP: {payload.get('take_profit')}\n"
            f"Algo: {payload.get('algo')} ({payload.get('strategy')})\n"
            f"As Of UTC: {payload.get('as_of_utc')}"
        )

    def _monitoring_dict(self) -> dict[str, Any]:
        stats = self.performance_tracker.stats(portfolio_heat_pct=self._portfolio_heat())
        calibration = self.auto_corrector.check(self.performance_tracker.trades)
        return {
            'recent_win_rate': stats.recent_win_rate,
            'current_drawdown_pct': stats.current_drawdown_pct,
            'portfolio_heat_pct': stats.portfolio_heat_pct,
            'auto_pause': False,
            'loss_streak': stats.loss_streak,
            'trades_count': stats.trades_count,
            'avg_rr': stats.avg_rr,
            'avg_mae': stats.avg_mae,
            'avg_mfe': stats.avg_mfe,
            'calibrated': calibration.calibrated,
            'calibration_reason': calibration.reason,
            'confidence_gap': calibration.confidence_gap,
        }

    def _portfolio_heat(self) -> float:
        if self.paper.open_position is None:
            return 0.0
        return float(settings.risk_per_trade_pct)

    async def health(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        ws_stats = await self.hub.stats()
        return {
            'status': 'ok',
            'running': self._running,
            'ws': ws_stats,
            'last_ws_update_utc': snap.get('last_ws_update_utc', ''),
            'latency': snap.get('latency', {}),
            'buffers': {k: len(v) for k, v in snap.get('candles', {}).items()},
            'model_version': self.model.version,
            'paper_open': self.paper.open_position is not None,
            'paper_closed_trades': len(self.paper.closed),
            'monitoring': snap.get('monitoring_stats', {}),
            'redis_state': {
                'enabled': bool(settings.redis_state_enabled),
                'connected': bool(self.redis_state.connected),
                'url': settings.redis_url,
            },
            'server_time_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        }

    async def latest_signal(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        out = dict(snap.get('latest_signal', {}))
        # Buffer/Redis may lag until the next 15m bar; model artifacts can reload independently.
        out['model_version'] = self.model.version
        return out

    async def signal_history(self) -> list[dict[str, Any]]:
        snap = await self.buffer.snapshot()
        return snap.get('signal_history', [])[:100]

    async def latest_regime(self) -> dict[str, Any]:
        sig = await self.latest_signal()
        return {
            'market_regime': sig.get('market_regime', 'unknown'),
            'as_of_utc': sig.get('as_of_utc', ''),
            'reason': sig.get('reason', ''),
        }

    async def current_features(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        if not self._has_required_candles(snap):
            return {'status': 'warming_up'}
        state = build_feature_state(snap)
        vec, mapping = state.to_vector(direction='LONG', regime='sideways_range')
        return {'feature_vector': vec.squeeze(0).tolist(), 'feature_map': mapping}

    async def model_performance(self) -> dict[str, Any]:
        root = Path(settings.model_dir)
        payload: dict[str, Any] = {}
        default_meta = root / 'metadata.json'
        if default_meta.exists():
            payload['default'] = json.loads(default_meta.read_text(encoding='utf-8'))

        for regime in ['bullish_trend', 'bearish_trend', 'sideways_range', 'breakout']:
            path = root / regime / 'metadata.json'
            if path.exists():
                payload[regime] = json.loads(path.read_text(encoding='utf-8'))

        if not payload:
            return {'status': 'no_metadata'}
        return payload

    async def monitoring_stats(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        return snap.get('monitoring_stats', {})

    async def monitoring_edges(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        return {'edges': snap.get('edge_stats', {}), 'priors': self.stacker.current_edges()}

    async def trigger_model_retrain(self, payload: dict[str, Any]) -> dict[str, Any]:
        data_path = str(payload.get('data_path', '')).strip()
        if not data_path:
            return {'started': False, 'reason': 'data_path is required'}
        out = str(payload.get('out_dir', settings.model_dir))
        return await self.retrainer.run(data_path=data_path, out_dir=out)

    async def open_paper_trade(self, payload: dict[str, Any]) -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        regime_label = str(payload.get('regime', 'SIDEWAYS'))
        can_open, throttle_reason = self.trade_throttle.can_trade(regime=regime_label, now=now_utc)
        if not can_open:
            logger.info(
                'Manual paper trade blocked by throttle: regime=%s reason=%s',
                regime_label,
                throttle_reason,
            )
            return {'opened': False, 'open_position': self.paper.open_position is not None, 'reason': throttle_reason}
        opened = self.paper.open(
            direction=str(payload.get('direction', 'LONG')).upper(),
            entry=float(payload.get('entry', 0.0)),
            stop=float(payload.get('stop', 0.0)),
            tp1=float(payload.get('tp1', 0.0)),
            tp2=float(payload.get('tp2', 0.0)),
            tp3=float(payload.get('tp3', 0.0)),
            size_btc=float(payload.get('size_btc', 0.0)),
            confidence=float(payload.get('confidence', 100.0)),
            regime=regime_label,
        )
        return {'opened': opened, 'open_position': self.paper.open_position is not None}

    async def close_paper_trade(self, payload: dict[str, Any]) -> dict[str, Any]:
        closed = self.paper.close(float(payload.get('exit_price', 0.0)), reason=str(payload.get('reason', 'manual')))
        if closed:
            row = {**closed, 'rr_achieved': self._rr_from_reason(str(closed.get('reason', 'manual'))), 'mae': 0.0, 'mfe': 0.0}
            self.performance_tracker.record_trade(row)
            direction = str(closed.get('direction', 'LONG')).upper()
            entry = float(closed.get('entry', 0.0))
            exit_price = float(closed.get('exit', 0.0))
            if entry > 0:
                pnl_pct = ((entry - exit_price) / entry) * 100.0 if direction == 'SHORT' else ((exit_price - entry) / entry) * 100.0
            else:
                pnl_pct = 0.0
            closed_at = str(closed.get('closed_at_utc', '')).strip()
            try:
                closed_now = datetime.fromisoformat(closed_at.replace('Z', '+00:00')) if closed_at else datetime.now(timezone.utc)
            except Exception:
                closed_now = datetime.now(timezone.utc)
            self.trade_throttle.record_trade(pnl_pct=pnl_pct, now=closed_now)
            reason = str(closed.get('reason', 'timeout')).lower()
            outcome = 'TP' if reason.startswith('tp') else 'SL' if reason in {'stop', 'sl'} else 'TIMEOUT'
            if self._open_signal_state is not None:
                self._close_btc_signal_row(
                    signal_ref=str(self._open_signal_state.get('signal_ref', '')),
                    is_minimal_schema=bool(self._open_signal_state.get('is_minimal_schema', False)),
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                )
            self._open_signal_state = None
            self._last_open_signal_id = None
            self._last_saved_btc_signal = 'HOLD'
        return {'closed': closed}

    async def market_klines(self, timeframe: str, limit: int) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        rows = snap.get('candles', {}).get(timeframe, [])
        return {'symbol': 'BTCUSDT', 'timeframe': timeframe, 'rows': rows[-limit:]}

    async def market_history(self, timeframe: str, limit: int) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.get(
                f'{settings.binance_rest_base}/fapi/v1/klines',
                params={'symbol': 'BTCUSDT', 'interval': timeframe, 'limit': min(limit, 5000)},
            )
            resp.raise_for_status()
            rows = resp.json()
        parsed = [
            {
                'open_time': int(r[0]),
                'open': float(r[1]),
                'high': float(r[2]),
                'low': float(r[3]),
                'close': float(r[4]),
                'volume': float(r[5]),
                'close_time': int(r[6]),
            }
            for r in rows
        ]
        return {'symbol': 'BTCUSDT', 'timeframe': timeframe, 'rows': parsed}

    async def _live_intelligence_bundle(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        if not self._has_required_candles(snap):
            return {
                "status": "warming_up",
                "as_of_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            }
        state = build_feature_state(snap)
        regime = classify_regime(snap, state)
        get_ic_monitor().maybe_record_daily(snap, state, regime.regime)
        latest_signal = snap.get("latest_signal", {})
        bundle = self._compute_intelligence_bundle(
            snapshot=snap,
            state=state,
            regime_name=str(regime.regime),
            signal_payload=latest_signal if isinstance(latest_signal, dict) else None,
            regime_state_probs=regime.state_probs,
        )
        self._latest_intelligence_bundle = bundle
        return bundle

    async def orderflow_intelligence(self) -> dict[str, Any]:
        bundle = await self._live_intelligence_bundle()
        if bundle.get("status") == "warming_up":
            return {"status": "warming_up", "decision_state": "NO_TRADE"}
        payload = dict(bundle.get("order_flow", {}))
        decision_block = bundle.get("decision_engine", {})
        payload["decision"] = decision_block.get("decision", "HOLD")
        payload["confidence"] = decision_block.get("confidence", 0.0)
        payload["reason"] = decision_block.get("reason", payload.get("reason", ""))
        payload["trade_triggers"] = decision_block.get("trade_triggers", [])
        payload["blockers"] = decision_block.get("blockers", [])
        payload["as_of_utc"] = bundle.get("as_of_utc") or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('orderflow', payload, ttl_seconds=10)
        return payload

    async def decision_intelligence(self) -> dict[str, Any]:
        bundle = await self._live_intelligence_bundle()
        if bundle.get("status") == "warming_up":
            return {"status": "warming_up", "decision": "HOLD"}
        payload = dict(bundle.get("decision_engine", {}))
        payload["decision_breakdown"] = bundle.get("decision_breakdown", {})
        payload["factor_contributions"] = bundle.get("factor_contributions", [])
        payload["trade_verdict"] = bundle.get("trade_verdict", {})
        payload["meta_decision"] = bundle.get("meta_decision", {})
        payload["meta_labeling"] = bundle.get("meta_labeling", {})
        payload["validation_engine"] = bundle.get("validation_engine", {})
        payload["strategy_selection"] = bundle.get("strategy_selection", {})
        payload["data_drift"] = bundle.get("data_drift", {})
        payload["meta_output"] = bundle.get("meta_output", {})
        payload["adaptive_learning"] = bundle.get("adaptive_learning", {})
        payload["as_of_utc"] = bundle.get("as_of_utc")
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('decision', payload, ttl_seconds=10)
        return payload

    async def probability_intelligence(self) -> dict[str, Any]:
        bundle = await self._live_intelligence_bundle()
        if bundle.get("status") == "warming_up":
            return {"status": "warming_up", "up_prob": 0.0, "down_prob": 0.0, "sideways_prob": 100.0}
        payload = dict(bundle.get("probability", {}))
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('probability', payload, ttl_seconds=10)
        return payload

    async def execution_plan_intelligence(self) -> dict[str, Any]:
        bundle = await self._live_intelligence_bundle()
        if bundle.get("status") == "warming_up":
            return {"status": "warming_up", "slippage_risk": "HIGH"}
        payload = dict(bundle.get("execution_plan", {}))
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('execution_plan', payload, ttl_seconds=10)
        return payload

    async def intelligence_bundle(self) -> dict[str, Any]:
        bundle = await self._live_intelligence_bundle()
        if bundle.get("status") == "warming_up":
            return bundle
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('intelligence', bundle, ttl_seconds=10)
        return bundle

    async def volume_profile_intelligence(self, window_minutes: int = 45, bins: int = 24) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        payload = trade_based_volume_profile(
            agg_trades=snap.get('agg_trades', []),
            window_minutes=int(max(30, min(window_minutes, 60))),
            n_bins=int(max(12, min(bins, 48))),
        )
        payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('volprofile', payload, ttl_seconds=10)
        return payload

    async def volatility_intelligence(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        if not self._has_required_candles(snap):
            return {'status': 'warming_up', 'tradeability': 'NO_TRADE'}
        state = build_feature_state(snap)
        payload = volatility_tradeability(state.volatility)
        await self.buffer.set_volatility_tradeability(payload)
        payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('volatility', payload, ttl_seconds=10)
        return payload

    def _build_execution_payload(self, snapshot: dict[str, Any], state, direction: str | None = None) -> dict[str, Any]:
        latest = snapshot.get('latest_signal', {}) or {}
        resolved_direction = str(direction or latest.get('signal', 'LONG')).upper()
        if resolved_direction not in {'LONG', 'SHORT'}:
            resolved_direction = 'LONG'
        side = 'buy' if resolved_direction == 'LONG' else 'sell'

        fallback_vol = volatility_tradeability(state.volatility)
        check = evaluate_execution(
            snapshot.get('depth', {}),
            snapshot.get('agg_trades', []),
            resolved_direction,
            order_flow=state.order_flow,
            derivatives=state.derivatives,
            market_state=snapshot,
            fallback_tradeability=str(fallback_vol.get('tradeability', 'ALLOW')),
        )
        max_btc = recommended_max_position_btc(
            depth=snapshot.get('depth', {}),
            side=side,
            slippage_limit_pct=settings.slippage_reject_pct,
            min_qty_btc=0.01,
            max_qty_btc=8.0,
        )
        bid, ask, mid = MarketDataBuffer.best_bid_ask(snapshot.get('depth', {}))
        adverse = compute_adverse_selection(
            snapshot.get('depth', {}),
            snapshot.get('agg_trades', []),
            resolved_direction,
        )
        return {
            'direction': resolved_direction,
            'accepted': bool(check.accepted),
            'trigger_ready': bool(check.trigger_ready),
            'quality': check.quality,
            'spread_pct': round(float(check.spread_pct), 6),
            'slippage_pct': round(float(check.slippage_pct), 6),
            'price_move_30s_pct': round(float(check.price_move_30s_pct), 6),
            'recommended_max_btc': float(max_btc),
            'recommended_max_notional_usdt': round(float(max_btc * mid), 2) if mid > 0 else 0.0,
            'best_bid': round(float(bid), 2),
            'best_ask': round(float(ask), 2),
            'mid_price': round(float(mid), 2),
            'rejection_reason': check.reason if not check.accepted else '',
            'rejection_code': execution_rejection_code(check.reason) if not check.accepted else '',
            'adverse_selection_flag': bool(adverse.adverse_selection_flag),
            'execution_mode_recommendation': adverse.execution_mode_recommendation,
            'adverse_selection': {
                'spread_widening': adverse.spread_widening,
                'mid_drift_pct': adverse.mid_drift_pct,
                'spread_pct_book': adverse.spread_pct,
                'spread_baseline_proxy_pct': adverse.spread_baseline_proxy_pct,
                'noise_floor_pct': adverse.noise_floor_pct,
                'reason': adverse.reason,
            },
            'as_of_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        }

    async def execution_intelligence(self, direction: str | None = None) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        if not self._has_required_candles(snap):
            return {'status': 'warming_up', 'accepted': False, 'decision_state': 'NO_TRADE'}
        state = build_feature_state(snap)
        payload = self._build_execution_payload(snapshot=snap, state=state, direction=direction)
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('execution', payload, ttl_seconds=10)
        return payload

