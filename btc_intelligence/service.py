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
from btc_intelligence.signals.intelligence import (
    combined_btc_signal,
    execution_rejection_code,
    order_flow_decision_state,
    trade_based_volume_profile,
    volatility_tradeability,
)
from btc_intelligence.signals.probability_stacker import ProbabilityStacker
from btc_intelligence.state import RedisStateStore
from btc_intelligence.utils.notifier import TelegramNotifier


logger = logging.getLogger(__name__)
MIN_CONFIDENCE = 25.0
MAX_OPEN_SIGNALS = 1
MIN_HOLD_SECONDS = 300
MIN_PRICE_MOVE_PCT = 0.05
MIN_CONFIDENCE_CHANGE = 10.0


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
        self.paper = PaperTradeBook()
        self.hub = LiveWebSocketHub()
        self.notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)

        self.performance_tracker = PerformanceTracker(settings.monitoring_log_path)
        self.auto_pause_manager = AutoPauseManager()
        self.auto_corrector = AutoCorrector()
        self.retrainer = ModelRetrainer(self.model)

        self._signal_task: asyncio.Task | None = None
        self._running = False
        self.feature_seq: deque[np.ndarray] = deque(maxlen=200)
        self._open_trade_context: dict[str, Any] = {}
        self._shared_signals_db = Path('data/signals.db')
        self._shared_signals_db.parent.mkdir(parents=True, exist_ok=True)
        self._last_saved_btc_signal = 'HOLD'
        self._last_open_signal_id: str | None = None
        self._open_signal_state: dict[str, Any] | None = None
        self._last_open_pnl_update_ts: float = 0.0
        self._last_signal_time: float | None = None
        self._last_signal_direction: str | None = None
        self._last_signal_confidence: float | None = None

        self.signal_log_path = Path(settings.signal_log_path)
        self.signal_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_shared_db_schema()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if settings.redis_state_enabled:
            await self.redis_state.connect()

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
        logger.info('Runtime started')

    async def stop(self) -> None:
        self._running = False
        if self._signal_task:
            self._signal_task.cancel()
            await asyncio.gather(self._signal_task, return_exceptions=True)

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
        logger.info('Runtime stopped')

    async def _signal_loop(self) -> None:
        while self._running:
            try:
                snapshot = await self.buffer.snapshot()
                monitoring = self._monitoring_dict()
                pause = self.auto_pause_manager.evaluate(self.performance_tracker.stats(portfolio_heat_pct=monitoring['portfolio_heat_pct']))
                monitoring['auto_pause'] = pause.paused
                monitoring['auto_pause_reason'] = pause.reason
                await self.buffer.set_monitoring(monitoring)
                await self.buffer.set_edge_stats(self.stacker.current_edges())

                if not self._has_required_candles(snapshot):
                    await asyncio.sleep(settings.feature_eval_interval_sec)
                    continue

                last_close = int(snapshot.get('last_15m_close_time', 0))
                if last_close <= int(snapshot.get('last_feature_eval_close_time', 0)):
                    self._mark_to_market(snapshot)
                    state = build_feature_state(snapshot)
                    await self._publish_intelligence_state(snapshot, state)
                    await asyncio.sleep(2)
                    continue

                state = build_feature_state(snapshot)
                vol_payload = volatility_tradeability(state.volatility)
                await self.buffer.set_volatility_tradeability(vol_payload)

                regime = classify_regime(snapshot, state)
                payload = self.engine.build(
                    snapshot=snapshot,
                    state=state,
                    regime=regime,
                    sequence_vectors=list(self.feature_seq),
                    monitoring_stats=monitoring,
                    auto_pause=bool(monitoring['auto_pause']),
                )

                vec, _ = state.to_vector(direction='LONG', regime=regime.regime)
                self.feature_seq.append(vec.squeeze(0))

                await self.buffer.set_latest_signal(payload)
                await self.buffer.mark_feature_eval(last_close)
                await self._append_signal_log(payload)
                await self.hub.broadcast(payload)
                await self._publish_intelligence_state(snapshot, state)

                self._handle_paper_trade(payload, snapshot)

                if payload.get('signal') in {'LONG', 'SHORT'}:
                    await self.notifier.send(self._format_telegram(payload))

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

    @staticmethod
    def _calc_signal_pnl_pct(signal: str, entry_price: float, current_price: float) -> float:
        if entry_price <= 0 or current_price <= 0:
            return 0.0
        if str(signal).upper() == 'SHORT':
            return ((entry_price - current_price) / entry_price) * 100.0
        return ((current_price - entry_price) / entry_price) * 100.0

    def _next_btc_signal_id(self, cur: sqlite3.Cursor) -> str:
        cur.execute("SELECT COUNT(*) FROM signals WHERE ticker = 'BTCUSDT'")
        count = int((cur.fetchone() or [0])[0] or 0)
        return f"BTC-{count + 1:03d}"

    def _insert_btc_signal_row(
        self,
        signal: str,
        confidence: float,
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
                            ticker, signal, confidence, outcome, pnl_pct,
                            entry_price, sl, tp1, tp2, tp3, rr_ratio,
                            size_multiplier, signal_note, timestamp
                        )
                        VALUES ('BTCUSDT', ?, ?, 'OPEN', 0.0, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                        """,
                        (
                            str(signal).upper(),
                            float(confidence),
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
                        signal, strength, confidence, alpha_score, regime,
                        entry_price, stop_loss, take_profit, sl, tp1, tp2, tp3, rr_ratio, position_size_pct,
                        kelly_fraction, hurst_exponent, factor_scores, ic_weights,
                        slippage_cost_pct, cost_pct, net_alpha_score, cs_alpha_score,
                        execution_price, implementation_shortfall_pct,
                        outcome, result, pnl_pct, size_multiplier, signal_note
                    )
                    VALUES (?, 'BTCUSDT', ?, 'BTCUSDT', 'crypto', 'intraday',
                            ?, 'MEDIUM', ?, 0.0, 'SIDEWAYS',
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
                    params.append(int(rowid))
                    cur.execute(
                        f"UPDATE signals SET {', '.join(set_parts)} WHERE rowid = ?",
                        params,
                    )

                    state_ref = str(self._open_signal_state.get('signal_ref', '')) if self._open_signal_state else ''
                    if state_ref and state_ref in {str(rowid), str(signal_ref)}:
                        self._open_signal_state = None
                        self._last_open_signal_id = None
                        self._last_saved_btc_signal = 'HOLD'

                conn.commit()
                return len(rows)
        except Exception as exc:
            logger.error('Failed to close BTC open signals: %s', exc, exc_info=True)
            return 0

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

                cur.execute(
                    f"""
                    SELECT
                        rowid,
                        {signal_id_expr} AS signal_ref,
                        COALESCE(signal, 'HOLD') AS signal,
                        {entry_expr} AS entry_price,
                        {sl_expr} AS sl,
                        {tp1_expr} AS tp1,
                        CAST((strftime('%s','now') - strftime('%s', COALESCE(timestamp, datetime('now')))) AS INTEGER) AS hold_seconds
                    FROM signals
                    WHERE {ticker_expr} AND {open_expr}
                    """
                )
                rows = cur.fetchall()
                if not rows:
                    return

                for rowid, signal_ref, signal, entry_price, sl, tp1, hold_seconds in rows:
                    side = str(signal or 'HOLD').upper()
                    if side not in {'LONG', 'SHORT'}:
                        continue
                    entry = float(entry_price or 0.0)
                    if entry <= 0:
                        continue
                    self._update_mfe_mae(
                        signal_id=str(signal_ref),
                        current_price=float(current_price),
                        entry_price=float(entry),
                        signal_direction=side,
                    )
                    sl_val = float(sl or 0.0)
                    tp1_val = float(tp1 or 0.0)
                    hold_secs = int(max(0, float(hold_seconds or 0)))
                    price_move_pct = abs(((float(current_price) - float(entry)) / float(entry)) * 100.0)
                    can_check_exit = hold_secs >= MIN_HOLD_SECONDS and price_move_pct > MIN_PRICE_MOVE_PCT

                    outcome: str | None = None
                    if can_check_exit:
                        if side == 'LONG':
                            if sl_val > 0 and current_price <= sl_val:
                                outcome = 'SL'
                                logger.info("BTC SL hit: %s entry=%s sl=%s", side, format(entry, "g"), format(sl_val, "g"))
                            elif tp1_val > 0 and current_price >= tp1_val:
                                outcome = 'TP1'
                                logger.info("BTC TP1 hit: %s entry=%s tp1=%s", side, format(entry, "g"), format(tp1_val, "g"))
                        else:
                            if sl_val > 0 and current_price >= sl_val:
                                outcome = 'SL'
                                logger.info("BTC SL hit: %s entry=%s sl=%s", side, format(entry, "g"), format(sl_val, "g"))
                            elif tp1_val > 0 and current_price <= tp1_val:
                                outcome = 'TP1'
                                logger.info("BTC TP1 hit: %s entry=%s tp1=%s", side, format(entry, "g"), format(tp1_val, "g"))

                    pnl_pct = self._calc_signal_pnl_pct(side, entry, current_price)

                    set_parts = ["pnl_pct = ?"]
                    params: list[Any] = [float(pnl_pct)]
                    if outcome is not None:
                        if 'outcome' in cols:
                            set_parts.append("outcome = ?")
                            params.append(outcome)
                        if 'result' in cols:
                            set_parts.append("result = ?")
                            params.append(outcome)
                    params.append(int(rowid))
                    cur.execute(
                        f"UPDATE signals SET {', '.join(set_parts)} WHERE rowid = ?",
                        params,
                    )

                    if outcome is not None and self._open_signal_state is not None:
                        state_ref = str(self._open_signal_state.get('signal_ref', ''))
                        if state_ref and state_ref == str(signal_ref):
                            self._open_signal_state = None
                            self._last_open_signal_id = None
                            self._last_saved_btc_signal = 'HOLD'

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
        if not settings.redis_state_enabled:
            return
        signal_payload = await self.redis_state.get_json('signal', default={})
        if not isinstance(signal_payload, dict):
            return

        signal = str(signal_payload.get('signal', 'HOLD')).upper()
        if signal not in {'LONG', 'SHORT'}:
            return

        confidence = float(signal_payload.get('confidence', 0.0))
        if confidence < MIN_CONFIDENCE:
            return
        now_ts = float(time_module.time())
        if self._last_signal_time is not None:
            elapsed = now_ts - float(self._last_signal_time)
            direction_flipped = (
                self._last_signal_direction is not None
                and signal != str(self._last_signal_direction).upper()
            )
            confidence_changed = (
                self._last_signal_confidence is None
                or abs(float(confidence) - float(self._last_signal_confidence)) > MIN_CONFIDENCE_CHANGE
            )
            hold_elapsed = elapsed >= MIN_HOLD_SECONDS
            if not (direction_flipped or confidence_changed or hold_elapsed):
                logger.debug(
                    "Skipping signal — %.0fs since last, no flip, confidence delta <= %.1f",
                    elapsed,
                    MIN_CONFIDENCE_CHANGE,
                )
                return
            if signal == str(self._last_signal_direction).upper() and not (confidence_changed or hold_elapsed):
                logger.debug("Skipping — same direction %s repeated", signal)
                return
        size_multiplier = float(signal_payload.get('size_multiplier', 1.0))
        signal_note = str(signal_payload.get('signal_note', 'normal'))
        entry_price = float(signal_payload.get('entry_price', current_price))
        sl = float(signal_payload.get('sl', entry_price))
        tp1 = float(signal_payload.get('tp1', entry_price))
        tp2 = float(signal_payload.get('tp2', entry_price))
        tp3 = float(signal_payload.get('tp3', entry_price))
        rr_ratio = float(signal_payload.get('rr_ratio', 1.0))
        if entry_price <= 0:
            entry_price = float(current_price)

        open_count = await asyncio.to_thread(self._count_open_btc_signals)
        if open_count >= MAX_OPEN_SIGNALS:
            await asyncio.to_thread(self._close_all_open_btc_signals, float(current_price), 'CLOSED')

        open_state = self._open_signal_state
        if open_state is not None:
            open_signal = str(open_state.get('signal', 'HOLD')).upper()

            # Safety close for stale in-memory state that may not be in DB open set anymore.
            prev_entry = float(open_state.get('entry_price', 0.0))
            prev_pnl_pct = self._calc_signal_pnl_pct(open_signal, prev_entry, current_price)
            await asyncio.to_thread(
                self._close_btc_signal_row,
                str(open_state.get('signal_ref', '')),
                bool(open_state.get('is_minimal_schema', False)),
                prev_pnl_pct,
                'CLOSED',
            )
            self._open_signal_state = None
            self._last_open_signal_id = None

        insert_meta = await asyncio.to_thread(
            self._insert_btc_signal_row,
            signal,
            confidence,
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
            self._open_signal_state = {
                "signal_ref": str(insert_meta.get("signal_ref", "")),
                "is_minimal_schema": bool(insert_meta.get("is_minimal_schema", False)),
                "signal": signal,
                "entry_price": float(entry_price),
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": float(tp2),
                "opened_at_ts": float(time_module.time()),
            }
            self._last_open_signal_id = str(insert_meta.get("signal_ref", ""))
            self._last_saved_btc_signal = signal
            self._last_signal_time = now_ts
            self._last_signal_direction = signal
            self._last_signal_confidence = float(confidence)
            logger.info(
                "BTC signal saved: %s conf=%.1f%% size=%sx note=%s",
                signal,
                confidence,
                format(size_multiplier, "g"),
                signal_note,
            )

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

    async def _publish_intelligence_state(self, snapshot: dict[str, Any], state) -> None:
        if not (settings.redis_state_enabled and self.redis_state.connected):
            return
        orderflow_payload = order_flow_decision_state(state.order_flow)
        orderflow_payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        vol_payload = volatility_tradeability(state.volatility)
        vol_payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        volume_payload = trade_based_volume_profile(
            agg_trades=snapshot.get('agg_trades', []),
            window_minutes=45,
            n_bins=24,
        )
        volume_payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        preferred_direction = 'LONG' if str(orderflow_payload.get('decision_state', '')).upper() != 'FAVOR_SHORT' else 'SHORT'
        execution_payload = self._build_execution_payload(snapshot=snapshot, state=state, direction=preferred_direction)

        await self.redis_state.set_json('orderflow', orderflow_payload, ttl_seconds=10)
        await self.redis_state.set_json('volatility', vol_payload, ttl_seconds=10)
        await self.redis_state.set_json('volprofile', volume_payload, ttl_seconds=10)
        await self.redis_state.set_json('execution', execution_payload, ttl_seconds=10)
        current_price = self._extract_current_price(snapshot)
        mtf_bias = await self._read_mtf_bias()
        combined_payload = combined_btc_signal(
            orderflow_payload,
            vol_payload,
            current_price=current_price,
            mtf_bias=mtf_bias,
        )
        combined_payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        await self.redis_state.set_json('signal', combined_payload, ttl_seconds=10)
        await self._persist_btc_signal_from_redis(current_price=current_price)
        await asyncio.to_thread(self._refresh_btc_open_signals, current_price)

    def _handle_paper_trade(self, payload: dict[str, Any], snapshot: dict[str, Any]) -> None:
        if not settings.paper_trade:
            return

        signal = str(payload.get('signal', 'HOLD'))
        if signal in {'LONG', 'SHORT'} and self.paper.open_position is None:
            entry_zone = payload.get('entry_zone', [0, 0])
            entry = (float(entry_zone[0]) + float(entry_zone[1])) / 2.0
            opened = self.paper.open(
                direction=signal,
                entry=entry,
                stop=float(payload.get('stop_loss', 0.0)),
                tp1=float(payload.get('take_profit', {}).get('TP1', 0.0)),
                tp2=float(payload.get('take_profit', {}).get('TP2', 0.0)),
                tp3=float(payload.get('take_profit', {}).get('TP3', 0.0)),
                size_btc=float(payload.get('position_size_btc', 0.0)),
            )
            if opened:
                self._open_trade_context = {
                    'factors': list(payload.get('factors_present', [])),
                    'confidence': float(payload.get('confidence', 0.0)),
                    'strategy': str(payload.get('strategy', 'NONE')),
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
        return self.paper.mark_to_market(last_price)

    @staticmethod
    def _rr_from_reason(reason: str) -> float:
        if reason == 'tp1':
            return 1.5
        if reason == 'tp2':
            return 2.5
        if reason == 'tp3':
            return 4.0
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
        return snap.get('latest_signal', {})

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
        opened = self.paper.open(
            direction=str(payload.get('direction', 'LONG')).upper(),
            entry=float(payload.get('entry', 0.0)),
            stop=float(payload.get('stop', 0.0)),
            tp1=float(payload.get('tp1', 0.0)),
            tp2=float(payload.get('tp2', 0.0)),
            tp3=float(payload.get('tp3', 0.0)),
            size_btc=float(payload.get('size_btc', 0.0)),
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

    async def orderflow_intelligence(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        if not self._has_required_candles(snap):
            return {'status': 'warming_up', 'decision_state': 'NO_TRADE'}
        state = build_feature_state(snap)
        payload = order_flow_decision_state(state.order_flow)
        vol_payload = volatility_tradeability(state.volatility)
        current_price = self._extract_current_price(snap)
        mtf_bias = await self._read_mtf_bias()
        combined = combined_btc_signal(
            payload,
            vol_payload,
            current_price=current_price,
            mtf_bias=mtf_bias,
        )
        payload.update(
            {
                "signal": combined.get("signal", "HOLD"),
                "confidence": combined.get("confidence", 0.0),
                "entry_price": combined.get("entry_price", 0.0),
                "sl": combined.get("sl", 0.0),
                "tp1": combined.get("tp1", 0.0),
                "tp2": combined.get("tp2", 0.0),
                "tp3": combined.get("tp3", 0.0),
                "rr_ratio": combined.get("rr_ratio", 1.0),
                "size_multiplier": combined.get("size_multiplier", 1.0),
                "signal_note": combined.get("signal_note", "normal"),
                "volatility_regime": combined.get("volatility_regime", vol_payload.get("volatility_regime", "NORMAL")),
                "tradeability": combined.get("tradeability", vol_payload.get("tradeability", "ALLOW")),
                "mtf_bias_4h": combined.get("mtf_bias_4h", "UNKNOWN"),
            }
        )
        payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        if settings.redis_state_enabled and self.redis_state.connected:
            await self.redis_state.set_json('orderflow', payload, ttl_seconds=10)
        return payload

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
