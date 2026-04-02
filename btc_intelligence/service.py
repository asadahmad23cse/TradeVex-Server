from __future__ import annotations

import asyncio
import json
import logging
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
    execution_rejection_code,
    order_flow_decision_state,
    trade_based_volume_profile,
    volatility_tradeability,
)
from btc_intelligence.signals.probability_stacker import ProbabilityStacker
from btc_intelligence.state import RedisStateStore
from btc_intelligence.utils.notifier import TelegramNotifier


logger = logging.getLogger(__name__)


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

        self.signal_log_path = Path(settings.signal_log_path)
        self.signal_log_path.parent.mkdir(parents=True, exist_ok=True)

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
                    await asyncio.sleep(2)
                    continue

                state = build_feature_state(snapshot)
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
        payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        return payload

    async def volume_profile_intelligence(self, window_minutes: int = 45, bins: int = 24) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        payload = trade_based_volume_profile(
            agg_trades=snap.get('agg_trades', []),
            window_minutes=int(max(30, min(window_minutes, 60))),
            n_bins=int(max(12, min(bins, 48))),
        )
        payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        return payload

    async def volatility_intelligence(self) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        if not self._has_required_candles(snap):
            return {'status': 'warming_up', 'tradeability': 'NO_TRADE'}
        state = build_feature_state(snap)
        payload = volatility_tradeability(state.volatility)
        payload['as_of_utc'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        return payload

    async def execution_intelligence(self, direction: str | None = None) -> dict[str, Any]:
        snap = await self.buffer.snapshot()
        if not self._has_required_candles(snap):
            return {'status': 'warming_up', 'accepted': False, 'decision_state': 'NO_TRADE'}

        latest = snap.get('latest_signal', {}) or {}
        resolved_direction = str(direction or latest.get('signal', 'LONG')).upper()
        if resolved_direction not in {'LONG', 'SHORT'}:
            resolved_direction = 'LONG'
        side = 'buy' if resolved_direction == 'LONG' else 'sell'

        state = build_feature_state(snap)
        check = evaluate_execution(
            snap.get('depth', {}),
            snap.get('agg_trades', []),
            resolved_direction,
            order_flow=state.order_flow,
            derivatives=state.derivatives,
        )
        max_btc = recommended_max_position_btc(
            depth=snap.get('depth', {}),
            side=side,
            slippage_limit_pct=settings.slippage_reject_pct,
            min_qty_btc=0.01,
            max_qty_btc=8.0,
        )
        bid, ask, mid = MarketDataBuffer.best_bid_ask(snap.get('depth', {}))

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
