"""
Layer 10 â€” FastAPI Dashboard + WebSocket.

Binds to 127.0.0.1 (localhost only).
WebSocket at /ws broadcasts new signals in real-time.
5 pages: Live Signals, Portfolio, History, Factor Analysis, Regime Monitor.
"""

from datetime import datetime, timedelta, timezone
import asyncio
import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import httpx
import redis
from fastapi import (  # type: ignore[import]
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, Response  # type: ignore[import]
from fastapi.staticfiles import StaticFiles  # type: ignore[import]
from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore[import]
from src.data.news_feed import get_btc_news
<<<<<<< HEAD
from src.data.signal_history import (
    get_history as get_signal_history,
    get_stats as get_signal_stats,
    record_signal,
)
from src.dashboard.btc_service import INTERVAL_TO_MS
=======
from src.data.signal_history import get_history as get_signal_history, get_stats as get_signal_stats
from src.dashboard.btc_service import INTERVAL_TO_MS, BitcoinMarketService
>>>>>>> origin/main
from src.dashboard.focus_engine import FocusQuantEngine
from src.options import ExpiryTracker, OptionsEngine
from src.compliance import SEBIComplianceEngine
from src.execution.broker import create_executor
from src.risk.kelly import KellyCalculator
from src.utils.notifiers import NotificationManager
from src.webhook.receiver import WebhookReceiver

try:
    import jwt  # type: ignore[import]
    _JWT = True
except ImportError:
    jwt = None  # type: ignore[assignment]
    _JWT = False

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="QuantTrader Dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state injected at startup
_store = None
_portfolio = None
_focus_engine: FocusQuantEngine | None = None
_btc_service: Any = None  # deprecated: local BTC engine removed; kept name for backward compat
_live_runner = None
_connected_ws: List[WebSocket] = []
_app_loop: asyncio.AbstractEventLoop | None = None
_dashboard_cfg: dict = {}
_options_engine: OptionsEngine | None = None
_expiry_tracker = ExpiryTracker()
_compliance_engine = SEBIComplianceEngine()

_webhook_receiver: WebhookReceiver | None = None
_webhook_enabled: bool = False
_webhook_request_counts: dict[str, list[float]] = {}
_options_provider: Optional[object] = None
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
_btc_proxy_cache: dict[str, dict[str, Any]] = {}
_DASHBOARD_STALE_CACHE_PREFIX = "dashboard:stale:"
_DASHBOARD_STALE_TTL_SEC = 300

BTC_INTEL_BASE = os.environ.get("BTC_INTELLIGENCE_BASE", "http://127.0.0.1:9000").rstrip("/")


def init_webhook(
    cfg: dict,
    store,
    broker,
    portfolio,
    kelly,
    notifier,
) -> None:
    """Enable WebhookReceiver when webhook.enabled and a valid secret are set."""
    global _webhook_receiver, _webhook_enabled
    _webhook_receiver = None
    _webhook_enabled = False
    try:
        wh_cfg = cfg.get("webhook") or {}
        if not wh_cfg.get("enabled", False):
            logger.info("Webhook bridge: disabled in config")
            return

        secret = str(wh_cfg.get("secret_token", "") or "").strip()
        if not secret or secret == "CHANGE_ME":
            logger.warning(
                "Webhook bridge: secret_token not set — webhook disabled for security",
            )
            return

        _webhook_receiver = WebhookReceiver(
            secret_token=secret,
            broker=broker,
            portfolio_tracker=portfolio,
            kelly_calculator=kelly,
            store=store,
            notifier=notifier,
            max_kelly=float(wh_cfg.get("max_webhook_kelly", 0.05)),
        )
        _webhook_enabled = True
        logger.info("Webhook bridge: enabled")
    except Exception as exc:
        logger.error("Webhook init failed (webhook disabled): %s", exc)


def _webhook_rate_limit_per_min() -> int:
    try:
        return int(((_dashboard_cfg.get("webhook") or {}).get("rate_limit_per_min", 60)))
    except Exception:
        return 60


def _check_rate_limit(client_ip: str) -> bool:
    limit = _webhook_rate_limit_per_min()
    now = time.time()
    window_start = now - 60.0
    counts = _webhook_request_counts.get(client_ip, [])
    counts = [t for t in counts if t > window_start]
    counts.append(now)
    _webhook_request_counts[client_ip] = counts
    return len(counts) <= limit


def init_dashboard(store, portfolio, config: dict | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
    """Inject SignalStore and PortfolioTracker into the dashboard."""
    global _store, _portfolio, _dashboard_cfg, _focus_engine, _btc_service, _options_engine
    _store = store
    _portfolio = portfolio
    _dashboard_cfg = config or {}
    _focus_engine = FocusQuantEngine(_dashboard_cfg)
    _btc_service = None
    if _options_engine is None:
        _options_engine = OptionsEngine()
    try:
        from src.paper_trading import get_auto_executor, get_paper_engine

        _ = get_paper_engine()
        _ = get_auto_executor()
        logger.info("Paper trading engine initialized")
    except Exception as e:
        logger.warning("Paper engine init failed: %s", e)

    risk = (_dashboard_cfg.get("risk") or {})
    broker = kwargs.get("broker") or create_executor(_dashboard_cfg)
    kelly = kwargs.get("kelly") or KellyCalculator(
        kelly_fraction=float(risk.get("kelly_fraction", 0.25)),
        max_position_pct=float(risk.get("max_position_size_pct", 5.0)) / 100.0,
        cold_start_pct=float(risk.get("cold_start_position_pct", 2.0)) / 100.0,
        min_trades=int(risk.get("min_kelly_trades", 30)),
    )
    notifier = kwargs.get("notifier") or NotificationManager(
        _dashboard_cfg.get("notifications", {}),
    )
    init_webhook(_dashboard_cfg, _store, broker, _portfolio, kelly, notifier)

    global _options_provider
    _options_provider = None
    try:
        from src.api.options_alt_data import OptionsAltDataProvider

        _options_provider = OptionsAltDataProvider(_dashboard_cfg)
    except Exception as exc:
        logger.warning("OptionsAltDataProvider init failed: %s", exc)


def set_live_runner(runner) -> None:  # type: ignore[no-untyped-def]
    """Attach/detach a LiveRunner instance for runtime telemetry endpoints."""
    global _live_runner
    _live_runner = runner


def _auth_config() -> dict:
    return (_dashboard_cfg.get("dashboard", {}) or {}).get("auth", {})


def _auth_enabled() -> bool:
    return bool(_auth_config().get("enabled", False) and _JWT)


def _issue_token(username: str) -> str:
    assert jwt is not None, "PyJWT is required for auth"
    cfg = _auth_config()
    secret = cfg.get("jwt_secret", "change-me")
    return jwt.encode({"sub": username}, secret, algorithm="HS256")


def _verify_token(token: str) -> bool:
    assert jwt is not None, "PyJWT is required for auth"
    cfg = _auth_config()
    secret = cfg.get("jwt_secret", "change-me")
    try:
        jwt.decode(token, secret, algorithms=["HS256"])
        return True
    except Exception:
        return False


class DashboardAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if not _auth_enabled():
            return await call_next(request)
        open_paths = {"/auth/token"}
        if (
            request.url.path.startswith("/static")
            or request.url.path.startswith("/webhook")
            or request.url.path == "/api/options-intelligence"
            or request.url.path in open_paths
        ):
            return await call_next(request)
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        if not token and "token" in request.query_params:
            token = request.query_params["token"]
        if not token or not _verify_token(token):
            raise HTTPException(status_code=401, detail="Dashboard auth required")
        return await call_next(request)


app.add_middleware(DashboardAuthMiddleware)

_response_cache: dict[str, dict[str, object]] = {}
_response_cache_ttl = {
    "/api/stock-signal": 30,
    "/api/chart-data": 60,
    "/api/options/chain": 60,
    "/api/options/signal": 30,
    "/api/options/iv-surface": 60,
    "/api/options/expiry": 300,
    "/api/market-overview": 30,
    "/api/btc/candles": 30,
    "/api/btc/sr-levels": 60,
    "/api/btc/liquidity-zones": 120,
    "/api/btc/markers": 60,
    "/api/btc/market-context": 10,
    "/api/btc/system-report": 10,
    "/api/btc/signal": 10,
    "/api/btc/signal/history": 10,
    "/api/btc/decision-intelligence": 10,
    "/api/btc/probability": 10,
    "/api/btc/execution-plan": 10,
    "/api/btc/news": 120,
    "/api/news": 120,
    "/api/portfolio": 15,
}


def _cache_ttl_for_path(path: str) -> int:
    for prefix, ttl in _response_cache_ttl.items():
        if path.startswith(prefix):
            return ttl
    return 0


def _db_sqlite_path() -> Path:
    db_url = ((_dashboard_cfg.get("database") or {}).get("url") or "data/signals.db")
    if "://" in str(db_url):
        # Keep local-dashboard fallback simple: only local sqlite file supported here.
        return Path("data/signals.db")
    return Path(str(db_url))


def _normalize_signal_row(row: dict[str, Any], idx_hint: int = 0) -> dict[str, Any]:
    signal_raw = str(row.get("signal", row.get("requested_signal", "HOLD"))).upper()
    if signal_raw == "BUY":
        signal_norm = "LONG"
    elif signal_raw == "SELL":
        signal_norm = "SHORT"
    else:
        signal_norm = signal_raw

    ticker = (
        row.get("ticker")
        or row.get("asset")
        or row.get("symbol")
        or "BTCUSDT"
    )
    timestamp = row.get("timestamp") or row.get("time") or row.get("as_of_utc") or ""
    outcome = row.get("outcome") or row.get("result") or ("OPEN" if row.get("close_price") is None else "CLOSED")
    status = str(outcome or "OPEN").upper()
    pnl_val = row.get("pnl_pct")
    return {
        "id": row.get("id") or row.get("signal_id") or idx_hint,
        "time": timestamp,
        "ticker": str(ticker).upper(),
        "type": row.get("type") or signal_norm,
        "signal": signal_norm,
        "requested_signal": row.get("requested_signal") or signal_norm,
        "confidence": float(row.get("confidence") or 0.0),
        "entry": row.get("entry") if row.get("entry") is not None else row.get("entry_price"),
        "stop_loss": row.get("stop_loss"),
        "tp1": row.get("tp1") if row.get("tp1") is not None else row.get("take_profit"),
        "risk_reward": row.get("risk_reward"),
        "result": status,
        "status": status,
        "reason": row.get("reason", ""),
        "pnl_pct": pnl_val,
    }


def _fetch_sqlite_signal_rows(limit: int = 200) -> list[dict[str, Any]]:
    db_path = _db_sqlite_path()
    if not db_path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cols = {str(r[1]) for r in cur.execute("PRAGMA table_info(signals)").fetchall()}
            if "signals" not in {str(r[0]) for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}:
                return []
            is_minimal_schema = (
                {"ticker", "signal", "confidence", "outcome", "pnl_pct", "timestamp"}.issubset(cols)
                and "signal_id" not in cols
                and "asset" not in cols
            )
            if is_minimal_schema:
                cur.execute(
                    """
                    SELECT rowid AS id, ticker, signal, confidence, outcome, pnl_pct, timestamp
                    FROM signals
                    ORDER BY datetime(timestamp) DESC, rowid DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                )
            else:
                # Main terminal schema.
                select_cols = [
                    "signal_id",
                    "timestamp",
                    "asset",
                    "signal",
                    "confidence",
                    "entry_price",
                    "stop_loss",
                    "take_profit",
                    "outcome",
                    "pnl_pct",
                ]
                if "result" in cols:
                    select_cols.append("result")
                cur.execute(
                    f"""
                    SELECT {", ".join(select_cols)}
                    FROM signals
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                )
            for i, row in enumerate(cur.fetchall(), start=1):
                out.append(_normalize_signal_row(dict(row), idx_hint=i))
    except Exception as exc:
        logger.warning("Failed to read combined signals from sqlite: %s", exc)
    return out


def _combined_signal_rows(limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if _store is not None:
        try:
            store_rows = _store.get_recent_signals(limit=max(limit, 50))
            rows.extend(_normalize_signal_row(x, idx_hint=i + 1) for i, x in enumerate(store_rows))
        except Exception as exc:
            logger.warning("Store signals fetch failed: %s", exc)
    sqlite_rows = _fetch_sqlite_signal_rows(limit=max(limit, 50))
    rows.extend(sqlite_rows)

    # Deduplicate by strong id if present, else by (ticker,time,signal).
    dedup: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows, start=1):
        key = str(row.get("id") or f"{row.get('ticker')}|{row.get('time')}|{row.get('signal')}|{i}")
        if key not in dedup:
            dedup[key] = row
    merged = list(dedup.values())
    merged.sort(key=lambda x: str(x.get("time") or ""), reverse=True)
    return merged[:limit]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _parse_utc(ts: Any) -> datetime | None:
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return datetime.fromisoformat(raw)
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except Exception as exc:
                logger.warning("parse_iso format %r failed for %r: %s", fmt, raw, exc)
                continue
    return None


def _iso_utc(dt: datetime | None) -> str:
    if dt is None:
        return ""
    try:
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return str(dt)


def _normalize_direction(raw_signal: Any) -> str:
    s = str(raw_signal or "").upper()
    if s in {"BUY", "LONG"}:
        return "LONG"
    if s in {"SELL", "SHORT"}:
        return "SHORT"
    return s or "HOLD"


def _fetch_trade_report_rows(limit: int = 200, ticker: str | None = None) -> list[dict[str, Any]]:
    db_path = _db_sqlite_path()
    if not db_path.exists():
        return []

    out: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            table_names = {str(r[0]) for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "signals" not in table_names:
                return []

            cols = {str(r[1]) for r in cur.execute("PRAGMA table_info(signals)").fetchall()}

            def coalesce_expr(candidates: list[str], default_sql: str) -> str:
                available = [c for c in candidates if c in cols]
                if not available:
                    return default_sql
                joined = ", ".join(available)
                return f"COALESCE({joined}, {default_sql})"

            signal_id_expr = (
                "COALESCE(signal_id, 'BTC-' || printf('%03d', rowid))"
                if "signal_id" in cols
                else "'BTC-' || printf('%03d', rowid)"
            )
            ticker_expr = coalesce_expr(["ticker", "asset", "symbol"], "'BTCUSDT'")
            signal_expr = coalesce_expr(["signal", "requested_signal"], "'HOLD'")
            confidence_expr = coalesce_expr(["confidence"], "0.0")
            entry_expr = coalesce_expr(["entry_price", "entry"], "0.0")
            exit_expr = coalesce_expr(["exit_price", "close_price"], "0.0")
            sl_expr = coalesce_expr(["sl", "stop_loss"], "0.0")
            tp1_expr = coalesce_expr(["tp1", "take_profit"], "0.0")
            outcome_expr = coalesce_expr(["outcome", "result"], "'OPEN'")
            pnl_expr = coalesce_expr(["pnl_pct"], "0.0")
            mfe_expr = coalesce_expr(["mfe_pct"], "0.0")
            mae_expr = coalesce_expr(["mae_pct"], "0.0")
            duration_expr = coalesce_expr(["duration_seconds"], "0")
            ts_expr = coalesce_expr(["timestamp", "time", "as_of_utc"], "datetime('now')")
            size_expr = coalesce_expr(["size_multiplier"], "1.0")
            rr_expr = coalesce_expr(["rr_ratio", "risk_reward"], "0.0")

            where_sql = ""
            params: list[Any] = []
            ticker_norm = str(ticker or "").upper().strip()
            if ticker_norm:
                where_sql = f"WHERE UPPER({ticker_expr}) = ?"
                params.append(ticker_norm)
            params.append(int(limit))

            cur.execute(
                f"""
                SELECT
                    rowid AS rowid,
                    {signal_id_expr} AS signal_id,
                    {ticker_expr} AS ticker,
                    {signal_expr} AS signal,
                    {confidence_expr} AS confidence,
                    {entry_expr} AS entry_price,
                    {exit_expr} AS exit_price,
                    {sl_expr} AS sl,
                    {tp1_expr} AS tp1,
                    {outcome_expr} AS outcome,
                    {pnl_expr} AS pnl_pct,
                    {mfe_expr} AS mfe_pct,
                    {mae_expr} AS mae_pct,
                    {duration_expr} AS duration_seconds,
                    {ts_expr} AS entry_time,
                    {size_expr} AS size_multiplier,
                    {rr_expr} AS rr_ratio
                FROM signals
                {where_sql}
                ORDER BY datetime({ts_expr}) DESC, rowid DESC
                LIMIT ?
                """,
                tuple(params),
            )

            for row in cur.fetchall():
                r = dict(row)
                direction = _normalize_direction(r.get("signal"))
                entry_price = _safe_float(r.get("entry_price"), 0.0)
                exit_price = _safe_float(r.get("exit_price"), 0.0)
                sl = _safe_float(r.get("sl"), 0.0)
                tp1 = _safe_float(r.get("tp1"), 0.0)
                pnl_pct = _safe_float(r.get("pnl_pct"), 0.0)
                mfe_pct = max(0.0, _safe_float(r.get("mfe_pct"), 0.0))
                mae_pct = max(0.0, _safe_float(r.get("mae_pct"), 0.0))
                duration_seconds = int(_safe_float(r.get("duration_seconds"), 0.0))
                outcome = str(r.get("outcome") or "OPEN").upper()
                entry_time_raw = str(r.get("entry_time") or "")
                entry_dt = _parse_utc(entry_time_raw)
                if outcome == "OPEN":
                    exit_time = ""
                elif duration_seconds > 0 and entry_dt is not None:
                    exit_time = _iso_utc(entry_dt + timedelta(seconds=int(duration_seconds)))
                else:
                    exit_time = entry_time_raw

                risk_pct = 0.0
                if entry_price > 0 and sl > 0:
                    risk_pct = abs((entry_price - sl) / entry_price) * 100.0
                if risk_pct > 0:
                    rr_achieved = pnl_pct / risk_pct
                else:
                    rr_achieved = _safe_float(r.get("rr_ratio"), 0.0)
                    if rr_achieved == 0.0 and outcome.startswith("TP"):
                        rr_achieved = 1.0
                    elif rr_achieved == 0.0 and outcome == "SL":
                        rr_achieved = -1.0

                out.append(
                    {
                        "signal_id": str(r.get("signal_id") or f"BTC-{int(r.get('rowid') or 0):03d}"),
                        "ticker": str(r.get("ticker") or "BTCUSDT").upper(),
                        "direction": direction,
                        "entry_time": entry_time_raw,
                        "exit_time": exit_time,
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "sl": round(sl, 4),
                        "tp1": round(tp1, 4),
                        "confidence": round(_safe_float(r.get("confidence"), 0.0), 2),
                        "outcome": outcome,
                        "pnl_pct": round(pnl_pct, 4),
                        "mfe_pct": round(mfe_pct, 4),
                        "mae_pct": round(mae_pct, 4),
                        "duration_seconds": duration_seconds,
                        "rr_achieved": round(rr_achieved, 4),
                        "size_multiplier": round(_safe_float(r.get("size_multiplier"), 1.0), 4),
                    }
                )
    except Exception as exc:
        logger.warning("Failed to build trade report rows: %s", exc)
    return out


def _dashboard_stale_redis_key(name: str) -> str:
    return f"{_DASHBOARD_STALE_CACHE_PREFIX}{name}"


def _proxy_cache_store_stale(
    name: str,
    payload: dict[str, Any],
    *,
    request_id: str,
) -> None:
    data = dict(payload)
    _btc_proxy_cache[name] = data
    try:
        r.setex(_dashboard_stale_redis_key(name), _DASHBOARD_STALE_TTL_SEC, json.dumps(data))
    except Exception as exc:
        logger.warning(
            "dashboard stale cache Redis SET failed name=%r request_id=%s: %s",
            name,
            request_id,
            exc,
        )


def _proxy_cache_get(name: str, *, request_id: str) -> dict[str, Any] | None:
    key = _dashboard_stale_redis_key(name)
    try:
        raw = r.get(key)
        if raw:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out = dict(obj)
                out["stale"] = True
                return out
    except Exception as exc:
        logger.warning(
            "dashboard stale cache Redis GET failed name=%r request_id=%s: %s",
            name,
            request_id,
            exc,
        )
    cached = _btc_proxy_cache.get(name)
    if not cached:
        return None
    payload = dict(cached)
    payload["stale"] = True
    return payload


def _as_key_list(value: Union[str, List[str]]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if str(v).strip()]


def _gate_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_confidence_pct(conf: Any) -> float | None:
    f = _gate_float(conf)
    if f is None:
        return None
    if 0.0 <= f <= 1.0:
        f *= 100.0
    return max(0.0, min(100.0, f))


def _decision_engine_cost_blocked(de: dict[str, Any]) -> bool:
    if not isinstance(de, dict):
        return False
    blockers = de.get("blockers")
    if not isinstance(blockers, list):
        return False
    for b in blockers:
        low = str(b).lower()
        if "cost gate" in low or "alpha barrier" in low:
            return True
    return False


def _apply_dashboard_signal_pipeline_fields(out: dict[str, Any]) -> None:
    """
    Populate confidence / cost gate fields expected by Decision Pipeline UI
    (validation_checks.confidence_ok, cost_ok; confidence; adjusted_confidence_threshold;
    net_alpha_score_raw). Sources: top-level btc:signal, decision_engine, meta_output,
    meta_decision / meta_decision_details, adaptive_learning.
    """
    de = out.get("decision_engine") if isinstance(out.get("decision_engine"), dict) else {}
    db_top = out.get("decision_breakdown") if isinstance(out.get("decision_breakdown"), dict) else {}
    meta_dec = out.get("meta_decision") if isinstance(out.get("meta_decision"), dict) else {}
    meta_out = out.get("meta_output") if isinstance(out.get("meta_output"), dict) else {}
    meta_details = out.get("meta_decision_details") if isinstance(out.get("meta_decision_details"), dict) else {}
    adaptive = out.get("adaptive_learning") if isinstance(out.get("adaptive_learning"), dict) else {}
    de_break = de.get("decision_breakdown") if isinstance(de.get("decision_breakdown"), dict) else {}
    db = db_top if db_top else de_break

    nc = _normalize_confidence_pct(out.get("confidence"))
    if nc is None:
        nc = _normalize_confidence_pct(meta_out.get("confidence"))
    if nc is None:
        nc = _normalize_confidence_pct(de.get("confidence"))
    if nc is None:
        nc = _normalize_confidence_pct(meta_dec.get("adjusted_confidence"))
    if nc is not None:
        out["confidence"] = nc

    th = _gate_float(out.get("adjusted_confidence_threshold"))
    if th is None:
        th = _gate_float(de.get("confidence_threshold"))
    if th is None:
        th = _gate_float(de.get("adjusted_confidence_threshold"))
    if th is None:
        th = _gate_float(meta_out.get("confidence_threshold"))
    if th is None:
        th = _gate_float(meta_dec.get("confidence_threshold"))
    if th is None:
        th = _gate_float(meta_dec.get("min_confidence"))
    if th is None:
        th = _gate_float(meta_details.get("confidence_threshold"))
    if th is None:
        th = _gate_float(adaptive.get("min_confidence"))
    if th is None:
        th = 55.0
    out["adjusted_confidence_threshold"] = max(0.0, min(100.0, float(th)))

    nar = _gate_float(out.get("net_alpha_score_raw"))
    if nar is None:
        nar = _gate_float(out.get("net_alpha_raw"))
    if nar is None:
        nar = _gate_float(out.get("net_alpha"))
    if nar is None:
        nar = _gate_float(out.get("net_alpha_score"))
    if nar is None:
        nar = _gate_float(meta_out.get("net_alpha"))
    if nar is None and isinstance(db, dict):
        nar = _gate_float(db.get("net_alpha"))
    if nar is not None:
        out["net_alpha_score_raw"] = nar

    vc_pre = out.get("validation_checks")
    vc: dict[str, Any] = dict(vc_pre) if isinstance(vc_pre, dict) else {}

    top_checks = out.get("checks")
    if isinstance(top_checks, dict):
        for k in (
            "confidence_ok",
            "cost_ok",
            "data_quality_ok",
            "mtf_ok",
            "etf_ok",
            "fear_greed_ok",
            "liquidation_ok",
            "oi_ok",
            "sl_distance_ok",
        ):
            if k in top_checks and k not in vc:
                vc[k] = bool(top_checks[k])

    ve = out.get("validation_engine") if isinstance(out.get("validation_engine"), dict) else {}
    ve_checks = ve.get("checks") if isinstance(ve.get("checks"), dict) else {}
    if ve_checks:
        if "calibration_ok" in ve_checks and "calibration_ok" not in vc:
            vc["calibration_ok"] = bool(ve_checks.get("calibration_ok"))
        if "alpha_barrier" in ve_checks and "alpha_barrier_ok" not in vc:
            vc["alpha_barrier_ok"] = bool(ve_checks.get("alpha_barrier"))

    if isinstance(meta_dec, dict):
        for k in ("confidence_ok", "cost_ok"):
            if k in meta_dec and k not in vc:
                vc[k] = bool(meta_dec[k])

    conf_disp = _gate_float(out.get("confidence")) or 0.0
    th_disp = float(out["adjusted_confidence_threshold"])
    if "confidence_ok" not in vc:
        vc["confidence_ok"] = bool(conf_disp >= th_disp - 1e-9)

    cost_blocked = _decision_engine_cost_blocked(de)
    if "cost_ok" not in vc:
        nar_v = out.get("net_alpha_score_raw")
        if nar_v is not None:
            vc["cost_ok"] = bool(float(nar_v) >= 0.0) and not cost_blocked
        else:
            vc["cost_ok"] = not cost_blocked

    out["validation_checks"] = vc

    if out.get("requested_signal") is None and isinstance(de, dict):
        rs = de.get("requested_signal") or de.get("raw_signal")
        if rs is not None:
            out["requested_signal"] = str(rs).upper()


def _normalize_dashboard_signal_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Map btc_intelligence /signal payload to fields the vanilla terminal expects."""
    if not isinstance(raw, dict):
        return {}
    out = dict(raw)
    sig = str(out.get("signal", "HOLD")).upper()
    val = bool(out.get("validated", False))
    out["validated_signal"] = sig if val and sig in ("LONG", "SHORT") else "HOLD"
    tp = out.get("take_profit")
    if isinstance(tp, dict):
        out.setdefault("tp1", tp.get("TP1"))
        out.setdefault("tp2", tp.get("TP2"))
        out.setdefault("tp3", tp.get("TP3"))
    ez = out.get("entry_zone")
    if isinstance(ez, (list, tuple)) and len(ez) >= 2:
        try:
            lo, hi = float(ez[0]), float(ez[1])
            out["entry_zone_low"] = lo
            out["entry_zone_high"] = hi
            if not out.get("entry_price"):
                out["entry_price"] = round((lo + hi) / 2.0, 2)
        except (TypeError, ValueError):
            pass
    if out.get("stop_loss") is not None and out.get("sl") is None:
        out["sl"] = out["stop_loss"]
    if out.get("net_alpha_score") is None and out.get("net_alpha") is not None:
        try:
            out["net_alpha_score"] = float(out["net_alpha"])
        except (TypeError, ValueError):
            out["net_alpha_score"] = 0.0
    if not out.get("regime"):
        mr = out.get("market_regime")
        if mr:
            out["regime"] = str(mr).replace("_", " ").upper()
    macro = out.get("macro") if isinstance(out.get("macro"), dict) else {}
    if out.get("fear_greed") is None and macro.get("fear_greed") is not None:
        out["fear_greed"] = macro.get("fear_greed")
    mtf = out.get("mtf")
    if isinstance(mtf, dict) and not isinstance(out.get("mtf_bias"), dict):
        b4 = mtf.get("4h") or mtf.get("bias_4h")
        b1d = mtf.get("1d") or mtf.get("bias_1d")
        if b4 or b1d:
            out["mtf_bias"] = {
                "bias_4h": str(b4 or "NEUTRAL").upper(),
                "bias_1d": str(b1d or "NEUTRAL").upper(),
            }
    if isinstance(out.get("derivatives"), dict) and out.get("funding_rate_pct") is None:
        d = out["derivatives"]
        if d.get("funding_rate") is not None:
            try:
                out["funding_rate_pct"] = float(d["funding_rate"]) * 100.0
            except (TypeError, ValueError):
                pass
    if isinstance(out.get("kelly_sizing"), dict):
        ks = out["kelly_sizing"]
        out["position_sizing"] = {
            "position_size_pct": ks.get("position_pct"),
            "position_size_usd": ks.get("position_size_usd"),
            "raw_kelly": ks.get("raw_kelly"),
            "p": ks.get("p"),
            "b": ks.get("b"),
            "method": ks.get("method", "bayesian_prior"),
            "bucket_key": ks.get("bucket_key"),
            "trades_in_bucket": ks.get("trades_in_bucket"),
        }
    if isinstance(out.get("execution_plan"), dict):
        ep = out["execution_plan"]
        tr = ep.get("tail_risk_sizing")
        if isinstance(tr, dict):
            out.setdefault("tail_risk_sizing", tr)
        if ep.get("expected_rr") is not None and out.get("risk_reward") is None:
            out["risk_reward"] = ep.get("expected_rr")
    # --- Signal Quality gauge mapping ---
    if out.get("signal_strength") is None:
        _sq = None
        for _fld in ("quality_score", "alpha_score", "net_alpha_score", "net_alpha"):
            _v = out.get(_fld)
            if _v is not None:
                try:
                    _f = float(_v)
                    _sq = abs(_f) * 100.0 if abs(_f) <= 1.0 else abs(_f)
                    break
                except (TypeError, ValueError):
                    pass
        if _sq is None:
            _agg = out.get("signal_aggregation")
            if isinstance(_agg, dict):
                for _k in ("raw_score", "confidence"):
                    _av = _agg.get(_k)
                    if _av is not None:
                        try:
                            _f = float(_av)
                            _sq = abs(_f) * 100.0 if abs(_f) <= 1.0 else abs(_f)
                            break
                        except (TypeError, ValueError):
                            pass
        if _sq is None:
            for _fld in ("meta_confidence", "raw_confidence"):
                _v = out.get(_fld)
                if _v is not None:
                    try:
                        _f = float(_v)
                        _sq = _f * 100.0 if _f <= 1.0 else _f
                        break
                    except (TypeError, ValueError):
                        pass
        if _sq is not None:
            out["signal_strength"] = max(0.0, min(100.0, round(_sq, 1)))
    if out.get("quality_score") is None and out.get("signal_strength") is not None:
        out["quality_score"] = out["signal_strength"]
    # --- Order Flow fields ---
    _of = out.get("orderflow") or out.get("order_flow") or {}
    if isinstance(_of, dict):
        if out.get("flow_decision") is None:
            out["flow_decision"] = _of.get("decision") or _of.get("decision_state")
        if out.get("obi") is None:
            out["obi"] = _of.get("obi") or _of.get("obi_imbalance")
        if out.get("flow_strength") is None:
            out["flow_strength"] = _of.get("flow_strength") or _of.get("strength")
    _apply_dashboard_signal_pipeline_fields(out)
    return out


def _candle_rows_to_chart_data(rows: List[Any]) -> List[dict[str, Any]]:
    data: List[dict[str, Any]] = []
    if not isinstance(rows, list):
        return data
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            ot = int(r.get("open_time", 0) or 0)
            t_sec = ot // 1000 if ot > 1_000_000_000_000 else ot
            data.append(
                {
                    "time": int(t_sec),
                    "open": round(float(r.get("open", 0.0)), 2),
                    "high": round(float(r.get("high", 0.0)), 2),
                    "low": round(float(r.get("low", 0.0)), 2),
                    "close": round(float(r.get("close", 0.0)), 2),
                    "volume": round(float(r.get("volume", 0.0)), 6),
                }
            )
        except (TypeError, ValueError):
            continue
    data.sort(key=lambda x: int(x["time"]))
    return data


def _cluster_price_levels(levels: List[float], band_pct: float = 0.003) -> List[float]:
    """Merge levels within band_pct (e.g. 0.003 = 0.3%) of cluster mean; return cluster averages."""
    vals = sorted(float(x) for x in levels if x is not None and math.isfinite(float(x)))
    if not vals:
        return []
    clusters: List[List[float]] = []
    for x in vals:
        if not clusters:
            clusters.append([x])
            continue
        m = sum(clusters[-1]) / len(clusters[-1])
        if m > 0 and abs(x - m) / m <= band_pct:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [round(sum(c) / len(c), 2) for c in clusters]


def _nearest_sr_to_price(
    current: float,
    supports: List[float],
    resistances: List[float],
    n: int = 3,
) -> tuple[List[float], List[float]]:
    """Top n supports and resistances nearest to current (prefer below / above)."""

    def pick_side(levels: List[float], prefer_below: bool) -> List[float]:
        if not levels:
            return []
        if prefer_below:
            below = sorted([p for p in levels if p < current], key=lambda p: current - p)
            out = below[:n]
            if len(out) < n:
                rest = sorted([p for p in levels if p >= current], key=lambda p: abs(p - current))
                for p in rest:
                    if len(out) >= n:
                        break
                    if p not in out:
                        out.append(p)
            return [round(float(x), 2) for x in out[:n]]
        above = sorted([p for p in levels if p > current], key=lambda p: p - current)
        out = above[:n]
        if len(out) < n:
            rest = sorted([p for p in levels if p <= current], key=lambda p: abs(p - current))
            for p in rest:
                if len(out) >= n:
                    break
                if p not in out:
                    out.append(p)
        return [round(float(x), 2) for x in out[:n]]

    return pick_side(supports, True), pick_side(resistances, False)


def _levels_within_pct_of(current: float, levels: List[float], pct: float) -> List[float]:
    if current <= 0 or not levels:
        return []
    return [p for p in levels if abs(float(p) - current) / current <= pct]


def _compute_sr_from_ohlc(rows: List[dict[str, Any]]) -> dict[str, Any]:
    """Pivot S/R from highs/lows, cluster (0.8%), nearest 2+2 within 3% of last close."""
    if len(rows) < 3:
        cur = float(rows[-1].get("close", 0.0)) if rows else 0.0
        return {"support": [], "resistance": [], "current": round(cur, 2)}
    highs = [float(rows[i]["high"]) for i in range(len(rows))]
    lows = [float(rows[i]["low"]) for i in range(len(rows))]
    resistance_raw: List[float] = []
    support_raw: List[float] = []
    for i in range(1, len(rows) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            resistance_raw.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            support_raw.append(lows[i])
    current = float(rows[-1]["close"])
    resistance = _cluster_price_levels(resistance_raw, band_pct=0.008)
    support = _cluster_price_levels(support_raw, band_pct=0.008)
    support = _levels_within_pct_of(current, support, 0.03)
    resistance = _levels_within_pct_of(current, resistance, 0.03)
    sup2, res2 = _nearest_sr_to_price(current, support, resistance, 2)
    return {
        "support": sup2,
        "resistance": res2,
        "current": round(current, 2),
    }


def _compute_liquidity_zones_payload(data: List[dict[str, Any]]) -> dict[str, Any]:
    """High-volume 15m nodes (vs 20-bar vol MA at 2.0×) + long-wick sweep candidates."""
    n = len(data)
    if n < 21:
        return {"zones": [], "sweep_levels": []}

    vol_mult = 2.0
    volumes = [float(d.get("volume", 0.0) or 0.0) for d in data]
    zones: List[dict[str, Any]] = []
    for i in range(20, n):
        window = volumes[i - 20 : i]
        ma = sum(window) / 20.0
        if ma <= 0:
            continue
        v = volumes[i]
        if v > vol_mult * ma:
            ratio = v / ma
            strength = min(1.0, max(0.0, (ratio - vol_mult) / 1.0))
            hi = round(float(data[i]["high"]), 2)
            lo = round(float(data[i]["low"]), 2)
            st = round(strength, 3)
            zones.append({"price": hi, "type": "high_vol", "strength": st})
            zones.append({"price": lo, "type": "high_vol", "strength": st})

    sweep_raw: List[float] = []
    wick_ratios: List[float] = []
    for d in data:
        o = float(d["open"])
        h = float(d["high"])
        l_ = float(d["low"])
        c = float(d["close"])
        tr = h - l_
        if tr <= 1e-12:
            continue
        body = abs(c - o)
        uw = h - max(o, c)
        lw = min(o, c) - l_
        wick_ratios.extend([uw / tr, lw / tr])

    wick_ratios.sort()
    thr_idx = int(len(wick_ratios) * 0.80) if wick_ratios else 0
    wick_thr = wick_ratios[thr_idx] if wick_ratios else 0.55

    for d in data:
        o = float(d["open"])
        h = float(d["high"])
        l_ = float(d["low"])
        c = float(d["close"])
        tr = h - l_
        if tr <= 1e-12:
            continue
        body = abs(c - o)
        uw = h - max(o, c)
        lw = min(o, c) - l_
        ur = uw / tr
        lr = lw / tr
        long_body = max(body, tr * 0.08)
        if ur >= wick_thr and uw >= 1.2 * long_body:
            sweep_raw.append(round(h, 2))
        if lr >= wick_thr and lw >= 1.2 * long_body:
            sweep_raw.append(round(l_, 2))

    current = float(data[-1]["close"])
    if current > 0:
        sweep_raw = [p for p in sweep_raw if abs(float(p) - current) / current <= 0.02]
    sweep_clustered = _cluster_price_levels(sweep_raw, band_pct=0.005)
    sweep_levels = sorted(sweep_clustered)

    zones.sort(key=lambda z: float(z.get("strength", 0.0) or 0.0), reverse=True)
    zones = zones[:4]

    return {"zones": zones, "sweep_levels": sweep_levels}


async def _btc_fetch_json(urls: List[str]) -> Any:
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            for url in urls:
                for _ in range(3):
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        return resp.json()
                    except Exception as exc:
                        logger.warning("_btc_fetch_json attempt failed for %s: %s", url, exc)
                        await asyncio.sleep(0.75)
    except Exception as exc:
        logger.warning("_btc_fetch_json outer failure: %s", exc)
    return None


# Intervals not streamed into the live buffer — fetch full candles from Binance via market/history.
_BTC_CHART_REST_HISTORY_INTERVALS = frozenset({"3m", "30m", "2h", "6h", "12h", "3d"})


async def _btc_history_payload_from_upstream(interval: str, limit: int) -> dict[str, Any]:
    """Chart: live buffer klines for streamed TFs; REST klines (history route) for the rest; 1d uses daily history."""
    interval = interval if interval in INTERVAL_TO_MS else "15m"
    limit = int(max(50, min(limit, 5000)))
    if interval == "1d":
        url = f"{BTC_INTEL_BASE}/market/history?timeframe=1d&limit={limit}"
    elif interval in _BTC_CHART_REST_HISTORY_INTERVALS:
        url = f"{BTC_INTEL_BASE}/market/history?timeframe={interval}&limit={limit}"
    else:
        url = f"{BTC_INTEL_BASE}/market/klines?timeframe={interval}&limit={limit}"
    payload = await _btc_fetch_json([url])
    rows: List[Any] = []
    if isinstance(payload, dict):
        rows = list(payload.get("rows") or [])
    data = _candle_rows_to_chart_data(rows)
    if not data:
        return {"asset": "BTCUSDT", "interval": interval, "data": [], "error": "No history", "points": 0}
    return {
        "asset": "BTCUSDT",
        "interval": interval,
        "points": len(data),
        "start_utc": datetime.utcfromtimestamp(int(data[0]["time"])).isoformat() + "Z",
        "end_utc": datetime.utcfromtimestamp(int(data[-1]["time"])).isoformat() + "Z",
        "data": data,
    }


def _extract_probability_payload(payload: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return block
    if any(k in payload for k in ("up_prob", "down_prob", "sideways_prob")):
        block = dict(payload)
    elif isinstance(payload.get("probability"), dict):
        block = dict(payload.get("probability", {}))

    if not block:
        return {}

    up_prob = float(block.get("up_prob", 0.0) or 0.0)
    down_prob = float(block.get("down_prob", 0.0) or 0.0)
    sideways_prob = float(block.get("sideways_prob", max(0.0, 100.0 - up_prob - down_prob)) or 0.0)
    dominant_state = str(block.get("dominant_state") or block.get("dominant") or "SIDEWAYS").upper()

    calibration_score = block.get("calibration_score")
    if calibration_score is None:
        platt_prob = block.get("platt_probability")
        if platt_prob is not None:
            calibration_score = float(platt_prob) * 100.0
        else:
            calibration_score = float(max(up_prob, down_prob, sideways_prob))

    block["up_prob"] = round(up_prob, 2)
    block["down_prob"] = round(down_prob, 2)
    block["sideways_prob"] = round(sideways_prob, 2)
    block["dominant_state"] = dominant_state
    block["dominant"] = dominant_state
    block["calibration_score"] = round(float(calibration_score), 2)
    return block


<<<<<<< HEAD
=======
def _btc_redis_payload_is_unusable(payload: dict[str, Any]) -> bool:
    """
    True when a Redis JSON blob is only an error/placeholder so we should try
    the next key (e.g. btc:signal) or upstream instead of treating it as data.
    """
    err = payload.get("error")
    if err is None or err == "":
        return False
    if not isinstance(err, str):
        return False
    if isinstance(payload.get("decision_engine"), dict) and payload["decision_engine"]:
        return False
    if payload.get("signal") is not None:
        return False
    if any(k in payload for k in ("up_prob", "down_prob", "probability")):
        return False
    return True


>>>>>>> origin/main
def _redis_get_json(key: str) -> dict[str, Any] | None:
    try:
        raw = r.get(key)
        if not raw:
            return None
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
<<<<<<< HEAD
    except Exception as exc:
        logger.warning("_redis_get_json failed for key %r: %s", key, exc)
=======
    except Exception:
>>>>>>> origin/main
        return None


def _minimal_decision_intelligence_from_signal(
    sig: dict[str, Any],
    *,
    source: str = "redis_signal_fallback",
) -> dict[str, Any]:
    """
    Build a decision-intelligence-shaped dict from btc:signal when upstream is down.
    Marked stale/degraded for UI.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    signal = str(sig.get("signal", "HOLD")).upper()
    conf = float(sig.get("confidence", 0.0) or 0.0)
    prob_in = sig.get("probability") if isinstance(sig.get("probability"), dict) else {}
    up = float(prob_in.get("up_prob", 0.0) or 0.0)
    down = float(prob_in.get("down_prob", 0.0) or 0.0)
    sw = float(prob_in.get("sideways_prob", max(0.0, 100.0 - up - down)) or 0.0)
    dom = str(prob_in.get("dominant") or prob_in.get("dominant_state") or "SIDEWAYS").upper()
    if signal == "LONG":
        dom = "LONG"
    elif signal == "SHORT":
        dom = "SHORT"

    de = (
        sig.get("decision_engine")
        if isinstance(sig.get("decision_engine"), dict)
        else {
            "final_score": conf,
            "decision": signal if signal in ("LONG", "SHORT") else "HOLD",
        }
    )
    out: dict[str, Any] = {
        "decision_engine": dict(de),
        "decision_breakdown": dict(sig.get("decision_breakdown", {}))
        if isinstance(sig.get("decision_breakdown"), dict)
        else {},
        "probability": {
            "up_prob": round(up, 2),
            "down_prob": round(down, 2),
            "sideways_prob": round(sw, 2),
            "dominant_state": dom,
            "dominant": dom,
            "calibration_score": round(conf, 2),
        },
        "execution_plan": dict(sig.get("execution_plan", {}))
        if isinstance(sig.get("execution_plan"), dict)
        else {},
        "trade_verdict": dict(sig.get("trade_verdict", {}))
        if isinstance(sig.get("trade_verdict"), dict)
        else {},
        "meta_decision": dict(sig.get("meta_decision", {}))
        if isinstance(sig.get("meta_decision"), dict)
        else {},
        "meta_labeling": dict(sig.get("meta_labeling", {}))
        if isinstance(sig.get("meta_labeling"), dict)
        else {},
        "as_of_utc": sig.get("as_of_utc") or sig.get("timestamp") or now,
        "stale": True,
        "degraded": True,
        "source": source,
    }
<<<<<<< HEAD
    for k in ("regime_state_probs", "anti_crowding", "hibernated_factors", "tail_risk_sizing", "validation_engine"):
        if k in sig and sig[k] is not None:
            out[k] = sig[k]
    return out


def _enrich_decision_intel(out: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Lift institutional fields for the vanilla terminal + proxy consumers."""
    if not isinstance(out, dict):
        return out
    src = payload if isinstance(payload, dict) else {}
    for k in ("regime_state_probs", "anti_crowding", "hibernated_factors"):
        if k in src and src[k] is not None:
            out[k] = src[k]
    ep = out.get("execution_plan")
    if isinstance(ep, dict) and isinstance(ep.get("tail_risk_sizing"), dict):
        out["tail_risk_sizing"] = ep["tail_risk_sizing"]

    de = out.get("decision_engine") if isinstance(out.get("decision_engine"), dict) else {}
    mo = src.get("meta_output") if isinstance(src.get("meta_output"), dict) else {}
    md = src.get("meta_decision") if isinstance(src.get("meta_decision"), dict) else {}
    if not mo and isinstance(out.get("meta_output"), dict):
        mo = out["meta_output"]
    if not md and isinstance(out.get("meta_decision"), dict):
        md = out["meta_decision"]

    conf_src = None
    if mo.get("confidence") is not None:
        conf_src = mo.get("confidence")
    elif de.get("confidence") is not None:
        conf_src = de.get("confidence")
    elif src.get("confidence") is not None:
        conf_src = src.get("confidence")
    nc = _normalize_confidence_pct(conf_src)
    if nc is not None:
        out["confidence"] = nc

    adj_c = _gate_float(md.get("adjusted_confidence"))
    if adj_c is not None:
        out["adjusted_confidence"] = max(0.0, min(100.0, adj_c))
        if out.get("confidence") is None:
            out["confidence"] = out["adjusted_confidence"]

    th = _gate_float(src.get("adjusted_confidence_threshold"))
    if th is None:
        th = _gate_float(de.get("confidence_threshold"))
    if th is None:
        th = _gate_float(de.get("adjusted_confidence_threshold"))
    if th is None:
        th = _gate_float(mo.get("confidence_threshold"))
    if th is None:
        th = _gate_float(md.get("confidence_threshold"))
    if th is not None:
        out["adjusted_confidence_threshold"] = max(0.0, min(100.0, float(th)))

    nar = _gate_float(src.get("net_alpha_score_raw"))
    if nar is None:
        nar = _gate_float(src.get("net_alpha"))
    if nar is not None:
        out["net_alpha_score_raw"] = nar

=======
>>>>>>> origin/main
    return out


def _normalize_decision_intelligence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    if isinstance(payload.get("decision_engine"), dict):
        out = dict(payload)
        out["decision_breakdown"] = (
            dict(out.get("decision_breakdown", {}))
            if isinstance(out.get("decision_breakdown"), dict)
            else {}
        )
        out["probability"] = _extract_probability_payload(out)
        out["execution_plan"] = dict(out.get("execution_plan", {})) if isinstance(out.get("execution_plan"), dict) else {}
        return _enrich_decision_intel(out, payload)

    decision_engine = dict(payload)
    out: dict[str, Any] = {
        "decision_engine": decision_engine,
        "decision_breakdown": (
            dict(payload.get("decision_breakdown", {}))
            if isinstance(payload.get("decision_breakdown"), dict)
            else {}
        ),
        "factor_contributions": list(payload.get("factor_contributions", []))
        if isinstance(payload.get("factor_contributions"), list)
        else [],
        "trade_verdict": dict(payload.get("trade_verdict", {}))
        if isinstance(payload.get("trade_verdict"), dict)
        else {},
        "meta_decision": dict(payload.get("meta_decision", {}))
        if isinstance(payload.get("meta_decision"), dict)
        else {},
        "meta_labeling": dict(payload.get("meta_labeling", {}))
        if isinstance(payload.get("meta_labeling"), dict)
        else {},
        "validation_engine": dict(payload.get("validation_engine", {}))
        if isinstance(payload.get("validation_engine"), dict)
        else {},
        "strategy_selection": dict(payload.get("strategy_selection", {}))
        if isinstance(payload.get("strategy_selection"), dict)
        else {},
        "data_drift": dict(payload.get("data_drift", {}))
        if isinstance(payload.get("data_drift"), dict)
        else {},
        "meta_output": dict(payload.get("meta_output", {}))
        if isinstance(payload.get("meta_output"), dict)
        else {},
        "adaptive_learning": dict(payload.get("adaptive_learning", {}))
        if isinstance(payload.get("adaptive_learning"), dict)
        else {},
        "as_of_utc": payload.get("as_of_utc"),
    }
    out["probability"] = _extract_probability_payload(payload)
    out["execution_plan"] = dict(payload.get("execution_plan", {})) if isinstance(payload.get("execution_plan"), dict) else {}
    return _enrich_decision_intel(out, payload)


def _btc_redis_payload_is_unusable(payload: Any) -> bool:
    """
    True when a Redis JSON blob should be skipped for the BTC proxy fast-path:
    not a dict, missing as_of_utc/timestamp/as_of, unparseable time, or older than 60 seconds.
    """
    if not isinstance(payload, dict):
        return True
    raw: Any = None
    for key in ("as_of_utc", "timestamp", "as_of"):
        v = payload.get(key)
        if v is not None and str(v).strip() != "":
            raw = v
            break
    if raw is None:
        return True
    try:
        if isinstance(raw, (int, float)):
            ts = float(raw)
            if ts > 1e12:
                ts = ts / 1000.0
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            s = str(raw).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_sec = (now - dt.astimezone(timezone.utc)).total_seconds()
        if age_sec > 60.0:
            return True
    except Exception as exc:
        logger.warning("Redis payload timestamp unusable (parse): %s", exc)
        return True
    return False


async def _btc_proxy_payload(
    name: str,
    redis_key: Union[str, List[str]],
    upstream_url: Union[str, List[str]],
) -> dict[str, Any]:
    request_id = str(uuid4())
    redis_keys = _as_key_list(redis_key)
    upstream_urls = _as_key_list(upstream_url)
    proxy_headers = {"X-Request-ID": request_id}

    # 1) Redis fast-path
    for btc_key in redis_keys:
        try:
            raw = r.get(btc_key)
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    if _btc_redis_payload_is_unusable(payload):
                        continue
                    payload["stale"] = False
                    _proxy_cache_store_stale(name, payload, request_id=request_id)
                    return payload
        except Exception as exc:
            logger.warning(
                "_btc_proxy_payload redis key %r read failed request_id=%s: %s",
                btc_key,
                request_id,
                exc,
            )
            continue

<<<<<<< HEAD
    # 1b) SQLite checkpoint mirror (1h TTL); OK if older than 60s — intentional degraded fallback
=======
    # 2) Upstream fallback (retry each URL attempt up to 3 times, 1s between tries)
>>>>>>> origin/main
    try:
        raw_p = r.get("btc:signal:persistent")
        if raw_p:
            payload = json.loads(raw_p)
            if isinstance(payload, dict):
                payload["stale"] = True
                payload["degraded"] = True
                payload["source"] = "sqlite_checkpoint"
                _proxy_cache_store_stale(name, payload, request_id=request_id)
                return payload
    except Exception as exc:
        logger.warning(
            "_btc_proxy_payload redis key 'btc:signal:persistent' read failed request_id=%s: %s",
            request_id,
            exc,
        )

    # 2) Upstream fallback — short timeouts so the dashboard browser fetch (15–60s) does not abort
    #    before we can return Redis/stale data.
    try:
        _px_timeout = httpx.Timeout(4.0, connect=2.0)
        async with httpx.AsyncClient(timeout=_px_timeout) as client:
            for url in upstream_urls:
                last_exc: Exception | None = None
<<<<<<< HEAD
                for attempt in range(2):
                    try:
                        resp = await client.get(url, headers=proxy_headers)
=======
                for attempt in range(3):
                    try:
                        resp = await client.get(url)
>>>>>>> origin/main
                        resp.raise_for_status()
                        payload = resp.json()
                        if isinstance(payload, dict):
                            payload["stale"] = False
<<<<<<< HEAD
                            _proxy_cache_store_stale(name, payload, request_id=request_id)
                            return payload
                        wrapped = {"data": payload, "stale": False}
                        _proxy_cache_store_stale(name, wrapped, request_id=request_id)
                        return wrapped
                    except Exception as exc:
                        last_exc = exc
                        if attempt < 1:
                            await asyncio.sleep(0.5)
                        logger.warning(
                            "BTC proxy GET %s attempt failed request_id=%s: %s",
                            url,
                            request_id,
                            exc,
                        )
                        continue
                if last_exc is not None:
                    logger.debug(
                        "BTC proxy upstream failed after retries %s request_id=%s: %s",
                        url,
                        request_id,
                        last_exc,
                    )
    except Exception as exc:
        logger.warning(
            "_btc_proxy_payload upstream client failure request_id=%s: %s",
            request_id,
            exc,
        )
=======
                            _btc_proxy_cache[name] = dict(payload)
                            return payload
                        wrapped = {"data": payload, "stale": False}
                        _btc_proxy_cache[name] = dict(wrapped)
                        return wrapped
                    except Exception as exc:
                        last_exc = exc
                        if attempt < 2:
                            await asyncio.sleep(1.0)
                        continue
                if last_exc is not None:
                    logger.debug(
                        "BTC proxy upstream failed after retries %s: %s",
                        url,
                        last_exc,
                    )
    except Exception:
        pass
>>>>>>> origin/main

    # 3) Last cached stale payload
    cached = _proxy_cache_get(name, request_id=request_id)
    if cached is not None:
        return cached
    return {"error": "upstream_unavailable", "stale": True}


@app.middleware("http")
async def response_cache_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    if request.method != "GET":
        return await call_next(request)

    ttl = _cache_ttl_for_path(request.url.path)
    if ttl <= 0:
        return await call_next(request)

    key = str(request.url)
    now = time.time()
    cached = _response_cache.get(key)
    if cached and (now - float(cached.get("ts", 0.0)) < ttl):
        return Response(
            content=bytes(cached.get("body", b"")),
            status_code=int(cached.get("status", 200)),
            media_type=str(cached.get("media_type", "application/json")),
        )

    response = await call_next(request)
    ctype = str(response.headers.get("content-type", ""))
    if response.status_code >= 400 or "application/json" not in ctype.lower():
        return response

    body = b""
    async for chunk in response.body_iterator:
        body += chunk

    _response_cache[key] = {
        "ts": now,
        "body": body,
        "status": int(response.status_code),
        "media_type": response.media_type or "application/json",
    }

    # Lightweight eviction.
    if len(_response_cache) > 500:
        cutoff = now - 120
        for k in [k for k, v in _response_cache.items() if float(v.get("ts", 0.0)) < cutoff]:
            _response_cache.pop(k, None)

    return Response(
        content=body,
        status_code=response.status_code,
        media_type=response.media_type,
        headers=dict(response.headers),
    )


async def broadcast(data: dict) -> None:
    """Broadcast a message to all connected WebSocket clients."""
    msg = json.dumps(data)
    dead = []
    for ws in _connected_ws:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connected_ws.remove(ws)


def push_broadcast_threadsafe(data: dict) -> None:
    """Schedule a WebSocket broadcast from non-async worker threads."""
    loop = _app_loop
    if loop is None or loop.is_closed():
        return
    asyncio.run_coroutine_threadsafe(broadcast(data), loop)  # type: ignore[arg-type]


@app.on_event("startup")
async def _capture_loop() -> None:
    global _app_loop
    _app_loop = asyncio.get_running_loop()


@app.on_event("shutdown")
async def _shutdown_live_runner() -> None:
    """Ensure scheduler/broker loop is stopped when the API server exits."""
    runner = _live_runner
    if runner is None:
        return
    try:
        runner.stop()
    except Exception as exc:
        logger.warning("Live runner shutdown hook failed: %s", exc)


# ------------------------------------------------------------------
# WebSocket endpoint
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token", "")
    if _auth_enabled() and not _verify_token(token):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    _connected_ws.append(websocket)
    try:
        while True:
            # Keep connection alive; actual messages are pushed by broadcast()
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _connected_ws:
            _connected_ws.remove(websocket)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _round2(val: object) -> float:
    """Round a numeric value to 2 decimal places, safe for Pyre2."""
    return round(float(val), 2)  # type: ignore[call-overload]


def _round4(val: object) -> float:
    """Round a numeric value to 4 decimal places, safe for Pyre2."""
    return round(float(val), 4)  # type: ignore[call-overload]


def _clean_factor_scores(raw: Any) -> Dict[str, float]:
    """Return factor scores as a plain float dict."""
    parsed: Any = raw if raw is not None else {}
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            parsed = {}
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in parsed.items():
        try:
            out[str(k)] = float(v)
        except Exception as exc:
            logger.warning("_clean_factor_scores skipped key %r: %s", k, exc)
            continue
    return out


def _attach_factor_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach normalized factor_scores + explicit F1/F2/F3/F5 fields."""
    clean = _clean_factor_scores(payload.get("factor_scores", {}))
    payload["factor_scores"] = clean
    payload["F1_momentum"] = float(clean.get("F1", 0.0))
    payload["F2_mean_rev"] = float(clean.get("F2", 0.0))
    payload["F3_volume"] = float(clean.get("F3", 0.0))
    payload["F5_volatility"] = float(clean.get("F5", 0.0))
    return payload


def _is_us_stock_ticker(ticker: str) -> bool:
    """
    Heuristic for US equities in this dashboard context:
    - not Indian suffix (.NS/.BO)
    - not index symbols (^...)
    - not FX pairs (...=X)
    - not crypto symbols used elsewhere
    """
    t = (ticker or "").upper().strip()
    if not t:
        return False
    if t.endswith(".NS") or t.endswith(".BO"):
        return False
    if t.startswith("^"):
        return False
    if "=" in t:
        return False
    if t in {"BTC", "ETH", "XAUUSD", "XAGUSD"}:
        return False
    return True


def _get_options_engine() -> OptionsEngine:
    global _options_engine
    if _options_engine is None:
        _options_engine = OptionsEngine()
    return _options_engine


def _calc_max_pain(calls: List[Dict[str, Any]], puts: List[Dict[str, Any]]) -> float:
    """Calculate max-pain strike from CE/PE OI buckets."""
    try:
        strikes = sorted(
            {
                float(c.get("strike", 0.0))
                for c in calls
                if isinstance(c.get("strike"), (int, float))
            }
            | {
                float(p.get("strike", 0.0))
                for p in puts
                if isinstance(p.get("strike"), (int, float))
            }
        )
        if not strikes:
            return 0.0

        min_pain = float("inf")
        max_pain_strike = strikes[len(strikes) // 2]
        for s in strikes:
            call_pain = sum(max(0.0, s - float(c.get("strike", 0.0))) * float(c.get("oi", 0.0)) for c in calls)
            put_pain = sum(max(0.0, float(p.get("strike", 0.0)) - s) * float(p.get("oi", 0.0)) for p in puts)
            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                max_pain_strike = s
        return float(max_pain_strike)
    except Exception:
        return 0.0


def _synthetic_option_chain(symbol: str) -> Dict[str, Any]:
    """
    Fallback synthetic option chain when NSE blocks API access.
    Uses yfinance spot and approximates CE/PE surface around ATM.
    """
    import yfinance as yf  # type: ignore[import]

    ticker_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    }
    sym = (symbol or "NIFTY").upper().strip()
    yf_ticker = ticker_map.get(sym, f"{sym}.NS")

    try:
        hist = yf.Ticker(yf_ticker).history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else 22000.0
    except Exception:
        spot = 22000.0

    round_to = 100 if spot > 20000 else 50
    atm = round(spot / round_to) * round_to
    strikes = [atm + (i * round_to) for i in range(-10, 11)]
    calls: List[Dict[str, Any]] = []
    puts: List[Dict[str, Any]] = []

    for strike in strikes:
        diff = float(strike - spot)
        iv = 15.0 + abs(diff / max(spot, 1.0)) * 50.0

        if diff <= 0:
            call_ltp = max(0.0, spot - strike) + iv / 100.0 * 50.0
        else:
            call_ltp = max(5.0, (iv / 100.0) * spot * 0.3 * math.exp(-abs(diff) / max(spot, 1.0) * 5.0))

        if diff >= 0:
            put_ltp = max(0.0, strike - spot) + iv / 100.0 * 50.0
        else:
            put_ltp = max(5.0, (iv / 100.0) * spot * 0.3 * math.exp(-abs(diff) / max(spot, 1.0) * 5.0))

        oi = max(100, int(10000 * math.exp(-abs(diff) / max(spot, 1.0) * 20.0)))
        calls.append(
            {
                "strike": float(strike),
                "oi": oi,
                "oi_change": int(oi * 0.05),
                "iv": round(iv, 1),
                "ltp": round(call_ltp, 2),
                "volume": oi * 2,
            }
        )
        puts.append(
            {
                "strike": float(strike),
                "oi": oi,
                "oi_change": int(oi * 0.03),
                "iv": round(iv, 1),
                "ltp": round(put_ltp, 2),
                "volume": oi * 2,
            }
        )

    total_put_oi = float(sum(p["oi"] for p in puts))
    total_call_oi = float(sum(c["oi"] for c in calls))
    pcr = total_put_oi / max(total_call_oi, 1.0)
    return {
        "symbol": symbol,
        "spot_price": round(spot, 2),
        "expiry_dates": ["Synthetic"],
        "selected_expiry": "Synthetic",
        "calls": calls,
        "puts": puts,
        "pcr": round(pcr, 4),
        "max_pain": _calc_max_pain(calls, puts),
        "source": "synthetic",
    }


# ------------------------------------------------------------------
# REST endpoints (data for dashboard pages)
# ------------------------------------------------------------------

@app.get("/api/watchlist")
def get_watchlist():
    """Return full watchlist with latest prices fetched from yfinance."""
    import yfinance as yf  # type: ignore[import]

    cfg = _dashboard_cfg or {}
    stocks = []
    for asset_class in ("indian_stocks", "us_stocks"):
        items = cfg.get("watchlist", {}).get(asset_class, [])
        for item in items:
            stocks.append({
                "symbol": item.get("symbol", ""),
                "yf_ticker": item.get("yf_ticker", ""),
                "name": item.get("name", ""),
                "asset_class": asset_class,
            })
    return stocks


@app.get("/api/chart-data")
def get_chart_data(ticker: str = "RELIANCE.NS", interval: str = "5m", period: str = "1d"):
    """
    Return OHLCV data for a stock â€” used by dashboard charts.
    interval: 1m, 5m, 15m, 1h, 1d
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y
    """
    try:
        import yfinance as yf  # type: ignore[import]
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if not df.empty:
            # Handle MultiIndex columns from yfinance
            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.get_level_values(0)

            records = []
            for idx, row in df.iterrows():
                ts: Union[int, str] = int(idx.timestamp()) if hasattr(idx, 'timestamp') else str(idx)  # type: ignore[union-attr]
                records.append({
                    "time": ts,
                    "open": _round2(row.get("Open", 0)),
                    "high": _round2(row.get("High", 0)),
                    "low": _round2(row.get("Low", 0)),
                    "close": _round2(row.get("Close", 0)),
                    "volume": int(row.get("Volume", 0)),
                })
            return {"ticker": ticker, "data": records}

        # Secondary fallback for US stocks only.
        if _is_us_stock_ticker(ticker):
            try:
                from src.data.alpha_vantage import get_daily_candles, get_intraday_candles  # type: ignore[import]

                iv = (interval or "").lower().strip()
                av_data = (
                    get_intraday_candles(ticker, interval=iv)
                    if iv in {"5m", "15m", "1h", "5min", "15min", "60min"}
                    else get_daily_candles(ticker)
                )
                if av_data:
                    logger.warning("yfinance empty for %s, using Alpha Vantage fallback", ticker)
                    return {"ticker": ticker, "data": av_data}
            except Exception as e:
                logger.warning("Alpha Vantage chart fallback failed for %s: %s", ticker, e)

        return {"ticker": ticker, "data": [], "error": "No data"}
    except Exception as e:
        return {"ticker": ticker, "data": [], "error": str(e)}


@app.get("/api/focus-assets")
def get_focus_assets():
    """Return the dedicated Gold/Silver/Bitcoin focus universe."""
    if _focus_engine is None:
        return []
    return _focus_engine.list_assets()


@app.get("/api/focus-chart")
def get_focus_chart(symbol: str = "XAUUSD", interval: str = "5m", period: str = "5d"):
    """Return live candles for focus assets."""
    if _focus_engine is None:
        return {"symbol": symbol, "data": [], "error": "Focus engine unavailable"}
    return _focus_engine.get_chart_data(symbol_or_ticker=symbol, interval=interval, period=period)


@app.get("/api/focus-trade")
def get_focus_trade(symbol: str = "XAUUSD", interval: str = "5m", period: str = "5d"):
    """
    Return real-time AI trade with validation gates.
    The validated signal is only non-HOLD when all checks pass.
    """
    if _focus_engine is None:
        return {"symbol": symbol, "signal": "HOLD", "validated_signal": "HOLD", "validated": False}
    return _focus_engine.get_focus_trade(symbol_or_ticker=symbol, interval=interval, period=period)


@app.get("/api/focus-trades")
def get_focus_trades(interval: str = "5m"):
    """Batch endpoint for all focus assets."""
    if _focus_engine is None:
        return []
    return _focus_engine.get_focus_trades(interval=interval)


@app.get("/api/btc/history")
async def get_btc_history(interval: str = "1d"):
    """Historical BTCUSDT candles proxied from btc_intelligence (Binance-aligned)."""
    return await _btc_history_payload_from_upstream(interval=interval, limit=4000)


@app.get("/api/btc/candles")
async def get_btc_candles(interval: str = "15m", limit: int = 200):
    """Recent BTCUSDT candles for the terminal chart (proxied from btc_intelligence)."""
    limit = max(50, min(limit, 5000))
    return await _btc_history_payload_from_upstream(interval=interval, limit=limit)


@app.get("/api/btc/sr-levels")
async def get_btc_sr_levels():
    """Pivot S/R from last 100×1h; 0.8% cluster, ≤3% from spot, nearest 2 support + 2 resistance."""
    url = f"{BTC_INTEL_BASE}/market/klines?timeframe=1h&limit=100"
    payload = await _btc_fetch_json([url])
    rows_in: List[Any] = []
    if isinstance(payload, dict):
        rows_in = list(payload.get("rows") or [])
    data = _candle_rows_to_chart_data(rows_in)
    if len(data) < 3:
        return {
            "support": [],
            "resistance": [],
            "current": float(data[-1]["close"]) if data else 0.0,
            "error": "insufficient_candles" if not data else "need_at_least_3_bars",
        }
    return _compute_sr_from_ohlc(data)


@app.get("/api/btc/liquidity-zones")
async def get_btc_liquidity_zones():
    """High-volume liquidity nodes + sweep wicks from last 50×15m candles."""
    url = f"{BTC_INTEL_BASE}/market/klines?timeframe=15m&limit=50"
    payload = await _btc_fetch_json([url])
    rows_in: List[Any] = []
    if isinstance(payload, dict):
        rows_in = list(payload.get("rows") or [])
    data = _candle_rows_to_chart_data(rows_in)
    if len(data) < 21:
        return {
            "zones": [],
            "sweep_levels": [],
            "error": "insufficient_candles" if data else "no_data",
        }
    return _compute_liquidity_zones_payload(data)


def _btc_signal_placeholder_payload() -> dict[str, Any]:
    """Degraded HOLD payload so the UI never treats /api/btc/signal as a hard offline error."""
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "signal": "HOLD",
        "confidence": 0.0,
        "validated": False,
        "stale": True,
        "degraded": True,
        "source": "no_upstream",
        "as_of_utc": now,
        "reason": "BTC intelligence unreachable — start btc_intelligence on :9000 (or Redis btc:signal).",
    }


def _parse_signal_utc(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _maybe_record_btc_signal_to_local_history(normalized: dict[str, Any]) -> None:
    """Append validated LONG/SHORT to data/signal_history.json if as_of_utc is new vs last entry."""
    try:
        sig = str(normalized.get("signal") or "").upper()
        if sig not in {"LONG", "SHORT"}:
            return
        if not bool(normalized.get("validated")):
            return
        ts_new = _parse_signal_utc(normalized.get("as_of_utc"))
        if ts_new is None:
            return
        recent = get_signal_history(1)
        if recent:
            ts_last = _parse_signal_utc(recent[-1].get("time"))
            if ts_last is not None and ts_new <= ts_last:
                return
        record_signal(normalized)
    except Exception as exc:
        logger.warning("local signal_history record skipped: %s", exc)


@app.get("/api/btc/signal")
async def get_btc_signal(interval: str = "5m"):
    """Real-time BTC signal — proxied from btc_intelligence (Redis fast path + HTTP)."""
    _ = interval
    payload = await _btc_proxy_payload(
        name="btc_signal_route",
        redis_key="btc:signal",
        upstream_url=f"{BTC_INTEL_BASE}/signal",
    )
    raw: dict[str, Any]
    if isinstance(payload, dict) and payload.get("error") == "upstream_unavailable":
        sig_only = _redis_get_json("btc:signal")
        if sig_only and isinstance(sig_only, dict):
            degraded = dict(sig_only)
            degraded["stale"] = True
            degraded["degraded"] = True
            if not degraded.get("source"):
                degraded["source"] = "redis_signal_fallback"
            raw = degraded
        else:
            sig_cp = _redis_get_json("btc:signal:persistent")
            if sig_cp and isinstance(sig_cp, dict):
                degraded = dict(sig_cp)
                degraded["stale"] = True
                degraded["degraded"] = True
                degraded["source"] = "sqlite_checkpoint"
                raw = degraded
            else:
                raw = _btc_signal_placeholder_payload()
    elif not isinstance(payload, dict):
        raw = _btc_signal_placeholder_payload()
    else:
        raw = payload

    normalized = _normalize_dashboard_signal_payload(raw)
    _maybe_record_btc_signal_to_local_history(normalized)
    return normalized


@app.get("/api/btc/market-context")
async def get_btc_market_context(interval: str = "5m"):
    """Lightweight context slice derived from the latest institutional signal."""
    _ = interval
    try:
        payload = await _btc_proxy_payload(
            name="btc_signal_context",
            redis_key="btc:signal",
            upstream_url=f"{BTC_INTEL_BASE}/signal",
        )
        if not isinstance(payload, dict) or payload.get("error") == "upstream_unavailable":
            return {}
        p = _normalize_dashboard_signal_payload(payload)
        return {
            "ticker": p.get("ticker", "BTCUSDT"),
            "regime": p.get("regime") or p.get("market_regime"),
            "derivatives": p.get("derivatives"),
            "macro": p.get("macro"),
            "order_flow": p.get("order_flow"),
            "as_of_utc": p.get("as_of_utc"),
        }
    except Exception:
        return {}


@app.get("/api/btc/signal/history")
async def signal_history(limit: int = Query(50, ge=1, le=200)):
    lim = int(limit)
    try:
        data = await _btc_fetch_json([f"{BTC_INTEL_BASE}/signal/history?limit={lim}"])
        if isinstance(data, list):
            signals = [x for x in data if isinstance(x, dict)]
            return {
                "signals": signals[:lim],
                "stats": get_signal_stats(),
                "source": "btc_intelligence",
            }
        if isinstance(data, dict) and isinstance(data.get("signals"), list):
            sigs_raw = data.get("signals") or []
            signals = [x for x in sigs_raw if isinstance(x, dict)]
            return {
                "signals": signals[:lim],
                "stats": get_signal_stats(),
                "source": "btc_intelligence",
            }
    except Exception as exc:
        logger.warning("/api/btc/signal/history upstream failed: %s", exc)
    try:
        return {
            "signals": get_signal_history(lim),
            "stats": get_signal_stats(),
            "source": "local_fallback",
        }
    except Exception as exc:
        logger.warning("/api/btc/signal/history local fallback failed: %s", exc)
        return {"signals": [], "stats": get_signal_stats(), "source": "local_fallback"}


@app.get("/api/btc/signal/stats")
def signal_stats():
    return get_signal_stats()


@app.get("/api/btc/system-report")
async def btc_system_report(interval: str = "5m"):
    _ = interval
    report: dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "runtime": {"engine": "btc_intelligence", "base": BTC_INTEL_BASE},
        "known_gaps": [],
    }
    health = await _btc_fetch_json([f"{BTC_INTEL_BASE}/health"])
    report["upstream_health"] = health if isinstance(health, dict) else {"status": "unknown"}

    try:
        raw = await _btc_proxy_payload(
            name="btc_signal_report",
            redis_key="btc:signal",
            upstream_url=f"{BTC_INTEL_BASE}/signal",
        )
        signal = _normalize_dashboard_signal_payload(raw if isinstance(raw, dict) else {})
        report["signal_snapshot"] = {
            "signal": signal.get("signal"),
            "validated_signal": signal.get("validated_signal"),
            "validated": signal.get("validated"),
            "confidence": signal.get("confidence"),
            "reason": signal.get("reason"),
            "regime": signal.get("regime"),
            "as_of_utc": signal.get("as_of_utc"),
        }
    except Exception as exc:
        report["signal_snapshot"] = {"error": str(exc)}

    try:
        from src.paper_trading import get_auto_executor, get_paper_engine

        engine = get_paper_engine()
        executor = get_auto_executor()
        report["automation"] = {
            "paper_engine_ready": True,
            "mode": engine._state.get("mode", "manual"),
            "open_positions": len(engine.get_open_positions()),
            "closed_trades": len(engine.get_closed_trades(500)),
            "pending_signals": len(executor.get_pending_signals()),
            "auto_executor_running": bool(executor._running),
        }
    except Exception as exc:
        report["automation"] = {
            "paper_engine_ready": False,
            "error": str(exc),
        }

    return report


def _markers_from_institutional_history(rows: List[Any], limit: int) -> List[dict[str, Any]]:
    markers: List[dict[str, Any]] = []
    if not isinstance(rows, list):
        return markers
    for row in rows:
        if not isinstance(row, dict):
            continue
        sig = str(row.get("signal") or row.get("validated_signal") or "HOLD").upper()
        if sig not in {"LONG", "SHORT"}:
            continue
        raw_ts = row.get("as_of_utc") or row.get("timestamp") or row.get("as_of")
        t_unix = None
        if isinstance(raw_ts, (int, float)):
            t_unix = int(raw_ts) // 1000 if float(raw_ts) > 1e12 else int(raw_ts)
        elif isinstance(raw_ts, str) and raw_ts.strip():
            try:
                iso = raw_ts.replace("Z", "+00:00")
                t_unix = int(datetime.fromisoformat(iso).timestamp())
            except Exception as exc:
                logger.warning("_markers_from_institutional_history bad timestamp %r: %s", raw_ts, exc)
                continue
        if t_unix is None:
            continue
        conf = float(row.get("confidence") or 0.0)
        markers.append(
            {
                "time": int(t_unix),
                "position": "belowBar" if sig == "LONG" else "aboveBar",
                "color": "#34d399" if sig == "LONG" else "#f87171",
                "shape": "arrowUp" if sig == "LONG" else "arrowDown",
                "text": "",
                "signal": sig,
                "confidence": conf,
            }
        )
    markers.sort(key=lambda m: int(m["time"]))
    return markers[-limit:]


@app.get("/api/btc/markers")
async def get_btc_markers(interval: str = "1d", limit: int = 1000):
    """Chart markers from btc_intelligence recent signal history (institutional path)."""
    _ = interval
    limit = int(max(50, min(limit, 500)))
    hist = await _btc_fetch_json([f"{BTC_INTEL_BASE}/signal/history"])
    rows: List[Any] = []
    if isinstance(hist, list):
        rows = hist
    return _markers_from_institutional_history(rows, limit)


@app.get("/api/btc/news")
def btc_news(limit: int = 8):
    return {"news": get_btc_news(limit)}


@app.get("/api/btc/orderflow")
async def btc_orderflow_proxy():
    return await _btc_proxy_payload(
        name="orderflow",
        redis_key="btc:orderflow",
        upstream_url=f"{BTC_INTEL_BASE}/api/orderflow",
    )


@app.get("/api/btc/volume")
async def btc_volume_proxy():
    return await _btc_proxy_payload(
        name="volume",
        redis_key="btc:volprofile",
        upstream_url=f"{BTC_INTEL_BASE}/api/volume-profile",
    )


@app.get("/api/btc/volatility")
async def btc_volatility_proxy():
    return await _btc_proxy_payload(
        name="volatility",
        redis_key="btc:volatility",
        upstream_url=f"{BTC_INTEL_BASE}/api/volatility",
    )


@app.get("/api/btc/execution")
async def btc_execution_proxy():
    return await _btc_proxy_payload(
        name="execution",
        redis_key="btc:execution",
        upstream_url=f"{BTC_INTEL_BASE}/api/execution",
    )


@app.get("/api/btc/decision-intelligence")
async def btc_decision_intelligence_proxy(interval: str = Query("15m")):
<<<<<<< HEAD
    _ = interval if interval in INTERVAL_TO_MS else "15m"
=======
    iv = interval if interval in INTERVAL_TO_MS else "15m"
>>>>>>> origin/main
    payload = await _btc_proxy_payload(
        name="decision_intelligence",
        redis_key=["btc:intelligence", "btc:decision", "btc:signal"],
        upstream_url=[
            f"{BTC_INTEL_BASE}/api/intelligence",
            f"{BTC_INTEL_BASE}/api/decision",
            f"{BTC_INTEL_BASE}/signal",
        ],
    )
    if not isinstance(payload, dict):
        payload = {}

    if payload.get("error") == "upstream_unavailable":
        sig_only = _redis_get_json("btc:signal")
        if sig_only and (
            sig_only.get("signal") is not None
            or "confidence" in sig_only
            or isinstance(sig_only.get("probability"), dict)
        ):
            payload = _minimal_decision_intelligence_from_signal(sig_only)

<<<<<<< HEAD
    if payload.get("error") == "upstream_unavailable":
        sig_cp = _redis_get_json("btc:signal:persistent")
        if sig_cp and (
            sig_cp.get("signal") is not None
            or "confidence" in sig_cp
            or isinstance(sig_cp.get("probability"), dict)
        ):
            payload = _minimal_decision_intelligence_from_signal(sig_cp, source="sqlite_checkpoint")
=======
    if payload.get("error") == "upstream_unavailable" and _btc_service is not None:
        try:
            live_sig = _btc_service.get_realtime_signal(interval=iv)
            if isinstance(live_sig, dict) and (
                live_sig.get("signal") is not None
                or "confidence" in live_sig
                or isinstance(live_sig.get("probability"), dict)
            ):
                payload = _minimal_decision_intelligence_from_signal(
                    live_sig,
                    source="dashboard_signal_fallback",
                )
        except Exception as exc:
            logger.warning("decision-intelligence dashboard signal fallback failed: %s", exc)
>>>>>>> origin/main

    if payload.get("error") == "upstream_unavailable":
        return {"error": "upstream_unavailable", "stale": True}

    normalized = _normalize_decision_intelligence_payload(payload)
    if normalized.get("error"):
        if isinstance(normalized.get("decision_engine"), dict) and normalized["decision_engine"]:
            normalized.pop("error", None)
        else:
            return normalized

    if not normalized.get("probability"):
        prob_payload = await _btc_proxy_payload(
            name="probability_for_decision",
            redis_key=["btc:probability", "btc:intelligence", "btc:signal"],
            upstream_url=[
                f"{BTC_INTEL_BASE}/api/probability",
                f"{BTC_INTEL_BASE}/api/intelligence",
                f"{BTC_INTEL_BASE}/api/decision",
                f"{BTC_INTEL_BASE}/signal",
            ],
        )
        normalized["probability"] = _extract_probability_payload(prob_payload if isinstance(prob_payload, dict) else {})

    if not normalized.get("execution_plan"):
        exec_payload = await _btc_proxy_payload(
            name="execution_plan_for_decision",
            redis_key=["btc:execution_plan", "btc:intelligence"],
            upstream_url=[
                f"{BTC_INTEL_BASE}/api/execution-plan",
                f"{BTC_INTEL_BASE}/api/intelligence",
                f"{BTC_INTEL_BASE}/api/execution",
            ],
        )
        if isinstance(exec_payload, dict):
            if isinstance(exec_payload.get("execution_plan"), dict):
                normalized["execution_plan"] = dict(exec_payload.get("execution_plan", {}))
            elif any(k in exec_payload for k in ("entry_zone", "stop_loss", "take_profit", "take_profit_2", "expected_rr", "slippage_risk")):
                normalized["execution_plan"] = dict(exec_payload)

    if not isinstance(normalized.get("decision_breakdown"), dict):
        normalized["decision_breakdown"] = {}
    for meta_k in ("stale", "degraded", "source"):
        if meta_k in payload:
            normalized[meta_k] = payload[meta_k]
<<<<<<< HEAD
    return _enrich_decision_intel(normalized, payload)
=======
    return normalized
>>>>>>> origin/main


@app.get("/api/btc/probability")
async def btc_probability_proxy():
    payload = await _btc_proxy_payload(
        name="probability",
        redis_key=["btc:probability", "btc:intelligence", "btc:signal"],
        upstream_url=[
            f"{BTC_INTEL_BASE}/api/probability",
            f"{BTC_INTEL_BASE}/api/intelligence",
            f"{BTC_INTEL_BASE}/api/decision",
            f"{BTC_INTEL_BASE}/signal",
        ],
    )
    if not isinstance(payload, dict):
        return {"up_prob": 0.0, "down_prob": 0.0, "sideways_prob": 0.0, "dominant": "SIDEWAYS", "calibration_score": 0.0, "stale": True}

    if payload.get("error"):
        return payload

    prob = _extract_probability_payload(payload)
    if prob:
        prob["stale"] = bool(payload.get("stale", False))
        return prob
    return {"up_prob": 0.0, "down_prob": 0.0, "sideways_prob": 0.0, "dominant": "SIDEWAYS", "calibration_score": 0.0, "stale": True}


@app.get("/api/btc/execution-plan")
async def btc_execution_plan_proxy():
    return await _btc_proxy_payload(
        name="execution_plan",
        redis_key="btc:execution_plan",
        upstream_url=f"{BTC_INTEL_BASE}/api/execution-plan",
    )


@app.get("/api/news")
def get_news(symbol: str = "BTC", asset_class: str = "crypto", limit: int = 8):
    from src.data.news_feed import get_news_for_asset
    cls = (asset_class or "crypto").lower().strip()
    if cls in {"indian_stock", "us_stock", "stocks"}:
        cls = "stock"
    return {"news": get_news_for_asset(symbol, cls, limit)}


@app.get("/api/options/chain")
async def get_options_chain(symbol: str = "NIFTY", expiry: str = "nearest"):
    """
    Fetch NSE option chain with proper browser headers + session cookie.
    Falls back to synthetic chain when NSE blocks requests.
    """
    import requests  # type: ignore[import]

    sym_upper = (symbol or "NIFTY").upper().strip()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    }
    session = requests.Session()
    try:
        # Prime cookies before API hit.
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        session.get("https://www.nseindia.com/option-chain", headers=headers, timeout=10)

        url_map = {
            "NIFTY": "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
            "BANKNIFTY": "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY",
            "FINNIFTY": "https://www.nseindia.com/api/option-chain-indices?symbol=FINNIFTY",
        }
        if sym_upper in url_map:
            url = url_map[sym_upper]
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={sym_upper}"

        response = session.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return _synthetic_option_chain(symbol)

        data = response.json()
        records = data.get("records", {}) if isinstance(data, dict) else {}
        if not records:
            return _synthetic_option_chain(symbol)

        spot = float(records.get("underlyingValue", 0.0) or 0.0)
        expiry_dates = list(records.get("expiryDates", []) or [])
        selected_expiry = None
        if expiry_dates:
            req_exp = str(expiry or "nearest").strip()
            if req_exp and req_exp.lower() != "nearest" and req_exp in expiry_dates:
                selected_expiry = req_exp
            else:
                selected_expiry = expiry_dates[0]

        calls: List[Dict[str, Any]] = []
        puts: List[Dict[str, Any]] = []
        for item in list(records.get("data", []) or []):
            if selected_expiry and item.get("expiryDate") != selected_expiry:
                continue
            strike = float(item.get("strikePrice", 0.0) or 0.0)
            ce = item.get("CE")
            pe = item.get("PE")
            if isinstance(ce, dict):
                calls.append(
                    {
                        "strike": strike,
                        "oi": int(ce.get("openInterest", 0) or 0),
                        "oi_change": int(ce.get("changeinOpenInterest", 0) or 0),
                        "iv": float(ce.get("impliedVolatility", 0.0) or 0.0),
                        "ltp": float(ce.get("lastPrice", 0.0) or 0.0),
                        "volume": int(ce.get("totalTradedVolume", 0) or 0),
                    }
                )
            if isinstance(pe, dict):
                puts.append(
                    {
                        "strike": strike,
                        "oi": int(pe.get("openInterest", 0) or 0),
                        "oi_change": int(pe.get("changeinOpenInterest", 0) or 0),
                        "iv": float(pe.get("impliedVolatility", 0.0) or 0.0),
                        "ltp": float(pe.get("lastPrice", 0.0) or 0.0),
                        "volume": int(pe.get("totalTradedVolume", 0) or 0),
                    }
                )

        if not calls and not puts:
            return _synthetic_option_chain(symbol)

        total_put_oi = float(sum(p.get("oi", 0.0) for p in puts))
        total_call_oi = float(sum(c.get("oi", 0.0) for c in calls))
        pcr = total_put_oi / max(total_call_oi, 1.0)
        return {
            "symbol": symbol,
            "spot_price": round(spot, 2),
            "expiry_dates": expiry_dates,
            "selected_expiry": selected_expiry,
            "calls": calls,
            "puts": puts,
            "pcr": round(pcr, 4),
            "max_pain": _calc_max_pain(calls, puts),
            "source": "NSE",
        }
    except Exception as e:
        logger.warning("NSE option chain failed for %s: %s", symbol, e)
        return _synthetic_option_chain(symbol)
    finally:
        try:
            session.close()
        except Exception as exc:
            logger.warning("NSE session close failed: %s", exc)


@app.get("/api/options/signal")
def get_options_signal(symbol: str = "NIFTY"):
    try:
        eng = _get_options_engine()
        return eng.get_options_signal(symbol=symbol)
    except Exception as e:
        logger.warning("Options signal endpoint failed for %s: %s", symbol, e)
        return {"symbol": symbol, "strategy": "HOLD", "confidence": 0.0, "error": str(e)}


@app.get("/api/options/iv-surface")
def get_options_iv_surface(symbol: str = "NIFTY", expiry: str = "nearest"):
    try:
        eng = _get_options_engine()
        return eng.get_iv_surface(symbol=symbol, expiry=expiry)
    except Exception as e:
        logger.warning("Options IV endpoint failed for %s: %s", symbol, e)
        return {"symbol": symbol, "points": [], "iv_percentile": 0.0, "error": str(e)}


@app.get("/api/options/max-pain")
def get_options_max_pain(symbol: str = "NIFTY", expiry: str = "nearest"):
    try:
        eng = _get_options_engine()
        chain = eng.get_option_chain(symbol=symbol, expiry=expiry)
        return {
            "symbol": symbol,
            "spot_price": chain.get("spot_price", 0.0),
            "pcr": chain.get("pcr", 0.0),
            "max_pain": chain.get("max_pain", 0.0),
            "selected_expiry": chain.get("selected_expiry", ""),
        }
    except Exception as e:
        logger.warning("Options max pain endpoint failed for %s: %s", symbol, e)
        return {"symbol": symbol, "max_pain": 0.0, "error": str(e)}


@app.get("/api/options/greeks")
def get_options_greeks(
    symbol: str = "NIFTY",
    strike: float = 22000.0,
    expiry: str = "nearest",
    type: str = "CE",  # noqa: A002
):
    try:
        eng = _get_options_engine()
        chain = eng.get_option_chain(symbol=symbol, expiry="nearest")
        spot = float(chain.get("spot_price", 0.0))
        expiry_days = _expiry_tracker.get_days_to_expiry(symbol)
        if expiry and expiry.lower() != "nearest":
            parsed = None
            for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"):
                try:
                    parsed = datetime.strptime(expiry, fmt).date()
                    break
                except Exception as exc:
                    logger.warning("options greeks expiry parse %r with %r: %s", expiry, fmt, exc)
                    continue
            if parsed is not None:
                expiry_days = max((parsed - datetime.utcnow().date()).days, 1)

        iv_surface = eng.get_iv_surface(symbol=symbol, expiry=expiry)
        iv = float(iv_surface.get("atm_iv", 15.0) or 15.0)
        greeks = eng.calculate_greeks(
            spot=spot if spot > 0 else strike,
            strike=float(strike),
            expiry_days=max(expiry_days, 1),
            iv=iv,
            option_type=type,
        )
        return {
            "symbol": symbol,
            "spot_price": spot,
            "strike": float(strike),
            "expiry_days": max(expiry_days, 1),
            "option_type": type.upper(),
            "iv": iv,
            "greeks": greeks,
        }
    except Exception as e:
        logger.warning("Options greeks endpoint failed for %s: %s", symbol, e)
        return {"symbol": symbol, "greeks": {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}, "error": str(e)}


@app.get("/api/options/expiry")
def get_options_expiry(symbol: str = "NIFTY"):
    try:
        next_exp = _expiry_tracker.get_next_expiry(symbol)
        return {
            "symbol": symbol,
            "next_expiry": next_exp.isoformat(),
            "days_to_expiry": _expiry_tracker.get_days_to_expiry(symbol),
            "is_expiry_week": _expiry_tracker.is_expiry_week(symbol),
            "is_expiry_day": _expiry_tracker.is_expiry_day(symbol),
            "monthly_expiry": _expiry_tracker.get_monthly_expiry(next_exp.month, next_exp.year).isoformat(),
        }
    except Exception as e:
        logger.warning("Options expiry endpoint failed for %s: %s", symbol, e)
        return {"symbol": symbol, "error": str(e)}


@app.get("/api/av/status")
def get_alpha_vantage_status():
    try:
        from src.data.alpha_vantage import get_status  # type: ignore[import]

        return get_status()
    except Exception as e:
        logger.warning("Alpha Vantage status unavailable: %s", e)
        return {"remaining_requests": 0, "daily_limit": 23}


@app.get("/api/compliance/status")
def get_compliance_status():
    return _compliance_engine.get_status()


@app.get("/api/compliance/algo-ids")
def get_compliance_algo_ids():
    return {"algo_ids": _compliance_engine.list_algo_ids()}


@app.post("/api/compliance/kill-switch")
def trigger_compliance_kill_switch(payload: dict):
    reason = str(payload.get("reason", "manual_trigger"))
    _compliance_engine.kill_switch(reason)
    return {"ok": True, "reason": reason}


@app.get("/api/compliance/audit")
def get_compliance_audit(date: str | None = None):
    day = date or datetime.utcnow().strftime("%Y-%m-%d")
    return {"date": day, "rows": _compliance_engine.get_audit(day)}


@app.get("/api/compliance/tax-summary")
def get_compliance_tax_summary():
    return _compliance_engine.tax_summary()


@app.get("/api/compliance/true-cost")
def get_compliance_true_cost(trade: str | None = None):
    payload: dict[str, Any] = {}
    if trade:
        try:
            payload = json.loads(trade)
        except Exception:
            payload = {}
    return _compliance_engine.calculate_true_costs(payload)


@app.get("/api/compliance/report")
def get_compliance_report(date: str | None = None):
    day = date or datetime.utcnow().strftime("%Y-%m-%d")
    return _compliance_engine.generate_compliance_report(day)


@app.get("/api/compliance/export-itr3")
def export_itr3_csv():
    return {"path": _compliance_engine.export_itr3_csv()}


@app.post("/api/backtest")
async def run_backtest_endpoint(payload: dict):
    """
    Run backtest in simple / wfo / cpcv mode.
    """
    ticker = payload.get("ticker", "RELIANCE.NS")
    strategy = str(payload.get("strategy", "quant_alpha")).lower()
    mode = str(payload.get("mode", "simple")).lower().strip()
    from_date = payload.get("from_date")  # "YYYY-MM-DD"
    to_date = payload.get("to_date")      # "YYYY-MM-DD"
    capital = float(payload.get("capital", 100000))

    try:
        import yfinance as yf  # type: ignore[import]
        import numpy as np  # type: ignore[import]
        import pandas as pd  # type: ignore[import]
        from src.features.engineer import FeatureEngineer  # type: ignore[import]
        from src.alpha.factor_model import get_cached_alpha_model  # type: ignore[import]
        from src.research.validation import CPCVValidator  # type: ignore[import]
        from src.validator import WFOValidator  # type: ignore[import]
    except Exception as e:
        return {"error": f"Backtest dependency missing: {e}"}

    # Fetch data
    try:
        df = yf.download(ticker, start=from_date, end=to_date, progress=False)
    except Exception as e:
        return {"error": f"Data fetch failed: {e}"}
    if df.empty or len(df) < 30:
        return {"error": "Insufficient data for backtest"}

    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    # Compute features
    eng = FeatureEngineer()
    feat_df = eng.compute_all_features(df, timeframe="daily", ticker=ticker)
    if feat_df.empty:
        return {"error": "Feature computation failed"}

    close = feat_df["Close"].astype(float)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi14 = feat_df.get("RSI_14", pd.Series([50.0] * len(feat_df), index=feat_df.index)).astype(float)
    roc5 = feat_df.get("ROC_5d", close.pct_change(5) * 100).astype(float)

    def _build_signals(frame: pd.DataFrame) -> list[str]:
        sigs = ["HOLD"] * len(frame)
        if strategy == "quant_alpha":
            ac = "indian_stock" if str(ticker).upper().endswith((".NS", ".BO")) else "us_stock"
            model = get_cached_alpha_model(asset_class=ac, alpha_threshold=0.25)
            for i in range(50, len(frame)):
                try:
                    out = model.score(frame.iloc[: i + 1], asset=str(ticker), asset_class=ac)
                    sigs[i] = str(out.get("signal", "HOLD")).upper()
                except Exception:
                    sigs[i] = "HOLD"
        elif strategy == "momentum_only":
            for i in range(50, len(frame)):
                if pd.notna(roc5.iloc[i]) and pd.notna(sma20.iloc[i]):
                    if roc5.iloc[i] > 0.5 and close.iloc[i] > sma20.iloc[i]:
                        sigs[i] = "BUY"
                    elif roc5.iloc[i] < -0.5 and close.iloc[i] < sma20.iloc[i]:
                        sigs[i] = "SELL"
        elif strategy == "mean_reversion_only":
            for i in range(50, len(frame)):
                if pd.notna(rsi14.iloc[i]):
                    if rsi14.iloc[i] < 35:
                        sigs[i] = "BUY"
                    elif rsi14.iloc[i] > 65:
                        sigs[i] = "SELL"
        elif strategy == "ma_crossover":
            prev_spread = None
            for i in range(50, len(frame)):
                if pd.notna(sma20.iloc[i]) and pd.notna(sma50.iloc[i]):
                    spread = float(sma20.iloc[i] - sma50.iloc[i])
                    if prev_spread is not None:
                        if spread > 0 and prev_spread <= 0:
                            sigs[i] = "BUY"
                        elif spread < 0 and prev_spread >= 0:
                            sigs[i] = "SELL"
                    prev_spread = spread
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return sigs

    def _simulate(frame: pd.DataFrame, signals: list[str]) -> dict[str, object]:
        frame = frame.copy()
        frame["signal"] = signals
        position = 0.0
        cash = capital
        equity = [capital]
        trades: list[dict[str, Any]] = []
        entry_price = 0.0
        entry_date: str | None = None
        entry_idx: int | None = None

        for i in range(1, len(frame)):
            price = float(frame["Close"].iloc[i])
            sig = str(frame["signal"].iloc[i]).upper()
            dt = str(frame.index[i].date())

            if sig == "BUY" and position == 0:
                position = cash / price if price > 0 else 0
                entry_price = price
                entry_date = dt
                entry_idx = i
                cash = 0
            elif sig == "SELL" and position > 0:
                exit_value = position * price
                pnl_pct = round((price - entry_price) / max(entry_price, 1e-12) * 100, 2)
                hold_days = max(1, (i - (entry_idx or i)))
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": dt,
                        "signal": "LONG",
                        "entry": round(entry_price, 2),
                        "exit": round(price, 2),
                        "pnl_pct": pnl_pct,
                        "result": "WIN" if pnl_pct > 0 else "LOSS",
                        "hold_days": hold_days,
                    }
                )
                cash = exit_value
                position = 0.0
                entry_idx = None

            current_value = cash + (position * price if position > 0 else 0)
            equity.append(round(current_value, 2))

        if position > 0:
            price = float(frame["Close"].iloc[-1])
            dt = str(frame.index[-1].date())
            pnl_pct = round((price - entry_price) / max(entry_price, 1e-12) * 100, 2)
            hold_days = max(1, (len(frame) - 1 - (entry_idx or len(frame) - 1)))
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "signal": "LONG",
                    "entry": round(entry_price, 2),
                    "exit": round(price, 2),
                    "pnl_pct": pnl_pct,
                    "result": "WIN" if pnl_pct > 0 else "LOSS",
                    "hold_days": hold_days,
                }
            )
            cash = position * price
            equity[-1] = round(cash, 2)

        final_value = equity[-1]
        total_return = round((final_value - capital) / max(capital, 1e-12) * 100, 2)
        returns = np.diff(equity) / np.maximum(np.array(equity[:-1]), 1e-12)
        returns = returns[np.isfinite(returns)]
        returns = returns[returns != 0]
        sharpe = round(float(np.mean(returns) / max(np.std(returns), 1e-10)) * np.sqrt(252), 2) if len(returns) > 1 else 0.0
        wins = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]

        peak = capital
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (v - peak) / max(peak, 1e-12) * 100
            if dd < max_dd:
                max_dd = dd

        avg_hold = round(float(np.mean([t.get("hold_days", 0) for t in trades])) if trades else 0.0, 1)
        step = max(1, len(equity) // 200)
        dates = [str(frame.index[min(i, len(frame) - 1)].date()) for i in range(0, len(equity), step)]
        eq_sampled = equity[::step]

        return {
            "metrics": {
                "total_return_pct": total_return,
                "final_value": round(final_value, 2),
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": round(max_dd, 2),
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate_pct": round(len(wins) / max(len(trades), 1) * 100, 1),
                "best_trade_pct": max((t["pnl_pct"] for t in trades), default=0),
                "worst_trade_pct": min((t["pnl_pct"] for t in trades), default=0),
                "avg_hold_days": avg_hold,
            },
            "equity_curve": [{"time": d, "value": v} for d, v in zip(dates, eq_sampled)],
            "trades": trades[-50:],
        }

    if mode == "wfo":
        cfg = (_dashboard_cfg.get("wfo", {}) or {})
        validator = WFOValidator(ticker=ticker, train_window=int(cfg.get("min_train_days", 252)), wfo_config=cfg)
        return {"mode": "wfo", "result": validator.run_validation()}

    if mode == "cpcv":
        cpcv = CPCVValidator(
            n_splits=int(payload.get("n_splits", 6)),
            n_test_splits=int(payload.get("n_test_splits", 2)),
            embargo_pct=float(payload.get("embargo_pct", 0.01)),
        )

        close_arr = feat_df["Close"].astype(float).values
        ret_arr = np.diff(close_arr, prepend=close_arr[0]) / np.maximum(close_arr, 1e-12)

        def _strategy_fn(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
            sigs = _build_signals(frame)
            pos = np.array([1.0 if s == "BUY" else 0.0 for s in sigs], dtype=float)
            pos = np.roll(pos, 1)
            pos[0] = 0.0
            strat_ret = ret_arr * pos
            is_ret = strat_ret[train_idx] if len(train_idx) else np.array([], dtype=float)
            oos_ret = strat_ret[test_idx] if len(test_idx) else np.array([], dtype=float)
            oos_curve = np.cumprod(1.0 + np.clip(oos_ret, -0.95, 10.0)).tolist()
            eq = [
                {"time": str(frame.index[int(i)].date()), "value": float(v)}
                for i, v in zip(test_idx[: len(oos_curve)], oos_curve)
            ]
            return {"is_returns": is_ret, "oos_returns": oos_ret, "equity_curve": eq}

        cpcv_result = cpcv.run_cpcv(feat_df, _strategy_fn)
        simple = _simulate(feat_df, _build_signals(feat_df))
        return {"mode": "cpcv", "cpcv": cpcv_result, **simple}

    try:
        signals = _build_signals(feat_df)
    except Exception as e:
        return {"error": str(e)}
    return {"mode": "simple", **_simulate(feat_df, signals)}


@app.get("/api/backtest/regime-analysis")
async def get_regime_analysis():
    """
    Returns last saved regime analysis.
    Reads from store — no computation here.
    Returns empty dict if none available yet.
    JWT: same protection as other GET routes.
    """
    try:
        if _store is None:
            return {}
        data = _store.get_latest_regime_analysis()
        return data or {}
    except Exception:
        return {}


@app.get("/api/stock-signal")
def get_stock_signal(ticker: str = "RELIANCE.NS"):
    """Return AI signal for a specific stock â€” live alpha computation."""
    # First check stored signals
    if _store is not None:
        signals = _store.get_recent_signals(limit=200)
        for s in signals:
            if s.get("asset") == ticker or s.get("asset", "").replace(".NS", "") == ticker.replace(".NS", ""):
                return _attach_factor_fields(dict(s))

    def _av_fallback_signal() -> dict | None:
        if not _is_us_stock_ticker(ticker):
            return None
        try:
            from src.data.alpha_vantage import get_us_stock_signal  # type: ignore[import]
            av_signal = get_us_stock_signal(ticker)
            if av_signal:
                logger.warning("Using Alpha Vantage fallback signal for %s", ticker)
                return dict(av_signal)
        except Exception as e:
            logger.warning("Alpha Vantage signal fallback failed for %s: %s", ticker, e)
        return None

    # Primary source for stocks: yfinance-based live computation.
    try:
        import yfinance as yf  # type: ignore[import]
        from src.features.engineer import FeatureEngineer  # type: ignore[import]
        from src.alpha.factor_model import get_cached_alpha_model  # type: ignore[import]
        from src.alpha.signal_quality import SignalQualityScorer  # type: ignore[import]

        # Fallback to fetching minimum 5 days to ensure FeatureEngineer has enough data for ATR_Percentile (104 bars)
        df = yf.download(ticker, interval="5m", period="5d", progress=False)
        if df.empty or len(df) < 20:
            # Try daily data
            df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or len(df) < 30:
            av = _av_fallback_signal()
            if av is not None:
                return _attach_factor_fields(av)
            return _attach_factor_fields({"ticker": ticker, "signal": "HOLD", "confidence": 50, "alpha_score": 0.0, "message": "Insufficient data"})

        # Handle MultiIndex
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)

        eng = FeatureEngineer()
        tf = "daily" if len(df) > 100 else "intraday"
        feat_df = eng.compute_all_features(df, timeframe=tf, ticker=ticker)
        if feat_df.empty:
            av = _av_fallback_signal()
            if av is not None:
                return _attach_factor_fields(av)
            return _attach_factor_fields({"ticker": ticker, "signal": "HOLD", "confidence": 50, "alpha_score": 0.0, "message": "Feature computation failed"})

        # Very low threshold for dashboard activity demonstration
        asset_class = "indian_stock" if ticker.upper().endswith((".NS", ".BO")) else "us_stock"
        model = get_cached_alpha_model(asset_class=asset_class, alpha_threshold=0.01)
        result = model.score(feat_df, asset=ticker, asset_class=asset_class)
        atr_pct = float(feat_df.get("ATR_Percentile", 50.0).iloc[-1]) if "ATR_Percentile" in feat_df.columns else 50.0
        sqs_min = float((_dashboard_cfg.get("signal", {}) or {}).get("min_sqs", 55))
        sqs_obj = SignalQualityScorer(min_sqs=sqs_min).score(
            signal=str(result.get("signal", "HOLD")),
            factor_scores=result.get("factor_scores", {}),
            ic_weights=result.get("ic_weights", {}),
            regime=str(result.get("regime_for_confidence", "SIDEWAYS")),
            atr_percentile=atr_pct,
        )
        signal_out = str(result.get("signal", "HOLD"))
        if signal_out in {"BUY", "SELL"} and not sqs_obj.passes:
            signal_out = "HOLD"
        sqs_message = (
            f"Signal blocked by SQS gate ({sqs_obj.sqs} < {sqs_min})"
            if str(result.get("signal", "HOLD")) in {"BUY", "SELL"} and not sqs_obj.passes
            else ""
        )

        entry_price: float = float(feat_df["Close"].iloc[-1])
        atr: float = float(feat_df.get("ATR_14", feat_df["Close"] * 0.02).iloc[-1])

        sl: float = entry_price - 2 * atr if result["signal"] == "BUY" else entry_price + 2 * atr
        tp: float = entry_price + 3 * atr if result["signal"] == "BUY" else entry_price - 3 * atr

        return _attach_factor_fields({
            "ticker": ticker,
            "asset": ticker,
            "signal": signal_out,
            "strength": result["strength"],
            "confidence": result["confidence"],
            "alpha_score": result["alpha_score"],
            "entry_price": _round2(entry_price),
            "stop_loss": _round2(sl),
            "take_profit": _round2(tp),
            "factor_scores": result.get("factor_scores", {}),
            "ic_weights": result.get("ic_weights", {}),
            "sqs": sqs_obj.sqs,
            "sqs_passed": sqs_obj.passes,
            "sqs_components": sqs_obj.components,
            "message": sqs_message,
            "live": True,
        })
    except Exception as e:
        av = _av_fallback_signal()
        if av is not None:
            return _attach_factor_fields(av)
        return _attach_factor_fields({"ticker": ticker, "signal": "HOLD", "confidence": 50, "alpha_score": 0.0, "message": str(e)})


def _norm_signal_dir(raw_signal: Any) -> str:
    s = str(raw_signal or "").upper().strip()
    if s in {"BUY", "LONG"}:
        return "LONG"
    if s in {"SELL", "SHORT"}:
        return "SHORT"
    return "HOLD"


def _latest_store_signal_for_ticker(ticker: str) -> dict[str, Any] | None:
    if _store is None:
        return None
    try:
        needle = str(ticker or "").upper().strip()
        needle_alt = needle.replace(".NS", "").replace(".BO", "")
        rows = _store.get_recent_signals(limit=400)
        for row in rows:
            asset = str(row.get("asset") or row.get("ticker") or "").upper().strip()
            if not asset:
                continue
            asset_alt = asset.replace(".NS", "").replace(".BO", "")
            if asset == needle or asset_alt == needle_alt:
                return dict(row)
    except Exception:
        return None
    return None


def _download_stock_frame(ticker: str, interval: str = "15m", period: str = "5d"):
    try:
        import yfinance as yf  # type: ignore[import]

        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in df.columns:
                return None
        return df
    except Exception:
        return None


def _calc_rsi_from_closes(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = float(closes[i]) - float(closes[i - 1])
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss <= 1e-9:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_rr_ratio(entry_price: float, sl: float, tp1: float) -> float:
    risk = abs(float(entry_price) - float(sl))
    reward = abs(float(tp1) - float(entry_price))
    if risk <= 1e-9:
        return 0.0
    return round(reward / risk, 2)


def _bucket_from_confidence(conf: float) -> str:
    if conf >= 80:
        return "STRONG"
    if conf >= 65:
        return "MODERATE"
    return "WEAK"


def _build_terminal_signal_payload(ticker: str, source: dict[str, Any]) -> dict[str, Any]:
    src = _attach_factor_fields(dict(source or {}))
    signal_dir = _norm_signal_dir(src.get("signal"))

    conf_raw = _safe_float(src.get("confidence"), 0.0)
    confidence = conf_raw * 100.0 if 0.0 <= conf_raw <= 1.0 else conf_raw
    confidence = max(0.0, min(100.0, confidence))

    alpha_raw = _safe_float(src.get("alpha_score"), 0.0)
    strength = abs(alpha_raw) * 100.0 if abs(alpha_raw) <= 1.0 else abs(alpha_raw)
    signal_strength = max(0.0, min(100.0, round(strength, 1)))

    edge_raw = src.get("net_alpha_score", src.get("edge_after_costs", signal_strength))
    edge_val = _safe_float(edge_raw, signal_strength)
    if abs(edge_val) <= 1.0:
        edge_val *= 100.0
    edge_after_costs = max(0.0, min(100.0, round(edge_val, 1)))

    entry_price = _safe_float(src.get("entry_price", src.get("entry")), 0.0)
    sl = _safe_float(src.get("sl", src.get("stop_loss")), 0.0)
    tp1 = _safe_float(src.get("tp1", src.get("take_profit")), 0.0)
    rr_ratio = _safe_float(src.get("rr_ratio", src.get("risk_reward")), 0.0)
    if rr_ratio <= 0 and entry_price > 0 and sl > 0 and tp1 > 0:
        rr_ratio = _compute_rr_ratio(entry_price, sl, tp1)
    rr_ratio = round(rr_ratio, 2)

    regime = str(src.get("regime") or src.get("regime_for_confidence") or "SIDEWAYS").upper()
    reason = str(src.get("reason") or src.get("message") or "Live terminal signal")

    f1 = _safe_float(src.get("F1_momentum"), 0.0)
    f2 = _safe_float(src.get("F2_mean_rev"), 0.0)
    f3 = _safe_float(src.get("F3_volume"), 0.0)
    f5 = _safe_float(src.get("F5_volatility"), 0.0)

    confidence_threshold = 65.0
    cost_gate_val = round(max(0.01, (100.0 - edge_after_costs) / 275.0), 3)
    data_quality = bool(entry_price > 0 and confidence > 0)
    confidence_gate = bool(confidence >= confidence_threshold)
    cost_gate = bool(edge_after_costs >= 20.0)
    mtf_alignment = True
    if signal_dir == "LONG" and f1 < 0:
        mtf_alignment = False
    if signal_dir == "SHORT" and f1 > 0:
        mtf_alignment = False
    fear_greed = True
    regime_filter = "HIGH_VOL" not in regime and "HALT" not in regime

    pipeline_output = (
        signal_dir
        if signal_dir in {"LONG", "SHORT"} and data_quality and confidence_gate and cost_gate and mtf_alignment and fear_greed and regime_filter
        else "HOLD"
    )

    trade_rows = _fetch_trade_report_rows(limit=800, ticker=ticker)
    closed_outcomes = {"TP", "TP1", "TP2", "TP3", "SL", "STOPPED", "CLOSED", "TIMEOUT"}
    closed_rows = [r for r in trade_rows if str(r.get("outcome", "")).upper() in closed_outcomes]
    trades_count = len(closed_rows)
    wins = [r for r in closed_rows if _safe_float(r.get("pnl_pct"), 0.0) > 0]
    win_rate = (len(wins) / trades_count * 100.0) if trades_count else 0.0
    rr_for_kelly = max(rr_ratio, 0.1)
    p = win_rate / 100.0
    kelly_f = max(0.0, (p - (1.0 - p) / rr_for_kelly) * 100.0)
    method = "KELLY" if trades_count >= 30 else "COLD START"
    if method == "KELLY":
        size_pct = max(0.25, min(5.0, kelly_f / 10.0))
    else:
        size_pct = max(0.25, min(1.50, (confidence / 100.0) * 1.2))
    size_pct = round(size_pct, 2)
    size_usd = round(size_pct * 10.0, 2)

    strength_bucket = _bucket_from_confidence(confidence)
    dir_bucket = signal_dir if signal_dir in {"LONG", "SHORT"} else "NEUTRAL"
    bucket = f"{dir_bucket}/{strength_bucket}/{regime}"

    return {
        "ticker": str(ticker),
        "signal": signal_dir,
        "confidence": round(confidence, 1),
        "regime": regime,
        "signal_strength": signal_strength,
        "edge_after_costs": edge_after_costs,
        "entry_price": round(entry_price, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "rr_ratio": rr_ratio,
        "reason": reason,
        "gate_status": {
            "data_quality": data_quality,
            "confidence_gate": confidence_gate,
            "cost_gate": cost_gate,
            "mtf_alignment": mtf_alignment,
            "fear_greed": fear_greed,
            "regime_filter": regime_filter,
        },
        "pipeline": {
            "raw_signal": signal_dir,
            "regime_gate": regime,
            "regime_pass": regime_filter,
            "confidence_val": round(confidence, 1),
            "confidence_threshold": confidence_threshold,
            "confidence_pass": confidence_gate,
            "cost_gate_val": cost_gate_val,
            "cost_pass": cost_gate,
            "output": pipeline_output,
        },
        "position_sizing": {
            "size_pct": size_pct,
            "size_usd": size_usd,
            "method": method,
            "kelly_f": round(kelly_f, 1),
            "win_rate": round(win_rate, 1),
            "trades_count": trades_count,
            "rr": rr_ratio,
            "bucket": bucket,
            "algo": "quant_alpha_factor_model_v1",
        },
        "factors": {
            "F1_momentum": round(f1, 3),
            "F2_mean_rev": round(f2, 3),
            "F3_volume": round(f3, 3),
            "F5_volatility": round(f5, 3),
        },
        "as_of_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


@app.get("/api/terminal-signal")
def get_terminal_signal(ticker: str = "RELIANCE.NS"):
    src = _latest_store_signal_for_ticker(ticker)
    if src is None:
        src = get_stock_signal(ticker=ticker)
    if not isinstance(src, dict):
        src = {"ticker": ticker, "signal": "HOLD", "confidence": 0.0}
    return _build_terminal_signal_payload(ticker=ticker, source=src)


@app.get("/api/stock/flow")
def get_stock_flow(ticker: str = "RELIANCE.NS"):
    df = _download_stock_frame(ticker=ticker, interval="15m", period="5d")
    if df is None or len(df) < 20:
        return {
            "ticker": ticker,
            "decision": "NO_TRADE",
            "trend": "NEUTRAL",
            "money_flow": 0.0,
            "smart_money": "NEUTRAL",
            "rsi": 50.0,
            "reason": "Insufficient intraday data",
            "as_of_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }

    view = df.tail(40)
    highs = [float(v) for v in view["High"].tolist()]
    lows = [float(v) for v in view["Low"].tolist()]
    closes = [float(v) for v in view["Close"].tolist()]
    volumes = [float(v) for v in view["Volume"].tolist()]

    mf_num = 0.0
    mf_den = 0.0
    avg_vol = sum(volumes) / max(len(volumes), 1)
    smart_score = 0
    for i in range(len(closes)):
        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        op = float(view["Open"].iloc[i])
        vol = volumes[i]
        span = hi - lo
        if span > 1e-9:
            mfm = ((cl - lo) - (hi - cl)) / span
            mf_num += mfm * vol
            mf_den += vol
        body_ratio = abs(cl - op) / max(span, 1e-9)
        if vol > avg_vol * 1.3 and body_ratio > 0.55:
            smart_score += 1 if cl >= op else -1

    money_flow = (mf_num / mf_den) if mf_den > 0 else 0.0
    rsi = _calc_rsi_from_closes(closes[-20:], period=14)
    if money_flow > 0.05:
        trend = "ACCUMULATION"
    elif money_flow < -0.05:
        trend = "DISTRIBUTION"
    else:
        trend = "NEUTRAL"

    if smart_score >= 2:
        smart_money = "BUYER_DOMINANT"
    elif smart_score <= -2:
        smart_money = "SELLER_DOMINANT"
    else:
        smart_money = "NEUTRAL"

    if trend == "ACCUMULATION" and rsi < 70:
        decision = "FAVOR_LONG"
        reason = "Money flow accumulation with healthy RSI"
    elif trend == "DISTRIBUTION" and rsi > 30:
        decision = "FAVOR_SHORT"
        reason = "Money flow distribution with downside pressure"
    else:
        decision = "NO_TRADE"
        reason = "Flow mixed or edge not strong"

    return {
        "ticker": ticker,
        "decision": decision,
        "trend": trend,
        "money_flow": round(money_flow, 4),
        "smart_money": smart_money,
        "rsi": round(rsi, 2),
        "reason": reason,
        "as_of_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


@app.get("/api/stock/volume")
def get_stock_volume_profile(ticker: str = "RELIANCE.NS"):
    df = _download_stock_frame(ticker=ticker, interval="15m", period="5d")
    if df is None or len(df) < 20:
        return {
            "ticker": ticker,
            "decision": "NO_TRADE",
            "support": 0.0,
            "resistance": 0.0,
            "poc_price": 0.0,
            "distance_from_vwap_pct": 0.0,
            "reason": "Insufficient intraday data",
            "as_of_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }

    view = df.tail(60)
    last_close = _safe_float(view["Close"].iloc[-1], 0.0)
    support = _safe_float(view["Low"].tail(20).min(), 0.0)
    resistance = _safe_float(view["High"].tail(20).max(), 0.0)

    volumes = [float(v) for v in view["Volume"].tolist()]
    closes = [float(v) for v in view["Close"].tolist()]
    total_vol = sum(volumes)
    vwap = (sum(c * v for c, v in zip(closes, volumes)) / total_vol) if total_vol > 0 else last_close
    dist_pct = ((last_close - vwap) / last_close * 100.0) if last_close > 0 else 0.0

    if last_close > vwap and dist_pct > 0.10:
        decision = "FAVOR_LONG"
        reason = "Price holding above VWAP with positive structure"
    elif last_close < vwap and dist_pct < -0.10:
        decision = "FAVOR_SHORT"
        reason = "Price below VWAP with negative structure"
    else:
        decision = "NO_TRADE"
        reason = "Price near fair value (VWAP)"

    return {
        "ticker": ticker,
        "decision": decision,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "poc_price": round(vwap, 2),
        "distance_from_vwap_pct": round(dist_pct, 4),
        "reason": reason,
        "as_of_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


@app.get("/api/stock/volatility")
def get_stock_volatility(ticker: str = "RELIANCE.NS"):
    df = _download_stock_frame(ticker=ticker, interval="15m", period="5d")
    if df is None or len(df) < 20:
        return {
            "ticker": ticker,
            "tradeability": "NO_TRADE",
            "regime": "UNKNOWN",
            "atr_pct": 0.0,
            "reason": "Insufficient intraday data",
            "as_of_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }

    view = df.tail(120).copy()
    highs = view["High"].astype(float).tolist()
    lows = view["Low"].astype(float).tolist()
    closes = view["Close"].astype(float).tolist()

    tr_values: list[float] = []
    prev_close = closes[0] if closes else 0.0
    for hi, lo, cl in zip(highs, lows, closes):
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        tr_values.append(float(tr))
        prev_close = cl

    atr = sum(tr_values[-14:]) / max(len(tr_values[-14:]), 1)
    last_price = closes[-1] if closes else 0.0
    atr_pct = (atr / last_price * 100.0) if last_price > 0 else 0.0

    if atr_pct < 0.10:
        regime = "LOW"
        tradeability = "REDUCE_SIZE"
        reason = "Low volatility; reduce position size"
    elif atr_pct < 0.80:
        regime = "NORMAL"
        tradeability = "ALLOW"
        reason = "Normal tradable volatility"
    elif atr_pct < 1.50:
        regime = "EXPANSION"
        tradeability = "CAUTION"
        reason = "Volatility expansion; trade cautiously"
    else:
        regime = "HIGH_VOL"
        tradeability = "NO_TRADE"
        reason = "Extreme volatility regime"

    return {
        "ticker": ticker,
        "tradeability": tradeability,
        "regime": regime,
        "atr_pct": round(atr_pct, 4),
        "reason": reason,
        "as_of_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


@app.get("/api/signal-quality")
def get_signal_quality(ticker: str = "RELIANCE.NS"):
    """Return composite Signal Quality Score (SQS) for an asset."""
    try:
        import yfinance as yf  # type: ignore[import]
        from src.features.engineer import FeatureEngineer  # type: ignore[import]
        from src.alpha.factor_model import get_cached_alpha_model  # type: ignore[import]
        from src.alpha.signal_quality import SignalQualityScorer  # type: ignore[import]

        df = yf.download(ticker, interval="5m", period="5d", progress=False)
        if df.empty or len(df) < 40:
            df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or len(df) < 40:
            return {"ticker": ticker, "error": "Insufficient data"}

        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

        eng = FeatureEngineer()
        tf = "daily" if len(df) > 100 else "intraday"
        feat_df = eng.compute_all_features(df, timeframe=tf, ticker=ticker)
        if feat_df.empty:
            return {"ticker": ticker, "error": "Feature computation failed"}

        asset_class = "indian_stock" if ticker.upper().endswith((".NS", ".BO")) else "us_stock"
        model = get_cached_alpha_model(
            asset_class=asset_class,
            alpha_threshold=float((_dashboard_cfg.get("signal", {}) or {}).get("alpha_score_threshold", 0.15)),
        )
        result = model.score(feat_df, asset=ticker, asset_class=asset_class)

        atr_pct = float(feat_df.get("ATR_Percentile", 50.0).iloc[-1]) if "ATR_Percentile" in feat_df.columns else 50.0
        sqs_min = float((_dashboard_cfg.get("signal", {}) or {}).get("min_sqs", 55))
        sqs = SignalQualityScorer(min_sqs=sqs_min).score(
            signal=str(result.get("signal", "HOLD")),
            factor_scores=result.get("factor_scores", {}),
            ic_weights=result.get("ic_weights", {}),
            regime=str(result.get("regime_for_confidence", "SIDEWAYS")),
            atr_percentile=atr_pct,
        )

        return {
            "ticker": ticker,
            "asset_class": asset_class,
            "signal": result.get("signal", "HOLD"),
            "alpha_score": result.get("alpha_score", 0.0),
            "confidence": result.get("confidence", 50.0),
            "sqs": sqs.sqs,
            "sqs_passed": sqs.passes,
            "threshold": sqs.threshold,
            "components": sqs.components,
            "factor_scores": result.get("factor_scores", {}),
            "ic_weights": result.get("ic_weights", {}),
            "as_of_utc": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@app.get("/api/signal-audit/{signal_id}")
async def get_signal_audit(signal_id: str):
    """Full factor breakdown + confluence + SQS for one signal."""
    try:
        if _store is None:
            return {"error": "store unavailable"}
        signal = _store.get_signal_by_id(signal_id)
        if not signal:
            return {"error": "signal not found"}
        return {
            "signal_id": signal_id,
            "ticker": signal.get("asset"),
            "direction": signal.get("signal"),
            "alpha_score": signal.get("alpha_score"),
            "factor_breakdown": signal.get("factor_breakdown", {}),
            "confluence_grade": signal.get("confluence_grade"),
            "confluence_pct": signal.get("confluence_pct"),
            "sqs": signal.get("sqs"),
            "sqs_grade": signal.get("sqs_grade"),
            "size_multiplier_used": signal.get("size_multiplier_used"),
            "top_aligned_factors": signal.get("top_aligned_factors", []),
            "top_drag_factors": signal.get("top_drag_factors", []),
        }
    except Exception:
        return {"error": "audit unavailable"}


@app.get("/api/signal-quality-stats")
async def get_signal_quality_stats():
    """Aggregated SQS stats for last 7 days."""
    try:
        if _store is None:
            return {}
        return _store.get_signal_quality_stats(days=7)
    except Exception:
        return {}


@app.get("/api/chart-markers")
def get_chart_markers(ticker: str = "RELIANCE.NS", interval: str = "5m", period: str = "1d"):
    """
    Computes historical BUY/SELL signals over the chart period so they can be 
    plotted as markers on the frontend Lightweight chart.
    """
    try:
        import yfinance as yf  # type: ignore[import]
        from src.features.engineer import FeatureEngineer  # type: ignore[import]
        
        fetch_period = "5d" if (interval == "5m" and period == "1d") else period
        df = yf.download(ticker, interval=interval, period=fetch_period, progress=False)
        if df.empty or len(df) < 30:
            return []

        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)

        eng = FeatureEngineer()
        feat_df = eng.compute_all_features(df, timeframe="intraday" if "m" in interval or "h" in interval else "daily")
        if feat_df.empty: return []

        # We will do a fast vectorised combination of the primary momentum + mean rev factors
        # since AlphaFactorModel.score() is point-in-time for the latest bar.
        # This provides the visual representation of the quant logic on the chart.
        roc = feat_df.get("ROC_5d", feat_df["Close"].pct_change(5))
        sma = feat_df.get("SMA_20", feat_df["Close"].rolling(20).mean())
        
        markers = []
        current_pos = 0 # 1 for buy, -1 for sell
        
        for idx in range(20, len(feat_df)):
            close = float(feat_df["Close"].iloc[idx])
            sma_val = float(sma.iloc[idx])
            roc_val = float(roc.iloc[idx])
            
            # Simple combined proxy of the quant logic for historical visual markers
            # (Momentum > 0 and Price > SMA)
            if roc_val > 0.5 and close > sma_val * 1.002:
                sig = 1
            elif roc_val < -0.5 and close < sma_val * 0.998:
                sig = -1
            else:
                sig = 0
                
            if sig == 1 and current_pos <= 0:
                current_pos = 1
                ts = feat_df.index[idx]
                markers.append({
                    "time": int(ts.timestamp()) if hasattr(ts, 'timestamp') else str(ts),
                    "position": "belowBar", 
                    "color": "#34d399", 
                    "shape": "arrowUp", 
                    "text": "BUY"
                })
            elif sig == -1 and current_pos >= 0:
                current_pos = -1
                ts = feat_df.index[idx]
                markers.append({
                    "time": int(ts.timestamp()) if hasattr(ts, 'timestamp') else str(ts),
                    "position": "aboveBar", 
                    "color": "#f87171", 
                    "shape": "arrowDown", 
                    "text": "SELL"
                })
                
        return markers
    except Exception as e:
        logger.error(f"Marker error: {e}")
        return []

@app.get("/api/market-overview")
def get_market_overview():
    """Quick snapshot of NIFTY 50, NIFTY Bank, and key metrics."""
    import yfinance as yf  # type: ignore[import]

    results: Dict[str, Dict[str, Any]] = {}
    indices = {"NIFTY_50": "^NSEI", "NIFTY_BANK": "^NSEBANK", "SENSEX": "^BSESN", "INDIA_VIX": "^INDIAVIX"}
    for name, ticker in indices.items():
        try:
            info = yf.Ticker(ticker)
            hist = info.history(period="2d")
            if len(hist) >= 2:
                prev: float = float(hist["Close"].iloc[-2])
                curr: float = float(hist["Close"].iloc[-1])
                change_pct: float = _round2((curr - prev) / prev * 100)
                results[name] = {"price": _round2(curr), "change_pct": change_pct}
            elif len(hist) == 1:
                results[name] = {"price": _round2(hist["Close"].iloc[-1]), "change_pct": 0.0}
            else:
                results[name] = {"price": 0, "change_pct": 0}
        except Exception:
            results[name] = {"price": 0, "change_pct": 0}
    return results


@app.get("/api/prices/batch")
async def get_batch_prices(tickers: str):
    """
    Get current prices for multiple tickers.
    Query: tickers=AAPL,RELIANCE.NS,BTCUSDT
    """
    import requests  # type: ignore[import]
    import yfinance as yf  # type: ignore[import]

    ticker_list = [t.strip().upper() for t in str(tickers or "").split(",") if t.strip()]
    result: Dict[str, Union[float, None]] = {}
    for ticker in ticker_list[:20]:
        try:
            if "USDT" in ticker:
                r = requests.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": ticker},
                    timeout=5,
                )
                if r.ok:
                    payload = r.json()
                    result[ticker] = float(payload.get("price", 0.0) or 0.0)
                else:
                    result[ticker] = None
            else:
                hist = yf.Ticker(ticker).history(period="1d", interval="1m")
                if not hist.empty:
                    result[ticker] = float(hist["Close"].iloc[-1])
                else:
                    # fallback to latest daily close
                    hist_d = yf.Ticker(ticker).history(period="5d", interval="1d")
                    result[ticker] = float(hist_d["Close"].iloc[-1]) if not hist_d.empty else None
        except Exception:
            result[ticker] = None
    return result

@app.get("/api/signals")
def get_signals(limit: int = 50):
    lim = max(1, min(int(limit), 1000))
    return _combined_signal_rows(limit=lim)


@app.get("/api/portfolio")
def get_portfolio():
    if _portfolio is None:
        return {}
    return {
        "metrics": _portfolio.get_metrics(),
        "positions": _portfolio.get_open_positions_list(),
    }


# ------------------------------------------------------------------
# Paper Trading Endpoints
# ------------------------------------------------------------------

_PAPER_DEFAULT_CAPITAL = 10000.0


def _paper_fmt_duration(seconds: float) -> str:
    total = max(0, int(round(_safe_float(seconds, 0.0))))
    if total < 60:
        return f"{total}s"
    mins, sec = divmod(total, 60)
    if mins < 60:
        return f"{mins}m {sec}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m"


def _paper_result_label(reason: Any) -> str:
    r = str(reason or "").strip().lower()
    if not r:
        return "CLOSED"
    if "tp2" in r:
        return "TP2"
    if "tp1" in r or "tp_hit" in r or r == "tp":
        return "TP1"
    if "timeout" in r:
        return "TIMEOUT"
    if "sl" in r or "stop" in r:
        return "SL"
    if "close" in r or "manual" in r:
        return "CLOSED"
    return str(reason).upper()


def _paper_session_bucket(ts_any: Any) -> str:
    dt = _parse_utc(ts_any)
    if dt is None:
        return "asia"
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)
    hour = int(dt_utc.hour)
    if 13 <= hour < 21:
        return "ny"
    if 8 <= hour < 13:
        return "london"
    return "asia"


def _paper_extract_price(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("mark_price", "price", "current_price", "last_price", "entry_price"):
        px = _safe_float(payload.get(key), 0.0)
        if px > 0:
            return float(px)
    deriv = payload.get("derivatives")
    if isinstance(deriv, dict):
        px = _safe_float(deriv.get("mark_price"), 0.0)
        if px > 0:
            return float(px)
    return None


def _latest_btc_price_for_paper(default: float = 0.0) -> float:
    # Prefer hot local cache from btc_intelligence redis mirrors.
    for redis_key in ("btc:signal", "btc:signal:persistent", "btc:execution", "btc:intelligence"):
        obj = _redis_get_json(redis_key)
        if not isinstance(obj, dict):
            continue
        normalized = _normalize_dashboard_signal_payload(obj) if "signal" in obj else obj
        px = _paper_extract_price(normalized)
        if px and px > 0:
            return float(px)

    # Fallback: best effort from local signal store.
    try:
        src = _latest_store_signal_for_ticker("BTCUSDT")
        px = _paper_extract_price(src)
        if px and px > 0:
            return float(px)
    except Exception:
        pass

    # Final fallback: direct ticker call.
    try:
        resp = httpx.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            timeout=4.0,
        )
        if resp.status_code == 200:
            payload = resp.json()
            px = _safe_float(payload.get("price"), 0.0)
            if px > 0:
                return float(px)
    except Exception:
        pass
    return float(default)


def _paper_suggested_size() -> float:
    # Return "size %" from the same kelly/risk signal payload used by terminal.
    sig = _redis_get_json("btc:signal") or _redis_get_json("btc:signal:persistent")
    if isinstance(sig, dict):
        norm = _normalize_dashboard_signal_payload(sig)
        ps = norm.get("position_sizing")
        if isinstance(ps, dict):
            for key in ("size_pct", "position_size_pct", "position_pct"):
                v = _safe_float(ps.get(key), -1.0)
                if v >= 0:
                    return float(round(v, 4))
    # Fallback to configured cold start risk budget.
    try:
        risk_cfg = (_dashboard_cfg.get("risk") or {})
        cold = _safe_float(risk_cfg.get("cold_start_position_pct"), 1.0)
        return float(round(max(0.1, cold), 4))
    except Exception:
        return 1.0


def _paper_portfolio_payload(engine) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    metrics = engine.get_portfolio_metrics() or {}
    open_positions = engine.get_open_positions() or []
    closed = engine.get_closed_trades(5000) or []

    capital = _safe_float(metrics.get("capital"), 0.0)
    initial = _safe_float(metrics.get("initial_capital"), _PAPER_DEFAULT_CAPITAL)
    total_pnl = _safe_float(metrics.get("total_pnl"), 0.0)
    total_pnl_pct = _safe_float(metrics.get("total_pnl_pct"), 0.0)
    win_rate = _safe_float(metrics.get("win_rate"), 0.0)
    total_trades = int(metrics.get("total_trades") or len(closed) or 0)
    sharpe = _safe_float(metrics.get("sharpe_ratio", metrics.get("sharpe")), 0.0)
    max_drawdown = _safe_float(metrics.get("max_drawdown"), 0.0)
    profit_factor = _safe_float(metrics.get("profit_factor"), 0.0)

    now_utc = datetime.now(timezone.utc)
    today_key = now_utc.strftime("%Y-%m-%d")
    today_pnl = 0.0
    dur_seconds_list: list[float] = []
    session_acc = {
        "asia": {"pnl_pct": 0.0, "trades": 0},
        "london": {"pnl_pct": 0.0, "trades": 0},
        "ny": {"pnl_pct": 0.0, "trades": 0},
    }

    for tr in closed:
        if not isinstance(tr, dict):
            continue
        pnl_val = _safe_float(tr.get("pnl"), 0.0)
        pnl_pct_val = _safe_float(tr.get("pnl_pct"), 0.0)
        close_ts = tr.get("closed_at") or tr.get("exit_time") or tr.get("closed_time")
        close_dt = _parse_utc(close_ts)
        if close_dt is not None:
            dt_utc = close_dt.astimezone(timezone.utc) if close_dt.tzinfo else close_dt.replace(tzinfo=timezone.utc)
            if dt_utc.strftime("%Y-%m-%d") == today_key:
                today_pnl += pnl_val
        held_hours = _safe_float(tr.get("held_hours"), 0.0)
        if held_hours > 0:
            dur_seconds_list.append(held_hours * 3600.0)

        bucket = _paper_session_bucket(tr.get("opened_at") or tr.get("entry_time") or close_ts)
        if bucket not in session_acc:
            bucket = "asia"
        session_acc[bucket]["trades"] += 1
        session_acc[bucket]["pnl_pct"] += pnl_pct_val

    avg_duration_sec = (sum(dur_seconds_list) / len(dur_seconds_list)) if dur_seconds_list else 0.0
    avg_duration = _paper_fmt_duration(avg_duration_sec)
    today_pnl_pct = (today_pnl / initial * 100.0) if initial > 0 else 0.0

    live_btc_px = _latest_btc_price_for_paper(default=0.0)
    open_payload: list[dict[str, Any]] = []
    for pos in open_positions:
        if not isinstance(pos, dict):
            continue
        ticker = str(pos.get("ticker") or "BTCUSDT").upper()
        direction = _normalize_direction(pos.get("direction") or "LONG")
        entry = _safe_float(pos.get("entry_price"), 0.0)
        qty = _safe_float(pos.get("quantity"), 0.0)
        current = _safe_float(pos.get("current_price"), entry)
        if ticker in {"BTCUSDT", "BTC-USD", "XBTUSD"} and live_btc_px > 0:
            current = live_btc_px
        if current <= 0:
            current = entry
        if qty <= 0 and entry > 0:
            value = _safe_float(pos.get("value"), 0.0)
            qty = (value / entry) if value > 0 else 0.0

        pnl_usd = (current - entry) * qty if direction == "LONG" else (entry - current) * qty
        base = entry * qty
        pnl_pct = (pnl_usd / base * 100.0) if base > 0 else _safe_float(pos.get("unrealized_pnl_pct"), 0.0)

        duration_sec = _safe_float(pos.get("held_hours"), 0.0) * 3600.0
        if duration_sec <= 0:
            opened_dt = _parse_utc(pos.get("opened_at"))
            if opened_dt is not None:
                opened_utc = opened_dt.astimezone(timezone.utc) if opened_dt.tzinfo else opened_dt.replace(tzinfo=timezone.utc)
                duration_sec = max(0.0, (now_utc - opened_utc).total_seconds())

        open_payload.append(
            {
                "id": str(pos.get("trade_id") or pos.get("id") or ticker),
                "ticker": ticker,
                "direction": direction if direction in {"LONG", "SHORT"} else "LONG",
                "entry_price": float(round(entry, 6)),
                "current_price": float(round(current, 6)),
                "pnl_usd": float(round(pnl_usd, 6)),
                "pnl_pct": float(round(pnl_pct, 6)),
                "sl": float(round(_safe_float(pos.get("stop_loss"), 0.0), 6)),
                "tp1": float(round(_safe_float(pos.get("take_profit"), 0.0), 6)),
                "duration_str": _paper_fmt_duration(duration_sec),
            }
        )

    session_payload = {
        "asia": {
            "pnl_pct": float(round(session_acc["asia"]["pnl_pct"], 6)),
            "trades": int(session_acc["asia"]["trades"]),
        },
        "london": {
            "pnl_pct": float(round(session_acc["london"]["pnl_pct"], 6)),
            "trades": int(session_acc["london"]["trades"]),
        },
        "ny": {
            "pnl_pct": float(round(session_acc["ny"]["pnl_pct"], 6)),
            "trades": int(session_acc["ny"]["trades"]),
        },
    }

    return {
        "capital": float(round(capital, 6)),
        "initial_capital": float(round(initial, 6)),
        "total_pnl": float(round(total_pnl, 6)),
        "total_pnl_pct": float(round(total_pnl_pct, 6)),
        "today_pnl": float(round(today_pnl, 6)),
        "today_pnl_pct": float(round(today_pnl_pct, 6)),
        "win_rate": float(round(win_rate, 6)),
        "total_trades": int(total_trades),
        "avg_duration": avg_duration,
        "sharpe": float(round(sharpe, 6)),
        "max_drawdown": float(round(max_drawdown, 6)),
        "profit_factor": float(round(profit_factor, 6)),
        "session_pnl": session_payload,
        "suggested_size": float(round(_paper_suggested_size(), 6)),
        "open_positions": open_payload,
    }


@app.get("/api/paper/portfolio")
def paper_portfolio():
    """Paper account state in terminal-compatible shape."""
    try:
        from src.paper_trading import get_paper_engine

        engine = get_paper_engine()
        return _paper_portfolio_payload(engine)
    except Exception as e:
        logger.error("Paper portfolio error: %s", e)
        return {
            "capital": float(_PAPER_DEFAULT_CAPITAL),
            "initial_capital": float(_PAPER_DEFAULT_CAPITAL),
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "today_pnl": 0.0,
            "today_pnl_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "avg_duration": "0s",
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "session_pnl": {
                "asia": {"pnl_pct": 0.0, "trades": 0},
                "london": {"pnl_pct": 0.0, "trades": 0},
                "ny": {"pnl_pct": 0.0, "trades": 0},
            },
            "suggested_size": 1.0,
            "open_positions": [],
        }


@app.get("/api/paper/trades")
def paper_trades(limit: int = 10):
    """Recent closed trades in compact terminal shape."""
    from src.paper_trading import get_paper_engine

    lim = max(1, min(int(limit), 200))
    rows = get_paper_engine().get_closed_trades(lim)
    out: list[dict[str, Any]] = []
    for tr in rows:
        if not isinstance(tr, dict):
            continue
        raw_time = tr.get("closed_at") or tr.get("exit_time") or tr.get("closed_time") or tr.get("opened_at")
        dt = _parse_utc(raw_time)
        if dt is None:
            time_str = str(raw_time or "")
        else:
            dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            time_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        out.append(
            {
                "id": str(tr.get("trade_id") or tr.get("id") or uuid4()),
                "time": time_str,
                "ticker": str(tr.get("ticker") or "BTCUSDT").upper(),
                "direction": _normalize_direction(tr.get("direction")),
                "entry_price": float(round(_safe_float(tr.get("entry_price"), 0.0), 6)),
                "exit_price": float(round(_safe_float(tr.get("exit_price"), 0.0), 6)),
                "pnl_usd": float(round(_safe_float(tr.get("pnl"), 0.0), 6)),
                "pnl_pct": float(round(_safe_float(tr.get("pnl_pct"), 0.0), 6)),
                "result": _paper_result_label(tr.get("reason")),
            }
        )
    return out


@app.post("/api/paper/execute")
async def paper_execute(payload: dict):
    """Manual paper execution. Body: { ticker, direction, entry_price, mode }."""
    from src.paper_trading import get_paper_engine

    body = payload or {}
    ticker = str(body.get("ticker") or "BTCUSDT").strip().upper()
    direction = _normalize_direction(body.get("direction") or body.get("signal"))
    mode = str(body.get("mode") or "manual").strip().lower() or "manual"
    if direction not in {"LONG", "SHORT"}:
        raise HTTPException(status_code=400, detail="direction must be LONG or SHORT")

    entry_price = _safe_float(body.get("entry_price"), 0.0)
    if entry_price <= 0:
        entry_price = _latest_btc_price_for_paper(default=0.0)
    if entry_price <= 0:
        raise HTTPException(status_code=400, detail="Valid entry_price is required")

    atr_pct = 0.0
    vol_payload = _redis_get_json("btc:volatility")
    if isinstance(vol_payload, dict):
        atr_pct = _safe_float(vol_payload.get("atr_pct"), 0.0)
    if atr_pct <= 0:
        atr_pct = _safe_float(body.get("atr_pct"), 0.0)
    if atr_pct <= 0:
        # fixed fallback if ATR unavailable
        atr_pct = 1.0

    atr_abs = entry_price * (atr_pct / 100.0)
    risk_dist = max(entry_price * 0.003, 1.5 * atr_abs)
    if direction == "LONG":
        sl = entry_price - risk_dist
        tp = entry_price + risk_dist
    else:
        sl = entry_price + risk_dist
        tp = entry_price - risk_dist
    if sl <= 0:
        raise HTTPException(status_code=400, detail="Could not compute stop loss")
    if tp <= 0:
        tp = entry_price * (1.01 if direction == "LONG" else 0.99)

    signal = {
        "ticker": ticker,
        "signal": direction,
        "entry_price": float(round(entry_price, 6)),
        "stop_loss": float(round(sl, 6)),
        "take_profit": float(round(tp, 6)),
        "confidence": _safe_float(body.get("confidence"), 70.0),
        "asset_class": str(body.get("asset_class") or "crypto"),
        "strength": str(body.get("strength") or "MODERATE"),
    }

    engine = get_paper_engine()
    result = engine.execute_trade(signal, mode=mode)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=str(result.get("message") or "Trade rejected"))

    size = _safe_float(result.get("quantity"), 0.0)
    push_broadcast_threadsafe(
        {
            "type": "paper_trade_update",
            "action": "opened",
            "ticker": ticker,
            "pnl_usd": 0.0,
        }
    )
    return {"success": True, "trade": result, "size": float(round(size, 6))}


@app.post("/api/paper/close")
async def paper_close(payload: dict):
    """Close open paper position. Body: { ticker, exit_price }."""
    from src.paper_trading import get_paper_engine

    body = payload or {}
    engine = get_paper_engine()
    ticker = str(body.get("ticker", "")).strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    exit_price = _safe_float(body.get("exit_price"), 0.0)
    if exit_price <= 0:
        exit_price = _latest_btc_price_for_paper(default=0.0)
    if exit_price <= 0:
        raise HTTPException(status_code=400, detail="Valid exit_price is required")

    result = engine.close_position(ticker, float(exit_price), str(body.get("reason") or "manual"))
    if not result:
        raise HTTPException(status_code=404, detail="No open position for ticker")

    pnl_usd = _safe_float(result.get("pnl"), 0.0)
    pnl_pct = _safe_float(result.get("pnl_pct"), 0.0)
    res_label = _paper_result_label(result.get("reason"))
    push_broadcast_threadsafe(
        {
            "type": "paper_trade_update",
            "action": "closed",
            "ticker": ticker,
            "pnl_usd": float(round(pnl_usd, 6)),
        }
    )
    return {
        "success": True,
        "pnl_usd": float(round(pnl_usd, 6)),
        "pnl_pct": float(round(pnl_pct, 6)),
        "result": res_label,
    }


@app.post("/api/paper/mode")
async def paper_set_mode(payload: dict):
    """
    Set auto/manual mode.
    Body: { mode: "auto" | "manual" }
    """
    import threading

    from src.paper_trading import get_auto_executor, get_paper_engine

    mode = str(payload.get("mode", "manual")).lower()
    mode = "auto" if mode == "auto" else "manual"
    engine = get_paper_engine()
    engine.set_mode(mode)

    executor = get_auto_executor()
    if mode == "auto" and not executor._running:
        t = threading.Thread(target=executor.start, daemon=True)
        t.start()
    elif mode == "manual":
        executor.stop()

    return {"mode": mode, "success": True}


@app.post("/api/paper/reset")
async def paper_reset(payload: dict | None = None):
    """Reset paper account. Body: { capital: float }"""
    from src.paper_trading import get_paper_engine

    body = payload or {}
    capital = _safe_float(body.get("capital"), _PAPER_DEFAULT_CAPITAL)
    if capital <= 0:
        capital = _PAPER_DEFAULT_CAPITAL
    get_paper_engine().reset(capital)
    return {"success": True, "capital": float(round(capital, 6))}


@app.get("/api/paper/pending")
def paper_pending():
    """Get signals waiting for manual approval."""
    from src.paper_trading import get_auto_executor

    return get_auto_executor().get_pending_signals()


@app.post("/api/paper/approve")
async def paper_approve(payload: dict):
    """Approve a pending signal. Body: { ticker, trade_id }"""
    from src.paper_trading import get_auto_executor

    return get_auto_executor().approve_signal(payload["ticker"], payload["trade_id"])


@app.post("/api/paper/reject")
async def paper_reject(payload: dict):
    """Reject a pending signal. Body: { ticker, trade_id }"""
    from src.paper_trading import get_auto_executor

    get_auto_executor().reject_signal(payload["ticker"], payload["trade_id"])
    return {"success": True}


@app.get("/api/history")
def get_history(limit: int = 100, tab: str = "all", ticker: str | None = None):
    lim = max(1, min(int(limit), 1000))
    rows = _fetch_trade_report_rows(limit=lim, ticker=ticker)
    tab_norm = str(tab or "all").strip().lower()
    if tab_norm in {"tp", "takeprofit", "take_profit"}:
        allowed = {"TP1", "TP2", "TP"}
        return [r for r in rows if str(r.get("outcome", "")).upper() in allowed]
    if tab_norm in {"sl", "stop", "stoploss", "stop_loss"}:
        allowed = {"SL", "STOPPED"}
        return [r for r in rows if str(r.get("outcome", "")).upper() in allowed]
    if tab_norm in {"active", "open"}:
        return [r for r in rows if str(r.get("outcome", "")).upper() == "OPEN"]
    # all: intentionally unfiltered to include TIMEOUT and all terminal outcomes.
    return rows


@app.get("/api/equity-curve")
def get_equity_curve(limit: int = 5000, ticker: str | None = None):
    lim = max(1, min(int(limit), 20000))
    rows = _fetch_trade_report_rows(limit=lim, ticker=ticker)
    closed_outcomes = {"TP1", "TP2", "TP3", "SL", "CLOSED"}
    closed = [r for r in rows if str(r.get("outcome", "")).upper() in closed_outcomes]
    closed.sort(key=lambda x: str(x.get("exit_time") or x.get("entry_time") or ""))

    pnl_values = [float(r.get("pnl_pct") or 0.0) for r in closed]
    timestamps = [str(r.get("exit_time") or r.get("entry_time") or "") for r in closed]

    cumulative_pnl: list[float] = []
    drawdown: list[float] = []
    running = 0.0
    peak = 0.0
    for pnl in pnl_values:
        running += pnl
        peak = max(peak, running)
        cumulative_pnl.append(round(running, 4))
        drawdown.append(round(running - peak, 4))

    total_trades = len(pnl_values)
    wins = [x for x in pnl_values if x > 0]
    losses = [x for x in pnl_values if x < 0]
    total_pnl = round(sum(pnl_values), 4)
    max_drawdown = round(min(drawdown), 4) if drawdown else 0.0
    win_rate = (len(wins) / total_trades) if total_trades else 0.0

    mean_ret = (sum(pnl_values) / total_trades) if total_trades else 0.0
    std_ret = 0.0
    if total_trades > 1:
        var = sum((x - mean_ret) ** 2 for x in pnl_values) / (total_trades - 1)
        std_ret = math.sqrt(max(var, 0.0))
    # Trade-count Sharpe (not time-annualized) for irregular high-frequency trade durations.
    sharpe_raw = (mean_ret / std_ret) * math.sqrt(total_trades) if (std_ret > 0 and total_trades > 0) else 0.0
    sharpe = max(-5.0, min(5.0, sharpe_raw))

    sum_wins = sum(wins)
    sum_losses_abs = abs(sum(losses))
    if sum_losses_abs > 0:
        profit_factor = sum_wins / sum_losses_abs
    else:
        profit_factor = sum_wins if sum_wins > 0 else 0.0

    avg_duration_seconds = (
        sum(float(r.get("duration_seconds") or 0.0) for r in closed) / total_trades
        if total_trades
        else 0.0
    )
    open_positions = sum(1 for r in rows if str(r.get("outcome", "")).upper() == "OPEN")

    sessions = {
        "asia": {"name": "Asia Session", "trades": 0, "wins": 0, "pnl_pct": 0.0},
        "london": {"name": "London Session", "trades": 0, "wins": 0, "pnl_pct": 0.0},
        "ny": {"name": "NY Session", "trades": 0, "wins": 0, "pnl_pct": 0.0},
    }

    for row in closed:
        dt = _parse_utc(row.get("entry_time"))
        hour = dt.hour if dt is not None else 0
        if 13 <= hour < 21:
            bucket = "ny"
        elif 8 <= hour < 16:
            bucket = "london"
        elif 0 <= hour < 8:
            bucket = "asia"
        else:
            bucket = "asia"
        sessions[bucket]["trades"] += 1
        pnl = float(row.get("pnl_pct") or 0.0)
        sessions[bucket]["pnl_pct"] += pnl
        if pnl > 0:
            sessions[bucket]["wins"] += 1

    session_payload = {}
    for key, payload in sessions.items():
        trades = int(payload["trades"])
        wins_count = int(payload["wins"])
        session_payload[key] = {
            "name": payload["name"],
            "trades": trades,
            "win_rate": (wins_count / trades) if trades else 0.0,
            "pnl_pct": round(float(payload["pnl_pct"]), 4),
        }

    return {
        "timestamps": timestamps,
        "cumulative_pnl": cumulative_pnl,
        "drawdown": drawdown,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "win_rate": round(win_rate, 6),
        "total_trades": total_trades,
        "sharpe_ratio": round(sharpe, 4),
        "avg_win": round((sum(wins) / len(wins)) if wins else 0.0, 4),
        "avg_loss": round((sum(losses) / len(losses)) if losses else 0.0, 4),
        "profit_factor": round(profit_factor, 4),
        "best_trade": round(max(pnl_values), 4) if pnl_values else 0.0,
        "worst_trade": round(min(pnl_values), 4) if pnl_values else 0.0,
        "avg_duration_seconds": int(avg_duration_seconds),
        "open_positions": int(open_positions),
        "sessions": session_payload,
    }


@app.get("/api/snapshot")
def get_snapshot():
    if _store is None:
        return {}
    return _store.get_latest_snapshot() or {}


@app.post("/auth/token")
async def issue_dashboard_token(payload: dict):
    cfg = _auth_config()
    username = payload.get("username", "")
    password = payload.get("password", "")
    if username != cfg.get("username") or password != cfg.get("password"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": _issue_token(username), "token_type": "bearer"}


@app.get("/api/factors")
def get_factors(limit: int = 100):
    """
    Returns per-asset IC weights and factor scores from recent signals.
    Used by the Factor Analysis dashboard page.
    """
    if _store is None:
        return []
    signals = _store.get_recent_signals(limit=limit)
    result = []
    for s in signals:
        try:
            factor_scores = json.loads(s["factor_scores"]) if isinstance(s.get("factor_scores"), str) else (s.get("factor_scores") or {})
            ic_weights = json.loads(s["ic_weights"]) if isinstance(s.get("ic_weights"), str) else (s.get("ic_weights") or {})
        except (json.JSONDecodeError, TypeError):
            factor_scores = {}
            ic_weights = {}
        result.append({
            "asset": s.get("asset"),
            "asset_class": s.get("asset_class"),
            "timestamp": s.get("timestamp"),
            "hurst_exponent": s.get("hurst_exponent"),
            "alpha_score": s.get("alpha_score"),
            "confidence": s.get("confidence"),
            "factor_scores": factor_scores,
            "ic_weights": ic_weights,
        })
    return result


@app.get("/api/regime")
def get_regime():
    """
    Returns the most recent regime state per asset class.
    Derived from the latest signals in the store.
    Used by the Regime Monitor dashboard page.
    """
    if _store is None:
        return {}
    signals = _store.get_recent_signals(limit=200)
    # Find latest regime per asset_class
    regimes: dict = {}
    for s in signals:
        ac = s.get("asset_class", "unknown")
        if ac not in regimes:
            regimes[ac] = {
                "asset_class": ac,
                "regime": s.get("regime", "SIDEWAYS"),
                "timestamp": s.get("timestamp"),
                "assets": {},
            }
        asset = s.get("asset", "")
        if asset and asset not in regimes[ac]["assets"]:
            regimes[ac]["assets"][asset] = {
                "regime": s.get("regime"),
                "hurst_exponent": s.get("hurst_exponent"),
                "alpha_score": s.get("alpha_score"),
                "timestamp": s.get("timestamp"),
            }
    return list(regimes.values())


@app.get("/api/orders")
def get_orders(limit: int = 50):
    if _store is None:
        return []
    return _store.get_recent_orders(limit=limit)


@app.get("/api/data-quality")
def get_data_quality(limit: int = 50):
    if _store is None:
        return []
    return _store.get_recent_data_quality_events(limit=limit)


@app.get("/api/reconciliation")
def get_reconciliation(limit: int = 50):
    if _store is None:
        return []
    return _store.get_recent_reconciliation_events(limit=limit)


@app.get("/api/model-validation")
def get_model_validation(limit: int = 20):
    if _store is None:
        return []
    return _store.get_recent_model_validation(limit=limit)


@app.get("/api/system-health")
def get_system_health(limit: int = 50):
    if _store is None:
        return []
    return _store.get_recent_system_health(limit=limit)


# ------------------------------------------------------------------
# Step 9 â€” Real-Time Health Monitor
# ------------------------------------------------------------------

@app.get("/api/health")
def get_health():
    """
    Real-time system health dashboard data.
    Reports: Sharpe decay, factor IC breakdown, drawdown speed, status.
    """
    import time

    result: Dict[str, Any] = {
        "status": "HEALTHY",
        "uptime_seconds": int(time.time()),
        "pipeline": {
            "signal_store": _store is not None,
            "portfolio_tracker": _portfolio is not None,
        },
    }

    if _portfolio is not None:
        try:
            metrics = _portfolio.get_metrics()
            sharpe_all_val: float = float(metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0)))
            equity = metrics.get("equity_curve", [])

            # Rolling 30-day Sharpe
            sharpe_30d_val: float = 0.0
            if len(equity) >= 30:
                recent = equity[-30:]
                import numpy as np  # type: ignore[import]
                returns = np.diff(recent) / np.array(recent[:-1])
                sharpe_30d_val = float(np.mean(returns) / max(np.std(returns), 1e-10)) * float(np.sqrt(252))

            result["sharpe"] = {
                "all_time": _round4(sharpe_all_val),
                "rolling_30d": _round4(sharpe_30d_val),
                "decaying": sharpe_30d_val < sharpe_all_val * 0.7 if sharpe_all_val > 0 else False,
            }

            dd_current: float = float(metrics.get("current_drawdown_pct", 0.0))
            dd_max: float = float(metrics.get("max_drawdown_pct", metrics.get("max_drawdown", 0.0)))
            result["drawdown"] = {
                "current_pct": _round2(dd_current),
                "max_pct": _round2(dd_max),
                "circuit_breaker_active": dd_current < -5.0,
            }
        except Exception:
            result["sharpe"] = {"all_time": 0, "rolling_30d": 0, "decaying": False}
            result["drawdown"] = {"current_pct": 0, "max_pct": 0, "circuit_breaker_active": False}

    if _store is not None:
        try:
            signals = _store.get_recent_signals(limit=100)
            # Factor IC breakdown from latest signals
            factor_ics = {}  # type: ignore
            signals_list: list = list(signals)
            for s in signals_list[:20]:  # type: ignore[index]
                try:
                    raw_ic = s.get("ic_weights", {})
                    parsed: Any = json.loads(raw_ic) if isinstance(raw_ic, str) else raw_ic
                    ic_dict: Dict[str, Any] = dict(parsed) if parsed else {}
                    for f_name, f_val in ic_dict.items():
                        v_float: float = float(f_val)
                        if f_name not in factor_ics:
                            factor_ics[f_name] = []  # type: ignore
                        factor_ics[f_name].append(v_float)  # type: ignore
                except Exception as exc:
                    logger.warning("factor_ic row parse skipped: %s", exc)
            result["factor_ic"] = {  # type: ignore[call-overload]
                f: _round4(sum(vals) / max(len(vals), 1))
                for f, vals in factor_ics.items()
            }
            result["signals_24h"] = len(signals)
        except Exception as exc:
            logger.warning("portfolio factor_ic block failed: %s", exc)
            result["factor_ic"] = {}
            result["signals_24h"] = 0

    # Health verdict
    sharpe_info: Any = result.get("sharpe", {})
    drawdown_info: Any = result.get("drawdown", {})
    sharpe_30: float = float(sharpe_info.get("rolling_30d", 0) if isinstance(sharpe_info, dict) else 0)
    dd: float = float(drawdown_info.get("current_pct", 0) if isinstance(drawdown_info, dict) else 0)
    if dd < -5.0:
        result["status"] = "CRITICAL â€” Drawdown Circuit Breaker Active"
    elif isinstance(sharpe_info, dict) and sharpe_info.get("decaying", False):
        result["status"] = "WARNING â€” Sharpe Decay Detected"
    elif sharpe_30 < 0:
        result["status"] = "DEGRADED â€” Negative Rolling Sharpe"

    return result


# ------------------------------------------------------------------
# Live Capital Validation + Latency
# ------------------------------------------------------------------

@app.get("/api/live-validation")
def live_validation():
    """Paper -> Live graduation status and performance metrics."""
    if _store is None:
        raise HTTPException(503, "Store not initialised")

    if _live_runner is not None and hasattr(_live_runner, "get_live_validation_report"):
        try:
            return _live_runner.get_live_validation_report()
        except Exception as exc:
            logger.warning("Live validation fetch failed from runner: %s", exc)

    # Fallback structure for dashboard-only mode
    return {
        "graduation": {
            "stage_name": "Paper Validation",
            "stage_index": 0,
            "capital_pct": 0.0,
            "can_advance": False,
            "stages": [
                {"name": "Paper Validation", "capital_pct": 0.0, "status": "active"},
                {"name": "Seed Capital", "capital_pct": 0.10, "status": "pending"},
                {"name": "Quarter Capital", "capital_pct": 0.25, "status": "pending"},
                {"name": "Half Capital", "capital_pct": 0.50, "status": "pending"},
                {"name": "Full Capital", "capital_pct": 1.00, "status": "pending"},
            ],
        },
        "performance": {
            "days": 0,
            "sharpe": 0.0,
            "cumulative_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
        },
        "as_of_utc": "",
        "mode": "dashboard_only",
    }


@app.get("/api/latency")
def latency_report():
    """Latest async pipeline latency breakdown."""
    if _live_runner is not None and hasattr(_live_runner, "get_latency_report"):
        try:
            return _live_runner.get_latency_report()
        except Exception as exc:
            logger.warning("Latency fetch failed from runner: %s", exc)

    # Fallback for dashboard-only mode
    return {
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


# ------------------------------------------------------------------
# Inbound webhooks (HMAC in JSON body; no dashboard JWT)
# ------------------------------------------------------------------


async def _handle_webhook_ingress(request: Request, default_source: str) -> JSONResponse:
    if not _webhook_enabled or _webhook_receiver is None:
        return JSONResponse({"error": "webhook disabled"}, status_code=503)

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        logger.warning("Webhook rate limit exceeded for IP: %s", client_ip)
        return JSONResponse({"error": "rate_limit_exceeded"}, status_code=429)

    try:
        raw_payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    if isinstance(raw_payload, dict):
        if not str(raw_payload.get("source", "")).strip():
            raw_payload["source"] = default_source
    else:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    result = await asyncio.to_thread(_webhook_receiver.process, raw_payload)
    status_code = 200 if result.get("status") == "success" else 400
    return JSONResponse(result, status_code=status_code)


@app.post("/webhook/tradingview")
async def webhook_tradingview(request: Request):
    return await _handle_webhook_ingress(request, "tradingview")


@app.post("/webhook/custom")
async def webhook_custom(request: Request):
    return await _handle_webhook_ingress(request, "custom")


@app.get("/api/webhook-log")
def get_webhook_log():
    if _store is None:
        return []
    try:
        return _store.get_webhook_log(limit=50)
    except Exception as exc:
        logger.warning("webhook-log fetch failed: %s", exc)
        return []


@app.get("/api/options-intelligence")
def get_options_intelligence():
    """
    Latest F15/F16/F17-style snapshot for Indian watchlist tickers.
    Public (no dashboard JWT). Returns {} when options intelligence is off or empty.
    """
    try:
        if _options_provider is None:
            return {}

        wl = _dashboard_cfg.get("watchlist") or {}
        indian_tickers: list[str] = []
        for row in wl.get("indian_stocks") or []:
            if isinstance(row, dict) and row.get("yf_ticker"):
                indian_tickers.append(str(row["yf_ticker"]))
        if not indian_tickers:
            is_in = getattr(_options_provider, "is_indian_ticker", None)
            if callable(is_in):
                for row in wl.get("assets") or []:
                    if isinstance(row, dict) and row.get("yf_ticker"):
                        t = str(row["yf_ticker"])
                        if is_in(t):
                            indian_tickers.append(t)

        raw = _options_provider.get_all_tickers_snapshot(indian_tickers)

        result: dict[str, dict] = {}
        for ticker, factors in raw.items():
            if not factors.get("available"):
                continue
            iv = float(factors.get("F15_iv_skew") or 0.0)
            mp = float(factors.get("F16_max_pain") or 0.0)
            gx = float(factors.get("F17_gex") or 0.0)
            result[ticker] = {
                "iv_skew": round(iv, 4),
                "iv_skew_signal": (
                    "BEARISH" if iv > 0.5 else "BULLISH" if iv < -0.5 else "NEUTRAL"
                ),
                "max_pain_level": factors.get("max_pain_level"),
                "max_pain_signal": (
                    "PULL_DOWN" if mp > 0.2 else "PULL_UP" if mp < -0.2 else "NEUTRAL"
                ),
                "gex": round(gx, 4),
                "gex_signal": (
                    "STABILIZING" if gx > 0.3 else "AMPLIFYING" if gx < -0.3 else "NEUTRAL"
                ),
                "available": True,
            }
        return result
    except Exception as exc:
        logger.warning("options-intelligence endpoint failed: %s", exc)
        return {}


# ------------------------------------------------------------------
# HTML Pages
# ------------------------------------------------------------------

def _read_html(name: str) -> str:
    p = STATIC_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else f"<h1>{name} not found</h1>"


@app.get("/", response_class=HTMLResponse)
def index():
    return _read_html("index.html")


@app.get("/terminal", response_class=HTMLResponse)
def terminal_page():
    return _read_html("terminal.html")


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio_page():
    return _read_html("portfolio.html")


@app.get("/history", response_class=HTMLResponse)
def history_page():
    return _read_html("history.html")


@app.get("/factors", response_class=HTMLResponse)
def factors_page():
    return _read_html("factors.html")


@app.get("/regime", response_class=HTMLResponse)
def regime_page():
    return _read_html("regime.html")


@app.get("/focus", response_class=HTMLResponse)
def focus_page():
    return _read_html("focus.html")


@app.get("/stock-terminal", response_class=HTMLResponse)
def stock_terminal_page():
    return _read_html("stock_terminal.html")

