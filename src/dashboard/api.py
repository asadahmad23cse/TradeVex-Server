"""
Layer 10 â€” FastAPI Dashboard + WebSocket.

Binds to 127.0.0.1 (localhost only).
WebSocket at /ws broadcasts new signals in real-time.
5 pages: Live Signals, Portfolio, History, Factor Analysis, Regime Monitor.
"""

from datetime import datetime, timedelta
import asyncio
import json
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import redis
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # type: ignore[import]
from fastapi.responses import HTMLResponse, JSONResponse, Response  # type: ignore[import]
from fastapi.staticfiles import StaticFiles  # type: ignore[import]
from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore[import]
from src.data.news_feed import get_btc_news
from src.data.signal_history import get_history as get_signal_history, get_stats as get_signal_stats
from src.dashboard.btc_service import BitcoinMarketService
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
_btc_service: BitcoinMarketService | None = None
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
    _btc_service = BitcoinMarketService(_dashboard_cfg)
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
    except Exception:
        pass


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
    "/api/btc/markers": 60,
    "/api/btc/market-context": 10,
    "/api/btc/system-report": 10,
    "/api/btc/signal": 10,
    "/api/btc/signal/history": 10,
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
            except Exception:
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


def _proxy_cache_get(name: str) -> dict[str, Any] | None:
    cached = _btc_proxy_cache.get(name)
    if not cached:
        return None
    payload = dict(cached)
    payload["stale"] = True
    return payload


async def _btc_proxy_payload(name: str, redis_key: str, upstream_url: str) -> dict[str, Any]:
    # 1) Redis fast-path
    try:
        raw = r.get(redis_key)
        if raw:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                payload["stale"] = False
                _btc_proxy_cache[name] = dict(payload)
                return payload
    except Exception:
        pass

    # 2) Upstream fallback
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(upstream_url)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict):
                payload["stale"] = False
                _btc_proxy_cache[name] = dict(payload)
                return payload
            wrapped = {"data": payload, "stale": False}
            _btc_proxy_cache[name] = dict(wrapped)
            return wrapped
    except Exception:
        pass

    # 3) Last cached stale payload
    cached = _proxy_cache_get(name)
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
        except Exception:
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
def get_btc_history(interval: str = "1d"):
    """All-time historical BTCUSDT candles from Binance."""
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    return _btc_service.get_all_time_history(interval=interval)


@app.get("/api/btc/candles")
def get_btc_candles(interval: str = "15m", limit: int = 200):
    """Recent BTCUSDT candles for trading chart windows."""
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    limit = max(50, min(limit, 1000))
    return _btc_service.get_recent_candles(interval=interval, limit=limit)


@app.get("/api/btc/signal")
def get_btc_signal(interval: str = "5m"):
    """Real-time BTC signal using the project's quant factor algorithm."""
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    return _btc_service.get_realtime_signal(interval=interval)


@app.get("/api/btc/market-context")
def get_btc_market_context(interval: str = "5m"):
    """Current BTC macro/derivatives context used by the live signal."""
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    return _btc_service.get_market_context(interval=interval)


@app.get("/api/btc/signal/history")
def signal_history(limit: int = 50):
    return {"signals": get_signal_history(limit), "stats": get_signal_stats()}


@app.get("/api/btc/signal/stats")
def signal_stats():
    return get_signal_stats()


@app.get("/api/btc/system-report")
def btc_system_report(interval: str = "5m"):
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "runtime": _btc_service.get_runtime_stats(),
        "known_gaps": [
            {
                "key": "btc_replay_backtest",
                "severity": "high",
                "status": "missing",
                "summary": "No dedicated bar-by-bar BTC replay backtest for the live BitcoinMarketService pipeline.",
            },
            {
                "key": "liquidation_heatmap_clusters",
                "severity": "medium",
                "status": "missing",
                "summary": "System detects liquidation events after the move, not pre-positioned heatmap clusters.",
            },
            {
                "key": "llm_transport_live",
                "severity": "medium",
                "status": "partial",
                "summary": "LLM validation logic exists, but the transport is still safe-fallback unless a real provider is wired.",
            },
        ],
    }

    try:
        signal = _btc_service.get_realtime_signal(interval=interval)
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


@app.get("/api/btc/markers")
def get_btc_markers(interval: str = "1d", limit: int = 1000):
    """Historical LONG/SHORT markers for BTC chart overlay."""
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    return _btc_service.get_signal_markers(interval=interval, limit=limit)


@app.get("/api/btc/news")
def btc_news(limit: int = 8):
    return {"news": get_btc_news(limit)}


@app.get("/api/btc/orderflow")
async def btc_orderflow_proxy():
    return await _btc_proxy_payload(
        name="orderflow",
        redis_key="btc:orderflow",
        upstream_url="http://127.0.0.1:9000/api/orderflow",
    )


@app.get("/api/btc/volume")
async def btc_volume_proxy():
    return await _btc_proxy_payload(
        name="volume",
        redis_key="btc:volprofile",
        upstream_url="http://127.0.0.1:9000/api/volume-profile",
    )


@app.get("/api/btc/volatility")
async def btc_volatility_proxy():
    return await _btc_proxy_payload(
        name="volatility",
        redis_key="btc:volatility",
        upstream_url="http://127.0.0.1:9000/api/volatility",
    )


@app.get("/api/btc/execution")
async def btc_execution_proxy():
    return await _btc_proxy_payload(
        name="execution",
        redis_key="btc:execution",
        upstream_url="http://127.0.0.1:9000/api/execution",
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
        except Exception:
            pass


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
                except Exception:
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

@app.get("/api/paper/portfolio")
def paper_portfolio():
    """Live portfolio metrics + positions."""
    try:
        from src.paper_trading import get_paper_engine

        engine = get_paper_engine()
        metrics = engine.get_portfolio_metrics()
        positions = engine.get_open_positions()
        return {
            "metrics": metrics,
            "open_positions": positions,
            "mode": engine._state.get("mode", "manual"),
            "success": True,
        }
    except Exception as e:
        logger.error("Paper portfolio error: %s", e)
        return {
            "metrics": {
                "capital": 100000.0,
                "initial_capital": 100000.0,
                "portfolio_value": 100000.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "total_trades": 0,
                "open_positions": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "best_trade_pct": 0.0,
                "worst_trade_pct": 0.0,
                "mode": "manual",
            },
            "open_positions": [],
            "mode": "manual",
            "success": False,
            "error": str(e),
        }


@app.get("/api/paper/trades")
def paper_trades(limit: int = 50):
    """Closed trade history."""
    from src.paper_trading import get_paper_engine

    return get_paper_engine().get_closed_trades(limit)


@app.post("/api/paper/execute")
async def paper_execute(payload: dict):
    """
    Manually execute a paper trade.
    Body: { ticker, signal, entry_price, stop_loss,
            take_profit, confidence, asset_class }
    """
    from src.paper_trading import get_paper_engine

    engine = get_paper_engine()
    result = engine.execute_trade(payload, mode="manual")
    if result.get("success"):
        try:
            nm = NotificationManager((_dashboard_cfg or {}).get("notifications", {}))
            nm.notify(
                "PAPER TRADE EXECUTED",
                (
                    f"{result.get('direction')} {result.get('ticker')}\n"
                    f"Entry: {result.get('entry_price')}\n"
                    f"Qty: {result.get('quantity')}\n"
                    f"SL: {result.get('stop_loss')} | TP: {result.get('take_profit')}"
                ),
                severity="INFO",
            )
        except Exception:
            pass
    return result


@app.post("/api/paper/close")
async def paper_close(payload: dict):
    """
    Close an open position.
    Body: { ticker, exit_price, reason }
    """
    from src.paper_trading import get_paper_engine

    engine = get_paper_engine()
    ticker = str(payload.get("ticker", "")).strip().upper()
    if not ticker:
        return {"success": False, "error": "Ticker is required"}

    exit_price = payload.get("exit_price")
    if not exit_price:
        open_map = {str(p.get("ticker")).upper(): p for p in engine.get_open_positions()}
        pos = open_map.get(ticker)
        if pos:
            exit_price = pos.get("current_price") or pos.get("entry_price")
        if not exit_price:
            try:
                import yfinance as yf  # type: ignore[import]

                hist = yf.Ticker(ticker).history(period="1d", interval="1m")
                exit_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
            except Exception:
                return {"success": False, "error": "Could not fetch current price"}

    result = engine.close_position(ticker, float(exit_price), payload.get("reason", "manual"))
    if result:
        try:
            nm = NotificationManager((_dashboard_cfg or {}).get("notifications", {}))
            pnl = float(result.get("pnl", 0.0))
            nm.notify(
                "PAPER TRADE CLOSED",
                (
                    f"{result.get('direction')} {ticker}\n"
                    f"P&L: {pnl:+.2f} ({float(result.get('pnl_pct', 0.0)):+.2f}%)\n"
                    f"Reason: {result.get('reason')}"
                ),
                severity="INFO",
            )
        except Exception:
            pass
        return {"success": True, **result}
    return {"success": False, "error": "No open position"}


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
async def paper_reset(payload: dict):
    """Reset paper account. Body: { capital: float }"""
    from src.paper_trading import get_paper_engine

    capital = float(payload.get("capital", 100000))
    get_paper_engine().reset(capital)
    return {"success": True, "capital": capital}


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
def get_equity_curve(limit: int = 5000):
    lim = max(1, min(int(limit), 20000))
    rows = _fetch_trade_report_rows(limit=lim)
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
                except Exception:
                    pass
            result["factor_ic"] = {  # type: ignore[call-overload]
                f: _round4(sum(vals) / max(len(vals), 1))
                for f, vals in factor_ics.items()
            }
            result["signals_24h"] = len(signals)
        except Exception:
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

