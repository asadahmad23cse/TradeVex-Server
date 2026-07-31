"""
Layer 10 â€” FastAPI Dashboard + WebSocket.

Binds to 127.0.0.1 (localhost only).
WebSocket at /ws broadcasts new signals in real-time.
5 pages: Live Signals, Portfolio, History, Factor Analysis, Regime Monitor.
"""

from datetime import datetime, timedelta, timezone
import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import pandas as pd
import redis
import yaml  # type: ignore[import]
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
from src.data.signal_history import get_history as get_signal_history
from src.dashboard.btc_service import INTERVAL_TO_MS, BitcoinMarketService
from src.dashboard.altcoin_service import AltcoinMarketService
from src.dashboard.focus_engine import FocusQuantEngine
from src.options import ExpiryTracker, OptionsEngine
from src.compliance import SEBIComplianceEngine
from src.execution.broker import create_executor
from src.risk.kelly import KellyCalculator
from src.utils.notifiers import NotificationManager
from src.webhook.receiver import WebhookReceiver

try:
    from dotenv import load_dotenv  # type: ignore[import]
except Exception:
    load_dotenv = None  # type: ignore[assignment]

try:
    import jwt  # type: ignore[import]
    _JWT = True
except ImportError:
    jwt = None  # type: ignore[assignment]
    _JWT = False

try:
    from cryptography.fernet import Fernet  # type: ignore[import]
    _HAS_FERNET = True
except Exception:
    Fernet = None  # type: ignore[assignment]
    _HAS_FERNET = False

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
ADMIN_EMAILS = {"techieasad01@gmail.com"}

if load_dotenv is not None:
    load_dotenv()

app = FastAPI(title="QuantTrader Dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state injected at startup
_store = None
_portfolio = None
_focus_engine: FocusQuantEngine | None = None
_btc_service: BitcoinMarketService | None = None
_altcoin_service: AltcoinMarketService | None = None
_live_runner = None
_connected_ws: List[WebSocket] = []
_binance_account_cache: dict[str, Any] = {}   # { user_key: {"ts": float, "data": dict} }
_BINANCE_ACCOUNT_CACHE_TTL = 12.0              # seconds
_app_loop: asyncio.AbstractEventLoop | None = None
_btc_signal_task: asyncio.Task | None = None
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
_supabase_token_cache: dict[str, tuple[float, str, str]] = {}
_user_profile_file = Path("data/user_profiles.json")
_user_profile_lock = threading.Lock()
_user_profile_cipher: Any = None


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
    global _store, _portfolio, _dashboard_cfg, _focus_engine, _btc_service, _altcoin_service, _options_engine
    _store = store
    _portfolio = portfolio
    _dashboard_cfg = config or {}
    _focus_engine = FocusQuantEngine(_dashboard_cfg)
    _btc_service = BitcoinMarketService(_dashboard_cfg)
    _altcoin_service = AltcoinMarketService(_dashboard_cfg)
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


def _ensure_dashboard_cfg_loaded() -> None:
    global _dashboard_cfg
    if _dashboard_cfg:
        return
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        os.getenv("DASHBOARD_CONFIG_PATH", "").strip(),
        os.getenv("QUANT_CONFIG_PATH", "").strip(),
        "config.runtime.8001.yaml",
        "config.yaml",
        str((repo_root / "config.runtime.8001.yaml").resolve()),
        str((repo_root / "config.yaml").resolve()),
    ]
    for cand in candidates:
        if not cand:
            continue
        path = Path(cand)
        if not path.exists():
            continue
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                _dashboard_cfg = loaded
                logger.info("Dashboard config fallback loaded from %s", path)
                return
        except Exception as exc:
            logger.warning("Failed to load dashboard config from %s: %s", path, exc)


def set_live_runner(runner) -> None:  # type: ignore[no-untyped-def]
    """Attach/detach a LiveRunner instance for runtime telemetry endpoints."""
    global _live_runner
    _live_runner = runner


def _auth_config() -> dict:
    _ensure_dashboard_cfg_loaded()
    return (_dashboard_cfg.get("dashboard", {}) or {}).get("auth", {})


def _auth_provider() -> str:
    cfg = _auth_config()
    return str(cfg.get("provider", "local") or "local").strip().lower()


def _supabase_config() -> dict[str, str]:
    cfg = _auth_config()
    url = str(cfg.get("supabase_url") or os.getenv("SUPABASE_URL", "")).strip().rstrip("/")
    anon_key = str(cfg.get("supabase_anon_key") or os.getenv("SUPABASE_ANON_KEY", "")).strip()
    return {"url": url, "anon_key": anon_key}


def _supabase_allowed_emails() -> set[str]:
    cfg = _auth_config()
    allowed: set[str] = set()

    cfg_val = cfg.get("allowed_emails") or []
    if isinstance(cfg_val, list):
        for v in cfg_val:
            email = str(v or "").strip().lower()
            if email:
                allowed.add(email)
    elif isinstance(cfg_val, str):
        for part in cfg_val.split(","):
            email = part.strip().lower()
            if email:
                allowed.add(email)

    env_val = str(os.getenv("DASHBOARD_ALLOWED_EMAILS", "")).strip()
    if env_val:
        for part in env_val.split(","):
            email = part.strip().lower()
            if email:
                allowed.add(email)

    return allowed


def _is_supabase_email_allowed(email: str) -> bool:
    allowed = _supabase_allowed_emails()
    if not allowed:
        return True
    return email.strip().lower() in allowed


def _supabase_requires_active_subscription() -> bool:
    cfg = _auth_config()
    raw = os.getenv("DASHBOARD_REQUIRE_ACTIVE_SUBSCRIPTION", "")
    if str(raw).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return bool(cfg.get("require_active_subscription", False))


def _parse_supabase_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


async def _verify_supabase_profile_access(conf: dict[str, str], token: str, user_id: str) -> bool:
    if not _supabase_requires_active_subscription():
        return True
    try:
        async with httpx.AsyncClient(timeout=6.0, trust_env=False) as client:
            res = await client.get(
                f"{conf['url']}/rest/v1/users",
                params={
                    "id": f"eq.{user_id}",
                    "select": "is_active,subscription_expires_at",
                    "limit": "1",
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": conf["anon_key"],
                    "Accept": "application/json",
                },
            )
        if res.status_code != 200:
            logger.warning("Supabase profile access check failed: status=%s", res.status_code)
            return False
        rows = res.json() if res.content else []
        if not isinstance(rows, list) or not rows:
            return False
        row = rows[0] or {}
        if row.get("is_active") is False:
            return False
        expires_at = _parse_supabase_timestamp(row.get("subscription_expires_at"))
        if expires_at is None:
            return False
        return expires_at > datetime.now(timezone.utc)
    except Exception as exc:
        logger.warning("Supabase profile access check failed: %s", exc)
        return False


def _email_from_jwt_unverified(token: str) -> str:
    try:
        parts = str(token or "").split(".")
        if len(parts) < 2:
            return ""
        payload = parts[1]
        padding = "=" * ((4 - len(payload) % 4) % 4)
        raw = base64.urlsafe_b64decode(payload + padding).decode("utf-8", errors="ignore")
        obj = json.loads(raw) if raw else {}
        return str((obj or {}).get("email") or "").strip().lower()
    except Exception:
        return ""


def _auth_enabled() -> bool:
    if not bool(_auth_config().get("enabled", False)):
        return False
    provider = _auth_provider()
    if provider == "supabase":
        return True
    return bool(_JWT)


def _issue_token(username: str) -> str:
    assert jwt is not None, "PyJWT is required for auth"
    cfg = _auth_config()
    secret = cfg.get("jwt_secret", "change-me")
    return jwt.encode({"sub": username}, secret, algorithm="HS256")


def _verify_local_token(token: str) -> bool:
    assert jwt is not None, "PyJWT is required for auth"
    cfg = _auth_config()
    secret = cfg.get("jwt_secret", "change-me")
    try:
        jwt.decode(token, secret, algorithms=["HS256"])
        return True
    except Exception:
        return False


def _local_token_subject(token: str) -> str:
    if not _JWT or jwt is None:
        return ""
    cfg = _auth_config()
    secret = cfg.get("jwt_secret", "change-me")
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return str(payload.get("sub") or "").strip()
    except Exception:
        return ""


async def _verify_supabase_token(token: str) -> bool:
    now = time.time()
    cached = _supabase_token_cache.get(token)
    if cached and cached[0] > now:
        return _is_supabase_email_allowed(cached[2])

    conf = _supabase_config()
    if not conf["url"] or not conf["anon_key"]:
        return False
    try:
        async with httpx.AsyncClient(timeout=6.0, trust_env=False) as client:
            res = await client.get(
                f"{conf['url']}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": conf["anon_key"],
                },
            )
        if res.status_code != 200:
            return False
        payload = res.json() if res.content else {}
        user_id = str(payload.get("id") or "")
        email = str(payload.get("email") or "").strip().lower()
        if not email:
            email = _email_from_jwt_unverified(token)
        if not user_id:
            return False
        if not _is_supabase_email_allowed(email):
            return False
        if email in ADMIN_EMAILS:
            _supabase_token_cache[token] = (now + 45.0, user_id, email)
            return True
        if not await _verify_supabase_profile_access(conf, token, user_id):
            return False
        _supabase_token_cache[token] = (now + 45.0, user_id, email)
        return True
    except Exception:
        return False


async def _verify_request_token(token: str) -> bool:
    provider = _auth_provider()
    if provider == "supabase":
        return await _verify_supabase_token(token)
    return _verify_local_token(token)


def _extract_auth_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    if not token and "token" in request.query_params:
        token = str(request.query_params["token"])
    return token


def _identity_from_token_cache(token: str) -> tuple[str, str]:
    if not token:
        return "", ""
    if _auth_provider() == "supabase":
        cached = _supabase_token_cache.get(token)
        if cached:
            return str(cached[1] or ""), str(cached[2] or "")
        return "", ""
    sub = _local_token_subject(token)
    return sub, sub


def _request_user_identity(request: Request) -> tuple[str, str]:
    uid = str(getattr(request.state, "auth_user_id", "") or "")
    email = str(getattr(request.state, "auth_email", "") or "")
    if uid:
        return uid, email
    token = _extract_auth_token(request)
    return _identity_from_token_cache(token)


def _profile_key() -> str:
    raw = str(os.getenv("DASHBOARD_PROFILE_ENCRYPTION_KEY", "") or "").strip()
    if raw:
        key_bytes = raw.encode("utf-8")
        if _HAS_FERNET and Fernet is not None:
            try:
                Fernet(key_bytes)
                return raw
            except Exception:
                pass
        digest = hashlib.sha256(key_bytes).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8")

    # Fallback deterministic key (replace in production with DASHBOARD_PROFILE_ENCRYPTION_KEY).
    seed = str(_auth_config().get("jwt_secret") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or "change-this-secret")
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8")


def _profile_cipher() -> Any:
    global _user_profile_cipher
    if _user_profile_cipher is not None:
        return _user_profile_cipher
    if not _HAS_FERNET or Fernet is None:
        return None
    try:
        _user_profile_cipher = Fernet(_profile_key().encode("utf-8"))
        return _user_profile_cipher
    except Exception as exc:
        logger.warning("Profile encryption key invalid; falling back to plaintext storage: %s", exc)
        return None


def _encrypt_profile_secret(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    cipher = _profile_cipher()
    if cipher is None:
        return raw
    try:
        return "enc:" + cipher.encrypt(raw.encode("utf-8")).decode("utf-8")
    except Exception:
        return raw


def _decrypt_profile_secret(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    if raw.startswith("enc:"):
        cipher = _profile_cipher()
        if cipher is None:
            return ""
        token = raw[4:]
        try:
            return cipher.decrypt(token.encode("utf-8")).decode("utf-8")
        except Exception:
            return ""
    return raw


def _mask_secret(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    if len(raw) <= 8:
        return "*" * len(raw)
    return f"{raw[:4]}{'*' * (len(raw) - 8)}{raw[-4:]}"


def _load_user_profiles() -> dict[str, dict[str, Any]]:
    with _user_profile_lock:
        if not _user_profile_file.exists():
            return {}
        try:
            obj = json.loads(_user_profile_file.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _save_user_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    with _user_profile_lock:
        _user_profile_file.parent.mkdir(parents=True, exist_ok=True)
        _user_profile_file.write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _profile_summary(user_key: str, email: str, rec: dict[str, Any]) -> dict[str, Any]:
    api_key = _decrypt_profile_secret(str(rec.get("binance_api_key", "") or ""))
    api_secret = _decrypt_profile_secret(str(rec.get("binance_api_secret", "") or ""))
    access_token = _decrypt_profile_secret(str(rec.get("binance_access_token", "") or ""))
    return {
        "user_id": user_key,
        "email": str(rec.get("email") or email or ""),
        "display_name": str(rec.get("display_name") or ""),
        "created_at": str(rec.get("created_at") or ""),
        "updated_at": str(rec.get("updated_at") or ""),
        "binance": {
            "api_key_masked": _mask_secret(api_key),
            "api_secret_set": bool(api_secret),
            "access_token_masked": _mask_secret(access_token),
            "access_token_set": bool(access_token),
            "ready_for_real_trading": bool(api_key and (api_secret or access_token)),
        },
    }


def _open_dashboard_paths() -> set[str]:
    return {
        "/",
        "/terminal",
        "/crypto-terminal",
        "/asset-terminal",
        "/portfolio",
        "/history",
        "/factors",
        "/regime",
        "/focus",
        "/stock-terminal",
        "/auth/token",
    }


def _open_dashboard_readonly_prefixes() -> tuple[str, ...]:
    return (
        "/api/btc/candles",
        "/api/btc/markers",
        "/api/btc/sr-levels",
        "/api/btc/liquidity-zones",
        "/api/btc/market-context",
        "/api/btc/signal",
        "/api/btc/orderflow",
        "/api/btc/volume",
        "/api/btc/volatility",
        "/api/btc/decision-intelligence",
        "/api/btc/probability",
        "/api/btc/execution-plan",
        "/api/btc/news",
        "/api/crypto/candles",
        "/api/crypto/market-context",
        "/api/crypto/deep-signal",
    )

class DashboardAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if not _auth_enabled():
            return await call_next(request)
        open_paths = _open_dashboard_paths()
        open_readonly = (
            request.method.upper() in {"GET", "HEAD", "OPTIONS"}
            and request.url.path.startswith(_open_dashboard_readonly_prefixes())
        )
        if (
            request.url.path.startswith("/static")
            or request.url.path.startswith("/webhook")
            or request.url.path == "/api/options-intelligence"
            or request.url.path == "/favicon.ico"
            or request.url.path in open_paths
            or open_readonly
        ):
            return await call_next(request)
        token = _extract_auth_token(request)
        if not token or not await _verify_request_token(token):
            return JSONResponse({"detail": "Dashboard auth required"}, status_code=401)
        user_id, email = _identity_from_token_cache(token)
        request.state.auth_user_id = user_id
        request.state.auth_email = email
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
    "/api/btc/signal/history": 0,
    "/api/btc/decision-intelligence": 10,
    "/api/btc/probability": 10,
    "/api/btc/execution-plan": 10,
    "/api/crypto/candles": 10,
    "/api/crypto/deep-signal": 10,
    "/api/crypto/market-context": 10,
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
        with closing(sqlite3.connect(db_path)) as conn:
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


def _epoch_or_iso_to_iso(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        n = float(value)
        if n > 0:
            if n > 1e12:
                n /= 1000.0
            return datetime.fromtimestamp(n, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    dt = _parse_utc(value)
    return _iso_utc(dt) if dt is not None else str(value)


def _history_ticker_key(value: Any, default: str = "BTCUSDT") -> str:
    raw = str(value or default).strip().upper()
    if not raw:
        return ""
    compact = raw.replace("-", "")
    if compact in {"BTC", "BTCUSDT", "ETH", "ETHUSDT", "SOL", "SOLUSDT"} or compact.endswith("USDT"):
        return _normalize_crypto_symbol(compact)
    return raw


def _history_signal_id(raw: Any, ticker: Any, fallback: Any) -> str:
    ticker_key = _history_ticker_key(ticker)
    prefix = _asset_short(ticker_key) or "BTC"
    existing = str(raw or "").strip()
    if existing:
        if ticker_key != "BTCUSDT" and existing.upper().startswith("BTC-"):
            return f"{prefix}-{existing.split('-', 1)[1]}"
        if not existing.isdigit():
            return existing
    try:
        numeric = int(existing or fallback)
        return f"{prefix}-{numeric:03d}"
    except Exception:
        return f"{prefix}-{str(fallback or existing or '000')}"


def _signal_history_report_rows(limit: int = 5000, ticker: str | None = None) -> list[dict[str, Any]]:
    """Normalize signal_history.json rows for equity/session reporting."""
    try:
        history_rows = get_signal_history(limit)
    except Exception as exc:
        logger.warning("Failed to read signal_history rows for equity curve: %s", exc)
        return []

    ticker_norm = _history_ticker_key(ticker, "") if ticker else ""
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(history_rows, start=1):
        if not isinstance(rec, dict):
            continue
        rec_ticker = _history_ticker_key(rec.get("ticker") or "BTCUSDT")
        if ticker_norm and rec_ticker != ticker_norm:
            continue

        direction = _normalize_direction(rec.get("signal") or rec.get("direction"))
        if direction not in {"LONG", "SHORT"}:
            continue

        status = str(rec.get("status") or rec.get("result") or rec.get("outcome") or "OPEN").upper()
        result = str(rec.get("result") or rec.get("outcome") or status).upper()
        entry_time = _epoch_or_iso_to_iso(rec.get("time") or rec.get("entry_time") or rec.get("open_timestamp"))
        exit_time = "" if status in {"OPEN", "BLOCKED"} else _epoch_or_iso_to_iso(rec.get("closed_time") or rec.get("exit_time") or rec.get("time"))

        opened = _safe_float(rec.get("open_timestamp"), 0.0)
        closed = _safe_float(rec.get("closed_time"), 0.0)
        duration = int(_safe_float(rec.get("duration_seconds"), 0.0))
        if duration <= 0 and opened > 0 and closed >= opened:
            duration = int(round(closed - opened))

        signal_id = _history_signal_id(rec.get("signal_id") or rec.get("id"), rec_ticker, idx)

        out.append(
            {
                "signal_id": signal_id,
                "trade_id": str(rec.get("trade_id") or ""),
                "ticker": rec_ticker,
                "direction": direction,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": round(_safe_float(rec.get("entry_price", rec.get("entry")), 0.0), 4),
                "exit_price": round(_safe_float(rec.get("exit_price", rec.get("exit")), 0.0), 4),
                "sl": round(_safe_float(rec.get("sl", rec.get("stop_loss")), 0.0), 4),
                "tp1": round(_safe_float(rec.get("tp1", rec.get("take_profit")), 0.0), 4),
                "confidence": round(_safe_float(rec.get("confidence"), 0.0), 2),
                "outcome": result,
                "status": status,
                "pnl_pct": round(_safe_float(rec.get("pnl_pct"), 0.0), 4),
                "mfe_pct": round(max(0.0, _safe_float(rec.get("mfe_pct"), 0.0)), 4),
                "mae_pct": round(max(0.0, _safe_float(rec.get("mae_pct"), 0.0)), 4),
                "duration_seconds": duration,
                "rr_achieved": round(_safe_float(rec.get("rr_achieved", rec.get("risk_reward")), 0.0), 4),
                "size_multiplier": round(_safe_float(rec.get("size_multiplier"), 1.0), 4),
            }
        )
    return out


def _fetch_trade_report_rows(limit: int = 200, ticker: str | None = None) -> list[dict[str, Any]]:
    db_path = _db_sqlite_path()
    if not db_path.exists():
        return []

    out: list[dict[str, Any]] = []
    try:
        with closing(sqlite3.connect(db_path)) as conn:
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
            ticker_norm = _history_ticker_key(ticker, "") if ticker else ""
            if ticker_norm:
                where_sql = f"WHERE REPLACE(UPPER({ticker_expr}), '-', '') = ?"
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
                        "signal_id": _history_signal_id(r.get("signal_id"), r.get("ticker") or "BTCUSDT", r.get("rowid")),
                        "trade_id": str(r.get("trade_id") or ""),
                        "ticker": _history_ticker_key(r.get("ticker") or "BTCUSDT"),
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


def _paper_trade_outcome(reason: Any, pnl_pct: float) -> str:
    reason_l = str(reason or "").strip().lower()
    if reason_l in {"tp", "tp_hit", "tp1", "tp1_hit"}:
        return "TP1"
    if reason_l in {"tp2", "tp2_hit"}:
        return "TP2"
    if reason_l in {"tp3", "tp3_hit"}:
        return "TP3"
    if reason_l in {"sl", "stop", "sl_hit", "stop_loss"}:
        return "SL"
    if pnl_pct > 0:
        return "WIN"
    if pnl_pct < 0:
        return "LOSS"
    return "FLAT"


def _paper_trade_report_rows(engine: Any, limit: int = 5000, ticker: str | None = None) -> list[dict[str, Any]]:
    """Normalize paper-trading open/closed rows into the Signal History shape."""
    if engine is None:
        return []
    lim = max(1, min(int(limit), 5000))
    ticker_norm = _history_ticker_key(ticker, "") if ticker else ""
    out: list[dict[str, Any]] = []

    def include_symbol(value: Any) -> tuple[bool, str]:
        key = _history_ticker_key(value or "BTCUSDT")
        return (not ticker_norm or key == ticker_norm), key

    try:
        closed_rows = engine.get_closed_trades(lim) or []
    except Exception as exc:
        logger.debug("Paper closed trades unavailable for history: %s", exc)
        closed_rows = []

    for idx, row in enumerate(closed_rows, start=1):
        if not isinstance(row, dict):
            continue
        ok, row_ticker = include_symbol(row.get("ticker") or row.get("asset") or row.get("symbol"))
        if not ok:
            continue
        direction = _normalize_direction(row.get("direction") or row.get("signal"))
        if direction not in {"LONG", "SHORT"}:
            continue
        entry = _safe_float(row.get("entry_price") or row.get("entry"), 0.0)
        exit_price = _safe_float(row.get("exit_price") or row.get("exit"), 0.0)
        pnl_pct = _safe_float(row.get("pnl_pct"), 0.0)
        outcome = _paper_trade_outcome(row.get("reason") or row.get("result") or row.get("status"), pnl_pct)
        opened_at = _epoch_or_iso_to_iso(row.get("opened_at") or row.get("entry_time") or row.get("time"))
        closed_at = _epoch_or_iso_to_iso(row.get("closed_at") or row.get("exit_time") or row.get("closed_time"))
        duration = int(round(max(0.0, _safe_float(row.get("held_hours"), 0.0) * 3600.0)))
        if duration <= 0:
            opened_dt = _history_dt(opened_at)
            closed_dt = _history_dt(closed_at)
            if opened_dt is not None and closed_dt is not None and closed_dt >= opened_dt:
                duration = int(round((closed_dt - opened_dt).total_seconds()))
        risk_pct = abs((entry - _safe_float(row.get("stop_loss") or row.get("sl"), 0.0)) / entry) * 100.0 if entry > 0 else 0.0
        rr = (pnl_pct / risk_pct) if risk_pct > 0 else 0.0
        trade_id = str(row.get("trade_id") or "")
        out.append(
            {
                "signal_id": _history_signal_id(row.get("signal_id") or trade_id, row_ticker, idx),
                "trade_id": trade_id,
                "ticker": row_ticker,
                "direction": direction,
                "entry_time": opened_at,
                "exit_time": closed_at,
                "entry_price": round(entry, 4),
                "exit_price": round(exit_price, 4),
                "sl": round(_safe_float(row.get("stop_loss") or row.get("sl"), 0.0), 4),
                "tp1": round(_safe_float(row.get("take_profit") or row.get("tp1") or row.get("tp"), 0.0), 4),
                "confidence": round(_safe_float(row.get("confidence"), 0.0), 2),
                "outcome": outcome,
                "status": "CLOSED",
                "pnl_pct": round(pnl_pct, 4),
                "mfe_pct": max(0.0, round(pnl_pct, 4)),
                "mae_pct": max(0.0, round(-pnl_pct, 4)),
                "duration_seconds": duration,
                "rr_achieved": round(rr, 4),
                "size_multiplier": 1.0,
                "source": "paper_trading",
                "mode": str(row.get("mode") or "").upper() or "MANUAL",
            }
        )

    try:
        open_rows = engine.get_open_positions() or []
    except Exception as exc:
        logger.debug("Paper open positions unavailable for history: %s", exc)
        open_rows = []

    for idx, row in enumerate(open_rows, start=1):
        if not isinstance(row, dict):
            continue
        ok, row_ticker = include_symbol(row.get("ticker") or row.get("asset") or row.get("symbol"))
        if not ok:
            continue
        direction = _normalize_direction(row.get("direction") or row.get("signal"))
        if direction not in {"LONG", "SHORT"}:
            continue
        trade_id = str(row.get("trade_id") or "")
        out.append(
            {
                "signal_id": _history_signal_id(row.get("signal_id") or trade_id, row_ticker, f"O{idx}"),
                "trade_id": trade_id,
                "ticker": row_ticker,
                "direction": direction,
                "entry_time": _epoch_or_iso_to_iso(row.get("opened_at") or row.get("entry_time") or row.get("time")),
                "exit_time": "",
                "entry_price": round(_safe_float(row.get("entry_price") or row.get("entry"), 0.0), 4),
                "exit_price": round(_safe_float(row.get("current_price"), 0.0), 4),
                "sl": round(_safe_float(row.get("stop_loss") or row.get("sl"), 0.0), 4),
                "tp1": round(_safe_float(row.get("take_profit") or row.get("tp1") or row.get("tp"), 0.0), 4),
                "confidence": round(_safe_float(row.get("confidence"), 0.0), 2),
                "outcome": "OPEN",
                "status": "OPEN",
                "pnl_pct": round(_safe_float(row.get("unrealized_pnl_pct") or row.get("pnl_pct"), 0.0), 4),
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "duration_seconds": int(round(max(0.0, _safe_float(row.get("held_hours"), 0.0) * 3600.0))),
                "rr_achieved": 0.0,
                "size_multiplier": 1.0,
                "source": "paper_trading",
                "mode": str(row.get("mode") or "").upper() or "MANUAL",
            }
        )

    return out


def _history_status(row: dict[str, Any]) -> str:
    return str(row.get("outcome") or row.get("status") or row.get("result") or "OPEN").upper()


def _history_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        n = float(value)
        if n > 0:
            if n > 1e12:
                n /= 1000.0
            return datetime.fromtimestamp(n, tz=timezone.utc)
    except Exception:
        pass
    dt = _parse_utc(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _history_event_dt(row: dict[str, Any]) -> datetime | None:
    status = _history_status(row)
    if status not in {"OPEN", "BLOCKED"}:
        dt = _history_dt(row.get("closed_time") or row.get("exit_time"))
        if dt is not None:
            return dt
    return _history_dt(row.get("open_timestamp") or row.get("entry_time") or row.get("time"))


def _history_dedupe_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    trade_id = str(row.get("trade_id") or "").strip()
    ticker_key = _history_ticker_key(row.get("ticker") or "BTCUSDT")
    direction = _normalize_direction(row.get("direction") or row.get("signal"))
    if trade_id:
        return (ticker_key, trade_id, "", direction, "")
    event_dt = _history_event_dt(row)
    event_key = event_dt.isoformat() if event_dt is not None else str(row.get("entry_time") or row.get("time") or "")
    return (
        ticker_key,
        str(row.get("signal_id") or row.get("id") or ""),
        event_key,
        direction,
        _history_status(row),
    )


def _combined_signal_history_rows(limit: int = 200, ticker: str | None = None, paper_engine: Any | None = None) -> list[dict[str, Any]]:
    lim = max(1, min(int(limit), 5000))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    # signal_history.json carries live signal events; SQLite and paper state carry executed trades.
    source_rows = (
        _signal_history_report_rows(limit=lim, ticker=ticker)
        + _fetch_trade_report_rows(limit=lim, ticker=ticker)
        + _paper_trade_report_rows(paper_engine, limit=lim, ticker=ticker)
    )
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        key = _history_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    rows.sort(key=lambda row: (_history_event_dt(row) or datetime.min.replace(tzinfo=timezone.utc)).timestamp(), reverse=True)
    return rows[:lim]


def _signal_history_stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in rows if _history_status(row) not in {"OPEN", "BLOCKED"}]
    blocked = [row for row in rows if _history_status(row) == "BLOCKED"]
    open_rows = [row for row in rows if _history_status(row) == "OPEN"]
    wins = [row for row in closed if _safe_float(row.get("pnl_pct"), 0.0) > 0]
    losses = [row for row in closed if _safe_float(row.get("pnl_pct"), 0.0) <= 0]
    total_pnl = sum(_safe_float(row.get("pnl_pct"), 0.0) for row in closed)
    today = datetime.now(timezone.utc).date()

    def is_today(row: dict[str, Any]) -> bool:
        dt = _history_event_dt(row)
        return bool(dt and dt.date() == today)

    directional_today = [
        row
        for row in rows
        if is_today(row)
        and _normalize_direction(row.get("direction") or row.get("signal")) in {"LONG", "SHORT"}
        and _history_status(row) != "BLOCKED"
    ]
    long_closed = [row for row in closed if _normalize_direction(row.get("direction") or row.get("signal")) == "LONG"]
    short_closed = [row for row in closed if _normalize_direction(row.get("direction") or row.get("signal")) == "SHORT"]
    long_wins = [row for row in long_closed if _safe_float(row.get("pnl_pct"), 0.0) > 0]
    short_wins = [row for row in short_closed if _safe_float(row.get("pnl_pct"), 0.0) > 0]
    long_pnl = sum(_safe_float(row.get("pnl_pct"), 0.0) for row in long_closed)
    short_pnl = sum(_safe_float(row.get("pnl_pct"), 0.0) for row in short_closed)

    return {
        "total": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(closed), 2) if closed else 0,
        "open_signals": len(open_rows),
        "blocked_total": len(blocked),
        "blocked_today": len([row for row in blocked if is_today(row)]),
        "long_total": len(long_closed),
        "short_total": len(short_closed),
        "long_signals_today": len([row for row in directional_today if _normalize_direction(row.get("direction") or row.get("signal")) == "LONG"]),
        "short_signals_today": len([row for row in directional_today if _normalize_direction(row.get("direction") or row.get("signal")) == "SHORT"]),
        "long_win_rate": round(len(long_wins) / len(long_closed) * 100, 1) if long_closed else 0,
        "short_win_rate": round(len(short_wins) / len(short_closed) * 100, 1) if short_closed else 0,
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
    }


def _proxy_cache_get(name: str) -> dict[str, Any] | None:
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


def _redis_get_json(key: str) -> dict[str, Any] | None:
    try:
        raw = r.get(key)
        if not raw:
            return None
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _minimal_decision_intelligence_from_signal(
    sig: dict[str, Any],
    *,
    source: str = "redis_signal_fallback",
    stale: bool = True,
    degraded: bool = True,
) -> dict[str, Any]:
    """
    Build a decision-intelligence-shaped dict from btc:signal.
    Caller controls stale/degraded semantics.
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
        "stale": bool(stale),
        "degraded": bool(degraded),
        "source": source,
    }
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
        return out

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
    return out


def _normalize_crypto_symbol(symbol: str | None) -> str:
    raw = str(symbol or "BTCUSDT").strip().upper()
    alias = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
    return alias.get(raw, raw)


def _is_alt_symbol(symbol: str | None) -> bool:
    return _normalize_crypto_symbol(symbol) in {"ETHUSDT", "SOLUSDT"}


def _asset_short(symbol: str | None) -> str:
    sym = _normalize_crypto_symbol(symbol)
    return sym[:-4] if sym.endswith("USDT") else sym


def _ensure_altcoin_service() -> AltcoinMarketService | None:
    global _altcoin_service
    if _altcoin_service is not None:
        return _altcoin_service
    try:
        _altcoin_service = AltcoinMarketService(_dashboard_cfg)
    except Exception:
        _altcoin_service = None
    return _altcoin_service


def _alt_signal_payload(symbol: str, interval: str = "15m") -> dict[str, Any]:
    svc = _ensure_altcoin_service()
    if svc is None:
        base = {
            "asset": _normalize_crypto_symbol(symbol),
            "signal": "HOLD",
            "validated_signal": "HOLD",
            "validated": False,
            "reason": "Altcoin service not initialised",
            "confidence": 0.0,
            "alpha_score": 0.0,
            "net_alpha_score": 0.0,
            "regime": "SIDEWAYS",
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
        }
        return _alt_enrich_signal_payload(base, symbol=symbol)
    raw = svc.get_realtime_signal(
        symbol=_normalize_crypto_symbol(symbol),
        interval=interval,
    )
    if not isinstance(raw, dict):
        raw = {
            "asset": _normalize_crypto_symbol(symbol),
            "signal": "HOLD",
            "validated_signal": "HOLD",
            "validated": False,
            "reason": "Altcoin service returned invalid payload",
            "confidence": 0.0,
            "alpha_score": 0.0,
            "net_alpha_score": 0.0,
            "regime": "SIDEWAYS",
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
        }
    return _alt_enrich_signal_payload(raw, symbol=symbol)


def _alt_frame(symbol: str, interval: str = "15m", limit: int = 300) -> pd.DataFrame:
    svc = _ensure_altcoin_service()
    if svc is None:
        return pd.DataFrame()
    payload = svc.get_recent_candles(
        symbol=_normalize_crypto_symbol(symbol),
        interval=interval,
        limit=limit,
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            parsed.append(
                {
                    "time": datetime.fromtimestamp(int(row.get("time")), tz=timezone.utc),
                    "Open": float(row.get("open", 0.0)),
                    "High": float(row.get("high", 0.0)),
                    "Low": float(row.get("low", 0.0)),
                    "Close": float(row.get("close", 0.0)),
                    "Volume": float(row.get("volume", 0.0)),
                },
            )
        except Exception:
            continue
    if not parsed:
        return pd.DataFrame()
    return pd.DataFrame(parsed).set_index("time").sort_index()


def _alt_clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _alt_session_label() -> str:
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 8:
        return "Asia Session"
    if 8 <= hour < 16:
        return "London Session"
    return "NY Session"


def _alt_fear_greed_label(score: float) -> str:
    if score <= 20:
        return "Extreme Fear"
    if score <= 40:
        return "Fear"
    if score < 60:
        return "Neutral"
    if score < 80:
        return "Greed"
    return "Extreme Greed"


def _alt_volatility_label(atr_pct: float) -> str:
    if atr_pct >= 2.8:
        return "HIGH"
    if atr_pct <= 0.7:
        return "LOW"
    return "NORMAL"


def _alt_validation_checks_from_gate(sig: dict[str, Any], fear_greed: float) -> dict[str, bool]:
    gate = sig.get("gate_status") if isinstance(sig.get("gate_status"), dict) else {}
    liq_gate = bool(gate.get("liquidity_gate", True))
    checks = {
        "data_quality_ok": bool(gate.get("data_quality", True)),
        "confidence_ok": bool(gate.get("confidence_gate", False)),
        "cost_ok": bool(gate.get("cost_gate", False)),
        "mtf_ok": bool(gate.get("mtf_alignment", True)),
        "etf_ok": True,
        "fear_greed_ok": 12.0 <= fear_greed <= 88.0,
        "liquidation_ok": liq_gate,
        "oi_ok": True,
        "sl_distance_ok": True,
    }
    return checks


def _alt_first_blocker(sig: dict[str, Any], checks: dict[str, bool]) -> str:
    requested = str(sig.get("requested_signal") or sig.get("signal") or "HOLD").upper()
    validated = str(sig.get("validated_signal") or "HOLD").upper()
    if requested in {"HOLD", "WAIT", ""}:
        return "alpha_threshold"
    if validated in {"LONG", "SHORT"}:
        return ""
    for key in (
        "data_quality_ok",
        "confidence_ok",
        "cost_ok",
        "mtf_ok",
        "fear_greed_ok",
        "liquidation_ok",
        "oi_ok",
        "sl_distance_ok",
    ):
        if checks.get(key) is False:
            return key
    return "validation_block"


def _alt_flow_state(obi: float) -> str:
    if obi >= 0.08:
        return "FAVOR_LONG"
    if obi <= -0.08:
        return "FAVOR_SHORT"
    return "NO_TRADE"


def _alt_rsi_zone(rsi: float) -> str:
    if rsi >= 70:
        return "OVERBOUGHT"
    if rsi <= 30:
        return "OVERSOLD"
    return "NEUTRAL"


def _alt_market_overview(sig: dict[str, Any], fear_greed: float) -> dict[str, Any]:
    ctx = sig.get("market_context") if isinstance(sig.get("market_context"), dict) else {}
    regime = str(sig.get("regime") or "SIDEWAYS").upper()
    atr_pct = float(ctx.get("atr_pct", 0.0) or 0.0)
    return {
        "price_change_1h": round(float(ctx.get("price_change_1h", 0.0) or 0.0), 3),
        "price_change_24h": round(float(ctx.get("price_change_24h", 0.0) or 0.0), 3),
        "fear_greed_label": _alt_fear_greed_label(fear_greed),
        "volatility": _alt_volatility_label(atr_pct),
        "atr_pct": round(atr_pct, 3),
        "regime": regime,
        "session": _alt_session_label(),
    }


def _alt_mtf_bias(sig: dict[str, Any]) -> dict[str, Any]:
    ctx = sig.get("market_context") if isinstance(sig.get("market_context"), dict) else {}
    mtf = ctx.get("mtf") if isinstance(ctx.get("mtf"), dict) else {}
    details = mtf.get("details") if isinstance(mtf.get("details"), dict) else {}
    b4 = str(details.get("4h", "NEUTRAL")).upper()
    b1h = str(details.get("1h", "NEUTRAL")).upper()
    regime = str(sig.get("regime") or "SIDEWAYS").upper()
    if "BULL" in regime:
        b1d = "BULLISH"
    elif "BEAR" in regime:
        b1d = "BEARISH"
    else:
        b1d = "NEUTRAL"
    return {
        "bias_4h": b4 if b4 in {"BULLISH", "BEARISH", "NEUTRAL", "SIDEWAYS"} else "NEUTRAL",
        "bias_1d": b1d if b1d in {"BULLISH", "BEARISH", "NEUTRAL"} else "NEUTRAL",
        "bias_1h": b1h if b1h in {"BULLISH", "BEARISH", "NEUTRAL", "SIDEWAYS"} else "NEUTRAL",
        "aligned": bool(mtf.get("aligned", True)),
        "score": float(mtf.get("score", 1.0) or 1.0),
    }


def _alt_factor_contributions(sig: dict[str, Any]) -> list[dict[str, Any]]:
    fs = sig.get("factor_scores") if isinstance(sig.get("factor_scores"), dict) else {}
    iw = sig.get("ic_weights") if isinstance(sig.get("ic_weights"), dict) else {}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(fs.keys()) | set(iw.keys())):
        score = float(fs.get(key, 0.0) or 0.0)
        weight = float(iw.get(key, 0.0) or 0.0)
        contrib = score * (weight if abs(weight) > 0 else 1.0)
        rows.append(
            {
                "factor": key,
                "score": round(score, 4),
                "weight": round(weight, 4),
                "contribution": round(contrib, 4),
            },
        )
    rows.sort(key=lambda r: abs(float(r.get("contribution", 0.0) or 0.0)), reverse=True)
    return rows[:12]


def _alt_breakdown_from_signal(sig: dict[str, Any], checks: dict[str, bool], obi: float) -> dict[str, Any]:
    conf = float(sig.get("confidence", 0.0) or 0.0)
    thr = float(sig.get("adjusted_confidence_threshold", 65.0) or 65.0)
    net_alpha = float(sig.get("net_alpha_score_raw", sig.get("net_alpha_score", 0.0)) or 0.0)
    ctx = sig.get("market_context") if isinstance(sig.get("market_context"), dict) else {}
    regime = str(sig.get("regime") or "SIDEWAYS").upper()
    mom = float(ctx.get("price_change_1h", 0.0) or 0.0) * 12.0 + (float(ctx.get("rsi", 50.0) or 50.0) - 50.0)
    momentum_score = _alt_clip(mom, -100.0, 100.0)
    flow_score = _alt_clip(obi * 100.0, -100.0, 100.0)
    cost_score = _alt_clip(net_alpha * 1200.0, -100.0, 100.0)
    regime_score = 35.0 if checks.get("mtf_ok", True) and checks.get("liquidation_ok", True) else -25.0
    if "BEAR" in regime:
        regime_score -= 5.0
    elif "BULL" in regime:
        regime_score += 5.0
    confidence_edge = conf - thr
    final_score = _alt_clip((confidence_edge * 1.4) + (momentum_score * 0.15) + (flow_score * 0.2) + (cost_score * 0.25), -100.0, 100.0)
    return {
        "regime_score": round(regime_score, 2),
        "momentum_score": round(momentum_score, 2),
        "flow_score": round(flow_score, 2),
        "cost_score": round(cost_score, 2),
        "final_score": round(final_score, 2),
        "explanation": str(sig.get("reason") or "Altcoin decision stack active"),
    }


def _alt_trade_verdict_from_signal(sig: dict[str, Any], checks: dict[str, bool], breakdown: dict[str, Any]) -> dict[str, Any]:
    decision = str(sig.get("validated_signal") or "HOLD").upper()
    final_verdict = "TRADE" if decision in {"LONG", "SHORT"} else "AVOID"
    liq_ok = checks.get("liquidation_ok", True)
    mtf_ok = checks.get("mtf_ok", True)
    regime_alignment = "HIGH" if mtf_ok and liq_ok else "MEDIUM" if mtf_ok else "LOW"
    mo = sig.get("market_overview") if isinstance(sig.get("market_overview"), dict) else {}
    vol = str(mo.get("volatility", "NORMAL")).upper()
    vol_state = "NOT_TRADEABLE" if vol == "HIGH" and final_verdict != "TRADE" else vol
    return {
        "final_verdict": final_verdict,
        "decision_confidence": round(float(sig.get("confidence", 0.0) or 0.0), 2),
        "regime_alignment": regime_alignment,
        "liquidity_quality": "STRONG" if liq_ok else "WEAK",
        "volatility_state": vol_state,
        "decision": decision,
        "reason": str(breakdown.get("explanation") or sig.get("reason") or ""),
    }


def _alt_decision_intelligence_from_signal(sig: dict[str, Any]) -> dict[str, Any]:
    enriched = _alt_enrich_signal_payload(sig, symbol=str(sig.get("symbol") or sig.get("asset") or "ETHUSDT"))
    checks = enriched.get("validation_checks") if isinstance(enriched.get("validation_checks"), dict) else {}
    order_flow = enriched.get("order_flow") if isinstance(enriched.get("order_flow"), dict) else {}
    obi = float(order_flow.get("obi", 0.0) or 0.0)
    breakdown = _alt_breakdown_from_signal(enriched, checks, obi)
    probability = _alt_prob_from_signal(enriched)
    execution_plan = _alt_execution_plan_from_signal(enriched)
    trade_verdict = _alt_trade_verdict_from_signal(enriched, checks, breakdown)
    blockers = [k for k, v in checks.items() if v is False]
    decision = str(enriched.get("validated_signal") or "HOLD").upper()
    requested = str(enriched.get("requested_signal") or enriched.get("signal") or "HOLD").upper()
    regime = str(enriched.get("regime") or "SIDEWAYS").upper()
    cal_prob = max(float(probability.get("up_prob", 0.0) or 0.0), float(probability.get("down_prob", 0.0) or 0.0), float(probability.get("sideways_prob", 0.0) or 0.0)) / 100.0

    return {
        "as_of_utc": enriched.get("as_of_utc"),
        "decision_engine": {
            "decision": decision,
            "requested_signal": requested,
            "final_score": float(breakdown.get("final_score", 0.0) or 0.0),
            "confidence": float(enriched.get("confidence", 0.0) or 0.0),
            "quality_score": float(enriched.get("signal_strength", 0.0) or 0.0),
            "regime": regime,
            "reason": str(enriched.get("reason") or ""),
            "blockers": blockers,
            "trade_triggers": [] if blockers else [f"{decision} setup confirmed on {enriched.get('symbol', '')}"],
            "flow_decision": str(order_flow.get("decision_state") or "NO_TRADE"),
        },
        "decision_breakdown": breakdown,
        "factor_contributions": _alt_factor_contributions(enriched),
        "probability": probability,
        "execution_plan": execution_plan,
        "trade_verdict": trade_verdict,
        "meta_decision": {
            "final_decision": decision,
            "reasons": blockers if blockers else [str(enriched.get("reason") or "")],
            "calibration": {"calibrated_prob": round(cal_prob, 4)},
        },
        "meta_labeling": {
            "inputs": {
                "rf_veto_applied": False,
                "rf_fp_probability": None,
            },
        },
        "validation_engine": {
            "checks": {
                "alpha_barrier": bool(checks.get("cost_ok", False)),
                "confidence_gate": bool(checks.get("confidence_ok", False)),
                "data_quality": bool(checks.get("data_quality_ok", False)),
            },
        },
        "strategy_selection": {"active": "altcoin_spot_quant_v1", "symbol": enriched.get("symbol")},
        "data_drift": {"status": "stable"},
        "meta_output": {"status": "live"},
        "adaptive_learning": {"enabled": True},
        "order_flow": order_flow,
        "regime": regime,
        "regime_state_probs": {
            "trending": 0.7 if "TREND" in regime else 0.35,
            "mean_reverting": 0.2 if "SIDEWAYS" in regime or "RANGE" in regime else 0.45,
            "liquidity_cascade": 0.1 if "VOLATILITY" not in regime else 0.2,
        },
        "signal_aggregation": {
            "requested_signal": requested,
            "validated_signal": decision,
            "blocked_by": enriched.get("blocked_by"),
        },
        "execution_gate": {"open": decision in {"LONG", "SHORT"}},
        "kelly_sizing": {
            "position_size_pct": float(enriched.get("position_size_pct", 0.0) or 0.0),
            "method": ((enriched.get("position_sizing") or {}).get("method")),
        },
        "net_alpha": float(enriched.get("net_alpha_score_raw", 0.0) or 0.0),
        "monitoring_summary": {
            "stale": bool(enriched.get("stale", False)),
            "degraded": bool(enriched.get("degraded", False)),
            "source": "altcoin_deep_signal",
        },
        "source": "altcoin_deep_signal",
        "stale": bool(enriched.get("stale", False)),
        "degraded": bool(enriched.get("degraded", False)),
    }


def _alt_enrich_signal_payload(payload: dict[str, Any], symbol: str | None = None) -> dict[str, Any]:
    out = dict(payload if isinstance(payload, dict) else {})
    sym = _normalize_crypto_symbol(str(out.get("symbol") or out.get("asset") or symbol or "ETHUSDT"))
    out["symbol"] = sym
    out["asset"] = sym
    raw_signal = str(out.get("signal") or "HOLD").upper()
    if raw_signal == "BUY":
        raw_signal = "LONG"
    elif raw_signal == "SELL":
        raw_signal = "SHORT"
    out["signal"] = raw_signal

    validated = str(out.get("validated_signal") or raw_signal).upper()
    if validated == "BUY":
        validated = "LONG"
    elif validated == "SELL":
        validated = "SHORT"
    if validated not in {"LONG", "SHORT"}:
        validated = "HOLD"
    out["validated_signal"] = validated
    out["validated"] = bool(validated in {"LONG", "SHORT"})

    ctx = out.get("market_context") if isinstance(out.get("market_context"), dict) else {}
    rsi = float(ctx.get("rsi", 50.0) or 50.0)
    p24 = float(ctx.get("price_change_24h", 0.0) or 0.0)
    obv = float(ctx.get("obv_slope", 0.0) or 0.0)
    fear_greed = _alt_clip(50.0 + (rsi - 50.0) * 0.9 + p24 * 3.2 + obv * 12.0, 0.0, 100.0)
    out["fear_greed"] = int(round(fear_greed))
    out["session"] = _alt_session_label()

    mo = _alt_market_overview(out, fear_greed)
    out["market_overview"] = mo
    out["market_regime"] = str(out.get("regime") or mo.get("regime") or "SIDEWAYS").upper()
    out["volatility_regime"] = str(mo.get("volatility") or "NORMAL")
    out["volatility"] = {
        "volatility_regime": out["volatility_regime"],
        "atr_pct": mo.get("atr_pct"),
    }
    out["mtf_bias"] = _alt_mtf_bias(out)

    gate = out.get("gate_status") if isinstance(out.get("gate_status"), dict) else {}
    liq = ctx.get("liquidity") if isinstance(ctx.get("liquidity"), dict) else {}
    bbook = liq.get("binance") if isinstance(liq.get("binance"), dict) else {}
    obi = float(bbook.get("book_imbalance", 0.0) or 0.0)
    volume_ratio = float(ctx.get("volume_ratio", 1.0) or 1.0)
    cmf = float(ctx.get("cmf", 0.0) or 0.0)
    volume_trend = "HIGH" if volume_ratio >= 1.2 else "LOW" if volume_ratio <= 0.85 else "NORMAL"
    obv_trend = "RISING" if obv >= 0 else "FALLING"
    cmf_signal = "ACCUMULATION" if cmf > 0.05 else "DISTRIBUTION" if cmf < -0.05 else "NEUTRAL"
    of = {
        "decision_state": _alt_flow_state(obi),
        "volume_trend": volume_trend,
        "obv_trend": obv_trend,
        "cmf": round(cmf, 4),
        "cmf_signal": cmf_signal,
        "rsi": round(rsi, 2),
        "rsi_zone": _alt_rsi_zone(rsi),
        "obi": round(obi, 4),
    }
    out["order_flow"] = of
    out["orderflow"] = of
    out["obi"] = round(obi, 4)

    mark_price = float(out.get("entry_price", 0.0) or 0.0)
    if mark_price <= 0:
        mark_price = float(out.get("entry", 0.0) or 0.0)
    top_notional = float(bbook.get("top30_notional_usd", 0.0) or 0.0)
    out["mark_price"] = round(mark_price, 6) if mark_price > 0 else None
    out["open_interest_btc"] = round((top_notional / max(mark_price, 1.0)), 3) if top_notional > 0 and mark_price > 0 else 0.0
    out["funding_rate"] = 0.0
    out["funding_rate_pct"] = 0.0
    out["funding_sentiment"] = "SPOT_ONLY"

    pos = out.get("position_sizing") if isinstance(out.get("position_sizing"), dict) else {}
    out["position_size_pct"] = float(pos.get("size_pct", 0.0) or 0.0)
    out["meta_confidence"] = float(out.get("confidence", 0.0) or 0.0)
    out["signal_strength"] = int(round(_alt_clip(float(out.get("confidence", 0.0) or 0.0) if out["validated"] else float(out.get("confidence", 0.0) or 0.0) * 0.55, 0.0, 100.0)))
    out["quality_score"] = out["signal_strength"]

    out["requested_signal"] = raw_signal
    out["alpha_score_raw"] = float(out.get("alpha_score", 0.0) or 0.0)
    out["net_alpha_score_raw"] = float(out.get("net_alpha_score", 0.0) or 0.0)

    checks = _alt_validation_checks_from_gate(out, fear_greed)
    out["validation_checks"] = checks
    out["blocked_by"] = _alt_first_blocker(out, checks)

    entry = float(out.get("entry_price", 0.0) or 0.0)
    sl = float(out.get("stop_loss", 0.0) or 0.0)
    tp1 = float(out.get("tp1", 0.0) or 0.0)
    tp2 = float(out.get("tp2", 0.0) or 0.0)
    tp3 = float(out.get("tp3", 0.0) or 0.0)
    if entry > 0:
        out["entry_zone_low"] = round(entry * 0.998, 6)
        out["entry_zone_high"] = round(entry * 1.002, 6)
        out["sl_pct"] = round(abs(entry - sl) / entry * 100.0, 3) if sl > 0 else None
        out["tp1_pct"] = round(abs(tp1 - entry) / entry * 100.0, 3) if tp1 > 0 else None
        out["tp2_pct"] = round(abs(tp2 - entry) / entry * 100.0, 3) if tp2 > 0 else None
        out["tp3_pct"] = round(abs(tp3 - entry) / entry * 100.0, 3) if tp3 > 0 else None
    else:
        out["entry_zone_low"] = None
        out["entry_zone_high"] = None
        out["sl_pct"] = None
        out["tp1_pct"] = None
        out["tp2_pct"] = None
        out["tp3_pct"] = None

    interval = str(out.get("interval") or "15m")
    out["signal_validity_seconds"] = int(max(60, int(INTERVAL_TO_MS.get(interval, 900000) / 1000)))
    out.setdefault("hibernated_factors", [])
    out["source"] = "altcoin_deep_signal"
    out["stale"] = bool(out.get("stale", False))
    out["degraded"] = bool(out.get("degraded", False))
    return out


def _alt_prob_from_signal(sig: dict[str, Any]) -> dict[str, Any]:
    conf = float(sig.get("confidence", 0.0) or 0.0)
    vsig = str(sig.get("validated_signal") or sig.get("signal") or "HOLD").upper()
    if vsig in {"BUY", "LONG"}:
        up = max(0.0, min(100.0, conf))
        down = max(0.0, round((100.0 - up) * 0.45, 2))
    elif vsig in {"SELL", "SHORT"}:
        down = max(0.0, min(100.0, conf))
        up = max(0.0, round((100.0 - down) * 0.45, 2))
    else:
        up = down = max(0.0, round(conf * 0.35, 2))
    sideways = max(0.0, round(100.0 - up - down, 2))
    if up >= down and up >= sideways:
        dom = "LONG"
    elif down >= up and down >= sideways:
        dom = "SHORT"
    else:
        dom = "SIDEWAYS"
    return {
        "up_prob": round(up, 2),
        "down_prob": round(down, 2),
        "sideways_prob": round(sideways, 2),
        "dominant_state": dom,
        "dominant": dom,
        "calibration_score": round(conf, 2),
    }


def _alt_execution_plan_from_signal(sig: dict[str, Any]) -> dict[str, Any]:
    entry = sig.get("entry_price")
    sl = sig.get("stop_loss")
    tp = sig.get("take_profit")
    rr = sig.get("risk_reward")
    gate = sig.get("gate_status") if isinstance(sig.get("gate_status"), dict) else {}
    slippage_risk = "LOW"
    if gate.get("liquidity_gate") is False:
        slippage_risk = "HIGH"
    elif gate.get("liquidity_gate") is True and gate.get("cost_gate") is False:
        slippage_risk = "MEDIUM"
    return {
        "entry_zone": [entry, entry] if entry else None,
        "stop_loss": sl,
        "take_profit": tp,
        "take_profit_2": sig.get("tp2"),
        "expected_rr": rr,
        "slippage_risk": slippage_risk,
        "position_size_pct": ((sig.get("position_sizing") or {}).get("size_pct")),
        "regime": sig.get("regime"),
    }


def _alt_markers(symbol: str, interval: str = "15m", limit: int = 300) -> list[dict[str, Any]]:
    frame = _alt_frame(symbol, interval=interval, limit=limit)
    if frame.empty or len(frame) < 80:
        return []
    close = frame["Close"].astype(float)
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    out: list[dict[str, Any]] = []
    prev_state = 0
    for i in range(1, len(frame)):
        state = 1 if ema20.iloc[i] > ema50.iloc[i] else -1
        if state == prev_state:
            continue
        prev_state = state
        ts = int(pd.Timestamp(frame.index[i]).timestamp())
        if state > 0:
            out.append({"time": ts, "position": "belowBar", "color": "#34d399", "shape": "arrowUp", "text": "BUY"})
        else:
            out.append({"time": ts, "position": "aboveBar", "color": "#f87171", "shape": "arrowDown", "text": "SELL"})
    return out[-200:]


async def _btc_proxy_payload(
    name: str,
    redis_key: Union[str, List[str]],
    upstream_url: Union[str, List[str]],
    *,
    per_try_timeout: float = 8.0,
    retries_per_url: int = 3,
    retry_sleep_sec: float = 1.0,
) -> dict[str, Any]:
    redis_keys = _as_key_list(redis_key)
    upstream_urls = _as_key_list(upstream_url)

    # 1) Redis fast-path
    for key in redis_keys:
        try:
            raw = r.get(key)
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    if _btc_redis_payload_is_unusable(payload):
                        continue
                    payload["stale"] = False
                    _btc_proxy_cache[name] = dict(payload)
                    return payload
        except Exception:
            continue

    # 2) Upstream fallback
    try:
        timeout_s = max(0.5, float(per_try_timeout))
        max_retries = max(1, int(retries_per_url))
        sleep_s = max(0.0, float(retry_sleep_sec))
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            for url in upstream_urls:
                last_exc: Exception | None = None
                for attempt in range(max_retries):
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        payload = resp.json()
                        if isinstance(payload, dict):
                            payload["stale"] = False
                            _btc_proxy_cache[name] = dict(payload)
                            return payload
                        wrapped = {"data": payload, "stale": False}
                        _btc_proxy_cache[name] = dict(wrapped)
                        return wrapped
                    except Exception as exc:
                        last_exc = exc
                        if attempt < (max_retries - 1) and sleep_s > 0:
                            await asyncio.sleep(sleep_s)
                        continue
                if last_exc is not None:
                    logger.debug(
                        "BTC proxy upstream failed after retries %s: %s",
                        url,
                        last_exc,
                    )
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


def _btc_background_signal_enabled() -> bool:
    dashboard_cfg = (_dashboard_cfg.get("dashboard") or {}) if isinstance(_dashboard_cfg, dict) else {}
    raw = dashboard_cfg.get("btc_background_signal_enabled", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


def _btc_background_signal_interval_seconds() -> float:
    dashboard_cfg = (_dashboard_cfg.get("dashboard") or {}) if isinstance(_dashboard_cfg, dict) else {}
    try:
        return max(5.0, float(dashboard_cfg.get("btc_background_signal_seconds", 15)))
    except Exception:
        return 15.0


def _btc_background_signal_intervals() -> list[str]:
    dashboard_cfg = (_dashboard_cfg.get("dashboard") or {}) if isinstance(_dashboard_cfg, dict) else {}
    raw = dashboard_cfg.get("btc_background_signal_intervals", ["15m"])
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(part or "").strip() for part in raw]
    else:
        values = ["15m"]
    allowed = [value for value in values if value in INTERVAL_TO_MS]
    return allowed or ["15m"]


async def _btc_background_signal_loop() -> None:
    """Keep BTC signal/history warm even when no browser tab is polling."""
    await asyncio.sleep(3)
    logger.info(
        "BTC background signal poller started: intervals=%s every=%.1fs",
        ",".join(_btc_background_signal_intervals()),
        _btc_background_signal_interval_seconds(),
    )
    while True:
        try:
            svc = _btc_service
            if svc is not None:
                for interval in _btc_background_signal_intervals():
                    payload = await asyncio.to_thread(svc.get_realtime_signal, interval=interval)
                    signal = str((payload or {}).get("signal") or "HOLD").upper()
                    validated = bool((payload or {}).get("validated", False))
                    logger.debug("BTC background signal tick: interval=%s signal=%s validated=%s", interval, signal, validated)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("BTC background signal poll failed: %s", exc)
        await asyncio.sleep(_btc_background_signal_interval_seconds())


@app.on_event("startup")
async def _capture_loop() -> None:
    global _app_loop, _btc_service, _altcoin_service, _btc_signal_task
    _app_loop = asyncio.get_running_loop()
    # Auto-init BTC service when running standalone via uvicorn (without main.py)
    if _btc_service is None:
        try:
            _btc_service = BitcoinMarketService(_dashboard_cfg)
        except Exception as exc:  # pragma: no cover
            import logging
            logging.getLogger(__name__).warning("BTC service auto-init failed: %s", exc)
    if _altcoin_service is None:
        try:
            _altcoin_service = AltcoinMarketService(_dashboard_cfg)
        except Exception as exc:  # pragma: no cover
            import logging
            logging.getLogger(__name__).warning("Altcoin service auto-init failed: %s", exc)
    if _btc_background_signal_enabled() and _btc_signal_task is None:
        _btc_signal_task = asyncio.create_task(_btc_background_signal_loop())


@app.on_event("shutdown")
async def _shutdown_live_runner() -> None:
    """Ensure scheduler/broker loop is stopped when the API server exits."""
    global _btc_signal_task
    if _btc_signal_task is not None:
        _btc_signal_task.cancel()
        try:
            await _btc_signal_task
        except asyncio.CancelledError:
            pass
        _btc_signal_task = None
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
    if _auth_enabled() and not await _verify_request_token(token):
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
def get_btc_history(interval: str = "1d", symbol: str = "BTCUSDT"):
    """Historical candles; BTC default, ETH/SOL compatible via symbol parameter."""
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        svc = _ensure_altcoin_service()
        if svc is None:
            raise HTTPException(503, "Altcoin service not initialised")
        return svc.get_recent_candles(symbol=sym, interval=interval, limit=1000)
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    return _btc_service.get_all_time_history(interval=interval)


@app.get("/api/btc/candles")
def get_btc_candles(interval: str = "15m", limit: int = 200, symbol: str = "BTCUSDT"):
    """Recent candles for trading chart windows (BTC default, ETH/SOL compatible)."""
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        svc = _ensure_altcoin_service()
        if svc is None:
            raise HTTPException(503, "Altcoin service not initialised")
        limit = max(50, min(limit, 1200))
        return svc.get_recent_candles(symbol=sym, interval=interval, limit=limit)
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    limit = max(50, min(limit, 1000))
    return _btc_service.get_recent_candles(interval=interval, limit=limit)


@app.get("/api/crypto/candles")
def get_crypto_candles(symbol: str = "ETHUSDT", interval: str = "15m", limit: int = 300):
    """Recent spot candles for non-BTC crypto terminal (ETH/SOL)."""
    svc = _ensure_altcoin_service()
    if svc is None:
        raise HTTPException(503, "Altcoin service not initialised")
    return svc.get_recent_candles(symbol=symbol, interval=interval, limit=limit)


@app.get("/api/crypto/deep-signal")
def get_crypto_deep_signal(symbol: str = "ETHUSDT", interval: str = "15m"):
    """BTC-style deep signal pipeline for ETH/SOL without touching BTC service."""
    svc = _ensure_altcoin_service()
    if svc is None:
        raise HTTPException(503, "Altcoin service not initialised")
    return svc.get_realtime_signal(symbol=symbol, interval=interval)


@app.get("/api/crypto/market-context")
def get_crypto_market_context(symbol: str = "ETHUSDT", interval: str = "15m"):
    """Return market context block from the deep signal payload."""
    svc = _ensure_altcoin_service()
    if svc is None:
        raise HTTPException(503, "Altcoin service not initialised")
    payload = svc.get_realtime_signal(symbol=symbol, interval=interval)
    return {
        "symbol": str(payload.get("symbol") or symbol).upper(),
        "interval": interval,
        "market_context": payload.get("market_context", {}),
        "gate_status": payload.get("gate_status", {}),
        "as_of_utc": payload.get("as_of_utc"),
    }


@app.get("/api/btc/signal")
def get_btc_signal(interval: str = "5m", symbol: str = "BTCUSDT"):
    """Real-time signal endpoint with BTC default; ETH/SOL compatible via symbol."""
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        try:
            return _alt_signal_payload(sym, interval=interval)
        except Exception:
            return {
                "asset": sym,
                "signal": "HOLD",
                "validated_signal": "HOLD",
                "validated": False,
                "reason": "altcoin signal unavailable",
                "is_binding": False,
                "signal_authority": "dashboard_fallback_advisory",
                "signal_note": "altcoin service unavailable - advisory only",
            }
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    try:
        return _btc_service.get_realtime_signal(interval=interval)
    except Exception:
        try:
            return {
                "asset": "BTCUSDT",
                "signal": "HOLD",
                "validated_signal": "HOLD",
                "validated": False,
                "reason": "dashboard BTC signal unavailable",
                "is_binding": False,
                "signal_authority": "dashboard_fallback_advisory",
                "signal_note": "btc_intelligence unavailable - advisory only",
            }
        except Exception:
            raise HTTPException(503, "BTC signal unavailable")


@app.get("/api/btc/market-context")
def get_btc_market_context(interval: str = "5m", symbol: str = "BTCUSDT"):
    """Current context used by the live signal (BTC default, ETH/SOL compatible)."""
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        sig = _alt_signal_payload(sym, interval=interval)
        return sig.get("market_context", {})
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    return _btc_service.get_market_context(interval=interval)


@app.get("/api/btc/signal/history")
def signal_history(request: Request, response: Response, limit: int = 50, symbol: str = "BTCUSDT"):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    sym = _normalize_crypto_symbol(symbol)
    ticker_filter = sym if (_is_alt_symbol(sym) or sym == "BTCUSDT") else None
    lim = max(1, min(int(limit), 1000))
    stats_rows = _combined_signal_history_rows(
        limit=max(lim, 5000),
        ticker=ticker_filter,
        paper_engine=_paper_engine_for_request(request),
    )
    return {"signals": stats_rows[:lim], "stats": _signal_history_stats_from_rows(stats_rows)}


@app.get("/api/btc/signal/stats")
def signal_stats(response: Response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    rows = _combined_signal_history_rows(limit=5000, ticker="BTCUSDT")
    return _signal_history_stats_from_rows(rows)


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
def get_btc_markers(interval: str = "1d", limit: int = 1000, symbol: str = "BTCUSDT"):
    """Historical LONG/SHORT markers (BTC default, ETH/SOL compatible)."""
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        return _alt_markers(sym, interval=interval, limit=min(max(limit, 100), 1200))
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    return _btc_service.get_signal_markers(interval=interval, limit=limit)


@app.get("/api/btc/sr-levels")
def get_sr_levels(interval: str = "15m", limit: int = 200, symbol: str = "BTCUSDT"):
    """Support and resistance levels computed from recent candles."""
    sym = _normalize_crypto_symbol(symbol)
    if (not _is_alt_symbol(sym)) and _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    try:
        if _is_alt_symbol(sym):
            df = _alt_frame(sym, interval=interval, limit=limit)
        else:
            df = _btc_service.get_recent_frame(interval=interval, limit=limit)
        if df is None or df.empty:
            return {"support": [], "resistance": [], "interval": interval}
        highs = df["High"].values
        lows  = df["Low"].values
        closes = df["Close"].values
        current = float(closes[-1])
        # Simple pivot-based S/R: use rolling window peaks/troughs
        window = 10
        supports: list[float] = []
        resistances: list[float] = []
        for i in range(window, len(lows) - window):
            lo = float(lows[i])
            hi = float(highs[i])
            if lo == min(lows[i - window: i + window]):
                supports.append(round(lo, 2))
            if hi == max(highs[i - window: i + window]):
                resistances.append(round(hi, 2))
        # Cluster nearby levels (within 0.3%)
        def cluster(levels: list[float]) -> list[float]:
            if not levels:
                return []
            levels = sorted(set(levels))
            clustered: list[float] = [levels[0]]
            for lvl in levels[1:]:
                if abs(lvl - clustered[-1]) / max(clustered[-1], 1) > 0.003:
                    clustered.append(lvl)
            return clustered
        sup_clean  = [s for s in cluster(supports)   if s < current][-6:]
        res_clean  = [r for r in cluster(resistances) if r > current][:6]
        return {
            "support":    sup_clean,
            "resistance": res_clean,
            "current":    round(current, 2),
            "interval":   interval,
        }
    except Exception as exc:
        return {"support": [], "resistance": [], "error": str(exc)}


@app.get("/api/btc/liquidity-zones")
def get_liquidity_zones(interval: str = "15m", limit: int = 300, symbol: str = "BTCUSDT"):
    """Liquidity zones (high-volume price clusters) from recent candles."""
    sym = _normalize_crypto_symbol(symbol)
    if (not _is_alt_symbol(sym)) and _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    try:
        if _is_alt_symbol(sym):
            df = _alt_frame(sym, interval=interval, limit=limit)
        else:
            df = _btc_service.get_recent_frame(interval=interval, limit=limit)
        if df is None or df.empty:
            return {"zones": [], "interval": interval}
        closes  = df["Close"].values
        volumes = df["Volume"].values if "Volume" in df.columns else df.get("volume", df["Close"] * 0).values
        current = float(closes[-1])
        # Find high-volume candles (top 15%)
        vol_threshold = float(sorted(volumes)[int(len(volumes) * 0.85)])
        zones: list[dict] = []
        for i, (c, v) in enumerate(zip(closes, volumes)):
            if float(v) >= vol_threshold:
                lo = float(df["Low"].values[i])
                hi = float(df["High"].values[i])
                zones.append({
                    "price":  round(float(c), 2),
                    "low":    round(lo, 2),
                    "high":   round(hi, 2),
                    "volume": round(float(v), 4),
                    "type":   "supply" if float(c) > current else "demand",
                })
        # Keep top 10 by volume
        zones = sorted(zones, key=lambda z: z["volume"], reverse=True)[:10]
        return {"zones": zones, "current": round(current, 2), "interval": interval}
    except Exception as exc:
        return {"zones": [], "error": str(exc)}


@app.get("/api/btc/news")
def btc_news(limit: int = 8, symbol: str = "BTCUSDT"):
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        from src.data.news_feed import get_news_for_asset
        return {"news": get_news_for_asset(_asset_short(sym), "crypto", limit)}
    return {"news": get_btc_news(limit)}


@app.get("/api/btc/orderflow")
async def btc_orderflow_proxy(symbol: str = "BTCUSDT", interval: str = Query("15m")):
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        sig = _alt_signal_payload(sym, interval=interval)
        ctx = sig.get("market_context", {}) if isinstance(sig.get("market_context"), dict) else {}
        liq = ctx.get("liquidity", {}) if isinstance(ctx.get("liquidity"), dict) else {}
        binance = liq.get("binance", {}) if isinstance(liq.get("binance"), dict) else {}
        decision = str(sig.get("validated_signal") or sig.get("signal") or "HOLD").upper()
        if decision == "LONG":
            decision_state = "FAVOR_LONG"
        elif decision == "SHORT":
            decision_state = "FAVOR_SHORT"
        else:
            decision_state = "NO_TRADE"
        return {
            "decision_state": decision_state,
            "decision": decision_state,
            "reason": sig.get("reason", "--"),
            "obi": float(binance.get("book_imbalance", 0.0) or 0.0),
            "cvd_slope": float(ctx.get("obv_slope", 0.0) or 0.0),
            "stale": False,
        }
    return await _btc_proxy_payload(
        name="orderflow",
        redis_key="btc:orderflow",
        upstream_url="http://127.0.0.1:9000/api/orderflow",
    )


@app.get("/api/btc/volume")
async def btc_volume_proxy(symbol: str = "BTCUSDT", interval: str = Query("15m")):
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        sig = _alt_signal_payload(sym, interval=interval)
        entry = float(sig.get("entry_price", 0.0) or 0.0)
        tp = float(sig.get("take_profit", 0.0) or 0.0)
        sl = float(sig.get("stop_loss", 0.0) or 0.0)
        vol_ratio = float(((sig.get("market_context") or {}).get("volume_ratio", 1.0) or 1.0))
        decision = str(sig.get("validated_signal") or sig.get("signal") or "HOLD").upper()
        if decision == "LONG":
            decision_state = "FAVOR_LONG"
        elif decision == "SHORT":
            decision_state = "FAVOR_SHORT"
        else:
            decision_state = "NO_TRADE"
        return {
            "decision_state": decision_state,
            "decision": decision_state,
            "poc_price": round(entry, 4) if entry > 0 else None,
            "hvn": [round(tp, 4)] if tp > 0 else [],
            "lvn": [round(sl, 4)] if sl > 0 else [],
            "distance_from_poc_pct": 0.0,
            "window_minutes": 120,
            "volume_ratio": round(vol_ratio, 4),
            "stale": False,
        }
    return await _btc_proxy_payload(
        name="volume",
        redis_key="btc:volprofile",
        upstream_url="http://127.0.0.1:9000/api/volume-profile",
    )


@app.get("/api/btc/volatility")
async def btc_volatility_proxy(symbol: str = "BTCUSDT", interval: str = Query("15m")):
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        sig = _alt_signal_payload(sym, interval=interval)
        ctx = sig.get("market_context", {}) if isinstance(sig.get("market_context"), dict) else {}
        atr_pct = float(ctx.get("atr_pct", 0.0) or 0.0)
        if atr_pct < 0.6:
            regime = "LOW"
            tradeability = "REDUCE_SIZE"
            size_multiplier = 0.7
        elif atr_pct < 1.8:
            regime = "NORMAL"
            tradeability = "ALLOW"
            size_multiplier = 1.0
        elif atr_pct < 2.8:
            regime = "EXPANSION"
            tradeability = "CAUTION"
            size_multiplier = 0.75
        else:
            regime = "HIGH_VOL"
            tradeability = "NO_TRADE"
            size_multiplier = 0.4
        return {
            "tradeability": tradeability,
            "regime": regime,
            "atr_pct": round(atr_pct, 4),
            "size_multiplier": size_multiplier,
            "stale": False,
        }
    return await _btc_proxy_payload(
        name="volatility",
        redis_key="btc:volatility",
        upstream_url="http://127.0.0.1:9000/api/volatility",
    )


@app.get("/api/btc/execution")
async def btc_execution_proxy(symbol: str = "BTCUSDT", interval: str = Query("15m")):
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        sig = _alt_signal_payload(sym, interval=interval)
        plan = _alt_execution_plan_from_signal(sig)
        return {
            "symbol": sym,
            "decision": sig.get("validated_signal") or sig.get("signal") or "HOLD",
            "execution_plan": plan,
            "entry_zone": plan.get("entry_zone"),
            "stop_loss": plan.get("stop_loss"),
            "take_profit": plan.get("take_profit"),
            "take_profit_2": plan.get("take_profit_2"),
            "expected_rr": plan.get("expected_rr"),
            "slippage_risk": plan.get("slippage_risk"),
            "stale": False,
        }
    return await _btc_proxy_payload(
        name="execution",
        redis_key="btc:execution",
        upstream_url="http://127.0.0.1:9000/api/execution",
    )


@app.get("/api/btc/decision-intelligence")
async def btc_decision_intelligence_proxy(interval: str = Query("15m"), symbol: str = "BTCUSDT"):
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        sig = _alt_signal_payload(sym, interval=interval)
        payload = _alt_decision_intelligence_from_signal(sig)
        normalized = _normalize_decision_intelligence_payload(payload)
        if not isinstance(normalized.get("decision_breakdown"), dict):
            normalized["decision_breakdown"] = {}
        return normalized

    iv = interval if interval in INTERVAL_TO_MS else "15m"
    payload = await _btc_proxy_payload(
        name="decision_intelligence",
        redis_key=["btc:intelligence", "btc:decision", "btc:signal"],
        upstream_url=[
            "http://127.0.0.1:9000/api/decision",
            "http://127.0.0.1:9000/api/intelligence",
            "http://127.0.0.1:9000/signal",
        ],
        per_try_timeout=2.5,
        retries_per_url=1,
        retry_sleep_sec=0.0,
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
                    source="dashboard_signal_live",
                    stale=False,
                    degraded=False,
                )
        except Exception as exc:
            logger.warning("decision-intelligence dashboard signal fallback failed: %s", exc)

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
                "http://127.0.0.1:9000/api/probability",
                "http://127.0.0.1:9000/api/intelligence",
                "http://127.0.0.1:9000/api/decision",
                "http://127.0.0.1:9000/signal",
            ],
            per_try_timeout=2.5,
            retries_per_url=1,
            retry_sleep_sec=0.0,
        )
        normalized["probability"] = _extract_probability_payload(prob_payload if isinstance(prob_payload, dict) else {})

    if not normalized.get("execution_plan"):
        exec_payload = await _btc_proxy_payload(
            name="execution_plan_for_decision",
            redis_key=["btc:execution_plan", "btc:intelligence"],
            upstream_url=[
                "http://127.0.0.1:9000/api/execution-plan",
                "http://127.0.0.1:9000/api/intelligence",
                "http://127.0.0.1:9000/api/execution",
            ],
            per_try_timeout=2.5,
            retries_per_url=1,
            retry_sleep_sec=0.0,
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
    return normalized


@app.get("/api/btc/probability")
async def btc_probability_proxy(symbol: str = "BTCUSDT", interval: str = Query("15m")):
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        sig = _alt_signal_payload(sym, interval=interval)
        prob = _alt_prob_from_signal(sig)
        prob["stale"] = False
        return prob

    payload = await _btc_proxy_payload(
        name="probability",
        redis_key=["btc:probability", "btc:intelligence", "btc:signal"],
        upstream_url=[
            "http://127.0.0.1:9000/api/probability",
            "http://127.0.0.1:9000/api/intelligence",
            "http://127.0.0.1:9000/api/decision",
            "http://127.0.0.1:9000/signal",
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
async def btc_execution_plan_proxy(symbol: str = "BTCUSDT", interval: str = Query("15m")):
    sym = _normalize_crypto_symbol(symbol)
    if _is_alt_symbol(sym):
        sig = _alt_signal_payload(sym, interval=interval)
        return _alt_execution_plan_from_signal(sig)
    return await _btc_proxy_payload(
        name="execution_plan",
        redis_key="btc:execution_plan",
        upstream_url="http://127.0.0.1:9000/api/execution-plan",
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

def _paper_user_key(request: Request) -> str:
    user_id, email = _request_user_identity(request)
    key = str(user_id or "").strip()
    if key:
        return key
    fallback = str(email or "").strip()
    if fallback:
        return fallback
    return "local-default"


def _paper_engine_for_request(request: Request):
    from src.paper_trading import get_user_paper_engine

    return get_user_paper_engine(_paper_user_key(request))


def _paper_executor_for_request(request: Request):
    from src.paper_trading import get_user_auto_executor

    return get_user_auto_executor(_paper_user_key(request))


def _start_paper_executor_if_needed(executor: Any) -> None:
    if bool(getattr(executor, "_running", False)):
        return
    import threading

    thread = threading.Thread(target=executor.start, daemon=True, name="paper-auto-executor")
    thread.start()


def _ensure_paper_auto_runtime(engine: Any, executor: Any) -> None:
    try:
        if str(getattr(engine, "mode", "manual")).lower() == "auto":
            _start_paper_executor_if_needed(executor)
    except Exception as exc:
        logger.debug("Paper auto runtime ensure skipped: %s", exc)


def _live_btc_mark_price(interval: str = "5m") -> float | None:
    """Best-effort live BTC price used to guard paper execution and refresh PnL."""
    try:
        if _btc_service is None:
            return None
        sig = _btc_service.get_realtime_signal(interval=interval) or {}
        candidates = [
            sig.get("mark_price"),
            ((sig.get("market_context") or {}).get("futures") or {}).get("mark_price"),
            sig.get("current_price"),
            sig.get("price"),
            sig.get("last_price"),
            sig.get("close"),
        ]
        for candidate in candidates:
            try:
                px = float(candidate)
            except Exception:
                px = 0.0
            if px > 0:
                return px
    except Exception as exc:
        logger.debug("Live BTC mark lookup skipped: %s", exc)
    return None


def _live_crypto_mark_price(symbol: str = "BTCUSDT", interval: str = "5m") -> float | None:
    sym = _normalize_crypto_symbol(symbol)
    if sym == "BTCUSDT":
        return _live_btc_mark_price(interval=interval)
    if not _is_alt_symbol(sym):
        return None
    try:
        svc = _ensure_altcoin_service()
        if svc is None:
            return None
        sig = svc.get_realtime_signal(symbol=sym, interval=interval) or {}
        candidates = [
            sig.get("mark_price"),
            sig.get("current_price"),
            sig.get("entry_price"),
            sig.get("entry"),
            sig.get("price"),
            sig.get("close"),
        ]
        for candidate in candidates:
            try:
                px = float(candidate)
            except Exception:
                px = 0.0
            if px > 0:
                return px
        candles = svc.get_recent_candles(symbol=sym, interval=interval, limit=80) or {}
        data = candles.get("data") if isinstance(candles, dict) else []
        if isinstance(data, list) and data:
            px = float(data[-1].get("close", 0.0) or 0.0)
            return px if px > 0 else None
    except Exception as exc:
        logger.debug("Live %s mark lookup skipped: %s", sym, exc)
    return None


def _paper_execution_market_guard(
    signal: dict[str, Any],
    live_price: float | None,
    *,
    entry_zone_low: Any = None,
    entry_zone_high: Any = None,
    enforce_entry_zone: bool = False,
) -> str | None:
    try:
        live = float(live_price or 0.0)
    except Exception:
        live = 0.0
    if live <= 0:
        return None

    symbol = _normalize_crypto_symbol(str(signal.get("ticker") or "BTCUSDT"))
    symbol_label = _asset_short(symbol)
    direction = str(signal.get("signal") or signal.get("direction") or "").strip().upper()
    try:
        entry = float(signal.get("entry_price") or 0.0)
        stop = float(signal.get("stop_loss") or 0.0)
        take_profit = float(signal.get("take_profit") or 0.0)
    except Exception:
        return None
    if entry <= 0 or stop <= 0 or take_profit <= 0:
        return None

    min_tolerance = 5.0 if symbol == "BTCUSDT" else 0.01
    tolerance = max(abs(entry) * 0.0005, min_tolerance)
    if enforce_entry_zone:
        try:
            zone_low = float(entry_zone_low or 0.0)
            zone_high = float(entry_zone_high or 0.0)
        except Exception:
            zone_low = 0.0
            zone_high = 0.0
        if zone_low > 0 and zone_high > 0:
            low, high = sorted((zone_low, zone_high))
            if live < low - tolerance or live > high + tolerance:
                return (
                    f"Live {symbol_label} price {live:.2f} is outside execution entry zone "
                    f"{low:.2f}-{high:.2f}. Refresh the plan before trading."
                )

    if direction == "LONG":
        if live <= stop:
            return f"Live {symbol_label} price {live:.2f} is already at/below LONG stop loss {stop:.2f}."
        if live >= take_profit:
            return f"Live {symbol_label} price {live:.2f} is already at/above LONG take profit {take_profit:.2f}."
    elif direction == "SHORT":
        if live >= stop:
            return f"Live {symbol_label} price {live:.2f} is already at/above SHORT stop loss {stop:.2f}."
        if live <= take_profit:
            return f"Live {symbol_label} price {live:.2f} is already at/below SHORT take profit {take_profit:.2f}."
    return None


@app.get("/api/paper/portfolio")
def paper_portfolio(request: Request):
    """Live portfolio metrics + positions."""
    try:
        engine = _paper_engine_for_request(request)
        executor = _paper_executor_for_request(request)
        _ensure_paper_auto_runtime(engine, executor)

        # Push live crypto marks into paper engine so SL/TP auto-close works for manual trades.
        try:
            open_rows = engine.get_open_positions()
            price_map: dict[str, float] = {}
            for row in open_rows:
                ticker_raw = str(row.get("ticker", "")).upper().strip()
                if not ticker_raw:
                    continue
                sym = _normalize_crypto_symbol(ticker_raw.replace("-", ""))
                live_price = _live_crypto_mark_price(sym, interval="5m")
                if live_price and live_price > 0:
                    price_map[ticker_raw] = live_price
                    price_map[ticker_raw.replace("-", "")] = live_price
                    if sym.endswith("USDT"):
                        price_map[sym] = live_price
                        price_map[f"{sym[:-4]}-USDT"] = live_price
            if price_map:
                closed_rows = engine.update_prices(price_map) or []
                for closed in closed_rows:
                    push_broadcast_threadsafe(
                        {
                            "type": "paper_trade_update",
                            "action": "closed",
                            "ticker": str(closed.get("ticker", "")),
                            "pnl_usd": float(closed.get("pnl", 0.0) or 0.0),
                            "reason": str(closed.get("reason", "")),
                        },
                    )
        except Exception as sync_exc:
            logger.debug("Paper price sync skipped: %s", sync_exc)

        metrics = engine.get_portfolio_metrics()
        raw_positions = engine.get_open_positions()

        def _duration_str(held_hours: float) -> str:
            try:
                total_seconds = max(0, int(round(float(held_hours) * 3600.0)))
            except Exception:
                total_seconds = 0
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            if hours > 0:
                return f"{hours}h {minutes}m"
            if minutes > 0:
                return f"{minutes}m {seconds}s"
            return f"{seconds}s"

        positions = []
        for row in raw_positions:
            try:
                entry_price = float(row.get("entry_price", 0.0) or 0.0)
            except Exception:
                entry_price = 0.0
            try:
                current_price = float(row.get("current_price", entry_price) or entry_price)
            except Exception:
                current_price = entry_price
            try:
                quantity = float(row.get("quantity", 0.0) or 0.0)
            except Exception:
                quantity = 0.0

            pnl_usd = float(row.get("unrealized_pnl", row.get("pnl_usd", 0.0)) or 0.0)
            pnl_pct_raw = row.get("unrealized_pnl_pct", row.get("pnl_pct"))
            if pnl_pct_raw is None:
                denom = entry_price * quantity
                pnl_pct = (pnl_usd / denom * 100.0) if denom > 0 else 0.0
            else:
                try:
                    pnl_pct = float(pnl_pct_raw)
                except Exception:
                    pnl_pct = 0.0

            try:
                sl_val = float(row.get("stop_loss", row.get("sl", 0.0)) or 0.0)
            except Exception:
                sl_val = 0.0
            try:
                tp_val = float(row.get("take_profit", row.get("tp1", row.get("tp", 0.0))) or 0.0)
            except Exception:
                tp_val = 0.0

            held_hours = float(row.get("held_hours", 0.0) or 0.0)
            position = dict(row)
            position.update(
                {
                    "id": str(row.get("trade_id", row.get("id", ""))),
                    "pnl_usd": round(pnl_usd, 6),
                    "pnl_pct": round(pnl_pct, 6),
                    "sl": sl_val if sl_val > 0 else None,
                    "tp1": tp_val if tp_val > 0 else None,
                    "duration_str": _duration_str(held_hours),
                },
            )
            positions.append(position)
        return {
            "metrics": metrics,
            "open_positions": positions,
            "mode": engine._state.get("mode", "manual"),
            "auto_running": bool(getattr(executor, "_running", False)),
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
def paper_trades(request: Request, limit: int = 50):
    """Closed trade history."""
    return _paper_engine_for_request(request).get_closed_trades(limit)


@app.post("/api/paper/execute")
async def paper_execute(payload: dict, request: Request):
    """
    Manually execute a paper trade.
    Body: { ticker, direction|signal, entry_price, stop_loss|sl,
            take_profit|tp1|tp, confidence, asset_class, mode,
            sizing_mode, capital_to_use, capital_pct }
    """
    engine = _paper_engine_for_request(request)
    mode = "auto" if str(payload.get("mode", "manual")).lower() == "auto" else "manual"
    signal = {
        "ticker": str(payload.get("ticker") or payload.get("asset") or "").strip().upper(),
        "signal": str(payload.get("signal") or payload.get("direction") or "").strip().upper(),
        "entry_price": payload.get("entry_price"),
        "stop_loss": payload.get("stop_loss", payload.get("sl")),
        "take_profit": payload.get("take_profit", payload.get("tp1", payload.get("tp"))),
        "confidence": payload.get("confidence"),
        "alpha_score": payload.get("alpha_score", payload.get("confidence")),
        "regime": payload.get("regime"),
        "asset_class": payload.get("asset_class", "crypto"),
        "strength": payload.get("strength"),
        "sizing_mode": payload.get("sizing_mode"),
        "capital_to_use": payload.get("capital_to_use", payload.get("trade_capital_usd")),
        "capital_pct": payload.get("capital_pct", payload.get("trade_capital_pct")),
        "position_size_pct": payload.get("position_size_pct"),
        "position_sizing": payload.get("position_sizing"),
        "meta_controls": payload.get("meta_controls"),
        "last_calibration_timestamp": payload.get("last_calibration_timestamp"),
    }
    signal["ticker"] = _normalize_crypto_symbol(signal["ticker"] or "BTCUSDT")
    before_metrics = engine.get_portfolio_metrics()
    capital_before = float(before_metrics.get("capital", 0.0) or 0.0)
    try:
        payload_price = float(payload.get("current_price_snapshot") or payload.get("market_price") or 0.0)
    except Exception:
        payload_price = 0.0
    if _is_alt_symbol(signal["ticker"]) and payload_price > 0:
        live_price = payload_price
    else:
        live_price = _live_crypto_mark_price(signal["ticker"], interval="5m")
        if live_price is None and payload_price > 0:
            live_price = payload_price
    guard_msg = _paper_execution_market_guard(
        signal,
        live_price,
        entry_zone_low=payload.get("entry_zone_low"),
        entry_zone_high=payload.get("entry_zone_high"),
        enforce_entry_zone=str(payload.get("execution_source") or "").strip().lower() == "execution_plan",
    )
    if guard_msg:
        return {
            "success": False,
            "message": guard_msg,
            "live_price": round(float(live_price or 0.0), 2),
            "blocked_by": "live_price_guard",
        }
    result = engine.execute_trade(signal, mode=mode)
    if result.get("success"):
        used_capital = float(result.get("value", 0.0) or 0.0)
        after_metrics = engine.get_portfolio_metrics()
        capital_after = float(after_metrics.get("capital", 0.0) or 0.0)
        base_capital = capital_before if capital_before > 0 else (capital_after + used_capital)
        result["capital_before"] = round(base_capital, 4)
        result["capital_after"] = round(capital_after, 4)
        result["capital_used"] = round(used_capital, 4)
        result["capital_used_pct"] = round(((used_capital / base_capital) * 100.0) if base_capital > 0 else 0.0, 4)
        result["risk_budget"] = round(base_capital * 0.02, 4)
        result["max_trade_value"] = round(base_capital * 0.10, 4)
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
async def paper_close(payload: dict, request: Request):
    """
    Close an open position.
    Body: { ticker, exit_price, reason }
    """
    engine = _paper_engine_for_request(request)
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
        # [ADDITIVE] Mark any stale OPEN signal_history record as CLOSED
        # Prevents TRADE ACTIVE banner showing after manual close
        try:
            import time as _time
            from src.data.signal_history import _load as _sh_load, _save as _sh_save
            _sh_hist = _sh_load()
            _updated = False
            for _rec in reversed(_sh_hist):
                if str(_rec.get("status", "")).upper() == "OPEN":
                    _rec["status"] = "CLOSED"
                    _rec["result"] = "WIN" if float(result.get("pnl", 0)) > 0 else "LOSS"
                    _rec["exit_price"] = float(result.get("exit_price", 0))
                    _rec["pnl_pct"] = round(float(result.get("pnl_pct", 0)), 2)
                    _rec["closed_time"] = _time.time()
                    _updated = True
                    break
            if _updated:
                _sh_save(_sh_hist)
        except Exception:
            pass
        return {"success": True, **result}
    return {"success": False, "error": "No open position"}


@app.post("/api/paper/mode")
async def paper_set_mode(payload: dict, request: Request):
    """
    Set auto/manual mode.
    Body: { mode: "auto" | "manual" }
    """
    mode = str(payload.get("mode", "manual")).lower()
    mode = "auto" if mode == "auto" else "manual"
    engine = _paper_engine_for_request(request)
    engine.set_mode(mode)

    executor = _paper_executor_for_request(request)
    if mode == "auto":
        _start_paper_executor_if_needed(executor)
    elif mode == "manual":
        executor.stop()

    return {"mode": mode, "auto_running": bool(getattr(executor, "_running", False)), "success": True}


@app.post("/api/paper/reset")
async def paper_reset(payload: dict, request: Request):
    """Reset paper account. Body: { capital: float }"""
    capital = float(payload.get("capital", 100000))
    _paper_engine_for_request(request).reset(capital)
    return {"success": True, "capital": capital}


@app.get("/api/paper/pending")
def paper_pending(request: Request):
    """Get signals waiting for manual approval."""
    engine = _paper_engine_for_request(request)
    executor = _paper_executor_for_request(request)
    _ensure_paper_auto_runtime(engine, executor)
    return executor.get_pending_signals()


@app.post("/api/paper/approve")
async def paper_approve(payload: dict, request: Request):
    """Approve a pending signal. Body: { ticker, trade_id }"""
    return _paper_executor_for_request(request).approve_signal(payload["ticker"], payload["trade_id"])


@app.post("/api/paper/reject")
async def paper_reject(payload: dict, request: Request):
    """Reject a pending signal. Body: { ticker, trade_id }"""
    _paper_executor_for_request(request).reject_signal(payload["ticker"], payload["trade_id"])
    return {"success": True}


# ─── BINANCE LIVE ACCOUNT ────────────────────────────────────────────────────

def _binance_signed_request(api_key: str, api_secret: str, path: str, params: dict | None = None) -> Any:
    """Make an HMAC-SHA256 signed GET request to Binance REST API."""
    import hmac as _hmac
    base_params = dict(params or {})
    base_params["timestamp"] = int(time.time() * 1000)
    base_params["recvWindow"] = 10000
    query_string = "&".join(f"{k}={v}" for k, v in sorted(base_params.items()))
    signature = _hmac.new(
        api_secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    url = f"https://api.binance.com{path}?{query_string}&signature={signature}"
    resp = httpx.get(url, headers={"X-MBX-APIKEY": api_key}, timeout=8.0)
    if resp.status_code == 200:
        return resp.json()
    raise ValueError(f"Binance {resp.status_code}: {resp.text[:200]}")


def _binance_get_usdt_prices(assets: list[str]) -> dict[str, float]:
    """Fetch USDT prices for a list of assets using public ticker endpoint."""
    prices: dict[str, float] = {"USDT": 1.0}
    symbols_needed = [a for a in assets if a != "USDT"]
    if not symbols_needed:
        return prices
    try:
        symbols_param = "[" + ",".join(f'"{a}USDT"' for a in symbols_needed) + "]"
        resp = httpx.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbols": symbols_param},
            timeout=6.0,
        )
        if resp.status_code == 200:
            for item in resp.json():
                sym = str(item.get("symbol", ""))
                if sym.endswith("USDT"):
                    asset = sym[:-4]
                    try:
                        prices[asset] = float(item["price"])
                    except Exception:
                        pass
    except Exception:
        pass
    return prices


@app.get("/api/binance/account")
def binance_account(request: Request):
    """Return live Binance spot account balance. Requires user to have saved API keys."""
    uid, _ = _request_user_identity(request)
    user_key = uid or "default"

    # Serve from cache if fresh
    cached = _binance_account_cache.get(user_key)
    if cached and (time.time() - cached["ts"]) < _BINANCE_ACCOUNT_CACHE_TTL:
        return cached["data"]

    # Load and decrypt API keys from user profile
    profiles = _load_user_profiles()
    rec = profiles.get(user_key, {})
    api_key = _decrypt_profile_secret(str(rec.get("binance_api_key", "") or ""))
    api_secret = _decrypt_profile_secret(str(rec.get("binance_api_secret", "") or ""))

    if not api_key or not api_secret:
        result = {"connected": False, "error": "No API keys configured", "balances": []}
        return result

    try:
        account_data = _binance_signed_request(api_key, api_secret, "/api/v3/account")

        can_trade = bool(account_data.get("canTrade", False))
        raw_balances = account_data.get("balances", [])

        # Filter to non-zero balances only
        non_zero = [
            b for b in raw_balances
            if float(b.get("free", 0)) > 0 or float(b.get("locked", 0)) > 0
        ]

        asset_names = [b["asset"] for b in non_zero]
        prices = _binance_get_usdt_prices(asset_names)

        balances = []
        total_usdt = 0.0
        for b in non_zero:
            asset = str(b["asset"])
            free = float(b.get("free", 0))
            locked = float(b.get("locked", 0))
            price = prices.get(asset, 0.0)
            usdt_val = round((free + locked) * price, 2) if price > 0 else None
            if usdt_val is not None:
                total_usdt += usdt_val
            balances.append({
                "asset": asset,
                "free": round(free, 8),
                "locked": round(locked, 8),
                "usdt_value": usdt_val,
            })

        # Sort by USDT value descending
        balances.sort(key=lambda x: x["usdt_value"] or 0, reverse=True)

        result = {
            "connected": True,
            "can_trade": can_trade,
            "total_usdt_value": round(total_usdt, 2),
            "balances": balances,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        _binance_account_cache[user_key] = {"ts": time.time(), "data": result}
        return result

    except ValueError as e:
        err = str(e)
        if "401" in err or "403" in err or "-2014" in err or "-2015" in err:
            return {"connected": False, "error": "Invalid API keys — check key permissions", "balances": []}
        logger.warning("Binance account fetch failed: %s", e)
        return {"connected": False, "error": "Binance request failed", "balances": []}
    except Exception as e:
        logger.warning("Binance account endpoint error: %s", e)
        return {"connected": False, "error": "Internal error", "balances": []}


@app.get("/api/binance/open-orders")
def binance_open_orders(request: Request):
    """Return all open Binance spot orders for the authenticated user."""
    uid, _ = _request_user_identity(request)
    user_key = uid or "default"

    profiles = _load_user_profiles()
    rec = profiles.get(user_key, {})
    api_key = _decrypt_profile_secret(str(rec.get("binance_api_key", "") or ""))
    api_secret = _decrypt_profile_secret(str(rec.get("binance_api_secret", "") or ""))

    if not api_key or not api_secret:
        return []

    try:
        orders = _binance_signed_request(api_key, api_secret, "/api/v3/openOrders")
        result = []
        for o in (orders if isinstance(orders, list) else []):
            result.append({
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "type": o.get("type"),
                "origQty": o.get("origQty"),
                "executedQty": o.get("executedQty"),
                "price": o.get("price"),
                "stopPrice": o.get("stopPrice"),
                "status": o.get("status"),
                "time": o.get("time"),
            })
        return result
    except Exception as e:
        logger.warning("Binance open-orders endpoint error: %s", e)
        return []


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
def get_equity_curve(request: Request, limit: int = 5000, ticker: str | None = None):
    lim = max(1, min(int(limit), 20000))
    rows = _combined_signal_history_rows(
        limit=min(lim, 5000),
        ticker=ticker,
        paper_engine=_paper_engine_for_request(request),
    )
    closed = [
        r
        for r in rows
        if str(r.get("status") or r.get("outcome") or "").upper() not in {"OPEN", "BLOCKED"}
    ]
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
    if _auth_provider() == "supabase":
        raise HTTPException(status_code=404, detail="Local token auth disabled (provider=supabase)")
    cfg = _auth_config()
    username = payload.get("username", "")
    password = payload.get("password", "")
    if username != cfg.get("username") or password != cfg.get("password"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": _issue_token(username), "token_type": "bearer"}


@app.get("/api/auth-check")
def auth_check(request: Request):
    user_id, email = _request_user_identity(request)
    return {"ok": True, "user_id": user_id, "email": email}


@app.get("/api/user/profile")
def get_user_profile(request: Request):
    user_id, email = _request_user_identity(request)
    user_key = str(user_id or email or "").strip()
    if not user_key:
        raise HTTPException(status_code=401, detail="Dashboard auth required")
    profiles = _load_user_profiles()
    rec = profiles.get(user_key, {})
    return _profile_summary(user_key, email, rec)


@app.post("/api/user/profile")
def update_user_profile(payload: dict, request: Request):
    user_id, email = _request_user_identity(request)
    user_key = str(user_id or email or "").strip()
    if not user_key:
        raise HTTPException(status_code=401, detail="Dashboard auth required")

    profiles = _load_user_profiles()
    now_iso = datetime.now(timezone.utc).isoformat()
    rec: dict[str, Any] = dict(profiles.get(user_key, {}))
    if not rec.get("created_at"):
        rec["created_at"] = now_iso

    rec["user_id"] = user_key
    rec["email"] = str(email or rec.get("email") or "").strip().lower()
    rec["updated_at"] = now_iso

    if "display_name" in payload:
        rec["display_name"] = str(payload.get("display_name") or "").strip()

    if bool(payload.get("clear_binance_keys")):
        rec["binance_api_key"] = ""
        rec["binance_api_secret"] = ""
        rec["binance_access_token"] = ""
    else:
        if "binance_api_key" in payload:
            api_key = str(payload.get("binance_api_key") or "").strip()
            if api_key:
                rec["binance_api_key"] = _encrypt_profile_secret(api_key)
        if "binance_api_secret" in payload:
            api_secret = str(payload.get("binance_api_secret") or "").strip()
            if api_secret:
                rec["binance_api_secret"] = _encrypt_profile_secret(api_secret)
        if "binance_access_token" in payload:
            access_token = str(payload.get("binance_access_token") or "").strip()
            if access_token:
                rec["binance_access_token"] = _encrypt_profile_secret(access_token)

    profiles[user_key] = rec
    _save_user_profiles(profiles)

    return {"success": True, "profile": _profile_summary(user_key, email, rec)}


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

def _frontend_auth_config() -> dict[str, Any]:
    supa = _supabase_config()
    return {
        "enabled": _auth_enabled(),
        "provider": _auth_provider(),
        "supabaseUrl": supa["url"],
        "supabaseAnonKey": supa["anon_key"],
    }


def _supabase_auth_bootstrap() -> str:
    cfg_json = json.dumps(_frontend_auth_config())
    bootstrap = """
<script>
(function () {
  const AUTH_CFG = __AUTH_CFG__;
  window.__DASHBOARD_AUTH__ = AUTH_CFG;
  if (!AUTH_CFG || !AUTH_CFG.enabled || AUTH_CFG.provider !== "supabase") return;

  const SUPABASE_URL = String(AUTH_CFG.supabaseUrl || "").trim();
  const SUPABASE_ANON_KEY = String(AUTH_CFG.supabaseAnonKey || "").trim();
  const SUPABASE_CONFIG_READY = !!(SUPABASE_URL && SUPABASE_ANON_KEY);

  let supabaseClient = null;
  let supabasePromise = null;
  let currentToken = "";
  let uiReady = false;

  function projectRefFromUrl(url) {
    try {
      return new URL(url).hostname.split(".")[0] || "";
    } catch (_err) {
      return "";
    }
  }

  const projectRef = projectRefFromUrl(SUPABASE_URL);
  const storageKey = projectRef ? ("sb-" + projectRef + "-auth-token") : "";

  function readStoredToken() {
    if (!storageKey) return "";
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return "";
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        if (typeof parsed.access_token === "string" && parsed.access_token) return parsed.access_token;
        if (parsed.currentSession && typeof parsed.currentSession.access_token === "string") return parsed.currentSession.access_token;
        if (Array.isArray(parsed) && parsed[0] && typeof parsed[0].access_token === "string") return parsed[0].access_token;
      }
    } catch (_err) {}
    return "";
  }

  function readHandoffTokens() {
    const out = { accessToken: "", refreshToken: "" };
    try {
      const hash = window.location.hash && window.location.hash.startsWith("#")
        ? new URLSearchParams(window.location.hash.slice(1))
        : new URLSearchParams();
      const query = new URLSearchParams(window.location.search || "");
      out.accessToken = hash.get("access_token") || query.get("access_token") || "";
      out.refreshToken = hash.get("refresh_token") || query.get("refresh_token") || "";
    } catch (_err) {}
    return out;
  }

  function hasCompleteHandoff(handoff) {
    return !!(handoff && handoff.accessToken && handoff.refreshToken);
  }

  function shouldForcePortalEntry(handoff) {
    return window.location.hostname === "terminal.tradevex.live" && !hasCompleteHandoff(handoff);
  }

  function clearHandoffTokens() {
    try {
      const url = new URL(window.location.href);
      url.hash = "";
      url.searchParams.delete("access_token");
      url.searchParams.delete("refresh_token");
      url.searchParams.delete("source");
      window.history.replaceState({}, document.title, url.pathname + url.search);
    } catch (_err) {}
  }

  currentToken = readStoredToken();

  function isInternalApi(urlValue) {
    try {
      const u = new URL(urlValue, window.location.origin);
      if (u.origin !== window.location.origin) return false;
      return u.pathname.startsWith("/api/");
    } catch (_err) {
      return false;
    }
  }

  const nativeFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    let targetUrl = "";
    if (typeof input === "string") targetUrl = input;
    else if (input && input.url) targetUrl = input.url;

    if (!isInternalApi(targetUrl)) {
      return nativeFetch(input, init);
    }

    const headers = new Headers((init && init.headers) || (input instanceof Request ? input.headers : undefined));
    if (currentToken) headers.set("Authorization", "Bearer " + currentToken);

    if (input instanceof Request) {
      const nextInit = Object.assign({}, init || {});
      nextInit.headers = headers;
      return nativeFetch(input, nextInit);
    }
    const nextInit = Object.assign({}, init || {});
    nextInit.headers = headers;
    return nativeFetch(input, nextInit);
  };

  const NativeWebSocket = window.WebSocket;
  function AuthWebSocket(url, protocols) {
    let nextUrl = url;
    try {
      const u = new URL(url, window.location.origin);
      if (u.origin === window.location.origin && u.pathname === "/ws" && currentToken && !u.searchParams.get("token")) {
        u.searchParams.set("token", currentToken);
      }
      nextUrl = u.toString();
    } catch (_err) {}
    return protocols ? new NativeWebSocket(nextUrl, protocols) : new NativeWebSocket(nextUrl);
  }
  AuthWebSocket.prototype = NativeWebSocket.prototype;
  Object.setPrototypeOf(AuthWebSocket, NativeWebSocket);
  window.WebSocket = AuthWebSocket;

  function ensureUi() {
    if (uiReady) return;
    if (!document.body) return;
    uiReady = true;
    const style = document.createElement("style");
    style.textContent = `
      .sb-auth-overlay{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:#020617;z-index:2147483647}
      .sb-auth-card{width:min(420px,92vw);padding:20px;border-radius:14px;border:1px solid rgba(148,163,184,.3);background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif}
      .sb-auth-title{font-size:18px;font-weight:700;margin:0 0 8px}
      .sb-auth-sub{font-size:12px;color:#94a3b8;margin:0 0 14px}
      .sb-auth-tabs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:0 0 10px}
      .sb-auth-tab{padding:8px 10px;border-radius:8px;border:1px solid rgba(148,163,184,.35);background:#111c31;color:#b7c6de;font-weight:700;cursor:pointer}
      .sb-auth-tab.active{background:#1d4ed8;border-color:#1d4ed8;color:#eaf2ff}
      .sb-auth-pane{display:none}
      .sb-auth-pane.active{display:block}
      .sb-auth-inp{width:100%;padding:10px;border-radius:8px;border:1px solid rgba(148,163,184,.35);background:#0b1220;color:#e2e8f0;margin-bottom:8px}
      .sb-auth-row{display:flex;gap:8px}
      .sb-auth-btn{flex:1;padding:10px;border-radius:8px;border:1px solid rgba(148,163,184,.35);background:#1e293b;color:#e2e8f0;font-weight:600;cursor:pointer}
      .sb-auth-btn.primary{background:#0ea5e9;border-color:#0ea5e9;color:#06253a}
      .sb-auth-btn.ghost{background:#0b1220;color:#b7c6de}
      .sb-auth-msg{min-height:18px;font-size:12px;color:#fda4af;margin-top:8px}
      .sb-auth-user{display:none;align-items:center;gap:8px;padding:8px 10px;border-radius:999px;border:1px solid rgba(148,163,184,.35);background:rgba(15,23,42,.95);color:#e2e8f0;font:12px/1.2 system-ui,sans-serif;max-width:min(94vw,480px)}
      .sb-auth-user button{padding:4px 8px;border-radius:999px;border:1px solid rgba(148,163,184,.35);background:#0b1220;color:#e2e8f0;cursor:pointer;flex-shrink:0}
      .sb-auth-user #sb-auth-user-email{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .sb-auth-user-row{position:relative;z-index:9000;display:flex;justify-content:flex-end;padding:10px 16px 0 16px}
      .sb-profile-fab{display:none;position:fixed;right:16px;bottom:16px;z-index:9002;padding:10px 14px;border-radius:999px;border:1px solid rgba(148,163,184,.4);background:#0b1220;color:#dbeafe;font-weight:700;cursor:pointer}
      .sb-profile-panel{display:none;position:fixed;right:16px;bottom:64px;z-index:9002;width:min(420px,94vw);padding:14px;border-radius:12px;border:1px solid rgba(148,163,184,.35);background:rgba(2,6,23,.98);color:#e2e8f0}
      .sb-profile-title{font-size:15px;font-weight:700;margin:0 0 8px}
      .sb-profile-sub{font-size:11px;color:#93c5fd;margin:0 0 10px}
      .sb-profile-label{display:block;font-size:11px;color:#94a3b8;margin:8px 0 4px}
      .sb-profile-inp{width:100%;padding:9px;border-radius:8px;border:1px solid rgba(148,163,184,.35);background:#0b1220;color:#e2e8f0}
      .sb-profile-help{font-size:11px;color:#94a3b8;margin-top:8px}
      .sb-profile-actions{display:flex;gap:8px;margin-top:10px}
      .sb-profile-btn{flex:1;padding:9px;border-radius:8px;border:1px solid rgba(148,163,184,.35);background:#1e293b;color:#e2e8f0;font-weight:700;cursor:pointer}
      .sb-profile-btn.primary{background:#0284c7;border-color:#0284c7;color:#e0f2fe}
      .sb-profile-msg{min-height:16px;font-size:12px;color:#fda4af;margin-top:8px}
      body.sb-auth-locked > *:not(#sb-auth-overlay){display:none !important}
      body.sb-auth-locked #sb-profile-open, body.sb-auth-locked #sb-profile-panel{display:none !important}
    `;
    document.head.appendChild(style);
    document.body.insertAdjacentHTML("beforeend", `
      <div id="sb-auth-overlay" class="sb-auth-overlay">
        <div class="sb-auth-card">
          <h3 class="sb-auth-title">Secure Login</h3>
          <p class="sb-auth-sub">Use your Supabase account to access trading dashboard.</p>
          <div class="sb-auth-tabs">
            <button id="sb-tab-login" class="sb-auth-tab active" type="button">Login</button>
            <button id="sb-tab-signup" class="sb-auth-tab" type="button">Sign Up</button>
          </div>

          <div id="sb-pane-login" class="sb-auth-pane active">
            <input id="sb-auth-email" class="sb-auth-inp" type="email" placeholder="Email" autocomplete="username" />
            <input id="sb-auth-password" class="sb-auth-inp" type="password" placeholder="Password" autocomplete="current-password" />
            <div class="sb-auth-row">
              <button id="sb-auth-login" class="sb-auth-btn primary" type="button">Login</button>
            </div>
            <div class="sb-auth-row">
              <button id="sb-auth-forgot" class="sb-auth-btn ghost" type="button">Forgot password</button>
            </div>
          </div>

          <div id="sb-pane-signup" class="sb-auth-pane">
            <input id="sb-signup-email" class="sb-auth-inp" type="email" placeholder="Email" autocomplete="email" />
            <input id="sb-signup-password" class="sb-auth-inp" type="password" placeholder="Create password" autocomplete="new-password" />
            <input id="sb-signup-confirm" class="sb-auth-inp" type="password" placeholder="Confirm password" autocomplete="new-password" />
            <div class="sb-auth-row">
              <button id="sb-auth-signup" class="sb-auth-btn primary" type="button">Create Account</button>
            </div>
          </div>
          <div id="sb-auth-msg" class="sb-auth-msg"></div>
        </div>
      </div>
      <div class="sb-auth-user-row">
        <div id="sb-auth-user" class="sb-auth-user">
          <span id="sb-auth-user-email"></span>
          <button id="sb-auth-logout" type="button">Logout</button>
        </div>
      </div>
      <button id="sb-profile-open" class="sb-profile-fab" type="button">Profile</button>
      <div id="sb-profile-panel" class="sb-profile-panel">
        <h4 class="sb-profile-title">Account Profile</h4>
        <p class="sb-profile-sub">Add Binance keys for your own real account execution.</p>
        <label class="sb-profile-label" for="sb-profile-email">Email</label>
        <input id="sb-profile-email" class="sb-profile-inp" type="text" readonly />
        <label class="sb-profile-label" for="sb-profile-display">Display Name</label>
        <input id="sb-profile-display" class="sb-profile-inp" type="text" placeholder="Optional name" />
        <label class="sb-profile-label" for="sb-profile-api-key">Binance API Key</label>
        <input id="sb-profile-api-key" class="sb-profile-inp" type="text" placeholder="Paste API key" />
        <label class="sb-profile-label" for="sb-profile-access-token">Binance Access Token (optional)</label>
        <input id="sb-profile-access-token" class="sb-profile-inp" type="text" placeholder="Paste access token" />
        <label class="sb-profile-label" for="sb-profile-api-secret">Binance API Secret</label>
        <input id="sb-profile-api-secret" class="sb-profile-inp" type="password" placeholder="Leave blank to keep saved secret" />
        <div class="sb-profile-help" id="sb-profile-status"></div>
        <div class="sb-profile-actions">
          <button id="sb-profile-save" class="sb-profile-btn primary" type="button">Save</button>
          <button id="sb-profile-clear" class="sb-profile-btn" type="button">Clear Keys</button>
          <button id="sb-profile-close" class="sb-profile-btn" type="button">Close</button>
        </div>
        <div id="sb-profile-msg" class="sb-profile-msg"></div>
      </div>
    `);
    document.body.classList.add("sb-auth-locked");
  }

  function setMessage(msg, isError) {
    const el = document.getElementById("sb-auth-msg");
    if (!el) return;
    el.textContent = msg || "";
    el.style.color = isError ? "#fda4af" : "#86efac";
  }

  async function verifyDashboardAccess(token) {
    if (!token) return false;
    try {
      const res = await nativeFetch("/api/auth-check", {
        headers: { "Authorization": "Bearer " + token },
        cache: "no-store",
      });
      return !!res.ok;
    } catch (_err) {
      return false;
    }
  }

  async function renderSession(session) {
    const overlay = document.getElementById("sb-auth-overlay");
    const userChip = document.getElementById("sb-auth-user");
    const userEmail = document.getElementById("sb-auth-user-email");
    const profileOpen = document.getElementById("sb-profile-open");
    const profilePanel = document.getElementById("sb-profile-panel");
    const email = session && session.user ? (session.user.email || session.user.id || "") : "";
    currentToken = session && session.access_token ? session.access_token : "";
    
    if (!session) {
      window.location.href = "https://tradevex.live/terminal";
      return;
    }

    const canAccess = await verifyDashboardAccess(currentToken);
    if (!canAccess) {
      try {
        const c = await loadSupabaseClient();
        await c.auth.signOut();
      } catch (_err) {}
      currentToken = "";
      window.location.href = "https://tradevex.live/terminal?error=unauthorized";
      return;
    }

    if (overlay) overlay.style.display = "none";
    if (document.body) document.body.classList.remove("sb-auth-locked");
    if (userChip) userChip.style.display = "inline-flex";
    if (userEmail) userEmail.textContent = email || "";
    if (profileOpen) profileOpen.style.display = "inline-flex";
  }

  async function loadSupabaseClient() {
    if (!SUPABASE_CONFIG_READY) {
      throw new Error("SUPABASE_URL / SUPABASE_ANON_KEY missing on server config");
    }
    if (supabaseClient) return supabaseClient;
    if (!supabasePromise) {
      supabasePromise = new Promise(function (resolve, reject) {
        if (window.supabase && typeof window.supabase.createClient === "function") {
          resolve(window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY));
          return;
        }
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2";
        script.async = true;
        script.onload = function () {
          if (!window.supabase || typeof window.supabase.createClient !== "function") {
            reject(new Error("Supabase SDK unavailable"));
            return;
          }
          resolve(window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY));
        };
        script.onerror = function () { reject(new Error("Failed to load Supabase SDK")); };
        document.head.appendChild(script);
      });
    }
    supabaseClient = await supabasePromise;
    return supabaseClient;
  }

  async function initAuthUi() {
    ensureUi();
    if (!SUPABASE_CONFIG_READY) {
      const overlay = document.getElementById("sb-auth-overlay");
      if (overlay) overlay.style.display = "flex";
      setMessage("Server auth config missing. Set SUPABASE_URL and SUPABASE_ANON_KEY.", true);
      return;
    }
    try {
      const client = await loadSupabaseClient();
      const handoff = readHandoffTokens();
      if (shouldForcePortalEntry(handoff)) {
        try { await client.auth.signOut(); } catch (_err) {}
        currentToken = "";
        window.location.replace("https://tradevex.live/");
        return;
      }
      if (handoff.accessToken || handoff.refreshToken) {
        if (handoff.accessToken && handoff.refreshToken) {
          const handoffRes = await client.auth.setSession({
            access_token: handoff.accessToken,
            refresh_token: handoff.refreshToken,
          });
          clearHandoffTokens();
          if (handoffRes && handoffRes.error) {
            setMessage("Secure terminal link expired. Please log in again from tradevex.live.", true);
          } else {
            currentToken = handoff.accessToken;
          }
        } else {
          clearHandoffTokens();
          setMessage("Incomplete terminal handoff. Please log in again from tradevex.live.", true);
        }
      }
      const sessionRes = await client.auth.getSession();
      await renderSession(sessionRes && sessionRes.data ? sessionRes.data.session : null);

      const tabLogin = document.getElementById("sb-tab-login");
      const tabSignup = document.getElementById("sb-tab-signup");
      const paneLogin = document.getElementById("sb-pane-login");
      const paneSignup = document.getElementById("sb-pane-signup");
      const loginBtn = document.getElementById("sb-auth-login");
      const signupBtn = document.getElementById("sb-auth-signup");
      const forgotBtn = document.getElementById("sb-auth-forgot");
      const logoutBtn = document.getElementById("sb-auth-logout");
      const emailEl = document.getElementById("sb-auth-email");
      const passEl = document.getElementById("sb-auth-password");
      const signupEmailEl = document.getElementById("sb-signup-email");
      const signupPassEl = document.getElementById("sb-signup-password");
      const signupConfirmEl = document.getElementById("sb-signup-confirm");
      const profileOpenBtn = document.getElementById("sb-profile-open");
      const profilePanel = document.getElementById("sb-profile-panel");
      const profileEmailEl = document.getElementById("sb-profile-email");
      const profileDisplayEl = document.getElementById("sb-profile-display");
      const profileApiKeyEl = document.getElementById("sb-profile-api-key");
      const profileAccessTokenEl = document.getElementById("sb-profile-access-token");
      const profileApiSecretEl = document.getElementById("sb-profile-api-secret");
      const profileStatusEl = document.getElementById("sb-profile-status");
      const profileMsgEl = document.getElementById("sb-profile-msg");
      const profileSaveBtn = document.getElementById("sb-profile-save");
      const profileClearBtn = document.getElementById("sb-profile-clear");
      const profileCloseBtn = document.getElementById("sb-profile-close");

      function setMode(mode) {
        const isSignup = mode === "signup";
        if (tabLogin) tabLogin.classList.toggle("active", !isSignup);
        if (tabSignup) tabSignup.classList.toggle("active", isSignup);
        if (paneLogin) paneLogin.classList.toggle("active", !isSignup);
        if (paneSignup) paneSignup.classList.toggle("active", isSignup);
        if (isSignup && signupEmailEl && emailEl && !signupEmailEl.value) {
          signupEmailEl.value = String(emailEl.value || "");
        }
        if (!isSignup && emailEl && signupEmailEl && !emailEl.value) {
          emailEl.value = String(signupEmailEl.value || "");
        }
        setMessage("", false);
      }

      function setProfileMessage(msg, isError) {
        if (!profileMsgEl) return;
        profileMsgEl.textContent = msg || "";
        profileMsgEl.style.color = isError ? "#fda4af" : "#86efac";
      }

      async function loadProfileData() {
        try {
          const res = await fetch("/api/user/profile", { cache: "no-store" });
          if (!res.ok) {
            setProfileMessage("Profile load failed.", true);
            return;
          }
          const data = await res.json();
          const bin = data && data.binance ? data.binance : {};
          if (profileEmailEl) profileEmailEl.value = String(data.email || "");
          if (profileDisplayEl) profileDisplayEl.value = String(data.display_name || "");
          if (profileApiKeyEl) profileApiKeyEl.value = String(bin.api_key_masked || "");
          if (profileAccessTokenEl) profileAccessTokenEl.value = String(bin.access_token_masked || "");
          if (profileApiSecretEl) profileApiSecretEl.value = "";
          if (profileStatusEl) {
            const ready = !!bin.ready_for_real_trading;
            const secretSet = !!bin.api_secret_set;
            profileStatusEl.textContent = ready
              ? "Binance profile ready for real-account wiring."
              : (secretSet ? "API secret saved. Add API key/access token to complete setup." : "Add Binance keys to enable real-account wiring.");
          }
          setProfileMessage("", false);
        } catch (_err) {
          setProfileMessage("Profile load failed.", true);
        }
      }

      if (tabLogin) tabLogin.addEventListener("click", function () { setMode("login"); });
      if (tabSignup) tabSignup.addEventListener("click", function () { setMode("signup"); });
      if (profileOpenBtn) profileOpenBtn.addEventListener("click", async function () {
        if (profilePanel) profilePanel.style.display = "block";
        await loadProfileData();
      });
      if (profileCloseBtn) profileCloseBtn.addEventListener("click", function () {
        if (profilePanel) profilePanel.style.display = "none";
      });

      const lastEmail = String(localStorage.getItem("sb-auth-last-email") || "").trim();
      if (lastEmail) {
        if (emailEl && !emailEl.value) emailEl.value = lastEmail;
        if (signupEmailEl && !signupEmailEl.value) signupEmailEl.value = lastEmail;
      }
      const activeSession = !!(sessionRes && sessionRes.data && sessionRes.data.session);
      setMode(activeSession || lastEmail ? "login" : "signup");

      if (loginBtn) loginBtn.addEventListener("click", async function () {
        const email = String(emailEl && emailEl.value ? emailEl.value : "").trim();
        const password = String(passEl && passEl.value ? passEl.value : "");
        if (!email || !password) {
          setMessage("Email/password required.", true);
          return;
        }
        setMessage("Signing in...", false);
        const result = await client.auth.signInWithPassword({ email: email, password: password });
        if (result.error) {
          setMessage(result.error.message || "Login failed.", true);
          return;
        }
        try { localStorage.setItem("sb-auth-last-email", email); } catch (_err) {}
        await renderSession(result.data ? result.data.session : null);
      });

      if (signupBtn) signupBtn.addEventListener("click", async function () {
        const email = String(signupEmailEl && signupEmailEl.value ? signupEmailEl.value : "").trim();
        const password = String(signupPassEl && signupPassEl.value ? signupPassEl.value : "");
        const confirm = String(signupConfirmEl && signupConfirmEl.value ? signupConfirmEl.value : "");
        if (!email || !password) {
          setMessage("Email/password required.", true);
          return;
        }
        if (password.length < 8) {
          setMessage("Password must be at least 8 characters.", true);
          return;
        }
        if (password !== confirm) {
          setMessage("Password and confirm password do not match.", true);
          return;
        }
        setMessage("Creating account...", false);
        const result = await client.auth.signUp({ email: email, password: password });
        if (result.error) {
          setMessage(result.error.message || "Signup failed.", true);
          return;
        }
        try { localStorage.setItem("sb-auth-last-email", email); } catch (_err) {}
        if (result.data && result.data.session) {
          await renderSession(result.data.session);
          return;
        }
        if (emailEl) emailEl.value = email;
        if (passEl) passEl.value = password;
        setMode("login");
        setMessage("Signup created. Verify email, then login.", false);
      });

      if (profileSaveBtn) profileSaveBtn.addEventListener("click", async function () {
        const payload = {};
        const displayName = String(profileDisplayEl && profileDisplayEl.value ? profileDisplayEl.value : "").trim();
        const apiKey = String(profileApiKeyEl && profileApiKeyEl.value ? profileApiKeyEl.value : "").trim();
        const accessToken = String(profileAccessTokenEl && profileAccessTokenEl.value ? profileAccessTokenEl.value : "").trim();
        const apiSecret = String(profileApiSecretEl && profileApiSecretEl.value ? profileApiSecretEl.value : "").trim();
        if (displayName) payload.display_name = displayName;
        if (apiKey && apiKey.indexOf("*") === -1) payload.binance_api_key = apiKey;
        if (accessToken && accessToken.indexOf("*") === -1) payload.binance_access_token = accessToken;
        if (apiSecret) payload.binance_api_secret = apiSecret;
        setProfileMessage("Saving profile...", false);
        try {
          const res = await fetch("/api/user/profile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!res.ok) {
            const err = await res.text();
            setProfileMessage(err || "Profile save failed.", true);
            return;
          }
          setProfileMessage("Profile saved.", false);
          await loadProfileData();
        } catch (_err) {
          setProfileMessage("Profile save failed.", true);
        }
      });

      if (profileClearBtn) profileClearBtn.addEventListener("click", async function () {
        const ok = window.confirm("Clear saved Binance API key, secret and access token?");
        if (!ok) return;
        setProfileMessage("Clearing keys...", false);
        try {
          const res = await fetch("/api/user/profile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ clear_binance_keys: true }),
          });
          if (!res.ok) {
            setProfileMessage("Could not clear keys.", true);
            return;
          }
          await loadProfileData();
          setProfileMessage("Saved keys cleared.", false);
        } catch (_err) {
          setProfileMessage("Could not clear keys.", true);
        }
      });

      if (forgotBtn) forgotBtn.addEventListener("click", async function () {
        const email = String(emailEl && emailEl.value ? emailEl.value : "").trim();
        if (!email) {
          setMessage("Enter email first, then click forgot password.", true);
          return;
        }
        setMessage("Sending reset email...", false);
        const out = await client.auth.resetPasswordForEmail(email, {
          redirectTo: window.location.origin + "/",
        });
        if (out && out.error) {
          setMessage(out.error.message || "Reset email failed.", true);
          return;
        }
        setMessage("Password reset email sent. Check inbox/spam.", false);
      });

      if (logoutBtn) logoutBtn.addEventListener("click", async function () {
        await client.auth.signOut();
        if (profilePanel) profilePanel.style.display = "none";
        await renderSession(null);
      });

      client.auth.onAuthStateChange(function (_event, session) {
        void renderSession(session);
      });
    } catch (err) {
      setMessage("Supabase auth init failed.", true);
      console.error(err);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAuthUi);
  } else {
    initAuthUi();
  }
})();
</script>
"""
    return bootstrap.replace("__AUTH_CFG__", cfg_json)


def _inject_html_auth_bootstrap(html: str) -> str:
    if not (_auth_enabled() and _auth_provider() == "supabase"):
        return html
    snippet = _supabase_auth_bootstrap()
    if "</head>" in html:
        return html.replace("</head>", f"{snippet}\n</head>", 1)
    return f"{snippet}\n{html}"


def _read_html(name: str) -> str:
    p = STATIC_DIR / name
    raw = p.read_text(encoding="utf-8") if p.exists() else f"<h1>{name} not found</h1>"
    return _inject_html_auth_bootstrap(raw)


_DASHBOARD_HTML_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _html_response(html: str) -> HTMLResponse:
    return HTMLResponse(content=html, headers=dict(_DASHBOARD_HTML_HEADERS))


def _render_asset_terminal_html(symbol: str = "ETHUSDT") -> str:
    sym = _normalize_crypto_symbol(symbol)
    if sym not in {"ETHUSDT", "SOLUSDT"}:
        sym = "ETHUSDT"
    short = _asset_short(sym)
    html = _read_html("index.html")
    html = html.replace("BTCUSDT", sym)
    html = html.replace("btcusdt", sym.lower())
    html = html.replace("<title>BTC Quant Terminal</title>", f"<title>{short} Quant Terminal</title>", 1)
    html = html.replace("BTC Quant Terminal", f"{short} Quant Terminal", 1)

    bootstrap = f"""
<script>
(function(){{
  const ALT_SYMBOL = {json.dumps(sym)};
  const ALT_SHORT = {json.dumps(short)};
  window.__ALT_SYMBOL__ = ALT_SYMBOL;
  window.__ALT_SHORT__ = ALT_SHORT;

  function rewriteUrl(inputUrl){{
    let url = String(inputUrl || "");
    if(!url) return url;
    try{{
      if(url.startsWith("/api/btc/")){{
        const u = new URL(url, window.location.origin);
        if(!u.searchParams.has("symbol")) u.searchParams.set("symbol", ALT_SYMBOL);
        url = u.pathname + u.search + u.hash;
      }}
      url = url.replace(/symbol=BTC\\b/g, "symbol=" + ALT_SHORT);
      url = url.replace(/symbol=BTCUSDT/g, "symbol=" + ALT_SYMBOL);
      url = url.replace(/btcusdt/g, ALT_SYMBOL.toLowerCase());
      url = url.replace(/BTCUSDT/g, ALT_SYMBOL);
    }}catch(_e){{}}
    return url;
  }}

  const nativeFetch = window.fetch.bind(window);
  window.fetch = function(input, init){{
    try{{
      if(typeof input === "string"){{
        return nativeFetch(rewriteUrl(input), init);
      }}
      if(input && input.url){{
        const next = rewriteUrl(input.url);
        if(next !== input.url){{
          return nativeFetch(new Request(next, input), init);
        }}
      }}
    }}catch(_e){{}}
    return nativeFetch(input, init);
  }};

  const NativeWebSocket = window.WebSocket;
  function WrappedWebSocket(url, protocols){{
    return protocols ? new NativeWebSocket(rewriteUrl(url), protocols) : new NativeWebSocket(rewriteUrl(url));
  }}
  WrappedWebSocket.prototype = NativeWebSocket.prototype;
  Object.setPrototypeOf(WrappedWebSocket, NativeWebSocket);
  window.WebSocket = WrappedWebSocket;

  window.addEventListener("DOMContentLoaded", function(){{
    try {{
      const assetNameByShort = {{ ETH: "Ethereum", SOL: "Solana" }};
      const assetName = assetNameByShort[ALT_SHORT] || ALT_SHORT;
      const title = document.querySelector(".title");
      if(title){{
        title.innerHTML = '<span>' + ALT_SHORT + '</span> Quant Terminal <span style=\"color:#00c853\">●</span>';
      }}
      const priceLabel = Array.from(document.querySelectorAll(".grid .card .k"))
        .find((el) => String(el.textContent || "").trim() === "Bitcoin Price (USDT)");
      if(priceLabel){{
        priceLabel.textContent = assetName + " Price (USDT)";
      }}
      const chartHead = document.querySelector(".head-title");
      if(chartHead){{
        chartHead.textContent = ALT_SYMBOL + " Trading Chart";
      }}
      const oiUnit = document.getElementById("open-interest-unit");
      if(oiUnit){{
        oiUnit.textContent = ALT_SHORT;
      }}
      const newsTitle = Array.from(document.querySelectorAll(".panel-title"))
        .find((el) => String(el.textContent || "").trim() === "Live BTC News");
      if(newsTitle){{
        newsTitle.textContent = "Live " + ALT_SHORT + " News";
      }}
      const modalTitle = document.getElementById("pt-modal-title");
      if(modalTitle){{
        modalTitle.textContent = "▲ LONG " + ALT_SYMBOL;
      }}
    }} catch(_e) {{}}
  }});
}})();
</script>
"""
    return html.replace("</head>", bootstrap + "\n</head>", 1)


@app.get("/", response_class=HTMLResponse)
def index():
    return _html_response(_read_html("index.html"))


@app.get("/terminal", response_class=HTMLResponse)
def terminal_page():
    return _html_response(_read_html("terminal.html"))


@app.get("/crypto-terminal", response_class=HTMLResponse)
def crypto_terminal_page():
    return _html_response(_read_html("crypto_terminal.html"))


@app.get("/asset-terminal", response_class=HTMLResponse)
def asset_terminal_page(symbol: str = "ETHUSDT"):
    return _html_response(_render_asset_terminal_html(symbol))


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio_page():
    return _html_response(_read_html("portfolio.html"))


@app.get("/history", response_class=HTMLResponse)
def history_page():
    return _html_response(_read_html("history.html"))


@app.get("/factors", response_class=HTMLResponse)
def factors_page():
    return _html_response(_read_html("factors.html"))


@app.get("/regime", response_class=HTMLResponse)
def regime_page():
    return _html_response(_read_html("regime.html"))


@app.get("/focus", response_class=HTMLResponse)
def focus_page():
    return _html_response(_read_html("focus.html"))


@app.get("/stock-terminal", response_class=HTMLResponse)
def stock_terminal_page():
    return _html_response(_read_html("stock_terminal.html"))

