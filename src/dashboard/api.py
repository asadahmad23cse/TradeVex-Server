"""
Layer 10 â€” FastAPI Dashboard + WebSocket.

Binds to 127.0.0.1 (localhost only).
WebSocket at /ws broadcasts new signals in real-time.
5 pages: Live Signals, Portfolio, History, Factor Analysis, Regime Monitor.
"""

from datetime import datetime
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # type: ignore[import]
from fastapi.responses import HTMLResponse, Response  # type: ignore[import]
from fastapi.staticfiles import StaticFiles  # type: ignore[import]
from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore[import]
from src.data.news_feed import get_btc_news
from src.data.signal_history import get_history as get_signal_history, get_stats as get_signal_stats
from src.dashboard.btc_service import BitcoinMarketService
from src.dashboard.focus_engine import FocusQuantEngine
from src.options import ExpiryTracker, OptionsEngine
from src.compliance import SEBIComplianceEngine

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

def init_dashboard(store, portfolio, config: dict | None = None) -> None:  # type: ignore[no-untyped-def]
    """Inject SignalStore and PortfolioTracker into the dashboard."""
    global _store, _portfolio, _dashboard_cfg, _focus_engine, _btc_service, _options_engine
    _store = store
    _portfolio = portfolio
    _dashboard_cfg = config or {}
    _focus_engine = FocusQuantEngine(_dashboard_cfg)
    _btc_service = BitcoinMarketService(_dashboard_cfg)
    if _options_engine is None:
        _options_engine = OptionsEngine()


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
        if request.url.path.startswith("/static") or request.url.path in open_paths:
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
    "/api/market-overview": 30,
}


def _cache_ttl_for_path(path: str) -> int:
    for prefix, ttl in _response_cache_ttl.items():
        if path.startswith(prefix):
            return ttl
    return 0


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


# ------------------------------------------------------------------
# REST endpoints (data for dashboard pages)
# ------------------------------------------------------------------

@app.get("/api/watchlist")
def get_watchlist():
    """Return full watchlist with latest prices fetched from yfinance."""
    import yfinance as yf  # type: ignore[import]

    cfg = _dashboard_cfg or {}
    stocks = []
    for asset_class in ("indian_stocks", "us_stocks", "forex"):
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
        # US stocks: Alpha Vantage first, then yfinance fallback.
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
                    return {"ticker": ticker, "data": av_data}
            except Exception as e:
                logger.warning("Alpha Vantage chart fallback for %s: %s", ticker, e)

        import yfinance as yf  # type: ignore[import]
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            return {"ticker": ticker, "data": [], "error": "No data"}

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


@app.get("/api/btc/signal/history")
def signal_history(limit: int = 50):
    return {"signals": get_signal_history(limit), "stats": get_signal_stats()}


@app.get("/api/btc/signal/stats")
def signal_stats():
    return get_signal_stats()


@app.get("/api/btc/markers")
def get_btc_markers(interval: str = "1d", limit: int = 1000):
    """Historical LONG/SHORT markers for BTC chart overlay."""
    if _btc_service is None:
        raise HTTPException(503, "BTC service not initialised")
    return _btc_service.get_signal_markers(interval=interval, limit=limit)


@app.get("/api/btc/news")
def btc_news(limit: int = 8):
    return {"news": get_btc_news(limit)}


@app.get("/api/news")
def get_news(symbol: str = "BTC", asset_class: str = "crypto", limit: int = 8):
    from src.data.news_feed import get_news_for_asset
    cls = (asset_class or "crypto").lower().strip()
    if cls in {"indian_stock", "us_stock", "stocks"}:
        cls = "stock"
    return {"news": get_news_for_asset(symbol, cls, limit)}


@app.get("/api/options/chain")
def get_options_chain(symbol: str = "NIFTY", expiry: str = "nearest"):
    try:
        eng = _get_options_engine()
        return eng.get_option_chain(symbol=symbol, expiry=expiry)
    except Exception as e:
        logger.warning("Options chain endpoint failed for %s: %s", symbol, e)
        return {
            "symbol": symbol,
            "calls": [],
            "puts": [],
            "spot_price": 0.0,
            "expiry_dates": [],
            "pcr": 0.0,
            "max_pain": 0.0,
            "error": str(e),
        }


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


@app.get("/api/stock-signal")
def get_stock_signal(ticker: str = "RELIANCE.NS"):
    """Return AI signal for a specific stock â€” live alpha computation."""
    # First check stored signals
    if _store is not None:
        signals = _store.get_recent_signals(limit=200)
        for s in signals:
            if s.get("asset") == ticker or s.get("asset", "").replace(".NS", "") == ticker.replace(".NS", ""):
                return s

    # US stocks: Alpha Vantage signal first, yfinance fallback next.
    if _is_us_stock_ticker(ticker):
        try:
            from src.data.alpha_vantage import get_us_stock_signal  # type: ignore[import]

            av_signal = get_us_stock_signal(ticker)
            if av_signal:
                return av_signal
        except Exception as e:
            logger.warning("Alpha Vantage signal fallback for %s: %s", ticker, e)

    # Live computation fallback
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
            return {"ticker": ticker, "signal": "HOLD", "confidence": 50, "alpha_score": 0.0, "message": "Insufficient data"}

        # Handle MultiIndex
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)

        eng = FeatureEngineer()
        tf = "daily" if len(df) > 100 else "intraday"
        feat_df = eng.compute_all_features(df, timeframe=tf, ticker=ticker)
        if feat_df.empty:
            return {"ticker": ticker, "signal": "HOLD", "confidence": 50, "alpha_score": 0.0, "message": "Feature computation failed"}

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

        return {
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
        }
    except Exception as e:
        return {"ticker": ticker, "signal": "HOLD", "confidence": 50, "alpha_score": 0.0, "message": str(e)}


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

@app.get("/api/signals")
def get_signals(limit: int = 50):
    if _store is None:
        return []
    return _store.get_recent_signals(limit=limit)


@app.get("/api/portfolio")
def get_portfolio():
    if _portfolio is None:
        return {}
    return {
        "metrics": _portfolio.get_metrics(),
        "positions": _portfolio.get_open_positions_list(),
    }


@app.get("/api/history")
def get_history(limit: int = 100):
    if _store is None:
        return []
    return _store.get_recent_signals(limit=limit)


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

