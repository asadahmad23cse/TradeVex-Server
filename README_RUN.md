# Run Guide

This is the practical runbook for the current project structure and UI.

## 1. Environment Setup

From the project root:

```powershell
cd c:\Users\Asad\Desktop\Trading
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Optional Credentials

Create `.env` if you want live integrations:

```env
ALPHA_VANTAGE_API_KEY=
ZERODHA_API_KEY=
ZERODHA_ACCESS_TOKEN=
```

Most runtime settings are in `config.yaml`, not `.env`.

Useful config sections before first run:

- `dashboard.host`
- `dashboard.port`
- `dashboard.auth`
- `execution.broker`
- `database.url`
- `notifications`
- `secondary_validation`

## 3. Main Commands

### Live system

Starts APScheduler, signal pipeline, persistence, and dashboard:

```powershell
python main.py --mode live
```

Open:

```text
http://127.0.0.1:8000
```

The frontend is now Bitcoin-only and uses:
- Binance WebSocket live stream: `wss://stream.binance.com:9443/ws/btcusdt@trade`
- Binance REST fallback every 15 seconds: `https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT`
- All-time BTC history endpoint (Binance paginated klines): `/api/btc/history?interval=1d`
- Real-time BTC quant signal endpoint: `/api/btc/signal?interval=5m`

If you still see `404` for `/api/btc/signal`, restart the server process; that means an older app instance is still running.

### Dashboard only

Reads from the configured DB and serves the UI without starting the scheduler:

```powershell
python main.py --mode dashboard
```

### Terminal signals only

No web UI:

```powershell
python main.py --mode signals
```

### Walk-forward validation

```powershell
python main.py --mode backtest --engine wfo --ticker AAPL --train_days 252
```

### Full event-driven backtest

```powershell
python main.py --mode backtest --engine full --ticker AAPL
```

### Capacity analysis

Single ticker:

```powershell
python main.py --mode capacity --ticker RELIANCE.NS
```

Watchlist approximation:

```powershell
python main.py --mode capacity --ticker ALL
```

## 4. Dashboard Routes

Main pages:

- `/`
- `/portfolio`
- `/history`
- `/factors`
- `/regime`

Operational APIs:

- `/api/orders`
- `/api/data-quality`
- `/api/reconciliation`
- `/api/model-validation`
- `/api/system-health`
- `/api/health`
- `/api/live-validation`
- `/api/latency`
- `/api/focus-assets`
- `/api/focus-chart`
- `/api/focus-trade`
- `/api/focus-trades`

## 5. If Auth Is Enabled

Set this in `config.yaml`:

```yaml
dashboard:
  auth:
    enabled: true
    username: admin
    password: change-me
    jwt_secret: change-this-secret
```

Get a token:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/auth/token `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"username":"admin","password":"change-me"}'
```

Use the returned bearer token for API calls and pass `?token=...` for WebSocket clients if you build a remote client around it.

The bundled dashboard UI will also prompt for username/password and cache the token locally when auth is enabled.

## 6. Common Local Workflow

1. Start in dashboard mode and confirm the UI loads.
2. Run tests.
3. Switch to `--mode live` with `execution.broker: paper`.
4. Review:
   - `/api/health`
   - `/api/data-quality`
   - `/api/reconciliation`
   - `/api/model-validation`
5. Only then move broker settings away from paper mode.

## 7. Troubleshooting

If the UI loads but looks empty:

- confirm `data/signals.db` exists or `database.url` points to the right DB
- confirm you started `--mode live` if you expect fresh signals
- check `/api/health` and `/api/system-health`

If live mode starts but no orders appear:

- check `execution.broker`
- check `/api/data-quality` for suppressed assets
- check `/api/reconciliation` and `/api/model-validation`
- check the scheduler watchdog events in `/api/system-health`

If TensorFlow or tree-model packages fail to install:

- the system still runs, but ML coverage degrades
- LSTM and some ensemble functionality will be reduced or disabled
