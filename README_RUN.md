# QuantTrader Runbook

This is the quick operational guide for running QuantTrader locally.

## 1) Environment setup

```powershell
cd c:\Users\ASAD AHMAD\OneDrive\Desktop\Trading
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Recommended startup (port 8001)

```powershell
python main.py --mode live --config config.runtime.8001.yaml
```

Open:
- `http://127.0.0.1:8001/` (BTC terminal)
- `http://127.0.0.1:8001/terminal` (multi-asset terminal)

## 3) Other modes

```powershell
# API/dashboard only (no scheduler)
python main.py --mode dashboard --config config.runtime.8001.yaml

# Signals only in terminal
python main.py --mode signals --config config.runtime.8001.yaml

# WFO backtest
python main.py --mode backtest --engine wfo --ticker RELIANCE.NS --train_days 252 --config config.runtime.8001.yaml

# Full event-driven backtest
python main.py --mode backtest --engine full --ticker AAPL --config config.runtime.8001.yaml

# Combined project validation (default quick: WFO + full backtest + accuracy report)
python main.py --mode validate --ticker AAPL --train_days 252 --config config.runtime.8001.yaml --start 2023-01-01 --end 2025-12-31 --output data/validation_aapl.json --validation_profile quick

# Full validation (slower, adds CPCV robustness test)
python main.py --mode validate --ticker AAPL --train_days 252 --config config.runtime.8001.yaml --start 2023-01-01 --end 2025-12-31 --output data/validation_aapl_full.json --validation_profile full

# Capacity
python main.py --mode capacity --ticker RELIANCE.NS --config config.runtime.8001.yaml
python main.py --mode capacity --ticker ALL --config config.runtime.8001.yaml
```

## 4) Core checks after startup

Health:
- `GET /api/health`
- `GET /api/system-health`
- `GET /api/live-validation`

Trading data:
- `GET /api/market-overview`
- `GET /api/stock-signal?ticker=RELIANCE.NS`
- `GET /api/options/chain?symbol=NIFTY`
- `GET /api/paper/portfolio`

## 5) Paper trading quick checks

Execute sample:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8001/api/paper/execute `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"ticker":"RELIANCE.NS","signal":"BUY","entry_price":2500,"stop_loss":2450,"take_profit":2600,"confidence":70,"asset_class":"indian_stock","strength":"MODERATE"}'
```

Then verify:
- `GET /api/paper/portfolio`
- `GET /api/paper/trades?limit=20`

## 6) Common issues

- Port busy: change `dashboard.port` in config or stop old process.
- Empty UI: verify `data/signals.db` and `/api/health`.
- Backtest feature error: use longer range / enough rows.
- Option chain unavailable: NSE can block; backend returns synthetic fallback.

## 7) Auth (optional)

Two auth modes are supported via `dashboard.auth.provider`:

- `local`: set `enabled: true`, then use `POST /auth/token` and send `Authorization: Bearer <token>`.
- `supabase`: set `enabled: true`, `provider: supabase`, and configure `SUPABASE_URL` + `SUPABASE_ANON_KEY` (or in config).  
  Login/Signup UI is auto-injected on dashboard pages and API/`/ws` are protected by Supabase access token.
  To restrict access to specific users only, set `dashboard.auth.allowed_emails` and/or `DASHBOARD_ALLOWED_EMAILS` (comma-separated).
