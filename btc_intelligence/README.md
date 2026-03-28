# Advanced Bitcoin Trade Intelligence System v2.0

Production-grade BTC/USDT perpetual intelligence stack with:
- Async multi-source ingestion: Binance, Bybit, OKX, Deribit, Coinglass, Glassnode, Whale, CryptoPanic, macro feeds
- 51-feature engineering pipeline with SMC, order flow microstructure, derivatives/options, on-chain, macro
- Regime engine + strategy engine (S1-S5) with strict HOLD-first filtering
- ML confidence ensemble + probability stacking
- Monitoring with auto-pause triggers and retrain endpoint
- FastAPI backend + WebSocket + Next.js dashboard

## Folder Structure

```text
btc_intelligence/
├── main.py
├── config.py
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── service.py
├── ingestion/
│   ├── binance_ws.py
│   ├── binance_rest.py
│   ├── multi_exchange_ws.py
│   ├── coinglass.py
│   ├── glassnode.py
│   ├── deribit.py
│   ├── whale_tracker.py
│   ├── macro_data.py
│   ├── cryptopanic.py
│   └── data_buffer.py
├── features/
│   ├── price_action.py
│   ├── smc.py
│   ├── order_flow.py
│   ├── volatility.py
│   ├── derivatives.py
│   ├── onchain.py
│   ├── macro.py
│   └── feature_vector.py
├── regime/
│   └── classifier.py
├── signals/
│   ├── engine.py
│   ├── alpha.py
│   ├── risk.py
│   ├── execution.py
│   ├── no_trade_filter.py
│   └── probability_stacker.py
├── monitoring/
│   ├── performance_tracker.py
│   ├── auto_pause.py
│   ├── auto_correct.py
│   └── weekly_report.py
├── models/
│   ├── train_lgbm.py
│   ├── train_xgb.py
│   ├── train_lstm.py
│   ├── train_attention_lstm.py
│   ├── retrainer.py
│   ├── ensemble.py
│   ├── inference.py
│   ├── label_generator.py
│   └── artifacts/
├── api/
│   ├── routes.py
│   └── websocket.py
├── backtesting/
│   ├── engine.py
│   ├── metrics.py
│   └── paper_trade.py
├── utils/
│   ├── logger.py
│   ├── validator.py
│   └── notifier.py
├── tests/
│   └── test_features.py
└── frontend/
    ├── app/
    │   ├── page.tsx
    │   ├── analysis/page.tsx
    │   ├── history/page.tsx
    │   ├── regime/page.tsx
    │   ├── monitoring/page.tsx
    │   └── options/page.tsx
    └── components/
```

## Local Setup (8 Steps)

1. Backend env + deps:
```powershell
cd c:\Users\ASAD AHMAD\OneDrive\Desktop\Trading
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r btc_intelligence\requirements.txt
```

2. Configure env:
```powershell
Copy-Item btc_intelligence\.env.example btc_intelligence\.env
```

3. Start backend:
```powershell
uvicorn btc_intelligence.main:app --host 0.0.0.0 --port 9000 --reload
```

4. Install frontend deps:
```powershell
cd btc_intelligence\frontend
npm install
```

5. Set frontend API URL:
```powershell
Set-Content -Path .env.local -Value "NEXT_PUBLIC_API_BASE=http://localhost:9000"
```

6. Start frontend:
```powershell
npm run dev
```

7. Open apps:
- Frontend: `http://localhost:3000`
- Backend docs: `http://localhost:9000/docs`

8. Run tests:
```powershell
python -m unittest btc_intelligence.tests.test_features
```

## API Endpoints

- `GET /signal`
- `GET /signal/history`
- `GET /regime`
- `GET /features`
- `GET /health`
- `GET /models/performance`
- `GET /monitoring/stats`
- `GET /monitoring/edges`
- `POST /models/retrain`
- `POST /paper_trade/open`
- `POST /paper_trade/close`
- `GET /market/klines?timeframe=15m&limit=500`
- `GET /market/history?timeframe=1d&limit=2000`
- `WS /ws/live`

## Model Training

Generate labels:
```powershell
python -m btc_intelligence.models.label_generator --input data.csv --output labeled.csv
```

Train regime models manually:
```powershell
python -m btc_intelligence.models.train_lgbm --data labeled.csv --out btc_intelligence/models/artifacts/bullish_trend --regime bullish_trend
python -m btc_intelligence.models.train_xgb --data labeled.csv --out btc_intelligence/models/artifacts/bullish_trend --regime bullish_trend
python -m btc_intelligence.models.train_attention_lstm --data labeled.csv --out btc_intelligence/models/artifacts/bullish_trend --regime bullish_trend
```

Trigger retrain by API:
```bash
curl -X POST http://localhost:9000/models/retrain -H "Content-Type: application/json" -d '{"data_path":"labeled.csv"}'
```

## Backtesting

```powershell
python -m btc_intelligence.backtesting.engine --data labeled.csv --model btc_intelligence/models/artifacts/lightgbm.pkl --out btc_intelligence/backtesting/report.json
```

Report includes Sharpe, max drawdown, win rate, average MAE, average MFE.

## Docker

```powershell
cd btc_intelligence
Copy-Item .env.example .env
docker compose up --build
```

Services:
- Backend: `http://localhost:9000`
- Frontend: `http://localhost:3000`
- Redis: `localhost:6379`
