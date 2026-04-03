@echo off
echo Starting TradeVex...
echo.

echo [1/3] Starting Redis...
start "Redis" redis-server
timeout /t 2

echo [2/3] Starting BTC Intelligence (port 9000)...
start "BTC Intel" cmd /k "python -m uvicorn btc_intelligence.main:app --host 127.0.0.1 --port 9000"
timeout /t 3

echo [3/3] Starting Main Dashboard (port 8000)...
start "TradeVex" cmd /k "python main.py --mode live"

echo.
echo TradeVex started! Open http://127.0.0.1:8000
pause
