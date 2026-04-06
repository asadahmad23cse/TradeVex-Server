@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo Starting TradeVex...
echo.

echo [1/3] Starting Redis...
start "Redis" redis-server
timeout /t 2 >nul

if exist "%ROOT%.venv\Scripts\python.exe" (
  set "PYEXE=%ROOT%.venv\Scripts\python.exe"
) else (
  set "PYEXE=python"
)

echo [2/3] Starting BTC Intelligence (port 9000)...
start "BTC Intel" cmd /k "cd /d \"%ROOT%\" && \"%PYEXE%\" -m uvicorn btc_intelligence.main:app --host 127.0.0.1 --port 9000"

echo Waiting for BTC Intelligence on port 9000...
"%PYEXE%" "%ROOT%scripts\wait_for_port.py" 127.0.0.1 9000 45
if errorlevel 1 (
  echo WARNING: Port 9000 did not open in time. Intelligence pill may show degraded until service is up.
) else (
  echo Port 9000 is ready.
)

echo [2b] BTC Intel watchdog (restarts service if port 9000 drops^)...
start "BTC Intel Watchdog" cmd /k "cd /d \"%ROOT%\" && \"%PYEXE%\" scripts\btc_intel_watchdog.py"

echo [3/3] Starting Main Dashboard (port 8000)...
start "TradeVex" cmd /k "cd /d \"%ROOT%\" && \"%PYEXE%\" main.py --mode live"

echo.
echo TradeVex started! Open http://127.0.0.1:8000
pause
