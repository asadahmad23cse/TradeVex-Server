@echo off
setlocal EnableDelayedExpansion

echo Starting BTC Quant Terminal...
echo Redis: localhost:6379 (optional)
echo BTC intelligence: http://127.0.0.1:9000  (new window)
echo Dashboard:       http://127.0.0.1:8000  (new window)
echo.

cd /d "%~dp0"

where redis-server >nul 2>&1
if %ERRORLEVEL%==0 (
  start "Redis" /MIN redis-server
  timeout /t 2 /nobreak >nul
) else (
  echo [warn] redis-server not in PATH — start Redis manually if needed.
)

start "BTC Intelligence :9000" cmd /k "cd /d ""%~dp0"" && python -m uvicorn btc_intelligence.main:app --host 0.0.0.0 --port 9000 --reload"

set ATTEMPTS=0
:waitbtc
curl.exe -sf http://127.0.0.1:9000/health >nul 2>&1
if not errorlevel 1 (
  echo BTC intelligence health OK.
  goto btcready
)
set /a ATTEMPTS+=1
if !ATTEMPTS! GEQ 60 (
  echo WARNING: btc_intelligence did not become healthy — check BTC Intelligence window.
  goto btcready
)
timeout /t 1 /nobreak >nul
goto waitbtc

:btcready
start "Dashboard :8000" cmd /k "cd /d ""%~dp0"" && python -m uvicorn src.dashboard.api:app --host 127.0.0.1 --port 8000 --reload"
echo.
echo Both servers launched — see "BTC Intelligence :9000" and "Dashboard :8000" windows.
echo Open: http://127.0.0.1:8000
echo.
