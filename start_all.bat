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
  if errorlevel 1 (
    echo ERROR: Failed to start redis-server.
    exit /b 1
  )
  timeout /t 2 /nobreak >nul
) else (
  echo [warn] redis-server not in PATH — start Redis manually if btc_intelligence needs it.
)

rem Separate window so 9000 always runs (no silent /B). Matches: uvicorn ... --host 0.0.0.0 --port 9000
start "BTC Intelligence :9000" cmd /k "cd /d ""%~dp0"" && python -m uvicorn btc_intelligence.main:app --host 0.0.0.0 --port 9000"

set ATTEMPTS=0
:waitbtc
curl.exe -sf http://127.0.0.1:9000/health >nul 2>&1
if not errorlevel 1 (
  echo BTC intelligence health OK.
  goto btcready
)
set /a ATTEMPTS+=1
if !ATTEMPTS! GEQ 60 (
  echo ERROR: btc_intelligence did not become healthy in time ^(http://127.0.0.1:9000/health^).
  echo        Check the "BTC Intelligence :9000" window for errors.
  exit /b 1
)
timeout /t 1 /nobreak >nul
goto waitbtc

:btcready
start "Dashboard :8000" cmd /k "cd /d ""%~dp0"" && python -m uvicorn src.dashboard.api:app --host 127.0.0.1 --port 8000 --reload"
echo.
echo Both servers launched — see windows "BTC Intelligence :9000" and "Dashboard :8000".
echo You can close this launcher; servers keep running until those windows are closed.
echo.
