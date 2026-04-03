@echo off
setlocal

echo Starting BTC Quant Terminal...
echo Backend: http://127.0.0.1:9000
echo Dashboard: http://127.0.0.1:8000

pushd "%~dp0btc_intelligence"
start /B python -m uvicorn main:app --host 127.0.0.1 --port 9000 --reload
popd

timeout /t 3 /nobreak >nul

cd /d "%~dp0src\dashboard"
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
