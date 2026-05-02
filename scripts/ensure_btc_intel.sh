#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/asad/TradeVex-Server}"
PY_BIN="${PY_BIN:-$ROOT_DIR/venv/bin/python}"
LOG_DIR="$ROOT_DIR/logs"
OUT_LOG="$LOG_DIR/tradevex-btc.out.log"
ERR_LOG="$LOG_DIR/tradevex-btc.err.log"
PID_FILE="$LOG_DIR/tradevex-btc.pid"

if curl -fsS --max-time 2 "http://127.0.0.1:9000/health" >/dev/null 2>&1; then
  exit 0
fi

if command -v pgrep >/dev/null 2>&1; then
  while IFS= read -r pid; do
    if [[ -n "${pid:-}" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done < <(pgrep -f "uvicorn btc_intelligence.main:app --host 127.0.0.1 --port 9000" || true)
fi

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

nohup "$PY_BIN" -m uvicorn btc_intelligence.main:app --host 127.0.0.1 --port 9000 >>"$OUT_LOG" 2>>"$ERR_LOG" < /dev/null &
echo "$!" > "$PID_FILE"

sleep 2
if ! curl -fsS --max-time 4 "http://127.0.0.1:9000/health" >/dev/null 2>&1; then
  echo "[ensure_btc_intel] failed to start btc_intelligence on :9000" >&2
  exit 1
fi

