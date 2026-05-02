#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/asad/TradeVex-Server}"

latest_pid_matching() {
  local pattern="$1"
  ps -eo pid=,cmd= \
    | grep "$pattern" \
    | grep -v grep \
    | awk '{print $1}' \
    | tail -n 1
}

kill_all_but_latest() {
  local pattern="$1"
  local latest
  latest="$(latest_pid_matching "$pattern" || true)"
  local all
  all="$(ps -eo pid=,cmd= | grep "$pattern" | grep -v grep | awk '{print $1}' || true)"
  if [[ -z "${all:-}" ]]; then
    return
  fi
  for p in $all; do
    if [[ -n "${latest:-}" && "$p" == "$latest" ]]; then
      continue
    fi
    kill "$p" >/dev/null 2>&1 || true
  done
}

kill_all_but_latest "main.py --mode live --config config.yaml"
kill_all_but_latest "uvicorn btc_intelligence.main:app --host 127.0.0.1 --port 9000"

if ! ps -eo pid,cmd | grep 'main.py --mode live --config config.yaml' | grep -v grep >/dev/null; then
  # Give systemd-managed service a chance to recover first.
  sleep 8
fi

if ! ps -eo pid,cmd | grep 'main.py --mode live --config config.yaml' | grep -v grep >/dev/null; then
  cd "$ROOT_DIR"
  nohup ./venv/bin/python main.py --mode live --config config.yaml > logs/live.out.log 2>&1 < /dev/null &
fi

if ! ps -eo pid,cmd | grep 'uvicorn btc_intelligence.main:app --host 127.0.0.1 --port 9000' | grep -v grep >/dev/null; then
  "$ROOT_DIR/scripts/ensure_btc_intel.sh" || true
fi

sleep 6

echo "MAIN PROC:"
ps -eo pid,user,cmd | grep 'main.py --mode live --config config.yaml' | grep -v grep || true
echo "BTC PROC:"
ps -eo pid,user,cmd | grep 'uvicorn btc_intelligence.main:app --host 127.0.0.1 --port 9000' | grep -v grep || true

echo "API /health:"
curl -sS --max-time 12 http://127.0.0.1:8000/api/health | head -c 280 || true
echo
echo "BTC /health:"
curl -sS --max-time 12 http://127.0.0.1:9000/health | head -c 280 || true
echo
