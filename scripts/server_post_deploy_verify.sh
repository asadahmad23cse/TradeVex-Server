#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/asad/TradeVex-Server}"
PY_BIN="${PY_BIN:-$ROOT_DIR/venv/bin/python}"

main_pids() {
  ps -eo pid=,cmd= | awk '/main.py --mode live --config config.yaml/ && $0 !~ /awk/ {print $1}'
}

wait_health() {
  local url="$1"
  local attempts="${2:-12}"
  local sleep_sec="${3:-2}"
  local i
  for ((i=1; i<=attempts; i++)); do
    if curl -sS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_sec"
  done
  return 1
}

echo "[1/4] Python syntax check"
"$PY_BIN" -m py_compile \
  "$ROOT_DIR/src/dashboard/btc_service.py" \
  "$ROOT_DIR/src/dashboard/api.py" \
  "$ROOT_DIR/src/paper_trading/paper_engine.py"

echo "[2/4] Restart main live process"
PIDS="$(main_pids || true)"
if [[ -n "${PIDS:-}" ]]; then
  # shellcheck disable=SC2086
  kill $PIDS || true
fi
sleep 4

if [[ -z "$(main_pids || true)" ]]; then
  # Give systemd-managed unit time to auto-restart first.
  sleep 8
fi

if [[ -z "$(main_pids || true)" ]]; then
  # Fallback only when no service auto-restart happened.
  cd "$ROOT_DIR"
  nohup ./venv/bin/python main.py --mode live --config config.yaml > logs/live.out.log 2>&1 < /dev/null &
  sleep 6
fi

echo "[3/4] Ensure BTC intelligence watchdog permissions"
if [[ -f "$ROOT_DIR/scripts/ensure_btc_intel.sh" ]]; then
  sed -i 's/\r$//' "$ROOT_DIR/scripts/ensure_btc_intel.sh"
  chmod +x "$ROOT_DIR/scripts/ensure_btc_intel.sh"
fi

echo "[4/4] Health checks"
wait_health "http://127.0.0.1:8000/api/health" 20 2 || true
wait_health "http://127.0.0.1:9000/health" 20 2 || true
echo "MAIN PROC:"
ps -eo pid,user,cmd | grep 'main.py --mode live --config config.yaml' | grep -v grep || true
echo "BTC PROC:"
ps -eo pid,user,cmd | grep 'btc_intelligence.main:app' | grep -v grep || true
echo "API /health:"
curl -sS --max-time 8 http://127.0.0.1:8000/api/health | head -c 280 || true
echo
echo "BTC /health:"
curl -sS --max-time 8 http://127.0.0.1:9000/health | head -c 280 || true
echo
