#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting BTC Quant Terminal..."
echo "Redis: localhost:6379 (optional)"
echo "Backend: http://127.0.0.1:9000"
echo "Dashboard: http://127.0.0.1:8000"

BTC_PID=""

cleanup() {
  if [[ -n "${BTC_PID}" ]] && kill -0 "${BTC_PID}" 2>/dev/null; then
    kill "${BTC_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if command -v redis-server >/dev/null 2>&1; then
  redis-server --daemonize yes 2>/dev/null || true
  sleep 1
else
  echo "[warn] redis-server not found — start Redis manually if btc_intelligence needs it."
fi

cd "${ROOT_DIR}"
python -m uvicorn btc_intelligence.main:app --host 127.0.0.1 --port 9000 --reload &
BTC_PID=$!

ok=0
for _i in {1..30}; do
  if curl -sf "http://127.0.0.1:9000/health" >/dev/null; then
    ok=1
    echo "BTC intelligence health OK (http://127.0.0.1:9000/health)."
    break
  fi
  sleep 1
done

if [[ "${ok}" -ne 1 ]]; then
  echo "error: btc_intelligence did not return HTTP 200 on /health within 30 attempts (30s)." >&2
  exit 1
fi

cd "${ROOT_DIR}"
python -m uvicorn src.dashboard.api:app --host 127.0.0.1 --port 8000 --reload
