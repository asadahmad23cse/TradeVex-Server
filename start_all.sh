#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting BTC Quant Terminal..."
echo "Backend: http://127.0.0.1:9000"
echo "Dashboard: http://127.0.0.1:8000"

(
  cd "${ROOT_DIR}/btc_intelligence"
  python -m uvicorn main:app --host 127.0.0.1 --port 9000 --reload
) &

sleep 3

cd "${ROOT_DIR}/src/dashboard"
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
