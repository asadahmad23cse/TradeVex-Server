"""
Watch port 9000; if BTC intelligence stops responding, restart uvicorn.

Run in a separate console alongside the dashboard (optional). Uses project
root and .venv Python when present.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.is_file() else sys.executable
HOST, PORT = "127.0.0.1", 9000
INTERVAL_SEC = 12


def _port_open() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=2.0):
            return True
    except OSError:
        return False


def main() -> None:
    proc: subprocess.Popen | None = None
    print(f"BTC Intel watchdog: monitoring {HOST}:{PORT}", flush=True)
    while True:
        time.sleep(INTERVAL_SEC)
        if _port_open():
            continue
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                pass
        print("BTC Intel: port down — starting uvicorn...", flush=True)
        try:
            proc = subprocess.Popen(
                [
                    PY,
                    "-m",
                    "uvicorn",
                    "btc_intelligence.main:app",
                    "--host",
                    HOST,
                    "--port",
                    str(PORT),
                ],
                cwd=str(ROOT),
            )
        except Exception as exc:
            print(f"BTC Intel watchdog spawn failed: {exc}", flush=True)
            proc = None


if __name__ == "__main__":
    main()
