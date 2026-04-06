"""Block until a TCP port accepts connections (used by Start_tradevex.bat)."""
from __future__ import annotations

import socket
import sys
import time


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
    max_wait = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    for _ in range(max_wait):
        try:
            with socket.create_connection((host, port), timeout=2.0):
                print(f"OK: {host}:{port} is open", flush=True)
                sys.exit(0)
        except OSError:
            time.sleep(1)
    print(f"TIMEOUT: {host}:{port} not reachable after {max_wait}s", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
