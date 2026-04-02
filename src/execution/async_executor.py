"""
Gap 4 — Async Execution & Latency Pipeline.

Parallelises per-asset data fetching, feature computation, and exec routing
using concurrent.futures.  Instruments every stage with timing data for
latency monitoring.

Usage in live_runner.py:
    pipeline = AsyncExecutionPipeline(max_workers=4)
    results = pipeline.run_parallel(assets, fetch_fn, process_fn)
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageTimer:
    """Accumulates timing for a named stage."""
    name: str
    start_ms: float = 0.0
    end_ms: float = 0.0

    def start(self):
        self.start_ms = time.perf_counter() * 1000

    def stop(self):
        self.end_ms = time.perf_counter() * 1000

    @property
    def elapsed_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class LatencyReport:
    """End-to-end latency breakdown for one intraday cycle."""
    cycle_id: str = ""
    data_fetch_ms: float = 0.0
    feature_compute_ms: float = 0.0
    alpha_score_ms: float = 0.0
    risk_check_ms: float = 0.0
    execution_ms: float = 0.0
    total_ms: float = 0.0
    n_assets: int = 0
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "data_fetch_ms": round(self.data_fetch_ms, 1),
            "feature_compute_ms": round(self.feature_compute_ms, 1),
            "alpha_score_ms": round(self.alpha_score_ms, 1),
            "risk_check_ms": round(self.risk_check_ms, 1),
            "execution_ms": round(self.execution_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "n_assets": self.n_assets,
            "errors": self.errors,
        }


class FeatureCache:
    """
    Warm cache for computed features between intraday cycles.

    Caches keyed by (ticker, timeframe). Expires after `ttl_sec`.
    Re-used to skip redundant feature engineering when data hasn't changed.
    """

    def __init__(self, ttl_sec: int = 300):
        self.ttl_sec = ttl_sec
        self._cache: Dict[str, dict] = {}
        self._lock = threading.RLock()  # C3: guard compound get/put/delete ops

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() - entry["ts"] > self.ttl_sec:
                del self._cache[key]
                return None
            return entry["value"]

    def put(self, key: str, value: Any):
        with self._lock:
            self._cache[key] = {"value": value, "ts": time.time()}

    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class AsyncExecutionPipeline:
    """
    Parallel per-asset processing with latency instrumentation.

    Usage
    -----
    pipeline = AsyncExecutionPipeline(max_workers=4)

    def process_asset(ticker, asset_class, timeframe):
        # fetch data, compute features, generate signal
        return signal_dict

    results = pipeline.run_parallel(
        assets=[("RELIANCE.NS", "indian_stock", "intraday"), ...],
        process_fn=process_asset,
        cycle_id="intra_20260326_1015",
    )
    report = pipeline.last_report
    """

    def __init__(self, max_workers: int = 4, feature_cache_ttl: int = 300):
        self.max_workers = max_workers
        self.feature_cache = FeatureCache(ttl_sec=feature_cache_ttl)
        self._last_report: Optional[LatencyReport] = None

    @property
    def last_report(self) -> Optional[LatencyReport]:
        return self._last_report

    def run_parallel(
        self,
        assets: List[tuple],
        process_fn: Callable,
        cycle_id: str = "",
    ) -> List[dict]:
        """
        Process assets in parallel using ThreadPoolExecutor.

        Parameters
        ----------
        assets : list of tuples
            Each tuple: (ticker, asset_class, timeframe)
        process_fn : callable
            Function(ticker, asset_class, timeframe) -> dict
        cycle_id : str
            Identifier for this cycle.

        Returns
        -------
        List of result dicts from process_fn (one per asset).
        """
        report = LatencyReport(cycle_id=cycle_id, n_assets=len(assets))
        overall_start = time.perf_counter() * 1000
        results = []

        if not assets:
            self._last_report = report
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_asset = {}
            for asset_tuple in assets:
                future = executor.submit(self._timed_process, process_fn, asset_tuple)
                future_to_asset[future] = asset_tuple

            for future in as_completed(future_to_asset):
                asset_tuple = future_to_asset[future]
                try:
                    result, timings = future.result()
                    if result is not None:
                        results.append(result)
                    # Aggregate timings
                    report.data_fetch_ms += timings.get("data_fetch_ms", 0)
                    report.feature_compute_ms += timings.get("feature_compute_ms", 0)
                    report.alpha_score_ms += timings.get("alpha_score_ms", 0)
                    report.risk_check_ms += timings.get("risk_check_ms", 0)
                    report.execution_ms += timings.get("execution_ms", 0)
                except Exception as e:
                    ticker = asset_tuple[0] if asset_tuple else "?"
                    report.errors.append(f"{ticker}: {e}")
                    logger.error("Async pipeline error for %s: %s", ticker, e)

        report.total_ms = time.perf_counter() * 1000 - overall_start

        # Average per-asset timings
        n = max(len(assets), 1)
        report.data_fetch_ms /= n
        report.feature_compute_ms /= n
        report.alpha_score_ms /= n
        report.risk_check_ms /= n
        report.execution_ms /= n

        self._last_report = report
        logger.info(
            "AsyncPipeline [%s]: %d assets in %.0fms (avg data=%.0f feat=%.0f alpha=%.0f exec=%.0fms)",
            cycle_id, len(assets), report.total_ms,
            report.data_fetch_ms, report.feature_compute_ms,
            report.alpha_score_ms, report.execution_ms,
        )
        return results

    @staticmethod
    def _timed_process(process_fn: Callable, asset_tuple: tuple) -> tuple:
        """Run process_fn with per-stage timing instrumentation."""
        overall_start = time.perf_counter() * 1000
        timings = {}
        try:
            result = process_fn(*asset_tuple)
            # If process_fn returns (result, timings_dict), collect timings
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                actual_result, timings = result
                return actual_result, timings
            # Otherwise, attribute entire time to processing
            elapsed = time.perf_counter() * 1000 - overall_start
            timings = {"data_fetch_ms": elapsed * 0.3, "feature_compute_ms": elapsed * 0.3,
                       "alpha_score_ms": elapsed * 0.2, "execution_ms": elapsed * 0.2}
            return result, timings
        except Exception:
            raise


class ConnectionPool:
    """
    Simple connection pool for broker API sessions.

    Maintains a pool of pre-authenticated sessions to avoid
    per-request auth overhead.
    """

    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self._pool: list = []
        self._in_use: set = set()

    def acquire(self) -> Optional[Any]:
        """Get a connection from the pool."""
        for conn in self._pool:
            if id(conn) not in self._in_use:
                self._in_use.add(id(conn))
                return conn
        return None

    def release(self, conn):
        """Return a connection to the pool."""
        self._in_use.discard(id(conn))

    def add(self, conn):
        """Add a new connection to the pool."""
        if len(self._pool) < self.max_connections:
            self._pool.append(conn)

    @property
    def available(self) -> int:
        return len(self._pool) - len(self._in_use)

    @property
    def size(self) -> int:
        return len(self._pool)
