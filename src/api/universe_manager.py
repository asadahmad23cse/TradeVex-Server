"""
Gap 5 Fix — Point-in-Time Universe Manager (Survivorship Bias Fix).

Problem:
    If you backtest on today's NIFTY 50 constituents, you're looking at
    stocks that SURVIVED. This inflates returns because you skip the
    delisted/merged/failed stocks.

Solution:
    Maintain a versioned universe registry:
        1. Record the watchlist as it was on each date
        2. Use only the universe that was active at time T for signals at time T
        3. Flag any asset that was delisted or merged during the backtest window

    For live trading: no change (always use current watchlist).
    For backtesting: use point-in-time universe snapshots.

Free data source:
    NSE publishes NIFTY 50 constituent changes on their website.
    We store these as a JSON registry in data/universe_snapshots.json.

Usage:
    um = UniverseManager()
    um.snapshot_current(watchlist_config)  # save today's universe
    universe = um.get_universe_at(date(2023, 6, 15))  # point-in-time lookup
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

UNIVERSE_FILE = Path("data/universe_snapshots.json")


class UniverseManager:
    """
    Manages point-in-time universe snapshots for survivorship-bias-free backtesting.

    The registry is a JSON file at data/universe_snapshots.json:
    [
        {
            "date": "2024-01-15",
            "indian_stocks": ["RELIANCE", "INFY", ...],
            "us_stocks": ["AAPL", "TSLA", ...],
            "forex": ["EURUSD", "GBPUSD", ...]
        },
        ...
    ]
    """

    def __init__(self, path: str | Path = UNIVERSE_FILE):
        self._path = Path(path)
        self._snapshots: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._snapshots = json.load(f)
                logger.info("Loaded %d universe snapshots", len(self._snapshots))
            except Exception as exc:
                logger.warning("Universe file load failed: %s", exc)
                self._snapshots = []
        else:
            self._snapshots = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._snapshots, f, indent=2, default=str)

    def snapshot_current(self, watchlist_config: dict) -> None:
        """
        Save today's watchlist as a point-in-time snapshot.

        Parameters
        ----------
        watchlist_config : dict
            Watchlist section from config.yaml.
            Expected format:
                {"indian_stocks": [...], "us_stocks": [...], "forex": [...]}
        """
        today = date.today().isoformat()

        # Don't duplicate if already saved today
        if any(s["date"] == today for s in self._snapshots):
            logger.debug("Universe snapshot for %s already exists", today)
            return

        snapshot = {
            "date": today,
            "indian_stocks": [
                a["symbol"] if isinstance(a, dict) else a
                for a in watchlist_config.get("indian_stocks", [])
            ],
            "us_stocks": [
                a["symbol"] if isinstance(a, dict) else a
                for a in watchlist_config.get("us_stocks", [])
            ],
            "forex": [
                a["symbol"] if isinstance(a, dict) else a
                for a in watchlist_config.get("forex", [])
            ],
        }

        self._snapshots.append(snapshot)
        self._snapshots.sort(key=lambda s: s["date"])
        self._save()
        logger.info("Universe snapshot saved for %s: %d assets", today, sum(
            len(snapshot[k]) for k in ["indian_stocks", "us_stocks", "forex"]
        ))

    def get_universe_at(self, target_date: date) -> dict:
        """
        Return the universe that was active on the given date.

        Uses the latest snapshot that is on or before target_date.
        If no snapshot exists before that date, returns the earliest available.

        Returns
        -------
        dict with keys: indian_stocks, us_stocks, forex → list of symbols
        """
        target = target_date.isoformat()

        # Find latest snapshot on or before target
        valid = [s for s in self._snapshots if s["date"] <= target]
        if valid:
            return valid[-1]

        # If no valid snapshot, return earliest
        if self._snapshots:
            return self._snapshots[0]

        return {"date": target, "indian_stocks": [], "us_stocks": [], "forex": []}

    def get_all_snapshots(self) -> list[dict]:
        return self._snapshots

    def get_delisted_between(self, start: date, end: date) -> dict[str, list[str]]:
        """
        Find symbols that were present in universe at `start` but missing by `end`.

        Returns
        -------
        dict[asset_class → list of symbols that disappeared]
        """
        start_uni = self.get_universe_at(start)
        end_uni = self.get_universe_at(end)

        result = {}
        for key in ["indian_stocks", "us_stocks", "forex"]:
            start_set = set(start_uni.get(key, []))
            end_set = set(end_uni.get(key, []))
            removed = start_set - end_set
            if removed:
                result[key] = sorted(removed)

        return result

    def count_snapshots(self) -> int:
        return len(self._snapshots)
