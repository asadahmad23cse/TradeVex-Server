"""
Scheduler watchdog for APScheduler jobs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class JobHealth:
    name: str
    expected_interval_min: int
    last_run_at: datetime | None = None
    last_alert_at: datetime | None = None
    stale: bool = False
    message: str = ""


class SchedulerHealthWatchdog:
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self.alert_cooldown_min = int(cfg.get("alert_cooldown_min", 30))
        self.jobs: dict[str, JobHealth] = {}

    def register_job(self, name: str, expected_interval_min: int) -> None:
        self.jobs[name] = JobHealth(name=name, expected_interval_min=expected_interval_min)

    def heartbeat(self, name: str, when: datetime | None = None) -> None:
        if name not in self.jobs:
            self.register_job(name, 5)
        job = self.jobs[name]
        job.last_run_at = when or datetime.utcnow()
        job.stale = False
        job.message = "OK"

    def check(self, now: datetime | None = None) -> list[JobHealth]:
        if not self.enabled:
            return []
        current = now or datetime.utcnow()
        stale_jobs: list[JobHealth] = []
        for job in self.jobs.values():
            if job.last_run_at is None:
                continue
            max_delay = timedelta(minutes=job.expected_interval_min)
            if current - job.last_run_at > max_delay:
                job.stale = True
                lag_min = (current - job.last_run_at).total_seconds() / 60.0
                job.message = (
                    f"{job.name} stale for {lag_min:.1f} min "
                    f"(expected <= {job.expected_interval_min} min)"
                )
                stale_jobs.append(job)
        return stale_jobs

    def due_for_alert(self, job: JobHealth, now: datetime | None = None) -> bool:
        current = now or datetime.utcnow()
        if job.last_alert_at is None:
            return True
        return current - job.last_alert_at >= timedelta(minutes=self.alert_cooldown_min)

    def mark_alerted(self, job: JobHealth, when: datetime | None = None) -> None:
        job.last_alert_at = when or datetime.utcnow()
        logger.warning("Watchdog alert marked for %s", job.name)
