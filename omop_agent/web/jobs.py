"""Pipeline runs started from the web app.

Each run is a separate process (``python -m omop_agent.web.worker <job.json>``) so a long run
survives a slow page and a crashed run cannot take the web app down. Its record and log live
in ``<workdir>/jobs/``. The worker holds an exclusive lock on ``jobs/run.lock`` while it runs
and writes its job id to ``jobs/current``: the lock, not the record, says whether a run is in
progress, so a run that died without updating its record shows as interrupted.
"""

from __future__ import annotations

import os
import re
import secrets
import signal
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..fsutil import atomic_write_text, file_lock
from ..workspace import Workspace, utcnow

JOB_ID = re.compile(r"^\d{8}T\d{6}-[0-9a-f]{6}$")
# What each kind of run passes to Pipeline.run_all.
KINDS: dict[str, dict[str, bool]] = {
    "continue": {},
    "remap": {"remap": True},
    "redesign": {"redesign": True},
    "fresh": {"fresh": True},
}
KIND_LABELS = {
    "continue": "Continue the pipeline",
    "remap": "Map every code again",
    "redesign": "Redraft the ETL design",
    "fresh": "Start over from the source data",
}
JobStatus = Literal[
    "queued", "running", "waiting_review", "passed", "failed_quality", "failed", "cancelled", "interrupted"
]
FINISHED = {"waiting_review", "passed", "failed_quality", "failed", "cancelled"}
STARTUP_GRACE = timedelta(seconds=30)  # a queued run may take this long to take its lock
_start_lock = threading.Lock()


class JobBusy(RuntimeError):
    """A run of this project is already in progress."""


class Job(BaseModel):
    id: str
    kind: str
    by: str
    config_path: str
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    status: JobStatus = "queued"
    message: str = ""
    stage: str | None = None  # the review gate a run stopped at
    pid: int | None = None

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)


class JobStore:
    def __init__(self, ws: Workspace):
        self.dir = ws.root / "jobs"

    @property
    def lock_path(self) -> Path:
        return self.dir / "run.lock"

    def path(self, job_id: str) -> Path:
        if not JOB_ID.match(job_id):
            raise ValueError(f"not a job id: {job_id!r}")
        return self.dir / f"{job_id}.json"

    def log_path(self, job_id: str) -> Path:
        return self.path(job_id).with_suffix(".log")

    # ---- records -----------------------------------------------------------------------
    def save(self, job: Job) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path(job.id), job.model_dump_json(indent=2) + "\n")

    def get(self, job_id: str) -> Job | None:
        try:
            path = self.path(job_id)
        except ValueError:
            return None
        return Job.model_validate_json(path.read_text()) if path.exists() else None

    def list(self, limit: int = 50) -> list[Job]:
        if not self.dir.exists():
            return []
        names = sorted((p.stem for p in self.dir.glob("*.json") if JOB_ID.match(p.stem)), reverse=True)
        return [j for j in (self.get(n) for n in names[:limit]) if j is not None]

    def running(self) -> bool:
        """True while a worker holds the run lock."""
        if not self.lock_path.exists():
            return False
        with file_lock(self.lock_path, blocking=False) as got:
            return not got

    @property
    def current_path(self) -> Path:
        return self.dir / "current"

    def holder(self) -> str | None:
        """Id of the job holding the run lock ("" if it has not written it yet), else None."""
        if not self.running():
            return None
        try:
            return self.current_path.read_text().strip()
        except FileNotFoundError:
            return ""

    def status(self, job: Job) -> str:
        """The job's status, with runs that died without a word shown as interrupted."""
        if job.status in FINISHED:
            return job.status
        if self.holder() == job.id:
            return "running"
        if job.status == "queued" and utcnow() - job.created_at < STARTUP_GRACE:
            return "queued"
        return "interrupted"

    def active(self) -> Job | None:
        """The run in progress (or just started), if any."""
        for job in self.list(limit=5):
            if self.status(job) in ("queued", "running"):
                return job
        return None

    def log_tail(self, job_id: str, max_lines: int = 400) -> str:
        path = self.log_path(job_id)
        if not path.exists():
            return ""
        lines = path.read_text(errors="replace").splitlines()
        skipped = len(lines) - max_lines
        tail = lines[-max_lines:]
        return (f"… {skipped} earlier lines not shown\n" if skipped > 0 else "") + "\n".join(tail)

    # ---- actions -----------------------------------------------------------------------
    def start(self, kind: str, by: str, config_path: Path, command: list[str] | None = None) -> Job:
        """Start a run in the background. Raises JobBusy if one is already going."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind of run: {kind}")
        with _start_lock:
            if self.running() or self.active() is not None:
                raise JobBusy("a run of this project is already in progress")
            now = utcnow()
            job = Job(
                id=f"{now:%Y%m%dT%H%M%S}-{secrets.token_hex(3)}", kind=kind, by=by,
                config_path=str(config_path), created_at=now,
            )
            self.save(job)
            cmd = command or [sys.executable, "-m", "omop_agent.web.worker", str(self.path(job.id))]
            with self.log_path(job.id).open("ab") as log:
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    cwd=str(Path(config_path).parent), start_new_session=True, close_fds=True,
                )
            job.pid = proc.pid
            self.save(job)
            # Reap the child when it exits so it does not linger as a zombie.
            threading.Thread(target=proc.wait, daemon=True).start()
            return job

    def cancel(self, job_id: str) -> bool:
        """Ask a running job to stop (SIGTERM to its process group). False if it is not running."""
        job = self.get(job_id)
        if job is None or job.pid is None or self.status(job) not in ("queued", "running"):
            return False
        try:
            os.killpg(job.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False
        return True
