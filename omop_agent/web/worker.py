"""Runs one pipeline job for the web app: ``python -m omop_agent.web.worker <jobs/ID.json>``.

The run uses the reviewers' recorded approvals at the review gates: it continues past a gate
only if the current ETL spec or mappings were approved, and otherwise stops with status
``waiting_review``. Output goes to stdout, which the web app sends to the job's log file.
"""

from __future__ import annotations

import os
import signal
import sys
import traceback
from pathlib import Path

from ..approvals import GATE_PHRASES, recorded_approver
from ..config import load_config
from ..fsutil import atomic_write_text
from ..runlock import RunInProgress, run_lock
from ..workspace import Workspace, utcnow
from .jobs import KINDS, Job, JobStore


def _say(message: str) -> None:
    print(message, flush=True)


def _terminate(signum, frame):  # noqa: ARG001 - signal handler signature
    raise SystemExit(128 + signum)


def run_job(job_path: Path) -> Job:
    job_path = Path(job_path)
    job = Job.model_validate_json(job_path.read_text())
    try:
        cfg = load_config(job.config_path)
    except Exception as e:
        job.status, job.ended_at = "failed", utcnow()
        job.message = f"Could not read the project configuration: {type(e).__name__}: {str(e)[:400]}"
        atomic_write_text(job_path, job.model_dump_json(indent=2) + "\n")
        _say(job.message)
        return job
    ws = Workspace(cfg.workdir)
    store = JobStore(ws)
    if store.path(job.id).resolve() != job_path.resolve():
        raise SystemExit(f"{job_path} is not a job record of this project")

    try:
        with run_lock(ws, job.id, wait_s=10):
            _run(job, cfg, ws, store)
    except RunInProgress as e:
        job.status, job.message, job.ended_at = "failed", str(e).capitalize() + ".", utcnow()
        store.save(job)
        _say(job.message)
    return job


def _run(job: Job, cfg, ws: Workspace, store: JobStore) -> None:
    from ..pipeline import Pipeline, PipelineStopped, SpecInvalid
    from .projects import schema_conflicts_for

    job.status, job.started_at, job.pid = "running", utcnow(), os.getpid()
    store.save(job)
    ws.audit("web_run_started", job=job.id, kind=job.kind, by=job.by)
    _say(f"Run {job.id} ({job.label}) started by {job.by} at {job.started_at:%Y-%m-%d %H:%M:%S} UTC")
    previous = signal.signal(signal.SIGTERM, _terminate)
    p = None
    try:
        clashes = schema_conflicts_for(Path(job.config_path).parent, cfg)
        if clashes:
            raise RuntimeError(
                "this project writes to database schemas another project uses ("
                + "; ".join(clashes) + "). Give it its own schema names in omop_agent.yaml."
            )
        p = Pipeline(cfg, log=_say)
        p.approver = recorded_approver(p.ws)
        report = p.run_all(**KINDS[job.kind])
        job.status = "passed" if report.passed else "failed_quality"
        job.message = (
            "The CDM passed the data-quality gate." if report.passed
            else f"{len(report.blocking_failures)} blocking data-quality check(s) failed; see the report."
        )
        if report.passed:
            from ..report.inspection import review_caveats

            caveats = review_caveats(cfg, ws)
            if caveats:
                job.message += " Not fully reviewed: " + "; ".join(caveats) + "."
    except PipelineStopped as e:
        job.status, job.stage = "waiting_review", e.stage
        what = GATE_PHRASES.get(e.stage or "", "artifacts")
        job.message = f"Waiting for a reviewer to approve the {what}."
    except SpecInvalid as e:
        job.status, job.message = "failed", f"The ETL spec has errors: {str(e)[:500]}"
    except SystemExit:
        job.status, job.message = "cancelled", "The run was stopped."
    except Exception as e:
        traceback.print_exc()
        job.status, job.message = "failed", f"{type(e).__name__}: {str(e)[:500]}"
    finally:
        signal.signal(signal.SIGTERM, previous)
        if p is not None:
            p.close()
        job.ended_at = utcnow()
        store.save(job)
        ws.audit("web_run_finished", job=job.id, status=job.status, message=job.message)
        _say(f"Finished: {job.status}. {job.message}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m omop_agent.web.worker <jobs/ID.json>", file=sys.stderr)
        return 2
    job = run_job(Path(argv[0]))
    return 0 if job.status in ("passed", "waiting_review") else 1


if __name__ == "__main__":
    sys.exit(main())
