"""One pipeline run per project at a time, whether it is started from the command line or the
web app. A run holds ``<workdir>/jobs/run.lock`` and names itself in ``jobs/current``; the web
app also takes the lock for a moment while it saves a reviewer's mapping edit, so a run never
overwrites an edit it did not read.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .fsutil import atomic_write_text, file_lock
from .workspace import Workspace


class RunInProgress(RuntimeError):
    """Another run (or an edit) holds this project's run lock."""


def lock_holder(ws: Workspace) -> str | None:
    """Who holds the run lock ("" if they have not said yet), or None when it is free."""
    if not ws.run_lock.exists():
        return None
    with file_lock(ws.run_lock, blocking=False) as free:
        if free:
            return None
    try:
        return ws.run_holder.read_text().strip()
    except FileNotFoundError:
        return ""


@contextmanager
def run_lock(ws: Workspace, holder: str, wait_s: float = 5.0) -> Iterator[None]:
    """Hold the project's run lock for the ``with`` block, or raise RunInProgress."""
    with file_lock(ws.run_lock, blocking=False, wait_s=wait_s) as got:
        if not got:
            try:
                who = ws.run_holder.read_text().strip()
            except FileNotFoundError:
                who = ""
            raise RunInProgress(
                "another run of this project is in progress" + (f" ({who})" if who else "")
            )
        atomic_write_text(ws.run_holder, holder + "\n")
        try:
            yield
        finally:
            try:
                ws.run_holder.unlink()
            except FileNotFoundError:
                pass
