"""Small file helpers shared by the CLI, the pipeline worker and the web app.

Several processes can touch a project's workspace at once (a reviewer in the web app, a
pipeline run in the background, someone editing a CSV on the server). Files that more than one
of them writes are replaced atomically and updated under an advisory lock.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # POSIX only; the web app and the Docker image run on Linux
    import fcntl
except ImportError:  # pragma: no cover - Windows: locking is skipped
    fcntl = None  # type: ignore[assignment]


_UMASK = os.umask(0)
os.umask(_UMASK)


def atomic_write_text(path: Path, text: str, mode: int | None = None) -> Path:
    """Write ``text`` to ``path`` so readers see either the old or the new file, never half.

    The file keeps its current permissions (new files get the usual umask-based ones) unless
    ``mode`` is given.
    """
    path = Path(path)
    if mode is None:
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o666 & ~_UMASK
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


@contextmanager
def file_lock(path: Path, blocking: bool = True) -> Iterator[bool]:
    """Hold an exclusive advisory lock on ``path`` (created if missing) for the ``with`` block.

    Yields True when the lock is held. With ``blocking=False`` it yields False instead of
    waiting when another process holds the lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        if fcntl is None:
            yield True
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
