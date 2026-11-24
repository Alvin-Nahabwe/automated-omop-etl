"""Mapping memory: the code mappings people approved in earlier projects, retrieved when new
source terms are mapped.

The memory is one SQLite file (``mapping.memory_path``), normally shared by all the projects on
a server. Each project's entries are replaced whenever its decisions are written to
``source_to_concept_map``, so the memory always holds the project's current reviewed answers.
Only decisions a person approved are remembered: a reviewer's own concept (method ``manual``)
or a suggestion a reviewer approved in the web app (a ``reviewed_at`` stamp). Automatic
approvals (exact matches, confident model answers, earlier memory reuse) are not, so the
memory never learns from itself. Terms seen fewer than ``privacy.min_cell_count`` times are
left out.

When terms are mapped (``vocab.mapper``), the memory is used in two ways:

* **reuse**: a term whose description is the same as a remembered one (after normalisation),
  where every remembered answer names the same concept and that concept is still a valid
  standard concept of a domain the term accepts, takes that concept (method ``memory``)
  without asking the model;
* **retrieval**: for the terms the model maps, the concepts reviewers chose for the most similar
  remembered terms are added to the candidates, and listed in the prompt as reviewed examples.

A description without a word of three or more letters ("1", "M", "2b") is never reused or
retrieved: such local codes mean different things in different sources.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..models import MappingDecision
from ..workspace import utcnow

SIMILAR_MIN_SCORE = 0.6  # text_score of two descriptions for a remembered term to be retrieved
SIMILAR_LIMIT = 3  # remembered terms retrieved per source term
SIMILAR_POOL = 200  # remembered terms scored per source term (those sharing the most word starts)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mapping (
    project TEXT NOT NULL,
    project_name TEXT,
    source_vocabulary_id TEXT NOT NULL,
    source_code TEXT NOT NULL,
    source_name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    target_domain TEXT,
    target_concept_id INTEGER NOT NULL,
    target_concept_name TEXT,
    target_domain_id TEXT,
    target_vocabulary_id TEXT,
    equivalence TEXT,
    reviewed_by TEXT,
    reviewed_at TEXT,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (project, source_vocabulary_id, source_code)
);
CREATE INDEX IF NOT EXISTS mapping_name_key ON mapping (name_key);
"""

_WORD = re.compile(r"[a-z]{3,}")


@dataclass(frozen=True)
class Remembered:
    """One remembered mapping."""

    project: str
    project_name: str | None
    source_vocabulary_id: str
    source_code: str
    source_name: str
    target_concept_id: int
    target_concept_name: str | None
    target_domain_id: str | None
    equivalence: str | None
    reviewed_by: str | None
    reviewed_at: str | None
    score: float = 1.0  # similarity of its description to the term it was retrieved for

    def where(self) -> str:
        """Who approved it, in which project (for rationales)."""
        who = f" by {self.reviewed_by}" if self.reviewed_by else ""
        when = f" on {self.reviewed_at[:10]}" if self.reviewed_at else ""
        return f"in project {self.project_name or self.project}{who}{when}"


def name_key(text: str) -> str:
    """The form descriptions are compared in (the mapper's normalisation)."""
    from .mapper import normalize

    return normalize(text)


def meaningful(key: str) -> bool:
    """A normalised description that says what it is (a word of three letters or more), unlike
    local codes such as '1' or 'M'."""
    return bool(_WORD.search(key))


def _prefixes(key: str) -> set[str]:
    return {w[:3] for w in key.split() if len(w) >= 3 and w[:3].isalpha()}


def human_approved(d: MappingDecision) -> bool:
    """A mapping a person chose or approved (not an automatic approval)."""
    return d.status == "APPROVED" and d.target_concept_id != 0 and (d.method == "manual" or bool(d.reviewed_at))


class MappingMemory:
    """The SQLite store. Opened per use; safe to share between projects and processes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rows: list[Remembered] | None = None
        self._keys: list[str] = []  # name_key of each row
        self._by_key: dict[str, list[int]] = {}
        self._by_prefix: dict[str, set[int]] = {}  # first 3 letters of a word -> rows (similar())

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    def record(
        self, project: str, project_name: str | None, decisions: Iterable[MappingDecision], min_count: int = 0
    ) -> int:
        """Replace ``project``'s entries with its human-approved decisions; returns how many."""
        now = utcnow().isoformat(timespec="seconds")
        rows = [
            (project, project_name, d.source_vocabulary_id, d.source_code, d.source_name, name_key(d.source_name),
             d.target_domain, d.target_concept_id, d.target_concept_name, d.target_domain_id,
             d.target_vocabulary_id, d.equivalence, d.reviewed_by, d.reviewed_at, now)
            for d in decisions
            if human_approved(d) and d.frequency >= min_count
        ]
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM mapping WHERE project = ?", (project,))
                conn.executemany(f"INSERT INTO mapping VALUES ({', '.join('?' * 15)})", rows)
        finally:
            conn.close()
        self._rows = None
        return len(rows)

    def _all(self) -> list[Remembered]:
        if self._rows is None:
            if not self.path.exists():
                self._rows = []
                return self._rows
            conn = self._connect()
            try:
                self._rows = [
                    Remembered(
                        project=r["project"], project_name=r["project_name"],
                        source_vocabulary_id=r["source_vocabulary_id"], source_code=r["source_code"],
                        source_name=r["source_name"], target_concept_id=r["target_concept_id"],
                        target_concept_name=r["target_concept_name"], target_domain_id=r["target_domain_id"],
                        equivalence=r["equivalence"], reviewed_by=r["reviewed_by"], reviewed_at=r["reviewed_at"],
                    )
                    for r in conn.execute("SELECT * FROM mapping ORDER BY project, source_vocabulary_id, source_code")
                ]
            finally:
                conn.close()
            self._keys = [name_key(r.source_name) for r in self._rows]
            self._by_key, self._by_prefix = {}, {}
            for i, key in enumerate(self._keys):
                if meaningful(key):
                    self._by_key.setdefault(key, []).append(i)
                    for p in _prefixes(key):
                        self._by_prefix.setdefault(p, set()).add(i)
        return self._rows

    def same(self, source_name: str) -> list[Remembered]:
        """Remembered terms with the same (normalised, meaningful) description."""
        key = name_key(source_name)
        if not meaningful(key):
            return []
        rows = self._all()
        return [rows[i] for i in self._by_key.get(key, [])]

    def similar(self, source_name: str, limit: int = SIMILAR_LIMIT, min_score: float = SIMILAR_MIN_SCORE
                ) -> list[Remembered]:
        """The remembered terms whose descriptions are most like ``source_name`` (best first, one per
        concept)."""
        from .mapper import text_score

        key = name_key(source_name)
        if not meaningful(key):
            return []
        rows = self._all()
        # only the rows that share the start of a word with the description can score: the
        # SIMILAR_POOL rows that share the most are scored
        shared: Counter[int] = Counter()
        for p in _prefixes(key):
            shared.update(self._by_prefix.get(p, ()))
        near = sorted(shared, key=lambda i: (-shared[i], i))[:SIMILAR_POOL]
        scored: dict[int, Remembered] = {}
        for i in near:
            r = rows[i]
            s = round(text_score(key, self._keys[i]), 3)
            if s >= min_score and (r.target_concept_id not in scored or s > scored[r.target_concept_id].score):
                scored[r.target_concept_id] = Remembered(**{**r.__dict__, "score": s})
        return sorted(scored.values(), key=lambda r: (-r.score, r.target_concept_id))[:limit]

    def stats(self) -> dict[str, int]:
        rows = self._all()
        return {
            "mappings": len(rows),
            "projects": len({r.project for r in rows}),
            "concepts": len({r.target_concept_id for r in rows}),
        }


__all__ = ["MappingMemory", "Remembered", "human_approved", "meaningful", "name_key"]
