"""Human review of vocabulary mappings (the Usagi review screen, as files).

``export_review`` writes ``mappings/decisions.json`` (the full decisions, candidates included)
and one CSV per source vocabulary (``mappings/<source_vocabulary_id>.csv``) with Usagi-like
columns. A reviewer edits the CSVs in a spreadsheet: sets ``status`` (APPROVED, UNCHECKED,
FLAGGED, REJECTED), or puts another ``target_concept_id`` (0 = no mapping), optionally taking
one from ``alternatives``. ``import_review`` applies those edits on top of ``decisions.json``.
Only cells that differ from ``decisions.json`` count as edits, so importing twice gives the
same result.

The web app's review screen edits the same CSVs (``update_review_row``), so spreadsheet,
command-line and browser reviews are interchangeable. It also fills ``reviewed_by``,
``reviewed_at`` and ``review_note``, which are carried into the decisions and kept on re-runs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from pathlib import Path
from typing import Mapping, get_args

import psycopg

from ..db import Schemas
from ..etl.routing import VocabularyDomains
from ..fsutil import atomic_write_text, file_lock
from ..models import Equivalence, MappingDecision, MappingStatus
from ..workspace import Workspace
from .mapper import accepted_domains, check_concept

REVIEW_COLUMNS = [
    "status",
    "source_vocabulary_id",
    "source_code",
    "source_name",
    "frequency",
    "target_domain",
    "target_concept_id",
    "target_concept_name",
    "target_domain_id",
    "target_vocabulary_id",
    "equivalence",
    "confidence",
    "method",
    "rationale",
    "alternatives",
    "review_note",
    "reviewed_by",
    "reviewed_at",
]
DECISIONS_FILE = "decisions.json"
_STATUSES = set(get_args(MappingStatus))
_EQUIVALENCES = set(get_args(Equivalence))


def decisions_path(ws: Workspace) -> Path:
    return ws.mappings_dir / DECISIONS_FILE


def _csv_name(source_vocabulary_id: str) -> str:
    """``<id>.csv``; ids with other characters get a hash suffix, so two ids never share a file."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_vocabulary_id).strip("_.") or "vocabulary"
    if safe != source_vocabulary_id:
        safe += "-" + hashlib.sha1(source_vocabulary_id.encode()).hexdigest()[:8]
    return safe + ".csv"


def _alternatives(d: MappingDecision) -> str:
    return "; ".join(f"{c.concept_id}:{c.concept_name}" for c in d.candidates if c.concept_id != d.target_concept_id)


def _row(d: MappingDecision) -> dict[str, object]:
    return {
        "status": d.status,
        "source_vocabulary_id": d.source_vocabulary_id,
        "source_code": d.source_code,
        "source_name": d.source_name,
        "frequency": d.frequency,
        "target_domain": d.target_domain or "",
        "target_concept_id": d.target_concept_id,
        "target_concept_name": d.target_concept_name or "",
        "target_domain_id": d.target_domain_id or "",
        "target_vocabulary_id": d.target_vocabulary_id or "",
        "equivalence": d.equivalence,
        "confidence": f"{d.confidence:.2f}",
        "method": d.method,
        "rationale": d.rationale,
        "alternatives": _alternatives(d),
        "review_note": d.review_note,
        "reviewed_by": d.reviewed_by or "",
        "reviewed_at": d.reviewed_at or "",
    }


def _lock_path(ws: Workspace) -> Path:
    return ws.mappings_dir / ".review.lock"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    atomic_write_text(path, buf.getvalue())


def export_review(ws: Workspace, decisions: list[MappingDecision]) -> list[Path]:
    """Write ``decisions.json`` and one review CSV per source vocabulary. Returns the paths."""
    with file_lock(_lock_path(ws)):
        paths = [ws.save_json(decisions_path(ws), decisions)]
        by_vocab: dict[str, list[MappingDecision]] = {}
        for d in decisions:
            by_vocab.setdefault(d.source_vocabulary_id, []).append(d)
        for svid in sorted(by_vocab):
            path = ws.mappings_dir / _csv_name(svid)
            _write_csv(path, [_row(d) for d in by_vocab[svid]])
            paths.append(path)
    return paths


def update_review_row(ws: Workspace, source_vocabulary_id: str, source_code: str, changes: dict[str, str]) -> None:
    """Change cells of one term's row in its review CSV, as a reviewer would in a spreadsheet.

    ``import_review`` then applies the change like any other CSV edit. The CSV is rewritten
    atomically under the review lock, so concurrent edits of different terms are not lost.
    Raises KeyError when the term is not in the review files.
    """
    unknown = set(changes) - set(REVIEW_COLUMNS)
    if unknown:
        raise ValueError(f"not review columns: {', '.join(sorted(unknown))}")
    with file_lock(_lock_path(ws)):
        path = ws.mappings_dir / _csv_name(source_vocabulary_id)
        if path.exists():
            rows: list[dict[str, object]] = list(_read_csv(path))
        else:
            decisions = ws.load_json_list(decisions_path(ws), MappingDecision)
            rows = [_row(d) for d in decisions if d.source_vocabulary_id == source_vocabulary_id]
        for r in rows:
            if r.get("source_vocabulary_id") == source_vocabulary_id and r.get("source_code") == source_code:
                r.update(changes)
                break
        else:  # the row was deleted from the CSV: restore it from decisions.json
            decisions = ws.load_json_list(decisions_path(ws), MappingDecision)
            d = next(
                (x for x in decisions if x.source_vocabulary_id == source_vocabulary_id and x.source_code == source_code),
                None,
            )
            if d is None:
                raise KeyError(f"{source_vocabulary_id} {source_code!r} is not in the review files")
            rows.append({**_row(d), **changes})
        _write_csv(path, rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:  # tolerate a BOM from spreadsheet tools
        return [{(k or "").strip(): (v or "") for k, v in r.items()} for r in csv.DictReader(fh)]


def _parse_id(text: str) -> int | None:
    text = text.strip()
    if text == "":
        return 0
    try:
        return int(float(text))  # spreadsheets sometimes write 2000003002.0
    except ValueError:
        return None


def _apply_edit(
    d: MappingDecision, row: dict[str, str], conn: psycopg.Connection | None, schemas: Schemas | None
) -> MappingDecision:
    out = _apply_cells(d, row, conn, schemas)
    review_note = row.get("review_note")
    if review_note is not None and review_note.strip() != d.review_note:
        out.review_note = review_note.strip()
    # Who reviewed: a new reviewed_at stamp (the web app always writes one) names the reviewer.
    # An edit of the status or concept without a new stamp (a spreadsheet edit) is not credited
    # to whoever reviewed the term before.
    stamp = (row.get("reviewed_at") or "").strip()
    if stamp and stamp != (d.reviewed_at or ""):
        out.reviewed_at = stamp
        out.reviewed_by = (row.get("reviewed_by") or "").strip() or None
    elif (out.status, out.target_concept_id) != (d.status, d.target_concept_id):
        out.reviewed_by = out.reviewed_at = None
    return out


def _apply_cells(
    d: MappingDecision, row: dict[str, str], conn: psycopg.Connection | None, schemas: Schemas | None
) -> MappingDecision:
    d = d.model_copy(deep=True)
    status = row.get("status", "").strip().upper()
    status_changed = status in _STATUSES and status != d.status
    if status_changed:
        d.status = status  # type: ignore[assignment]
    elif status and status not in _STATUSES:
        d.status = "FLAGGED"
        d.rationale = (
            f"Unknown status '{row['status'].strip()}' in the review file; use one of {', '.join(sorted(_STATUSES))}."
        )
        return d

    eq = row.get("equivalence", "").strip().upper()
    if eq in _EQUIVALENCES and eq != d.equivalence:
        d.equivalence = eq  # type: ignore[assignment]
    note = row.get("rationale")
    if note is not None and note.strip() and note != d.rationale:
        d.rationale = note

    if "target_concept_id" not in row:
        return d
    new_id = _parse_id(row["target_concept_id"])
    if new_id is None:
        d.status = "FLAGGED"
        d.rationale = f"Reviewer entered target_concept_id '{row['target_concept_id']}', which is not a number."
        return d
    if new_id == d.target_concept_id:
        return d

    d.method = "manual"
    d.confidence = 1.0
    d.target_concept_id = new_id
    if not status_changed:
        d.status = "APPROVED"
    if new_id == 0:
        d.target_concept_name = d.target_domain_id = d.target_vocabulary_id = None
        d.equivalence = "UNMATCHED"
        return d
    known = next((c for c in d.candidates if c.concept_id == new_id), None)
    if conn is not None and schemas is not None:
        known, why = check_concept(conn, schemas, new_id, accepted_domains(d))
        if known is None:
            d.status = "FLAGGED"
            d.rationale = f"Reviewer's target_concept_id rejected: {why}."
    if known is not None:
        d.target_concept_name, d.target_domain_id, d.target_vocabulary_id = (
            known.concept_name,
            known.domain_id,
            known.vocabulary_id,
        )
    else:
        d.target_concept_name = row.get("target_concept_name") or None
        d.target_domain_id = row.get("target_domain_id") or None
        d.target_vocabulary_id = row.get("target_vocabulary_id") or None
    if d.equivalence == "UNMATCHED" and d.status != "FLAGGED":
        d.equivalence = "EQUIVALENT"
    return d


def import_review(
    ws: Workspace,
    conn: psycopg.Connection | None = None,
    schemas: Schemas | None = None,
    *,
    domains: Mapping[str, VocabularyDomains] | None = None,
) -> list[MappingDecision]:
    """``decisions.json`` with the reviewer's CSV edits applied (matched on vocabulary + code).

    ``domains`` (``vocab.mapper.source_vocabulary_domains`` of the current spec) first refreshes
    each decision's ``target_domain`` and ``accepted_domains``, so the edits are checked against
    the domains the spec accepts now (the spec may have changed since the terms were mapped).

    * a changed ``status`` is taken as is;
    * a changed ``target_concept_id`` sets method ``manual``, confidence 1.0 and, unless the
      status was changed too, status APPROVED. With a connection, the concept is looked up to
      refresh its name, domain and vocabulary. An id that does not exist, is not standard and
      valid, or is outside the term's accepted domains (its target domain, plus the other
      domains of a table routed by domain) is FLAGGED.

    Only ``mappings/<source_vocabulary_id>.csv`` files are read; rows whose code is not in
    ``decisions.json`` are ignored, and an unknown status value FLAGs the term. Nothing is
    written back.
    """
    path = decisions_path(ws)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the mapping step first")
    decisions = ws.load_json_list(path, MappingDecision)
    if domains:
        decisions = [
            d.model_copy(update={"target_domain": vd.target_domain, "accepted_domains": list(vd.accepted)})
            if (vd := domains.get(d.source_vocabulary_id)) is not None else d
            for d in decisions
        ]
    edits: dict[tuple[str, str], dict[str, str]] = {}
    for svid in sorted({d.source_vocabulary_id for d in decisions}):
        csv_path = ws.mappings_dir / _csv_name(svid)  # only the files export_review wrote
        if csv_path.exists():
            for row in _read_csv(csv_path):
                if row.get("source_vocabulary_id", "") == svid:
                    edits[(svid, row.get("source_code", ""))] = row
    return [
        (
            _apply_edit(d, edits[(d.source_vocabulary_id, d.source_code)], conn, schemas)
            if (d.source_vocabulary_id, d.source_code) in edits
            else d
        )
        for d in decisions
    ]
