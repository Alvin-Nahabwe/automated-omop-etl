"""Recorded reviewer approvals for the pipeline's review gates.

A reviewer approves what a gate shows: what will be sent to Claude (``privacy``, see
privacy_review.py), the ETL design (``design``) or the code mappings (``vocab_map``), in the web
app, at the terminal during ``omop-agent run``, or with ``omop-agent approve``. The approval is stored in
``<workdir>/approvals.json`` together with a digest of what was approved, and a copy of it is
kept under ``<workdir>/approvals/`` so a later change can be shown as a diff.

``recorded_approver`` lets a non-interactive run pass a gate only while that digest still
matches. A spec that changed since it was approved (hand edit, redraft, the repair agent) or
mappings that changed (an edit, new source codes) need a fresh review. The design digest
compares meaning, not formatting, and leaves out spec fields added later while they hold their
default (``models.DIGEST_IGNORED_DEFAULTS``), so upgrading the tool does not void approvals.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from .fsutil import atomic_write_text, file_lock
from .models import DIGEST_IGNORED_DEFAULTS, ETLSpec
from .workspace import Workspace, utcnow

GATES = {
    "privacy": "Privacy review",
    "design": "ETL design",
    "vocab_map": "Code mappings",
}
# The same, for use inside a sentence.
GATE_PHRASES = {
    "privacy": "data that will be sent to Claude",
    "design": "ETL design",
    "vocab_map": "code mappings",
}
APPROVALS_FILE = "approvals.json"


class Approval(BaseModel):
    stage: str
    by: str
    at: datetime
    digest: str
    note: str = ""


class GateState(BaseModel):
    stage: str
    ready: bool  # the artifact exists
    digest: str | None = None  # of the artifact as it is now
    approval: Approval | None = None
    current: bool = False  # approved, and unchanged since

    @property
    def label(self) -> str:
        if not self.ready:
            return "not ready"
        if self.current:
            return "approved"
        return "changed since approval" if self.approval else "awaiting review"


def _path(ws: Workspace) -> Path:
    return ws.root / APPROVALS_FILE


def _snapshot_path(ws: Workspace, stage: str) -> Path:
    return ws.root / "approvals" / (f"{stage}.yaml" if stage == "design" else f"{stage}.json")


def _privacy_payload(ws: Workspace) -> Any | None:
    from .privacy_review import digest_payload, review_path

    path = review_path(ws)
    return digest_payload(json.loads(path.read_text())) if path.exists() else None


def _check_stage(stage: str) -> None:
    if stage not in GATES:
        raise ValueError(f"unknown review gate '{stage}'; use one of: {', '.join(GATES)}")


def _spec_payload_of(text: str) -> Any:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return {"unparsable_yaml": text}  # still has a digest: a broken file is "changed"
    try:  # compare meaning, not formatting: defaults and key order do not matter
        payload = ETLSpec.model_validate(raw).model_dump(mode="json")
    except ValidationError:
        return raw
    return _without_new_defaults(payload)


def without_default_lines(text: str) -> str:
    """Spec YAML ``text`` without the lines that only write a DIGEST_IGNORED_DEFAULTS field at its
    default (``route_by_domain: true``, ``dedup_order: null``). A spec approved before those fields
    existed gains them the first time this version saves it; a diff of the two texts should show
    the changes someone made, not those lines (they change no digest either)."""
    import re

    def literal(value: Any) -> str:
        return {True: "true", False: "false", None: "(?:null|~)?"}.get(value, re.escape(str(value)))

    names = {**DIGEST_IGNORED_DEFAULTS["spec"], **DIGEST_IGNORED_DEFAULTS["table_mapping"]}
    pattern = re.compile(r"^\s*(?:" + "|".join(f"{re.escape(n)}:\\s*{literal(v)}" for n, v in names.items())
                         + r")\s*$")
    return "\n".join(line for line in text.splitlines() if not pattern.match(line))


def _without_new_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """``payload`` without the fields of DIGEST_IGNORED_DEFAULTS that hold their default: a spec
    approved before those fields existed keeps its digest; setting one changes it."""

    def drop(d: dict[str, Any], defaults: dict[str, Any]) -> None:
        for name, default in defaults.items():
            if name in d and d[name] == default:
                del d[name]

    drop(payload, DIGEST_IGNORED_DEFAULTS["spec"])
    for tm in payload.get("table_mappings") or []:
        drop(tm, DIGEST_IGNORED_DEFAULTS["table_mapping"])
    return payload


def _spec_payload(ws: Workspace) -> Any | None:
    if not ws.spec_yaml.exists():
        return None
    return _spec_payload_of(ws.spec_yaml.read_text())


def _mapping_rows(ws: Workspace) -> list[list[Any]] | None:
    from .vocab.review import import_review

    if not ws.decisions_json.exists():
        return None
    return sorted(
        [d.source_vocabulary_id, d.source_code, d.target_concept_id, d.status] for d in import_review(ws)
    )


def _payload(ws: Workspace, stage: str) -> Any | None:
    _check_stage(stage)
    if stage == "privacy":
        return _privacy_payload(ws)
    return _spec_payload(ws) if stage == "design" else _mapping_rows(ws)


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def artifact_digest(ws: Workspace, stage: str) -> str | None:
    """Digest of what the gate reviews now, or None when that artifact does not exist yet."""
    payload = _payload(ws, stage)
    return None if payload is None else _digest(payload)


def load_approvals(ws: Workspace) -> dict[str, Approval]:
    path = _path(ws)
    if not path.exists():
        return {}
    data = json.loads(path.read_text() or "{}")
    return {k: Approval.model_validate(v) for k, v in data.items() if k in GATES}


def gate_state(ws: Workspace, stage: str) -> GateState:
    digest = artifact_digest(ws, stage)
    approval = load_approvals(ws).get(stage)
    return GateState(
        stage=stage,
        ready=digest is not None,
        digest=digest,
        approval=approval,
        current=digest is not None and approval is not None and approval.digest == digest,
    )


class ApprovalConflict(RuntimeError):
    """What would be approved is not what the reviewer saw."""


def record_approval(
    ws: Workspace, stage: str, by: str, note: str = "", expected_digest: str | None = None
) -> Approval:
    """Approve the gate's current artifact as ``by``.

    With ``expected_digest`` (what the reviewer's page showed), raises ApprovalConflict if the
    artifact has changed since. The digest and the kept copy come from one read of the
    artifact, under the approvals lock. Raises FileNotFoundError if there is nothing to approve.
    """
    by = by.strip()
    if not by:
        raise ValueError("an approval needs the reviewer's name")
    _check_stage(stage)
    with file_lock(ws.root / ".approvals.lock"):
        if stage == "design":
            text = ws.spec_yaml.read_text() if ws.spec_yaml.exists() else None
            payload = None if text is None else _spec_payload_of(text)
        else:
            payload = _privacy_payload(ws) if stage == "privacy" else _mapping_rows(ws)
            text = None if payload is None else json.dumps(payload, indent=1, sort_keys=True) + "\n"
        if payload is None or text is None:
            what = {
                "privacy": "the privacy review (profile the source first)",
                "design": "etl_spec.yaml",
                "vocab_map": "mappings/decisions.json",
            }[stage]
            raise FileNotFoundError(f"nothing to approve yet: {what} does not exist")
        digest = _digest(payload)
        if expected_digest is not None and expected_digest != digest:
            raise ApprovalConflict(f"the {GATE_PHRASES[stage]} changed after the page was loaded")
        approval = Approval(stage=stage, by=by, at=utcnow(), digest=digest, note=note.strip())
        snap = _snapshot_path(ws, stage)
        snap.parent.mkdir(exist_ok=True)
        atomic_write_text(snap, text)
        data = {k: v.model_dump(mode="json") for k, v in load_approvals(ws).items()}
        data[stage] = approval.model_dump(mode="json")
        atomic_write_text(_path(ws), json.dumps(data, indent=2) + "\n")
    ws.audit("review_approved", stage=stage, by=approval.by, digest=approval.digest, note=approval.note)
    return approval


def approved_snapshot(ws: Workspace, stage: str) -> str | None:
    """The text of what was last approved (YAML for the design, JSON rows for mappings)."""
    _check_stage(stage)
    snap = _snapshot_path(ws, stage)
    return snap.read_text() if snap.exists() else None


def mapping_changes(ws: Workspace) -> list[dict[str, Any]]:
    """Mappings added, removed or changed since the last approval of ``vocab_map``."""
    snap = approved_snapshot(ws, "vocab_map")
    now = _mapping_rows(ws) or []
    before = {(r[0], r[1]): r for r in json.loads(snap)} if snap else {}
    after = {(r[0], r[1]): r for r in now}
    out: list[dict[str, Any]] = []
    for key in sorted(before.keys() | after.keys()):
        b, a = before.get(key), after.get(key)
        if b == a:
            continue
        out.append(
            {
                "source_vocabulary_id": key[0],
                "source_code": key[1],
                "before": None if b is None else {"target_concept_id": b[2], "status": b[3]},
                "after": None if a is None else {"target_concept_id": a[2], "status": a[3]},
            }
        )
    return out


def recorded_approver(ws: Workspace):
    """An approver for ``Pipeline`` that passes a gate only with a current recorded approval."""

    def approve(stage: str, message: str, paths: list[str]) -> bool:
        if stage not in GATES:
            return True
        return gate_state(ws, stage).current

    approve.mode = "recorded"  # type: ignore[attr-defined]
    return approve
