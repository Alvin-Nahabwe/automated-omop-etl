"""Recorded approvals for the review gates, and reviewer fields on the mapping review files."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest
import yaml

from omop_agent.approvals import (
    approved_snapshot,
    artifact_digest,
    gate_state,
    mapping_changes,
    record_approval,
    recorded_approver,
)
from omop_agent.models import ConceptCandidate, MappingDecision
from omop_agent.vocab.review import export_review, import_review, update_review_row
from omop_agent.workspace import Workspace

SPEC = Path(__file__).parent / "fixtures" / "demo_etl_spec.yaml"


def _candidate(cid: int, name: str) -> ConceptCandidate:
    return ConceptCandidate(
        concept_id=cid, concept_name=name, domain_id="Drug", vocabulary_id="RxNorm",
        concept_class_id="Clinical Drug", standard_concept="S", concept_code=str(cid),
    )


def decisions() -> list[MappingDecision]:
    return [
        MappingDecision(
            source_vocabulary_id="SRC_DRUG", source_code="Paracetamol 500mg", source_name="Paracetamol 500mg",
            frequency=900, target_domain="Drug", target_concept_id=11, target_concept_name="acetaminophen 500 MG",
            equivalence="EQUAL", confidence=0.85, status="UNCHECKED", method="llm",
            candidates=[_candidate(11, "acetaminophen 500 MG"), _candidate(12, "acetaminophen 500 MG Oral Tablet")],
        ),
        MappingDecision(
            source_vocabulary_id="SRC_DRUG", source_code="TLD (TDF/3TC/DTG)", source_name="TLD (TDF/3TC/DTG)",
            frequency=287, target_domain="Drug", status="UNCHECKED", method="llm",
        ),
    ]


@pytest.fixture
def ws(tmp_path) -> Workspace:
    ws = Workspace(tmp_path / "work")
    shutil.copy(SPEC, ws.spec_yaml)
    export_review(ws, decisions())
    return ws


def test_gates_start_unapproved_and_not_ready_without_artifacts(tmp_path):
    empty = Workspace(tmp_path / "empty")
    assert gate_state(empty, "design").label == "not ready"
    assert gate_state(empty, "vocab_map").label == "not ready"
    with pytest.raises(FileNotFoundError):
        record_approval(empty, "design", "alice")
    with pytest.raises(ValueError):
        gate_state(empty, "build")


def test_design_approval_holds_until_the_spec_changes(ws):
    approve = recorded_approver(ws)
    assert not approve("design", "", [])
    assert approve("build", "", [])  # not a review gate

    a = record_approval(ws, "design", "alice", note="looks right")
    assert a.by == "alice" and a.note == "looks right"
    assert gate_state(ws, "design").label == "approved"
    assert approve("design", "", [])
    assert approved_snapshot(ws, "design") == ws.spec_yaml.read_text()

    # Re-saving the same spec with different formatting is not a change ...
    data = yaml.safe_load(ws.spec_yaml.read_text())
    ws.spec_yaml.write_text("# reformatted\n" + yaml.safe_dump(data, sort_keys=True))
    assert approve("design", "", [])
    # ... but changing what it says is.
    data["assumptions"] = data.get("assumptions", []) + ["a hand edit"]
    ws.spec_yaml.write_text(yaml.safe_dump(data))
    assert gate_state(ws, "design").label == "changed since approval"
    assert not approve("design", "", [])

    events = [e for e in ws.read_audit() if e["event"] == "review_approved"]
    assert events[-1]["by"] == "alice" and events[-1]["stage"] == "design"


def test_design_digest_ignores_fields_added_at_their_default():
    """New ETLSpec / TableMapping fields must not void the approvals of specs written before them:
    register each one in models.DIGEST_IGNORED_DEFAULTS. This pins the reference spec's digest."""
    from omop_agent.approvals import _digest, _spec_payload_of

    text = SPEC.read_text()
    assert _digest(_spec_payload_of(text))[:16] == "c89d2809a0188caa"
    data = yaml.safe_load(text)
    assert _digest(_spec_payload_of(yaml.safe_dump({**data, "derive_eras": True})))[:16] == "c89d2809a0188caa"
    # a field that is set to something else changes what was approved
    assert _digest(_spec_payload_of(text + "derive_eras: false\n"))[:16] != "c89d2809a0188caa"
    assert _digest(_spec_payload_of(text + "route_by_domain: false\n"))[:16] != "c89d2809a0188caa"
    data["table_mappings"][-1]["dedup_order"] = "x.dod::date DESC"
    assert _digest(_spec_payload_of(yaml.safe_dump(data)))[:16] != "c89d2809a0188caa"


def test_mapping_approval_tracks_targets_and_statuses(ws):
    record_approval(ws, "vocab_map", "bob")
    digest = artifact_digest(ws, "vocab_map")
    assert gate_state(ws, "vocab_map").current
    assert mapping_changes(ws) == []

    # A reviewer's note alone does not change what the ETL loads.
    update_review_row(ws, "SRC_DRUG", "TLD (TDF/3TC/DTG)", {"review_note": "ask the pharmacy"})
    assert artifact_digest(ws, "vocab_map") == digest

    update_review_row(ws, "SRC_DRUG", "Paracetamol 500mg", {"target_concept_id": "12", "reviewed_by": "carol"})
    assert not gate_state(ws, "vocab_map").current
    [change] = mapping_changes(ws)
    assert change["source_code"] == "Paracetamol 500mg"
    assert change["before"] == {"target_concept_id": 11, "status": "UNCHECKED"}
    assert change["after"] == {"target_concept_id": 12, "status": "APPROVED"}


def test_reviewer_fields_survive_a_re_export(ws):
    update_review_row(
        ws, "SRC_DRUG", "Paracetamol 500mg",
        {"status": "APPROVED", "reviewed_by": "alice", "reviewed_at": "2026-10-01T09:00:00+00:00", "review_note": "ok"},
    )
    merged = import_review(ws)
    para = next(d for d in merged if d.source_code == "Paracetamol 500mg")
    assert (para.status, para.reviewed_by, para.reviewed_at, para.review_note) == (
        "APPROVED", "alice", "2026-10-01T09:00:00+00:00", "ok",
    )
    export_review(ws, merged)  # what the next pipeline run does
    with (ws.mappings_dir / "SRC_DRUG.csv").open(newline="") as fh:
        row = next(r for r in csv.DictReader(fh) if r["source_code"] == "Paracetamol 500mg")
    assert (row["status"], row["reviewed_by"], row["review_note"]) == ("APPROVED", "alice", "ok")
    para = next(d for d in import_review(ws) if d.source_code == "Paracetamol 500mg")
    assert para.reviewed_by == "alice"


def test_update_review_row_rejects_unknown_terms_and_columns(ws):
    with pytest.raises(KeyError):
        update_review_row(ws, "SRC_DRUG", "Aspirin", {"status": "APPROVED"})
    with pytest.raises(ValueError):
        update_review_row(ws, "SRC_DRUG", "Paracetamol 500mg", {"frequency_x": "1"})


def test_update_review_row_rebuilds_a_missing_csv(ws):
    (ws.mappings_dir / "SRC_DRUG.csv").unlink()
    update_review_row(ws, "SRC_DRUG", "TLD (TDF/3TC/DTG)", {"status": "FLAGGED"})
    tld = next(d for d in import_review(ws) if d.source_code.startswith("TLD"))
    assert tld.status == "FLAGGED"
    para = next(d for d in import_review(ws) if d.source_code.startswith("Paracetamol"))
    assert para.status == "UNCHECKED"
