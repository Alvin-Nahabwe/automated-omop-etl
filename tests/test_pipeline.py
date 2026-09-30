"""End-to-end: the offline demo (recorded LLM answers) on PostgreSQL."""

from __future__ import annotations

import csv
import json
import os

import pytest
from typer.testing import CliRunner

from omop_agent.cli import app
from omop_agent.config import load_config
from omop_agent.db import Schemas, connect, execute, qi, scalar
from omop_agent.demo import create_demo_project
from omop_agent.models import ETLSpec, RepairAction
from omop_agent.pipeline import Pipeline, PipelineStopped

DB_URL = os.environ.get("OMOP_AGENT_TEST_DB_URL")


def _drop(prefix: str) -> None:
    conn = connect(DB_URL, autocommit=True)  # type: ignore[arg-type]
    for s in ("cdm", "vocab", "src", "etl", "results"):
        execute(conn, f"DROP SCHEMA IF EXISTS {qi(f'{prefix}_{s}')} CASCADE")
    conn.close()


@pytest.fixture
def demo_project(tmp_path):
    import uuid

    prefix = f"t{uuid.uuid4().hex[:8]}demo"
    cfg_path = create_demo_project(tmp_path / "demo", DB_URL, provider="replay", schema_prefix=prefix)  # type: ignore[arg-type]
    yield cfg_path
    _drop(prefix)


@pytest.mark.postgres
def test_demo_end_to_end(demo_project):
    cfg = load_config(demo_project)
    logs: list[str] = []
    report = Pipeline(cfg, log=logs.append).run_all()
    ws_root = cfg.workdir

    assert report.passed, report.blocking_failures
    # the repair loop fixed both deliberate design bugs
    repairs = [RepairAction.model_validate(r) for r in json.loads((ws_root / "repairs.json").read_text())]
    fields = {(r.target_table, r.target_field) for r in repairs}
    assert ("visit_occurrence", "visit_end_date") in fields
    assert ("condition_occurrence", "condition_concept_id") in fields
    spec = ETLSpec.model_validate(__import__("yaml").safe_load((ws_root / "etl_spec.yaml").read_text()))
    assert "COALESCE" in next(f.expression for f in spec.mapping_for("visit_occurrence").fields if f.target_field == "visit_end_date")

    schemas = Schemas.from_config(cfg.target)
    conn = connect(DB_URL)  # type: ignore[arg-type]
    try:
        assert scalar(conn, f"SELECT COUNT(*) FROM {schemas.cdm}.person") == 1500
        unmapped = scalar(conn, f"SELECT COUNT(*) FROM {schemas.cdm}.condition_occurrence WHERE condition_concept_id = 0")
        assert unmapped == 0
        # rare drug names were never sent to the LLM and stay unmapped for human review
        mapped_quinine = scalar(
            conn,
            f"SELECT COUNT(*) FROM {schemas.cdm}.drug_exposure WHERE drug_source_value LIKE 'Quinine%' AND drug_concept_id <> 0",
        )
        assert mapped_quinine == 0
    finally:
        conn.close()
    prompts = " ".join(p.read_text() for p in (ws_root / "llm_calls").glob("*.json"))
    # rare source values and identifiers never reach a prompt (vocabulary concept names such as
    # "quinine dihydrochloride ..." may appear as mapping candidates; that is not source data)
    for secret in ("Quinine inj 600mg", "Ceftriaxone 1g inj", "Achieng", "Odhiambo", "KH000001"):
        assert secret not in prompts
    assert (ws_root / "report.html").exists() and "PASSED" in (ws_root / "report.md").read_text()


@pytest.mark.postgres
def test_remap_keeps_review_edits(demo_project):
    cfg = load_config(demo_project)
    p = Pipeline(cfg, log=lambda m: None)
    p.stage_and_profile()
    p.load_vocabulary()
    p.design()
    p.map_vocabulary()
    csv_path = p.ws.mappings_dir / "SRC_DRUG.csv"
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows, header = list(reader), reader.fieldnames
    for row in rows:  # the reviewer maps the rare quinine product by hand
        if row["source_code"] == "Quinine inj 600mg":
            row["target_concept_id"] = "2000003010"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    decisions = p.map_vocabulary()  # re-run: keeps the manual decision, calls no LLM
    quinine = next(d for d in decisions if d.source_code == "Quinine inj 600mg")
    assert quinine.target_concept_id == 2000003010 and quinine.method == "manual"
    p.close()


@pytest.mark.postgres
def test_review_gate_stops(demo_project):
    cfg = load_config(demo_project)
    p = Pipeline(cfg, approver=lambda stage, msg, paths: stage != "design", log=lambda m: None)
    with pytest.raises(PipelineStopped):
        p.run_all()
    assert p.ws.stage_status("design") == "awaiting_review"
    assert any(e["event"] == "human_approval" and not e["approved"] for e in p.ws.read_audit())


def test_cli_init(tmp_path):
    result = CliRunner().invoke(app, ["init", str(tmp_path), "--name", "kenya-hdss"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "omop_agent.yaml").read_text()
    assert "claude-opus-5-5" in text and "min_cell_count: 5" in text
