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
from omop_agent.pipeline import Pipeline, PipelineStopped, SpecInvalid

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
def test_rerun_keeps_edited_spec_and_redesign_backs_it_up(demo_project):
    cfg = load_config(demo_project)
    Pipeline(cfg, log=lambda m: None).run_all()
    spec_path = cfg.workdir / "etl_spec.yaml"
    spec_path.write_text(spec_path.read_text() + "\n# reviewer note\n")
    llm_calls = len(list((cfg.workdir / "llm_calls").glob("*.json")))
    report = Pipeline(cfg, log=lambda m: None).run_all()
    assert report.passed
    assert "# reviewer note" in spec_path.read_text()  # not redrafted
    new_calls = sorted((cfg.workdir / "llm_calls").glob("*.json"))[llm_calls:]
    assert not [p for p in new_calls if "design" in p.name or "annotate" in p.name or "vocab_map" in p.name]
    Pipeline(cfg, log=lambda m: None).run_all(redesign=True)
    assert "# reviewer note" in (cfg.workdir / "etl_spec.1.yaml").read_text()


@pytest.mark.postgres
def test_review_gate_stops(demo_project):
    cfg = load_config(demo_project)
    p = Pipeline(cfg, approver=lambda stage, msg, paths: stage != "design", log=lambda m: None)
    with pytest.raises(PipelineStopped):
        p.run_all()
    assert p.ws.stage_status("design") == "awaiting_review"
    assert any(e["event"] == "human_approval" and not e["approved"] for e in p.ws.read_audit())


def test_cli_init(tmp_path, monkeypatch):
    monkeypatch.delenv("PGUSER", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    result = CliRunner().invoke(app, ["init", str(tmp_path), "--name", "APHRC: Nairobi HDSS", "--source-type", "CSV"])
    assert result.exit_code == 0, result.output
    cfg = load_config(tmp_path / "omop_agent.yaml")
    assert cfg.name == "APHRC: Nairobi HDSS" and cfg.source.type == "csv"
    assert cfg.target.url == "postgresql://postgres:@localhost:5432/omop"
    assert cfg.llm.model == "claude-opus-5-5" and cfg.privacy.min_cell_count == 5
    bad = CliRunner().invoke(app, ["init", str(tmp_path / "x"), "--source-type", "excel"])
    assert bad.exit_code != 0


@pytest.mark.postgres
def test_harmful_repair_is_reverted(demo_project, tmp_path):
    import shutil

    from omop_agent.demo import FIXTURE_REPLAY

    replay = tmp_path / "replay"
    shutil.copytree(FIXTURE_REPLAY, replay)
    it3 = replay / "repair" / "iteration_3.json"
    data = json.loads(it3.read_text())
    data["tool_calls"].append({"name": "update_field_mapping", "input": {
        "target_table": "visit_occurrence", "target_field": "visit_end_date", "kind": "expression",
        "expression": "v.discharge_date::date", "rationale": "a change that compiles but fails at runtime",
    }})
    it3.write_text(json.dumps(data))
    cfg = load_config(demo_project)
    cfg.llm.replay_dir = replay
    report = Pipeline(cfg, log=lambda m: None).run_all()
    assert report.passed
    events = [e["event"] for e in json.loads("[" + ",".join((cfg.workdir / "audit.jsonl").read_text().split("\n")[:-1]) + "]")]
    assert "repair_reverted" in events
    spec = ETLSpec.model_validate(__import__("yaml").safe_load((cfg.workdir / "etl_spec.yaml").read_text()))
    end = next(f for f in spec.mapping_for("visit_occurrence").fields if f.target_field == "visit_end_date")
    assert "COALESCE" in end.expression


@pytest.mark.postgres
def test_unsafe_hand_edit_is_refused_at_build(demo_project):
    import yaml

    cfg = load_config(demo_project)
    p = Pipeline(cfg, log=lambda m: None)
    p.run_all()
    spec = yaml.safe_load((cfg.workdir / "etl_spec.yaml").read_text())
    person = next(m for m in spec["table_mappings"] if m["target_table"] == "person")
    person["fields"].append({"target_field": "person_source_value", "kind": "expression", "expression": "i.first_name"})
    person["fields"] = [f for f in person["fields"] if f["target_field"] != "person_source_value" or f["expression"] == "i.first_name"]
    (cfg.workdir / "etl_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    p2 = Pipeline(cfg, log=lambda m: None)
    with pytest.raises(SpecInvalid, match="first_name"):
        p2.build_check_repair()
    p2.close()
