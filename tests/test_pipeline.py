"""End-to-end: the offline demo (recorded LLM answers) on PostgreSQL."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

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
    # the missing visit end dates no longer fail the load: the build gives them the start date and
    # reports it, and that report is what the agent repairs
    visit_end = next(r for r in repairs if r.target_field == "visit_end_date")
    assert visit_end.iteration == 1 and visit_end.trigger == "etlDefaultedValue.visit_occurrence.visit_end_date"
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
    # decisions of an older version (or of an earlier spec): no accepted domains of their own
    stale = json.loads(p.ws.decisions_json.read_text())
    for d in stale:
        d["accepted_domains"] = []
    p.ws.decisions_json.write_text(json.dumps(stale))
    decisions = p.map_vocabulary()  # re-run: keeps the manual decision, calls no LLM
    quinine = next(d for d in decisions if d.source_code == "Quinine inj 600mg")
    assert quinine.target_concept_id == 2000003010 and quinine.method == "manual"
    # the kept decisions take the domains the spec accepts now (drug_exposure is routed by domain)
    assert {tuple(d.accepted_domains) for d in decisions if d.source_vocabulary_id == "SRC_DRUG"} == {("Drug", "Device")}
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
    assert any(e["event"] == "review_gate" and e["stage"] == "design" and e["outcome"] == "stopped"
               for e in p.ws.read_audit())


@pytest.mark.postgres
def test_start_over_survives_a_privacy_stop(demo_project, monkeypatch):
    """A run that stops at the privacy review is continued by a plain run: what the operator asked
    for (start over) still happens, and the approved review leaves no unfinished stage behind."""
    import omop_agent.privacy_review as privacy_review
    from omop_agent.approvals import record_approval, recorded_approver

    cfg = load_config(demo_project)
    Pipeline(cfg, log=lambda m: None).run_all()
    spec_path = cfg.workdir / "etl_spec.yaml"
    spec_path.write_text(spec_path.read_text() + "\n# reviewer note\n")
    annotated_at = (cfg.workdir / "annotations.json").stat().st_mtime_ns

    monkeypatch.setattr(privacy_review, "sends_to_llm", lambda c: True)  # as if the project used Claude
    p = Pipeline(cfg, log=lambda m: None)
    ws, approver = p.ws, recorded_approver(p.ws)

    def run(**kw):
        with pytest.raises(PipelineStopped) as e:
            Pipeline(cfg, approver=approver, log=lambda m: None).run_all(**kw)
        return e.value.stage

    assert run(fresh=True) == "privacy"
    assert json.loads(ws.pending_json.read_text()) == {"annotate": True, "redesign": True, "remap": True}
    record_approval(ws, "privacy", "rita")
    assert run() == "design"
    assert (cfg.workdir / "annotations.json").stat().st_mtime_ns != annotated_at  # interpreted again
    assert "# reviewer note" not in spec_path.read_text()  # redrafted...
    assert "# reviewer note" in (cfg.workdir / "etl_spec.1.yaml").read_text()  # ...after a backup
    assert json.loads(ws.pending_json.read_text()) == {"remap": True}
    record_approval(ws, "design", "rita")
    assert run() == "vocab_map"
    assert not ws.pending_json.exists()
    record_approval(ws, "vocab_map", "rita")
    report = Pipeline(cfg, approver=approver, log=lambda m: None).run_all()
    assert report.passed and ws.stage_status("privacy") == "done"
    assert "Verdict: PASSED" in ws.report_md.read_text()


def test_cli_init(tmp_path, monkeypatch):
    monkeypatch.delenv("PGUSER", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    result = CliRunner().invoke(app, ["init", str(tmp_path), "--name", "APHRC: Nairobi HDSS", "--source-type", "CSV"])
    assert result.exit_code == 0, result.output
    cfg = load_config(tmp_path / "omop_agent.yaml")
    assert cfg.name == "APHRC: Nairobi HDSS" and cfg.source.type == "csv"
    assert cfg.target.url == "postgresql://postgres:@localhost:5432/omop"
    assert cfg.llm.model == "claude-opus-5-5" and cfg.privacy.min_cell_count == 5
    assert (cfg.etl.timezone, cfg.etl.partial_dates, cfg.source.na_values) == (None, "widest", None)
    # the commented settings work once uncommented
    text = (tmp_path / "omop_agent.yaml").read_text()
    assert "# timezone: Africa/Nairobi" in text and "# source_release_date: 2024-06-30" in text
    on = (text.replace("# timezone:", "timezone:").replace("# na_values:", "na_values:")
          .replace("# source_release_date:", "source_release_date:"))
    (tmp_path / "on.yaml").write_text(on)
    cfg = load_config(tmp_path / "on.yaml")
    assert cfg.etl.timezone == "Africa/Nairobi" and cfg.source.na_values == ["NA", "NULL", "\\N", "."]
    assert str(cfg.cdm_source.source_release_date) == "2024-06-30"
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
    # back to the design bug: outpatient visits have no end date again, so the build gives them their
    # start date and fails etlDefaultedValue, one more repairable failure than the build before
    data["tool_calls"].append({"name": "update_field_mapping", "input": {
        "target_table": "visit_occurrence", "target_field": "visit_end_date", "kind": "expression",
        "expression": "v.discharge_date::date", "rationale": "a change that leaves most visits without an end date",
    }})
    it3.write_text(json.dumps(data))
    cfg = load_config(demo_project)
    cfg.llm.replay_dir = replay
    report = Pipeline(cfg, log=lambda m: None).run_all()
    assert report.passed
    events = json.loads("[" + ",".join((cfg.workdir / "audit.jsonl").read_text().split("\n")[:-1]) + "]")
    assert "repair_reverted" in [e["event"] for e in events]
    assert not any(e["event"] == "etl_run" and not e["ok"] for e in events)  # every build loaded the visits
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


FAKE_R = Path(__file__).parent / "fixtures" / "fake_r" / "Rscript"


@pytest.mark.postgres
def test_official_dqd_runs_once_after_repairs_and_failures_are_visible(demo_project, tmp_path, monkeypatch):
    """The R bridge, end to end with a stand-in Rscript: results are merged without duplicates,
    R runs once (after the repair loop), and a failed R run is reported, or fails the gate when
    quality.require_r_dqd is set."""
    calls = tmp_path / "r_calls.txt"
    monkeypatch.setenv("FAKE_R_CALLS", str(calls))
    cfg = load_config(demo_project)
    cfg.quality.use_r_dqd = True
    cfg.quality.rscript = str(FAKE_R)
    logs: list[str] = []
    p = Pipeline(cfg, log=logs.append)
    report = p.run_all(fresh=True)
    assert report.passed and report.engine == "builtin+dqd"
    assert calls.read_text().count("run") == 1  # the demo's repair loop builds 3 times; R ran once
    keys = [r.key for r in report.results]
    assert len(keys) == len(set(keys))
    gender = next(r for r in report.results if r.check_name == "plausibleGenderUseDescendants")
    assert gender.concept_id == 4090861
    assert any("R DataQualityDashboard: 3 checks" in m for m in logs)
    assert "builtin+dqd" in p.ws.report_md.read_text()

    monkeypatch.setenv("FAKE_R_MODE", "fail")
    logs.clear()
    p = Pipeline(cfg, log=logs.append)
    report = p.check()
    assert report.passed and report.engine == "builtin"
    assert any("✗ R Achilles/DataQualityDashboard failed" in m and "SIGPIPE" in m for m in logs)

    cfg.quality.require_r_dqd = True
    p = Pipeline(cfg, log=logs.append)
    report = p.check()
    assert not report.passed and any(b.startswith("officialDqdRun") for b in report.blocking_failures)

    monkeypatch.setenv("FAKE_R_MODE", "fatal")
    cfg.quality.require_r_dqd = False
    report = Pipeline(cfg, log=logs.append).check()
    assert not report.passed and any(b.startswith("cdmField.person.gender_concept_id") for b in report.blocking_failures)
    p.close()


def test_upgrade_notice_is_given_once(tmp_path, monkeypatch):
    """The first build of a project last built by an older version (its steps carry no kind) says
    once, in the log and the audit trail, what builds now do differently."""
    import yaml

    from omop_agent.config import ProjectConfig
    from omop_agent.etl import runner
    from omop_agent.models import ETLRunResult, TableLoadResult
    from omop_agent.workspace import Workspace

    cfg = ProjectConfig.model_validate({
        "name": "t", "workdir": str(tmp_path / "work"), "source": {"type": "csv", "path": str(tmp_path)},
        "target": {"url": "postgresql://unused"}, "llm": {"provider": "none"}})
    ws = Workspace(cfg.workdir)
    assert runner.upgrade_notice(ws) == []  # never built: nothing to announce
    old = ETLRunResult(started_at="2025-01-01T00:00:00Z",
                       steps=[TableLoadResult(target_table="person", step="person", rows_inserted=3)])
    ws.save_json(ws.etl_run_json, old)
    assert runner.upgrade_notice(ws) == list(runner.UPGRADE_BEHAVIOURS)

    def fake_run_etl(conn, schemas, spec, cfg, ws):
        new = ETLRunResult(started_at="2026-01-01T00:00:00Z", steps=[
            TableLoadResult(target_table="person", step="03_person", kind="insert", rows_inserted=3),
            TableLoadResult(target_table="drug_era", step="19_drug_era", kind="derived", blocking=False,
                            error="permission denied for table concept_ancestor")])
        ws.save_json(ws.etl_run_json, new)
        return new

    monkeypatch.setattr(runner, "run_etl", fake_run_etl)
    monkeypatch.setattr(Pipeline, "conn", property(lambda self: None))
    spec = ETLSpec.model_validate(yaml.safe_load((Path(__file__).parent / "fixtures" / "demo_etl_spec.yaml").read_text()))
    logs: list[str] = []
    p = Pipeline(cfg, log=logs.append)
    # `build --no-repair` with a spec the validator refuses builds nothing: the notice is kept for the
    # build that runs
    from omop_agent.pipeline import SpecInvalid

    def refused(self):
        raise SpecInvalid("etl_spec.yaml has 1 error(s)")

    monkeypatch.setattr(Pipeline, "validated_spec", refused)
    with pytest.raises(SpecInvalid):
        p.build()
    assert not [e for e in ws.read_audit() if e["event"] == "upgrade_notice"]
    assert not any("the first build of this project with the new version" in m for m in logs)
    assert runner.upgrade_notice(ws) == list(runner.UPGRADE_BEHAVIOURS)
    p.build(spec)
    notices = [e for e in ws.read_audit() if e["event"] == "upgrade_notice"]
    assert len(notices) == 1 and notices[0]["behaviours"] == list(runner.UPGRADE_BEHAVIOURS)
    assert notices[0]["notice"] == runner.UPGRADE_NOTICE
    assert sum("the first build of this project with the new version" in m for m in logs) == 1
    assert any(m.endswith("1 errors (1 in steps that do not block the build)") for m in logs)
    assert "  ✗ 19_drug_era (not blocking): permission denied for table concept_ancestor" in logs
    # a build by the new version is not an upgrade, and the notice is given once even when the old
    # result comes back (a build that failed before writing its own)
    assert runner.upgrade_notice(ws) == []
    ws.save_json(ws.etl_run_json, old)
    assert runner.upgrade_notice(ws) == []
    p.build(spec)
    assert len([e for e in ws.read_audit() if e["event"] == "upgrade_notice"]) == 1
