"""Eras: the condition_era and drug_era steps, their place in the build, the OHDSI era logic (hand
cases, ties, the published scripts), the checks, and the repair, design, history and report rules
around them."""

from __future__ import annotations

import random
import re
from datetime import date, datetime, timedelta, timezone

import pytest
import yaml

from omop_agent.cdm.ddl import create_cdm_tables, create_etl_tables
from omop_agent.cdm.spec import DERIVED_ERAS, LOAD_ORDER
from omop_agent.db import Schemas, execute, fetch_all, scalar
from omop_agent.design.designer import _normalise_plan, _plan_prompt
from omop_agent.design.render import eras_text, render_spec_md
from omop_agent.design.validate import validate_spec
from omop_agent.etl.compiler import (
    SPEC_KINDS,
    CompiledStep,
    compile_spec,
    derived_era_tables,
    era_status,
    spec_era_tables,
    target_tables,
)
from omop_agent.etl.runner import _run_step, run_etl
from omop_agent.models import (
    CheckResult,
    CustomSQL,
    DQReport,
    ETLRunResult,
    FieldMapping,
    KeySpec,
    PlannedTable,
    TableLoadResult,
    TableMapping,
    TablePlan,
)
from omop_agent.pipeline import _BuildHistory
from omop_agent.quality.checks import (
    _populated_tables,
    derived_tables,
    etl_checks,
    gate,
    repairable,
    run_builtin_checks,
)
from omop_agent.quality.repair import (
    REPAIR_SYSTEM,
    RepairState,
    build_repair_prompt,
    build_repair_tools,
    era_issues,
    repair_triggers,
)
from omop_agent.report.inspection import (
    _design_section,
    _dq_section,
    _etl_section,
    collect_report_data,
    render_html,
    render_markdown,
)
from omop_agent.workspace import Workspace
from tests.conftest import FIXTURES
from tests.test_etl import S, check_step_invariants, count, fixed_ref, load_ref, step
from tests.test_quality import plain_config
from tests.test_report import full_ws  # noqa: F401 (fixture)

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
D = date.fromisoformat


def labels(steps) -> list[str]:
    return [s.label for s in steps]


# --------------------------------------------------------------------------------------
# the era steps in the compiled build (no database)
# --------------------------------------------------------------------------------------


def test_step_order_and_targets():
    spec = load_ref()
    steps = compile_spec(spec, S)
    # plan section 2: the eras after routing and the observation periods, before cdm_source
    assert [s.name for s in steps][-6:] == [
        "15_observation_period", "16_observation_period_merge", "17_observation_period_fill", "18_condition_era",
        "19_drug_era", "20_cdm_source"]
    for era, source in DERIVED_ERAS.items():
        s = step(steps, era)
        assert (s.kind, s.target_table, s.blocking, s.requires, s.returns) == ("derived", era, False, source, "rowcount")
        assert s.slot == (LOAD_ORDER.index(era), 0, 0) and s.kind not in SPEC_KINDS
        assert s.source_count_sql is None and s.retained_count_sql is None and s.count_sites == ()
        assert s.statements == (s.sql,) and s.sql.startswith(f"-- {era}: the OHDSI")
        # a unique tiebreaker in both windows, so they see one order
        assert "ORDER BY t.start_date, t.end_date, t.rid" in s.sql and "ORDER BY m.start_date, m.end_date, m.rid" in s.sql
        assert f"INSERT INTO cdm.{era} (" in s.sql and "{" not in s.sql
    ce, de = step(steps, "condition_era").sql, step(steps, "drug_era").sql
    assert "co.condition_occurrence_id AS rid" in ce and "WHERE co.condition_concept_id <> 0" in ce
    assert "ELSE co.condition_start_date + 1 END AS end_date" in ce and "m.prev_end + 30" in ce
    assert "gap_days" not in ce
    assert "d.drug_exposure_id AS rid" in de and "AND COALESCE(d.days_supply, 0) >= 0" in de
    assert "WHEN d.days_supply BETWEEN 1 AND 365 THEN d.drug_exposure_start_date + d.days_supply - 1" in de
    assert "ELSE d.drug_exposure_start_date END AS end_date" in de
    assert "c.vocabulary_id IN ('RxNorm', 'RxNorm Extension')" in de and "c.standard_concept = 'S'" in de
    assert "GREATEST(start_date - prev_end, 0)" in de and "e.gap_days::integer" in de
    assert target_tables(spec) == [
        "person", "visit_occurrence", "condition_occurrence", "drug_exposure", "measurement", "death",
        "observation_period", "condition_era", "drug_era", "cdm_source"]
    check_step_invariants(steps)
    assert derived_era_tables(spec) == ["condition_era", "drug_era"]
    assert era_status(spec) == {"condition_era": "derived", "drug_era": "derived"}


def test_derive_eras_off_and_overrides():
    spec = load_ref()
    off = spec.model_copy(update={"derive_eras": False})
    assert not [s for s in compile_spec(off, S) if s.target_table in DERIVED_ERAS]
    assert not set(DERIVED_ERAS) & set(target_tables(off)) and derived_era_tables(off) == []
    assert era_status(off) == {"condition_era": "off", "drug_era": "off"}

    # spec SQL that loads an era table replaces the built-in step for that table: by its target ...
    own = spec.model_copy(deep=True)
    own.custom_sql = [CustomSQL(name="my eras", target_table="condition_era", sql="SELECT 1")]
    steps = compile_spec(own, S)
    assert labels(steps)[-4:] == ["observation_period_fill", "custom_my_eras", "drug_era", "cdm_source"]
    assert era_status(own) == {"condition_era": "spec", "drug_era": "derived"}
    # ... or by an INSERT INTO {cdm}.<era> in custom SQL of another table (any case, quoted)
    for sql in ("INSERT INTO {cdm}.drug_era SELECT * FROM x", 'insert  into {cdm} . "drug_era" (drug_era_id) SELECT 1'):
        other = spec.model_copy(deep=True)
        other.custom_sql = [CustomSQL(name="drugs", target_table="drug_exposure", sql=sql)]
        assert spec_era_tables(other) == {"drug_era"}
        assert [s.label for s in compile_spec(other, S) if s.kind == "derived"][-1] == "condition_era"
        assert "drug_era" in target_tables(other)  # emptied before each build, checked
    # ... also under the schema's own name or unqualified, not in another schema
    for sql, found in (("INSERT INTO cdm.condition_era SELECT 1", {"condition_era"}),
                       ("insert into drug_era select 1", {"drug_era"}),
                       ("INSERT INTO {etl}.drug_era SELECT 1", set()),
                       ("INSERT INTO {cdm}.drug_era_backup SELECT 1", set())):
        other = spec.model_copy(deep=True)
        other.custom_sql = [CustomSQL(name="eras", target_table="metadata", sql=sql)]
        assert spec_era_tables(other) == found, sql
    # such SQL runs at the era table's place, after routing and the observation periods, also when
    # its target_table comes earlier
    early = spec.model_copy(deep=True)
    early.custom_sql = [CustomSQL(name="eras", target_table="drug_exposure", sql="INSERT INTO cdm.drug_era SELECT 1")]
    steps = compile_spec(early, S)
    assert labels(steps)[-4:] == ["observation_period_fill", "condition_era", "custom_eras", "cdm_source"]
    assert step(steps, "custom_eras").target_table == "drug_exposure"
    late = spec.model_copy(deep=True)
    late.custom_sql = [CustomSQL(name="eras", target_table="metadata", sql="INSERT INTO cdm.drug_era SELECT 1")]
    assert labels(compile_spec(late, S))[-3:] == ["condition_era", "custom_eras", "cdm_source"]
    # the validator says that spec SQL replaces the era step (the README advised adding eras so)
    warned = [i for i in validate_spec(early) if "era step" in i.message]
    assert [(i.severity, i.target_table) for i in warned] == [("warning", "drug_era")]
    assert "remove that mapping or custom SQL to let the build derive drug_era" in warned[0].message
    assert not [i for i in validate_spec(early.model_copy(update={"derive_eras": False})) if "era step" in i.message]
    assert not [i for i in validate_spec(spec) if "era step" in i.message]
    # the text in a string literal or a comment is not SQL that loads it
    quoted = spec.model_copy(deep=True)
    quoted.custom_sql = [CustomSQL(name="note", target_table="metadata",
                                   sql="SELECT 'INSERT INTO {cdm}.drug_era' -- INSERT INTO {cdm}.condition_era")]
    assert spec_era_tables(quoted) == set() and derived_era_tables(quoted) == ["condition_era", "drug_era"]
    # a mapping into an era table replaces the step too
    mapped = spec.model_copy(deep=True)
    mapped.table_mappings.append(TableMapping(target_table="drug_era", source={"table": "prescriptions", "alias": "r"},
                                              fields=[]))
    assert era_status(mapped)["drug_era"] == "spec"

    # an era needs its event table: loaded by the spec, or a route destination
    no_drugs = spec.model_copy(deep=True)
    no_drugs.table_mappings = [tm for tm in no_drugs.table_mappings if tm.target_table != "drug_exposure"]
    assert derived_era_tables(no_drugs) == ["condition_era", "drug_era"]  # drugs may be moved there
    unrouted = no_drugs.model_copy(update={"route_by_domain": False})
    assert derived_era_tables(unrouted) == ["condition_era"]
    assert era_status(unrouted)["drug_era"] == "no_records"
    assert "drug_era" not in target_tables(unrouted)


def test_eras_described_for_review(tmp_path):
    spec = load_ref()
    text = eras_text(spec)
    assert text.startswith("condition_era derived by the build from condition_occurrence (OHDSI era logic); drug_era "
                           "derived by the build from drug_exposure")
    assert "start at most 30 days after the earlier ones ended" in text
    assert f"- **Eras:** {text}" in render_spec_md(spec)
    assert eras_text(spec.model_copy(update={"derive_eras": False})) == (
        "condition_era not built (derive_eras: false); drug_era not built (derive_eras: false)")
    design = _design_section(spec)
    assert design["eras"] == text and design["era_status"] == {"condition_era": "derived", "drug_era": "derived"}
    # the repair agent is told how the eras are built
    cfg = plain_config(tmp_path)
    prompt = build_repair_prompt(spec, 1, [], [], cfg)
    assert ("Eras: condition_era and drug_era are derived by the build from condition_occurrence and drug_exposure"
            in prompt) and "never add a mapping into them" in prompt
    assert "Eras: the build derives no eras (derive_eras: false)." in build_repair_prompt(
        spec.model_copy(update={"derive_eras": False}), 1, [], [], cfg)
    assert "a new mapping into condition_era or drug_era is refused" in " ".join(REPAIR_SYSTEM.split())
    # one era derived: its own words
    conditions_only = spec.model_copy(update={"route_by_domain": False}, deep=True)
    conditions_only.table_mappings = [tm for tm in conditions_only.table_mappings if tm.target_table != "drug_exposure"]
    line = next(x for x in build_repair_prompt(conditions_only, 1, [], [], cfg).splitlines() if x.startswith("Eras:"))
    assert line == ("Eras: condition_era is derived by the build from condition_occurrence with the OHDSI era logic "
                    "(step *_condition_era, generated): never add a mapping into it; its checks are not yours to fix, "
                    "fix the condition records instead.")
    # spec SQL replaces the era step only when the build derives eras
    own = spec.model_copy(deep=True)
    own.custom_sql = [CustomSQL(name="eras", target_table="drug_era", sql="INSERT INTO {cdm}.drug_era SELECT 1")]
    assert "drug_era loaded by the spec's own SQL (it replaces the build's era step)" in eras_text(own)
    assert eras_text(own.model_copy(update={"derive_eras": False})) == (
        "condition_era not built (derive_eras: false); drug_era loaded by the spec's own SQL")
    assert "Do not plan condition_era or drug_era" in " ".join(_plan_prompt("", "").split())


def test_normalise_plan_skips_eras():
    plan = TablePlan(
        person_key=KeySpec(table="individuals", expression="individual_id"),
        tables=[PlannedTable(target_table="person", source_tables=["individuals"], rationale="r"),
                PlannedTable(target_table="Condition_Era", source_tables=["diagnoses"], rationale="r"),
                PlannedTable(target_table="drug_era", source_tables=["prescriptions"], rationale="r")],
    )
    out = _normalise_plan(plan)
    assert [t.target_table for t in out.tables] == ["person"]
    assert [q for q in out.open_questions if "era" in q] == [
        "The plan named condition_era, which the build derives from condition_occurrence with the OHDSI era logic; "
        "it was skipped.",
        "The plan named drug_era, which the build derives from drug_exposure with the OHDSI era logic; it was skipped."]


def _era_mapping(table: str = "condition_era") -> TableMapping:
    return TableMapping(target_table=table, source={"table": "diagnoses", "alias": "d"}, fields=[
        FieldMapping(target_field="person_id", kind="person_id", expression="d.individual_id")])


def test_repair_rejects_era_targets(tmp_path):
    spec = load_ref()
    cfg = plain_config(tmp_path)
    state = RepairState(spec=spec, conn=None, schemas=S, cfg=cfg)
    tools = {t.name: t for t in build_repair_tools(state)}
    replace = tools["replace_table_mapping"]
    out = replace.invoke({"target_table": "condition_era", "mapping_index": 0, "rationale": "r",
                          "mapping_yaml": yaml.safe_dump(_era_mapping().model_dump())})
    assert out.startswith("REJECTED: the build derives condition_era from condition_occurrence with the OHDSI era "
                          "logic (derive_eras)") and "fix the condition_occurrence records" in out
    assert state.actions == [] and state.spec is spec
    # update_field_mapping edits an existing mapping only: there is no era mapping to edit
    out = tools["update_field_mapping"].invoke({"target_table": "condition_era", "mapping_index": 0,
                                                "target_field": "person_id", "kind": "person_id",
                                                "expression": "d.individual_id", "rationale": "r"})
    assert out.startswith("REJECTED: no mapping #0 for condition_era") and state.actions == []
    # without derive_eras, or an edit of an era mapping the spec already has (it replaced the step), passes
    assert era_issues(spec.model_copy(update={"derive_eras": False}), _era_mapping()) == []
    own = spec.model_copy(deep=True)
    own.table_mappings.append(_era_mapping("drug_era"))
    assert era_issues(own, _era_mapping("drug_era"), own.table_mappings[-1]) == [] and era_issues(own, _era_mapping())
    assert era_issues(spec, spec.table_mappings[0]) == []
    # a second writer of an era table the spec's own SQL loads would load its eras twice
    assert era_issues(own, _era_mapping("drug_era"))[0].startswith(
        "the spec's own SQL already loads drug_era, and a new mapping into it would load its eras twice")
    custom = spec.model_copy(deep=True)
    custom.custom_sql = [CustomSQL(name="eras", target_table="condition_era", sql="INSERT INTO {cdm}.condition_era "
                                   "SELECT 1")]
    state = RepairState(spec=custom, conn=None, schemas=S, cfg=cfg)
    out = {t.name: t for t in build_repair_tools(state)}["replace_table_mapping"].invoke(
        {"target_table": "condition_era", "mapping_index": 0, "rationale": "r",
         "mapping_yaml": yaml.safe_dump(_era_mapping().model_dump())})
    assert out.startswith("REJECTED: the spec's own SQL already loads condition_era") and state.actions == []
    # an era whose event table the spec does not load: the reason says so
    no_drugs = spec.model_copy(update={"route_by_domain": False}, deep=True)
    no_drugs.table_mappings = [tm for tm in no_drugs.table_mappings if tm.target_table != "drug_exposure"]
    assert era_issues(no_drugs, _era_mapping("drug_era")) == [
        "the build derives drug_era from the drug_exposure records (derive_eras), and the spec loads none: load the "
        "drug_exposure records instead of mapping into drug_era"]


def _era_result(table: str = "drug_era", **kw) -> TableLoadResult:
    return TableLoadResult(target_table=table, step=f"19_{table}", kind="derived", blocking=False, **kw)


def test_era_step_errors_and_checks_are_not_repaired(tmp_path):
    cfg = plain_config(tmp_path)
    person = TableLoadResult(target_table="person", step="03_person", kind="insert", rows_inserted=10, source_rows=10)
    era = _era_result(error="permission denied for table concept_ancestor")
    etl = ETLRunResult(started_at=NOW, steps=[person, _era_result("condition_era", rows_inserted=4), era])
    assert etl.ok and etl.blocking_errors() == [] and derived_tables(etl) == {"condition_era", "drug_era"}
    (err,) = [r for r in etl_checks(etl) if r.check_name == "etlStepError"]
    assert (err.severity, err.repairable, err.step) == ("convention", False, "drug_era")
    assert err.description.startswith("ETL step '19_drug_era' deriving drug_era from drug_exposure failed, so drug_era "
                                      "is empty; the build goes on without it")
    sev = {"fatal", "convention"}

    def res(table: str, **kw) -> CheckResult:
        return CheckResult(**{"check_name": "measurePersonCompleteness", "check_level": "TABLE", "table": table,
                              "description": "d", "severity": "convention", "failed": True, **kw})

    failures = [err, res("drug_era"), res("condition_era"), res("drug_exposure")]
    dq = DQReport(run_at=NOW, passed=True, results=failures)
    assert gate(failures, cfg, etl) == (True, [])
    # the checks of the era tables the build derived are never given to the repair agent
    assert repair_triggers(dq, etl, cfg) == ([], [failures[3]])
    assert [r.table for r in failures if repairable(r, sev, derived=derived_tables(etl))] == ["drug_exposure"]
    # an era table the spec loads (no era step) is repaired like any other
    spec_loaded = ETLRunResult(started_at=NOW, steps=[person, TableLoadResult(
        target_table="drug_era", step="19_custom_eras", kind="custom", rows_inserted=3)])
    assert derived_tables(spec_loaded) == set() and derived_tables(None) == set()
    assert [r.table for r in repair_triggers(dq, spec_loaded, cfg)[1]] == ["drug_era", "condition_era", "drug_exposure"]

    from omop_agent.pipeline import Pipeline

    p = Pipeline(cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"provider": "none"})}), log=lambda m: None)
    only_eras = DQReport(run_at=NOW, passed=True, results=failures[:3])
    assert not p._repairable(only_eras, etl) and p._repairable(only_eras, spec_loaded)
    assert p._score(etl, only_eras)[-1] == 0 and p._score(spec_loaded, only_eras)[-1] == -2


def test_checks_of_spec_loaded_eras_are_left_to_a_person(tmp_path):
    """While the build derives eras, an era table the spec's own SQL loads in its place is a
    person's decision (the validator warns): its checks are not given to the repair agent."""
    from omop_agent.pipeline import Pipeline

    cfg = plain_config(tmp_path)
    spec = load_ref().model_copy(deep=True)
    spec.custom_sql = [CustomSQL(name="eras", target_table="drug_era", sql="INSERT INTO {cdm}.drug_era SELECT 1")]
    run = ETLRunResult(started_at=NOW, steps=[TableLoadResult(
        target_table="drug_era", step="19_eras", kind="custom", rows_inserted=3)])
    assert derived_tables(run) == set() and derived_tables(run, spec) == {"drug_era"}
    off = spec.model_copy(update={"derive_eras": False})
    assert derived_tables(run, off) == set()
    failure = CheckResult(check_name="measurePersonCompleteness", check_level="TABLE", table="drug_era",
                          description="d", severity="convention", failed=True, num_violated_rows=1,
                          num_denominator_rows=2)
    dq = DQReport(run_at=NOW, passed=True, results=[failure])
    assert repair_triggers(dq, run, cfg, spec) == ([], [])
    # without the spec, or when the spec turned the build's eras off, it is repaired like any other
    assert repair_triggers(dq, run, cfg) == ([], [failure]) == repair_triggers(dq, run, cfg, off)
    p = Pipeline(cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"provider": "none"})}), log=lambda m: None)
    assert not p._repairable(dq, run, spec) and p._repairable(dq, run, off)
    assert p._score(run, dq, None, spec)[-1] == 0 and p._score(run, dq, None, off)[-1] == -1
    # the agent is told why
    line = next(x for x in build_repair_prompt(spec, 1, [], [], cfg).splitlines() if x.startswith("Eras:"))
    assert ("drug_era is loaded by the spec's own SQL in place of the build's era step: its checks are not yours to "
            "fix (a person decides") in line
    line = next(x for x in build_repair_prompt(off, 1, [], [], cfg).splitlines() if x.startswith("Eras:"))
    assert "drug_era is loaded by the spec's own SQL;" in line and "in place of" not in line


def test_rows_and_history_count_eras_apart():
    person = TableLoadResult(target_table="person", step="03_person", kind="insert", rows_inserted=80)
    cond = TableLoadResult(target_table="condition_occurrence", step="07_condition_occurrence", kind="insert",
                           rows_inserted=500)
    eras = [_era_result("condition_era", rows_inserted=300), _era_result(rows_inserted=200)]
    run = ETLRunResult(started_at=NOW, steps=[person, cond, *eras])
    assert run.rows_loaded() == 580 and run.rows_derived() == 500
    section = _etl_section(run)
    assert (section["total_rows"], section["derived_rows"]) == (580, 500)
    # the eras follow the condition and drug records, which the history tracks: not counted, not a loss
    assert _BuildHistory._rows(run) == {"person": 80, "condition_occurrence": 500}
    history = _BuildHistory()
    applicable = [CheckResult(check_name="isRequired", check_level="FIELD", table=t, field="person_id",
                              description="d") for t in ("drug_era", "condition_occurrence")]
    history.add(run, DQReport(run_at=NOW, passed=True, results=applicable))
    emptied = [r.model_copy(update={"not_applicable": True}) for r in applicable]
    no_eras = ETLRunResult(started_at=NOW, steps=[person, cond, *[e.model_copy(update={"rows_inserted": 0})
                                                                  for e in eras]])
    assert history.losses(no_eras, DQReport(run_at=NOW, passed=True, results=emptied[:1])) == []
    assert history.losses(no_eras, DQReport(run_at=NOW, passed=True, results=emptied))[0].startswith(
        "1 check(s) no longer apply, e.g. isRequired.condition_occurrence.person_id")


def test_report_groups_era_failures(tmp_path):
    cfg = plain_config(tmp_path)

    def res(table: str, check: str = "measurePersonCompleteness") -> CheckResult:
        return CheckResult(check_name=check, check_level="TABLE", table=table, description="d", severity="convention",
                           failed=True, num_violated_rows=1, num_denominator_rows=2)

    dq = DQReport(run_at=NOW, passed=True, results=[
        res("drug_era"), res("condition_occurrence"), res("drug_era", "measureDrugEraCoverage"), res("condition_era")])
    section = _dq_section(dq, cfg, {"condition_era", "drug_era"})
    assert section["n_failed"] == 4 and [f["key"] for f in section["failed"]] == [
        "measurePersonCompleteness.condition_occurrence"]
    assert [(g["table"], g["source"], [f["key"] for f in g["failed"]]) for g in section["derived_failed"]] == [
        ("condition_era", "condition_occurrence", ["measurePersonCompleteness.condition_era"]),
        ("drug_era", "drug_exposure", ["measureDrugEraCoverage.drug_era", "measurePersonCompleteness.drug_era"])]
    # an era table the spec loads is listed with the others
    assert len(_dq_section(dq, cfg)["failed"]) == 4 and _dq_section(dq, cfg)["derived_failed"] == []


def test_report_shows_eras(full_ws):  # noqa: F811 (the fixture imported above)
    cfg, ws = full_ws
    run = ws.load_json(ws.etl_run_json, ETLRunResult)
    run.steps += [_era_result("condition_era", rows_inserted=300), _era_result(rows_inserted=200)]
    ws.save_json(ws.etl_run_json, run)
    dq = ws.load_json(ws.dq_json, DQReport)
    dq.results.append(CheckResult(check_name="measureDrugEraCoverage", check_level="TABLE", table="drug_era",
                                  description="Exposures that get no drug_era.", severity="convention", failed=True,
                                  num_violated_rows=7, num_denominator_rows=70, threshold_pct=5.0))
    ws.save_json(ws.dq_json, dq)
    data = collect_report_data(cfg, ws)
    assert data["design"]["eras"].startswith("condition_era derived by the build from condition_occurrence")
    assert (data["etl"]["total_rows"], data["etl"]["derived_rows"]) == (398, 500)
    assert data["dq"]["n_failed"] == 3 and len(data["dq"]["failed"]) == 2
    md, html = render_markdown(data), render_html(data)
    assert "Eras: condition_era derived by the build from condition_occurrence (OHDSI era logic)" in md
    assert "398 rows loaded, 500 rows derived by the build (observation periods, eras)" in md
    assert "**FAILED**, 3 failed" in md
    assert "**Failed checks of drug_era (derived by the build from drug_exposure)**" in md
    assert "| measureDrugEraCoverage.drug_era | convention |" in md
    assert ("The eras follow the drug_exposure records: a failure here comes from those records, or is expected "
            "(drugs without a standard ingredient, such as vaccines, get no drug era). These checks are never sent to "
            "the repair agent.") in md
    assert "**Failed checks of the loaded tables**" in md
    assert "Failed checks of drug_era <span class=\"muted\">(derived by the build from drug_exposure)</span>" in html
    assert "<dt>Eras</dt><dd>condition_era derived by the build" in html and "<h3>Failed checks of the loaded tables</h3>" in html
    # only era failures: no empty "Failed checks" table next to a non-zero failed count
    data["dq"] = {**data["dq"], **_dq_section(DQReport(run_at=NOW, passed=True, results=dq.results[-1:]), cfg,
                                              {"drug_era"})}
    md, html = render_markdown(data), render_html(data)
    dq_md = md.split("## 8. Data quality")[1].split("\n## ")[0]
    assert "1 failed" in dq_md and "**Failed checks**" not in dq_md and "_None._" not in dq_md
    assert "**Failed checks of drug_era (derived by the build from drug_exposure)**" in dq_md
    assert "Failed checks of the loaded tables" not in dq_md and "<h3>Failed checks</h3>" not in html
    assert "Failed checks of drug_era" in html


# --------------------------------------------------------------------------------------
# the OHDSI era logic on PostgreSQL
# --------------------------------------------------------------------------------------

# id, name, domain, vocabulary, class, standard
CONCEPTS = [
    (1001, "ingredient A", "Drug", "RxNorm", "Ingredient", "S"),
    (1002, "ingredient B", "Drug", "RxNorm", "Ingredient", "S"),
    (1003, "ingredient C (RxNorm Extension only)", "Drug", "RxNorm Extension", "Ingredient", "S"),
    (1004, "ingredient D (deprecated)", "Drug", "RxNorm", "Ingredient", None),
    (2001, "A 500 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S"),
    (2002, "B 10 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S"),
    (2004, "A 250 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S"),
    (2012, "A / B Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S"),
    (2003, "C 10 MG Oral Tablet [Kenya brand]", "Drug", "RxNorm Extension", "Marketed Product", "S"),
    (2009, "vaccine without ingredient", "Drug", "CVX", "CVX", "S"),
    (3001, "ATC class of A", "Drug", "ATC", "ATC 4th", "C"),
    (5001, "malaria", "Condition", "SNOMED", "Clinical Finding", "S"),
    (5002, "pneumonia", "Condition", "SNOMED", "Clinical Finding", "S"),
    (5003, "anaemia", "Condition", "SNOMED", "Clinical Finding", "S"),
]
ANCESTOR = [  # (ancestor, descendant)
    (1001, 1001), (1002, 1002), (1003, 1003), (2001, 2001), (2002, 2002), (2004, 2004), (2012, 2012), (2003, 2003),
    (2009, 2009), (1001, 2001), (1002, 2002), (1001, 2004), (1001, 2012), (1002, 2012), (1003, 2003), (3001, 2001),
    (3001, 1001), (1004, 2001),
]


@pytest.fixture
def cdm(pg):
    """Empty CDM, ETL and vocabulary tables with drug and condition concepts and their ancestors."""
    conn, schemas = pg
    create_cdm_tables(conn, schemas, include_vocab=True)
    create_etl_tables(conn, schemas)
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {schemas.vocab}.concept (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, "
            "standard_concept, concept_code, valid_start_date, valid_end_date) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, DATE '1970-01-01', DATE '2099-12-31')",
            [(*c, str(c[0])) for c in CONCEPTS])
        cur.executemany(
            f"INSERT INTO {schemas.vocab}.concept_ancestor (ancestor_concept_id, descendant_concept_id, "
            "min_levels_of_separation, max_levels_of_separation) VALUES (%s, %s, 0, 0)", ANCESTOR)
    conn.commit()
    return conn, schemas


def add_conditions(conn, schemas, rows: list[tuple]) -> None:
    """(person, concept, start, end) rows, ids in order."""
    first = int(scalar(conn, f"SELECT COALESCE(MAX(condition_occurrence_id), 0) FROM {schemas.cdm}.condition_occurrence"))
    with conn.cursor() as cur:
        with cur.copy(f"COPY {schemas.cdm}.condition_occurrence (condition_occurrence_id, person_id, condition_concept_id, "
                      "condition_start_date, condition_end_date, condition_type_concept_id) FROM STDIN") as cp:
            for i, (p, c, s, e) in enumerate(rows, start=first + 1):
                cp.write_row((i, p, c, s, e, 32817))
    conn.commit()


def add_drugs(conn, schemas, rows: list[tuple]) -> None:
    """(person, drug concept, start, end, days_supply) rows, ids in order."""
    first = int(scalar(conn, f"SELECT COALESCE(MAX(drug_exposure_id), 0) FROM {schemas.cdm}.drug_exposure"))
    with conn.cursor() as cur:
        with cur.copy(f"COPY {schemas.cdm}.drug_exposure (drug_exposure_id, person_id, drug_concept_id, "
                      "drug_exposure_start_date, drug_exposure_end_date, days_supply, drug_type_concept_id) "
                      "FROM STDIN") as cp:
            for i, (p, c, s, e, ds) in enumerate(rows, start=first + 1):
                cp.write_row((i, p, c, s, e, ds, 32817))
    conn.commit()


def era_steps(schemas: Schemas) -> dict[str, CompiledStep]:
    return {s.label: s for s in compile_spec(load_ref(), schemas) if s.label in DERIVED_ERAS}


def run_eras(conn, schemas) -> dict[str, TableLoadResult]:
    out = {}
    for label, s in era_steps(schemas).items():
        out[label] = _run_step(conn, s, None)
        conn.commit()
        assert out[label].error is None, (label, out[label].error)
    return out


def condition_eras(conn, schemas) -> set[tuple]:
    return {(r["person_id"], r["condition_concept_id"], str(r["condition_era_start_date"]),
             str(r["condition_era_end_date"]), r["condition_occurrence_count"])
            for r in fetch_all(conn, f"SELECT * FROM {schemas.cdm}.condition_era")}


def drug_eras(conn, schemas) -> set[tuple]:
    return {(r["person_id"], r["drug_concept_id"], str(r["drug_era_start_date"]), str(r["drug_era_end_date"]),
             r["drug_exposure_count"], r["gap_days"]) for r in fetch_all(conn, f"SELECT * FROM {schemas.cdm}.drug_era")}


def _dates(*rows: tuple) -> list[tuple]:
    return [tuple(D(x) if isinstance(x, str) else x for x in r) for r in rows]


@pytest.mark.postgres
def test_condition_era_cases(cdm):
    conn, schemas = cdm
    add_conditions(conn, schemas, _dates(
        (1, 5001, "2020-01-01", "2020-01-10"),                                         # single
        (2, 5001, "2020-01-01", None),                                                 # no end: start + 1
        (3, 5001, "2020-01-01", "2020-01-10"), (3, 5001, "2020-02-09", None),          # 30 days apart: one era
        (4, 5001, "2020-01-01", "2020-01-10"), (4, 5001, "2020-02-10", "2020-02-10"),  # 31 days: two eras
        (5, 5001, "2020-01-01", "2020-03-01"), (5, 5001, "2020-01-15", "2020-01-20"),
        (5, 5001, "2020-02-15", "2020-04-01"),                                         # overlaps: one era
        (6, 5001, "2020-01-01", "2020-06-30"), (6, 5001, "2020-01-10", "2020-01-11"),
        (6, 5001, "2020-07-20", "2020-07-21"),                                         # the latest end so far counts
        (7, 5001, "2020-01-01", "2020-01-05"), (7, 5002, "2020-01-02", "2020-01-03"),  # one era per concept
        (8, 0, "2020-01-01", "2020-01-05"), (8, 5001, "2021-01-01", None),             # concept 0: no era
        (9, 5001, "2020-03-03", "2020-03-03"), (9, 5001, "2020-03-03", "2020-03-03"),  # same-day duplicates
        (10, 5001, "2020-05-10", "2020-05-01"),                                        # ends before it starts
    ))
    # many identical records in two eras: the tie order must be the same in both windows
    add_conditions(conn, schemas, _dates(*[(11, 5003, "2020-01-01", "2020-01-05")] * 1000,
                                         *[(11, 5003, "2020-03-01", "2020-03-05")] * 1000))
    res = run_eras(conn, schemas)["condition_era"]
    assert res.rows_inserted == 14 and res.kind == "derived" and not res.blocking
    assert condition_eras(conn, schemas) == {
        (1, 5001, "2020-01-01", "2020-01-10", 1),
        (2, 5001, "2020-01-01", "2020-01-02", 1),
        (3, 5001, "2020-01-01", "2020-02-10", 2),
        (4, 5001, "2020-01-01", "2020-01-10", 1), (4, 5001, "2020-02-10", "2020-02-10", 1),
        (5, 5001, "2020-01-01", "2020-04-01", 3),
        (6, 5001, "2020-01-01", "2020-07-21", 3),
        (7, 5001, "2020-01-01", "2020-01-05", 1), (7, 5002, "2020-01-02", "2020-01-03", 1),
        (8, 5001, "2021-01-01", "2021-01-02", 1),
        (9, 5001, "2020-03-03", "2020-03-03", 2),
        (10, 5001, "2020-05-10", "2020-05-11", 1),
        (11, 5003, "2020-01-01", "2020-01-05", 1000), (11, 5003, "2020-03-01", "2020-03-05", 1000),
    }
    ids = [r["condition_era_id"] for r in fetch_all(conn, f"SELECT condition_era_id FROM {schemas.cdm}.condition_era "
                                                          "ORDER BY condition_era_id")]
    assert ids == list(range(1, 15))


@pytest.mark.postgres
def test_drug_era_cases(cdm):
    conn, schemas = cdm
    add_drugs(conn, schemas, _dates(
        (21, 2001, "2020-01-01", "2020-01-10", 10), (21, 2001, "2020-01-20", "2020-01-29", 10),  # gap 10
        (22, 2001, "2020-01-01", "2020-01-10", 10), (22, 2001, "2020-02-10", "2020-02-19", 10),  # gap 31
        (23, 2001, "2020-01-01", "2020-01-31", 31), (23, 2001, "2020-01-15", "2020-02-14", 31),
        (23, 2001, "2020-03-01", "2020-03-10", 10),                                              # overlap and gap
        (24, 2012, "2020-01-01", "2020-01-10", 10),                                              # two ingredients
        (25, 2012, "2020-01-01", "2020-01-10", 10), (25, 2001, "2020-01-20", "2020-01-30", 11),
        (26, 2001, "2020-05-10", "2020-05-01", 14),                       # ends before it starts: days_supply
        (27, 2001, "2020-05-10", "2020-05-01", None),                     # ... without one: on its start
        (28, 2001, "2020-05-10", "2020-05-01", 0),
        (29, 2001, "2020-01-01", "2020-01-10", -5),                       # negative days_supply: no era
        (30, 0, "2020-01-01", "2020-01-10", 10),                          # concept 0: no era
        (31, 2009, "2020-01-01", "2020-01-01", 1),                        # no ingredient (vaccine): no era
        (32, 2003, "2020-01-01", "2020-01-30", 30),                       # RxNorm Extension ingredient
        (33, 2001, "2020-03-01", "2020-03-01", 1), (33, 2001, "2020-03-01", "2020-03-01", 1),    # duplicates
        (34, 2001, "2020-01-01", "2020-01-10", 10), (34, 2001, "2020-01-10", "2020-01-20", 11),  # touching
        (35, 2001, "2020-01-01", "2020-06-30", 180), (35, 2001, "2020-01-10", "2020-01-11", 2),
        (35, 2001, "2020-07-15", "2020-07-20", 6),                        # the latest end so far counts
        (36, 2001, "2020-05-10", "2020-05-01", 2147483647),              # implausible days_supply: no error
        (37, 2001, "2020-05-10", "2020-05-01", 3_000_000),
        (38, 2001, "2020-05-10", "2020-05-01", 365), (39, 2001, "2020-05-10", "2020-05-01", 366),
    ))
    add_drugs(conn, schemas, _dates(*[(40, 2002, "2020-01-01", "2020-01-05", 5)] * 1000,
                                    *[(40, 2002, "2020-03-01", "2020-03-05", 5)] * 1000))
    res = run_eras(conn, schemas)["drug_era"]
    assert res.error is None and res.rows_inserted == 21
    assert drug_eras(conn, schemas) == {
        (21, 1001, "2020-01-01", "2020-01-29", 2, 10),
        (22, 1001, "2020-01-01", "2020-01-10", 1, 0), (22, 1001, "2020-02-10", "2020-02-19", 1, 0),
        (23, 1001, "2020-01-01", "2020-03-10", 3, 16),  # 2020-02-14 to 03-01: 16 days (leap year)
        (24, 1001, "2020-01-01", "2020-01-10", 1, 0), (24, 1002, "2020-01-01", "2020-01-10", 1, 0),
        (25, 1001, "2020-01-01", "2020-01-30", 2, 10), (25, 1002, "2020-01-01", "2020-01-10", 1, 0),
        (26, 1001, "2020-05-10", "2020-05-23", 1, 0),  # start + 14 - 1
        (27, 1001, "2020-05-10", "2020-05-10", 1, 0), (28, 1001, "2020-05-10", "2020-05-10", 1, 0),
        (32, 1003, "2020-01-01", "2020-01-30", 1, 0),
        (33, 1001, "2020-03-01", "2020-03-01", 2, 0),
        (34, 1001, "2020-01-01", "2020-01-20", 2, 0),
        (35, 1001, "2020-01-01", "2020-07-20", 3, 15),
        (36, 1001, "2020-05-10", "2020-05-10", 1, 0), (37, 1001, "2020-05-10", "2020-05-10", 1, 0),
        (38, 1001, "2020-05-10", "2021-05-09", 1, 0), (39, 1001, "2020-05-10", "2020-05-10", 1, 0),
        (40, 1002, "2020-01-01", "2020-01-05", 1000, 0), (40, 1002, "2020-03-01", "2020-03-05", 1000, 0),
    }
    # no era for the deprecated ingredient (1004) or the ATC class (3001) the drug also rolls up to
    assert not count(conn, schemas, "drug_era", "drug_concept_id IN (1004, 3001)")


@pytest.mark.postgres
def test_drug_era_checks_with_vaccines_and_routed_drugs(cdm, tmp_path):
    """Exposures that get no era (vaccines) are measureDrugEraCoverage's to count, not
    measureDrugEraCompleteness's; and the drug eras of records only moved to drug_exposure by domain
    are checked like those of loaded records, also when there are none."""
    conn, schemas = cdm
    cfg = plain_config(tmp_path)
    spec = load_ref().model_copy(deep=True)
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "drug_exposure"]
    assert derived_era_tables(spec) == ["condition_era", "drug_era"] and spec.route_by_domain

    def era_checks(s=spec) -> dict[str, CheckResult]:
        return {r.check_name: r for r in run_builtin_checks(conn, schemas, cfg, s).results
                if r.table in DERIVED_ERAS and r.check_level == "TABLE"}

    def populated() -> set[str]:
        existing = {r["table_name"] for r in fetch_all(conn, "SELECT table_name FROM information_schema.tables WHERE "
                                                             "table_schema = %s", (schemas.cdm,))}
        out = _populated_tables(conn, schemas, existing, spec)
        conn.rollback()
        return out

    # no drug moved there: drug_exposure and its eras are not populated, the checks do not apply
    run_eras(conn, schemas)
    assert not {"drug_exposure", "drug_era"} & populated()
    checks = era_checks()
    assert all(checks[c].not_applicable for c in ("measureDrugEraCompleteness", "measureDrugEraCoverage"))
    assert checks["measureConditionEraCompleteness"].not_applicable  # no condition records
    # one vaccine moved there (an immunisation coded in a procedure list): it gets no era, which only
    # the coverage check counts; drug_era is populated although it has no rows
    add_drugs(conn, schemas, _dates((1, 2009, "2020-01-01", "2020-01-01", 1)))
    run_eras(conn, schemas)
    assert not count(conn, schemas, "drug_era") and {"drug_exposure", "drug_era"} <= populated()
    checks = era_checks()
    complete, coverage = checks["measureDrugEraCompleteness"], checks["measureDrugEraCoverage"]
    assert not (complete.not_applicable or complete.failed or complete.error) and complete.num_denominator_rows == 0
    assert "not built" not in complete.description and "has no rows" not in complete.description
    assert coverage.failed and (coverage.num_violated_rows, coverage.num_denominator_rows) == (1, 1)
    # a drug with an ingredient gets an era; when it has none the era build is incomplete
    add_drugs(conn, schemas, _dates((2, 2001, "2020-01-01", "2020-01-10", 10)))
    execute(conn, f"DELETE FROM {schemas.cdm}.drug_era")
    conn.commit()
    run_eras(conn, schemas)
    checks = era_checks()
    complete, coverage = checks["measureDrugEraCompleteness"], checks["measureDrugEraCoverage"]
    assert not complete.failed and (complete.num_violated_rows, complete.num_denominator_rows) == (0, 1)
    assert coverage.failed and (coverage.num_violated_rows, coverage.num_denominator_rows) == (1, 2)
    execute(conn, f"DELETE FROM {schemas.cdm}.drug_era")
    complete = era_checks()["measureDrugEraCompleteness"]
    assert complete.failed and (complete.num_violated_rows, complete.num_denominator_rows) == (1, 1)
    # without a spec the empty drug_era is not populated, and the description says why
    no_spec = {r.check_name: r for r in run_builtin_checks(conn, schemas, cfg).results if r.table == "drug_era"}
    conn.rollback()
    assert no_spec["measureDrugEraCompleteness"].not_applicable
    assert no_spec["measureDrugEraCompleteness"].description.endswith(" drug_era has no rows.")


def reference_sql(schemas: Schemas) -> dict[str, str]:
    """The published era scripts, translated (tests/fixtures/era_reference.sql), per table."""
    text = (FIXTURES / "era_reference.sql").read_text()
    parts = re.split(r"(?m)^-- @@ (\w+)\n", text)[1:]
    return {name: schemas.render(sql.strip().rstrip(";")) for name, sql in zip(parts[::2], parts[1::2])}


def random_records(seed: int, persons: int = 2000) -> tuple[list[tuple], list[tuple]]:
    """Valid records (ends on or after the start, or none for conditions) for the comparison with
    the published scripts: up to 12 conditions and 12 drug exposures per person within a year, with
    exact duplicates, of RxNorm drugs whose ingredients are all standard (the published script also
    rolls drugs up to deprecated ingredients, such as 1004 of drug 2001)."""
    rnd = random.Random(seed)
    day0 = date(2020, 1, 1)
    conditions, drugs = [], []
    for p in range(1, persons + 1):
        for _ in range(rnd.randint(1, 12)):
            if conditions and conditions[-1][0] == p and rnd.random() < 0.1:
                conditions.append(conditions[-1])  # an exact duplicate
                continue
            start = day0 + timedelta(days=rnd.randint(0, 365))
            end = None if rnd.random() < 0.2 else start + timedelta(days=rnd.choice([0, 0, 1, 3, 7, 20, 45]))
            conditions.append((p, rnd.choice([5001, 5002, 5003]), start, end))
        for _ in range(rnd.randint(0, 12)):
            if drugs and drugs[-1][0] == p and rnd.random() < 0.1:
                drugs.append(drugs[-1])
                continue
            start = day0 + timedelta(days=rnd.randint(0, 365))
            days = rnd.choice([1, 3, 7, 14, 28, 30, 60])
            supply = rnd.choice([None, 0, days, days, days, -1])
            drugs.append((p, rnd.choice([2002, 2004, 2012, 1001]), start, start + timedelta(days=days - 1), supply))
    return conditions, drugs


@pytest.mark.postgres
def test_eras_equal_published_scripts(cdm):
    conn, schemas = cdm
    conditions, drugs = random_records(seed=7)
    add_conditions(conn, schemas, conditions + [(1, 0, date(2020, 1, 1), None)])  # concept 0: no era in either
    add_drugs(conn, schemas, drugs)
    run_eras(conn, schemas)
    ref = reference_sql(schemas)
    ours = {
        "condition_era": "SELECT person_id, condition_concept_id, condition_era_start_date, condition_era_end_date, "
                         f"condition_occurrence_count::bigint FROM {schemas.cdm}.condition_era",
        "drug_era": "SELECT person_id, drug_concept_id, drug_era_start_date, drug_era_end_date, "
                    f"drug_exposure_count::bigint, gap_days::bigint FROM {schemas.cdm}.drug_era",
    }
    for table in DERIVED_ERAS:
        execute(conn, f"CREATE TABLE {schemas.cdm}.{table}_ref AS {ref[table]}")
        cols = "*" if table == "condition_era" else "person_id, drug_concept_id, drug_era_start_date, " \
            "drug_era_end_date, drug_exposure_count::bigint, gap_days::bigint"
        theirs = f"SELECT {cols} FROM {schemas.cdm}.{table}_ref"
        diff = fetch_all(conn, f"SELECT (SELECT COUNT(*) FROM ({ours[table]} EXCEPT ALL {theirs}) a) AS only_ours, "
                               f"(SELECT COUNT(*) FROM ({theirs} EXCEPT ALL {ours[table]}) b) AS only_published")[0]
        assert diff == {"only_ours": 0, "only_published": 0}, table
    conn.rollback()
    # the comparison is not trivial: eras of several records, with gaps, and exposures left out
    n_cond = count(conn, schemas, "condition_era")
    assert n_cond > 5000 and count(conn, schemas, "condition_era", "condition_occurrence_count > 1") > 1000
    assert count(conn, schemas, "drug_era", "gap_days > 0") > 500
    assert count(conn, schemas, "drug_exposure", "days_supply < 0") > 100


# --------------------------------------------------------------------------------------
# a whole build on the demo data
# --------------------------------------------------------------------------------------


def map_drugs(conn, schemas) -> int:
    """Map every drug of the demo prescriptions to a clinical drug of the fixture vocabulary (the
    staged fixture has no code mappings, so every drug_concept_id would be 0)."""
    drugs = [r["drug"] for r in fetch_all(conn, f"SELECT DISTINCT drug FROM {schemas.src}.prescriptions ORDER BY 1")]
    targets = [r["concept_id"] for r in fetch_all(conn, f"SELECT concept_id FROM {schemas.vocab}.concept WHERE "
                                                        "concept_class_id = 'Clinical Drug' ORDER BY 1")]
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO {schemas.etl}.source_to_concept_map (source_code, source_vocabulary_id, "
                        "target_concept_id) VALUES (%s, 'SRC_DRUG', %s)",
                        [(d, targets[i % len(targets)]) for i, d in enumerate(drugs)])
    conn.commit()
    return len(drugs)


@pytest.mark.postgres
def test_era_checks_on_built_cdm(staged):
    conn, schemas, cfg = staged
    assert map_drugs(conn, schemas) > 3
    spec = fixed_ref()
    ws = Workspace(cfg.workdir)
    result = run_etl(conn, schemas, spec, cfg, ws)
    assert result.ok and result.errors() == []
    by = {s.step.split("_", 1)[1]: s for s in result.steps}
    n_ce, n_de = by["condition_era"].rows_inserted, by["drug_era"].rows_inserted
    assert n_ce == count(conn, schemas, "condition_era") > 0 and n_de == count(conn, schemas, "drug_era") > 0
    assert result.rows_derived() == n_ce + n_de + by["observation_period_fill"].rows_inserted
    # every person with a mapped condition has a condition era
    assert not count(conn, schemas, "condition_occurrence co", f"co.condition_concept_id <> 0 AND NOT EXISTS (SELECT 1 "
                     f"FROM {schemas.cdm}.condition_era e WHERE e.person_id = co.person_id AND e.condition_concept_id "
                     "= co.condition_concept_id AND co.condition_start_date BETWEEN e.condition_era_start_date AND "
                     "e.condition_era_end_date)")
    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    checks = {r.key: r for r in report.results}
    assert report.passed, report.blocking_failures
    cec, dec = checks["measureConditionEraCompleteness.condition_era"], checks["measureDrugEraCompleteness.drug_era"]
    cov = checks["measureDrugEraCoverage.drug_era"]
    for r in (cec, dec, cov):
        assert not (r.not_applicable or r.error or r.failed) and r.num_denominator_rows > 0 and r.repairable is False, r.key
        assert r.severity == "convention" and r.check_level == "TABLE"
    assert cec.num_violated_rows == 0 and cec.threshold_pct == 0 and cec.category == "Completeness"
    assert cov.threshold_pct == 5.0 and "BCG 36882756" in cov.description
    # the era tables are populated: their field checks run
    assert not checks["isRequired.condition_era.person_id"].not_applicable
    assert not checks["withinObservationPeriod.drug_era.drug_era_start_date"].not_applicable
    assert "measureConditionEraCompleteness" not in {r.check_name for r in report.results if r.not_applicable}

    # persons without eras and exposures without an ingredient are counted, and never sent to the agent
    execute(conn, f"DELETE FROM {schemas.cdm}.condition_era WHERE person_id IN (SELECT DISTINCT person_id FROM "
                  f"{schemas.cdm}.condition_era ORDER BY person_id LIMIT 3)")
    execute(conn, f"DELETE FROM {schemas.cdm}.drug_era")
    execute(conn, f"UPDATE {schemas.cdm}.drug_exposure SET days_supply = -1 WHERE drug_exposure_id <= 10")
    checks = {r.key: r for r in run_builtin_checks(conn, schemas, cfg, spec, result).results}
    conn.rollback()
    assert checks["measureConditionEraCompleteness.condition_era"].num_violated_rows == 3
    assert checks["measureConditionEraCompleteness.condition_era"].failed
    assert checks["measureDrugEraCompleteness.drug_era"].pct_violated_rows == 1.0
    assert checks["measureDrugEraCoverage.drug_era"].num_violated_rows == 10
    failed = DQReport(run_at=NOW, passed=True, results=list(checks.values()))
    assert not [r for r in repair_triggers(failed, result, cfg)[1] if r.table in DERIVED_ERAS]

    # without eras (the tables the last build wrote are emptied): the DQD check still runs (and
    # fails), but it is for a person to decide
    off = spec.model_copy(update={"derive_eras": False})
    result = run_etl(conn, schemas, off, cfg, ws)
    assert result.ok and not count(conn, schemas, "condition_era") and not count(conn, schemas, "drug_era")
    checks = {r.key: r for r in run_builtin_checks(conn, schemas, cfg, off, result).results}
    cec = checks["measureConditionEraCompleteness.condition_era"]
    assert cec.failed and cec.pct_violated_rows == 1.0 and cec.repairable is False
    assert "condition_era is not built: derive_eras is false and no spec SQL loads it" in cec.description
    assert checks["measureDrugEraCoverage.drug_era"].not_applicable
    # our own drug era check does not apply to eras switched off (DQD's condition check does)
    dec = checks["measureDrugEraCompleteness.drug_era"]
    assert dec.not_applicable and "drug_era is not built: derive_eras is false" in dec.description
    assert checks["fieldLevelChecks.condition_era"].not_applicable
    # without conditions, the DQD check does not apply
    execute(conn, f"TRUNCATE {schemas.cdm}.condition_occurrence")
    checks = {r.key: r for r in run_builtin_checks(conn, schemas, cfg, off, result).results}
    conn.rollback()
    assert checks["measureConditionEraCompleteness.condition_era"].not_applicable


@pytest.mark.postgres
def test_era_failure_is_not_blocking(staged):
    conn, schemas, cfg = staged
    role = f"{schemas.cdm}_role"
    conn.rollback()
    execute(conn, f"CREATE ROLE {role} NOLOGIN")
    execute(conn, f"GRANT USAGE ON SCHEMA {schemas.src}, {schemas.vocab}, {schemas.cdm}, {schemas.etl} TO {role}")
    execute(conn, f"GRANT SELECT ON ALL TABLES IN SCHEMA {schemas.src}, {schemas.vocab} TO {role}")
    execute(conn, f"GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA {schemas.cdm}, {schemas.etl} TO {role}")
    execute(conn, f"REVOKE SELECT ON {schemas.vocab}.concept_ancestor FROM {role}")
    conn.commit()
    map_drugs(conn, schemas)
    try:
        cfg = cfg.model_copy(update={"target": cfg.target.model_copy(update={"etl_role": role})})
        spec = fixed_ref()
        result = run_etl(conn, schemas, spec, cfg)
        (err,) = result.errors()
        assert err.step.endswith("_drug_era") and "permission denied" in err.error and not err.blocking
        assert result.ok and result.blocking_errors() == []
        assert count(conn, schemas, "condition_era") > 0 and count(conn, schemas, "drug_era") == 0
        assert {s.step.split("_", 1)[1] for s in result.steps if s.error is None} >= {"condition_era", "cdm_source"}
        report = run_builtin_checks(conn, schemas, cfg, spec, result)
        assert report.passed, report.blocking_failures  # the DQ checks run and the gate passes
        checks = {r.key: r for r in report.results}
        step_error = checks["etlStepError.drug_era"]
        assert (step_error.severity, step_error.repairable) == ("convention", False)
        assert checks["measureDrugEraCompleteness.drug_era"].failed
        assert repair_triggers(report, result, cfg) == ([], [r for r in repair_triggers(report, result, cfg)[1]
                                                             if r.table not in DERIVED_ERAS])
        assert not [r for r in repair_triggers(report, result, cfg)[1] if r.table in DERIVED_ERAS
                    or r.check_name == "etlStepError"]
    finally:
        conn.rollback()
        execute(conn, f"DROP OWNED BY {role}")
        execute(conn, f"DROP ROLE {role}")
        conn.commit()
    # the validator says the step cannot run, as a warning (the build goes on without it)
    execute(conn, f"ALTER TABLE {schemas.vocab}.concept_ancestor RENAME TO concept_ancestor_gone")
    conn.commit()
    try:
        issues = validate_spec(spec, None, conn, schemas, cfg=cfg.model_copy(
            update={"target": cfg.target.model_copy(update={"etl_role": None})}))
        conn.rollback()
        era = [i for i in issues if i.target_table == "drug_era"]
        assert [(i.severity, i.message.split(":")[0]) for i in era] == [
            ("warning", "step drug_era cannot run (the step would fail and be reported, without failing the "
                        "build)")], issues
        assert "concept_ancestor" in era[0].message and not [i for i in issues if i.severity == "error"]
    finally:
        execute(conn, f"ALTER TABLE {schemas.vocab}.concept_ancestor_gone RENAME TO concept_ancestor")
        conn.commit()


@pytest.mark.postgres
def test_eras_skipped_when_their_event_table_fails(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref()
    spec.mapping_for("drug_exposure").where = "1/0 = 1"
    result = run_etl(conn, schemas, spec, cfg)
    by = {s.step.split("_", 1)[1]: s for s in result.steps}
    assert by["drug_exposure"].error and not result.ok
    assert by["drug_era"].skipped == "a step loading drug_exposure failed" and by["drug_era"].error is None
    assert by["condition_era"].error is None and by["condition_era"].rows_inserted > 0
    # a rebuild empties the eras first
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    n = count(conn, schemas, "condition_era")
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok and count(conn, schemas, "condition_era") == n
