"""The mini HDSS fixture end to end: ``tests/fixtures/hdss/`` (one source row per named case,
listed in ``tests/hdss_cases.py``) built once with ``tests/fixtures/hdss_etl_spec.yaml``, the
fixture vocabulary plus ``hdss_vocab.sql`` and the reviewed mappings of ``hdss_stcm.sql`` (no LLM).
Every case is checked against the CDM, and so are the build's step statistics, the check results
and the gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from psycopg.rows import tuple_row

from omop_agent.cdm.ddl import create_cdm_tables, create_etl_tables
from omop_agent.config import ProjectConfig
from omop_agent.db import Schemas, connect, execute, scalar
from omop_agent.design.validate import validate_spec
from omop_agent.etl.compiler import step_label
from omop_agent.etl.runner import run_etl
from omop_agent.models import DQReport, ETLRunResult, ETLSpec, FieldMapping
from omop_agent.profiling.loader import stage_source
from omop_agent.profiling.profiler import profile_staging
from omop_agent.quality.checks import run_builtin_checks
from omop_agent.quality.repair import repair_triggers
from omop_agent.report.inspection import build_report
from omop_agent.vocab.loader import load_athena
from omop_agent.workspace import Workspace
from tests import hdss_cases as hc
from tests.conftest import DB_URL, FIXTURE_VOCAB, FIXTURES, _drop, _new_schemas

pytestmark = pytest.mark.postgres

HDSS = FIXTURES / "hdss"


def load_spec() -> ETLSpec:
    return ETLSpec.model_validate(yaml.safe_load((FIXTURES / "hdss_etl_spec.yaml").read_text()))


def hdss_config(work: Path, schemas: Schemas, **changes) -> ProjectConfig:
    return ProjectConfig.model_validate({
        "name": "hdss",
        "workdir": str(work),
        "source": {"type": "csv", "path": str(HDSS)},
        "target": {"url": DB_URL, "cdm_schema": schemas.cdm, "vocab_schema": schemas.vocab,
                   "staging_schema": schemas.src, "etl_schema": schemas.etl, "results_schema": schemas.results},
        "vocabulary": {"athena_dir": str(FIXTURE_VOCAB)},
        "llm": {"provider": "none"},
        **hc.CONFIG,
        **changes,
    })


@dataclass
class Built:
    conn: object
    schemas: Schemas
    cfg: ProjectConfig
    ws: Workspace
    spec: ETLSpec
    manifest: dict
    etl: ETLRunResult
    report: DQReport

    def rows(self, sql: str) -> list[tuple]:
        # rolled back, so a query that fails leaves the shared connection usable for the other tests
        with self.conn.transaction(force_rollback=True), self.conn.cursor(row_factory=tuple_row) as cur:  # type: ignore[attr-defined]
            cur.execute(self.schemas.render(sql))
            return list(cur.fetchall())

    def step(self, label: str):
        return next(s for s in self.etl.steps if step_label(s.step) == label)

    def results(self) -> dict:
        return {r.key: r for r in self.report.results}


def build(conn, schemas: Schemas, work: Path, spec: ETLSpec) -> Built:
    """Stage the fixture, load the vocabulary and the reviewed mappings, build and check."""
    cfg = hdss_config(work, schemas)
    ws = Workspace(cfg.workdir)
    manifest = stage_source(cfg, conn, schemas, ws)
    load_athena(conn, schemas, FIXTURE_VOCAB)
    execute(conn, schemas.render((FIXTURES / "hdss_vocab.sql").read_text()))
    create_cdm_tables(conn, schemas, include_vocab=False)
    create_etl_tables(conn, schemas)
    execute(conn, schemas.render((FIXTURES / "hdss_stcm.sql").read_text()))
    conn.commit()
    etl = run_etl(conn, schemas, spec, cfg, ws)
    report = run_builtin_checks(conn, schemas, cfg, spec, etl)
    ws.save_json(ws.dq_json, report)
    ws.save_yaml(ws.spec_yaml, spec)
    # the cases name people and visits by their source keys
    execute(conn, f"CREATE TEMP VIEW pk AS SELECT source_value AS key, person_id FROM {schemas.etl}.person_map")
    execute(conn, f"CREATE TEMP VIEW vk AS SELECT source_value AS key, visit_occurrence_id FROM {schemas.etl}.visit_map")
    conn.commit()
    return Built(conn, schemas, cfg, ws, spec, manifest, etl, report)


@pytest.fixture(scope="module")
def hdss(tmp_path_factory):
    schemas = _new_schemas()
    conn = connect(DB_URL)  # type: ignore[arg-type]
    try:
        yield build(conn, schemas, tmp_path_factory.mktemp("hdss") / "work", load_spec())
    finally:
        conn.rollback()
        conn.close()
        _drop(schemas)


@pytest.mark.parametrize("case", list(hc.CASES))
def test_case(hdss, case):
    for sql, expected in hc.CASES[case]:
        assert hdss.rows(sql) == expected, sql


def test_staging_and_the_spec(hdss):
    # the NA marker of H06's dob is staged as NULL, and counted; no other value is a marker
    markers = {(t, c["name"]): c["na_markers"] for t, m in hdss.manifest.items() for c in m["columns"]
               if c.get("na_markers")}
    assert markers == {("individuals", "dob"): {"NA": 1}}
    assert scalar(hdss.conn, f"SELECT COUNT(*) FROM {hdss.schemas.src}.individuals WHERE dob IS NULL") == 1
    # the spec is valid for the profiled source; without etl.timezone, the offsets of submitted_at ('+03:00',
    # and one 'Z', too rare to name) are an error
    profile = profile_staging(hdss.conn, hdss.schemas, hdss.cfg)
    issues = validate_spec(hdss.spec, profile, hdss.conn, hdss.schemas, cfg=hdss.cfg)
    assert [i for i in issues if i.severity == "error"] == [], issues
    unset = hdss.cfg.model_copy(update={"etl": hdss.cfg.etl.model_copy(update={"timezone": None})})
    errors = [i.message for i in validate_spec(hdss.spec, profile, hdss.conn, hdss.schemas, cfg=unset)
              if i.severity == "error"]
    assert len(errors) == 1 and errors[0].startswith("v.submitted_at holds timestamps written with several time zone "
                                                     "offsets ('+03:00' and others") and "etl.timezone" in errors[0]
    hdss.conn.rollback()


def test_steps(hdss):
    etl = hdss.etl
    assert etl.ok and etl.errors() == [], [(s.step, s.error) for s in etl.errors()]
    assert [step_label(s.step) for s in etl.steps] == [
        "person_map", "person_map_2", "visit_map", "visit_map_2", "person", "person_2", "person_map_prune",
        "visit_occurrence", "visit_occurrence_2", "visit_map_prune", "condition_occurrence", "drug_exposure",
        "measurement", "observation", "death", "death_2", "route_prepare", "route_condition_occurrence",
        "route_drug_exposure", "route_measurement", "observation_period", "observation_period_merge",
        "observation_period_fill", "condition_era", "drug_era", "cdm_source"]
    # id maps: every key of the register, the clinic patients not in the HDSS, the outpatient visits and the
    # admissions; the pruning retires H06 and H07, and the visits 1006 (H06), 1010 (invalid) and 1011 (EMERG)
    assert [hdss.step(s).rows_inserted for s in ("person_map", "person_map_2", "visit_map", "visit_map_2",
                                                 "person_map_prune", "visit_map_prune")] == [30, 4, 30, 2, 2, 3]
    for label, counts in hc.STEP_COUNTS.items():
        s = hdss.step(label)
        assert (s.source_rows, s.id_retained_rows, s.retained_rows, s.rows_inserted) == counts, label
        assert s.value_counts == hc.VALUE_COUNTS[label], label
    for label, stats in hc.KEY_STATS.items():
        assert hdss.step(label).stats == stats, label
    for label, links in hc.VISIT_LINKS.items():
        assert hdss.step(label).stats["visit_links"] == links, label
    for label, moved in hc.MOVED.items():
        s = hdss.step(label)
        assert s.error is None and [(m.to_table, m.domain_id, m.moved_rows) for m in s.moved] == moved, label
        assert all(m.rows_with_dropped_fields == 0 for m in s.moved)
    assert hdss.step("observation_period_merge").stats == hc.MERGE_STATS
    assert hdss.step("observation_period_fill").stats == hc.FILL_STATS
    assert {t: hdss.step(t).rows_inserted for t in hc.DERIVED_ROWS} == hc.DERIVED_ROWS
    assert etl.rows_moved() == 4
    # routed records get ids after every id their destination had: none is given twice
    assert hdss.rows("SELECT COUNT(*), COUNT(DISTINCT observation_id), MAX(observation_id) FROM {cdm}.observation") == [
        (9, 9, 9)]
    assert hdss.rows("SELECT source_release_date FROM {cdm}.cdm_source") == [(hc.RELEASE,)]


def test_checks_and_gate(hdss):
    report, results = hdss.report, hdss.results()
    failed = {r.key: r.severity for r in report.results if (r.failed or r.error) and not r.not_applicable}
    assert failed == hc.FAILED_CHECKS
    assert report.passed and report.blocking_failures == []
    # the integrity checks hold on data that tries them: a visit of another person, two deaths of one person,
    # overlapping episodes, a visit key shared by an outpatient visit and an admission
    for key in ("fkVisitPerson.condition_occurrence.visit_occurrence_id", "measureDuplicateDeaths.death",
                "measureObservationPeriodOverlap.observation_period",
                "isPrimaryKey.visit_occurrence.visit_occurrence_id", "isPrimaryKey.person.person_id",
                "measurePersonCompleteness.observation_period", "measureDrugEraCompleteness.drug_era",
                "measureConditionEraCompleteness.condition_era"):
        r = results[key]
        assert not r.not_applicable and not r.failed and r.num_violated_rows == 0, key
    # H03's rows disagree on sex, but its dedup_order picks a row: a conflict, not an undecided key
    conflict = results["etlKeyConflict.person"]
    assert not conflict.failed and "1 of 28 keys" in conflict.description and "For 0 of 28 keys" in conflict.description
    # not for the repair agent: they are for people (vocabulary, release date, route-only completeness)
    for key in ("etlKeyFanOut.death.death_2", "etlSourceReleaseDate.observation_period.observation_period_merge",
                "measurePersonCompleteness.procedure_occurrence", "etlDuplicateRecords.person",
                "measureDrugEraCoverage.drug_era"):
        assert results[key].repairable is False, key
    # an unreadable value inside a COALESCE does not make the field NULL: the description says so, and a required
    # field none of whose rows was NULL is characterization, so the repair agent is not told to change a right
    # expression
    end = results["etlInvalidValue.drug_exposure.drug_exposure_end_date"]
    assert "unless its expression falls back to another value" in end.description and "loaded as NULL" not in end.description
    assert "no row was left with this required field NULL" in end.description
    assert hdss.rows("SELECT COUNT(*) FROM {cdm}.drug_exposure WHERE drug_exposure_end_date IS NULL") == [(0,)]
    _, triggers = repair_triggers(report, hdss.etl, hdss.cfg, hdss.spec)
    assert triggers and end.key not in {r.key for r in triggers}
    # H20's exit after the configured release date: counted (descriptions give counts, never a person's key)
    late = results["etlSourceReleaseDate.observation_period.observation_period_merge"].description
    assert "1 of 32 episodes" in late and "H20" not in late
    # moves are reported, never failed
    assert [(r.key, r.failed, r.num_violated_rows) for r in report.results if r.check_name == "etlDomainRouting"] == [
        ("etlDomainRouting.condition_occurrence.route_condition_occurrence", False, 3),
        ("etlDomainRouting.measurement.route_measurement", False, 1)]
    # the result list stays a few hundred rows, most of them applicable (measured: 757, 208 not applicable)
    n_na = sum(r.not_applicable for r in report.results)
    assert len(report.results) < 850 and n_na < len(report.results) / 2


def test_report(hdss):
    md_path, html_path = build_report(hdss.cfg, hdss.ws)
    md = md_path.read_text()
    assert "PASSED" in md and "PASSED" in html_path.read_text()
    assert "Records moved to the table of their concept's domain" in md
    assert "Values not loaded as written" in md and "offset" in md.lower()
    saved = json.loads(hdss.ws.etl_run_json.read_text())
    assert len(saved["steps"]) == len(hdss.etl.steps)


def test_shared_visit_key_still_fails_is_primary_key(tmp_path):
    """Without the IPD: prefix, admissions 1001 and 1002 have the keys of H01's outpatient visits: the build
    loads each visit once per person, so the visit key is not unique and the fatal isPrimaryKey blocks."""
    spec = load_spec()
    ipd = spec.table_mappings[4]
    assert ipd.source.table == "ipd_wards"
    key = next(f for f in ipd.fields if f.target_field == "visit_occurrence_id")
    ipd.fields[ipd.fields.index(key)] = FieldMapping(target_field="visit_occurrence_id", kind="visit_id",
                                                     expression="w.admission_no")
    schemas = _new_schemas()
    conn = connect(DB_URL)  # type: ignore[arg-type]
    try:
        built = build(conn, schemas, tmp_path / "work", spec)
        results = built.results()
        pk = results["isPrimaryKey.visit_occurrence.visit_occurrence_id"]
        assert pk.failed and pk.severity == "fatal" and pk.num_violated_rows == 4
        assert not built.report.passed
        assert any(b.startswith("isPrimaryKey.visit_occurrence.visit_occurrence_id") for b in built.report.blocking_failures)
        assert built.step("visit_occurrence_2").stats["conflict_keys"] == 2
        assert results["etlKeyConflict.visit_occurrence.visit_occurrence_2"].num_violated_rows == 2
        assert built.rows("SELECT vk.key, pk.key, v.visit_concept_id FROM {cdm}.visit_occurrence v "
                          "JOIN vk USING (visit_occurrence_id) JOIN pk USING (person_id) "
                          "WHERE vk.key IN ('1001', '1002') ORDER BY 1, 3") == [
            ("1001", "H02", 9201), ("1001", "H01", 9202), ("1002", "H10", 9201), ("1002", "H01", 9202)]
    finally:
        conn.rollback()
        conn.close()
        _drop(schemas)
