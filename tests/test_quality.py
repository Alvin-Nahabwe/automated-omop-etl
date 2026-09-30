"""Quality stage: built-in DQD checks and the gate, the R DQD bridge, and the repair agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from omop_agent.config import ProjectConfig
from omop_agent.db import execute, qi, scalar
from omop_agent.llm.base import NullLLM
from omop_agent.llm.replay import ReplayLLM, save_recording
from omop_agent.models import (
    CheckResult,
    ColumnProfile,
    DQReport,
    ETLRunResult,
    ETLSpec,
    SourceProfile,
    TableLoadResult,
    TableProfile,
)
from omop_agent.quality import dqd_r
from omop_agent.quality.checks import etl_checks, gate, run_builtin_checks, translate_sqlrender
from omop_agent.quality.repair import RepairState, build_repair_tools, redact_error, repair_spec, repair_triggers
from omop_agent.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
VISIT_END_FIX = "COALESCE(v.discharge_date::date, v.visit_date::date)"
NOT_NULL_ERROR = 'null value in column "visit_end_date" of relation "visit_occurrence" violates not-null constraint'


def load_ref() -> ETLSpec:
    return ETLSpec.model_validate(yaml.safe_load((FIXTURES / "demo_etl_spec.yaml").read_text()))


def plain_config(tmp_path: Path, **quality) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "name": "t",
            "workdir": str(tmp_path / "work"),
            "source": {"type": "csv", "path": str(tmp_path)},
            "target": {"url": "postgresql://omop:s3cret@db.example:6543/omop_prod", "cdm_schema": "cdm_x"},
            "quality": quality,
        }
    )


def check(results: list[CheckResult], key: str) -> CheckResult:
    matches = [r for r in results if r.key == key]
    assert len(matches) == 1, f"{key}: {len(matches)} results"
    return matches[0]


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------


def test_translate_sqlrender():
    assert translate_sqlrender("'19500101'") == "(DATE '1950-01-01')"
    assert translate_sqlrender("DATEADD(dd,1,GETDATE())") == "((CURRENT_DATE + 1))"
    assert translate_sqlrender("YEAR(GETDATE())+1") == "(EXTRACT(YEAR FROM CURRENT_DATE)+1)"
    assert translate_sqlrender("1850") == "(1850)"
    with pytest.raises(ValueError):
        translate_sqlrender("(SELECT 1); DROP TABLE person")


def test_gate(tmp_path):
    cfg = plain_config(tmp_path)

    def res(name, severity, **kw):
        return CheckResult(check_name=name, check_level="FIELD", table="person", field="x", description="d",
                           severity=severity, **kw)

    ok = [res("a", "fatal"), res("b", "convention", failed=True), res("c", "characterization", failed=True),
          res("d", "fatal", failed=True, not_applicable=True)]
    assert gate(ok, cfg) == (True, [])
    passed, blocking = gate(ok + [res("e", "fatal", failed=True, num_violated_rows=2, num_denominator_rows=8,
                                      pct_violated_rows=0.25)], cfg)
    assert not passed and blocking == ["e.person.x (fatal): 2 of 8 (25.00%) violated, threshold 0%"]
    passed, blocking = gate(ok + [res("f", "fatal", error="boom")], cfg)
    assert not passed and "ERROR boom" in blocking[0]
    strict = plain_config(tmp_path, blocking_severities=["fatal", "convention"])
    assert gate(ok, strict)[1] == ["b.person.x (convention): 0 of 0 (0.00%) violated, threshold 0%"]


def test_etl_checks_and_gate(tmp_path):
    cfg = plain_config(tmp_path)
    etl = ETLRunResult(
        started_at=NOW,
        steps=[
            TableLoadResult(target_table="person_map", step="01_person_map", rows_inserted=100),
            TableLoadResult(target_table="person", step="03_person", rows_inserted=100, source_rows=100),
            TableLoadResult(target_table="visit_occurrence", step="04_visit_occurrence", rows_inserted=90, source_rows=100),
            TableLoadResult(target_table="condition_occurrence", step="05_condition_occurrence", rows_inserted=95,
                            source_rows=100),
            TableLoadResult(target_table="measurement", step="06_measurement", rows_inserted=120, source_rows=100),
            TableLoadResult(target_table="drug_exposure", step="07_drug_exposure", error="boom", sql="INSERT ...",
                            source_rows=50),
        ],
    )
    results = etl_checks(etl)
    assert [(r.check_name, r.table) for r in results] == [
        ("etlRecordRetention", "person"),
        ("etlRecordRetention", "visit_occurrence"),
        ("etlRecordRetention", "condition_occurrence"),
        ("etlRecordRetention", "measurement"),
        ("etlStepError", "drug_exposure"),
    ]
    visit = results[1]
    assert (visit.num_violated_rows, visit.num_denominator_rows, visit.pct_violated_rows) == (10, 100, 0.1)
    assert visit.failed and visit.severity == "convention" and visit.check_level == "ETL"
    assert "04_visit_occurrence" in visit.description and "dropped" in visit.description
    assert not results[2].failed  # exactly 5% is not more than the 5% threshold
    assert results[3].num_violated_rows == 0 and not results[3].failed
    err = results[4]
    assert err.failed and err.error == "boom" and err.severity == "fatal" and err.sql == "INSERT ..."

    passed, blocking = gate(results, cfg, etl)
    assert not passed and len(blocking) == 1 and blocking[0].startswith("etlStepError.drug_exposure")
    # the ETL error blocks even when the checks were computed without it
    passed, blocking = gate(results[:4], cfg, etl)
    assert not passed and blocking == ["etlStepError.drug_exposure (step 07_drug_exposure): boom"]
    assert gate(results[:4], cfg) == (True, [])


def test_redact_error():
    msg = redact_error(NOT_NULL_ERROR + " DETAIL: Failing row contains (1, 2, 9202, 2019-04-22, null).", "INSERT ...")
    assert msg == NOT_NULL_ERROR + " DETAIL: (withheld: may contain record values)"
    assert redact_error('invalid input syntax for type date: "31/02/1990"', "SELECT d.dob::date") == (
        'invalid input syntax for type date: "<value withheld>"'
    )
    assert redact_error('column "dx_code" does not exist', "SELECT d.dx_code") == 'column "dx_code" does not exist'
    assert redact_error('duplicate key value violates unique constraint "xpk_person" DETAIL: Key (person_id)=(42) '
                        "already exists. HINT: h") == (
        'duplicate key value violates unique constraint "xpk_person" DETAIL: (withheld: may contain record values) HINT: h'
    )
    assert redact_error("date field value out of range: 1800-13-01") == "date field value out of range: <value withheld>"


def test_repair_triggers(tmp_path):
    cfg = plain_config(tmp_path)

    def res(name, severity, pct, **kw):
        return CheckResult(check_name=name, check_level="FIELD", table="t", description="d", severity=severity,
                           pct_violated_rows=pct, **kw)

    dq = DQReport(run_at=NOW, passed=False, results=[
        res("conv", "convention", 0.5, failed=True),
        res("fatal_small", "fatal", 0.01, failed=True),
        res("fatal_big", "fatal", 0.9, failed=True),
        res("char", "characterization", 1.0, failed=True),
        res("na", "fatal", 1.0, failed=True, not_applicable=True),
        res("err", "convention", 0.0, error="x"),
        res("ok", "fatal", 0.0),
    ])
    etl_errors, checks = repair_triggers(dq, None, cfg)
    assert etl_errors == [] and [r.check_name for r in checks] == ["fatal_big", "fatal_small", "conv", "err"]
    broken = ETLRunResult(started_at=NOW, steps=[TableLoadResult(target_table="x", step="01_x", error="e")])
    etl_errors, checks = repair_triggers(dq, broken, cfg)
    assert [s.step for s in etl_errors] == ["01_x"] and checks == []


# --------------------------------------------------------------------------------------
# R DataQualityDashboard bridge (R is not needed: rendering and parsing only)
# --------------------------------------------------------------------------------------


def test_write_r_script_has_no_password(tmp_path):
    cfg = plain_config(tmp_path)
    ws = Workspace(tmp_path / "work")
    path = dqd_r.write_r_script(cfg, ws)
    assert path == ws.root / "r" / "run_achilles_dqd.R"
    text = path.read_text()
    assert "s3cret" not in text and "{{" not in text
    assert 'Sys.getenv("OMOP_AGENT_DB_PASSWORD")' in text
    assert 'server          <- "db.example/omop_prod"' in text
    assert "port            <- 6543" in text and 'user            <- "omop"' in text
    assert 'cdmSchema       <- "cdm_x"' in text and 'vocabSchema     <- "cdm"' in text
    assert f'outputFolder    <- "{(ws.root / "dqd").resolve()}"' in text
    assert 'resultsFile     <- "dqd_results.json"' in text
    assert "DataQualityDashboard::executeDqChecks(" in text and "Achilles::achilles(" in text
    assert dqd_r.dqd_results_path(ws) == ws.root / "dqd" / "dqd_results.json"


def test_r_unavailable(tmp_path):
    cfg = plain_config(tmp_path, rscript="no-such-rscript-binary")
    assert not dqd_r.r_available(cfg)
    assert dqd_r.run_r_dqd(cfg, Workspace(tmp_path / "work")) is None


def test_parse_dqd_json(tmp_path):
    sample = {
        "startTimestamp": "2024-01-01 10:00:00",
        "Metadata": [{"cdmSourceName": "demo"}],
        "CheckResults": [
            {
                "numViolatedRows": 12, "pctViolatedRows": 0.012, "numDenominatorRows": 1000,
                "queryText": "SELECT num_violated_rows FROM ...", "checkName": "isRequired", "checkLevel": "FIELD",
                "checkDescription": "The number and percent of records with a NULL value in the VISIT_END_DATE ...",
                "cdmTableName": "VISIT_OCCURRENCE", "cdmFieldName": "VISIT_END_DATE", "category": "Conformance",
                "subcategory": "Relational", "context": "Validation", "failed": 1, "passed": 0, "isError": 0,
                "notApplicable": 0, "notApplicableReason": "NA", "thresholdValue": 0,
            },
            {
                "numViolatedRows": 3, "pctViolatedRows": 0.3, "numDenominatorRows": 10,
                "checkName": "plausibleGender", "checkLevel": "CONCEPT", "cdmTableName": "CONDITION_OCCURRENCE",
                "cdmFieldName": "CONDITION_CONCEPT_ID", "conceptId": 194997, "failed": 1, "thresholdValue": 5,
                "category": "Plausibility", "subcategory": "Atemporal",
            },
            {
                "checkName": "measurePersonCompleteness", "checkLevel": "TABLE", "cdmTableName": "DEATH",
                "numViolatedRows": "NA", "pctViolatedRows": "NA", "isError": 1,
                "error": "relation \"death\" does not exist", "failed": 0, "thresholdValue": "NA",
            },
            {"checkName": "isPrimaryKey", "cdmTableName": "NOTE", "cdmFieldName": "NOTE_ID", "notApplicable": 1},
        ],
    }
    path = tmp_path / "dqd_results.json"
    path.write_text(json.dumps(sample))
    cfg = plain_config(tmp_path)
    results = dqd_r.parse_dqd_json(path, cfg)
    assert len(results) == 4
    req, gender, death, note = results
    assert (req.table, req.field, req.check_level, req.severity) == ("visit_occurrence", "visit_end_date", "FIELD", "fatal")
    assert (req.num_violated_rows, req.num_denominator_rows, req.pct_violated_rows) == (12, 1000, 0.012)
    assert req.failed and req.sql and req.category == "Conformance" and req.key == "isRequired.visit_occurrence.visit_end_date"
    assert (gender.check_level, gender.concept_id, gender.severity, gender.threshold_pct) == ("CONCEPT", 194997, "characterization", 5)
    assert death.error and "does not exist" in death.error and death.num_violated_rows == 0 and not death.failed
    assert death.severity == "convention" and death.field is None
    assert note.not_applicable and note.check_level == "FIELD" and note.severity == "fatal"
    passed, blocking = gate(results, cfg)
    assert not passed and blocking == [
        "isRequired.visit_occurrence.visit_end_date (fatal): 12 of 1000 (1.20%) violated, threshold 0%"
    ]


# --------------------------------------------------------------------------------------
# Built-in checks on PostgreSQL
# --------------------------------------------------------------------------------------


def _load_cdm_rows(conn, schemas) -> None:
    """A tiny hand-made CDM with one deliberate problem per check under test."""
    c = qi(schemas.cdm)
    execute(conn, f"ALTER TABLE {c}.visit_occurrence ALTER COLUMN visit_end_date DROP NOT NULL")
    execute(conn, f"""INSERT INTO {c}.person (person_id, gender_concept_id, year_of_birth, month_of_birth, day_of_birth,
        birth_datetime, race_concept_id, ethnicity_concept_id, person_source_value, gender_source_value) VALUES
        (1, 8507, 1980, 5, 1, '1980-05-01', 0, 0, 'P1', '1'),
        (2, 8532, 1800, NULL, NULL, NULL, 0, 0, 'P2', '2'),       -- born before 1850
        (3, 8507, 1990, NULL, NULL, NULL, 0, 0, 'P3', '1'),
        (4, 0, 2000, NULL, NULL, NULL, 0, 0, 'P4', '9')           -- unmapped gender""")
    execute(conn, f"""INSERT INTO {c}.observation_period VALUES
        (1, 1, '2000-01-01', '2010-12-31', 32817),
        (2, 1, '2010-06-01', '2015-12-31', 32817),                -- overlaps period 1
        (3, 2, '2000-01-01', '2020-12-31', 32817),
        (4, 3, '2000-01-01', '2020-12-31', 32817),
        (5, 4, '2000-01-01', '2020-12-31', 32817)""")
    execute(conn, f"""INSERT INTO {c}.visit_occurrence (visit_occurrence_id, person_id, visit_concept_id,
        visit_start_date, visit_end_date, visit_type_concept_id, visit_source_value) VALUES
        (1, 1, 9202, '2012-01-01', '2012-01-01', 32817, 'OPD'),
        (1, 1, 9202, '2012-02-01', '2012-02-01', 32817, 'OPD'),   -- duplicate primary key
        (2, 99, 9201, '2012-03-01', '2012-03-05', 32817, 'IPD'),  -- person 99 does not exist
        (3, 3, 9202, '1985-01-01', '1985-01-01', 32817, 'OPD'),   -- before birth (1990)
        (4, 2, 9203, '2013-01-01', NULL, 32817, 'EMERG')          -- missing end date""")
    execute(conn, f"""INSERT INTO {c}.condition_occurrence (condition_occurrence_id, person_id, condition_concept_id,
        condition_start_date, condition_type_concept_id, condition_source_value, condition_source_concept_id) VALUES
        (1, 1, 2000001001, '2012-01-01', 32817, 'B54', 2000002001),
        (2, 1, 2000002002, '2012-01-01', 32817, 'J18.9', 2000002002),  -- non-standard ICD10 concept
        (3, 2, 0, '2012-01-01', 32817, 'J189', 0),                     -- unmapped
        (4, 3, 2000001010, '2012-01-01', 32817, 'V89', 0),             -- Observation domain, after death
        (5, 3, 2000001003, '2012-01-01', 32817, 'A09', 2000002003)     -- after death""")
    execute(conn, f"INSERT INTO {c}.death (person_id, death_date, death_type_concept_id) VALUES (3, '2011-01-01', 32817)")
    conn.commit()


@pytest.mark.postgres
def test_builtin_checks(staged):
    conn, schemas, cfg = staged
    _load_cdm_rows(conn, schemas)
    report = run_builtin_checks(conn, schemas, cfg, load_ref())
    assert report.engine == "builtin" and conn.info.transaction_status == 0  # left idle
    results = report.results
    applicable = [r for r in results if not r.not_applicable]
    assert not [r for r in applicable if r.error], [r.error for r in applicable if r.error]

    # fatal problems block the gate, and nothing else does
    fatal = {r.key for r in applicable if r.failed and r.severity == "fatal"}
    assert fatal == {
        "isPrimaryKey.visit_occurrence.visit_occurrence_id",
        "isForeignKey.visit_occurrence.person_id",
        "isRequired.visit_occurrence.visit_end_date",
        "measureObservationPeriodOverlap.observation_period",
    }
    assert not report.passed and len(report.blocking_failures) == 4

    pk = check(results, "isPrimaryKey.visit_occurrence.visit_occurrence_id")
    assert (pk.num_violated_rows, pk.num_denominator_rows, pk.pct_violated_rows) == (2, 5, 0.4)
    assert pk.sql and "GROUP BY" in pk.sql and "does not exist" not in pk.description
    fk = check(results, "isForeignKey.visit_occurrence.person_id")
    assert fk.num_violated_rows == 1 and "person table" in fk.description
    overlap = check(results, "measureObservationPeriodOverlap.observation_period")
    assert (overlap.num_violated_rows, overlap.num_denominator_rows, overlap.check_level) == (1, 4, "TABLE")

    # concept checks read the vocabulary schema
    std = check(results, "isStandardValidConcept.condition_occurrence.condition_concept_id")
    assert std.failed and std.num_violated_rows == 1 and std.severity == "convention"
    domain = check(results, "fkDomain.condition_occurrence.condition_concept_id")
    assert domain.failed and domain.num_violated_rows == 1 and "Condition" in domain.description
    assert not check(results, "isForeignKey.condition_occurrence.condition_concept_id").failed
    zero = check(results, "standardConceptRecordCompleteness.condition_occurrence.condition_concept_id")
    assert zero.failed and (zero.num_violated_rows, zero.threshold_pct) == (1, 5)
    gender = check(results, "standardConceptRecordCompleteness.person.gender_concept_id")
    assert gender.failed and gender.pct_violated_rows == 0.25
    assert not check(results, "standardConceptRecordCompleteness.person.race_concept_id").failed  # threshold 100
    source_values = check(results, "sourceValueCompleteness.condition_occurrence.condition_source_value")
    assert source_values.failed and (source_values.num_violated_rows, source_values.num_denominator_rows) == (1, 5)
    assert check(results, "sourceConceptRecordCompleteness.condition_occurrence.condition_source_concept_id").num_violated_rows == 2

    # plausibility (characterization)
    low = check(results, "plausibleValueLow.person.year_of_birth")
    assert low.failed and low.num_violated_rows == 1 and low.severity == "characterization"
    assert "1850" in low.description and "< (1850)" in (low.sql or "")
    assert not check(results, "plausibleValueHigh.visit_occurrence.visit_start_date").failed
    birth = check(results, "plausibleAfterBirth.visit_occurrence.visit_start_date")
    assert birth.failed and (birth.num_violated_rows, birth.num_denominator_rows) == (1, 5)
    death = check(results, "plausibleBeforeDeath.condition_occurrence.condition_start_date")
    assert death.failed and (death.num_violated_rows, death.num_denominator_rows) == (2, 2)
    start_end = check(results, "plausibleStartBeforeEnd.visit_occurrence.visit_start_date")
    assert not start_end.failed and start_end.num_denominator_rows == 4
    assert check(results, "isRequired.visit_occurrence.visit_end_date").num_violated_rows == 1
    assert check(results, "measureValueCompleteness.visit_occurrence.visit_end_date").severity == "characterization"

    # table coverage
    assert not check(results, "measurePersonCompleteness.visit_occurrence").failed  # 1 of 4 persons, threshold 95%
    drug = check(results, "measurePersonCompleteness.drug_exposure")  # mapped by the spec but empty
    assert drug.failed and drug.pct_violated_rows == 1.0 and not drug.not_applicable
    assert all(r.not_applicable for r in results if r.table == "drug_exposure" and r.check_level == "FIELD")
    procedure = [r for r in results if r.table == "procedure_occurrence"]  # not in the spec
    assert procedure and all(r.not_applicable and not r.sql for r in procedure)
    # all-NULL fields are not applicable except for completeness checks
    assert check(results, "plausibleValueLow.visit_occurrence.visit_end_datetime").not_applicable
    assert not check(results, "measureValueCompleteness.visit_occurrence.visit_end_datetime").not_applicable


@pytest.mark.postgres
def test_builtin_checks_without_spec_and_sql_errors(staged):
    conn, schemas, cfg = staged
    _load_cdm_rows(conn, schemas)
    etl = ETLRunResult(started_at=NOW, steps=[
        TableLoadResult(target_table="visit_occurrence", step="04_visit_occurrence", rows_inserted=5, source_rows=10),
    ])
    report = run_builtin_checks(conn, schemas, cfg, None, etl)
    # without a spec, "populated" means "has rows"
    assert check(report.results, "measurePersonCompleteness.drug_exposure").not_applicable
    assert not check(report.results, "measurePersonCompleteness.death").not_applicable
    retention = check(report.results, "etlRecordRetention.visit_occurrence")
    assert retention.failed and retention.pct_violated_rows == 0.5

    # a broken check is reported as an error and does not stop the others
    execute(conn, f"ALTER TABLE {qi(schemas.vocab)}.concept RENAME TO concept_gone")
    conn.commit()
    report = run_builtin_checks(conn, schemas, cfg, load_ref())
    broken = check(report.results, "isForeignKey.person.gender_concept_id")
    assert broken.error and "concept" in broken.error and broken.sql
    assert not check(report.results, "isPrimaryKey.person.person_id").error
    assert any(b.startswith("isForeignKey.person.gender_concept_id (fatal): ERROR") for b in report.blocking_failures)


# --------------------------------------------------------------------------------------
# Repair agent
# --------------------------------------------------------------------------------------


def _profile() -> SourceProfile:
    def col(name, pii=False):
        return ColumnProfile(name=name, inferred_type="string", n_rows=400, n_null=0, n_distinct=10, unique=False,
                             is_pii=pii, pii_reason="column name" if pii else None)

    return SourceProfile(
        source_type="csv", generated_at=NOW, min_cell_count=5,
        tables=[TableProfile(name="individuals", n_rows=400,
                             columns=[col("individual_id"), col("sex"), col("first_name", pii=True), col("phone", pii=True)])],
    )


@pytest.fixture
def repair_env(staged):
    conn, schemas, cfg = staged
    state = RepairState(spec=load_ref(), conn=conn, schemas=schemas, cfg=cfg, profile=_profile(), iteration=1,
                        actions=[], trigger="isRequired.visit_occurrence.visit_end_date")
    tools = {t.name: t for t in build_repair_tools(state)}
    assert set(tools) == {"get_table_mapping", "get_cdm_table", "get_source_profile", "run_aggregate_query",
                          "update_field_mapping", "replace_table_mapping"}
    return state, tools


@pytest.mark.postgres
def test_run_aggregate_query(repair_env):
    state, tools = repair_env
    q = tools["run_aggregate_query"].invoke

    out = q({"sql": "SELECT CASE WHEN individual_id = 'KH000001' THEN 'tiny_group' ELSE 'rest' END AS grp, "
                    "COUNT(*) AS n FROM {src}.individuals GROUP BY 1 ORDER BY 1;"})
    assert "rest" in out and "399" in out and "tiny_group" not in out
    assert "1 row(s) withheld because n < 5" in out

    assert q({"sql": "SELECT COUNT(*) AS n FROM {src}.individuals WHERE false GROUP BY sex"}) == "0 rows."
    refused = q({"sql": "SELECT first_name, COUNT(*) AS n FROM {src}.individuals GROUP BY 1"})
    assert refused.startswith("ERROR: refused") and "first_name" in refused
    assert "PII" in q({"sql": 'SELECT "Phone", COUNT(*) AS n FROM {src}.individuals GROUP BY 1'})
    assert "named n" in q({"sql": "SELECT sex, COUNT(*) AS cnt FROM {src}.individuals GROUP BY 1"})
    assert q({"sql": "DELETE FROM {src}.individuals"}).startswith("ERROR: only a single")
    assert q({"sql": "SELECT 1 AS n; SELECT 2 AS n"}).startswith("ERROR: only one statement")
    assert "SELECT *" in q({"sql": "SELECT *, 100 AS n FROM {src}.individuals"})
    assert "string_agg" in q({"sql": "SELECT string_agg(sex, ',') AS s, 100 AS n FROM {src}.individuals"})
    assert q({"sql": "SELECT no_such_column, COUNT(*) AS n FROM {src}.individuals GROUP BY 1"}).startswith("ERROR: column")
    # values quoted in error messages are withheld
    out = q({"sql": "SELECT individual_id::int AS x, COUNT(*) AS n FROM {src}.individuals GROUP BY 1"})
    assert out.startswith("ERROR: invalid input syntax") and "<value withheld>" in out and "KH0" not in out
    # data-modifying CTEs pass the text checks but fail in the read-only transaction
    out = q({"sql": "WITH d AS (DELETE FROM {src}.individuals RETURNING 1) SELECT COUNT(*) AS n FROM d"})
    assert out.startswith("ERROR:") and "read-only" in out
    assert scalar(state.conn, f"SELECT COUNT(*) FROM {qi(state.schemas.src)}.individuals") == 400

    state.profile = None  # without a profile the configured name patterns guard identifiers
    assert "PII" in q({"sql": "SELECT last_name, COUNT(*) AS n FROM {src}.individuals GROUP BY 1"})


@pytest.mark.postgres
def test_read_tools(repair_env):
    state, tools = repair_env
    text = tools["get_table_mapping"].invoke({"target_table": "visit_occurrence"})
    assert "target_field: visit_end_date" in text and "INSERT INTO" in text and state.schemas.cdm in text
    assert tools["get_table_mapping"].invoke({"target_table": "specimen"}).startswith("ERROR: no mapping")
    assert "visit_end_date" in tools["get_cdm_table"].invoke({"table": "visit_occurrence"})
    assert tools["get_cdm_table"].invoke({"table": "nope"}).startswith("ERROR")
    profile = tools["get_source_profile"].invoke({"table": "individuals"})
    assert "first_name" in profile and "PII - values withheld" in profile
    assert tools["get_source_profile"].invoke({"table": "nope"}).startswith("ERROR")
    state.profile = None
    assert tools["get_source_profile"].invoke({"table": "individuals"}) == "profile not available"
    with pytest.raises(Exception):  # unknown input fields are rejected by the input model
        tools["get_cdm_table"].invoke({"table": "person", "extra": 1})


@pytest.mark.postgres
def test_update_field_mapping(repair_env):
    state, tools = repair_env
    original = state.spec
    update = tools["update_field_mapping"].invoke

    out = update({"target_table": "visit_occurrence", "target_field": "visit_end_date", "kind": "expression",
                  "expression": "v.no_such_column::date", "rationale": "typo"})
    assert out.startswith("REJECTED: EXPLAIN failed") and "no_such_column" in out
    assert update({"target_table": "visit_occurrence", "target_field": "bogus", "kind": "null",
                   "rationale": "x"}).startswith("REJECTED")
    assert update({"target_table": "specimen", "target_field": "specimen_id", "kind": "auto_id",
                   "rationale": "x"}).startswith("REJECTED: no mapping")
    assert update({"target_table": "visit_occurrence", "target_field": "visit_end_date", "kind": "vocab_code",
                   "expression": "v.visit_date", "rationale": "x"}).startswith("REJECTED: does not compile")
    assert state.spec is original and state.actions == []

    out = update({"target_table": "visit_occurrence", "target_field": "visit_end_date", "kind": "expression",
                  "expression": VISIT_END_FIX, "rationale": "Outpatient visits end on their start date."})
    assert out.startswith("OK:")
    fm = next(f for f in state.spec.mapping_for("visit_occurrence").fields if f.target_field == "visit_end_date")
    assert fm.expression == VISIT_END_FIX and fm.rationale.startswith("Outpatient")
    old = next(f for f in original.mapping_for("visit_occurrence").fields if f.target_field == "visit_end_date")
    assert old.expression == "v.discharge_date::date"  # the input spec is not mutated
    (action,) = state.actions
    assert (action.iteration, action.target_table, action.target_field) == (1, "visit_occurrence", "visit_end_date")
    assert action.change == f"expression: v.discharge_date::date → expression: {VISIT_END_FIX}"
    assert action.trigger == "isRequired.visit_occurrence.visit_end_date"

    # a field that was not mapped yet is added
    out = update({"target_table": "visit_occurrence", "target_field": "visit_end_datetime", "kind": "expression",
                  "expression": "v.visit_date::timestamp", "rationale": "x"})
    assert out.startswith("OK:") and state.actions[-1].change.startswith("(unmapped) → expression")


@pytest.mark.postgres
def test_replace_table_mapping(repair_env):
    state, tools = repair_env
    replace = tools["replace_table_mapping"].invoke
    tm = state.spec.mapping_for("condition_occurrence").model_dump(mode="json")
    normalised = "CASE WHEN d.icd10 ~ '^[A-Z][0-9]{3}$' THEN substr(d.icd10,1,3)||'.'||substr(d.icd10,4) ELSE d.icd10 END"
    for fm in tm["fields"]:
        if fm["target_field"] == "condition_concept_id":
            fm["expression"] = normalised
    tm["where"] = "d.icd10 IS NOT NULL"

    assert replace({"target_table": "condition_occurrence", "mapping_yaml": "fields: [", "rationale": "x"}).startswith(
        "REJECTED: invalid YAML")
    assert "not 'condition_occurrence'" in replace({"target_table": "condition_occurrence",
                                                    "mapping_yaml": "target_table: person", "rationale": "x"})
    assert replace({"target_table": "condition_occurrence", "mapping_yaml": "source: 3",
                    "rationale": "x"}).startswith("REJECTED: not a valid TableMapping")
    assert replace({"target_table": "condition_occurrence", "mapping_yaml": yaml.safe_dump(tm), "rationale": "x",
                    "mapping_index": 5}).startswith("REJECTED: condition_occurrence has 1 mapping")

    out = replace({"target_table": "condition_occurrence", "mapping_yaml": yaml.safe_dump(tm),
                   "rationale": "ICD-10 codes are sometimes written without the dot."})
    assert out.startswith("OK:"), out
    (action,) = state.actions
    assert action.target_field is None and "condition_concept_id: vocab_code: d.icd10" in action.change
    assert "where: (none) → d.icd10 IS NOT NULL" in action.change
    assert state.spec.mapping_for("condition_occurrence").where == "d.icd10 IS NOT NULL"


@pytest.mark.postgres
def test_repair_spec_replay(staged, tmp_path):
    conn, schemas, cfg = staged
    ws = Workspace(tmp_path / "ws")
    replay_dir = tmp_path / "replay"
    save_recording(replay_dir, "repair", "iteration:1", {
        "kind": "agent",
        "tool_calls": [
            {"name": "get_table_mapping", "input": {"target_table": "visit_occurrence"}},
            {"name": "run_aggregate_query", "input": {"sql": "SELECT v.visit_type, COUNT(*) AS n FROM {src}.clinic_visits v "
                                                             "WHERE v.discharge_date IS NULL GROUP BY 1 ORDER BY 1"}},
            {"name": "update_field_mapping", "input": {
                "target_table": "visit_occurrence", "target_field": "visit_end_date", "kind": "expression",
                "expression": VISIT_END_FIX,
                "rationale": "Only inpatient visits have a discharge date; OMOP ends other visits on the start date."}},
        ],
        "final_text": "visit_end_date now falls back to the visit date. Nothing left for a human.",
    })
    spec = load_ref()
    ws.save_yaml(ws.spec_yaml, spec)
    etl = ETLRunResult(started_at=NOW, steps=[
        TableLoadResult(target_table="person", step="03_person", rows_inserted=400, source_rows=400),
        TableLoadResult(target_table="visit_occurrence", step="05_visit_occurrence",
                        error=NOT_NULL_ERROR + " DETAIL: Failing row contains (1, 2, 9202, 2019-04-22, null).",
                        sql="INSERT INTO cdm.visit_occurrence ...", source_rows=900),
    ])
    llm = ReplayLLM(replay_dir, ws)
    new_spec, actions = repair_spec(spec, iteration=1, dq=None, etl_result=etl, llm=llm, conn=conn, schemas=schemas,
                                    cfg=cfg, ws=ws, profile=None)

    (action,) = actions
    assert action.target_field == "visit_end_date" and action.trigger == "etl:05_visit_occurrence"
    fm = next(f for f in new_spec.mapping_for("visit_occurrence").fields if f.target_field == "visit_end_date")
    assert fm.expression == VISIT_END_FIX
    saved = ws.load_yaml(ws.spec_yaml, ETLSpec)
    assert saved == new_spec and ws.spec_yaml.read_text().startswith("# ETL specification")
    assert json.loads(ws.repairs_json.read_text())[0]["target_field"] == "visit_end_date"

    (call,) = [json.loads(p.read_text()) for p in (ws.root / "llm_calls").glob("*repair*.json")]
    assert NOT_NULL_ERROR in call["prompt"] and "INSERT INTO cdm.visit_occurrence" in call["prompt"]
    assert "Failing row" not in call["prompt"] and "2019-04-22" not in call["prompt"]  # no record values
    assert "visit_occurrence (mapping_index 0) <- {src}.clinic_visits v" in call["prompt"]
    results = [c["result"] for c in call["tool_calls"]]
    assert "INSERT INTO" in results[0] and "OPD" in results[1] and results[2].startswith("OK:")
    audit = [e for e in ws.read_audit() if e["event"] == "repair"]
    assert audit and audit[-1]["actions"] == 1 and "falls back" in audit[-1]["summary"]

    # a second iteration appends to repairs.json
    dq = DQReport(run_at=NOW, passed=True, results=[CheckResult(
        check_name="standardConceptRecordCompleteness", check_level="FIELD", table="condition_occurrence",
        field="condition_concept_id", description="d", severity="convention", failed=True, pct_violated_rows=0.13)])
    save_recording(replay_dir, "repair", "iteration:2", {"kind": "agent", "final_text": "done", "tool_calls": [
        {"name": "update_field_mapping", "input": {
            "target_table": "condition_occurrence", "target_field": "condition_concept_id", "kind": "vocab_code",
            "expression": "CASE WHEN d.icd10 ~ '^[A-Z][0-9]{3}$' THEN substr(d.icd10,1,3)||'.'||substr(d.icd10,4) "
                          "ELSE d.icd10 END",
            "vocabulary": "ICD10", "rationale": "normalise codes written without the dot"}}]})
    new_spec2, actions2 = repair_spec(new_spec, iteration=2, dq=dq, etl_result=None, llm=llm, conn=conn,
                                      schemas=schemas, cfg=cfg, ws=ws, profile=None)
    assert actions2[0].trigger == "standardConceptRecordCompleteness.condition_occurrence.condition_concept_id"
    assert [r["iteration"] for r in json.loads(ws.repairs_json.read_text())] == [1, 2]
    assert new_spec2.mapping_for("visit_occurrence") == new_spec.mapping_for("visit_occurrence")


@pytest.mark.postgres
def test_repair_spec_without_llm_or_failures(staged, tmp_path):
    conn, schemas, cfg = staged
    ws = Workspace(tmp_path / "ws")
    spec = load_ref()
    etl = ETLRunResult(started_at=NOW, steps=[TableLoadResult(target_table="visit_occurrence", step="05_v", error="x")])
    kwargs = dict(llm=NullLLM(ws), conn=conn, schemas=schemas, cfg=cfg, ws=ws, profile=None)
    assert repair_spec(spec, iteration=1, dq=None, etl_result=etl, **kwargs) == (spec, [])
    assert [e["event"] for e in ws.read_audit()] == ["repair_skipped"]
    # nothing to repair: the model is not called
    clean = DQReport(run_at=NOW, passed=True, results=[])
    assert repair_spec(spec, iteration=1, dq=clean, etl_result=None, **kwargs) == (spec, [])
    assert len(ws.read_audit()) == 1 and not ws.repairs_json.exists()
