"""Built-in checks: retention with vocabulary fan-out, cdm_source coverage, compact not-applicable
results, withinVisitDates and the DQD checks the built-in engine leaves to the R package."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from omop_agent.db import execute
from omop_agent.etl.runner import run_etl
from omop_agent.models import CheckResult, ETLRunResult, ETLSpec, FieldMapping, TableLoadResult
from omop_agent.quality.checks import (
    FIELD_SUMMARY,
    NOT_IMPLEMENTED,
    _matches_filter,
    check_catalog,
    etl_checks,
    run_builtin_checks,
)

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def fixed_ref() -> ETLSpec:
    spec = ETLSpec.model_validate(yaml.safe_load((FIXTURES / "demo_etl_spec.yaml").read_text()))
    tm = spec.mapping_for("visit_occurrence")
    i = next(i for i, f in enumerate(tm.fields) if f.target_field == "visit_end_date")
    tm.fields[i] = FieldMapping(target_field="visit_end_date", kind="expression",
                                expression="COALESCE(v.discharge_date::date, v.visit_date::date)")
    return spec


def by_key(results: list[CheckResult]) -> dict[str, CheckResult]:
    keys = [r.key for r in results]
    assert len(keys) == len(set(keys)), "check keys must be unique"
    return {r.key: r for r in results}


# --------------------------------------------------------------------------------------
# record retention
# --------------------------------------------------------------------------------------


def retention(source: int, inserted: int, retained: int | None) -> CheckResult:
    step = TableLoadResult(target_table="measurement", step="09_measurement", rows_inserted=inserted,
                           source_rows=source, retained_rows=retained)
    (res,) = etl_checks(ETLRunResult(started_at=NOW, steps=[step]))
    assert res.check_name == "etlRecordRetention"
    return res


def test_retention_is_not_netted_against_fan_out():
    # 20 of 100 rows dropped, and 30 extra rows from codes with two 'Maps to' targets: the net
    # comparison (110 inserted >= 100 source) hid the drops
    res = retention(100, 110, 80)
    assert (res.num_violated_rows, res.num_denominator_rows, res.pct_violated_rows) == (20, 100, 0.2)
    assert res.failed
    assert "20 of 100 records dropped" in res.description
    assert "30 of 110 rows inserted are extra rows added by 1-to-many vocabulary mappings" in res.description
    # every count is "<n> of <m>", so the repair prompt suppresses the small ones (the fan-out and the
    # rows inserted would otherwise give away a suppressed number of dropped records)
    from omop_agent.quality.repair import _check_summary

    _, text = _check_summary(retention(19, 19, 18), 5)
    assert "<5 of 19 records dropped" in text and "<5 of 19 rows inserted are extra rows" in text
    assert " 1 " not in text and "1 rows" not in text

    res = retention(100, 150, 100)  # fan-out only
    assert res.num_violated_rows == 0 and not res.failed and "50 of 150 rows inserted are extra" in res.description
    res = retention(100, 96, 96)  # plain drops, no fan-out: 4% is under the 5% threshold
    assert res.num_violated_rows == 4 and not res.failed and "1-to-many" not in res.description
    # results written before retained_rows existed fall back to the net comparison
    res = retention(100, 90, None)
    assert res.num_violated_rows == 10 and res.failed and "1-to-many" not in res.description
    assert retention(100, 120, None).num_violated_rows == 0


def test_evaluation_filters():
    row = {"isForeignKey": "Yes", "fkDomain": "Gender", "fkClass": "", "cdmDatatype": "integer"}
    assert _matches_filter(row, "isForeignKey=='Yes' & fkDomain!= ''")
    assert not _matches_filter(row, "isForeignKey=='Yes' & fkClass!=''")
    assert _matches_filter(row, "cdmDatatype=='integer'")
    assert not _matches_filter({}, "plausibleTemporalAfter=='Yes'")
    assert not _matches_filter(row, "something odd")
    # every DQD filter can be read
    assert all(i.evaluation_filter and all("==" in c or "!=" in c for c in i.evaluation_filter.split("&"))
               for i in check_catalog().values())


# --------------------------------------------------------------------------------------
# checks on a built CDM (PostgreSQL)
# --------------------------------------------------------------------------------------


@pytest.fixture
def built(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref()
    etl = run_etl(conn, schemas, spec, cfg)
    assert etl.ok, [(e.step, e.error) for e in etl.errors()]
    return conn, schemas, cfg, spec, etl


@pytest.mark.postgres
def test_cdm_source_is_checked(built):
    conn, schemas, cfg, spec, etl = built
    for with_spec in (spec, None):  # populated through the spec, or because it has a row
        results = by_key(run_builtin_checks(conn, schemas, cfg, with_spec, etl).results)
        cdm_source = [r for r in results.values() if r.table == "cdm_source" and r.field]
        assert len(cdm_source) >= 20 and not [r.key for r in cdm_source if r.not_applicable or r.error or r.failed]
        assert results["isRequired.cdm_source.cdm_source_name"].num_denominator_rows == 1
        assert not results["isForeignKey.cdm_source.cdm_version_concept_id"].failed
        assert f"{FIELD_SUMMARY}.cdm_source" not in results


@pytest.mark.postgres
def test_tables_without_rows_get_one_summary(built):
    conn, schemas, cfg, spec, etl = built
    report = run_builtin_checks(conn, schemas, cfg, spec, etl)
    results = by_key(report.results)
    assert report.passed, report.blocking_failures

    # not populated: the table-level checks plus one summary, never a result per field and check
    procedure = [r for r in report.results if r.table == "procedure_occurrence"]
    assert sorted(r.key for r in procedure) == [
        f"{FIELD_SUMMARY}.procedure_occurrence", "measurePersonCompleteness.procedure_occurrence"
    ]
    assert all(r.not_applicable and not r.sql for r in procedure)
    summary = results[f"{FIELD_SUMMARY}.procedure_occurrence"]
    assert summary.check_level == "TABLE" and "the ETL does not populate it" in summary.description
    assert "isRequired" in summary.description and "withinVisitDates" in summary.description
    assert [r.key for r in report.results if r.table == "location"] == [f"{FIELD_SUMMARY}.location"]
    assert not [r for r in report.results if r.check_level == "FIELD" and r.not_applicable
                and r.table not in {"person", "observation_period", "visit_occurrence", "condition_occurrence",
                                    "drug_exposure", "measurement", "death", "condition_era", "cdm_source"}]
    # drug_era is derived, but no drug of the fixture is mapped: summarised like a table without rows
    assert [r.key for r in report.results if r.table == "drug_era" and r.check_level == "FIELD"] == []
    assert "it has no rows" in results[f"{FIELD_SUMMARY}.drug_era"].description
    # a table the spec loads but that ended up empty is summarised too
    execute(conn, f"TRUNCATE {schemas.cdm}.death")
    empty = by_key(run_builtin_checks(conn, schemas, cfg, spec, etl).results)
    assert "it has no rows" in empty[f"{FIELD_SUMMARY}.death"].description
    assert not [k for k in empty if k.startswith("isRequired.death.")]
    conn.rollback()

    # the result list stays small: a few hundred checks, not thousands of not-applicable rows. Measured:
    # 550 here (518 before the build derived eras; 585 when the drugs are mapped and drug_era has rows)
    n_na = sum(r.not_applicable for r in report.results)
    assert len(report.results) < 650 and n_na < len(report.results) / 2


@pytest.mark.postgres
def test_checks_left_to_the_r_package(built):
    conn, schemas, cfg, spec, etl = built
    results = by_key(run_builtin_checks(conn, schemas, cfg, spec, etl).results)
    expected = {
        "cdmTable.measurement": "TABLE",
        "cdmField.measurement": "FIELD",
        "cdmDatatype.measurement": "FIELD",
        "plausibleTemporalAfter.measurement": "FIELD",
        "plausibleGender.condition_occurrence": "CONCEPT",
        "plausibleUnitConceptIds.measurement": "CONCEPT",
        "cdmTable.cdm_source": "TABLE",
    }
    for key, level in expected.items():
        r = results[key]
        assert r.not_applicable and r.check_level == level and r.field is None, key
        assert r.description.endswith(NOT_IMPLEMENTED), key
        assert r.severity == check_catalog()[r.check_name].severity
    assert "fields of measurement" in results["cdmField.measurement"].description
    assert "plausibleGender.death" not in results  # no concept-level rows for death
    # the build derives condition_era, and the built-in engine checks its completeness (DQD 2.8.9)
    era = results["measureConditionEraCompleteness.condition_era"]
    assert not (era.not_applicable or era.error or era.failed) and era.num_denominator_rows > 0
    assert era.check_level == "TABLE" and era.severity == check_catalog()[era.check_name].severity == "convention"
    assert "cdmTable.procedure_occurrence" not in results  # covered by the table's summary


@pytest.mark.postgres
def test_within_visit_dates(built):
    conn, schemas, cfg, spec, etl = built
    results = by_key(run_builtin_checks(conn, schemas, cfg, spec, etl).results)
    for t, f in [("condition_occurrence", "condition_start_date"), ("drug_exposure", "drug_exposure_start_date"),
                 ("measurement", "measurement_date")]:
        r = results[f"withinVisitDates.{t}.{f}"]
        assert not (r.not_applicable or r.error), (r.key, r.error)
        assert r.num_denominator_rows > 0 and r.severity == "characterization" and "one week" in r.description

    base = results["withinVisitDates.condition_occurrence.condition_start_date"].num_violated_rows
    c = schemas.cdm
    execute(conn, f"""UPDATE {c}.condition_occurrence co SET condition_start_date = vo.visit_start_date - 8
        FROM {c}.visit_occurrence vo WHERE vo.visit_occurrence_id = co.visit_occurrence_id
        AND co.condition_occurrence_id IN (1, 2)""")
    execute(conn, f"""UPDATE {c}.condition_occurrence co SET condition_start_date = vo.visit_end_date + 7
        FROM {c}.visit_occurrence vo WHERE vo.visit_occurrence_id = co.visit_occurrence_id
        AND co.condition_occurrence_id = 3""")  # exactly a week after: still within
    results = by_key(run_builtin_checks(conn, schemas, cfg, spec, etl).results)
    r = results["withinVisitDates.condition_occurrence.condition_start_date"]
    assert r.num_violated_rows == base + 2 and "vo.visit_start_date - 7" in (r.sql or "")

    # without visits the check does not apply
    execute(conn, f"TRUNCATE {c}.visit_occurrence")
    results = by_key(run_builtin_checks(conn, schemas, cfg, spec, etl).results)
    assert results["withinVisitDates.condition_occurrence.condition_start_date"].not_applicable
    conn.rollback()


@pytest.mark.postgres
def test_retention_after_a_build_with_fan_out(staged):
    conn, schemas, cfg = staged
    execute(conn, f"""INSERT INTO {schemas.etl}.source_to_concept_map (source_code, source_vocabulary_id, target_concept_id)
        VALUES ('HB', 'SRC_LAB', 2000004001), ('HB', 'SRC_LAB', 2000004002)""")
    conn.commit()
    etl = run_etl(conn, schemas, fixed_ref(), cfg)
    assert etl.ok
    m = next(r for r in etl_checks(etl) if r.table == "measurement")
    assert m.num_violated_rows == 12  # lab results of people missing from the registry
    assert "1-to-many vocabulary mappings" in m.description and not m.failed
