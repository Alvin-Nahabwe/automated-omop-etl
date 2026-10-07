"""ETL design: static/DB validation, the replayed design flow with revisions, markdown rendering."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from omop_agent.config import PrivacyConfig, ProjectConfig
from omop_agent.db import Schemas, execute, list_columns, list_tables, row_count, scalar
from omop_agent.design.designer import DESIGN_SYSTEM, design_etl, reference_concepts_text
from omop_agent.design.render import render_spec_md
from omop_agent.design.validate import SpecIssue, sql_safety_issues, validate_spec, validate_table_mapping
from omop_agent.llm import NullLLM, ReplayLLM
from omop_agent.llm.replay import save_recording
from omop_agent.models import (
    ColumnProfile,
    CustomSQL,
    ETLSpec,
    FieldMapping,
    JoinSpec,
    PlannedTable,
    SourceAnnotations,
    SourceProfile,
    TablePlan,
    TableProfile,
)
from omop_agent.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"


def load_ref() -> ETLSpec:
    return ETLSpec.model_validate(yaml.safe_load((FIXTURES / "demo_etl_spec.yaml").read_text()))


def set_field(spec: ETLSpec, table: str, field: str, **changes) -> ETLSpec:
    spec = spec.model_copy(deep=True)
    tm = spec.mapping_for(table)
    fm = next((f for f in tm.fields if f.target_field == field), None)
    if fm is None:
        tm.fields.append(FieldMapping(target_field=field, **changes))
    else:
        tm.fields[tm.fields.index(fm)] = fm.model_copy(update=changes)
    return spec


def drop_field(spec: ETLSpec, table: str, field: str) -> ETLSpec:
    spec = spec.model_copy(deep=True)
    tm = spec.mapping_for(table)
    tm.fields = [f for f in tm.fields if f.target_field != field]
    return spec


def profile_from_db(conn, schemas) -> SourceProfile:
    """A minimal profile of the staged tables (names and row counts only)."""
    tables = []
    for t in list_tables(conn, schemas.src):
        n = row_count(conn, schemas.src, t)
        cols = [
            ColumnProfile(name=c["column_name"], inferred_type="string", n_rows=n, n_null=0, n_distinct=1, unique=False)
            for c in list_columns(conn, schemas.src, t)
        ]
        tables.append(TableProfile(name=t, n_rows=n, columns=cols))
    return SourceProfile(source_type="csv", generated_at=datetime.now(timezone.utc), min_cell_count=5, tables=tables)


DEMO_COLUMNS = {
    "individuals": ["individual_id", "sex", "dob", "village", "hh_id", "first_name", "last_name", "phone"],
    "residency_episodes": ["individual_id", "start_date", "start_event", "end_date", "end_event"],
    "clinic_visits": ["visit_id", "individual_id", "visit_date", "discharge_date", "visit_type", "facility"],
    "diagnoses": ["visit_id", "individual_id", "icd10", "dx_text", "dx_date"],
    "prescriptions": ["visit_id", "individual_id", "drug", "qty", "days", "date_prescribed"],
    "lab_results": ["visit_id", "individual_id", "test", "result", "unit", "test_date"],
    "deaths": ["individual_id", "dod", "va_cause"],
}
PII = {"first_name", "last_name", "phone"}


def demo_profile() -> SourceProfile:
    """The demo dataset's tables and columns, with its direct identifiers flagged (no database)."""
    def col(name: str) -> ColumnProfile:
        return ColumnProfile(name=name, inferred_type="string", n_rows=10, n_null=0, n_distinct=10, unique=False,
                             is_pii=name in PII, pii_reason="name" if name in PII else None)

    tables = [TableProfile(name=t, n_rows=10, columns=[col(c) for c in cols]) for t, cols in DEMO_COLUMNS.items()]
    return SourceProfile(source_type="csv", generated_at=datetime.now(timezone.utc), min_cell_count=5, tables=tables)


def errors(issues: list[SpecIssue]) -> list[SpecIssue]:
    return [i for i in issues if i.severity == "error"]


def has(issues: list[SpecIssue], table: str | None, field: str | None, text: str) -> bool:
    return any(
        i.severity == "error" and i.target_table == table and i.target_field == field and text in i.message
        for i in issues
    )


# --------------------------------------------------------------------------------------
# static validation (no database)
# --------------------------------------------------------------------------------------


def test_reference_spec_is_statically_valid():
    for profile in (None, demo_profile()):
        issues = validate_spec(load_ref(), profile)
        assert errors(issues) == []
        # the verbal-autopsy cause mapping has no target domain: the only warning
        assert [(i.target_table, i.target_field) for i in issues] == [("death", "cause_concept_id")]


@pytest.mark.parametrize(
    "make, table, field, text",
    [
        (lambda s: drop_field(s, "person", "year_of_birth"), "person", "year_of_birth", "required field is not mapped"),
        (lambda s: drop_field(s, "condition_occurrence", "condition_occurrence_id"), "condition_occurrence",
         "condition_occurrence_id", "auto_id"),
        (lambda s: set_field(s, "person", "race_concept_id", kind="null", expression=None), "person", "race_concept_id", "NULL"),
        (lambda s: set_field(s, "person", "gender_concept_id", expression="CASE z.sex WHEN '1' THEN 8507 ELSE 0 END"),
         "person", "gender_concept_id", "alias 'z' is not declared"),
        (lambda s: set_field(s, "person", "shoe_size", kind="constant", expression="42"), "person", "shoe_size", "not a field"),
        (lambda s: set_field(s, "drug_exposure", "drug_concept_id", vocabulary=None), "drug_exposure", "drug_concept_id",
         "needs a vocabulary"),
        (lambda s: set_field(s, "condition_occurrence", "condition_concept_id", vocabulary=None), "condition_occurrence",
         "condition_concept_id", "needs a vocabulary"),
        (lambda s: set_field(s, "person", "year_of_birth", kind="auto_id", expression=None), "person", "year_of_birth",
         "primary key"),
        (lambda s: set_field(s, "person", "gender_source_value", expression=None), "person", "gender_source_value",
         "needs an expression"),
    ],
)
def test_static_errors(make, table, field, text):
    issues = validate_spec(make(load_ref()))
    assert has(issues, table, field, text), issues


def test_static_spec_level_errors():
    spec = load_ref().model_copy(deep=True)
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table not in ("observation_period", "person")]
    spec.mapping_for("death").fields.append(spec.mapping_for("death").fields[0].model_copy())
    spec.table_mappings[0].target_table = "visits"
    issues = validate_spec(spec)
    assert has(issues, "person", None, "no table mapping loads person")
    assert has(issues, "observation_period", None, "no mapping loads observation_period")
    assert has(issues, "death", "person_id", "mapped 2 times")
    assert has(issues, "visits", None, "not an OMOP CDM v5.4 table")

    spec = load_ref().model_copy(update={"visit_key": None})
    assert has(validate_spec(spec), "visit_occurrence", "visit_occurrence_id", "needs a visit_key")


def test_duplicate_aliases_and_function_qualifiers():
    spec = set_field(load_ref(), "person", "birth_datetime", expression="public.to_timestamp(i.dob, 'DD/MM/YYYY')")
    assert errors(validate_spec(spec)) == []  # a schema-qualified function call is not an alias
    spec = spec.model_copy(deep=True)
    spec.mapping_for("person").joins.append(JoinSpec(table="individuals", alias="i", on="TRUE"))
    assert has(validate_spec(spec), "person", None, "alias 'i' is declared twice")
    # system catalogs are refused by the SQL guard
    spec = set_field(load_ref(), "person", "birth_datetime", expression="pg_catalog.to_timestamp(i.dob, 'DD/MM/YYYY')")
    assert has(validate_spec(spec), "person", "birth_datetime", "not allowed in ETL SQL: pg_catalog")
    # ... and public.<table> is not a silent read of another schema
    spec = set_field(load_ref(), "person", "gender_source_value", expression="public.secrets.x")
    assert has(validate_spec(spec), "person", "gender_source_value", "alias 'public' is not declared")


# --------------------------------------------------------------------------------------
# person/visit links, SQL safety, direct identifiers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make, table, field, text",
    [
        # an auto_id person_id follows row order, while events resolve the key through the map
        (lambda s: set_field(s, "person", "person_id", kind="auto_id", expression=None), "person", "person_id",
         "must use kind person_id"),
        (lambda s: set_field(s, "person", "person_id", kind="column", expression="i.individual_id"), "person",
         "person_id", "must use kind person_id"),
        (lambda s: set_field(s, "measurement", "person_id", kind="expression", expression="r.individual_id::int"),
         "measurement", "person_id", "must use kind person_id"),
        (lambda s: drop_field(s, "death", "person_id"), "death", "person_id", "use kind person_id with the column"),
        (lambda s: drop_field(s, "person", "person_id"), "person", "person_id", "use kind person_id with the source key"),
        (lambda s: set_field(s, "visit_occurrence", "visit_occurrence_id", kind="column", expression="v.visit_id"),
         "visit_occurrence", "visit_occurrence_id", "must use kind visit_id"),
        (lambda s: set_field(s, "visit_occurrence", "visit_occurrence_id", kind="auto_id", expression=None),
         "visit_occurrence", "visit_occurrence_id", "auto_id is not allowed here"),
    ],
)
def test_person_and_visit_ids_come_from_the_id_maps(make, table, field, text):
    assert has(validate_spec(make(load_ref())), table, field, text)


def test_visit_links_without_a_visit_key():
    spec = load_ref().model_copy(update={"visit_key": None}, deep=True)
    for tm in spec.table_mappings:
        tm.fields = [f for f in tm.fields if f.kind != "visit_id"]
    spec = set_field(spec, "visit_occurrence", "visit_occurrence_id", kind="expression", expression="substr(v.visit_id, 2)::int")
    assert errors(validate_spec(spec)) == []  # without a visit map any expression will do ...
    spec = set_field(spec, "visit_occurrence", "visit_occurrence_id", kind="auto_id", expression=None)
    assert has(validate_spec(spec), "visit_occurrence", "visit_occurrence_id", "auto_id is not allowed here")  # ... but not auto_id
    # an event's visit link that bypasses the visit map is suspicious
    spec = set_field(load_ref(), "measurement", "visit_occurrence_id", kind="expression", expression="r.visit_id::int")
    issues = validate_spec(spec)
    assert errors(issues) == []
    assert any(i.severity == "warning" and i.target_field == "visit_occurrence_id" and "visit_id" in i.message for i in issues)


@pytest.mark.parametrize(
    "make, table, field, text",
    [
        (lambda s: set_field(s, "person", "gender_source_value", expression="pg_read_file('/etc/passwd')"),
         "person", "gender_source_value", "pg_read_file()"),
        (lambda s: set_field(s, "person", "gender_source_value", expression="i.sex); DROP TABLE x; SELECT (1"),
         "person", "gender_source_value", "contains ';'"),
        (lambda s: set_field(s, "person", "gender_source_value", expression="current_setting('server_version')"),
         "person", "gender_source_value", "current_setting()"),
        (lambda s: set_field(s, "person", "gender_source_value", expression="$$x$$"),
         "person", "gender_source_value", "dollar-quoted"),
        (lambda s: set_field(s, "person", "gender_source_value",
                             expression="(SELECT MIN(p.gender_source_value) FROM {cdm}.person p)"),
         "person", "gender_source_value", "table '{cdm}.person' may not be read"),
        (lambda s: set_field(s, "person", "gender_source_value",
                             expression="(SELECT MIN(r.rolname) FROM other.roles AS r)"),
         "person", "gender_source_value", "table 'other.roles' may not be read"),
        (lambda s: _replace(s, "death", where="EXISTS (SELECT 1 FROM information_schema.tables)"),
         "death", None, "information_schema"),
        (lambda s: _replace(s, "death", joins=[JoinSpec(table="{cdm}.person", alias="cp", on="cp.person_id = 1")]),
         "death", None, "table '{cdm}.person' may not be read"),
        (lambda s: _replace(s, "death", joins=[JoinSpec(table="individuals", alias="q", on="pg_sleep(10) IS NULL")]),
         "death", None, "pg_sleep()"),
        (lambda s: _replace(s, "death", source={"table": "cdm.death", "alias": "x"}), "death", None,
         "table 'cdm.death' may not be read"),
    ],
)
def test_unsafe_sql_is_rejected(make, table, field, text):
    spec = load_ref().model_copy(deep=True)
    assert has(validate_spec(make(spec)), table, field, text)


def _replace(spec: ETLSpec, table: str, **update) -> ETLSpec:
    spec = spec.model_copy(deep=True)
    i = next(i for i, tm in enumerate(spec.table_mappings) if tm.target_table == table)
    spec.table_mappings[i] = spec.table_mappings[i].model_validate({**spec.table_mappings[i].model_dump(), **update})
    return spec


def test_placeholders_and_keys():
    spec = _replace(load_ref(), "death", source={"table": "{src}.deaths", "alias": "x"},
                    joins=[JoinSpec(table="{vocab}.concept", alias="vc", on="vc.concept_code = x.va_cause")])
    assert errors(validate_spec(spec, demo_profile())) == []
    spec = load_ref().model_copy(deep=True)
    spec.person_key.expression = "individual_id || current_setting('x')"
    spec.visit_key.table = "{cdm}.visit_occurrence"
    issues = errors(validate_spec(spec, demo_profile()))
    assert any(i.message.startswith("person_key: unsafe SQL in the key expression") for i in issues)
    assert any(i.message.startswith("visit_key: table '{cdm}.visit_occurrence' may not be read") for i in issues)
    assert sql_safety_issues(spec.person_key) and sql_safety_issues(spec.visit_key)
    assert sql_safety_issues(load_ref().person_key) == [] and sql_safety_issues(load_ref().mapping_for("death")) == []


def test_custom_sql_is_reviewed_and_guarded():
    spec = load_ref().model_copy(deep=True)
    spec.custom_sql = [
        CustomSQL(name="fix ends", target_table="drug_exposure", sql=(
            "UPDATE {cdm}.drug_exposure SET drug_exposure_end_date = drug_exposure_start_date "
            "WHERE drug_exposure_end_date < drug_exposure_start_date;\n"
            "DELETE FROM {cdm}.drug_exposure WHERE drug_concept_id IS NULL;")),
    ]
    issues = validate_spec(spec)
    assert errors(issues) == []  # several statements and DML are what custom SQL is for
    assert any(i.severity == "warning" and "executed as written; review it" in i.message for i in issues)
    spec.custom_sql.append(CustomSQL(name="evil", target_table="death",
                                     sql="INSERT INTO {cdm}.death SELECT * FROM {src}.deaths; DROP TABLE {cdm}.person"))
    spec.custom_sql.append(CustomSQL(name="files", target_table="note", sql="SELECT pg_read_file('/etc/passwd')"))
    assert has(validate_spec(spec), "death", None, "custom SQL 'evil': unsafe custom SQL")
    assert has(validate_spec(spec), "note", None, "pg_read_file()")
    assert sql_safety_issues(spec.custom_sql[0]) == [] and "DROP" in sql_safety_issues(spec.custom_sql[1])[0]
    # statements that could leave the ETL role or the step's transaction
    for sql in ("RESET ROLE; DELETE FROM {cdm}.death", "SET ROLE NONE", "COMMIT; INSERT INTO {cdm}.death SELECT 1",
                "DO 'BEGIN PERFORM 1; END'", "-- note\nCALL {etl}.p()"):
        problems = sql_safety_issues(CustomSQL(name="x", target_table="death", sql=sql))
        assert problems and all(p.startswith("custom SQL 'x': unsafe custom SQL") for p in problems), sql
    ok = "WITH d AS (SELECT 1 AS x) SELECT x FROM d; -- trailing comment\n"
    assert sql_safety_issues(CustomSQL(name="x", target_table="death", sql=ok)) == []


def test_direct_identifiers_are_not_loaded():
    profile = demo_profile()
    spec = set_field(load_ref(), "person", "person_source_value", expression="i.first_name || ' ' || i.last_name")
    issues = validate_spec(spec, profile)
    assert has(issues, "person", None, "loads direct identifier column(s) individuals.first_name, individuals.last_name")
    # identifiers in filters, join conditions and id-map keys are not loaded into the CDM
    spec = _replace(load_ref(), "death", joins=[JoinSpec(table="individuals", alias="q", on="q.individual_id = x.individual_id")],
                    where="q.phone IS NOT NULL")
    assert errors(validate_spec(spec, profile)) == []
    # privacy.pii_columns applies without a profile too
    spec = set_field(load_ref(), "person", "gender_source_value", expression="i.village")
    assert errors(validate_spec(spec)) == []
    assert has(validate_spec(spec, None, privacy=PrivacyConfig(pii_columns=["individuals.village"])),
               "person", None, "individuals.village")
    # ... and through a correlated subquery
    spec = set_field(load_ref(), "death", "cause_source_value",
                     expression="(SELECT MIN(q.phone) FROM {src}.individuals q WHERE q.individual_id = x.individual_id)")
    assert has(validate_spec(spec, profile), "death", "cause_source_value", "direct identifier column 'individuals.phone'")


def test_correlated_subquery_aliases():
    profile = demo_profile()
    last_residence = (
        "COALESCE(x.dod::date, (SELECT MAX(r.end_date::date) FROM {src}.residency_episodes r "
        "WHERE r.individual_id = x.individual_id))"
    )
    spec = set_field(load_ref(), "death", "death_date", expression=last_residence)
    assert errors(validate_spec(spec)) == [] and errors(validate_spec(spec, profile)) == []
    joined = (
        "(SELECT MIN(v.visit_date::date) FROM {src}.clinic_visits AS v JOIN {src}.diagnoses dd "
        "ON dd.visit_id = v.visit_id WHERE v.individual_id = x.individual_id AND dd.icd10 = 'B54')"
    )
    spec = set_field(load_ref(), "death", "death_datetime", kind="expression", expression=joined)
    assert errors(validate_spec(spec, profile)) == []
    # columns of subquery aliases are still checked
    spec = set_field(load_ref(), "death", "death_date", expression=last_residence.replace("r.end_date", "r.end_dt"))
    assert has(validate_spec(spec, profile), "death", "death_date",
               "column 'r.end_dt' does not exist in source table 'residency_episodes'")
    # a bare source table inside an expression would be looked up in the search_path, not {src}
    spec = set_field(load_ref(), "death", "death_date", expression=last_residence.replace("{src}.", ""))
    assert has(validate_spec(spec, profile), "death", "death_date", "write the source table as {src}.residency_episodes")
    # an alias used outside any declaration is still reported; EXTRACT(... FROM <column>) declares nothing
    spec = set_field(load_ref(), "person", "year_of_birth",
                     expression="EXTRACT(YEAR FROM to_date(i.dob, 'DD/MM/YYYY'))::int + 0 * EXTRACT(YEAR FROM z.dob::date)")
    assert has(validate_spec(spec, profile), "person", "year_of_birth", "alias 'z' is not declared")
    spec = set_field(load_ref(), "person", "year_of_birth", expression="EXTRACT(YEAR FROM i.dob::date AT TIME ZONE 'UTC')::int")
    assert errors(validate_spec(spec, profile)) == []


# --------------------------------------------------------------------------------------
# validation against the staged source and the vocabulary
# --------------------------------------------------------------------------------------


@pytest.mark.postgres
def test_reference_spec_is_valid_against_db(staged):
    conn, schemas, cfg = staged
    profile = profile_from_db(conn, schemas)
    # neither deliberate bug is detectable before running: the dot bug is semantic and the
    # visit_end_date NULLs only exist in the data
    assert errors(validate_spec(load_ref(), profile, conn, schemas)) == []
    assert errors(validate_spec(load_ref(), None, conn, schemas)) == []  # columns from the staging schema


@pytest.mark.postgres
@pytest.mark.parametrize("use_profile", [True, False])
@pytest.mark.parametrize(
    "make, table, field, text",
    [
        (lambda s: set_field(s, "person", "person_source_value", expression="i.patient_no"),
         "person", "person_source_value", "column 'i.patient_no' does not exist in source table 'individuals'"),
        (lambda s: set_field(s, "person", "race_concept_id", expression="999999999"),
         "person", "race_concept_id", "concept_id 999999999 does not exist"),
        (lambda s: set_field(s, "person", "gender_concept_id", expression="CASE i.sex WHEN '1' THEN 9202 WHEN '2' THEN 8532 ELSE 0 END"),
         "person", "gender_concept_id", "not in the expected domain Gender"),
        (lambda s: set_field(s, "visit_occurrence", "visit_type_concept_id", expression="8507"),
         "visit_occurrence", "visit_type_concept_id", "not in the expected domain Type Concept"),
        (lambda s: set_field(s, "condition_occurrence", "condition_type_concept_id", expression="2000002001"),
         "condition_occurrence", "condition_type_concept_id", "not a standard concept"),
        (lambda s: set_field(s, "condition_occurrence", "condition_concept_id", vocabulary="ICD9CM"),
         "condition_occurrence", "condition_concept_id", "vocabulary 'ICD9CM' is not loaded"),
    ],
)
def test_db_errors(staged, use_profile, make, table, field, text):
    conn, schemas, cfg = staged
    profile = profile_from_db(conn, schemas) if use_profile else None
    issues = validate_spec(make(load_ref()), profile, conn, schemas)
    assert has(issues, table, field, text), issues


@pytest.mark.postgres
def test_explain_and_source_table_errors(staged):
    conn, schemas, cfg = staged
    profile = profile_from_db(conn, schemas)
    spec = set_field(load_ref(), "person", "year_of_birth", expression="no_such_function(i.dob)")
    issues = validate_spec(spec, profile, conn, schemas)
    assert any(i.target_table == "person" and "EXPLAIN failed" in i.message and "no_such_function" in i.message for i in issues)
    # the other tables are still EXPLAINed cleanly, and the connection is usable afterwards
    assert [i.target_table for i in errors(issues)] == ["person"]

    spec = load_ref().model_copy(deep=True)
    spec.mapping_for("death").source.table = "mortality"
    spec.person_key.table = "people"
    issues = validate_spec(spec, profile, conn, schemas)
    assert has(issues, "death", None, "source table 'mortality' does not exist")
    assert any(i.severity == "error" and "person_key table 'people'" in i.message for i in issues)

    spec = load_ref().model_copy(deep=True)
    spec.person_key.expression = "no_such_column"
    issues = validate_spec(spec, None, conn, schemas)
    assert any("person_key column 'no_such_column'" in i.message for i in errors(issues))

    tm_issues = validate_table_mapping(load_ref().mapping_for("measurement"), load_ref(), profile, conn, schemas)
    assert errors(tm_issues) == []
    assert conn.execute("SELECT 1").fetchone()


@pytest.mark.postgres
def test_explain_runs_under_the_build_session_settings(staged):
    """EXPLAIN folds constants, under the session settings of the build (DateStyle 'ISO, YMD'):
    a month-first date the server's own DateStyle would accept is an error, as in the build."""
    conn, schemas, cfg = staged
    server_style = scalar(conn, "SELECT current_setting('DateStyle')")
    assert scalar(conn, "SELECT '03/04/2020'::date IS NOT NULL") is True  # the server's DateStyle reads it
    spec = set_field(load_ref(), "death", "death_date", kind="expression", expression="'03/04/2020'::date")
    issues = validate_table_mapping(spec.mapping_for("death"), spec, None, conn, schemas, cfg=cfg)
    assert any("EXPLAIN failed" in i.message and "out of range" in i.message for i in errors(issues)), issues
    assert has(validate_spec(spec, None, conn, schemas, cfg=cfg), "death", None, "EXPLAIN failed")
    spec = set_field(load_ref(), "death", "death_date", kind="expression", expression="'2020/03/04'::date")
    assert errors(validate_table_mapping(spec.mapping_for("death"), spec, None, conn, schemas, cfg=cfg)) == []
    assert scalar(conn, "SELECT current_setting('DateStyle')") == server_style  # rolled back


@pytest.mark.postgres
def test_correlated_subquery_against_db(staged):
    conn, schemas, cfg = staged
    spec = set_field(load_ref(), "death", "death_date", expression=(
        "COALESCE(x.dod::date, (SELECT MAX(r.end_date::date) FROM {src}.residency_episodes r "
        "WHERE r.individual_id = x.individual_id))"))
    assert errors(validate_spec(spec, None, conn, schemas)) == []
    spec = set_field(spec, "death", "death_datetime", kind="expression",
                     expression="(SELECT MAX(r.end_date::timestamp) FROM residency_episodes r)")
    issues = validate_spec(spec, None, conn, schemas)  # a bare name inside an expression is not the staged table
    assert has(issues, "death", "death_datetime", "write the source table as {src}.residency_episodes")
    spec = set_field(spec, "death", "death_datetime", expression="(SELECT MAX(r.x::timestamp) FROM elsewhere r)")
    issues = validate_spec(spec, None, conn, schemas)  # unknown to the catalog: EXPLAIN reports it
    assert any(i.target_table == "death" and "EXPLAIN failed" in i.message and "elsewhere" in i.message
               for i in errors(issues))


@pytest.mark.postgres
def test_reference_concepts_text(staged):
    conn, schemas, cfg = staged
    text = reference_concepts_text(conn, schemas)
    lines = text.splitlines()
    assert lines[0] == "concept_id | concept_name | domain | vocabulary"
    assert "8507 | MALE | Gender | Gender" in lines
    assert "9202 | Outpatient Visit | Visit | Visit" in lines
    assert "32817 | EHR | Type Concept | Type Concept" in lines
    assert not any("SNOMED" in ln for ln in lines)
    assert lines.index("8507 | MALE | Gender | Gender") < lines.index("9201 | Inpatient Visit | Visit | Visit")
    assert len(reference_concepts_text(conn, schemas, limit_per_domain=1, type_concept_limit=1).splitlines()) == 1 + 3

    # relevance, not concept_id: the domain's own vocabulary first, common provenance types first
    execute(conn, f"""INSERT INTO {schemas.vocab}.concept (concept_id, concept_name, domain_id, vocabulary_id,
        concept_class_id, standard_concept, concept_code, valid_start_date, valid_end_date) VALUES
        (8, 'Ambulance - Land', 'Visit', 'CMS Place of Service', 'Visit', 'S', '41', '1970-01-01', '2099-12-31'),
        (9, 'Aardvark legacy type', 'Type Concept', 'Type Concept', 'Type Concept', 'S', 'X1', '1970-01-01', '2099-12-31'),
        (10, 'Registry', 'Type Concept', 'Type Concept', 'Type Concept', 'S', 'X2', '1970-01-01', '2099-12-31')""")
    lines = reference_concepts_text(conn, schemas).splitlines()
    visits = [ln.split(" | ")[1] for ln in lines if " | Visit | " in ln]
    assert visits == ["Emergency Room Visit", "Inpatient Visit", "Outpatient Visit", "Ambulance - Land"]
    types = [ln.split(" | ")[1] for ln in lines if " | Type Concept | " in ln]
    assert types == ["EHR", "Registry", "Aardvark legacy type"]
    lines = reference_concepts_text(conn, schemas, limit_per_domain=1, type_concept_limit=2).splitlines()
    assert [ln.split(" | ")[1] for ln in lines[1:]] == ["FEMALE", "Emergency Room Visit", "EHR", "Registry"]
    conn.rollback()


# --------------------------------------------------------------------------------------
# design flow with recorded answers
# --------------------------------------------------------------------------------------


def record_reference(replay_dir: Path, spec: ETLSpec) -> None:
    plan = TablePlan(
        person_key=spec.person_key,
        visit_key=spec.visit_key,
        observation_period=spec.observation_period,
        tables=[
            PlannedTable(target_table=tm.target_table, source_tables=[tm.source.table], rationale=tm.rationale)
            for tm in spec.table_mappings
        ],
        assumptions=spec.assumptions,
        open_questions=spec.open_questions,
    )
    save_recording(replay_dir, "design", "plan", {"kind": "structured", "output": plan.model_dump(mode="json")})
    for tm in spec.table_mappings:
        save_recording(
            replay_dir, "design", f"table:{tm.target_table}", {"kind": "structured", "output": tm.model_dump(mode="json")}
        )


@pytest.mark.postgres
def test_design_reproduces_reference_spec(staged, tmp_path):
    conn, schemas, cfg = staged
    ref = load_ref()
    replay = tmp_path / "replay"
    record_reference(replay, ref)
    ws = Workspace(tmp_path / "work")
    llm = ReplayLLM(replay, ws)

    spec = design_etl(profile_from_db(conn, schemas), None, cfg, llm, conn, schemas, ws)
    assert spec.model_dump() == ref.model_dump()

    text = ws.spec_yaml.read_text()
    assert text.startswith("# ETL specification for omop-agent")
    assert ws.load_yaml(ws.spec_yaml, ETLSpec).model_dump() == ref.model_dump()
    assert "## Table mappings" in ws.spec_md.read_text()
    issues = json.loads((ws.root / "design_issues.json").read_text())
    assert [i for i in issues if i["severity"] == "error"] == []
    audit = ws.read_audit()
    done = [e for e in audit if e["event"] == "design_complete"]
    assert done and done[0]["tables"] == [tm.target_table for tm in ref.table_mappings] and done[0]["errors"] == 0

    calls = sorted((ws.root / "llm_calls").glob("*.json"))
    assert len(calls) == 1 + len(ref.table_mappings)
    person_call = json.loads(next(c for c in calls if c.name.endswith("table_person.json")).read_text())
    # one system prompt for every call (cached by the provider): instructions + shared reference
    system = person_call["system"]
    assert system.startswith(DESIGN_SYSTEM)
    assert {json.loads(c.read_text())["system"] for c in calls} == {system}
    assert "concept_map  local codes" in system and "8507 | MALE" in system
    assert "all columns typed text" in system and "VOCABULARIES LOADED: " in system
    assert "to_date(x, 'DD/MM/YYYY')" in system and "Never pass a strftime pattern" in system
    assert "never auto_id" in system and "normally kind auto_id" not in system
    prompt = person_call["prompt"]
    assert "TABLE person:" in prompt and "SOURCE TABLE individuals" in prompt
    assert "SOURCE TABLE diagnoses" not in prompt  # only this table's sources
    assert "concept_map  local codes" not in prompt and "8507 | MALE" not in prompt  # not repeated per table
    assert "Person key: individuals.individual_id" in prompt


@pytest.mark.postgres
def test_design_prompts_carry_the_notes(staged, tmp_path):
    conn, schemas, cfg = staged
    ref = load_ref()
    replay = tmp_path / "replay"
    record_reference(replay, ref)
    ws = Workspace(tmp_path / "work")
    notes = ["individuals.sex code 9 means unknown.", "12 lab results belong to people missing from the registry."]
    annotations = SourceAnnotations(tables=[], columns=[], notes=notes)
    design_etl(profile_from_db(conn, schemas), annotations, cfg, ReplayLLM(replay, ws), conn, schemas, ws)
    calls = [json.loads(c.read_text()) for c in (ws.root / "llm_calls").glob("*.json")]
    assert len(calls) == 1 + len(ref.table_mappings)
    for call in calls:  # the plan and every table see the notes
        assert "NOTES ON THE SOURCE DATA" in call["system"]
        assert all(f"- {n}" in call["system"] for n in notes)


@pytest.mark.postgres
def test_design_revises_invalid_mapping(staged, tmp_path):
    conn, schemas, cfg = staged
    ref = load_ref()
    replay = tmp_path / "replay"
    record_reference(replay, ref)
    broken = drop_field(ref, "person", "year_of_birth")
    broken = set_field(broken, "person", "gender_concept_id", expression="CASE p.sex WHEN '1' THEN 8507 ELSE 0 END")
    save_recording(replay, "design", "table:person",
                   {"kind": "structured", "output": broken.mapping_for("person").model_dump(mode="json")})
    save_recording(replay, "design", "table:person:revision1",
                   {"kind": "structured", "output": ref.mapping_for("person").model_dump(mode="json")})
    ws = Workspace(tmp_path / "work")

    spec = design_etl(profile_from_db(conn, schemas), None, cfg, ReplayLLM(replay, ws), conn, schemas, ws)
    assert spec.model_dump() == ref.model_dump()
    rev = [e for e in ws.read_audit() if e["event"] == "design_revision"]
    assert [e["key"] for e in rev] == ["table:person:revision1"]
    assert any("year_of_birth" in x for x in rev[0]["issues"])
    call = next((ws.root / "llm_calls").glob("*_table_person_revision1.json"))
    prompt = json.loads(call.read_text())["prompt"]
    assert "YOUR PREVIOUS MAPPING" in prompt and "alias 'p' is not declared" in prompt


@pytest.mark.postgres
def test_design_keeps_remaining_issues(staged, tmp_path):
    conn, schemas, cfg = staged
    ref = load_ref()
    replay = tmp_path / "replay"
    record_reference(replay, ref)
    broken = drop_field(ref, "death", "death_date")
    save_recording(replay, "design", "table:death",
                   {"kind": "structured", "output": broken.mapping_for("death").model_dump(mode="json")})
    ws = Workspace(tmp_path / "work")

    spec = design_etl(profile_from_db(conn, schemas), None, cfg, ReplayLLM(replay, ws), conn, schemas, ws)
    assert spec.mapping_for("death").model_dump() == broken.mapping_for("death").model_dump()
    issues = json.loads((ws.root / "design_issues.json").read_text())
    assert any(i["severity"] == "error" and i["target_field"] == "death_date" for i in issues)
    assert [e["key"] for e in ws.read_audit() if e["event"] == "design_revision_unavailable"] == ["table:death:revision1"]


def test_design_needs_an_llm(tmp_path):
    profile = SourceProfile(source_type="csv", generated_at=datetime.now(timezone.utc), min_cell_count=5, tables=[])
    cfg = ProjectConfig.model_validate(
        {"name": "t", "source": {"type": "csv", "path": str(tmp_path)}, "target": {"url": "postgresql://unused"}}
    )
    with pytest.raises(RuntimeError, match="needs an LLM"):
        design_etl(profile, None, cfg, NullLLM(), None, Schemas("cdm", "cdm", "src", "etl", "results"))


# --------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------


def test_render_spec_md():
    spec = set_field(
        load_ref(), "condition_occurrence", "condition_concept_id",
        expression="CASE WHEN d.icd10 ~ '^[A-Z][0-9]{3}$' THEN substr(d.icd10,1,3)||'.'||substr(d.icd10,4) ELSE d.icd10 END",
    )
    md = render_spec_md(spec)
    assert md.startswith("# ETL specification")
    assert "**Person key:** `individuals`.`individual_id`" in md
    assert "**Observation period:** loaded by a table mapping" in md
    assert "| 1 | individuals | → | person | 11 |" in md
    assert "### condition_occurrence ← `diagnoses` (d)" in md
    assert "| CDM field | kind | source / expression | vocabulary | rationale |" in md
    assert "\\|\\|'.'\\|\\|" in md  # pipes in expressions are escaped
    assert "| condition_concept_id | vocab_code |" in md and "ICD10 (standard)" in md
    assert "SRC_DRUG → Drug" in md
    assert "| drug_exposure_id | auto_id | *generated surrogate key* |" in md
    assert "## Assumptions" in md and "- HDSS residency episodes define observation periods." in md
    assert "## Open questions" in md
    # escaped pipes keep the row at five cells
    rows = [ln for ln in md.splitlines() if ln.startswith("| condition_concept_id")]
    assert rows and rows[0].replace("\\|", "").count("|") == 6
