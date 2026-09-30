"""ETL design: static/DB validation, the replayed design flow with revisions, markdown rendering."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from omop_agent.config import ProjectConfig
from omop_agent.db import Schemas, list_columns, list_tables, row_count
from omop_agent.design.designer import DESIGN_SYSTEM, design_etl, reference_concepts_text
from omop_agent.design.render import render_spec_md
from omop_agent.design.validate import SpecIssue, validate_spec, validate_table_mapping
from omop_agent.llm import NullLLM, ReplayLLM
from omop_agent.llm.replay import save_recording
from omop_agent.models import (
    ColumnProfile,
    ETLSpec,
    FieldMapping,
    JoinSpec,
    PlannedTable,
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
    issues = validate_spec(load_ref())
    assert errors(issues) == []
    # the verbal-autopsy cause mapping has no target domain: a warning, not an error
    assert any(i.severity == "warning" and i.target_field == "cause_concept_id" for i in issues)


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
    spec = set_field(load_ref(), "person", "birth_datetime", expression="pg_catalog.to_timestamp(i.dob, 'DD/MM/YYYY')")
    assert errors(validate_spec(spec)) == []
    spec = spec.model_copy(deep=True)
    spec.mapping_for("person").joins.append(JoinSpec(table="individuals", alias="i", on="TRUE"))
    assert has(validate_spec(spec), "person", None, "alias 'i' is declared twice")


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
    assert len(reference_concepts_text(conn, schemas, limit_per_domain=1).splitlines()) == 1 + 3


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
    assert person_call["system"] == DESIGN_SYSTEM
    prompt = person_call["prompt"]
    assert "TABLE person:" in prompt and "SOURCE TABLE individuals" in prompt
    assert "SOURCE TABLE diagnoses" not in prompt  # only this table's sources
    assert "concept_map  local codes" in prompt and "8507 | MALE" in prompt
    assert "all columns typed text" in prompt


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
