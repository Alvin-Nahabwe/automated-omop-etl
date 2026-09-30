"""ETL compiler and runner: SQL shape (no database) and loads of the demo data (PostgreSQL)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from omop_agent.db import Schemas, execute, fetch_all, scalar
from omop_agent.etl.compiler import CompileError, compile_spec, compile_table, target_tables
from omop_agent.etl.runner import run_etl
from omop_agent.models import CustomSQL, ETLSpec, FieldMapping, JoinSpec, TableMapping
from omop_agent.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"
S = Schemas(cdm="cdm", vocab="vocab", src="src", etl="etl", results="results")
ICD10_DOT = "CASE WHEN d.icd10 ~ '^[A-Z][0-9]{3}$' THEN substr(d.icd10,1,3)||'.'||substr(d.icd10,4) ELSE d.icd10 END"


def load_ref() -> ETLSpec:
    return ETLSpec.model_validate(yaml.safe_load((FIXTURES / "demo_etl_spec.yaml").read_text()))


def set_field(spec: ETLSpec, table: str, field: str, **changes) -> ETLSpec:
    """Copy of ``spec`` with one field mapping changed (or added when missing)."""
    spec = spec.model_copy(deep=True)
    tm = spec.mapping_for(table)
    assert tm is not None
    fm = next((f for f in tm.fields if f.target_field == field), None)
    if fm is None:
        tm.fields.append(FieldMapping(target_field=field, **changes))
    else:
        tm.fields[tm.fields.index(fm)] = fm.model_copy(update=changes)
    return spec


def fixed_ref() -> ETLSpec:
    return set_field(load_ref(), "visit_occurrence", "visit_end_date",
                     expression="COALESCE(v.discharge_date::date, v.visit_date::date)")


def step(steps, name_suffix: str):
    return next(s for s in steps if s.name.split("_", 1)[1] == name_suffix)


# --------------------------------------------------------------------------------------
# compiler (no database)
# --------------------------------------------------------------------------------------


def test_step_order_and_targets():
    spec = load_ref()
    steps = compile_spec(spec, S)
    assert [s.name for s in steps] == [
        "01_person_map",
        "02_visit_map",
        "03_person",
        "04_person_map_prune",
        "05_visit_occurrence",
        "06_visit_map_prune",
        "07_condition_occurrence",
        "08_drug_exposure",
        "09_measurement",
        "10_death",
        "11_observation_period",
        "12_cdm_source",
    ]
    assert [s.kind for s in steps] == ["id_map", "id_map", "insert", "id_map", "insert", "id_map"] + ["insert"] * 5 + [
        "metadata"
    ]
    assert [s.requires for s in steps if s.requires] == ["person", "visit_occurrence"]
    assert target_tables(spec) == [
        "person", "visit_occurrence", "condition_occurrence", "drug_exposure",
        "measurement", "death", "observation_period", "cdm_source",
    ]
    # placeholders are rendered everywhere
    assert all("{" + k + "}" not in s.sql for s in steps for k in ("cdm", "src", "etl", "vocab"))


def test_id_maps():
    spec = load_ref()
    steps = compile_spec(spec, S)
    pm = steps[0].sql
    # stable across runs: never truncated; only new keys, numbered after the largest existing id
    assert "TRUNCATE" not in pm
    assert "SELECT DISTINCT (individual_id)::text AS source_value" in pm
    assert "FROM src.individuals" in pm and "WHERE (individual_id) IS NOT NULL" in pm
    assert (
        "((SELECT COALESCE(MAX(person_id), 0) FROM etl.person_map) + ROW_NUMBER() OVER (ORDER BY k.source_value))"
        in pm
    )
    assert "WHERE NOT EXISTS (SELECT 1 FROM etl.person_map pm WHERE pm.source_value = k.source_value)" in pm
    assert pm.rstrip().endswith("ON CONFLICT DO NOTHING;")
    assert steps[0].origins == (spec.person_key,)
    vm = steps[1].sql
    assert "TRUNCATE" not in vm
    assert "INSERT INTO etl.visit_map (source_value, visit_occurrence_id, person_source_value)" in vm
    assert "MIN((v.individual_id)::text)" in vm  # person column taken from the visit mapping
    assert "FROM src.clinic_visits v" in vm
    assert "(SELECT COALESCE(MAX(visit_occurrence_id), 0) FROM etl.visit_map)" in vm
    assert steps[1].origins == (spec.visit_key, spec.mapping_for("visit_occurrence"))
    # pruning after the person / visit loads
    prune = step(steps, "person_map_prune")
    assert prune.target_table is None and prune.requires == "person"
    assert "DELETE FROM etl.person_map pm\nWHERE NOT EXISTS (SELECT 1 FROM cdm.person p WHERE p.person_id = pm.person_id);" in prune.sql
    vprune = step(steps, "visit_map_prune")
    assert "DELETE FROM etl.visit_map vm" in vprune.sql and "FROM cdm.visit_occurrence v" in vprune.sql
    # no visit key, no visit map and no visit pruning
    spec = load_ref().model_copy(update={"visit_key": None})
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table == "person"]
    assert [s.name for s in compile_spec(spec, S)] == ["01_person_map", "02_person", "03_person_map_prune",
                                                        "04_cdm_source"]


def test_person_prune_follows_person_custom_sql():
    spec = load_ref().model_copy(deep=True)
    spec.custom_sql = [CustomSQL(name="extra people", target_table="person", sql="SELECT 1")]
    names = [s.name for s in compile_spec(spec, S)]
    assert names[2:5] == ["03_person", "04_custom_extra_people", "05_person_map_prune"]


def test_casts_and_person_join():
    sql = compile_table(load_ref().mapping_for("person"), load_ref(), S)
    assert sql.splitlines()[0].startswith("-- person <- individuals (i)")
    assert "INSERT INTO cdm.person (" in sql
    assert "INNER JOIN etl.person_map pm1 ON pm1.source_value = (i.individual_id)::text" in sql
    assert "(pm1.person_id)::integer AS person_id,  -- person_id" in sql
    assert "(CASE i.sex WHEN '1' THEN 8507 WHEN '2' THEN 8532 ELSE 0 END)::integer AS gender_concept_id" in sql
    assert "(to_date(i.dob, 'DD/MM/YYYY')::timestamp)::timestamp AS birth_datetime" in sql
    assert "LEFT((i.individual_id)::text, 50) AS person_source_value" in sql
    assert "(0)::integer AS race_concept_id,  -- constant" in sql
    # unmapped CDM fields are left out of the column list
    assert "location_id" not in sql and "race_source_value" not in sql


def test_visit_joins_and_auto_id():
    spec = load_ref()
    vo = compile_table(spec.mapping_for("visit_occurrence"), spec, S)
    assert "INNER JOIN etl.visit_map vm1 ON vm1.source_value = (v.visit_id)::text" in vo  # required PK
    co = compile_table(spec.mapping_for("condition_occurrence"), spec, S)
    assert "LEFT JOIN etl.visit_map vm1 ON vm1.source_value = (d.visit_id)::text" in co  # optional FK
    assert (
        "((SELECT COALESCE(MAX(condition_occurrence_id), 0) FROM cdm.condition_occurrence)"
        " + ROW_NUMBER() OVER ())::integer AS condition_occurrence_id" in co
    )
    de = compile_table(spec.mapping_for("drug_exposure"), spec, S)
    assert "(p.qty::numeric)::numeric AS quantity" in de


def test_vocab_code_lookup_and_null_semantics():
    spec = load_ref()
    sql = compile_table(spec.mapping_for("condition_occurrence"), spec, S)
    # standard + source roles share one concept join
    assert sql.count("JOIN vocab.concept sc") == 1
    assert "LEFT JOIN vocab.concept sc1 ON sc1.concept_code = (d.icd10)::text AND sc1.vocabulary_id = 'ICD10'" in sql
    assert (
        "LEFT JOIN vocab.concept_relationship cr1 ON cr1.concept_id_1 = sc1.concept_id"
        " AND cr1.relationship_id = 'Maps to' AND cr1.invalid_reason IS NULL" in sql
    )
    assert (
        "LEFT JOIN vocab.concept tc1 ON tc1.concept_id = cr1.concept_id_2"
        " AND tc1.standard_concept = 'S' AND tc1.invalid_reason IS NULL" in sql
    )
    # required field: NULL source -> 0; optional field: NULL source -> NULL
    assert "(CASE WHEN (d.icd10) IS NULL THEN 0 ELSE COALESCE(tc1.concept_id, 0) END)::integer AS condition_concept_id" in sql
    assert "(CASE WHEN (d.icd10) IS NULL THEN NULL ELSE COALESCE(sc1.concept_id, 0) END)::integer AS condition_source_concept_id" in sql


def test_standard_role_added_after_source_role():
    # fields compile in CDM order; a later standard-role field on the same (code, vocabulary)
    # must add its Maps-to joins right after the concept join it shares
    spec = set_field(load_ref(), "condition_occurrence", "condition_concept_id", concept_role="source")
    spec = set_field(spec, "condition_occurrence", "condition_status_concept_id", kind="vocab_code",
                     expression="d.dx_text", vocabulary="SNOMED")
    spec = set_field(spec, "condition_occurrence", "condition_source_concept_id", concept_role="standard")
    sql = compile_table(spec.mapping_for("condition_occurrence"), spec, S)
    lines = [ln for ln in sql.splitlines() if ln.startswith("LEFT JOIN vocab.")]
    assert [ln.split()[3] for ln in lines] == ["sc1", "cr2", "tc2", "sc2", "cr1", "tc1"]
    assert "COALESCE(sc1.concept_id, 0) END)::integer AS condition_concept_id" in sql
    assert "COALESCE(tc2.concept_id, 0) END)::integer AS condition_source_concept_id" in sql


def test_concept_map_joins_are_shared():
    spec = set_field(load_ref(), "measurement", "measurement_source_concept_id", kind="concept_map",
                     expression="r.test", vocabulary="SRC_LAB", concept_role="source")
    sql = compile_table(spec.mapping_for("measurement"), spec, S)
    assert (
        "LEFT JOIN etl.source_to_concept_map m1 ON m1.source_code = (r.test)::text"
        " AND m1.source_vocabulary_id = 'SRC_LAB' AND m1.invalid_reason IS NULL" in sql
    )
    assert sql.count("source_to_concept_map m") == 3  # SRC_LAB (shared), SRC_RESULT, SRC_UNIT
    assert "COALESCE(m1.target_concept_id, 0) END)::integer AS measurement_concept_id" in sql
    assert "COALESCE(m1.source_concept_id, 0) END)::integer AS measurement_source_concept_id" in sql
    assert "THEN NULL ELSE COALESCE(m3.target_concept_id, 0) END)::integer AS unit_concept_id" in sql


def test_source_count_sql_excludes_lookups():
    steps = compile_spec(load_ref(), S)
    m = step(steps, "measurement")
    assert m.source_table == "lab_results"
    assert m.source_count_sql == "SELECT COUNT(*) AS n\nFROM src.lab_results r"
    # retained: + the INNER person-map join; the optional visit link and the lookups can't drop rows
    assert m.retained_count_sql == (
        "SELECT COUNT(*) AS n\nFROM src.lab_results r\n"
        "INNER JOIN etl.person_map pm1 ON pm1.source_value = (r.individual_id)::text"
    )
    vo = step(steps, "visit_occurrence")  # the required visit key is an INNER join too
    assert vo.retained_count_sql == (
        "SELECT COUNT(*) AS n\nFROM src.clinic_visits v\n"
        "INNER JOIN etl.visit_map vm1 ON vm1.source_value = (v.visit_id)::text\n"
        "INNER JOIN etl.person_map pm1 ON pm1.source_value = (v.individual_id)::text"
    )
    assert "concept" not in step(steps, "condition_occurrence").retained_count_sql
    assert all(s.source_count_sql is None and s.retained_count_sql is None for s in steps if s.kind != "insert")


def test_where_joins_and_custom_sql():
    spec = load_ref().model_copy(deep=True)
    tm = spec.mapping_for("drug_exposure")
    tm.joins.append(JoinSpec(table="clinic_visits", alias="cv", join_type="inner", on="cv.visit_id = p.visit_id"))
    tm.where = "cv.visit_type <> 'EMERG'"
    spec.custom_sql = [
        CustomSQL(name="drop orphans", target_table="drug_exposure", sql="DELETE FROM {cdm}.drug_exposure WHERE person_id IS NULL"),
        CustomSQL(name="note", target_table="note", sql="SELECT 1"),
    ]
    steps = compile_spec(spec, S)
    names = [s.name for s in steps]
    assert names.index("09_custom_drop_orphans") == names.index("08_drug_exposure") + 1
    assert names[-2] == "13_custom_note"  # remaining custom SQL after the inserts
    de = step(steps, "drug_exposure")
    assert "INNER JOIN src.clinic_visits cv ON cv.visit_id = p.visit_id" in de.sql
    assert de.sql.rstrip().endswith("WHERE (cv.visit_type <> 'EMERG');")
    assert de.source_count_sql.endswith("INNER JOIN src.clinic_visits cv ON cv.visit_id = p.visit_id\nWHERE (cv.visit_type <> 'EMERG')")
    assert de.retained_count_sql.endswith(
        "INNER JOIN src.clinic_visits cv ON cv.visit_id = p.visit_id\n"
        "INNER JOIN etl.person_map pm1 ON pm1.source_value = (p.individual_id)::text\n"
        "WHERE (cv.visit_type <> 'EMERG')"
    )
    custom = steps[names.index("09_custom_drop_orphans")]
    assert "DELETE FROM cdm.drug_exposure" in custom.sql and custom.origins == (spec.custom_sql[0],)
    assert "note" in target_tables(spec)


def test_from_events_and_cdm_source_steps():
    spec = load_ref().model_copy(update={"observation_period": "from_events"})
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "observation_period"]
    steps = compile_spec(spec, S)
    op = step(steps, "observation_period")
    assert op.kind == "derived"
    for col in ("visit_start_date", "visit_end_date", "condition_start_date", "drug_exposure_end_date",
                "measurement_date", "death_date"):
        assert col in op.sql
    assert "procedure_date" not in op.sql  # procedure_occurrence is not loaded
    assert "32817" in op.sql
    cs = steps[-1]
    assert cs.name.endswith("_cdm_source") and cs.kind == "metadata"
    assert "vocabulary_id = 'CDM' AND concept_code = 'CDM v5.4'" in cs.sql
    assert "WHERE vocabulary_id = 'None'), 'unknown')" in cs.sql


def test_trailing_comment_in_expression_is_safe():
    spec = set_field(load_ref(), "person", "year_of_birth",
                     expression="EXTRACT(YEAR FROM to_date(i.dob, 'DD/MM/YYYY'))::int -- day-first dates")
    sql = compile_table(spec.mapping_for("person"), spec, S)
    assert "-- day-first dates\n)::integer AS year_of_birth" in sql


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda s: s.table_mappings.append(TableMapping(target_table="patients", source={"table": "x", "alias": "x"}, fields=[])), "not an OMOP CDM"),
        (lambda s: s.table_mappings[0].fields.append(FieldMapping(target_field="shoe_size", kind="constant", expression="1")), "not a field"),
        (lambda s: setattr(s.table_mappings[4].fields[2], "vocabulary", None), "needs a vocabulary"),
        (lambda s: setattr(s.table_mappings[3].fields[2], "vocabulary", None), "needs a vocabulary"),
        (lambda s: setattr(s.table_mappings[0].fields[2], "expression", None), "needs an expression"),
        (lambda s: setattr(s.table_mappings[0].fields[2], "kind", "auto_id"), "primary key"),
        # person/visit keys must come from the id maps, or events attach to the wrong rows
        (lambda s: setattr(s.table_mappings[0].fields[0], "kind", "auto_id"), "use kind person_id"),
        (lambda s: setattr(s.table_mappings[2].fields[0], "kind", "auto_id"), "use kind visit_id"),
        (lambda s: s.table_mappings[0].fields.append(s.table_mappings[0].fields[1].model_copy()), "more than once"),
        (lambda s: setattr(s, "visit_key", None), "visit_key"),
    ],
)
def test_compile_errors(change, message):
    spec = load_ref().model_copy(deep=True)
    change(spec)
    with pytest.raises(CompileError, match=message):
        compile_spec(spec, S)
    steps = compile_spec(spec, S, strict=False)
    assert any(s.error and message in s.error for s in steps)


# --------------------------------------------------------------------------------------
# runner (PostgreSQL)
# --------------------------------------------------------------------------------------


def count(conn, schemas, table: str, where: str = "TRUE") -> int:
    return int(scalar(conn, f"SELECT COUNT(*) FROM {schemas.cdm}.{table} WHERE {where}"))


@pytest.mark.postgres
def test_run_reference_spec(staged):
    conn, schemas, cfg = staged
    ws = Workspace(cfg.workdir)
    (ws.sql_dir / "99_stale.sql").write_text("-- old\n")
    result = run_etl(conn, schemas, load_ref(), cfg, ws)

    errors = result.errors()
    assert len(errors) == 1, [(e.step, e.error) for e in errors]
    assert errors[0].target_table == "visit_occurrence"
    assert "null value" in errors[0].error and "visit_end_date" in errors[0].error
    assert errors[0].source_rows == int(scalar(conn, f"SELECT COUNT(*) FROM {schemas.src}.clinic_visits"))
    assert count(conn, schemas, "person") == 400
    assert count(conn, schemas, "visit_occurrence") == 0
    by_table = {s.target_table: s for s in result.steps}
    for t in ("condition_occurrence", "drug_exposure", "measurement", "death", "observation_period"):
        assert by_table[t].error is None and by_table[t].rows_inserted > 0, t
        assert by_table[t].rows_inserted == count(conn, schemas, t)
    # retention: lab results of people missing from the registry are dropped by the person map
    m = by_table["measurement"]
    assert m.source_rows - m.rows_inserted == 12
    assert m.source_rows - m.retained_rows == 12 and m.retained_rows == m.rows_inserted
    assert by_table["person_map"].rows_inserted == 400
    assert by_table["person_map_prune"].rows_inserted == 0 and by_table["person_map_prune"].error is None
    assert count(conn, schemas, "cdm_source") == 1
    # the visit load failed: its id map is not pruned (the ids stay stable for the next run)
    prune = next(s for s in result.steps if s.step.endswith("visit_map_prune"))
    assert prune.error is None and prune.rows_inserted == 0
    assert scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.visit_map") == errors[0].source_rows
    audit = [e for e in ws.read_audit() if e["event"] == "etl_step" and e["step"] == prune.step]
    assert audit[0]["skipped"] == "a step loading visit_occurrence failed"
    assert [e["retained_rows"] for e in ws.read_audit() if e.get("step") == "09_measurement"] == [m.retained_rows]

    files = sorted(p.name for p in ws.sql_dir.glob("*.sql"))
    assert files[0] == "01_person_map.sql" and "07_condition_occurrence.sql" in files
    assert "99_stale.sql" not in files
    saved = json.loads(ws.etl_run_json.read_text())
    assert len(saved["steps"]) == len(result.steps)
    events = [e["event"] for e in ws.read_audit()]
    assert events.count("etl_step") == len(result.steps) and events[-1] == "etl_run"
    assert ws.read_audit()[-1]["ok"] is False


@pytest.mark.postgres
def test_fixed_spec_and_icd10_normalisation(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref()
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    assert count(conn, schemas, "visit_occurrence") == int(scalar(conn, f"SELECT COUNT(*) FROM {schemas.src}.clinic_visits"))
    assert count(conn, schemas, "visit_occurrence", "visit_end_date < visit_start_date") == 0
    n_cond = count(conn, schemas, "condition_occurrence")
    unmapped = count(conn, schemas, "condition_occurrence", "condition_concept_id = 0")
    assert 0 < unmapped < n_cond / 3  # the dot bug: codes written without the dot
    assert count(conn, schemas, "condition_occurrence", "condition_source_concept_id = 0") == unmapped
    assert count(conn, schemas, "condition_occurrence", "visit_occurrence_id IS NULL") == 0

    spec = set_field(spec, "condition_occurrence", "condition_concept_id", expression=ICD10_DOT)
    spec = set_field(spec, "condition_occurrence", "condition_source_concept_id", expression=ICD10_DOT)
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok
    assert count(conn, schemas, "condition_occurrence") == n_cond  # rebuilt, not appended
    assert count(conn, schemas, "condition_occurrence", "condition_concept_id = 0") == 0
    assert count(conn, schemas, "condition_occurrence", "condition_source_concept_id = 0") == 0
    assert count(conn, schemas, "person") == 400


@pytest.mark.postgres
def test_concept_map_fields_follow_source_to_concept_map(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref()
    assert run_etl(conn, schemas, spec, cfg).ok
    assert count(conn, schemas, "drug_exposure", "drug_concept_id <> 0") == 0
    assert count(conn, schemas, "measurement", "measurement_concept_id <> 0") == 0
    # numeric results give a NULL value_as_concept_id; empty units a NULL unit_concept_id
    assert count(conn, schemas, "measurement", "value_as_number IS NOT NULL AND value_as_concept_id IS NOT NULL") == 0
    assert count(conn, schemas, "measurement", "value_source_value IN ('POS', 'NEG') AND value_as_concept_id = 0") > 0
    assert count(conn, schemas, "measurement", "unit_source_value IS NULL AND unit_concept_id IS NOT NULL") == 0

    execute(
        conn,
        f"""INSERT INTO {schemas.etl}.source_to_concept_map (source_code, source_vocabulary_id, target_concept_id)
            VALUES ('Paracetamol 500mg', 'SRC_DRUG', 2000003002), ('POS', 'SRC_RESULT', 2000006001),
                   ('NEG', 'SRC_RESULT', 2000006002), ('g/dL', 'SRC_UNIT', 2000005001), ('HB', 'SRC_LAB', 2000004001)""",
    )
    execute(
        conn,
        f"""INSERT INTO {schemas.etl}.source_to_concept_map (source_code, source_vocabulary_id, target_concept_id, invalid_reason)
            VALUES ('ORS sachets', 'SRC_DRUG', 2000003004, 'D')""",
    )
    conn.commit()
    assert run_etl(conn, schemas, spec, cfg).ok
    para = count(conn, schemas, "drug_exposure", "drug_source_value = 'Paracetamol 500mg'")
    assert para > 0
    assert count(conn, schemas, "drug_exposure", "drug_concept_id = 2000003002") == para
    assert count(conn, schemas, "drug_exposure", "drug_source_value = 'ORS sachets' AND drug_concept_id <> 0") == 0
    assert count(conn, schemas, "measurement", "value_source_value = 'POS' AND value_as_concept_id <> 2000006001") == 0
    assert count(conn, schemas, "measurement", "value_source_value = 'NEG' AND value_as_concept_id <> 2000006002") == 0
    assert count(conn, schemas, "measurement", "unit_source_value = 'g/dL' AND unit_concept_id <> 2000005001") == 0
    assert count(conn, schemas, "measurement", "measurement_source_value = 'HB' AND measurement_concept_id <> 2000004001") == 0
    assert count(conn, schemas, "measurement", "measurement_source_value = 'HB'") > 0


@pytest.mark.postgres
def test_auto_id_with_two_mappings_into_one_table(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref()
    base = spec.mapping_for("measurement")
    numeric = base.model_copy(deep=True, update={"where": r"r.result ~ '^-?[0-9]+(\.[0-9]+)?$'"})
    other = base.model_copy(deep=True, update={"where": r"r.result !~ '^-?[0-9]+(\.[0-9]+)?$'"})
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "measurement"] + [numeric, other]
    names = [s.name for s in compile_spec(spec, schemas)]
    assert "09_measurement" in names and "10_measurement_2" in names

    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok
    parts = [s for s in result.steps if s.target_table == "measurement"]
    assert len(parts) == 2 and all(p.rows_inserted > 0 for p in parts)
    total = count(conn, schemas, "measurement")
    assert total == sum(p.rows_inserted for p in parts)
    row = fetch_all(conn, f"SELECT COUNT(DISTINCT measurement_id) AS d, MIN(measurement_id) AS lo, MAX(measurement_id) AS hi FROM {schemas.cdm}.measurement")[0]
    assert (row["d"], row["lo"], row["hi"]) == (total, 1, total)


@pytest.mark.postgres
def test_observation_period_from_events(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref().model_copy(update={"observation_period": "from_events", "observation_period_type_concept_id": 32817})
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "observation_period"]
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok
    op = next(s for s in result.steps if s.target_table == "observation_period")
    assert op.step.endswith("_observation_period") and op.rows_inserted > 0
    with_events = scalar(
        conn,
        f"""SELECT COUNT(DISTINCT person_id) FROM (
              SELECT person_id FROM {schemas.cdm}.visit_occurrence UNION SELECT person_id FROM {schemas.cdm}.condition_occurrence
              UNION SELECT person_id FROM {schemas.cdm}.drug_exposure UNION SELECT person_id FROM {schemas.cdm}.measurement
              UNION SELECT person_id FROM {schemas.cdm}.death) x""",
    )
    assert count(conn, schemas, "observation_period") == with_events
    assert count(conn, schemas, "observation_period", "period_type_concept_id <> 32817") == 0
    assert count(conn, schemas, "observation_period", "observation_period_start_date > observation_period_end_date") == 0
    outside = scalar(
        conn,
        f"""SELECT COUNT(*) FROM {schemas.cdm}.drug_exposure d
            JOIN {schemas.cdm}.observation_period o ON o.person_id = d.person_id
            WHERE d.drug_exposure_start_date < o.observation_period_start_date
               OR d.drug_exposure_end_date > o.observation_period_end_date""",
    )
    assert outside == 0


@pytest.mark.postgres
def test_cdm_source_row(staged):
    conn, schemas, cfg = staged
    cfg = cfg.model_copy(update={"cdm_source": cfg.cdm_source.model_copy(update={"cdm_source_name": "Demo HDSS", "cdm_holder": "Test"})})
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    rows = fetch_all(conn, f"SELECT * FROM {schemas.cdm}.cdm_source")
    assert len(rows) == 1
    r = rows[0]
    assert r["cdm_source_name"] == "Demo HDSS" and r["cdm_holder"] == "Test"
    assert r["cdm_version"] == "5.4"
    assert r["vocabulary_version"] == "v5.0 FIXTURE"
    assert r["cdm_version_concept_id"] == 0  # the fixture vocabulary has no CDM vocabulary
    assert r["cdm_release_date"] is not None and r["source_release_date"] is not None


@pytest.mark.postgres
def test_compile_error_does_not_stop_other_tables(staged, tmp_path):
    conn, schemas, cfg = staged
    ws = Workspace(tmp_path / "ws")
    spec = set_field(fixed_ref(), "death", "shoe_size", kind="constant", expression="42")
    # a table dropped from the spec is emptied on the next run
    execute(conn, f"INSERT INTO {schemas.cdm}.specimen (specimen_id, person_id, specimen_concept_id, specimen_type_concept_id, specimen_date) VALUES (1, 1, 0, 32817, DATE '2020-01-01')")
    conn.commit()
    ws.etl_run_json.write_text(json.dumps({"started_at": "2026-01-01T00:00:00Z", "steps": [{"target_table": "specimen", "step": "09_specimen"}]}))
    result = run_etl(conn, schemas, spec, cfg, ws)
    errors = result.errors()
    assert [e.target_table for e in errors] == ["death"]
    assert errors[0].error.startswith("compile error:")
    assert count(conn, schemas, "death") == 0 and count(conn, schemas, "person") == 400
    assert count(conn, schemas, "specimen") == 0
    assert (ws.sql_dir / "10_death.sql").read_text().startswith("-- compile error")


def id_map(conn, schemas, name: str) -> dict[str, int]:
    col = "person_id" if name == "person_map" else "visit_occurrence_id"
    return {r["source_value"]: r[col] for r in fetch_all(conn, f"SELECT source_value, {col} FROM {schemas.etl}.{name}")}


def person_ids(conn, schemas) -> dict[str, int]:
    rows = fetch_all(conn, f"SELECT person_source_value, person_id FROM {schemas.cdm}.person")
    return {r["person_source_value"]: r["person_id"] for r in rows}


def orphans(conn, schemas) -> dict[str, int]:
    """Rows whose person_id is not in person, or whose visit_occurrence_id is not in visit_occurrence."""
    out = {}
    for t in ("visit_occurrence", "condition_occurrence", "drug_exposure", "measurement", "death", "observation_period"):
        out[t] = count(conn, schemas, t, f"person_id NOT IN (SELECT person_id FROM {schemas.cdm}.person)")
    for t in ("condition_occurrence", "drug_exposure", "measurement"):
        out[t + ".visit"] = count(
            conn, schemas, t, f"visit_occurrence_id NOT IN (SELECT visit_occurrence_id FROM {schemas.cdm}.visit_occurrence)"
        )
    return out


@pytest.mark.postgres
def test_id_maps_are_stable_across_runs(staged):
    conn, schemas, cfg = staged
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    people, visits, persons = id_map(conn, schemas, "person_map"), id_map(conn, schemas, "visit_map"), person_ids(conn, schemas)
    assert len(people) == 400 and sorted(people.values()) == list(range(1, 401))

    # a new individual whose key sorts before everyone else: renumbering would shift every id
    execute(conn, f"INSERT INTO {schemas.src}.individuals (individual_id, sex, dob) VALUES ('AA000001', '1', '01/02/1990')")
    execute(conn, f"INSERT INTO {schemas.src}.clinic_visits (visit_id, individual_id, visit_date, visit_type) "
                  "VALUES ('AAV1', 'AA000001', '2020-01-01', 'OPD')")
    conn.commit()
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert result.ok
    assert next(s for s in result.steps if s.target_table == "person_map").rows_inserted == 1
    people2, visits2 = id_map(conn, schemas, "person_map"), id_map(conn, schemas, "visit_map")
    assert {k: people2[k] for k in people} == people and people2["AA000001"] == 401
    assert {k: visits2[k] for k in visits} == visits and visits2["AAV1"] == max(visits.values()) + 1
    persons2 = person_ids(conn, schemas)
    assert {k: persons2[k] for k in persons} == persons and persons2["AA000001"] == 401
    assert count(conn, schemas, "visit_occurrence", f"visit_occurrence_id = {visits2['AAV1']} AND person_id = 401") == 1


@pytest.mark.postgres
def test_id_maps_are_pruned_to_loaded_people_and_visits(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref().model_copy(deep=True)
    spec.mapping_for("person").where = "i.sex <> '9'"
    spec.mapping_for("visit_occurrence").where = "v.visit_type <> 'EMERG'"
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    src = schemas.src
    kept = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.individuals WHERE sex <> '9'"))
    assert 0 < kept < 400 and count(conn, schemas, "person") == kept
    by_step = {s.step.split("_", 1)[1]: s for s in result.steps}
    assert by_step["person_map_prune"].rows_inserted == 400 - kept
    assert int(scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.person_map")) == kept
    assert int(scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.visit_map")) == count(conn, schemas, "visit_occurrence")
    # nothing points at a person or visit that was filtered out ...
    assert set(orphans(conn, schemas).values()) == {0}
    # ... events of the unknown-sex people are dropped, and links to emergency visits are NULL
    assert count(conn, schemas, "condition_occurrence", "visit_occurrence_id IS NULL") > 0
    co = by_step["condition_occurrence"]
    dropped = int(scalar(conn, f"""SELECT COUNT(*) FROM {src}.diagnoses d
        WHERE NOT EXISTS (SELECT 1 FROM {src}.individuals i WHERE i.individual_id = d.individual_id AND i.sex <> '9')"""))
    assert co.source_rows - co.retained_rows == dropped > 0 and co.rows_inserted == co.retained_rows

    # without the filter, the pruned people come back with new ids after the largest; the rest keep theirs
    before = person_ids(conn, schemas)
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    after = person_ids(conn, schemas)
    assert len(after) == 400 and {k: after[k] for k in before} == before
    assert min(v for k, v in after.items() if k not in before) > max(before.values())


@pytest.mark.postgres
def test_failed_person_load_keeps_the_id_map(staged, tmp_path):
    conn, schemas, cfg = staged
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    people = id_map(conn, schemas, "person_map")
    broken = set_field(fixed_ref(), "person", "year_of_birth", expression="i.sex::int / 0")
    ws = Workspace(tmp_path / "ws")
    result = run_etl(conn, schemas, broken, cfg, ws)
    assert [e.target_table for e in result.errors()] == ["person"] and "division by zero" in result.errors()[0].error
    prune = next(s for s in result.steps if s.step.endswith("person_map_prune"))
    assert prune.error is None and prune.rows_inserted == 0
    assert id_map(conn, schemas, "person_map") == people  # not wiped although person is empty
    assert any(e.get("skipped") for e in ws.read_audit() if e.get("step") == prune.step)
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    assert id_map(conn, schemas, "person_map") == people


@pytest.mark.postgres
def test_retained_rows_ignore_vocabulary_fan_out(staged):
    conn, schemas, cfg = staged
    # 'HB' maps to two concepts: each HB result is inserted twice
    execute(conn, f"""INSERT INTO {schemas.etl}.source_to_concept_map (source_code, source_vocabulary_id, target_concept_id)
        VALUES ('HB', 'SRC_LAB', 2000004001), ('HB', 'SRC_LAB', 2000004002)""")
    conn.commit()
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert result.ok
    m = next(s for s in result.steps if s.target_table == "measurement")
    extra = count(conn, schemas, "measurement", "measurement_concept_id = 2000004002")
    assert extra > 0 and m.rows_inserted == m.retained_rows + extra == count(conn, schemas, "measurement")
    assert m.source_rows - m.retained_rows == 12  # the real drops, not netted against the fan-out
    assert m.source_rows - m.rows_inserted < 0


@pytest.mark.postgres
def test_steps_run_under_the_etl_role(staged):
    conn, schemas, cfg = staged
    role = f"{schemas.cdm}_role"
    conn.rollback()
    execute(conn, f"CREATE ROLE {role} NOLOGIN")
    execute(conn, f"GRANT USAGE ON SCHEMA {schemas.src}, {schemas.vocab}, {schemas.cdm}, {schemas.etl} TO {role}")
    execute(conn, f"GRANT SELECT ON ALL TABLES IN SCHEMA {schemas.src}, {schemas.vocab} TO {role}")
    execute(conn, f"GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA {schemas.cdm}, {schemas.etl} TO {role}")
    execute(conn, f"CREATE SCHEMA {schemas.results}")
    execute(conn, f"CREATE TABLE {schemas.results}.secret AS SELECT 1 AS x")
    conn.commit()
    try:
        cfg = cfg.model_copy(update={"target": cfg.target.model_copy(update={"etl_role": role})})
        spec = fixed_ref().model_copy(deep=True)
        spec.custom_sql = [
            CustomSQL(name="who", target_table="metadata", sql=(
                "INSERT INTO {cdm}.metadata (metadata_id, metadata_concept_id, metadata_type_concept_id, name, "
                "value_as_string) SELECT 1, 0, 0, 'etl user', current_user")),
            CustomSQL(name="peek", target_table="metadata",
                      sql="DELETE FROM {cdm}.metadata WHERE metadata_id IN (SELECT x FROM {results}.secret)"),
        ]
        result = run_etl(conn, schemas, spec, cfg)
        errors = [(e.step, e.error) for e in result.errors()]
        assert len(errors) == 1 and errors[0][0].endswith("custom_peek") and "permission denied" in errors[0][1], errors
        assert fetch_all(conn, f"SELECT value_as_string FROM {schemas.cdm}.metadata") == [{"value_as_string": role}]
        assert count(conn, schemas, "person") == 400 and count(conn, schemas, "visit_occurrence") > 0
        assert scalar(conn, "SELECT current_user") == scalar(conn, "SELECT session_user")  # SET LOCAL only
        # the same spec without the role reads the other schema
        cfg = cfg.model_copy(update={"target": cfg.target.model_copy(update={"etl_role": None})})
        assert run_etl(conn, schemas, spec, cfg).ok
    finally:
        conn.rollback()
        execute(conn, f"DROP OWNED BY {role}")
        execute(conn, f"DROP ROLE {role}")
        conn.commit()


@pytest.mark.postgres
def test_runner_refuses_unsafe_spec_sql(staged):
    conn, schemas, cfg = staged
    spec = set_field(fixed_ref(), "person", "gender_source_value", expression="pg_read_file('/etc/hostname')")
    spec.custom_sql = [CustomSQL(name="drop", target_table="death", sql="DROP TABLE {cdm}.death")]
    result = run_etl(conn, schemas, spec, cfg)
    errors = {e.target_table: e.error for e in result.errors()}
    assert set(errors) == {"person", "death"}
    assert errors["person"].startswith("refused: unsafe SQL in the spec: person.gender_source_value:")
    assert "pg_read_file()" in errors["person"] and "DROP" in errors["death"]
    assert count(conn, schemas, "person") == 0 and count(conn, schemas, "death") > 0  # the table still exists
