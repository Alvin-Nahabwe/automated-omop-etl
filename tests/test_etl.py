"""ETL compiler and runner: SQL shape (no database) and loads of the demo data (PostgreSQL)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from omop_agent.cdm.ddl import create_etl_tables
from omop_agent.cdm.spec import LOAD_ORDER
from omop_agent.db import Schemas, execute, fetch_all, scalar
from omop_agent.etl.compiler import (
    SPEC_KINDS,
    CompiledStep,
    CompileError,
    compile_spec,
    compile_table,
    step_label,
    target_tables,
)
from omop_agent.etl.runner import _analyzed_after, _run_step, run_etl
from omop_agent.etl.session import session_sql
from omop_agent.models import CustomSQL, ETLSpec, FieldMapping, JoinSpec, TableLoadResult, TableMapping
from omop_agent.sqlguard import strip_literals
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
    assert [s.kind for s in steps] == ["id_map", "id_map", "insert", "prune", "insert", "prune"] + ["insert"] * 5 + [
        "metadata"
    ]
    assert [s.requires for s in steps if s.requires] == []  # pruning runs after a failed load too
    # sort keys: (load index of the table, band, sequence); id maps first, cdm_source last
    assert [s.slot for s in steps] == sorted(s.slot for s in steps)
    assert steps[0].slot[0] == steps[1].slot[0] == -1 and steps[-1].slot[0] > len(LOAD_ORDER) - 1
    assert step(steps, "person_map_prune").slot == (LOAD_ORDER.index("person"), 1, 0)
    assert [s.label for s in steps][:3] == ["person_map", "visit_map", "person"]
    # every generated step is blocking and reads a row count; its SQL is its statements (two for
    # the id maps and the visit pruning); the spec kinds
    assert all(s.blocking and s.returns == "rowcount" and s.count_sites == () for s in steps)
    assert all("\n".join(s.statements) == s.sql for s in steps)
    assert [s.label for s in steps if len(s.statements) > 1] == ["person_map", "visit_map", "visit_map_prune"]
    assert {s.kind for s in steps if s.kind in SPEC_KINDS} == {"id_map", "insert"}
    assert target_tables(spec) == [
        "person", "visit_occurrence", "condition_occurrence", "drug_exposure",
        "measurement", "death", "observation_period", "cdm_source",
    ]
    # placeholders are rendered everywhere
    assert all("{" + k + "}" not in s.sql for s in steps for k in ("cdm", "src", "etl", "vocab"))


def test_id_maps():
    spec = load_ref()
    steps = compile_spec(spec, S)
    pm = steps[0]
    # stable across runs: never truncated; retired keys that are back get their old id, then only
    # new keys are added, numbered after the largest id ever given out (map or retired)
    assert "TRUNCATE" not in pm.sql
    restore, new = pm.statements
    keys = "    SELECT DISTINCT (individual_id)::text AS source_value\n    FROM src.individuals\n" \
           "    WHERE (individual_id) IS NOT NULL\n"
    assert f"WITH k AS (\n{keys})," in restore and f"FROM (\n{keys}) k" in new
    assert (
        "back AS (\n    DELETE FROM etl.person_map_retired r USING k\n    WHERE r.source_value = k.source_value\n"
        "    RETURNING r.source_value, r.person_id\n)\nINSERT INTO etl.person_map (source_value, person_id)\n"
        "SELECT source_value, person_id FROM back\nON CONFLICT DO NOTHING;"
    ) in restore
    assert (
        "    (GREATEST((SELECT COALESCE(MAX(person_id), 0) FROM etl.person_map),\n"
        "              (SELECT COALESCE(MAX(person_id), 0) FROM etl.person_map_retired))\n"
        "     + ROW_NUMBER() OVER (ORDER BY k.source_value))::integer\n"
    ) in new
    assert "WHERE NOT EXISTS (SELECT 1 FROM etl.person_map pm WHERE pm.source_value = k.source_value)" in new
    assert new.rstrip().endswith("ON CONFLICT DO NOTHING;")  # the step's row count: the new keys
    assert pm.origins == (spec.person_key,)
    vm = steps[1]
    assert "TRUNCATE" not in vm.sql and len(vm.statements) == 2
    restore, new = vm.statements
    assert "INSERT INTO etl.visit_map (source_value, visit_occurrence_id, person_source_value)" in new
    assert "MIN((v.individual_id)::text)" in new  # person column taken from the visit mapping
    assert "FROM src.clinic_visits v" in restore and "FROM src.clinic_visits v" in new
    # a restored visit takes the person the visit table gives it now
    assert "DELETE FROM etl.visit_map_retired r USING k" in restore
    assert ("RETURNING r.source_value, r.visit_occurrence_id, "
            "COALESCE(k.person_source_value, r.person_source_value) AS person_source_value") in restore
    assert "(SELECT COALESCE(MAX(visit_occurrence_id), 0) FROM etl.visit_map_retired)" in new
    assert vm.origins == (spec.visit_key, spec.mapping_for("visit_occurrence"))
    # pruning after the person / visit loads retires keys (never deletes them) and requires nothing:
    # it runs after a failed load too
    prune = step(steps, "person_map_prune")
    assert (prune.kind, prune.target_table, prune.requires) == ("prune", None, None)
    assert (
        "WITH gone AS (\n    DELETE FROM etl.person_map pm\n"
        "    WHERE NOT EXISTS (SELECT 1 FROM cdm.person p WHERE p.person_id = pm.person_id)\n"
        "    RETURNING pm.source_value, pm.person_id\n)\n"
        "INSERT INTO etl.person_map_retired (source_value, person_id)\nSELECT source_value, person_id FROM gone\n"
    ) in prune.sql
    vprune = step(steps, "visit_map_prune")
    assert (vprune.kind, vprune.requires) == ("prune", None)
    refresh, retire = vprune.statements
    # 1) each loaded visit's person is recorded (DELETE + INSERT: the ETL role has no UPDATE)
    assert "USING cdm.visit_occurrence v, etl.person_map pm" in refresh
    assert "AND vm.person_source_value IS DISTINCT FROM pm.source_value" in refresh
    assert "UPDATE" not in strip_literals(refresh)
    # 2) then the keys without a visit are retired with their person
    assert "DELETE FROM etl.visit_map vm" in retire and "FROM cdm.visit_occurrence v" in retire
    assert "INSERT INTO etl.visit_map_retired (source_value, visit_occurrence_id, person_source_value)" in retire
    # no visit key, no visit map and no visit pruning
    spec = load_ref().model_copy(update={"visit_key": None})
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table == "person"]
    assert [s.name for s in compile_spec(spec, S)] == ["01_person_map", "02_person", "03_person_map_prune",
                                                        "04_cdm_source"]


def test_person_prune_follows_person_custom_sql():
    spec = load_ref().model_copy(deep=True)
    spec.custom_sql = [CustomSQL(name="extra people", target_table="person", sql="SELECT 1")]
    steps = compile_spec(spec, S)
    names = [s.name for s in steps]
    assert names[2:5] == ["03_person", "04_custom_extra_people", "05_person_map_prune"]
    assert [s.kind for s in steps[2:5]] == ["insert", "custom", "prune"]
    assert steps[3].statements == ()  # custom SQL is not split into statements

    # people loaded only by custom SQL: the prune still follows them, before the visits
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "person"]
    names = [s.name for s in compile_spec(spec, S)]
    assert names[2:5] == ["03_custom_extra_people", "04_person_map_prune", "05_visit_occurrence"]


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
    # custom SQL of a table without a mapping runs at that table's place in the load order
    assert names[-4:] == ["11_death", "12_custom_note", "13_observation_period", "14_cdm_source"]
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


def test_custom_sql_runs_at_its_tables_load_order_slot():
    spec = load_ref().model_copy(deep=True)
    spec.custom_sql = [
        CustomSQL(name="eras", target_table="drug_era", sql="SELECT 1"),
        CustomSQL(name="sites", target_table="care_site", sql="SELECT 1"),
        CustomSQL(name="va deaths", target_table="death", sql="SELECT 1"),
        CustomSQL(name="specimens", target_table="specimen", sql="SELECT 2"),
        CustomSQL(name="more specimens", target_table="specimen", sql="SELECT 3"),
        CustomSQL(name="dates", target_table="observation_period", sql="SELECT 4"),
        CustomSQL(name="other", target_table="not_a_cdm_table", sql="SELECT 5"),
    ]
    labels = [s.label for s in compile_spec(spec, S)]
    assert labels == [
        "person_map", "visit_map", "custom_sites", "person", "person_map_prune", "visit_occurrence",
        "visit_map_prune", "condition_occurrence", "drug_exposure", "measurement", "death", "custom_va_deaths",
        "custom_specimens", "custom_more_specimens", "observation_period", "custom_dates", "custom_eras",
        "custom_other", "cdm_source",
    ]
    # a derived observation period follows every spec writer of observation_period
    spec = spec.model_copy(update={"observation_period": "from_events"})
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "observation_period"]
    labels = [s.label for s in compile_spec(spec, S)]
    assert labels[labels.index("custom_specimens"):labels.index("custom_eras")] == [
        "custom_specimens", "custom_more_specimens", "custom_dates", "observation_period"]
    assert step_label("05_person_2") == "person_2" and step_label("person") == "person"


def check_step_invariants(steps: list[CompiledStep]) -> None:
    """What the runner relies on in every compiled step list."""
    by_label = {s.label: s for s in steps}
    for s in steps:
        # a skip is not an error: a blocking step may only require blocking steps, or the build
        # would pass without it
        if s.requires and s.blocking:
            needed = [by_label[s.requires]] if s.requires in by_label else [
                w for w in steps if w.target_table == s.requires and w.kind != "route"]
            assert needed and all(n.blocking for n in needed), s.name
        # EXPLAIN takes one statement at a time
        if s.sql and s.kind != "custom":
            assert s.statements, s.name
            assert all(";" not in strip_literals(x).strip().rstrip(";") for x in s.statements), s.name


def test_step_invariants():
    spec = load_ref().model_copy(deep=True)
    spec.custom_sql = [CustomSQL(name="extra people", target_table="person", sql="SELECT 1; SELECT 2"),
                       CustomSQL(name="notes", target_table="note", sql="SELECT 3")]
    check_step_invariants(compile_spec(spec, S))
    spec = spec.model_copy(update={"observation_period": "from_events"})
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "observation_period"]
    check_step_invariants(compile_spec(spec, S))
    # what the check catches
    with pytest.raises(AssertionError, match="02_x"):
        check_step_invariants([CompiledStep("01_r", None, "route", "SELECT 1", blocking=False, statements=("SELECT 1",)),
                               CompiledStep("02_x", None, "derived", "SELECT 1", requires="r", statements=("SELECT 1",))])
    with pytest.raises(AssertionError, match="01_two"):
        check_step_invariants([CompiledStep("01_two", None, "merge", "SELECT 1; SELECT 2",
                                            statements=("SELECT 1; SELECT 2",))])


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
    assert "vocabulary_id = 'CDM' AND concept_code IN ('CDM v5.4.0', 'CDM v5.4')" in cs.sql
    assert "WHERE vocabulary_id = 'None'), 'unknown')" in cs.sql
    assert "    'omop-agent',\n    CURRENT_DATE,\n    CURRENT_DATE," in cs.sql
    # a configured extract date is the source release date
    from datetime import date

    from omop_agent.config import CdmSourceConfig

    configured = compile_spec(spec, S, CdmSourceConfig(source_release_date=date(2023, 12, 31)))[-1]
    assert "    'omop-agent',\n    DATE '2023-12-31',\n    CURRENT_DATE," in configured.sql


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
    # the visit load failed: pruning retires every visit key with its id (the ids stay stable for
    # the next run), and the events get no visit instead of one missing from visit_occurrence
    prune = next(s for s in result.steps if s.step.endswith("visit_map_prune"))
    assert prune.error is None and prune.rows_inserted == errors[0].source_rows
    assert scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.visit_map") == 0
    assert scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.visit_map_retired") == errors[0].source_rows
    assert count(conn, schemas, "condition_occurrence", "visit_occurrence_id IS NOT NULL") == 0
    audit = [e for e in ws.read_audit() if e["event"] == "etl_step" and e["step"] == prune.step]
    assert "skipped" not in audit[0] and audit[0]["rows_inserted"] == prune.rows_inserted
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
    """source key -> id of an id map (person_map, visit_map) or of the keys it retired (*_retired)."""
    col = "person_id" if name.startswith("person_map") else "visit_occurrence_id"
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
def test_etl_tables(staged):
    conn, schemas, _ = staged

    def columns(table: str) -> list[str]:
        rows = fetch_all(conn, "SELECT column_name FROM information_schema.columns WHERE table_schema = %s "
                               "AND table_name = %s ORDER BY ordinal_position", (schemas.etl, table))
        return [r["column_name"] for r in rows]

    assert columns("person_map_retired") == ["source_value", "person_id"]
    assert columns("visit_map_retired") == ["source_value", "visit_occurrence_id", "person_source_value"]
    assert columns("id_high_water") == ["table_name", "max_id"]
    assert columns("build_facts") == ["name", "value"]
    assert create_etl_tables(conn, schemas) == []  # all there
    # a missing table is created; a role that does not exist is not granted anything (the build
    # reports it when it sets the role)
    execute(conn, f"DROP TABLE {schemas.etl}.build_facts")
    assert create_etl_tables(conn, schemas, f"{schemas.cdm}_no_such_role") == ["build_facts"]
    conn.commit()


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
    # the pruned keys are retired with their ids, not deleted
    retired, visits_retired = id_map(conn, schemas, "person_map_retired"), id_map(conn, schemas, "visit_map_retired")
    assert len(retired) == 400 - kept and not set(retired) & set(id_map(conn, schemas, "person_map"))
    assert len(visits_retired) == by_step["visit_map_prune"].rows_inserted > 0
    # nothing points at a person or visit that was filtered out ...
    assert set(orphans(conn, schemas).values()) == {0}
    # ... events of the unknown-sex people are dropped, and links to emergency visits are NULL
    assert count(conn, schemas, "condition_occurrence", "visit_occurrence_id IS NULL") > 0
    co = by_step["condition_occurrence"]
    dropped = int(scalar(conn, f"""SELECT COUNT(*) FROM {src}.diagnoses d
        WHERE NOT EXISTS (SELECT 1 FROM {src}.individuals i WHERE i.individual_id = d.individual_id AND i.sex <> '9')"""))
    assert co.source_rows - co.retained_rows == dropped > 0 and co.rows_inserted == co.retained_rows

    # without the filter, the pruned people and visits come back with their old ids; the rest keep theirs
    before, visits = person_ids(conn, schemas), id_map(conn, schemas, "visit_map")
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert result.ok and next(s for s in result.steps if s.target_table == "person_map").rows_inserted == 0
    after = person_ids(conn, schemas)
    assert len(after) == 400 and after == before | retired
    assert id_map(conn, schemas, "visit_map") == visits | visits_retired
    assert id_map(conn, schemas, "person_map_retired") == id_map(conn, schemas, "visit_map_retired") == {}


@pytest.mark.postgres
def test_failed_person_load_keeps_the_id_map(staged, tmp_path):
    conn, schemas, cfg = staged
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    people, visits = id_map(conn, schemas, "person_map"), id_map(conn, schemas, "visit_map")
    broken = set_field(fixed_ref(), "person", "year_of_birth", expression="i.sex::int / 0")
    ws = Workspace(tmp_path / "ws")
    result = run_etl(conn, schemas, broken, cfg, ws)
    assert [e.target_table for e in result.errors()] == ["person"] and "division by zero" in result.errors()[0].error
    # pruning still runs: everyone is retired with their id (none is lost), so no event points at
    # a person who is not in person
    prune = next(s for s in result.steps if s.step.endswith("person_map_prune"))
    assert prune.error is None and prune.skipped is None and prune.rows_inserted == 400
    assert id_map(conn, schemas, "person_map") == {} and id_map(conn, schemas, "person_map_retired") == people
    assert not any(e.get("skipped") for e in ws.read_audit() if e.get("step") == prune.step)
    assert set(orphans(conn, schemas).values()) == {0} and count(conn, schemas, "condition_occurrence") == 0
    assert id_map(conn, schemas, "visit_map") == {} and id_map(conn, schemas, "visit_map_retired") == visits
    # the next good build restores every id
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    assert id_map(conn, schemas, "person_map") == people and id_map(conn, schemas, "visit_map") == visits
    assert id_map(conn, schemas, "person_map_retired") == id_map(conn, schemas, "visit_map_retired") == {}


@pytest.mark.postgres
def test_ids_never_reused_and_restored(staged):
    conn, schemas, cfg = staged
    src = schemas.src

    def build(filtered: bool):
        spec = fixed_ref().model_copy(deep=True)
        if filtered:  # the person and the visit with the largest keys (and ids) are not loaded
            spec.mapping_for("person").where = "i.individual_id <> 'KH000400'"
            spec.mapping_for("visit_occurrence").where = "v.visit_id <> 'V0000890'"
        result = run_etl(conn, schemas, spec, cfg)
        assert result.ok, [(e.step, e.error) for e in result.errors()]
        return {s.step.split("_", 1)[1]: s.rows_inserted for s in result.steps}

    def source(person: str, visit: str, add: bool = True) -> None:
        if add:
            execute(conn, f"INSERT INTO {src}.individuals (individual_id, sex, dob) VALUES ('{person}', '1', '01/02/1990')")
            execute(conn, f"INSERT INTO {src}.clinic_visits (visit_id, individual_id, visit_date, visit_type) "
                          f"VALUES ('{visit}', '{person}', '2020-01-01', 'OPD')")
        else:
            execute(conn, f"DELETE FROM {src}.clinic_visits WHERE visit_id = '{visit}'")
            execute(conn, f"DELETE FROM {src}.individuals WHERE individual_id = '{person}'")
        conn.commit()

    def maps(name: str) -> tuple[dict[str, int], dict[str, int]]:
        return id_map(conn, schemas, name), id_map(conn, schemas, f"{name}_retired")

    # 1) the largest keys are filtered out: retired with their ids
    build(filtered=True)
    people, people_retired = maps("person_map")
    visits, visits_retired = maps("visit_map")
    assert people_retired == {"KH000400": 400} and max(people.values()) == 399
    assert visits_retired["V0000890"] == 890 and max(visits.values()) < 890
    # 2) new keys are numbered after every id ever given out: the retired ids are not reused
    source("KH000401", "V0000891")
    assert build(filtered=True)["person_map"] == 1
    assert maps("person_map") == (people | {"KH000401": 401}, people_retired)
    assert maps("visit_map") == (visits | {"V0000891": 891}, visits_retired)
    # 3) without the filter the retired keys come back with their old ids; the newest keys leave
    #    the source and are retired in turn
    source("KH000401", "V0000891", add=False)
    build(filtered=False)
    assert maps("person_map") == (people | people_retired, {"KH000401": 401})
    assert maps("visit_map") == (visits | visits_retired, {"V0000891": 891})
    assert person_ids(conn, schemas)["KH000400"] == 400
    assert count(conn, schemas, "visit_occurrence", "visit_occurrence_id = 890 AND person_id = 400") == 1
    # 4) a new key gets the next id, not the one of the retired key at the top (the old maps,
    #    which deleted pruned keys, gave it out again)
    source("KH000402", "V0000892")
    assert build(filtered=False)["person_map"] == 1
    assert maps("person_map") == (people | people_retired | {"KH000402": 402}, {"KH000401": 401})
    assert maps("visit_map") == (visits | visits_retired | {"V0000892": 892}, {"V0000891": 891})
    # 5) a key that returns gets its old id, and is not counted as a new key
    source("KH000401", "V0000891")
    assert build(filtered=False)["person_map"] == 0
    assert maps("person_map") == (people | people_retired | {"KH000401": 401, "KH000402": 402}, {})
    assert maps("visit_map") == (visits | visits_retired | {"V0000891": 891, "V0000892": 892}, {})
    assert {k: person_ids(conn, schemas)[k] for k in ("KH000401", "KH000402")} == {"KH000401": 401, "KH000402": 402}


@pytest.mark.postgres
def test_visit_person_follows_the_source_between_runs(staged):
    conn, schemas, cfg = staged
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    visits = id_map(conn, schemas, "visit_map")
    persons = person_ids(conn, schemas)

    def visit_people() -> dict[str, str]:
        rows = fetch_all(conn, f"SELECT source_value, person_source_value FROM {schemas.etl}.visit_map")
        return {r["source_value"]: r["person_source_value"] for r in rows}

    assert visit_people()["V0000001"] == "KH000001"
    # the extract now gives the visit to another person; maps written by older versions may hold
    # no person at all
    execute(conn, f"UPDATE {schemas.src}.clinic_visits SET individual_id = 'KH000002' WHERE visit_id = 'V0000001'")
    execute(conn, f"UPDATE {schemas.etl}.visit_map SET person_source_value = NULL WHERE source_value <> 'V0000001'")
    conn.commit()
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    # the visit keeps its id; the map records the person it was loaded for
    assert id_map(conn, schemas, "visit_map") == visits
    people = visit_people()
    assert people["V0000001"] == "KH000002"
    loaded = fetch_all(conn, f"SELECT v.visit_occurrence_id, p.person_source_value FROM {schemas.cdm}.visit_occurrence v "
                             f"JOIN {schemas.cdm}.person p ON p.person_id = v.person_id")
    by_id = {v: k for k, v in visits.items()}
    assert {by_id[r["visit_occurrence_id"]]: r["person_source_value"] for r in loaded} == people
    assert count(conn, schemas, "visit_occurrence",
                 f"visit_occurrence_id = {visits['V0000001']} AND person_id = {persons['KH000002']}") == 1


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
    # an ETL schema of an older version, granted before the upgrade: the build creates the tables
    # added since then and grants them to the role
    new_tables = ["person_map_retired", "visit_map_retired", "id_high_water", "build_facts"]
    for t in new_tables:
        execute(conn, f"DROP TABLE {schemas.etl}.{t}")
    conn.commit()
    try:
        cfg = cfg.model_copy(update={"target": cfg.target.model_copy(update={"etl_role": role})})
        spec = fixed_ref().model_copy(deep=True)
        # filtered visits are retired under the role: the grants cover the retired-key tables
        spec.mapping_for("visit_occurrence").where = "v.visit_type <> 'EMERG'"
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
        assert len(id_map(conn, schemas, "visit_map_retired")) > 0
        for t in new_tables:
            assert scalar(conn, "SELECT has_table_privilege(%s, %s, 'SELECT, INSERT, DELETE')", (role, f"{schemas.etl}.{t}"))
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
    assert count(conn, schemas, "person") == 0
    assert scalar(conn, f"SELECT to_regclass('{schemas.cdm}.death') IS NOT NULL")  # the table still exists


# --------------------------------------------------------------------------------------
# step mechanics (kinds, returns, counters, skips, session settings, ANALYZE)
# --------------------------------------------------------------------------------------


def test_analyze_policy():
    def res(**kw):
        return TableLoadResult(target_table="x", step="01_x", **kw)

    insert = CompiledStep("05_condition_occurrence", "condition_occurrence", "insert", "INSERT ...")
    assert _analyzed_after(insert, res(rows_inserted=3), S, last_id_map=False) == ["cdm.condition_occurrence"]
    assert _analyzed_after(insert, res(error="boom"), S, last_id_map=False) == []
    assert _analyzed_after(insert, res(skipped="a step loading person failed"), S, last_id_map=False) == []
    # the id maps and their retired keys: after the last id-map step and after each prune
    id_map = CompiledStep("02_visit_map", None, "id_map", "INSERT ...")
    maps = ["etl.person_map", "etl.visit_map", "etl.person_map_retired", "etl.visit_map_retired"]
    assert _analyzed_after(id_map, res(), S, last_id_map=True) == maps
    assert _analyzed_after(id_map, res(), S, last_id_map=False) == []
    prune = CompiledStep("04_person_map_prune", None, "prune", "WITH gone AS (DELETE ...")
    assert _analyzed_after(prune, res(), S, last_id_map=False) == maps
    # a route step: its source table and the destinations of what it moved
    route = CompiledStep("12_route_measurement", "measurement", "route", "WITH ...", returns="rows")
    moved = res(moved=[{"from_table": "measurement", "to_table": "observation", "domain_id": "Observation",
                        "moved_rows": 4}])
    assert _analyzed_after(route, moved, S, last_id_map=False) == ["cdm.measurement", "cdm.observation"]
    # the cdm_source row and custom SQL into a non-CDM table are not analyzed
    assert _analyzed_after(CompiledStep("20_cdm_source", "cdm_source", "metadata", "INSERT ..."), res(), S,
                           last_id_map=False) == []
    assert _analyzed_after(CompiledStep("09_custom_x", "elsewhere", "custom", "SELECT 1"), res(), S,
                           last_id_map=False) == []


def test_session_sql():
    from datetime import date

    from omop_agent.config import CdmSourceConfig, EtlConfig

    # without a configured zone the session keeps the server's
    assert session_sql() == [
        "SET LOCAL datestyle = 'ISO, YMD';",
        "SELECT set_config('omop_agent.imputation_cap', CURRENT_DATE::text, true);",
    ]
    cfg = make_test_config().model_copy(update={
        "etl": EtlConfig(timezone="Africa/Nairobi"),
        "cdm_source": CdmSourceConfig(source_release_date=date(2023, 12, 31)),
    })
    assert session_sql(cfg)[1:] == [
        "SET LOCAL timezone = 'Africa/Nairobi';",
        "SELECT set_config('omop_agent.imputation_cap', '2023-12-31', true);",
    ]


def make_test_config():
    from omop_agent.config import ProjectConfig

    return ProjectConfig.model_validate({"name": "t", "source": {"type": "csv", "path": "."},
                                         "target": {"url": "postgresql://unused"}})


@pytest.mark.postgres
def test_step_returns_counters_and_session(staged):
    from datetime import date

    from omop_agent.config import CdmSourceConfig, EtlConfig

    conn, schemas, cfg = staged
    cfg = cfg.model_copy(update={
        "etl": EtlConfig(timezone="Africa/Nairobi"),
        "cdm_source": CdmSourceConfig(source_release_date=date(2023, 12, 31)),
    })
    # returns="row": one row of statistics, 'inserted' is the row count; the settings are pinned
    stats = CompiledStep(
        "01_merge", "observation_period", "merge",
        "SELECT 1;\nSELECT 7 AS inserted, 3::numeric AS episodes, DATE '2020-01-31' AS release_date,"
        " current_setting('DateStyle') AS datestyle, current_setting('TimeZone') AS tz,"
        " current_setting('omop_agent.imputation_cap') AS cap",
        returns="row",
    )
    res = _run_step(conn, stats, cfg)
    assert res.error is None and (res.kind, res.blocking) == ("merge", True)
    assert res.rows_inserted == 7
    assert res.stats == {"inserted": 7, "episodes": 3, "release_date": "2020-01-31", "datestyle": "ISO, YMD",
                         "tz": "Africa/Nairobi", "cap": "2023-12-31"}
    # SET LOCAL: the settings end with the step's transaction
    assert scalar(conn, "SELECT current_setting('omop_agent.imputation_cap', true)") in (None, "")
    # without a configured zone and release date: the session's own zone, and the cap is the current date
    with conn.transaction(force_rollback=True):
        execute(conn, "SET LOCAL timezone = 'Asia/Kolkata'")  # e.g. the server's zone
        res = _run_step(conn, stats, None)
    assert res.stats["tz"] == "Asia/Kolkata" and res.stats["cap"] == date.today().isoformat()

    # returns="rows": the records a route step moved; a non-blocking step reports its kind
    route = CompiledStep(
        "02_route_measurement", "measurement", "route",
        "SELECT * FROM (VALUES ('observation', 'Observation', 4, 1), ('condition_occurrence', 'Condition', 2, 0))"
        " AS t(to_table, domain_id, moved_rows, rows_with_dropped_fields)",
        returns="rows", blocking=False,
    )
    res = _run_step(conn, route, cfg)
    assert res.error is None and res.blocking is False and res.rows_inserted == 0
    assert [(m.from_table, m.to_table, m.domain_id, m.moved_rows, m.rows_with_dropped_fields) for m in res.moved] == [
        ("measurement", "observation", "Observation", 4, 1), ("measurement", "condition_occurrence", "Condition", 2, 0)]

    # counters of the step's sites go to value_counts; dropped rows leave retained_rows
    counted = CompiledStep(
        "03_person", "person", "insert",
        "SELECT set_config('omop_agent.invalid__year_of_birth', '2', true),"
        " set_config('omop_agent.defaulted__gender_concept_id', '5', true),"
        " set_config('omop_agent.dropped__row', '1', true),"
        " set_config('omop_agent.invalid__not_a_site', '9', true)",
        count_sites=("year_of_birth", "gender_concept_id", "row"),
        retained_count_sql="SELECT 5 AS n, 3 AS visit_key_rows",
    )
    res = _run_step(conn, counted, cfg)
    assert res.error is None
    assert res.value_counts == {"invalid": {"year_of_birth": 2}, "defaulted": {"gender_concept_id": 5},
                                "row": {"dropped": 1}}
    assert (res.id_retained_rows, res.retained_rows) == (5, 4)
    assert res.stats == {"visit_links": {"visit_key_rows": 3}}
    # counters set earlier in an enclosing transaction are reset at step start
    with conn.transaction():
        execute(conn, "SELECT set_config('omop_agent.invalid__year_of_birth', '8', true)")
        quiet = CompiledStep("04_person", "person", "insert", "SELECT 1", count_sites=("year_of_birth",))
        assert _run_step(conn, quiet, cfg).value_counts == {}
    # a failed step reports no counts
    broken = CompiledStep("05_person", "person", "insert",
                          "SELECT set_config('omop_agent.dropped__row', '3', true); SELECT 1/0",
                          count_sites=("row",), retained_count_sql="SELECT 4 AS n")
    res = _run_step(conn, broken, cfg)
    assert "division by zero" in res.error and res.value_counts == {} and res.retained_rows == 4


@pytest.mark.postgres
def test_a_step_result_that_cannot_be_read_fails_that_step(staged):
    conn, schemas, cfg = staged
    execute(conn, f"CREATE TABLE {schemas.etl}.p_result_probe (n int)")
    conn.commit()
    probe = f"INSERT INTO {schemas.etl}.p_result_probe VALUES (1);\n"
    try:
        # a moved-rows row without to_table: the step fails (rolled back), the build goes on
        route = CompiledStep("12_route_measurement", "measurement", "route",
                             probe + "SELECT 'Observation' AS domain_id, 3 AS moved_rows", returns="rows",
                             blocking=False)
        res = _run_step(conn, route, cfg)
        assert res.error == "unexpected step result: to_table: Field required" and res.moved == []
        merge = CompiledStep("16_observation_period_merge", "observation_period", "merge",
                             probe + "SELECT 'n/a' AS inserted", returns="row")
        res = _run_step(conn, merge, cfg)
        assert res.error.startswith("unexpected step result: invalid literal for int()") and res.stats == {}
        counted = CompiledStep("03_person", "person", "insert",
                               probe + "SELECT set_config('omop_agent.invalid__year_of_birth', 'x', true)",
                               count_sites=("year_of_birth",))
        assert _run_step(conn, counted, cfg).error.startswith("unexpected step result:")
        assert scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.p_result_probe") == 0  # every step rolled back
    finally:
        conn.rollback()
        execute(conn, f"DROP TABLE {schemas.etl}.p_result_probe")
        conn.commit()


@pytest.mark.postgres
def test_skips_propagate_and_non_blocking_errors_do_not_fail_the_build(staged, tmp_path, monkeypatch):
    import omop_agent.etl.runner as runner

    conn, schemas, cfg = staged
    ws = Workspace(tmp_path / "ws")
    steps = [
        CompiledStep("01_route_prepare", None, "route", "SELECT 1/0", blocking=False),
        CompiledStep("02_route_measurement", "measurement", "route", "SELECT 1", requires="route_prepare",
                     blocking=False, returns="rows"),
        CompiledStep("03_after_route", None, "derived", "SELECT 1", requires="route_measurement", blocking=False),
        CompiledStep("04_condition_era", "condition_era", "derived", "SELECT 1/0", blocking=False),
        CompiledStep("05_drug_era", "drug_era", "derived", "SELECT 1", requires="condition_era", blocking=False),
        CompiledStep("06_cdm_source", "cdm_source", "metadata", "SELECT 1"),
        # a route step's own table is not marked failed: steps loading measurement still run
        CompiledStep("07_measurement_check", None, "derived", "SELECT 1", requires="measurement"),
    ]
    monkeypatch.setattr(runner, "compile_spec", lambda *a, **k: steps)
    result = run_etl(conn, schemas, fixed_ref(), cfg, ws)
    by = {s.step: s for s in result.steps}
    assert [s.kind for s in result.steps] == ["route", "route", "derived", "derived", "derived", "metadata", "derived"]
    assert by["02_route_measurement"].skipped == "step route_prepare failed"
    assert by["03_after_route"].skipped == "step route_measurement failed"  # the skip propagates
    assert by["05_drug_era"].skipped == "a step loading condition_era failed"
    assert by["07_measurement_check"].skipped is None and by["06_cdm_source"].skipped is None
    assert [s.step for s in result.errors()] == ["01_route_prepare", "04_condition_era"]
    assert result.ok and result.blocking_errors() == []  # only non-blocking steps failed
    audit = {e["step"]: e for e in ws.read_audit() if e["event"] == "etl_step"}
    assert audit["04_condition_era"]["blocking"] is False and audit["04_condition_era"]["kind"] == "derived"
    assert audit["03_after_route"]["skipped"] == "step route_measurement failed"
    run = [e for e in ws.read_audit() if e["event"] == "etl_run"][-1]
    assert run["ok"] is True and run["totals"]["errors"] == 2 and run["totals"]["blocking_errors"] == 0
    assert run["totals"]["skipped"] == 3
    # each SQL file records the session settings it ran under
    assert (ws.sql_dir / "06_cdm_source.sql").read_text().startswith("-- session: SET LOCAL datestyle = 'ISO, YMD';")

    steps[5] = CompiledStep("06_cdm_source", "cdm_source", "metadata", "SELECT 1/0")
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert not result.ok and [s.step for s in result.blocking_errors()] == ["06_cdm_source"]


@pytest.mark.postgres
def test_spec_sql_is_read_under_the_session_settings(staged):
    from omop_agent.config import EtlConfig
    from omop_agent.vocab.evaluate import query_rows

    conn, schemas, cfg = staged
    cfg = cfg.model_copy(update={"etl": EtlConfig(timezone="Asia/Kolkata")})
    sql = "SELECT current_setting('DateStyle') AS ds, current_setting('TimeZone') AS tz"
    assert query_rows(conn, sql, cfg) == [{"ds": "ISO, YMD", "tz": "Asia/Kolkata"}]
    server_zone = scalar(conn, "SELECT current_setting('TimeZone')")
    assert query_rows(conn, sql) == [{"ds": "ISO, YMD", "tz": server_zone}]
    # the build: a timestamptz literal is read in the site zone
    spec = fixed_ref().model_copy(deep=True)
    spec.custom_sql = [CustomSQL(name="when", target_table="metadata", sql=(
        "INSERT INTO {cdm}.metadata (metadata_id, metadata_concept_id, metadata_type_concept_id, name, "
        "value_as_string, metadata_date) SELECT 1, 0, 0, 'zone', "
        "to_char('2020-01-01 23:30:00+00'::timestamptz, 'YYYY-MM-DD HH24:MI'), "
        "('2020-01-01 23:30:00+00'::timestamptz)::date"))]
    assert run_etl(conn, schemas, spec, cfg).ok
    rows = fetch_all(conn, f"SELECT value_as_string, metadata_date FROM {schemas.cdm}.metadata")
    assert [(r["value_as_string"], r["metadata_date"].isoformat()) for r in rows] == [("2020-01-02 05:00", "2020-01-02")]
