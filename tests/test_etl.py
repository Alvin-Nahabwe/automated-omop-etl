"""ETL compiler and runner: SQL shape (no database) and loads of the demo data (PostgreSQL)."""

from __future__ import annotations

import json
from datetime import date
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
    _compile,
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
        "11_route_prepare",
        "12_route_condition_occurrence",
        "13_route_drug_exposure",
        "14_route_measurement",
        "15_observation_period",
        "16_observation_period_merge",
        "17_observation_period_fill",
        "18_condition_era",
        "19_drug_era",
        "20_cdm_source",
    ]
    assert [s.kind for s in steps] == ["id_map", "id_map", "insert", "prune", "insert", "prune"] + ["insert"] * 4 + [
        "route"] * 4 + ["insert", "merge", "derived", "derived", "derived", "metadata"]
    # pruning, the merge and the fill run after a failed load too; the route steps need route_prepare,
    # the eras their event table
    assert [(s.label, s.requires) for s in steps if s.requires] == [
        ("route_condition_occurrence", "route_prepare"), ("route_drug_exposure", "route_prepare"),
        ("route_measurement", "route_prepare"), ("condition_era", "condition_occurrence"),
        ("drug_era", "drug_exposure")]
    # sort keys: (load index of the table, band, sequence); id maps first, cdm_source last
    assert [s.slot for s in steps] == sorted(s.slot for s in steps)
    assert steps[0].slot[0] == steps[1].slot[0] == -1 and steps[-1].slot[0] > len(LOAD_ORDER) - 1
    assert step(steps, "person_map_prune").slot == (LOAD_ORDER.index("person"), 1, 0)
    assert [s.label for s in steps][:3] == ["person_map", "visit_map", "person"]
    # every step but the route and era steps is blocking and reads a row count, except the tables that keep
    # one row per key, whose statement returns its statistics, and the route steps, which return
    # the records they moved; its SQL is its statements (two for the id maps, the visit pruning and
    # route_prepare); the spec kinds. The inserts count the required values they complete or drop
    # (per field, and the dropped rows), the generated steps count nothing
    assert [s.label for s in steps if not s.blocking] == [
        "route_prepare", "route_condition_occurrence", "route_drug_exposure", "route_measurement", "condition_era",
        "drug_era"]
    assert [s.label for s in steps if s.returns != "rowcount"] == [
        "person", "visit_occurrence", "death", "route_condition_occurrence", "route_drug_exposure",
        "route_measurement", "observation_period_merge", "observation_period_fill"]
    assert {s.returns for s in steps if s.label in ("person", "visit_occurrence", "death")} == {"row"}
    assert all(s.count_sites == () for s in steps if s.kind != "insert")
    # and the tolerant conversions count the values they cannot read, per field
    assert step(steps, "visit_occurrence").count_sites == (
        "visit_concept_id", "visit_start_date", "visit_start_datetime", "visit_end_date", "visit_type_concept_id", "row")
    assert step(steps, "death").count_sites == ("death_date", "row")
    assert all("\n".join(s.statements) == s.sql for s in steps)
    assert [s.label for s in steps if len(s.statements) > 1] == [
        "person_map", "visit_map", "visit_map_prune", "route_prepare", "observation_period_merge"]
    assert {s.kind for s in steps if s.kind in SPEC_KINDS} == {"id_map", "insert"}
    assert target_tables(spec) == [
        "person", "visit_occurrence", "condition_occurrence", "drug_exposure",
        "measurement", "death", "observation_period", "condition_era", "drug_era", "cdm_source",
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
    # 1) each loaded visit's person is recorded (DELETE + INSERT: the ETL role has no UPDATE); a
    #    visit loaded for two people records the same one on every run (the smaller key, byte-wise)
    assert ("WITH loaded AS (\n    SELECT v.visit_occurrence_id, MIN(pm.source_value COLLATE \"C\") AS person_source_value\n"
            "    FROM cdm.visit_occurrence v\n    JOIN etl.person_map pm ON pm.person_id = v.person_id\n"
            "    GROUP BY v.visit_occurrence_id\n)") in refresh
    assert "AND vm.person_source_value IS DISTINCT FROM l.person_source_value" in refresh
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
    # expression values convert to the CDM type tolerantly (an integer converts as itself), and so
    # do the casts and to_date calls inside them; a constant keeps its cast
    assert ("etl.omop_as_integer_at('gender_concept_id', 'reject', (CASE i.sex WHEN '1' THEN 8507 WHEN '2' THEN 8532"
            " ELSE 0 END)) AS gender_concept_id") in sql
    assert ("etl.omop_as_timestamp_at('birth_datetime', 'birth', etl.omop_as_date_at('birth_datetime', 'birth', i.dob,"
            " 'DD/MM/YYYY')) AS birth_datetime") in sql
    assert ("etl.omop_as_integer_at('year_of_birth', 'year', (EXTRACT(YEAR FROM etl.omop_as_date_at('year_of_birth',"
            " 'year', i.dob, 'DD/MM/YYYY'))::int)) AS year_of_birth") in sql
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
    assert "etl.omop_as_number_at('quantity', 'reject', p.qty) AS quantity" in de


def test_vocab_code_lookup_and_null_semantics():
    spec = load_ref()
    sql = compile_table(spec.mapping_for("condition_occurrence"), spec, S)
    # standard + source roles share one concept join, on the code computed once per source row
    assert sql.count("JOIN vocab.concept sc") == 1
    assert "            (d.icd10) AS _k1  -- lookup source\n" in sql
    assert "LEFT JOIN vocab.concept sc1 ON sc1.concept_code = (s._k1)::text AND sc1.vocabulary_id = 'ICD10'" in sql
    assert (
        "LEFT JOIN vocab.concept_relationship cr1 ON cr1.concept_id_1 = sc1.concept_id"
        " AND cr1.relationship_id = 'Maps to' AND cr1.invalid_reason IS NULL" in sql
    )
    assert (
        "LEFT JOIN vocab.concept tc1 ON tc1.concept_id = cr1.concept_id_2"
        " AND tc1.standard_concept = 'S' AND tc1.invalid_reason IS NULL" in sql
    )
    # required field: NULL source -> 0; optional field: NULL source -> NULL
    assert "(CASE WHEN (s._k1) IS NULL THEN 0 ELSE COALESCE(tc1.concept_id, 0) END)::integer AS condition_concept_id" in sql
    assert "(CASE WHEN (s._k1) IS NULL THEN NULL ELSE COALESCE(sc1.concept_id, 0) END)::integer AS condition_source_concept_id" in sql


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
        "LEFT JOIN etl.source_to_concept_map m1 ON m1.source_code = (s._k1)::text"
        " AND m1.source_vocabulary_id = 'SRC_LAB' AND m1.invalid_reason IS NULL" in sql
    )
    assert sql.count("source_to_concept_map m") == 3  # SRC_LAB (shared), SRC_RESULT, SRC_UNIT
    # one layer-1 column per distinct code expression, shared by the lookups on it
    assert [ln.strip() for ln in sql.splitlines() if ln.strip().startswith("v._k")] == ["v._k1,", "v._k2,", "v._k3"]
    assert "(r.test) AS _k1," in sql and "(r.unit) AS _k3  -- lookup source" in sql
    assert "COALESCE(m1.target_concept_id, 0) END)::integer AS measurement_concept_id" in sql
    assert "COALESCE(m1.source_concept_id, 0) END)::integer AS measurement_source_concept_id" in sql
    assert "THEN NULL ELSE COALESCE(m3.target_concept_id, 0) END)::integer AS unit_concept_id" in sql


def test_source_count_sql_excludes_lookups():
    steps = compile_spec(load_ref(), S)
    m = step(steps, "measurement")
    assert m.source_table == "lab_results"
    assert m.source_count_sql == "SELECT COUNT(*) AS n\nFROM src.lab_results r"
    # retained: + the INNER person-map join; the optional visit link and the lookups can't drop
    # rows. The same pass counts the visit links: rows with a visit key, linked to a visit of their
    # person, naming another person's visit, or a key of the visit mapping's source rows (before its
    # filter) without a visit (filtered out); the visit map and the distinct source keys are joined
    # on their keys, so they do not change n
    assert m.retained_count_sql == (
        "SELECT COUNT(*) AS n,\n"
        "       COUNT(*) FILTER (WHERE (r.visit_id) IS NOT NULL) AS with_key,\n"
        "       COUNT(*) FILTER (WHERE vm2.person_source_value = pm1.source_value) AS linked,\n"
        "       COUNT(*) FILTER (WHERE vm2.person_source_value <> pm1.source_value) AS other_person,\n"
        "       COUNT(*) FILTER (WHERE vm2.source_value IS NULL AND vs1.source_value IS NOT NULL) AS filtered\n"
        "FROM src.lab_results r\n"
        "INNER JOIN etl.person_map pm1 ON pm1.source_value = (r.individual_id)::text\n"
        "LEFT JOIN etl.visit_map vm2 ON vm2.source_value = (r.visit_id)::text\n"
        "LEFT JOIN (\n"
        "    SELECT DISTINCT k.source_value\n"
        "    FROM (\n"
        "        SELECT (v.visit_id)::text AS source_value\n"
        "        FROM src.clinic_visits v\n"
        "        WHERE (v.visit_id) IS NOT NULL\n"
        "    ) k\n"
        ") vs1 ON vs1.source_value = (r.visit_id)::text"
    )
    # the visit mapping's SQL runs in the count, so the runner re-checks it with the step's own
    assert m.origins == (load_ref().mapping_for("measurement"), load_ref().mapping_for("visit_occurrence"))
    vo = step(steps, "visit_occurrence")  # the required visit key is an INNER join too, after the person's
    assert vo.retained_count_sql == (
        "SELECT COUNT(*) AS n\nFROM src.clinic_visits v\n"
        "INNER JOIN etl.person_map pm1 ON pm1.source_value = (v.individual_id)::text\n"
        "INNER JOIN etl.visit_map vm1 ON vm1.source_value = (v.visit_id)::text"
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
    assert names[10:12] == ["11_death", "12_custom_note"]  # before the route block, which follows specimen
    assert names[12:] == ["13_route_prepare", "14_route_condition_occurrence", "15_route_drug_exposure",
                          "16_route_measurement", "17_observation_period", "18_observation_period_merge",
                          "19_observation_period_fill", "20_condition_era", "21_drug_era", "22_cdm_source"]
    de = step(steps, "drug_exposure")
    # the joins and the filter apply to the source rows (layer 1), before the id-map joins' rows
    # are counted and before the required-field policies
    assert (
        "        FROM src.prescriptions p\n"
        "        INNER JOIN src.clinic_visits cv ON cv.visit_id = p.visit_id\n"
        "        INNER JOIN etl.person_map pm1 ON pm1.source_value = (p.individual_id)::text\n"
        "        LEFT JOIN etl.visit_map vm1 ON vm1.source_value = (p.visit_id)::text"
        " AND vm1.person_source_value = pm1.source_value\n"
        "        WHERE (cv.visit_type <> 'EMERG')\n"
        "        OFFSET 0\n"
        "    ) v\n"
    ) in de.sql
    assert de.source_count_sql.endswith("INNER JOIN src.clinic_visits cv ON cv.visit_id = p.visit_id\nWHERE (cv.visit_type <> 'EMERG')")
    assert de.retained_count_sql.endswith(
        "INNER JOIN src.clinic_visits cv ON cv.visit_id = p.visit_id\n"
        "INNER JOIN etl.person_map pm1 ON pm1.source_value = (p.individual_id)::text\n"
        "LEFT JOIN etl.visit_map vm2 ON vm2.source_value = (p.visit_id)::text\n"
        "LEFT JOIN (\n"
        "    SELECT DISTINCT k.source_value\n"
        "    FROM (\n"
        "        SELECT (v.visit_id)::text AS source_value\n"
        "        FROM src.clinic_visits v\n"
        "        WHERE (v.visit_id) IS NOT NULL\n"
        "    ) k\n"
        ") vs1 ON vs1.source_value = (p.visit_id)::text\n"
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
        "custom_specimens", "custom_more_specimens", "route_prepare", "route_condition_occurrence",
        "route_drug_exposure", "route_measurement", "observation_period", "custom_dates",
        "observation_period_merge", "observation_period_fill", "condition_era", "custom_eras", "custom_other",
        "cdm_source",
    ]
    # a derived observation period follows every spec writer of observation_period, the merge follows it
    spec = spec.model_copy(update={"observation_period": "from_events"})
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "observation_period"]
    labels = [s.label for s in compile_spec(spec, S)]
    assert labels[labels.index("route_measurement") + 1:labels.index("custom_eras")] == [
        "custom_dates", "observation_period", "observation_period_merge", "condition_era"]
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
    assert "visit_detail_start_date" not in op.sql  # visit_detail is not loaded
    # the route destinations: their records were moved there from the loaded tables
    assert "procedure_date" in op.sql and "observation_date" in op.sql
    off = step(compile_spec(spec.model_copy(update={"route_by_domain": False}), S), "observation_period")
    assert "procedure_date" not in off.sql  # procedure_occurrence is not loaded
    assert "32817" in op.sql
    cs = steps[-1]
    assert cs.name.endswith("_cdm_source") and cs.kind == "metadata"
    assert "vocabulary_id = 'CDM' AND concept_code IN ('CDM v5.4.0', 'CDM v5.4')" in cs.sql
    assert "WHERE vocabulary_id = 'None'), 'unknown')" in cs.sql
    # the release date the merge derived (in build_facts), else the build date
    assert ("    'omop-agent',\n    COALESCE((SELECT MAX(value::date) FROM etl.build_facts"
            " WHERE name = 'source_release_date'), CURRENT_DATE),\n    CURRENT_DATE,") in cs.sql
    # a configured extract date is the source release date
    from datetime import date

    from omop_agent.config import CdmSourceConfig

    configured = compile_spec(spec, S, CdmSourceConfig(source_release_date=date(2023, 12, 31)))[-1]
    assert "    'omop-agent',\n    DATE '2023-12-31',\n    CURRENT_DATE," in configured.sql


def test_trailing_comment_in_expression_is_safe():
    spec = set_field(load_ref(), "person", "year_of_birth",
                     expression="EXTRACT(YEAR FROM to_date(i.dob, 'DD/MM/YYYY'))::int -- day-first dates")
    sql = compile_table(spec.mapping_for("person"), spec, S)
    assert "::int -- day-first dates\n)) AS year_of_birth" in sql


def test_compiled_insert_shape():
    spec = load_ref()
    vo = compile_spec(spec, S)[4]
    assert vo.label == "visit_occurrence"
    sql = vo.sql
    # four layers: the source rows, the required-field policies (each fenced), the candidate rows
    # (the outer SELECT of the other tables) and one row per visit and person
    one, two = sql.index("-- 1) the source rows"), sql.index("-- 2) required values that are NULL")
    three, four = sql.index("-- 3) the candidate rows"), sql.index("-- 4) one row per visit_occurrence_id")
    assert four < three < sql.index("\n    SELECT\n        s.visit_occurrence_id,") < two < one
    assert one < sql.index("INSERT INTO cdm.visit_occurrence (")
    assert sql.count("OFFSET 0") == 2 and "        OFFSET 0\n    ) s\n),\npicked AS (" in sql
    # layer 1 computes every value from the source once, with tolerant conversions (a start date
    # reads an unknown day as the first, an end date as the last); a constant keeps its cast
    assert ("                etl.omop_as_date_at('visit_start_date', 'first', v.visit_date) AS visit_start_date,"
            "  -- expression\n") in sql
    assert ("                etl.omop_as_date_at('visit_end_date', 'last', v.discharge_date) AS visit_end_date,"
            "  -- expression\n") in sql
    assert "                (32817)::integer AS visit_type_concept_id,  -- constant\n" in sql
    # layer 2: a NULL required concept becomes 0, a NULL end the start, and a NULL start drops the
    # row; each counted, and only on the NULL branch
    assert ("            CASE WHEN v.visit_type_concept_id IS NULL THEN etl.omop_default_int('visit_type_concept_id', 0)"
            " ELSE v.visit_type_concept_id END AS visit_type_concept_id,  -- NULL -> 0 (counted)\n") in sql
    assert ("            COALESCE(v.visit_end_date, etl.omop_derived_date('visit_end_date', v.visit_start_date))"
            " AS visit_end_date,  -- NULL -> visit_start_date (counted)\n") in sql
    assert ("        WHERE CASE WHEN v.visit_start_date IS NOT NULL THEN true\n"
            "                   ELSE etl.omop_drop(ARRAY['visit_start_date'], ARRAY[v.visit_start_date IS NULL]) END\n"
            "        OFFSET 0\n    ) s\n),") in sql
    # optional fields, id-map values and the lookups' codes pass through
    assert "            v.visit_start_datetime,\n" in sql and "            v.person_id,\n" in sql
    assert vo.count_sites == (
        "visit_concept_id", "visit_start_date", "visit_start_datetime", "visit_end_date", "visit_type_concept_id", "row")

    # an end date without a start date to take is dropped instead
    tm = set_field(spec, "drug_exposure", "drug_exposure_start_date", kind="null").mapping_for("drug_exposure")
    assert _compile(tm, spec).required == {"drug_exposure_end_date": "drop", "drug_type_concept_id": "zero"}
    # several fields that drop a row: each NULL one is counted, the row once
    note = TableMapping(target_table="note", source={"table": "clinic_visits", "alias": "n"}, fields=[
        FieldMapping(target_field="note_id", kind="auto_id"),
        FieldMapping(target_field="person_id", kind="person_id", expression="n.individual_id"),
        FieldMapping(target_field="note_date", kind="expression", expression="n.visit_date::date"),
        FieldMapping(target_field="note_type_concept_id", kind="constant", expression="32817"),
        FieldMapping(target_field="note_class_concept_id", kind="constant", expression="0"),
        FieldMapping(target_field="note_text", kind="column", expression="n.facility"),
        FieldMapping(target_field="encoding_concept_id", kind="constant", expression="0"),
        FieldMapping(target_field="language_concept_id", kind="constant", expression="0")])
    c = _compile(note, spec)
    assert [k for k, p in c.required.items() if p == "drop"] == ["note_date", "note_text"]
    assert c.count_sites[-1] == "row" and len(c.count_sites) == 7
    assert ("    WHERE CASE WHEN v.note_date IS NOT NULL AND v.note_text IS NOT NULL THEN true\n"
            "               ELSE {etl}.omop_drop(ARRAY['note_date', 'note_text'],"
            " ARRAY[v.note_date IS NULL, v.note_text IS NULL]) END\n") in c.insert
    # the observation period end ('open') is loaded as 9999-12-31 when NULL, uncounted: the merge
    # counts the open episodes and ends them; its counter site is the conversion's (invalid dates)
    op = _compile(spec.mapping_for("observation_period"), spec)
    assert op.required["observation_period_end_date"] == "open" and op.count_sites == (
        "observation_period_start_date", "observation_period_end_date", "period_type_concept_id", "row")
    assert ("        COALESCE(v.observation_period_end_date, DATE '9999-12-31') AS observation_period_end_date,"
            "  -- NULL = still open (ended by the observation_period_merge step)\n") in op.insert
    constant_end = set_field(spec, "observation_period", "observation_period_end_date", kind="constant",
                             expression="NULL::date")
    op = _compile(constant_end.mapping_for("observation_period"), constant_end)
    assert op.count_sites == ("observation_period_start_date", "period_type_concept_id", "row")
    assert "COALESCE(v.observation_period_end_date, DATE '9999-12-31') AS observation_period_end_date" in op.insert

    # no required field computed from the source: no layer 2
    loc = TableMapping(target_table="location", source={"table": "clinic_visits", "alias": "v"}, fields=[
        FieldMapping(target_field="location_id", kind="auto_id"),
        FieldMapping(target_field="city", kind="column", expression="v.facility")])
    c = _compile(loc, spec)
    assert (c.required, c.count_sites) == ({}, ()) and "-- 2)" not in c.insert and c.insert.count("OFFSET 0") == 1
    assert "    s.city\nFROM (\n    -- 1) the source rows" in c.insert


def test_conversions_in_the_compiled_sql():
    from omop_agent.config import EtlConfig

    spec = load_ref()
    steps = compile_spec(spec, S)
    person = step(steps, "person")
    # key and lookup expressions stay as written (join keys); constants keep their cast
    assert "INNER JOIN etl.person_map pm1 ON pm1.source_value = (i.individual_id)::text" in person.sql
    assert "(0)::integer AS race_concept_id,  -- constant" in person.sql
    # THEMIS for the birth fields, the widest interval for start / end dates, the 15th elsewhere
    assert "etl.omop_as_date_at('month_of_birth', 'month', i.dob, 'DD/MM/YYYY')" in person.sql
    assert "etl.omop_as_date_at('day_of_birth', 'reject', i.dob, 'DD/MM/YYYY')" in person.sql
    op = step(steps, "observation_period")
    assert "etl.omop_as_date_at('observation_period_start_date', 'first', r.start_date)" in op.sql
    assert "etl.omop_as_date_at('observation_period_end_date', 'last', r.end_date)" in op.sql
    assert "etl.omop_as_date_at('measurement_date', 'mid', r.test_date)" in step(steps, "measurement").sql
    # etl.partial_dates: reject leaves only the birth and death rules
    strict = compile_spec(spec, S, etl=EtlConfig(partial_dates="reject"))
    assert "etl.omop_as_date_at('observation_period_start_date', 'reject', r.start_date)" in step(strict,
                                                                                                  "observation_period").sql
    assert "etl.omop_as_date_at('month_of_birth', 'month', i.dob, 'DD/MM/YYYY')" in step(strict, "person").sql
    # a strftime template never reads right: the mapping does not compile
    bad = set_field(spec, "person", "birth_datetime", expression="to_date(i.dob, '%d/%m/%Y')::timestamp")
    with pytest.raises(CompileError, match="person.birth_datetime: unsupported date template '%d/%m/%Y'"):
        compile_spec(bad, S)
    bad = spec.model_copy(deep=True)
    bad.mapping_for("death").where = "to_date(x.dod, '%Y') IS NOT NULL"
    with pytest.raises(CompileError, match="death where: unsupported date template"):
        compile_spec(bad, S)
    assert any(s.error and "strftime" in s.error for s in compile_spec(bad, S, strict=False))


def test_required_null_policy_follows_the_cdm():
    from omop_agent.etl.policies import REQUIRED_NULL_POLICY, required_null_policy

    assert REQUIRED_NULL_POLICY["person"] == {
        "person_id": "drop", "gender_concept_id": "zero", "year_of_birth": "drop", "race_concept_id": "zero",
        "ethnicity_concept_id": "zero"}
    assert REQUIRED_NULL_POLICY["visit_occurrence"]["visit_end_date"] == "start"
    assert REQUIRED_NULL_POLICY["visit_detail"]["visit_detail_end_date"] == "start"
    assert REQUIRED_NULL_POLICY["drug_exposure"]["drug_exposure_end_date"] == "start"
    assert REQUIRED_NULL_POLICY["observation_period"]["observation_period_end_date"] == "open"
    assert REQUIRED_NULL_POLICY["death"] == {"person_id": "drop", "death_date": "drop"}
    assert REQUIRED_NULL_POLICY["fact_relationship"] == {
        "domain_concept_id_1": "zero", "fact_id_1": "drop", "domain_concept_id_2": "zero", "fact_id_2": "drop",
        "relationship_concept_id": "zero"}
    assert required_null_policy("note", "note_text") == required_null_policy("cost", "cost_domain_id") == "drop"
    assert required_null_policy("metadata", "name") == "drop"
    assert required_null_policy("person", "birth_datetime") is None  # optional: stays NULL
    assert required_null_policy("not_a_table", "x") is None


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
    from omop_agent.quality.checks import etl_checks

    conn, schemas, cfg = staged
    ws = Workspace(cfg.workdir)
    (ws.sql_dir / "99_stale.sql").write_text("-- old\n")
    result = run_etl(conn, schemas, load_ref(), cfg, ws)

    # the design bug (visit_end_date from the discharge date, empty for outpatient visits) no longer
    # fails the visit load: those visits end on their start date, and the build counts them
    assert result.ok and result.errors() == [], [(e.step, e.error) for e in result.errors()]
    src = schemas.src
    visits = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.clinic_visits"))
    no_discharge = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.clinic_visits WHERE discharge_date IS NULL"))
    assert 0 < no_discharge < visits
    assert count(conn, schemas, "person") == 400 and count(conn, schemas, "visit_occurrence") == visits
    by_table = {s.target_table: s for s in result.steps if s.kind not in ("merge", "derived", "route")}
    vo = by_table["visit_occurrence"]
    assert vo.value_counts == {"defaulted": {"visit_end_date": no_discharge}}
    assert (vo.source_rows, vo.id_retained_rows, vo.retained_rows, vo.rows_inserted) == (visits,) * 4
    assert count(conn, schemas, "visit_occurrence", "visit_end_date = visit_start_date") >= no_discharge
    assert count(conn, schemas, "visit_occurrence", "visit_end_date < visit_start_date") == 0
    defaulted = [r for r in etl_checks(result) if r.check_name == "etlDefaultedValue"]
    assert [(r.key, r.severity, r.failed, r.num_violated_rows, r.num_denominator_rows) for r in defaulted] == [
        ("etlDefaultedValue.visit_occurrence.visit_end_date", "convention", True, no_discharge, visits)]
    assert "took the start date (visit_start_date)" in defaulted[0].description
    assert not any(r.check_name == "etlRequiredFieldNull" for r in etl_checks(result))
    for t in ("condition_occurrence", "drug_exposure", "measurement", "death", "observation_period"):
        assert by_table[t].error is None and by_table[t].rows_inserted > 0, t
        assert by_table[t].rows_inserted == count(conn, schemas, t)
        assert by_table[t].value_counts == {}, t
    # retention: lab results of people missing from the registry are dropped by the person map
    m = by_table["measurement"]
    assert m.source_rows - m.rows_inserted == 12
    assert m.source_rows - m.retained_rows == 12 and m.retained_rows == m.rows_inserted == m.id_retained_rows
    assert by_table["person_map"].rows_inserted == 400
    assert by_table["person_map_prune"].rows_inserted == 0 and by_table["person_map_prune"].error is None
    assert by_table["visit_map_prune"].rows_inserted == 0
    assert count(conn, schemas, "cdm_source") == 1
    # one closed residency episode per person: the merge only renumbers them, the fill finds nobody
    # without a period, and the release date is derived from the exit dates (the demo's END)
    merge = next(s for s in result.steps if s.kind == "merge")
    assert merge.error is None and merge.source_rows is None and merge.value_counts == {}
    assert {k: merge.stats[k] for k in ("inserted", "episodes", "dropped_inverted", "open_episodes",
                                        "persons_overlapping", "persons_without_period", "release_date",
                                        "configured", "fact_rows")} == {
        "inserted": by_table["observation_period"].rows_inserted, "episodes": by_table["observation_period"].rows_inserted,
        "dropped_inverted": 0, "open_episodes": 0, "persons_overlapping": 0, "persons_without_period": 0,
        "release_date": "2023-12-31", "configured": False, "fact_rows": 1}
    assert merge.rows_inserted == merge.stats["inserted"]
    fill = next(s for s in result.steps if s.step.endswith("_observation_period_fill"))
    assert fill.error is None and fill.rows_inserted == 0
    assert scalar(conn, f"SELECT source_release_date FROM {schemas.cdm}.cdm_source") == date(2023, 12, 31)
    # the demo's codes map to concepts of their tables' domains: the route steps move nothing
    routes = [s for s in result.steps if s.kind == "route"]
    assert [(s.step, s.error, s.moved) for s in routes] == [
        ("11_route_prepare", None, []), ("12_route_condition_occurrence", None, []),
        ("13_route_drug_exposure", None, []), ("14_route_measurement", None, [])]
    assert result.rows_moved() == 0 and count(conn, schemas, "observation") == 0
    audit = [e for e in ws.read_audit() if e["event"] == "etl_step" and e["step"] == vo.step]
    assert audit[0]["value_counts"] == vo.value_counts and audit[0]["retained_rows"] == visits
    assert [e["retained_rows"] for e in ws.read_audit() if e.get("step") == "09_measurement"] == [m.retained_rows]

    files = sorted(p.name for p in ws.sql_dir.glob("*.sql"))
    assert files[0] == "01_person_map.sql" and "07_condition_occurrence.sql" in files
    assert "99_stale.sql" not in files
    saved = json.loads(ws.etl_run_json.read_text())
    assert len(saved["steps"]) == len(result.steps)
    assert saved["steps"][4]["value_counts"] == {"defaulted": {"visit_end_date": no_discharge}}
    events = [e["event"] for e in ws.read_audit()]
    assert events.count("etl_step") == len(result.steps) and events[-1] == "etl_run"
    assert ws.read_audit()[-1]["ok"] is True


@pytest.mark.postgres
def test_failed_visit_load_retires_every_visit_key(staged):
    conn, schemas, cfg = staged
    ws = Workspace(cfg.workdir)
    # a runtime error in every row (not a value the build converts or defaults)
    broken = set_field(fixed_ref(), "visit_occurrence", "visit_concept_id",
                       expression="9202 / (length(v.visit_id) - length(v.visit_id))")
    result = run_etl(conn, schemas, broken, cfg, ws)
    errors = result.errors()
    assert [e.target_table for e in errors] == ["visit_occurrence"], [(e.step, e.error) for e in errors]
    assert "division by zero" in errors[0].error and errors[0].value_counts == {}
    assert errors[0].source_rows == int(scalar(conn, f"SELECT COUNT(*) FROM {schemas.src}.clinic_visits"))
    assert count(conn, schemas, "visit_occurrence") == 0 and count(conn, schemas, "person") == 400
    # pruning retires every visit key with its id (the ids stay stable for the next run), and the
    # events get no visit instead of one missing from visit_occurrence
    prune = next(s for s in result.steps if s.step.endswith("visit_map_prune"))
    assert prune.error is None and prune.rows_inserted == errors[0].source_rows
    assert scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.visit_map") == 0
    assert scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.visit_map_retired") == errors[0].source_rows
    assert count(conn, schemas, "condition_occurrence") > 0
    assert count(conn, schemas, "condition_occurrence", "visit_occurrence_id IS NOT NULL") == 0
    audit = [e for e in ws.read_audit() if e["event"] == "etl_step" and e["step"] == prune.step]
    assert "skipped" not in audit[0] and audit[0]["rows_inserted"] == prune.rows_inserted
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
    parts = [s for s in result.steps if s.target_table == "measurement" and s.kind == "insert"]
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
def test_required_null_policies(staged):
    from omop_agent.quality.checks import etl_checks, gate

    conn, schemas, cfg = staged
    src = schemas.src
    # sex codes other than 1/2 have no branch, so gender is NULL: it becomes 0. One person (with
    # diagnoses) has no birth year: they are dropped, and so are their events. Prescriptions of
    # three days or less have no end date: they end on their start date
    who = scalar(conn, f"""SELECT d.individual_id FROM {src}.diagnoses d JOIN {src}.individuals i
        ON i.individual_id = d.individual_id GROUP BY 1 ORDER BY COUNT(*) DESC, 1 LIMIT 1""")
    spec = set_field(fixed_ref(), "person", "gender_concept_id",
                     expression="CASE i.sex WHEN '1' THEN 8507 WHEN '2' THEN 8532 END")
    spec = set_field(spec, "person", "year_of_birth", expression=(
        f"CASE WHEN i.individual_id <> '{who}' THEN EXTRACT(YEAR FROM to_date(i.dob, 'DD/MM/YYYY'))::int END"))
    spec = set_field(spec, "drug_exposure", "drug_exposure_end_date",
                     expression="CASE WHEN p.days::int > 3 THEN p.date_prescribed::date + p.days::int - 1 END")
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    by = {step_label(r.step): r for r in result.steps}

    unknown_sex = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.individuals WHERE sex NOT IN ('1', '2')"))
    person = by["person"]
    assert 0 < unknown_sex and person.value_counts == {
        "defaulted": {"gender_concept_id": unknown_sex}, "required_null": {"year_of_birth": 1}, "row": {"dropped": 1}}
    assert (person.source_rows, person.id_retained_rows, person.retained_rows, person.rows_inserted) == (400, 400, 399, 399)
    assert count(conn, schemas, "person") == 399
    assert count(conn, schemas, "person", "gender_concept_id = 0") == unknown_sex
    assert count(conn, schemas, "person", f"person_source_value = '{who}'") == 0
    # the dropped person is retired from the person map, so their events are dropped too
    assert by["person_map_prune"].rows_inserted == 1 and who in id_map(conn, schemas, "person_map_retired")
    assert set(orphans(conn, schemas).values()) == {0}
    events = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.diagnoses WHERE individual_id = '{who}'"))
    co = by["condition_occurrence"]
    assert events > 0 and co.source_rows - co.retained_rows >= events and co.value_counts == {}

    short = int(scalar(conn, f"""SELECT COUNT(*) FROM {src}.prescriptions p
        JOIN {schemas.etl}.person_map pm ON pm.source_value = p.individual_id WHERE p.days::int <= 3"""))
    drug = by["drug_exposure"]
    assert 0 < short < drug.retained_rows
    assert drug.value_counts == {"defaulted": {"drug_exposure_end_date": short}}
    assert count(conn, schemas, "drug_exposure", "drug_exposure_end_date = drug_exposure_start_date") >= short
    assert count(conn, schemas, "drug_exposure", "drug_exposure_end_date IS NULL") == 0

    results = {r.key: r for r in etl_checks(result)}
    yob = results["etlRequiredFieldNull.person.year_of_birth"]
    assert (yob.num_violated_rows, yob.num_denominator_rows, yob.threshold_pct, yob.failed) == (1, 400, 0.0, True)
    assert yob.severity == "convention" and yob.repairable is None and yob.step == "person"
    gender = results["etlDefaultedValue.person.gender_concept_id"]
    assert (gender.num_violated_rows, gender.num_denominator_rows) == (unknown_sex, 400)
    assert "became 0 (no matching concept)" in gender.description
    assert results["etlDefaultedValue.drug_exposure.drug_exposure_end_date"].num_violated_rows == short
    retention = results["etlRecordRetention.person"]
    assert (retention.num_violated_rows, retention.failed) == (1, False)  # the drop is a retention loss
    assert gate(list(results.values()), cfg, result) == (True, [])  # convention: reported, not blocking


@pytest.mark.postgres
def test_counters_are_per_source_row(staged):
    conn, schemas, cfg = staged
    # 'HB' maps to two concepts, so each HB result is inserted twice. Results of some people have
    # no date and are dropped, and results of others have no type, which becomes 0: both are
    # counted once per source row, below the fan-out
    execute(conn, f"""INSERT INTO {schemas.etl}.source_to_concept_map (source_code, source_vocabulary_id, target_concept_id)
        VALUES ('HB', 'SRC_LAB', 2000004001), ('HB', 'SRC_LAB', 2000004002)""")
    conn.commit()
    no_date, no_type = "right(r.individual_id, 1) IN ('1', '2')", "right(r.individual_id, 1) IN ('3', '4')"
    spec = set_field(fixed_ref(), "measurement", "measurement_date",
                     expression=f"CASE WHEN NOT ({no_date}) THEN r.test_date::date END")
    spec = set_field(spec, "measurement", "measurement_type_concept_id", kind="expression",
                     expression=f"CASE WHEN NOT ({no_type}) THEN 32817 END")
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok
    m = next(s for s in result.steps if s.target_table == "measurement")

    def source_rows(where: str) -> int:
        return int(scalar(conn, f"""SELECT COUNT(*) FROM {schemas.src}.lab_results r
            JOIN {schemas.etl}.person_map pm ON pm.source_value = r.individual_id WHERE {where}"""))

    dropped, hb_dropped = source_rows(no_date), source_rows(f"{no_date} AND r.test = 'HB'")
    defaulted, hb_defaulted = source_rows(no_type), source_rows(f"{no_type} AND r.test = 'HB'")
    assert dropped > hb_dropped > 0 and defaulted > hb_defaulted > 0
    assert m.value_counts == {"required_null": {"measurement_date": dropped}, "row": {"dropped": dropped},
                              "defaulted": {"measurement_type_concept_id": defaulted}}
    assert count(conn, schemas, "measurement", "measurement_type_concept_id = 0") == defaulted + hb_defaulted
    assert m.retained_rows == m.id_retained_rows - dropped and m.source_rows - m.id_retained_rows == 12
    extra = count(conn, schemas, "measurement", "measurement_concept_id = 2000004002")
    assert extra > 0 and m.rows_inserted == m.retained_rows + extra == count(conn, schemas, "measurement")


@pytest.mark.postgres
def test_bad_value_does_not_abort_table(staged):
    from omop_agent.quality.checks import etl_checks

    conn, schemas, cfg = staged
    src = schemas.src
    # unreadable values: a visit date that is no date, a lab date with a day that does not exist, a
    # quantity with text in it, a days_supply that is not a number
    execute(conn, f"UPDATE {src}.clinic_visits SET visit_date = '2019-13-01' WHERE visit_id IN "
                  f"(SELECT visit_id FROM {src}.clinic_visits ORDER BY visit_id LIMIT 3)")
    execute(conn, f"UPDATE {src}.lab_results SET test_date = '2019-02-30' WHERE ctid IN "
                  f"(SELECT ctid FROM {src}.lab_results WHERE individual_id IN (SELECT individual_id FROM "
                  f"{src}.individuals) ORDER BY test_date LIMIT 4)")
    execute(conn, f"UPDATE {src}.prescriptions SET qty = '2 tabs', days = 'NaN' WHERE ctid IN "
                  f"(SELECT ctid FROM {src}.prescriptions ORDER BY date_prescribed LIMIT 5)")
    conn.commit()
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert result.ok and result.errors() == [], [(e.step, e.error) for e in result.errors()]
    by = {step_label(r.step): r for r in result.steps}
    vo, m, de = by["visit_occurrence"], by["measurement"], by["drug_exposure"]
    visits = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.clinic_visits"))
    # the visits without a readable start date are dropped and counted; the others load
    assert vo.value_counts["invalid"]["visit_start_date"] == 3 and vo.value_counts["row"] == {"dropped": 3}
    assert vo.value_counts["required_null"] == {"visit_start_date": 3}
    assert count(conn, schemas, "visit_occurrence") == vo.rows_inserted == visits - 3 == vo.retained_rows
    assert m.value_counts["invalid"] == {"measurement_date": 4} and m.value_counts["row"] == {"dropped": 4}
    # optional values are loaded as NULL; the end date computed from an unreadable days is the start date
    assert de.value_counts["invalid"] == {"quantity": 5, "days_supply": 5, "drug_exposure_end_date": 5}
    assert count(conn, schemas, "drug_exposure", "quantity IS NULL") == 5
    assert count(conn, schemas, "drug_exposure", "days_supply IS NULL") == 5
    checks = {r.key: r for r in etl_checks(result)}
    assert checks["etlInvalidValue.visit_occurrence.visit_start_date"].severity == "convention"
    assert checks["etlInvalidValue.drug_exposure.quantity"].severity == "characterization"
    assert checks["etlInvalidValue.measurement.measurement_date"].num_violated_rows == 4


@pytest.mark.postgres
def test_where_is_counted_in_the_insert_and_not_in_the_counts(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    execute(conn, f"UPDATE {src}.clinic_visits SET visit_date = 'unknown' WHERE visit_id IN "
                  f"(SELECT visit_id FROM {src}.clinic_visits ORDER BY visit_id LIMIT 2)")
    conn.commit()
    spec = fixed_ref().model_copy(deep=True)
    spec.mapping_for("visit_occurrence").where = "v.visit_date::date >= DATE '2015-01-01'"
    steps = compile_spec(spec, schemas, etl=cfg.etl)
    vo = next(s for s in steps if s.label == "visit_occurrence")
    # counted at site 'where' in the INSERT; uncounted (parallel safe) in the count queries
    assert f"WHERE ({schemas.etl}.omop_as_date_at('where', 'mid', v.visit_date) >= DATE '2015-01-01')" in vo.sql
    assert vo.source_count_sql.endswith(f"WHERE ({schemas.etl}.try_date(v.visit_date, 'ISO', 'mid') >= DATE '2015-01-01')")
    assert "omop_as" not in vo.retained_count_sql and "where" in vo.count_sites
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    r = next(s for s in result.steps if s.target_table == "visit_occurrence")
    # the WHERE read the two values as unknown and left the rows out; the counts use the same filter
    assert r.value_counts == {"invalid": {"where": 2}}
    visits = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.clinic_visits"))
    assert r.source_rows == r.retained_rows == r.rows_inserted == visits - 2
    # but those rows are lost records, not filtered ones: the retention checks count them
    from omop_agent.quality.checks import etl_checks, gate

    checks = {c.key: c for c in etl_checks(result, etl=cfg.etl)}
    kept = checks["etlRecordRetention.visit_occurrence"]
    assert (kept.num_violated_rows, kept.num_denominator_rows) == (2, visits)
    # a WHERE that can read none of its values empties the table: the gate fails
    execute(conn, f"UPDATE {src}.clinic_visits SET visit_date = to_char(visit_date::date, 'DD/MM/YYYY') "
                  "WHERE visit_date <> 'unknown'")
    conn.commit()
    result = run_etl(conn, schemas, spec, cfg)
    r = next(s for s in result.steps if s.target_table == "visit_occurrence")
    assert r.rows_inserted == 0 and r.source_rows == 0
    checks = list(etl_checks(result, cfg.quality.min_record_retention_pct, cfg.etl))
    assert any(c.key == "etlRecordLoss.visit_occurrence" and c.failed for c in checks)
    passed, blocking = gate(checks, cfg, result)
    assert not passed and any("etlRecordLoss.visit_occurrence" in b for b in blocking)
    # the vocabulary stage reads the same rows
    from omop_agent.vocab.evaluate import source_from_sql

    assert "WHERE ({etl}.try_date(v.visit_date, 'ISO', 'mid') >= DATE '2015-01-01')" in source_from_sql(
        spec.mapping_for("visit_occurrence"))


@pytest.mark.postgres
def test_invalid_values_are_counted_per_source_row(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    # 'HB' maps to two concepts, so each HB result is inserted twice; the unreadable dates of HB
    # results are still counted once each
    execute(conn, f"""INSERT INTO {schemas.etl}.source_to_concept_map (source_code, source_vocabulary_id, target_concept_id)
        VALUES ('HB', 'SRC_LAB', 2000004001), ('HB', 'SRC_LAB', 2000004002)""")
    execute(conn, f"UPDATE {src}.lab_results SET test_date = '31/02/2019' WHERE test = 'HB' AND right(individual_id, 1) = '1'")
    conn.commit()
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert result.ok
    bad = int(scalar(conn, f"""SELECT COUNT(*) FROM {src}.lab_results r JOIN {schemas.etl}.person_map pm
        ON pm.source_value = r.individual_id WHERE r.test_date = '31/02/2019'"""))
    m = next(s for s in result.steps if s.target_table == "measurement")
    assert bad > 0 and m.value_counts["invalid"] == {"measurement_date": bad}
    assert m.value_counts["row"] == {"dropped": bad} and m.retained_rows == m.id_retained_rows - bad


@pytest.mark.postgres
def test_partial_birth_and_death_dates(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    people = [r["individual_id"] for r in fetch_all(conn, f"""SELECT i.individual_id FROM {src}.individuals i
        WHERE EXISTS (SELECT 1 FROM {src}.deaths d WHERE d.individual_id = i.individual_id)
        ORDER BY 1 LIMIT 3""")]
    assert len(people) == 3
    a, b, c = people
    execute(conn, f"UPDATE {src}.individuals SET dob = '00/06/1975' WHERE individual_id = %s", (a,))  # month, no day
    execute(conn, f"UPDATE {src}.individuals SET dob = '00/00/1980' WHERE individual_id = %s", (b,))  # a year only
    execute(conn, f"UPDATE {src}.deaths SET dod = '2019-00-00' WHERE individual_id = %s", (a,))
    execute(conn, f"UPDATE {src}.deaths SET dod = '2020-02-00' WHERE individual_id = %s", (b,))
    execute(conn, f"UPDATE {src}.deaths SET dod = '9999-00-00' WHERE individual_id = %s", (c,))  # nothing known
    conn.commit()
    spec = set_field(fixed_ref(), "person", "birth_datetime", expression="to_date(i.dob, 'DD/MM/YYYY')::timestamp")
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    rows = {r["person_source_value"]: r for r in fetch_all(conn, f"""SELECT person_source_value, year_of_birth,
        month_of_birth, day_of_birth, birth_datetime FROM {schemas.cdm}.person WHERE person_source_value = ANY(%s)""",
        ([a, b],))}
    # THEMIS: the known year and month are kept, a missing day is the 1st; a year alone gives no birth_datetime
    from datetime import datetime as dt

    assert (rows[a]["year_of_birth"], rows[a]["month_of_birth"], rows[a]["day_of_birth"]) == (1975, 6, None)
    assert rows[a]["birth_datetime"] == dt(1975, 6, 1)
    assert (rows[b]["year_of_birth"], rows[b]["month_of_birth"], rows[b]["day_of_birth"]) == (1980, None, None)
    assert rows[b]["birth_datetime"] is None
    person = next(s for s in result.steps if s.target_table == "person")
    assert person.value_counts["imputed"] == {"birth_datetime": 1}
    assert person.value_counts["partial_null"] == {"birth_datetime": 1, "day_of_birth": 2, "month_of_birth": 1}
    # a death date takes the last day of the known period; with nothing known the row is dropped
    deaths = {r["person_source_value"]: r["death_date"] for r in fetch_all(conn, f"""SELECT p.person_source_value,
        d.death_date FROM {schemas.cdm}.death d JOIN {schemas.cdm}.person p ON p.person_id = d.person_id""")}
    assert deaths[a] == date(2019, 12, 31) and deaths[b] == date(2020, 2, 29) and c not in deaths
    death = next(s for s in result.steps if s.target_table == "death")
    assert death.value_counts["imputed"] == {"death_date": 2} and death.value_counts["invalid"] == {"death_date": 1}


@pytest.mark.postgres
@pytest.mark.parametrize("zone", [None, "Africa/Nairobi"])
def test_offsets_and_the_site_time_zone(staged, zone):
    from datetime import datetime as dt

    from omop_agent.config import EtlConfig

    conn, schemas, cfg = staged
    src = schemas.src
    cfg = cfg.model_copy(update={"etl": EtlConfig(timezone=zone)})
    v1, v2, v3 = [r["visit_id"] for r in fetch_all(conn, f"SELECT visit_id FROM {src}.clinic_visits ORDER BY 1 LIMIT 3")]
    execute(conn, f"UPDATE {src}.clinic_visits SET visit_date = '2020-01-01T23:30:00Z' WHERE visit_id = %s", (v1,))
    execute(conn, f"UPDATE {src}.clinic_visits SET visit_date = '2020-01-01T23:30:00+03:00' WHERE visit_id = %s", (v2,))
    execute(conn, f"UPDATE {src}.clinic_visits SET visit_date = '2020-01-01 23:30' WHERE visit_id = %s", (v3,))
    conn.commit()
    spec = set_field(fixed_ref(), "visit_occurrence", "visit_end_datetime", kind="expression",
                     expression="(v.visit_date::timestamptz AT TIME ZONE 'Africa/Nairobi')")
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    visits = {r["visit_source_value"]: r for r in fetch_all(conn, f"""SELECT m.source_value AS visit_source_value,
        v.visit_start_date, v.visit_start_datetime, v.visit_end_datetime FROM {schemas.cdm}.visit_occurrence v
        JOIN {schemas.etl}.visit_map m ON m.visit_occurrence_id = v.visit_occurrence_id WHERE m.source_value = ANY(%s)""",
        ([v1, v2, v3],))}
    vo = next(s for s in result.steps if s.target_table == "visit_occurrence")
    if zone is None:
        # unset: an offset is ignored, the wall clock kept as written, and counted unless it is the
        # session zone's own (the server's zone: honouring it would have changed nothing)
        assert [visits[v]["visit_start_datetime"] for v in (v1, v2, v3)] == [dt(2020, 1, 1, 23, 30)] * 3
        assert [visits[v]["visit_start_date"] for v in (v1, v2, v3)] == [date(2020, 1, 1)] * 3
        n = scalar(conn, "SELECT count(*) FROM unnest(ARRAY['2020-01-01T23:30:00Z', '2020-01-01T23:30:00+03:00']) x "
                         "WHERE x::timestamptz <> x::timestamp::timestamptz")
        assert n >= 1
        assert {k: n for k, n in vo.value_counts.get("offset_ignored", {}).items() if k.startswith("visit_start")} == {
            "visit_start_date": n, "visit_start_datetime": n}
    else:
        # set: converted to the site's zone ('Z' 23:30 is 02:30 the next day in Nairobi) and counted;
        # the zone's own offset (+03:00) changes nothing and is not counted
        assert [visits[v]["visit_start_datetime"] for v in (v1, v2, v3)] == [
            dt(2020, 1, 2, 2, 30), dt(2020, 1, 1, 23, 30), dt(2020, 1, 1, 23, 30)]
        assert [visits[v]["visit_start_date"] for v in (v1, v2, v3)] == [date(2020, 1, 2), date(2020, 1, 1),
                                                                         date(2020, 1, 1)]
        assert {k: n for k, n in vo.value_counts["offset_converted"].items() if k.startswith("visit_start")} == {
            "visit_start_date": 1, "visit_start_datetime": 1}
    # the AT TIME ZONE idiom reads the instant, whatever the setting: as the hard cast did
    assert visits[v1]["visit_end_datetime"] == dt(2020, 1, 2, 2, 30)
    assert visits[v2]["visit_end_datetime"] == dt(2020, 1, 1, 23, 30)


@pytest.mark.postgres
def test_etl_functions_are_installed_once_per_version(staged):
    from datetime import datetime

    import psycopg

    from omop_agent.cdm.ddl import FUNCTIONS_VERSION, install_etl_functions, installed_functions_version

    conn, schemas, _ = staged
    etl = schemas.etl
    assert installed_functions_version(conn, schemas) == FUNCTIONS_VERSION  # by create_etl_tables
    assert install_etl_functions(conn, schemas) is False  # the same version: nothing to do
    # the counters: transaction-local settings, set only for the rows they count
    with conn.transaction(force_rollback=True):
        assert scalar(conn, f"SELECT {etl}.omop_default_int('f', 0)") == 0
        assert scalar(conn, f"SELECT {etl}.omop_derived_date('f', NULL)") is None
        assert scalar(conn, f"SELECT {etl}.omop_derived_ts('g', TIMESTAMP '2020-01-02 03:04')") == datetime(2020, 1, 2, 3, 4)
        assert scalar(conn, f"SELECT {etl}.omop_drop(ARRAY['a', 'b'], ARRAY[true, false])") is False
        assert scalar(conn, f"SELECT {etl}.omop_drop(ARRAY['a'], ARRAY[true])") is False
        names = ["defaulted__f", "defaulted__g", "required_null__a", "required_null__b", "dropped__row"]
        rows = fetch_all(conn, "SELECT n, current_setting('omop_agent.' || n, true) AS v FROM unnest(%s::text[]) n",
                         (names,))
        assert {r["n"]: r["v"] for r in rows} == {"defaulted__f": "1", "defaulted__g": "1", "required_null__a": "2",
                                                  "required_null__b": None, "dropped__row": "2"}
    conn.commit()
    assert scalar(conn, "SELECT current_setting('omop_agent.dropped__row', true)") in (None, "")

    # an older version is replaced, functions it lacked included, and functions whose parameter
    # names or return type changed (which CREATE OR REPLACE refuses)
    execute(conn, f"CREATE OR REPLACE FUNCTION {etl}.omop_agent_functions_version() RETURNS text "
                  "LANGUAGE sql AS $$ SELECT 'old' $$")
    execute(conn, f"DROP FUNCTION {etl}.omop_drop(text[], boolean[])")
    execute(conn, f"DROP FUNCTION {etl}.omop_count(text, text)")
    execute(conn, f"CREATE FUNCTION {etl}.omop_count(field text, what text) RETURNS void LANGUAGE sql AS $$ SELECT $$")
    assert install_etl_functions(conn, schemas) is True
    assert installed_functions_version(conn, schemas) == FUNCTIONS_VERSION
    assert scalar(conn, "SELECT to_regprocedure(%s) IS NOT NULL", (f"{etl}.omop_drop(text[], boolean[])",))
    assert scalar(conn, f"SELECT {etl}.omop_count('f', 'defaulted')") is True
    conn.commit()

    # a role that does not own them cannot replace them: the error says what to do
    role = f"{schemas.cdm}_other"
    execute(conn, f"CREATE ROLE {role} NOLOGIN")
    execute(conn, f"GRANT USAGE, CREATE ON SCHEMA {etl} TO {role}")
    conn.commit()
    try:
        with conn.transaction(force_rollback=True):
            execute(conn, f"CREATE OR REPLACE FUNCTION {etl}.omop_agent_functions_version() RETURNS text "
                          "LANGUAGE sql AS $$ SELECT 'old' $$")
            execute(conn, f"SET LOCAL ROLE {role}")
            with pytest.raises(psycopg.errors.InsufficientPrivilege, match=(
                    f"the ETL helper functions in schema {etl} belong to another role and must be updated: "
                    "run `omop-agent build` once as their owner")):
                install_etl_functions(conn, schemas)
        # installing grants EXECUTE to the build's role
        with conn.transaction(force_rollback=True):
            execute(conn, f"CREATE OR REPLACE FUNCTION {etl}.omop_agent_functions_version() RETURNS text "
                          "LANGUAGE sql AS $$ SELECT 'old' $$")
            assert install_etl_functions(conn, schemas, role) is True
            assert scalar(conn, """SELECT EXISTS (SELECT 1 FROM pg_proc p, aclexplode(p.proacl) a
                WHERE p.oid = to_regprocedure(%s) AND a.grantee = to_regrole(%s) AND a.privilege_type = 'EXECUTE')""",
                          (f"{etl}.omop_count(text, text)", role))
    finally:
        conn.rollback()
        execute(conn, f"DROP OWNED BY {role}")
        execute(conn, f"DROP ROLE {role}")
        conn.commit()


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
        # the reference spec: visits without a discharge date end on their start date, and the
        # counters (settings set by the ETL helper functions) work under the role; results without
        # a date are dropped and counted the same way
        spec = set_field(load_ref(), "measurement", "measurement_date",
                         expression="CASE WHEN r.test <> 'HB' THEN r.test_date::date END")
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
        by_table = {s.target_table: s for s in result.steps if s.kind != "route"}
        no_discharge = int(scalar(conn, f"""SELECT COUNT(*) FROM {schemas.src}.clinic_visits
            WHERE visit_type <> 'EMERG' AND discharge_date IS NULL"""))
        hb = int(scalar(conn, f"""SELECT COUNT(*) FROM {schemas.src}.lab_results r
            JOIN {schemas.etl}.person_map pm ON pm.source_value = r.individual_id WHERE r.test = 'HB'"""))
        assert no_discharge > 0 and hb > 0
        assert by_table["visit_occurrence"].value_counts == {"defaulted": {"visit_end_date": no_discharge}}
        assert by_table["measurement"].value_counts == {
            "required_null": {"measurement_date": hb}, "row": {"dropped": hb}}
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

    # without a configured zone the session keeps the server's, and offsets are not converted
    assert session_sql() == [
        "SET LOCAL datestyle = 'ISO, YMD';",
        "SET LOCAL standard_conforming_strings = on;",
        "SELECT set_config('omop_agent.imputation_cap', CURRENT_DATE::text, true),"
        " set_config('omop_agent.timezone', '', true);",
    ]
    cfg = make_test_config().model_copy(update={
        "etl": EtlConfig(timezone="Africa/Nairobi"),
        "cdm_source": CdmSourceConfig(source_release_date=date(2023, 12, 31)),
    })
    assert session_sql(cfg)[2:] == [
        "SET LOCAL timezone = 'Africa/Nairobi';",
        "SELECT set_config('omop_agent.imputation_cap', '2023-12-31', true),"
        " set_config('omop_agent.timezone', 'Africa/Nairobi', true);",
    ]


def make_test_config():
    from omop_agent.config import ProjectConfig

    return ProjectConfig.model_validate({"name": "t", "source": {"type": "csv", "path": "."},
                                         "target": {"url": "postgresql://unused"}})


@pytest.mark.postgres
def test_step_returns_counters_and_session(staged):
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

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
    # (the current date in that zone: the container's clock may be on the day before)
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    assert res.stats["tz"] == "Asia/Kolkata" and res.stats["cap"] == today

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
        retained_count_sql="SELECT 5 AS n, 3 AS with_key",
    )
    res = _run_step(conn, counted, cfg)
    assert res.error is None
    assert res.value_counts == {"invalid": {"year_of_birth": 2}, "defaulted": {"gender_concept_id": 5},
                                "row": {"dropped": 1}}
    assert (res.id_retained_rows, res.retained_rows) == (5, 4)
    assert res.stats == {"visit_links": {"with_key": 3}}
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
    # a backslash is an ordinary character in '...' strings, as sqlguard reads them, whatever the
    # server's or role's setting
    execute(conn, "SET standard_conforming_strings = off")
    try:
        assert query_rows(conn, r"SELECT length('\') AS n") == [{"n": 1}]
    finally:
        execute(conn, "RESET standard_conforming_strings")
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
