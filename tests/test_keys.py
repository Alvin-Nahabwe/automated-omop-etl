"""Keys: id maps fed by several person and visit sources, one row per key (person, death,
visit_occurrence), links only to a visit of the record's own person, and the checks, validator
rules and repair guards around them."""

from __future__ import annotations

import re

import pytest

from omop_agent.db import execute, fetch_all, scalar
from omop_agent.design.validate import key_shape, sql_safety_issues, validate_spec
from omop_agent.etl.compiler import compile_spec, compile_table, key_sources
from omop_agent.etl.runner import _refuse_unsafe, run_etl
from omop_agent.models import CustomSQL, ETLRunResult, FieldMapping, TableLoadResult, TableMapping
from omop_agent.quality.checks import etl_checks, gate, run_builtin_checks
from omop_agent.quality.repair import _mapping_diff, _same_mapping_logic, data_loss_issues, identity_issues
from test_etl import S, count, fixed_ref, id_map, set_field
from test_quality import repair_env as repair_env  # a fixture

EMR_KEY = "COALESCE({a}.hdss_id, 'EMR:' || {a}.patient_no)"


def emr_person(key: str = "'EMR:' || e.patient_no") -> TableMapping:
    """Clinic patients who are not in the HDSS register."""
    return TableMapping(
        target_table="person", source={"table": "emr_patients", "alias": "e"}, where="e.hdss_id IS NULL",
        fields=[
            FieldMapping(target_field="person_id", kind="person_id", expression=key),
            FieldMapping(target_field="gender_concept_id", kind="constant", expression="0"),
            FieldMapping(target_field="year_of_birth", kind="expression", expression="EXTRACT(YEAR FROM e.dob::date)::int"),
            FieldMapping(target_field="race_concept_id", kind="constant", expression="0"),
            FieldMapping(target_field="ethnicity_concept_id", kind="constant", expression="0"),
            FieldMapping(target_field="person_source_value", kind="expression", expression=key),
        ],
    )


def ipd_visit(table: str = "ipd_admissions", key: str = "'IPD:' || a.admission_id",
              person: str = EMR_KEY.format(a="a")) -> TableMapping:
    """Admissions of the clinic system, numbered like the outpatient visits (hence the prefix)."""
    return TableMapping(
        target_table="visit_occurrence", source={"table": table, "alias": "a"},
        fields=[
            FieldMapping(target_field="visit_occurrence_id", kind="visit_id", expression=key),
            FieldMapping(target_field="person_id", kind="person_id", expression=person),
            FieldMapping(target_field="visit_concept_id", kind="constant", expression="9201"),
            FieldMapping(target_field="visit_start_date", kind="expression", expression="a.admit_date::date"),
            FieldMapping(target_field="visit_end_date", kind="expression", expression="a.discharge_date::date"),
            FieldMapping(target_field="visit_type_concept_id", kind="constant", expression="32817"),
        ],
    )


def emr_labs() -> TableMapping:
    return TableMapping(
        target_table="measurement", source={"table": "emr_labs", "alias": "l"},
        fields=[
            FieldMapping(target_field="measurement_id", kind="auto_id"),
            FieldMapping(target_field="person_id", kind="person_id", expression=EMR_KEY.format(a="l")),
            FieldMapping(target_field="measurement_concept_id", kind="constant", expression="0"),
            FieldMapping(target_field="measurement_date", kind="expression", expression="l.test_date::date"),
            FieldMapping(target_field="measurement_type_concept_id", kind="constant", expression="32817"),
            FieldMapping(target_field="visit_occurrence_id", kind="visit_id", expression="'IPD:' || l.admission_id"),
        ],
    )


def va_deaths(dedup_order: str | None = None) -> TableMapping:
    """Verbal-autopsy deaths: a second death source, two rows per person."""
    return TableMapping(
        target_table="death", source={"table": "va_deaths", "alias": "w"}, dedup_order=dedup_order,
        fields=[
            FieldMapping(target_field="person_id", kind="person_id", expression="w.individual_id"),
            FieldMapping(target_field="death_date", kind="expression", expression="w.va_date::date"),
            FieldMapping(target_field="death_type_concept_id", kind="constant", expression="32817"),
            FieldMapping(target_field="cause_source_value", kind="column", expression="w.verified"),
        ],
    )


def multi_spec(dedup_order: str | None = None):
    spec = fixed_ref().model_copy(deep=True)
    spec.table_mappings += [emr_person(), ipd_visit(), emr_labs(), va_deaths(dedup_order)]
    return spec


def by_label(steps) -> dict:
    return {s.label: s for s in steps}


# --------------------------------------------------------------------------------------
# compiler (no database)
# --------------------------------------------------------------------------------------


def test_id_maps_take_keys_from_every_person_and_visit_mapping():
    spec = multi_spec()
    steps = compile_spec(spec, S)
    assert [s.name for s in steps][:4] == ["01_person_map", "02_person_map_2", "03_visit_map", "04_visit_map_2"]
    assert [s.kind for s in steps[:4]] == ["id_map"] * 4 and [s.slot[0] for s in steps[:4]] == [-1] * 4
    by = by_label(steps)
    assert by["person_map"].sql == compile_spec(fixed_ref(), S)[0].sql  # the person_key step is unchanged
    # the extra sources: the keys of the rows the mapping reads (its FROM, joins and WHERE), restored
    # or numbered after every id given out, like the person_key's; a key already in the map keeps its id
    restore, new = by["person_map_2"].statements
    keys = ("    SELECT DISTINCT ('EMR:' || e.patient_no)::text AS source_value\n    FROM src.emr_patients e\n"
            "    WHERE (e.hdss_id IS NULL)\n      AND ('EMR:' || e.patient_no) IS NOT NULL\n")
    assert f"WITH k AS (\n{keys})," in restore and f"FROM (\n{keys}) k" in new
    assert "DELETE FROM etl.person_map_retired r USING k" in restore
    assert "(SELECT COALESCE(MAX(person_id), 0) FROM etl.person_map_retired))" in new
    assert "WHERE NOT EXISTS (SELECT 1 FROM etl.person_map pm WHERE pm.source_value = k.source_value)" in new
    assert by["person_map_2"].origins == (spec.table_mappings[-4],) and by["person_map_2"].source_table == "emr_patients"
    # a visit source guesses each visit's person from its person_id expression
    restore, new = by["visit_map_2"].statements
    assert ("    SELECT ('IPD:' || a.admission_id)::text AS source_value, "
            "MIN((COALESCE(a.hdss_id, 'EMR:' || a.patient_no))::text) AS person_source_value\n"
            "    FROM src.ipd_admissions a\n    WHERE ('IPD:' || a.admission_id) IS NOT NULL\n    GROUP BY 1\n") in new
    assert [(tm.source.table, expr) for tm, expr, _ in key_sources(spec, "visit_occurrence")] == [
        ("ipd_admissions", "'IPD:' || a.admission_id")]
    # a mapping that reads the key table with the key expression adds nothing (even filtered, or
    # written with other spacing or case outside its literals)
    spec = fixed_ref().model_copy(deep=True)
    spec.mapping_for("person").where = "i.sex <> '9'"
    spec = set_field(spec, "person", "person_id", expression="( I.Individual_ID )")
    assert [s.name for s in compile_spec(spec, S)][:3] == ["01_person_map", "02_visit_map", "03_person"]
    # ... but another form of the key does (KB11), and so does another literal
    spec = set_field(fixed_ref(), "person", "person_id", expression="trim(i.individual_id)")
    assert [s.label for s in compile_spec(spec, S)][:3] == ["person_map", "person_map_2", "visit_map"]
    assert key_sources(multi_spec(), "person")[0][1] == "'EMR:' || e.patient_no"
    two = multi_spec().model_copy(deep=True)
    two.table_mappings.append(emr_person("'emr:' || e.patient_no"))
    assert [s.label for s in compile_spec(two, S)][:3] == ["person_map", "person_map_2", "person_map_3"]
    # a mapping whose WHERE cannot compile gives that step an error (and not the others)
    bad = multi_spec()
    bad.table_mappings[-4].where = "to_date(e.dob, '%d/%m/%Y') IS NOT NULL"
    steps = by_label(compile_spec(bad, S, strict=False))
    assert steps["person_map_2"].error.startswith("compile error:") and steps["person_map"].error is None


def test_event_visits_must_belong_to_the_same_person():
    spec = multi_spec()
    co = compile_table(spec.mapping_for("condition_occurrence"), spec, S)
    # the person is joined first; the visit must be one of the row's person
    assert ("        INNER JOIN etl.person_map pm1 ON pm1.source_value = (d.individual_id)::text\n"
            "        LEFT JOIN etl.visit_map vm1 ON vm1.source_value = (d.visit_id)::text"
            " AND vm1.person_source_value = pm1.source_value\n") in co
    # the visit table's own key and its preceding visit are not checked (their person is recorded
    # after the visit load)
    spec = set_field(spec, "visit_occurrence", "preceding_visit_occurrence_id", kind="visit_id",
                     expression="v.previous_visit")
    vo = compile_table(spec.mapping_for("visit_occurrence"), spec, S)
    assert "INNER JOIN etl.visit_map vm1 ON vm1.source_value = (v.visit_id)::text\n" in vo
    assert "LEFT JOIN etl.visit_map vm2 ON vm2.source_value = (v.previous_visit)::text\n" in vo
    # the person join comes first even when the visit field is listed before it
    tm = spec.mapping_for("drug_exposure").model_copy(deep=True)
    tm.fields.sort(key=lambda f: f.target_field != "visit_occurrence_id")
    de = compile_table(tm, spec, S)
    assert de.index("JOIN etl.person_map pm1") < de.index("JOIN etl.visit_map vm1 ON vm1.source_value = (p.visit_id)"
                                                          "::text AND vm1.person_source_value = pm1.source_value")
    # the link classes are counted with the retained rows of the records, not of the visits
    steps = by_label(compile_spec(spec, S))
    assert "AS other_person" in steps["condition_occurrence"].retained_count_sql
    assert "AS other_person" in steps["measurement_2"].retained_count_sql
    assert "FILTER" not in steps["visit_occurrence"].retained_count_sql
    assert "FILTER" not in steps["death"].retained_count_sql


def test_one_row_per_key_tables_are_deduplicated():
    spec = multi_spec("w.verified = '1' DESC")
    steps = by_label(compile_spec(spec, S))
    for name in ("person", "person_2", "death", "death_2"):
        sql = steps[name].sql
        assert steps[name].returns == "row" and "    SELECT DISTINCT ON (c.person_id)\n" in sql, name
        # rows whose required fields are NULL are dropped in layer 2, inside the candidates
        assert sql.index("etl.omop_drop(") < sql.index("picked AS (") < sql.index("ins AS (") < sql.index("per_key AS (")
        # the anti-join on the loaded keys runs on the chosen rows, outside DISTINCT ON
        assert f"    FROM picked p\n    WHERE NOT EXISTS (SELECT 1 FROM cdm.{steps[name].target_table} x " \
               "WHERE x.person_id = p.person_id)\n    RETURNING 1\n" in sql, name
        # a hash of the source row's text breaks ties byte-wise, whatever the server's collation; it
        # is computed at the scan of the source (a fenced subquery), so a wide source row is not
        # carried through the joins and sorts
        assert "AS _src_text  -- the last tie-break (a hash of the source row)" in sql
        assert 'c._src_text COLLATE "C",' in sql and "._omop_src_hash" in sql
        assert re.search(r"FROM \(SELECT (\w+)\.\*, md5\(\(\1\.\*\)::text\) AS _omop_src_hash FROM src\.\w+ \1 "
                         r"OFFSET 0\) \1\n", sql), name
        assert "::text AS _src_text" not in sql
    for name in ("visit_occurrence", "visit_occurrence_2"):
        sql = steps[name].sql
        assert "    SELECT DISTINCT ON (c.visit_occurrence_id, c.person_id)\n" in sql
        # a visit spans all its rows
        assert "        MIN(c.visit_start_date) OVER w AS visit_start_date,  -- the visit spans all its rows\n" in sql
        assert "        MAX(c.visit_end_date) OVER w AS visit_end_date,  -- the visit spans all its rows\n" in sql
        assert "    WINDOW w AS (PARTITION BY c.visit_occurrence_id, c.person_id)\n" in sql
        assert "WHERE x.visit_occurrence_id = p.visit_occurrence_id AND x.person_id = p.person_id)" in sql
        # a visit an earlier mapping loaded for another person is a conflicting key
        assert "JOIN cdm.visit_occurrence x ON x.visit_occurrence_id = c.visit_occurrence_id AND x.person_id <> " \
               "c.person_id" in sql
        # rows of one visit key that are separate stays: a row starting more than a day after the
        # earlier rows ended (only ends read from the source: one the 'start' policy completed does
        # not end a row)
        assert "v.visit_end_date IS NOT NULL AS _end_known  -- visit_end_date as read" in sql
        assert ("        SELECT c.*, c.visit_start_date > MAX(CASE WHEN c._end_known THEN c.visit_end_date END) OVER (\n"
                "            PARTITION BY c.visit_occurrence_id, c.person_id ORDER BY c.visit_start_date, "
                "c.visit_end_date\n            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) + 1 AS _gap\n") in sql
        assert "COUNT(*) FILTER (WHERE k.gap) AS gap_keys" in sql and "undecided_keys" not in sql
    person = steps["person"].sql
    # completeness counts a concept 0 as missing, so a known sex wins over an unknown one; then the
    # source row's text, then the values: deterministic, and no sex, concept or value is preferred
    assert "         num_nulls(NULLIF(c.gender_concept_id, 0), c.year_of_birth, c.month_of_birth, " in person
    assert '         c._src_text COLLATE "C",\n         c.gender_concept_id,' in person
    assert "COUNT(DISTINCT NULLIF(c.gender_concept_id, 0)) > 1 OR COUNT(DISTINCT c.year_of_birth) > 1" in person
    # conflicts: among the rows, or with the person an earlier mapping loaded; undecided: the keys
    # this mapping loads whose rows (without a dedup_order: all of them) disagree
    assert ("UNION SELECT c.person_id FROM cand c JOIN cdm.person x ON x.person_id = c.person_id WHERE "
            "NULLIF(x.gender_concept_id, 0) <> NULLIF(c.gender_concept_id, 0) OR x.year_of_birth <> c.year_of_birth"
            ) in person
    assert ("(SELECT COUNT(*) FROM (SELECT c.person_id FROM cand c WHERE NOT EXISTS (SELECT 1 FROM cdm.person x "
            "WHERE x.person_id = c.person_id) GROUP BY c.person_id HAVING ") in person
    assert "    FALSE AS dedup_order\nFROM per_key k;" in person
    for column in ("inserted", "candidate_rows", "source_rows_kept", "keys", "duplicate_rows", "already_loaded_keys",
                   "conflict_keys", "undecided_keys", "fan_out_keys"):
        assert f" AS {column}" in person
    # dedup_order ranks the rows of a key over the source aliases (layer 1); then the latest death
    # (NULLS LAST: a row whose ranking value is missing never wins); then a death date known as
    # written before one completed from a year or month, then the latest death
    death = steps["death_2"].sql
    assert ("                RANK() OVER (PARTITION BY (pm1.person_id)::integer ORDER BY w.verified = '1' DESC"
            " NULLS LAST) AS _pref,  -- dedup_order\n") in death
    assert ("                (etl.try_date(w.va_date, 'ISO', 'reject')) IS NOT NULL AS _exact,  -- death_date known as "
            "written (not a completed partial date)\n") in death
    assert ("    ORDER BY c.person_id,\n         c._pref,\n         c._exact DESC,\n"
            "         c.death_date DESC NULLS LAST,\n") in death
    assert "    TRUE AS dedup_order\n" in death
    assert ("c.person_id) AS _best FROM cand c) c WHERE c._pref = c._best AND NOT EXISTS (SELECT 1 FROM cdm.death x "
            "WHERE x.person_id = c.person_id)") in death
    assert "c._pref" not in steps["death"].sql and "ORDER BY c.person_id,\n         c._exact DESC,\n" \
        "         c.death_date DESC" in steps["death"].sql
    # its conversions are tolerant and uncounted, and a trailing comment cannot swallow the parenthesis
    tm = va_deaths("w.va_date::date DESC -- latest first")
    sql = compile_table(tm, spec, S)
    assert "ORDER BY etl.try_date(w.va_date, 'ISO', 'mid') DESC NULLS LAST -- latest first\n) AS _pref" in sql
    # every item gets NULLS LAST unless it says where NULLs go
    tm = va_deaths("w.verified = '1' DESC, coalesce(w.va_date, '') NULLS FIRST, w.va_date")
    assert ("ORDER BY w.verified = '1' DESC NULLS LAST, coalesce(w.va_date, '') NULLS FIRST, w.va_date NULLS LAST)"
            in compile_table(tm, spec, S))
    # a field left NULL is a typed NULL: the rows pass through the candidates
    spec = set_field(fixed_ref(), "person", "location_id", kind="null")
    spec = set_field(spec, "visit_occurrence", "visit_end_datetime", kind="null")
    assert "(NULL)::integer AS location_id,  -- null" in compile_table(spec.mapping_for("person"), spec, S)
    assert "(NULL)::timestamp AS visit_end_datetime" in compile_table(spec.mapping_for("visit_occurrence"), spec, S)
    # a visit from several rows: where the patient came from is as in its first row, where they
    # went as in its last
    tm = set_field_tm(ward_visit(), "admitted_from_source_value", "a.came_from", kind="column")
    tm = set_field_tm(tm, "discharged_to_source_value", "a.went_to", kind="column")
    sql = compile_table(tm, spec, S)
    assert "FIRST_VALUE(c.admitted_from_source_value) OVER w_first AS admitted_from_source_value" in sql
    assert "FIRST_VALUE(c.discharged_to_source_value) OVER w_last AS discharged_to_source_value" in sql
    assert ("    WINDOW w AS (PARTITION BY c.visit_occurrence_id, c.person_id),\n"
            "       w_first AS (PARTITION BY c.visit_occurrence_id, c.person_id ORDER BY c.visit_start_date, "
            'c._src_text COLLATE "C"),\n'
            "       w_last AS (PARTITION BY c.visit_occurrence_id, c.person_id ORDER BY c.visit_end_date DESC "
            'NULLS LAST, c._src_text COLLATE "C")\n') in sql
    # other tables insert every row
    co = steps["condition_occurrence"]
    assert co.returns == "rowcount" and "DISTINCT" not in co.sql and "_src_text" not in co.sql


def test_validator_key_warnings():
    def keyed(spec) -> list[str]:
        return [str(i) for i in validate_spec(spec) if "id space" in i.message or "another form" in i.message
                or "key table" in i.message or "dedup_order" in i.message]

    assert keyed(multi_spec()) == []  # prefixed keys, visit keys written as the visit mappings do
    spec = multi_spec()
    spec.table_mappings[-4] = emr_person("e.patient_no")  # no prefix: shares the HDSS id space
    spec = set_field(spec, "drug_exposure", "visit_occurrence_id", expression="lower(p.visit_id)")
    spec.mapping_for("condition_occurrence").dedup_order = "d.dx_date DESC"
    msgs = keyed(spec)
    assert any(m.startswith("WARNING person.person_id: the mapping from emr_patients adds its keys (e.patient_no) to "
                            "the same id space as individuals") for m in msgs), msgs
    assert any(m.startswith("WARNING drug_exposure.visit_occurrence_id: the visit key is written as lower(p.visit_id)")
               for m in msgs), msgs
    assert any("condition_occurrence: dedup_order is ignored" in m for m in msgs), msgs
    # the IPD keys have the shape of the IPD visit mapping, not of the visit_key: no warning
    assert not any("measurement" in m for m in msgs)
    # KB11: the person key table with another form of the key
    spec = set_field(fixed_ref(), "person", "person_id", expression="upper(i.individual_id)")
    assert any("WARNING person.person_id: the mapping from individuals reads the person key table but writes the key "
               "as upper(i.individual_id), not as the person_key (individual_id)" in m for m in keyed(spec))
    assert key_shape("d.visit_id") == key_shape("(visit_id)") != key_shape("lower(d.visit_id)")
    # a dedup_order that would close the window clause it is written in is refused
    for bad in ("x.dod; DROP TABLE x", "x.dod) AS _pref, (SELECT 1", "(x.dod"):
        spec = fixed_ref()
        spec.mapping_for("death").dedup_order = bad
        assert any(i.severity == "error" and i.target_table == "death" and "dedup_order" in i.message
                   for i in validate_spec(spec)), bad


def test_dedup_order_paren_refused_at_runtime():
    tm = va_deaths("w.verified) AS _pref, (SELECT current_user")
    assert any("dedup_order: unbalanced parentheses" in p for p in sql_safety_issues(tm))
    spec = fixed_ref().model_copy(deep=True)
    spec.table_mappings.append(tm)
    steps = compile_spec(spec, S, strict=False)
    _refuse_unsafe(steps)
    refused = [s.label for s in steps if s.error and "unbalanced parentheses" in s.error]
    assert refused == ["death_2"]
    # a balanced one inside literals or comments is fine
    assert sql_safety_issues(va_deaths("w.verified = ')' DESC")) == []


def test_dedup_order_must_not_draw():
    """A dedup_order whose value changes from call to call ranks every row of a key apart, so
    etlKeyConflict sees every key decided while the row loaded is a draw that changes between
    builds: the validator and the repair agent refuse it."""
    from omop_agent.design.validate import dedup_order_issue

    for order in ("random()", "w.verified DESC, RANDOM()", "clock_timestamp()", "md5(gen_random_uuid()::text)"):
        assert "changes from call to call" in dedup_order_issue(order), order
        spec = multi_spec(order)
        assert any(i.severity == "error" and i.target_table == "death" and "changes from call to call" in i.message
                   for i in validate_spec(spec)), order
    for order in ("w.verified = '1' DESC", "to_date(w.va_date, 'DD/MM/YYYY') DESC", "'random()'", None):
        assert dedup_order_issue(order) is None, order


def test_identity_edits_refused():
    ref = fixed_ref()
    person, visit, death = ref.mapping_for("person"), ref.mapping_for("visit_occurrence"), ref.mapping_for("death")
    cond = ref.mapping_for("condition_occurrence")

    def edit(tm: TableMapping, field: str, **changes) -> TableMapping:
        return set_field(ref, tm.target_table, field, **changes).mapping_for(tm.target_table)

    # a person_id expression, in any table, re-attributes records
    (msg,) = identity_issues(cond, edit(cond, "person_id", expression="trim(d.individual_id)"))
    assert "changes the person key of condition_occurrence.person_id" in msg
    assert identity_issues(person, edit(person, "person_id", expression="upper(i.individual_id)"))
    # the visit key of visit_occurrence decides which visits exist; a record's visit link may change
    assert identity_issues(visit, edit(visit, "visit_occurrence_id", expression="lower(v.visit_id)"))
    assert identity_issues(cond, edit(cond, "visit_occurrence_id", expression="upper(d.visit_id)")) == []
    # a new mapping into person, visit_occurrence or death adds entities; elsewhere it adds records
    assert identity_issues(None, va_deaths()) and identity_issues(None, emr_person())
    assert identity_issues(None, emr_labs()) == []
    # a removed or changed filter of those tables
    filtered = person.model_copy(update={"where": "i.sex <> '9'"})
    assert "removes the filter of person" in identity_issues(filtered, person)[0]
    assert identity_issues(cond.model_copy(update={"where": "d.icd10 IS NOT NULL"}), cond) == []
    # dedup_order only chooses among the rows of a key: no loss, no identity change, but a change
    ranked = death.model_copy(update={"dedup_order": "x.dod::date DESC"})
    assert identity_issues(death, ranked) == [] and data_loss_issues(death, ranked) == []
    assert not _same_mapping_logic(death, ranked)
    assert "dedup_order: (none) → x.dod::date DESC" in _mapping_diff(death, ranked)


def _step(name: str, table: str, **kw) -> TableLoadResult:
    return TableLoadResult(target_table=table, step=name, kind="insert", **kw)


def test_key_and_link_checks():
    def checks(*steps: TableLoadResult) -> dict[str, object]:
        run = ETLRunResult(started_at="2026-01-01T00:00:00Z", steps=list(steps))
        return {r.key: r for r in etl_checks(run, 50.0)}

    stats = {"inserted": 95, "candidate_rows": 106, "source_rows_kept": 100, "keys": 96, "duplicate_rows": 4,
             "already_loaded_keys": 1, "conflict_keys": 2, "fan_out_keys": 3, "dedup_order": False}
    r = checks(_step("03_person", "person", source_rows=100, retained_rows=100, rows_inserted=95, stats=stats))
    dup, conflict, fan = r["etlDuplicateRecords.person"], r["etlKeyConflict.person"], r["etlKeyFanOut.person"]
    # person: duplicates within the mapping (not the keys an earlier mapping loaded), convention at
    # 5%; only a person can choose another source table, so it is not for the repair agent
    assert (dup.severity, dup.num_violated_rows, dup.num_denominator_rows, dup.failed) == ("convention", 4, 100, False)
    assert dup.repairable is False
    # conflicts without undecided_keys (older results): every conflict counts
    assert (conflict.severity, conflict.num_violated_rows, conflict.num_denominator_rows, conflict.failed) == (
        "convention", 2, 96, True)
    assert conflict.repairable is None and "set dedup_order" in conflict.description
    assert "a person must fix the key" in conflict.description
    assert (fan.severity, fan.failed, fan.repairable) == ("convention", True, False)
    assert r["etlRecordRetention.person"].num_violated_rows == 0  # rows that lost to another row are not drops
    # the keys whose best-ranked rows the dedup_order tells apart pass; the others still fail (a
    # dedup_order that ranks every row alike decides nothing)
    r = checks(_step("03_person", "person", stats=stats | {"dedup_order": True, "fan_out_keys": 0, "undecided_keys": 0}))
    conflict = r["etlKeyConflict.person"]
    assert (conflict.severity, conflict.num_violated_rows, conflict.failed) == ("convention", 0, False)
    assert "2 of 96 keys" in conflict.description and "etlKeyFanOut.person" not in r
    r = checks(_step("03_person", "person", stats=stats | {"dedup_order": True, "undecided_keys": 1}))
    assert (r["etlKeyConflict.person"].num_violated_rows, r["etlKeyConflict.person"].failed) == (1, True)
    # death and visits: characterization, the keys an earlier mapping loaded included; per step
    r = checks(_step("10_death_2", "death", stats=stats))
    dup = r["etlDuplicateRecords.death.death_2"]
    assert (dup.severity, dup.num_violated_rows, dup.failed) == ("characterization", 5, False)
    assert (r["etlKeyConflict.death.death_2"].severity, r["etlKeyConflict.death.death_2"].failed) == (
        "characterization", False)

    # visits rolled up from rows that are separate stays: a visit key coarser than one visit, fatal
    # above 5% of the visits, for a person to fix
    visits = stats | {"gap_keys": 4}
    grain = checks(_step("05_visit_occurrence", "visit_occurrence", stats=visits))["etlVisitGrain.visit_occurrence"]
    assert (grain.severity, grain.num_violated_rows, grain.num_denominator_rows, grain.failed, grain.repairable) == (
        "fatal", 4, 96, False, False)
    grain = checks(_step("05_visit_occurrence", "visit_occurrence", stats=visits | {"gap_keys": 5}))
    assert grain["etlVisitGrain.visit_occurrence"].failed
    assert "etlVisitGrain.person" not in checks(_step("03_person", "person", stats=visits))

    # visit links: of 100 rows with a key, 10 filtered out by the visit mapping, 80 linked, 4 name
    # another person's visit and 6 a key never loaded: 10 of 90
    links = {"with_key": 100, "linked": 80, "other_person": 4, "filtered": 10}
    r = checks(_step("07_condition_occurrence", "condition_occurrence", source_rows=120, retained_rows=120,
                     rows_inserted=120, stats={"visit_links": links}))
    link = r["etlVisitLinkage.condition_occurrence.visit_occurrence_id"]
    assert (link.severity, link.num_violated_rows, link.num_denominator_rows, link.failed) == ("convention", 10, 90, True)
    assert "Some name a loaded visit of another person, and some a visit key that no visit mapping reads" in link.description
    assert "etlVisitLinkLoss.condition_occurrence.visit_occurrence_id" not in r
    # unknown keys beyond 100 - min_record_retention_pct: fatal
    lost = {"with_key": 100, "linked": 30, "other_person": 0, "filtered": 10}
    r = checks(_step("07_condition_occurrence", "condition_occurrence", stats={"visit_links": lost}))
    loss = r["etlVisitLinkLoss.condition_occurrence.visit_occurrence_id"]
    assert (loss.severity, loss.num_violated_rows, loss.num_denominator_rows, loss.failed) == ("fatal", 60, 90, True)
    # a failed visit (or person) load is the cause: no link checks
    failed_visits = _step("05_visit_occurrence", "visit_occurrence", error="boom")
    r = checks(failed_visits, _step("07_condition_occurrence", "condition_occurrence", stats={"visit_links": lost}))
    assert not any(k.startswith("etlVisitLink") for k in r)


# --------------------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------------------


def stage_extra(conn, src: str) -> None:
    """A clinic system beside the HDSS: patients (40 of 60 not in the HDSS), admissions numbered like
    the outpatient visits, their lab results; verbal-autopsy deaths (two rows per person); ten
    register rows written twice; and a diagnosis on another person's visit."""
    for sql in [
        f"""CREATE TABLE {src}.emr_patients AS
            SELECT g::text AS patient_no, CASE WHEN g > 40 THEN i.individual_id END AS hdss_id, '1990-01-01' AS dob
            FROM generate_series(1, 60) g
            JOIN (SELECT individual_id, ROW_NUMBER() OVER (ORDER BY individual_id) n FROM {src}.individuals) i ON i.n = g""",
        f"""CREATE TABLE {src}.ipd_admissions AS
            SELECT 'V' || lpad(g::text, 7, '0') AS admission_id, p.patient_no, p.hdss_id,
                   '2020-01-01' AS admit_date, '2020-01-03' AS discharge_date
            FROM generate_series(1, 60) g JOIN {src}.emr_patients p ON p.patient_no = g::text""",
        f"CREATE TABLE {src}.emr_labs AS SELECT admission_id, patient_no, hdss_id, admit_date AS test_date "
        f"FROM {src}.ipd_admissions",
        f"""CREATE TABLE {src}.va_deaths AS
            SELECT individual_id, (dod::date + 30)::text AS va_date, '0' AS verified FROM {src}.deaths
            UNION ALL SELECT individual_id, (dod::date - 30)::text, '1' FROM {src}.deaths
            UNION ALL SELECT i.individual_id, '2022-02-02', '0' FROM (SELECT individual_id FROM {src}.individuals
                WHERE individual_id NOT IN (SELECT individual_id FROM {src}.deaths) ORDER BY 1 DESC LIMIT 7) i
            UNION ALL SELECT i.individual_id, '2022-01-01', '1' FROM (SELECT individual_id FROM {src}.individuals
                WHERE individual_id NOT IN (SELECT individual_id FROM {src}.deaths) ORDER BY 1 DESC LIMIT 7) i""",
        f"INSERT INTO {src}.individuals SELECT * FROM {src}.individuals ORDER BY individual_id LIMIT 10",
        f"""INSERT INTO {src}.diagnoses (visit_id, individual_id, icd10, dx_date)
            SELECT v.visit_id, 'KH000400', 'B54', v.visit_date FROM {src}.clinic_visits v
            WHERE v.individual_id <> 'KH000400' ORDER BY v.visit_id LIMIT 1""",
    ]:
        execute(conn, sql)
    conn.commit()


@pytest.mark.postgres
def test_several_sources_load_once(staged):
    conn, schemas, cfg = staged
    stage_extra(conn, schemas.src)
    cdm, src = schemas.cdm, schemas.src
    result = run_etl(conn, schemas, multi_spec(), cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]

    def q(sql: str) -> int:
        return int(scalar(conn, sql))

    by = {s.step.split("_", 1)[1]: s for s in result.steps}
    # people: the register (ten rows written twice load once) + 40 clinic patients without an HDSS id
    assert q(f"SELECT COUNT(*) FROM {cdm}.person") == q(f"SELECT COUNT(DISTINCT person_id) FROM {cdm}.person") == 440
    assert {k: by["person"].stats[k] for k in ("inserted", "keys", "duplicate_rows", "already_loaded_keys",
                                                "conflict_keys", "fan_out_keys")} == {
        "inserted": 400, "keys": 400, "duplicate_rows": 10, "already_loaded_keys": 0, "conflict_keys": 0,
        "fan_out_keys": 0}
    assert by["person"].rows_inserted == 400 and by["person"].retained_rows == by["person"].source_rows == 410
    assert by["person_map_2"].rows_inserted == 40 and by["person_2"].rows_inserted == 40
    assert len(id_map(conn, schemas, "person_map")) == 440
    # visits: the outpatient visits + 60 admissions numbered like them, kept apart by the IPD: prefix
    n_opd = q(f"SELECT COUNT(*) FROM {src}.clinic_visits")
    assert q(f"SELECT COUNT(DISTINCT visit_occurrence_id) FROM {cdm}.visit_occurrence") == n_opd + 60
    assert by["visit_map_2"].rows_inserted == 60 and by["visit_occurrence_2"].rows_inserted == 60
    # clinic labs: all retained, each linked to an admission of their own person
    m2 = by["measurement_2"]
    assert m2.retained_rows == m2.source_rows == m2.rows_inserted == 60
    assert m2.stats["visit_links"] == {"with_key": 60, "linked": 60, "other_person": 0, "filtered": 0}
    # deaths: one per person; the register wins, then (no dedup_order) the latest VA date
    n_reg = q(f"SELECT COUNT(*) FROM {src}.deaths")
    assert q(f"SELECT COUNT(*) FROM {cdm}.death") == q(f"SELECT COUNT(DISTINCT person_id) FROM {cdm}.death") == n_reg + 7
    d2 = by["death_2"].stats
    assert (d2["keys"], d2["already_loaded_keys"], d2["duplicate_rows"], d2["inserted"]) == (
        n_reg + 7, n_reg, n_reg + 7, 7)
    assert d2["conflict_keys"] == n_reg + 7 and d2["dedup_order"] is False
    assert q(f"SELECT COUNT(*) FROM {cdm}.death WHERE death_date = '2022-02-02'") == 7
    # the diagnosis on another person's visit is loaded without a visit, and counted
    co = by["condition_occurrence"]
    links = co.stats["visit_links"]
    assert links["other_person"] == 1 and links["with_key"] - links["linked"] == 1 and links["filtered"] == 0
    assert q(f"""SELECT COUNT(*) FROM {cdm}.condition_occurrence c JOIN {cdm}.visit_occurrence v
                 USING (visit_occurrence_id) WHERE v.person_id <> c.person_id""") == 0
    checks = {r.key: r for r in etl_checks(result)}
    assert not checks["etlVisitLinkage.condition_occurrence.visit_occurrence_id"].failed
    assert checks["etlVisitLinkage.condition_occurrence.visit_occurrence_id"].num_violated_rows == 1
    assert not checks["etlDuplicateRecords.person"].failed and checks["etlDuplicateRecords.person"].num_violated_rows == 10
    assert not checks["etlKeyConflict.person"].failed
    assert checks["etlDuplicateRecords.death.death_2"].num_violated_rows == 2 * n_reg + 7
    assert checks["etlKeyConflict.death.death_2"].severity == "characterization"
    assert not any(r.failed for k, r in checks.items() if k.startswith(("etlKey", "etlDuplicate", "etlVisitLink")))
    # the built-in checks agree: one death per person, records link to visits of their person
    report = run_builtin_checks(conn, schemas, cfg, multi_spec(), result)
    integrity = {r.key: r for r in report.results if r.check_name in ("measureDuplicateDeaths", "fkVisitPerson")}
    assert not integrity["measureDuplicateDeaths.death"].failed
    assert integrity["fkVisitPerson.measurement.visit_occurrence_id"].num_denominator_rows > 0
    assert not any(r.failed for r in integrity.values())

    # with dedup_order, the verified VA row wins among the two VA rows of a person
    result = run_etl(conn, schemas, multi_spec("w.verified = '1' DESC"), cfg)
    assert result.ok
    assert q(f"SELECT COUNT(*) FROM {cdm}.death WHERE death_date = '2022-01-01'") == 7
    assert next(s for s in result.steps if s.step.endswith("death_2")).stats["dedup_order"] is True
    # rebuilt: the same ids
    assert len(id_map(conn, schemas, "person_map")) == 440 and count(conn, schemas, "person") == 440


def stage_wards(conn, src: str, rows: list[tuple]) -> None:
    execute(conn, f"CREATE TABLE {src}.wards (admission_no text, patient_ref text, ward_in text, ward_out text)")
    for r in rows:
        execute(conn, f"INSERT INTO {src}.wards VALUES (%s, %s, %s, %s)", r)
    conn.commit()


def ward_visit(key: str = "'IPD:' || a.admission_no") -> TableMapping:
    tm = ipd_visit("wards", key, "a.patient_ref")
    return set_field_tm(set_field_tm(tm, "visit_start_date", "a.ward_in::date"), "visit_end_date", "a.ward_out::date")


def set_field_tm(tm: TableMapping, field: str, expression: str, kind: str | None = None) -> TableMapping:
    """Copy of ``tm`` with one field's expression changed (the field is added, of ``kind``, when missing)."""
    tm = tm.model_copy(deep=True)
    fm = next((f for f in tm.fields if f.target_field == field), None)
    if fm is None:
        tm.fields.append(FieldMapping(target_field=field, kind=kind or "expression", expression=expression))
    else:
        fm.expression = expression
    return tm


@pytest.mark.postgres
def test_visit_key_shared_by_two_persons_still_fails(staged):
    conn, schemas, cfg = staged
    # X1 is one admission number written for two people; V0000001 is an outpatient visit of
    # KH000001 that the wards (unprefixed keys) also give to KH000002
    owner = scalar(conn, f"SELECT individual_id FROM {schemas.src}.clinic_visits WHERE visit_id = 'V0000001'")
    other = "KH000002" if owner != "KH000002" else "KH000003"
    stage_wards(conn, schemas.src, [("X1", "KH000001", "2021-02-01", "2021-02-02"),
                                    ("X1", other, "2021-02-01", "2021-02-03"),
                                    ("V0000001", other, "2021-03-01", "2021-03-01")])
    spec = fixed_ref().model_copy(deep=True)
    spec.table_mappings.append(ward_visit("a.admission_no"))
    for _ in range(2):  # the visit map records the same person for the shared keys on every run
        result = run_etl(conn, schemas, spec, cfg)
        assert result.ok, [(e.step, e.error) for e in result.errors()]
        people = {r["source_value"]: r["person_source_value"] for r in fetch_all(
            conn, f"SELECT source_value, person_source_value FROM {schemas.etl}.visit_map "
                  "WHERE source_value IN ('X1', 'V0000001')")}
        assert people == {"X1": min("KH000001", other), "V0000001": min(owner, other)}
    # each person gets a row: the key is not unique, and the fatal isPrimaryKey still blocks
    vid = id_map(conn, schemas, "visit_map")["X1"]
    assert count(conn, schemas, "visit_occurrence", f"visit_occurrence_id = {vid}") == 2
    assert count(conn, schemas, "visit_occurrence", "visit_occurrence_id = "
                 f"{id_map(conn, schemas, 'visit_map')['V0000001']}") == 2
    wards = next(s for s in result.steps if s.step.endswith("visit_occurrence_2"))
    assert wards.stats["conflict_keys"] == 2 and wards.stats["already_loaded_keys"] == 0
    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    by_key = {r.key: r for r in report.results}
    pk = by_key["isPrimaryKey.visit_occurrence.visit_occurrence_id"]
    assert pk.failed and pk.severity == "fatal" and not report.passed and pk.repairable is False
    conflict = by_key["etlKeyConflict.visit_occurrence.visit_occurrence_2"]
    assert (conflict.num_violated_rows, conflict.severity, conflict.failed) == (2, "characterization", False)
    assert "include the person key" in conflict.description
    passed, blocking = gate(report.results, cfg, result)
    assert not passed and any(b.startswith("isPrimaryKey.visit_occurrence") for b in blocking)
    # the records of the shared visits link to a row of their own person: not a foreign visit
    assert not any(r.failed for r in report.results if r.check_name == "fkVisitPerson")


@pytest.mark.postgres
def test_conflicting_reregistration(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    known = [r["individual_id"] for r in fetch_all(
        conn, f"SELECT individual_id FROM {src}.individuals WHERE sex IN ('1', '2') ORDER BY individual_id LIMIT 2")]
    fixed, unknown = known  # a sex corrected at re-registration; a re-registration with the sex unknown
    sex = {r["individual_id"]: r["sex"] for r in fetch_all(
        conn, f"SELECT individual_id, sex FROM {src}.individuals WHERE individual_id = ANY(%s)", (known,))}
    execute(conn, f"ALTER TABLE {src}.individuals ADD COLUMN reg_date text")
    execute(conn, f"UPDATE {src}.individuals SET reg_date = '2010-01-01'")
    execute(conn, f"""INSERT INTO {src}.individuals (individual_id, sex, dob, reg_date)
        SELECT individual_id, CASE sex WHEN '1' THEN '2' ELSE '1' END, dob, '2020-01-01'
        FROM {src}.individuals WHERE individual_id = %s""", (fixed,))
    execute(conn, f"""INSERT INTO {src}.individuals (individual_id, sex, dob, reg_date)
        SELECT individual_id, '9', dob, '2005-01-01' FROM {src}.individuals WHERE individual_id = %s""", (unknown,))
    conn.commit()
    gender = {"1": 8507, "2": 8532}

    def persons() -> dict[str, int]:
        rows = fetch_all(conn, f"SELECT person_source_value, gender_concept_id FROM {schemas.cdm}.person "
                               "WHERE person_source_value = ANY(%s)", (known,))
        return {r["person_source_value"]: r["gender_concept_id"] for r in rows}

    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert result.ok and count(conn, schemas, "person") == 400
    checks = {r.key: r for r in etl_checks(result)}
    conflict = checks["etlKeyConflict.person"]
    # only the corrected sex is a conflict: an unknown sex (concept 0) is not a disagreement, and
    # never wins over a known one
    assert (conflict.severity, conflict.num_violated_rows, conflict.failed) == ("convention", 1, True)
    assert persons()[unknown] == gender[sex[unknown]]
    dup = checks["etlDuplicateRecords.person"]
    assert (dup.num_violated_rows, dup.num_denominator_rows, dup.failed) == (2, 402, False)

    # the latest registration wins with a dedup_order, and the conflict is then decided: reported,
    # not failed. A registration without a readable date ranks last (NULLS LAST is added), even
    # under DESC, where PostgreSQL puts NULLs first
    execute(conn, f"""INSERT INTO {src}.individuals (individual_id, sex, dob, reg_date)
        SELECT individual_id, sex, dob, d FROM {src}.individuals, (VALUES (NULL), ('31/02/2009')) v(d)
        WHERE individual_id = %s AND reg_date = '2010-01-01'""", (fixed,))
    conn.commit()
    spec = fixed_ref().model_copy(deep=True)
    spec.mapping_for("person").dedup_order = "i.reg_date::date DESC"
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok
    assert persons() == {fixed: gender["2" if sex[fixed] == "1" else "1"], unknown: gender[sex[unknown]]}
    person = next(s for s in result.steps if s.target_table == "person")
    assert (person.stats["conflict_keys"], person.stats["undecided_keys"], person.value_counts) == (1, 0, {})
    conflict = next(r for r in etl_checks(result) if r.key == "etlKeyConflict.person")
    assert (conflict.severity, conflict.num_violated_rows, conflict.failed) == ("convention", 0, False)
    # a dedup_order that ranks every row of a key alike decides nothing: still a failure
    spec.mapping_for("person").dedup_order = "i.individual_id"
    result = run_etl(conn, schemas, spec, cfg)
    conflict = next(r for r in etl_checks(result) if r.key == "etlKeyConflict.person")
    assert (conflict.num_violated_rows, conflict.failed) == (1, True)

    # a second person mapping whose rows disagree with the person already loaded: a conflicting
    # key (the earlier mapping's person is kept), not one this mapping can decide
    again = spec.mapping_for("person").model_copy(deep=True, update={"dedup_order": None})
    again.source.table, again.where = "individuals", f"i.individual_id = '{fixed}'"
    again = set_field_tm(again, "person_id", "'' || i.individual_id")
    spec.table_mappings.insert(1, again)
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok
    person_2 = next(s for s in result.steps if s.step.endswith("person_2"))
    assert (person_2.stats["inserted"], person_2.stats["conflict_keys"], person_2.stats["undecided_keys"]) == (0, 1, 0)
    assert not next(r for r in etl_checks(result) if r.key == "etlKeyConflict.person.person_2").failed


@pytest.mark.postgres
def test_visit_rollup_from_rows(staged):
    conn, schemas, cfg = staged
    stage_wards(conn, schemas.src, [("A1", "KH000001", "2021-02-01", "2021-02-03"),
                                    ("A1", "KH000001", "2021-02-03", "2021-02-07"),
                                    ("A2", "KH000002", "2021-03-01", None)])
    execute(conn, f"""INSERT INTO {schemas.src}.diagnoses (visit_id, individual_id, icd10, dx_date)
        VALUES ('IPD:A1', 'KH000001', 'B54', '2021-02-06')""")
    execute(conn, f"ALTER TABLE {schemas.src}.wards ADD COLUMN came_from text, ADD COLUMN went_to text")
    execute(conn, f"""UPDATE {schemas.src}.wards SET came_from = CASE WHEN ward_in = '2021-02-01' THEN 'home'
        ELSE 'ward' END, went_to = CASE WHEN ward_in = '2021-02-01' THEN 'ward' ELSE 'died' END""")
    conn.commit()
    spec = fixed_ref().model_copy(deep=True)
    ward = set_field_tm(ward_visit(), "admitted_from_source_value", "a.came_from", kind="column")
    spec.table_mappings.append(set_field_tm(ward, "discharged_to_source_value", "a.went_to", kind="column"))
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    vm = id_map(conn, schemas, "visit_map")
    rows = {r["visit_occurrence_id"]: (str(r["visit_start_date"]), str(r["visit_end_date"]), r["admitted_from_source_value"],
                                       r["discharged_to_source_value"]) for r in fetch_all(
        conn, f"SELECT * FROM {schemas.cdm}.visit_occurrence WHERE visit_occurrence_id IN (%s, %s)",
        (vm["IPD:A1"], vm["IPD:A2"]))}
    # the two ward stays are one visit, from the earliest start to the latest end, from where the
    # patient came at the first stay to where they went after the last; a missing end is the start
    assert rows == {vm["IPD:A1"]: ("2021-02-01", "2021-02-07", "home", "died"),
                    vm["IPD:A2"]: ("2021-03-01", "2021-03-01", "ward", "died")}
    wards = next(s for s in result.steps if s.step.endswith("visit_occurrence_2"))
    assert (wards.rows_inserted, wards.stats["duplicate_rows"], wards.stats["gap_keys"], wards.value_counts) == (
        2, 1, 0, {"defaulted": {"visit_end_date": 1}})
    # the diagnosis made during the second ward stay is linked to the admission, within its dates
    assert count(conn, schemas, "condition_occurrence",
                 f"visit_occurrence_id = {vm['IPD:A1']} AND condition_start_date = DATE '2021-02-06'") == 1
    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    within = [r for r in report.results if r.check_name == "withinVisitDates" and r.table == "condition_occurrence"]
    assert within and all(r.num_violated_rows == 0 and not r.error for r in within)


@pytest.mark.postgres
def test_failed_visit_source_keeps_links(staged):
    conn, schemas, cfg = staged
    stage_extra(conn, schemas.src)
    spec = multi_spec()
    broken = next(tm for tm in spec.table_mappings if tm.source.table == "ipd_admissions")
    next(f for f in broken.fields if f.target_field == "visit_concept_id").kind = "expression"
    next(f for f in broken.fields if f.target_field == "visit_concept_id").expression = \
        "9201 / (length(a.admission_id) - length(a.admission_id))"
    result = run_etl(conn, schemas, spec, cfg)
    assert [e.step.split("_", 1)[1] for e in result.errors()] == ["visit_occurrence_2"]
    # the pruning still runs: the outpatient visits keep their person, the admissions are retired
    # (with their ids), and the outpatient diagnoses keep their visits
    retired = id_map(conn, schemas, "visit_map_retired")
    assert len(retired) == 60 and all(k.startswith("IPD:") for k in retired)
    with_visit = count(conn, schemas, "condition_occurrence", "visit_occurrence_id IS NOT NULL")
    assert with_visit == count(conn, schemas, "condition_occurrence") - 1  # the one on another person's visit
    # the failed visit load is the cause of the clinic labs without a visit: no link checks
    assert not any(r.check_name.startswith("etlVisitLink") for r in etl_checks(result))
    # the next good build restores the admissions' ids
    assert run_etl(conn, schemas, multi_spec(), cfg).ok
    assert {k: id_map(conn, schemas, "visit_map")[k] for k in retired} == retired


@pytest.mark.postgres
def test_link_classes(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    execute(conn, f"""INSERT INTO {src}.diagnoses (visit_id, individual_id, icd10, dx_date)
        SELECT v.visit_id, 'KH000400', 'B54', v.visit_date FROM {src}.clinic_visits v
        WHERE v.individual_id <> 'KH000400' AND v.visit_type <> 'EMERG' ORDER BY v.visit_id LIMIT 2""")
    execute(conn, f"""INSERT INTO {src}.diagnoses (visit_id, individual_id, icd10, dx_date)
        VALUES ('ipd:7', 'KH000001', 'B54', '2020-01-01')""")
    conn.commit()
    spec = fixed_ref().model_copy(deep=True)
    spec.mapping_for("visit_occurrence").where = "v.visit_type <> 'EMERG'"
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok
    on_emerg = int(scalar(conn, f"""SELECT COUNT(*) FROM {src}.diagnoses d JOIN {src}.clinic_visits v
        USING (visit_id) WHERE v.visit_type = 'EMERG'"""))
    total = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.diagnoses"))
    co = next(s for s in result.steps if s.target_table == "condition_occurrence")
    assert on_emerg > 0 and co.stats["visit_links"] == {
        "with_key": total, "linked": total - on_emerg - 3, "other_person": 2, "filtered": on_emerg}
    link = next(r for r in etl_checks(result) if r.key == "etlVisitLinkage.condition_occurrence.visit_occurrence_id")
    # the filtered visits are deliberate: not counted; 2 other-person rows and 1 unknown key are
    assert (link.num_violated_rows, link.num_denominator_rows, link.failed) == (3, total - on_emerg, False)
    assert not any(r.check_name == "etlVisitLinkLoss" for r in etl_checks(result))

    # a visit key written in another form loses (nearly) every link: fatal
    broken = set_field(spec, "condition_occurrence", "visit_occurrence_id", expression="lower(d.visit_id)")
    result = run_etl(conn, schemas, broken, cfg)
    assert result.ok
    checks = {r.key: r for r in etl_checks(result, cfg.quality.min_record_retention_pct)}
    loss = checks["etlVisitLinkLoss.condition_occurrence.visit_occurrence_id"]
    assert loss.failed and loss.severity == "fatal" and loss.num_denominator_rows == total
    assert checks["etlVisitLinkage.condition_occurrence.visit_occurrence_id"].failed
    assert not checks["etlVisitLinkage.drug_exposure.visit_occurrence_id"].failed
    report = run_builtin_checks(conn, schemas, cfg, broken, result)
    assert not report.passed and any(b.startswith("etlVisitLinkLoss") for b in report.blocking_failures)


@pytest.mark.postgres
def test_post_load_integrity_checks(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref().model_copy(deep=True)
    spec.custom_sql = [
        CustomSQL(name="second deaths", target_table="death", sql=(
            "INSERT INTO {cdm}.death (person_id, death_date, death_type_concept_id) "
            "SELECT person_id, death_date - 1, death_type_concept_id FROM {cdm}.death ORDER BY person_id LIMIT 3")),
        CustomSQL(name="foreign visit", target_table="condition_occurrence", sql=(
            "INSERT INTO {cdm}.condition_occurrence (condition_occurrence_id, person_id, condition_concept_id, "
            "condition_start_date, condition_type_concept_id, visit_occurrence_id) "
            "SELECT 999999, p.person_id, 0, v.visit_start_date, 32817, v.visit_occurrence_id "
            "FROM {cdm}.visit_occurrence v JOIN {cdm}.person p ON p.person_id <> v.person_id "
            "ORDER BY v.visit_occurrence_id, p.person_id LIMIT 1")),
    ]
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    by_key = {r.key: r for r in report.results}
    deaths = by_key["measureDuplicateDeaths.death"]
    assert (deaths.num_violated_rows, deaths.severity, deaths.failed, deaths.repairable) == (3, "fatal", True, False)
    assert deaths.num_denominator_rows == count(conn, schemas, "death", "TRUE") - 3
    fk = by_key["fkVisitPerson.condition_occurrence.visit_occurrence_id"]
    assert (fk.num_violated_rows, fk.severity, fk.failed, fk.repairable) == (1, "fatal", True, False)
    assert not by_key["fkVisitPerson.drug_exposure.visit_occurrence_id"].failed
    assert not report.passed and any(b.startswith("measureDuplicateDeaths") for b in report.blocking_failures)


@pytest.mark.postgres
def test_permission_errors_are_not_sent_to_the_repair_agent(staged):
    from omop_agent.quality.repair import repair_triggers

    conn, schemas, cfg = staged
    role = f"{schemas.cdm}_role"
    conn.rollback()
    execute(conn, f"CREATE ROLE {role} NOLOGIN")
    execute(conn, f"GRANT USAGE ON SCHEMA {schemas.src}, {schemas.vocab}, {schemas.cdm}, {schemas.etl} TO {role}")
    execute(conn, f"GRANT SELECT ON ALL TABLES IN SCHEMA {schemas.src}, {schemas.vocab} TO {role}")
    execute(conn, f"GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA {schemas.cdm}, {schemas.etl} TO {role}")
    # an ETL table an administrator has not granted (e.g. one a new version added)
    execute(conn, f"REVOKE ALL ON {schemas.etl}.person_map_retired FROM {role}")
    conn.commit()
    try:
        cfg = cfg.model_copy(update={"target": cfg.target.model_copy(update={"etl_role": role})})
        result = run_etl(conn, schemas, fixed_ref(), cfg)
        pm = next(s for s in result.steps if s.step.endswith("_person_map"))
        assert "permission denied" in pm.error and not pm.repairable and not result.ok
        assert f"GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA {schemas.cdm}, {schemas.etl} TO {role}" in pm.error
        error = next(r for r in etl_checks(result) if r.check_name == "etlStepError" and r.step == "person_map")
        assert error.repairable is False and error.severity == "fatal"
        assert repair_triggers(None, result, cfg) == ([], [])
    finally:
        conn.rollback()
        execute(conn, f"DROP OWNED BY {role}")
        execute(conn, f"DROP ROLE {role}")
        conn.commit()


@pytest.mark.postgres
def test_validator_explains_the_generated_steps(staged, monkeypatch):
    from omop_agent.db import rollback_only
    from omop_agent.design import validate

    conn, schemas, cfg = staged
    stage_extra(conn, schemas.src)
    spec = multi_spec()
    assert [str(i) for i in validate_spec(spec, conn=conn, schemas=schemas, cfg=cfg) if i.severity == "error"] == []

    def explained(clean: list[TableMapping]) -> list[str]:
        with rollback_only(conn):  # a visit map table that is not there: its steps cannot run
            execute(conn, f"DROP TABLE {schemas.etl}.visit_map_retired")
            return [str(i) for i in validate._explain_generated(spec, clean, [], conn, schemas, cfg)]

    # every id-map step (the visit_key's and the other sources') and the pruning are EXPLAINed: a
    # blocking step that cannot run is an error
    issues = explained(spec.table_mappings)
    assert [i.split(":")[0] for i in issues] == [
        "ERROR step visit_map cannot run (the build would fail)",
        "ERROR step visit_map_2 cannot run (the build would fail)",
        "ERROR step visit_map_prune cannot run (the build would fail)",
    ], issues
    assert all("visit_map_retired" in i and "does not exist" in i for i in issues)
    # a step that runs the SQL of a mapping with errors is left out (its errors are reported already)
    issues = explained([tm for tm in spec.table_mappings if tm.source.table != "ipd_admissions"])
    assert not any("visit_map_2" in i for i in issues) and any("visit_map_prune" in i for i in issues)
    # a non-blocking step that cannot run is a warning (its error would be reported, and the build goes on)
    real = validate.compile_spec

    def prune_not_blocking(*args, **kwargs):
        steps = real(*args, **kwargs)
        for s in steps:
            s.blocking = s.kind != "prune"
        return steps

    monkeypatch.setattr(validate, "compile_spec", prune_not_blocking)
    assert any(i.startswith("WARNING step visit_map_prune cannot run (the step would fail and be reported, "
                            "without failing the build)") for i in explained(spec.table_mappings))


@pytest.mark.postgres
def test_visit_links_follow_the_source_between_runs(staged):
    conn, schemas, cfg = staged
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    vid = id_map(conn, schemas, "visit_map")["V0000001"]
    owner = scalar(conn, f"SELECT individual_id FROM {schemas.src}.clinic_visits WHERE visit_id = 'V0000001'")
    other = "KH000400" if owner != "KH000400" else "KH000399"
    n_dx = int(scalar(conn, f"SELECT COUNT(*) FROM {schemas.src}.diagnoses WHERE visit_id = 'V0000001'"))
    assert n_dx > 0 and count(conn, schemas, "condition_occurrence", f"visit_occurrence_id = {vid}") == n_dx
    # the extract gives the visit to another person; its diagnoses still name the first one
    execute(conn, f"UPDATE {schemas.src}.clinic_visits SET individual_id = %s WHERE visit_id = 'V0000001'", (other,))
    conn.commit()
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert result.ok and id_map(conn, schemas, "visit_map")["V0000001"] == vid
    assert count(conn, schemas, "condition_occurrence", f"visit_occurrence_id = {vid}") == 0
    co = next(s for s in result.steps if s.target_table == "condition_occurrence")
    assert co.stats["visit_links"]["other_person"] == n_dx


def test_report_and_spec_document_show_keys():
    from omop_agent.design.render import render_spec_md
    from omop_agent.report.inspection import _design_section, _etl_section

    spec = multi_spec("w.verified = '1' DESC")
    design = _design_section(spec)
    assert design["person_key_also"] == "emr_patients: 'EMR:' || e.patient_no"
    assert design["visit_key_also"] == "ipd_admissions: 'IPD:' || a.admission_id"
    assert [t["dedup_order"] for t in design["tables"] if t["target"] == "death"] == ["", "w.verified = '1' DESC"]
    md = render_spec_md(spec)
    assert "- **Person key:** `individuals`.`individual_id`; also from `emr_patients`.`'EMR:' || e.patient_no`" in md
    assert "One row per key, best row first by: `w.verified = '1' DESC`" in md

    stats = {"keys": 96, "duplicate_rows": 30, "already_loaded_keys": 45, "conflict_keys": 3, "fan_out_keys": 0}
    run = ETLRunResult(started_at="2026-01-01T00:00:00Z", steps=[
        _step("03_person", "person", rows_inserted=21, source_rows=100, retained_rows=96, stats=stats),
        _step("07_condition_occurrence", "condition_occurrence", rows_inserted=90, stats={
            "visit_links": {"with_key": 100, "linked": 90, "other_person": 3, "filtered": 0}}),
        _step("10_death", "death", rows_inserted=5, stats=stats | {"duplicate_rows": 2, "fan_out_keys": 4}),
    ])
    steps = _etl_section(run)["steps"]
    assert [(s["one_row_per_key"], s["visit_links"]) for s in steps] == [
        ("30 dup / 45 earlier / 3 conflicting", ""), ("", "90/100, 3 other person"),
        ("2 dup / 45 earlier / 3 conflicting / 4 fan-out", "")]
    # retention of a step that keeps one row per key: the source rows kept, not the rows inserted
    assert steps[0]["retention"] == 96.0
    # the shareable report withholds the small counts
    steps = _etl_section(run, 5, shareable=True)["steps"]
    assert steps[0]["one_row_per_key"] == "30 dup / 45 earlier / <5 conflicting"
    assert steps[1]["visit_links"] == "90/100, <5 other person"


def test_build_history_spots_lost_visit_links():
    from omop_agent.pipeline import _BuildHistory

    def run(with_key: int, linked: int) -> ETLRunResult:
        links = {"with_key": with_key, "linked": linked, "other_person": 0, "filtered": 0}
        return ETLRunResult(started_at="2026-01-01T00:00:00Z", steps=[
            _step("07_condition_occurrence", "condition_occurrence", rows_inserted=100, stats={"visit_links": links})])

    h = _BuildHistory()
    h.add(run(100, 90), None)
    assert h.losses(run(100, 95), None) == []
    # visit keys made NULL leave the link checks without anything to count: a loss all the same
    assert h.losses(run(0, 0), None) == ["condition_occurrence has 0 records with a visit key, 100 before"]
    assert h.losses(run(100, 10), None) == ["condition_occurrence has 10 records linked to their visit, 90 before"]


def test_key_texts_compare_as_the_id_maps_do():
    # a cast to text is no other form of the key: the id map compares keys as text anyway
    for written in ("i.individual_id::text", "(i.individual_id)::varchar", "( I.individual_id :: TEXT )"):
        spec = set_field(fixed_ref(), "person", "person_id", expression=written)
        assert key_sources(spec, "person") == [], written
        assert not any("person key table" in i.message for i in validate_spec(spec)), written
    spec = set_field(fixed_ref(), "person", "person_id", expression="i.individual_id::int")
    assert [tm.source.table for tm, _, _ in key_sources(spec, "person")] == ["individuals"]


def test_visit_key_shapes_follow_the_visit_mappings():
    # the visit mapping writes the key in another form than the visit_key: records written like the
    # visit_key link to nothing, so their shape is reported
    spec = set_field(fixed_ref(), "visit_occurrence", "visit_occurrence_id", expression="'OPD:' || v.visit_id")
    msgs = [str(i) for i in validate_spec(spec) if "another form than the visit keys" in i.message]
    assert any(m.startswith("WARNING condition_occurrence.visit_occurrence_id: the visit key is written as d.visit_id")
               for m in msgs), msgs
    assert not any("(visit_key, clinic_visits)" in m for m in msgs)


@pytest.mark.postgres
def test_null_fields_of_one_row_per_key_tables(staged):
    conn, schemas, cfg = staged
    spec = fixed_ref()
    for table, field in [("person", "location_id"), ("person", "race_source_value"), ("person", "care_site_id"),
                         ("death", "death_datetime"), ("death", "cause_concept_id"),
                         ("visit_occurrence", "visit_end_datetime"), ("visit_occurrence", "care_site_id")]:
        spec = set_field(spec, table, field, kind="null", expression=None)
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    assert count(conn, schemas, "person") == 400 and count(conn, schemas, "visit_occurrence", "visit_end_datetime IS NULL") \
        == count(conn, schemas, "visit_occurrence")


@pytest.mark.postgres
def test_visit_links_count_against_the_visit_mappings_keys(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    total = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.diagnoses"))

    def links_and_loss(spec):
        result = run_etl(conn, schemas, spec, cfg)
        assert result.ok, [(e.step, e.error) for e in result.errors()]
        co = next(s for s in result.steps if s.target_table == "condition_occurrence")
        checks = {r.key: r for r in etl_checks(result, cfg.quality.min_record_retention_pct)}
        return co.stats["visit_links"], checks.get("etlVisitLinkLoss.condition_occurrence.visit_occurrence_id")

    # the visit mapping writes 'OPD:' || visit_id, the diagnoses the visit_key's plain form: the
    # visit_key's keys are never visits, so they are not "filtered out": the links are lost (fatal)
    opd = set_field(fixed_ref(), "visit_occurrence", "visit_occurrence_id", expression="'OPD:' || v.visit_id")
    links, loss = links_and_loss(opd)
    assert links == {"with_key": total, "linked": 0, "other_person": 0, "filtered": 0}
    assert loss is not None and loss.failed and loss.num_violated_rows == total
    # the same when the format changes between builds: the keys an older spec retired are not
    # filtered out by this one
    assert run_etl(conn, schemas, fixed_ref(), cfg).ok
    opd.visit_key.expression = "'OPD:' || visit_id"
    links, loss = links_and_loss(opd)
    assert (links["filtered"], links["linked"]) == (0, 0) and loss is not None and loss.failed
    # written as the visit mapping writes it, every diagnosis links again
    links, loss = links_and_loss(set_field(opd, "condition_occurrence", "visit_occurrence_id",
                                           expression="'OPD:' || d.visit_id"))
    assert links["linked"] == total - 0 and loss is None


@pytest.mark.postgres
def test_a_filter_of_another_visit_source_is_filtered_not_lost(staged):
    conn, schemas, cfg = staged
    stage_extra(conn, schemas.src)
    spec = multi_spec()
    ipd = next(tm for tm in spec.table_mappings if tm.source.table == "ipd_admissions")
    ipd.where = "a.patient_no::int > 20"  # 20 of 60 admissions left out on purpose
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    m2 = next(s for s in result.steps if s.step.endswith("measurement_2"))
    assert m2.stats["visit_links"] == {"with_key": 60, "linked": 40, "other_person": 0, "filtered": 20}
    checks = {r.key: r for r in etl_checks(result, cfg.quality.min_record_retention_pct)}
    link = checks["etlVisitLinkage.measurement.visit_occurrence_id.measurement_2"]
    assert (link.num_violated_rows, link.num_denominator_rows, link.failed) == (0, 40, False)
    assert not any(k.startswith("etlVisitLinkLoss") for k in checks)


@pytest.mark.postgres
def test_visit_key_coarser_than_a_visit_fails(staged):
    conn, schemas, cfg = staged
    # one visit per person: every visit of a person is rolled up into one spanning years
    spec = fixed_ref().model_copy(deep=True)
    spec.visit_key.expression = "individual_id"
    for tm in spec.table_mappings:
        for fm in tm.fields:
            if fm.kind == "visit_id":
                fm.expression = f"{tm.source.alias}.individual_id"
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    grain = next(r for r in report.results if r.key == "etlVisitGrain.visit_occurrence")
    assert grain.failed and grain.severity == "fatal" and grain.repairable is False
    assert grain.num_violated_rows > grain.num_denominator_rows / 2
    passed, blocking = gate(report.results, cfg, result)
    assert not passed and any(b.startswith("etlVisitGrain.visit_occurrence") for b in blocking)
    # the visits of the demo are one row each: no separate stays
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    visits = next(s for s in result.steps if s.target_table == "visit_occurrence")
    assert visits.stats["gap_keys"] == 0


@pytest.mark.postgres
def test_a_death_date_known_to_the_day_wins(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    execute(conn, f"CREATE TABLE {src}.va_deaths (individual_id text, va_date text, verified text)")
    execute(conn, f"""INSERT INTO {src}.va_deaths VALUES ('KH000001', '2022-07-19', NULL), ('KH000001', '2022', '1'),
        ('KH000002', '2021', NULL), ('KH000002', '2021-06', NULL), ('KH000003', '2020-05-05', '1'),
        ('KH000003', 'garbage', '1')""")
    conn.commit()
    spec = fixed_ref().model_copy(deep=True)
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "death"] + [va_deaths()]
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]

    def deaths() -> dict[str, str]:
        rows = fetch_all(conn, f"""SELECT p.person_source_value, d.death_date FROM {schemas.cdm}.death d
            JOIN {schemas.cdm}.person p USING (person_id)""")
        return {r["person_source_value"]: str(r["death_date"]) for r in rows}

    # the precise date of the same death wins over the year (completed to 31 December); between two
    # partial dates the latest completion; an unreadable date is dropped before the choice
    assert deaths() == {"KH000001": "2022-07-19", "KH000002": "2021-12-31", "KH000003": "2020-05-05"}
    death = next(s for s in result.steps if s.target_table == "death")
    assert death.value_counts["imputed"] == {"death_date": 3} and death.stats["conflict_keys"] == 2
    # a dedup_order comes first: the verified row wins, and a missing flag ranks last even under DESC
    spec.table_mappings[-1].dedup_order = "w.verified = '1' DESC"
    assert run_etl(conn, schemas, spec, cfg).ok
    assert deaths() == {"KH000001": "2022-12-31", "KH000002": "2021-12-31", "KH000003": "2020-05-05"}


@pytest.mark.postgres
def test_repair_keeps_visit_keys_and_fixes_keys_that_never_ran(repair_env):
    state, tools = repair_env
    update = tools["update_field_mapping"].invoke
    assert run_etl(state.conn, state.schemas, state.spec, state.cfg).ok  # the guard counts over the id maps

    def edit(field: str, kind: str, expression: str) -> str:
        return update({"target_table": "condition_occurrence", "target_field": field, "kind": kind,
                       "expression": expression, "rationale": "x"})

    # making visit keys NULL hides the records that lost their visit from the link checks: refused
    for kind, expression in (("visit_id", "NULLIF(d.visit_id, d.visit_id)"),
                             ("visit_id", "CASE WHEN d.visit_id ~ '^X' THEN d.visit_id END"), ("constant", "NULL")):
        out = edit("visit_occurrence_id", kind, expression)
        assert out.startswith("REJECTED: it would take the visit key away from"), (expression, out)
    assert state.actions == []
    # another form of the key is applied (the link checks count what it links); a blank or NA-marker
    # key is no key, and may become NULL
    assert edit("visit_occurrence_id", "visit_id", "upper(d.visit_id)").startswith("OK:")
    execute(state.conn, f"UPDATE {state.schemas.src}.diagnoses SET visit_id = 'NA' WHERE icd10 LIKE 'A%'")
    state.conn.commit()
    assert edit("visit_occurrence_id", "visit_id", "NULLIF(upper(d.visit_id), 'NA')").startswith("OK:")
    # a person key may not change while the mapping loads records; once it cannot run at all (a
    # misspelled column), fixing it re-attributes nothing
    out = edit("person_id", "person_id", "trim(d.individual_id)")
    assert out.startswith("REJECTED: it changes the person key"), out
    cond = state.spec.mapping_for("condition_occurrence")
    next(f for f in cond.fields if f.target_field == "person_id").expression = "d.individualid"
    assert edit("person_id", "person_id", "d.individual_id").startswith("OK:")


def test_designer_lists_the_drafted_keys():
    from omop_agent.design.designer import _keys_text

    spec = multi_spec()
    text = _keys_text(spec.person_key, spec.visit_key, spec.table_mappings)
    assert ("Person mappings drafted so far (a record refers to a person with exactly the same key text as one of "
            "them):\n  - individuals (i): i.individual_id\n  - emr_patients (e): 'EMR:' || e.patient_no") in text
    assert ("Visit mappings drafted so far (their visit keys):\n  - clinic_visits (v): v.visit_id\n"
            "  - ipd_admissions (a): 'IPD:' || a.admission_id") in text
    assert "drafted so far" not in _keys_text(spec.person_key, spec.visit_key, [])


def test_prompts_say_the_first_mapping_wins():
    """Rows of one key are ranked and spanned within one mapping; between the mappings of a table the
    first in spec order that loads a key wins outright. The prompts, the field description, the
    review text and the validator say so (a VA death after the register's loses its cause)."""
    from omop_agent.design.designer import STAGING_CONVENTIONS, _plan_prompt
    from omop_agent.design.render import ONE_ROW_PER_KEY_TEXT
    from omop_agent.quality.repair import REPAIR_SYSTEM

    conventions = " ".join(STAGING_CONVENTIONS.split())
    assert "the first mapping that loads a key wins outright" in conventions
    assert "List the authoritative source first" in conventions and "must come through that mapping" in conventions
    plan = " ".join(_plan_prompt("p", "a").split())
    assert "(outpatient visits, admissions, ward stays)" not in plan
    assert "load them through the admission's mapping (or as visit_detail)" in plan
    assert "the first mapping that loads a key wins outright, so list the authoritative source first" in plan
    assert "the first in spec order that loads a key wins" in TableMapping.model_fields["dedup_order"].description
    assert "the first in spec order that loads a key wins" in " ".join(REPAIR_SYSTEM.split())
    assert "the first (in spec order) that loads a key wins outright" in ONE_ROW_PER_KEY_TEXT
    spec = fixed_ref().model_copy(deep=True)
    emr = emr_person()
    key = next(f for f in emr.fields if f.target_field == "person_id")
    key.expression = "e.patient_no"  # no prefix: a key found in both registers is one person
    spec.table_mappings.append(emr)
    assert any("loaded from the mapping that comes first in the spec" in i.message for i in validate_spec(spec))
