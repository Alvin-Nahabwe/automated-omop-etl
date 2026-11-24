"""Mechanical fixes to drafted table mappings (design/tidy.py)."""

from __future__ import annotations

from omop_agent.design.tidy import tidy_mapping
from omop_agent.models import TableMapping


def mapping(target: str, fields: list[dict], **kw) -> TableMapping:
    return TableMapping.model_validate({"target_table": target, "source": {"table": "src", "alias": "s"},
                                        "fields": fields, **kw})


def test_missing_surrogate_key_is_generated():
    tm, notes = tidy_mapping(mapping("observation_period", [{"target_field": "person_id", "kind": "person_id",
                                                             "expression": "s.id"}]))
    assert (tm.fields[0].target_field, tm.fields[0].kind) == ("observation_period_id", "auto_id")
    assert notes == ["observation_period.observation_period_id: the primary key was not mapped; it is generated "
                     "(auto_id)."]
    # keys that must come from the id maps are left for the validator to report
    for target in ("person", "visit_occurrence"):
        tm, notes = tidy_mapping(mapping(target, []))
        assert tm.fields == [] and notes == []


def test_helper_functions_get_the_schema_placeholder():
    tm, notes = tidy_mapping(mapping(
        "measurement",
        [{"target_field": "value_as_number", "kind": "expression", "expression": "ETL.try_number(s.result)"},
         {"target_field": "measurement_date", "kind": "expression", "expression": "{etl}.try_date(s.d)"}],
        where="etl.try_number(s.result) IS NOT NULL", joins=[{"table": "v", "alias": "v", "on": "v.id = s.vid"}]))
    expr = {f.target_field: f.expression for f in tm.fields}
    assert expr["value_as_number"] == "{etl}.try_number(s.result)"
    assert expr["measurement_date"] == "{etl}.try_date(s.d)"  # already right
    assert tm.where == "{etl}.try_number(s.result) IS NOT NULL"
    assert any("(value_as_number, where)" in n for n in notes)
    # an alias called etl is a table, not the schema
    tm, notes = tidy_mapping(mapping("measurement", [{"target_field": "value_as_number", "kind": "expression",
                                                      "expression": "etl.try_number(s.x)"}],
                                     joins=[{"table": "lab", "alias": "etl", "on": "etl.id = s.id"}]))
    assert [f.expression for f in tm.fields if f.target_field == "value_as_number"] == ["etl.try_number(s.x)"]
    assert not any("placeholder" in n for n in notes)


def test_zero_foreign_keys_become_null():
    tm, notes = tidy_mapping(mapping("person", [
        {"target_field": "care_site_id", "kind": "constant", "expression": "0"},
        {"target_field": "provider_id", "kind": "constant", "expression": "'0'::integer"},
        {"target_field": "race_concept_id", "kind": "constant", "expression": "0"},  # a concept: 0 is right
        {"target_field": "location_id", "kind": "constant", "expression": "7"},
    ]))
    kinds = {f.target_field: (f.kind, f.expression) for f in tm.fields}
    assert kinds == {"care_site_id": ("null", None), "provider_id": ("null", None),
                     "race_concept_id": ("constant", "0"), "location_id": ("constant", "7")}
    assert len(notes) == 2


def test_a_clean_mapping_is_unchanged():
    tm = mapping("death", [{"target_field": "person_id", "kind": "person_id", "expression": "s.id"}])
    assert tidy_mapping(tm) == (tm, [])


def test_a_visit_key_the_plan_left_out_is_taken_from_the_visit_mapping():
    from omop_agent.design.designer import _derived_visit_key
    from omop_agent.models import KeySpec

    visit = TableMapping.model_validate({
        "target_table": "visit_occurrence", "source": {"table": "clinic_visits", "alias": "v"},
        "fields": [{"target_field": "visit_occurrence_id", "kind": "visit_id",
                    "expression": "v.individual_id || ':' || V.visit_no"}]})
    key, note = _derived_visit_key(None, [mapping("person", []), visit])
    assert key == KeySpec(table="clinic_visits", expression="individual_id || ':' || visit_no")
    assert note and "clinic_visits.individual_id" in note
    planned = KeySpec(table="t", expression="k")
    assert _derived_visit_key(planned, [visit]) == (planned, None)
    joined = TableMapping.model_validate({**visit.model_dump(), "joins": [{"table": "x", "alias": "x",
                                                                         "on": "x.id = v.id"}]})
    joined.fields[0].expression = "x.visit_no"
    assert _derived_visit_key(None, [joined]) == (None, None)
