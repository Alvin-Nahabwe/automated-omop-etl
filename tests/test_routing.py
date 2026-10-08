"""Routing by domain: the route steps, their place in the build, the fields they carry, the
accepted domains of the mapping side, and the checks, history and report around them."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from omop_agent.cdm.ddl import create_cdm_tables, create_etl_tables
from omop_agent.cdm.spec import ACCEPTED_DOMAINS, EVENT_FIELD_ROLES, ROUTE_SOURCES, ROUTE_TABLES, load_cdm_spec
from omop_agent.db import Schemas, execute, fetch_all, scalar
from omop_agent.design.designer import DESIGN_SYSTEM, STAGING_CONVENTIONS, _plan_prompt
from omop_agent.design.render import render_spec_md
from omop_agent.design.validate import validate_spec
from omop_agent.etl.compiler import CompiledStep, compile_spec, target_tables
from omop_agent.etl.routing import (
    accepted_domains_for,
    pair_fields,
    route_only_tables,
    route_prepare_sql,
    route_sources,
    route_sql,
    route_targets,
    vocab_indexed,
    vocabulary_domains,
)
from omop_agent.etl.runner import _run_step, run_etl
from omop_agent.models import (
    CheckResult,
    CustomSQL,
    DQReport,
    ETLRunResult,
    FieldMapping,
    MappingDecision,
    RoutedRows,
    TableLoadResult,
)
from omop_agent.pipeline import _BuildHistory
from omop_agent.quality.checks import etl_checks, repairable, run_builtin_checks
from omop_agent.quality.repair import REPAIR_SYSTEM, data_loss_issues, repair_triggers, routing_issues
from omop_agent.report.inspection import _etl_section
from omop_agent.sqlguard import strip_literals
from omop_agent.vocab import review
from omop_agent.vocab.mapper import (
    _exact_match,
    _format_term,
    _Hit,
    _lexical,
    _validate_row,
)
from omop_agent.workspace import Workspace
from tests.test_etl import S, check_step_invariants, count, fixed_ref, load_ref, set_field, step
from tests.test_quality import plain_config

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def labels(steps) -> list[str]:
    return [s.label for s in steps]


# --------------------------------------------------------------------------------------
# the route block in the compiled build (no database)
# --------------------------------------------------------------------------------------


def test_route_steps_follow_event_loads():
    spec = load_ref().model_copy(deep=True)
    spec.custom_sql = [
        CustomSQL(name="specimens", target_table="specimen", sql="SELECT 1"),
        CustomSQL(name="costs", target_table="cost", sql="SELECT 2"),
        CustomSQL(name="plans", target_table="payer_plan_period", sql="SELECT 3"),
        CustomSQL(name="eras", target_table="drug_era", sql="SELECT 4"),
        CustomSQL(name="links", target_table="fact_relationship", sql="SELECT 5"),
    ]
    steps = compile_spec(spec, S)
    names = labels(steps)
    route = ["route_prepare", "route_condition_occurrence", "route_drug_exposure", "route_measurement"]
    i = names.index("route_prepare")
    assert names[i:i + 4] == route
    # after every writer of the event tables (specimen is the last), before payer_plan_period, cost,
    # the observation periods and the eras
    assert names[:i][-2:] == ["death", "custom_specimens"]
    assert names[i + 4:] == ["custom_plans", "custom_costs", "observation_period", "observation_period_merge",
                             "observation_period_fill", "condition_era", "custom_eras", "custom_links", "cdm_source"]
    by = {s.label: s for s in steps}
    # non-blocking route steps; the per-table steps require route_prepare and return the moves
    assert all(by[n].kind == "route" and not by[n].blocking for n in route)
    assert [by[n].requires for n in route] == [None, "route_prepare", "route_prepare", "route_prepare"]
    assert [by[n].returns for n in route] == ["rowcount", "rows", "rows", "rows"]
    assert [by[n].target_table for n in route] == [None, "condition_occurrence", "drug_exposure", "measurement"]
    assert by["route_prepare"].slot == (14, 2, 0) and len(by["route_prepare"].statements) == 2
    assert "INSERT INTO etl.id_high_water (table_name, max_id)" in by["route_prepare"].sql
    assert "UNION ALL SELECT 'observation', COALESCE(MAX(observation_id), 0) FROM cdm.observation;" in by[
        "route_prepare"].sql
    check_step_invariants(steps)
    # the route destinations are emptied by the build, but are not the spec's own targets
    assert route_sources(spec) == ["condition_occurrence", "drug_exposure", "measurement"]
    assert route_targets(spec) == list(ROUTE_TABLES)
    assert "observation" not in target_tables(spec)
    assert route_only_tables(spec) == {"procedure_occurrence", "device_exposure", "observation"}


def test_route_by_domain_off():
    spec = load_ref().model_copy(update={"route_by_domain": False})
    steps = compile_spec(spec, S)
    assert not any(s.kind == "route" for s in steps)
    assert route_sources(spec) == [] and route_targets(spec) == [] and route_only_tables(spec) == set()
    # without routing a concept_map accepts its target domain only
    tm = spec.mapping_for("measurement")
    fm = next(f for f in tm.fields if f.target_field == "measurement_concept_id")
    assert accepted_domains_for(spec, tm, fm) == ["Measurement"]
    assert vocabulary_domains(spec)["SRC_LAB"].accepted == []


def test_route_steps_after_custom_only_event_sql():
    """An event table loaded only by custom SQL is routed too, after that SQL; the periods derived
    from the records read the route destinations."""
    spec = load_ref().model_copy(update={"observation_period": "from_events"}, deep=True)
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "observation_period"]
    spec.custom_sql = [CustomSQL(name="procedures", target_table="procedure_occurrence", sql="SELECT 1")]
    steps = compile_spec(spec, S)
    names = labels(steps)
    assert names.index("custom_procedures") < names.index("route_prepare")
    assert names[names.index("route_prepare"):names.index("route_prepare") + 5] == [
        "route_prepare", "route_condition_occurrence", "route_procedure_occurrence", "route_drug_exposure",
        "route_measurement"]
    op = step(steps, "observation_period")
    assert names.index("route_measurement") < names.index("observation_period")
    for col in ("observation_date", "device_exposure_start_date", "procedure_date", "condition_start_date"):
        assert col in op.sql, col
    # without routing the derived periods read only what the spec loads
    off = step(compile_spec(spec.model_copy(update={"route_by_domain": False}), S), "observation_period")
    assert "observation_date" not in off.sql and "device_exposure_start_date" not in off.sql
    # the fill reads the route destinations too
    fill = step(compile_spec(load_ref(), S), "observation_period_fill")
    assert "observation_date" in fill.sql and "procedure_date" in fill.sql


def test_pair_fields_correspondence():
    cdm = load_cdm_spec()
    pairs = [(a, b) for a in ROUTE_TABLES for b in ROUTE_TABLES if a != b]
    assert len(pairs) == 30
    for src, dst in pairs:
        p = pair_fields(src, dst)
        written = [f for f, _ in p.values]
        # every required field of the destination is filled (the route step numbers the id)
        required = {f.name for f in cdm.table(dst).fields if f.required} - {EVENT_FIELD_ROLES[dst]["id"]}
        assert required <= set(written), (src, dst, required - set(written))
        assert len(written) == len(set(written)) and EVENT_FIELD_ROLES[dst]["id"] not in written
        assert all(cdm.table(dst).field(f) is not None for f in written)
        values = dict(p.values)
        assert values["person_id"] == "m.person_id" and values["visit_occurrence_id"] == "m.visit_occurrence_id"
        assert values[EVENT_FIELD_ROLES[dst]["concept"]] == f"m.{EVENT_FIELD_ROLES[src]['concept']}"
    # the start date and the type carry over by role
    p = dict(pair_fields("condition_occurrence", "observation").values)
    assert p["observation_date"] == "m.condition_start_date"
    assert p["observation_type_concept_id"] == "m.condition_type_concept_id"
    assert p["observation_source_value"] == "m.condition_source_value"
    # drug_exposure_end_date is required: the record's end, else its start
    assert dict(pair_fields("condition_occurrence", "drug_exposure").values)["drug_exposure_end_date"] == (
        "COALESCE(m.condition_end_date, m.condition_start_date)")
    assert dict(pair_fields("measurement", "drug_exposure").values)["drug_exposure_end_date"] == "m.measurement_date"
    # a quantity becomes an integer only when it is one (NaN fails BETWEEN), else NULL and counted
    guarded = ("CASE WHEN m.quantity = trunc(m.quantity) AND m.quantity BETWEEN -2147483648 AND 2147483647"
               " THEN (m.quantity)::integer END")
    to_proc = pair_fields("drug_exposure", "procedure_occurrence")
    assert dict(to_proc.values)["quantity"] == guarded
    assert f"(m.quantity IS NOT NULL AND {guarded} IS NULL)" in to_proc.lost
    assert dict(pair_fields("procedure_occurrence", "drug_exposure").values)["quantity"] == "m.quantity"
    # the values the destination has no place for are counted as dropped
    to_obs = pair_fields("measurement", "observation")
    assert set(to_obs.lost) == {f"m.{f} IS NOT NULL" for f in (
        "measurement_time", "operator_concept_id", "range_low", "range_high", "unit_source_concept_id")}
    assert dict(to_obs.values)["value_as_number"] == "m.value_as_number"
    assert "m.days_supply IS NOT NULL" in pair_fields("drug_exposure", "observation").lost


def test_join_form_without_concept_index():
    sql = route_sql("measurement")
    assert "WHERE c.concept_id = ANY(ARRAY(SELECT DISTINCT t.measurement_concept_id FROM {cdm}.measurement t" in sql
    assert "t.measurement_concept_id <> 0" in sql  # only the standard concept decides; concept 0 stays
    assert "c.domain_id IN ('Condition', 'Procedure', 'Drug', 'Device', 'Observation')" in sql
    # a measured result goes to observation; the moves are returned per destination and domain
    assert "THEN 'observation'  -- a measured result is never dropped" in sql
    assert "(t.value_as_number IS NOT NULL OR t.value_as_concept_id IS NOT NULL OR t.value_source_value IS NOT NULL)" in sql
    assert "GREATEST((SELECT COALESCE(MAX(observation_id), 0) FROM {cdm}.observation)" in sql
    assert "(SELECT max_id FROM {etl}.id_high_water WHERE table_name = 'observation'))" in sql
    assert sql.rstrip().endswith("FROM moved m\nGROUP BY 1, 2\nORDER BY 1, 2;")
    assert "to_measurement" not in sql
    # the other sources have no value rule
    assert "'observation'  -- a measured result" not in route_sql("condition_occurrence")
    # without an index on concept(concept_id) a lookup of the list reads the concept table per value:
    # the concepts are joined instead
    join = route_sql("measurement", vocab_indexed=False)
    assert "ANY(ARRAY(" not in join
    assert "FROM (SELECT DISTINCT t.measurement_concept_id AS id FROM {cdm}.measurement t" in join
    assert "JOIN {vocab}.concept c ON c.concept_id = d.id" in join
    steps = compile_spec(load_ref(), S, vocab_indexed=False)
    assert all("ANY(ARRAY(" not in s.sql for s in steps if s.kind == "route")
    assert "JOIN vocab.concept c ON c.concept_id = d.id" in step(steps, "route_measurement").sql
    assert all(";" not in strip_literals(x).strip().rstrip(";") for x in route_prepare_sql())  # one statement each


# --------------------------------------------------------------------------------------
# accepted domains (no database)
# --------------------------------------------------------------------------------------


def _concept_field(spec, table: str, field: str, **changes):
    spec = set_field(spec, table, field, **changes)
    tm = spec.mapping_for(table)
    return spec, tm, next(f for f in tm.fields if f.target_field == field)


def test_accepted_domains_per_table():
    assert ACCEPTED_DOMAINS == {
        "condition_occurrence": ("Condition", "Observation", "Procedure", "Measurement"),
        "procedure_occurrence": ("Procedure", "Observation", "Measurement"),
        "measurement": ("Measurement", "Observation"),
        "drug_exposure": ("Drug", "Device"),
        "device_exposure": ("Device", "Drug"),
    }
    spec = load_ref()
    tm = spec.mapping_for("measurement")
    by = {f.target_field: f for f in tm.fields}
    # the main concept field of a routed table: its own domain first, then the others
    assert accepted_domains_for(spec, tm, by["measurement_concept_id"]) == ["Measurement", "Observation"]
    # other concept fields, and observation (not routed), accept their own domain only
    assert accepted_domains_for(spec, tm, by["unit_concept_id"]) == ["Unit"]
    assert accepted_domains_for(spec, tm, by["value_as_concept_id"]) == ["Meas Value"]
    drug = spec.mapping_for("drug_exposure")
    assert accepted_domains_for(spec, drug, next(f for f in drug.fields if f.target_field == "drug_concept_id")) == [
        "Drug", "Device"]
    spec2, cond, fm = _concept_field(spec, "condition_occurrence", "condition_concept_id", kind="concept_map",
                                     vocabulary="SRC_DX", target_domain="condition")
    assert accepted_domains_for(spec2, cond, fm) == ["condition", "Observation", "Procedure", "Measurement"]
    # a source concept field and a target domain the table does not accept: that domain only
    src = fm.model_copy(update={"target_field": "condition_source_concept_id", "concept_role": "source"})
    assert accepted_domains_for(spec2, cond, src) == ["condition"]
    assert accepted_domains_for(spec2, cond, fm.model_copy(update={"target_domain": "Drug"})) == ["Drug"]
    assert accepted_domains_for(spec2, cond, fm.model_copy(update={"target_domain": None})) == []
    # per source vocabulary, as the terms get them
    vd = vocabulary_domains(spec)
    assert (vd["SRC_LAB"].target_domain, vd["SRC_LAB"].accepted) == ("Measurement", ["Measurement", "Observation"])
    assert (vd["SRC_UNIT"].target_domain, vd["SRC_UNIT"].accepted) == ("Unit", [])
    assert (vd["SRC_VA_CAUSE"].target_domain, vd["SRC_VA_CAUSE"].accepted) == (None, [])
    assert all(v.conflict is None for v in vd.values())


def test_union_for_routed_shared_vocab():
    """Fields that share a source vocabulary share one accepted set: the union over routed fields
    (a diagnosis list read into condition and procedure keeps Condition), the intersection over the
    others, and a warning when a field that is not routed can get a concept it does not accept."""
    spec, _, _ = _concept_field(load_ref(), "condition_occurrence", "condition_concept_id", kind="concept_map",
                                expression="d.icd10", vocabulary="SRC_DX", target_domain="Condition")
    spec = set_field(spec, "condition_occurrence", "condition_source_concept_id", kind="concept_map",
                     expression="d.icd10", vocabulary="SRC_DX", target_domain="Condition", concept_role="source")
    proc = spec.mapping_for("condition_occurrence").model_copy(deep=True, update={"target_table": "procedure_occurrence"})
    proc.fields = [f.model_copy(update={"target_field": f.target_field.replace("condition_start_date", "procedure_date")
                                        .replace("condition_", "procedure_")}) for f in proc.fields
                   if f.target_field not in ("condition_start_datetime",)]
    spec.table_mappings.append(proc)
    vd = vocabulary_domains(spec)["SRC_DX"]
    assert vd.target_domain == "Condition"
    assert vd.accepted == ["Condition", "Observation", "Procedure", "Measurement"]  # Condition is kept
    assert vd.conflict is None
    assert vd.fields == ["condition_occurrence.condition_concept_id", "condition_occurrence.condition_source_concept_id",
                         "procedure_occurrence.procedure_concept_id", "procedure_occurrence.procedure_source_concept_id"]
    # a field that is not routed (observation) can get a Condition concept: warned
    obs = spec.mapping_for("condition_occurrence").model_copy(deep=True, update={"target_table": "observation"})
    obs.fields = [FieldMapping(target_field="observation_concept_id", kind="concept_map", expression="d.icd10",
                               vocabulary="SRC_DX", target_domain="Observation")]
    shared = spec.model_copy(update={"table_mappings": spec.table_mappings + [obs]})
    vd = vocabulary_domains(shared)["SRC_DX"]
    assert vd.accepted == ["Condition", "Observation", "Procedure", "Measurement"]
    assert "observation.observation_concept_id (Observation)" in vd.conflict
    assert "can get a concept of a domain it does not accept (Condition, Procedure, Measurement)" in vd.conflict
    # fields that are not routed and share no domain: their union, and a warning
    lab = set_field(load_ref(), "measurement", "unit_concept_id", vocabulary="SRC_LAB")
    vd = vocabulary_domains(lab)["SRC_LAB"]
    assert vd.target_domain == "Measurement" and vd.accepted == ["Measurement", "Observation", "Unit"]
    assert "measurement.unit_concept_id (Unit)" in vd.conflict
    issues = [i.message for i in validate_spec(lab) if "source vocabulary SRC_LAB" in i.message]
    assert len(issues) == 1 and "Give each field a source vocabulary of its own" in issues[0]
    # without routing the routed field accepts its target domain only, so the intersection decides
    off = vocabulary_domains(spec.model_copy(update={"route_by_domain": False}))["SRC_DX"]
    assert (off.target_domain, off.accepted) == ("Condition", [])


def _hit(cid: int, name: str, domain: str, score: float) -> _Hit:
    from omop_agent.models import ConceptCandidate

    return _Hit(ConceptCandidate(concept_id=cid, concept_name=name, domain_id=domain, vocabulary_id="X",
                                 concept_class_id="X", standard_concept="S", concept_code=str(cid), score=score),
                [name])


def test_auto_picks_only_in_preferred_domain():
    """DB1: the exact-match and lexical shortcuts never pick a concept outside the target domain."""
    from omop_agent.models import SourceTerm

    term = SourceTerm(source_vocabulary_id="SRC_DRUG", source_code="Ferrous sulphate", source_name="Ferrous sulphate",
                      frequency=40, target_domain="Drug", accepted_domains=["Drug", "Device"])
    device = _hit(1, "Ferrous sulphate", "Device", 1.0)
    drug = _hit(2, "Ferrous sulfate 200 MG Oral Tablet", "Drug", 0.86)
    assert _exact_match(term, [device, drug]) is None
    lex = _lexical(term, [device, drug], "no LLM")
    assert lex.target_concept_id == 2 and lex.status == "UNCHECKED" and lex.accepted_domains == ["Drug", "Device"]
    assert _lexical(term, [device], "no LLM").target_concept_id == 0
    exact = _exact_match(term, [device, _hit(3, "Ferrous Sulphate", "Drug", 1.0)])
    assert exact is not None and exact.target_concept_id == 3 and exact.status == "APPROVED"
    # a term without a target domain accepts any concept, as before
    anything = term.model_copy(update={"target_domain": None, "accepted_domains": []})
    assert _exact_match(anything, [device]).target_concept_id == 1
    # the prompt says which domains are accepted and shows each candidate's domain
    lines = "\n".join(_format_term(1, term, [device, drug], 10, None))
    assert ("target domain: Drug (also accepted: Device; the record is moved to the table of the chosen concept's "
            "domain)") in lines
    assert "1 | Ferrous sulphate | Device | X | X | 1.00" in lines
    single = "\n".join(_format_term(1, anything.model_copy(update={"target_domain": "Drug"}), [drug], 10, None))
    assert "target domain: Drug\n" in single and "candidates (concept_id | name | class | vocabulary | score)" in single
    # the checks of a chosen id accept the term's domains
    row = {"concept_id": 1, "concept_name": "Ferrous sulphate", "domain_id": "Device", "vocabulary_id": "X",
           "concept_class_id": "X", "standard_concept": "S", "concept_code": "1", "invalid_reason": None}
    assert _validate_row(row, 1, ["Drug", "Device"])[0] is not None
    assert _validate_row(row, 1, "Drug")[1] == "concept_id 1 (Ferrous sulphate) is in domain Device, not Drug"
    assert _validate_row(row, 1, ["Condition", "Observation", "Procedure"])[1].endswith(
        "not Condition, Observation or Procedure")


def test_off_domain_llm_pick_is_never_approved(tmp_path):
    from omop_agent.models import SourceTerm, TermChoice
    from omop_agent.vocab.mapper import _VocabMapper

    vm = _VocabMapper(None, None, plain_config(tmp_path), "SRC_DX")  # type: ignore[arg-type]
    term = SourceTerm(source_vocabulary_id="SRC_DX", source_code="Z34.9", source_name="Z34.9", frequency=40,
                      target_domain="Condition", accepted_domains=["Condition", "Observation", "Procedure", "Measurement"])
    vm.hits["Z34.9"] = [_hit(10, "Routine antenatal care", "Observation", 0.9), _hit(11, "Pregnancy", "Condition", 0.5)]
    pick = TermChoice(source_code="Z34.9", chosen_concept_id=10, equivalence="EQUIVALENT", confidence=0.99,
                      rationale="An antenatal care visit.")
    d = vm._apply(term, pick)
    assert (d.target_concept_id, d.target_domain_id, d.status) == (10, "Observation", "UNCHECKED")
    assert "A Observation concept, not Condition: the record moves to the table of the Observation domain" in d.rationale
    d = vm._apply(term, pick.model_copy(update={"chosen_concept_id": 11}))
    assert (d.target_concept_id, d.status) == (11, "APPROVED")


def test_other_domains_never_take_the_target_domains_places():
    """DB1: the best candidates of the other accepted domains are shown on top of the target
    domain's ``k`` (prompt) and DECISION_CANDIDATES (reviewer), never instead of them."""
    from omop_agent.models import SourceTerm
    from omop_agent.vocab.mapper import _decision

    term = SourceTerm(source_vocabulary_id="SRC_DX", source_code="HTN", source_name="hypertension", frequency=40,
                      target_domain="Condition", accepted_domains=["Condition", "Observation", "Procedure", "Measurement"])
    cond = [_hit(100 + i, f"Hypertension {i}", "Condition", round(0.95 - 0.03 * i, 2)) for i in range(10)]
    other = [_hit(200 + 10 * j + i, f"{d} {i}", d, round(0.90 - 0.01 * i, 2))
             for j, d in enumerate(("Observation", "Procedure", "Measurement")) for i in range(3)]
    hits = sorted(cond + other, key=lambda h: (-h.candidate.score, h.candidate.domain_id != "Condition",
                                               h.candidate.concept_id))
    lines = "\n".join(_format_term(1, term, hits, 10, None))
    assert all(f"{h.candidate.concept_id} | " in lines for h in cond + other)
    d = _decision(term, hits, status="UNCHECKED", method="llm")
    assert [c.domain_id for c in d.candidates].count("Condition") == 5 and len(d.candidates) == 14
    # with a retry, more of each domain are found: still k of the target domain and 3 of each other
    more = hits + [_hit(300 + i, f"Observation x{i}", "Observation", 0.5) for i in range(4)]
    assert len([x for x in _format_term(1, term, more, 10, None) if "| Observation |" in x]) == 3


def test_vocabulary_shared_by_tables_of_different_domains(tmp_path):
    """One vocabulary read by the main concept fields of two routed tables of different domains
    (an events table split by row type): every code would be searched in the first field's domain
    and its picks moved silently, so the spec is warned about and no concept is approved without a
    reviewer."""
    from omop_agent.models import SourceTerm, TermChoice
    from omop_agent.vocab.mapper import _VocabMapper

    spec, _, _ = _concept_field(load_ref(), "condition_occurrence", "condition_concept_id", kind="concept_map",
                                expression="d.icd10", vocabulary="SRC_LAB", target_domain="Condition")
    vd = vocabulary_domains(spec)["SRC_LAB"]
    assert vd.target_domain == "Condition" and not vd.auto_approve
    assert ("measurement.measurement_concept_id: Measurement" in vd.conflict
            and "no concept is approved for them without a reviewer" in vd.conflict)
    assert [i for i in validate_spec(spec) if "expect concepts of different domains" in i.message]
    assert vocabulary_domains(load_ref())["SRC_LAB"].auto_approve
    term = SourceTerm(source_vocabulary_id="SRC_LAB", source_code="HB", source_name="Haemoglobin", frequency=40,
                      target_domain="Condition", accepted_domains=vd.accepted, auto_approve=False)
    exact = _exact_match(term, [_hit(1, "Haemoglobin", "Condition", 1.0)])
    assert (exact.target_concept_id, exact.status) == (1, "UNCHECKED") and "different domains" in exact.rationale
    vm = _VocabMapper(None, None, plain_config(tmp_path), "SRC_LAB")  # type: ignore[arg-type]
    vm.hits["HB"] = [_hit(1, "Haemoglobin", "Condition", 1.0)]
    d = vm._apply(term, TermChoice(source_code="HB", chosen_concept_id=1, equivalence="EQUAL", confidence=0.99,
                                   rationale="Same name."))
    assert (d.target_concept_id, d.status) == (1, "UNCHECKED")


def test_import_review_refreshes_domains_first(tmp_path, monkeypatch):
    """DA9: a reviewer's concept is checked against the domains the current spec accepts."""
    ws = Workspace(tmp_path / "work")
    d = MappingDecision(source_vocabulary_id="SRC_LAB", source_code="MRDT", source_name="MRDT", frequency=50,
                        target_domain="Measurement", target_concept_id=0)
    review.export_review(ws, [d])
    review.update_review_row(ws, "SRC_LAB", "MRDT", {"target_concept_id": "77"})
    seen: list = []

    def check(conn, schemas, concept_id, domain):
        seen.append(domain)
        from omop_agent.models import ConceptCandidate

        return ConceptCandidate(concept_id=concept_id, concept_name="Malaria test", domain_id="Observation",
                                vocabulary_id="X", concept_class_id="X", standard_concept="S", concept_code="x"), None

    monkeypatch.setattr(review, "check_concept", check)
    out = review.import_review(ws, object(), S, domains=vocabulary_domains(load_ref()))  # type: ignore[arg-type]
    assert seen == [["Measurement", "Observation"]]
    assert out[0].accepted_domains == ["Measurement", "Observation"] and out[0].target_concept_id == 77
    # without the spec's domains, the decision's own (here its target domain only)
    seen.clear()
    review.import_review(ws, object(), S)  # type: ignore[arg-type]
    assert seen == [["Measurement"]]


def test_reapplied_review_edits_use_the_current_spec(tmp_path, monkeypatch):
    """DA9: ``apply_reviewed_mappings`` hands import_review the domains of the spec as it is now."""
    import shutil

    from omop_agent.pipeline import Pipeline
    from omop_agent.vocab import mapper
    from tests.test_etl import FIXTURES

    cfg = plain_config(tmp_path)
    p = Pipeline(cfg, log=lambda _: None)
    shutil.copy(FIXTURES / "demo_etl_spec.yaml", p.ws.spec_yaml)
    seen: list = []

    def fake_import(ws, conn, schemas, *, domains=None):
        seen.append(domains)
        return []

    class Conn:
        closed = False

        def commit(self):
            pass

    monkeypatch.setattr(review, "import_review", fake_import)
    monkeypatch.setattr(mapper, "write_source_to_concept_map", lambda *a, **k: 0)
    p._conn = Conn()  # type: ignore[assignment]
    assert p.apply_reviewed_mappings() == 0
    assert seen[0]["SRC_LAB"].accepted == ["Measurement", "Observation"]


# --------------------------------------------------------------------------------------
# prompts, validator, repair agent (no database)
# --------------------------------------------------------------------------------------


def test_prompts_explain_routing():
    assert "the build then moves the record to that domain's table" in DESIGN_SYSTEM
    assert "Only the standard concept decides: unmapped records (concept 0) stay where you load them" in DESIGN_SYSTEM
    assert "concepts of other domains (values, units, types) are never moved" in DESIGN_SYSTEM
    assert "never split a source table by the domain of its codes yourself" in DESIGN_SYSTEM
    assert "never by the domain of its\ncode" not in STAGING_CONVENTIONS
    assert "never by the domain of its code: the build moves records by domain" in " ".join(STAGING_CONVENTIONS.split())
    assert "Do not plan mappings that only split a source table by the domain of its codes" in " ".join(
        _plan_prompt("p", "a").split())
    text = " ".join(REPAIR_SYSTEM.split())
    assert "Never route records by hand" in text and "route_* steps, etlDomainRouting" in text
    assert "route the observation-domain codes" not in (data_loss_issues.__doc__ or "")
    md = render_spec_md(load_ref())
    assert "**Routing by domain:** after loading" in md
    assert "**Routing by domain:** off" in render_spec_md(load_ref().model_copy(update={"route_by_domain": False}))


HAND_ROUTED_WHERE = ("EXISTS (SELECT 1 FROM {vocab}.concept c JOIN {vocab}.concept_relationship r ON "
                     "r.concept_id_1 = c.concept_id JOIN {vocab}.concept t ON t.concept_id = r.concept_id_2 "
                     "WHERE c.concept_code = d.icd10 AND t.domain_id = 'Observation')")


def _hand_routed_observation(spec):
    cond = spec.mapping_for("condition_occurrence")
    obs = cond.model_copy(deep=True, update={"target_table": "observation", "where": HAND_ROUTED_WHERE})
    names = {"condition_occurrence_id": "observation_id", "condition_concept_id": "observation_concept_id",
             "condition_start_date": "observation_date", "condition_start_datetime": "observation_datetime",
             "condition_type_concept_id": "observation_type_concept_id",
             "condition_source_value": "observation_source_value",
             "condition_source_concept_id": "observation_source_concept_id"}
    obs.fields = [f.model_copy(update={"target_field": names.get(f.target_field, f.target_field)}) for f in obs.fields]
    return obs


def test_validator_warns_about_hand_routing():
    spec = fixed_ref()
    obs = _hand_routed_observation(spec)
    spec = spec.model_copy(update={"table_mappings": spec.table_mappings + [obs]})
    issues = validate_spec(spec)
    assert not [i for i in issues if i.severity == "error"], [str(i) for i in issues if i.severity == "error"]
    routing = [i for i in issues if "routing by hand loads such records twice" in i.message]
    assert [(i.severity, i.target_table) for i in routing] == [("warning", "observation")]
    twice = [i for i in issues if "read the same codes" in i.message]
    assert [(i.severity, i.target_table) for i in twice] == [("warning", "observation")]
    assert "the mappings from diagnoses into condition_occurrence and observation read the same codes (d.icd10)" in (
        twice[0].message)
    # without routing by the build, hand-made routing is the way to do it
    off = validate_spec(spec.model_copy(update={"route_by_domain": False}))
    assert not [i for i in off if "routing by hand" in i.message or "read the same codes" in i.message]
    # the repair agent may not add either
    added = routing_issues(None, obs, fixed_ref(), spec)
    assert len(added) == 2 and "observation: it filters records by the domain of their concept" in added[0]
    same = obs.model_copy(update={"where": None})
    assert routing_issues(None, same, fixed_ref(), fixed_ref().model_copy(
        update={"table_mappings": fixed_ref().table_mappings + [same]}))[0].startswith("the mappings from diagnoses")
    # an edit of a mapping that routed by hand before is not refused for it
    assert routing_issues(obs, obs, spec, spec) == []


def _with(spec, *added):
    return spec.model_copy(update={"table_mappings": spec.table_mappings + list(added)})


def test_lookups_and_disjoint_splits_are_not_hand_routing():
    """A LEFT join's domain test only blanks a lookup, a domain_id test against a domain without an
    event table routes nothing, and two mappings that split one source table by disjoint filters
    load each row once: none of them is routing by hand, and the repair agent may make them."""
    from omop_agent.models import JoinSpec

    base = fixed_ref()
    meas = base.mapping_for("measurement")
    unit = JoinSpec(table="{vocab}.concept", alias="u", join_type="left",
                    on="u.concept_code = r.unit AND u.vocabulary_id = 'UCUM' AND u.domain_id = 'Unit'")
    looked_up = meas.model_copy(deep=True, update={"joins": [*meas.joins, unit]})
    spec = base.model_copy(update={"table_mappings": [looked_up if tm is meas else tm for tm in base.table_mappings]})
    assert not [i for i in validate_spec(spec) if "routing by hand" in i.message]
    assert routing_issues(meas, looked_up, base, spec) == []
    # an inner join on the domain of an event concept, or a WHERE on a LEFT-joined concept's domain, is
    inner = unit.model_copy(update={"join_type": "inner", "on": "u.concept_code = r.test AND u.domain_id = 'Observation'"})
    assert routing_issues(meas, meas.model_copy(update={"joins": [*meas.joins, inner]}), base, base)
    where = meas.model_copy(update={"joins": [*meas.joins, unit.model_copy(update={"on": "u.concept_code = r.test"})],
                                    "where": "u.domain_id <> 'Observation'"})
    assert routing_issues(meas, where, base, base)
    # numeric results into measurement, text answers into observation: disjoint filters
    numeric = meas.model_copy(update={"where": "r.result ~ '^[0-9.]+$'"})
    names = {"measurement_id": "observation_id", "measurement_concept_id": "observation_concept_id",
             "measurement_date": "observation_date", "measurement_type_concept_id": "observation_type_concept_id",
             "measurement_source_value": "observation_source_value"}
    text = meas.model_copy(deep=True, update={"target_table": "observation", "where": "NOT (r.result ~ '^[0-9.]+$')"})
    text.fields = [f.model_copy(update={"target_field": names.get(f.target_field, f.target_field)})
                   for f in text.fields if f.target_field != "value_as_number"]
    split = base.model_copy(update={"table_mappings": [numeric if tm is meas else tm for tm in base.table_mappings]})
    assert not [i for i in validate_spec(_with(split, text)) if "read the same codes" in i.message]
    assert routing_issues(None, text, split, _with(split, text)) == []
    # the same filter, or none on one side, selects the same rows: loaded twice
    same = text.model_copy(update={"where": numeric.where})
    assert routing_issues(None, same, split, _with(split, same))[0].startswith("the mappings from lab_results")


def test_the_same_fix_to_both_mappings_of_a_pair_is_not_refused():
    """A spec that already loads one source table's codes into two event tables (written before the
    build routed): the repair agent may make one fix to both mappings, one after the other."""
    base = fixed_ref()
    cond = base.mapping_for("condition_occurrence")
    obs = _hand_routed_observation(base).model_copy(update={"where": None})
    spec = _with(base, obs)
    dotted = "CASE WHEN d.icd10 ~ '^[A-Z][0-9]{3}$' THEN substr(d.icd10, 1, 3) || '.' || substr(d.icd10, 4) ELSE d.icd10 END"

    def fixed(tm, field):
        return tm.model_copy(deep=True, update={"fields": [
            f.model_copy(update={"expression": dotted}) if f.target_field == field else f for f in tm.fields]})

    cond2 = fixed(cond, "condition_concept_id")
    step1 = spec.model_copy(update={"table_mappings": [cond2 if tm is cond else tm for tm in spec.table_mappings]})
    assert routing_issues(cond, cond2, spec, step1) == []
    obs2 = fixed(obs, "observation_concept_id")
    step2 = step1.model_copy(update={"table_mappings": step1.table_mappings[:-1] + [obs2]})
    assert routing_issues(obs, obs2, step1, step2) == []
    # the pair is still reported by the validator and by every build
    assert [i for i in validate_spec(step2) if "read the same codes" in i.message]


def test_hand_routing_is_reported_by_the_build():
    """A spec that routed by hand before the build routed by domain: each build reports the mapping
    (etlHandRouting, convention, failed, not repairable) instead of only a log line."""
    from omop_agent.quality.checks import _hand_routing_checks

    base = fixed_ref()
    obs = _hand_routed_observation(base)
    spec = _with(base, obs)
    run = ETLRunResult(started_at=NOW, steps=[
        TableLoadResult(target_table="condition_occurrence", step="07_condition_occurrence", kind="insert",
                        rows_inserted=40),
        TableLoadResult(target_table="observation", step="09_observation", kind="insert", rows_inserted=12)])
    [r] = _hand_routing_checks(spec, run)
    assert (r.key, r.severity, r.failed, r.repairable, r.num_violated_rows) == (
        "etlHandRouting.observation", "convention", True, False, 12)
    assert "routes records by the domain of their concept by hand" in r.description
    assert _hand_routing_checks(spec.model_copy(update={"route_by_domain": False}), run) == []


# --------------------------------------------------------------------------------------
# checks, build history, report (no database)
# --------------------------------------------------------------------------------------


def route_result(table: str, moved: list[tuple[str, str, int, int]], **kw) -> TableLoadResult:
    return TableLoadResult(
        target_table=table, step=f"12_route_{table}", kind="route", blocking=False,
        moved=[RoutedRows(from_table=table, to_table=t, domain_id=d, moved_rows=n, rows_with_dropped_fields=x)
               for t, d, n, x in moved], **kw)


def test_domain_routing_check_and_report(tmp_path):
    loaded = TableLoadResult(target_table="measurement", step="09_measurement", kind="insert", rows_inserted=200,
                             source_rows=200, retained_rows=200, id_retained_rows=200)
    routed = route_result("measurement", [("condition_occurrence", "Condition", 3, 0), ("observation", "Observation", 9, 2)])
    nothing = route_result("drug_exposure", [])
    failed = route_result("condition_occurrence", [], error="relation \"concept\" does not exist")
    # route_prepare's count is the bookkeeping of {etl}.id_high_water, not CDM rows
    prepare = TableLoadResult(target_table="route_prepare", step="11_route_prepare", kind="route", blocking=False,
                              rows_inserted=6)
    run = ETLRunResult(started_at=NOW, steps=[loaded, prepare, routed, nothing, failed])
    checks = {r.key: r for r in etl_checks(run)}
    r = checks["etlDomainRouting.measurement.route_measurement"]
    assert (r.severity, r.failed, r.repairable, r.num_violated_rows, r.num_denominator_rows) == (
        "characterization", False, False, 12, 200)
    assert "12 of 200 rows (3 to condition_occurrence (Condition); 9 to observation (Observation))" in r.description
    assert "2 of them had a value the destination table has no place for" in r.description
    assert not [k for k in checks if k.startswith("etlDomainRouting.drug_exposure")]  # nothing moved
    # a failed route step: a convention-level step error, never sent to the repair agent
    err = checks["etlStepError.condition_occurrence.route_condition_occurrence"]
    assert (err.severity, err.repairable) == ("convention", False)
    assert run.ok and run.rows_moved() == 12 and run.rows_loaded() == 200
    assert repair_triggers(None, run, plain_config(tmp_path)) == ([], [])
    # the report: route steps show what they moved, and a table lists the moves
    etl = _etl_section(run)
    assert [s["moved"] for s in etl["steps"]] == ["", "", "moved 12", "moved 0", ""]
    assert [s["rows"] for s in etl["steps"]] == [200, None, None, None, None]
    assert etl["moved"] == [
        {"from_table": "measurement", "to_table": "condition_occurrence", "domain_id": "Condition", "rows": "3",
         "dropped": "0"},
        {"from_table": "measurement", "to_table": "observation", "domain_id": "Observation", "rows": "9",
         "dropped": "2"}]
    assert etl["moved_rows"] == "12"
    shareable = _etl_section(run, min_cell_count=5, shareable=True)
    assert [m["rows"] for m in shareable["moved"]] == ["<5", "9"] and shareable["moved"][1]["dropped"] == "<5"


def test_route_only_table_not_repaired():
    """DB3: measurePersonCompleteness of a table that only receives moved records is not repairable,
    and a repair after which it receives none is not a loss of data."""
    sev = {"fatal", "convention"}
    completeness = CheckResult(check_name="measurePersonCompleteness", check_level="TABLE", table="observation",
                               description="d", severity="convention", failed=True, num_violated_rows=99,
                               num_denominator_rows=100, pct_violated_rows=0.99, threshold_pct=95)
    from omop_agent.quality.checks import _route_only_note

    noted = _route_only_note(completeness, route_only_tables(load_ref()))
    assert noted.repairable is False and "No mapping loads observation" in noted.description
    assert not repairable(noted, sev)
    kept = _route_only_note(completeness.model_copy(update={"table": "measurement"}), route_only_tables(load_ref()))
    assert kept.repairable is None and repairable(kept, sev)

    insert = TableLoadResult(target_table="condition_occurrence", step="07_condition_occurrence", kind="insert",
                             rows_inserted=100)
    moved = ETLRunResult(started_at=NOW, steps=[insert, route_result("condition_occurrence", [
        ("observation", "Observation", 7, 0)])])
    none = ETLRunResult(started_at=NOW, steps=[insert, route_result("condition_occurrence", [])])
    applies = CheckResult(check_name="isRequired", check_level="FIELD", table="observation",
                          field="observation_date", description="d")
    report = DQReport(run_at=NOW, results=[applies], passed=True)
    blanked = DQReport(run_at=NOW, results=[applies.model_copy(update={"not_applicable": True})], passed=True)
    h = _BuildHistory()
    h.add(moved, report)
    assert h.route_only == {"observation"}
    # moves change no row total, and the route-only table emptied by a repair is not a loss
    assert _BuildHistory._rows(moved) == _BuildHistory._rows(none) == {"condition_occurrence": 100}
    assert h.losses(none, blanked) == []
    # a table that a mapping loads is still a loss when its checks stop applying
    other = applies.model_copy(update={"table": "condition_occurrence", "field": "condition_start_date"})
    h.add(moved, DQReport(run_at=NOW, results=[other], passed=True))
    assert h.losses(none, DQReport(run_at=NOW, results=[other.model_copy(update={"not_applicable": True})],
                                   passed=True))


# --------------------------------------------------------------------------------------
# the route steps on PostgreSQL
# --------------------------------------------------------------------------------------

# concept id -> (name, domain)
CONCEPTS = {
    101: ("Malaria", "Condition"),
    102: ("Routine antenatal care", "Observation"),
    103: ("Appendectomy", "Procedure"),
    104: ("Haemoglobin", "Measurement"),
    105: ("Ferrous sulfate 200 MG Oral Tablet", "Drug"),
    106: ("Bandage", "Device"),
    107: ("Positive", "Meas Value"),
    108: ("gram per deciliter", "Unit"),
}


@pytest.fixture
def cdm(pg):
    """Empty CDM, ETL and vocabulary tables with a few concepts of every domain (no index on
    concept(concept_id): the route steps join the concept table)."""
    conn, schemas = pg
    create_cdm_tables(conn, schemas, include_vocab=True)
    create_etl_tables(conn, schemas)
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {schemas.vocab}.concept (concept_id, concept_name, domain_id, vocabulary_id, "
            "concept_class_id, standard_concept, concept_code, valid_start_date, valid_end_date) "
            "VALUES (%s, %s, %s, 'X', 'X', 'S', %s, DATE '1970-01-01', DATE '2099-12-31')",
            [(cid, name, domain, str(cid)) for cid, (name, domain) in CONCEPTS.items()],
        )
    conn.commit()
    return conn, schemas


def route_steps(schemas: Schemas, sources=ROUTE_SOURCES, indexed: bool = True) -> list[CompiledStep]:
    prep = [schemas.render(x) for x in route_prepare_sql()]
    out = [CompiledStep("01_route_prepare", None, "route", "\n".join(prep), blocking=False, statements=tuple(prep))]
    for i, src in enumerate(sources, start=2):
        out.append(CompiledStep(f"{i:02d}_route_{src}", src, "route", schemas.render(route_sql(src, indexed)),
                                blocking=False, requires="route_prepare", returns="rows"))
    return out


def run_routes(conn, schemas, sources=ROUTE_SOURCES, indexed: bool = True) -> dict[str, TableLoadResult]:
    out = {}
    for s in route_steps(schemas, sources, indexed):
        res = _run_step(conn, s, None)
        conn.commit()
        assert res.error is None, (s.name, res.error)
        out[s.label] = res
    return out


def insert(conn, schemas, table: str, rows: list[dict]) -> None:
    for r in rows:
        cols = ", ".join(r)
        execute(conn, f"INSERT INTO {schemas.cdm}.{table} ({cols}) VALUES ({', '.join(['%s'] * len(r))})",
                tuple(r.values()))
    conn.commit()


def cond(i: int, concept: int, person: int = 1, **kw) -> dict:
    return {"condition_occurrence_id": i, "person_id": person, "condition_concept_id": concept,
            "condition_start_date": date(2020, 1, i % 28 + 1), "condition_type_concept_id": 32817,
            "condition_source_value": f"C{i}", **kw}


def moves(res: TableLoadResult) -> list[tuple]:
    return [(m.to_table, m.domain_id, m.moved_rows, m.rows_with_dropped_fields) for m in res.moved]


def all_rows(conn, schemas) -> int:
    return sum(int(scalar(conn, f"SELECT COUNT(*) FROM {schemas.cdm}.{t}")) for t in ROUTE_TABLES)


def misplaced(conn, schemas, table: str) -> int:
    """fkDomain: records of ``table`` whose concept is in another event domain."""
    concept = EVENT_FIELD_ROLES[table]["concept"]
    domain = {"condition_occurrence": "Condition", "procedure_occurrence": "Procedure", "drug_exposure": "Drug",
              "device_exposure": "Device", "measurement": "Measurement"}[table]
    return int(scalar(conn, f"""SELECT COUNT(*) FROM {schemas.cdm}.{table} t JOIN {schemas.vocab}.concept c
        ON c.concept_id = t.{concept} WHERE c.domain_id <> '{domain}' AND c.domain_id IN
        ('Condition', 'Procedure', 'Drug', 'Device', 'Measurement', 'Observation')"""))


@pytest.mark.postgres
@pytest.mark.parametrize("indexed", [True, False])
def test_routing_moves_records(cdm, indexed):
    conn, schemas = cdm
    assert vocab_indexed(conn, schemas) is False  # no index on concept(concept_id) in this fixture
    insert(conn, schemas, "condition_occurrence", [
        cond(1, 101), cond(2, 102, person=2, visit_occurrence_id=7, condition_source_concept_id=55,
                           condition_start_datetime=datetime(2020, 1, 3, 10, 30)),
        cond(3, 103, person=3, condition_end_date=date(2020, 2, 1)), cond(4, 104), cond(5, 0),
        cond(6, 101, person=4, condition_source_value="O80"), cond(7, 102, person=4, condition_source_value="O80"),
        cond(8, 105), cond(9, 106, condition_status_concept_id=32902),
    ])
    insert(conn, schemas, "observation", [{"observation_id": 1, "person_id": 9, "observation_concept_id": 102,
                                           "observation_date": date(2020, 1, 1), "observation_type_concept_id": 32817}])
    before = all_rows(conn, schemas)
    res = run_routes(conn, schemas, ["condition_occurrence"], indexed)
    assert res["route_prepare"].error is None
    assert fetch_all(conn, f"SELECT table_name, max_id FROM {schemas.etl}.id_high_water ORDER BY 1") == [
        {"table_name": t, "max_id": {"condition_occurrence": 9, "observation": 1}.get(t, 0)} for t in sorted(ROUTE_TABLES)]
    # one row per destination and domain; a field without a counterpart (condition_status) is dropped
    assert moves(res["route_condition_occurrence"]) == [
        ("device_exposure", "Device", 1, 1), ("drug_exposure", "Drug", 1, 0), ("measurement", "Measurement", 1, 0),
        ("observation", "Observation", 2, 0), ("procedure_occurrence", "Procedure", 1, 0)]
    assert all_rows(conn, schemas) == before  # nothing lost, nothing added
    assert misplaced(conn, schemas, "condition_occurrence") == 0
    # concept 0 and the concepts of the condition domain stay; a code with two targets in two domains
    # ends up as one row in each table
    assert fetch_all(conn, f"SELECT condition_occurrence_id AS id FROM {schemas.cdm}.condition_occurrence ORDER BY 1") == [
        {"id": 1}, {"id": 5}, {"id": 6}]
    obs = fetch_all(conn, f"SELECT * FROM {schemas.cdm}.observation ORDER BY observation_id")
    assert [(o["observation_id"], o["person_id"], o["observation_source_value"]) for o in obs] == [
        (1, 9, None), (2, 2, "C2"), (3, 4, "O80")]  # ids after the largest, in the order of the moved ids
    o = obs[1]
    assert (o["observation_concept_id"], o["observation_date"], o["observation_datetime"], o["observation_type_concept_id"],
            o["visit_occurrence_id"], o["observation_source_concept_id"]) == (
        102, date(2020, 1, 3), datetime(2020, 1, 3, 10, 30), 32817, 7, 55)
    proc = fetch_all(conn, f"SELECT * FROM {schemas.cdm}.procedure_occurrence")
    assert [(p["procedure_occurrence_id"], p["procedure_concept_id"], p["procedure_date"], p["procedure_end_date"])
            for p in proc] == [(1, 103, date(2020, 1, 4), date(2020, 2, 1))]
    drug = fetch_all(conn, f"SELECT * FROM {schemas.cdm}.drug_exposure")
    assert [(d["drug_concept_id"], d["drug_exposure_start_date"], d["drug_exposure_end_date"]) for d in drug] == [
        (105, date(2020, 1, 9), date(2020, 1, 9))]  # a required end: the start
    assert count(conn, schemas, "device_exposure", "device_concept_id = 106") == 1
    assert count(conn, schemas, "measurement", "measurement_concept_id = 104") == 1
    # a second run moves nothing
    again = run_routes(conn, schemas, ["condition_occurrence"], indexed)
    assert moves(again["route_condition_occurrence"]) == []


@pytest.mark.postgres
def test_non_event_domain_stays(cdm, tmp_path):
    """DA3: a concept of a domain without an event table (Meas Value, Unit) is a mapping error: the
    record stays, so fkDomain keeps reporting it."""
    conn, schemas = cdm
    insert(conn, schemas, "condition_occurrence", [cond(1, 107), cond(2, 108), cond(3, 101)])
    res = run_routes(conn, schemas, ["condition_occurrence"])
    assert moves(res["route_condition_occurrence"]) == []
    assert count(conn, schemas, "condition_occurrence") == 3
    assert count(conn, schemas, "observation") == 0
    report = run_builtin_checks(conn, schemas, plain_config(tmp_path))
    fk = {r.key: r for r in report.results}["fkDomain.condition_occurrence.condition_concept_id"]
    assert (fk.failed, fk.num_violated_rows, fk.repairable) == (True, 2, None)  # a lookup to fix
    # when the route step failed, the misplaced records are the build's, not the spec's: not repairable
    failed = ETLRunResult(started_at=NOW, steps=[
        route_result("condition_occurrence", [], error="boom")])
    after = run_builtin_checks(conn, schemas, plain_config(tmp_path), etl_result=failed)
    results = {r.key: r for r in after.results}
    fk = results["fkDomain.condition_occurrence.condition_concept_id"]
    assert (fk.failed, fk.repairable) == (True, False) and "step '12_route_condition_occurrence' failed" in fk.description
    err = results["etlStepError.condition_occurrence.route_condition_occurrence"]
    assert err.description.startswith("ETL step '12_route_condition_occurrence' moving condition_occurrence records to "
                                      "the table of their concept's domain failed; they stay in condition_occurrence")


@pytest.mark.postgres
def test_no_id_reuse_across_route_steps(cdm):
    """DA4: condition 15 moves to observation, then a drug record moves to condition: it gets 16,
    not the 15 the first move freed."""
    conn, schemas = cdm
    insert(conn, schemas, "condition_occurrence", [cond(i, 101) for i in range(1, 15)] + [cond(15, 102)])
    insert(conn, schemas, "drug_exposure", [
        {"drug_exposure_id": 1, "person_id": 3, "drug_concept_id": 101, "drug_exposure_start_date": date(2020, 5, 1),
         "drug_exposure_end_date": date(2020, 5, 3), "drug_type_concept_id": 32817, "days_supply": 3}])
    res = run_routes(conn, schemas)
    assert moves(res["route_condition_occurrence"]) == [("observation", "Observation", 1, 0)]
    assert moves(res["route_drug_exposure"]) == [("condition_occurrence", "Condition", 1, 1)]  # days_supply dropped
    ids = [r["condition_occurrence_id"] for r in fetch_all(
        conn, f"SELECT condition_occurrence_id FROM {schemas.cdm}.condition_occurrence ORDER BY 1")]
    assert ids == list(range(1, 15)) + [16]
    moved = fetch_all(conn, f"SELECT * FROM {schemas.cdm}.condition_occurrence WHERE condition_occurrence_id = 16")[0]
    assert (moved["person_id"], moved["condition_start_date"], moved["condition_end_date"]) == (
        3, date(2020, 5, 1), date(2020, 5, 3))


@pytest.mark.postgres
def test_nan_and_overflow_quantity(cdm):
    """DA6: a quantity that is not an integer (a fraction, NaN, out of range) becomes NULL in the
    integer quantity of procedure_occurrence and is counted; the step does not fail."""
    conn, schemas = cdm
    rows = []
    for i, q in enumerate(["3", "2.5", "NaN", "1e12", None], start=1):
        rows.append({"drug_exposure_id": i, "person_id": 1, "drug_concept_id": 103,
                     "drug_exposure_start_date": date(2020, 1, i), "drug_exposure_end_date": date(2020, 1, i),
                     "drug_type_concept_id": 32817, "quantity": q})
    for r in rows:
        q = r.pop("quantity")
        insert(conn, schemas, "drug_exposure", [r])
        execute(conn, f"UPDATE {schemas.cdm}.drug_exposure SET quantity = %s::numeric WHERE drug_exposure_id = %s",
                (q, r["drug_exposure_id"]))
    conn.commit()
    res = run_routes(conn, schemas, ["drug_exposure"])
    assert moves(res["route_drug_exposure"]) == [("procedure_occurrence", "Procedure", 5, 3)]
    got = fetch_all(conn, f"SELECT procedure_date AS d, quantity AS q FROM {schemas.cdm}.procedure_occurrence ORDER BY 1")
    assert [g["q"] for g in got] == [3, None, None, None, None]


@pytest.mark.postgres
def test_end_on_the_start_is_not_a_lost_value(cdm):
    """drug_exposure_end_date is required (usually the start date when the source has no end): a
    drug record moved to observation, which has no end, loses a value only when its end is another day."""
    conn, schemas = cdm
    insert(conn, schemas, "drug_exposure", [
        {"drug_exposure_id": i, "person_id": 1, "drug_concept_id": 102, "drug_exposure_start_date": date(2020, 1, 1),
         "drug_exposure_end_date": end, "drug_type_concept_id": 32817}
        for i, end in ((1, date(2020, 1, 1)), (2, date(2020, 1, 30)))])
    res = run_routes(conn, schemas, ["drug_exposure"])
    assert moves(res["route_drug_exposure"]) == [("observation", "Observation", 2, 1)]


@pytest.mark.postgres
def test_measurement_result_never_dropped(cdm):
    conn, schemas = cdm

    def meas(i: int, concept: int, **kw) -> dict:
        return {"measurement_id": i, "person_id": 1, "measurement_concept_id": concept,
                "measurement_date": date(2020, 3, i), "measurement_type_concept_id": 32817, **kw}

    insert(conn, schemas, "measurement", [
        meas(1, 101, value_as_number=7.5, unit_concept_id=108, unit_source_value="g/dL", range_low=5),
        meas(2, 101),  # a condition without a result: condition_occurrence
        meas(3, 102, value_as_concept_id=107, value_source_value="POS"),
        meas(4, 103, value_source_value="done"),
        meas(5, 107, value_as_number=1),  # Meas Value: stays
        meas(6, 104, value_as_number=12.1),  # its own domain: stays
    ])
    res = run_routes(conn, schemas, ["measurement"])
    assert moves(res["route_measurement"]) == [
        ("condition_occurrence", "Condition", 1, 0), ("observation", "Condition", 1, 1),
        ("observation", "Observation", 1, 0), ("observation", "Procedure", 1, 0)]
    obs = fetch_all(conn, f"""SELECT observation_concept_id AS c, value_as_number AS n, value_as_concept_id AS vc,
                                     value_source_value AS v, unit_concept_id AS u, unit_source_value AS us
                              FROM {schemas.cdm}.observation ORDER BY observation_id""")
    assert [tuple(o.values()) for o in obs] == [
        (101, 7.5, None, None, 108, "g/dL"), (102, None, 107, "POS", None, None), (103, None, None, "done", None, None)]
    assert count(conn, schemas, "condition_occurrence", "condition_concept_id = 101") == 1
    assert count(conn, schemas, "measurement") == 2


@pytest.mark.postgres
def test_route_step_error_is_not_blocking(cdm):
    conn, schemas = cdm
    insert(conn, schemas, "condition_occurrence", [cond(1, 102)])
    execute(conn, f"ALTER TABLE {schemas.vocab}.concept RENAME TO concept_gone")
    conn.commit()
    steps = route_steps(schemas, ["condition_occurrence"])
    res = _run_step(conn, steps[1], None)
    conn.commit()
    assert res.error and not res.blocking and res.kind == "route" and res.moved == []
    assert count(conn, schemas, "condition_occurrence") == 1  # rolled back: the record stays


# --------------------------------------------------------------------------------------
# a whole build on the demo data, with codes of other domains
# --------------------------------------------------------------------------------------

# ICD-10 codes added to the fixture vocabulary: Z34.9 maps to an observation, Z30.0 to a
# procedure, O80 to a condition and an observation (two targets in two domains)
EXTRA_CONCEPTS = [
    (2000100001, "Supervision of normal pregnancy, unspecified", "Condition", "ICD10", "ICD10 code", None, "Z34.9"),
    (2000100002, "Routine antenatal care", "Observation", "SNOMED", "Procedure", "S", "FX2000100002"),
    (2000100003, "General counselling and advice on contraception", "Condition", "ICD10", "ICD10 code", None, "Z30.0"),
    (2000100004, "Contraceptive counselling", "Procedure", "SNOMED", "Procedure", "S", "FX2000100004"),
    (2000100005, "Single spontaneous delivery", "Condition", "ICD10", "ICD10 code", None, "O80"),
    (2000100006, "Normal delivery", "Condition", "SNOMED", "Clinical Finding", "S", "FX2000100006"),
    (2000100007, "Delivery normal", "Observation", "SNOMED", "Observable Entity", "S", "FX2000100007"),
]
EXTRA_MAPS = [(2000100001, 2000100002), (2000100003, 2000100004), (2000100005, 2000100006), (2000100005, 2000100007)]


def add_codes(conn, schemas, per_code: int = 4) -> dict[str, int]:
    """The extra concepts, and ``per_code`` diagnoses of each extra code for persons with a visit."""
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {schemas.vocab}.concept (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, "
            "standard_concept, concept_code, valid_start_date, valid_end_date) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, DATE '1970-01-01', DATE '2099-12-31')", EXTRA_CONCEPTS)
        cur.executemany(
            f"INSERT INTO {schemas.vocab}.concept_relationship (concept_id_1, concept_id_2, relationship_id, "
            "valid_start_date, valid_end_date) VALUES (%s, %s, 'Maps to', DATE '1970-01-01', DATE '2099-12-31')",
            EXTRA_MAPS)
    visits = fetch_all(conn, f"SELECT visit_id, individual_id, visit_date FROM {schemas.src}.clinic_visits "
                             f"ORDER BY visit_id LIMIT {per_code}")
    for code in ("Z34.9", "Z30.0", "O80"):
        for v in visits:
            execute(conn, f"INSERT INTO {schemas.src}.diagnoses (visit_id, individual_id, icd10, dx_text, dx_date) "
                          "VALUES (%s, %s, %s, 'extra', %s)", (v["visit_id"], v["individual_id"], code, v["visit_date"]))
    conn.commit()
    return {"visits": len(visits)}


@pytest.mark.postgres
def test_build_routes_records_and_reports_them(staged, tmp_path):
    conn, schemas, cfg = staged
    n = add_codes(conn, schemas)["visits"]
    ws = Workspace(cfg.workdir)
    spec = fixed_ref()
    result = run_etl(conn, schemas, spec, cfg, ws)
    assert result.ok and result.errors() == [], [(e.step, e.error) for e in result.errors()]
    assert vocab_indexed(conn, schemas) is True  # load_athena indexes concept(concept_id)
    assert "ANY(ARRAY(SELECT DISTINCT" in (ws.sql_dir / "12_route_condition_occurrence.sql").read_text()
    by = {s.step.split("_", 1)[1]: s for s in result.steps}
    cond_load = by["condition_occurrence"]
    # O80 has two targets: two rows at load, one in each table after routing
    assert moves(by["route_condition_occurrence"]) == [
        ("observation", "Observation", 2 * n, 0), ("procedure_occurrence", "Procedure", n, 0)]
    assert moves(by["route_drug_exposure"]) == [] and moves(by["route_measurement"]) == []
    routed = count(conn, schemas, "condition_occurrence") + count(conn, schemas, "observation") + count(
        conn, schemas, "procedure_occurrence")
    assert routed == cond_load.rows_inserted  # rows conserved
    assert count(conn, schemas, "observation", "observation_source_value = 'Z34.9'") == n
    assert count(conn, schemas, "observation", "observation_source_value = 'O80'") == n
    assert count(conn, schemas, "condition_occurrence", "condition_source_value = 'O80'") == n
    assert count(conn, schemas, "procedure_occurrence", "procedure_source_value = 'Z30.0'") == n
    assert count(conn, schemas, "observation", "visit_occurrence_id IS NULL") == 0
    for t in ("condition_occurrence", "observation", "procedure_occurrence"):
        pk = EVENT_FIELD_ROLES[t]["id"]
        assert scalar(conn, f"SELECT COUNT(DISTINCT {pk}) = COUNT(*) FROM {schemas.cdm}.{t}"), t
    # the build's log and audit
    assert result.rows_moved() == 3 * n
    audit = [e for e in ws.read_audit() if e["event"] == "etl_run"][-1]
    assert audit["totals"]["moved_rows"] == 3 * n

    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    checks = {r.key: r for r in report.results}
    # retention is measured at load, before routing; nothing is misplaced afterwards
    retention = checks["etlRecordRetention.condition_occurrence"]
    assert retention.num_violated_rows == cond_load.source_rows - cond_load.retained_rows
    fk = checks["fkDomain.condition_occurrence.condition_concept_id"]
    assert (fk.failed, fk.num_violated_rows) == (False, 0)
    routing = checks["etlDomainRouting.condition_occurrence.route_condition_occurrence"]
    assert (routing.num_violated_rows, routing.severity, routing.failed) == (3 * n, "characterization", False)
    # observation and procedure_occurrence received records: their checks run. No mapping loads them,
    # so their person completeness is not sent to the repair agent
    for t in ("observation", "procedure_occurrence"):
        pc = checks[f"measurePersonCompleteness.{t}"]
        assert not pc.not_applicable and pc.failed and pc.repairable is False, t
        assert not repairable(pc, {"fatal", "convention"})
        assert not checks[f"isRequired.{t}.person_id"].not_applicable
    assert checks["measurePersonCompleteness.device_exposure"].not_applicable  # received nothing
    _, failures = repair_triggers(report, result, cfg)
    assert not [r for r in failures if r.check_name == "measurePersonCompleteness"]


@pytest.mark.postgres
def test_rebuild_truncates_route_targets(staged):
    conn, schemas, cfg = staged
    n = add_codes(conn, schemas, per_code=2)["visits"]
    ws = Workspace(cfg.workdir)
    spec = fixed_ref()
    assert run_etl(conn, schemas, spec, cfg, ws).ok
    assert count(conn, schemas, "observation") == 2 * n
    ws.etl_run_json.unlink()  # no record of the earlier moves: the route destinations are emptied anyway
    assert run_etl(conn, schemas, spec, cfg, ws).ok
    assert count(conn, schemas, "observation") == 2 * n and count(conn, schemas, "procedure_occurrence") == n
    # without routing, the tables an earlier build moved records to are emptied too
    off = run_etl(conn, schemas, spec.model_copy(update={"route_by_domain": False}), cfg, ws)
    assert off.ok and not any(s.kind == "route" for s in off.steps)
    assert count(conn, schemas, "observation") == 0 and count(conn, schemas, "procedure_occurrence") == 0
    assert count(conn, schemas, "condition_occurrence", "condition_source_value IN ('Z34.9', 'Z30.0')") == 2 * n


# --------------------------------------------------------------------------------------
# the mapping side and the vocabulary indexes on PostgreSQL
# --------------------------------------------------------------------------------------

ROAD_TRAFFIC_ACCIDENT = 2000001010  # the fixture vocabulary's only Observation concept


@pytest.mark.postgres
def test_search_adds_other_accepted_domains(staged):
    from omop_agent.vocab.mapper import collect_terms, search_candidates

    conn, schemas, _ = staged
    alone = search_candidates(conn, schemas, "Road traffic accident", "Measurement", limit=5)
    assert alone and {c.domain_id for c in alone} == {"Measurement"}
    both = search_candidates(conn, schemas, "Road traffic accident", "Measurement", limit=5, also=["Observation"])
    # the target domain's candidates as before, plus the best of the other accepted domain, by score
    assert [c.concept_id for c in both if c.domain_id == "Measurement"] == [c.concept_id for c in alone]
    assert both[0].concept_id == ROAD_TRAFFIC_ACCIDENT and len(both) == len(alone) + 1
    # the terms of a routed concept field carry their accepted domains; the others none
    terms = {t.source_vocabulary_id: t for t in collect_terms(conn, schemas, load_ref())}
    assert terms["SRC_LAB"].accepted_domains == ["Measurement", "Observation"]
    assert terms["SRC_DRUG"].accepted_domains == ["Drug", "Device"]
    assert terms["SRC_UNIT"].accepted_domains == [] and terms["SRC_UNIT"].target_domain == "Unit"


@pytest.mark.postgres
@pytest.mark.parametrize("pool", [300, 2])
def test_one_retrieval_for_all_accepted_domains(staged, monkeypatch, pool):
    """The domains too large to score in full are retrieved together, per term (the name and
    synonym scans do not depend on the domain), with per domain the same candidates as a search of
    that domain alone: the target domain's recall is unchanged. The retrieval runs first at a higher
    similarity threshold, which a full vocabulary answers much faster, and again at the normal one
    only for the domains whose pool that left short (on the fixture vocabulary: all of them, unless
    the pool is tiny); the pools are those of a retrieval at the normal threshold."""
    from omop_agent.vocab import mapper

    conn, schemas, _ = staged
    monkeypatch.setattr(mapper, "EXHAUSTIVE_DOMAIN_SIZE", 0)  # every domain is too large to score in full
    monkeypatch.setattr(mapper, "POOL_SIZE", pool)
    if not mapper._Searcher(conn, schemas).trigram:
        pytest.skip("pg_trgm is not available")

    def key(h):
        return -h.candidate.score, h.candidate.domain_id != "Measurement", h.candidate.concept_id

    def counting(searcher):
        retrievals: list[tuple[str, str]] = []
        real = searcher._rows

        def counted(sql, params=None, setup=None):
            if "word_similarity" in sql:
                retrievals.append((setup, ",".join(params.get("domains") or [params.get("domain") or ""])))
            return real(sql, params, setup)

        searcher._rows = counted  # type: ignore[method-assign]
        return retrievals

    domains = "Measurement,Observation,Condition"
    shortcut = 0
    for text in ("haemoglobin", "Road traffic accident", "malaria test", "HB"):
        searcher = mapper._Searcher(conn, schemas)
        retrievals = counting(searcher)
        both = searcher.search(text, "Measurement", 5, also=["Observation", "Condition"])
        first = f"SET LOCAL pg_trgm.word_similarity_threshold = {mapper.TRGM_FIRST_THRESHOLD}"
        assert retrievals[0] == (first, domains) and len(retrievals) <= 2, text
        if len(retrievals) == 2:  # again at the normal threshold, for the domains left short only
            assert retrievals[1][0] == f"SET LOCAL pg_trgm.word_similarity_threshold = {mapper.TRGM_WORD_THRESHOLD}"
            assert set(retrievals[1][1].split(",")) <= set(domains.split(","))
        shortcut += len(retrievals) == 1 or len(retrievals[1][1].split(",")) < 3
        alone = mapper._Searcher(conn, schemas)
        expected = (alone.search(text, "Measurement", 5) + alone.search(text, "Observation", 3)
                    + alone.search(text, "Condition", 3))
        assert [(h.candidate.concept_id, h.candidate.score) for h in both] == [
            (h.candidate.concept_id, h.candidate.score) for h in sorted(expected, key=key)], text
        # the same pools as one retrieval at the normal threshold
        with monkeypatch.context() as m:
            m.setattr(mapper, "TRGM_FIRST_THRESHOLD", mapper.TRGM_WORD_THRESHOLD)
            once = mapper._Searcher(conn, schemas)
            single = once.search(text, "Measurement", 5, also=["Observation", "Condition"])
        assert [(h.candidate.concept_id, h.candidate.score) for h in both] == [
            (h.candidate.concept_id, h.candidate.score) for h in single], text
    assert shortcut > 0 if pool == 2 else shortcut == 0


@pytest.mark.postgres
def test_off_domain_choice_is_unchecked_not_flagged(staged):
    from omop_agent.models import SourceTerm
    from omop_agent.vocab.mapper import map_terms
    from tests.test_vocab import CapturingLLM

    conn, schemas, cfg = staged
    routed = SourceTerm(source_vocabulary_id="SRC_X", source_code="RTA", source_name="Road traffic accident",
                        frequency=50, target_domain="Measurement", accepted_domains=["Measurement", "Observation"])
    llm = CapturingLLM({"RTA": ROAD_TRAFFIC_ACCIDENT})
    d = map_terms(conn, schemas, [routed], llm, cfg)[0]
    assert (d.target_concept_id, d.target_domain_id, d.status, d.method) == (
        ROAD_TRAFFIC_ACCIDENT, "Observation", "UNCHECKED", "llm")
    assert d.accepted_domains == ["Measurement", "Observation"]
    prompt = llm.calls[0][1]
    assert "target domain: Measurement (also accepted: Observation;" in prompt
    assert f"{ROAD_TRAFFIC_ACCIDENT} | Road traffic accident | Observation |" in prompt
    # a term that accepts its target domain only: the same choice is rejected
    strict = routed.model_copy(update={"accepted_domains": [], "source_vocabulary_id": "SRC_Y"})
    d = map_terms(conn, schemas, [strict], CapturingLLM({"RTA": ROAD_TRAFFIC_ACCIDENT}), cfg)[0]
    assert (d.target_concept_id, d.status) == (0, "FLAGGED") and "domain Observation, not Measurement" in d.rationale


@pytest.mark.postgres
def test_self_loaded_vocabulary_gets_its_indexes(staged, tmp_path):
    """DB2: a vocabulary loaded by someone else gets the index on concept(concept_id) too (the route
    steps look concepts up by it); the runner compiles the join form without it."""
    from omop_agent.pipeline import Pipeline

    conn, schemas, cfg = staged
    execute(conn, f"DROP INDEX {schemas.vocab}.idx_concept_concept_id")
    conn.commit()
    assert vocab_indexed(conn, schemas) is False
    ws = Workspace(cfg.workdir)
    run_etl(conn, schemas, fixed_ref(), cfg, ws)
    assert f"JOIN {schemas.vocab}.concept c ON c.concept_id = d.id" in (
        ws.sql_dir / "12_route_condition_occurrence.sql").read_text()
    self_loaded = cfg.model_copy(update={"vocabulary": cfg.vocabulary.model_copy(update={"athena_dir": None})})
    p = Pipeline(self_loaded, log=lambda _: None)
    try:
        p.load_vocabulary()
    finally:
        p.close()
    assert vocab_indexed(conn, schemas) is True
    run_etl(conn, schemas, fixed_ref(), cfg, ws)
    assert "ANY(ARRAY(SELECT DISTINCT" in (ws.sql_dir / "12_route_condition_occurrence.sql").read_text()


@pytest.mark.postgres
def test_vocabulary_indexes_never_fail_the_stage(staged):
    """A vocabulary served as views (or foreign tables, or read-only) cannot be indexed: the
    vocab_load stage logs and audits it instead of failing, and the build joins the concept table.
    An index under another name over the same columns (the OHDSI primary key) counts as present."""
    from omop_agent.pipeline import Pipeline
    from omop_agent.vocab.loader import VOCAB_FILES, ensure_indexes

    conn, schemas, cfg = staged
    # the OHDSI DDL's primary key serves the lookups by id: no second index on concept_id
    execute(conn, f"DROP INDEX {schemas.vocab}.idx_concept_concept_id")
    execute(conn, f"ALTER TABLE {schemas.vocab}.concept ADD CONSTRAINT xpk_concept PRIMARY KEY (concept_id)")
    conn.commit()
    assert ensure_indexes(conn, schemas) == []
    assert not scalar(conn, "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = 'idx_concept_concept_id'",
                      (schemas.vocab,))
    assert vocab_indexed(conn, schemas) is True
    # the vocabulary tables moved to another schema, and served as views
    execute(conn, f"CREATE SCHEMA IF NOT EXISTS {schemas.results}")
    for t in VOCAB_FILES:
        execute(conn, f"ALTER TABLE {schemas.vocab}.{t} SET SCHEMA {schemas.results}")
        execute(conn, f"CREATE VIEW {schemas.vocab}.{t} AS SELECT * FROM {schemas.results}.{t}")
    conn.commit()
    assert "idx_concept_code" in ensure_indexes(conn, schemas)
    logged: list[str] = []
    self_loaded = cfg.model_copy(update={"vocabulary": cfg.vocabulary.model_copy(update={"athena_dir": None})})
    p = Pipeline(self_loaded, log=logged.append)
    try:
        p.load_vocabulary()
    finally:
        p.close()
    assert any("vocabulary indexes not created" in line for line in logged), logged
    assert [e for e in Workspace(cfg.workdir).read_audit() if e["event"] == "vocab_indexes_not_created"]
    assert vocab_indexed(conn, schemas) is False
