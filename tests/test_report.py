"""Inspection report: artifact collection, verdict, rendering, escaping, CDM characterisation."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from omop_agent.config import ProjectConfig
from omop_agent.db import execute, qi
from omop_agent.models import (
    CheckResult,
    ColumnProfile,
    DQReport,
    ETLRunResult,
    MappingDecision,
    RepairAction,
    SourceAnnotations,
    SourceProfile,
    TableAnnotation,
    TableLoadResult,
    TableProfile,
    ValueFreq,
)
from omop_agent.report.inspection import (
    build_report,
    characterize_cdm,
    collect_report_data,
    fmt_pct,
    md_escape,
    small_cell,
)
from omop_agent.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
SECTIONS = [
    "Summary",
    "Pipeline timeline",
    "Source summary",
    "ETL design",
    "Vocabulary",
    "Vocabulary mapping",
    "ETL run",
    "Data quality",
    "Repair log",
    "CDM characterization",
    "LLM usage",
    "Next steps",
]
XSS = "<script>alert('x')</script>"


def _cfg(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "name": "demo",
            "workdir": str(tmp_path / "work"),
            "source": {"type": "csv", "path": str(tmp_path)},
            "target": {"url": "postgresql://unused"},
            "llm": {"provider": "anthropic", "model": "claude-opus-5-5"},
        }
    )


def _col(name: str, **kw) -> ColumnProfile:
    base = {"inferred_type": "string", "n_rows": 400, "n_null": 0, "n_distinct": 10, "unique": False}
    return ColumnProfile(name=name, **{**base, **kw})


def _check(name: str, table: str, field: str | None = None, **kw) -> CheckResult:
    base = {"check_level": "FIELD", "description": f"{name} on {table}", "category": "Conformance"}
    return CheckResult(check_name=name, table=table, field=field, **{**base, **kw})


def _passing_run(ws: Workspace) -> None:
    ws.save_json(
        ws.etl_run_json,
        ETLRunResult(
            started_at=T0 + timedelta(minutes=10),
            finished_at=T0 + timedelta(minutes=11),
            steps=[TableLoadResult(target_table="person", step="person", rows_inserted=400, source_rows=400)],
        ),
    )
    ws.save_json(
        ws.dq_json,
        DQReport(
            run_at=T0 + timedelta(minutes=12),
            results=[_check("isRequired", "person", "person_id", severity="fatal", num_denominator_rows=400)],
            passed=True,
        ),
    )


@pytest.fixture
def full_ws(tmp_path: Path) -> tuple[ProjectConfig, Workspace]:
    cfg = _cfg(tmp_path)
    ws = Workspace(cfg.workdir)
    ws.save_json(
        ws.profile_json,
        SourceProfile(
            source_type="csv",
            generated_at=T0,
            min_cell_count=5,
            tables=[
                TableProfile(
                    name="individuals",
                    n_rows=400,
                    columns=[
                        _col("individual_id", n_distinct=400, unique=True),
                        _col("first_name", is_pii=True, pii_reason="column name matches /name/"),
                        _col("dob", inferred_type="date", date_format="%d/%m/%Y"),
                        _col("sex", top_values=[ValueFreq(value="1", count=200)]),
                    ],
                ),
                TableProfile(name="diagnoses", n_rows=1000, columns=[_col("icd10", n_rows=1000)]),
            ],
        ),
    )
    ws.save_json(
        ws.annotations_json,
        SourceAnnotations(
            tables=[TableAnnotation(table="individuals", description="HDSS registry", grain="one row per person")],
            columns=[],
            notes=["Sex code 9 probably means unknown: confirm with the data owner."],
        ),
    )
    shutil.copy(FIXTURES / "demo_etl_spec.yaml", ws.spec_yaml)
    ws.vocab_eval_json.write_text(
        json.dumps(
            {
                "vocabulary_version": "v5.0 FIXTURE",
                "loaded": [{"vocabulary_id": "ICD10", "version": "2026", "concepts": 9, "standard_concepts": 0}],
                "required": [
                    {"vocabulary_id": "ICD10", "reason": "diagnoses.icd10", "present": True},
                    {"vocabulary_id": "LOINC", "reason": "lab codes", "present": False},
                ],
                "coverage": [
                    {
                        "target_table": "condition_occurrence",
                        "target_field": "condition_concept_id",
                        "vocabulary_id": "ICD10",
                        "source_table": "diagnoses",
                        "distinct_codes": 9,
                        "codes_found": 8,
                        "codes_mapped_to_standard": 8,
                        "records": 1000,
                        "records_mapped": 900,
                        "unmatched": [{"value": "J189", "count": 97}, {"value": "Z999", "count": 3}],
                    }
                ],
                "issues": ["LOINC is required but not loaded"],
            }
        )
    )
    ws.save_json(
        ws.mappings_dir / "decisions.json",
        [
            MappingDecision(
                source_vocabulary_id="SRC_DRUG", source_code="AL 20/120", source_name="AL 20/120 tabs", frequency=260,
                target_concept_id=2000003001, target_concept_name="artemether / lumefantrine", status="APPROVED",
                equivalence="EQUAL", confidence=0.97, method="llm",
            ),
            MappingDecision(
                source_vocabulary_id="SRC_DRUG", source_code="PCM", source_name="Paracetamol 500mg", frequency=300,
                target_concept_id=2000003002, target_concept_name="acetaminophen 500 MG Oral Tablet",
                status="UNCHECKED", equivalence="EQUIVALENT", confidence=0.8, method="llm",
            ),
            MappingDecision(
                source_vocabulary_id="SRC_DRUG", source_code="herb", source_name="Mystery herb", frequency=3,
                status="UNCHECKED", method="suppressed",
            ),
            MappingDecision(
                source_vocabulary_id="SRC_LAB", source_code="HB", source_name="Haemoglobin", frequency=50,
                target_concept_id=2000004001, target_concept_name="Hemoglobin [Mass/volume] in Blood",
                status="FLAGGED", equivalence="INEXACT", confidence=0.4, method="llm",
            ),
        ],
    )
    ws.save_json(
        ws.etl_run_json,
        ETLRunResult(
            started_at=T0 + timedelta(minutes=10),
            finished_at=T0 + timedelta(minutes=11),
            steps=[
                TableLoadResult(target_table="person", step="person", rows_inserted=398, source_rows=400, seconds=0.5),
                TableLoadResult(
                    target_table="measurement", step="measurement", source_rows=1200, seconds=0.2,
                    error='column "x.unit" does not exist',
                ),
            ],
        ),
    )
    ws.save_json(
        ws.dq_json,
        DQReport(
            run_at=T0 + timedelta(minutes=12),
            results=[
                _check(
                    "isRequired", "person", "gender_concept_id", severity="fatal", num_violated_rows=4,
                    num_denominator_rows=398, pct_violated_rows=4 / 398, threshold_pct=0, failed=True,
                ),
                _check(
                    "plausibleValueLow", "person", "year_of_birth", severity="convention", category="Plausibility",
                    num_violated_rows=12, num_denominator_rows=398, pct_violated_rows=12 / 398, threshold_pct=1,
                    failed=True,
                ),
                _check("measurePersonCompleteness", "person", check_level="TABLE", category="Completeness"),
                _check("isForeignKey", "measurement", "unit_concept_id", not_applicable=True),
                _check("cdmField", "measurement", "value_as_number", error="relation does not exist"),
            ],
            passed=False,
            blocking_failures=["isRequired.person.gender_concept_id (fatal): 4 of 398 (1.01%) violated, threshold 0%"],
        ),
    )
    ws.save_json(
        ws.repairs_json,
        [
            RepairAction(
                iteration=1, target_table="person", target_field="gender_concept_id",
                change="ELSE branch maps sex 9 to 0", rationale=f"Unknown sex {XSS} must still get a concept.",
                trigger="isRequired.person.gender_concept_id",
            )
        ],
    )
    for stage in ("profile", "annotate", "design", "vocab_map", "build", "quality"):
        ws.mark(stage, "done")
    ws.audit("review_gate", stage="design", outcome="approved", mode="interactive",
             artifacts=[str(ws.spec_md), str(ws.spec_yaml)])
    if ws.spec_yaml.exists():
        from omop_agent.approvals import record_approval

        record_approval(ws, "design", "alice", note="keys checked")
    ws.audit("repair", iteration=1, actions=1, trigger="isRequired.person.gender_concept_id", summary="Mapped sex 9.")
    ws.audit("repair_skipped", iteration=2, reason="no recording for repair/iteration:2", trigger="etl:measurement")
    usage = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 500}
    call = {"provider": "anthropic", "model": "claude-opus-5-5"}
    ws.audit("llm_call", stage="annotate", key="profile", seconds=2.5, usage=usage, error=None, **call)
    ws.audit("llm_call", stage="design", key="person", seconds=4.0, usage=usage, error=None, **call)
    ws.audit("llm_call", stage="design", key="drug", seconds=1.0, usage={}, error="overloaded", **call)
    return cfg, ws


def test_build_report_writes_both_formats(full_ws):
    cfg, ws = full_ws
    md_path, html_path = build_report(cfg, ws)
    assert (md_path, html_path) == (ws.report_md, ws.report_html)
    md, html = md_path.read_text(), html_path.read_text()
    for i, title in enumerate(SECTIONS, start=1):
        assert f"## {i}. {title}" in md
        assert f'<span class="sn">{i}</span>{title}</h2>' in html
    for text in ("FAILED", "claude-opus-5-5", "v5.0 FIXTURE", "individuals", "first_name", "%d/%m/%Y",
                 "Sex code 9 probably means unknown", "Should care sites be created", "LOINC",
                 'column "x.unit" does not exist', "Haemoglobin", "isRequired.person.gender_concept_id",
                 "use_r_dqd", "source_daimon", "llm_calls/"):
        assert text in md, text
    html_text = html.replace("&#34;", '"').replace("&#39;", "'")
    for text in ("FAILED", "claude-opus-5-5", "Haemoglobin", 'column "x.unit" does not exist', "blocking"):
        assert text in html_text, text
    assert html.lstrip().lower().startswith("<!doctype html>")
    # self-contained: no scripts, stylesheets or remote assets
    assert "<script" not in html and "<link" not in html and "src=" not in html
    assert "prefers-color-scheme: dark" in html
    assert "No database connection was given" in md


def test_report_warns_that_it_lists_source_terms(full_ws):
    cfg, ws = full_ws
    md, html = (p.read_text() for p in build_report(cfg, ws))
    notice = "Share it only within your data-governance boundary."
    head = md.split("## 1. Summary")[0]
    assert "**Confidential.**" in head and "including rare ones, needed for mapping review" in head and notice in head
    assert notice in html.split('<section id="s1"')[0]
    # the full report keeps rare terms: reviewers need them
    assert "Mystery herb" in md and "Z999 (&lt;5)" in md and "Mystery herb" in html


def test_shareable_report_withholds_rare_source_terms(full_ws):
    cfg, ws = full_ws
    data = collect_report_data(cfg, ws, shareable=True)
    herb = next(r for r in data["mapping"]["review"] if r["frequency"] == "<5")
    assert (herb["code"], herb["name"], herb["concept"]) == ("<withheld>", "<withheld>", "none (0)")
    frequent = next(r for r in data["mapping"]["review"] if r["code"] == "PCM")
    assert frequent["name"] == "Paracetamol 500mg"
    assert data["vocab"]["coverage"][0]["unmatched"] == "J189 (97), <withheld> (<5)"

    full_md, full_html = build_report(cfg, ws)
    md_path, html_path = build_report(cfg, ws, shareable=True)
    assert (md_path.name, html_path.name) == ("report.shareable.md", "report.shareable.html")
    assert full_md.read_text() != md_path.read_text() and "Mystery herb" in full_md.read_text()  # full copy kept
    md, html = md_path.read_text(), html_path.read_text()
    for text in ("Mystery herb", "herb", "Z999"):
        assert text not in md and text not in html, text
    assert "&lt;withheld&gt;" in md and "&lt;withheld&gt;" in html
    assert "**Shareable version.**" in md and "fewer than 5 times" in md and "Shareable version" in html
    assert "data-governance boundary" not in md
    assert "Paracetamol 500mg" in md and "J189" in md  # frequent terms are still shown


def test_html_and_markdown_escape_untrusted_text(full_ws):
    cfg, ws = full_ws
    md_path, html_path = build_report(cfg, ws)
    html, md = html_path.read_text(), md_path.read_text()
    assert "<script" not in html
    assert "&lt;script&gt;alert(" in html
    assert "<script" not in md and "&lt;script&gt;" in md
    assert md_escape("a|b\nc") == "a\\|b c"


def test_verdict(full_ws):
    cfg, ws = full_ws
    data = collect_report_data(cfg, ws)
    assert data["verdict"] == "FAILED"
    assert any("ETL step" in r for r in data["verdict_reasons"])
    assert any("data-quality gate failed" in r for r in data["verdict_reasons"])

    _passing_run(ws)
    data = collect_report_data(cfg, ws)
    assert data["verdict"] == "PASSED", data["verdict_reasons"]
    assert "PASSED" in build_report(cfg, ws)[1].read_text()

    # a failed stage fails the run even if the last checks passed
    ws.mark("build", "failed", error="boom")
    assert collect_report_data(cfg, ws)["verdict"] == "FAILED"
    ws.mark("build", "done")

    # a stage waiting at a review gate means the results may be stale
    ws.mark("design", "awaiting_review")
    data = collect_report_data(cfg, ws)
    assert data["verdict"] == "INCOMPLETE" and data["verdict_reasons"] == ["stage 'design' is awaiting_review"]
    ws.mark("design", "done")

    # checks older than the latest ETL run do not count
    run = ws.load_json(ws.etl_run_json, ETLRunResult)
    ws.save_json(ws.etl_run_json, run.model_copy(update={"started_at": T0 + timedelta(hours=1)}))
    assert collect_report_data(cfg, ws)["verdict"] == "INCOMPLETE"


def test_sections_content(full_ws):
    cfg, ws = full_ws
    data = collect_report_data(cfg, ws)

    src = data["source"]
    assert (src["n_tables"], src["n_rows"], src["n_columns"]) == (2, 1400, 5)
    assert src["pii"] == [{"table": "individuals", "column": "first_name", "reason": "column name matches /name/"}]
    assert src["tables"][0]["grain"] == "one row per person"

    design = data["design"]
    assert design["n_mappings"] >= 5 and design["open_questions"]
    person = next(t for t in design["tables"] if t["target"] == "person")
    assert person["source"] == "individuals i" and "person_id 1" in person["kinds"]

    vocab = data["vocab"]
    assert vocab["n_missing"] == 1 and vocab["coverage"][0]["pct"] == pytest.approx(90.0)
    assert "Z999 (<5)" in vocab["coverage"][0]["unmatched"]

    mapping = data["mapping"]
    drug = next(v for v in mapping["vocabularies"] if v["source_vocabulary_id"] == "SRC_DRUG")
    assert (drug["terms"], drug["approved"], drug["unchecked"], drug["unmapped"], drug["suppressed"]) == (3, 1, 2, 1, 1)
    assert drug["records_mapped"] == 560 and drug["pct"] == pytest.approx(100 * 560 / 563)
    assert mapping["review_total"] == 3
    assert [r["status"] for r in mapping["review"]] == ["FLAGGED", "UNCHECKED", "UNCHECKED"]
    assert mapping["review"][-1]["frequency"] == "<5"  # small cell, never exact

    etl = data["etl"]
    assert etl["n_errors"] == 1 and etl["steps"][0]["retention"] == pytest.approx(99.5)

    dq = data["dq"]
    fatal = next(r for r in dq["by_severity"] if r["name"] == "fatal")
    conv = next(r for r in dq["by_severity"] if r["name"] == "convention")
    assert (fatal["failed"], conv["failed"], conv["error"], conv["not_applicable"], conv["passed"]) == (1, 1, 1, 1, 1)
    assert dq["by_severity"][-1]["total"] == 5
    assert {r["name"] for r in dq["by_category"]} >= {"Conformance", "Completeness", "Plausibility"}
    assert [f["blocking"] for f in dq["failed"]] == [True, False]
    assert dq["failed"][0]["pct"] == pytest.approx(100 * 4 / 398)

    repairs = data["repairs"]
    assert repairs["n_actions"] == 1 and repairs["iterations"][0]["actions"][0]["target"] == "person.gender_concept_id"
    assert [(e["iteration"], e["outcome"]) for e in repairs["events"]] == [(1, "done"), (2, "skipped")]

    llm = data["llm"]
    assert llm["totals"]["calls"] == 3 and llm["totals"]["input_tokens"] == 2000
    assert llm["totals"]["output_tokens"] == 400 and llm["totals"]["errors"] == 1
    assert [s["stage"] for s in llm["stages"]] == ["annotate", "design"]
    assert data["project"]["llm_used"] == ["anthropic / claude-opus-5-5"]

    timeline = data["timeline"]
    assert next(s for s in timeline["stages"] if s["stage"] == "build")["status"] == "done"
    assert next(s for s in timeline["stages"] if s["stage"] == "report")["status"] == "not run"
    reviews = {r["gate"]: r for r in data["reviews"]}
    assert (reviews["ETL design"]["state"], reviews["ETL design"]["by"], reviews["ETL design"]["note"]) == (
        "approved", "alice", "keys checked")
    assert reviews["ETL design"]["last_pass"] == "approved"

    steps = {s["text"].split(" ")[0]: s["done"] for s in data["next_steps"]}
    assert steps["Review"] is False and steps["Register"] is False


def test_empty_workspace_is_incomplete(tmp_path):
    cfg = _cfg(tmp_path)
    ws = Workspace(cfg.workdir)
    data = collect_report_data(cfg, ws)
    assert data["verdict"] == "INCOMPLETE"
    assert all(data[k] is None for k in ("source", "design", "vocab", "mapping", "etl", "dq", "repairs", "cdm"))
    md_path, html_path = build_report(cfg, ws)
    md, html = md_path.read_text(), html_path.read_text()
    assert "INCOMPLETE" in md and "INCOMPLETE" in html
    assert md.count("Not run") >= 8 and html.count("Not run") >= 8


def test_unreadable_artifacts_are_reported_not_fatal(tmp_path):
    cfg = _cfg(tmp_path)
    ws = Workspace(cfg.workdir)
    ws.dq_json.write_text("{not json")
    ws.spec_yaml.write_text("version: 1\ntable_mappings: nope\n")
    ws.audit_log.write_text('{"event": "llm_call", "stage": "design", "usage": {"input_tokens": 5}}\ngarbage\n')
    data = collect_report_data(cfg, ws)
    assert data["verdict"] == "INCOMPLETE"
    assert {p.split(":")[0] for p in data["problems"]} == {"dq_results.json", "etl_spec.yaml", "audit.jsonl"}
    assert data["llm"]["totals"]["input_tokens"] == 5
    md, html = (p.read_text() for p in build_report(cfg, ws))
    assert "Unreadable artifacts" in md and "Unreadable artifacts" in html


def test_fmt_pct_never_rounds_to_the_bounds():
    assert (fmt_pct(None), fmt_pct(0), fmt_pct(100.0), fmt_pct(42.04)) == ("—", "0.0%", "100.0%", "42.0%")
    assert fmt_pct(99.97) == ">99.9%" and fmt_pct(0.01) == "<0.1%"
    assert fmt_pct(99.999, 2) == ">99.99%"


def test_small_cell():
    assert small_cell(3, 5) == {"count": None, "display": "<5", "suppressed": True}
    assert small_cell(5, 5)["display"] == "5"
    assert small_cell(12345, 5)["display"] == "12,345"
    assert small_cell(0, 5)["display"] == "0"


@pytest.mark.postgres
def test_characterize_cdm_suppresses_small_cells(staged, tmp_path):
    conn, schemas, _ = staged
    cdm = qi(schemas.cdm)
    # 9 men born in the 1980s, 3 women born in the 1990s
    people = [(i, 8507, 1980 + i % 10) for i in range(1, 10)] + [(i, 8532, 1990 + i % 10) for i in range(10, 13)]
    for pid, gender, yob in people:
        execute(
            conn,
            f"INSERT INTO {cdm}.person (person_id, gender_concept_id, year_of_birth, race_concept_id, "
            "ethnicity_concept_id) VALUES (%s, %s, %s, 0, 0)",
            (pid, gender, yob),
        )
        execute(
            conn,
            f"INSERT INTO {cdm}.observation_period VALUES (%s, %s, DATE '2020-01-01', DATE '2020-01-01' + %s, 32817)",
            (pid, pid, pid * 10),
        )
    # 6 malaria, 2 pneumonia
    for i, concept in enumerate([2000001001] * 6 + [2000001002] * 2, start=1):
        execute(
            conn,
            f"INSERT INTO {cdm}.condition_occurrence (condition_occurrence_id, person_id, condition_concept_id, "
            "condition_start_date, condition_type_concept_id) VALUES (%s, %s, %s, DATE '2021-03-01', 32817)",
            (i, i, concept),
        )

    out = characterize_cdm(conn, schemas, min_cell_count=5)
    assert out["errors"] == []
    counts = {r["table"]: r["display"] for r in out["record_counts"]}
    assert counts == {"person": "12", "condition_occurrence": "8", "observation_period": "12"}

    gender = {r["concept_name"]: r for r in out["gender"]}
    assert gender["MALE"]["display"] == "9" and gender["MALE"]["bar"] == 100.0
    assert gender["FEMALE"] == {
        "concept_id": 8532, "concept_name": "FEMALE", "count": None, "display": "<5", "suppressed": True, "bar": 0.0
    }
    decades = {r["decade"]: r["display"] for r in out["birth_decades"]}
    assert decades == {"1980s": "9", "1990s": "<5"}

    op = out["observation_period"]
    assert op["periods"] == "12" and op["median_days"] == 65 and not op["suppressed"]
    assert op["p10_days"] < op["median_days"] < op["p90_days"]

    top = {t["table"]: t["rows"] for t in out["top_concepts"]}
    assert set(top) == {"condition_occurrence"}
    assert [(r["concept_name"], r["display"]) for r in top["condition_occurrence"]] == [
        ("Malaria", "6"),
        ("Pneumonia", "<5"),
    ]
    assert out["vocabulary_version"] == "v5.0 FIXTURE"
    # the exact small counts (3 women, 2 pneumonia records) never appear
    assert '"count": 3' not in json.dumps(out) and '"count": 2' not in json.dumps(out)

    cfg = _cfg(tmp_path)
    ws = Workspace(cfg.workdir)
    md_path, html_path = build_report(cfg, ws, conn, schemas)
    md, html = md_path.read_text(), html_path.read_text()
    assert "Persons by gender" in html and "&lt;5" in html and 'class="bar"' in html
    assert "| FEMALE | 8532 | &lt;5 |" in md
    assert "v5.0 FIXTURE" in md  # vocabulary version falls back to the vocab schema


@pytest.mark.postgres
def test_characterize_cdm_withholds_small_distributions(pg):
    conn, schemas = pg
    from omop_agent.cdm.ddl import create_cdm_tables

    create_cdm_tables(conn, schemas, include_vocab=False)
    cdm = qi(schemas.cdm)
    for pid in (1, 2):
        execute(
            conn,
            f"INSERT INTO {cdm}.observation_period VALUES (%s, %s, DATE '2020-01-01', DATE '2021-01-01', 32817)",
            (pid, pid),
        )
    out = characterize_cdm(conn, schemas, min_cell_count=5)
    assert out["record_counts"] == [
        {"table": "observation_period", "count": None, "display": "<5", "suppressed": True, "bar": 0.0}
    ]
    assert out["observation_period"]["suppressed"] and out["observation_period"]["median_days"] is None
    assert out["gender"] == [] and out["vocabulary_version"] is None


def test_shareable_report_withholds_values_in_database_errors():
    from omop_agent.report.inspection import _redact_errors

    data = {
        "etl": {"steps": [{"error": 'null value in column "visit_end_date" violates not-null constraint '
                                    "DETAIL: Failing row contains (1, 2, 9202, 2019-04-22, KH000123)"}]},
        "verdict_reasons": ["stage 'build' failed: invalid input syntax for type date: \"31/02/1990\""],
        "mapping": {"rationale": 'kept "Hemoglobin"'},
    }
    out = _redact_errors(data)
    assert "KH000123" not in str(out) and "31/02/1990" not in str(out)
    assert out["mapping"]["rationale"] == 'kept "Hemoglobin"'  # only error fields are touched


def test_shareable_report_withholds_values_in_the_stage_timeline(full_ws):
    cfg, ws = full_ws
    ws.mark("vocab_map", "failed", error='invalid input syntax for type integer: "Wanjiku Otieno 0712345678"')
    for shareable in (False, True):
        timeline = str(collect_report_data(cfg, ws, shareable=shareable)["timeline"])
        assert ("Wanjiku" in timeline) is not shareable


def test_report_counts_only_mappings_that_reach_the_cdm():
    from omop_agent.models import MappingDecision
    from omop_agent.report.inspection import _mapping_row

    def d(code, status, target, n):
        return MappingDecision(source_vocabulary_id="SRC", source_code=code, source_name=code, frequency=n,
                               target_concept_id=target, status=status)

    ds = [d("a", "APPROVED", 1, 100), d("b", "UNCHECKED", 2, 50), d("c", "FLAGGED", 3, 30), d("e", "REJECTED", 4, 20)]
    assert _mapping_row("SRC", ds)["records_mapped"] == 150
    assert _mapping_row("SRC", ds, include_unchecked=False)["records_mapped"] == 100
