"""Vocabulary evaluation, Usagi-style mapping and the mapping review round trip."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from omop_agent.db import execute, fetch_all
from omop_agent.llm import LLMClient, NullLLM, ReplayLLM
from omop_agent.llm.replay import save_recording
from omop_agent.config import PrivacyConfig
from omop_agent.models import (
    ColumnProfile,
    ConceptCandidate,
    ETLSpec,
    FieldMapping,
    MappingBatchDecision,
    MappingDecision,
    SourceProfile,
    SourceTerm,
    TableProfile,
)
from omop_agent.vocab import mapper
from omop_agent.vocab.evaluate import evaluate_vocabulary, render_vocab_evaluation_md
from omop_agent.vocab.mapper import (
    MAPPER_SYSTEM,
    collect_terms,
    map_terms,
    mapping_summary,
    normalize,
    pii_concept_fields,
    search_candidates,
    text_score,
    write_source_to_concept_map,
)
from omop_agent.vocab.review import REVIEW_COLUMNS, export_review, import_review
from omop_agent.workspace import Workspace

SPEC_PATH = Path(__file__).parent / "fixtures" / "demo_etl_spec.yaml"
RARE = ["Quinine inj 600mg", "Ceftriaxone 1g inj", "HIV/AIDS related death"]

# concept ids of the fixture vocabulary used below
ACETAMINOPHEN_500_TAB = 2000003002
ACETAMINOPHEN = 2000003103
ARTEMETHER_LUMEFANTRINE_TAB = 2000003001
ORS = 2000003004
TLD_TAB = 2000003008
MALARIA = 2000001001
GRAM_PER_DL = 2000005001


def load_spec() -> ETLSpec:
    return ETLSpec.model_validate(yaml.safe_load(SPEC_PATH.read_text()))


def choice(code: str, cid: int, conf: float = 0.95, eq: str = "EQUAL", retry: list[str] | None = None) -> dict:
    return {
        "source_code": code,
        "chosen_concept_id": cid,
        "equivalence": eq if cid else "UNMATCHED",
        "confidence": conf,
        "rationale": f"test choice for {code}",
        "retry_search_terms": retry or [],
    }


def record(root: Path, key: str, choices: list[dict]) -> None:
    save_recording(root, "vocab_map", key, {"kind": "structured", "output": {"choices": choices}})


class CapturingLLM(LLMClient):
    """Answers every term with 0 and remembers (key, prompt) of each call."""

    provider = "capture"

    def __init__(self, answers: dict[str, int] | None = None):
        super().__init__(None)
        self.calls: list[tuple[str, str]] = []
        self.answers = answers or {}

    def structured(self, *, stage, key, system, prompt, output):
        assert stage == "vocab_map" and system == MAPPER_SYSTEM and output is MappingBatchDecision
        self.calls.append((key, prompt))
        codes = [
            json.loads(line.split("source_code: ", 1)[1]) for line in prompt.splitlines() if "source_code: " in line
        ]
        return MappingBatchDecision.model_validate({"choices": [choice(c, self.answers.get(c, 0)) for c in codes]})

    def run_agent(self, **kw):  # pragma: no cover - not used
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# pure functions
# --------------------------------------------------------------------------------------


def test_normalize():
    assert normalize("Paracetamol 500mg") == "paracetamol 500 mg"
    assert normalize("Co-trimoxazole 960mg") == "cotrimoxazole 960 mg"
    assert normalize("TLD (TDF/3TC/DTG)") == "tld tdf 3 tc dtg"
    assert normalize("quinine 0.5mg/ML") == "quinine 0.5 mg ml"
    assert normalize("Hemoglobin A1c/Hemoglobin.total") == "hemoglobin a1c hemoglobin total"


def test_text_score_prefers_the_clinical_drug_over_the_ingredient():
    q = normalize("Paracetamol 500mg")
    drug = text_score(q, normalize("paracetamol 500 MG Oral Tablet"))
    ingredient = text_score(q, normalize("paracetamol"))
    other_strength = text_score(q, normalize("acetaminophen 325 MG Oral Tablet"))
    assert drug > ingredient > other_strength
    assert text_score("malaria", "malaria") == 1.0
    assert text_score("malaria", "male") < 0.2
    assert text_score("pos", "positive") > 0.85  # abbreviation by prefix
    assert text_score("amoxicillin 250 mg caps", normalize("amoxicillin 250 MG Oral Capsule")) > text_score(
        "amoxicillin 250 mg caps", normalize("amoxicillin 500 MG Oral Capsule")
    )


def test_mapping_summary():
    def d(code, status, target, freq):
        return MappingDecision(
            source_vocabulary_id="SRC_X",
            source_code=code,
            source_name=code,
            frequency=freq,
            status=status,
            target_concept_id=target,
        )

    rows = mapping_summary(
        [
            d("a", "APPROVED", 1, 50),
            d("b", "UNCHECKED", 2, 30),
            d("c", "UNCHECKED", 0, 10),
            d("d", "FLAGGED", 0, 5),
            d("e", "REJECTED", 3, 5),
        ]
    )
    assert rows == [
        {
            "source_vocabulary_id": "SRC_X",
            "terms": 5,
            "approved": 1,
            "unchecked": 1,
            "flagged": 1,
            "rejected": 1,
            "unmapped": 1,
            "records": 100,
            "records_mapped": 80,
            "pct_records_mapped": 80.0,
        }
    ]
    assert (
        mapping_summary([d("a", "APPROVED", 1, 50), d("b", "UNCHECKED", 2, 50)], include_unchecked=False)[0][
            "records_mapped"
        ]
        == 50
    )


# --------------------------------------------------------------------------------------
# evaluate vocabulary
# --------------------------------------------------------------------------------------


@pytest.mark.postgres
def test_evaluate_vocabulary(staged):
    conn, schemas, cfg = staged
    ev = evaluate_vocabulary(conn, schemas, load_spec(), cfg)

    required = {r.vocabulary_id: r for r in ev.required}
    for vocab in ["ICD10", "RxNorm", "LOINC", "UCUM", "Gender", "Visit", "Type Concept"]:
        assert required[vocab].present, vocab
    assert "codes in condition_occurrence.condition_concept_id" in required["ICD10"].reason
    assert required["Gender"].reason == "OMOP conventions"
    assert "SNOMED" not in required  # no concept_map field targets Condition/Observation/Procedure
    assert {v.vocabulary_id for v in ev.loaded} >= {"ICD10", "RxNorm", "SNOMED"}
    assert ev.vocabulary_version == "v5.0 FIXTURE"

    cov = next(c for c in ev.coverage if c.target_field == "condition_concept_id")
    assert cov.vocabulary_id == "ICD10" and cov.source_table == "diagnoses"
    assert 0 < cov.records_mapped < cov.records
    assert cov.codes_found == cov.codes_mapped_to_standard < cov.distinct_codes
    unmatched = {u.value: u.count for u in cov.unmatched}
    assert {"J189", "J069"} <= set(unmatched)  # undotted ICD-10 codes are not in the vocabulary
    assert all(n >= cfg.privacy.min_cell_count for n in unmatched.values())
    # coverage matches the source file
    with (Path(cfg.source.path) / "diagnoses.csv").open() as fh:
        counts = Counter(r["icd10"] for r in csv.DictReader(fh))
    assert cov.records == sum(counts.values())
    assert unmatched["J189"] == counts["J189"]

    issues = " ".join(ev.issues)
    assert "condition_occurrence.condition_concept_id" in issues and "%" in issues and "J189" in issues
    assert "condition_source_concept_id" not in issues  # same codes as the standard field: reported once
    assert "FIXTURE" in issues
    md = render_vocab_evaluation_md(ev)
    assert "J189" in md and "| ICD10 | yes |" in md


@pytest.mark.postgres
def test_evaluate_vocabulary_reports_missing_vocabulary_and_bad_expression(staged):
    conn, schemas, cfg = staged
    spec = load_spec()
    cond = spec.mapping_for("condition_occurrence")
    fm = next(f for f in cond.fields if f.target_field == "condition_concept_id")
    fm.vocabulary = "ICD10CM"
    src = next(f for f in cond.fields if f.target_field == "condition_source_concept_id")
    src.expression = "d.no_such_column"
    drug = next(f for f in spec.mapping_for("drug_exposure").fields if f.target_field == "drug_concept_id")
    drug.target_domain = "Procedure"

    ev = evaluate_vocabulary(conn, schemas, spec, cfg)
    required = {r.vocabulary_id: r for r in ev.required}
    assert not required["ICD10CM"].present
    assert required["SNOMED"].present
    issues = " ".join(ev.issues)
    assert "Required vocabulary ICD10CM is not loaded" in issues
    assert "condition_source_concept_id: coverage query failed" in issues
    bad = next(c for c in ev.coverage if c.target_field == "condition_concept_id")
    assert bad.codes_found == 0 and bad.records_mapped == 0
    # the failed query did not break the connection's transaction
    assert fetch_all(conn, "SELECT 1 AS x") == [{"x": 1}]


# --------------------------------------------------------------------------------------
# collect terms and search
# --------------------------------------------------------------------------------------


@pytest.mark.postgres
def test_collect_terms(staged):
    conn, schemas, cfg = staged
    terms = collect_terms(conn, schemas, load_spec())
    assert terms == sorted(terms, key=lambda t: (t.source_vocabulary_id, -t.frequency, t.source_code))

    with (Path(cfg.source.path) / "prescriptions.csv").open() as fh:
        expected = Counter(r["drug"] for r in csv.DictReader(fh))
    drugs = {t.source_code: t for t in terms if t.source_vocabulary_id == "SRC_DRUG"}
    assert {k: t.frequency for k, t in drugs.items()} == dict(expected)
    assert all(t.target_domain == "Drug" and t.source_name == t.source_code for t in drugs.values())
    for rare in RARE[:2]:
        assert 0 < drugs[rare].frequency < cfg.privacy.min_cell_count

    by_vocab: dict[str, set[str]] = {}
    for t in terms:
        by_vocab.setdefault(t.source_vocabulary_id, set()).add(t.source_code)
    assert by_vocab["SRC_LAB"] == {"HB", "MRDT", "RBS", "WT", "TEMP", "SBP", "DBP"}
    assert by_vocab["SRC_RESULT"] == {"POS", "NEG"}  # numeric results are excluded by the expression
    assert by_vocab["SRC_UNIT"] == {"mmHg", "g/dL", "mmol/L", "C", "kg"}  # NULL units are skipped
    va = [t for t in terms if t.source_vocabulary_id == "SRC_VA_CAUSE"]
    assert {t.source_code for t in va} >= {"Malaria", "Pneumonia"} and all(t.target_domain is None for t in va)


@pytest.mark.postgres
def test_collect_terms_skips_fields_that_read_identifiers(staged, tmp_path):
    conn, schemas, cfg = staged
    spec = load_spec()
    person = spec.mapping_for("person")
    person.fields += [
        FieldMapping(target_field="race_source_concept_id", kind="concept_map", expression="i.first_name",
                     vocabulary="SRC_NAME", target_domain="Race"),
        # unqualified column, inside a function call
        FieldMapping(target_field="ethnicity_source_concept_id", kind="concept_map", expression="upper(last_name)",
                     vocabulary="SRC_SURNAME", target_domain="Ethnicity"),
    ]
    ws = Workspace(tmp_path / "work")
    terms = collect_terms(conn, schemas, spec, None, cfg.privacy, ws)
    vocabs = {t.source_vocabulary_id for t in terms}
    assert vocabs == {"SRC_DRUG", "SRC_LAB", "SRC_RESULT", "SRC_UNIT", "SRC_VA_CAUSE"}
    with (Path(cfg.source.path) / "individuals.csv").open() as fh:
        names = {v for r in csv.DictReader(fh) for v in (r["first_name"], r["last_name"], r["last_name"].upper())}
    assert not names & {t.source_code for t in terms}
    skipped = [
        "person.race_source_concept_id reads individuals.first_name",
        "person.ethnicity_source_concept_id reads individuals.last_name",
    ]
    assert pii_concept_fields(spec, None, cfg.privacy) == skipped
    assert next(e for e in ws.read_audit() if e["event"] == "vocab_pii_fields_skipped")["fields"] == skipped
    # the old positional call applies the default PII name patterns too
    assert {t.source_vocabulary_id for t in collect_terms(conn, schemas, spec)} == vocabs

    # columns listed in privacy.pii_columns, or flagged in the profile, are skipped as well; a
    # WHERE on an identifier only filters rows, its values do not become terms
    drug = spec.mapping_for("drug_exposure")
    drug.where = "p.individual_id IS NOT NULL"
    privacy = PrivacyConfig(pii_columns=["prescriptions.individual_id", "lab_results.unit"])
    profile = SourceProfile(
        source_type="csv",
        generated_at=datetime.now(timezone.utc),
        min_cell_count=5,
        tables=[
            TableProfile(
                name="deaths",
                n_rows=10,
                columns=[
                    ColumnProfile(name="va_cause", inferred_type="string", n_rows=10, n_null=0, n_distinct=3,
                                  unique=False, is_pii=True, pii_reason="test"),
                ],
            )
        ],
    )
    vocabs = {t.source_vocabulary_id for t in collect_terms(conn, schemas, spec, profile, privacy)}
    assert vocabs == {"SRC_DRUG", "SRC_LAB", "SRC_RESULT"}
    assert pii_concept_fields(spec, profile, privacy)[2:] == [
        "measurement.unit_concept_id reads lab_results.unit",
        "death.cause_concept_id reads deaths.va_cause",
    ]


@pytest.fixture(params=["exhaustive", "trigram", "ilike"])
def search_path(request, monkeypatch):
    """Run a search test through each retrieval path (the fixture's domains are all small)."""
    if request.param != "exhaustive":
        monkeypatch.setattr(mapper, "EXHAUSTIVE_DOMAIN_SIZE", 0)
    if request.param == "ilike":
        monkeypatch.setattr(mapper, "has_trigram", lambda conn: False)
    return request.param


@pytest.mark.postgres
def test_search_candidates(staged, search_path):
    conn, schemas, _ = staged
    cands = search_candidates(conn, schemas, "Paracetamol 500mg", "Drug")
    ids = [c.concept_id for c in cands]
    assert ids[0] == ACETAMINOPHEN_500_TAB  # found through the synonym 'paracetamol 500 MG Oral Tablet'
    assert ACETAMINOPHEN in ids[:3]
    assert cands[0].score > cands[1].score
    assert all(c.domain_id == "Drug" and c.standard_concept == "S" for c in cands)
    assert all(0 <= c.score <= 1 for c in cands) and len(cands) <= 10

    hb = search_candidates(conn, schemas, "HB", "Measurement")
    assert all(c.score < 0.5 for c in hb)  # an abbreviation: this is what the LLM is for

    unit = search_candidates(conn, schemas, "g/dL", "Unit", limit=3)
    assert unit[0].concept_id == GRAM_PER_DL and unit[0].score >= 0.9  # exact concept_code match
    assert len(unit) <= 3

    assert search_candidates(conn, schemas, "Malaria", None)[0].concept_id == MALARIA
    assert search_candidates(conn, schemas, "Paracetamol 500mg", "drug")[0].concept_id == ACETAMINOPHEN_500_TAB
    assert not search_candidates(conn, schemas, "Unspecified malaria", "Condition") or all(
        c.vocabulary_id != "ICD10" for c in search_candidates(conn, schemas, "Unspecified malaria", "Condition")
    )


# --------------------------------------------------------------------------------------
# map terms
# --------------------------------------------------------------------------------------


def demo_recordings(root: Path) -> None:
    record(
        root,
        "SRC_DRUG:batch1",
        [
            choice("Paracetamol 500mg", ACETAMINOPHEN_500_TAB, 0.97),
            choice("AL 20/120 tabs", 0, 0.7, retry=["artemether lumefantrine"]),
            choice("ORS sachets", ORS, 0.85, "EQUIVALENT"),
            choice("Zinc 20mg", 2000003005, 0.93),
            choice("Amoxicillin 250mg caps", 2000003003, 0.97),
            choice("Amlodipine 5mg", 2000003007, 0.97),
            choice("Cotrimoxazole 960mg", 2000003009, 0.95),
            choice("TLD (TDF/3TC/DTG)", 2999999999, 0.9),  # hallucinated id
            choice("Metformin 500mg", 2000003006, 0.93),
        ],
    )
    record(root, "SRC_DRUG:retry1", [choice("AL 20/120 tabs", ARTEMETHER_LUMEFANTRINE_TAB, 0.95)])
    record(
        root,
        "SRC_LAB:batch1",
        [
            choice("HB", 2000004001, 0.93, "EQUIVALENT"),
            choice("MRDT", 2000004002, 0.9),
            choice("RBS", 2000004003, 0.8, "EQUIVALENT"),
            choice("DBP", 2000004007, 0.95),
            choice("SBP", 2000004006, 0.95),
            choice("TEMP", 2000004005, 0.95),
            choice("WT", 2000004004, 0.95),
        ],
    )
    record(root, "SRC_RESULT:batch1", [choice("NEG", 2000006002, 0.97), choice("POS", 2000006001, 0.97)])
    record(
        root,
        "SRC_UNIT:batch1",
        [
            choice("mmHg", 2000005005),
            choice("g/dL", GRAM_PER_DL, 0.99),
            choice("mmol/L", 2000005002, 0.99),
            choice("C", 2000005004, 0.9),
            choice("kg", 2000005003, 0.99),
        ],
    )
    record(root, "SRC_VA_CAUSE:batch1", [choice("Diarrhoeal diseases", 2000001011, 0.8, "EQUIVALENT")])


@pytest.mark.postgres
def test_map_terms_with_replay(staged, tmp_path):
    conn, schemas, cfg = staged
    ws = Workspace(tmp_path / "work")
    replay = tmp_path / "replay"
    demo_recordings(replay)
    terms = collect_terms(conn, schemas, load_spec())
    decisions = map_terms(conn, schemas, terms, ReplayLLM(replay, ws), cfg, ws)
    assert [(d.source_vocabulary_id, d.source_code) for d in decisions] == [
        (t.source_vocabulary_id, t.source_code) for t in terms
    ]
    by = {(d.source_vocabulary_id, d.source_code): d for d in decisions}

    para = by["SRC_DRUG", "Paracetamol 500mg"]
    assert (para.target_concept_id, para.status, para.method) == (ACETAMINOPHEN_500_TAB, "APPROVED", "llm")
    assert para.target_concept_name == "acetaminophen 500 MG Oral Tablet" and para.target_vocabulary_id == "RxNorm"
    assert len(para.candidates) == 5

    al = by["SRC_DRUG", "AL 20/120 tabs"]  # mapped on the second search
    assert (al.target_concept_id, al.status) == (ARTEMETHER_LUMEFANTRINE_TAB, "APPROVED")
    assert al.candidates[0].concept_id == ARTEMETHER_LUMEFANTRINE_TAB

    assert by["SRC_DRUG", "ORS sachets"].status == "UNCHECKED"  # below auto_approve_confidence
    assert by["SRC_LAB", "RBS"].status == "UNCHECKED"

    tld = by["SRC_DRUG", "TLD (TDF/3TC/DTG)"]
    assert (tld.status, tld.target_concept_id) == ("FLAGGED", 0)
    assert "2999999999" in tld.rationale and "does not exist" in tld.rationale

    for code in RARE[:2]:
        d = by["SRC_DRUG", code]
        assert (d.method, d.status, d.target_concept_id) == ("suppressed", "UNCHECKED", 0)
        assert d.candidates  # local lexical candidates for the reviewer
    assert by["SRC_DRUG", "Quinine inj 600mg"].candidates[0].concept_id == 2000003010
    assert by["SRC_VA_CAUSE", "HIV/AIDS related death"].method == "suppressed"

    malaria = by["SRC_VA_CAUSE", "Malaria"]
    assert (malaria.method, malaria.status, malaria.equivalence, malaria.target_concept_id) == (
        "exact",
        "APPROVED",
        "EQUAL",
        MALARIA,
    )
    assert by["SRC_VA_CAUSE", "Road traffic accident"].method == "exact"
    assert by["SRC_UNIT", "g/dL"].target_concept_id == GRAM_PER_DL

    # privacy: exactly the expected calls, and no rare or exact-matched term in any prompt
    calls = [json.loads(p.read_text()) for p in sorted((ws.root / "llm_calls").glob("*.json"))]
    assert [c["key"] for c in calls] == [
        "SRC_DRUG:batch1",
        "SRC_DRUG:retry1",
        "SRC_LAB:batch1",
        "SRC_RESULT:batch1",
        "SRC_UNIT:batch1",
        "SRC_VA_CAUSE:batch1",
    ]
    for c in calls:
        assert c["system"] == MAPPER_SYSTEM
        for rare in RARE:
            assert rare not in c["prompt"]
        assert 'source_code: "Malaria"' not in c["prompt"]
    prompts = {c["key"]: c["prompt"] for c in calls}
    assert "2000003002 | acetaminophen 500 MG Oral Tablet | Clinical Drug | RxNorm |" in prompts["SRC_DRUG:batch1"]
    retry = prompts["SRC_DRUG:retry1"]
    assert "AL 20/120 tabs" in retry and "artemether lumefantrine" in retry and "Paracetamol" not in retry

    audit = [e for e in ws.read_audit() if e["event"] == "vocab_mapped"]
    assert {e["source_vocabulary_id"] for e in audit} == {
        "SRC_DRUG",
        "SRC_LAB",
        "SRC_RESULT",
        "SRC_UNIT",
        "SRC_VA_CAUSE",
    }
    drug_audit = next(e for e in audit if e["source_vocabulary_id"] == "SRC_DRUG")
    assert drug_audit["rejected_ids"] == 1 and drug_audit["retried"] == 1 and drug_audit["methods"]["suppressed"] == 2

    summary = {r["source_vocabulary_id"]: r for r in mapping_summary(decisions)}
    assert summary["SRC_DRUG"]["flagged"] == 1 and summary["SRC_DRUG"]["unmapped"] == 2
    assert summary["SRC_UNIT"]["pct_records_mapped"] == 100.0


@pytest.mark.postgres
def test_map_terms_validates_choices_and_batches(staged):
    conn, schemas, cfg = staged
    cfg.mapping.llm_batch_size = 2
    cfg.mapping.candidates_per_term = 2
    terms = [
        SourceTerm(
            source_vocabulary_id="SRC_X",
            source_code=code,
            source_name=code,
            frequency=freq,
            target_domain="Measurement",
        )
        for code, freq in [("SBP", 90), ("MRDT", 80), ("RBS", 70), ("WT", 60), ("DBP", 50), ("rare test", 2)]
    ]
    llm = CapturingLLM(
        {
            "SBP": 2000004006,  # valid standard Measurement concept outside the 2 candidates: accepted
            "MRDT": MALARIA,  # standard, but Condition domain: rejected
            "RBS": 2000002001,  # ICD10, not standard: rejected
            "WT": 2000004004,
        }
    )
    by = {d.source_code: d for d in map_terms(conn, schemas, terms, llm, cfg)}
    assert [k for k, _ in llm.calls] == ["SRC_X:batch1", "SRC_X:batch2", "SRC_X:batch3"]
    assert "SBP" in llm.calls[0][1] and "MRDT" in llm.calls[0][1] and "DBP" in llm.calls[2][1]
    assert all("rare test" not in p for _, p in llm.calls)
    assert all(p.count(" | LOINC | ") <= 2 * 2 for _, p in llm.calls)  # candidates_per_term

    assert by["SBP"].target_concept_id == 2000004006 and by["SBP"].target_concept_name == "Systolic blood pressure"
    assert by["MRDT"].status == "FLAGGED" and "domain Condition" in by["MRDT"].rationale
    assert by["RBS"].status == "FLAGGED" and "not a standard concept" in by["RBS"].rationale
    assert by["DBP"].target_concept_id == 0 and by["DBP"].status == "UNCHECKED" and by["DBP"].method == "llm"
    assert by["rare test"].method == "suppressed"


@pytest.mark.postgres
@pytest.mark.parametrize("which", ["null", "missing_replay"])
def test_map_terms_without_llm_falls_back_to_lexical(staged, tmp_path, which):
    conn, schemas, cfg = staged
    llm = NullLLM() if which == "null" else ReplayLLM(tmp_path / "empty")
    by = {
        (d.source_vocabulary_id, d.source_code): d
        for d in map_terms(conn, schemas, collect_terms(conn, schemas, load_spec()), llm, cfg)
    }

    gdl = by["SRC_UNIT", "g/dL"]
    assert (gdl.method, gdl.status, gdl.target_concept_id) == ("lexical", "UNCHECKED", GRAM_PER_DL)
    assert gdl.confidence == pytest.approx(0.95)
    para = by["SRC_DRUG", "Paracetamol 500mg"]
    assert (para.method, para.target_concept_id) == ("lexical", ACETAMINOPHEN_500_TAB)
    assert by["SRC_RESULT", "POS"].target_concept_id == 2000006001
    hb = by["SRC_LAB", "HB"]
    assert (hb.method, hb.target_concept_id, hb.status) == ("lexical", 0, "UNCHECKED") and "manually" in hb.rationale
    assert by["SRC_VA_CAUSE", "Malaria"].method == "exact"  # no LLM needed
    assert by["SRC_DRUG", "Quinine inj 600mg"].method == "suppressed"
    assert all(d.status != "APPROVED" for d in by.values() if d.method == "lexical")


@pytest.mark.postgres
def test_send_values_to_llm_false_never_calls_the_llm(staged, tmp_path):
    conn, schemas, cfg = staged
    cfg.privacy.send_values_to_llm = False
    ws = Workspace(tmp_path / "work")
    spy = CapturingLLM()
    replay = tmp_path / "replay"
    demo_recordings(replay)  # a recording exists for every batch, and must not be used
    terms = collect_terms(conn, schemas, load_spec())
    for llm in (spy, ReplayLLM(replay, ws)):
        by = {(d.source_vocabulary_id, d.source_code): d for d in map_terms(conn, schemas, terms, llm, cfg, ws)}
        para = by["SRC_DRUG", "Paracetamol 500mg"]
        assert (para.method, para.status, para.target_concept_id) == ("lexical", "UNCHECKED", ACETAMINOPHEN_500_TAB)
        assert "privacy.send_values_to_llm" in para.rationale
        hb = by["SRC_LAB", "HB"]
        assert (hb.method, hb.target_concept_id) == ("lexical", 0) and "privacy.send_values_to_llm" in hb.rationale
        assert by["SRC_VA_CAUSE", "Malaria"].method == "exact"  # a vocabulary lookup, not an LLM call
        assert by["SRC_DRUG", "Quinine inj 600mg"].method == "suppressed"
        assert {d.method for d in by.values()} == {"lexical", "exact", "suppressed"}
    assert spy.calls == []
    assert not (ws.root / "llm_calls").exists()
    audit = [e for e in ws.read_audit() if e["event"] == "vocab_mapped"]
    assert audit and all(e["llm_calls"] == 0 and "send_values_to_llm" in e["llm_disabled"] for e in audit)
    assert not any(e["event"] == "llm_call" for e in ws.read_audit())


@pytest.mark.postgres
def test_unmatched_choice_is_never_mapped(staged, tmp_path):
    conn, schemas, cfg = staged
    terms = [
        SourceTerm(source_vocabulary_id="SRC_X", source_code=code, source_name=code, frequency=freq,
                   target_domain="Measurement")
        for code, freq in [("SBP", 90), ("MRDT", 80)]
    ]
    unmatched = {"equivalence": "UNMATCHED", "confidence": 0.96, "rationale": "Not the same test."}
    record(
        tmp_path,
        "SRC_X:batch1",
        [
            {"source_code": "SBP", "chosen_concept_id": 2000004006, **unmatched},
            {"source_code": "MRDT", "chosen_concept_id": MALARIA, **unmatched, "retry_search_terms": ["malaria test"]},
        ],
    )
    record(tmp_path, "SRC_X:retry1", [choice("MRDT", 2000004002, 0.95)])
    by = {d.source_code: d for d in map_terms(conn, schemas, terms, ReplayLLM(tmp_path), cfg)}
    sbp = by["SBP"]
    assert (sbp.target_concept_id, sbp.target_concept_name, sbp.status, sbp.equivalence, sbp.method) == (
        0, None, "UNCHECKED", "UNMATCHED", "llm"
    )
    assert sbp.rationale.startswith("Not the same test.") and "2000004006 was not used" in sbp.rationale
    assert by["MRDT"].target_concept_id == 2000004002  # UNMATCHED with suggested phrases is searched again
    assert write_source_to_concept_map(conn, schemas, list(by.values())) == 1
    stcm = f"{schemas.etl}.source_to_concept_map"
    rows = fetch_all(conn, f"SELECT source_code FROM {stcm} WHERE source_vocabulary_id = 'SRC_X'")
    assert rows == [{"source_code": "MRDT"}]


# --------------------------------------------------------------------------------------
# source_to_concept_map and review
# --------------------------------------------------------------------------------------


def _decision(svid: str, code: str, target: int, status: str, **kw) -> MappingDecision:
    return MappingDecision(
        source_vocabulary_id=svid,
        source_code=code,
        source_name=code,
        frequency=10,
        target_concept_id=target,
        status=status,
        **kw,
    )


@pytest.mark.postgres
def test_write_source_to_concept_map(staged):
    conn, schemas, _ = staged
    stcm = f"{schemas.etl}.source_to_concept_map"
    execute(
        conn,
        f"INSERT INTO {stcm} (source_code, source_vocabulary_id, target_concept_id) VALUES "
        "('old', 'SRC_DRUG', 1), ('keep', 'SRC_OTHER', 2)",
    )
    long_code = "x" * 300
    decisions = [
        _decision("SRC_DRUG", "Paracetamol 500mg", ACETAMINOPHEN_500_TAB, "APPROVED", target_vocabulary_id="RxNorm"),
        _decision("SRC_DRUG", long_code, ACETAMINOPHEN, "UNCHECKED"),  # vocabulary looked up
        _decision("SRC_DRUG", "flagged", ORS, "FLAGGED"),
        _decision("SRC_DRUG", "rejected", ORS, "REJECTED"),
        _decision("SRC_DRUG", "unmapped", 0, "UNCHECKED"),
        _decision("SRC_UNIT", "g/dL", GRAM_PER_DL, "UNCHECKED", target_vocabulary_id="UCUM"),
    ]
    assert write_source_to_concept_map(conn, schemas, decisions) == 3
    rows = fetch_all(conn, f"SELECT * FROM {stcm} ORDER BY source_vocabulary_id, source_code")
    got = {(r["source_vocabulary_id"], r["source_code"]): r for r in rows}
    assert set(got) == {
        ("SRC_DRUG", "Paracetamol 500mg"),
        ("SRC_DRUG", "x" * 255),
        ("SRC_OTHER", "keep"),
        ("SRC_UNIT", "g/dL"),
    }
    assert got["SRC_DRUG", "x" * 255]["target_vocabulary_id"] == "RxNorm"
    para = got["SRC_DRUG", "Paracetamol 500mg"]
    assert (para["source_concept_id"], para["target_concept_id"], para["invalid_reason"]) == (
        0,
        ACETAMINOPHEN_500_TAB,
        None,
    )
    assert para["source_code_description"] == "Paracetamol 500mg"
    assert str(para["valid_start_date"]) == "1970-01-01"

    assert write_source_to_concept_map(conn, schemas, decisions, include_unchecked=False) == 1
    n = fetch_all(conn, f"SELECT COUNT(*) AS n FROM {stcm} WHERE source_vocabulary_id IN ('SRC_DRUG', 'SRC_UNIT')")
    assert n[0]["n"] == 1


def _edit_csv(path: Path, edits: dict[str, dict[str, str]]) -> None:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r.update(edits.get(r["source_code"], {}))
    with path.open("w", newline="", encoding="utf-8-sig") as fh:  # as a spreadsheet would save it
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(rows)


@pytest.mark.postgres
def test_review_round_trip(staged, tmp_path):
    conn, schemas, cfg = staged
    ws = Workspace(tmp_path / "work")
    replay = tmp_path / "replay"
    demo_recordings(replay)
    decisions = map_terms(conn, schemas, collect_terms(conn, schemas, load_spec()), ReplayLLM(replay), cfg)

    paths = export_review(ws, decisions)
    names = sorted(p.name for p in paths)
    assert names == [
        "SRC_DRUG.csv",
        "SRC_LAB.csv",
        "SRC_RESULT.csv",
        "SRC_UNIT.csv",
        "SRC_VA_CAUSE.csv",
        "decisions.json",
    ]
    drug_csv = ws.mappings_dir / "SRC_DRUG.csv"
    with drug_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == REVIEW_COLUMNS
        rows = {r["source_code"]: r for r in reader}
    assert rows["Paracetamol 500mg"]["target_concept_id"] == str(ACETAMINOPHEN_500_TAB)
    assert f"{ACETAMINOPHEN}:acetaminophen" in rows["Paracetamol 500mg"]["alternatives"]
    assert str(ACETAMINOPHEN_500_TAB) not in rows["Paracetamol 500mg"]["alternatives"]

    # unedited files import unchanged; stray copies in the folder are not read
    (ws.mappings_dir / "SRC_DRUG copy.csv").write_text(drug_csv.read_text().replace("APPROVED", "REJECTED"))
    assert import_review(ws, conn, schemas) == decisions

    _edit_csv(
        drug_csv,
        {
            "TLD (TDF/3TC/DTG)": {"target_concept_id": str(TLD_TAB)},
            "ORS sachets": {"status": "REJECTED"},
            "Zinc 20mg": {"target_concept_id": "2000002001"},  # ICD10 code: not standard
            "Metformin 500mg": {"target_concept_id": str(MALARIA)},  # wrong domain
            "Amlodipine 5mg": {"status": "approved", "target_concept_id": "0"},  # reviewer unmaps it
            "Quinine inj 600mg": {"target_concept_id": "2000003010", "status": "UNCHECKED"},  # status as before
            "Ceftriaxone 1g inj": {"target_concept_id": "2000003011", "status": "FLAGGED"},  # explicit status wins
            "Cotrimoxazole 960mg": {"status": "OK"},  # not a status
        },
    )
    imported = {d.source_code: d for d in import_review(ws, conn, schemas)}
    tld = imported["TLD (TDF/3TC/DTG)"]
    assert (tld.method, tld.status, tld.target_concept_id) == ("manual", "APPROVED", TLD_TAB)
    assert tld.target_concept_name.startswith("dolutegravir 50 MG") and tld.target_vocabulary_id == "RxNorm"
    assert imported["ORS sachets"].status == "REJECTED" and imported["ORS sachets"].method == "llm"
    assert imported["Zinc 20mg"].status == "FLAGGED" and "not a standard concept" in imported["Zinc 20mg"].rationale
    assert imported["Metformin 500mg"].status == "FLAGGED"
    aml = imported["Amlodipine 5mg"]
    assert (aml.target_concept_id, aml.status, aml.method, aml.target_concept_name) == (0, "APPROVED", "manual", None)
    quinine = imported["Quinine inj 600mg"]
    assert (quinine.status, quinine.method, quinine.target_concept_name) == (
        "APPROVED",
        "manual",
        "quinine dihydrochloride 300 MG/ML Injectable Solution",
    )
    ceftriaxone = imported["Ceftriaxone 1g inj"]
    assert (ceftriaxone.status, ceftriaxone.method, ceftriaxone.target_concept_id) == ("FLAGGED", "manual", 2000003011)
    assert ceftriaxone.target_concept_name == "ceftriaxone 1000 MG Injection"
    assert imported["Paracetamol 500mg"] == next(d for d in decisions if d.source_code == "Paracetamol 500mg")

    # without a connection the name comes from the candidates, and nothing is checked
    offline = {d.source_code: d for d in import_review(ws)}
    assert offline["Quinine inj 600mg"].target_concept_name == "quinine dihydrochloride 300 MG/ML Injectable Solution"
    assert offline["Zinc 20mg"].status == "APPROVED"

    write_source_to_concept_map(conn, schemas, list(imported.values()))
    codes = {
        r["source_code"]: r["target_concept_id"]
        for r in fetch_all(
            conn,
            f"SELECT source_code, target_concept_id FROM {schemas.etl}.source_to_concept_map "
            "WHERE source_vocabulary_id = 'SRC_DRUG'",
        )
    }
    assert codes["TLD (TDF/3TC/DTG)"] == TLD_TAB and codes["Quinine inj 600mg"] == 2000003010
    assert not {"ORS sachets", "Zinc 20mg", "Metformin 500mg", "Amlodipine 5mg", "Ceftriaxone 1g inj"} & set(codes)
    assert import_review(ws, conn, schemas) == list(imported.values())  # importing twice gives the same result
    assert (
        imported["Cotrimoxazole 960mg"].status == "FLAGGED"
        and "Unknown status 'OK'" in imported["Cotrimoxazole 960mg"].rationale
    )


def test_import_review_needs_decisions(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_review(Workspace(tmp_path / "w"))


def test_export_review_formats(tmp_path):
    ws = Workspace(tmp_path / "w")
    d = _decision(
        "SRC Weird/ID",
        "a,b",
        5,
        "UNCHECKED",
        confidence=0.123456,
        candidates=[
            ConceptCandidate(
                concept_id=5,
                concept_name="five",
                domain_id="Drug",
                vocabulary_id="RxNorm",
                concept_class_id="Ingredient",
                standard_concept="S",
                concept_code="5",
            ),
            ConceptCandidate(
                concept_id=6,
                concept_name="six",
                domain_id="Drug",
                vocabulary_id="RxNorm",
                concept_class_id="Ingredient",
                standard_concept="S",
                concept_code="6",
            ),
        ],
    )
    paths = export_review(ws, [d])
    assert paths[1].name == "SRC_Weird_ID.csv"
    with paths[1].open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["source_code"] == "a,b" and row["confidence"] == "0.12" and row["alternatives"] == "6:six"
    assert import_review(ws) == [d]
