"""The mapping memory: reviewed mappings kept across projects, reused and retrieved."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from omop_agent.config import MappingConfig, load_config
from omop_agent.llm import LLMClient
from omop_agent.models import MappingBatchDecision, MappingDecision, SourceTerm
from omop_agent.vocab.mapper import map_terms
from omop_agent.vocab.memory import MappingMemory, human_approved, meaningful, name_key
from omop_agent.vocab.review import recheck

AL_TAB = 2000003001  # artemether 20 MG / lumefantrine 120 MG Oral Tablet
PCM_TAB = 2000003002  # acetaminophen 500 MG Oral Tablet
MALARIA = 2000001001  # Condition
ICD10_MALARIA = 2000002001  # not standard
WEIGHT = 2000004004
SBP = 2000004006


def decision(code: str, name: str, cid: int, *, status: str = "APPROVED", method: str = "llm",
             reviewed_at: str | None = "2026-11-01T10:00:00Z", freq: int = 50, svid: str = "SRC_DRUG",
             domain: str = "Drug") -> MappingDecision:
    return MappingDecision(
        source_vocabulary_id=svid, source_code=code, source_name=name, frequency=freq, target_domain=domain,
        target_concept_id=cid, target_concept_name=f"concept {cid}", target_domain_id=domain,
        equivalence="EQUAL" if cid else "UNMATCHED", status=status, method=method,
        reviewed_by="amina" if reviewed_at else None, reviewed_at=reviewed_at,
    )


# --------------------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------------------


def test_only_decisions_a_person_approved_are_remembered():
    assert human_approved(decision("A", "a", 1))                                        # approved in the web app
    assert human_approved(decision("A", "a", 1, method="manual", reviewed_at=None))     # a reviewer's concept
    assert not human_approved(decision("A", "a", 1, reviewed_at=None))                  # automatic approval
    assert not human_approved(decision("A", "a", 1, method="memory", reviewed_at=None))  # reused, not re-reviewed
    assert not human_approved(decision("A", "a", 1, status="UNCHECKED"))
    assert not human_approved(decision("A", "a", 0, method="manual"))                   # no concept


def test_meaningful_descriptions():
    assert meaningful(name_key("AL 20/120 tabs")) and meaningful(name_key("MAL-RDT+"))
    assert not meaningful(name_key("1")) and not meaningful(name_key("M")) and not meaningful(name_key("2b"))


def test_record_replaces_a_projects_entries(tmp_path):
    mem = MappingMemory(tmp_path / "sub" / "memory.sqlite")
    assert mem.stats() == {"mappings": 0, "projects": 0, "concepts": 0}  # no file yet
    n = mem.record("/p/one", "one", [
        decision("AL", "AL 20/120 tabs", AL_TAB),
        decision("PCM", "Paracetamol 500mg tab", PCM_TAB, reviewed_at=None),  # automatic: not remembered
        decision("RARE", "Quinine inj 600mg", 99, freq=2),                     # below the small-cell threshold
    ], min_count=5)
    assert n == 1
    mem.record("/p/two", "two", [decision("PCM", "Paracetamol 500mg tab", PCM_TAB, method="manual")])
    assert mem.stats() == {"mappings": 2, "projects": 2, "concepts": 2}
    # the reviewer of project one changed their mind: its entries are replaced, two's are kept
    mem.record("/p/one", "one", [decision("AL", "AL 20/120 tabs", AL_TAB, status="REJECTED")])
    fresh = MappingMemory(mem.path)
    assert fresh.stats() == {"mappings": 1, "projects": 1, "concepts": 1}
    assert [r.target_concept_id for r in fresh.same("paracetamol 500 MG tab")] == [PCM_TAB]
    with sqlite3.connect(mem.path) as conn:
        assert conn.execute("SELECT project, reviewed_by FROM mapping").fetchall() == [("/p/two", "amina")]


def test_same_and_similar(tmp_path):
    mem = MappingMemory(tmp_path / "m.sqlite")
    mem.record("/p/a", "a", [
        decision("AL", "AL 20/120 tabs", AL_TAB),
        decision("PCM", "Paracetamol 500mg tablet", PCM_TAB),
        decision("1", "1", WEIGHT, svid="SRC_SEX"),
    ])
    assert [r.source_code for r in mem.same("al 20/120 TABS")] == ["AL"]
    assert mem.same("1") == [] and mem.similar("1") == []  # local codes are never matched
    sim = mem.similar("Paracetamol 500 mg tabs")
    assert [r.target_concept_id for r in sim] == [PCM_TAB] and 0.6 <= sim[0].score < 1
    assert mem.similar("ceftriaxone injection") == []


def test_memory_path_config(tmp_path):
    assert MappingConfig(memory_path="").memory_path is None
    (tmp_path / "p.yaml").write_text(
        "name: x\nsource: {type: csv, path: .}\ntarget: {url: 'postgresql://u'}\n"
        "mapping: {memory_path: shared/memory.sqlite}\n"
    )
    assert load_config(tmp_path / "p.yaml").mapping.memory_path == tmp_path / "shared" / "memory.sqlite"


def test_recheck_treats_memory_approvals_as_automatic():
    d = decision("AL", "AL 20/120 tabs", AL_TAB, method="memory", reviewed_at=None)
    d.accepted_domains = ["Drug"]
    assert recheck(d, auto_approve=False).status == "UNCHECKED"


# --------------------------------------------------------------------------------------
# mapping with the memory
# --------------------------------------------------------------------------------------


class CapturingLLM(LLMClient):
    provider = "capture"

    def __init__(self, answers: dict[str, int] | None = None):
        super().__init__(None)
        self.prompts: list[str] = []
        self.answers = answers or {}

    def structured(self, *, stage, key, system, prompt, output):
        self.prompts.append(prompt)
        codes = [json.loads(x.split("source_code: ", 1)[1]) for x in prompt.splitlines() if "source_code: " in x]
        return MappingBatchDecision.model_validate({"choices": [
            {"source_code": c, "chosen_concept_id": self.answers.get(c, 0),
             "equivalence": "EQUAL" if self.answers.get(c) else "UNMATCHED", "confidence": 0.95, "rationale": "t"}
            for c in codes]})

    def run_agent(self, **kw):  # pragma: no cover - not used
        raise NotImplementedError


def term(code: str, name: str, freq: int = 40, domain: str = "Drug", svid: str = "SRC_RX", **kw) -> SourceTerm:
    return SourceTerm(source_vocabulary_id=svid, source_code=code, source_name=name, frequency=freq,
                      target_domain=domain, **kw)


@pytest.mark.postgres
def test_a_term_reviewers_mapped_before_is_reused(staged, tmp_path):
    conn, schemas, cfg = staged
    mem = MappingMemory(tmp_path / "m.sqlite")
    mem.record("/p/earlier", "Kisumu HDSS", [
        decision("AL", "Coartem tabs", AL_TAB),
        decision("PCM", "Panadol", PCM_TAB),
        decision("PCM2", "Panadol", AL_TAB),              # reviewers disagree: not reused
        decision("MAL", "Fever tabs", MALARIA, domain="Condition"),   # another domain: not reused for a drug
        decision("ICD", "Old code tabs", ICD10_MALARIA),  # no longer a standard concept
        decision("1", "1", AL_TAB),                       # a bare local code
    ])
    llm = CapturingLLM()
    terms = [term("X1", "coartem TABS"), term("X2", "Panadol"), term("X3", "Fever tabs"), term("X4", "Old code tabs"),
             term("1", "1"), term("X5", "Coartem tabs", freq=2)]
    by = {d.source_code: d for d in map_terms(conn, schemas, terms, llm, cfg, memory=mem)}

    for code in ("X1", "X5"):  # reused without the model, even below the small-cell threshold
        d = by[code]
        assert (d.method, d.target_concept_id, d.status, d.confidence) == ("memory", AL_TAB, "APPROVED", 1.0)
        assert "Kisumu HDSS by amina on 2026-11-01" in d.rationale
        assert d.candidates[0].concept_id == AL_TAB
    for code in ("X2", "X3", "X4", "1"):
        assert by[code].method == "llm"
    sent = "\n".join(llm.prompts)
    assert "coartem" not in sent.lower()


@pytest.mark.postgres
def test_reused_concepts_wait_for_a_reviewer_when_configured(staged, tmp_path):
    conn, schemas, cfg = staged
    cfg.mapping.memory_auto_approve = False
    mem = MappingMemory(tmp_path / "m.sqlite")
    mem.record("/p/a", "a", [decision("AL", "Coartem tabs", AL_TAB)])
    mixed = term("X1", "Coartem tabs", accepted_domains=["Drug", "Observation"], auto_approve=False)
    [d] = map_terms(conn, schemas, [mixed], CapturingLLM(), cfg, memory=mem)
    assert (d.method, d.status) == ("memory", "UNCHECKED")


@pytest.mark.postgres
@pytest.mark.parametrize("memory_text", [True, False])
def test_similar_reviewed_terms_are_retrieved_for_the_model(staged, tmp_path, memory_text):
    conn, schemas, cfg = staged
    cfg.mapping.candidates_per_term = 2
    mem = MappingMemory(tmp_path / "m.sqlite")
    mem.record("/p/a", "a", [decision("AL", "Artemether lumefantrine 20/120 tabs", AL_TAB)])
    llm = CapturingLLM({"AL6": AL_TAB})
    [d] = map_terms(conn, schemas, [term("AL6", "Artemether lumefantrine tabs x6")], llm, cfg, memory=mem,
                    memory_text=memory_text)
    [prompt] = llm.prompts
    assert "approved by reviewers in earlier projects:" in prompt
    assert f"-> {AL_TAB} | artemether 20 MG / lumefantrine 120 MG Oral Tablet | Drug" in prompt
    assert ('"Artemether lumefantrine 20/120 tabs" ->' in prompt) is memory_text
    assert "a similar term ->" in prompt if not memory_text else True
    assert prompt.count(f"   {AL_TAB} | ") == 1  # among the (2) candidates shown
    assert (d.method, d.target_concept_id) == ("llm", AL_TAB)


@pytest.mark.postgres
def test_pipeline_records_reviewed_mappings(staged, tmp_path):
    from omop_agent.pipeline import Pipeline

    conn, schemas, cfg = staged
    cfg.mapping.memory_path = tmp_path / "shared" / "m.sqlite"
    p = Pipeline(cfg, log=lambda _m: None)
    assert p.remember_mappings([decision("AL", "Coartem tabs", AL_TAB), decision("PCM", "Panadol", PCM_TAB,
                                                                               reviewed_at=None)]) == 1
    st = MappingMemory(cfg.mapping.memory_path).stats()
    assert st == {"mappings": 1, "projects": 1, "concepts": 1}
    cfg.mapping.memory_path = tmp_path  # a directory: cannot be opened as a database
    assert p.remember_mappings([decision("AL", "Coartem tabs", AL_TAB)]) is None
    p.close()
