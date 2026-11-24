"""``omop-agent llm-check``: is the configured model server ready for the pipeline?

Uses synthetic prompts only (no project data), so it can run before the privacy review. It checks:

1. every answer schema the pipeline sends is accepted by the server's constrained decoding
   (llama.cpp compiles JSON Schemas to grammars, and some constructs are refused);
2. a structured answer: a small code-mapping task with known right answers;
3. a tool call round trip, as the repair agent makes them;

and reports how long each took, which is what decides whether a model is usable on a CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..models import MappingBatchDecision, SourceAnnotations, TableMapping, TablePlan
from .base import AgentTool, LLMClient, LLMError, LLMUnavailable

PIPELINE_SCHEMAS: list[type[BaseModel]] = [SourceAnnotations, TablePlan, TableMapping, MappingBatchDecision]

MAPPING_SYSTEM = (
    "You map local clinical terms to OMOP standard concepts. For each term choose the concept_id "
    "from its candidate list that means the same thing, or 0 if none does. Answer in JSON."
)
MAPPING_PROMPT = """\
Term 1: source_code "MAL-RDT+" (label "Malaria RDT positive", 412 records, domain Condition)
  candidates:
    concept_id 4182128  "Malaria"                         Condition SNOMED  score 0.41
    concept_id 4071214  "Falciparum malaria"              Condition SNOMED  score 0.35
    concept_id 4299535  "Fever"                           Condition SNOMED  score 0.12
Term 2: source_code "PCM500" (label "PCM 500mg tab", 1250 records, domain Drug)
  candidates:
    concept_id 1125315  "acetaminophen"                   Drug RxNorm Ingredient  score 0.10
    concept_id 19020053 "acetaminophen 500 MG Oral Tablet" Drug RxNorm Clinical Drug  score 0.09
    concept_id 1112807  "aspirin"                         Drug RxNorm Ingredient  score 0.08
Term 3: source_code "XQZ" (label "XQZ", 30 records, domain Measurement)
  candidates:
    concept_id 3000963  "Hemoglobin [Mass/volume] in Blood" Measurement LOINC  score 0.05
"""
EXPECTED = {"MAL-RDT+": {4182128, 4071214}, "PCM500": {19020053}, "XQZ": {0}}


class _Count(BaseModel):
    table: str = Field(..., description="A CDM table name, e.g. condition_occurrence.")


@dataclass
class Check:
    name: str
    ok: bool
    seconds: float
    detail: str = ""


@dataclass
class CheckReport:
    provider: str
    model: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _timed(report: CheckReport, name: str, fn) -> Any:
    started = time.time()
    try:
        ok, detail, value = fn()
    except (LLMError, LLMUnavailable) as e:
        ok, detail, value = False, str(e)[:400], None
    report.checks.append(Check(name, ok, round(time.time() - started, 1), detail))
    return value


def run_checks(llm: LLMClient, schemas: bool = True) -> CheckReport:
    report = CheckReport(provider=llm.provider, model=llm.model)
    if not llm.available:
        report.checks.append(Check("model configured", False, 0.0, f"llm.provider is {llm.provider!r}"))
        return report

    if schemas and hasattr(llm, "schema_accepted"):
        for model in PIPELINE_SCHEMAS:
            _timed(report, f"schema {model.__name__} accepted", lambda m=model: (*llm.schema_accepted(m), None))
        if not all(c.ok for c in report.checks):
            return report  # the server cannot hold answers to these schemas; nothing else will work

    def mapping():
        out = llm.structured(stage="llm_check", key="mapping", system=MAPPING_SYSTEM, prompt=MAPPING_PROMPT,
                             output=MappingBatchDecision)
        got = {c.source_code: c.chosen_concept_id for c in out.choices}
        wrong = [f"{code}: chose {got.get(code)}" for code, ok_ids in EXPECTED.items() if got.get(code) not in ok_ids]
        return not wrong, ("; ".join(wrong) if wrong else "3 of 3 terms mapped as expected"), out

    _timed(report, "structured answer (code mapping)", mapping)

    def tools():
        asked: list[str] = []

        def count(a: _Count) -> str:
            asked.append(a.table)
            return "412 records" if a.table == "condition_occurrence" else "0 records"

        res = llm.run_agent(
            stage="llm_check", key="tools", system="You answer questions by calling the tools you are given.",
            prompt="How many records does the condition_occurrence table have? Use the count_records tool, "
            "then answer with the number.",
            tools=[AgentTool("count_records", "Count the records in one CDM table.", _Count, count)],
            max_iterations=4,
        )
        ok = asked[:1] == ["condition_occurrence"] and "412" in res.final_text
        return ok, f"tool calls {asked}, answer {res.final_text[:120]!r}", res

    _timed(report, "tool call round trip", tools)
    return report
