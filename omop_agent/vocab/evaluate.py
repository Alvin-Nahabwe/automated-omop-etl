"""Evaluate the loaded vocabulary against the ETL spec (the "Evaluate Vocabulary" step).

Before any mapping is done this step answers two questions:

* are the vocabularies the spec relies on loaded? (``vocab_code`` vocabularies, the usual
  target vocabulary of each ``concept_map`` domain, and the vocabularies every CDM needs);
* for every ``vocab_code`` field, how many source codes are found in the vocabulary and map
  to a standard concept, counted both as distinct codes and as source records?

Only aggregate counts are computed. Unmatched codes seen fewer than
``privacy.min_cell_count`` times are left out of the result.
"""

from __future__ import annotations

import psycopg
from pydantic import BaseModel, Field

from ..config import ProjectConfig
from ..db import Schemas, fetch_all, qi, ql, short_error
from ..models import ETLSpec, FieldMapping, TableMapping, ValueFreq
from .loader import vocabulary_inventory, vocabulary_version

COVERAGE_THRESHOLD = 0.95
MAX_UNMATCHED = 50

# concept_map target domain -> vocabularies that can hold its standard concepts (first = named)
DOMAIN_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "drug": ("RxNorm", "RxNorm Extension"),
    "measurement": ("LOINC",),
    "condition": ("SNOMED",),
    "observation": ("SNOMED",),
    "procedure": ("SNOMED",),
    "unit": ("UCUM",),
    "meas value": ("LOINC", "SNOMED"),
}
INFRASTRUCTURE_VOCABULARIES = ("Gender", "Visit", "Type Concept")


class VocabularyStatus(BaseModel):
    vocabulary_id: str
    version: str | None = None
    concepts: int
    standard_concepts: int


class RequiredVocabulary(BaseModel):
    vocabulary_id: str
    reason: str
    present: bool


class CodeCoverage(BaseModel):
    target_table: str
    target_field: str
    vocabulary_id: str
    source_table: str
    distinct_codes: int = Field(..., description="Distinct non-null source codes.")
    codes_found: int = Field(..., description="Codes that exist in the vocabulary (code + vocabulary_id).")
    codes_mapped_to_standard: int = Field(..., description="Codes with a 'Maps to' a valid standard concept.")
    records: int = Field(..., description="Source records with a non-null code.")
    records_mapped: int = Field(
        ...,
        description="Records whose code maps to a standard concept (concept_role 'standard') or is "
        "found in the vocabulary (concept_role 'source').",
    )
    unmatched: list[ValueFreq] = Field(
        default_factory=list,
        description="Codes not mapped, most frequent first; codes seen fewer than min_cell_count times are left out.",
    )
    concept_role: str = "standard"
    error: str | None = None

    @property
    def pct_records_mapped(self) -> float:
        return 100.0 * self.records_mapped / self.records if self.records else 100.0


class VocabEvaluation(BaseModel):
    vocabulary_version: str | None = None
    loaded: list[VocabularyStatus] = Field(default_factory=list)
    required: list[RequiredVocabulary] = Field(default_factory=list)
    coverage: list[CodeCoverage] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# SQL helpers shared with the mapper
# --------------------------------------------------------------------------------------


def paren(expr: str) -> str:
    """Parenthesise an expression; a trailing ``--`` comment must not swallow what follows."""
    expr = expr.strip()
    return f"({expr}\n)" if "--" in expr else f"({expr})"


def source_ref(table: str) -> str:
    """Bare table names live in ``{src}``; qualified names (or placeholders) are kept as written."""
    table = table.strip()
    if "." in table or "{" in table:
        return table
    return "{src}." + qi(table)


def source_from_sql(tm: TableMapping) -> str:
    """``FROM <source> <alias> [JOIN ...] [WHERE ...]`` of a table mapping, placeholders unrendered.

    Mirrors the FROM clause the ETL compiler generates, so counts here match what the ETL reads.
    """
    lines = [f"FROM {source_ref(tm.source.table)} {qi(tm.source.alias)}"]
    for j in tm.joins:
        jt = "INNER" if j.join_type == "inner" else "LEFT"
        lines.append(f"{jt} JOIN {source_ref(j.table)} {qi(j.alias)} ON {j.on.strip()}")
    if tm.where and tm.where.strip():
        lines.append(f"WHERE {paren(tm.where)}")
    return "\n".join(lines)


def code_counts_sql(tm: TableMapping, expression: str) -> str:
    """``code, n``: frequency of each value of ``expression`` (as text) over the mapping's rows."""
    return f"SELECT {paren(expression)}::text AS code, COUNT(*) AS n\n{source_from_sql(tm)}\nGROUP BY 1"


def query_rows(conn: psycopg.Connection, sql: str) -> list[dict]:
    """Run a read-only statement in a savepoint that is always rolled back.

    An error (a bad expression in the spec, say) leaves the caller's transaction usable. The
    statement is sent without parameters, so ``%`` in spec expressions needs no escaping.
    """
    with conn.transaction(force_rollback=True):
        return fetch_all(conn, sql)


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------


def _vocab_fields(spec: ETLSpec, kind: str) -> list[tuple[TableMapping, FieldMapping]]:
    out = []
    for tm in spec.table_mappings:
        for fm in tm.fields:
            if fm.kind == kind and (fm.expression or "").strip() and (fm.vocabulary or "").strip():
                out.append((tm, fm))
    return out


def _required(spec: ETLSpec, present: set[str]) -> list[RequiredVocabulary]:
    reasons: dict[str, list[str]] = {}
    accepted: dict[str, tuple[str, ...]] = {}

    def need(vocab: str, reason: str, alternatives: tuple[str, ...] = ()) -> None:
        lst = reasons.setdefault(vocab, [])
        if reason not in lst:
            lst.append(reason)
        accepted[vocab] = tuple(dict.fromkeys(accepted.get(vocab, ()) + (vocab,) + alternatives))

    for tm, fm in _vocab_fields(spec, "vocab_code"):
        need(fm.vocabulary.strip(), f"codes in {tm.target_table}.{fm.target_field}")  # type: ignore[union-attr]
    for tm, fm in _vocab_fields(spec, "concept_map"):
        domain = (fm.target_domain or "").strip()
        vocabs = DOMAIN_VOCABULARIES.get(domain.lower())
        if vocabs:
            alt = f"; {' or '.join(vocabs)} accepted" if len(vocabs) > 1 else ""
            need(
                vocabs[0],
                f"{domain} targets for {fm.vocabulary} ({tm.target_table}.{fm.target_field}){alt}",
                vocabs[1:],
            )
    for vocab in INFRASTRUCTURE_VOCABULARIES:
        need(vocab, "OMOP conventions")

    return [
        RequiredVocabulary(
            vocabulary_id=v,
            reason="; ".join(rs),
            present=any(a in present for a in accepted[v]),
        )
        for v, rs in reasons.items()
    ]


def _coverage(
    conn: psycopg.Connection, schemas: Schemas, tm: TableMapping, fm: FieldMapping, min_cell_count: int
) -> CodeCoverage:
    vocab = fm.vocabulary.strip()  # type: ignore[union-attr]
    v = "{vocab}"
    sql = f"""
WITH codes AS (
{code_counts_sql(tm, fm.expression or "")}
)
SELECT k.code, k.n,
       EXISTS (SELECT 1 FROM {v}.concept c
               WHERE c.concept_code = k.code AND c.vocabulary_id = {ql(vocab)}) AS found,
       EXISTS (SELECT 1 FROM {v}.concept c
               JOIN {v}.concept_relationship r ON r.concept_id_1 = c.concept_id
                    AND r.relationship_id = 'Maps to' AND r.invalid_reason IS NULL
               JOIN {v}.concept t ON t.concept_id = r.concept_id_2
                    AND t.standard_concept = 'S' AND t.invalid_reason IS NULL
               WHERE c.concept_code = k.code AND c.vocabulary_id = {ql(vocab)}) AS mapped
FROM codes k
WHERE k.code IS NOT NULL
ORDER BY k.n DESC, k.code"""
    base = dict(
        target_table=tm.target_table,
        target_field=fm.target_field,
        vocabulary_id=vocab,
        source_table=tm.source.table,
        concept_role=fm.concept_role,
    )
    try:
        rows = query_rows(conn, schemas.render(sql))
    except psycopg.Error as e:
        return CodeCoverage(
            **base,
            distinct_codes=0,
            codes_found=0,
            codes_mapped_to_standard=0,
            records=0,
            records_mapped=0,
            error=short_error(e),
        )
    ok_key = "mapped" if fm.concept_role == "standard" else "found"
    unmatched = [ValueFreq(value=r["code"], count=int(r["n"])) for r in rows if not r[ok_key]]
    return CodeCoverage(
        **base,
        distinct_codes=len(rows),
        codes_found=sum(1 for r in rows if r["found"]),
        codes_mapped_to_standard=sum(1 for r in rows if r["mapped"]),
        records=sum(int(r["n"]) for r in rows),
        records_mapped=sum(int(r["n"]) for r in rows if r[ok_key]),
        unmatched=[u for u in unmatched if u.count >= min_cell_count][:MAX_UNMATCHED],
    )


def _coverage_issue(cov: CodeCoverage) -> str:
    what = "map to a standard concept" if cov.concept_role == "standard" else "are found in the vocabulary"
    msg = (
        f"{cov.target_table}.{cov.target_field}: {cov.pct_records_mapped:.1f}% of {cov.records:,} records "
        f"({cov.vocabulary_id} codes from {cov.source_table}) {what}"
    )
    if cov.unmatched:
        top = ", ".join(f"{u.value} ({u.count:,})" for u in cov.unmatched[:5])
        msg += f"; most frequent unmatched codes: {top}"
    return msg + ". Check the code format (dots, case, padding) or map these codes with concept_map."


def evaluate_vocabulary(
    conn: psycopg.Connection, schemas: Schemas, spec: ETLSpec, cfg: ProjectConfig
) -> VocabEvaluation:
    """Check vocabulary availability and ``vocab_code`` coverage for ``spec``."""
    ev = VocabEvaluation()
    try:
        with conn.transaction():
            inventory = vocabulary_inventory(conn, schemas)
    except psycopg.Error as e:
        inventory = []
        ev.issues.append(f"Vocabulary tables could not be read in schema {schemas.vocab}: {short_error(e)}")
    ev.vocabulary_version = vocabulary_version(conn, schemas)
    ev.loaded = [
        VocabularyStatus(
            vocabulary_id=r["vocabulary_id"],
            version=r["vocabulary_version"],
            concepts=int(r["concepts"]),
            standard_concepts=int(r["standard_concepts"]),
        )
        for r in inventory
        if r["vocabulary_id"] != "None" and int(r["concepts"]) > 0
    ]
    present = {v.vocabulary_id for v in ev.loaded}
    ev.required = _required(spec, present)

    fields = _vocab_fields(spec, "vocab_code")
    ev.coverage = [_coverage(conn, schemas, tm, fm, cfg.privacy.min_cell_count) for tm, fm in fields]

    # ---- issues ----------------------------------------------------------------------
    if inventory and not ev.loaded:
        ev.issues.append(f"No concepts are loaded in {schemas.vocab}.concept.")
    for req in ev.required:
        if not req.present:
            ev.issues.append(
                f"Required vocabulary {req.vocabulary_id} is not loaded ({req.reason}). "
                "Add it to the Athena download and reload the vocabulary."
            )
    # A 'source' field that reads the same codes as a 'standard' field would repeat its message.
    keys = [
        (c.source_table, c.vocabulary_id, " ".join((fm.expression or "").split()))
        for c, (_, fm) in zip(ev.coverage, fields)
    ]
    standard_keys = {k for k, c in zip(keys, ev.coverage) if c.concept_role == "standard"}
    for key, cov in zip(keys, ev.coverage):
        if cov.error:
            ev.issues.append(f"{cov.target_table}.{cov.target_field}: coverage query failed: {cov.error}")
        elif cov.concept_role == "source" and key in standard_keys:
            continue
        elif cov.records and cov.records_mapped < COVERAGE_THRESHOLD * cov.records:
            ev.issues.append(_coverage_issue(cov))
    if ev.vocabulary_version is None:
        ev.issues.append("Vocabulary version is unknown (no 'None' row in the vocabulary table).")
    elif "fixture" in ev.vocabulary_version.lower():
        ev.issues.append(
            f"Vocabulary version is '{ev.vocabulary_version}': the synthetic demo fixture. "
            "Never use it for a production CDM; load an Athena download instead."
        )
    return ev


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_vocab_evaluation_md(ev: VocabEvaluation) -> str:
    """Human-readable summary of a :class:`VocabEvaluation`."""
    out = ["# Vocabulary evaluation", ""]
    out.append(f"Vocabulary version: **{_cell(ev.vocabulary_version or 'unknown')}**")
    out.append("")

    out += ["## Issues", ""]
    out += [f"- {_cell(i)}" for i in ev.issues] or ["No issues found."]
    out.append("")

    out += ["## Required vocabularies", "", "| Vocabulary | Loaded | Needed for |", "|---|---|---|"]
    for r in ev.required:
        out.append(f"| {_cell(r.vocabulary_id)} | {'yes' if r.present else '**no**'} | {_cell(r.reason)} |")
    out.append("")

    out += ["## Code coverage", ""]
    if not ev.coverage:
        out += ["The ETL spec has no `vocab_code` fields.", ""]
    else:
        out += [
            "| Field | Vocabulary | Source table | Distinct codes | Found | Mapped to standard | Records | Records mapped |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for c in ev.coverage:
            role = "" if c.concept_role == "standard" else " (source)"
            recs = "error" if c.error else f"{c.records_mapped:,} ({c.pct_records_mapped:.1f}%)"
            out.append(
                f"| {_cell(c.target_table)}.{_cell(c.target_field)}{role} | {_cell(c.vocabulary_id)} | "
                f"{_cell(c.source_table)} | {c.distinct_codes:,} | {c.codes_found:,} | "
                f"{c.codes_mapped_to_standard:,} | {c.records:,} | {recs} |"
            )
        out.append("")
        for c in ev.coverage:
            if c.unmatched:
                out.append(
                    f"Unmatched codes in {_cell(c.target_table)}.{_cell(c.target_field)}: "
                    + ", ".join(f"`{_cell(u.value)}` ({u.count:,})" for u in c.unmatched)
                )
                out.append("")

    out += ["## Loaded vocabularies", "", "| Vocabulary | Version | Concepts | Standard |", "|---|---|---:|---:|"]
    for v in ev.loaded:
        out.append(
            f"| {_cell(v.vocabulary_id)} | {_cell(v.version or '')} | {v.concepts:,} | {v.standard_concepts:,} |"
        )
    out.append("")
    return "\n".join(out)
