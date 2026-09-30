"""Draft the ETL spec with Claude (the "Rabbit-in-a-Hat" stage), then check it with code.

The design runs in two passes. A plan call decides which source tables feed which CDM tables,
the person and visit keys, and the observation-period strategy. Then one call per planned table
returns a ``TableMapping``. Each mapping is validated (``design/validate.py``: CDM structure,
aliases, source columns, concept ids, ``EXPLAIN``), and mappings with errors go back to the
model with the validator's findings, up to ``max_revisions`` times. Problems that remain are
recorded for the human reviewer rather than raised.

Prompts carry aggregate profile data only (``privacy.profile_prompt_text``).
"""

from __future__ import annotations

import inspect
import json

import psycopg
import yaml

from ..cdm.spec import load_cdm_spec
from ..config import ProjectConfig
from ..db import Schemas, fetch_all, qi, short_error
from ..llm import LLMClient, LLMUnavailable
from ..models import (
    ETLSpec,
    FieldMapping,
    KeySpec,
    PlannedTable,
    SourceAnnotations,
    SourceProfile,
    TableMapping,
    TablePlan,
)
from ..privacy import profile_prompt_text
from ..workspace import Workspace, utcnow
from .render import render_spec_md
from .validate import SpecIssue, format_issues, validate_spec, validate_table_mapping

REFERENCE_DOMAINS = ["Gender", "Race", "Ethnicity", "Visit", "Type Concept"]

# CDM tables a source extract is normally mapped into (eras, metadata and cdm_source are
# derived or written by the build itself).
PLANNABLE_TABLES = [
    "person",
    "observation_period",
    "visit_occurrence",
    "visit_detail",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "device_exposure",
    "measurement",
    "observation",
    "death",
    "note",
    "specimen",
    "location",
    "care_site",
    "provider",
    "payer_plan_period",
    "cost",
]

DESIGN_SYSTEM = """\
You are an OHDSI ETL architect. You design how a source health dataset is converted into the \
OMOP Common Data Model (CDM) version 5.4, in the way a data team would with Rabbit-in-a-Hat. \
Your design is written as structured data that is compiled into PostgreSQL, validated by code, \
and reviewed by people before it runs, so precision matters more than coverage.

What you can see: an aggregate profile of the source (table and column names, types, counts, \
ranges and frequent values), an interpretation of that profile, the CDM table definitions, \
and reference concepts from the loaded vocabulary. You never see patient rows; values seen \
fewer than the small-cell threshold and every value of identifying columns (names, phone \
numbers, ids) are withheld. Do not guess at withheld values; design logic that works for any \
value in the column.

OMOP conventions to follow (OHDSI THEMIS):
- One person row per individual. person_id is never taken from the source directly: fields \
of kind person_id resolve the source patient key through a person id map, and rows whose key \
has no person are dropped.
- Every event row records its provenance in a *_type_concept_id field (for example 32817 \
"EHR"). Choose type concepts from the reference list.
- Concept fields hold standard concepts from the domain the CDM field expects. Use 0 ("No \
matching concept") when a source value exists but cannot be mapped. Only use concept ids that \
appear in the reference list; anything else must come from a lookup (concept_map or \
vocab_code), never from memory.
- Keep the verbatim source value in the matching *_source_value field, and fill \
*_source_concept_id from the source vocabulary when a vocab_code lookup is used (0 otherwise).
- Required fields must never be NULL. When a required date can be missing in the source, \
derive a sensible fallback (for example an end date equal to the start date) and say so in \
the rationale.
- Observation periods are the spans during which a person's events would be recorded. Use a \
source table of enrollment, residency or follow-up episodes when one exists; otherwise derive \
them from the events.
- Records go to the table of their content's domain: diagnoses to condition_occurrence, \
prescriptions and dispensings to drug_exposure, lab results and vital signs to measurement, \
other facts to observation.

Answer with the requested JSON structure only. Write SQL for PostgreSQL. Put the reasoning a \
reviewer needs into the rationale fields, briefly.
"""

STAGING_CONVENTIONS = """\
Staging conventions for the SQL you write:
- Every source table has been copied to schema {src} with all columns typed text. Cast \
explicitly: ::date or ::timestamp for ISO dates, to_date(x, '<format>') for other date \
formats shown in the profile, ::integer or ::numeric for numbers.
- Empty source values were loaded as NULL.
- Refer to columns through the aliases you declare in source.alias and joins[].alias (for \
example v.visit_date::date). Join other source tables by their bare name; use the placeholders \
{src}, {cdm}, {vocab}, {etl} if you need another schema.
- Constant concept ids must come from the reference concept list below, or be 0.
- Use kind concept_map for local codes, labels and free-text categories. Name the \
vocabulary 'SRC_<NAME>' (for example 'SRC_DRUG') and set target_domain to the OMOP domain of \
the concepts they should map to; the codes are mapped to standard concepts in a later step.
- Use kind vocab_code for codes that already belong to an OMOP vocabulary (ICD10, ICD10CM, \
LOINC, RxNorm, SNOMED, ...). The expression must produce the code exactly as it appears in \
concept.concept_code, so normalise the format in SQL when the profile shows it differs (for \
example, add the dot to ICD-10 codes written without one).
- Map every required field. The primary key is normally kind auto_id; visit_occurrence_id in \
visit_occurrence uses kind visit_id with the visit key so that other tables can link to it. \
person_id always uses kind person_id with the column holding the patient key.
- Use where to keep only the rows that belong in this table when one source table feeds \
several CDM tables.
"""


# --------------------------------------------------------------------------------------
# prompt context
# --------------------------------------------------------------------------------------


def reference_concepts_text(conn: psycopg.Connection | None, schemas: Schemas, limit_per_domain: int = 80) -> str:
    """Standard, valid concepts of the Gender, Race, Ethnicity, Visit and Type Concept domains."""
    if conn is None:
        return "(no vocabulary connection: use 0 for concept constants)"
    sql = f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id FROM (
            SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id,
                   ROW_NUMBER() OVER (PARTITION BY c.domain_id ORDER BY c.concept_id) AS rn
            FROM {qi(schemas.vocab)}.concept c
            WHERE c.domain_id = ANY(%(domains)s)
              AND c.standard_concept = 'S' AND c.invalid_reason IS NULL
        ) x
        WHERE rn <= %(limit)s
        ORDER BY array_position(%(domains)s::text[], domain_id), concept_id"""
    try:
        with conn.transaction():
            rows = fetch_all(conn, sql, {"domains": REFERENCE_DOMAINS, "limit": limit_per_domain})
    except psycopg.Error as e:
        return f"(reference concepts unavailable: {short_error(e)})"
    if not rows:
        return "(no reference concepts found in the vocabulary)"
    lines = ["concept_id | concept_name | domain | vocabulary"]
    lines += [f"{r['concept_id']} | {r['concept_name']} | {r['domain_id']} | {r['vocabulary_id']}" for r in rows]
    return "\n".join(lines)


def _vocabularies_text(conn: psycopg.Connection | None, schemas: Schemas) -> str:
    if conn is None:
        return "(unknown)"
    try:
        with conn.transaction():
            rows = fetch_all(
                conn,
                f"SELECT vocabulary_id FROM {qi(schemas.vocab)}.vocabulary "
                f"WHERE vocabulary_id <> 'None' ORDER BY vocabulary_id",
            )
    except psycopg.Error:
        return "(unknown)"
    return ", ".join(r["vocabulary_id"] for r in rows) or "(none)"


def _annotations_text(annotations: SourceAnnotations | None, tables: list[str] | None = None) -> str:
    if annotations is None:
        return "(no interpretation available)"
    keep = None if tables is None else {t.lower() for t in tables}
    lines: list[str] = []
    for t in annotations.tables:
        if keep is not None and t.table.lower() not in keep:
            continue
        bits = [f"grain: {t.grain}"]
        if t.person_key:
            bits.append(f"person key: {t.person_key}")
        if t.candidate_cdm_tables:
            bits.append("candidate CDM tables: " + ", ".join(t.candidate_cdm_tables))
        lines.append(f"TABLE {t.table}: {t.description} ({'; '.join(bits)})")
        for c in annotations.columns:
            if c.table.lower() != t.table.lower():
                continue
            coding = f", coding system {c.coding_system}" if c.coding_system else ""
            lines.append(f"  - {c.column}: {c.semantic_type}{coding}. {c.description}")
    if annotations.notes and tables is None:
        lines.append("NOTES:")
        lines += [f"  - {n}" for n in annotations.notes]
    return "\n".join(lines) or "(none)"


def _cdm_tables_text() -> str:
    cdm = load_cdm_spec()
    lines = []
    for name in PLANNABLE_TABLES:
        t = cdm.table(name)
        first = t.description.replace("\n", " ").split(". ")[0].strip().rstrip(".")
        lines.append(f"- {name}: {first[:220]}.")
    return "\n".join(lines)


def _keys_text(person_key: KeySpec, visit_key: KeySpec | None) -> str:
    lines = [
        f"Person key: {person_key.table}.{person_key.expression}. Fill person_id with kind person_id "
        "and the column that holds this patient key in the table you are mapping."
    ]
    if visit_key is not None:
        lines.append(
            f"Visit key: {visit_key.table}.{visit_key.expression}. Fill visit_occurrence_id with kind "
            "visit_id and the column that holds this visit key (NULL when a record has no visit)."
        )
    else:
        lines.append("There is no visit key: leave visit_occurrence_id fields unmapped.")
    return "\n".join(lines)


def _plan_prompt(profile_text: str, annotations_text: str, reference: str) -> str:
    return f"""\
Plan the ETL of the source dataset below into OMOP CDM v5.4.

Decide:
1. person_key: the source table that lists each individual once, and the SQL expression of \
the patient key over that table's bare column names.
2. visit_key: the source table of visits/encounters and its key expression, if there is one.
3. observation_period: 'table_mapping' if a source table records enrollment, residency or \
follow-up periods (then plan an observation_period table from it), otherwise 'from_events'.
4. tables: one entry per table mapping, in load order (person first). For each, the CDM \
target table and its source tables (the first one provides the rows; any others are joined). \
A CDM table may appear more than once when several source tables feed it. Only plan tables \
the data supports.
5. unmapped_source_tables: source tables you leave out, with the reason in open_questions.
6. assumptions you made and open_questions for the data owner.

SOURCE PROFILE (aggregates only)
{profile_text}
INTERPRETATION OF THE PROFILE
{annotations_text}

CDM TABLES YOU CAN TARGET
{_cdm_tables_text()}

REFERENCE CONCEPTS (standard, valid)
{reference}
"""


def _table_prompt(
    pt: PlannedTable,
    plan: TablePlan,
    profile: SourceProfile,
    annotations: SourceAnnotations | None,
    cfg: ProjectConfig,
    reference: str,
    vocabularies: str,
) -> str:
    cdm = load_cdm_spec()
    others = [f"{t.target_table} <- {', '.join(t.source_tables)}" for t in plan.tables if t is not pt]
    return f"""\
Write the table mapping that loads CDM table {pt.target_table} from source table(s) \
{', '.join(pt.source_tables)}. The first source table is the driving table (source); join the \
others if you need them.

Why this table was planned: {pt.rationale}
Other planned mappings: {'; '.join(others) or 'none'}
Observation period strategy: {plan.observation_period}

{_keys_text(plan.person_key, plan.visit_key)}

TARGET TABLE
{cdm.describe(pt.target_table)}

SOURCE PROFILE (aggregates only)
{profile_prompt_text(profile, cfg.privacy, tables=pt.source_tables)}
INTERPRETATION
{_annotations_text(annotations, pt.source_tables)}

FIELD KINDS
{inspect.cleandoc(FieldMapping.__doc__ or '')}

{STAGING_CONVENTIONS}
VOCABULARIES LOADED: {vocabularies}

REFERENCE CONCEPTS (standard, valid)
{reference}
"""


def _revision_prompt(base_prompt: str, previous: TableMapping, issues: list[SpecIssue]) -> str:
    prev = yaml.safe_dump(previous.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=110)
    return f"""{base_prompt}
YOUR PREVIOUS MAPPING FOR THIS TABLE
{prev}
The validator found these problems in it:
{format_issues(issues)}

Return the complete corrected table mapping. Keep the parts that were fine.
"""


# --------------------------------------------------------------------------------------
# design flow
# --------------------------------------------------------------------------------------


def _need_llm(e: Exception) -> RuntimeError:
    return RuntimeError(
        "Designing the ETL needs an LLM: set llm.provider to 'anthropic' (with an API key) or to "
        "'replay' with recorded answers, or write etl_spec.yaml by hand and skip the design stage. "
        f"({e})"
    )


def _normalise_plan(plan: TablePlan) -> TablePlan:
    """Keep only CDM targets, make sure person is planned, and a consistent period strategy."""
    cdm = load_cdm_spec()
    tables: list[PlannedTable] = []
    notes: list[str] = []
    for pt in plan.tables:
        name = pt.target_table.strip().lower()
        if not cdm.has_table(name) or cdm.table(name).schema_class != "CDM":
            notes.append(f"The plan named '{pt.target_table}', which is not a CDM v5.4 table; it was skipped.")
            continue
        if not pt.source_tables:
            notes.append(f"The plan for {name} named no source table; it was skipped.")
            continue
        tables.append(pt.model_copy(update={"target_table": name}))
    if not any(t.target_table == "person" for t in tables):
        tables.insert(
            0,
            PlannedTable(
                target_table="person",
                source_tables=[plan.person_key.table],
                rationale="Every CDM needs a person table; it is loaded from the person key table.",
            ),
        )
    update: dict = {"tables": tables, "open_questions": plan.open_questions + notes}
    if plan.observation_period == "table_mapping" and not any(t.target_table == "observation_period" for t in tables):
        update["observation_period"] = "from_events"
        update["assumptions"] = plan.assumptions + [
            "No observation_period mapping was planned, so observation periods are derived from events."
        ]
    return plan.model_copy(update=update)


def design_etl(
    profile: SourceProfile,
    annotations: SourceAnnotations | None,
    cfg: ProjectConfig,
    llm: LLMClient,
    conn: psycopg.Connection | None,
    schemas: Schemas,
    ws: Workspace | None = None,
    max_revisions: int = 2,
) -> ETLSpec:
    """Plan, map each table, validate and revise. Returns the spec (also saved to ``ws``)."""
    reference = reference_concepts_text(conn, schemas)
    vocabularies = _vocabularies_text(conn, schemas)

    try:
        plan = llm.structured(
            stage="design",
            key="plan",
            system=DESIGN_SYSTEM,
            prompt=_plan_prompt(
                profile_prompt_text(profile, cfg.privacy), _annotations_text(annotations), reference
            ),
            output=TablePlan,
        )
    except LLMUnavailable as e:
        raise _need_llm(e) from e
    plan = _normalise_plan(plan)

    # one mapping per planned table
    keys: list[str] = []
    prompts: list[str] = []
    mappings: list[TableMapping] = []
    seen: dict[str, int] = {}
    for pt in plan.tables:
        seen[pt.target_table] = seen.get(pt.target_table, 0) + 1
        key = f"table:{pt.target_table}" + (f":{seen[pt.target_table]}" if seen[pt.target_table] > 1 else "")
        prompt = _table_prompt(pt, plan, profile, annotations, cfg, reference, vocabularies)
        try:
            tm = llm.structured(stage="design", key=key, system=DESIGN_SYSTEM, prompt=prompt, output=TableMapping)
        except LLMUnavailable as e:
            raise _need_llm(e) from e
        keys.append(key)
        prompts.append(prompt)
        mappings.append(tm.model_copy(update={"target_table": pt.target_table}))

    open_questions = list(plan.open_questions)
    if plan.unmapped_source_tables:
        open_questions.append("Source tables not mapped to the CDM: " + ", ".join(plan.unmapped_source_tables) + ".")
    spec = ETLSpec(
        person_key=plan.person_key,
        visit_key=plan.visit_key,
        observation_period=plan.observation_period,
        table_mappings=mappings,
        assumptions=list(plan.assumptions),
        open_questions=open_questions,
    )

    # validate each mapping; send the ones with errors back with the findings
    for i in range(len(mappings)):
        for n in range(1, max_revisions + 1):
            errors = [
                x for x in validate_table_mapping(spec.table_mappings[i], spec, profile, conn, schemas)
                if x.severity == "error"
            ]
            if not errors:
                break
            rkey = f"{keys[i]}:revision{n}"
            try:
                revised = llm.structured(
                    stage="design",
                    key=rkey,
                    system=DESIGN_SYSTEM,
                    prompt=_revision_prompt(prompts[i], spec.table_mappings[i], errors),
                    output=TableMapping,
                )
            except LLMUnavailable as e:
                if ws is not None:
                    ws.audit("design_revision_unavailable", key=rkey, error=str(e))
                break
            target = spec.table_mappings[i].target_table
            spec.table_mappings[i] = revised.model_copy(update={"target_table": target})
            if ws is not None:
                ws.audit("design_revision", key=rkey, issues=[str(x) for x in errors])

    issues = validate_spec(spec, profile, conn, schemas)
    if ws is not None:
        save_design(ws, spec, issues)
        ws.audit(
            "design_complete",
            tables=[tm.target_table for tm in spec.table_mappings],
            issues=[str(x) for x in issues],
            errors=sum(x.severity == "error" for x in issues),
            warnings=sum(x.severity == "warning" for x in issues),
        )
    return spec


SPEC_HEADER = """\
# ETL specification for omop-agent (drafted {when}).
# This file is the editable hand-off between design and build, like a Rabbit-in-a-Hat document.
# Edit it freely: each table_mappings entry compiles to one INSERT ... SELECT (see sql/ after a
# build), and field kinds are described in the FieldMapping docstring (omop_agent/models.py).
# Expressions are PostgreSQL over the source aliases; {{src}}, {{cdm}}, {{vocab}}, {{etl}} are
# replaced by the configured schemas. Changes are picked up by the next build."""


def save_design(ws: Workspace, spec: ETLSpec, issues: list[SpecIssue]) -> None:
    """Write etl_spec.yaml, etl_spec.md and design_issues.json."""
    ws.save_yaml(ws.spec_yaml, spec, header=SPEC_HEADER.format(when=utcnow().strftime("%Y-%m-%d %H:%M UTC")))
    ws.spec_md.write_text(render_spec_md(spec))
    (ws.root / "design_issues.json").write_text(
        json.dumps([x.model_dump(mode="json") for x in issues], indent=2, ensure_ascii=False) + "\n"
    )


__all__ = [
    "DESIGN_SYSTEM",
    "design_etl",
    "reference_concepts_text",
    "save_design",
]
