"""Draft the ETL spec with Claude (the "Rabbit-in-a-Hat" stage), then check it with code.

The design runs in two passes. A plan call decides which source tables feed which CDM tables,
the person and visit keys, and the observation-period strategy. Then one call per planned table
returns a ``TableMapping``. Each mapping is validated (``design/validate.py``: CDM structure,
aliases, source columns, concept ids, ``EXPLAIN``), and mappings with errors go back to the
model with the validator's findings, up to ``max_revisions`` times. Problems that remain are
recorded for the human reviewer rather than raised.

Prompts carry aggregate profile data only (``privacy.profile_prompt_text``). Every design call
shares one system prompt: ``DESIGN_SYSTEM`` followed by the reference material all tables need
(field kinds, staging conventions, loaded vocabularies, notes on the data, reference concepts),
so the provider can cache it; each user prompt carries only what is specific to its table.
"""

from __future__ import annotations

import inspect
import json

import psycopg
import yaml

from ..cdm.spec import DERIVED_ERAS, LOAD_ORDER, load_cdm_spec
from ..config import EtlConfig, ProjectConfig
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
vocab_code), never from memory. The main concept field of condition_occurrence, \
procedure_occurrence, drug_exposure, device_exposure and measurement may also receive a concept \
of another domain: the build then moves the record to that domain's table. A local code \
(concept_map) of a condition may map to a Condition, Observation, Procedure or Measurement \
concept; of a procedure to a Procedure, Observation or Measurement concept; of a measurement to a \
Measurement or Observation concept; of a drug or a device to a Drug or Device concept. Only the \
standard concept decides: unmapped records (concept 0) stay where you load them, and concepts of \
other domains (values, units, types) are never moved.
- Keep the verbatim source value in the matching *_source_value field, and fill \
*_source_concept_id from the source vocabulary when a vocab_code lookup is used (0 otherwise).
- Required fields: when a required value can be missing or unreadable, the build applies the \
CDM convention and counts it: a required concept becomes 0; visit_end_date, \
visit_detail_end_date and drug_exposure_end_date fall back to the start date; rows without a \
required start date, birth year or death date are dropped. Write an explicit fallback when the \
source offers a better one (for example start + days_supply - 1 for drug_exposure_end_date) \
and say so in the rationale. Never give observation_period_end_date a fallback (see observation \
periods).
- Observation periods are the spans during which a person's events would be recorded. When a \
source table records enrollment, residency or follow-up episodes, map it to observation_period \
one row per episode: start = entry date, end = exit date, end NULL when the episode is still \
open (never CURRENT_DATE, never the start date, never a placeholder date). The build merges each \
person's overlapping and back-to-back periods, ends open ones at cdm_source.source_release_date \
or at death, drops episodes that end before they start, and gives persons without a valid \
episode a period from their first to their last event. So do not merge, de-duplicate or filter \
episodes yourself and do not write correlated subqueries for it. Without such a table use \
'from_events'.
- Map each source table of events to the table of its content: diagnoses to \
condition_occurrence, prescriptions and dispensings to drug_exposure, lab results and vital signs \
to measurement, other facts to observation. After loading, the build moves every \
condition_occurrence, procedure_occurrence, drug_exposure, device_exposure and measurement record \
whose standard concept belongs to another of these domains or to Observation to that domain's \
table (an ICD-10 Z code for antenatal care to observation, a procedure code to \
procedure_occurrence); a measurement with a result whose concept is not a measurement or an \
observation goes to observation, so the result is kept. So never split a source table by the \
domain of its codes yourself (a second mapping of the same codes, or a filter on \
{vocab}.concept.domain_id): those records would be loaded twice.

Answer with the requested JSON structure only. Write SQL for PostgreSQL. Put the reasoning a \
reviewer needs into the rationale fields, briefly.
"""

STAGING_CONVENTIONS = """\
Staging conventions for the SQL you write:
- Every source table has been copied to schema {src} with all columns typed text. Convert \
explicitly: ::date or ::timestamp for ISO dates (YYYY-MM-DD; the year-first YYYYMMDD and \
YYYY/MM/DD read too), ::integer or ::numeric for numbers. For any other date format the profile \
shows the PostgreSQL template to use, for example "format %d/%m/%Y (PostgreSQL: to_date(x, \
'DD/MM/YYYY'))": write to_date(x, 'DD/MM/YYYY'); a ::date cast reads none of the values of such \
a column. Never pass a strftime pattern such as '%d/%m/%Y' to to_date or to_timestamp: \
PostgreSQL does not reject it and silently returns a wrong date, so the validator rejects it.
- The build reads these conversions tolerantly: a value that cannot be read (31/02/2019, \
'unknown', a number with text in it) becomes NULL and is counted per field, instead of failing \
the whole table. Do not add regex guards only to avoid cast errors; do guard columns that \
legitimately mix numbers and text (lab results), so that the text goes to value_as_string. \
Casts in join conditions and in person_id, visit_id, vocab_code and concept_map expressions are \
keys and are evaluated as written: where such a value can be unreadable, use \
{etl}.try_date(x, 'DD/MM/YYYY') (or try_timestamp, try_integer, try_number), which gives NULL.
- <PARTIAL_DATES>
- <OFFSETS>
- Empty source values, and the missing-value markers the profile counts (source.na_values, by \
default NA, NULL, \\N and '.' in CSV files), were loaded as NULL. The profile also counts the \
values that do not fit a column's type or date format, and lists value shapes (9 for a digit, \
a / A for a letter): read them before you choose a template.
- Refer to columns through the aliases you declare in source.alias and joins[].alias (for \
example v.visit_date::date). Join other source tables by their bare name, and vocabulary or \
ETL tables as {vocab}.<table> or {etl}.<table>. Inside an expression (a correlated subquery) \
qualify source tables as {src}.<table>. Nothing else may be read: not the CDM being built, not \
system catalogs, not other schemas.
- Each expression, where and join condition is a single SQL expression: no ';', no DDL or \
DML, no functions that read server files or settings. The validator rejects them.
- Never load direct identifiers (columns marked PII in the profile: names, phone numbers, \
national ids, addresses, GPS) into any CDM field. A patient key column may still be used with \
kind person_id, because only the resulting integer person_id is stored.
- Constant concept ids must come from the reference concept list, or be 0.
- Use kind concept_map for local codes, labels and free-text categories. Name the \
vocabulary 'SRC_<NAME>' (for example 'SRC_DRUG') and set target_domain to the OMOP domain of \
the concepts they should map to; the codes are mapped to standard concepts in a later step. For \
the main concept field of an event table this is the table's own domain: the mapper also accepts \
the other domains listed above for it. Give every field its own source vocabulary unless the \
fields mean the same codes: one concept is chosen per code for every field that shares one.
- Use kind vocab_code for codes that already belong to an OMOP vocabulary (ICD10, ICD10CM, \
LOINC, RxNorm, SNOMED, ...). The expression must produce the code exactly as it appears in \
concept.concept_code, so normalise the format in SQL when the profile shows it differs (for \
example, add the dot to ICD-10 codes written without one).
- Map every required field. person_id always uses kind person_id with the column holding the \
patient key, in every table including person itself; visit_occurrence_id in \
visit_occurrence uses kind visit_id with the visit key. These two keys come from the id maps \
that every other table links through, so they are never auto_id: an auto_id person_id would \
follow row order and attach every event to the wrong person. Other tables' own primary keys \
(condition_occurrence_id, drug_exposure_id, measurement_id, ...) use kind auto_id.
- Use where to keep only the rows that belong in this table when one source table feeds \
several CDM tables, by what a row records (a visit type, a form), never by the domain of its \
code: the build moves records by domain.
- A correlated subquery ({src}.<table> inside an expression) runs once per source row, which is \
slow on large tables; prefer a join. Observation period episodes never need one: the build merges \
them.
- Every person and visit_occurrence mapping adds the keys of the rows it loads to the person or \
visit id map, so people and visits can come from several source tables (the HDSS register and a \
clinic register; outpatient visits and admissions). A key found in two sources is one person \
(visit), and the mappings of one table are applied in spec order: the first mapping that loads a \
key wins outright, and a later mapping's rows of that key are skipped (counted); spanning, \
dedup_order and the death ranking compare the rows of one mapping only. List the authoritative \
source first (the HDSS register before a clinic register; the death source whose date should win) \
and bring in what another source adds (a verbal autopsy's cause) through a join of that mapping. \
When a source numbers its people or visits independently of the person_key or \
visit_key table, prefix its keys (for example 'EMR:' || e.patient_no or 'IPD:' || a.admission_no), \
here and in every mapping that refers to them.
- person, death and visit_occurrence keep one row per key: person and death one per person, \
visit_occurrence one per visit and person. Several rows of one visit (ward stays, diagnoses) may \
feed one visit_occurrence mapping: the build keeps one row per visit and person, taking the earliest \
start and the latest end of that mapping's rows (admitted_from from the first row, discharged_to \
from the last). Rows that belong to a visit another mapping loads (the ward stays of an admission \
table) must come through that mapping (as its driving table or a join), or go to visit_detail: a \
mapping of their own adds nothing to the visits loaded before it. Rows of one visit \
key that are separate stays (days apart) fail etlVisitGrain: the visit key must identify one \
visit. Set dedup_order (an ORDER BY list over the aliases, best row first, for example \
"x.verified = '1' DESC" or "i.reg_date::date DESC"; a missing or unreadable value ranks last) when \
the rows of one key can disagree and one should win; otherwise the most complete row is kept (for \
death, a date known to the day, then the latest).
- A record links only to a visit of its own person, and only when its visit key gives exactly the \
text of the visit mapping's key (case, padding, separators, prefix, date format): write it the same \
way.
- When visit numbers restart for each person, the visit key must include the person key, e.g. \
v.individual_id || ':' || v.visit_no (in the visit_key, the visit mapping and every record).
"""


# --------------------------------------------------------------------------------------
# prompt context
# --------------------------------------------------------------------------------------


# Type concepts a source dataset most often needs (EHR, registries, surveys, ...), listed first.
_PREFERRED_TYPE_CONCEPTS = (
    r"\m(EHR|registry|survey|self-report|case report form|death certificate|verbal autopsy|"
    r"clinical study|research|lab|healthcare|claim|prescription|dispens|administration)"
)


def reference_concepts_text(
    conn: psycopg.Connection | None, schemas: Schemas, limit_per_domain: int = 80, type_concept_limit: int = 150
) -> str:
    """Standard, valid concepts of the Gender, Race, Ethnicity, Visit and Type Concept domains.

    Within a domain the most relevant concepts come first: those of the domain's own vocabulary
    (e.g. vocabulary Visit before CMS Place of Service), then, for Type Concept, the common
    provenance types (EHR, registry, survey, ...), then by name. Type Concept is a large domain,
    so it has its own, higher limit.
    """
    if conn is None:
        return "(no vocabulary connection: use 0 for concept constants)"
    sql = f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id FROM (
            SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.domain_id
                       ORDER BY (c.vocabulary_id = c.domain_id) DESC,
                                (c.domain_id = 'Type Concept' AND c.concept_name ~* %(preferred)s) DESC,
                                c.concept_name, c.concept_id
                   ) AS rn
            FROM {qi(schemas.vocab)}.concept c
            WHERE c.domain_id = ANY(%(domains)s)
              AND c.standard_concept = 'S' AND c.invalid_reason IS NULL
        ) x
        WHERE rn <= CASE WHEN domain_id = 'Type Concept' THEN %(type_limit)s ELSE %(limit)s END
        ORDER BY array_position(%(domains)s::text[], domain_id), rn"""
    params = {
        "domains": REFERENCE_DOMAINS,
        "limit": limit_per_domain,
        "type_limit": type_concept_limit,
        "preferred": _PREFERRED_TYPE_CONCEPTS,
    }
    try:
        with conn.transaction():
            rows = fetch_all(conn, sql, params)
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


def _notes_text(annotations: SourceAnnotations | None) -> str:
    if annotations is None or not annotations.notes:
        return "(none)"
    return "\n".join(f"- {n}" for n in annotations.notes)


def staging_conventions(cfg: ProjectConfig | None = None) -> str:
    """``STAGING_CONVENTIONS`` with the date rules of this project (``cfg.etl``: the site time zone
    and the partial-date rule; defaults without it)."""
    etl = cfg.etl if cfg is not None else EtlConfig()
    if etl.partial_dates == "widest":
        partial = ("start dates take the first day, end dates the last day (the widest interval; an end computed "
                   "from the start's columns reads them as the start does), other dates the 15th")
    else:
        partial = "they are left NULL (etl.partial_dates: reject)"
    if etl.timezone:
        offsets = (f"are converted to the site time zone ({etl.timezone}); write ::timestamptz AT TIME ZONE "
                   "'<zone>' only when you need another zone")
    else:
        offsets = ("keep the time as written (no site time zone is configured, so the offset is ignored and "
                   "counted). A column whose profile shows 'Z', '+00:00' or several offsets needs etl.timezone: "
                   "the validator stops the build until a person sets it, whatever the expression, so write the "
                   "plain conversion and say so in open_questions")
    return STAGING_CONVENTIONS.replace("<PARTIAL_DATES>", (
        "Dates with an unknown day or month (00/06/1975, 1975-00-00, or a template without DD or MM) are "
        f"completed by the build and counted: {partial}. Birth and death dates follow THEMIS: year_of_birth and "
        "month_of_birth keep the known parts, a missing birth day is the 1st and a year alone gives no "
        "birth_datetime; a death date takes the last day of the known month or year. Write the plain conversion; "
        "do not handle 00 days yourself."
    )).replace("<OFFSETS>", f"Timestamps with an offset ('Z', '+03:00') {offsets}.")


def design_system(
    reference: str, vocabularies: str, annotations: SourceAnnotations | None, cfg: ProjectConfig | None = None
) -> str:
    """The system prompt of every design call: instructions plus the shared reference material
    (``cfg`` gives the project's date rules, see ``staging_conventions``).

    It is identical for the plan and every table (and revision) call of a run, so the provider
    caches it once; table-specific content goes into the user prompt.
    """
    return f"""{DESIGN_SYSTEM}
The reference material below applies to every request in this design.

FIELD KINDS (TableMapping.fields[].kind)
{inspect.cleandoc(FieldMapping.__doc__ or '')}

{staging_conventions(cfg)}
VOCABULARIES LOADED: {vocabularies}

NOTES ON THE SOURCE DATA (from the interpretation of the profile; they apply to every table \
they mention)
{_notes_text(annotations)}

REFERENCE CONCEPTS (standard, valid)
{reference}
"""


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
    return "\n".join(lines) or "(none)"


def _cdm_tables_text() -> str:
    cdm = load_cdm_spec()
    lines = []
    for name in PLANNABLE_TABLES:
        t = cdm.table(name)
        first = t.description.replace("\n", " ").split(". ")[0].strip().rstrip(".")
        lines.append(f"- {name}: {first[:220]}.")
    return "\n".join(lines)


def _keys_text(person_key: KeySpec, visit_key: KeySpec | None, drafted: list[TableMapping] | None = None) -> str:
    """The patient and visit keys every mapping must write the same way: the plan's keys, plus the
    keys of the person and visit_occurrence mappings drafted so far (each adds its keys to the map)."""

    def written(table: str, field: str) -> list[str]:
        out = []
        for tm in drafted or []:
            fm = next((f for f in tm.fields if f.target_field == field and (f.expression or "").strip()), None)
            if tm.target_table == table and fm is not None and fm.kind in ("person_id", "visit_id"):
                out.append(f"  - {tm.source.table} ({tm.source.alias}): {' '.join((fm.expression or '').split())}")
        return out

    lines = [
        f"Person key: {person_key.table}.{person_key.expression}. Fill person_id with kind person_id "
        "and the column that holds this patient key in the table you are mapping."
    ]
    people = written("person", "person_id")
    if people:
        lines.append("Person mappings drafted so far (a record refers to a person with exactly the same key text as "
                     "one of them):\n" + "\n".join(people))
    if visit_key is not None:
        lines.append(
            f"Visit key: {visit_key.table}.{visit_key.expression}. Fill visit_occurrence_id with kind "
            "visit_id and the column that holds this visit key (NULL when a record has no visit). Write it "
            "exactly as the visit mapping does: a key that gives other text (case, padding, separators, prefix, "
            "date format) leaves the record without a visit, and a record links only to a visit of its own person."
        )
        visits = written("visit_occurrence", "visit_occurrence_id")
        if visits:
            lines.append("Visit mappings drafted so far (their visit keys):\n" + "\n".join(visits))
    else:
        lines.append("There is no visit key: leave visit_occurrence_id fields unmapped.")
    return "\n".join(lines)


def _plan_prompt(profile_text: str, annotations_text: str) -> str:
    return f"""\
Plan the ETL of the source dataset below into OMOP CDM v5.4.

Decide:
1. person_key: the source table that lists each individual once, and the SQL expression of \
the patient key over that table's bare column names. Plan one person mapping per source table \
that lists people (for example the HDSS register, and a clinic register of patients who are not \
in it): each adds the people it loads to the person id map.
2. visit_key: the source table of visits/encounters and its key expression, if there is one. \
Plan one visit_occurrence mapping per source of visits (outpatient visits, admissions): each adds \
its visits to the visit id map. The ward stays of an admission are rows of that admission's \
visit: load them through the admission's mapping (or as visit_detail), not as a mapping of their \
own. If visit numbers restart for each person, \
include the person key in the visit key.
3. observation_period: 'table_mapping' if a source table records enrollment, residency or \
follow-up episodes (then plan an observation_period table from it, one row per episode: the build \
merges each person's episodes, ends open ones and gives persons without an episode a period from \
their events), otherwise 'from_events'.
4. tables: one entry per table mapping, in load order (person first). For each, the CDM \
target table and its source tables (the first one provides the rows; any others are joined). \
A CDM table may appear more than once when several source tables feed it; for person, death and \
visit_occurrence the first mapping that loads a key wins outright, so list the authoritative \
source first. Only plan tables \
the data supports. Do not plan mappings that only split a source table by the domain of its \
codes: the build moves each record to the table of its concept's domain. Do not plan \
condition_era or drug_era: the build derives them from the condition and drug records.
5. unmapped_source_tables: source tables you leave out, with the reason in open_questions.
6. assumptions you made and open_questions for the data owner.

SOURCE PROFILE (aggregates only)
{profile_text}
INTERPRETATION OF THE PROFILE
{annotations_text}

CDM TABLES YOU CAN TARGET
{_cdm_tables_text()}
"""


def _table_prompt(
    pt: PlannedTable,
    plan: TablePlan,
    profile: SourceProfile,
    annotations: SourceAnnotations | None,
    cfg: ProjectConfig,
    drafted: list[TableMapping] | None = None,
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

{_keys_text(plan.person_key, plan.visit_key, drafted)}

TARGET TABLE
{cdm.describe(pt.target_table)}

SOURCE PROFILE (aggregates only)
{profile_prompt_text(profile, cfg.privacy, tables=pt.source_tables)}
INTERPRETATION
{_annotations_text(annotations, pt.source_tables)}

Follow the field kinds, staging conventions and notes on the source data in the system prompt.
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
    """Keep only CDM targets (not the era tables the build derives), make sure person is planned,
    and a consistent period strategy."""
    cdm = load_cdm_spec()
    tables: list[PlannedTable] = []
    notes: list[str] = []
    for pt in plan.tables:
        name = pt.target_table.strip().lower()
        if not cdm.has_table(name) or cdm.table(name).schema_class != "CDM":
            notes.append(f"The plan named '{pt.target_table}', which is not a CDM v5.4 table; it was skipped.")
            continue
        if name in DERIVED_ERAS:  # a mapping into an era table would replace the build's OHDSI eras
            notes.append(f"The plan named {name}, which the build derives from {DERIVED_ERAS[name]} with the OHDSI "
                         "era logic; it was skipped.")
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
    planned_op = any(t.target_table == "observation_period" for t in tables)
    if plan.observation_period == "table_mapping" and not planned_op:
        update["observation_period"] = "from_events"
        update["assumptions"] = plan.assumptions + [
            "No observation_period mapping was planned, so observation periods are derived from events."
        ]
    elif plan.observation_period == "from_events" and planned_op:
        # the planned episodes are the periods; persons without one get a period from their events
        update["observation_period"] = "table_mapping"
        update["assumptions"] = plan.assumptions + [
            "An observation_period mapping was planned, so observation periods come from it ('table_mapping'); "
            "persons without a valid episode get a period from their first to their last event."
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
    system = design_system(reference_concepts_text(conn, schemas), _vocabularies_text(conn, schemas), annotations, cfg)

    try:
        plan = llm.structured(
            stage="design",
            key="plan",
            system=system,
            prompt=_plan_prompt(profile_prompt_text(profile, cfg.privacy), _annotations_text(annotations)),
            output=TablePlan,
        )
    except LLMUnavailable as e:
        raise _need_llm(e) from e
    plan = _normalise_plan(plan)

    # one mapping per planned table, drafted in CDM load order (the person and visit mappings before
    # the records that refer to their keys, which their prompts list) but kept in plan order (the
    # order among the mappings of one table is their priority). The keys of the calls follow the
    # plan order, so recorded answers replay whatever the drafting order
    keys: list[str] = []
    seen: dict[str, int] = {}
    for pt in plan.tables:
        seen[pt.target_table] = seen.get(pt.target_table, 0) + 1
        keys.append(f"table:{pt.target_table}" + (f":{seen[pt.target_table]}" if seen[pt.target_table] > 1 else ""))
    rank = {t: i for i, t in enumerate(LOAD_ORDER)}
    draft_order = sorted(range(len(plan.tables)), key=lambda i: (rank.get(plan.tables[i].target_table, len(rank)), i))
    prompts: list[str] = [""] * len(plan.tables)
    drafted: dict[int, TableMapping] = {}
    for i in draft_order:
        pt = plan.tables[i]
        prompt = _table_prompt(pt, plan, profile, annotations, cfg, [drafted[j] for j in sorted(drafted)])
        try:
            tm = llm.structured(stage="design", key=keys[i], system=system, prompt=prompt, output=TableMapping)
        except LLMUnavailable as e:
            raise _need_llm(e) from e
        prompts[i] = prompt
        drafted[i] = tm.model_copy(update={"target_table": pt.target_table})
    mappings = [drafted[i] for i in range(len(plan.tables))]

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
            issues = validate_table_mapping(spec.table_mappings[i], spec, profile, conn, schemas, cfg.privacy, cfg=cfg)
            errors = [x for x in issues if x.severity == "error"]
            if not errors:
                break
            rkey = f"{keys[i]}:revision{n}"
            try:
                revised = llm.structured(
                    stage="design",
                    key=rkey,
                    system=system,
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

    issues = validate_spec(spec, profile, conn, schemas, cfg.privacy, cfg=cfg)
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
    "design_system",
    "reference_concepts_text",
    "save_design",
]
