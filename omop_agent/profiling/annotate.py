"""Interpret the source profile: what each table and column means (stage 1b).

Claude reads the privacy-filtered scan report and describes each table (grain, patient key,
candidate OMOP tables) and each column (semantic type, coding system). The answer is then
checked against the profile: unknown tables/columns are dropped, PII flags from the profiler
always win, and anything the model left out is filled in by the rule-based
``heuristic_annotations``, which is also the complete fallback when no LLM is available.
"""

from __future__ import annotations

import re

from ..config import ProjectConfig
from ..llm.base import LLMClient, LLMUnavailable
from ..models import (
    ColumnAnnotation,
    ColumnProfile,
    SemanticType,
    SourceAnnotations,
    SourceProfile,
    TableAnnotation,
    TableProfile,
)
from ..privacy import profile_prompt_text, suppress_small_cells
from .profiler import strftime_to_pg

# OMOP CDM v5.4 tables a source table can feed (clinical data, health system, economics,
# episodes). Derived eras, vocabulary and metadata tables are built by the ETL itself.
CDM_TARGET_TABLES = (
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
    "note_nlp",
    "specimen",
    "fact_relationship",
    "location",
    "care_site",
    "provider",
    "payer_plan_period",
    "cost",
    "episode",
    "episode_event",
)

ANNOTATE_SYSTEM = """\
You are an experienced OHDSI data analyst preparing the ETL of a source dataset into the OMOP \
Common Data Model v5.4. You are reading a scan report of the source, in the style of White \
Rabbit: for every table, its row count; for every column, the inferred type, share of nulls, \
number of distinct values, range, detected date format and most frequent values.

The report contains aggregates only. Values seen fewer than the small-cell threshold have been \
withheld, and so has every value of columns flagged as direct identifiers (PII) and of \
high-cardinality key columns. A value missing from a list therefore does not mean it is absent \
from the data. Do not try to guess or reconstruct withheld values.

Your job is to interpret the scan so the ETL can be designed:
- for each table: what it holds, its grain (what one row represents), the column that \
identifies the patient (if any), and the OMOP CDM tables it could feed;
- for each column: its semantic type, and for coded or categorical columns the coding system - \
the OMOP vocabulary_id (for example ICD10, ICD10CM, ICD9CM, LOINC, RxNorm, SNOMED, ATC, CPT4) \
when the codes belong to a standard vocabulary, or 'local' for site-specific codes and labels.

Work from the evidence in the report (names, types, formats, value lists, counts) and stay \
conservative: when the evidence is thin, say so in the description instead of guessing.

Use notes to flag data issues and questions for the data owner, for example: non-ISO or \
ambiguous date formats, coded values whose meaning must be confirmed (such as sex coded 1/2/9), \
missing end or discharge dates, codes written inconsistently (such as ICD-10 codes without the \
dot), numbers and text mixed in one result column, records that may not link to a registered \
person, and identifier columns that must not be loaded into the CDM. Keep each note to one or \
two sentences."""

_SEMANTIC_GUIDE = """\
semantic_type values:
- person_identifier: identifies the patient/person (the key the whole dataset links on)
- record_identifier: unique key of this table's rows (e.g. a visit or episode id)
- foreign_key: refers to rows of another table (e.g. a visit id in a diagnosis table)
- date / datetime: when something happened
- coded: codes from a coding system (ICD-10, LOINC, local drug or test codes...)
- categorical: a small set of labels or codes (sex, visit type, status...)
- numeric_measurement: a quantity or measured value
- free_text: narrative or unstructured text
- pii: a direct identifier (name, phone, national id, address, GPS) - never loaded into the CDM
- other: anything else"""


def _instructions(profile: SourceProfile) -> str:
    return f"""\
Interpret the scan report above.

1. Return exactly one table annotation for each of these source tables, using the exact names: \
{", ".join(t.name for t in profile.tables)}.
2. Return one column annotation for every column listed in the report, with the exact table and \
column names.
3. {_SEMANTIC_GUIDE}
4. coding_system: for coded and categorical columns, the OMOP vocabulary_id or 'local'; \
otherwise null.
5. candidate_cdm_tables: choose among {", ".join(CDM_TARGET_TABLES)}. A table may feed several.
6. person_key: the column of that table that identifies the patient, or null if there is none.
7. Columns marked "PII - values withheld" must get semantic_type 'pii'.
8. notes: data issues and questions for the data owner."""


def annotate(profile: SourceProfile, cfg: ProjectConfig, llm: LLMClient) -> SourceAnnotations:
    """Ask the LLM to interpret the profile; fall back to heuristics when no LLM is available."""
    prompt = (
        "SOURCE SCAN REPORT (aggregates only; rare values and identifiers withheld)\n\n"
        + profile_prompt_text(profile, cfg.privacy)
        + "\n"
        + _instructions(profile)
    )
    try:
        answer = llm.structured(
            stage="annotate", key="source", system=ANNOTATE_SYSTEM, prompt=prompt, output=SourceAnnotations
        )
    except LLMUnavailable:
        return heuristic_annotations(profile)
    return reconcile(answer, profile)


def reconcile(answer: SourceAnnotations, profile: SourceProfile) -> SourceAnnotations:
    """Validate LLM annotations against the profile and complete them with heuristics."""
    fallback = heuristic_annotations(profile)
    fb_tables = {t.table: t for t in fallback.tables}
    fb_cols = {(c.table, c.column): c for c in fallback.columns}
    got_tables: dict[str, TableAnnotation] = {}
    for t in answer.tables:
        got_tables.setdefault(t.table, t)
    got_cols: dict[tuple[str, str], ColumnAnnotation] = {}
    for c in answer.columns:
        got_cols.setdefault((c.table, c.column), c)

    tables: list[TableAnnotation] = []
    columns: list[ColumnAnnotation] = []
    for tp in profile.tables:
        names = {c.name for c in tp.columns}
        ta = got_tables.get(tp.name)
        if ta is None:
            tables.append(fb_tables[tp.name])
        else:
            tables.append(
                ta.model_copy(
                    update={
                        "person_key": ta.person_key if ta.person_key in names else None,
                        "candidate_cdm_tables": _clean_cdm_tables(ta.candidate_cdm_tables),
                    }
                )
            )
        for cp in tp.columns:
            ca = got_cols.get((tp.name, cp.name))
            if ca is None:
                columns.append(fb_cols[(tp.name, cp.name)])
            elif cp.is_pii and ca.semantic_type != "pii":
                columns.append(ca.model_copy(update={"semantic_type": "pii", "coding_system": None, "origin": "llm"}))
            else:
                columns.append(ca.model_copy(update={"origin": "llm"}))
    return SourceAnnotations(tables=tables, columns=columns, notes=list(answer.notes))


def _clean_cdm_tables(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        n = n.strip().lower().rsplit(".", 1)[-1]
        if n in CDM_TARGET_TABLES and n not in out:
            out.append(n)
    return out


# --------------------------------------------------------------------------------------
# Heuristics
# --------------------------------------------------------------------------------------

ICD10_RE = re.compile(r"^[A-TV-Z][0-9]{2}(\.?[0-9A-Z]{0,4})?$")
LOINC_RE = re.compile(r"^\d{1,5}-\d$")
_ID_NAME = re.compile(
    r"(^|_)id$|^id_|(^|_)(uuid|guid|key)$|(visit|encounter|record|episode|admission|household)id$", re.I
)
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")
_PERSON_NAME = re.compile(r"patient|person|individual|client|subject|participant|member", re.I)
_PERSON_ID_NAME = re.compile(r"(^|_)(id|no|num|number|uuid)$|id$", re.I)
_CATEGORY_NAME = re.compile(
    r"(^|_)(sex|gender|status|type|category|class|group|flag|marital|religion|ethnicity|race|"
    r"education|occupation|outcome|result_code|unit|units)($|_)",
    re.I,
)
CODED_MIN_SHARE = 0.8
CATEGORICAL_MAX_DISTINCT = 50
FREE_TEXT_MIN_LENGTH = 50

# (keywords matched in table/column names, CDM table)
_CDM_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("person", "individual", "patient", "demog"), "person"),
    (("residen", "enrol", "episode"), "observation_period"),
    (("visit", "encounter"), "visit_occurrence"),
    (("diag", "condition"), "condition_occurrence"),
    (("drug", "prescri", "medic", "pharm"), "drug_exposure"),
    (("procedure",), "procedure_occurrence"),
    (("lab", "test", "result", "vital", "obs"), "measurement"),
    (("death", "dod"), "death"),
]


def _is_person_id(name: str) -> bool:
    return bool(_PERSON_NAME.search(name) and _PERSON_ID_NAME.search(name))


def _share_matching(values: list[str], pattern: re.Pattern[str]) -> float:
    if not values:
        return 0.0
    return sum(bool(pattern.match(v.strip())) for v in values) / len(values)


def _column_semantics(c: ColumnProfile, min_cell_count: int) -> tuple[SemanticType, str | None]:
    t = c.inferred_type
    if c.is_pii:
        return "pii", None
    if _is_person_id(c.name):
        return "person_identifier", None
    if _ID_NAME.search(c.name) and t not in ("date", "datetime"):
        return ("record_identifier" if c.unique else "foreign_key"), None
    if t in ("date", "datetime"):
        return t, None  # type: ignore[return-value]
    shown = [v.value for v in suppress_small_cells(c.top_values, min_cell_count)[0]]
    if t == "string" and _share_matching(shown, ICD10_RE) >= CODED_MIN_SHARE:
        return "coded", "ICD10"
    if t == "string" and _share_matching(shown, LOINC_RE) >= CODED_MIN_SHARE:
        return "coded", "LOINC"
    if t == "boolean" or _CATEGORY_NAME.search(c.name):
        return "categorical", "local"
    if t == "string" and 0 < c.n_distinct <= CATEGORICAL_MAX_DISTINCT:
        return "categorical", "local"
    if t in ("integer", "float") or (t == "string" and _share_matching(shown, _NUMBER_RE) >= 0.5):
        return "numeric_measurement", None
    if t == "string" and (c.max_length or 0) >= FREE_TEXT_MIN_LENGTH:
        return "free_text", None
    return "other", None


def _describe_column(c: ColumnProfile, sem: SemanticType, coding: str | None) -> str:
    null_pct = 100.0 * c.n_null / c.n_rows if c.n_rows else 0.0
    what = {
        "person_identifier": "Patient identifier",
        "record_identifier": "Unique record identifier",
        "foreign_key": "Reference to another table's key",
        "date": f"Date ({c.date_format})" if c.date_format else "Date",
        "datetime": f"Date and time ({c.date_format})" if c.date_format else "Date and time",
        "coded": f"Codes ({coding})" if coding else "Codes",
        "categorical": f"Categorical value with {c.n_distinct} distinct values",
        "numeric_measurement": f"Numeric value ({c.inferred_type})",
        "free_text": "Free text",
        "pii": "Direct identifier (PII), withheld",
        "other": f"Time of day ({c.date_format})" if c.inferred_type == "time" else f"Column of type {c.inferred_type}",
    }[sem]
    return f"{what}; {null_pct:.0f}% null (inferred by name/type rules)."


def _candidate_tables(t: TableProfile, col_sem: dict[str, SemanticType]) -> list[str]:
    def match(text: str) -> list[str]:
        text = text.lower()
        return [cdm for keys, cdm in _CDM_KEYWORDS if any(k in text for k in keys)]

    found = match(t.name)
    if not found:  # fall back to descriptive (non-key) column names
        names = " ".join(
            c.name for c in t.columns
            if col_sem[c.name] not in ("person_identifier", "record_identifier", "foreign_key", "pii")
        )
        found = match(names)
    return list(dict.fromkeys(found))


def _grain(t: TableProfile, person_key: str | None, col_sem: dict[str, SemanticType]) -> str:
    by_name = {c.name: c for c in t.columns}
    if person_key and by_name[person_key].unique and by_name[person_key].n_null == 0:
        return f"one row per person ({person_key} is unique)"
    rid = next((c.name for c in t.columns if col_sem[c.name] == "record_identifier" and c.n_null == 0), None)
    if rid:
        return f"one row per {rid} ({rid} is unique)"
    return f"one row per {t.name} record (no single unique key column)"


def heuristic_annotations(profile: SourceProfile) -> SourceAnnotations:
    """Rule-based annotations from column names, inferred types and listed values."""
    tables: list[TableAnnotation] = []
    columns: list[ColumnAnnotation] = []
    notes = ["Annotations were produced by name/type rules, not by a model; review them before designing the ETL."]
    for t in profile.tables:
        col_sem: dict[str, SemanticType] = {}
        for c in t.columns:
            sem, coding = _column_semantics(c, profile.min_cell_count)
            col_sem[c.name] = sem
            columns.append(
                ColumnAnnotation(
                    table=t.name, column=c.name, semantic_type=sem, coding_system=coding,
                    description=_describe_column(c, sem, coding),
                )
            )
            notes += _column_notes(t, c)
        person_key = next((c.name for c in t.columns if col_sem[c.name] == "person_identifier"), None)
        tables.append(
            TableAnnotation(
                table=t.name,
                description=f"Source table {t.name} with {t.n_rows:,} rows and {len(t.columns)} columns "
                "(described by rules, not by a model).",
                grain=_grain(t, person_key, col_sem),
                person_key=person_key,
                candidate_cdm_tables=_candidate_tables(t, col_sem),
            )
        )
    pii = [f"{t.name}.{c.name}" for t in profile.tables for c in t.columns if c.is_pii]
    if pii:
        notes.append(f"Direct identifiers withheld and not to be loaded into the CDM: {', '.join(pii)}.")
    return SourceAnnotations(tables=tables, columns=columns, notes=notes)


def _column_notes(t: TableProfile, c: ColumnProfile) -> list[str]:
    notes = []
    iso = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
    if c.date_format and c.date_format not in iso and c.inferred_type != "time":
        try:
            pg = f" (PostgreSQL template '{strftime_to_pg(c.date_format)}')"
        except ValueError:
            pg = ""
        notes.append(f"{t.name}.{c.name} is not ISO formatted: it uses {c.date_format}{pg}; parse it explicitly.")
    if c.inferred_type in ("date", "datetime") and c.n_rows and c.n_null / c.n_rows >= 0.1:
        notes.append(f"{t.name}.{c.name} is empty in {100.0 * c.n_null / c.n_rows:.0f}% of rows.")
    return notes
