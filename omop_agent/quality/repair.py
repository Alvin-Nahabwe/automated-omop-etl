"""The repair agent: Claude diagnoses failed ETL steps and data-quality checks and fixes the spec.

The agent never touches the data. It reads context through tools (the table mapping and its
compiled SQL, the CDM table conventions, the privacy-filtered source profile), tests hypotheses
with ``count_records``, and edits the ``ETLSpec`` through two tools that apply a change only if
it passes the static SQL and privacy checks and the table still compiles and ``EXPLAIN``s
cleanly. Each applied change is recorded as a ``RepairAction`` (``repairs.json`` and the audit
log); the pipeline then rebuilds and re-checks.

Privacy:

* ``count_records`` is not SQL. The model names a table, up to three columns to group by and a
  few filters; the tool builds the query itself (identifiers quoted, values bound as
  parameters) and runs it read-only. It returns counts only: groups with fewer than
  ``min_cell_count`` records are withheld (their number is reported) and a total below the
  threshold is shown as ``<k``. Patient, visit and record keys, direct identifiers (PII), CDM
  ``*_id`` columns (except ``*_concept_id``), CDM fields the spec derives from identifiers or
  computes across records, and the person/visit id maps can never be grouped or filtered.
  Vocabulary tables are public reference data and are not suppressed. With
  ``privacy.send_values_to_llm: false`` only ``*_concept_id`` columns (and vocabulary tables)
  can be grouped.
* Spec edits are rejected when their SQL fails ``sqlguard.fragment_issues``, reads a direct
  identifier (``privacy.pii_references``), reads tables outside the staged source, vocabulary
  and ETL schemas, or breaks the person/visit linkage (``auto_id`` person or visit keys, a
  ``person_id`` field that bypasses the person id map).
* Database errors shown to the model (failed ETL steps, check errors) go through
  ``redact_error``: DETAIL is dropped, and quoted text is kept only when it is exactly a known
  identifier; every value slot is withheld. Small violated-row counts of failed checks are
  shown as ``<k``.

Residual risk: exact counts over model-chosen filters allow differencing ("tracker") attacks,
and whether a build succeeds is itself one bit. These channels are low-bandwidth and need a
deliberate, multi-step effort; noise or query auditing would be needed to close them.

Tool names and input fields are part of recorded replays: do not rename them.
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Iterator, Literal

import psycopg
import yaml
from psycopg import sql as pgsql
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..cdm.spec import load_cdm_spec
from ..config import PrivacyConfig, ProjectConfig
from ..db import Schemas, execute, fetch_one, rollback_only, short_error
from ..etl.compiler import LAYER_1_MARKER, SPEC_KINDS
from ..etl.session import apply_session
from ..llm.base import AgentTool, LLMClient, LLMError, LLMUnavailable
from ..models import (
    CheckResult,
    DQReport,
    ETLRunResult,
    ETLSpec,
    FieldKind,
    FieldMapping,
    RepairAction,
    SourceProfile,
    TableLoadResult,
    TableMapping,
)
from ..privacy import name_rule, pii_columns, pii_references, profile_prompt_text
from ..profiling.profiler import KEY_MAX_LISTED_DISTINCT, _is_key_column
from ..sqlguard import fragment_issues, strip_literals
from ..workspace import Workspace, utcnow
from .checks import derived_tables, repairable

MAX_FAILURES_IN_PROMPT = 30
COUNT_TIMEOUT_MS = 30_000
MAX_FILTER_VALUES = 100

REPAIR_SYSTEM = """\
You are the data-quality repair agent of an OMOP CDM v5.4 ETL pipeline: the ETL developer who \
reads the Data Quality Dashboard results and fixes the ETL.

The ETL is defined by an ETL spec. Each table mapping compiles to one
  INSERT INTO cdm.<table> SELECT ... FROM src.<source table> <alias> [JOIN ...] [WHERE ...]
with one field mapping per CDM field. Source columns are all text, so cast explicitly. Field kinds:
  column / expression / constant  a PostgreSQL expression over the source aliases, cast to the CDM type
  person_id    the source patient key, resolved through the person id map (rows without a person are
               dropped). Every person_id field must use this kind. Every person mapping adds the keys of
               its rows to the map, so people can come from several source tables.
  visit_id     the source visit key, resolved through the visit id map. An optional field (e.g.
               condition_occurrence.visit_occurrence_id) keeps the row with NULL when there is no match;
               the required visit_occurrence primary key drops rows without a visit. A record links only
               to a visit of its own person, and only when its visit key is written exactly as in the
               visit mapping (same text: case, padding, separators, prefix, date format).
  auto_id      generated surrogate key (only for the table's own primary key; never for person.person_id
               or visit_occurrence.visit_occurrence_id, which come from the id maps)
  vocab_code   a code of an OMOP vocabulary: expression = the code as in concept_code, vocabulary = its
               vocabulary_id; concept_role 'standard' follows 'Maps to', 'source' returns the source concept
  concept_map  a local code/label mapped through the project's source_to_concept_map: vocabulary = the
               source_vocabulary_id, target_domain = the OMOP domain of the targets; concept_role 'source'
               returns the map's source_concept_id
  null         leave the field NULL
person, death and visit_occurrence keep one row per key (person and death: one per person; visit_occurrence:
one per visit and person, from the earliest start to the latest end of its rows). A mapping's dedup_order (an
ORDER BY list over its aliases, best row first, e.g. "x.verified = '1' DESC"; a missing or unreadable value
ranks last) says which row of a key is loaded; without one the most complete row is (for death, a date
known to the day, then the latest). Set it when etlKeyConflict reports person rows that disagree on sex or
birth year, ranking the row that should win first. Who exists in the CDM is decided by people: changes to a
person_id expression (in any table, unless the mapping cannot run at all), to the key of person or
visit_occurrence, to the filter of person, visit_occurrence or death, and new mappings into those tables
are refused. A record that names another person's visit, or a key no visit mapping reads (etlVisitLinkage,
etlVisitLinkLoss), is fixed in its visit_id expression so that it gives the visit mapping's key, never by
changing its person_id and never by making the visit key NULL: an edit that takes the visit key away from
records that have one is refused.
Concept lookups yield 0 when a source value exists but has no mapping. Casts (::date, ::numeric, ...)
and to_date / to_timestamp in column and expression fields and in the filter are tolerant: a value that
cannot be read as written (31/02/2019, a date in another layout than the template, a number with text in
it) becomes NULL and is counted per field (etlInvalidValue) instead of failing the whole table. Fix the
expression when the source holds the value in another form (the to_date template the profile shows; a
regex guard for a column that mixes numbers and text, so the text goes to value_as_string), never by
filtering rows, and never by hiding the values: {etl}.try_date and the other try_* functions are counted
like a cast in a field or the filter, and a regex guard that makes unreadable dates or numbers NULL
before the cast only hides them. A date with an unknown day or month (00/06/1975) <PARTIAL_DATES> and is
reported as etlPartialDate. A required value that comes out NULL follows the CDM convention
automatically and is counted: required concepts default to 0 and visit_end_date, visit_detail_end_date
and drug_exposure_end_date to the start date (etlDefaultedValue tells you where a better fallback may
exist); a row without another required value (a start date, year_of_birth, death_date) is dropped
(etlRequiredFieldNull), and an edit that makes such a field NULL for more rows is refused because it
would drop those records. observation_period is loaded one row per episode, with a NULL end for an
episode that is still open: the build merges each person's overlapping and back-to-back episodes,
ends open ones at cdm_source.source_release_date or at death, drops episodes that end before they
start (etlObservationPeriodMerge) and gives persons without a valid episode a period from their
events; never write merging, filters or a fallback into the observation_period end date (an edit that
opens episodes with an exit date, or closes open ones, is refused). After loading, the build moves every
condition_occurrence, procedure_occurrence, drug_exposure, device_exposure and measurement record whose
standard concept belongs to another event domain (Condition, Procedure, Drug, Device, Measurement,
Observation) to that domain's table (route_* steps, etlDomainRouting; a measurement with a result goes to
observation). Never route records by hand: a mapping that filters or joins on the domain of
{vocab}.concept, or a second mapping of the same codes into another table, loads records twice and is
refused. fkDomain failures that remain on those tables come from concepts outside the six event domains
(a value, unit or type concept in the main concept field: fix the lookup) or from records the build could
not move (its route step failed). Steps named
*_map_prune, route_*, observation_period_merge, observation_period_fill, condition_era, drug_era and
cdm_source (and observation_period when periods are 'from_events') are generated by the build and
cannot be changed with the tools; the eras are built from the condition and drug records as loaded,
so a new mapping into condition_era or drug_era is refused (fix those records instead). Source and join
tables are staged source table names (or {src}., {vocab}., {etl}.-qualified names). Spec SQL must be a
single expression or condition: no statement separators, no catalog, file or session functions, and no
direct identifiers (names, phone numbers, national ids, addresses, GPS): those are never loaded into the
CDM.

How to work:
1. Diagnose the root cause of each failure before changing anything. Read the mapping and its compiled
   SQL (get_table_mapping), the CDM conventions (get_cdm_table) and the source profile
   (get_source_profile), and test hypotheses with count_records.
2. Fix the ETL spec, not the data: update_field_mapping changes one field, replace_table_mapping changes
   the source, joins, filter or several fields at once. A change is applied only if it passes the SQL and
   privacy checks and the table still compiles and EXPLAINs cleanly; if it comes back REJECTED, read the
   reason and try again.
3. Prefer the smallest change that follows OMOP conventions (for example, a visit without an end date
   ends on its start date; an unknown value maps to concept 0 rather than to a guessed concept).
4. Do not paper over real data problems. Never filter out or delete records just to make a check pass,
   and never invent concept ids, dates or values. Edits that can only remove data are refused: a new or
   changed filter, a new or changed inner join, another source table, or blanking a populated field. When the cause lies in the source data or in the
   vocabulary coverage (unmapped local codes, implausible dates recorded at source), leave the spec as
   it is and explain the problem for a human.
5. Privacy: you only ever see aggregates. count_records returns counts of records, optionally grouped by
   up to three columns: groups with fewer records than the minimum cell count are withheld (only their
   number is reported) and a small total is shown as "<k". Patient, visit and record identifiers, direct
   identifiers and CDM *_id columns (except *_concept_id) cannot be grouped or filtered. Never try to
   retrieve or reconstruct individual records or rare values.

Finish with a short summary: what you changed and why, and what remains for a human to review."""


# --------------------------------------------------------------------------------------
# State and tool inputs
# --------------------------------------------------------------------------------------


@dataclass
class RepairState:
    """What the repair tools read and change during one repair iteration."""

    spec: ETLSpec
    conn: Any
    schemas: Schemas
    cfg: ProjectConfig
    profile: SourceProfile | None = None
    iteration: int = 1
    actions: list[RepairAction] = field(default_factory=list)
    trigger: str = ""
    table_triggers: dict[str, list[str]] = field(default_factory=dict)

    def trigger_for(self, table: str) -> str:
        """The failures on ``table`` if there are any, else every failure of this iteration."""
        keys = self.table_triggers.get(table.lower())
        return ", ".join(keys)[:500] if keys else self.trigger

    @property
    def min_cell_count(self) -> int:
        return self.cfg.privacy.min_cell_count

    def known_identifiers(self) -> set[str]:
        """Identifiers that may appear verbatim in a redacted error message."""
        return known_identifiers(self.profile, self.schemas)


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetTableMappingInput(_ToolInput):
    target_table: str = Field(..., description="CDM table the mapping loads, e.g. 'visit_occurrence'.")
    mapping_index: int = Field(0, ge=0, description="Which mapping when several feed the same table (0 = the first).")


class GetCdmTableInput(_ToolInput):
    table: str = Field(..., description="OMOP CDM v5.4 table name, e.g. 'condition_occurrence'.")


class GetSourceProfileInput(_ToolInput):
    table: str = Field(..., description="Source table name as staged, e.g. 'clinic_visits'.")


FilterOp = Literal["=", "<>", "<", ">", "<=", ">=", "is_null", "not_null", "in", "not_in", "like"]


class RecordFilter(_ToolInput):
    column: str = Field(..., description="A column of the counted table.")
    op: FilterOp = Field(
        ...,
        description="Comparison: =, <>, <, >, <=, >= and like (SQL LIKE pattern with % and _) take `value`; "
        "in / not_in take `values`; is_null / not_null take neither.",
    )
    value: str | int | float | None = Field(None, description="The value compared with (cast to the column's type).")
    values: list[str | int | float] = Field(
        default_factory=list, max_length=MAX_FILTER_VALUES, description="The list of values for in / not_in."
    )


class CountRecordsInput(_ToolInput):
    table: str = Field(
        ...,
        description="'<schema>.<table>' where schema is src (staged source tables), cdm (CDM tables), vocab "
        "(vocabulary tables) or etl (only etl.source_to_concept_map), e.g. 'cdm.condition_occurrence', "
        "'src.clinic_visits', 'vocab.concept'.",
    )
    group_by: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 columns of the table to count by. Empty: one total count.",
    )
    filters: list[RecordFilter] = Field(
        default_factory=list, max_length=10, description="Conditions on columns of the table, combined with AND."
    )
    limit: int = Field(50, ge=1, le=50, description="Largest groups shown (at most 50).")


class UpdateFieldMappingInput(_ToolInput):
    target_table: str = Field(..., description="CDM table, e.g. 'visit_occurrence'.")
    target_field: str = Field(..., description="CDM field to (re)map, e.g. 'visit_end_date'.")
    kind: FieldKind = Field(..., description="Field kind (see the system prompt).")
    expression: str | None = Field(
        None, description="SQL expression over the source aliases (required for every kind except auto_id and null)."
    )
    vocabulary: str | None = Field(
        None, description="vocab_code: the OMOP vocabulary_id of the codes; concept_map: the source_vocabulary_id."
    )
    target_domain: str | None = Field(None, description="concept_map: OMOP domain of the target concepts.")
    concept_role: Literal["standard", "source"] = Field(
        "standard",
        description="vocab_code / concept_map: 'standard' yields the standard concept, 'source' the source concept "
        "(for *_source_concept_id fields).",
    )
    rationale: str = Field(..., description="Why this change fixes the failure (kept in the spec and repairs.json).")
    mapping_index: int = Field(0, ge=0, description="Which mapping when several feed the same table (0 = the first).")


class ReplaceTableMappingInput(_ToolInput):
    target_table: str = Field(..., description="CDM table the mapping loads.")
    mapping_yaml: str = Field(
        ...,
        description="The complete new TableMapping as YAML, in the format get_table_mapping returns "
        "(target_table, source {table, alias}, joins, where, dedup_order, fields, rationale).",
    )
    rationale: str = Field(..., description="Why this change fixes the failure.")
    mapping_index: int = Field(
        0,
        ge=0,
        description="Which mapping to replace when several feed the same table (0 = the first). Use the number "
        "of existing mappings for the table to add a new one.",
    )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _yaml(model: BaseModel) -> str:
    return yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=110)


def _locate(spec: ETLSpec, table: str, index: int) -> tuple[int | None, int]:
    """(position in spec.table_mappings, number of mappings for the table)."""
    positions = [i for i, tm in enumerate(spec.table_mappings) if tm.target_table.lower() == table.lower()]
    return (positions[index] if 0 <= index < len(positions) else None), len(positions)


def _mapped_tables(spec: ETLSpec) -> str:
    return ", ".join(sorted({tm.target_table for tm in spec.table_mappings})) or "(none)"


def _fm_text(fm: FieldMapping | None) -> str:
    if fm is None:
        return "(unmapped)"
    text = f"{fm.kind}: {fm.expression}" if fm.expression is not None else fm.kind
    if fm.vocabulary:
        role = f", {fm.concept_role}" if fm.kind in {"vocab_code", "concept_map"} else ""
        text += f" [{fm.vocabulary}{role}]"
    return text


def _same_logic(a: FieldMapping, b: FieldMapping) -> bool:
    return a.model_dump(exclude={"rationale"}) == b.model_dump(exclude={"rationale"})


def _same_mapping_logic(a: TableMapping, b: TableMapping) -> bool:
    """Same source, joins, filter, dedup_order and field logic (rationales and field order aside)."""
    if a.target_table.lower() != b.target_table.lower() or (a.source, a.joins) != (b.source, b.joins):
        return False
    if (a.where or "").strip() != (b.where or "").strip():
        return False
    if (a.dedup_order or "").strip() != (b.dedup_order or "").strip():
        return False
    fa = {f.target_field: f for f in a.fields}
    fb = {f.target_field: f for f in b.fields}
    return fa.keys() == fb.keys() and all(_same_logic(fa[n], fb[n]) for n in fa)


def _mapping_diff(old: TableMapping | None, new: TableMapping) -> str:
    if old is None:
        return f"added table mapping {new.target_table} <- {new.source.table} ({len(new.fields)} fields)"
    parts: list[str] = []
    if old.source != new.source:
        parts.append(f"source {old.source.table} {old.source.alias} → {new.source.table} {new.source.alias}")
    if old.joins != new.joins:
        parts.append("joins changed")
    if (old.where or "") != (new.where or ""):
        parts.append(f"where: {old.where or '(none)'} → {new.where or '(none)'}")
    if (old.dedup_order or "").strip() != (new.dedup_order or "").strip():
        parts.append(f"dedup_order: {old.dedup_order or '(none)'} → {new.dedup_order or '(none)'}")
    old_f = {f.target_field: f for f in old.fields}
    new_f = {f.target_field: f for f in new.fields}
    for name, fm in new_f.items():
        if name in old_f and not _same_logic(old_f[name], fm):
            parts.append(f"{name}: {_fm_text(old_f[name])} → {_fm_text(fm)}")
    added = [n for n in new_f if n not in old_f]
    removed = [n for n in old_f if n not in new_f]
    if added:
        parts.append("added " + ", ".join(f"{n} ({_fm_text(new_f[n])})" for n in added))
    if removed:
        parts.append("removed " + ", ".join(removed))
    return "replaced table mapping: " + ("; ".join(parts) if parts else "no functional change")


@contextmanager
def _readonly(conn: psycopg.Connection, timeout_ms: int = COUNT_TIMEOUT_MS) -> Iterator[psycopg.Connection]:
    """A READ ONLY transaction (or savepoint) with a statement timeout, always rolled back."""
    with conn.transaction(force_rollback=True):
        execute(conn, "SET TRANSACTION READ ONLY")
        execute(conn, f"SET LOCAL statement_timeout = {int(timeout_ms)}")
        yield conn


def _rows(conn: psycopg.Connection, query: Any, params: Iterable[Any] = (), limit: int | None = None) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, list(params))
        if not cur.description:
            return []
        return list(cur.fetchmany(limit) if limit is not None else cur.fetchall())


def _tables_in(conn: psycopg.Connection, schema: str) -> list[str]:
    rows = _rows(conn, "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", [schema])
    return sorted(r["table_name"] for r in rows)


def _columns_of(conn: psycopg.Connection, schema: str, table: str) -> dict[str, str]:
    """column name -> data type, in table order."""
    rows = _rows(
        conn,
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        [schema, table],
    )
    return {r["column_name"]: r["data_type"] for r in rows}


def _find_name(name: str, names: Iterable[str]) -> str | None:
    """``name`` as it exists in ``names``: exact, else the one case-insensitive match."""
    names = list(names)
    if name in names:
        return name
    matches = [n for n in names if n.lower() == name.lower()]
    return matches[0] if len(matches) == 1 else None


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_QUOTED_IDENT = re.compile(r'"((?:[^"]|"")+)"')


def _tokens(text: str | None) -> set[str]:
    """Lower-cased identifiers and words of a SQL fragment, literals and comments removed."""
    bare = strip_literals(text or "")
    quoted = {m.replace('""', '"').lower() for m in _QUOTED_IDENT.findall(bare)}
    return quoted | {m.lower() for m in _IDENT.findall(_QUOTED_IDENT.sub(" ", bare))}


# ---- which columns may be grouped or filtered ----------------------------------------


def _pii_pattern(name: str, state: RepairState) -> bool:
    return name_rule(name, state.cfg.privacy) is not None


def source_column_denial(state: RepairState, table: str, column: str) -> str | None:
    """Why a staged source column may not be grouped or filtered (None if it may)."""
    col = column.lower()
    if col in pii_columns(state.profile, state.cfg.privacy).get(table.lower(), set()) or _pii_pattern(col, state):
        return "a direct identifier (PII)"
    tp = state.profile.table(table) if state.profile is not None else None
    if tp is None and state.profile is not None:
        tp = next((t for t in state.profile.tables if t.name.lower() == table.lower()), None)
    cp = next((c for c in tp.columns if c.name.lower() == col), None) if tp is not None else None
    if cp is None:  # no profile for it: judge by the name alone, as if it had very many values
        if _is_key_column(col, sys.maxsize):
            return "a patient, visit or record key"
        return None
    if cp.is_pii:
        return "a direct identifier (PII)"
    if _is_key_column(cp.name, cp.n_distinct):
        return "a patient, visit or record key"
    if cp.unique and cp.n_distinct >= state.min_cell_count and (
        col.endswith("id") or (cp.n_distinct > KEY_MAX_LISTED_DISTINCT and cp.inferred_type in ("string", "integer"))
    ):
        return "a unique record identifier"
    return None


def _source_table_name(ref: str) -> str | None:
    """The staged table a mapping's source/join table reference points at, or None."""
    ref = ref.strip()
    m = re.fullmatch(r"(?:\{src\}\.)?\"?([^\".{}]+)\"?", ref)
    return m.group(1) if m else None


_WINDOW_OR_SUBQUERY = re.compile(r"\b(select|over)\b", re.I)


def _derived_from_identifier(state: RepairState, fm: FieldMapping, tm: TableMapping) -> str | None:
    """Why a CDM field's values may reveal identifiers or individual records, per its spec mapping."""
    if fm.kind not in ("column", "expression", "constant") or not fm.expression:
        return None
    bare = strip_literals(fm.expression)
    if _WINDOW_OR_SUBQUERY.search(bare):
        return "computed across records in the ETL spec (subquery or window function)"
    for alias in [tm.source.alias] + [j.alias for j in tm.joins]:
        if re.search(rf"(?<![\w.$]){re.escape(alias)}\b(?!\s*\.)", bare, re.I):
            return f"built from whole source rows ({alias}) in the ETL spec"
    tables = [t for t in (_source_table_name(r) for r in [tm.source.table] + [j.table for j in tm.joins]) if t]
    for token in sorted(_tokens(fm.expression)):
        for t in tables or [""]:
            reason = source_column_denial(state, t, token)
            if reason:
                return f"loaded from {token}, {reason}, in the ETL spec"
    return None


def cdm_column_denial(state: RepairState, table: str, column: str) -> str | None:
    """Why a CDM column may not be grouped or filtered (None if it may)."""
    col = column.lower()
    if col in ("person_id", "person_source_value"):
        return "a person identifier"
    if col.endswith("_id") and not col.endswith("_concept_id"):
        return "a record identifier (CDM *_id column)"
    if _pii_pattern(col, state):
        return "a direct identifier (PII)"
    for tm in state.spec.table_mappings:
        if tm.target_table.lower() != table.lower():
            continue
        fm = next((f for f in tm.fields if f.target_field.lower() == col), None)
        reason = _derived_from_identifier(state, fm, tm) if fm is not None else None
        if reason:
            return reason
    return None


# ---- count_records -------------------------------------------------------------------

LOGICAL_SCHEMAS = ("src", "cdm", "vocab", "etl")
ETL_COUNTABLE = ("source_to_concept_map",)
ETL_DENIED = ("person_map", "visit_map", "person_map_retired", "visit_map_retired")
# ETL tables that only generated steps use. Their columns are generic words ('name', 'value') that
# spec SQL never names, so redact_error does not treat them as identifiers.
_ETL_BOOKKEEPING = ("id_high_water", "build_facts")


@lru_cache(maxsize=1)
def _cdm_tables() -> frozenset[str]:
    return frozenset(t.name for t in load_cdm_spec().cdm_tables)


@lru_cache(maxsize=1)
def _public_vocab_tables() -> frozenset[str]:
    """The standardized vocabularies (Athena): public reference data. The source_to_concept_map is
    site-specific (it lists local source values) and is not part of it."""
    return frozenset(t.name for t in load_cdm_spec().vocab_tables if t.name != "source_to_concept_map")


def _resolve_table(state: RepairState, name: str) -> tuple[str, str, str] | str:
    """(logical schema, physical schema, table) or a message saying why the table can't be counted."""
    m = re.fullmatch(r"\s*\{?([A-Za-z_]+)\}?\s*\.\s*\"?([^\".]+?)\"?\s*", name)
    if not m or m.group(1).lower() not in LOGICAL_SCHEMAS:
        return (
            f"ERROR: table must be '<schema>.<table>' with schema one of {', '.join(LOGICAL_SCHEMAS)} "
            f"(e.g. 'cdm.condition_occurrence' or 'src.clinic_visits'), got {name!r}"
        )
    logical, wanted = m.group(1).lower(), m.group(2)
    if logical == "etl" and wanted.lower() in ETL_DENIED:
        return (
            f"REFUSED: etl.{wanted.lower()} maps source identifiers to CDM ids and cannot be queried. "
            "Count the CDM table instead (unmatched records are reported as etlRecordRetention)."
        )
    physical = getattr(state.schemas, logical)
    existing = _tables_in(state.conn, physical)
    if logical == "cdm":
        existing = [t for t in existing if t.lower() in _cdm_tables()]
    elif logical == "vocab":
        existing = [t for t in existing if t.lower() in _public_vocab_tables()]
    elif logical == "etl":
        existing = [t for t in existing if t.lower() in ETL_COUNTABLE]
    table = _find_name(wanted, existing)
    if table is None:
        listed = ", ".join(existing) or "(none)"
        return f"ERROR: no table {logical}.{wanted} that can be counted. {logical} tables: {listed}"
    return logical, physical, table


def _denial(state: RepairState, logical: str, table: str, column: str) -> str | None:
    if logical == "src":
        return source_column_denial(state, table, column)
    if logical == "cdm":
        return cdm_column_denial(state, table, column)
    return None  # vocab (public) and etl.source_to_concept_map


def _filter_sql(f: RecordFilter, column: str) -> tuple[pgsql.Composable, list[str]] | str:
    ident = pgsql.Identifier(column)
    if f.op in ("is_null", "not_null"):
        return pgsql.SQL("{} IS NULL" if f.op == "is_null" else "{} IS NOT NULL").format(ident), []
    if f.op in ("in", "not_in"):
        if not f.values:
            return f"filter on {f.column}: op '{f.op}' needs a non-empty `values` list"
        marks = pgsql.SQL(", ").join([pgsql.Placeholder()] * len(f.values))
        keyword = pgsql.SQL("IN" if f.op == "in" else "NOT IN")
        return pgsql.SQL("{} {} ({})").format(ident, keyword, marks), [str(v) for v in f.values]
    if f.value is None:
        return f"filter on {f.column}: op '{f.op}' needs a `value` (use is_null / not_null for NULL)"
    if f.op == "like":
        return pgsql.SQL("{}::text LIKE {}").format(ident, pgsql.Placeholder()), [str(f.value)]
    return pgsql.SQL("{} {} {}").format(ident, pgsql.SQL(f.op), pgsql.Placeholder()), [str(f.value)]


def _describe_filter(f: RecordFilter, column: str) -> str:
    if f.op in ("is_null", "not_null"):
        return f"{column} {'IS NULL' if f.op == 'is_null' else 'IS NOT NULL'}"
    if f.op in ("in", "not_in"):
        return f"{column} {'IN' if f.op == 'in' else 'NOT IN'} ({', '.join(repr(v) for v in f.values)})"
    return f"{column} {f.op.upper()} {f.value!r}"


def _cell(v: Any) -> str:
    s = "NULL" if v is None else str(v)
    s = " ".join(s.split())
    return s if len(s) <= 60 else s[:57] + "..."


def _table_text(rows: list[dict], cols: list[str]) -> list[str]:
    cells = [[_cell(r.get(c)) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    lines = [" | ".join(c.ljust(w) for c, w in zip(cols, widths)).rstrip(), "-+-".join("-" * w for w in widths)]
    lines += [" | ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip() for row in cells]
    return lines


def count_records(state: RepairState, inp: CountRecordsInput) -> str:
    """Counts only: the query is built here from the structured input, never from model SQL."""
    k = state.min_cell_count
    try:
        with _readonly(state.conn):
            apply_session(state.conn, state.cfg)  # compare dates and times as the build does
            resolved = _resolve_table(state, inp.table)
            if isinstance(resolved, str):
                return resolved
            logical, physical, table = resolved
            public = logical == "vocab"
            columns = _columns_of(state.conn, physical, table)

            def column(name: str) -> str | None:
                bare = name.strip().strip('"')
                if "." in bare and bare.split(".", 1)[0].lower() in (table.lower(), logical):
                    bare = bare.split(".", 1)[1]
                return _find_name(bare, columns)

            group_by: list[str] = []
            for name in inp.group_by:
                col = column(name)
                if col is None:
                    return f"ERROR: {logical}.{table} has no column {name!r}. Columns: {', '.join(columns)}"
                if col not in group_by:
                    group_by.append(col)
            filters: list[tuple[RecordFilter, str]] = []
            for f in inp.filters:
                col = column(f.column)
                if col is None:
                    return f"ERROR: {logical}.{table} has no column {f.column!r}. Columns: {', '.join(columns)}"
                filters.append((f, col))

            for col in dict.fromkeys(group_by + [c for _, c in filters]):
                reason = _denial(state, logical, table, col)
                if reason:
                    return (
                        f"REFUSED: {logical}.{table}.{col} is {reason}; it cannot be grouped or filtered. "
                        "Identifiers can only be counted: drop it from group_by and filters."
                    )
            if group_by and not public and not state.cfg.privacy.send_values_to_llm:
                values = [c for c in group_by if not c.lower().endswith("_concept_id")]
                if values:
                    return (
                        f"REFUSED: privacy.send_values_to_llm is false, so only *_concept_id columns (and vocabulary "
                        f"tables) can be grouped; {', '.join(values)} would show source values. Use filters and "
                        "totals instead."
                    )

            where: list[pgsql.Composable] = []
            params: list[str] = []
            for f, col in filters:
                built = _filter_sql(f, col)
                if isinstance(built, str):
                    return f"ERROR: {built}"
                where.append(built[0])
                params += built[1]
            src = pgsql.SQL("FROM {} WHERE {}").format(
                pgsql.Identifier(physical, table), pgsql.SQL(" AND ").join(where) if where else pgsql.SQL("TRUE")
            )
            label = f"{logical}.{table}"
            if filters:
                label += " WHERE " + " AND ".join(_describe_filter(f, c) for f, c in filters)

            if not group_by:
                n = int(_rows(state.conn, pgsql.SQL("SELECT COUNT(*) AS n {}").format(src), params)[0]["n"])
                if public or n >= k:
                    return f"{label}: n = {n}"
                return f"{label}: n < {k} (totals below the minimum cell count of {k} are not shown)"

            label += " GROUP BY " + ", ".join(group_by)
            idents = pgsql.SQL(", ").join(pgsql.Identifier(c) for c in group_by)
            n_name = "n" if "n" not in group_by else "n_records"
            threshold = 0 if public else k
            # positional ORDER BY: the count column, then the group columns
            order = ", ".join([f"{len(group_by) + 1} DESC"] + [f"{i + 1} NULLS LAST" for i, _ in enumerate(group_by)])
            main = pgsql.SQL(
                "SELECT {g}, COUNT(*) AS {n} {src} GROUP BY {g} HAVING COUNT(*) >= %s ORDER BY {o} LIMIT %s"
            ).format(g=idents, n=pgsql.Identifier(n_name), src=src, o=pgsql.SQL(order))
            rows = _rows(state.conn, main, [*params, threshold, inp.limit])
            stats = pgsql.SQL(
                "SELECT COUNT(*) FILTER (WHERE n >= %s) AS shown, COUNT(*) FILTER (WHERE n < %s) AS withheld "
                "FROM (SELECT COUNT(*) AS n {src} GROUP BY {g}) s"
            ).format(src=src, g=idents)
            st = _rows(state.conn, stats, [threshold, threshold, *params])[0]
    except psycopg.Error as e:
        values = {str(v) for f in inp.filters for v in [f.value, *f.values] if v is not None}
        return f"ERROR: {redact_error(short_error(e), known=state.known_identifiers() | values)}"

    shown, withheld = int(st["shown"]), int(st["withheld"])
    lines = [label]
    if rows:
        lines += _table_text(rows, group_by + [n_name])
    footer = f"({len(rows)} group(s) shown"
    if shown > len(rows):
        footer += f" of {shown}; raise limit (max 50) or add filters to see the rest"
    if public:
        footer += "; vocabulary tables are public reference data, not suppressed"
    else:
        footer += f"; {withheld} group(s) with fewer than {k} records withheld (small-cell suppression)"
    lines.append(footer + ")")
    return "\n".join(lines)


# ---- error redaction -----------------------------------------------------------------

WITHHELD = "<value withheld>"
_ERROR_DETAIL = re.compile(r"\s*DETAIL:.*?(?=\s*HINT:|$)", re.S)
_ERROR_HINT = re.compile(r"(\s*HINT:)")
_ERROR_OUT_OF_RANGE = re.compile(r"(out of range: )(?!\")(\S+)")
_ERROR_QUOTED = re.compile(r'"([^"]*)"')
# PostgreSQL message fragments that may separate two quoted identifiers, e.g.
# 'column "x" of relation "y"', 'relation "t" violates check constraint "c"'.
_ERROR_CONNECTOR = re.compile(
    r"\s+(of relation|of table|on table|in table|for|for table|or the column|and|violates "
    r"(check|foreign key|unique|not-null|exclusion) constraint)\s+",
    re.I,
)
# Text right before a quoted slot that holds a value, never an identifier: 'syntax for type date: "..."',
# 'value "..." is out of range', 'invalid value "..." for "MM"'.
_ERROR_VALUE_SLOT = re.compile(r"(:|\bvalue)\s*$", re.I)


@lru_cache(maxsize=1)
def _static_identifiers() -> frozenset[str]:
    from ..cdm.ddl import ETL_TABLES

    cdm = load_cdm_spec()
    names = {t.name for t in cdm.tables.values()} | {f.name for t in cdm.tables.values() for f in t.fields}
    names |= {f"xpk_{t.name}" for t in cdm.tables.values()}  # official primary key constraint names
    for table, ddl in ETL_TABLES.items():
        names |= {table, f"{table}_pkey"}
        if table not in _ETL_BOOKKEEPING:
            names |= set(re.findall(r"^\s+(\w+)\s", ddl, re.M))
    names |= {"person_map_person_id_key", "visit_map_visit_occurrence_id_key",
              "person_map_retired_person_id_key", "visit_map_retired_visit_occurrence_id_key"}
    return frozenset(n.lower() for n in names)


def known_identifiers(profile: SourceProfile | None, schemas: Schemas | None) -> set[str]:
    """CDM and ETL table/field names, staged source table/column names and the schema names."""
    known = set(_static_identifiers())
    if profile is not None:
        for t in profile.tables:
            known.add(t.name)
            known |= {c.name for c in t.columns}
    if schemas is not None:
        known |= {schemas.cdm, schemas.vocab, schemas.src, schemas.etl, schemas.results}
    return known


def _context_identifiers(context: str | None) -> set[str]:
    """Identifiers written in the SQL (as written and as PostgreSQL folds them); literals excluded."""
    bare = strip_literals(context or "")
    out = {m.replace('""', '"') for m in _QUOTED_IDENT.findall(bare)}
    for word in _IDENT.findall(_QUOTED_IDENT.sub(" ", bare)):
        out |= {word, word.lower()}
    return out


def _redact_segment(seg: str, known: set[str]) -> str:
    """Keep quoted text only when it is exactly a known identifier in an identifier slot.

    Values can contain double quotes, so from the first doubtful quoted token everything up to the
    last quote of the segment is withheld (text after it is PostgreSQL's own)."""
    if '"' not in seg:
        return seg
    tokens = list(_ERROR_QUOTED.finditer(seg))
    start: int | None = None
    for i, m in enumerate(tokens):
        before = seg[tokens[i - 1].end() if i else 0 : m.start()]
        if i and not _ERROR_CONNECTOR.fullmatch(before):
            start = tokens[i - 1].start()  # unexpected text between two quoted tokens: may be a value
            break
        value = m.group(1)
        if _ERROR_VALUE_SLOT.search(before) or not value or not all(p in known for p in value.split(".")):
            start = m.start()
            break
    if start is None:
        if seg.count('"') == 2 * len(tokens):
            return seg
        start = seg.index('"')  # a stray quote: the pairing above can't be trusted
    last = seg.rfind('"')
    return seg[:start] + f'"{WITHHELD}"' + (seg[last + 1 :] if last > start else "")


def redact_error(message: str | None, context: str | None = "", known: Iterable[str] | None = None) -> str:
    """A database error message with record values removed, fit for a prompt.

    PostgreSQL errors can quote data: ``DETAIL: Failing row contains (...)``, ``Key (id)=(42)``,
    ``invalid input syntax for type date: "31/02/1990"``, ``relation "<value>" does not exist``
    (a value cast to ``regclass``). DETAIL sections are dropped. Quoted text (in the message and
    in HINT) is kept only when it is exactly a known identifier - a CDM table/field name, a staged
    source table/column, a schema name (``known``) or an identifier written in ``context`` (the
    SQL) - and does not sit in a value slot; everything else becomes ``"<value withheld>"``.
    """
    msg = _ERROR_DETAIL.sub(" DETAIL: (withheld: may contain record values)", message or "")
    names = set(_static_identifiers()) | _context_identifiers(context) | set(known or ())
    parts = _ERROR_HINT.split(msg)  # [message, " HINT:", hint, ...]
    msg = "".join(p if i % 2 else _redact_segment(p, names) for i, p in enumerate(parts))
    return _ERROR_OUT_OF_RANGE.sub(rf"\1{WITHHELD}", msg)


# ---- spec edits ----------------------------------------------------------------------

_TABLE_REF = re.compile(
    r"(?:\{(?P<ph>src|vocab|etl)\}\.)?(?:(?P<bare>[A-Za-z_][A-Za-z0-9_$]*)|\"(?P<quoted>[^\"]+)\")"
)


def _table_ref_issue(state: RepairState, ref: str) -> str | None:
    m = _TABLE_REF.fullmatch(ref.strip())
    if not m:
        return (
            f"table {ref!r} must be a staged source table name, or a table qualified with {{src}}, {{vocab}} or "
            "{etl} (other schemas and subqueries are not allowed)"
        )
    schema = getattr(state.schemas, m.group("ph") or "src")
    # as the compiler emits it: a bare name is quoted as written, a qualified one is folded by PostgreSQL
    name = m.group("quoted") or (m.group("bare").lower() if m.group("ph") else m.group("bare"))
    if name not in _tables_in(state.conn, schema):
        where = f"{{{m.group('ph')}}}" if m.group("ph") else "the staged source"
        return f"table {ref!r} does not exist in {where}"
    return None


def _pii_tokens(state: RepairState, tm: TableMapping) -> list[str]:
    """Direct-identifier columns of the mapping's source tables named anywhere in its SQL (also
    unqualified); complements ``privacy.pii_references``, which follows alias.column references."""
    pii = pii_columns(state.profile, state.cfg.privacy)
    texts = [f.expression for f in tm.fields] + [tm.where, tm.dedup_order] + [j.on for j in tm.joins]
    tokens = set().union(*(_tokens(t) for t in texts))
    hits: list[str] = []
    for ref in [tm.source.table] + [j.table for j in tm.joins]:
        table = _source_table_name(ref)
        if table is None:
            continue
        cols = set(pii.get(table.lower(), set()))
        cols |= {c.lower() for c in _columns_of(state.conn, state.schemas.src, table) if _pii_pattern(c, state)}
        hits += [f"{table}.{c}" for c in sorted(tokens & cols)]
    return hits


_ID_MAP_KEYS = {
    ("person", "person_id"): "person id map (kind person_id)",
    ("visit_occurrence", "visit_occurrence_id"): "visit id map (kind visit_id)",
}


def mapping_issues(state: RepairState, tm: TableMapping) -> list[str]:
    """Why ``tm`` may not be applied (empty when it may); checked before compiling."""
    issues: list[str] = []
    for fm in tm.fields:
        issues += [f"{fm.target_field} expression {i}" for i in fragment_issues(fm.expression)]
    issues += [f"where {i}" for i in fragment_issues(tm.where)]
    issues += [f"dedup_order {i}" for i in fragment_issues(tm.dedup_order)]
    if tm.dedup_order:
        from ..design.validate import paren_problem

        problem = paren_problem(tm.dedup_order)
        if problem:
            issues.append(f"dedup_order has unbalanced parentheses ({problem}); it must be an ORDER BY list")
    for j in tm.joins:
        issues += [f"join {j.alias} condition {i}" for i in fragment_issues(j.on)]
    with _readonly(state.conn):
        for ref in [tm.source.table] + [j.table for j in tm.joins]:
            issue = _table_ref_issue(state, ref)
            if issue:
                issues.append(issue)
        pii = list(dict.fromkeys(pii_references(tm, state.profile, state.cfg.privacy) + _pii_tokens(state, tm)))
    if pii:
        issues.append(
            f"reads direct-identifier (PII) column(s) {', '.join(pii)}; direct identifiers must never be loaded "
            "into the CDM"
        )
    target = tm.target_table.lower()
    for fm in tm.fields:
        name = fm.target_field.lower()
        id_map = _ID_MAP_KEYS.get((target, name))
        if fm.kind == "auto_id" and id_map:
            issues.append(f"{target}.{name} cannot be auto_id: it must come from the {id_map}")
        if name == "person_id" and fm.kind != "person_id":
            issues.append(f"{target}.person_id must use kind person_id (the person id map), not {fm.kind}")
    return issues


def validate_mapping(
    tm: TableMapping,
    spec: ETLSpec,
    conn: psycopg.Connection,
    schemas: Schemas,
    cfg: ProjectConfig | None = None,
    old: TableMapping | None = None,
) -> str | None:
    """None if ``tm`` compiles and its INSERT EXPLAINs cleanly (under the build's session
    settings), else the error. With ``old`` (the mapping ``tm`` replaces) the edit is also
    dry-run: a change that makes a required field whose NULL drops the row (a start date,
    year_of_birth, death_date) NULL for rows where it was not would drop those records, one that
    makes a record's visit key NULL would hide the records that lose their visit, and one that
    opens closed observation period episodes or closes open ones on their start date would change
    observation time, which the repair agent may not do (``drop_guard``)."""
    from ..cdm.ddl import create_etl_tables
    from ..etl.compiler import compile_table

    etl = cfg.etl if cfg is not None else None
    try:
        sql = compile_table(tm, spec, schemas, etl=etl)
    except Exception as e:  # CompileError, or anything else the compiler trips over
        return f"does not compile: {e}"
    try:
        with rollback_only(conn):
            create_etl_tables(conn, schemas)  # the INSERT calls the ETL helper functions
            apply_session(conn, cfg)
            execute(conn, "EXPLAIN\n" + sql.rstrip().rstrip(";"))
    except psycopg.Error as e:
        return f"EXPLAIN failed: {redact_error(short_error(e), sql, known_identifiers(None, schemas))}"
    return drop_guard(old, tm, spec, conn, schemas, cfg) if old is not None else None


def repair_system(cfg: ProjectConfig | None = None) -> str:
    """``REPAIR_SYSTEM`` with the partial-date rule of this project (``cfg.etl.partial_dates``)."""
    if cfg is not None and cfg.etl.partial_dates == "reject":
        rule = ("is left NULL (etl.partial_dates: reject; birth and death dates follow THEMIS: the known year and "
                "month of birth are kept, a death date takes the last day of the known period)")
    else:
        rule = ("is completed by a fixed rule (a start date takes the first day, an end date the last, other dates "
                "the 15th; birth and death dates follow THEMIS)")
    return REPAIR_SYSTEM.replace("<PARTIAL_DATES>", rule)


def drop_guard(
    old: TableMapping, new: TableMapping, spec: ETLSpec, conn: psycopg.Connection, schemas: Schemas,
    cfg: ProjectConfig | None = None,
) -> str | None:
    """Why the edit ``old`` -> ``new`` may not be applied because it would drop records, take their
    visit key away, or change which observation period episodes are open (None: it may). For each
    changed required field whose NULL drops the row, and each changed visit key of a record's visit
    link, the rows where it becomes NULL are counted (``etl.compiler.nulling_count_sql``; for an
    open end, the episodes it opens or closes, ``_open_end_issue``) under ``COUNT_TIMEOUT_MS``;
    when the old value does not fit the new mapping's sources (a renamed alias), the NULLs of each
    mapping over its own rows are compared instead (not for an open end). A count that times out
    or fails lets the edit through: the build history still reverts a build that lost records."""
    from ..etl.compiler import nulling_count_sql

    try:
        counts = nulling_count_sql(old, new, spec, schemas, etl=cfg.etl if cfg is not None else None,
                                   na_values=cfg.source.effective_na_values if cfg is not None else ())
    except Exception:  # validate_mapping compiled it already; nothing to guard
        return None
    k = cfg.privacy.min_cell_count if cfg is not None else PrivacyConfig().min_cell_count
    for name, sql in counts.items():
        row = _guard_count(conn, schemas, cfg, sql.rows)
        if sql.what == "open":
            issue = _open_end_issue(row, new.target_table, name, k) if row is not None else None
            if issue:
                return issue
            continue
        if row is not None:
            n = int(row["n"])
        else:
            new_t = _guard_count(conn, schemas, cfg, sql.new_totals)
            old_t = _guard_count(conn, schemas, cfg, sql.old_totals) if sql.old_totals else None
            if new_t is None or old_t is None:
                continue
            n = (int(new_t["n"]) - int(new_t["v"])) - (int(old_t["n"]) - int(old_t["v"]))
        if n > 0 and sql.what == "visit_key":
            return (
                f"it would take the visit key away from {_small_count(n, k)} rows of {new.target_table}: "
                f"{new.target_table}.{name} becomes NULL for them (their key is not NULL with the current mapping), "
                "which hides records that lose their visit instead of linking them. Keep the key: write it so that it "
                "gives the visit mapping's key (etlVisitLinkage counts the records whose key links to no visit of "
                "their person)"
            )
        if n > 0:
            return (
                f"it would drop {_small_count(n, k)} rows of {new.target_table}: the required field "
                f"{new.target_table}.{name} becomes NULL for them (it is not NULL with the current mapping), and a "
                "record without it cannot be loaded. Keep the rows: fix the expression so it still gives a value "
                "for them"
            )
    return None


def _open_end_issue(row: dict, table: str, name: str, k: int) -> str | None:
    """Why an edit of an open end (``etl.compiler.nulling_count_sql``, what 'open') changes which
    episodes are open: it opens episodes with real, differing exit dates (one placeholder exit date
    shared by all of them may become NULL), or closes open ones on their start date."""
    opened, values, closed = int(row.get("n") or 0), int(row.get("v") or 0), int(row.get("m") or 0)
    if opened and values > 1:
        return (
            f"it would turn {_small_count(opened, k)} closed episodes of {table} into open ones that run to the "
            f"release date (or death): {table}.{name} becomes NULL for them where the current mapping reads an exit "
            "date. Keep the exit dates: an episode that ends before it starts is dropped and counted "
            "(etlObservationPeriodMerge), so fix how its dates are read, or leave it for a person to decide. (A "
            "placeholder exit date, one value for every such episode, may become NULL.)"
        )
    if closed:
        return (
            f"it would close {_small_count(closed, k)} open episodes of {table} on their start date: {table}.{name} "
            "gets the start date where the current mapping has no exit date. Leave the end NULL for open episodes: "
            "the build ends them at the release date or at death"
        )
    return None


def _guard_count(conn: psycopg.Connection, schemas: Schemas, cfg: ProjectConfig | None, sql: str) -> dict | None:
    """One read-only count of ``drop_guard`` under ``COUNT_TIMEOUT_MS`` (None: it failed or timed out)."""
    from ..cdm.ddl import create_etl_tables

    try:
        with rollback_only(conn, COUNT_TIMEOUT_MS):
            create_etl_tables(conn, schemas)  # the count calls the ETL helper functions
            apply_session(conn, cfg)
            execute(conn, "SET LOCAL transaction_read_only = on")
            return dict(fetch_one(conn, sql) or {}) or None
    except psycopg.Error:
        return None


def _norm_sql(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def data_loss_issues(old: TableMapping | None, new: TableMapping) -> list[str]:
    """Edits that can only remove records or data, which the repair agent may not make: a new or
    changed filter, a new or changed inner join, a different source table, or a populated field
    turned to NULL. (An added mapping may filter: it adds records, for example the visits of a
    clinic not loaded yet. It must not route records by the domain of their concept: the build
    moves them itself, so such a mapping loads them twice, see ``routing_issues``.)"""
    if old is None:
        return []
    out: list[str] = []
    if old.source.table.lower() != new.source.table.lower():
        out.append(f"it changes the source table ({old.source.table} → {new.source.table})")
    if new.where and _norm_sql(new.where) != _norm_sql(old.where):
        out.append(f"it {'changes' if old.where else 'adds'} the filter (where: {new.where})")
    old_joins = {j.alias.lower(): j for j in old.joins}
    for j in new.joins:
        before = old_joins.get(j.alias.lower())
        if j.join_type == "inner" and (
            before is None
            or before.join_type != "inner"
            or _norm_sql(before.on) != _norm_sql(j.on)
            or before.table.lower() != j.table.lower()
        ):
            out.append(f"it adds or changes an inner join ({j.table} {j.alias} on {j.on})")
    old_fields = {f.target_field: f for f in old.fields}
    for f in new.fields:
        before = old_fields.get(f.target_field)
        if f.kind == "null" and before is not None and before.kind != "null":
            out.append(f"it blanks {new.target_table}.{f.target_field} (was {before.kind})")
    for name, before in old_fields.items():
        if before.kind != "null" and name not in {f.target_field for f in new.fields}:
            out.append(f"it removes the mapping of {new.target_table}.{name}")
    return out


def routing_issues(old: TableMapping | None, new: TableMapping, spec: ETLSpec, new_spec: ETLSpec) -> list[str]:
    """Edits that route records by the domain of their concept by hand while the build routes them
    (``ETLSpec.route_by_domain``), which load records twice: a filter on the domain of
    {vocab}.concept that the mapping did not have (``etl.routing.hand_routing_issue``), or a new
    or retargeted mapping that reads the same codes from the same rows of a source table as one
    into another event table (``etl.routing.shared_codes``). The pairs are compared by source and
    target tables, so an edit of mappings that were already a pair (the same fix made to both) is
    not refused."""
    from ..etl.routing import hand_routing_issue, shared_codes, source_table_pairs

    out: list[str] = []
    issue = hand_routing_issue(new, new_spec)
    if issue and (old is None or hand_routing_issue(old, spec) is None):
        out.append(f"{new.target_table}: {issue}")
    before = source_table_pairs(spec)
    out += [p.message for p in shared_codes(new_spec) if p.key not in before]
    return out


def era_issues(spec: ETLSpec, new: TableMapping, old: TableMapping | None = None) -> list[str]:
    """A new mapping into an era table while the build derives eras (``ETLSpec.derive_eras``): it
    would replace the OHDSI era step (``etl.compiler.derived_era_tables``), or load the eras of an
    era table that the spec's own SQL loads a second time; either is for a person to decide. An
    edit of a mapping the spec already has into an era table (``old``: it replaced the step
    before) is allowed like any other."""
    from ..cdm.spec import DERIVED_ERAS
    from ..etl.compiler import derived_era_tables, spec_era_tables

    t = new.target_table.lower()
    if not spec.derive_eras or t not in DERIVED_ERAS or (old is not None and old.target_table.lower() == t):
        return []
    source = DERIVED_ERAS[t]
    if t in derived_era_tables(spec):
        return [f"the build derives {t} from {source} with the OHDSI era logic (derive_eras), and a mapping into {t} "
                f"would replace it: fix the {source} records (their concepts and dates) instead"]
    if t in spec_era_tables(spec):
        return [f"the spec's own SQL already loads {t}, and a new mapping into it would load its eras twice: fix the "
                f"{source} records instead"]
    return [f"the build derives {t} from the {source} records (derive_eras), and the spec loads none: load the "
            f"{source} records instead of mapping into {t}"]


def observation_period_end_issues(old: TableMapping | None, new: TableMapping) -> list[str]:
    """An edit of observation_period_end_date that closes the episodes the source leaves open, or
    opens those it closes (``design.validate.observation_period_end_issue``: a fallback onto the start
    date, a date literal or the build date, a constant, or a NULL that depends on the start date),
    which the repair agent may not make: it would change observation time to make a check pass. An
    end the edit leaves as it was is not judged again (it was approved with the spec)."""
    if new.target_table.lower() != "observation_period":
        return []
    from ..design.validate import observation_period_end_issue

    def end(tm: TableMapping | None) -> tuple[str, str] | None:
        fm = next((f for f in tm.fields if f.target_field == "observation_period_end_date"), None) if tm else None
        return (fm.kind, _norm_sql(fm.expression)) if fm is not None else None

    if old is not None and end(old) == end(new):
        return []
    issue = observation_period_end_issue(new)
    return [f"observation_period.observation_period_end_date: {issue}"] if issue else []


# Tables whose rows decide who exists in the CDM (people, visits, deaths).
IDENTITY_TABLES = ("person", "visit_occurrence", "death")
_KEY_FIELDS = {"person": ("person_id", "person_id"), "visit_occurrence": ("visit_occurrence_id", "visit_id")}


def identity_issues(old: TableMapping | None, new: TableMapping, *, old_runs: bool = True) -> list[str]:
    """Edits that change who exists in the CDM or whose records are whose, which only a person may
    decide: a changed person_id expression (in any table: it re-attributes records), a changed key
    of person or visit_occurrence, a new mapping into person, visit_occurrence or death, and a
    removed or changed filter of those tables (a new or changed filter is also data loss). When
    the old mapping cannot run at all (``old_runs`` False: it does not compile or EXPLAIN, e.g. a
    misspelled key column), it never loaded a record, so its keys may be fixed."""
    target = new.target_table.lower()
    if old is None:
        if target in IDENTITY_TABLES:
            return [f"it adds a mapping into {target}, which decides who (or which visits and deaths) exist in "
                    "the CDM"]
        return []
    out: list[str] = []
    before = {f.target_field: f for f in old.fields}
    after = {f.target_field: f for f in new.fields}
    key_field, key_kind = _KEY_FIELDS.get(target, (None, None))
    for name in sorted(set(before) | set(after)) if old_runs else []:
        a, b = before.get(name), after.get(name)
        ids = {f.kind for f in (a, b) if f is not None} & {"person_id", "visit_id"}
        if "person_id" not in ids and not (name == key_field and key_kind in ids):
            continue
        if a is None or b is None or (a.kind, _norm_sql(a.expression)) != (b.kind, _norm_sql(b.expression)):
            what = "the key" if name == key_field else "the person key"
            out.append(f"it changes {what} of {target}.{name} ({_fm_text(a)} → {_fm_text(b)}), which re-attributes "
                       "records or changes who exists in the CDM")
    if target in IDENTITY_TABLES and old.where and _norm_sql(old.where) != _norm_sql(new.where):
        out.append(f"it {'changes' if new.where else 'removes'} the filter of {target} (where: {old.where})")
    return out


def _apply(
    state: RepairState,
    pos: int | None,
    new_tm: TableMapping,
    *,
    target_field: str | None,
    change: str,
    rationale: str,
) -> str:
    old_tm = state.spec.table_mappings[pos] if pos is not None else None
    eras = era_issues(state.spec, new_tm, old_tm)
    if eras:
        return ("REJECTED: " + "; ".join(eras) + ". Describe the change you would make in your summary, for a person "
                "to decide.")
    loss = data_loss_issues(old_tm, new_tm)
    if loss:
        return (
            "REJECTED: " + "; ".join(loss) + ". The repair agent may not remove records or data to make a check "
            "pass. If records really should be excluded, leave the spec as it is and explain why in your summary, "
            "for a person to decide."
        )
    periods = observation_period_end_issues(old_tm, new_tm)
    if periods:
        return (
            "REJECTED: " + "; ".join(periods) + ". The repair agent may not change how long people are observed to "
            "make a check pass. If the exit dates are read wrongly, fix how they are read; otherwise leave the spec as "
            "it is and explain in your summary, for a person to decide."
        )
    mappings = list(state.spec.table_mappings)
    if pos is None:
        mappings.append(new_tm)
    else:
        mappings[pos] = new_tm
    new_spec = state.spec.model_copy(update={"table_mappings": mappings}, deep=True)
    routed = routing_issues(old_tm, new_tm, state.spec, new_spec)
    if routed:
        return (
            "REJECTED: " + "; ".join(routed) + ". The build moves every record to the table of its concept's domain "
            "itself. Leave the mappings as they are; fix a wrong concept in its lookup."
        )
    issues = mapping_issues(state, new_tm)
    if issues:
        return "REJECTED: " + "; ".join(issues)
    identity = identity_issues(old_tm, new_tm)
    if identity and old_tm is not None and validate_mapping(old_tm, state.spec, state.conn, state.schemas,
                                                            state.cfg) is not None:
        identity = identity_issues(old_tm, new_tm, old_runs=False)  # it never loaded a record
    if identity:
        return (
            "REJECTED: " + "; ".join(identity) + ". Who exists in the CDM, and whose records are whose, is decided "
            "by people: leave the spec as it is and describe the change you would make in your summary, for a "
            "person to decide."
        )
    error = validate_mapping(new_tm, new_spec, state.conn, state.schemas, state.cfg, old=old_tm)
    if error:
        return f"REJECTED: {error}"
    state.spec = new_spec
    state.actions.append(
        RepairAction(
            iteration=state.iteration,
            target_table=new_tm.target_table,
            target_field=target_field,
            change=change,
            rationale=rationale,
            trigger=state.trigger_for(new_tm.target_table),
        )
    )
    return (
        f"OK: {new_tm.target_table}{'.' + target_field if target_field else ''} updated ({change}). "
        "The table compiles and EXPLAINs cleanly; runtime errors (e.g. NOT NULL violations) show up when "
        "the ETL is rebuilt after you finish."
    )


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------


def build_repair_tools(state: RepairState) -> list[AgentTool]:
    """The repair agent's tools, bound to ``state`` (spec edits are applied to ``state.spec``)."""

    def get_table_mapping(inp: GetTableMappingInput) -> str:
        pos, n = _locate(state.spec, inp.target_table, inp.mapping_index)
        if pos is None:
            return (
                f"ERROR: no mapping #{inp.mapping_index} for {inp.target_table} (it has {n}). "
                f"Mapped tables: {_mapped_tables(state.spec)}"
            )
        tm = state.spec.table_mappings[pos]
        text = f"# TableMapping {tm.target_table} #{inp.mapping_index} ({n} mapping(s) feed this table)\n{_yaml(tm)}"
        from ..etl.compiler import compile_table

        try:
            text += "\n-- compiled SQL\n" + compile_table(tm, state.spec, state.schemas, etl=state.cfg.etl)
        except Exception as e:
            text += f"\n-- the mapping does not compile: {e}"
        return text

    def get_cdm_table(inp: GetCdmTableInput) -> str:
        try:
            return load_cdm_spec().describe(inp.table)
        except KeyError as e:
            return f"ERROR: {e.args[0] if e.args else e}"

    def get_source_profile(inp: GetSourceProfileInput) -> str:
        if state.profile is None:
            return "profile not available"
        if state.profile.table(inp.table) is None:
            names = ", ".join(t.name for t in state.profile.tables)
            return f"ERROR: no source table named {inp.table!r}. Profiled tables: {names}"
        return profile_prompt_text(state.profile, state.cfg.privacy, tables=[inp.table])

    def count(inp: CountRecordsInput) -> str:
        return count_records(state, inp)

    def update_field_mapping(inp: UpdateFieldMappingInput) -> str:
        pos, n = _locate(state.spec, inp.target_table, inp.mapping_index)
        if pos is None:
            return (
                f"REJECTED: no mapping #{inp.mapping_index} for {inp.target_table} (it has {n}); use "
                f"replace_table_mapping to add one. Mapped tables: {_mapped_tables(state.spec)}"
            )
        old_tm = state.spec.table_mappings[pos]
        target_field = inp.target_field.strip().lower()
        cdm = load_cdm_spec()
        if cdm.has_table(old_tm.target_table) and cdm.table(old_tm.target_table).field(target_field) is None:
            return f"REJECTED: {old_tm.target_table} has no field {target_field!r}"
        new_fm = FieldMapping(
            target_field=target_field,
            kind=inp.kind,
            expression=inp.expression,
            vocabulary=inp.vocabulary,
            target_domain=inp.target_domain,
            concept_role=inp.concept_role,
            rationale=inp.rationale,
        )
        old_fm = next((f for f in old_tm.fields if f.target_field == target_field), None)
        if old_fm is not None and _same_logic(old_fm, new_fm):
            return (
                f"UNCHANGED: {old_tm.target_table}.{target_field} already maps as {_fm_text(old_fm)}; nothing to "
                "apply (no repair recorded)."
            )
        if old_fm is not None:
            fields = [new_fm if f.target_field == target_field else f for f in old_tm.fields]
        else:
            fields = [*old_tm.fields, new_fm]
        new_tm = old_tm.model_copy(update={"fields": fields})
        change = f"{_fm_text(old_fm)} → {_fm_text(new_fm)}"
        return _apply(state, pos, new_tm, target_field=target_field, change=change, rationale=inp.rationale)

    def replace_table_mapping(inp: ReplaceTableMappingInput) -> str:
        try:
            data = yaml.safe_load(inp.mapping_yaml)
        except yaml.YAMLError as e:
            return f"REJECTED: invalid YAML: {e}"
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        if not isinstance(data, dict):
            return "REJECTED: mapping_yaml must be one TableMapping written as a YAML mapping"
        data.setdefault("target_table", inp.target_table)
        if str(data["target_table"]).lower() != inp.target_table.lower():
            return f"REJECTED: mapping_yaml is for {data['target_table']!r}, not {inp.target_table!r}"
        try:
            new_tm = TableMapping.model_validate(data)
        except ValidationError as e:
            return f"REJECTED: not a valid TableMapping: {e}"[:2000]
        pos, n = _locate(state.spec, inp.target_table, inp.mapping_index)
        if pos is None and inp.mapping_index != n:
            return f"REJECTED: {inp.target_table} has {n} mapping(s); mapping_index must be 0..{n}"
        old_tm = state.spec.table_mappings[pos] if pos is not None else None
        if old_tm is not None and _same_mapping_logic(old_tm, new_tm):
            return (
                f"UNCHANGED: the new {inp.target_table} mapping #{inp.mapping_index} is the same as the current "
                "one; nothing to apply (no repair recorded)."
            )
        change = _mapping_diff(old_tm, new_tm)
        return _apply(state, pos, new_tm, target_field=None, change=change, rationale=inp.rationale)

    k = state.min_cell_count
    return [
        AgentTool(
            "get_table_mapping",
            "Show one table mapping of the ETL spec as YAML, followed by the INSERT ... SELECT it compiles to.",
            GetTableMappingInput,
            get_table_mapping,
        ),
        AgentTool(
            "get_cdm_table",
            "Describe an OMOP CDM v5.4 table: its fields, which are required, and the OHDSI ETL conventions.",
            GetCdmTableInput,
            get_cdm_table,
        ),
        AgentTool(
            "get_source_profile",
            "Show the privacy-filtered profile of one source table: column types, null rates, distinct counts, "
            "ranges and frequent values (rare values and identifiers withheld).",
            GetSourceProfileInput,
            get_source_profile,
        ),
        AgentTool(
            "count_records",
            "Count the records of one table - a staged source table (src.<table>), a CDM table (cdm.<table>), a "
            "vocabulary table (vocab.<table>) or etl.source_to_concept_map - optionally grouped by up to 3 of "
            "its columns and filtered with simple conditions (combined with AND). Returns counts only, largest "
            f"groups first: groups with fewer than {k} records are withheld (their number is reported) and a "
            f"total below {k} is shown as '<{k}'. Vocabulary tables are public reference data and are not "
            "suppressed. Patient, visit and record identifiers, direct identifiers (PII) and CDM *_id columns "
            "(except *_concept_id) cannot be grouped or filtered, and the person/visit id maps cannot be counted.",
            CountRecordsInput,
            count,
        ),
        AgentTool(
            "update_field_mapping",
            "Set how one CDM field is populated (replaces the field's mapping, or adds it). Applied only if it "
            "passes the SQL and privacy checks and the table still compiles and EXPLAINs cleanly; otherwise "
            "returns REJECTED with the reason.",
            UpdateFieldMappingInput,
            update_field_mapping,
        ),
        AgentTool(
            "replace_table_mapping",
            "Replace a whole table mapping (source, joins, where, fields) with a new one given as YAML. Applied "
            "only if it passes the SQL and privacy checks and compiles and EXPLAINs cleanly; otherwise returns "
            "REJECTED with the reason.",
            ReplaceTableMappingInput,
            replace_table_mapping,
        ),
    ]


# --------------------------------------------------------------------------------------
# The repair step
# --------------------------------------------------------------------------------------


def repair_triggers(
    dq: DQReport | None, etl_result: ETLRunResult | None, cfg: ProjectConfig, spec: ETLSpec | None = None
) -> tuple[list[TableLoadResult], list[CheckResult]]:
    """The errors of blocking steps that run spec SQL, if there are any; otherwise the repairable DQ
    failures in ``repair_severities`` (``checks.repairable``). Errors of steps the build generates
    (id-map pruning, routing, eras, ...) and errors a spec change cannot fix (a missing privilege,
    ``TableLoadResult.repairable``) are never given to the agent: its tools cannot change them; nor
    are the checks of the era tables the build derived, or that ``spec`` loads itself while the
    build derives eras (``checks.derived_tables``)."""
    if etl_result is not None:
        fixable = [
            s for s in etl_result.blocking_errors() if (s.kind is None or s.kind in SPEC_KINDS) and s.repairable
        ]
        if fixable:
            return fixable, []
    if dq is None:
        return [], []
    severities = set(cfg.quality.repair_severities)
    order = {"fatal": 0, "convention": 1, "characterization": 2}
    derived = derived_tables(etl_result, spec)
    failures = [r for r in dq.results if repairable(r, severities, derived=derived)]
    return [], sorted(failures, key=lambda r: (order.get(r.severity, 3), -r.pct_violated_rows))


def _clip(text: str | None, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n... (truncated)"


def _clip_sql(sql: str | None, n: int) -> str:
    """``_clip`` for the SQL of a step or check. A table mapping's INSERT computes the spec's values
    in its innermost layer, at the end of the text, after the column list and the outer SELECTs:
    when it is too long, it is shown from that layer."""
    text = (sql or "").strip()
    start = text.find(LAYER_1_MARKER)
    if len(text) > n and start > 0:
        text = "-- (the INSERT column list and the outer SELECTs are left out)\n" + text[start:]
    return _clip(text, n)


def _small_count(n: int | None, k: int) -> str:
    """A record count for a prompt: counts between 1 and k-1 are shown as '<k'."""
    if n is None:
        return "?"
    return f"<{k}" if 0 < n < k else str(n)


# "<n> of <m>" in a check description: the form in which descriptions give counts (not inside a date
# or a decimal number)
_COUNT_PAIR = re.compile(r"(?<![\w.<-])(\d+) of (\d+)\b(?![.-]\d)")


def _check_summary(r: CheckResult, k: int) -> tuple[str, str]:
    """(one-line summary, description) with counts below k suppressed: the violated and denominator
    counts, and every other "<n> of <m>" count of the description."""
    v, d = r.num_violated_rows, r.num_denominator_rows
    description = _COUNT_PAIR.sub(lambda m: f"{_small_count(int(m[1]), k)} of {_small_count(int(m[2]), k)}",
                                  r.description)
    if not (0 < v < k or 0 < d < k):
        head = (
            f"{r.key} ({r.severity}): {v} of {d} ({r.pct_violated_rows:.2%}) violated, threshold {r.threshold_pct:g}%"
        )
        return head, description
    pct = f"<{k / d:.2%}" if d >= k and v < k else "percentage withheld"
    head = (
        f"{r.key} ({r.severity}): {_small_count(v, k)} of {_small_count(d, k)} ({pct}) violated, threshold "
        f"{r.threshold_pct:g}% (counts below {k} are not shown)"
    )
    return head, description


def build_repair_prompt(
    spec: ETLSpec,
    iteration: int,
    etl_errors: list[TableLoadResult],
    checks: list[CheckResult],
    cfg: ProjectConfig,
    *,
    profile: SourceProfile | None = None,
    schemas: Schemas | None = None,
) -> str:
    k = cfg.privacy.min_cell_count
    known = known_identifiers(profile, schemas)
    lines: list[str] = [f"Repair iteration {iteration} of at most {cfg.quality.max_repair_iterations}."]
    if etl_errors:
        lines.append(f"The last ETL build failed: {len(etl_errors)} step(s) raised an error.\n\nFAILED ETL STEPS")
        for i, s in enumerate(etl_errors, 1):
            lines.append(f"{i}. step {s.step} (target table {s.target_table})")
            lines.append(f"   error: {redact_error(s.error, s.sql, known)}")
            if s.source_rows is not None:
                lines.append(f"   source rows: {_small_count(s.source_rows, k)}")
            if s.sql:
                lines.append("   failing SQL:\n```sql\n" + _clip_sql(s.sql, 4000) + "\n```")
    else:
        lines.append(
            f"The data-quality checks found {len(checks)} failure(s) with a severity the pipeline repairs "
            f"({', '.join(cfg.quality.repair_severities)}; blocking: {', '.join(cfg.quality.blocking_severities)}). "
            "Percentages are of the check's denominator; a check fails when the percentage exceeds its threshold."
            f" Counts below the minimum cell count are shown as <{k}."
            "\n\nFAILED CHECKS"
        )
        for i, r in enumerate(checks[:MAX_FAILURES_IN_PROMPT], 1):
            if r.error:  # the description of an errored check can repeat the error text
                head = f"{r.key} ({r.severity}): ERROR {redact_error(r.error, r.sql, known)}"
                description = redact_error(r.description, r.sql, known)
            else:
                head, description = _check_summary(r, k)
            lines.append(f"{i}. {head}")
            lines.append(f"   {r.check_level} check: {description}")
            if r.sql:
                lines.append("   check SQL:\n```sql\n" + _clip_sql(r.sql, 1500) + "\n```")
        if len(checks) > MAX_FAILURES_IN_PROMPT:
            lines.append(f"... and {len(checks) - MAX_FAILURES_IN_PROMPT} more failures (fix the ones above first).")
    lines.append("\nTABLE MAPPINGS IN THE ETL SPEC")
    counts: dict[str, int] = {}
    for tm in spec.table_mappings:
        idx = counts.get(tm.target_table, 0)
        counts[tm.target_table] = idx + 1
        joins = f" + {', '.join(f'{j.table} {j.alias}' for j in tm.joins)}" if tm.joins else ""
        where = f" WHERE {tm.where}" if tm.where else ""
        order = f" dedup_order {tm.dedup_order}" if (tm.dedup_order or "").strip() else ""
        lines.append(
            f"- {tm.target_table} (mapping_index {idx}) <- {{src}}.{tm.source.table} {tm.source.alias}{joins}{where}"
            f"{order} [{len(tm.fields)} fields]"
        )
    if spec.custom_sql:
        lines.append(
            "Custom SQL steps (the tools cannot edit these; describe needed changes for a human): "
            + ", ".join(f"{c.name} -> {c.target_table}" for c in spec.custom_sql)
        )
    if spec.observation_period == "table_mapping":
        lines.append(
            "Observation periods: table_mapping (one row per episode). The build merges each person's episodes, ends "
            "open ones (NULL end) at cdm_source.source_release_date or at death, and gives persons without a valid "
            "episode a period from their events (steps *_observation_period_merge and *_observation_period_fill, "
            "generated); never write merging, filters or a fallback into the observation_period end date: leave it "
            "NULL for open episodes."
        )
    else:
        lines.append("Observation periods: from_events (one period per person from their first to their last record, "
                     "generated by the build).")
    lines.append(_eras_line(spec))
    lines.append(
        "\nDiagnose the root cause of each failure, inspect with the tools, and fix the ETL spec (not the data) "
        "with the smallest change that follows OMOP conventions. Do not paper over real data problems: never "
        "drop records just to pass a check; if the issue is in the source data or in vocabulary coverage, "
        "explain it instead of hacking around it. Finish with a short summary of what you changed and what "
        "remains for a human."
    )
    return "\n".join(lines)


def _eras_line(spec: ETLSpec) -> str:
    """How the spec's era tables are built, for the repair prompt."""
    from ..cdm.spec import DERIVED_ERAS
    from ..etl.compiler import derived_era_tables, spec_era_tables

    derived, own = derived_era_tables(spec), sorted(spec_era_tables(spec))
    parts = []
    if derived:
        one = len(derived) == 1
        records = " and ".join(DERIVED_ERAS[t].split("_")[0] for t in derived)  # condition, drug
        parts.append(f"{' and '.join(derived)} {'is' if one else 'are'} derived by the build from "
                     f"{' and '.join(DERIVED_ERAS[t] for t in derived)} with the OHDSI era logic "
                     f"(step{'' if one else 's'} {', '.join('*_' + t for t in derived)}, generated): never add a "
                     f"mapping into {'it' if one else 'them'}; {'its' if one else 'their'} checks are not yours to fix, "
                     f"fix the {records} records instead")
    if own:
        one = len(own) == 1
        parts.append(f"{' and '.join(own)} {'is' if one else 'are'} loaded by the spec's own SQL"
                     + (f" in place of the build's era step: {'its' if one else 'their'} checks are not yours to fix (a "
                        "person decides whether the build should derive the eras instead), and you may change that SQL "
                        "only to fix an error that stops it from running" if spec.derive_eras else ""))
    if not spec.derive_eras:
        parts.append("the build derives no eras (derive_eras: false)")
    return "Eras: " + ("; ".join(parts) if parts else "none (no condition or drug records are loaded)") + "."


SPEC_HEADER = """\
# ETL specification for omop-agent (updated by the repair agent, iteration {iteration}, {when}).
# Every change the repair agent made is listed in repairs.json with its rationale and trigger.
# Edit freely: each table_mappings entry compiles to one INSERT ... SELECT (see sql/ after a build);
# field kinds are described in the FieldMapping docstring (omop_agent/models.py). Changes are picked
# up by the next build."""


def _save_repairs(ws: Workspace, actions: list[RepairAction]) -> None:
    existing: list[RepairAction] = []
    if ws.repairs_json.exists():
        try:
            existing = ws.load_json_list(ws.repairs_json, RepairAction)
        except (ValueError, json.JSONDecodeError):
            ws.audit("repairs_json_unreadable", path=str(ws.repairs_json))
    ws.save_json(ws.repairs_json, existing + actions)


def repair_spec(
    spec: ETLSpec,
    *,
    iteration: int,
    dq: DQReport | None,
    etl_result: ETLRunResult | None,
    llm: LLMClient,
    conn: psycopg.Connection,
    schemas: Schemas,
    cfg: ProjectConfig,
    ws: Workspace | None,
    profile: SourceProfile | None,
) -> tuple[ETLSpec, list[RepairAction]]:
    """Let the repair agent fix ``spec`` for the current failures. Returns (new spec, actions)."""
    etl_errors, checks = repair_triggers(dq, etl_result, cfg, spec)
    if not etl_errors and not checks:
        return spec, []

    table_triggers: dict[str, list[str]] = {}
    if etl_errors:
        keys = [f"etl:{s.step}" for s in etl_errors]
        for s in etl_errors:
            table_triggers.setdefault(s.target_table.lower(), []).append(f"etl:{s.step}")
    else:
        keys = [r.key for r in checks]
        for r in checks:
            table_triggers.setdefault(r.table.lower(), []).append(r.key)
    state = RepairState(
        spec=spec,
        conn=conn,
        schemas=schemas,
        cfg=cfg,
        profile=profile,
        iteration=iteration,
        actions=[],
        trigger=", ".join(keys)[:500],
        table_triggers=table_triggers,
    )
    prompt = build_repair_prompt(spec, iteration, etl_errors, checks, cfg, profile=profile, schemas=schemas)
    try:
        result = llm.run_agent(
            stage="repair",
            key=f"iteration:{iteration}",
            system=repair_system(cfg),
            prompt=prompt,
            tools=build_repair_tools(state),
            max_iterations=cfg.llm.max_agent_iterations,
        )
        final_text = result.final_text
    except LLMUnavailable as e:
        if ws is not None:
            ws.audit("repair_skipped", iteration=iteration, reason=str(e), trigger=state.trigger)
        return spec, []
    except LLMError as e:  # keep the validated changes made before the model gave up
        final_text = f"(the repair agent stopped with an error: {e})"
        if ws is not None:
            ws.audit("repair_error", iteration=iteration, error=str(e))

    if ws is not None:
        _save_repairs(ws, state.actions)
        if state.actions:
            when = utcnow().strftime("%Y-%m-%d %H:%M UTC")
            ws.save_yaml(ws.spec_yaml, state.spec, header=SPEC_HEADER.format(iteration=iteration, when=when))
            try:
                from ..design.render import render_spec_md

                ws.spec_md.write_text(render_spec_md(state.spec))
            except Exception:  # the Markdown view is a convenience; the YAML is the source of truth
                pass
        ws.audit(
            "repair",
            iteration=iteration,
            actions=len(state.actions),
            changes=[f"{a.target_table}.{a.target_field or '*'}: {a.change}" for a in state.actions],
            trigger=state.trigger,
            summary=final_text[:2000],
        )
    return state.spec, state.actions
