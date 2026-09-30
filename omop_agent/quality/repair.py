"""The repair agent: Claude diagnoses failed ETL steps and data-quality checks and fixes the spec.

The agent never touches the data. It reads context through tools (the table mapping and its
compiled SQL, the CDM table conventions, the privacy-filtered source profile), tests hypotheses
with aggregate-only SQL, and edits the ``ETLSpec`` through two tools that apply a change only
if the table still compiles and ``EXPLAIN``s cleanly. Each applied change is recorded as a
``RepairAction`` (``repairs.json`` and the audit log); the pipeline then rebuilds and re-checks.

Privacy: ``run_aggregate_query`` accepts one read-only SELECT that returns a count column ``n``.
Rows with ``n < min_cell_count`` are withheld, and queries that mention a direct-identifier
column (from the profile or ``privacy.pii_columns``), select ``*``, or use row-concatenating or
file-access functions are refused.

Tool names and input fields are part of recorded replays: do not rename them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import psycopg
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..cdm.spec import load_cdm_spec
from ..config import ProjectConfig
from ..db import Schemas, execute, readonly_query, rollback_only, short_error
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
from ..privacy import profile_prompt_text
from ..workspace import Workspace, utcnow
from .checks import summarize

MAX_FAILURES_IN_PROMPT = 30
QUERY_FETCH_LIMIT = 200  # rows examined (for suppression counts)
QUERY_SHOW_LIMIT = 50  # rows shown to the model

REPAIR_SYSTEM = """\
You are the data-quality repair agent of an OMOP CDM v5.4 ETL pipeline: the ETL developer who \
reads the Data Quality Dashboard results and fixes the ETL.

The ETL is defined by an ETL spec. Each table mapping compiles to one
  INSERT INTO cdm.<table> SELECT ... FROM src.<source table> <alias> [JOIN ...] [WHERE ...]
with one field mapping per CDM field. Source columns are all text, so cast explicitly. Field kinds:
  column / expression / constant  a PostgreSQL expression over the source aliases, cast to the CDM type
  person_id    the source patient key, resolved through the person id map (rows without a person are dropped)
  visit_id     the source visit key, resolved through the visit id map (NULL when there is no match)
  auto_id      generated surrogate key (only for the table's own primary key)
  vocab_code   a code of an OMOP vocabulary: expression = the code as in concept_code, vocabulary = its
               vocabulary_id; concept_role 'standard' follows 'Maps to', 'source' returns the source concept
  concept_map  a local code/label mapped through the project's source_to_concept_map: vocabulary = the
               source_vocabulary_id, target_domain = the OMOP domain of the targets; concept_role 'source'
               returns the map's source_concept_id
  null         leave the field NULL
Concept lookups yield 0 when a source value exists but has no mapping. Expressions may use the schema
placeholders {src}, {cdm}, {vocab}, {etl}.

How to work:
1. Diagnose the root cause of each failure before changing anything. Read the mapping and its compiled
   SQL (get_table_mapping), the CDM conventions (get_cdm_table) and the source profile
   (get_source_profile), and test hypotheses with run_aggregate_query.
2. Fix the ETL spec, not the data: update_field_mapping changes one field, replace_table_mapping changes
   the source, joins, filter or several fields at once. A change is applied only if the table still
   compiles and EXPLAINs cleanly; if it comes back REJECTED, read the error and try again.
3. Prefer the smallest change that follows OMOP conventions (for example, a visit without an end date
   ends on its start date; an unknown value maps to concept 0 rather than to a guessed concept).
4. Do not paper over real data problems. Never filter out or delete records just to make a check pass,
   and never invent concept ids, dates or values. When the cause lies in the source data or in the
   vocabulary coverage (unmapped local codes, implausible dates recorded at source), leave the spec as
   it is and explain the problem for a human.
5. Privacy: you only ever see aggregates. run_aggregate_query must return a count column named n;
   groups with n below the minimum cell count are withheld, and queries that touch direct identifiers
   are refused. Never try to retrieve individual records.

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


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetTableMappingInput(_ToolInput):
    target_table: str = Field(..., description="CDM table the mapping loads, e.g. 'visit_occurrence'.")
    mapping_index: int = Field(0, ge=0, description="Which mapping when several feed the same table (0 = the first).")


class GetCdmTableInput(_ToolInput):
    table: str = Field(..., description="OMOP CDM v5.4 table name, e.g. 'condition_occurrence'.")


class GetSourceProfileInput(_ToolInput):
    table: str = Field(..., description="Source table name as staged, e.g. 'clinic_visits'.")


class RunAggregateQueryInput(_ToolInput):
    sql: str = Field(
        ...,
        description="One read-only PostgreSQL SELECT (or WITH ... SELECT) statement that aggregates and returns "
        "a count column named n, e.g. \"SELECT v.visit_type, COUNT(*) AS n FROM {src}.clinic_visits v WHERE "
        "v.discharge_date IS NULL GROUP BY 1\". Use the schema placeholders {src}, {cdm}, {vocab}, {etl}. "
        "Rows with n below the minimum cell count are withheld; at most 50 rows are shown.",
    )


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
        "(target_table, source {table, alias}, joins, where, fields, rationale).",
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


def validate_mapping(tm: TableMapping, spec: ETLSpec, conn: psycopg.Connection, schemas: Schemas) -> str | None:
    """None if ``tm`` compiles and its INSERT EXPLAINs cleanly, else the error."""
    from ..etl.compiler import compile_table

    try:
        sql = compile_table(tm, spec, schemas)
    except Exception as e:  # CompileError, or anything else the compiler trips over
        return f"does not compile: {e}"
    try:
        with rollback_only(conn):
            execute(conn, "EXPLAIN\n" + sql.rstrip().rstrip(";"))
    except psycopg.Error as e:
        return f"EXPLAIN failed: {short_error(e)}"
    return None


def _apply(
    state: RepairState,
    pos: int | None,
    new_tm: TableMapping,
    *,
    target_field: str | None,
    change: str,
    rationale: str,
) -> str:
    mappings = list(state.spec.table_mappings)
    if pos is None:
        mappings.append(new_tm)
    else:
        mappings[pos] = new_tm
    new_spec = state.spec.model_copy(update={"table_mappings": mappings}, deep=True)
    error = validate_mapping(new_tm, new_spec, state.conn, state.schemas)
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


# ---- aggregate queries ---------------------------------------------------------------

_LITERAL_OR_COMMENT = re.compile(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", re.S)
_QUOTED_IDENT = re.compile(r'"((?:[^"]|"")+)"')
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_STAR = re.compile(r"(\bselect\s+(?:all\s+|distinct\s+)?|,\s*|\.)\*", re.I)
_COUNT_STAR = re.compile(r"\bcount\s*\(\s*\*\s*\)", re.I)
_DENIED_FUNCTIONS = re.compile(
    r"\b(string_agg|array_agg|json_agg|jsonb_agg|json_object_agg|jsonb_object_agg|xmlagg|row_to_json|to_json|"
    r"to_jsonb|json_build_object|jsonb_build_object|json_build_array|jsonb_build_array|pg_read_file|"
    r"pg_read_binary_file|pg_ls_dir|pg_stat_file|lo_import|lo_export|lo_get|dblink\w*|pg_sleep\w*|"
    r"set_config|copy)\s*\(",
    re.I,
)


def _pii_columns(state: RepairState) -> set[str]:
    cols = {c.split(".")[-1].strip().lower() for c in state.cfg.privacy.pii_columns if c.strip()}
    if state.profile is not None:
        cols |= {c.name.lower() for t in state.profile.tables for c in t.columns if c.is_pii}
    return cols


def _identifiers(bare_sql: str) -> set[str]:
    quoted = {m.replace('""', '"').lower() for m in _QUOTED_IDENT.findall(bare_sql)}
    return quoted | {m.lower() for m in _IDENT.findall(_QUOTED_IDENT.sub(" ", bare_sql))}


def _check_query(sql: str, state: RepairState) -> str | None:
    """Why the query is refused, or None."""
    bare = _LITERAL_OR_COMMENT.sub(" ", sql).strip()
    if ";" in bare:
        return "only one statement is allowed."
    if not re.match(r"(?is)^(select|with)\b", bare):
        return "only a single read-only SELECT (or WITH ... SELECT) query is allowed."
    if _STAR.search(_COUNT_STAR.sub("count(1)", bare)):
        return "SELECT * is not allowed; name the columns you aggregate over."
    denied = _DENIED_FUNCTIONS.search(bare)
    if denied:
        return f"refused: {denied.group(1)}() is not allowed in aggregate queries."
    idents = _identifiers(bare)
    pii = sorted(idents & _pii_columns(state))
    if state.profile is None:  # no profile: fall back to the configured identifier name patterns
        pii += sorted(i for i in idents if any(re.search(p, i) for p in state.cfg.privacy.pii_name_patterns))
    if pii:
        return (
            f"refused: the query references direct-identifier (PII) columns ({', '.join(dict.fromkeys(pii))}). "
            "Only aggregate queries over non-identifying columns are allowed."
        )
    if "n" not in idents:
        return "the query must return a count column named n (e.g. COUNT(*) AS n)."
    return None


def _count(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cell(v: Any) -> str:
    s = "NULL" if v is None else str(v)
    s = " ".join(s.split())
    return s if len(s) <= 60 else s[:57] + "..."


def _format_rows(rows: list[dict], withheld: int, min_n: int, examined_all: bool) -> str:
    if not rows and not withheld:
        return "0 rows."
    lines: list[str] = []
    shown = rows[:QUERY_SHOW_LIMIT]
    if shown:
        cols = list(shown[0].keys())
        cells = [[_cell(r.get(c)) for c in cols] for r in shown]
        widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
        lines.append(" | ".join(c.ljust(w) for c, w in zip(cols, widths)).rstrip())
        lines.append("-+-".join("-" * w for w in widths))
        lines += [" | ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip() for row in cells]
    footer = f"({len(shown)} row(s) shown"
    if len(rows) > len(shown):
        footer += f" of {len(rows)}; aggregate further to see the rest"
    if withheld:
        footer += f"; {withheld} row(s) withheld because n < {min_n} (small-cell suppression)"
    if not examined_all:
        footer += f"; only the first {QUERY_FETCH_LIMIT} result rows were examined"
    lines.append(footer + ")")
    return "\n".join(lines)


def run_aggregate_query(state: RepairState, sql: str) -> str:
    """Run an aggregate-only, read-only query with small-cell suppression. Errors come back as text."""
    sql = sql.strip().rstrip(";").strip()
    refusal = _check_query(sql, state)
    if refusal:
        return f"ERROR: {refusal}"
    try:
        rows = readonly_query(state.conn, state.schemas.render(sql), limit=QUERY_FETCH_LIMIT)
    except psycopg.Error as e:
        return f"ERROR: {short_error(e)}"
    if rows and "n" not in rows[0]:
        return "ERROR: the result has no column named n; return the group size as COUNT(*) AS n."
    min_n = state.cfg.privacy.min_cell_count
    kept: list[dict] = []
    withheld = 0
    for r in rows:
        n = _count(r.get("n"))
        if n is None or n < min_n:
            withheld += 1
        else:
            kept.append(r)
    return _format_rows(kept, withheld, min_n, examined_all=len(rows) < QUERY_FETCH_LIMIT)


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
            text += "\n-- compiled SQL\n" + compile_table(tm, state.spec, state.schemas)
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

    def aggregate_query(inp: RunAggregateQueryInput) -> str:
        return run_aggregate_query(state, inp.sql)

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
        change = _mapping_diff(old_tm, new_tm)
        return _apply(state, pos, new_tm, target_field=None, change=change, rationale=inp.rationale)

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
            "run_aggregate_query",
            "Run one read-only aggregate SELECT against the source ({src}), CDM ({cdm}), vocabulary ({vocab}) "
            "or ETL working ({etl}) schemas. It must return a count column named n; groups with n below the "
            "minimum cell count are withheld and direct identifiers cannot be queried.",
            RunAggregateQueryInput,
            aggregate_query,
        ),
        AgentTool(
            "update_field_mapping",
            "Set how one CDM field is populated (replaces the field's mapping, or adds it). Applied only if the "
            "table still compiles and EXPLAINs cleanly; otherwise returns REJECTED with the error.",
            UpdateFieldMappingInput,
            update_field_mapping,
        ),
        AgentTool(
            "replace_table_mapping",
            "Replace a whole table mapping (source, joins, where, fields) with a new one given as YAML. Applied "
            "only if it compiles and EXPLAINs cleanly; otherwise returns REJECTED with the error.",
            ReplaceTableMappingInput,
            replace_table_mapping,
        ),
    ]


# --------------------------------------------------------------------------------------
# The repair step
# --------------------------------------------------------------------------------------


def repair_triggers(
    dq: DQReport | None, etl_result: ETLRunResult | None, cfg: ProjectConfig
) -> tuple[list[TableLoadResult], list[CheckResult]]:
    """ETL step errors if there are any; otherwise the DQ failures in ``repair_severities``."""
    if etl_result is not None and etl_result.errors():
        return etl_result.errors(), []
    if dq is None:
        return [], []
    severities = set(cfg.quality.repair_severities)
    order = {"fatal": 0, "convention": 1, "characterization": 2}
    failures = [r for r in dq.results if (r.failed or r.error) and r.severity in severities and not r.not_applicable]
    return [], sorted(failures, key=lambda r: (order.get(r.severity, 3), -r.pct_violated_rows))


def _clip(text: str | None, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "\n... (truncated)"


def build_repair_prompt(
    spec: ETLSpec, iteration: int, etl_errors: list[TableLoadResult], checks: list[CheckResult], cfg: ProjectConfig
) -> str:
    lines: list[str] = [f"Repair iteration {iteration} of at most {cfg.quality.max_repair_iterations}."]
    if etl_errors:
        lines.append(f"The last ETL build failed: {len(etl_errors)} step(s) raised an error.\n\nFAILED ETL STEPS")
        for i, s in enumerate(etl_errors, 1):
            lines.append(f"{i}. step {s.step} (target table {s.target_table})")
            lines.append(f"   error: {s.error}")
            if s.source_rows is not None:
                lines.append(f"   source rows: {s.source_rows}")
            if s.sql:
                lines.append("   failing SQL:\n```sql\n" + _clip(s.sql, 4000) + "\n```")
    else:
        lines.append(
            f"The data-quality checks found {len(checks)} failure(s) with a severity the pipeline repairs "
            f"({', '.join(cfg.quality.repair_severities)}; blocking: {', '.join(cfg.quality.blocking_severities)}). "
            "Percentages are of the check's denominator; a check fails when the percentage exceeds its threshold."
            "\n\nFAILED CHECKS"
        )
        for i, r in enumerate(checks[:MAX_FAILURES_IN_PROMPT], 1):
            lines.append(f"{i}. {summarize(r)}")
            lines.append(f"   {r.check_level} check: {r.description}")
            if r.sql:
                lines.append("   check SQL:\n```sql\n" + _clip(r.sql, 1500) + "\n```")
        if len(checks) > MAX_FAILURES_IN_PROMPT:
            lines.append(f"... and {len(checks) - MAX_FAILURES_IN_PROMPT} more failures (fix the ones above first).")
    lines.append("\nTABLE MAPPINGS IN THE ETL SPEC")
    counts: dict[str, int] = {}
    for tm in spec.table_mappings:
        idx = counts.get(tm.target_table, 0)
        counts[tm.target_table] = idx + 1
        joins = f" + {', '.join(f'{j.table} {j.alias}' for j in tm.joins)}" if tm.joins else ""
        where = f" WHERE {tm.where}" if tm.where else ""
        lines.append(
            f"- {tm.target_table} (mapping_index {idx}) <- {{src}}.{tm.source.table} {tm.source.alias}{joins}{where}"
            f" [{len(tm.fields)} fields]"
        )
    if spec.custom_sql:
        lines.append("Custom SQL steps: " + ", ".join(f"{c.name} -> {c.target_table}" for c in spec.custom_sql))
    lines.append(f"Observation periods: {spec.observation_period}.")
    lines.append(
        "\nDiagnose the root cause of each failure, inspect with the tools, and fix the ETL spec (not the data) "
        "with the smallest change that follows OMOP conventions. Do not paper over real data problems: never "
        "drop records just to pass a check; if the issue is in the source data or in vocabulary coverage, "
        "explain it instead of hacking around it. Finish with a short summary of what you changed and what "
        "remains for a human."
    )
    return "\n".join(lines)


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
    etl_errors, checks = repair_triggers(dq, etl_result, cfg)
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
    prompt = build_repair_prompt(spec, iteration, etl_errors, checks, cfg)
    try:
        result = llm.run_agent(
            stage="repair",
            key=f"iteration:{iteration}",
            system=REPAIR_SYSTEM,
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
