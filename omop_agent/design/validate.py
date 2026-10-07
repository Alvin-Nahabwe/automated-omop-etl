"""Check an ``ETLSpec`` before anything runs.

Static checks need only the spec (and optionally the source profile): tables and fields exist
in CDM v5.4, required fields are mapped, aliases and source columns resolve, field kinds carry
what they need, and person/visit keys come from the id maps (``person_id`` / ``visit_id``
kinds, never ``auto_id``) so that events link to the right person and visit. Because the spec's
SQL is written by an LLM and executed by the build, the static checks also refuse unsafe SQL
(``sqlguard.fragment_issues``: statement separators, server-file/session functions, catalog
access, DDL/DML), tables outside the staged source, vocabulary and ETL schemas, and mappings
that load direct identifiers (``privacy.pii_references``) into the CDM. Custom SQL is executed as
written, so it always gets a review warning. ``sql_safety_issues`` is the subset the ETL
runner re-checks before it executes a spec.

Conversions of dates and numbers are checked against the source profile: the build reads the
casts and to_date / to_timestamp calls of column and expression fields and of the WHERE
tolerantly (``etl.tolerant``), so a conversion it has to leave as written, or one in a join
condition, key or id / lookup expression (which stay as written), is a warning; an ISO cast
(``::date``, or try_date without a template) of a column profiled with a day-first or
month-first format, which reads none of its values under DateStyle 'ISO, YMD', is an error that
names the to_date template, also in custom SQL (a warning in a branch of a regex guard on the
column); so is a strftime template ('%d/%m/%Y'). A to_date template that differs from the
profiled format, and a mapped days_supply that the drug_exposure_end_date does not use, are
warnings.

With a database connection the validator also checks concept constants against the loaded
vocabulary, checks that ``vocab_code`` vocabularies are loaded, and ``EXPLAIN``s each compiled
INSERT inside a transaction that is always rolled back, under the build's session settings
(``etl.session``).

Semantic problems (a code format that does not match the vocabulary) and runtime problems (a
NOT NULL violation that only shows up with real rows) are left to the build and the
data-quality stage.
"""

from __future__ import annotations

import re
from typing import Literal

import psycopg
from pydantic import BaseModel

from ..cdm.ddl import create_cdm_tables, create_etl_tables
from ..cdm.spec import CdmTable, load_cdm_spec
from ..config import PrivacyConfig, ProjectConfig
from ..db import Schemas, execute, fetch_all, list_columns, qi, rollback_only, short_error
from ..etl.compiler import ID_MAP_KEYS, CompileError, alias_refs, compile_table, source_ref
from ..etl.session import apply_session
from ..etl.tolerant import DATE_KINDS, LexError, lex, rewrite, without_conversions
from ..models import CustomSQL, ETLSpec, KeySpec, SourceProfile, TableMapping
from ..privacy import pii_columns, pii_references
from ..sqlguard import fragment_issues, strip_literals

# Qualifiers that are schemas, not table aliases (e.g. pg_catalog.to_char(...)::pg_catalog.text).
# (Both are also refused by the SQL guard; ``public`` is not listed, so ``public.t`` is an
# undeclared alias rather than a silent read of another schema.)
_NOT_ALIASES = {"pg_catalog", "information_schema"}
_CASE_INT = re.compile(r"\bTHEN\s+(\d+)|\bELSE\s+(\d+)", re.I)
_INT_LITERAL = re.compile(r"^\(*\s*'?\s*(-?\d+)\s*'?\s*\)*$")
_NEEDS_EXPRESSION = {"column", "expression", "constant", "concept_map", "vocab_code", "person_id", "visit_id"}

# Schemas that spec SQL may read, as placeholders; bare table names are staged source tables.
_READABLE = ("{src}", "{vocab}", "{etl}")
_PLACEHOLDER_TABLE = re.compile(r'(\{\w+\})\s*\.\s*([A-Za-z_][A-Za-z0-9_]*|"[^"]+")')
_FROM_JOIN = re.compile(r"\b(?:from|join)\s+", re.I)
_TABLE_TOKEN = re.compile(r'\{\w+\}\s*\.\s*(?:[A-Za-z_]\w*|"[^"]+")|[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)?|"[^"]+"')
# Words that can follow FROM/JOIN <x> without being an alias (and are never table names).
_SQL_WORDS = {
    "all", "and", "any", "as", "asc", "at", "between", "by", "case", "cross", "desc", "distinct", "else", "end",
    "except", "false", "fetch", "for", "from", "full", "group", "having", "ilike", "in", "inner", "intersect", "into",
    "is", "isnull", "join", "lateral", "left", "like", "limit", "natural", "not", "notnull", "null", "offset", "on",
    "only", "or", "order", "outer", "returning", "right", "select", "set", "similar", "tablesample", "then", "true",
    "union", "using", "values", "when", "where", "window", "with", "without",
}
# DML is what custom SQL is for; the SQL guard's other rules still apply to it. Its statements
# must be plain DML/queries: SET ROLE, COMMIT, DO '...' and the like could leave the ETL role.
_DML = re.compile(r"\b(insert|update|delete)\b", re.I)
_CUSTOM_STATEMENTS = {"insert", "update", "delete", "select", "with"}


class SpecIssue(BaseModel):
    severity: Literal["error", "warning"]
    target_table: str | None = None
    target_field: str | None = None
    message: str

    def __str__(self) -> str:
        where = ".".join(x for x in (self.target_table, self.target_field) if x)
        return f"{self.severity.upper()} {where + ': ' if where else ''}{self.message}"


def format_issues(issues: list[SpecIssue]) -> str:
    """One issue per line, for prompts and logs."""
    return "\n".join(f"- {i}" for i in issues)


class _SourceCatalog:
    """Source table -> columns, from the profile or (without one) from the staged tables."""

    def __init__(self, profile: SourceProfile | None, conn: psycopg.Connection | None, schemas: Schemas | None):
        self.profile = profile
        self.conn = conn
        self.schemas = schemas
        self._cache: dict[str, set[str] | None] = {}

    @property
    def available(self) -> bool:
        return self.profile is not None or (self.conn is not None and self.schemas is not None)

    def columns(self, table: str) -> set[str] | None:
        """Lower-cased column names, or None when the table is unknown."""
        key = table.lower()
        if key in self._cache:
            return self._cache[key]
        cols: set[str] | None = None
        if self.profile is not None:
            t = next((t for t in self.profile.tables if t.name.lower() == key), None)
            cols = {c.name.lower() for c in t.columns} if t else None
        elif self.conn is not None and self.schemas is not None:
            try:
                with self.conn.transaction():
                    rows = list_columns(self.conn, self.schemas.src, table)
                cols = {r["column_name"].lower() for r in rows} or None
            except psycopg.Error:
                cols = None
        self._cache[key] = cols
        return cols


def _staged_name(table: str) -> str | None:
    """The staged source table a table reference names (bare or ``{src}.``-qualified), else None."""
    t = table.strip()
    if "." not in t and "{" not in t:
        return t
    m = _PLACEHOLDER_TABLE.fullmatch(t)
    if m and m.group(1) == "{src}":
        return m.group(2).strip('"')
    return None


def table_ref_issue(table: str) -> str | None:
    """Why a source/join/key table reference may not be read (None when it is allowed).

    Bare names are staged source tables; otherwise only the ``{src}``, ``{vocab}`` and ``{etl}``
    placeholders may qualify a table, so spec SQL cannot read the CDM being built, system
    catalogs or any other schema.
    """
    t = table.strip()
    if "." not in t and "{" not in t:
        return None
    m = _PLACEHOLDER_TABLE.fullmatch(t)
    if m and m.group(1) in _READABLE:
        return None
    return (
        f"table '{t}' may not be read: use a staged source table by its bare name, or qualify the table "
        "with {src}, {vocab} or {etl}"
    )


def _inner_tables(expr: str) -> list[tuple[str, str | None]]:
    """``(table, alias)`` declared by ``FROM``/``JOIN`` inside an expression (correlated subqueries).

    Function calls (``EXTRACT(YEAR FROM f(x))``, ``FROM generate_series(...)``) and casts are
    skipped. A dotted name may also be a column (``EXTRACT(YEAR FROM p.dob)``); callers decide.
    """
    text = strip_literals(expr)
    out: list[tuple[str, str | None]] = []
    for m in _FROM_JOIN.finditer(text):
        rest = text[m.end():]
        t = _TABLE_TOKEN.match(rest)
        if not t:
            continue
        after = rest[t.end():]
        if re.match(r"\s*(\(|::|\[)", after):
            continue
        table = re.sub(r"\s+", "", t.group(0))
        if table.lower() in _SQL_WORDS:
            continue
        a = re.match(r"\s+(?:as\s+)?([A-Za-z_]\w*)", after, re.I)
        alias = a.group(1).lower() if a and a.group(1).lower() not in _SQL_WORDS else None
        out.append((table, alias))
    return out


def _fragments(tm: TableMapping) -> list[tuple[str | None, str, str]]:
    """``(field, what, sql)`` for every SQL fragment of a table mapping."""
    out: list[tuple[str | None, str, str]] = [
        (fm.target_field, "expression", fm.expression) for fm in tm.fields if fm.expression
    ]
    out += [(None, f"join '{j.alias}' condition", j.on) for j in tm.joins]
    if tm.where:
        out.append((None, "where", tm.where))
    return out


def _mapping_sql_problems(tm: TableMapping) -> list[tuple[str | None, str]]:
    """``(field, problem)``: unsafe fragments and tables the mapping may not read."""
    out: list[tuple[str | None, str]] = []
    for table in [tm.source.table] + [j.table for j in tm.joins]:
        issue = table_ref_issue(table)
        if issue:
            out.append((None, issue))
    declared = {tm.source.alias.lower()} | {j.alias.lower() for j in tm.joins}
    for field, what, sql in _fragments(tm):
        out += [(field, f"unsafe SQL in {what}: {x}") for x in fragment_issues(sql)]
        for table, alias in _inner_tables(sql):
            if "{" not in table and "." in table and (table.split(".")[0].lower() in declared or alias is None):
                continue  # a column (EXTRACT(YEAR FROM p.dob)); an unknown qualifier is an undeclared alias
            if "." in table or "{" in table:
                issue = table_ref_issue(table)
                if issue:
                    out.append((field, f"{what}: {issue}"))
    return out


def _key_sql_problems(key: KeySpec) -> list[str]:
    out = [x for x in [table_ref_issue(key.table)] if x]
    out += [f"unsafe SQL in the key expression: {x}" for x in fragment_issues(key.expression)]
    return out


def _custom_sql_problems(cs: CustomSQL) -> list[str]:
    """Custom SQL may hold several INSERT/UPDATE/DELETE/SELECT/WITH statements; everything else
    the guard refuses is refused, and so is any other kind of statement."""
    out = [
        f"unsafe custom SQL: {x}"
        for x in fragment_issues(_DML.sub(" ", cs.sql))
        if not x.startswith("contains ';'")
    ]
    for stmt in strip_literals(cs.sql).split(";"):
        words = stmt.split()
        if words and words[0].lower() not in _CUSTOM_STATEMENTS:
            out.append(f"unsafe custom SQL: a '{words[0]}' statement is not allowed "
                       "(only INSERT, UPDATE, DELETE, SELECT or WITH)")
    return out


def sql_safety_issues(obj: TableMapping | KeySpec | CustomSQL) -> list[str]:
    """Reasons the spec SQL of ``obj`` must not be executed (empty when it may run).

    Used by the validator and, again, by the ETL runner right before a build, because the spec
    can change after validation (hand edits, the repair agent).
    """
    if isinstance(obj, TableMapping):
        return [f"{obj.target_table}{'.' + f if f else ''}: {msg}" for f, msg in _mapping_sql_problems(obj)]
    if isinstance(obj, KeySpec):
        return [f"key {obj.table}.{obj.expression}: {msg}" for msg in _key_sql_problems(obj)]
    if isinstance(obj, CustomSQL):
        return [f"custom SQL '{obj.name}': {msg}" for msg in _custom_sql_problems(obj)]
    return []


def _check_source_table(cat: _SourceCatalog, table: str, tt: str | None, what: str) -> SpecIssue | None:
    name = _staged_name(table)
    if not cat.available or name is None:
        return None
    if cat.columns(name) is None:
        where = "in the source profile" if cat.profile is not None else "in the staging schema"
        return SpecIssue(severity="error", target_table=tt, message=f"{what} '{table}' does not exist {where}")
    return None


def _pii_loaded(tm: TableMapping, profile: SourceProfile | None, privacy: PrivacyConfig) -> list[str]:
    """Direct-identifier columns whose values a mapping loads into the CDM.

    Fields of kind person_id / visit_id only resolve the key through the id maps (the CDM gets
    the integer id), and join conditions and filters only select rows, so those are not counted.
    """
    loading = tm.model_copy(
        update={
            "fields": [f for f in tm.fields if f.kind not in ("person_id", "visit_id")],
            "where": None,
            "joins": [j.model_copy(update={"on": "TRUE"}) for j in tm.joins],
        }
    )
    return pii_references(loading, profile, privacy)


# Kinds of field whose expression the build evaluates as written (keys of the id maps and lookups).
_KEY_KINDS = ("person_id", "visit_id", "vocab_code", "concept_map")
_TRY_FUNCTION = {"date": "try_date", "timestamp": "try_timestamp", "timestamptz": "try_timestamptz",
                 "integer": "try_integer", "bigint": "try_bigint", "number": "try_number"}


def _iso_readable(date_format: str) -> bool:
    """True when ``::date`` reads values of this profiled (strftime) format under DateStyle
    'ISO, YMD': year first, or an English month name with a four-digit year."""
    return date_format.startswith("%Y") or (("%b" in date_format or "%B" in date_format) and "%Y" in date_format)


def _pg_template(date_format: str) -> str | None:
    from ..profiling.profiler import strftime_to_pg

    try:
        return strftime_to_pg(date_format)
    except ValueError:
        return None


def _profiled_formats(operand: str, aliases: dict[str, str], cat: _SourceCatalog) -> list[tuple[str, str]]:
    """(alias.column, date_format) of the columns in ``operand`` that the profile gives a date format."""
    if cat.profile is None:
        return []
    out: list[tuple[str, str]] = []
    for alias, column in alias_refs(operand):
        table = _staged_name(aliases.get(alias, "")) if alias in aliases else None
        t = cat.profile.table(table) if table else None
        if t is None:
            t = next((x for x in cat.profile.tables if table and x.name.lower() == table.lower()), None)
        col = next((c for c in t.columns if c.name.lower() == column), None) if t is not None else None
        if col is not None and col.date_format and (f"{alias}.{column}", col.date_format) not in out:
            out.append((f"{alias}.{column}", col.date_format))
    return out


def _bare_formats(expr: str, table: str, cat: _SourceCatalog) -> list[tuple[str, str]]:
    """(column, date_format) of the bare column names in a key expression over ``table``."""
    if cat.profile is None:
        return []
    t = next((x for x in cat.profile.tables if x.name.lower() == (_staged_name(table) or "").lower()), None)
    if t is None:
        return []
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", strip_literals(expr)))
    return [(c.name, c.date_format) for c in t.columns if c.date_format and c.name in words]


def _conversion_problems(
    expr: str, where: str, *, hard: bool, formats: list[tuple[str, str]] | None = None,
    aliases: dict[str, str] | None = None, cat: _SourceCatalog | None = None,
) -> list[tuple[str, str]]:
    """(severity, message) for the conversions in one SQL fragment. ``hard``: the build evaluates
    the fragment as written (a key or join), so every conversion in it can fail the step."""
    rw = rewrite(expr, "f", "mid")
    out: list[tuple[str, str]] = []
    for _, problem in rw.bad_templates:
        out.append(("error", f"{where}: {problem}"))
    if not rw.conversions and rw.left_hard:  # the scan could not read the fragment
        out.append(("warning", f"{where}: the build cannot read this SQL to make its conversions tolerant, so one "
                               "value that does not convert fails the whole table"))
    for c in rw.conversions:
        # the columns the conversion reads as written (not those an inner conversion read first)
        operand = without_conversions(c.operand)
        if formats is not None:
            cols = [(col, fmt) for col, fmt in formats if re.search(rf"\b{re.escape(col)}\b", operand)]
        else:
            cols = _profiled_formats(operand, aliases or {}, cat) if cat else []
        bad = [(col, fmt) for col, fmt in cols if not _iso_readable(fmt)]
        iso = c.template is None and c.function in (None, f"try_{c.kind}")  # a cast, or try_* without a template
        if c.kind in DATE_KINDS and iso and bad:
            col, fmt = bad[0]
            pg = _pg_template(fmt)
            fn = "to_date" if c.kind == "date" else "to_timestamp"
            fix = f": write {fn}({col}, '{pg}'), which reads ISO values too" if pg else ""
            if re.search(rf"\b{re.escape(col)}\s*(!?~|similar\s+to)", expr, re.I):  # a branch of a regex guard
                out.append(("warning", f"{where}: {c.text} reads {col}, profiled as {fmt}, as ISO; the values in the "
                                       f"profiled format are NULL unless the guard keeps them out{fix}"))
            else:
                out.append(("error", f"{where}: {c.text} reads {col}, profiled as {fmt}, as ISO, which cannot read "
                                     f"that format under DateStyle 'ISO, YMD' (those values would be NULL){fix}"))
            continue
        if c.template is not None and not (c.note and "%" in c.template):
            for col, fmt in cols:
                pg = _pg_template(fmt)
                t = c.template.upper().replace("FM", "")
                if pg and t != pg.upper() and not pg.upper().startswith(t):
                    out.append(("warning", f"{where}: the template '{c.template}' differs from the profiled format "
                                           f"of {col} ({fmt}, PostgreSQL '{pg}')"))
        if c.function == f"try_{c.kind}":
            continue  # gives NULL for a value it cannot read (in a field or the WHERE: counted)
        if hard or not c.rewritten:
            if c.template is not None and c.note and "%" in c.template:
                continue  # reported as an error above
            why = f" ({c.note})" if c.note else ""
            fix = (f"use {{etl}}.{_TRY_FUNCTION[c.kind]}(...), which gives NULL for a value it cannot read"
                   if hard else "write a plain cast, or to_date / to_timestamp with a literal template")
            place = "is evaluated as written" if hard else "is left as written by the build"
            out.append(("warning", f"{where}: {c.text} {place}{why}: one value it cannot convert fails the whole "
                                   f"step; {fix}"))
    return out


def _days_supply_issue(tm: TableMapping) -> str | None:
    """A drug_exposure mapping whose end date ignores its mapped days_supply."""
    if tm.target_table != "drug_exposure":
        return None
    by = {f.target_field: f for f in tm.fields}
    days, end = by.get("days_supply"), by.get("drug_exposure_end_date")
    if days is None or end is None or days.kind not in ("column", "expression") or not days.expression:
        return None
    if end.kind not in ("column", "expression") or not end.expression:
        return None
    used = set(alias_refs(days.expression))
    if used and not used & set(alias_refs(end.expression)):
        return ("days_supply is mapped but drug_exposure_end_date does not use it: when the source has no end "
                "date, the end is the start + days_supply - 1 (the drug era logic reads the end date)")
    return None


def _static_mapping(
    tm: TableMapping, spec: ETLSpec, cat: _SourceCatalog, privacy: PrivacyConfig | None = None
) -> list[SpecIssue]:
    cdm = load_cdm_spec()
    privacy = privacy or PrivacyConfig()
    tt = tm.target_table
    issues: list[SpecIssue] = []

    def err(msg: str, field: str | None = None) -> None:
        issues.append(SpecIssue(severity="error", target_table=tt, target_field=field, message=msg))

    def warn(msg: str, field: str | None = None) -> None:
        issues.append(SpecIssue(severity="warning", target_table=tt, target_field=field, message=msg))

    if not cdm.has_table(tt) or cdm.table(tt).schema_class != "CDM":
        err(f"'{tt}' is not an OMOP CDM v5.4 table")
        return issues
    table: CdmTable = cdm.table(tt)

    # fields: existence, duplicates, coverage of required fields
    seen: dict[str, int] = {}
    for fm in tm.fields:
        seen[fm.target_field] = seen.get(fm.target_field, 0) + 1
        if table.field(fm.target_field) is None:
            err(f"'{fm.target_field}' is not a field of {tt}", fm.target_field)
    for name, n in seen.items():
        if n > 1:
            err(f"mapped {n} times; each CDM field may appear once", name)
    mapped = {fm.target_field: fm for fm in tm.fields}
    for f in table.fields:
        if not f.required:
            continue
        fm = mapped.get(f.name)
        if fm is None:
            if (tt, f.name) in ID_MAP_KEYS:
                hint = f" (use kind {ID_MAP_KEYS[(tt, f.name)]} with the source key)"
            elif f.name == "person_id":
                hint = " (use kind person_id with the column that holds the patient key)"
            else:
                hint = " (use kind auto_id for a surrogate key)" if f.is_pk else ""
            err(f"required field is not mapped{hint}", f.name)
        elif fm.kind == "null":
            err("required field is mapped to NULL", f.name)

    # per-field kind rules
    for fm in tm.fields:
        f = table.field(fm.target_field)
        if f is None:
            continue
        if fm.kind in _NEEDS_EXPRESSION and not (fm.expression or "").strip():
            err(f"kind '{fm.kind}' needs an expression", f.name)
        if fm.kind in ("concept_map", "vocab_code") and not (fm.vocabulary or "").strip():
            err(f"kind '{fm.kind}' needs a vocabulary", f.name)
        if fm.kind == "concept_map":
            if not fm.target_domain:
                warn("concept_map without target_domain: the mapping step cannot restrict candidates by domain", f.name)
            if fm.vocabulary and len(fm.vocabulary) > 50:
                err("source_vocabulary_id is longer than 50 characters", f.name)
        if fm.kind in ("concept_map", "vocab_code") and not f.is_concept_id:
            warn(f"kind '{fm.kind}' yields a concept_id but '{f.name}' is not a concept field", f.name)
        if fm.kind == "auto_id" and not f.is_pk:
            err(f"auto_id is only allowed on the primary key ({table.primary_key or 'this table has none'})", f.name)
        if fm.kind == "visit_id" and spec.visit_key is None:
            err("kind visit_id needs a visit_key in the spec", f.name)
        if fm.kind == "person_id" and f.name != "person_id":
            warn("kind person_id is meant for the person_id field", f.name)
        # links: person and visit ids must come from the id maps, or events attach to the wrong rows
        if f.name == "person_id" and fm.kind not in ("person_id", "null"):
            err(
                f"person_id must use kind person_id (the patient key resolved through the person id map), "
                f"not '{fm.kind}': any other value does not link to the person table's ids",
                f.name,
            )
        if (tt, f.name) == ("visit_occurrence", "visit_occurrence_id"):
            if fm.kind == "auto_id":
                err("auto_id is not allowed here: visit_occurrence_id must come from the visit id map "
                    "(kind visit_id with the visit key), or events link to the wrong visits", f.name)
            elif spec.visit_key is not None and fm.kind != "visit_id":
                err(f"visit_occurrence_id must use kind visit_id (the visit key resolved through the visit id map), "
                    f"not '{fm.kind}', so that other tables' visit_id fields link to it", f.name)
        elif (
            f.name == "visit_occurrence_id"
            and spec.visit_key is not None
            and fm.kind not in ("visit_id", "null")
        ):
            warn(f"visit_occurrence_id is kind '{fm.kind}': only kind visit_id links to the loaded visits", f.name)

    # SQL safety: fragments and the tables they read
    for field, msg in _mapping_sql_problems(tm):
        err(msg, field)
    pii = _pii_loaded(tm, cat.profile, privacy)
    if pii:
        err(
            f"loads direct identifier column(s) {', '.join(pii)} into the CDM; direct identifiers must never be "
            "loaded (leave these fields unmapped or NULL)"
        )

    # tables and aliases
    aliases: dict[str, str] = {tm.source.alias.lower(): tm.source.table}
    for j in tm.joins:
        if j.alias.lower() in aliases:
            err(f"alias '{j.alias}' is declared twice")
        aliases[j.alias.lower()] = j.table
    for t, what in [(tm.source.table, "source table")] + [(j.table, "joined table") for j in tm.joins]:
        issue = _check_source_table(cat, t, tt, what)
        if issue:
            issues.append(issue)

    pii_cols = pii_columns(cat.profile, privacy)
    reported: set[tuple[str, str]] = set()
    for field, _, expr in _fragments(tm):
        # aliases declared inside the expression (correlated subqueries): alias -> staged table,
        # or None when it cannot be resolved or shadows an outer alias
        inner: dict[str, str | None] = {}
        schemas_used: set[str] = set()
        for table, alias in _inner_tables(expr):
            dotted = "{" not in table and "." in table
            if dotted and (table.split(".")[0].lower() in aliases or table.split(".")[0].lower() in inner):
                continue  # a column: EXTRACT(YEAR FROM p.dob)
            if dotted and alias is not None:
                schemas_used.add(table.split(".")[0].lower())  # reported above as a table it may not read
            if "." not in table and "{" not in table and alias is not None and cat.available \
                    and cat.columns(table.strip('"')) is not None:
                # only source/join tables get {src} added; inside an expression the name would be
                # looked up in the search_path
                err(f"inside an expression write the source table as {{src}}.{table}, not '{table}'", field)
            if alias is not None:
                inner[alias] = None if alias in aliases else _staged_name(table)
        for alias, column in alias_refs(expr):
            if alias in _NOT_ALIASES or alias in schemas_used:
                continue
            if alias in inner:
                src = inner[alias]
            elif alias in aliases:
                src = _staged_name(aliases[alias])
            else:
                if (alias, "") not in reported:
                    reported.add((alias, ""))
                    err(f"alias '{alias}' is not declared (declared: {', '.join(sorted(aliases))})", field)
                continue
            if src is None:
                continue
            if alias in inner and column in pii_cols.get(src.lower(), set()) and (alias, column) not in reported:
                reported.add((alias, column))
                err(f"reads direct identifier column '{src}.{column}'; direct identifiers must never be loaded", field)
            if not cat.available:
                continue
            cols = cat.columns(src)
            if cols is not None and column not in cols and (alias, column) not in reported:
                reported.add((alias, column))
                err(f"column '{alias}.{column}' does not exist in source table '{src}'", field)

    # conversions: the build makes those of column / expression fields and the WHERE tolerant; keys,
    # lookups and join conditions are evaluated as written
    for fm in tm.fields:
        if not fm.expression or fm.kind not in ("column", "expression", *_KEY_KINDS):
            continue
        label = "expression" if fm.kind in ("column", "expression") else f"{fm.kind} expression"
        for severity, msg in _conversion_problems(fm.expression, label, hard=fm.kind in _KEY_KINDS,
                                                  aliases=aliases, cat=cat):
            (err if severity == "error" else warn)(msg, fm.target_field)
    if tm.where:
        for severity, msg in _conversion_problems(tm.where, "where", hard=False, aliases=aliases, cat=cat):
            (err if severity == "error" else warn)(msg)
    for j in tm.joins:
        for severity, msg in _conversion_problems(j.on, f"join '{j.alias}' condition", hard=True, aliases=aliases,
                                                  cat=cat):
            (err if severity == "error" else warn)(msg)
    days = _days_supply_issue(tm)
    if days:
        warn(days, "drug_exposure_end_date")
    return issues


def _static_spec(spec: ETLSpec, cat: _SourceCatalog) -> list[SpecIssue]:
    cdm = load_cdm_spec()
    issues: list[SpecIssue] = []
    for key, what in [(spec.person_key, "person_key"), (spec.visit_key, "visit_key")]:
        if key is None:
            continue
        unsafe = _key_sql_problems(key)
        issues += [SpecIssue(severity="error", message=f"{what}: {x}") for x in unsafe]
        if unsafe:
            continue
        issue = _check_source_table(cat, key.table, None, f"{what} table")
        if issue:
            issues.append(issue)
        elif cat.available and _staged_name(key.table) is not None:
            issues += _key_columns(key, cat, what)
        for severity, msg in _conversion_problems(key.expression, what, hard=True,
                                                  formats=_bare_formats(key.expression, key.table, cat)):
            issues.append(SpecIssue(severity=severity, message=msg))  # type: ignore[arg-type]
    if not spec.mapping_for("person"):
        issues.append(SpecIssue(severity="error", target_table="person", message="no table mapping loads person"))
    has_op_mapping = spec.mapping_for("observation_period") is not None or any(
        c.target_table.lower() == "observation_period" for c in spec.custom_sql
    )
    if spec.observation_period == "table_mapping" and not has_op_mapping:
        issues.append(
            SpecIssue(
                severity="error",
                target_table="observation_period",
                message="observation_period is 'table_mapping' but no mapping loads observation_period "
                "(add one or use 'from_events')",
            )
        )
    if spec.observation_period == "from_events" and has_op_mapping:
        issues.append(
            SpecIssue(
                severity="warning",
                target_table="observation_period",
                message="observation_period is 'from_events' but a mapping also loads observation_period; "
                "both will insert periods",
            )
        )
    for cs in spec.custom_sql:
        if not cdm.has_table(cs.target_table) or cdm.table(cs.target_table).schema_class != "CDM":
            issues.append(
                SpecIssue(severity="error", target_table=cs.target_table, message=f"custom SQL '{cs.name}' targets a non-CDM table")
            )
        issues.append(
            SpecIssue(severity="warning", target_table=cs.target_table,
                      message=f"custom SQL '{cs.name}' is executed as written; review it")
        )
        issues += [
            SpecIssue(severity="error", target_table=cs.target_table, message=f"custom SQL '{cs.name}': {x}")
            for x in _custom_sql_problems(cs)
        ]
        # custom SQL runs as written: its conversions are only checked against the profile
        issues += [
            SpecIssue(severity="error", target_table=cs.target_table, message=msg)
            for severity, msg in _conversion_problems(cs.sql, f"custom SQL '{cs.name}'", hard=True,
                                                      aliases=_custom_aliases(cs.sql), cat=cat)
            if severity == "error"
        ]
    return issues


_FROM_ITEM = re.compile(
    r"\b(?:from|join)\s+((?:\{src\}\s*\.\s*)?(?:\"[^\"]+\"|[a-z_][a-z0-9_]*))(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?",
    re.I,
)
_NOT_ALIAS = {"on", "using", "where", "join", "left", "right", "inner", "full", "cross", "natural", "group", "order",
              "limit", "union", "except", "intersect", "window", "having", "offset", "lateral", "returning"}


def _custom_aliases(sql: str) -> dict[str, str]:
    """alias -> staged table of the FROM and JOIN items of custom SQL (a table without an alias is
    its own alias)."""
    out: dict[str, str] = {}
    try:  # string literals and comments out, quoted names kept
        text = "".join(" " if t.kind in ("str", "estr", "dollar", "comment") else t.text for t in lex(sql))
    except LexError:
        return out
    for m in _FROM_ITEM.finditer(text):
        table = re.sub(r"\s+", "", m.group(1))
        alias = (m.group(2) or "").lower()
        if not alias or alias in _NOT_ALIAS:
            alias = table.split(".")[-1].strip('"').lower()
        out.setdefault(alias, table)
    return out


def _key_columns(key: KeySpec, cat: _SourceCatalog, what: str) -> list[SpecIssue]:
    """A key over a single bare column must name an existing column."""
    expr = key.expression.strip()
    cols = cat.columns(_staged_name(key.table) or key.table) or set()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr) and expr.lower() not in cols:
        return [SpecIssue(severity="error", message=f"{what} column '{expr}' does not exist in '{key.table}'")]
    return []


# --------------------------------------------------------------------------------------
# database checks
# --------------------------------------------------------------------------------------


def _concept_constants(tm: TableMapping) -> list[tuple[str, int]]:
    """(field, concept_id) literals used in *_concept_id fields."""
    out: list[tuple[str, int]] = []
    for fm in tm.fields:
        if not fm.target_field.endswith("_concept_id") or not fm.expression:
            continue
        if fm.kind == "constant":
            m = _INT_LITERAL.match(fm.expression.strip())
            if m:
                out.append((fm.target_field, int(m.group(1))))
        elif fm.kind == "expression":
            for a, b in _CASE_INT.findall(fm.expression):
                out.append((fm.target_field, int(a or b)))
    return out


def _db_concepts(tm: TableMapping, conn: psycopg.Connection, schemas: Schemas) -> list[SpecIssue]:
    cdm = load_cdm_spec()
    if not cdm.has_table(tm.target_table):
        return []
    table = cdm.table(tm.target_table)
    issues: list[SpecIssue] = []
    tt = tm.target_table

    for fm in tm.fields:
        if fm.target_field.endswith("_concept_id") and fm.kind == "constant" and fm.expression:
            if not _INT_LITERAL.match(fm.expression.strip()) and fm.expression.strip().upper() != "NULL":
                issues.append(
                    SpecIssue(severity="warning", target_table=tt, target_field=fm.target_field,
                              message=f"constant '{fm.expression}' is not an integer concept_id")
                )

    consts = [(f, c) for f, c in _concept_constants(tm) if c != 0]
    if consts:
        ids = sorted({c for _, c in consts})
        try:
            with conn.transaction():
                rows = fetch_all(
                    conn,
                    f"SELECT concept_id, concept_name, domain_id, standard_concept, invalid_reason "
                    f"FROM {qi(schemas.vocab)}.concept WHERE concept_id = ANY(%s)",
                    (ids,),
                )
        except psycopg.Error as e:
            return issues + [SpecIssue(severity="warning", target_table=tt, message=f"could not check concept ids: {short_error(e)}")]
        found = {r["concept_id"]: r for r in rows}
        for field, cid in consts:
            f = table.field(field)
            r = found.get(cid)
            if r is None:
                issues.append(SpecIssue(severity="error", target_table=tt, target_field=field,
                                        message=f"concept_id {cid} does not exist in the vocabulary"))
                continue
            label = f"concept_id {cid} ({r['concept_name']}, domain {r['domain_id']})"
            if r["invalid_reason"]:
                issues.append(SpecIssue(severity="error", target_table=tt, target_field=field,
                                        message=f"{label} is invalid (invalid_reason {r['invalid_reason']})"))
            if not field.endswith("_source_concept_id") and r["standard_concept"] != "S":
                issues.append(SpecIssue(severity="error", target_table=tt, target_field=field,
                                        message=f"{label} is not a standard concept"))
            if f is not None and f.fk_domain:
                allowed = {d.strip() for d in f.fk_domain.split(",")}
                if r["domain_id"] not in allowed:
                    issues.append(SpecIssue(severity="error", target_table=tt, target_field=field,
                                            message=f"{label} is not in the expected domain {f.fk_domain}"))

    vocabs = sorted({fm.vocabulary for fm in tm.fields if fm.kind == "vocab_code" and fm.vocabulary})
    if vocabs:
        try:
            with conn.transaction():
                rows = fetch_all(
                    conn,
                    f"SELECT vocabulary_id FROM {qi(schemas.vocab)}.vocabulary WHERE vocabulary_id = ANY(%s)",
                    (vocabs,),
                )
        except psycopg.Error as e:
            return issues + [SpecIssue(severity="warning", target_table=tt, message=f"could not check vocabularies: {short_error(e)}")]
        present = {r["vocabulary_id"] for r in rows}
        for fm in tm.fields:
            if fm.kind == "vocab_code" and fm.vocabulary and fm.vocabulary not in present:
                issues.append(SpecIssue(severity="error", target_table=tt, target_field=fm.target_field,
                                        message=f"vocabulary '{fm.vocabulary}' is not loaded"))
    return issues


def _explain(conn: psycopg.Connection, sql: str) -> str | None:
    try:
        with conn.transaction():
            execute(conn, "EXPLAIN " + sql.rstrip().rstrip(";"))
        return None
    except psycopg.Error as e:
        return short_error(e)


def _single_statement(sql: str) -> str | None:
    """The statement if ``sql`` is one explainable statement, else None."""
    body = re.sub(r"--[^\n]*", "", sql).strip().rstrip(";").strip()
    if ";" in body:
        return None
    return body if re.match(r"(?is)^(insert|update|delete|with|select)\b", body) else None


def _explain_all(
    spec: ETLSpec,
    tables: list[TableMapping],
    customs: list[CustomSQL],
    conn: psycopg.Connection,
    schemas: Schemas,
    keys: bool = True,
    cfg: ProjectConfig | None = None,
) -> list[SpecIssue]:
    issues: list[SpecIssue] = []
    compiled: list[tuple[TableMapping, str]] = []
    for tm in tables:
        try:
            compiled.append((tm, compile_table(tm, spec, schemas, etl=cfg.etl if cfg is not None else None)))
        except CompileError as e:
            issues.append(SpecIssue(severity="error", target_table=tm.target_table, message=f"does not compile: {e}"))
    try:
        with rollback_only(conn):
            with conn.transaction():
                create_cdm_tables(conn, schemas, include_vocab=False)
                create_etl_tables(conn, schemas)
            apply_session(conn, cfg)  # constants are folded at EXPLAIN: read them as the build does
            for key, what in [(spec.person_key, "person_key"), (spec.visit_key, "visit_key")]:
                if key is None or not keys or _key_sql_problems(key):
                    continue
                probe = schemas.render(f"SELECT DISTINCT ({key.expression})::text FROM {source_ref(key.table)}")
                error = _explain(conn, probe)
                if error:
                    issues.append(SpecIssue(severity="error", message=f"{what} expression fails: {error}"))
            for tm, sql in compiled:
                error = _explain(conn, sql)
                if error:
                    issues.append(SpecIssue(severity="error", target_table=tm.target_table, message=f"EXPLAIN failed: {error}"))
            for cs in customs:
                stmt = _single_statement(schemas.render(cs.sql))
                if stmt is None:
                    continue
                error = _explain(conn, stmt)
                if error:
                    issues.append(SpecIssue(severity="error", target_table=cs.target_table,
                                            message=f"custom SQL '{cs.name}': EXPLAIN failed: {error}"))
    except psycopg.Error as e:
        issues.append(SpecIssue(severity="warning", message=f"could not EXPLAIN the spec: {short_error(e)}"))
    return issues


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------


def validate_table_mapping(
    tm: TableMapping,
    spec: ETLSpec,
    profile: SourceProfile | None = None,
    conn: psycopg.Connection | None = None,
    schemas: Schemas | None = None,
    privacy: PrivacyConfig | None = None,
    *,
    cfg: ProjectConfig | None = None,
) -> list[SpecIssue]:
    """Issues of one table mapping, in the context of ``spec`` (its keys). ``cfg`` gives the
    session settings the EXPLAIN runs under (defaults without it)."""
    cat = _SourceCatalog(profile, conn, schemas)
    issues = _static_mapping(tm, spec, cat, privacy)
    if conn is not None and schemas is not None:
        issues += _db_concepts(tm, conn, schemas)
        if not any(i.severity == "error" for i in issues):
            issues += _explain_all(spec, [tm], [], conn, schemas, keys=False, cfg=cfg)
    return issues


def validate_spec(
    spec: ETLSpec,
    profile: SourceProfile | None = None,
    conn: psycopg.Connection | None = None,
    schemas: Schemas | None = None,
    privacy: PrivacyConfig | None = None,
    *,
    cfg: ProjectConfig | None = None,
) -> list[SpecIssue]:
    """All issues found in ``spec``. Errors block the build; warnings are for review.

    ``privacy`` (default ``PrivacyConfig()``) decides, with the profile's PII flags, which
    source columns are direct identifiers that must not be loaded. ``cfg`` gives the session
    settings the EXPLAINs run under (defaults without it).
    """
    cat = _SourceCatalog(profile, conn, schemas)
    issues = _static_spec(spec, cat)
    clean: list[TableMapping] = []
    for tm in spec.table_mappings:
        table_issues = _static_mapping(tm, spec, cat, privacy)
        if conn is not None and schemas is not None:
            table_issues += _db_concepts(tm, conn, schemas)
        issues += table_issues
        if not any(i.severity == "error" for i in table_issues):
            clean.append(tm)
    if conn is not None and schemas is not None:
        customs = [
            c for c in spec.custom_sql if load_cdm_spec().has_table(c.target_table) and not _custom_sql_problems(c)
        ]
        issues += _explain_all(spec, clean, customs, conn, schemas, cfg=cfg)
    return issues
