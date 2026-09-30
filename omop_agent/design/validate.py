"""Check an ``ETLSpec`` before anything runs.

Static checks need only the spec (and optionally the source profile): tables and fields exist
in CDM v5.4, required fields are mapped, aliases and source columns resolve, field kinds carry
what they need. With a database connection the validator also checks concept constants against
the loaded vocabulary, checks that ``vocab_code`` vocabularies are loaded, and ``EXPLAIN``s each
compiled INSERT inside a transaction that is always rolled back.

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
from ..db import Schemas, execute, fetch_all, list_columns, qi, rollback_only, short_error
from ..etl.compiler import CompileError, alias_refs, compile_table
from ..models import CustomSQL, ETLSpec, KeySpec, SourceProfile, TableMapping

# Qualifiers that are schemas, not table aliases (e.g. pg_catalog.to_char(...)::pg_catalog.text).
_NOT_ALIASES = {"pg_catalog", "public", "information_schema"}
_CASE_INT = re.compile(r"\bTHEN\s+(\d+)|\bELSE\s+(\d+)", re.I)
_INT_LITERAL = re.compile(r"^\(*\s*'?\s*(-?\d+)\s*'?\s*\)*$")
_NEEDS_EXPRESSION = {"column", "expression", "constant", "concept_map", "vocab_code", "person_id", "visit_id"}


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


def _is_qualified(table: str) -> bool:
    return "." in table or "{" in table


def _check_source_table(cat: _SourceCatalog, table: str, tt: str | None, what: str) -> SpecIssue | None:
    if not cat.available or _is_qualified(table):
        return None
    if cat.columns(table) is None:
        where = "in the source profile" if cat.profile is not None else "in the staging schema"
        return SpecIssue(severity="error", target_table=tt, message=f"{what} '{table}' does not exist {where}")
    return None


def _static_mapping(tm: TableMapping, spec: ETLSpec, cat: _SourceCatalog) -> list[SpecIssue]:
    cdm = load_cdm_spec()
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

    exprs: list[tuple[str | None, str]] = [(fm.target_field, fm.expression or "") for fm in tm.fields]
    exprs += [(None, j.on) for j in tm.joins]
    if tm.where:
        exprs.append((None, tm.where))
    reported: set[tuple[str, str]] = set()
    for field, expr in exprs:
        for alias, column in alias_refs(expr):
            if alias in _NOT_ALIASES:
                continue
            if alias not in aliases:
                if (alias, "") not in reported:
                    reported.add((alias, ""))
                    err(f"alias '{alias}' is not declared (declared: {', '.join(sorted(aliases))})", field)
                continue
            src = aliases[alias]
            if not cat.available or _is_qualified(src):
                continue
            cols = cat.columns(src)
            if cols is not None and column not in cols and (alias, column) not in reported:
                reported.add((alias, column))
                err(f"column '{alias}.{column}' does not exist in source table '{src}'", field)
    return issues


def _static_spec(spec: ETLSpec, cat: _SourceCatalog) -> list[SpecIssue]:
    cdm = load_cdm_spec()
    issues: list[SpecIssue] = []
    for key, what in [(spec.person_key, "person_key"), (spec.visit_key, "visit_key")]:
        if key is None:
            continue
        issue = _check_source_table(cat, key.table, None, f"{what} table")
        if issue:
            issues.append(issue)
        elif cat.available and not _is_qualified(key.table):
            issues += _key_columns(key, cat, what)
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
    return issues


def _key_columns(key: KeySpec, cat: _SourceCatalog, what: str) -> list[SpecIssue]:
    """A key over a single bare column must name an existing column."""
    expr = key.expression.strip()
    cols = cat.columns(key.table) or set()
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
) -> list[SpecIssue]:
    issues: list[SpecIssue] = []
    compiled: list[tuple[TableMapping, str]] = []
    for tm in tables:
        try:
            compiled.append((tm, compile_table(tm, spec, schemas)))
        except CompileError as e:
            issues.append(SpecIssue(severity="error", target_table=tm.target_table, message=f"does not compile: {e}"))
    try:
        with rollback_only(conn):
            with conn.transaction():
                create_cdm_tables(conn, schemas, include_vocab=False)
                create_etl_tables(conn, schemas)
            for key, what in [(spec.person_key, "person_key"), (spec.visit_key, "visit_key")]:
                if key is None or not keys:
                    continue
                probe = schemas.render(
                    f"SELECT DISTINCT ({key.expression})::text FROM "
                    + (key.table if _is_qualified(key.table) else "{src}." + qi(key.table))
                )
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
) -> list[SpecIssue]:
    """Issues of one table mapping, in the context of ``spec`` (its keys)."""
    cat = _SourceCatalog(profile, conn, schemas)
    issues = _static_mapping(tm, spec, cat)
    if conn is not None and schemas is not None:
        issues += _db_concepts(tm, conn, schemas)
        if not any(i.severity == "error" for i in issues):
            issues += _explain_all(spec, [tm], [], conn, schemas, keys=False)
    return issues


def validate_spec(
    spec: ETLSpec,
    profile: SourceProfile | None = None,
    conn: psycopg.Connection | None = None,
    schemas: Schemas | None = None,
) -> list[SpecIssue]:
    """All issues found in ``spec``. Errors block the build; warnings are for review."""
    cat = _SourceCatalog(profile, conn, schemas)
    issues = _static_spec(spec, cat)
    clean: list[TableMapping] = []
    for tm in spec.table_mappings:
        table_issues = _static_mapping(tm, spec, cat)
        if conn is not None and schemas is not None:
            table_issues += _db_concepts(tm, conn, schemas)
        issues += table_issues
        if not any(i.severity == "error" for i in table_issues):
            clean.append(tm)
    if conn is not None and schemas is not None:
        customs = [c for c in spec.custom_sql if load_cdm_spec().has_table(c.target_table)]
        issues += _explain_all(spec, clean, customs, conn, schemas)
    return issues
