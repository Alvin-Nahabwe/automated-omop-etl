"""Compile an ``ETLSpec`` into PostgreSQL.

Each ``TableMapping`` becomes one ``INSERT INTO {cdm}.<table> (...) SELECT ... FROM
{src}.<table> <alias> [JOIN ...] [WHERE ...]``. Field kinds compile as described in
``docs/architecture.md`` ("The ETL spec"): expressions are cast to the CDM column type,
``person_id`` / ``visit_id`` resolve through the id maps, ``auto_id`` continues after the
largest existing key, and concept lookups (``vocab_code``, ``concept_map``) become LEFT JOINs
whose result is 0 when a source value exists but has no mapping.

A full compile yields ordered steps: id maps, table inserts (in CDM load order, each followed
by its custom SQL), remaining custom SQL, the derived observation period, and the
``cdm_source`` row. The generated SQL is written to ``sql/*.sql`` for humans to review, so it
is laid out one field per line with the field kind as a trailing comment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..cdm.spec import EVENT_DATE_FIELDS, LOAD_ORDER, CdmField, CdmTable, load_cdm_spec
from ..config import CdmSourceConfig
from ..db import Schemas, qi, ql
from ..models import CustomSQL, ETLSpec, FieldMapping, KeySpec, TableMapping

StepKind = Literal["id_map", "insert", "custom", "derived", "metadata"]

# Kinds whose ``expression`` must be given.
_NEEDS_EXPRESSION = {"column", "expression", "constant", "concept_map", "vocab_code", "person_id", "visit_id"}
_ALIAS_REF = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


class CompileError(ValueError):
    """The spec cannot be turned into SQL (unknown table/field, incomplete field mapping, ...)."""


@dataclass
class CompiledStep:
    name: str
    target_table: str | None
    kind: StepKind
    sql: str
    source_table: str | None = None
    source_count_sql: str | None = None
    error: str | None = None  # set instead of raising when compile_spec(strict=False)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _p(expr: str) -> str:
    """Parenthesise an expression; a trailing ``--`` comment must not swallow what follows."""
    expr = expr.strip()
    return f"({expr}\n)" if "--" in expr else f"({expr})"


def _norm(expr: str) -> str:
    return " ".join(expr.split())


def cast_to(expr: str, datatype: str) -> str:
    """Cast ``expr`` to the SQL type of a CDM field datatype (``integer``, ``varchar(50)``, ...)."""
    dt = datatype.strip().lower()
    e = _p(expr)
    if dt in ("integer", "int", "int4"):
        return f"{e}::integer"
    if dt in ("bigint", "int8"):
        return f"{e}::bigint"
    if dt in ("float", "numeric", "double", "real"):
        return f"{e}::numeric"
    if dt == "date":
        return f"{e}::date"
    if dt in ("datetime", "timestamp"):
        return f"{e}::timestamp"
    m = re.fullmatch(r"varchar\((\d+)\)", dt)
    if m:
        return f"LEFT({e}::text, {m.group(1)})"
    if dt in ("varchar(max)", "text", "varchar"):
        return f"{e}::text"
    return e


def alias_refs(expr: str | None) -> list[tuple[str, str]]:
    """``(alias, column)`` pairs referenced as ``alias.column`` outside string literals."""
    if not expr:
        return []
    text = _STRING_LITERAL.sub("''", expr)
    out = []
    for m in _ALIAS_REF.finditer(text):
        # schema-qualified function calls such as pg_catalog.foo(...) are not column refs
        rest = text[m.end():].lstrip()
        if rest.startswith("("):
            continue
        out.append((m.group(1).lower(), m.group(2).lower()))
    return out


def source_ref(table: str) -> str:
    """A source table reference: bare names live in ``{src}``; qualified names are kept."""
    table = table.strip()
    if "." in table or "{" in table:
        return table
    return "{src}." + qi(table)


def _cdm_table(name: str) -> CdmTable:
    spec = load_cdm_spec()
    if not spec.has_table(name) or spec.table(name).schema_class != "CDM":
        raise CompileError(f"'{name}' is not an OMOP CDM v5.4 clinical/derived table")
    return spec.table(name)


def _comment(text: str) -> str:
    return "\n".join("-- " + line if line.strip() else "--" for line in text.strip().splitlines())


# --------------------------------------------------------------------------------------
# one table mapping -> INSERT ... SELECT
# --------------------------------------------------------------------------------------


class _Joins:
    """Lookup joins generated for one INSERT, with alias allocation that avoids user aliases."""

    def __init__(self, taken: set[str]):
        self.taken = set(taken)
        self.counters: dict[str, int] = {}
        self.lines: list[str] = []
        self.vocab: dict[tuple[str, str], dict[str, str | None]] = {}
        self.stcm: dict[tuple[str, str], str] = {}
        self._vocab_line_index: dict[tuple[str, str], int] = {}

    def alias(self, prefix: str) -> str:
        n = self.counters.get(prefix, 0)
        while True:
            n += 1
            name = f"{prefix}{n}"
            if name not in self.taken:
                break
        self.counters[prefix] = n
        self.taken.add(name)
        return name

    def person(self, expr: str) -> str:
        a = self.alias("pm")
        self.lines.append(f"INNER JOIN {{etl}}.person_map {a} ON {a}.source_value = {_p(expr)}::text")
        return f"{a}.person_id"

    def visit(self, expr: str, inner: bool) -> str:
        a = self.alias("vm")
        jt = "INNER" if inner else "LEFT"
        self.lines.append(f"{jt} JOIN {{etl}}.visit_map {a} ON {a}.source_value = {_p(expr)}::text")
        return f"{a}.visit_occurrence_id"

    def vocab_code(self, expr: str, vocabulary: str, standard: bool) -> str:
        key = (_norm(expr), vocabulary)
        g = self.vocab.get(key)
        if g is None:
            sc = self.alias("sc")
            g = {"sc": sc, "cr": None, "tc": None}
            self.vocab[key] = g
            self._vocab_line_index[key] = len(self.lines)
            self.lines.append(
                f"LEFT JOIN {{vocab}}.concept {sc} ON {sc}.concept_code = {_p(expr)}::text"
                f" AND {sc}.vocabulary_id = {ql(vocabulary)}"
            )
        if standard and g["tc"] is None:
            sc, cr, tc = g["sc"], self.alias("cr"), self.alias("tc")
            g["cr"], g["tc"] = cr, tc
            extra = [
                f"LEFT JOIN {{vocab}}.concept_relationship {cr} ON {cr}.concept_id_1 = {sc}.concept_id"
                f" AND {cr}.relationship_id = 'Maps to' AND {cr}.invalid_reason IS NULL",
                f"LEFT JOIN {{vocab}}.concept {tc} ON {tc}.concept_id = {cr}.concept_id_2"
                f" AND {tc}.standard_concept = 'S' AND {tc}.invalid_reason IS NULL",
            ]
            at = self._vocab_line_index[key] + 1
            self.lines[at:at] = extra
            for k, idx in self._vocab_line_index.items():
                if idx >= at:
                    self._vocab_line_index[k] = idx + len(extra)
        return f"{g['tc']}.concept_id" if standard else f"{g['sc']}.concept_id"

    def concept_map(self, expr: str, vocabulary: str, source_role: bool) -> str:
        key = (_norm(expr), vocabulary)
        a = self.stcm.get(key)
        if a is None:
            a = self.alias("m")
            self.stcm[key] = a
            self.lines.append(
                f"LEFT JOIN {{etl}}.source_to_concept_map {a} ON {a}.source_code = {_p(expr)}::text"
                f" AND {a}.source_vocabulary_id = {ql(vocabulary)} AND {a}.invalid_reason IS NULL"
            )
        return f"{a}.source_concept_id" if source_role else f"{a}.target_concept_id"


def _field_value(fm: FieldMapping, f: CdmField, table: CdmTable, tm: TableMapping, spec: ETLSpec, joins: _Joins) -> str:
    kind = fm.kind
    expr = (fm.expression or "").strip()
    where = f"{table.name}.{f.name}"
    if kind in _NEEDS_EXPRESSION and not expr:
        raise CompileError(f"{where}: kind '{kind}' needs an expression")
    if kind == "null":
        return "NULL"
    if kind in ("column", "expression", "constant"):
        return cast_to(expr, f.datatype)
    if kind == "auto_id":
        if not f.is_pk:
            pk = table.primary_key or "none"
            raise CompileError(f"{where}: auto_id is only allowed on the table's primary key ({pk})")
        value = f"(SELECT COALESCE(MAX({qi(f.name)}), 0) FROM {{cdm}}.{qi(table.name)}) + ROW_NUMBER() OVER ()"
        return cast_to(value, f.datatype)
    if kind == "person_id":
        return cast_to(joins.person(expr), f.datatype)
    if kind == "visit_id":
        if spec.visit_key is None:
            raise CompileError(f"{where}: kind visit_id needs spec.visit_key (the visit id map)")
        return cast_to(joins.visit(expr, inner=f.required), f.datatype)
    if kind in ("vocab_code", "concept_map"):
        if not (fm.vocabulary or "").strip():
            raise CompileError(f"{where}: kind '{kind}' needs a vocabulary")
        vocab = fm.vocabulary.strip()  # type: ignore[union-attr]
        if kind == "vocab_code":
            value = joins.vocab_code(expr, vocab, standard=fm.concept_role == "standard")
        else:
            value = joins.concept_map(expr, vocab, source_role=fm.concept_role == "source")
        when_null = "0" if f.required else "NULL"
        return cast_to(f"CASE WHEN {_p(expr)} IS NULL THEN {when_null} ELSE COALESCE({value}, 0) END", f.datatype)
    raise CompileError(f"{where}: unknown field kind '{kind}'")  # pragma: no cover - pydantic guards kinds


def _from_clause(tm: TableMapping) -> list[str]:
    lines = [f"FROM {source_ref(tm.source.table)} {qi(tm.source.alias)}"]
    for j in tm.joins:
        jt = "INNER" if j.join_type == "inner" else "LEFT"
        lines.append(f"{jt} JOIN {source_ref(j.table)} {qi(j.alias)} ON {j.on.strip()}")
    return lines


def _check_mapping(tm: TableMapping) -> CdmTable:
    table = _cdm_table(tm.target_table)
    seen: set[str] = set()
    for fm in tm.fields:
        if table.field(fm.target_field) is None:
            raise CompileError(f"{table.name}.{fm.target_field} is not a field of CDM table {table.name}")
        if fm.target_field in seen:
            raise CompileError(f"{table.name}.{fm.target_field} is mapped more than once")
        seen.add(fm.target_field)
    aliases = [tm.source.alias.lower()] + [j.alias.lower() for j in tm.joins]
    if len(set(aliases)) != len(aliases):
        raise CompileError(f"{table.name}: duplicate table alias in source/joins ({', '.join(aliases)})")
    return table


def _compile_insert(tm: TableMapping, spec: ETLSpec) -> str:
    """Unrendered INSERT (placeholders kept)."""
    table = _check_mapping(tm)
    by_name = {fm.target_field: fm for fm in tm.fields}
    ordered = [(f, by_name[f.name]) for f in table.fields if f.name in by_name]
    if not ordered:
        raise CompileError(f"{table.name}: the mapping has no fields")
    joins = _Joins({tm.source.alias.lower()} | {j.alias.lower() for j in tm.joins})
    values = [(f, fm, _field_value(fm, f, table, tm, spec, joins)) for f, fm in ordered]

    header = f"{table.name} <- {tm.source.table} ({tm.source.alias})"
    if tm.joins:
        header += " + " + ", ".join(f"{j.table} ({j.alias})" for j in tm.joins)
    out = [_comment(header)]
    if tm.rationale.strip():
        out.append(_comment(tm.rationale))
    out.append(f"INSERT INTO {{cdm}}.{qi(table.name)} (")
    out += [f"    {qi(f.name)}{',' if i < len(values) - 1 else ''}" for i, (f, _, _) in enumerate(values)]
    out.append(")")
    out.append("SELECT")
    for i, (f, fm, v) in enumerate(values):
        sep = "," if i < len(values) - 1 else ""
        out.append(f"    {v} AS {qi(f.name)}{sep}  -- {fm.kind}")
    out += _from_clause(tm)
    out += joins.lines
    if tm.where and tm.where.strip():
        out.append(f"WHERE {_p(tm.where)}")
    return "\n".join(out) + ";"


def _source_count(tm: TableMapping) -> str:
    lines = ["SELECT COUNT(*) AS n", *_from_clause(tm)]
    if tm.where and tm.where.strip():
        lines.append(f"WHERE {_p(tm.where)}")
    return "\n".join(lines)


def compile_table(tm: TableMapping, spec: ETLSpec, schemas: Schemas) -> str:
    """One rendered ``INSERT ... SELECT`` for a table mapping. Raises ``CompileError``."""
    return schemas.render(_compile_insert(tm, spec))


def source_count_sql(tm: TableMapping, schemas: Schemas) -> str:
    """``SELECT COUNT(*)`` of the mapping's source rows after user joins/WHERE, before id-map filtering."""
    return schemas.render(_source_count(tm))


# --------------------------------------------------------------------------------------
# id maps, derived tables, metadata
# --------------------------------------------------------------------------------------


def _person_map_sql(key: KeySpec) -> str:
    return "\n".join(
        [
            _comment(f"person id map: one person_id per distinct {key.table}.{_norm(key.expression)}"),
            "TRUNCATE TABLE {etl}.person_map;",
            "INSERT INTO {etl}.person_map (source_value, person_id)",
            "SELECT k.source_value, (ROW_NUMBER() OVER (ORDER BY k.source_value))::integer",
            "FROM (",
            f"    SELECT DISTINCT {_p(key.expression)}::text AS source_value",
            f"    FROM {source_ref(key.table)}",
            f"    WHERE {_p(key.expression)} IS NOT NULL",
            ") k;",
        ]
    )


def _visit_person_expr(spec: ETLSpec) -> tuple[str, str] | None:
    """(alias, person expression) from a mapping that reads the visit table, if one is usable."""
    vk = spec.visit_key
    if vk is None:
        return None
    for tm in spec.table_mappings:
        if tm.source.table != vk.table:
            continue
        pf = next((f for f in tm.fields if f.kind == "person_id" and f.expression), None)
        if pf is None:
            continue
        refs = {a for a, _ in alias_refs(pf.expression)}
        if refs <= {tm.source.alias.lower()}:
            return tm.source.alias, pf.expression  # type: ignore[return-value]
    return None


def _visit_map_sql(spec: ETLSpec) -> str:
    vk = spec.visit_key
    assert vk is not None
    person = _visit_person_expr(spec)
    alias = qi(person[0]) if person else "v"
    person_sql = f"MIN({_p(person[1])}::text)" if person else "NULL::text"
    return "\n".join(
        [
            _comment(f"visit id map: one visit_occurrence_id per distinct {vk.table}.{_norm(vk.expression)}"),
            "TRUNCATE TABLE {etl}.visit_map;",
            "INSERT INTO {etl}.visit_map (source_value, visit_occurrence_id, person_source_value)",
            "SELECT k.source_value, (ROW_NUMBER() OVER (ORDER BY k.source_value))::integer, k.person_source_value",
            "FROM (",
            f"    SELECT {_p(vk.expression)}::text AS source_value, {person_sql} AS person_source_value",
            f"    FROM {source_ref(vk.table)} {alias}",
            f"    WHERE {_p(vk.expression)} IS NOT NULL",
            "    GROUP BY 1",
            ") k;",
        ]
    )


def _observation_period_sql(spec: ETLSpec, loaded: list[str]) -> str | None:
    selects = []
    for t in loaded:
        if t not in EVENT_DATE_FIELDS:
            continue
        for col in EVENT_DATE_FIELDS[t]:
            if col:
                selects.append(f"    SELECT person_id, {qi(col)} AS event_date FROM {{cdm}}.{qi(t)}")
    if not selects:
        return None
    union = "\n    UNION ALL\n".join(selects)
    lines = [
        _comment("observation_period derived from events: one period per person, first to last event date"),
        "INSERT INTO {cdm}.observation_period (",
        "    observation_period_id,",
        "    person_id,",
        "    observation_period_start_date,",
        "    observation_period_end_date,",
        "    period_type_concept_id",
        ")",
        "SELECT",
        "    ((SELECT COALESCE(MAX(observation_period_id), 0) FROM {cdm}.observation_period)"
        " + ROW_NUMBER() OVER (ORDER BY ev.person_id))::integer,",
        "    ev.person_id,",
        "    MIN(ev.event_date),",
        "    MAX(ev.event_date),",
        f"    {int(spec.observation_period_type_concept_id)}",
        "FROM (",
        union,
        ") ev",
        "WHERE ev.event_date IS NOT NULL",
    ]
    if "person" in loaded:
        lines.append("  AND ev.person_id IN (SELECT person_id FROM {cdm}.person)")
    lines.append("GROUP BY ev.person_id;")
    return "\n".join(lines)


def _cdm_source_sql(cdm_source: CdmSourceConfig | None) -> str:
    cs = cdm_source or CdmSourceConfig()
    return "\n".join(
        [
            _comment("cdm_source: one row describing this CDM instance"),
            "INSERT INTO {cdm}.cdm_source (",
            "    cdm_source_name,",
            "    cdm_source_abbreviation,",
            "    cdm_holder,",
            "    source_description,",
            "    cdm_etl_reference,",
            "    source_release_date,",
            "    cdm_release_date,",
            "    cdm_version,",
            "    cdm_version_concept_id,",
            "    vocabulary_version",
            ")",
            "SELECT",
            f"    LEFT({ql(cs.cdm_source_name)}, 255),",
            f"    LEFT({ql(cs.cdm_source_abbreviation)}, 25),",
            f"    LEFT({ql(cs.cdm_holder)}, 255),",
            f"    {ql(cs.source_description)},",
            "    'omop-agent',",
            "    CURRENT_DATE,",
            "    CURRENT_DATE,",
            "    '5.4',",
            "    COALESCE((SELECT concept_id FROM {vocab}.concept"
            " WHERE vocabulary_id = 'CDM' AND concept_code = 'CDM v5.4' LIMIT 1), 0),",
            "    LEFT(COALESCE((SELECT vocabulary_version FROM {vocab}.vocabulary"
            " WHERE vocabulary_id = 'None'), 'unknown'), 20);",
        ]
    )


# --------------------------------------------------------------------------------------
# whole spec
# --------------------------------------------------------------------------------------


def _load_index(table: str) -> int:
    return LOAD_ORDER.index(table) if table in LOAD_ORDER else len(LOAD_ORDER)


def _is_cdm_table(name: str) -> bool:
    spec = load_cdm_spec()
    return spec.has_table(name) and spec.table(name).schema_class == "CDM"


def target_tables(spec: ETLSpec) -> list[str]:
    """Every CDM table the compiled spec writes (incl. observation_period, cdm_source), in load order."""
    names = {tm.target_table.lower() for tm in spec.table_mappings}
    names |= {cs.target_table.lower() for cs in spec.custom_sql}
    if spec.observation_period == "from_events":
        names.add("observation_period")
    names.add("cdm_source")
    return sorted((n for n in names if _is_cdm_table(n)), key=lambda n: (_load_index(n), n))


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower() or "step"


def compile_spec(
    spec: ETLSpec,
    schemas: Schemas,
    cdm_source: CdmSourceConfig | None = None,
    *,
    strict: bool = True,
) -> list[CompiledStep]:
    """Compile the whole spec into ordered steps.

    With ``strict=False`` a table mapping that does not compile yields a step with ``error``
    set (and empty SQL) instead of raising, so the other tables can still be loaded.
    """
    steps: list[CompiledStep] = []  # names are assigned at the end

    steps.append(CompiledStep("person_map", None, "id_map", _person_map_sql(spec.person_key), source_table=spec.person_key.table))
    if spec.visit_key is not None:
        steps.append(CompiledStep("visit_map", None, "id_map", _visit_map_sql(spec), source_table=spec.visit_key.table))

    ordered = sorted(enumerate(spec.table_mappings), key=lambda it: (_load_index(it[1].target_table.lower()), it[0]))
    customs_by_table: dict[str, list[CustomSQL]] = {}
    for cs in spec.custom_sql:
        customs_by_table.setdefault(cs.target_table.lower(), []).append(cs)
    last_index_for: dict[str, int] = {}
    for pos, (_, tm) in enumerate(ordered):
        last_index_for[tm.target_table.lower()] = pos

    seen_count: dict[str, int] = {}
    placed_custom: set[int] = set()
    for pos, (_, tm) in enumerate(ordered):
        t = tm.target_table.lower()
        seen_count[t] = seen_count.get(t, 0) + 1
        label = t if seen_count[t] == 1 else f"{t}_{seen_count[t]}"
        try:
            sql = _compile_insert(tm, spec)
            step = CompiledStep(label, t, "insert", sql, source_table=tm.source.table, source_count_sql=_source_count(tm))
        except CompileError as e:
            if strict:
                raise
            step = CompiledStep(label, t, "insert", "", source_table=tm.source.table, error=f"compile error: {e}")
        steps.append(step)
        if last_index_for[t] == pos:
            for cs in customs_by_table.get(t, []):
                placed_custom.add(id(cs))
                steps.append(_custom_step(cs))
    for cs in spec.custom_sql:
        if id(cs) not in placed_custom:
            steps.append(_custom_step(cs))

    if spec.observation_period == "from_events":
        loaded = [t for t in target_tables(spec) if t != "observation_period"]
        op = _observation_period_sql(spec, loaded)
        if op is not None:
            steps.append(CompiledStep("observation_period", "observation_period", "derived", op))

    steps.append(CompiledStep("cdm_source", "cdm_source", "metadata", _cdm_source_sql(cdm_source)))

    width = max(2, len(str(len(steps))))
    for i, s in enumerate(steps, start=1):
        s.name = f"{i:0{width}d}_{s.name}"
        s.sql = schemas.render(s.sql)
        if s.source_count_sql:
            s.source_count_sql = schemas.render(s.source_count_sql)
    return steps


def _custom_step(cs: CustomSQL) -> CompiledStep:
    sql = cs.sql.strip()
    header = _comment(f"custom SQL '{cs.name}' -> {cs.target_table}" + (f"\n{cs.rationale}" if cs.rationale.strip() else ""))
    return CompiledStep(f"custom_{_safe_name(cs.name)}", cs.target_table.lower(), "custom", header + "\n" + sql)
