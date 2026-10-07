"""Compile an ``ETLSpec`` into PostgreSQL.

Each ``TableMapping`` becomes one ``INSERT INTO {cdm}.<table> (...) SELECT ...`` over the
mapping's ``FROM {src}.<table> <alias> [JOIN ...] [WHERE ...]``. Field kinds compile as
described in ``docs/architecture.md`` ("The ETL spec"): column and expression values are
converted to the CDM column type tolerantly, and so are the casts and to_date / to_timestamp
calls inside them and in the WHERE (``etl.tolerant``: a value that cannot be read is NULL and
counted, instead of failing the table; partial dates follow ``etl.policies.partial_policy``);
constants keep a plain cast (they are checked when the statement is planned); ``person_id`` /
``visit_id`` resolve through the id maps, and their key expressions, like the lookup codes, are
left as written (they are join keys); ``auto_id`` continues after the largest existing key, and
concept lookups (``vocab_code``, ``concept_map``) become LEFT JOINs whose result is 0 when a
source value exists but has no mapping. The SELECT has layers, each fenced with ``OFFSET 0`` so
the planner neither merges them nor evaluates a value twice:

1. the source rows (joins, WHERE, id-map joins), with every value computed once per row and the
   code of every lookup as a column ``_k<n>``; the conversions count the values they could not
   read (``invalid``, ``imputed``, ``partial_null``, offsets) at the field's site, the WHERE at
   site ``where``;
2. the required-field policies (``etl.policies``): a NULL required concept becomes 0, a NULL
   visit / visit_detail / drug_exposure end date the start date, and a row with another NULL
   required value is dropped; each counted per field in transaction-local settings that the
   runner reads into ``value_counts`` (``CompiledStep.count_sites``);
3. the vocabulary lookups, ``auto_id`` and NULL fields, in the outer SELECT, so a code with several
   targets cannot multiply the counts;
4. for person, death and visit_occurrence, which keep one row per key (``ONE_ROW_PER``), one
   statement that keeps the best row of each key (the mapping's ``dedup_order`` with NULLs last,
   then for death a date known to the day and the latest date, then the most complete row, then
   the source row's text; a visit spans the earliest start to the latest end of its rows), skips
   the keys an earlier mapping loaded and returns its statistics.

A record's visit_occurrence_id links only to a visit of the row's person: the visit-map join also
matches the person key the visit was loaded for. The retained-row count of such a mapping counts
the visit links (linked, another person's visit, a visit that a visit mapping left out) in the
same pass.

A full compile yields ordered steps. Each step has a sort key ``slot`` = (load index of its
table, band, sequence), and the steps are stably sorted on it: the id maps come first; then,
table by table in CDM load order (``cdm.spec.LOAD_ORDER``), the table's writers (band 0: its
mappings in spec order, then its custom SQL; custom SQL of a table without a mapping runs at
that table's place too, not after everything) and the steps generated for it (band 1); the
derived observation period follows every spec writer of observation_period, and the
``cdm_source`` row comes last. After the person (and visit_occurrence) writers an id-map pruning
step retires the keys that did not become a row, so events of people or visits that a mapping
filtered out are dropped (or get a NULL visit) instead of pointing at rows that do not exist.
The id maps are never truncated, and ids are never reused or lost: they are kept across runs so
stored cohorts stay valid; a retired key moves to ``person_map_retired`` / ``visit_map_retired``
with its id and gets that id back when it returns; new keys get ids after the largest id in the
map or retired. Every person and visit_occurrence mapping whose key is not the
person_key / visit_key adds the keys of its rows in an id-map step of its own (``key_sources``),
so people and visits can come from several source tables. The generated SQL is written to
``sql/*.sql`` for humans to review, so it is laid out one field per line with the field kind as
a trailing comment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from ..cdm.spec import EVENT_DATE_FIELDS, LOAD_ORDER, CdmField, CdmTable, load_cdm_spec
from ..config import CdmSourceConfig, EtlConfig
from ..db import Schemas, qi, ql
from ..models import CustomSQL, ETLSpec, FieldMapping, KeySpec, TableMapping
from .policies import (
    END_FROM_START,
    POLICY_KINDS,
    ROW_SITE,
    WHERE_SITE,
    Policy,
    partial_policy,
    required_null_policy,
)
from .tolerant import LexError, Rewrite, kind_of, lex, rewrite, tolerant_value

StepKind = Literal["id_map", "insert", "custom", "prune", "route", "merge", "derived", "metadata"]
# Steps that run the spec's own SQL (keys, mappings, custom SQL): only their errors are sent to
# the repair agent. The other kinds are generated by the build and cannot be changed with its tools.
SPEC_KINDS = frozenset({"id_map", "insert", "custom"})

# Sequence numbers within a table's band-0 slot: custom SQL of a mapped table runs after its
# mappings; a step the build derives for a table runs after every spec writer of that table.
_CUSTOM_AFTER_MAPPINGS = 10_000
_AFTER_SPEC_WRITERS = 20_000

# Kinds whose ``expression`` must be given.
_NEEDS_EXPRESSION = {"column", "expression", "constant", "concept_map", "vocab_code", "person_id", "visit_id"}
# Kinds whose value layer 1 of the INSERT computes from the source row (the others are computed
# in layer 3: lookups, generated ids, NULL).
_LAYER_1_KINDS = frozenset({"column", "expression", "constant", "person_id", "visit_id"})
# Aliases of the layers inside the INSERT: the rows the outer SELECT reads (layer 2, or layer 1
# when no required-field policy applies), and layer 1 inside layer 2. The user's aliases are not
# visible there, so they cannot clash.
_ROWS = "s"
_VALUES = "v"
# The comment that opens layer 1, where the spec's expressions, joins and WHERE are: the repair
# prompt shows a long INSERT from there (quality.repair).
LAYER_1_MARKER = "-- 1) the source rows"
# Keys that other tables link to through an id map: (table, field) -> the kind that fills them.
ID_MAP_KEYS = {("person", "person_id"): "person_id", ("visit_occurrence", "visit_occurrence_id"): "visit_id"}
# Tables that keep one row per key, and the fields of the key (CDM v5.4: the person and visit
# primary keys; THEMIS: at most one death record per person). Several rows of one key, within a
# mapping or across mappings, load once. visit_occurrence keeps one row per visit and person: a
# visit key that two people share still gives two rows, so the fatal isPrimaryKey keeps failing.
ONE_ROW_PER: dict[str, tuple[str, ...]] = {
    "person": ("person_id",),
    "death": ("person_id",),
    "visit_occurrence": ("visit_occurrence_id", "person_id"),
}
# The fields the rows of one entity (a person, a death, a visit: the first key field) must agree
# on; an entity whose rows disagree is a conflicting key (etlKeyConflict). A concept 0 is unknown,
# not a disagreement.
IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "person": ("gender_concept_id", "year_of_birth"),
    "death": ("death_date",),
    "visit_occurrence": ("person_id",),
}
# How the rows of one key are ranked after the mapping's dedup_order (and ``EXACT_FIRST``), before
# completeness (THEMIS: the latest death date).
DEFAULT_PREFERENCE: dict[str, tuple[str, ...]] = {"death": ("death_date DESC NULLS LAST",)}
# The field whose value read as written ranks a row first, after the dedup_order: a death date
# completed from a year or month (THEMIS: 2022 -> 2022-12-31) must not win over the date known to
# the day of the same death.
EXACT_FIRST: dict[str, str] = {"death": "death_date"}
# visit_occurrence: a visit loaded from several rows (ward stays, diagnoses) spans all of them, from
# the earliest start to the latest end; where the patient came from is as in the row with the
# earliest start, where they went as in the row with the latest end.
VISIT_ROLLUP = {"visit_start_date": "MIN", "visit_start_datetime": "MIN", "visit_end_date": "MAX",
                "visit_end_datetime": "MAX", "admitted_from_concept_id": "FIRST", "admitted_from_source_value": "FIRST",
                "discharged_to_concept_id": "LAST", "discharged_to_source_value": "LAST"}
# The step label of the id-map steps of key sources besides the spec's person_key / visit_key.
_ID_MAP_LABELS = {"person": "person_map", "visit_occurrence": "visit_map"}
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
    # COUNT(*) of the source rows after the mapping's own joins and WHERE
    source_count_sql: str | None = None
    # ... that also survive the INNER person/visit id-map joins, without the vocabulary lookup
    # joins (which can fan out 1-to-many): source - retained = records dropped
    retained_count_sql: str | None = None
    # skip this step when a step loading this CDM table, or the step with this label (its name
    # without the number), failed or was skipped. A skip is not an error, so a blocking step must
    # never require a non-blocking one: the build would pass without it (tests check this for
    # generated steps). Id-map pruning requires nothing: retiring a key is undone when it returns
    requires: str | None = None
    # the spec objects (TableMapping, KeySpec, CustomSQL) whose SQL this step executes, so the
    # runner can re-check them before running anything
    origins: tuple[TableMapping | KeySpec | CustomSQL, ...] = ()
    error: str | None = None  # set instead of raising when compile_spec(strict=False)
    # False: an error is reported (etlStepError, convention) but does not fail the build
    blocking: bool = True
    # what the runner reads from the last statement: its row count; one row of statistics (its
    # ``inserted`` column is the row count); or one row per destination of moved records
    returns: Literal["rowcount", "row", "rows"] = "rowcount"
    # value-counter sites (field names, 'where', 'row') the runner reads after the step
    count_sites: tuple[str, ...] = ()
    # the statements of ``sql`` (rendered by compile_spec), for EXPLAIN; custom SQL is not split.
    # A generator whose SQL has several statements must list them here: otherwise compile_spec
    # takes the whole SQL as the one statement
    statements: tuple[str, ...] = ()
    # sort key (load index of the table, band, sequence); band 0: the table's writers, band 1: the
    # steps that follow them, band 2: the route block
    slot: tuple[int, int, int] = (0, 0, 0)

    @property
    def label(self) -> str:
        """The step name without its number (e.g. ``person_2``)."""
        return step_label(self.name)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _p(expr: str) -> str:
    """Parenthesise an expression; a trailing ``--`` comment must not swallow what follows."""
    expr = expr.strip()
    return f"({expr}\n)" if "--" in expr else f"({expr})"


def _norm(expr: str) -> str:
    return " ".join(expr.split())


def _is_datetime(f: CdmField) -> bool:
    return f.datatype.strip().lower() in ("datetime", "timestamp")


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


def typed_null(datatype: str) -> str:
    """NULL of the SQL type of a CDM field datatype. A bare NULL is text in a derived table or
    CTE, so a NULL field of a table that keeps one row per key (whose rows pass through one)
    could not be compared or inserted."""
    typed = cast_to("NULL", datatype)
    return "NULL::text" if typed.startswith("LEFT(") else typed


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
    """The joins generated for one INSERT, with alias allocation that avoids user aliases: the
    id-map joins of layer 1, and the vocabulary lookups of layer 3 with the layer-1 columns
    (``_k1``, ...) that carry their source expressions."""

    def __init__(self, taken: set[str]):
        self.taken = set(taken)
        self.counters: dict[str, int] = {}
        self.id_joins: list[str] = []  # person/visit id-map joins (layer 1)
        self.id_lines: list[str] = []  # the INNER ones: they drop rows (see retained_count_sql)
        self.lines: list[str] = []  # lookup joins (layer 3)
        self.keys: dict[str, str] = {}  # normalised lookup expression -> its layer-1 column
        self.key_exprs: list[tuple[str, str]] = []  # (layer-1 column, expression)
        self.vocab: dict[tuple[str, str], dict[str, str | None]] = {}
        self.stcm: dict[tuple[str, str], str] = {}
        self._vocab_line_index: dict[tuple[str, str], int] = {}
        self.sites: set[str] = set()  # counter sites of the tolerant conversions (layer 1)
        self.person_alias: str | None = None  # the first person-map join: the row's person
        self.visit_link: str | None = None  # the visit key of the row's (same-person) visit_occurrence_id

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

    def key(self, expr: str) -> str:
        """The source expression of a lookup, as a column of the rows layer 3 reads: it is evaluated
        once per source row in layer 1, below the lookups that can fan rows out."""
        k = _norm(expr)
        name = self.keys.get(k)
        if name is None:
            name = f"_k{len(self.keys) + 1}"
            self.keys[k] = name
            self.key_exprs.append((name, expr))
        return f"{_ROWS}.{name}"

    def person(self, expr: str) -> str:
        a = self.alias("pm")
        line = f"INNER JOIN {{etl}}.person_map {a} ON {a}.source_value = {_p(expr)}::text"
        self.id_joins.append(line)
        self.id_lines.append(line)
        if self.person_alias is None:
            self.person_alias = a
        return f"{a}.person_id"

    def visit(self, expr: str, inner: bool, same_person: bool = False) -> str:
        """Join the visit map. With ``same_person`` (and a person join made before) the visit must
        be one of the row's person: ``visit_map.person_source_value`` is the person key of the
        loaded visit (recorded by the visit pruning step), so a key that names another person's
        visit gives no visit."""
        a = self.alias("vm")
        jt = "INNER" if inner else "LEFT"
        cond = f"{a}.source_value = {_p(expr)}::text"
        if same_person and self.person_alias is not None:
            cond += f" AND {a}.person_source_value = {self.person_alias}.source_value"
        line = f"{jt} JOIN {{etl}}.visit_map {a} ON {cond}"
        self.id_joins.append(line)
        if inner:
            self.id_lines.append(line)  # a LEFT join on the map's unique key neither drops nor adds rows
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
                f"LEFT JOIN {{vocab}}.concept {sc} ON {sc}.concept_code = ({self.key(expr)})::text"
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
                f"LEFT JOIN {{etl}}.source_to_concept_map {a} ON {a}.source_code = ({self.key(expr)})::text"
                f" AND {a}.source_vocabulary_id = {ql(vocabulary)} AND {a}.invalid_reason IS NULL"
            )
        return f"{a}.source_concept_id" if source_role else f"{a}.target_concept_id"


def _refuse_bad_templates(rw: Rewrite, where: str) -> Rewrite:
    """``rw``, unless it holds a to_date / to_timestamp template that can never be right."""
    if rw.bad_templates:
        raise CompileError(f"{where}: {rw.bad_templates[0][1]}")
    return rw


def _tolerant_field(expr: str, f: CdmField, table: CdmTable, joins: _Joins, etl: EtlConfig | None) -> str:
    """A column or expression value converted tolerantly (``etl.tolerant``), counted at the field."""
    rw = _refuse_bad_templates(
        tolerant_value(expr, f.datatype, f.name, partial_policy(table.name, f.name, etl)), f"{table.name}.{f.name}")
    joins.sites.update(rw.sites)
    return rw.sql if kind_of(f.datatype) else cast_to(rw.sql, f.datatype)


def where_sql(tm: TableMapping, etl: EtlConfig | None = None, *, counted: bool = True) -> str | None:
    """The mapping's WHERE with its conversions tolerant: counted at site ``where`` (in the
    INSERT), or uncounted (in the count queries, which stay parallel). Placeholders unrendered;
    raises ``CompileError`` for a template that can never be right."""
    if not tm.where or not tm.where.strip():
        return None
    rw = rewrite(tm.where, WHERE_SITE, partial_policy(tm.target_table.lower(), WHERE_SITE, etl), counted=counted)
    return _refuse_bad_templates(rw, f"{tm.target_table} where").sql


def _field_value(
    fm: FieldMapping, f: CdmField, table: CdmTable, tm: TableMapping, spec: ETLSpec, joins: _Joins,
    etl: EtlConfig | None = None,
) -> str:
    kind = fm.kind
    expr = (fm.expression or "").strip()
    where = f"{table.name}.{f.name}"
    if kind in _NEEDS_EXPRESSION and not expr:
        raise CompileError(f"{where}: kind '{kind}' needs an expression")
    if kind == "null":
        return typed_null(f.datatype)
    if kind in ("column", "expression"):
        return _tolerant_field(expr, f, table, joins, etl)
    if kind == "constant":
        return cast_to(expr, f.datatype)
    if kind == "auto_id":
        if not f.is_pk:
            pk = table.primary_key or "none"
            raise CompileError(f"{where}: auto_id is only allowed on the table's primary key ({pk})")
        if (table.name, f.name) in ID_MAP_KEYS:
            raise CompileError(
                f"{where}: auto_id is not allowed here; use kind {ID_MAP_KEYS[(table.name, f.name)]} so the id "
                "comes from the id map that other tables link through (auto_id would follow row order and "
                "attach events to the wrong rows)"
            )
        value = f"(SELECT COALESCE(MAX({qi(f.name)}), 0) FROM {{cdm}}.{qi(table.name)}) + ROW_NUMBER() OVER ()"
        return cast_to(value, f.datatype)
    if kind == "person_id":
        return cast_to(joins.person(expr), f.datatype)
    if kind == "visit_id":
        if spec.visit_key is None:
            raise CompileError(f"{where}: kind visit_id needs spec.visit_key (the visit id map)")
        # a record's visit must be a visit of the same person; the visit table's own key and its
        # preceding visit are resolved before the visits' persons are recorded
        same_person = table.name != "visit_occurrence" and f.name != "preceding_visit_occurrence_id"
        value = joins.visit(expr, inner=f.required, same_person=same_person)
        if same_person and f.name == "visit_occurrence_id" and not f.required and joins.person_alias is not None:
            joins.visit_link = expr  # the links are counted with the retained rows (etlVisitLinkage)
        return cast_to(value, f.datatype)
    if kind in ("vocab_code", "concept_map"):
        if not (fm.vocabulary or "").strip():
            raise CompileError(f"{where}: kind '{kind}' needs a vocabulary")
        vocab = fm.vocabulary.strip()  # type: ignore[union-attr]
        if kind == "vocab_code":
            value = joins.vocab_code(expr, vocab, standard=fm.concept_role == "standard")
        else:
            value = joins.concept_map(expr, vocab, source_role=fm.concept_role == "source")
        when_null = "0" if f.required else "NULL"
        return cast_to(f"CASE WHEN ({joins.key(expr)}) IS NULL THEN {when_null} ELSE COALESCE({value}, 0) END",
                       f.datatype)
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


@dataclass
class _Compiled:
    """One table mapping, compiled (placeholders kept)."""

    insert: str  # the INSERT ... SELECT
    retained: str  # the retained-row COUNT (see retained_count_sql)
    count_sites: tuple[str, ...] = ()  # the value-counter sites the INSERT can set
    required: dict[str, Policy] = field(default_factory=dict)  # required field -> the NULL policy applied
    returns: Literal["rowcount", "row", "rows"] = "rowcount"
    id_lines: list[str] = field(default_factory=list)  # the INNER id-map joins (they drop rows)
    # the other mappings whose SQL the retained count runs (the visit sources of the link classes)
    count_origins: tuple[TableMapping, ...] = ()


def _compile_insert(tm: TableMapping, spec: ETLSpec, etl: EtlConfig | None = None) -> str:
    """Unrendered INSERT (placeholders kept)."""
    return _compile(tm, spec, etl).insert


def _select_list(items: list[tuple[str, str]]) -> list[str]:
    """``SELECT`` lines: one ``value AS name`` (or a bare column) per item, with an optional trailing comment."""
    lines = []
    for i, (item, note) in enumerate(items):
        sep = "," if i < len(items) - 1 else ""
        lines.append(f"    {item}{sep}" + (f"  -- {note}" if note else ""))
    return lines


def _required_policies(table: CdmTable, values: list[tuple[CdmField, FieldMapping, str]]) -> dict[str, Policy]:
    """The NULL policy (``etl.policies``) applied to each required field whose value layer 1 computes
    from the source. 'open' (observation_period_end_date) is left as written until the build merges
    observation periods, so a NULL end still fails that load."""
    in_layer_1 = {f.name for f, fm, _ in values if fm.kind in _LAYER_1_KINDS}
    out: dict[str, Policy] = {}
    for f, fm, _ in values:
        if not f.required or fm.kind not in POLICY_KINDS:
            continue
        policy = required_null_policy(table.name, f.name)
        if policy == "start" and END_FROM_START[(table.name, f.name)] not in in_layer_1:
            policy = "drop"  # no start date to take
        if policy in ("zero", "start", "drop"):
            out[f.name] = policy
    return out


def _source_rows(
    tm: TableMapping, joins: _Joins, layer_1: list[tuple[CdmField, FieldMapping, str]], where: str | None,
    extra: list[tuple[str, str, str]] | None = None,
) -> list[str]:
    """Layer 1: the source rows after the mapping's joins and WHERE (``where``, counted) and the
    id-map joins, with every value computed from the source once, the source expression of every
    lookup (``_k1``, ...), and the ``extra`` columns (name, value, note)."""
    items: list[tuple[str, str]] = [(f"{v} AS {qi(f.name)}", fm.kind) for f, fm, v in layer_1]
    items += [(f"{_p(expr)} AS {k}", "lookup source") for k, expr in joins.key_exprs]
    items += [(f"{value} AS {name}", note) for name, value, note in extra or []]
    rows = [
        f"{LAYER_1_MARKER}: every value computed once per row, below the lookups that can fan out",
        "SELECT", *_select_list(items), *_from_clause(tm), *joins.id_joins,
    ]
    if where:
        rows.append(f"WHERE {_p(where)}")
    return rows + ["OFFSET 0"]  # a fence: the planner neither merges the layers nor evaluates a value twice


def _required_values(
    table: CdmTable, layer_1: list[tuple[CdmField, FieldMapping, str]], joins: _Joins,
    required: dict[str, Policy], rows: list[str], extra: list[str] | None = None,
) -> list[str]:
    """Layer 2 over the layer-1 ``rows``: required values that are NULL get the CDM convention of
    their field (``required``), and each one is counted. Rows with a NULL drop-policy field are
    dropped; the counters are called only on the NULL branches. The ``extra`` columns of layer 1
    are passed on."""

    def col(name: str) -> str:
        return f"{_VALUES}.{qi(name)}"

    items: list[tuple[str, str]] = []
    for f, _, _ in layer_1:
        name, policy = f.name, required.get(f.name)
        if policy == "zero":
            items.append((f"CASE WHEN {col(name)} IS NULL THEN {{etl}}.omop_default_int({ql(name)}, 0)"
                          f" ELSE {col(name)} END AS {qi(name)}", "NULL -> 0 (counted)"))
        elif policy == "start":
            start = END_FROM_START[(table.name, name)]
            derived = "omop_derived_ts" if _is_datetime(f) else "omop_derived_date"
            items.append((f"COALESCE({col(name)}, {{etl}}.{derived}({ql(name)}, {col(start)})) AS {qi(name)}",
                          f"NULL -> {start} (counted)"))
        else:
            items.append((col(name), "NULL: the row is dropped (counted)" if policy == "drop" else ""))
    items += [(f"{_VALUES}.{k}", "") for k, _ in joins.key_exprs]
    items += [(f"{_VALUES}.{name}", "") for name in extra or []]
    out = [
        "-- 2) required values that are NULL: the CDM convention, counted per field",
        "SELECT", *_select_list(items), "FROM (", *_indent(rows), f") {_VALUES}",
    ]
    drops = [name for name, p in required.items() if p == "drop"]
    if drops:
        present = " AND ".join(f"{col(n)} IS NOT NULL" for n in drops)
        names = ", ".join(ql(n) for n in drops)
        nulls = ", ".join(f"{col(n)} IS NULL" for n in drops)
        out += [f"WHERE CASE WHEN {present} THEN true",
                f"           ELSE {{etl}}.omop_drop(ARRAY[{names}], ARRAY[{nulls}]) END"]
    return out + ["OFFSET 0"]


def _compile(tm: TableMapping, spec: ETLSpec, etl: EtlConfig | None = None) -> _Compiled:
    """One table mapping as an INSERT whose SELECT has three layers (see "How a mapping compiles"
    in docs/architecture.md), and its retained-row COUNT:

    1. the source rows (``_source_rows``), fenced so that each value is computed once per row, with
       the tolerant conversions (``etl`` gives the partial-date rule);
    2. the required-field policies (``_required_values``), only when one applies;
    3. the outer SELECT: vocabulary lookups (on the layer-1 ``_k`` columns), ``auto_id`` and NULL
       fields, and the other values as computed below;
    4. for a table that keeps one row per key (``ONE_ROW_PER``), the rows of layer 3 become the
       candidates of one statement that keeps one row per key (``_one_row_per_key``) and returns
       its statistics.

    person_id fields are resolved first, so a visit link can require the row's person.
    """
    table = _check_mapping(tm)
    by_name = {fm.target_field: fm for fm in tm.fields}
    ordered = [(f, by_name[f.name]) for f in table.fields if f.name in by_name]
    if not ordered:
        raise CompileError(f"{table.name}: the mapping has no fields")
    joins = _Joins({tm.source.alias.lower()} | {j.alias.lower() for j in tm.joins})
    first = sorted(ordered, key=lambda it: (it[1].kind != "person_id", it[0].name != "person_id"))
    computed = {f.name: _field_value(fm, f, table, tm, spec, joins, etl) for f, fm in first}
    values = [(f, fm, computed[f.name]) for f, fm in ordered]
    required = _required_policies(table, values)
    layer_1 = [(f, fm, v) for f, fm, v in values if fm.kind in _LAYER_1_KINDS]
    where = where_sql(tm, etl)
    keys = _one_row_keys(table, values)
    extra = _candidate_columns(tm, table, values, keys, etl) if keys else []
    rows = _source_rows(tm, joins, layer_1, where, extra)
    if required:
        rows = _required_values(table, layer_1, joins, required, rows, [name for name, _, _ in extra])
    items: list[tuple[str, str]] = [
        (f"{_ROWS}.{qi(f.name)}", "") if fm.kind in _LAYER_1_KINDS else (f"{v} AS {qi(f.name)}", fm.kind)
        for f, fm, v in values
    ]
    items += [(f"{_ROWS}.{name}", "") for name, _, _ in extra]
    select = ["SELECT", *_select_list(items), "FROM (", *_indent(rows), f") {_ROWS}", *joins.lines]

    header = f"{table.name} <- {tm.source.table} ({tm.source.alias})"
    if tm.joins:
        header += " + " + ", ".join(f"{j.table} ({j.alias})" for j in tm.joins)
    out = [_comment(header)]
    if tm.rationale.strip():
        out.append(_comment(tm.rationale))
    if keys:
        out += _one_row_per_key(table, values, select, keys, {n for n, _, _ in extra})
    else:
        out.append(f"INSERT INTO {{cdm}}.{qi(table.name)} (")
        out += [f"    {qi(f.name)}{',' if i < len(values) - 1 else ''}" for i, (f, _, _) in enumerate(values)]
        out += [")", *select]
    # the counter sites in field order (required-field policies, tolerant conversions), then the
    # WHERE's and the dropped rows'
    sites = tuple(f.name for f, _, _ in values if f.name in required or f.name in joins.sites)
    if where is not None and where != tm.where:
        sites += (WHERE_SITE,)
    if "drop" in required.values():
        sites += (ROW_SITE,)
    retained, count_origins = _retained_count(tm, spec, joins, etl)
    return _Compiled("\n".join(out) + ";", retained, count_sites=sites, required=required,
                     returns="row" if keys else "rowcount", id_lines=joins.id_lines, count_origins=count_origins)


def _visit_source_keys(spec: ETLSpec) -> tuple[list[str], tuple[TableMapping, ...]]:
    """The distinct visit keys of the rows of every visit_occurrence mapping before its WHERE (its
    joins as LEFT joins, so they do not filter either), as one query (lines) with a column
    ``source_value``; with custom SQL into visit_occurrence also the visit_key's keys, which that
    SQL resolves. ``[]`` when nothing writes visits. A record whose visit key is one of these but
    has no visit names a visit the visit mappings deliberately left out (filtered out, a person
    not loaded, a required value missing), not a key in another form. Also the visit mappings
    whose SQL the query runs (the runner re-checks it)."""
    parts: list[str] = []
    used: list[TableMapping] = []
    for tm in spec.table_mappings:
        kf = _key_field(tm, "visit_occurrence_id", "visit_id") if tm.target_table.lower() == "visit_occurrence" else None
        if kf is None:
            continue
        used.append(tm)
        key = _p(kf.expression or "")
        lines = [f"SELECT {key}::text AS source_value", f"FROM {source_ref(tm.source.table)} {qi(tm.source.alias)}"]
        lines += [f"LEFT JOIN {source_ref(j.table)} {qi(j.alias)} ON {j.on.strip()}" for j in tm.joins]
        parts.append("\n".join(lines + [f"WHERE {key} IS NOT NULL"]))
    vk = spec.visit_key
    if vk is not None and any(cs.target_table.lower() == "visit_occurrence" for cs in spec.custom_sql):
        parts.append(f"SELECT {_p(vk.expression)}::text AS source_value\nFROM {source_ref(vk.table)}\n"
                     f"WHERE {_p(vk.expression)} IS NOT NULL")
    if not parts:
        return [], ()
    return (["SELECT DISTINCT k.source_value", "FROM (", *_indent("\nUNION ALL\n".join(parts).splitlines()), ") k"],
            tuple(used))


def _retained_count(
    tm: TableMapping, spec: ETLSpec, joins: _Joins, etl: EtlConfig | None
) -> tuple[str, tuple[TableMapping, ...]]:
    """The retained-row COUNT (see ``retained_count_sql``) and, when the rows link to a visit of
    their person, the link classes of the rows (one pass; the visit map and the visit sources'
    keys are joined on unique keys, so they do not change the count): ``with_key`` (a visit key),
    ``linked`` (a loaded visit of the row's person), ``other_person`` (a loaded visit of another
    person), ``filtered`` (no loaded visit, but a key of a visit mapping's source rows, which that
    mapping left out: ``_visit_source_keys``). The rest of ``with_key`` names no visit of any
    visit source (``etlVisitLinkLoss``): a key in another form than the visit mappings write. The
    visit sources are read as they are now, so neither a visit_key that no visit mapping writes
    nor keys retired by builds of an older spec count as filtered. Also the visit mappings whose
    SQL the count runs."""
    if joins.visit_link is None or joins.person_alias is None:
        return _source_count(tm, joins.id_lines, etl), ()
    key, pm = _p(joins.visit_link), joins.person_alias
    vx = joins.alias("vm")
    lines = [f"LEFT JOIN {{etl}}.visit_map {vx} ON {vx}.source_value = {key}::text"]
    keys, used = _visit_source_keys(spec)
    filtered = "0"
    if keys:
        vs = joins.alias("vs")
        lines += ["LEFT JOIN (", *_indent(keys), f") {vs} ON {vs}.source_value = {key}::text"]
        filtered = f"COUNT(*) FILTER (WHERE {vx}.source_value IS NULL AND {vs}.source_value IS NOT NULL)"
    what = ",\n       ".join([
        "COUNT(*) AS n",
        f"COUNT(*) FILTER (WHERE {key} IS NOT NULL) AS with_key",
        f"COUNT(*) FILTER (WHERE {vx}.person_source_value = {pm}.source_value) AS linked",
        f"COUNT(*) FILTER (WHERE {vx}.person_source_value <> {pm}.source_value) AS other_person",
        f"{filtered} AS filtered",
    ])
    return _source_count(tm, joins.id_lines + lines, etl, what=what), used


# --------------------------------------------------------------------------------------
# one row per key (person, death, visit_occurrence)
# --------------------------------------------------------------------------------------


def _one_row_keys(table: CdmTable, values: list[tuple[CdmField, FieldMapping, str]]) -> tuple[str, ...]:
    """The key fields of a table that keeps one row per key, when the mapping fills all of them
    (else ``()``: a plain INSERT, which the validator reports)."""
    keys = ONE_ROW_PER.get(table.name, ())
    kinds = {f.name: fm.kind for f, fm, _ in values}
    if not keys or any(kinds.get(k) in (None, "null") for k in keys):
        return ()
    return keys


def _nulls_last(order: str) -> str:
    """An ORDER BY list with ``NULLS LAST`` added to every item that does not say where NULLs go.
    PostgreSQL sorts NULLs first under DESC, so a row whose ranking value is missing, or could not
    be read, would win over the rows that have one."""
    try:
        toks = lex(order)
    except LexError:
        return order
    ends: list[int] = []  # where each item that needs it gets NULLS LAST: after its last token
    depth, last, has_nulls = 0, None, False
    for t in [*toks, None]:
        if t is None or (depth == 0 and t.kind == "p" and t.text == ","):
            if last is not None and not has_nulls:
                ends.append(last)
            last, has_nulls = None, False
            continue
        if t.kind in ("ws", "comment"):
            continue
        if t.kind == "p" and t.text in "([":
            depth += 1
        elif t.kind == "p" and t.text in ")]":
            depth -= 1
        elif depth == 0 and t.kind == "id" and t.text.lower() == "nulls":
            has_nulls = True
        last = t.end
    for at in reversed(ends):
        order = order[:at] + " NULLS LAST" + order[at:]
    return order


def _dedup_order_sql(tm: TableMapping, etl: EtlConfig | None) -> str | None:
    """The mapping's dedup_order with its conversions tolerant and uncounted (a value it cannot read
    ranks as NULL instead of failing the table) and NULLs last in every item (``_nulls_last``).
    Raises ``CompileError`` for a template that can never be right."""
    if not (tm.dedup_order or "").strip():
        return None
    rw = rewrite(tm.dedup_order, "dedup_order", partial_policy(tm.target_table.lower(), WHERE_SITE, etl),
                 counted=False)
    return _nulls_last(_refuse_bad_templates(rw, f"{tm.target_table} dedup_order").sql.strip())


def _candidate_columns(
    tm: TableMapping, table: CdmTable, values: list[tuple[CdmField, FieldMapping, str]], keys: tuple[str, ...],
    etl: EtlConfig | None,
) -> list[tuple[str, str, str]]:
    """(name, layer-1 value, note) of the columns that rank the rows of one key: the source row
    (``_src_row``, to tell rows from vocabulary fan-out), the rank by the mapping's dedup_order
    (``_pref``, computed over the source aliases), whether the ``EXACT_FIRST`` field was read as
    written (``_exact``: its value, read uncounted with partial dates rejected, is not NULL) and
    the source row's text (``_src_text``, the neutral last tie-break; never stored)."""
    computed = {f.name: v for f, _, v in values}
    out = [("_src_row", "ROW_NUMBER() OVER ()", "one number per source row")]
    order = _dedup_order_sql(tm, etl)
    if order is not None:
        tail = "\n" if "--" in order else ""  # a trailing comment must not swallow the parenthesis
        partition = ", ".join(computed[k] for k in keys)
        out.append(("_pref", f"RANK() OVER (PARTITION BY {partition} ORDER BY {order}{tail})", "dedup_order"))
    exact = EXACT_FIRST.get(table.name)
    fm = next((fm for f, fm, _ in values if f.name == exact), None)
    if fm is not None and fm.kind in ("column", "expression") and (fm.expression or "").strip():
        f = table.field(exact)  # type: ignore[arg-type]
        rw = tolerant_value(fm.expression or "", f.datatype, f.name, "reject", counted=False)  # type: ignore[union-attr]
        out.append(("_exact", f"{_p(rw.sql)} IS NOT NULL", f"{exact} known as written (not a completed partial date)"))
    out.append(("_src_text", f"({qi(tm.source.alias)}.*)::text", "the last tie-break, never stored"))
    return out


def _is_concept_field(f: CdmField) -> bool:
    return f.is_concept_id or f.fk_table == "concept"


def _one_row_per_key(
    table: CdmTable, values: list[tuple[CdmField, FieldMapping, str]], candidates: list[str],
    keys: tuple[str, ...], extra: set[str],
) -> list[str]:
    """Layer 4: one INSERT that keeps one row per key of the ``candidates`` (the layer-3 SELECT) and
    returns its statistics in one row. Among the rows of one key the mapping's dedup_order wins
    (``_pref`` in ``extra``), then a value known as written (``_exact``, ``EXACT_FIRST``), then the
    table's ``DEFAULT_PREFERENCE``, then the most complete row (a concept 0 counts as missing), then
    the source row's text, then the values: the choice is deterministic and prefers no sex, concept
    or value (text compares byte-wise, COLLATE "C", so it does not depend on the server's
    collation). visit_occurrence takes the earliest start and the latest end of the rows of a
    visit, where the patient came from as in its first row and where they went as in its last
    (``VISIT_ROLLUP``). Rows whose drop-policy fields are NULL were dropped in layer 2, so they are
    never candidates. A key an earlier mapping of the table loaded is skipped; the anti-join runs
    on the chosen rows, outside DISTINCT ON (a planning fence).

    Statistics: ``inserted``; ``candidate_rows`` (after vocabulary fan-out); ``source_rows_kept``
    (the source rows that are candidates); ``keys``; ``duplicate_rows`` (source rows not loaded
    because another row of their key won); ``already_loaded_keys``; ``conflict_keys`` (entities
    whose rows disagree on ``IDENTITY_FIELDS``, among themselves or with the row an earlier mapping
    loaded; for a visit, a visit an earlier mapping loaded for another person); for person and
    death ``undecided_keys`` (entities this mapping loads whose best-ranked rows still disagree:
    the row loaded is arbitrary; all their rows rank best without a dedup_order); ``fan_out_keys``
    (keys with more candidates than source rows: a code with several concepts); for
    visit_occurrence ``gap_keys`` (visits whose rows are separate stays: a row starts more than a
    day after the earlier rows end, so the visit key is coarser than one visit); ``dedup_order``
    (whether the mapping ranks its rows)."""
    t = qi(table.name)
    names = [f.name for f, _, _ in values]
    cols = [qi(n) for n in names]
    key_cols = [f"c.{qi(k)}" for k in keys]
    partition = ", ".join(key_cols)
    ranked = "_pref" in extra
    src_text = 'c._src_text COLLATE "C"'
    rollup = {n: how for n, how in VISIT_ROLLUP.items() if n in names} if table.name == "visit_occurrence" else {}

    def present(*fields: str) -> list[str]:
        return [f"c.{qi(n)}" for n in fields if n in names]

    windows: dict[str, str] = {}
    if any(how in ("MIN", "MAX") for how in rollup.values()):
        windows["w"] = f"PARTITION BY {partition}"
    if "FIRST" in rollup.values():  # the row with the earliest start
        windows["w_first"] = (f"PARTITION BY {partition} ORDER BY "
                              + ", ".join(present("visit_start_date", "visit_start_datetime") + [src_text]))
    if "LAST" in rollup.values():  # the row with the latest end
        windows["w_last"] = (f"PARTITION BY {partition} ORDER BY "
                             + ", ".join([f"{c} DESC NULLS LAST" for c in present("visit_end_date", "visit_end_datetime")]
                                         + [src_text]))
    notes = {"MIN": "the visit spans all its rows", "MAX": "the visit spans all its rows",
             "FIRST": "as in the visit's first row", "LAST": "as in the visit's last row"}
    picked_items: list[tuple[str, str]] = []
    for n in names:
        how = rollup.get(n)
        if how in ("MIN", "MAX"):
            picked_items.append((f"{how}(c.{qi(n)}) OVER w AS {qi(n)}", notes[how]))
        elif how in ("FIRST", "LAST"):
            picked_items.append((f"FIRST_VALUE(c.{qi(n)}) OVER w_{how.lower()} AS {qi(n)}", notes[how]))
        else:
            picked_items.append((f"c.{qi(n)}", ""))
    order = list(key_cols) + (["c._pref"] if ranked else []) + (["c._exact DESC"] if "_exact" in extra else [])
    for pref in DEFAULT_PREFERENCE.get(table.name, ()):
        col, _, direction = pref.partition(" ")
        if col in names:
            order.append(f"c.{qi(col)} {direction}".rstrip())
    complete = [
        f"NULLIF(c.{qi(f.name)}, 0)" if _is_concept_field(f) else f"c.{qi(f.name)}"
        for f, _, _ in values if f.name not in keys
    ]
    if complete:
        order.append("num_nulls(" + ", ".join(complete) + ")")
    order.append(src_text)
    order += [f"c.{qi(n)}" for n in names if n not in keys]

    picked = [
        "SELECT DISTINCT ON (" + ", ".join(key_cols) + ")", *_select_list(picked_items), "FROM cand c",
        *(["WINDOW " + ",\n       ".join(f"{w} AS ({spec})" for w, spec in windows.items())] if windows else []),
        "ORDER BY " + ",\n         ".join(order),
    ]
    exists = " AND ".join(f"x.{qi(k)} = p.{qi(k)}" for k in keys)
    ins = [
        f"INSERT INTO {{cdm}}.{t} (",
        *[f"    {c}{',' if i < len(cols) - 1 else ''}" for i, c in enumerate(cols)],
        ")",
        "SELECT " + ", ".join(f"p.{c}" for c in cols),
        "FROM picked p",
        f"WHERE NOT EXISTS (SELECT 1 FROM {{cdm}}.{t} x WHERE {exists})",
        "RETURNING 1",
    ]
    gaps = rollup.get("visit_start_date") == "MIN" and rollup.get("visit_end_date") == "MAX"
    if gaps:  # a row that starts more than a day after every earlier row of its visit has ended
        rows = [
            "FROM (",
            "    SELECT c.*, c.visit_start_date > MAX(c.visit_end_date) OVER (",
            f"        PARTITION BY {partition} ORDER BY c.visit_start_date, c.visit_end_date",
            "        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) + 1 AS _gap",
            "    FROM cand c",
            ") c",
        ]
    else:
        rows = ["FROM cand c"]
    per_key = [f"SELECT {partition}, COUNT(*) AS n, COUNT(DISTINCT c._src_row) AS src_rows"
               + (", bool_or(c._gap) AS gap" if gaps else ""), *rows, f"GROUP BY {partition}"]
    conflicts, undecided = _conflicts(table, names, keys, ranked)
    stats = [
        "SELECT",
        *_select_list([
            ("(SELECT COUNT(*) FROM ins) AS inserted", ""),
            ("COALESCE(SUM(k.n), 0) AS candidate_rows", ""),
            ("COALESCE(SUM(k.src_rows), 0) AS source_rows_kept", ""),
            ("COUNT(*) AS keys", ""),
            ("COALESCE(SUM(k.src_rows), 0) - COUNT(*) AS duplicate_rows", "rows of a key that another row won"),
            ("COUNT(*) - (SELECT COUNT(*) FROM ins) AS already_loaded_keys", "loaded by an earlier mapping"),
            (f"({conflicts}) AS conflict_keys" if conflicts else "0 AS conflict_keys", "rows that disagree"),
            *([(f"({undecided}) AS undecided_keys", "the best-ranked rows disagree")] if undecided else []),
            ("COUNT(*) FILTER (WHERE k.n > k.src_rows) AS fan_out_keys", "a code with several concepts"),
            *([("COUNT(*) FILTER (WHERE k.gap) AS gap_keys", "rows that are separate stays")] if gaps else []),
            (f"{'TRUE' if ranked else 'FALSE'} AS dedup_order", ""),
        ]),
        "FROM per_key k",
    ]
    best = ["the dedup_order"] if ranked else []
    best += ["a death date known as written", "the latest death"] if table.name == "death" else []
    return [
        "-- 4) one row per " + " and ".join(keys) + ": the statement returns its statistics",
        "WITH cand AS MATERIALIZED (",
        "    -- 3) the candidate rows",
        *_indent(candidates),
        "),",
        "picked AS (",
        "    -- the best row of each key: "
        + ", then ".join(best + ["the most complete row (a concept 0 is missing)", "the source row's text"]),
        *_indent(picked),
        "),",
        "ins AS (",
        "    -- a key an earlier mapping loaded is skipped",
        *_indent(ins),
        "),",
        "per_key AS (",
        *_indent(per_key),
        ")",
        *stats,
    ]


def _conflicts(table: CdmTable, names: list[str], keys: tuple[str, ...], ranked: bool) -> tuple[str | None, str | None]:
    """Two COUNTs over the candidates (None when the mapping fills no ``IDENTITY_FIELDS``): the
    entities (first key field) whose rows disagree on the mapped identity fields, among themselves
    or with the row an earlier mapping loaded (a concept 0 or a NULL is unknown, not a
    disagreement; for a visit, a visit an earlier mapping loaded for another person: its rows are
    inserted too, so isPrimaryKey fails); and, for person and death, the entities this mapping
    loads whose best-ranked rows (by its dedup_order; all rows without one) still disagree."""
    entity = qi(keys[0])
    fields = [f for f in IDENTITY_FIELDS.get(table.name, ()) if f in names and f != keys[0]]
    if not fields:
        return None, None
    concept = {f.name for f in table.fields if _is_concept_field(f)}

    def value(alias: str, f: str) -> str:
        return f"NULLIF({alias}.{qi(f)}, 0)" if f in concept else f"{alias}.{qi(f)}"

    having = " OR ".join(f"COUNT(DISTINCT {value('c', f)}) > 1" for f in fields)
    t = f"{{cdm}}.{qi(table.name)}"
    inner = f"SELECT c.{entity} FROM cand c GROUP BY c.{entity} HAVING {having}"
    if table.name == "visit_occurrence":
        inner += f" UNION SELECT c.{entity} FROM cand c JOIN {t} x ON x.{entity} = c.{entity} AND x.person_id <> c.person_id"
        return f"SELECT COUNT(*) FROM ({inner}) e", None
    differs = " OR ".join(f"{value('x', f)} <> {value('c', f)}" for f in fields)
    inner += f" UNION SELECT c.{entity} FROM cand c JOIN {t} x ON x.{entity} = c.{entity} WHERE {differs}"
    rows, best = "cand c", ""
    if ranked:
        rows = f"(SELECT c.*, MIN(c._pref) OVER (PARTITION BY c.{entity}) AS _best FROM cand c) c"
        best = "c._pref = c._best AND "
    undecided = (f"SELECT COUNT(*) FROM (SELECT c.{entity} FROM {rows} WHERE {best}NOT EXISTS (SELECT 1 FROM {t} x "
                 f"WHERE x.{entity} = c.{entity}) GROUP BY c.{entity} HAVING {having}) e")
    return f"SELECT COUNT(*) FROM ({inner}) e", undecided


def _source_count(
    tm: TableMapping, id_joins: list[str] | None = None, etl: EtlConfig | None = None, also: list[str] | None = None,
    what: str = "COUNT(*) AS n",
) -> str:
    """``SELECT COUNT(*)`` (or ``what``) over the mapping's FROM, joins, ``id_joins`` and WHERE
    (uncounted), and the conditions ``also``."""
    lines = [f"SELECT {what}", *_from_clause(tm), *(id_joins or [])]
    where = where_sql(tm, etl, counted=False)
    conditions = ([_p(where)] if where else []) + (also or [])
    if conditions:
        lines.append("WHERE " + "\n  AND ".join(conditions))
    return "\n".join(lines)


def _uncounted_value(fm: FieldMapping, f: CdmField, table: CdmTable, etl: EtlConfig | None) -> str | None:
    """The value of a column, expression, constant or null field as the build computes it, without
    counters (None for other kinds)."""
    expr = (fm.expression or "").strip()
    if fm.kind == "null":
        return "NULL"
    if not expr:
        return None
    if fm.kind == "constant":
        return cast_to(expr, f.datatype)
    if fm.kind not in ("column", "expression"):
        return None
    rw = tolerant_value(expr, f.datatype, f.name, partial_policy(table.name, f.name, etl), counted=False)
    return rw.sql if kind_of(f.datatype) else cast_to(rw.sql, f.datatype)


@dataclass
class NullingCount:
    """The counts that tell whether an edit makes a required field (or a record's visit key) NULL
    for more rows (see ``nulling_count_sql``). Rendered SQL."""

    rows: str  # COUNT(*) AS n of the rows where the new value is NULL and the old one is not
    new_totals: str  # COUNT(*) AS n, COUNT(<new value>) AS v over the new mapping's rows
    old_totals: str | None  # the same for the old value over the old mapping's rows (None: it does not compile)
    # 'drop': a required field whose NULL drops the row; 'visit_key': the visit key of a record's
    # optional visit link, whose NULL takes the record out of the link checks
    what: Literal["drop", "visit_key"] = "drop"


def _totals(tm: TableMapping, id_lines: list[str], value: str, etl: EtlConfig | None) -> str:
    return _source_count(tm, id_lines, etl, what=f"COUNT(*) AS n, COUNT({_p(value)}) AS v")


def _key_value(fm: FieldMapping) -> str | None:
    """The key a visit_id field joins on, as written (None without an expression)."""
    return _p(fm.expression) if (fm.expression or "").strip() else None


def nulling_count_sql(
    old: TableMapping, new: TableMapping, spec: ETLSpec, schemas: Schemas, *, etl: EtlConfig | None = None,
    na_values: list[str] | tuple[str, ...] = (),
) -> dict[str, NullingCount]:
    """For an edit of a table mapping, per required field whose NULL drops the row (``etl.policies``,
    policy 'drop') and whose value changed: the count of the new mapping's source rows (joins,
    id-map joins, WHERE) where the field is NULL with the new value and was not with the old one,
    i.e. the records the edit would drop; and, for an old value that does not fit the new
    mapping's sources (a renamed alias), the NULL totals of each mapping over its own rows. The
    same for an optional visit_id field (a record's visit link): a record whose visit key becomes
    NULL leaves the link checks without a visit, so an edit that does that hides lost links; an
    old key that is blank or one of the source's ``na_values`` (compared without case) is no key,
    and may become NULL. Raises ``CompileError`` when the new mapping does not compile."""
    table = _check_mapping(new)
    c = _compile(new, spec, etl)
    try:
        old_lines: list[str] | None = _compile(old, spec, etl).id_lines
    except CompileError:
        old_lines = None
    before = {fm.target_field: fm for fm in old.fields}
    out: dict[str, NullingCount] = {}
    for fm in new.fields:
        f = table.field(fm.target_field)
        was = before.get(fm.target_field)
        visit_key = f is not None and was is not None and was.kind == "visit_id" and not f.required
        if f is None or (c.required.get(f.name) != "drop" and not visit_key):
            continue
        if was is not None and (was.kind, _norm(was.expression or "")) == (fm.kind, _norm(fm.expression or "")):
            continue
        new_value = _key_value(fm) if fm.kind == "visit_id" else _uncounted_value(fm, f, table, etl)
        if visit_key:
            old_value = _key_value(was)  # type: ignore[arg-type]
        else:
            old_value = _uncounted_value(was, f, table, etl) if was is not None else "NULL"
        if new_value is None or old_value is None:
            continue
        also = [f"{_p(new_value)} IS NULL", f"{_p(old_value)} IS NOT NULL"]
        if visit_key:
            markers = ", ".join(ql(v.strip().upper()) for v in ("", *na_values))
            also.append(f"upper(btrim({_p(old_value)}::text)) NOT IN ({markers})")
        rows = _source_count(new, c.id_lines, etl, also=also)
        out[f.name] = NullingCount(
            schemas.render(rows), schemas.render(_totals(new, c.id_lines, new_value, etl)),
            schemas.render(_totals(old, old_lines, old_value, etl)) if old_lines is not None else None,
            what="visit_key" if visit_key else "drop",
        )
    return out


def compile_table(tm: TableMapping, spec: ETLSpec, schemas: Schemas, *, etl: EtlConfig | None = None) -> str:
    """One rendered ``INSERT ... SELECT`` for a table mapping. Raises ``CompileError``."""
    return schemas.render(_compile_insert(tm, spec, etl))


def source_count_sql(tm: TableMapping, schemas: Schemas, *, etl: EtlConfig | None = None) -> str:
    """``SELECT COUNT(*)`` of the mapping's source rows after user joins/WHERE, before id-map filtering."""
    return schemas.render(_source_count(tm, etl=etl))


def retained_count_sql(tm: TableMapping, spec: ETLSpec, schemas: Schemas, *, etl: EtlConfig | None = None) -> str:
    """``SELECT COUNT(*)`` of the source rows that also survive the INNER person/visit id-map joins.

    The vocabulary lookups are left out: a code with several 'Maps to' targets (or several
    source_to_concept_map rows) inserts several CDM rows, and that fan-out must not hide rows
    that were dropped. Raises ``CompileError``.
    """
    return schemas.render(_compile(tm, spec, etl).retained)


# --------------------------------------------------------------------------------------
# id maps, derived tables, metadata
# --------------------------------------------------------------------------------------


_NUMBERED = re.compile(r"^\d+_(.+)$")


def step_label(name: str) -> str:
    """``05_person_2`` -> ``person_2``: the name of a step without the number compile_spec gives it."""
    m = _NUMBERED.match(name)
    return m.group(1) if m else name


def _indent(lines: list[str], n: int = 4) -> list[str]:
    return [" " * n + line for line in lines]


def _id_map_statements(id_map: str, id_col: str, keys: list[str], header: str) -> list[str]:
    """The two statements of an id-map step over ``keys`` (a query with one row per key): retired
    keys that are back get their old id again; then the new keys are numbered in key order after
    the largest id ever given out (in the map or retired), so an id is never reused. The step's
    row count is that of the last statement: the new keys."""
    a = "pm" if id_map == "person_map" else "vm"
    person = id_map == "visit_map"  # the visit map also records the person of each visit
    cols = ", ".join(["source_value", id_col] + (["person_source_value"] if person else []))
    restore = [
        _comment(header + "\n1) retired keys that are back get their old id again"),
        "WITH k AS (",
        *_indent(keys),
        "),",
        "back AS (",
        f"    DELETE FROM {{etl}}.{id_map}_retired r USING k",
        "    WHERE r.source_value = k.source_value",
        f"    RETURNING r.source_value, r.{id_col}"
        + (", COALESCE(k.person_source_value, r.person_source_value) AS person_source_value" if person else ""),
        ")",
        f"INSERT INTO {{etl}}.{id_map} ({cols})",
        f"SELECT {cols} FROM back",
        "ON CONFLICT DO NOTHING;",
    ]
    new = [
        _comment(
            "2) new keys, numbered in key order after the largest id ever given out (in the map or\n"
            "   retired), so an id is never reused"
        ),
        f"INSERT INTO {{etl}}.{id_map} ({cols})",
        "SELECT k.source_value,",
        f"    (GREATEST((SELECT COALESCE(MAX({id_col}), 0) FROM {{etl}}.{id_map}),",
        f"              (SELECT COALESCE(MAX({id_col}), 0) FROM {{etl}}.{id_map}_retired))",
        "     + ROW_NUMBER() OVER (ORDER BY k.source_value))::integer" + ("," if person else ""),
        *(["    k.person_source_value"] if person else []),
        "FROM (",
        *_indent(keys),
        ") k",
        f"WHERE NOT EXISTS (SELECT 1 FROM {{etl}}.{id_map} {a} WHERE {a}.source_value = k.source_value)",
        "ON CONFLICT DO NOTHING;",
    ]
    return ["\n".join(restore), "\n".join(new)]


def _person_map_sql(key: KeySpec) -> list[str]:
    """The statements of the person id-map step."""
    expr = _p(key.expression)
    keys = [f"SELECT DISTINCT {expr}::text AS source_value", f"FROM {source_ref(key.table)}",
            f"WHERE {expr} IS NOT NULL"]
    return _id_map_statements("person_map", "person_id", keys, (
        f"person id map: one person_id per distinct {key.table}.{_norm(key.expression)}.\n"
        "The map is kept across runs, so a person keeps their person_id on every rebuild (stored\n"
        "cohorts stay valid). Keys that do not become a person are retired, with their id, by the\n"
        "pruning step after the person load."
    ))


def _outside_literals(expr: str, fn: Callable[[str], str]) -> str:
    """``expr`` with ``fn`` applied to the text between its string literals."""
    out, last = [], 0
    for m in _STRING_LITERAL.finditer(expr):
        out += [fn(expr[last:m.start()]), m.group(0)]
        last = m.end()
    return "".join(out + [fn(expr[last:])])


_TEXT_CAST = re.compile(r"\s*::\s*(text|varchar|character\s+varying)\s*$")


def _bare(expr: str, alias: str | None = None) -> str:
    """A key expression in a comparable form: ``alias.`` qualifiers removed, white space and case
    outside string literals normalised, outer parentheses and trailing casts to text dropped (the
    id maps compare keys as text anyway)."""
    def norm(text: str) -> str:
        if alias:
            text = re.sub(rf"\b{re.escape(alias)}\s*\.\s*", "", text, flags=re.I)
        return re.sub(r"\s*([(),])\s*", r"\1", re.sub(r"\s+", " ", text.lower()))

    out = _outside_literals(expr.strip(), norm).strip()
    while True:
        if out.startswith("(") and out.endswith(")") and _balanced(out[1:-1]):
            out = out[1:-1].strip()
        elif _TEXT_CAST.search(out):
            out = _TEXT_CAST.sub("", out).strip()
        else:
            return out


def _balanced(text: str) -> bool:
    depth = 0
    for ch in _STRING_LITERAL.sub("''", text):
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def _key_field(tm: TableMapping, field: str, kind: str) -> FieldMapping | None:
    return next((f for f in tm.fields if f.target_field == field and f.kind == kind and (f.expression or "").strip()),
                None)


def _covered_by_key(tm: TableMapping, expr: str, key: KeySpec | None) -> bool:
    """The keys a mapping writes are keys of the spec's person_key / visit_key: it reads the key
    table and writes the key expression over its own source alias only (its joins and filter can
    only select fewer rows, so its keys are in the map already)."""
    if key is None or tm.source.table.strip().lower() != key.table.strip().lower():
        return False
    if {a for a, _ in alias_refs(expr)} - {tm.source.alias.lower()}:
        return False
    return _bare(expr, tm.source.alias) == _bare(key.expression)


def key_sources(spec: ETLSpec, table: str) -> list[tuple[TableMapping, str, str | None]]:
    """(mapping, key expression, person expression) of each mapping into ``table`` (person or
    visit_occurrence) whose keys are not those of the spec's person_key / visit_key: each such
    mapping adds the keys of its rows to the id map in its own step (``person_map_2``, ...), so
    people and visits can come from several source tables. The person expression is that of a
    visit mapping's person_id (None for person)."""
    field, kind = ("person_id", "person_id") if table == "person" else ("visit_occurrence_id", "visit_id")
    key = spec.person_key if table == "person" else spec.visit_key
    if key is None:
        return []
    out: list[tuple[TableMapping, str, str | None]] = []
    for tm in spec.table_mappings:
        kf = _key_field(tm, field, kind) if tm.target_table.lower() == table else None
        if kf is None or _covered_by_key(tm, kf.expression or "", key):
            continue
        pf = _key_field(tm, "person_id", "person_id") if table == "visit_occurrence" else None
        out.append((tm, kf.expression or "", pf.expression if pf else None))
    return out


def _mapping_keys_sql(table: str, tm: TableMapping, expr: str, person: str | None, etl: EtlConfig | None) -> list[str]:
    """The statements of the id-map step of one more key source (``key_sources``): the keys of the
    rows the mapping reads (its FROM, joins and WHERE, read tolerantly and uncounted), restored or
    numbered like those of the person_key / visit_key. A key that is in the map already keeps its
    id: a key found in two sources is one person (visit)."""
    key = _p(expr)
    where = where_sql(tm, etl, counted=False)
    conditions = [f"WHERE {_p(where)}", f"  AND {key} IS NOT NULL"] if where else [f"WHERE {key} IS NOT NULL"]
    what = "person" if table == "person" else "visit"
    header = (
        f"{what} id map: also the keys of the {table} mapping from {tm.source.table} ({tm.source.alias}):\n"
        f"{_norm(expr)}. A key that is in the map already keeps its id (the same {what})."
    )
    if table == "person":
        keys = [f"SELECT DISTINCT {key}::text AS source_value", *_from_clause(tm), *conditions]
        return _id_map_statements("person_map", "person_id", keys, header)
    person_sql = f"MIN({_p(person)}::text)" if person else "NULL::text"
    keys = [f"SELECT {key}::text AS source_value, {person_sql} AS person_source_value", *_from_clause(tm),
            *conditions, "GROUP BY 1"]
    return _id_map_statements("visit_map", "visit_occurrence_id", keys, header + (
        "\nperson_source_value is a first guess: the visit pruning step records the person each visit\n"
        "was loaded for."
    ))


def _person_map_prune_sql() -> str:
    return "\n".join(
        [
            _comment(
                "person id map: retire the keys that did not become a person (the person mapping's WHERE or\n"
                "joins filtered them out, or the person load failed), so the INNER JOIN on person_map drops\n"
                "their events instead of loading rows whose person_id is not in person. A retired key keeps\n"
                "its id in person_map_retired and gets it back when it becomes a person again; the ids of\n"
                "everyone kept are unchanged."
            ),
            "WITH gone AS (",
            "    DELETE FROM {etl}.person_map pm",
            "    WHERE NOT EXISTS (SELECT 1 FROM {cdm}.person p WHERE p.person_id = pm.person_id)",
            "    RETURNING pm.source_value, pm.person_id",
            ")",
            "INSERT INTO {etl}.person_map_retired (source_value, person_id)",
            "SELECT source_value, person_id FROM gone",
            "ON CONFLICT DO NOTHING;",
        ]
    )


def _visit_person_mapping(spec: ETLSpec) -> TableMapping | None:
    """A mapping that reads the visit table and whose person expression uses only its source alias."""
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
            return tm
    return None


def _visit_person_expr(spec: ETLSpec) -> tuple[str, str] | None:
    """(alias, person expression) from a mapping that reads the visit table, if one is usable."""
    tm = _visit_person_mapping(spec)
    if tm is None:
        return None
    pf = next(f for f in tm.fields if f.kind == "person_id" and f.expression)
    return tm.source.alias, pf.expression  # type: ignore[return-value]


def _visit_map_sql(spec: ETLSpec) -> list[str]:
    """The statements of the visit id-map step."""
    vk = spec.visit_key
    assert vk is not None
    person = _visit_person_expr(spec)
    alias = qi(person[0]) if person else "v"
    person_sql = f"MIN({_p(person[1])}::text)" if person else "NULL::text"
    keys = [
        f"SELECT {_p(vk.expression)}::text AS source_value, {person_sql} AS person_source_value",
        f"FROM {source_ref(vk.table)} {alias}",
        f"WHERE {_p(vk.expression)} IS NOT NULL",
        "GROUP BY 1",
    ]
    return _id_map_statements("visit_map", "visit_occurrence_id", keys, (
        f"visit id map: one visit_occurrence_id per distinct {vk.table}.{_norm(vk.expression)}.\n"
        "Kept across runs like the person id map; keys that do not become a visit are retired, with\n"
        "their id, by the pruning step after the visit_occurrence load. person_source_value is a first\n"
        "guess from the visit table: that step records the person each visit was loaded for."
    ))


def _visit_map_prune_sql() -> list[str]:
    """The statements of the visit id-map pruning step."""
    refresh = [
        _comment(
            "visit id map: 1) record the person each loaded visit belongs to, so records link only to\n"
            "   a visit of their own person and links follow the source when a visit key moves to\n"
            "   another person between extracts (the entry is replaced: the ETL role has no UPDATE).\n"
            "   A visit loaded for two people (a shared key: isPrimaryKey fails) records the smaller\n"
            "   person key (byte-wise), so the choice does not change from run to run or server to server"
        ),
        "WITH loaded AS (",
        '    SELECT v.visit_occurrence_id, MIN(pm.source_value COLLATE "C") AS person_source_value',
        "    FROM {cdm}.visit_occurrence v",
        "    JOIN {etl}.person_map pm ON pm.person_id = v.person_id",
        "    GROUP BY v.visit_occurrence_id",
        "),",
        "moved AS (",
        "    DELETE FROM {etl}.visit_map vm",
        "    USING loaded l",
        "    WHERE l.visit_occurrence_id = vm.visit_occurrence_id",
        "      AND vm.person_source_value IS DISTINCT FROM l.person_source_value",
        "    RETURNING vm.source_value, vm.visit_occurrence_id, l.person_source_value",
        ")",
        "INSERT INTO {etl}.visit_map (source_value, visit_occurrence_id, person_source_value)",
        "SELECT source_value, visit_occurrence_id, person_source_value FROM moved;",
    ]
    retire = [
        _comment(
            "2) retire the keys that did not become a visit (filtered out by the visit_occurrence\n"
            "   mapping, visits of people who were not loaded, or a failed load), so events that refer\n"
            "   to them get a NULL visit_occurrence_id instead of an id missing from visit_occurrence.\n"
            "   A retired key keeps its id in visit_map_retired and gets it back when it becomes a visit\n"
            "   again."
        ),
        "WITH gone AS (",
        "    DELETE FROM {etl}.visit_map vm",
        "    WHERE NOT EXISTS (SELECT 1 FROM {cdm}.visit_occurrence v"
        " WHERE v.visit_occurrence_id = vm.visit_occurrence_id)",
        "    RETURNING vm.source_value, vm.visit_occurrence_id, vm.person_source_value",
        ")",
        "INSERT INTO {etl}.visit_map_retired (source_value, visit_occurrence_id, person_source_value)",
        "SELECT source_value, visit_occurrence_id, person_source_value FROM gone",
        "ON CONFLICT DO NOTHING;",
    ]
    return ["\n".join(refresh), "\n".join(retire)]


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
            # the configured extract date, else the build date
            f"    {'DATE ' + ql(cs.source_release_date.isoformat()) if cs.source_release_date else 'CURRENT_DATE'},",
            "    CURRENT_DATE,",
            "    '5.4',",
            "    COALESCE((SELECT concept_id FROM {vocab}.concept"
            # Athena codes CDM versions as 'CDM v5.4.0' (concept 756265), 'CDM v5.4.1', ...
            " WHERE vocabulary_id = 'CDM' AND concept_code IN ('CDM v5.4.0', 'CDM v5.4')"
            " ORDER BY concept_code DESC LIMIT 1), 0),",
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
    etl: EtlConfig | None = None,
    strict: bool = True,
) -> list[CompiledStep]:
    """Compile the whole spec into ordered steps (see the module docstring for the order).
    ``etl`` (``ProjectConfig.etl``) gives the partial-date rule of the tolerant conversions.

    With ``strict=False`` a table mapping that does not compile yields a step with ``error``
    set (and empty SQL) instead of raising, so the other tables can still be loaded.
    """
    steps: list[CompiledStep] = []  # names are assigned at the end

    # 1. the id maps, before every insert: the person_key / visit_key, then the keys of every other
    #    person / visit_occurrence mapping (key_sources), each in its own step, so a source that
    #    fails loses only its own keys
    pm = _person_map_sql(spec.person_key)
    steps.append(CompiledStep("person_map", None, "id_map", "\n".join(pm), statements=tuple(pm),
                              source_table=spec.person_key.table, origins=(spec.person_key,)))
    steps += _key_source_steps(spec, "person", etl, strict)
    if spec.visit_key is not None:
        person_from = _visit_person_mapping(spec)
        vm = _visit_map_sql(spec)
        steps.append(CompiledStep("visit_map", None, "id_map", "\n".join(vm), statements=tuple(vm),
                                  source_table=spec.visit_key.table,
                                  origins=(spec.visit_key,) + ((person_from,) if person_from else ())))
        steps += _key_source_steps(spec, "visit_occurrence", etl, strict)
    for i, s in enumerate(steps):  # before every other step, in this order
        s.slot = (-1, 0, i)

    # 2. the spec's writers, each at its table's slot: the mappings in spec order, then the custom SQL
    seen_count: dict[str, int] = {}
    for i, tm in enumerate(spec.table_mappings):
        t = tm.target_table.lower()
        seen_count[t] = seen_count.get(t, 0) + 1
        label = t if seen_count[t] == 1 else f"{t}_{seen_count[t]}"
        slot = (_load_index(t), 0, i)
        try:
            c = _compile(tm, spec, etl)
            step = CompiledStep(
                label, t, "insert", c.insert, source_table=tm.source.table, source_count_sql=_source_count(tm, etl=etl),
                retained_count_sql=c.retained, origins=(tm, *c.count_origins), returns=c.returns,
                count_sites=c.count_sites, slot=slot,
            )
        except CompileError as e:
            if strict:
                raise
            step = CompiledStep(label, t, "insert", "", source_table=tm.source.table, origins=(tm,),
                                error=f"compile error: {e}", slot=slot)
        steps.append(step)
    for j, cs in enumerate(spec.custom_sql):
        t = cs.target_table.lower()
        seq = _CUSTOM_AFTER_MAPPINGS + j if t in seen_count else j
        steps.append(_custom_step(cs, slot=(_load_index(t), 0, seq)))

    # 3. the steps the build generates: prune each id map after the last writer of its table,
    #    before the events. Pruning always runs, also after a failed load: it only retires keys,
    #    and they get their ids back on the next build that loads them
    written = {s.target_table for s in steps if s.kind in ("insert", "custom")}
    if "person" in written:
        steps.append(CompiledStep("person_map_prune", None, "prune", _person_map_prune_sql(),
                                  slot=(_load_index("person"), 1, 0)))
    if spec.visit_key is not None and "visit_occurrence" in written:
        vp = _visit_map_prune_sql()
        steps.append(CompiledStep("visit_map_prune", None, "prune", "\n".join(vp), statements=tuple(vp),
                                  slot=(_load_index("visit_occurrence"), 1, 0)))

    if spec.observation_period == "from_events":
        loaded = [t for t in target_tables(spec) if t != "observation_period"]
        op = _observation_period_sql(spec, loaded)
        if op is not None:
            steps.append(CompiledStep("observation_period", "observation_period", "derived", op,
                                      slot=(_load_index("observation_period"), 0, _AFTER_SPEC_WRITERS)))

    steps.append(CompiledStep("cdm_source", "cdm_source", "metadata", _cdm_source_sql(cdm_source),
                              slot=(len(LOAD_ORDER) + 1, 0, 0)))  # last, after custom SQL of unknown tables

    # 4. a stable sort keeps spec order within a slot; 5. number and render
    steps.sort(key=lambda s: s.slot)
    width = max(2, len(str(len(steps))))
    for i, s in enumerate(steps, start=1):
        s.name = f"{i:0{width}d}_{s.name}"
        s.sql = schemas.render(s.sql)
        if s.statements:
            s.statements = tuple(schemas.render(x) for x in s.statements)
        elif s.sql and s.kind != "custom":
            s.statements = (s.sql,)  # a generated step of one statement
        if s.source_count_sql:
            s.source_count_sql = schemas.render(s.source_count_sql)
        if s.retained_count_sql:
            s.retained_count_sql = schemas.render(s.retained_count_sql)
    return steps


def _key_source_steps(spec: ETLSpec, table: str, etl: EtlConfig | None, strict: bool) -> list[CompiledStep]:
    """One id-map step (``person_map_2``, ``visit_map_2``, ...) per ``key_sources`` mapping."""
    out = []
    for n, (tm, expr, person) in enumerate(key_sources(spec, table), start=2):
        label = f"{_ID_MAP_LABELS[table]}_{n}"
        try:
            stmts = _mapping_keys_sql(table, tm, expr, person, etl)
        except CompileError as e:
            if strict:
                raise
            out.append(CompiledStep(label, None, "id_map", "", source_table=tm.source.table, origins=(tm,),
                                    error=f"compile error: {e}"))
            continue
        out.append(CompiledStep(label, None, "id_map", "\n".join(stmts), statements=tuple(stmts),
                                source_table=tm.source.table, origins=(tm,)))
    return out


def _custom_step(cs: CustomSQL, slot: tuple[int, int, int]) -> CompiledStep:
    sql = cs.sql.strip()
    header = _comment(f"custom SQL '{cs.name}' -> {cs.target_table}" + (f"\n{cs.rationale}" if cs.rationale.strip() else ""))
    return CompiledStep(f"custom_{_safe_name(cs.name)}", cs.target_table.lower(), "custom", header + "\n" + sql,
                        origins=(cs,), slot=slot)
