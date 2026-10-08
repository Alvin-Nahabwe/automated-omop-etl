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
   visit / visit_detail / drug_exposure end date the start date, a NULL observation_period end
   date is open (9999-12-31, ended by the merge, see below), and a row with another NULL
   required value is dropped; each counted per field in transaction-local settings that the
   runner reads into ``value_counts`` (``CompiledStep.count_sites``);
3. the vocabulary lookups, ``auto_id`` and NULL fields, in the outer SELECT, so a code with several
   targets cannot multiply the counts;
4. for person, death and visit_occurrence, which keep one row per key (``ONE_ROW_PER``), one
   statement that keeps the best row of each key (the mapping's ``dedup_order`` with NULLs last,
   then for death a date known to the day and the latest date, then the most complete row, then
   a hash of the source row's text; a visit spans the earliest start to the latest end of its rows), skips
   the keys an earlier mapping loaded and returns its statistics.

A record's visit_occurrence_id links only to a visit of the row's person: the visit-map join also
matches the person key the visit was loaded for. The retained-row count of such a mapping counts
the visit links (linked, another person's visit, a visit that a visit mapping left out) in the
same pass.

A full compile yields ordered steps. Each step has a sort key ``slot`` = (load index of its
table, band, sequence), and the steps are stably sorted on it: the id maps come first; then,
table by table in CDM load order (``cdm.spec.LOAD_ORDER``), the table's writers (band 0: its
mappings in spec order, then its custom SQL; custom SQL of a table without a mapping runs at
that table's place too, not after everything, and custom SQL that inserts into an era table at
that era table's place) and the steps generated for it (band 1); after
the last writer of the event tables (specimen, band 2) the route block moves records to the table
of their concept's domain (``etl.routing``: ``route_prepare``, then one non-blocking
``route_<table>`` step per routed table), before payer_plan_period, cost, the observation periods
and the eras, which read or refer to the records where they end up; the derived observation
period follows every spec writer of observation_period, and the ``cdm_source`` row comes last.
After the last writer of observation_period (and so after death) the
``observation_period_merge`` step replaces the loaded periods (residency episodes, or the
periods derived from events) by each person's merged periods: open episodes end at the source
release date (``cdm_source.source_release_date``, or one derived robustly from the exit dates and
stored in ``{etl}.build_facts`` for the cdm_source row) or at death, real exits are never clipped,
episodes that end before they start are dropped, and overlapping or back-to-back episodes become
one period (periods derived from the events are never open). With ``observation_period: table_mapping`` the ``observation_period_fill`` step then
gives every person without a period one from their first to their last event. With
``derive_eras`` the ``condition_era`` and ``drug_era`` steps then build the OHDSI eras from the
condition and drug records as routed (``derived_era_tables``: not when spec SQL loads the era
table, which replaces the step); they are not blocking and are skipped when a load of their event
table failed. After the person
(and visit_occurrence) writers an id-map pruning
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
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Literal

from ..cdm.spec import DERIVED_ERAS, EVENT_DATE_FIELDS, LOAD_ORDER, CdmField, CdmTable, load_cdm_spec
from ..config import CdmSourceConfig, EtlConfig
from ..db import Schemas, qi, ql
from ..models import CustomSQL, ETLSpec, FieldMapping, KeySpec, TableMapping
from ..sqlguard import code_text
from .policies import (
    END_FROM_START,
    OPEN_END,
    OPEN_ENDS,
    OPEN_FROM,
    POLICY_KINDS,
    ROW_SITE,
    WHERE_SITE,
    Policy,
    partial_policy,
    required_null_policy,
    start_end_pairs,
)
from .routing import ROUTE_PREPARE, route_prepare_sql, route_sources, route_sql, route_targets
from .tolerant import LexError, PolicyArg, Rewrite, column_refs, kind_of, lex, rewrite, tolerant_value

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
# The layer-2 column that says a visit's end was read from the source, not completed from its start.
_END_KNOWN = "_end_known"
# The column a keyed mapping's source subquery adds: a hash of the whole source row (_from_clause).
_SRC_HASH = "_omop_src_hash"
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


def _date_rule(tm: TableMapping | None, table: CdmTable, f: CdmField, etl: EtlConfig | None) -> PolicyArg:
    """The partial-date rule of ``table.f`` in the mapping ``tm`` (``etl.policies.partial_policy``).
    The end of a start/end pair takes the last day of an incomplete date (the widest interval), but
    only for values of its own: a conversion whose operand reads only columns the start field also
    reads (``start + days_supply - 1``, ``COALESCE(discharge_date, visit_date)``) takes the start's
    rule, so an end computed from the start is not stretched to the end of the month or year."""
    policy = partial_policy(table.name, f.name, etl)
    if tm is None or policy != "last":
        return policy
    start = next((s for (t, s), e in start_end_pairs().items() if t == table.name and e == f.name), None)
    sm = next((fm for fm in tm.fields if fm.target_field.lower() == start), None) if start else None
    if start is None or sm is None or sm.kind not in ("column", "expression"):
        return policy
    shared = column_refs(sm.expression)
    if not shared:
        return policy
    start_policy = partial_policy(table.name, start, etl)

    def rule(operand: str) -> str:
        refs = column_refs(operand)
        return start_policy if refs and refs <= shared else policy

    return rule


def _tolerant_field(
    expr: str, f: CdmField, table: CdmTable, joins: _Joins, etl: EtlConfig | None, tm: TableMapping | None = None,
) -> str:
    """A column or expression value converted tolerantly (``etl.tolerant``), counted at the field."""
    rw = _refuse_bad_templates(
        tolerant_value(expr, f.datatype, f.name, _date_rule(tm, table, f, etl)), f"{table.name}.{f.name}")
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
        return _tolerant_field(expr, f, table, joins, etl, tm)
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


def _from_clause(tm: TableMapping, hashed: bool = False) -> list[str]:
    """The mapping's FROM and joins. ``hashed`` reads the source table through a fenced subquery that
    adds ``_SRC_HASH``, a hash of the whole source row (the neutral last tie-break of the rows of
    one key): the hash is computed at the scan, so the columns no value reads are dropped there and
    wide source rows never travel through the joins, the candidates and their sorts."""
    a = qi(tm.source.alias)
    src = f"{source_ref(tm.source.table)} {a}"
    if hashed:
        src = f"(SELECT {a}.*, md5(({a}.*)::text) AS {_SRC_HASH} FROM {src} OFFSET 0) {a}"
    lines = [f"FROM {src}"]
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
    from the source. 'open' (observation_period_end_date) loads a NULL end as ``OPEN_END``, which
    the observation_period_merge step ends at the release date or at death."""
    in_layer_1 = {f.name for f, fm, _ in values if fm.kind in _LAYER_1_KINDS}
    out: dict[str, Policy] = {}
    for f, fm, _ in values:
        if not f.required or fm.kind not in POLICY_KINDS:
            continue
        policy = required_null_policy(table.name, f.name)
        if policy == "start" and END_FROM_START[(table.name, f.name)] not in in_layer_1:
            policy = "drop"  # no start date to take
        if policy is not None:
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
        "SELECT", *_select_list(items), *_from_clause(tm, any(name == "_src_text" for name, _, _ in extra or [])),
        *joins.id_joins,
    ]
    if where:
        rows.append(f"WHERE {_p(where)}")
    return rows + ["OFFSET 0"]  # a fence: the planner neither merges the layers nor evaluates a value twice


def _required_values(
    table: CdmTable, layer_1: list[tuple[CdmField, FieldMapping, str]], joins: _Joins,
    required: dict[str, Policy], rows: list[str], extra: list[str] | None = None,
    known: dict[str, str] | None = None,
) -> list[str]:
    """Layer 2 over the layer-1 ``rows``: required values that are NULL get the CDM convention of
    their field (``required``), and each one is counted. Rows with a NULL drop-policy field are
    dropped; the counters are called only on the NULL branches. The ``extra`` columns of layer 1
    are passed on, and for each ``known`` column (name -> field) whether the field's layer-1 value
    was there before its policy gave it one."""

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
        elif policy == "open":  # the merge counts the open episodes and ends them
            items.append((f"COALESCE({col(name)}, {OPEN_END}) AS {qi(name)}",
                          "NULL = still open (ended by the observation_period_merge step)"))
        else:
            items.append((col(name), "NULL: the row is dropped (counted)" if policy == "drop" else ""))
    items += [(f"{_VALUES}.{k}", "") for k, _ in joins.key_exprs]
    items += [(f"{_VALUES}.{name}", "") for name in extra or []]
    items += [(f"{col(field)} IS NOT NULL AS {name}", f"{field} as read, not completed by its policy")
              for name, field in (known or {}).items()]
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
    # a visit end the 'start' policy completed is not known: it must not split the visit (_gap)
    known = ({_END_KNOWN: "visit_end_date"} if keys and table.name == "visit_occurrence"
             and required.get("visit_end_date") == "start" else {})
    if required:
        rows = _required_values(table, layer_1, joins, required, rows, [name for name, _, _ in extra], known)
    items: list[tuple[str, str]] = [
        (f"{_ROWS}.{qi(f.name)}", "") if fm.kind in _LAYER_1_KINDS else (f"{v} AS {qi(f.name)}", fm.kind)
        for f, fm, v in values
    ]
    items += [(f"{_ROWS}.{name}", "") for name in [*(n for n, _, _ in extra), *known]]
    select = ["SELECT", *_select_list(items), "FROM (", *_indent(rows), f") {_ROWS}", *joins.lines]

    header = f"{table.name} <- {tm.source.table} ({tm.source.alias})"
    if tm.joins:
        header += " + " + ", ".join(f"{j.table} ({j.alias})" for j in tm.joins)
    out = [_comment(header)]
    if tm.rationale.strip():
        out.append(_comment(tm.rationale))
    if keys:
        out += _one_row_per_key(table, values, select, keys, {n for n, _, _ in extra} | set(known))
    else:
        out.append(f"INSERT INTO {{cdm}}.{qi(table.name)} (")
        out += [f"    {qi(f.name)}{',' if i < len(values) - 1 else ''}" for i, (f, _, _) in enumerate(values)]
        out += [")", *select]
    # the counter sites in field order (required-field policies, tolerant conversions), then the
    # WHERE's and the dropped rows'
    sites = tuple(f.name for f, _, _ in values
                  if (f.name in required and required[f.name] != "open") or f.name in joins.sites)
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
    a hash of the source row's text (``_src_text``, the neutral last tie-break; never stored; see
    ``_from_clause``)."""
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
    out.append(("_src_text", f"{qi(tm.source.alias)}.{_SRC_HASH}", "the last tie-break (a hash of the source row)"))
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
    a hash of the source row's text, then the values: the choice is deterministic and prefers no sex, concept
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
    day after the earlier rows end, so the visit key is coarser than one visit; only ends read from
    the source count, not an end the 'start' policy completed, ``_END_KNOWN``); ``dedup_order``
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
        # only an end read from the source tells that a row ended: one the 'start' policy completed
        # (a ward stay whose ward_out is missing or unreadable) may have lasted until the next row
        end = (f"CASE WHEN c.{_END_KNOWN} THEN c.visit_end_date END" if _END_KNOWN in extra
               else "c.visit_end_date")
        rows = [
            "FROM (",
            f"    SELECT c.*, c.visit_start_date > MAX({end}) OVER (",
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
        + ", then ".join(best + ["the most complete row (a concept 0 is missing)", "a hash of the source row"]),
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


def _uncounted_value(
    fm: FieldMapping, f: CdmField, table: CdmTable, etl: EtlConfig | None, tm: TableMapping | None = None,
) -> str | None:
    """The value of a column, expression, constant or null field of the mapping ``tm`` as the build
    computes it, without counters (None for other kinds)."""
    expr = (fm.expression or "").strip()
    if fm.kind == "null":
        return "NULL"
    if not expr:
        return None
    if fm.kind == "constant":
        return cast_to(expr, f.datatype)
    if fm.kind not in ("column", "expression"):
        return None
    rw = tolerant_value(expr, f.datatype, f.name, _date_rule(tm, table, f, etl), counted=False)
    return rw.sql if kind_of(f.datatype) else cast_to(rw.sql, f.datatype)


@dataclass
class NullingCount:
    """The counts that tell whether an edit makes a required field (or a record's visit key) NULL
    for more rows (see ``nulling_count_sql``). Rendered SQL."""

    rows: str  # COUNT(*) AS n of the rows where the new value is NULL and the old one is not
    new_totals: str  # COUNT(*) AS n, COUNT(<new value>) AS v over the new mapping's rows
    old_totals: str | None  # the same for the old value over the old mapping's rows (None: it does not compile)
    # 'drop': a required field whose NULL drops the row; 'visit_key': the visit key of a record's
    # optional visit link, whose NULL takes the record out of the link checks; 'open': an end
    # whose NULL means the episode is still open (``rows`` gives n, v and m, see nulling_count_sql)
    what: Literal["drop", "visit_key", "open"] = "drop"


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
    and may become NULL. For an end whose NULL means "still open" (policy 'open',
    observation_period_end_date) one count gives ``n``, the episodes the edit opens (the new end is
    NULL or a placeholder on or after ``OPEN_FROM`` where the old one was a real exit, other than
    the start date an old fallback gave), ``v``, the distinct old exits among them (one value is a
    placeholder exit that may become NULL), and ``m``, the open episodes it closes on their start
    date (a new fallback onto the start). Raises ``CompileError`` when the new mapping does not
    compile."""
    table = _check_mapping(new)
    c = _compile(new, spec, etl)
    try:
        old_lines: list[str] | None = _compile(old, spec, etl).id_lines
    except CompileError:
        old_lines = None
    before = {fm.target_field: fm for fm in old.fields}
    after = {fm.target_field: fm for fm in new.fields}
    out: dict[str, NullingCount] = {}
    for fm in new.fields:
        f = table.field(fm.target_field)
        was = before.get(fm.target_field)
        visit_key = f is not None and was is not None and was.kind == "visit_id" and not f.required
        if f is None or (c.required.get(f.name) not in ("drop", "open") and not visit_key):
            continue
        if was is not None and (was.kind, _norm(was.expression or "")) == (fm.kind, _norm(fm.expression or "")):
            continue
        if c.required.get(f.name) == "open":
            counts = _open_end_count(new, old, before, after, f, table, c.id_lines, etl)
            if counts is not None:
                out[f.name] = NullingCount(schemas.render(counts), schemas.render(counts), None, what="open")
            continue
        new_value = _key_value(fm) if fm.kind == "visit_id" else _uncounted_value(fm, f, table, etl, new)
        if visit_key:
            old_value = _key_value(was)  # type: ignore[arg-type]
        else:
            old_value = _uncounted_value(was, f, table, etl, old) if was is not None else "NULL"
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


def _without_source(sql: str, aliases: set[str]) -> str | None:
    """``sql`` with every column of the ``aliases`` read as a NULL text (staged columns are text);
    None when the scan cannot read it."""
    try:
        toks = lex(sql)
    except LexError:
        return None
    sig = [i for i, t in enumerate(toks) if t.kind not in ("ws", "comment")]
    cut: list[tuple[int, int]] = []
    for n, i in enumerate(sig[:-2]):
        a, d, c = toks[i], toks[sig[n + 1]], toks[sig[n + 2]]
        alias = a.text[1:-1] if a.kind == "qid" else a.text.lower()
        if a.kind in ("id", "qid") and alias in aliases and d.text == "." and c.kind in ("id", "qid"):
            follows = toks[sig[n + 3]].text if n + 3 < len(sig) else ""
            before = toks[sig[n - 1]].text if n > 0 else ""
            if follows != "(" and before != ".":
                cut.append((a.start, c.end))
    out, pos = [], 0
    for a, b in cut:
        out += [sql[pos:a], "NULL::text"]
        pos = b
    return "".join(out) + sql[pos:]


def filled_without_source_sql(
    old: TableMapping, new: TableMapping, schemas: Schemas, *, etl: EtlConfig | None = None,
) -> dict[str, tuple[str, str | None]]:
    """For an edit of a table mapping, per required field whose NULL drops the row (policy 'drop')
    and whose value changed: a query (new, old) that tells whether the field still has a value when
    every column of the mapping's source and joined tables is NULL (``filled``), i.e. whether its
    expression gives rows without a source value one of its own: a literal (a birth year 1900), a
    constant or the build date. The old query is None when the old mapping has no such field.
    Fields whose value cannot be read this way are left out."""
    table = _check_mapping(new)
    before = {fm.target_field: fm for fm in old.fields}

    def filled(tm: TableMapping, fm: FieldMapping | None, f: CdmField) -> str | None:
        if fm is None:
            return None
        value = _uncounted_value(fm, f, table, etl, tm)
        aliases = {tm.source.alias.lower()} | {j.alias.lower() for j in tm.joins}
        nulled = _without_source(value, aliases) if value is not None else None
        return schemas.render(f"SELECT ({_p(nulled)}) IS NOT NULL AS filled") if nulled is not None else None

    out: dict[str, tuple[str, str | None]] = {}
    for fm in new.fields:
        f = table.field(fm.target_field)
        if f is None or fm.kind not in POLICY_KINDS or required_null_policy(table.name, f.name) != "drop":
            continue
        was = before.get(fm.target_field)
        if was is not None and (was.kind, _norm(was.expression or "")) == (fm.kind, _norm(fm.expression or "")):
            continue
        new_sql = filled(new, fm, f)
        if new_sql is not None:
            out[f.name] = (new_sql, filled(old, was, f))
    return out


def _open_end_count(
    new: TableMapping, old: TableMapping, before: dict[str, FieldMapping], after: dict[str, FieldMapping],
    f: CdmField, table: CdmTable, id_lines: list[str], etl: EtlConfig | None,
) -> str | None:
    """The count of ``nulling_count_sql`` for an open end ``f`` (n, v, m) over the new mapping's
    rows, with the old (``before``, of ``old``) and new (``after``, of ``new``) field mappings;
    None when an end is not computed from the source."""
    was = before.get(f.name)
    new_end = _uncounted_value(after[f.name], f, table, etl, new)
    old_end = _uncounted_value(was, f, table, etl, old) if was is not None else "NULL"
    if new_end is None or old_end is None:
        return None
    new_end, old_end = _p(new_end), _p(old_end)
    start = table.field(OPEN_ENDS[(table.name, f.name)])

    def start_value(fields: dict[str, FieldMapping], tm: TableMapping) -> str | None:
        sm = fields.get(start.name) if start is not None else None
        value = _uncounted_value(sm, start, table, etl, tm) if sm is not None and start is not None else None
        return _p(value) if value is not None else None

    old_start, new_start = start_value(before, old), start_value(after, new)
    opened = f"({new_end} IS NULL OR {new_end} >= {OPEN_FROM}) AND {old_end} < {OPEN_FROM}"
    if old_start is not None:
        opened += f" AND {old_end} IS DISTINCT FROM {old_start}"  # an old fallback onto the start date
    closed = (f"({old_end} IS NULL OR {old_end} >= {OPEN_FROM}) AND {new_end} < {OPEN_FROM} AND {new_end} = {new_start}"
              if new_start is not None else "false")
    what = (f"COUNT(*) FILTER (WHERE {opened}) AS n,\n"
            f"       COUNT(DISTINCT ({old_end})::date) FILTER (WHERE {opened}) AS v,\n"
            f"       COUNT(*) FILTER (WHERE {closed}) AS m")
    return _source_count(new, id_lines, etl, what=what)


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


def mapping_labels(spec: ETLSpec) -> list[str]:
    """The step label of each table mapping, in spec order: its table, and ``<table>_<n>`` for the
    n-th mapping into one table."""
    seen: dict[str, int] = {}
    out = []
    for tm in spec.table_mappings:
        t = tm.target_table.lower()
        seen[t] = seen.get(t, 0) + 1
        out.append(t if seen[t] == 1 else f"{t}_{seen[t]}")
    return out


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


def _observation_period_sql(spec: ETLSpec, loaded: list[str], fill: bool = False) -> str | None:
    """One period per person from the first to the last date of their records in the ``loaded``
    tables: the from_events periods, or with ``fill`` the observation_period_fill step, which gives
    them only to the persons the merge left without a period (no valid episode) and returns one row:
    the periods ``inserted`` and the ``persons_without_period`` before it (with person loaded).
    Neither is clipped at the release date."""
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
    if not fill:
        title = "observation_period derived from events: one period per person, first to last event date"
        return "\n".join([_comment(title), *lines, "GROUP BY ev.person_id;"])
    lines += ["  AND NOT EXISTS (SELECT 1 FROM {cdm}.observation_period op WHERE op.person_id = ev.person_id)",
              "GROUP BY ev.person_id", "RETURNING 1"]
    # the persons without a period are counted in the snapshot before the insert
    without = ("(SELECT COUNT(*) FROM {cdm}.person p WHERE NOT EXISTS (SELECT 1 FROM {cdm}.observation_period op"
               " WHERE op.person_id = p.person_id))" if "person" in loaded else "NULL::bigint")
    return "\n".join([
        _comment("observation_period_fill: persons without a valid episode get one period, first to last event "
                 "date; returns\nthe periods inserted and the persons without a period before them"),
        "WITH ins AS (", *_indent("\n".join(lines).splitlines()), ")",
        "SELECT (SELECT COUNT(*) FROM ins) AS inserted,",
        f"       {without} AS persons_without_period;",
    ])


# The release date the merge derives when none is configured: the k-th latest exit of the valid,
# closed episodes, k = max(RELEASE_RANK_MIN, ceil(n / RELEASE_RANK_PER)), so that a few mistyped
# future exits (or junk rows) do not move it; it is independent of the build date.
RELEASE_RANK_MIN = 5
RELEASE_RANK_PER = 10_000
RELEASE_FACT = "source_release_date"


def release_rank(n: int) -> int:
    """k: the derived release date is the k-th latest exit of ``n`` valid, closed episodes."""
    return max(RELEASE_RANK_MIN, -(-n // RELEASE_RANK_PER))


def _release_rank(rows: str | None) -> str:
    """k - 1, the OFFSET of the k-th latest date of the episodes ``rows`` selects (``release_rank``)."""
    return (f"(SELECT GREATEST({RELEASE_RANK_MIN}, CEIL(COUNT(*) / {RELEASE_RANK_PER:.1f}))::int - 1 "
            f"FROM moved{f' WHERE {rows}' if rows else ''})")


def _observation_period_merge_sql(
    cdm_source: CdmSourceConfig | None, with_death: bool, from_events: bool = False
) -> list[str]:
    """The observation_period_merge step, two statements: forget the release date of an earlier
    run, then one statement that replaces the loaded episodes by each person's merged periods and
    returns its statistics (one row). The release date (``bounds``) is the configured
    ``cdm_source.source_release_date``, else derived (``RELEASE_RANK_MIN``); it is stored in
    ``{etl}.build_facts`` for cdm_source. An episode is open when its end is NULL (loaded as
    ``OPEN_END``) or a placeholder on or after ``OPEN_FROM``; it ends at the release date, or at
    the person's death before it; one that starts after the release date, or after the person's
    death, keeps one day. Real exits are never clipped. Episodes that end before they start are
    dropped (counted). The rest merge when they overlap or are back to back (DQD
    measureObservationPeriodOverlap: an end + 1 day = the next start), keeping the type of the
    first episode. With ``with_death`` (death is loaded before this step) the earliest death date
    of each person ends their open episodes. With ``from_events`` the periods were derived from
    the records and are never open (a far-future record date is a typo that plausibleValueHigh
    reports, not an open end): the merge only renumbers them and derives the release date."""
    release = cdm_source.source_release_date if cdm_source is not None else None
    if release is not None:
        bounds = [f"    SELECT DATE {ql(release.isoformat())} AS release_date, true AS configured"]
    else:
        valid_closed = "NOT is_open AND s <= e"
        bounds = [
            "    SELECT COALESCE(",
            f"             (SELECT e FROM moved WHERE {valid_closed} ORDER BY e DESC",
            f"              OFFSET {_release_rank(valid_closed)} LIMIT 1),",
            "             (SELECT s FROM moved ORDER BY s DESC",
            f"              OFFSET {_release_rank(None)} LIMIT 1),",
            "             (SELECT MIN(s) FROM moved)) AS release_date,",
            "           false AS configured",
        ]
    with_death = with_death and not from_events
    is_open = ("false AS is_open  -- periods derived from records are never open" if from_events else
               f"observation_period_end_date >= {OPEN_FROM} AS is_open  -- a NULL end was loaded as {OPEN_END}")
    at_death = "de.death_date >= m.s AND de.death_date < b.release_date"
    closed = [
        "    SELECT m.person_id, m.id, m.s, m.t, m.is_open,",
        "           CASE WHEN NOT m.is_open THEN m.e  -- a real exit is never clipped",
        *(["                WHEN de.death_date < m.s THEN m.s  -- open, but it starts after death: one day",
           f"                WHEN {at_death} THEN de.death_date"] if with_death else []),
        "                ELSE GREATEST(b.release_date, m.s) END AS e,  -- never dropped for the release date",
        f"           {f'm.is_open AND {at_death}' if with_death else 'false'} AS at_death,",
        f"           {'m.is_open AND de.death_date < m.s' if with_death else 'false'} AS starts_after_death,",
        "           m.is_open AND m.s > b.release_date AS starts_after_release,",
        "           NOT m.is_open AND m.s <= m.e AND m.e > b.release_date AS ends_after_release",
        "    FROM moved m",
        "    CROSS JOIN bounds b",
    ]
    if with_death:
        closed.append("    LEFT JOIN (SELECT person_id, MIN(death_date) AS death_date FROM {cdm}.death GROUP BY person_id) de"
                      " ON de.person_id = m.person_id")
    order = "ORDER BY c.s, c.e, c.id"
    if from_events:
        what = ("observation_period_merge: renumbers the periods derived from the records (they are never open) and "
                "derives the source release date ")
    else:
        what = ("observation_period_merge: each person's episodes become periods that neither overlap nor touch. "
                "Open episodes (no exit date) end at the source release date ")
    header = (
        what
        + ("(configured)" if release is not None else
           f"(derived: the k-th latest exit date of the valid episodes, k = max({RELEASE_RANK_MIN}, "
           f"n / {RELEASE_RANK_PER}))")
        + (", or at death before it (one day when they start after death)" if with_death else "")
        + "; real exit dates are kept. Episodes that end before they start are dropped. Overlapping and "
        "back-to-back episodes (DQD: an end + 1 day = the next start) merge into one period with the type of "
        "the first. Returns the statistics of the merge."
    )
    merge = [
        _comment(textwrap.fill(header, 100)),
        "WITH moved AS (",
        "    DELETE FROM {cdm}.observation_period",
        "    RETURNING person_id, observation_period_id AS id, observation_period_start_date AS s,",
        "              observation_period_end_date AS e, period_type_concept_id AS t,",
        f"              {is_open}",
        "),",
        "bounds AS (  -- the release date, computed once",
        *bounds,
        "),",
        "fact AS (",
        "    INSERT INTO {etl}.build_facts (name, value)",
        f"    SELECT {ql(RELEASE_FACT)}, release_date::text FROM bounds WHERE release_date IS NOT NULL",
        "    RETURNING 1",
        "),",
        "closed AS (",
        *closed,
        "),",
        "flagged AS (",
        "    SELECT c.*,",
        "           CASE WHEN c.s <= MAX(c.e) OVER w + 1 THEN 0 ELSE 1 END AS starts_island,  -- overlap or adjacent",
        "           COALESCE(c.s <= MAX(c.e) OVER w, false) AS overlaps_earlier,  -- a true overlap",
        f"           c.is_open AND LEAD(c.s) OVER (PARTITION BY c.person_id {order}) IS NOT NULL AS stale_open",
        "    FROM closed c",
        "    WHERE c.s <= c.e",
        f"    WINDOW w AS (PARTITION BY c.person_id {order} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)",
        "),",
        "islands AS (",
        "    SELECT f.*, SUM(f.starts_island) OVER (PARTITION BY f.person_id ORDER BY f.s, f.e, f.id"
        " ROWS UNBOUNDED PRECEDING) AS island",
        "    FROM flagged f",
        "),",
        "ins AS (",
        "    INSERT INTO {cdm}.observation_period (",
        "        observation_period_id,",
        "        person_id,",
        "        observation_period_start_date,",
        "        observation_period_end_date,",
        "        period_type_concept_id",
        "    )",
        "    SELECT (ROW_NUMBER() OVER (ORDER BY i.person_id, MIN(i.s)))::integer,",
        "           i.person_id, MIN(i.s), MAX(i.e), (ARRAY_AGG(i.t ORDER BY i.s, i.e, i.id))[1]",
        "    FROM islands i",
        "    GROUP BY i.person_id, i.island",
        "    RETURNING person_id",
        "),",
        "persons AS (SELECT COUNT(DISTINCT person_id) AS n FROM moved)  -- once: it sorts every episode",
        "SELECT (SELECT COUNT(*) FROM ins) AS inserted,",
        "       (SELECT COUNT(*) FROM moved) AS episodes,",
        "       (SELECT COUNT(*) FROM closed WHERE s > e) AS dropped_inverted,",
        "       (SELECT COUNT(*) FROM moved WHERE is_open) AS open_episodes,",
        "       (SELECT COUNT(*) FROM closed WHERE at_death) AS open_closed_at_death,",
        "       (SELECT COUNT(*) FROM closed WHERE starts_after_death) AS open_starting_after_death,",
        "       (SELECT COUNT(*) FROM closed WHERE starts_after_release) AS open_starting_after_release,",
        "       (SELECT COUNT(*) FROM closed WHERE ends_after_release) AS exits_after_release,",
        "       (SELECT COUNT(*) FROM closed c, bounds b WHERE c.s <= c.e AND c.e >= b.release_date)"
        " AS ending_from_release,",
        "       (SELECT COUNT(DISTINCT person_id) FROM flagged WHERE overlaps_earlier) AS persons_overlapping,",
        "       (SELECT COUNT(DISTINCT person_id) FROM flagged WHERE stale_open) AS persons_stale_open,",
        "       (SELECT n FROM persons) AS persons,",
        "       (SELECT n FROM persons) - (SELECT COUNT(DISTINCT person_id) FROM ins) AS persons_without_period,",
        "       (SELECT release_date FROM bounds)::text AS release_date,",
        "       (SELECT configured FROM bounds) AS configured,",
        "       (SELECT COUNT(*) FROM moved WHERE NOT is_open AND s <= e) AS closed_episodes,",
        "       (SELECT COUNT(*) FROM fact) AS fact_rows;",
    ]
    forget = f"DELETE FROM {{etl}}.build_facts WHERE name = {ql(RELEASE_FACT)};"
    return [forget, "\n".join(merge)]


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
            # the configured extract date, else the date the observation_period_merge step derived,
            # else (no observation periods) the build date
            f"    DATE {ql(cs.source_release_date.isoformat())}," if cs.source_release_date else
            f"    COALESCE((SELECT MAX(value::date) FROM {{etl}}.build_facts WHERE name = {ql(RELEASE_FACT)}),"
            " CURRENT_DATE),",
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
# eras
# --------------------------------------------------------------------------------------

# Records of one person and concept that start at most this many days after the earlier ones
# ended form one era (the OHDSI persistence window).
ERA_PAD_DAYS = 30
# The ingredients a drug exposure rolls up to (concept_ancestor): the standard Ingredient concepts
# of these vocabularies. The published script reads RxNorm only, which leaves the drugs that exist
# only in RxNorm Extension (many products marketed outside the US) without an era.
ERA_INGREDIENT_VOCABULARIES = ("RxNorm", "RxNorm Extension")
# A drug exposure that ends before it starts ends after its days_supply when that is plausible.
ERA_MAX_DAYS_SUPPLY = 365
# Spec SQL that loads an era table whatever its target_table says: an INSERT INTO the era table in
# {cdm}, in a schema named as written, or unqualified (not in {src}, {vocab}, {etl} or {results}).
_ERA_INSERT = re.compile(r"\binsert\s+into\s+(?:(?:\{cdm\}|\w+)\s*\.\s*)?(condition_era|drug_era)\b", re.I)


def era_inserts(sql: str) -> set[str]:
    """The era tables that custom SQL ``sql`` inserts into (``INSERT INTO {cdm}.drug_era``, also
    as ``cdm.drug_era`` or ``drug_era``), not counting text in string literals or comments."""
    return {m.group(1).lower() for m in _ERA_INSERT.finditer(code_text(sql))}


def spec_era_tables(spec: ETLSpec) -> set[str]:
    """The era tables the spec loads itself: the target of a mapping or custom SQL, or a table that
    custom SQL inserts into (``era_inserts``). Its SQL replaces the built-in step for that table
    and runs at that table's place (``compile_spec``)."""
    out = {t for t in (*(tm.target_table.lower() for tm in spec.table_mappings),
                       *(cs.target_table.lower() for cs in spec.custom_sql)) if t in DERIVED_ERAS}
    for cs in spec.custom_sql:
        out |= era_inserts(cs.sql)
    return out


def derived_era_tables(spec: ETLSpec, loaded: set[str] | list[str] | None = None) -> list[str]:
    """The era tables the build derives (``DERIVED_ERAS``), with ``derive_eras``: those whose event
    table is in ``loaded`` (default: the tables the spec writes and the route destinations) and that
    no spec SQL loads (``spec_era_tables``)."""
    if not spec.derive_eras:
        return []
    if loaded is None:
        written = {tm.target_table.lower() for tm in spec.table_mappings} | {cs.target_table.lower()
                                                                            for cs in spec.custom_sql}
        loaded = written | set(route_targets(spec))
    built = spec_era_tables(spec)
    return [era for era, source in DERIVED_ERAS.items() if source in loaded and era not in built]


def era_status(spec: ETLSpec) -> dict[str, str]:
    """How each era table of ``DERIVED_ERAS`` is built: 'derived' (the build's OHDSI era step),
    'spec' (the spec's own SQL), 'off' (``derive_eras: false``) or 'no_records' (its event table
    is not loaded)."""
    derived, own = derived_era_tables(spec), spec_era_tables(spec)
    return {era: "derived" if era in derived else "spec" if era in own else "off" if not spec.derive_eras
            else "no_records" for era in DERIVED_ERAS}


def _era_islands(gap_days: bool) -> list[str]:
    """The CTEs ``marked``, ``numbered`` and ``eras`` over a CTE ``target`` (rid, person_id,
    concept_id, start_date, end_date): the records of a person and concept, in order of start, end
    and a unique record id (so both windows see one order, also among ties), start a new era when
    they begin more than ``ERA_PAD_DAYS`` days after every earlier record ended; an era runs from
    its earliest start to its latest end. With ``gap_days`` the days between those records that no
    record covers are summed."""
    lines = [
        "marked AS (",
        "    -- the latest end of the earlier records of the person and concept",
        "    SELECT t.*,",
        "           MAX(t.end_date) OVER (PARTITION BY t.person_id, t.concept_id ORDER BY t.start_date, t.end_date, t.rid",
        "                                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end",
        "    FROM target t",
        "),",
        "numbered AS (",
        f"    -- a record starts a new era when it begins more than {ERA_PAD_DAYS} days after every earlier one ended",
        "    SELECT m.*,",
        f"           SUM(CASE WHEN m.start_date <= m.prev_end + {ERA_PAD_DAYS} THEN 0 ELSE 1 END)",
        "               OVER (PARTITION BY m.person_id, m.concept_id ORDER BY m.start_date, m.end_date, m.rid",
        "                     ROWS UNBOUNDED PRECEDING) AS era_no",
        "    FROM marked m",
        "),",
        "eras AS (",
        "    SELECT person_id, concept_id, MIN(start_date) AS start_date, MAX(end_date) AS end_date, COUNT(*) AS n"
        + ("," if gap_days else ""),
    ]
    if gap_days:
        lines.append(f"           SUM(CASE WHEN start_date <= prev_end + {ERA_PAD_DAYS} THEN GREATEST(start_date - prev_end, 0)"
                     " ELSE 0 END) AS gap_days  -- days no record covers")
    return lines + ["    FROM numbered", "    GROUP BY person_id, concept_id, era_no", ")"]


def _condition_era_sql() -> str:
    """The condition_era step: the OHDSI condition eras."""
    header = (
        "condition_era: the OHDSI condition eras (CommonDataModel / ETL-Synthea condition era script): the condition "
        f"records of one person and condition_concept_id that start at most {ERA_PAD_DAYS} days after the earlier "
        "ones ended form one era, from the earliest start to the latest end. A record without an end date, or "
        "with an end before its start, ends the day after it starts. Records with condition_concept_id 0 get no "
        "era. Ties are broken by condition_occurrence_id."
    )
    return "\n".join([
        _comment(textwrap.fill(header, 100)),
        "INSERT INTO {cdm}.condition_era (",
        "    condition_era_id,",
        "    person_id,",
        "    condition_concept_id,",
        "    condition_era_start_date,",
        "    condition_era_end_date,",
        "    condition_occurrence_count",
        ")",
        "WITH target AS (",
        "    SELECT co.condition_occurrence_id AS rid,",
        "           co.person_id,",
        "           co.condition_concept_id AS concept_id,",
        "           co.condition_start_date AS start_date,",
        "           CASE WHEN co.condition_end_date >= co.condition_start_date THEN co.condition_end_date",
        "                ELSE co.condition_start_date + 1 END AS end_date",
        "    FROM {cdm}.condition_occurrence co",
        "    WHERE co.condition_concept_id <> 0",
        "),",
        *_era_islands(gap_days=False),
        "SELECT",
        "    ((SELECT COALESCE(MAX(condition_era_id), 0) FROM {cdm}.condition_era)",
        "        + ROW_NUMBER() OVER (ORDER BY e.person_id, e.concept_id, e.start_date))::integer,",
        "    e.person_id,",
        "    e.concept_id,",
        "    e.start_date,",
        "    e.end_date,",
        "    e.n::integer",
        "FROM eras e;",
    ])


def _drug_era_sql() -> str:
    """The drug_era step: the OHDSI drug eras (non-stockpiling) per ingredient."""
    vocabularies = ", ".join(ql(v) for v in ERA_INGREDIENT_VOCABULARIES)
    header = (
        "drug_era: the OHDSI drug eras (CommonDataModel / ETL-Synthea drug era script, non-stockpiling). Each drug "
        "exposure rolls up to its ingredients (concept_ancestor: the standard Ingredient concepts of "
        f"{' and '.join(ERA_INGREDIENT_VOCABULARIES)}); the exposures of one person and ingredient that start at "
        f"most {ERA_PAD_DAYS} days after the earlier ones ended form one era, from the earliest start to the latest "
        "end, and gap_days counts the days of the era no exposure covers. An exposure that ends before it starts "
        f"ends after its days_supply (start + days_supply - 1, THEMIS) when that is 1 to {ERA_MAX_DAYS_SUPPLY} days, "
        "else on its start. Exposures with drug_concept_id 0 or a negative days_supply, and drugs without such an "
        "ingredient, get no era. Ties are broken by drug_exposure_id."
    )
    return "\n".join([
        _comment(textwrap.fill(header, 100)),
        "INSERT INTO {cdm}.drug_era (",
        "    drug_era_id,",
        "    person_id,",
        "    drug_concept_id,",
        "    drug_era_start_date,",
        "    drug_era_end_date,",
        "    drug_exposure_count,",
        "    gap_days",
        ")",
        "WITH ingredient AS (",
        "    SELECT ca.descendant_concept_id AS drug_concept_id, ca.ancestor_concept_id AS ingredient_concept_id",
        "    FROM {vocab}.concept_ancestor ca",
        "    JOIN {vocab}.concept c ON c.concept_id = ca.ancestor_concept_id",
        "    WHERE c.concept_class_id = 'Ingredient'",
        f"      AND c.vocabulary_id IN ({vocabularies})",
        "      AND c.standard_concept = 'S'",
        "      AND ca.descendant_concept_id IN (SELECT de.drug_concept_id FROM {cdm}.drug_exposure de"
        " WHERE de.drug_concept_id <> 0)",
        "),",
        "target AS (",
        "    SELECT d.drug_exposure_id AS rid,",
        "           d.person_id,",
        "           i.ingredient_concept_id AS concept_id,",
        "           d.drug_exposure_start_date AS start_date,",
        "           CASE WHEN d.drug_exposure_end_date >= d.drug_exposure_start_date THEN d.drug_exposure_end_date",
        f"                WHEN d.days_supply BETWEEN 1 AND {ERA_MAX_DAYS_SUPPLY}"
        " THEN d.drug_exposure_start_date + d.days_supply - 1",
        "                ELSE d.drug_exposure_start_date END AS end_date",
        "    FROM {cdm}.drug_exposure d",
        "    JOIN ingredient i ON i.drug_concept_id = d.drug_concept_id",
        "    WHERE d.drug_concept_id <> 0",
        "      AND COALESCE(d.days_supply, 0) >= 0",
        "),",
        *_era_islands(gap_days=True),
        "SELECT",
        "    ((SELECT COALESCE(MAX(drug_era_id), 0) FROM {cdm}.drug_era)",
        "        + ROW_NUMBER() OVER (ORDER BY e.person_id, e.concept_id, e.start_date))::integer,",
        "    e.person_id,",
        "    e.concept_id,",
        "    e.start_date,",
        "    e.end_date,",
        "    e.n::integer,",
        "    e.gap_days::integer",
        "FROM eras e;",
    ])


_ERA_SQL: dict[str, Callable[[], str]] = {"condition_era": _condition_era_sql, "drug_era": _drug_era_sql}


# --------------------------------------------------------------------------------------
# whole spec
# --------------------------------------------------------------------------------------


def _load_index(table: str) -> int:
    return LOAD_ORDER.index(table) if table in LOAD_ORDER else len(LOAD_ORDER)


def _is_cdm_table(name: str) -> bool:
    spec = load_cdm_spec()
    return spec.has_table(name) and spec.table(name).schema_class == "CDM"


def _in_load_order(names: set[str] | list[str]) -> list[str]:
    """The CDM tables of ``names`` in load order."""
    return sorted({n for n in names if _is_cdm_table(n)}, key=lambda n: (_load_index(n), n))


def target_tables(spec: ETLSpec) -> list[str]:
    """Every CDM table the compiled spec writes (incl. observation_period, the era tables the build
    derives or the spec's SQL loads, and cdm_source), in load order. The tables only the route
    steps write are not in it (``etl.routing.route_targets``): a route destination that received no
    record is not a populated table."""
    names = {tm.target_table.lower() for tm in spec.table_mappings}
    names |= {cs.target_table.lower() for cs in spec.custom_sql} | spec_era_tables(spec)
    if spec.observation_period == "from_events":
        names.add("observation_period")
    names.add("cdm_source")
    names |= set(derived_era_tables(spec, names | set(route_targets(spec))))
    return _in_load_order(names)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower() or "step"


def compile_spec(
    spec: ETLSpec,
    schemas: Schemas,
    cdm_source: CdmSourceConfig | None = None,
    *,
    etl: EtlConfig | None = None,
    vocab_indexed: bool = True,
    strict: bool = True,
) -> list[CompiledStep]:
    """Compile the whole spec into ordered steps (see the module docstring for the order).
    ``etl`` (``ProjectConfig.etl``) gives the partial-date rule of the tolerant conversions;
    ``vocab_indexed`` (``etl.routing.vocab_indexed``) says whether the route steps can look the
    concepts up by id or must join the concept table.

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
    for i, (tm, label) in enumerate(zip(spec.table_mappings, mapping_labels(spec))):
        t = tm.target_table.lower()
        seen_count[t] = seen_count.get(t, 0) + 1
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
        # custom SQL that loads an era table runs at that table's place, after routing and the
        # observation periods (it replaces the era step), also when its target_table comes earlier
        t = max([cs.target_table.lower(), *sorted(era_inserts(cs.sql))], key=_load_index)
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

    # the route block, after every writer of the event tables (specimen is the last of them) and
    # before payer_plan_period, cost, the observation periods and the eras, which read or refer to
    # the records where they end up. Not blocking: on an error the records stay where they were loaded
    sources = route_sources(spec)
    if sources:
        route_index = _load_index("specimen")
        rp = route_prepare_sql()
        steps.append(CompiledStep(ROUTE_PREPARE, None, "route", "\n".join(rp), statements=tuple(rp), blocking=False,
                                  slot=(route_index, 2, 0)))
        for i, src in enumerate(sources, start=1):
            steps.append(CompiledStep(f"route_{src}", src, "route", route_sql(src, vocab_indexed), blocking=False,
                                      requires=ROUTE_PREPARE, returns="rows", slot=(route_index, 2, i)))

    # observation periods: from the events (after every spec writer of observation_period), then the
    # merge of each person's periods after the last writer (and after death, which ends open
    # episodes); with 'table_mapping' the fill for persons left without a period. Both read the
    # event tables as routed (the route destinations too)
    op_index = _load_index("observation_period")
    loaded = [t for t in _in_load_order(set(target_tables(spec)) | set(route_targets(spec)))
              if t != "observation_period"]
    if spec.observation_period == "from_events":
        op = _observation_period_sql(spec, loaded)
        if op is not None:
            steps.append(CompiledStep("observation_period", "observation_period", "derived", op,
                                      slot=(op_index, 0, _AFTER_SPEC_WRITERS)))
    if any(s.target_table == "observation_period" for s in steps):
        # periods derived from the records only (from_events without a spec writer) are never open
        derived_only = all(s.kind == "derived" for s in steps if s.target_table == "observation_period")
        merge = _observation_period_merge_sql(cdm_source, with_death="death" in loaded, from_events=derived_only)
        steps.append(CompiledStep("observation_period_merge", "observation_period", "merge", "\n".join(merge),
                                  statements=tuple(merge), returns="row", slot=(op_index, 1, 0)))
        fill = _observation_period_sql(spec, loaded, fill=True) if spec.observation_period == "table_mapping" else None
        if fill is not None:
            steps.append(CompiledStep("observation_period_fill", "observation_period", "derived", fill,
                                      returns="row", slot=(op_index, 1, 1)))

    # the eras, at their tables' places: after routing (the records where they end up, deduplicated)
    # and after the observation periods. Each needs its event table loaded; not blocking: an error
    # (e.g. no concept_ancestor) is reported and the build goes on without that era table
    for era in derived_era_tables(spec, loaded):
        steps.append(CompiledStep(era, era, "derived", _ERA_SQL[era](), requires=DERIVED_ERAS[era], blocking=False,
                                  slot=(_load_index(era), 0, 0)))

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
