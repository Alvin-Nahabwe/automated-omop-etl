"""Move records to the table of their concept's domain (the route steps).

The CDM puts a record in the table of the domain of its standard concept: a diagnosis list
coded in ICD-10 holds conditions, but also observations (Z34.9 "supervision of normal
pregnancy"), procedures and lab findings, and a lab list holds observations. A spec maps each
source table to one CDM table, so after every writer of the event tables (``compile_spec`` puts
the route block after specimen, before payer_plan_period, cost, the observation periods and the
eras) the build moves every condition_occurrence, procedure_occurrence, drug_exposure,
device_exposure and measurement record (``ROUTE_SOURCES``) whose standard concept belongs to
another event domain (``DOMAIN_TABLE``: Condition, Procedure, Drug, Device, Measurement,
Observation) to that domain's table (``ETLSpec.route_by_domain``, on by default):

* only the standard concept (``*_concept_id``) decides. A record with concept 0 stays where it
  was loaded, and so does one whose concept is in a domain without an event table (Meas Value,
  Unit, Type Concept, ...): that is a mapping error, which fkDomain keeps reporting;
* observation is a destination only (DQD has no fkDomain on observation_concept_id);
* a measured result is never dropped: a measurement record with a value (``value_as_number``,
  ``value_as_concept_id`` or ``value_source_value``) whose concept is a condition, procedure,
  drug or device goes to observation, which has a place for the value;
* the fields are carried by role (``cdm.spec.EVENT_FIELD_ROLES``, ``pair_fields``):
  condition_start_date becomes observation_date, and so on. A value in a field the destination
  has no place for is dropped, and the moved rows that had one are counted
  (``rows_with_dropped_fields``; an end on the start day counts as no end). drug_exposure_end_date
  takes the start date when the record
  has no end (the CDM convention), and a quantity becomes an integer only when it is one (a
  fraction, NaN or a number out of range gives NULL and counts as dropped);
* a moved record gets a new id after the largest id its destination had before any record was
  moved (``route_prepare`` stores it in ``{etl}.id_high_water``) or has now, so an id that a
  moved record freed is never given to another record.

Each route step is one statement: it deletes the misplaced records of its table and inserts
them into their destinations, and returns one row per destination and domain (``RoutedRows``).
Route steps are not blocking: an error is reported (etlStepError, convention) but the build
goes on with the records where they were loaded. The misplaced concepts are found from the
distinct concepts of the table: with an index on ``{vocab}.concept (concept_id)`` by an
indexed lookup of that list, without one by a hash join (a lookup of a list without the index
reads the whole concept table for every value); the runner probes the index
(``vocab_indexed``).

The mapping side (``vocab.mapper``) lets a local code of the main concept field of a routed
table map to a concept of the other domains of ``ACCEPTED_DOMAINS``, its own first
(``accepted_domains_for``); fields that share a source vocabulary share one set
(``vocabulary_domains``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import psycopg

from ..cdm.spec import (
    ACCEPTED_DOMAINS,
    DOMAIN_TABLE,
    EVENT_FIELD_ROLES,
    ROUTE_SOURCES,
    ROUTE_TABLES,
    VALUE_ROLES,
    CdmField,
    load_cdm_spec,
)
from ..db import Schemas, qi, ql, scalar
from ..models import ETLSpec, FieldMapping, TableMapping
from ..sqlguard import strip_literals

ROUTE_PREPARE = "route_prepare"
_TABLE_DOMAIN = {t: d for d, t in DOMAIN_TABLE.items()}
_INTEGER_TYPES = ("integer", "int", "int4")
_INT_MIN, _INT_MAX = -2147483648, 2147483647


def _written(spec: ETLSpec) -> set[str]:
    """The CDM tables a mapping or custom SQL of the spec loads."""
    return {tm.target_table.lower() for tm in spec.table_mappings} | {cs.target_table.lower() for cs in spec.custom_sql}


def route_sources(spec: ETLSpec) -> list[str]:
    """The route sources (``ROUTE_SOURCES``) the spec writes, by a mapping or custom SQL, when it
    routes by domain; one route step each, in this order."""
    if not spec.route_by_domain:
        return []
    written = _written(spec)
    return [t for t in ROUTE_SOURCES if t in written]


def route_targets(spec: ETLSpec) -> list[str]:
    """Every table the route steps can write (``ROUTE_TABLES``) when there is a route step. They
    are emptied before each build, but only those with rows count as populated (a destination
    that received nothing is not checked)."""
    return list(ROUTE_TABLES) if route_sources(spec) else []


def route_only_tables(spec: ETLSpec) -> set[str]:
    """The route destinations that no mapping or custom SQL of the spec writes: their records are
    all moved there by domain, so few persons have any, and no change of the spec can give the
    others one (measurePersonCompleteness says nothing about the spec there)."""
    return set(route_targets(spec)) - _written(spec)


# --------------------------------------------------------------------------------------
# the moved fields
# --------------------------------------------------------------------------------------


@dataclass
class Pairing:
    """How the records of one route table are written into another (``pair_fields``)."""

    values: list[tuple[str, str]] = field(default_factory=list)  # (destination field, SQL over the moved row m)
    # SQL conditions over m, each true when a value of the record has no place in the destination
    # (a field without a counterpart, a quantity that is not an integer, text that is too long)
    lost: list[str] = field(default_factory=list)


def _width(f: CdmField) -> int | None:
    """n of a varchar(n) field; None for unbounded text and other types."""
    m = re.fullmatch(r"varchar\((\d+)\)", f.datatype.strip().lower())
    return int(m.group(1)) if m else None


def _convert(value: str, src: CdmField, dst: CdmField) -> tuple[str, str | None]:
    """``value`` (of ``src``) as a value of ``dst``, and the condition under which it is lost: a
    quantity that is not an integer becomes NULL in an integer field (a fraction, NaN or a number
    out of range: NaN fails the range test, so the cast never fails the step); text longer than
    the destination is cut."""
    s, d = src.datatype.strip().lower(), dst.datatype.strip().lower()
    if d in _INTEGER_TYPES and s not in _INTEGER_TYPES:
        guarded = (f"CASE WHEN {value} = trunc({value}) AND {value} BETWEEN {_INT_MIN} AND {_INT_MAX}"
                   f" THEN ({value})::integer END")
        return guarded, f"({value} IS NOT NULL AND {guarded} IS NULL)"
    n, m = _width(dst), _width(src)
    if n is not None and (m is None or m > n):
        return f"LEFT({value}, {n})", f"length({value}) > {n}"
    return value, None


def pair_fields(src: str, dst: str) -> Pairing:
    """The fields of a record of route table ``src`` written into route table ``dst`` (both in
    ``ROUTE_TABLES``), by role (``EVENT_FIELD_ROLES``), without the destination's id (the route
    step numbers it). drug_exposure_end_date, which the CDM requires, takes the start date when
    the record has none. ``lost`` lists the conditions under which a value of the record is
    dropped on the way."""
    cdm = load_cdm_spec()
    sr, dr = EVENT_FIELD_ROLES[src], EVENT_FIELD_ROLES[dst]
    src_table, dst_table = cdm.table(src), cdm.table(dst)
    role_of_dst = {f: r for r, f in dr.items()}
    start = f"m.{qi(sr['start_date'])}"
    out = Pairing()
    for f in dst_table.fields:
        role = role_of_dst.get(f.name)
        end_from_start = (dst, f.name) == ("drug_exposure", "drug_exposure_end_date")
        if role is None or role == "id" or role not in sr:
            if end_from_start:  # required: a record without an end ends on its start
                out.values.append((f.name, start))
            continue
        sf = src_table.field(sr[role])
        assert sf is not None, (src, role)
        value, lost = _convert(f"m.{qi(sf.name)}", sf, f)
        out.values.append((f.name, f"COALESCE({value}, {start})" if end_from_start else value))
        if lost:
            out.lost.append(lost)
    role_of_src = {f: r for r, f in sr.items()}
    start_dt = f"COALESCE(m.{qi(sr['start_datetime'])}, {start}::timestamp)"
    for f in src_table.fields:
        role = role_of_src.get(f.name)
        if role != "id" and (role is None or role not in dr):
            # an end on the start (drug_exposure_end_date's default) says nothing the start does not
            same = {"end_date": start, "end_datetime": start_dt}.get(role or "")
            out.lost.append(f"m.{qi(f.name)} IS NOT NULL" + (f" AND m.{qi(f.name)} <> {same}" if same else ""))
    return out


# --------------------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------------------


def route_prepare_sql() -> list[str]:
    """The route_prepare step, two statements: the largest id of every route table before any
    record is moved, in ``{etl}.id_high_water`` (a moved record gets an id above it, so the ids
    the moves free are never given again)."""
    tables = ", ".join(ql(t) for t in ROUTE_TABLES)
    rows = [
        f"SELECT {ql(t)}, COALESCE(MAX({qi(EVENT_FIELD_ROLES[t]['id'])}), 0) FROM {{cdm}}.{qi(t)}" for t in ROUTE_TABLES
    ]
    forget = "\n".join([
        "-- route_prepare: the largest id of each route table before any record moves to the table of its",
        "-- concept's domain; moved records get ids above it, so an id a move frees is never given again",
        f"DELETE FROM {{etl}}.id_high_water WHERE table_name IN ({tables});",
    ])
    insert = "\n".join(["INSERT INTO {etl}.id_high_water (table_name, max_id)", rows[0],
                        *(f"UNION ALL {r}" for r in rows[1:])]) + ";"
    return [forget, insert]


def _destination(src: str) -> str:
    """The route_table of a moved row of ``src`` (its concept's domain is ``c.domain_id``): the
    table of the domain, but observation for a record with a result when the domain's table has
    no place for one (a measured result is never dropped)."""
    by_domain = "".join(f"\n                     WHEN {ql(d)} THEN {ql(t)}" for d, t in DOMAIN_TABLE.items() if t != src)
    plain = f"CASE c.domain_id{by_domain} END"
    sr = EVENT_FIELD_ROLES[src]
    values = [sr[r] for r in VALUE_ROLES if r in sr]
    if not values:
        return plain
    holds_values = [d for d, t in DOMAIN_TABLE.items() if any(r in EVENT_FIELD_ROLES[t] for r in VALUE_ROLES)]
    has_value = " OR ".join(f"t.{qi(v)} IS NOT NULL" for v in values)
    return (f"CASE WHEN c.domain_id NOT IN ({', '.join(ql(d) for d in holds_values)})"
            f"\n                     AND ({has_value})"
            "\n                THEN 'observation'  -- a measured result is never dropped"
            f"\n                ELSE {plain} END")


def route_sql(src: str, vocab_indexed: bool = True) -> str:
    """The route step of ``src``: one statement that deletes the records whose standard concept
    belongs to another event domain, inserts them into the tables of their domains (new ids, see
    ``route_prepare_sql``) and returns one row per destination and domain: ``to_table``,
    ``domain_id``, ``moved_rows`` and ``rows_with_dropped_fields``. ``vocab_indexed``: the
    misplaced concepts are looked up by id (needs an index on ``{vocab}.concept (concept_id)``),
    else found by a hash join."""
    sr = EVENT_FIELD_ROLES[src]
    own = _TABLE_DOMAIN[src]
    concept, pk = qi(sr["concept"]), qi(sr["id"])
    domains = f"c.domain_id IN ({', '.join(ql(d) for d in DOMAIN_TABLE if d != own)})  -- the event domains but {own}"
    used = f"FROM {{cdm}}.{qi(src)} t WHERE t.{concept} <> 0"
    if vocab_indexed:
        misplaced = [
            "    SELECT c.concept_id, c.domain_id",
            "    FROM {vocab}.concept c",
            f"    WHERE c.concept_id = ANY(ARRAY(SELECT DISTINCT t.{concept} {used}))  -- by the index on concept_id",
            f"      AND {domains}",
        ]
    else:  # no index on concept_id: a lookup of the list would read the concept table once per value
        misplaced = [
            "    SELECT c.concept_id, c.domain_id",
            f"    FROM (SELECT DISTINCT t.{concept} AS id {used}) d",
            "    JOIN {vocab}.concept c ON c.concept_id = d.id",
            f"    WHERE {domains}",
        ]
    lines = [
        f"-- route_{src}: {src} records whose standard concept belongs to another event domain move to that",
        "-- domain's table (concept 0 and concepts of other domains stay); returns the records moved per",
        "-- destination and domain",
        "WITH misplaced AS MATERIALIZED (",
        *misplaced,
        "),",
        "moved AS (",
        f"    DELETE FROM {{cdm}}.{qi(src)} t USING misplaced c",
        f"    WHERE c.concept_id = t.{concept}",
        f"    RETURNING t.*, c.domain_id AS route_domain,\n           {_destination(src)} AS route_table",
        ")",
    ]
    lost_by_table: list[str] = []
    for dst in ROUTE_TABLES:
        if dst == src:
            continue
        pairing = pair_fields(src, dst)
        dr = EVENT_FIELD_ROLES[dst]
        cols = [dr["id"]] + [n for n, _ in pairing.values]
        new_id = (f"(GREATEST((SELECT COALESCE(MAX({qi(dr['id'])}), 0) FROM {{cdm}}.{qi(dst)}),\n"
                  f"                  (SELECT max_id FROM {{etl}}.id_high_water WHERE table_name = {ql(dst)}))\n"
                  f"         + ROW_NUMBER() OVER (ORDER BY m.{pk}))::integer")
        values = [new_id] + [v for _, v in pairing.values]
        lines[-1] += ","
        lines += [
            f"to_{dst} AS (",
            f"    INSERT INTO {{cdm}}.{qi(dst)} (",
            *(f"        {qi(c)}{',' if i < len(cols) - 1 else ''}" for i, c in enumerate(cols)),
            "    )",
            "    SELECT",
            *(f"        {v}{',' if i < len(values) - 1 else ''}" for i, v in enumerate(values)),
            "    FROM moved m",
            f"    WHERE m.route_table = {ql(dst)}",
            "    RETURNING 1",
            ")",
        ]
        if pairing.lost:
            lost_by_table.append(f"WHEN {ql(dst)} THEN " + "\n                    OR ".join(pairing.lost))
    lost = (["CASE m.route_table", *(f"                {x}" for x in lost_by_table), "                ELSE false END"]
            if lost_by_table else ["false"])
    lines += [
        "SELECT m.route_table AS to_table, m.route_domain AS domain_id, COUNT(*) AS moved_rows,",
        "       COUNT(*) FILTER (WHERE  -- a value the destination has no place for",
        *(f"           {x.strip()}" if i == 0 else x for i, x in enumerate(lost)),
        "       ) AS rows_with_dropped_fields",
        "FROM moved m",
        "GROUP BY 1, 2",
        "ORDER BY 1, 2;",
    ]
    return "\n".join(lines)


def vocab_indexed(conn: psycopg.Connection, schemas: Schemas) -> bool:
    """True when ``{vocab}.concept`` has an index whose first column is concept_id (the route
    steps then look the misplaced concepts up by id). A probe that fails counts as no index."""
    sql = (
        "SELECT 1 FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0] "
        "WHERE i.indrelid = to_regclass(%s) AND a.attname = 'concept_id' LIMIT 1"
    )
    try:
        with conn.transaction():
            return bool(scalar(conn, sql, (f"{qi(schemas.vocab)}.concept",)))
    except psycopg.Error:
        return False


# --------------------------------------------------------------------------------------
# routing by hand, which loads records twice while the build routes them
# --------------------------------------------------------------------------------------

_VOCAB_CONCEPT = re.compile(r"\{vocab\}\s*\.\s*\"?concept(?:_relationship)?\b", re.I)
_DOMAIN_ID = re.compile(r"\bdomain_id\b", re.I)
_EVENT_DOMAIN = re.compile("'(?:" + "|".join(DOMAIN_TABLE) + ")'", re.I)


def hand_routing_issue(tm: TableMapping, spec: ETLSpec) -> str | None:
    """Why ``tm`` routes records by the domain of their concept itself while the build routes every
    event table by domain (None: it does not): a condition that can remove rows (the WHERE, or the
    ON of an inner join) tests domain_id of {vocab}.concept or concept_relationship against an
    event domain ('Observation', ...). The build moves the records of the other domains again,
    so the mapping loads them twice (or, with a filter that drops them, the build had moved them
    already). A LEFT join's condition only blanks a lookup, and a test against another domain
    (a unit looked up with domain_id = 'Unit') routes nothing, so neither counts."""
    if not spec.route_by_domain or tm.target_table.lower() not in ROUTE_TABLES:
        return None
    raw = " ".join([tm.where or "", *(j.on for j in tm.joins if j.join_type == "inner")])
    filters = strip_literals(raw)
    reads = _VOCAB_CONCEPT.search(filters) or any(_VOCAB_CONCEPT.search(j.table) for j in tm.joins)
    if not reads or not _DOMAIN_ID.search(filters) or not _EVENT_DOMAIN.search(raw):
        return None
    return ("it filters records by the domain of their concept ({vocab}.concept domain_id): the build already moves "
            "every condition_occurrence, procedure_occurrence, drug_exposure, device_exposure and measurement record "
            "whose standard concept belongs to another domain to that domain's table, so routing by hand loads such "
            "records twice. Map the whole source table into one table and let the build route its records (or set "
            "route_by_domain: false)")


@dataclass
class SharedCodes:
    """Two mappings of one source table into two event tables that load the same codes of the same
    rows (``shared_codes``)."""

    key: tuple[str, str, str]  # (source table, the two target tables in order), lower case
    index: int  # the later mapping's position in spec.table_mappings
    message: str


def _main_code(tm: TableMapping) -> tuple[str, str] | None:
    """(the main concept field's code expression, the same with the source alias neutralised), for
    a mapping into an event table; None without one."""
    role = EVENT_FIELD_ROLES.get(tm.target_table.lower(), {}).get("concept")
    fm = next((f for f in tm.fields if f.target_field == role and f.kind in ("concept_map", "vocab_code")
               and (f.expression or "").strip()), None)
    if fm is None:
        return None
    expr = fm.expression or ""
    return " ".join(expr.split()), _neutral(tm, expr)


def _neutral(tm: TableMapping, sql: str) -> str:
    """``sql`` with the source alias neutralised, whitespace collapsed, lower case."""
    return " ".join(re.sub(rf"\b{re.escape(tm.source.alias)}\s*\.", "@.", sql, flags=re.I).split()).lower()


def _row_filter(tm: TableMapping) -> tuple[str, tuple[tuple[str, str], ...]]:
    """What decides which source rows ``tm`` loads: its WHERE and its inner joins (empty: every row)."""
    return (_neutral(tm, tm.where or ""),
            tuple(sorted((j.table.lower(), _neutral(tm, j.on)) for j in tm.joins if j.join_type == "inner")))


def _pair_key(a: TableMapping, b: TableMapping) -> tuple[str, str, str]:
    x, y = sorted((a.target_table.lower(), b.target_table.lower()))
    return a.source.table.lower(), x, y


def source_table_pairs(spec: ETLSpec) -> set[tuple[str, str, str]]:
    """(source table, target table, target table) of every two mappings of one source table into
    two different event tables (``ROUTE_TABLES``), whatever they read."""
    tms = [tm for tm in spec.table_mappings if tm.target_table.lower() in ROUTE_TABLES]
    return {_pair_key(a, b) for i, a in enumerate(tms) for b in tms[i + 1:]
            if a.source.table.lower() == b.source.table.lower() and a.target_table.lower() != b.target_table.lower()}


def shared_codes(spec: ETLSpec) -> list[SharedCodes]:
    """Two mappings of one source table into two event tables (``ROUTE_TABLES``) whose main
    concept fields read the same codes from the same rows, when the build routes by domain: each
    of those records is loaded into both tables (the hand-made way to route by domain), and the
    build moves them by their concept's domain on top of that. The rows are the same when both
    mappings filter them alike (or not at all), or when one of them loads every row of the source
    table (no WHERE, no inner join); mappings that filter differently (numeric results into
    measurement, text answers into observation) may split the rows, and are left alone."""
    if not spec.route_by_domain:
        return []
    out: list[SharedCodes] = []
    seen: list[tuple[TableMapping, str]] = []
    for i, tm in enumerate(spec.table_mappings):
        c = _main_code(tm) if tm.target_table.lower() in ROUTE_TABLES else None
        if c is None:
            continue
        for other, code in seen:
            if (code != c[1] or other.source.table.lower() != tm.source.table.lower()
                    or other.target_table.lower() == tm.target_table.lower()):
                continue
            fa, fb = _row_filter(other), _row_filter(tm)
            everything = ("", ())
            if fa == fb:
                rows = "from the same rows"
            elif everything in (fa, fb):
                wide, narrow = (other, tm) if fa == everything else (tm, other)
                rows = f"and the mapping into {wide.target_table} loads every row the one into {narrow.target_table} loads"
            else:
                continue
            out.append(SharedCodes(_pair_key(other, tm), i, (
                f"the mappings from {tm.source.table} into {other.target_table} and {tm.target_table} read the same "
                f"codes ({c[0]}) {rows}: each of those records is loaded into both tables. Map the source table "
                "into one table: the build moves each record to the table of its concept's domain")))
            break
        seen.append((tm, c[1]))
    return out


def same_table_twin(spec: ETLSpec, new: TableMapping) -> int | None:
    """The position in ``spec.table_mappings`` of a mapping into the same event table, from the same
    source table, whose main concept field reads the same codes as ``new`` (None: none). Whatever
    their filters, a source row both select is loaded twice; that they never overlap cannot be told
    from the SQL. ``shared_codes`` judges pairs of different tables."""
    c = _main_code(new) if new.target_table.lower() in EVENT_FIELD_ROLES else None
    if c is None:
        return None
    for i, tm in enumerate(spec.table_mappings):
        if (tm is not new and tm.target_table.lower() == new.target_table.lower()
                and tm.source.table.lower() == new.source.table.lower() and (_main_code(tm) or ("", ""))[1] == c[1]):
            return i
    return None


def hand_routed(spec: ETLSpec) -> list[tuple[int, str]]:
    """(position in spec.table_mappings, why) of every mapping that routes records by hand while
    the build routes them (``hand_routing_issue``, ``shared_codes``): its records are loaded twice."""
    out = [(i, issue) for i, tm in enumerate(spec.table_mappings) if (issue := hand_routing_issue(tm, spec))]
    done = {i for i, _ in out}
    return sorted(out + [(p.index, p.message) for p in shared_codes(spec) if p.index not in done])


# --------------------------------------------------------------------------------------
# the mapping side: the domains a concept_map target may belong to
# --------------------------------------------------------------------------------------


def _canonical(domain: str) -> str:
    """The spelling of an event domain ('observation' -> 'Observation'); other domains as given."""
    return next((d for d in DOMAIN_TABLE if d.lower() == domain.strip().lower()), domain.strip())


def _key(domain: str) -> str:
    """What two spellings of one domain share ('Meas Value', 'meas value'): domain_id is compared
    without case everywhere else too (the concept checks, the searcher)."""
    return domain.strip().lower()


def _distinct(domains: list[str]) -> list[str]:
    """``domains`` once each (compared by ``_key``), in order, in their first spelling."""
    out: dict[str, str] = {}
    for d in domains:
        out.setdefault(_key(d), _canonical(d))
    return list(out.values())


def routed_field(spec: ETLSpec, tm: TableMapping, fm: FieldMapping) -> bool:
    """``fm`` is the main concept field of a table the build routes by domain: its standard
    concept decides the table the record ends up in."""
    table = tm.target_table.lower()
    return (spec.route_by_domain and table in ROUTE_SOURCES and fm.concept_role == "standard"
            and fm.target_field == EVENT_FIELD_ROLES[table]["concept"])


def accepted_domains_for(spec: ETLSpec, tm: TableMapping, fm: FieldMapping) -> list[str]:
    """The domains a concept_map target of ``fm`` may belong to, its target_domain first. The main
    concept field of a table routed by domain (``routed_field``) whose target_domain is one of
    the table's ``ACCEPTED_DOMAINS`` accepts the others too (the record moves to the table of the
    concept's domain); every other field accepts its target_domain only. Empty when the field has
    no target_domain (any domain)."""
    domain = (fm.target_domain or "").strip()
    if not domain:
        return []
    accepted = ACCEPTED_DOMAINS.get(tm.target_table.lower(), ())
    if not routed_field(spec, tm, fm) or domain.lower() not in {d.lower() for d in accepted}:
        return [domain]
    return [domain] + [d for d in accepted if d.lower() != domain.lower()]


@dataclass
class VocabularyDomains:
    """The domains the targets of one source vocabulary may belong to (``vocabulary_domains``)."""

    target_domain: str | None  # searched first; exact and lexical picks only come from it
    accepted: list[str]  # target_domain first; [] = target_domain only (any domain without one)
    fields: list[str] = field(default_factory=list)  # table.field of every field that uses it
    conflict: str | None = None  # why one concept per code cannot fit every field that uses it
    # False: the fields that load its concepts expect different domains (a test and a diagnosis
    # read with one vocabulary), so no concept is approved for its codes without a reviewer
    auto_approve: bool = True


def _union(sets: list[list[str]]) -> list[str]:
    return _distinct([d for s in sets for d in s])


def _intersection(sets: list[list[str]]) -> list[str]:
    if not sets:
        return []
    keep = set.intersection(*({_key(d) for d in s} for s in sets))
    return [d for d in _union(sets) if _key(d) in keep]


@dataclass
class _VocabField:
    name: str  # table.field
    target: str | None
    routed: bool  # routed_field: a concept of another accepted domain moves the record
    standard: bool  # concept_role standard: the field loads the chosen concept itself
    accepted: list[str]  # accepted_domains_for; [] = any domain


def vocabulary_domains(
    spec: ETLSpec, skip: set[tuple[int, int]] | frozenset[tuple[int, int]] = frozenset()
) -> dict[str, VocabularyDomains]:
    """Per source vocabulary of the ``concept_map`` fields (except the ``skip`` ones: (mapping
    index, field index)), the domains its targets may belong to. One concept is chosen per code
    for every field that uses the vocabulary, so their accepted sets (``accepted_domains_for``)
    are merged: the routed main fields (``routed_field``) take the union of theirs (a diagnosis
    list read into condition_occurrence and procedure_occurrence keeps Condition), the other
    fields the intersection (their union when they share no domain), and the result is both.
    target_domain is the fields' shared one when they agree, else the first one given; a field
    without a target_domain accepts any domain and restricts nothing.

    ``conflict`` is set when the result holds a domain that a field which loads the chosen concept
    and is not routed does not accept (a lab test name and a unit in one vocabulary, or a
    diagnosis list read into condition_occurrence and observation): that field can then get a
    concept it has no place for. It is set too, and ``auto_approve`` is False, when the fields
    that load the chosen concept have different target domains, routed ones included (one events
    table split by row type into condition_occurrence and measurement, its codes read with one
    vocabulary): every code is searched in the first field's domain, so a test could be mapped to
    a disease and the build would move the record silently; a reviewer decides each code. The
    validator warns and the mapper audits it."""
    found: dict[str, list[_VocabField]] = {}
    for i, tm in enumerate(spec.table_mappings):
        for j, fm in enumerate(tm.fields):
            vocab = (fm.vocabulary or "").strip()
            if fm.kind != "concept_map" or not (fm.expression or "").strip() or not vocab or (i, j) in skip:
                continue
            found.setdefault(vocab, []).append(_VocabField(
                f"{tm.target_table}.{fm.target_field}", (fm.target_domain or "").strip() or None,
                routed_field(spec, tm, fm), fm.concept_role == "standard", accepted_domains_for(spec, tm, fm)))
    out: dict[str, VocabularyDomains] = {}
    for vocab, fields in found.items():
        targets = [f.target for f in fields if f.target]
        target = targets[0] if targets else None
        routed = [f.accepted for f in fields if f.routed and f.accepted]
        other = [f.accepted for f in fields if not f.routed and f.accepted]
        common = _intersection(other) or _union(other)
        accepted = _distinct(([target] if target else []) + _union(routed) + common)
        conflict = None
        for f in fields:
            extra = [d for d in accepted if _key(d) not in {_key(a) for a in f.accepted}]
            if f.standard and not f.routed and f.accepted and extra:
                others = "; ".join(f"{g.name} ({', '.join(g.accepted)})" for g in fields if g is not f and g.accepted)
                conflict = (f"source vocabulary {vocab} is used by {f.name} ({', '.join(f.accepted)}) and by {others}: "
                            f"one concept is chosen per code for all of them, so {f.name} can get a concept of a "
                            f"domain it does not accept ({', '.join(extra)}). Give each field a source vocabulary of "
                            "its own")
                break
        loading = [f for f in fields if f.standard and f.target]
        mixed = len({_key(f.target or "") for f in loading}) > 1
        if mixed:
            why = (f"the fields that use source vocabulary {vocab} expect concepts of different domains ("
                   + "; ".join(f"{f.name}: {f.target}" for f in loading)
                   + f"): its codes are searched in {target} first, and no concept is approved for them without a "
                   "reviewer. Give each field a source vocabulary of its own")
            conflict = f"{conflict}. Also, {why}" if conflict else why[0].upper() + why[1:]
        if accepted and target and [d.lower() for d in accepted] == [target.lower()]:
            accepted = []  # the target domain only, as without routing
        elif accepted and target:
            accepted[0] = target  # the spelling of the spec
        out[vocab] = VocabularyDomains(target, accepted, [f.name for f in fields], conflict, auto_approve=not mixed)
    return out
