"""Built-in Data Quality Dashboard checks, re-implemented in PostgreSQL.

Check names, thresholds, severities and descriptions come from DQD's own v5.4 metadata
(``resources/ohdsi/DQD_OMOP_CDMv5.4_*.csv``), and each check follows the logic of the
corresponding DQD SQL file, so results line up with the official R package:

* TABLE level: ``measurePersonCompleteness``, ``measureObservationPeriodOverlap``;
* FIELD level: ``isRequired``, ``isPrimaryKey``, ``isForeignKey``, ``fkDomain``, ``fkClass``,
  ``isStandardValidConcept``, ``measureValueCompleteness``, ``standardConceptRecordCompleteness``,
  ``sourceConceptRecordCompleteness``, ``sourceValueCompleteness``, ``plausibleValueLow/High``,
  ``plausibleStartBeforeEnd``, ``plausibleAfterBirth``, ``plausibleBeforeDeath``,
  ``plausibleDuringLife``, ``withinVisitDates``;
* ETL level (ours): ``etlRecordRetention`` (source rows that never reached the CDM, including
  rows dropped for a NULL required value), ``etlStepError`` (an ETL step that failed),
  ``etlRequiredFieldNull`` (rows dropped because a required field such as a start date or the
  year of birth was NULL, per field; threshold 0 for person, 5% elsewhere) and
  ``etlDefaultedValue`` (required values the build completed by the CDM convention: a concept 0,
  an end date equal to the start date; 5%), ``etlInvalidValue`` (values a conversion could not read,
  such as '31/02/2019' or a number with text in it, loaded as NULL; convention at 5% for date and
  required fields, characterization for other fields unless more than half of the values are
  unreadable, and for the mapping's WHERE, whose undecided rows etlRecordRetention counts as
  dropped) and
  ``etlPartialDate`` (dates with an unknown day or month, completed by the partial-date rule or
  left NULL; characterization, never failed); for person, death and visit_occurrence, which keep
  one row per key, ``etlDuplicateRecords`` (source rows that lost to another row of their key:
  convention at 5% for person, whose duplicates mean a source of another grain; characterization
  otherwise), ``etlKeyConflict`` (keys whose rows disagree: on sex or birth year, convention at 0
  for the person keys whose rows the mapping's dedup_order does not tell apart; on the date of
  death or a visit's person, characterization), ``etlKeyFanOut`` (keys with a code mapped to
  several concepts: convention at 0, not repairable) and ``etlVisitGrain`` (visits rolled up from
  rows that are separate stays: a visit key coarser than one visit; fatal above 5%, not
  repairable); for records that link to a visit, ``etlVisitLinkage`` (records whose visit key
  gives no visit of their own person; convention, 5%) and the fatal ``etlVisitLinkLoss`` (more
  than 100 - ``quality.min_record_retention_pct`` percent name a visit key that no visit mapping
  reads). Each carries the step's label (``step``): a second mapping into
  one table gets its own key, e.g. ``etlRecordRetention.measurement.measurement_2``;
* integrity (ours, re-checked after the load because custom SQL bypasses the build's INSERTs;
  fatal, threshold 0, not repairable): ``measureDuplicateDeaths`` (persons with more than one
  death record) and ``fkVisitPerson`` (records whose visit belongs to another person).

The other DQD checks (``cdmTable``, ``cdmField``, ``cdmDatatype``, ``plausibleTemporalAfter``,
``measureConditionEraCompleteness`` and the concept-level checks) are reported once per
populated table as not applicable, pointing to the R package (``quality.use_r_dqd``).

A check fails when ``pct_violated_rows * 100 > threshold``. Checks on tables the ETL does not
populate, on empty tables, and (except ``isRequired`` / ``measureValueCompleteness``) on
all-NULL fields are marked ``not_applicable``, as DQD does. A table without rows gets its
table-level checks plus one ``fieldLevelChecks`` summary instead of one result per field and
check, which keeps ``dq_results.json`` small. Every check runs as one query in its own
rolled-back savepoint, and the rendered SQL is kept on the result.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

import psycopg

from ..cdm.ddl import schema_for
from ..cdm.spec import RESOURCE_PKG, load_cdm_spec
from ..config import EtlConfig, ProjectConfig
from ..db import Schemas, execute, fetch_all, fetch_one, qi, ql, short_error
from ..etl.compiler import ONE_ROW_PER, SPEC_KINDS, step_label, target_tables
from ..etl.policies import END_FROM_START, THEMIS_PARTIAL, WHERE_SITE, partial_policy, required_null_policy
from ..models import CheckResult, DQReport, ETLRunResult, ETLSpec, TableLoadResult
from ..workspace import utcnow

CHECK_TIMEOUT_MS = 600_000
ETL_RETENTION_THRESHOLD = 5.0
# Required values the build had to complete or drop (etlDefaultedValue, etlRequiredFieldNull): the
# share of a step's rows that fails the check. A person dropped for a NULL required value (year of
# birth) takes every record of that person with them, so for person any drop fails.
ETL_VALUE_THRESHOLD = 5.0
ETL_PERSON_DROP_THRESHOLD = 0.0
# Values a conversion could not read (etlInvalidValue), as a share of the step's rows. In an
# optional field that is not a date it is characterization (a column that mixes numbers and text),
# unless more than ETL_INVALID_MOSTLY of its values are unreadable: then the expression is wrong.
ETL_INVALID_THRESHOLD = 5.0
ETL_INVALID_MOSTLY = 50.0
# One row per key (etlDuplicateRecords): the share of a person mapping's rows that lost to another
# row of the same person key (more means a source of another grain than one row per person).
ETL_DUPLICATE_THRESHOLD = 5.0
# Records with a visit key that got no visit of their own person (etlVisitLinkage).
ETL_VISIT_LINK_THRESHOLD = 5.0
# Visits whose rows are separate stays (etlVisitGrain): above this share of the visits of a
# visit_occurrence mapping, its visit key is coarser than one visit (a person or facility key).
ETL_VISIT_GRAIN_THRESHOLD = 5.0
# What each partial-date rule (etl.policies.partial_policy) does with an unknown day or month.
_PARTIAL_RULES = {
    "first": "took the first day of the known month or year",
    "last": "took the last day of the known month or year",
    "mid": "took the 15th of the month (June 15 when the month was unknown)",
    "birth": "took the 1st of the known month (THEMIS); a year alone gave no value",
    "month": "kept the known month",
    "year": "kept the known year",
    "reject": "were loaded as NULL (etl.partial_dates: reject)",
}
_PARTIAL_REJECTED = {
    "themis": "were loaded as NULL (THEMIS: a birth day that is not known is not completed)",
    "not_date": "were loaded as NULL (the field is not a date: a value computed from a partial date would be a guess)",
}

# Field-level checks we implement, in DQD order, with the metadata filter that enables each one
# (the ``evaluationFilter`` column of the Check_Descriptions file).
_YES = "Yes"
FIELD_CHECKS: list[tuple[str, Any]] = [
    ("isRequired", lambda r: r.get("isRequired") == _YES),
    ("isPrimaryKey", lambda r: r.get("isPrimaryKey") == _YES),
    ("isForeignKey", lambda r: r.get("isForeignKey") == _YES),
    ("fkDomain", lambda r: r.get("isForeignKey") == _YES and bool(r.get("fkDomain"))),
    ("fkClass", lambda r: r.get("isForeignKey") == _YES and bool(r.get("fkClass"))),
    ("isStandardValidConcept", lambda r: r.get("isStandardValidConcept") == _YES),
    ("measureValueCompleteness", lambda r: r.get("measureValueCompleteness") == _YES),
    ("standardConceptRecordCompleteness", lambda r: r.get("standardConceptRecordCompleteness") == _YES),
    ("sourceConceptRecordCompleteness", lambda r: r.get("sourceConceptRecordCompleteness") == _YES),
    ("sourceValueCompleteness", lambda r: r.get("sourceValueCompleteness") == _YES),
    ("plausibleValueLow", lambda r: bool(r.get("plausibleValueLow"))),
    ("plausibleValueHigh", lambda r: bool(r.get("plausibleValueHigh"))),
    ("plausibleStartBeforeEnd", lambda r: r.get("plausibleStartBeforeEnd") == _YES),
    ("plausibleAfterBirth", lambda r: r.get("plausibleAfterBirth") == _YES),
    ("plausibleBeforeDeath", lambda r: r.get("plausibleBeforeDeath") == _YES),
    ("plausibleDuringLife", lambda r: r.get("plausibleDuringLife") == _YES),
    ("withinVisitDates", lambda r: r.get("withinVisitDates") == _YES),
]
TABLE_CHECKS = ["measurePersonCompleteness", "measureObservationPeriodOverlap"]
IMPLEMENTED = {name for name, _ in FIELD_CHECKS} | set(TABLE_CHECKS)
NOT_IMPLEMENTED = "not implemented by the built-in engine; use the R DataQualityDashboard (quality.use_r_dqd)"
FIELD_SUMMARY = "fieldLevelChecks"

# Checks that still apply when the field is entirely NULL (DQD's applyNotApplicable).
_RUN_ON_EMPTY_FIELD = {"isRequired", "measureValueCompleteness"}

# Placeholders DQD uses in check descriptions -> Field_Level column holding the value.
_DESCRIPTION_KEYS = [
    "plausibleTemporalAfterFieldName",
    "plausibleTemporalAfterTableName",
    "plausibleStartBeforeEndFieldName",
    "plausibleValueLow",
    "plausibleValueHigh",
    "fkTableName",
    "fkDomain",
    "fkClass",
]


# --------------------------------------------------------------------------------------
# DQD metadata
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckInfo:
    """One row of DQD_OMOP_CDMv5.4_Check_Descriptions.csv."""

    level: str
    name: str
    description: str
    category: str
    subcategory: str
    severity: str
    evaluation_filter: str = ""


def _read_dqd(name: str) -> list[dict[str, str]]:
    text = resources.files(RESOURCE_PKG).joinpath(name).read_text(encoding="utf-8-sig")
    return [
        {k.strip(): (v or "").strip() for k, v in row.items() if k}
        for row in csv.DictReader(io.StringIO(text))
    ]


@lru_cache(maxsize=1)
def check_catalog() -> dict[str, CheckInfo]:
    out: dict[str, CheckInfo] = {}
    for r in _read_dqd("DQD_OMOP_CDMv5.4_Check_Descriptions.csv"):
        severity = r.get("severity", "").lower()
        out[r["checkName"]] = CheckInfo(
            level=r["checkLevel"].upper(),
            name=r["checkName"],
            description=" ".join(r["checkDescription"].split()),
            category=r.get("kahnCategory", ""),
            subcategory=r.get("kahnSubcategory", ""),
            severity=severity if severity in {"fatal", "convention", "characterization"} else "convention",
            evaluation_filter=r.get("evaluationFilter", ""),
        )
    return out


@lru_cache(maxsize=1)
def dqd_table_level() -> dict[str, dict[str, str]]:
    return {r["cdmTableName"].lower(): r for r in _read_dqd("DQD_OMOP_CDMv5.4_Table_Level.csv")}


@lru_cache(maxsize=1)
def dqd_field_level() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for r in _read_dqd("DQD_OMOP_CDMv5.4_Field_Level.csv"):
        out.setdefault(r["cdmTableName"].lower(), []).append(r)
    return out


@lru_cache(maxsize=1)
def dqd_concept_level() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for r in _read_dqd("DQD_OMOP_CDMv5.4_Concept_Level.csv"):
        out.setdefault(r["cdmTableName"].lower(), []).append(r)
    return out


_FILTER_CLAUSE = re.compile(r"^\s*(\w+)\s*(==|!=)\s*'([^']*)'\s*$")


def _matches_filter(row: dict[str, str], evaluation_filter: str) -> bool:
    """Evaluate a DQD ``evaluationFilter`` such as ``isForeignKey=='Yes' & fkDomain!= ''``."""
    for clause in evaluation_filter.split("&"):
        m = _FILTER_CLAUSE.match(clause)
        if not m:
            return False
        value = row.get(m.group(1), "")
        if (value == m.group(3)) != (m.group(2) == "=="):
            return False
    return True


def _threshold(row: dict[str, str], check: str) -> float:
    try:
        return float(row.get(f"{check}Threshold") or 0)
    except ValueError:
        return 0.0


def _description(check: str, table: str, fld: str | None, extra: dict[str, str] | None = None) -> str:
    info = check_catalog().get(check)
    text = info.description if info else check
    subs = {"cdmTableName": table, "cdmFieldName": fld or "", **(extra or {})}
    for key in sorted(subs, key=len, reverse=True):
        text = text.replace("@" + key, subs[key])
    return text


_DATE_LITERAL = re.compile(r"'(\d{4})(\d{2})(\d{2})'")
_TRANSLATED_TOKENS = re.compile(
    r"CURRENT_DATE|EXTRACT|YEAR|FROM|DATE '\d{4}-\d{2}-\d{2}'|\d+(?:\.\d+)?|[\s+\-*/()]"
)


def translate_sqlrender(expr: str) -> str:
    """Translate a DQD plausible-value bound (SqlRender dialect) into PostgreSQL.

    ``'19500101'`` -> ``DATE '1950-01-01'``; ``DATEADD(dd,1,GETDATE())`` -> ``CURRENT_DATE + 1``;
    ``YEAR(GETDATE())+1`` -> ``EXTRACT(YEAR FROM CURRENT_DATE)+1``; numbers are kept.
    """
    s = expr.strip()
    s = re.sub(
        r"DATEADD\(\s*(?:dd|d|day)\s*,\s*(-?\d+)\s*,\s*GETDATE\(\)\s*\)", r"(CURRENT_DATE + \1)", s, flags=re.I
    )
    s = re.sub(r"YEAR\(\s*GETDATE\(\)\s*\)", "EXTRACT(YEAR FROM CURRENT_DATE)", s, flags=re.I)
    s = re.sub(r"GETDATE\(\)", "CURRENT_DATE", s, flags=re.I)
    s = _DATE_LITERAL.sub(r"DATE '\1-\2-\3'", s)
    if not s or _TRANSLATED_TOKENS.sub("", s):
        raise ValueError(f"cannot translate DQD bound {expr!r} to PostgreSQL")
    return f"({s})"


# --------------------------------------------------------------------------------------
# SQL builders (each returns one SELECT with num_violated_rows and num_denominator_rows)
# --------------------------------------------------------------------------------------


class _NotApplicable(Exception):
    pass


def _q(violated: str, denominator: str) -> str:
    return f"SELECT\n  ({violated}) AS num_violated_rows,\n  ({denominator}) AS num_denominator_rows"


@dataclass
class _Ctx:
    conn: psycopg.Connection
    schemas: Schemas
    existing: set[str]
    populated: set[str]
    n_rows: dict[str, int] = field(default_factory=dict)
    columns: dict[str, set[str]] = field(default_factory=dict)
    non_null: dict[str, dict[str, int]] = field(default_factory=dict)

    def t(self, table: str) -> str:
        return f"{qi(schema_for(table, self.schemas))}.{qi(table)}"

    def concept(self) -> str:
        return f"{qi(self.schemas.vocab)}.concept"

    def need_column(self, table: str, col: str) -> None:
        if col not in self.columns.get(table, set()):
            raise ValueError(f"column {table}.{col} does not exist")


def _birth_date(p: str) -> str:
    """Birth date of person alias ``p``: birth_datetime, else year/month/day of birth."""
    return (
        f"COALESCE(CAST({p}.birth_datetime AS DATE), CASE WHEN {p}.year_of_birth BETWEEN 1 AND 9999 THEN "
        f"make_date({p}.year_of_birth, CASE WHEN {p}.month_of_birth BETWEEN 1 AND 12 THEN {p}.month_of_birth "
        f"ELSE 1 END, 1) + (CASE WHEN {p}.day_of_birth BETWEEN 1 AND 31 THEN {p}.day_of_birth ELSE 1 END - 1) END)"
    )


def _field_sql(ctx: _Ctx, check: str, table: str, fld: str, row: dict[str, str]) -> str:
    T, F = ctx.t(table), f"t.{qi(fld)}"
    count_all = f"SELECT COUNT(*) FROM {T} t"
    count_nonnull = f"SELECT COUNT(*) FROM {T} t WHERE {F} IS NOT NULL"
    is_date = row.get("cdmDatatype", "").lower() in {"date", "datetime"}
    as_date = f"CAST({F} AS DATE)"

    if check in {"isRequired", "measureValueCompleteness"}:
        return _q(f"SELECT COUNT(*) FROM {T} t WHERE {F} IS NULL", count_all)

    if check == "isPrimaryKey":
        return _q(
            f"SELECT COALESCE(SUM(d.n), 0) FROM (SELECT COUNT(*) AS n FROM {T} t WHERE {F} IS NOT NULL "
            f"GROUP BY {F} HAVING COUNT(*) > 1) d",
            count_all,
        )

    if check == "isForeignKey":
        fk_table = row.get("fkTableName", "").lower()
        fk_field = (row.get("fkFieldName") or "").lower()
        if not fk_table or not fk_field:
            raise ValueError(f"no fkTableName/fkFieldName for {table}.{fld}")
        spec = load_cdm_spec()
        if spec.has_table(fk_table) and spec.table(fk_table).schema_class != "VOCAB" and fk_table not in ctx.existing:
            raise ValueError(f"referenced table {fk_table} does not exist in schema {ctx.schemas.cdm}")
        R = ctx.t(fk_table)
        return _q(
            f"SELECT COUNT(*) FROM {T} t WHERE {F} IS NOT NULL "
            f"AND NOT EXISTS (SELECT 1 FROM {R} r WHERE r.{qi(fk_field)} = {F})",
            count_all,
        )

    if check in {"fkDomain", "fkClass", "isStandardValidConcept"}:
        if check == "fkDomain":
            cond = f"co.domain_id <> {ql(row['fkDomain'])}"
        elif check == "fkClass":
            cond = f"co.concept_class_id <> {ql(row['fkClass'])}"
        else:
            cond = "(co.standard_concept IS DISTINCT FROM 'S' OR co.invalid_reason IS NOT NULL)"
        return _q(
            f"SELECT COUNT(*) FROM {T} t JOIN {ctx.concept()} co ON co.concept_id = {F} "
            f"WHERE co.concept_id <> 0 AND {cond}",
            count_all,
        )

    if check in {"standardConceptRecordCompleteness", "sourceConceptRecordCompleteness"}:
        # DQD only counts units of records that carry a numeric value.
        only_numeric = fld == "unit_concept_id" and table in {"measurement", "observation"}
        extra = " AND t.value_as_number IS NOT NULL" if only_numeric else ""
        denom = f"{count_all} WHERE t.value_as_number IS NOT NULL" if only_numeric else count_all
        return _q(f"SELECT COUNT(*) FROM {T} t WHERE {F} = 0{extra}", denom)

    if check == "sourceValueCompleteness":
        std = (row.get("standardConceptFieldName") or "").lower()
        if not std:
            raise ValueError(f"no standardConceptFieldName for {table}.{fld}")
        ctx.need_column(table, std)
        return _q(
            f"SELECT COUNT(DISTINCT {F}) FROM {T} t WHERE t.{qi(std)} = 0",
            f"SELECT COUNT(DISTINCT {F}) FROM {T} t",
        )

    if check in {"plausibleValueLow", "plausibleValueHigh"}:
        bound = translate_sqlrender(row[check])
        op = "<" if check == "plausibleValueLow" else ">"
        value = as_date if is_date else F
        return _q(f"SELECT COUNT(*) FROM {T} t WHERE {value} {op} {bound}", count_nonnull)

    if check == "plausibleStartBeforeEnd":
        end = (row.get("plausibleStartBeforeEndFieldName") or "").lower()
        if not end:
            raise ValueError(f"no plausibleStartBeforeEndFieldName for {table}.{fld}")
        ctx.need_column(table, end)
        E = f"t.{qi(end)}"
        both = f"{F} IS NOT NULL AND {E} IS NOT NULL"
        return _q(
            f"SELECT COUNT(*) FROM {T} t WHERE {both} AND {as_date} > CAST({E} AS DATE)",
            f"SELECT COUNT(*) FROM {T} t WHERE {both}",
        )

    if check == "plausibleAfterBirth":
        if "person" not in ctx.existing:
            raise _NotApplicable
        ctx.need_column(table, "person_id")
        P = ctx.t("person")
        return _q(
            f"SELECT COUNT(*) FROM {T} t JOIN {P} p ON p.person_id = t.person_id "
            f"WHERE {F} IS NOT NULL AND {as_date} < {_birth_date('p')}",
            count_nonnull,
        )

    if check in {"plausibleBeforeDeath", "plausibleDuringLife"}:
        if "death" not in ctx.existing:
            raise _NotApplicable
        ctx.need_column(table, "person_id")
        D = ctx.t("death")
        violated = (
            f"SELECT COUNT(*) FROM {T} t JOIN {D} de ON de.person_id = t.person_id "
            f"WHERE {as_date} > de.death_date + 60"
        )
        if check == "plausibleBeforeDeath":
            denom = f"SELECT COUNT(*) FROM {T} t JOIN {D} de ON de.person_id = t.person_id WHERE {F} IS NOT NULL"
        else:
            denom = f"SELECT COUNT(*) FROM {T} t WHERE t.person_id IN (SELECT person_id FROM {D})"
        return _q(violated, denom)

    if check == "withinVisitDates":
        # DQD: records more than a week before the start or after the end of their visit
        if "visit_occurrence" not in ctx.populated or not ctx.n_rows.get("visit_occurrence"):
            raise _NotApplicable
        ctx.need_column(table, "visit_occurrence_id")
        join = f"FROM {T} t JOIN {ctx.t('visit_occurrence')} vo ON vo.visit_occurrence_id = t.visit_occurrence_id"
        return _q(
            f"SELECT COUNT(*) {join} WHERE {as_date} < vo.visit_start_date - 7 OR {as_date} > vo.visit_end_date + 7",
            f"SELECT COUNT(*) {join}",
        )

    raise ValueError(f"unknown check {check}")


def _person_completeness_sql(ctx: _Ctx, table: str) -> str:
    P, T = ctx.t("person"), ctx.t(table)
    return _q(
        f"SELECT COUNT(*) FROM {P} p WHERE NOT EXISTS (SELECT 1 FROM {T} t WHERE t.person_id = p.person_id)",
        f"SELECT COUNT(*) FROM {P} p",
    )


def _observation_period_overlap_sql(ctx: _Ctx) -> str:
    OP = ctx.t("observation_period")
    return _q(
        f"SELECT COUNT(DISTINCT a.person_id) FROM {OP} a JOIN {OP} b "
        f"ON b.person_id = a.person_id AND b.ctid <> a.ctid "
        f"WHERE a.observation_period_start_date <= b.observation_period_end_date + 1 "
        f"AND b.observation_period_start_date <= a.observation_period_end_date + 1",
        f"SELECT COUNT(DISTINCT person_id) FROM {OP}",
    )


# --------------------------------------------------------------------------------------
# Running and evaluating
# --------------------------------------------------------------------------------------


def _new(
    check: str, level: str, table: str, fld: str | None, threshold: float, extra: dict[str, str] | None = None
) -> CheckResult:
    """A not-yet-evaluated result carrying DQD's description, category and severity."""
    info = check_catalog().get(check)
    return CheckResult(
        check_name=check,
        check_level=level,  # type: ignore[arg-type]
        table=table,
        field=fld,
        description=_description(check, table, fld, extra),
        category=info.category if info else "",
        subcategory=info.subcategory if info else "",
        severity=(info.severity if info else "convention"),  # type: ignore[arg-type]
        threshold_pct=threshold,
    )


def _evaluate(ctx: _Ctx, res: CheckResult, sql: str) -> CheckResult:
    """Run one check query in its own rolled-back savepoint and fill in the result."""
    try:
        with ctx.conn.transaction(force_rollback=True):
            execute(ctx.conn, f"SET LOCAL statement_timeout = {CHECK_TIMEOUT_MS}")
            row = fetch_one(ctx.conn, sql) or {}
    except psycopg.Error as e:
        return res.model_copy(update={"error": short_error(e), "sql": sql})
    violated = int(row.get("num_violated_rows") or 0)
    denominator = int(row.get("num_denominator_rows") or 0)
    pct = violated / denominator if denominator else 0.0
    return res.model_copy(
        update={
            "num_violated_rows": violated,
            "num_denominator_rows": denominator,
            "pct_violated_rows": pct,
            # violated/denominator*100 > threshold, without floating-point surprises
            "failed": violated * 100 > res.threshold_pct * denominator,
            "sql": sql,
        }
    )


def _populated_tables(conn: psycopg.Connection, schemas: Schemas, existing: set[str], spec: ETLSpec | None) -> set[str]:
    if spec is not None:
        # every table the compiled spec writes, incl. the derived observation_period and cdm_source
        names = {m.target_table for m in spec.table_mappings} | {c.target_table for c in spec.custom_sql}
        return {n.lower() for n in names} | set(target_tables(spec)) | {"person", "observation_period", "cdm_source"}
    out = set()
    for t in existing:
        if fetch_one(conn, f"SELECT 1 AS x FROM {qi(schemas.cdm)}.{qi(t)} LIMIT 1"):
            out.add(t)
    return out


def _table_facts(ctx: _Ctx, table: str) -> None:
    """Row count and per-column non-null counts of a populated table (one query)."""
    cols = sorted(ctx.columns.get(table, set()))
    parts = ['COUNT(*) AS "__rows"'] + [f'COUNT({qi(c)}) AS "{c}"' for c in cols]
    try:
        with ctx.conn.transaction():
            row = fetch_one(ctx.conn, f"SELECT {', '.join(parts)} FROM {ctx.t(table)}") or {}
    except psycopg.Error:  # unknown facts: run every check and let each report its own error
        ctx.n_rows[table], ctx.non_null[table] = 1, {c: 1 for c in cols}
        return
    ctx.n_rows[table] = int(row.pop("__rows", 0) or 0)
    ctx.non_null[table] = {k: int(v or 0) for k, v in row.items()}


def _table_checks(ctx: _Ctx, table: str) -> list[CheckResult]:
    meta = dqd_table_level().get(table, {})
    out: list[CheckResult] = []
    catalog = check_catalog()
    if meta.get("measurePersonCompleteness") == _YES and "measurePersonCompleteness" in catalog:
        res = _new("measurePersonCompleteness", "TABLE", table, None, _threshold(meta, "measurePersonCompleteness"))
        if table not in ctx.populated or "person" not in ctx.existing:
            out.append(res.model_copy(update={"not_applicable": True}))
        else:
            out.append(_evaluate(ctx, res, _person_completeness_sql(ctx, table)))
    if meta.get("measureObservationPeriodOverlap") == _YES and "measureObservationPeriodOverlap" in catalog:
        res = _new(
            "measureObservationPeriodOverlap", "TABLE", table, None, _threshold(meta, "measureObservationPeriodOverlap")
        )
        if table not in ctx.populated or not ctx.n_rows.get(table):
            out.append(res.model_copy(update={"not_applicable": True}))
        else:
            out.append(_evaluate(ctx, res, _observation_period_overlap_sql(ctx)))
    return out


def _field_checks(ctx: _Ctx, table: str) -> list[CheckResult]:
    out: list[CheckResult] = []
    catalog = check_catalog()
    if table not in ctx.populated or not ctx.n_rows.get(table):
        summary = _field_summary(ctx, table)
        return [summary] if summary else []
    for row in dqd_field_level().get(table, []):
        fld = row["cdmFieldName"].lower()
        extra = {k: row.get(k, "").lower() if k.endswith("Name") else row.get(k, "") for k in _DESCRIPTION_KEYS}
        for check, applies in FIELD_CHECKS:
            if check not in catalog or not applies(row):
                continue
            res = _new(check, "FIELD", table, fld, _threshold(row, check), extra)
            if check == "isPrimaryKey" and table in ONE_ROW_PER:
                # the build loads one row per person (per visit and person): a duplicate is a key
                # shared by two people or custom SQL, which people fix (the agent may not change keys)
                res = res.model_copy(update={"repairable": False})
            if fld not in ctx.columns.get(table, set()):
                out.append(res.model_copy(update={"error": f"column {table}.{fld} does not exist"}))
                continue
            if check not in _RUN_ON_EMPTY_FIELD and not ctx.non_null.get(table, {}).get(fld):
                out.append(res.model_copy(update={"not_applicable": True}))
                continue
            try:
                sql = _field_sql(ctx, check, table, fld, row)
            except _NotApplicable:
                out.append(res.model_copy(update={"not_applicable": True}))
                continue
            except ValueError as e:
                out.append(res.model_copy(update={"error": str(e)}))
                continue
            out.append(_evaluate(ctx, res, sql))
    return out


def _field_summary(ctx: _Ctx, table: str) -> CheckResult | None:
    """One not-applicable result standing for every field-level check of a table without rows."""
    catalog = check_catalog()
    names: list[str] = []
    n = 0
    for row in dqd_field_level().get(table, []):
        for check, applies in FIELD_CHECKS:
            if check in catalog and applies(row):
                n += 1
                if check not in names:
                    names.append(check)
    if not n:
        return None
    reason = "the ETL does not populate it" if table not in ctx.populated else "it has no rows"
    return CheckResult(
        check_name=FIELD_SUMMARY,
        check_level="TABLE",
        table=table,
        description=f"The {n} field-level checks of {table} ({', '.join(names)}) are not applicable: {reason}.",
        not_applicable=True,
    )


def _integrity_checks(ctx: _Ctx, table: str) -> list[CheckResult]:
    """Checks of what the build guarantees, re-checked after the load because custom SQL bypasses
    the generated INSERTs (they are not DQD checks; fatal, threshold 0, never sent to the repair
    agent, whose tools cannot change custom SQL):

    * ``measureDuplicateDeaths`` (TABLE, death): persons with more than one death record (THEMIS:
      at most one);
    * ``fkVisitPerson`` (FIELD, ``visit_occurrence_id`` of every other table): records whose visit
      belongs to another person."""
    out: list[CheckResult] = []
    filled = table in ctx.populated and bool(ctx.n_rows.get(table))
    if table == "death":
        res = CheckResult(
            check_name="measureDuplicateDeaths", check_level="TABLE", table=table,
            description="The number and percent of persons in death with more than one death record (THEMIS: one "
            "death record per person). The build loads one per person, so a duplicate comes from custom SQL.",
            category="Plausibility", subcategory="Uniqueness", severity="fatal", threshold_pct=0.0, repairable=False,
        )
        if not filled:
            out.append(res.model_copy(update={"not_applicable": True}))
        else:
            out.append(_evaluate(ctx, res, _q(
                f"SELECT COUNT(*) FILTER (WHERE d.n > 1) FROM (SELECT person_id, COUNT(*) AS n FROM {ctx.t(table)} "
                "GROUP BY person_id) d",
                f"SELECT COUNT(DISTINCT person_id) FROM {ctx.t(table)}",
            )))
    cdm = load_cdm_spec()
    if table != "visit_occurrence" and filled and cdm.table(table).field("visit_occurrence_id") is not None:
        res = CheckResult(
            check_name="fkVisitPerson", check_level="FIELD", table=table, field="visit_occurrence_id",
            description=f"The number and percent of {table} records with a visit_occurrence_id whose visit belongs to "
            "another person. The build links a record only to a visit of its own person, so such a link comes from "
            "custom SQL.",
            category="Conformance", subcategory="Relational", severity="fatal", threshold_pct=0.0, repairable=False,
        )
        if "visit_occurrence" not in ctx.existing or not ctx.non_null.get(table, {}).get("visit_occurrence_id"):
            out.append(res.model_copy(update={"not_applicable": True}))
        else:
            # a visit key shared by two people (isPrimaryKey fails) has a row for each: a record of
            # either one links to a visit of its person
            V = ctx.t("visit_occurrence")
            with_visit = (f"FROM {ctx.t(table)} t WHERE EXISTS (SELECT 1 FROM {V} v "
                          "WHERE v.visit_occurrence_id = t.visit_occurrence_id)")
            out.append(_evaluate(ctx, res, _q(
                f"SELECT COUNT(*) {with_visit} AND NOT EXISTS (SELECT 1 FROM {V} v "
                "WHERE v.visit_occurrence_id = t.visit_occurrence_id AND v.person_id = t.person_id)",
                f"SELECT COUNT(*) {with_visit}",
            )))
    return out


def _not_implemented_checks(ctx: _Ctx, table: str) -> list[CheckResult]:
    """One not-applicable result per DQD check that applies to a populated table but is not built in."""
    if table not in ctx.populated or not ctx.n_rows.get(table):
        return []  # covered by the table's field-level summary
    rows_by_level = {
        "TABLE": [r for r in [dqd_table_level().get(table)] if r],
        "FIELD": dqd_field_level().get(table, []),
        "CONCEPT": dqd_concept_level().get(table, []),
    }
    unit = {"TABLE": "table", "FIELD": "field", "CONCEPT": "concept"}
    out: list[CheckResult] = []
    for info in check_catalog().values():
        if info.name in IMPLEMENTED or info.level not in rows_by_level:
            continue
        n = sum(_matches_filter(r, info.evaluation_filter) for r in rows_by_level[info.level])
        if not n:
            continue
        what = "" if info.level == "TABLE" else f" ({n} {unit[info.level]}{'s' if n > 1 else ''} of {table})"
        out.append(
            CheckResult(
                check_name=info.name,
                check_level=info.level,  # type: ignore[arg-type]
                table=table,
                description=f"{info.name}{what}: {NOT_IMPLEMENTED}",
                category=info.category,
                subcategory=info.subcategory,
                severity=info.severity,  # type: ignore[arg-type]
                not_applicable=True,
            )
        )
    return out


def etl_checks(
    etl_result: ETLRunResult, min_retention_pct: float = 50.0, etl: EtlConfig | None = None
) -> list[CheckResult]:
    """``etlStepError`` for failed steps; ``etlRecordRetention`` for steps that report source rows,
    and the fatal ``etlRecordLoss`` when a step keeps less than ``min_retention_pct`` of them;
    ``etlRequiredFieldNull`` and ``etlDefaultedValue`` per field whose required values a step
    dropped or completed, and ``etlInvalidValue`` / ``etlPartialDate`` per field (or the WHERE)
    with values its conversions could not read or that were partial dates
    (``TableLoadResult.value_counts``; ``etl`` gives the partial-date rule the build used);
    ``etlDuplicateRecords``, ``etlKeyConflict`` and ``etlKeyFanOut`` for the steps that keep one
    row per key (``_key_checks``), and ``etlVisitLinkage`` / the fatal ``etlVisitLinkLoss`` for
    the steps whose records link to a visit (``_visit_link_checks``).

    Every result carries the step label. The error of a non-blocking step is a convention-level
    failure, and the repair agent only gets the errors of steps that run spec SQL, and not those
    a change of the spec cannot fix (a missing privilege). When a person load (a mapping or
    custom SQL) failed, the other steps' retention and visit links are left out: its error is the
    cause of their drops, since the id-map pruning retires the keys of people not loaded; so are
    the visit links when a visit_occurrence load failed.
    """
    out: list[CheckResult] = []

    def load_failed(table: str) -> bool:
        return any(s.error and s.target_table == table and s.kind in (None, "insert", "custom") for s in etl_result.steps)

    person_failed, visit_failed = load_failed("person"), load_failed("visit_occurrence")
    for s in etl_result.steps:
        label = step_label(s.step)
        if s.error:
            out.append(
                CheckResult(
                    check_name="etlStepError",
                    check_level="ETL",
                    table=s.target_table,
                    description=f"ETL step '{s.step}' loading {s.target_table} failed: {s.error}",
                    category="Conformance",
                    subcategory="ETL",
                    severity="fatal" if s.blocking else "convention",
                    failed=True,
                    error=s.error,
                    sql=s.sql,
                    step=label,
                    repairable=(s.kind is None or s.kind in SPEC_KINDS) and s.repairable,
                )
            )
            continue
        if s.source_rows is not None and not person_failed:
            kept = _retention(s).model_copy(update={"step": label})
            out.append(kept)
            if kept.num_denominator_rows and (1 - kept.pct_violated_rows) * 100 < min_retention_pct:
                out.append(
                    kept.model_copy(
                        update={
                            "check_name": "etlRecordLoss",
                            "description": (
                                f"ETL step '{s.step}' loaded less than {min_retention_pct:g}% of its source rows into "
                                f"{s.target_table}: {kept.num_violated_rows} of {kept.num_denominator_rows} dropped. "
                                "Usually a broken person or visit key, or a join that matches nothing."
                            ),
                            "severity": "fatal",
                            "threshold_pct": 100 - min_retention_pct,
                            "failed": True,
                        }
                    )
                )
        out += _key_checks(s, label)
        # a failed person or visit load is the cause of the records left without their visit
        if s.stats.get("visit_links") and not person_failed and not visit_failed:
            out += _visit_link_checks(s, label, min_retention_pct)
        out += _required_value_checks(s, label)
        out += _conversion_checks(s, label, etl)
    return out


def _required_value_checks(s: TableLoadResult, label: str) -> list[CheckResult]:
    """``etlRequiredFieldNull`` per field with rows dropped for a NULL required value, and
    ``etlDefaultedValue`` per field whose NULL required values got the CDM default. Both count
    source rows (before vocabulary fan-out) out of the rows that survived the id-map joins."""
    rows = s.id_retained_rows if s.id_retained_rows is not None else int(s.source_rows or 0)
    out: list[CheckResult] = []
    for f, n in sorted(s.value_counts.get("required_null", {}).items()):
        threshold = ETL_PERSON_DROP_THRESHOLD if s.target_table == "person" else ETL_VALUE_THRESHOLD
        out.append(_value_check(
            s, label, "etlRequiredFieldNull", f, n, rows, threshold,
            f"Rows of ETL step '{s.step}' dropped because the required field {s.target_table}.{f} was NULL (missing "
            f"in the source, or the expression gave no value): {n} of {rows} rows. The CDM has no convention for a "
            "missing value of this field, so the record cannot be loaded; fix the expression if the source has the "
            "value in another form.",
        ))
    for f, n in sorted(s.value_counts.get("defaulted", {}).items()):
        start = END_FROM_START.get((s.target_table, f))
        if start:
            what = f"took the start date ({start})"
        elif required_null_policy(s.target_table, f) == "zero":
            what = "became 0 (no matching concept)"
        else:
            what = "got the CDM default"
        out.append(_value_check(
            s, label, "etlDefaultedValue", f, n, rows, ETL_VALUE_THRESHOLD,
            f"Rows of ETL step '{s.step}' whose required field {s.target_table}.{f} was NULL and {what}, by the CDM "
            f"convention: {n} of {rows} rows. Write an explicit fallback when the source has a better value.",
        ))
    return out


def _is_date_field(table: str, name: str) -> bool:
    cdm = load_cdm_spec()
    f = cdm.table(table).field(name) if cdm.has_table(table) else None
    return f is not None and f.datatype.strip().lower() in ("date", "datetime", "timestamp")


def _is_required_field(table: str, name: str) -> bool:
    cdm = load_cdm_spec()
    f = cdm.table(table).field(name) if cdm.has_table(table) else None
    return f is not None and f.required


def _partial_rule(table: str, name: str, etl: EtlConfig | None) -> str:
    """What the partial-date rule of ``table.name`` did with an unknown day or month."""
    policy = partial_policy(table, name, etl)
    if policy != "reject":
        return _PARTIAL_RULES[policy]
    if (table, name) in THEMIS_PARTIAL:
        return _PARTIAL_REJECTED["themis"]
    if name != WHERE_SITE and not _is_date_field(table, name):
        return _PARTIAL_REJECTED["not_date"]
    return _PARTIAL_RULES["reject"]


def _conversion_checks(s: TableLoadResult, label: str, etl: EtlConfig | None = None) -> list[CheckResult]:
    """``etlInvalidValue`` per field (or the WHERE) with values a conversion could not read, and
    ``etlPartialDate`` per field with partial dates. A field's count can exceed its rows when its
    expression converts several values (``COALESCE(a::date, b::date)``); the share is capped at 1.
    The descriptions give counts only as "<n> of <rows> rows", which the repair prompt can suppress
    below the minimum cell count."""
    rows = s.id_retained_rows if s.id_retained_rows is not None else int(s.source_rows or 0)
    t = s.target_table
    out: list[CheckResult] = []
    for f, n in sorted(s.value_counts.get("invalid", {}).items()):
        if f == WHERE_SITE:
            # the WHERE runs before the rows are counted: the rows it could not decide are not in source_rows
            denominator = int(s.source_rows or 0) + n
            out.append(_value_check(
                s, label, "etlInvalidValue", f, n, denominator, ETL_INVALID_THRESHOLD,
                f"Values in the WHERE of ETL step '{s.step}' that could not be read as written (an invalid date, a "
                f"number with text in it): {n} of {denominator} rows. The filter read them as unknown (NULL) and left "
                f"the rows out of {t}; they are counted as dropped records (etlRecordRetention). Fix the conversion "
                "if the source has these values in another form.",
                severity="characterization", category="Conformance",
            ))
            continue
        important = _is_date_field(t, f) or _is_required_field(t, f)
        mostly = not important and n * 100 > ETL_INVALID_MOSTLY * rows
        out.append(_value_check(
            s, label, "etlInvalidValue", f, n, rows, ETL_INVALID_MOSTLY if mostly else ETL_INVALID_THRESHOLD,
            f"Values of {t}.{f} in ETL step '{s.step}' that could not be read as written (an invalid date such as "
            f"31/02/2019, a date in another layout than the template, a number with text in it): {n} of {rows} rows, "
            "loaded as NULL"
            + (" (a required field then follows the CDM convention: see etlRequiredFieldNull and etlDefaultedValue)"
               if _is_required_field(t, f) else "")
            + (f"; more than {ETL_INVALID_MOSTLY:g}% of the values, so the expression is probably wrong for this "
               "source" if mostly else "")
            + ". Fix the expression if the source has these values in another form (the to_date template the "
            "profile shows, a decimal comma, a unit in the value, a guard for a column that mixes numbers and text).",
            severity="convention" if important or mostly else "characterization", category="Conformance",
        ))
    imputed, partial = s.value_counts.get("imputed", {}), s.value_counts.get("partial_null", {})
    for f in sorted(set(imputed) | set(partial)):
        n = imputed.get(f, 0) + partial.get(f, 0)
        policy = partial_policy(t, f, etl)
        out.append(_value_check(
            s, label, "etlPartialDate", f, n, rows, 100.0,
            f"Dates of {t}.{f} in ETL step '{s.step}' with an unknown day or month (00, 97-99, or no day or month "
            f"in the value): {n} of {rows} rows. The values {_partial_rule(t, f, etl)}"
            + ("; those the rule could not complete were left NULL" if partial.get(f) and policy != "reject" else "")
            + ". Reported for review only.",
            severity="characterization", failed=False,
        ))
    return out


def _value_check(
    s: TableLoadResult, label: str, name: str, target_field: str, n: int, rows: int, threshold: float,
    description: str, *, severity: str = "convention", category: str = "Completeness", failed: bool | None = None,
) -> CheckResult:
    return CheckResult(
        check_name=name,
        check_level="ETL",
        table=s.target_table,
        field=target_field,
        description=description,
        category=category,
        subcategory="ETL",
        severity=severity,  # type: ignore[arg-type]
        num_violated_rows=n,
        num_denominator_rows=rows,
        pct_violated_rows=min(n / rows, 1.0) if rows else 0.0,
        threshold_pct=threshold,
        failed=n * 100 > threshold * rows if failed is None else failed,
        sql=s.sql,
        step=label,
    )


def _retention(s: TableLoadResult) -> CheckResult:
    # rows the mapping's WHERE could not decide because a value in it could not be read: the filter
    # read it as NULL and left the row out, and source_rows (counted with the same filter) does not
    # have them; they are records lost, not rows filtered (up to: two such values in one row count twice)
    undecided = int(s.value_counts.get("invalid", {}).get(WHERE_SITE, 0))
    source = int(s.source_rows or 0) + undecided
    if s.retained_rows is not None:
        # rows lost before the vocabulary lookups; a code mapped to several concepts inserts
        # several rows, and that fan-out must not hide real drops
        dropped = max(source - s.retained_rows, 0)
        fan_out = max(s.rows_inserted - s.retained_rows, 0)
    else:  # older results: net comparison
        dropped = max(source - s.rows_inserted, 0)
        fan_out = 0
    pct = dropped / source if source else 0.0
    extra = (
        f" Another {fan_out} rows were added by 1-to-many vocabulary mappings ('Maps to' or "
        f"source_to_concept_map rows with several targets): {s.rows_inserted} rows inserted."
        if fan_out
        else ""
    )
    return CheckResult(
        check_name="etlRecordRetention",
        check_level="ETL",
        table=s.target_table,
        description=(
            f"The number and percent of source rows of ETL step '{s.step}' that were not loaded into "
            f"{s.target_table}: {dropped} of {source} records dropped (for example rows whose person or visit key "
            "has no match, or rows dropped because a required value was NULL: see etlRequiredFieldNull). Rows the "
            "mapping's own filter or inner joins leave out are not counted here"
            + (", except the rows its filter could not decide because a value in it could not be read (see "
               "etlInvalidValue): those are counted as dropped" if undecided else "")
            + f".{extra}"
        ),
        category="Completeness",
        subcategory="ETL",
        severity="convention",
        num_violated_rows=dropped,
        num_denominator_rows=source,
        pct_violated_rows=pct,
        threshold_pct=ETL_RETENTION_THRESHOLD,
        failed=dropped * 100 > ETL_RETENTION_THRESHOLD * source,
        sql=s.sql,
    )


_ONE_ROW_PER_WHAT = {"person": "person", "death": "person (THEMIS: one death record)", "visit_occurrence": "visit and person"}


def _key_check(
    s: TableLoadResult, label: str, name: str, v: int, d: int, description: str, *, severity: str, threshold: float,
    repairable: bool | None = None, subcategory: str = "Uniqueness",
) -> CheckResult:
    failed = severity != "characterization" and v * 100 > threshold * d  # characterization is for review only
    return CheckResult(
        check_name=name, check_level="ETL", table=s.target_table, description=description, category="Plausibility",
        subcategory=subcategory, severity=severity,  # type: ignore[arg-type]
        num_violated_rows=v, num_denominator_rows=d, pct_violated_rows=min(v / d, 1.0) if d else 0.0,
        threshold_pct=threshold if severity != "characterization" else 100.0, failed=failed, sql=s.sql, step=label,
        repairable=repairable,
    )


def _key_checks(s: TableLoadResult, label: str) -> list[CheckResult]:
    """For a step that keeps one row per key (its statistics, ``etl.compiler._one_row_per_key``):

    * ``etlDuplicateRecords``: source rows not loaded as a row of their own. For person, the rows
      of a key that another row of the mapping won (convention, 5%: many mean a source of another
      grain; not repairable, since only a person can choose another source table); for death and
      visit_occurrence also the keys an earlier mapping loaded (characterization: ward stays roll
      up into one visit);
    * ``etlKeyConflict``: keys whose rows disagree, among themselves or with the row an earlier
      mapping loaded (person: sex or year of birth; convention at 0 over the keys this mapping
      loads whose best-ranked rows still disagree, so that row is arbitrary: all of a key's rows
      without a dedup_order, the rows its dedup_order ranks first with one; death: the date; visit:
      the person; both characterization, a visit shared by two people fails isPrimaryKey);
    * ``etlKeyFanOut``: keys with several rows for one source row (a code mapped to several
      concepts in a field of a one-row-per-key table): convention at 0, not repairable (the
      vocabulary mapping is reviewed by people);
    * ``etlVisitGrain`` (visit_occurrence): visits whose rows are separate stays, a row starting
      more than a day after the earlier rows of the visit ended: the visit key is coarser than one
      visit (a person or facility key), and those stays were loaded as one visit spanning them all.
      Fatal above 5% of the visits (a coarse key used to fail isPrimaryKey), not repairable (the
      visit key is for people to change).

    The descriptions give counts only as "<n> of <rows>", which the repair prompt can suppress."""
    st = s.stats
    if "keys" not in st:
        return []
    t = s.target_table
    kept, keys = int(st.get("source_rows_kept") or 0), int(st.get("keys") or 0)
    dup, earlier = int(st.get("duplicate_rows") or 0), int(st.get("already_loaded_keys") or 0)
    conflicts, fan_out = int(st.get("conflict_keys") or 0), int(st.get("fan_out_keys") or 0)
    undecided = int(st.get("undecided_keys", conflicts) or 0)
    ranked = bool(st.get("dedup_order"))
    what = _ONE_ROW_PER_WHAT.get(t, "key")
    out: list[CheckResult] = []
    if t == "person":
        out.append(_key_check(
            s, label, "etlDuplicateRecords", dup, kept,
            f"person keeps one row per person. Source rows of ETL step '{s.step}' not loaded because another row of "
            f"the same person key was loaded instead: {dup} of {kept} rows ("
            + ("the mapping's dedup_order chose the row" if ranked else "the most complete row was kept")
            + "). Many such rows mean a source table of another grain than one row per person (one row per "
            "registration or per visit). Which table lists the people is for a person to decide (the repair agent "
            "may not change it): map a table that lists each person once, or keep this one and its dedup_order. Keys "
            "that an earlier person mapping loaded are skipped and not counted here.",
            severity="convention", threshold=ETL_DUPLICATE_THRESHOLD, repairable=False,
        ))
    else:
        rollup = (" Several rows of one visit (ward stays, diagnoses) are one visit, from the earliest start to the "
                  "latest end of its rows." if t == "visit_occurrence" else
                  " A death date known to the day was kept before one completed from a year or month, then the "
                  "latest, unless the mapping's dedup_order chose another.")
        out.append(_key_check(
            s, label, "etlDuplicateRecords", dup + earlier, kept,
            f"{t} keeps one row per {what}. Source rows of ETL step '{s.step}' not loaded as a row of their own, "
            f"because another row of the same key was loaded instead or an earlier mapping of {t} had loaded the "
            f"key: {dup + earlier} of {kept} rows.{rollup} Reported for review.",
            severity="characterization", threshold=100.0,
        ))
    if t == "person":
        severity, v = "convention", undecided
        description = (
            f"Person keys of ETL step '{s.step}' whose rows disagree on sex or year of birth (a re-registration that "
            "corrected them, or two people sharing one key), among themselves or with the person an earlier person "
            f"mapping loaded (which is kept): {conflicts} of {keys} keys. For {undecided} of {keys} keys this mapping "
            "loads, the rows it ranks best disagree, so the row loaded is arbitrary"
            + (" (the dedup_order does not tell them apart)" if ranked else " (it has no dedup_order)")
            + ": set dedup_order to rank the row that should win first (for example the latest registration). If the "
            "person key joins different people, a person must fix the key (the repair agent may not change it)."
        )
    elif t == "visit_occurrence":
        severity, v = "characterization", conflicts
        description = (
            f"Visit keys of ETL step '{s.step}' that belong to more than one person (in this mapping, or a visit an "
            f"earlier mapping loaded for another person): {conflicts} of {keys} keys. Each person gets a row, so the "
            "visit key is not unique and isPrimaryKey fails: when visit numbers restart for each person the visit "
            "key must include the person key (e.g. v.individual_id || ':' || v.visit_no); keys of two sources that "
            "number their visits independently need a prefix. A person must change the visit key."
        )
    else:
        severity, v = "characterization", conflicts
        description = (
            f"Persons of ETL step '{s.step}' whose death rows give different dates, among themselves or with the "
            f"death an earlier death mapping loaded (which is kept): {conflicts} of {keys} persons. One death per "
            "person was kept (a date known to the day before a completed one, then the latest, unless the mapping's "
            "dedup_order chose another). Reported for review."
        )
    out.append(_key_check(s, label, "etlKeyConflict", v, keys, description, severity=severity, threshold=0.0,
                          repairable=None if t == "person" else False))
    if fan_out:
        out.append(_key_check(
            s, label, "etlKeyFanOut", fan_out, keys,
            f"Keys of ETL step '{s.step}' with more than one candidate row for one source row: a source code maps to "
            f"several concepts ('Maps to', or several source_to_concept_map rows) in a field of {t}, which keeps one "
            f"row per {what}: {fan_out} of {keys} keys. One of the concepts was kept (the smallest id). Review the "
            "vocabulary mapping of these codes.",
            severity="convention", threshold=0.0, repairable=False, subcategory="Vocabulary",
        ))
    if t == "visit_occurrence" and "gap_keys" in st:
        gaps = int(st.get("gap_keys") or 0)
        out.append(_key_check(
            s, label, "etlVisitGrain", gaps, keys,
            f"Visits of ETL step '{s.step}' loaded from rows that are separate stays (a row starts more than a day "
            f"after the earlier rows of the visit ended): {gaps} of {keys} visits. The rows of one visit key are "
            "loaded as one visit from the earliest start to the latest end, which is right for the ward stays of one "
            "admission but not for separate visits: many such visits mean the visit key is coarser than one visit (a "
            "person or facility key), so each of them spans several visits. A person must fix the visit key (the "
            "repair agent may not change it).",
            severity="fatal", threshold=ETL_VISIT_GRAIN_THRESHOLD, repairable=False, subcategory="ETL",
        ))
    return out


def _visit_link_checks(s: TableLoadResult, label: str, min_retention_pct: float) -> list[CheckResult]:
    """``etlVisitLinkage``: the step's source rows with a visit key that got no visit of their own
    person (``stats['visit_links']``, counted with the retained rows), out of the rows with a key
    whose visit a visit mapping left out (a key of its source rows that it filtered out or could
    not load) (convention, 5%); and the fatal ``etlVisitLinkLoss`` when more than 100 -
    ``min_retention_pct`` percent of them name a visit key that no visit mapping reads: a visit key
    written in another form than in the visit mappings, which drops the links of most records.
    The agent cannot pass them by making visit keys NULL (``repair.drop_guard``)."""
    links = s.stats.get("visit_links") or {}
    with_key, linked = int(links.get("with_key") or 0), int(links.get("linked") or 0)
    other, filtered = int(links.get("other_person") or 0), int(links.get("filtered") or 0)
    unknown = max(with_key - linked - other - filtered, 0)
    d = max(with_key - filtered, 0)
    v = other + unknown
    if other and unknown:
        kinds = " Some name a loaded visit of another person, and some a visit key that no visit mapping reads."
    elif other:
        kinds = " They name a loaded visit of another person."
    elif unknown:
        kinds = " They name a visit key that no visit mapping reads."
    else:
        kinds = ""
    out = [CheckResult(
        check_name="etlVisitLinkage", check_level="ETL", table=s.target_table, field="visit_occurrence_id",
        description=(
            f"Source rows of ETL step '{s.step}' whose visit key gives no visit of their own person, so they were "
            f"loaded without a visit_occurrence_id: {v} of {d} rows with a visit key (rows whose visit the visit "
            f"mapping filtered out are not counted).{kinds} A record links only to a visit of its own person, and "
            "only when its visit key gives exactly the text of the visit mapping's key (case, padding, separators, "
            "prefix, date format). Fix the visit_id expression so it gives that text; never make it NULL, and never "
            "change a person_id expression to make a link."
        ),
        category="Conformance", subcategory="Relational", severity="convention",
        num_violated_rows=v, num_denominator_rows=d, pct_violated_rows=v / d if d else 0.0,
        threshold_pct=ETL_VISIT_LINK_THRESHOLD, failed=v * 100 > ETL_VISIT_LINK_THRESHOLD * d, sql=s.sql, step=label,
    )]
    loss = 100 - min_retention_pct
    if d and unknown * 100 > loss * d:
        out.append(out[0].model_copy(update={
            "check_name": "etlVisitLinkLoss",
            "description": (
                f"ETL step '{s.step}': rows whose visit key is a key that no visit mapping reads (not one a visit "
                f"mapping filtered out): {unknown} of {d} rows with a visit key, more than {loss:g}%. The visit key is "
                "almost certainly written in another form here than in the visit mapping (case, padding, separators, "
                "prefix, date format), so these records lost their visit."
            ),
            "severity": "fatal", "num_violated_rows": unknown, "pct_violated_rows": unknown / d,
            "threshold_pct": loss, "failed": True,
        }))
    return out


def summarize(r: CheckResult) -> str:
    """One-line summary of a check outcome, e.g. for gate messages and prompts."""
    if r.error:
        return f"{r.key} ({r.severity}): ERROR {r.error[:300]}"
    return (
        f"{r.key} ({r.severity}): {r.num_violated_rows} of {r.num_denominator_rows} "
        f"({r.pct_violated_rows:.2%}) violated, threshold {r.threshold_pct:g}%"
    )


def is_listed_check(r: CheckResult, entries: list[str]) -> bool:
    """True when ``r`` is named in ``entries`` by check name or by a key prefix."""
    return any(e == r.check_name or e == r.key or r.key.startswith(e + ".") for e in entries)


def repairable(
    r: CheckResult,
    severities: set[str] | list[str],
    *,
    derived: set[str] | frozenset[str] = frozenset(),
    route_only: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """True when ``r`` is a failure the repair agent should be given: failed or errored, applicable,
    with a severity in ``severities``, and caused by something the spec tools can change. Never:
    results marked ``repairable=False`` (e.g. errors of generated steps), checks on the tables this
    build derived from other tables rather than from the spec (``derived``, e.g. eras built by the
    build's own steps; an era table the spec loads stays repairable), and person completeness of
    tables that only receive records moved there by domain (``route_only``)."""
    if r.repairable is False or r.not_applicable or not (r.failed or r.error) or r.severity not in severities:
        return False
    if r.table in derived:
        return False
    return not (r.check_name == "measurePersonCompleteness" and r.table in route_only)


def gate(
    results: list[CheckResult], cfg: ProjectConfig, etl_result: ETLRunResult | None = None
) -> tuple[bool, list[str]]:
    """Pass iff no failed or errored check has a blocking severity or is listed in
    ``quality.blocking_checks``, and no blocking ETL step errored.

    Returns (passed, one line per blocking problem).
    """
    blocking_sev = set(cfg.quality.blocking_severities)
    always = list(cfg.quality.blocking_checks)
    blocking: list[str] = []
    step_errors_seen: set[tuple[str, str]] = set()
    for r in results:
        if r.not_applicable or not (r.failed or r.error):
            continue
        if r.severity not in blocking_sev and not is_listed_check(r, always):
            continue
        blocking.append(summarize(r))
        if r.check_name == "etlStepError":
            step_errors_seen.add((r.table, r.error or ""))
    if etl_result is not None:
        for s in etl_result.blocking_errors():
            if (s.target_table, s.error or "") not in step_errors_seen:
                blocking.append(f"etlStepError.{s.target_table} (step {s.step}): {(s.error or '')[:300]}")
    return not blocking, blocking


def run_builtin_checks(
    conn: psycopg.Connection,
    schemas: Schemas,
    cfg: ProjectConfig,
    spec: ETLSpec | None = None,
    etl_result: ETLRunResult | None = None,
) -> DQReport:
    """Run the built-in DQD subset over the CDM tables in ``schemas.cdm``.

    "Populated" tables are every table the compiled spec writes (``compiler.target_tables``,
    which includes cdm_source) plus person and observation_period, or, without a spec, the
    tables that have rows; checks on other tables are reported as not applicable.
    All statements run in a transaction that is rolled back, so the connection is left as found.
    """
    cdm = load_cdm_spec()
    results: list[CheckResult] = []
    with conn.transaction(force_rollback=True):
        rows = fetch_all(
            conn,
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = %s",
            (schemas.cdm,),
        )
        columns: dict[str, set[str]] = {}
        for r in rows:
            columns.setdefault(r["table_name"], set()).add(r["column_name"])
        cdm_tables = [t.name for t in cdm.cdm_tables]
        existing = {t for t in cdm_tables if t in columns}
        ctx = _Ctx(conn, schemas, existing, _populated_tables(conn, schemas, existing, spec), columns=columns)
        for t in cdm_tables:
            if t in existing and t in ctx.populated:
                _table_facts(ctx, t)
        for t in cdm_tables:
            if t in existing:
                results.extend(_table_checks(ctx, t))
        for t in cdm_tables:
            if t in existing:
                results.extend(_field_checks(ctx, t))
                results.extend(_integrity_checks(ctx, t))
                results.extend(_not_implemented_checks(ctx, t))
    if etl_result is not None:
        results.extend(etl_checks(etl_result, cfg.quality.min_record_retention_pct, cfg.etl))
    passed, blocking = gate(results, cfg, etl_result)
    return DQReport(run_at=utcnow(), engine="builtin", results=results, passed=passed, blocking_failures=blocking)
