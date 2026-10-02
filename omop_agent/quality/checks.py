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
* ETL level (ours): ``etlRecordRetention`` (source rows that never reached the CDM) and
  ``etlStepError`` (an ETL step that failed).

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
from ..config import ProjectConfig
from ..db import Schemas, execute, fetch_all, fetch_one, qi, ql, short_error
from ..etl.compiler import target_tables
from ..models import CheckResult, DQReport, ETLRunResult, ETLSpec, TableLoadResult
from ..workspace import utcnow

CHECK_TIMEOUT_MS = 600_000
ETL_RETENTION_THRESHOLD = 5.0

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


def etl_checks(etl_result: ETLRunResult, min_retention_pct: float = 50.0) -> list[CheckResult]:
    """``etlStepError`` for failed steps; ``etlRecordRetention`` for steps that report source rows,
    and the fatal ``etlRecordLoss`` when a step keeps less than ``min_retention_pct`` of them."""
    out: list[CheckResult] = []
    for s in etl_result.steps:
        if s.error:
            out.append(
                CheckResult(
                    check_name="etlStepError",
                    check_level="ETL",
                    table=s.target_table,
                    description=f"ETL step '{s.step}' loading {s.target_table} failed: {s.error}",
                    category="Conformance",
                    subcategory="ETL",
                    severity="fatal",
                    failed=True,
                    error=s.error,
                    sql=s.sql,
                )
            )
        elif s.source_rows is not None:
            kept = _retention(s)
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
    return out


def _retention(s: TableLoadResult) -> CheckResult:
    source = int(s.source_rows or 0)
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
            f"has no match). Rows the mapping's own filter or inner joins leave out are not counted here.{extra}"
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


def gate(
    results: list[CheckResult], cfg: ProjectConfig, etl_result: ETLRunResult | None = None
) -> tuple[bool, list[str]]:
    """Pass iff no failed or errored check has a blocking severity or is listed in
    ``quality.blocking_checks``, and no ETL step errored.

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
        for s in etl_result.errors():
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
                results.extend(_not_implemented_checks(ctx, t))
    if etl_result is not None:
        results.extend(etl_checks(etl_result, cfg.quality.min_record_retention_pct))
    passed, blocking = gate(results, cfg, etl_result)
    return DQReport(run_at=utcnow(), engine="builtin", results=results, passed=passed, blocking_failures=blocking)
