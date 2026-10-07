"""Run a compiled ETL spec against PostgreSQL.

Every step runs in its own transaction, so one failing table (say a NOT NULL violation in
``visit_occurrence``) is reported and the remaining tables still load. The SQL of each step is
written to ``sql/<step>.sql`` and the outcome to ``etl_run.json`` for review and for the
data-quality stage, which compares ``source_rows`` with ``retained_rows`` (record retention;
``rows_inserted`` can be larger than both when a code maps to several concepts).

Inside its transaction a step first pins the session settings (``etl.session``: DateStyle, the
configured site time zone, imputation cap) and counts its source rows. It then runs, and the
runner reads what its last statement returns (``CompiledStep.returns``): a row count, one row of
statistics (kept in ``stats``; its ``inserted`` column is the row count) or one row per
destination of the records it moved; a result it cannot read fails the step, which is rolled
back. Last, the value counters of the step's ``count_sites`` (transaction-local settings
``omop_agent.<category>__<site>``) go to ``value_counts``. A step whose ``requires`` (a CDM table
or a step label) failed or was skipped is skipped too, and so are the steps that require it in
turn. An error of a non-blocking step (``blocking=False``) is reported but does not fail the
build (``ETLRunResult.ok``). After the id maps, and after every step that wrote rows, the written
tables are ANALYZEd, so the next steps are planned with fresh statistics.

The spec's SQL is written by an LLM (and possibly edited by people), so before anything runs
each step's fragments are checked again by ``design.validate.sql_safety_issues``; unsafe steps
are refused and reported as errors. With ``target.etl_role`` set, every step also runs under
that role (``SET LOCAL ROLE``), which needs USAGE and SELECT on the staging and vocabulary
schemas, SELECT/INSERT/DELETE on the CDM and ETL schemas (DELETE for id-map pruning and custom
SQL), and membership of the connecting user. The CDM tables are emptied before the steps run,
and ANALYZEd, as the connecting user.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from pydantic import ValidationError

from ..cdm.ddl import create_cdm_tables, create_etl_tables, truncate_cdm_tables
from ..cdm.spec import load_cdm_spec
from ..config import ProjectConfig
from ..db import Schemas, execute, fetch_all, fetch_one, qi, scalar, short_error
from ..models import ETLRunResult, ETLSpec, RoutedRows, TableLoadResult
from ..workspace import Workspace, utcnow
from .compiler import CompiledStep, compile_spec, target_tables
from .session import apply_session, session_sql

# Value counters a step can report per site (a CDM field or 'where'), and the site whose
# 'dropped' counter counts rows dropped for a NULL required value: value_counts['row']['dropped'].
VALUE_COUNT_CATEGORIES = (
    "invalid", "imputed", "partial_null", "offset_ignored", "offset_converted", "defaulted", "required_null",
)
ROW_SITE = "row"
# Steps after which the tables they wrote are ANALYZEd.
_ANALYZED_KINDS = {"insert", "custom", "route", "merge", "derived"}
_ID_MAPS = ("person_map", "visit_map")


def _is_cdm_table(name: str | None) -> bool:
    cdm = load_cdm_spec()
    return bool(name) and cdm.has_table(name) and cdm.table(name).schema_class == "CDM"  # type: ignore[arg-type]


def _previous_tables(ws: Workspace | None) -> set[str]:
    """CDM tables loaded by the previous run (so tables dropped from the spec are emptied too)."""
    if ws is None or not ws.etl_run_json.exists():
        return set()
    try:
        prev = ws.load_json(ws.etl_run_json, ETLRunResult)
    except Exception:  # an unreadable old result must not block a new run
        return set()
    names = {s.target_table for s in prev.steps} | {m.to_table for s in prev.steps for m in s.moved}
    return {n for n in names if _is_cdm_table(n)}


def _write_sql(ws: Workspace, steps: list[CompiledStep], cfg: ProjectConfig | None = None) -> None:
    for old in ws.sql_dir.glob("*.sql"):
        old.unlink()
    settings = "".join(f"-- session: {stmt}\n" for stmt in session_sql(cfg))  # what the step runs under
    for step in steps:
        body = settings + step.sql if not step.error else f"-- {step.error}\n"
        (ws.sql_dir / f"{step.name}.sql").write_text(body.rstrip() + "\n")


def _plain(value: Any) -> Any:
    """A statistic as JSON-friendly Python (numeric -> int or float, dates -> ISO text)."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _counter_names(sites: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """(category, field, setting name) of every counter a step with these sites can set."""
    out = []
    for site in sites:
        if site == ROW_SITE:
            out.append((ROW_SITE, "dropped", f"omop_agent.dropped__{ROW_SITE}"))
        else:
            out += [(cat, site, f"omop_agent.{cat}__{site}") for cat in VALUE_COUNT_CATEGORIES]
    return out


def _reset_counters(conn: psycopg.Connection, step: CompiledStep) -> None:
    """Zero the step's counters, in case it runs inside a transaction that has set them before."""
    names = [n for _, _, n in _counter_names(step.count_sites)]
    if names:
        execute(conn, "SELECT set_config(n, '', true) FROM unnest(%s::text[]) AS n", (names,))


def _read_counters(conn: psycopg.Connection, step: CompiledStep) -> dict[str, dict[str, int]]:
    """The counters the step set, as ``value_counts``. They are placeholder settings, which
    ``pg_settings`` does not list, so each one is read by name."""
    names = _counter_names(step.count_sites)
    if not names:
        return {}
    rows = fetch_all(
        conn, "SELECT n, current_setting(n, true) AS v FROM unnest(%s::text[]) AS n", ([n for _, _, n in names],)
    )
    values = {r["n"]: r["v"] for r in rows}
    out: dict[str, dict[str, int]] = {}
    for cat, fld, name in names:
        n = int(values.get(name) or 0)
        if n:
            out.setdefault(cat, {})[fld] = n
    return out


class _UnreadableResult(Exception):
    """A step ran, but what it returned (or a counter it set) cannot be read: it fails, rolled back."""


def _read_result(
    cur: psycopg.Cursor, step: CompiledStep, table: str
) -> tuple[int, dict[str, Any], list[RoutedRows]]:
    """(rows_inserted, stats, moved) from the last statement of ``step``, as ``step.returns`` says."""
    if step.returns == "row":
        row = cur.fetchone() if cur.description else None
        stats = {k: _plain(v) for k, v in dict(row or {}).items()}
        return int(stats.get("inserted") or 0), stats, []
    if step.returns == "rows":
        rows = cur.fetchall() if cur.description else []
        return 0, {}, [RoutedRows.model_validate({"from_table": table, **dict(r)}) for r in rows]
    return max(cur.rowcount, 0), {}, []


def _result_error(e: ValueError | TypeError) -> str:
    """The error of a step whose result (``returns``) or counters cannot be read, e.g. a moved-rows
    row without ``to_table`` (pydantic's ValidationError is a ValueError)."""
    if isinstance(e, ValidationError):
        detail = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'row'}: {x['msg']}" for x in e.errors())
    else:
        detail = str(e)
    return f"unexpected step result: {detail}"[:500]


def _counts(conn: psycopg.Connection, step: CompiledStep, res: TableLoadResult) -> None:
    """Fill ``source_rows`` and ``id_retained_rows`` (plus the extra columns of the retained count,
    e.g. visit links, in ``stats``); a count that fails leaves its value unset."""
    if step.source_count_sql:
        try:
            with conn.transaction():  # a savepoint: a failed count does not abort the step
                res.source_rows = int(scalar(conn, step.source_count_sql) or 0)
        except psycopg.Error:
            pass
    if step.retained_count_sql:
        try:
            with conn.transaction():
                row = dict(fetch_one(conn, step.retained_count_sql) or {})
        except psycopg.Error:
            return
        res.id_retained_rows = int(row.pop("n", 0) or 0)
        if row:
            res.stats["visit_links"] = {k: _plain(v) for k, v in row.items()}


def _new_result(step: CompiledStep) -> TableLoadResult:
    return TableLoadResult(
        target_table=step.target_table or step.label,
        step=step.name,
        kind=step.kind,
        blocking=step.blocking,
        sql=step.sql or None,
    )


def _run_step(conn: psycopg.Connection, step: CompiledStep, cfg: ProjectConfig | None = None) -> TableLoadResult:
    res = _new_result(step)
    started = time.time()
    if step.error:
        res.error = step.error
        return res
    role = cfg.target.etl_role if cfg is not None else None
    try:
        with conn.transaction():
            if role:
                execute(conn, f"SET LOCAL ROLE {qi(role)}")
            apply_session(conn, cfg)
            _reset_counters(conn, step)
            _counts(conn, step, res)  # before the insert, which could change what the counts read
            with conn.cursor() as cur:
                cur.execute(step.sql)  # type: ignore[arg-type]
                while cur.nextset():  # multi-statement steps: the last statement reports
                    pass
                try:
                    rows_inserted, stats, moved = _read_result(cur, step, res.target_table)
                    value_counts = _read_counters(conn, step)
                except (ValueError, TypeError) as e:  # raised out of the transaction: rolled back
                    raise _UnreadableResult(_result_error(e)) from e
    except psycopg.Error as e:
        res.error = short_error(e)  # nothing was loaded: no statistics, moves or value counts
    except _UnreadableResult as e:
        res.error = str(e)
    else:
        res.rows_inserted, res.moved, res.value_counts = rows_inserted, moved, value_counts
        res.stats.update(stats)
    if res.id_retained_rows is not None:
        res.retained_rows = max(res.id_retained_rows - res.value_counts.get(ROW_SITE, {}).get("dropped", 0), 0)
    res.seconds = round(time.time() - started, 3)
    return res


def _refuse_unsafe(steps: list[CompiledStep]) -> None:
    """Mark steps whose spec SQL is unsafe to execute as errors, so they are not run."""
    from ..design.validate import sql_safety_issues

    for s in steps:
        if s.error:
            continue
        problems = [p for origin in s.origins for p in sql_safety_issues(origin)]
        if problems:
            s.error = "refused: unsafe SQL in the spec: " + "; ".join(problems)


def _skip_reason(step: CompiledStep, failed: set[str]) -> str | None:
    if not step.requires or step.requires not in failed:
        return None
    if _is_cdm_table(step.requires):
        return f"a step loading {step.requires} failed"
    return f"step {step.requires} failed"


def _analyze(conn: psycopg.Connection, relations: list[str]) -> None:
    """ANALYZE as the connecting user, in its own transaction; statistics are an optimisation only."""
    if not relations:
        return
    try:
        with conn.transaction():
            for rel in relations:
                execute(conn, f"ANALYZE {rel}")
    except psycopg.Error:
        pass


def _analyzed_after(step: CompiledStep, res: TableLoadResult, schemas: Schemas, last_id_map: bool) -> list[str]:
    """The relations to ANALYZE after ``step``: the id maps after the last id-map step and after
    pruning, else the tables a step wrote (and the destinations of the records it moved)."""
    id_maps = [f"{qi(schemas.etl)}.{m}" for m in _ID_MAPS]
    if last_id_map:
        return id_maps
    if res.error or res.skipped:
        return []
    if step.kind == "prune":
        return id_maps
    if step.kind not in _ANALYZED_KINDS:
        return []
    tables = [step.target_table] + [m.to_table for m in res.moved]
    return [f"{qi(schemas.cdm)}.{qi(t)}" for t in dict.fromkeys(tables) if _is_cdm_table(t)]


def run_etl(
    conn: psycopg.Connection,
    schemas: Schemas,
    spec: ETLSpec,
    cfg: ProjectConfig,
    ws: Workspace | None = None,
) -> ETLRunResult:
    """Create/empty the target tables, compile the spec and execute it step by step."""
    result = ETLRunResult(started_at=utcnow())

    create_cdm_tables(conn, schemas, include_vocab=False)
    create_etl_tables(conn, schemas)
    tables = sorted(set(target_tables(spec)) | _previous_tables(ws))
    truncate_cdm_tables(conn, schemas, tables)
    conn.commit()

    steps = compile_spec(spec, schemas, cfg.cdm_source, strict=False)
    _refuse_unsafe(steps)
    if ws is not None:
        _write_sql(ws, steps, cfg)

    last_id_map = max((i for i, s in enumerate(steps) if s.kind == "id_map"), default=-1)
    failed: set[str] = set()  # CDM tables and step labels whose steps failed or were skipped
    for i, step in enumerate(steps):
        skipped = _skip_reason(step, failed)
        if skipped:  # e.g. id-map pruning after a failed person load: keep the stable ids
            res = _new_result(step)
            res.skipped = skipped
        else:
            res = _run_step(conn, step, cfg)
        if res.error or res.skipped:
            failed.add(step.label)  # the steps that require this one are skipped too
            if step.target_table and step.kind != "route":  # route steps share their source table's name
                failed.add(step.target_table)
        result.steps.append(res)
        if ws is not None:
            ws.audit(
                "etl_step",
                step=res.step,
                kind=step.kind,
                blocking=step.blocking,
                target_table=res.target_table,
                rows_inserted=res.rows_inserted,
                source_rows=res.source_rows,
                retained_rows=res.retained_rows,
                seconds=res.seconds,
                error=res.error,
                value_counts=res.value_counts,
                stats=res.stats,
                moved=[m.model_dump() for m in res.moved],
                **({"skipped": skipped} if skipped else {}),
            )
        _analyze(conn, _analyzed_after(step, res, schemas, i == last_id_map))
    conn.commit()

    loaded = {r.target_table for r in result.steps if r.error is None and not r.skipped and r.kind != "route"}
    ok_tables = loaded & set(target_tables(spec))
    try:
        with conn.transaction():
            for t in sorted(ok_tables):
                execute(conn, f"ANALYZE {qi(schemas.cdm)}.{qi(t)}")
            for m in _ID_MAPS:
                execute(conn, f"ANALYZE {qi(schemas.etl)}.{m}")
    except psycopg.Error:
        pass  # statistics are an optimisation only
    conn.commit()

    result.finished_at = utcnow()
    if ws is not None:
        ws.save_json(ws.etl_run_json, result)
        ws.audit(
            "etl_run",
            ok=result.ok,
            errors=[f"{r.step}: {r.error}" for r in result.errors()],
            rows={r.step: r.rows_inserted for r in result.steps},
            totals={
                "rows_inserted": sum(r.rows_inserted for r in result.steps if r.error is None),
                "moved_rows": sum(m.moved_rows for r in result.steps for m in r.moved),
                "errors": len(result.errors()),
                "blocking_errors": len(result.blocking_errors()),
                "skipped": sum(1 for r in result.steps if r.skipped),
            },
        )
    return result
