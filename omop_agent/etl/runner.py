"""Run a compiled ETL spec against PostgreSQL.

Every step runs in its own transaction, so one failing table (say a NOT NULL violation in
``visit_occurrence``) is reported and the remaining tables still load. The SQL of each step is
written to ``sql/<step>.sql`` and the outcome to ``etl_run.json`` for review and for the
data-quality stage, which compares ``source_rows`` with ``rows_inserted`` (record retention).
"""

from __future__ import annotations

import time

import psycopg

from ..cdm.ddl import create_cdm_tables, create_etl_tables, truncate_cdm_tables
from ..cdm.spec import load_cdm_spec
from ..config import ProjectConfig
from ..db import Schemas, execute, qi, scalar, short_error
from ..models import ETLRunResult, ETLSpec, TableLoadResult
from ..workspace import Workspace, utcnow
from .compiler import CompiledStep, compile_spec, target_tables


def _previous_tables(ws: Workspace | None) -> set[str]:
    """CDM tables loaded by the previous run (so tables dropped from the spec are emptied too)."""
    if ws is None or not ws.etl_run_json.exists():
        return set()
    try:
        prev = ws.load_json(ws.etl_run_json, ETLRunResult)
    except Exception:  # an unreadable old result must not block a new run
        return set()
    cdm = load_cdm_spec()
    return {
        s.target_table
        for s in prev.steps
        if cdm.has_table(s.target_table) and cdm.table(s.target_table).schema_class == "CDM"
    }


def _write_sql(ws: Workspace, steps: list[CompiledStep]) -> None:
    for old in ws.sql_dir.glob("*.sql"):
        old.unlink()
    for step in steps:
        body = step.sql if not step.error else f"-- {step.error}\n"
        (ws.sql_dir / f"{step.name}.sql").write_text(body.rstrip() + "\n")


def _run_step(conn: psycopg.Connection, step: CompiledStep) -> TableLoadResult:
    res = TableLoadResult(
        target_table=step.target_table or step.name.split("_", 1)[-1],
        step=step.name,
        sql=step.sql or None,
    )
    started = time.time()
    if step.error:
        res.error = step.error
        return res
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(step.sql)  # type: ignore[arg-type]
                rows = cur.rowcount
                while cur.nextset():  # multi-statement steps: report the last statement
                    rows = cur.rowcount
                res.rows_inserted = max(rows, 0)
            if step.source_count_sql:
                res.source_rows = int(scalar(conn, step.source_count_sql) or 0)
    except psycopg.Error as e:
        res.error = short_error(e)
        if step.source_count_sql:
            try:
                with conn.transaction():
                    res.source_rows = int(scalar(conn, step.source_count_sql) or 0)
            except psycopg.Error:
                pass
    res.seconds = round(time.time() - started, 3)
    return res


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
    if ws is not None:
        _write_sql(ws, steps)

    for step in steps:
        res = _run_step(conn, step)
        result.steps.append(res)
        if ws is not None:
            ws.audit(
                "etl_step",
                step=res.step,
                kind=step.kind,
                target_table=res.target_table,
                rows_inserted=res.rows_inserted,
                source_rows=res.source_rows,
                seconds=res.seconds,
                error=res.error,
            )
    conn.commit()

    ok_tables = {r.target_table for r in result.steps if r.error is None} & set(target_tables(spec))
    try:
        with conn.transaction():
            for t in sorted(ok_tables):
                execute(conn, f"ANALYZE {qi(schemas.cdm)}.{qi(t)}")
            execute(conn, f"ANALYZE {qi(schemas.etl)}.person_map")
            execute(conn, f"ANALYZE {qi(schemas.etl)}.visit_map")
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
        )
    return result
