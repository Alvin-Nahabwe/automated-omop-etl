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
build (``ETLRunResult.ok``). A step that fails for a missing privilege is marked not repairable
(``TableLoadResult.repairable``), and its error names the grants to run: the repair agent cannot
fix it in the spec. After the id maps, and after every step that wrote rows, the written tables
are ANALYZEd, so the next steps are planned with fresh statistics. The route steps
(``etl.routing``) report the records they moved (``moved``) under the name of the table they
moved them from, which they never mark as failed or loaded; whether the vocabulary's concept
table is indexed on concept_id decides the form of their concept lookup.

The spec's SQL is written by an LLM (and possibly edited by people), so before anything runs
each step's fragments are checked again by ``design.validate.sql_safety_issues``; unsafe steps
are refused and reported as errors. With ``target.etl_role`` set, every step also runs under
that role (``SET LOCAL ROLE``), which needs USAGE and SELECT on the staging and vocabulary
schemas, SELECT/INSERT/DELETE on the CDM and ETL schemas (DELETE for retiring and restoring
id-map keys, and for custom SQL), and membership of the connecting user. ETL tables the build
creates (such as those a new version adds) are granted to the role. The CDM tables are emptied
before the steps run (with every table the route steps can write, and every table an earlier
build moved records to), and ANALYZEd, as the connecting user; the facts an earlier build derived
(``{etl}.build_facts``: the release date the observation-period merge stores for cdm_source) are
deleted in the same transaction.

``upgrade_notice`` tells the first build of a project last built by an older version what builds
now do differently (``UPGRADE_BEHAVIOURS``); the pipeline logs it once and records an
``upgrade_notice`` audit event.
"""

from __future__ import annotations

import json
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
from ..fsutil import atomic_write_text
from ..models import ETLRunResult, ETLSpec, RoutedRows, TableLoadResult
from ..workspace import Workspace, utcnow
from .compiler import CompiledStep, compile_spec, target_tables
from .policies import ROW_SITE
from .routing import route_sources, route_targets, vocab_indexed
from .session import apply_session, session_sql

# Value counters a step can report per site (a CDM field or 'where'); the site ROW_SITE has one
# counter, 'dropped', the rows dropped for a NULL required value: value_counts['row']['dropped'].
VALUE_COUNT_CATEGORIES = (
    "invalid", "imputed", "partial_null", "offset_ignored", "offset_converted", "defaulted", "required_null",
)
# Steps after which the tables they wrote are ANALYZEd.
_ANALYZED_KINDS = {"insert", "custom", "route", "merge", "derived"}
# The id maps and the keys they retired, ANALYZEd after the last id-map step and after pruning.
_ID_MAPS = ("person_map", "visit_map", "person_map_retired", "visit_map_retired")


def _is_cdm_table(name: str | None) -> bool:
    cdm = load_cdm_spec()
    return bool(name) and cdm.has_table(name) and cdm.table(name).schema_class == "CDM"  # type: ignore[arg-type]


def _previous_tables(ws: Workspace | None) -> set[str]:
    """CDM tables loaded by the previous run (so tables dropped from the spec are emptied too), and
    every table a build that never finished may have written (``Workspace.etl_tables_json``: a
    cancelled build keeps the rows of the steps it committed, e.g. records moved to another table
    or eras, though no etl_run.json names them)."""
    if ws is None:
        return set()
    names: set[str] = set()
    if ws.etl_tables_json.exists():
        try:
            names |= {str(n) for n in json.loads(ws.etl_tables_json.read_text())}
        except (ValueError, TypeError):
            pass
    if ws.etl_run_json.exists():
        try:
            prev = ws.load_json(ws.etl_run_json, ETLRunResult)
        except Exception:  # an unreadable old result must not block a new run
            prev = None
        if prev is not None:
            names |= {s.target_table for s in prev.steps} | {m.to_table for s in prev.steps for m in s.moved}
    return {n for n in names if _is_cdm_table(n)}


# The behaviours a build of an older version did not have, announced once per project by its first
# build after the update (an ``upgrade_notice`` audit event and one log line; README.md, "Upgrading").
UPGRADE_NOTICE = "hdss-correct-output"
UPGRADE_BEHAVIOURS = (
    "person, death and visit_occurrence keep one row per key, and every person and visit mapping adds its keys "
    "to the id maps",
    "id-map keys that are not loaded are retired with their ids instead of deleted",
    "values that cannot be read become NULL and are counted, and NULL required values follow the CDM conventions "
    "(concept 0, end date = start date, or the row is dropped)",
    "each person's observation periods are merged, and open episodes end at the source release date or at death",
    "records move to the table of their standard concept's domain (route_by_domain), and every table they can move "
    "to is emptied before each build, also one the spec does not load",
    "condition_era and drug_era are derived with the OHDSI era logic (derive_eras)",
    "custom SQL of a table without a mapping runs at that table's place in the load order",
)


# What the report and `check` say about a CDM whose last build an older version made (``built_by_older_version``).
OLDER_BUILD_NOTE = (
    "the CDM was built by an earlier version of omop-agent (its ETL run records no step kinds), without what this "
    "version's build does beyond the mappings (one row per key, merged observation periods, routing by domain, "
    "derived eras); the design section and the checks describe this version's build, so the era checks fail on "
    "era tables that were never built. Rebuild it (omop-agent build, or run) before relying on these results"
)


def built_by_older_version(run: ETLRunResult | None) -> bool:
    """Whether ``run`` was written by a version before these build behaviours (its steps carry no
    ``kind``): its CDM lacks them until the project is rebuilt."""
    return run is not None and bool(run.steps) and all(s.kind is None for s in run.steps)


def upgrade_notice(ws: Workspace) -> list[str]:
    """The new build behaviours (``UPGRADE_BEHAVIOURS``) when the project's last build was made by a
    version without them (its steps carry no ``kind``) and no ``upgrade_notice`` event announced them
    yet; else []. A project that was never built has nothing to be told."""
    if not ws.etl_run_json.exists():
        return []
    try:
        prev = ws.load_json(ws.etl_run_json, ETLRunResult)
    except Exception:  # an unreadable old result: no notice rather than a failed build
        return []
    if not built_by_older_version(prev):
        return []
    if ws.audit_log.exists():
        for line in ws.audit_log.read_text(encoding="utf-8").splitlines():
            if UPGRADE_NOTICE in line and '"upgrade_notice"' in line:
                return []
    return list(UPGRADE_BEHAVIOURS)


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
    except psycopg.errors.InsufficientPrivilege as e:
        res.error = f"{short_error(e)}. {_privilege_hint(cfg)}"[:1000]
        res.repairable = False  # a grant, not the spec: never sent to the repair agent
    except psycopg.Error as e:
        hint = _server_hint(e)
        res.error = f"{short_error(e)}. {hint}"[:1000] if hint else short_error(e)
        if hint:
            res.repairable = False  # the server's limits, not the spec: the repair agent cannot help
    except _UnreadableResult as e:
        res.error = str(e)
    if res.error:  # nothing was loaded: no statistics (the visit links read before it), moves or value counts
        res.stats = {}
    else:
        res.rows_inserted, res.moved, res.value_counts = rows_inserted, moved, value_counts
        res.stats.update(stats)
    if res.id_retained_rows is not None:
        res.retained_rows = max(res.id_retained_rows - res.value_counts.get(ROW_SITE, {}).get("dropped", 0), 0)
    res.seconds = round(time.time() - started, 3)
    return res


def _server_hint(e: psycopg.Error) -> str | None:
    """What to do when a step was stopped by the database server rather than by its SQL: out of
    disk, memory or connections (SQLSTATE class 53), a lock it waited for too long (55P03), a
    statement timeout, an administrator's cancel or a shutdown (class 57). None for other errors.
    Such a step is not sent to the repair agent: its tools cannot change the server."""
    state = getattr(e, "sqlstate", None) or ""
    message = str(e).lower()
    if state == "53100":
        return ("The database server ran out of disk space (a large sort or join writes temporary files); this is "
                "not a problem of the ETL spec. Free disk space on the server, or point temp_tablespaces at a larger "
                "disk, and build again.")
    if state.startswith("53"):
        return ("The database server ran out of a resource (memory or connections); this is not a problem of the "
                "ETL spec. Ask an administrator to check the server's limits, and build again.")
    if state == "55P03":
        return ("The step waited longer than the server's lock_timeout for a table another session was using; this "
                "is not a problem of the ETL spec. Build again when that session has finished, or raise lock_timeout "
                "for the build's user.")
    if state == "57014" and "statement timeout" in message:
        return ("The step ran longer than the server's statement_timeout (set for the database or the build's "
                "user); this is usually not a problem of the ETL spec. An administrator can raise it for the build's "
                "user (ALTER ROLE <user> SET statement_timeout = 0); if the step is slow because of its SQL (a "
                "subquery per row, a join without its key), a person should look at the spec.")
    if state.startswith("57"):
        return ("The step was stopped on the database server (cancelled by an administrator, or the server shut "
                "down); this is not a problem of the ETL spec. Build again.")
    return None


def _privilege_hint(cfg: ProjectConfig | None) -> str:
    """What to grant when a step fails for a missing privilege (e.g. ETL tables a new version
    added, which an administrator granted before they existed)."""
    if cfg is None:
        return "The build's database user lacks a privilege; this is not a problem of the ETL spec."
    t = cfg.target
    who = f"The ETL role {t.etl_role}" if t.etl_role else "The database user of target.url"
    to = qi(t.etl_role) if t.etl_role else "<user>"

    def names(*schemas: str) -> str:
        return ", ".join(qi(s) for s in dict.fromkeys(schemas))

    return (
        f"{who} lacks a privilege the build needs; this is not a problem of the ETL spec. An administrator can "
        f"grant it: GRANT USAGE ON SCHEMA {names(t.staging_schema, t.vocab_schema, t.cdm_schema, t.etl_schema)} "
        f"TO {to}; GRANT SELECT ON ALL TABLES IN SCHEMA {names(t.staging_schema, t.vocab_schema)} TO {to}; "
        f"GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA {names(t.cdm_schema, t.etl_schema)} TO {to} "
        "(see README.md, 'Privacy and safety')."
    )


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
    create_etl_tables(conn, schemas, cfg.target.etl_role)
    # the route destinations too, and the tables an earlier build moved records to
    tables = sorted(set(target_tables(spec)) | set(route_targets(spec)) | _previous_tables(ws))
    if ws is not None:  # until etl_run.json names them, a cancelled build must not leave rows behind
        atomic_write_text(ws.etl_tables_json, json.dumps(tables) + "\n")
    truncate_cdm_tables(conn, schemas, tables)
    # what an earlier build derived (the release date) goes with the rows it was derived from
    execute(conn, f"DELETE FROM {qi(schemas.etl)}.build_facts")
    conn.commit()

    indexed = vocab_indexed(conn, schemas) if route_sources(spec) else True
    steps = compile_spec(spec, schemas, cfg.cdm_source, etl=cfg.etl, vocab_indexed=indexed, strict=False)
    _refuse_unsafe(steps)
    if ws is not None:
        _write_sql(ws, steps, cfg)

    last_id_map = max((i for i, s in enumerate(steps) if s.kind == "id_map"), default=-1)
    failed: set[str] = set()  # CDM tables and step labels whose steps failed or were skipped
    for i, step in enumerate(steps):
        skipped = _skip_reason(step, failed)
        if skipped:  # e.g. a route step after a failed route_prepare
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
        ws.etl_tables_json.unlink(missing_ok=True)
        ws.audit(
            "etl_run",
            ok=result.ok,
            errors=[f"{r.step}: {r.error}" for r in result.errors()],
            rows={r.step: r.rows_inserted for r in result.steps},
            totals={
                # what the log and the report call the rows loaded: the mappings' and custom SQL's rows
                "rows_loaded": result.rows_loaded(),
                "derived_rows": result.rows_derived(),  # observation periods, eras (apart from rows_loaded)
                "moved_rows": result.rows_moved(),
                # every step's count but the routes' (they move rows; route_prepare's count is ETL
                # bookkeeping): the rows loaded and derived plus the id-map keys numbered and retired and
                # the merged periods (which replace episodes counted in rows_loaded)
                "rows_inserted": sum(r.rows_inserted for r in result.steps if r.error is None and r.kind != "route"),
                "errors": len(result.errors()),
                "blocking_errors": len(result.blocking_errors()),
                "skipped": sum(1 for r in result.steps if r.skipped),
            },
        )
    return result
