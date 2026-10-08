"""Session settings pinned wherever spec SQL is evaluated.

Spec expressions run in the build (``etl.runner``), in the vocabulary stage (``query_rows``,
used by ``collect_terms`` and the coverage queries), in the repair agent's counts and EXPLAINs,
and in the design validator's EXPLAINs. They must read dates, times and strings the same way in
every one of them, whatever the server or role defaults are:

* ``DateStyle = 'ISO, YMD'``: an ambiguous ``'03/04/2020'::date`` does not silently depend on
  the server's setting (MDY in a default install, DMY elsewhere). Under YMD a day-first or
  month-first text such as ``'31/12/2020'::date`` is an error ("date/time field value out of
  range"), not a guess, but one with a two-digit year (``'31/12/20'::date``) is read year-first
  (2031-12-20) without an error: such columns need ``to_date(x, 'DD/MM/YYYY')`` (the template the
  profile shows), as the design prompt already asks. ISO and other year-first text casts as before;
* ``TimeZone``: ``etl.timezone`` when it is set, so ``timestamptz`` values and ``AT TIME ZONE``
  are read in the site's zone, whatever the server's is. Unset, the session keeps the server's
  (or role's) zone, as builds always did;
* ``omop_agent.timezone``: ``etl.timezone``, or empty when it is unset. The tolerant conversions
  (``etl/functions.sql``) read it: a value with an offset ('Z', '+03:00') is converted to this zone,
  or, without one, kept at the wall clock as written;
* ``omop_agent.imputation_cap``: the latest date a partial date may be completed to, the
  configured ``cdm_source.source_release_date`` or else the current date;
* ``standard_conforming_strings = on`` (the default): a backslash is an ordinary character in
  '...' strings, as ``sqlguard`` reads them, and as the literals the compiler writes assume.

Every setting is transaction-local (``SET LOCAL``, ``set_config(..., true)``): run them inside
the transaction or savepoint of the statements they apply to.
"""

from __future__ import annotations

import psycopg

from ..config import ProjectConfig
from ..db import execute, ql


def session_sql(cfg: ProjectConfig | None = None) -> list[str]:
    """The statements that pin the session settings (defaults when ``cfg`` is None)."""
    zone = cfg.etl.timezone if cfg is not None else None
    release = cfg.cdm_source.source_release_date if cfg is not None else None
    cap = ql(release.isoformat()) if release is not None else "CURRENT_DATE::text"
    return [
        "SET LOCAL datestyle = 'ISO, YMD';",
        "SET LOCAL standard_conforming_strings = on;",
        *([f"SET LOCAL timezone = {ql(zone)};"] if zone else []),
        f"SELECT set_config('omop_agent.imputation_cap', {cap}, true),"
        f" set_config('omop_agent.timezone', {ql(zone or '')}, true);",
    ]


def apply_session(conn: psycopg.Connection, cfg: ProjectConfig | None = None) -> None:
    """Pin the settings for the rest of the current transaction (or savepoint)."""
    for stmt in session_sql(cfg):
        execute(conn, stmt)
