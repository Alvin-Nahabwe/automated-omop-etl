"""PostgreSQL helpers (psycopg 3)."""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import psycopg
from psycopg.rows import dict_row

from .config import TargetConfig

_SIMPLE_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


def normalize_url(url: str) -> str:
    """Accept SQLAlchemy-style URLs (``postgresql+psycopg://``) as well as libpq ones."""
    return re.sub(r"^postgres(ql)?\+\w+://", "postgresql://", url)


def connect(url: str, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(normalize_url(url), autocommit=autocommit, row_factory=dict_row)


def qi(name: str) -> str:
    """Quote an identifier for generated SQL text (left bare when it is already safe)."""
    if _SIMPLE_IDENT.match(name) and name not in _RESERVED:
        return name
    return '"' + name.replace('"', '""') + '"'


def ql(value: Any) -> str:
    """Render a Python value as a SQL literal for generated SQL text."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


_RESERVED = {
    "all", "and", "any", "as", "asc", "both", "case", "cast", "check", "column", "constraint",
    "create", "default", "desc", "distinct", "do", "else", "end", "except", "false", "for",
    "foreign", "from", "grant", "group", "having", "in", "into", "is", "join", "leading", "limit",
    "not", "null", "offset", "on", "only", "or", "order", "primary", "references", "select",
    "table", "then", "to", "trailing", "true", "union", "unique", "user", "using", "when",
    "where", "with", "window", "year", "month", "day", "date", "time", "timestamp",
}


@dataclass(frozen=True)
class Schemas:
    cdm: str
    vocab: str
    src: str
    etl: str
    results: str

    @classmethod
    def from_config(cls, t: TargetConfig) -> "Schemas":
        return cls(t.cdm_schema, t.vocab_schema, t.staging_schema, t.etl_schema, t.results_schema)

    def render(self, sql: str) -> str:
        """Substitute ``{cdm}``, ``{vocab}``, ``{src}``, ``{etl}``, ``{results}`` placeholders.

        Only these five tokens are replaced, so other braces in SQL (regex quantifiers, JSON)
        are left alone.
        """
        for key in ("cdm", "vocab", "src", "etl", "results"):
            sql = sql.replace("{" + key + "}", qi(getattr(self, key)))
        return sql


def fetch_all(conn: psycopg.Connection, sql: str, params: Sequence[Any] | dict | None = None) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params)  # type: ignore[arg-type]
        return list(cur.fetchall()) if cur.description else []


def fetch_one(conn: psycopg.Connection, sql: str, params: Sequence[Any] | dict | None = None) -> dict | None:
    rows = fetch_all(conn, sql, params)
    return rows[0] if rows else None


def scalar(conn: psycopg.Connection, sql: str, params: Sequence[Any] | dict | None = None) -> Any:
    row = fetch_one(conn, sql, params)
    return None if row is None else next(iter(row.values()))


def execute(conn: psycopg.Connection, sql: str, params: Sequence[Any] | dict | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params)  # type: ignore[arg-type]
        return cur.rowcount


def ensure_schema(conn: psycopg.Connection, schema: str) -> None:
    execute(conn, f"CREATE SCHEMA IF NOT EXISTS {qi(schema)}")


def table_exists(conn: psycopg.Connection, schema: str, table: str) -> bool:
    return bool(
        scalar(
            conn,
            "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
            (schema, table),
        )
    )


def list_tables(conn: psycopg.Connection, schema: str) -> list[str]:
    rows = fetch_all(
        conn,
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s ORDER BY table_name",
        (schema,),
    )
    return [r["table_name"] for r in rows]


def list_columns(conn: psycopg.Connection, schema: str, table: str) -> list[dict]:
    return fetch_all(
        conn,
        """SELECT column_name, data_type FROM information_schema.columns
           WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position""",
        (schema, table),
    )


def row_count(conn: psycopg.Connection, schema: str, table: str) -> int:
    return int(scalar(conn, f"SELECT COUNT(*) FROM {qi(schema)}.{qi(table)}") or 0)


@contextmanager
def rollback_only(conn: psycopg.Connection, timeout_ms: int = 60_000) -> Iterator[psycopg.Connection]:
    """Run statements in a transaction that is always rolled back (for dry runs)."""
    with conn.transaction(force_rollback=True):
        execute(conn, f"SET LOCAL statement_timeout = {int(timeout_ms)}")
        yield conn


def dry_run(conn: psycopg.Connection, sql: str, timeout_ms: int = 60_000) -> tuple[int | None, str | None]:
    """Execute ``sql`` and roll back. Returns (rowcount, error)."""
    try:
        with rollback_only(conn, timeout_ms):
            n = execute(conn, sql)
        return n, None
    except psycopg.Error as e:
        return None, short_error(e)


def readonly_query(
    conn: psycopg.Connection, sql: str, limit: int = 50, timeout_ms: int = 30_000
) -> list[dict]:
    """Run a query in a READ ONLY transaction that is rolled back, returning at most ``limit`` rows."""
    with conn.transaction(force_rollback=True):
        execute(conn, "SET TRANSACTION READ ONLY")
        execute(conn, f"SET LOCAL statement_timeout = {int(timeout_ms)}")
        with conn.cursor() as cur:
            cur.execute(sql)  # type: ignore[arg-type]
            return list(cur.fetchmany(limit)) if cur.description else []


def short_error(e: BaseException) -> str:
    msg = str(e).strip()
    diag = getattr(e, "diag", None)
    if diag is not None and getattr(diag, "message_primary", None):
        msg = diag.message_primary
        if diag.message_detail:
            msg += f" DETAIL: {diag.message_detail}"
        if diag.message_hint:
            msg += f" HINT: {diag.message_hint}"
    return msg[:2000]
