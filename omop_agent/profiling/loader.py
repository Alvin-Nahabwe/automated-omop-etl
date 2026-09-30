"""Stage the source data into PostgreSQL (``staging_schema``) as text columns.

Every source table is copied verbatim with every column typed ``text`` (the same stance
White Rabbit takes when scanning). The profile records inferred types, and the ETL SQL casts
explicitly, so CSV exports and database sources behave the same way downstream. Empty and
whitespace-only values become NULL.

A manifest (``staging_manifest.json``) keeps the original table/column names and any
declared source types.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

import psycopg

from ..config import ProjectConfig
from ..db import Schemas, ensure_schema, execute, qi
from ..workspace import Workspace

MANIFEST = "staging_manifest.json"


def sanitize_identifier(name: str, used: set[str] | None = None) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_") or "col"
    if s[0].isdigit():
        s = "c_" + s
    s = s[:63]
    if used is not None:
        base, n = s, 2
        while s in used:
            s = f"{base[:60]}_{n}"
            n += 1
        used.add(s)
    return s


def _clean(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return format(v, "f")
    if isinstance(v, bytes):
        return v.hex()
    s = str(v)
    return None if s.strip() == "" else s


def _create_and_copy(
    conn: psycopg.Connection, schema: str, table: str, columns: list[str], rows: Iterable[Iterable[Any]]
) -> int:
    execute(conn, f"DROP TABLE IF EXISTS {qi(schema)}.{qi(table)}")
    cols_sql = ", ".join(f"{qi(c)} text" for c in columns)
    execute(conn, f"CREATE TABLE {qi(schema)}.{qi(table)} ({cols_sql})")
    n = 0
    col_list = ", ".join(qi(c) for c in columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {qi(schema)}.{qi(table)} ({col_list}) FROM STDIN") as cp:
            for row in rows:
                vals = [_clean(v) for v in row]
                if len(vals) < len(columns):
                    vals += [None] * (len(columns) - len(vals))
                cp.write_row(vals[: len(columns)])
                n += 1
    return n


def _csv_rows(path: Path, delimiter: str, encoding: str) -> tuple[list[str], Iterator[list[str]]]:
    fh = path.open(newline="", encoding=encoding, errors="replace")
    reader = csv.reader(fh, delimiter=delimiter)
    header = next(reader, [])

    def gen() -> Iterator[list[str]]:
        try:
            yield from reader
        finally:
            fh.close()

    return header, gen()


def stage_csv(cfg: ProjectConfig, conn: psycopg.Connection, schemas: Schemas) -> dict[str, Any]:
    assert cfg.source.path is not None
    files = sorted(p for p in Path(cfg.source.path).iterdir() if p.suffix.lower() in {".csv", ".tsv", ".txt"})
    wanted = {t.lower() for t in cfg.source.tables} if cfg.source.tables else None
    manifest: dict[str, Any] = {}
    used_tables: set[str] = set()
    for f in files:
        table = sanitize_identifier(f.stem, used_tables)
        if wanted and table not in wanted and f.stem.lower() not in wanted:
            continue
        delim = "\t" if f.suffix.lower() == ".tsv" else cfg.source.delimiter
        header, rows = _csv_rows(f, delim, cfg.source.encoding)
        used_cols: set[str] = set()
        cols = [sanitize_identifier(h, used_cols) for h in header]
        n = _create_and_copy(conn, schemas.src, table, cols, rows)
        manifest[table] = {
            "source": f.name,
            "rows": n,
            "columns": [{"name": c, "source_name": h, "declared_type": None} for c, h in zip(cols, header)],
        }
    return manifest


def stage_database(cfg: ProjectConfig, conn: psycopg.Connection, schemas: Schemas) -> dict[str, Any]:
    import sqlalchemy as sa

    engine = sa.create_engine(cfg.source.url)  # type: ignore[arg-type]
    insp = sa.inspect(engine)
    names = insp.get_table_names(schema=cfg.source.db_schema)
    if cfg.source.tables:
        wanted = {t.lower() for t in cfg.source.tables}
        names = [n for n in names if n.lower() in wanted]
    manifest: dict[str, Any] = {}
    used_tables: set[str] = set()
    md = sa.MetaData()
    with engine.connect() as src:
        for name in sorted(names):
            t = sa.Table(name, md, autoload_with=engine, schema=cfg.source.db_schema)
            used_cols: set[str] = set()
            cols = [sanitize_identifier(c.name, used_cols) for c in t.columns]
            table = sanitize_identifier(name, used_tables)
            result = src.execution_options(stream_results=True, yield_per=10_000).execute(sa.select(t))
            n = _create_and_copy(conn, schemas.src, table, cols, (tuple(r) for r in result))
            manifest[table] = {
                "source": f"{cfg.source.db_schema + '.' if cfg.source.db_schema else ''}{name}",
                "rows": n,
                "columns": [
                    {"name": c, "source_name": col.name, "declared_type": str(col.type)}
                    for c, col in zip(cols, t.columns)
                ],
            }
    engine.dispose()
    return manifest


def stage_source(cfg: ProjectConfig, conn: psycopg.Connection, schemas: Schemas, ws: Workspace | None = None) -> dict[str, Any]:
    """Copy all source tables into ``schemas.src``. Returns the manifest (also saved to the workspace)."""
    ensure_schema(conn, schemas.src)
    manifest = stage_csv(cfg, conn, schemas) if cfg.source.type == "csv" else stage_database(cfg, conn, schemas)
    conn.commit()
    if ws is not None:
        (ws.root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        ws.audit("source_staged", tables={t: m["rows"] for t, m in manifest.items()}, schema=schemas.src)
    return manifest


def load_manifest(ws: Workspace) -> dict[str, Any] | None:
    p = ws.root / MANIFEST
    return json.loads(p.read_text()) if p.exists() else None
