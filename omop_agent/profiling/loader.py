"""Stage the source data into PostgreSQL (``staging_schema``) as text columns.

Every source table is copied verbatim with every column typed ``text`` (the same stance
White Rabbit takes when scanning). The profile records inferred types, and the ETL SQL casts
explicitly, so CSV exports and database sources behave the same way downstream. Empty and
whitespace-only values become NULL.

A manifest (``staging_manifest.json``) keeps the original table/column names and any
declared source types.

CSV files are read strictly, so a file is never staged in a corrupted form:

* text is decoded with ``source.encoding`` and no replacement characters; an undecodable file
  is an error that names the file and suggests setting ``source.encoding``;
* a row whose number of fields differs from the header's (usually a wrong
  ``source.delimiter`` or an unquoted delimiter inside a value) is never padded or cut. The
  manifest counts such "ragged" rows per file with a few example line numbers (never the row
  contents). With ``strict=True`` (the default) any ragged row is an error; with
  ``strict=False`` those rows are left out of the staged table.

Blank lines are skipped. Staging refuses to run when the staging schema is also the CDM,
vocabulary, ETL or results schema, since it drops and recreates tables there.
"""

from __future__ import annotations

import codecs
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
STRICT_ROWS = True  # default: a CSV row with the wrong number of fields stops staging
MAX_RAGGED_EXAMPLES = 5  # example line numbers kept per file


class SourceDataError(ValueError):
    """A source file cannot be staged faithfully (encoding, row shape) or the target is unsafe."""


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


def _check_staging_schema(schemas: Schemas) -> None:
    """Staging drops and recreates tables: never do that in the CDM, vocabulary, ETL or results schema."""
    roles = {"cdm": schemas.cdm, "vocab": schemas.vocab, "etl": schemas.etl, "results": schemas.results}
    clash = [role for role, name in roles.items() if name == schemas.src]
    if clash:
        raise SourceDataError(
            f"Refusing to stage source data into schema '{schemas.src}': it is also the {' and '.join(clash)} "
            "schema, and staging drops and recreates its tables. Set target.staging_schema to a separate schema."
        )


def _create_and_copy(
    conn: psycopg.Connection, schema: str, table: str, columns: list[str], rows: Iterable[Iterable[Any]]
) -> int:
    """(Re)create ``schema.table`` with text columns and COPY ``rows`` in. Each row must have
    exactly ``len(columns)`` values."""
    execute(conn, f"DROP TABLE IF EXISTS {qi(schema)}.{qi(table)}")
    cols_sql = ", ".join(f"{qi(c)} text" for c in columns)
    execute(conn, f"CREATE TABLE {qi(schema)}.{qi(table)} ({cols_sql})")
    n = 0
    col_list = ", ".join(qi(c) for c in columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {qi(schema)}.{qi(table)} ({col_list}) FROM STDIN") as cp:
            for row in rows:
                cp.write_row([_clean(v) for v in row])
                n += 1
    return n


def _decode_error_line(path: Path, encoding: str) -> tuple[int, str] | None:
    """(line number, 'byte 0x.. : reason') of the first undecodable byte, or None."""
    dec = codecs.getincrementaldecoder(encoding)(errors="strict")
    n = 0
    with path.open("rb") as fh:
        for n, line in enumerate(fh, 1):
            try:
                dec.decode(line)
            except UnicodeDecodeError as e:
                return n, f"byte 0x{e.object[e.start]:02x}: {e.reason}"
        try:
            dec.decode(b"", final=True)
        except UnicodeDecodeError as e:
            return n, f"{e.reason} at the end of the file"
    return None


class _CsvFile:
    """Reads one CSV file strictly: rows with the header's width only; ragged rows are counted.

    Decoding and CSV syntax errors are raised as :class:`SourceDataError` naming the file and
    the line, never quoting the data.
    """

    def __init__(self, path: Path, delimiter: str, encoding: str):
        self.path = path
        self.delimiter = delimiter
        # utf-8-sig also reads plain UTF-8, and keeps a byte-order mark out of the first column name
        self.encoding = "utf-8-sig" if codecs.lookup(encoding).name == "utf-8" else encoding
        self.config_encoding = encoding
        self.too_many = 0
        self.too_few = 0
        self.blank_lines = 0
        self.examples: list[int] = []
        self.header: list[str] = []
        self._fh = path.open(newline="", encoding=self.encoding, errors="strict")
        self._reader = csv.reader(self._fh, delimiter=delimiter)
        try:
            self.header = self._next() or []
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._fh.close()

    @property
    def ragged(self) -> int:
        return self.too_many + self.too_few

    def _next(self) -> list[str] | None:
        try:
            return next(self._reader)
        except StopIteration:
            return None
        except UnicodeDecodeError as e:
            # text is decoded in chunks, so find the line by scanning the file again
            where = _decode_error_line(self.path, self.encoding)
            detail = f"line {where[0]}, {where[1]}" if where else f"byte 0x{e.object[e.start]:02x}: {e.reason}"
            raise SourceDataError(
                f"{self.path.name} is not valid {self.config_encoding} text ({detail}). Set source.encoding in "
                "the project config to the file's encoding, e.g. latin-1 or cp1252."
            ) from None
        except csv.Error as e:
            raise SourceDataError(
                f"{self.path.name}, line {self._reader.line_num}: cannot parse the CSV ({e}). "
                f"Check source.delimiter (currently {self.delimiter!r}) and the file's quoting."
            ) from None

    def rows(self) -> Iterator[list[str]]:
        width = len(self.header)
        while True:
            start = self._reader.line_num + 1  # physical line on which the record starts
            row = self._next()
            if row is None:
                return
            if not row:
                self.blank_lines += 1
                continue
            if len(row) != width:
                if len(row) > width:
                    self.too_many += 1
                else:
                    self.too_few += 1
                if len(self.examples) < MAX_RAGGED_EXAMPLES:
                    self.examples.append(start)
                continue
            yield row

    def ragged_error(self) -> SourceDataError:
        lines = ", ".join(str(n) for n in self.examples) + (" ..." if self.ragged > len(self.examples) else "")
        return SourceDataError(
            f"{self.path.name}: {self.ragged} row(s) do not have the header's {len(self.header)} fields "
            f"({self.too_many} with more, {self.too_few} with fewer), e.g. line(s) {lines}. Staging stopped: "
            f"padding or cutting these rows would shift values into the wrong columns. Check "
            f"source.delimiter (currently {self.delimiter!r}) and the file's quoting (a value that contains "
            "the delimiter must be enclosed in double quotes)."
        )


def stage_csv(
    cfg: ProjectConfig, conn: psycopg.Connection, schemas: Schemas, strict: bool = STRICT_ROWS
) -> dict[str, Any]:
    """Stage every CSV/TSV file of ``source.path``. Raises :class:`SourceDataError` for an
    undecodable file, and (when ``strict``) for a file with ragged rows; otherwise ragged rows are
    left out and counted in the manifest."""
    assert cfg.source.path is not None
    _check_staging_schema(schemas)
    try:
        codecs.lookup(cfg.source.encoding)
    except LookupError:
        raise SourceDataError(
            f"source.encoding '{cfg.source.encoding}' is not a known text encoding (e.g. utf-8, latin-1, cp1252)."
        ) from None
    files = sorted(p for p in Path(cfg.source.path).iterdir() if p.suffix.lower() in {".csv", ".tsv", ".txt"})
    wanted = {t.lower() for t in cfg.source.tables} if cfg.source.tables else None
    manifest: dict[str, Any] = {}
    used_tables: set[str] = set()
    for f in files:
        table = sanitize_identifier(f.stem, used_tables)
        if wanted and table not in wanted and f.stem.lower() not in wanted:
            continue
        delim = "\t" if f.suffix.lower() == ".tsv" else cfg.source.delimiter
        src = _CsvFile(f, delim, cfg.source.encoding)
        header = src.header
        used_cols: set[str] = set()
        cols = [sanitize_identifier(h, used_cols) for h in header]
        try:
            n = _create_and_copy(conn, schemas.src, table, cols, src.rows())
        finally:
            src.close()
        if strict and src.ragged:
            raise src.ragged_error()
        manifest[table] = {
            "source": f.name,
            "rows": n,
            "ragged_rows": src.ragged,
            "ragged_too_many_fields": src.too_many,
            "ragged_too_few_fields": src.too_few,
            "ragged_example_lines": src.examples,
            "blank_lines": src.blank_lines,
            "columns": [{"name": c, "source_name": h, "declared_type": None} for c, h in zip(cols, header)],
        }
    return manifest


def stage_database(cfg: ProjectConfig, conn: psycopg.Connection, schemas: Schemas) -> dict[str, Any]:
    import sqlalchemy as sa

    _check_staging_schema(schemas)

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


def stage_source(
    cfg: ProjectConfig,
    conn: psycopg.Connection,
    schemas: Schemas,
    ws: Workspace | None = None,
    strict: bool = STRICT_ROWS,
) -> dict[str, Any]:
    """Copy all source tables into ``schemas.src``. Returns the manifest (also saved to the workspace).

    ``strict`` applies to CSV sources (see :func:`stage_csv`). On any error the transaction is
    rolled back, so an earlier staging is left as it was.
    """
    _check_staging_schema(schemas)
    try:
        ensure_schema(conn, schemas.src)
        if cfg.source.type == "csv":
            manifest = stage_csv(cfg, conn, schemas, strict=strict)
        else:
            manifest = stage_database(cfg, conn, schemas)
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    if ws is not None:
        (ws.root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        ragged = {t: m["ragged_rows"] for t, m in manifest.items() if m.get("ragged_rows")}
        ws.audit(
            "source_staged",
            tables={t: m["rows"] for t, m in manifest.items()},
            schema=schemas.src,
            **({"ragged_rows_skipped": ragged} if ragged else {}),
        )
    return manifest


def load_manifest(ws: Workspace) -> dict[str, Any] | None:
    p = ws.root / MANIFEST
    return json.loads(p.read_text()) if p.exists() else None
