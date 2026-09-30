"""Profile the staged source tables (the White Rabbit scan), computed in SQL.

Every statistic is an aggregate query over the ``text`` tables in the staging schema, so the
scan scales to millions of rows: a few full-table passes per table (counts, ranges, value
frequencies) plus a bounded sample of non-null values per column for type inference and PII
detection. No patient row is ever pulled into Python.

Privacy is enforced here, not only when prompting: top values below ``min_cell_count`` are
filtered in SQL (``HAVING``), lexical min/max of text columns are kept only when those values
are themselves frequent enough, and columns detected as direct identifiers get no values,
ranges or means at all. High-cardinality key columns (patient, visit, household ids) list no
values either: their frequencies would be per-person record counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import psycopg

from ..config import ProjectConfig
from ..db import Schemas, fetch_all, fetch_one, qi
from ..models import ColumnProfile, SourceProfile, TableProfile, ValueFreq
from ..privacy import detect_pii, suppress_small_cells
from ..workspace import utcnow

TYPE_MATCH_RATIO = 0.95  # share of sampled non-null values a pattern must match
SAMPLE_VALUES = 20_000  # non-null values inspected per column for type inference
TABLESAMPLE_ABOVE = 200_000  # tables larger than this are block-sampled for type inference
PII_SAMPLE_VALUES = 50  # distinct values handed to privacy.detect_pii
KEY_MAX_LISTED_DISTINCT = 50  # identifier-named columns with more distinct values list none
_MAX_TARGETS = 600  # select-list entries per aggregate query (Postgres caps it at 1664)
_KEY_NAME = re.compile(
    r"(^|_)(id|uuid|guid)(_|$)|(patient|person|individual|client|subject|member|visit|encounter)_?(id|no|num|number)$",
    re.I,
)

_TRIM_CHARS = "E' \\t\\r\\n'"
_BOOL_VALUES = ("true", "false", "yes", "no", "y", "n", "t", "f")
_INT_RE = r"^[+-]?\d+$"
_NUM_RE = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d{1,3})?$"

# Building blocks for strict date regexes (month 1-12, day 1-31, 4-digit year).
_Y = r"[1-9]\d{3}"
_M = r"(0?[1-9]|1[0-2])"
_D = r"(0?[1-9]|[12]\d|3[01])"
_MON = "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_YMD_PADDED = r"[1-9]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
_HM = r"([01]?\d|2[0-3]):[0-5]\d"
_TZ = r"(Z|[+-]\d{2}(:?\d{2})?)?"
_SEC = r":[0-5]\d(\.\d+)?"


def _lit(pattern: str) -> str:
    """Dollar-quoted SQL literal: immune to quoting and backslash settings."""
    return f"$re${pattern}$re$"


def _trim(expr: str) -> str:
    return f"btrim({expr}, {_TRIM_CHARS})"


def _ymd_key(y: str, m: str, d: str) -> str:
    return f"({y} || '-' || lpad({m}, 2, '0') || '-' || lpad({d}, 2, '0'))"


def _split(sep: str, i: int) -> str:
    return f"split_part(@V, '{sep}', {i})"


@dataclass(frozen=True)
class _DateFormat:
    """A date layout: strftime name, strict regex, and a SQL sort key over ``@V``."""

    fmt: str
    regex: str
    key: str

    def key_sql(self, value_sql: str) -> str:
        return self.key.replace("@V", value_sql)


def _dmy(sep: str, fmt: str, day_first: bool) -> _DateFormat:
    esc = "\\" + sep if sep == "." else sep
    a, b = (_D, _M) if day_first else (_M, _D)
    d_idx, m_idx = (1, 2) if day_first else (2, 1)
    return _DateFormat(fmt, f"^{a}{esc}{b}{esc}{_Y}$", _ymd_key(_split(sep, 3), _split(sep, m_idx), _split(sep, d_idx)))


# Order matters on ties: day-first layouts precede month-first ones, so dates whose day and
# month are both <= 12 everywhere are read as dd/mm (the common convention outside the US).
DATE_FORMATS: list[_DateFormat] = [
    _DateFormat("%Y-%m-%d", f"^{_Y}-{_M}-{_D}$", _ymd_key(_split("-", 1), _split("-", 2), _split("-", 3))),
    _DateFormat("%Y/%m/%d", f"^{_Y}/{_M}/{_D}$", _ymd_key(_split("/", 1), _split("/", 2), _split("/", 3))),
    _dmy("/", "%d/%m/%Y", True),
    _dmy("/", "%m/%d/%Y", False),
    _DateFormat(
        "%Y%m%d",
        r"^(18|19|20|21)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$",
        _ymd_key("substr(@V, 1, 4)", "substr(@V, 5, 2)", "substr(@V, 7, 2)"),
    ),
    _dmy("-", "%d-%m-%Y", True),
    _dmy("-", "%m-%d-%Y", False),
    _DateFormat(
        "%d-%b-%Y",
        f"^{_D}-{_MON}-{_Y}$",
        _ymd_key(
            _split("-", 3),
            "array_position(ARRAY['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],"
            f" lower({_split('-', 2)}))::text",
            _split("-", 1),
        ),
    ),
    _dmy(".", "%d.%m.%Y", True),
]
_DATE_BY_FMT = {f.fmt: f for f in DATE_FORMATS}

# ISO-like timestamps; the date part must be zero-padded. One sort key serves all variants.
DATETIME_FORMATS: list[tuple[str, str]] = [
    ("%Y-%m-%d %H:%M:%S", f"^{_YMD_PADDED} {_HM}{_SEC}{_TZ}$"),
    ("%Y-%m-%dT%H:%M:%S", f"^{_YMD_PADDED}T{_HM}{_SEC}{_TZ}$"),
    ("%Y-%m-%d %H:%M", f"^{_YMD_PADDED} {_HM}{_TZ}$"),
    ("%Y-%m-%dT%H:%M", f"^{_YMD_PADDED}T{_HM}{_TZ}$"),
]
_DATETIME_ANY = f"^{_YMD_PADDED}[ T]{_HM}({_SEC})?{_TZ}$"


_TZ_SUFFIX = r"(Z|[+-]\d{2}(:?\d{2})?)$"


def _datetime_key(v: str) -> str:
    t = f"regexp_replace(substr({v}, 12), {_lit(_TZ_SUFFIX)}, '')"
    return (
        f"(substr({v}, 1, 10) || ' ' || lpad(split_part({t}, ':', 1), 2, '0') || ':' || split_part({t}, ':', 2)"
        f" || ':' || coalesce(nullif(left(split_part({t}, ':', 3), 2), ''), '00'))"
    )


_PG_TOKENS = {
    "%Y": "YYYY", "%y": "YY", "%m": "MM", "%d": "DD", "%b": "Mon", "%B": "Month",
    "%H": "HH24", "%I": "HH12", "%p": "AM", "%M": "MI", "%S": "SS", "%f": "US", "%j": "DDD", "%%": "%",
}


def strftime_to_pg(fmt: str) -> str:
    """Translate a strftime pattern (as in ``ColumnProfile.date_format``) to a Postgres
    ``to_date``/``to_timestamp`` template, e.g. ``%d/%m/%Y`` -> ``DD/MM/YYYY``."""
    out: list[str] = []
    i = 0
    while i < len(fmt):
        if fmt[i] == "%" and i + 1 < len(fmt):
            tok = fmt[i : i + 2]
            if tok not in _PG_TOKENS:
                raise ValueError(f"unsupported strftime directive {tok!r} in {fmt!r}")
            out.append(_PG_TOKENS[tok])
            i += 2
        else:
            ch = fmt[i]
            out.append(f'"{ch}"' if ch.isalpha() else ch)
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------------------


@dataclass
class _Col:
    name: str
    expr: str  # SQL expression yielding the raw text value
    declared_type: str | None
    idx: int
    sample_n: int = 0
    inferred_type: str = "empty"
    date_format: str | None = None
    pii_reason: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def profile_staging(
    conn: psycopg.Connection,
    schemas: Schemas,
    cfg: ProjectConfig,
    manifest: dict | None = None,
    top_n: int = 50,
) -> SourceProfile:
    """Scan every table in ``schemas.src`` and return the privacy-safe source profile."""
    tables = _tables(conn, schemas.src, manifest)
    profiles = [
        _profile_table(conn, schemas.src, name, is_base, cfg, (manifest or {}).get(name), top_n)
        for name, is_base in tables
    ]
    return SourceProfile(
        source_type=cfg.source.type,
        generated_at=utcnow(),
        min_cell_count=cfg.privacy.min_cell_count,
        tables=profiles,
    )


def _tables(conn: psycopg.Connection, schema: str, manifest: dict | None) -> list[tuple[str, bool]]:
    rows = fetch_all(
        conn,
        "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = %s",
        (schema,),
    )
    kinds = {r["table_name"]: r["table_type"] == "BASE TABLE" for r in rows}
    order = [t for t in (manifest or {}) if t in kinds]
    order += sorted(t for t in kinds if t not in order)
    return [(t, kinds[t]) for t in order]


def _profile_table(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    is_base: bool,
    cfg: ProjectConfig,
    manifest_entry: dict | None,
    top_n: int,
) -> TableProfile:
    tbl = f"{qi(schema)}.{qi(table)}"
    declared = {c["name"]: c.get("declared_type") for c in (manifest_entry or {}).get("columns", [])}
    col_rows = fetch_all(
        conn,
        """SELECT column_name, data_type FROM information_schema.columns
           WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position""",
        (schema, table),
    )
    cols = [
        _Col(
            name=r["column_name"],
            expr=qi(r["column_name"]) if r["data_type"] in ("text", "character varying") else f"{qi(r['column_name'])}::text",
            declared_type=declared.get(r["column_name"]),
            idx=i,
        )
        for i, r in enumerate(col_rows)
    ]
    n_rows = int(fetch_one(conn, f"SELECT count(*) AS n FROM {tbl}")["n"])  # type: ignore[index]
    min_cell = cfg.privacy.min_cell_count

    for c in cols:
        _inspect_sample(conn, tbl, table, c, n_rows, is_base, cfg)
    _aggregate_pass(conn, tbl, cols)
    _verify_text_extremes(conn, tbl, cols, min_cell)

    profiles = []
    for c in cols:
        s = c.stats
        nn, nd = int(s["nn"]), int(s["nd"])
        top: list[ValueFreq] = []
        n_hidden = 0
        if c.pii_reason is None and nn:
            if _is_key_column(c.name, nd):
                # Frequencies of a patient/visit key are per-person record counts: withhold them.
                s["lo"] = s["hi"] = None
            elif not (nd == nn and min_cell > 1) and top_n > 0:
                top = _top_values(conn, tbl, c, min_cell, top_n)
            n_hidden = nd - len(top)
        profiles.append(
            ColumnProfile(
                name=c.name,
                declared_type=c.declared_type,
                inferred_type=c.inferred_type,  # type: ignore[arg-type]
                n_rows=n_rows,
                n_null=n_rows - nn,
                n_distinct=nd,
                unique=nn > 0 and nd == nn,
                min_value=None if c.pii_reason else s.get("lo"),
                max_value=None if c.pii_reason else s.get("hi"),
                mean=None if c.pii_reason else s.get("mean"),
                min_length=s.get("lmin"),
                max_length=s.get("lmax"),
                date_format=c.date_format,
                top_values=top,
                n_suppressed_values=n_hidden,
                is_pii=c.pii_reason is not None,
                pii_reason=c.pii_reason,
            )
        )
    return TableProfile(name=table, n_rows=n_rows, columns=profiles)


def _is_key_column(name: str, n_distinct: int) -> bool:
    """A high-cardinality identifier (person, visit, household key...): its values are not listed."""
    return n_distinct > KEY_MAX_LISTED_DISTINCT and bool(_KEY_NAME.search(name))


def _sample_sql(tbl: str, c: _Col, n_rows: int, block_sample: bool) -> str:
    ts = ""
    if block_sample:
        pct = min(100.0, 100.0 * 5 * SAMPLE_VALUES / max(n_rows, 1))
        ts = f" TABLESAMPLE SYSTEM ({pct:.6f}) REPEATABLE (42)"
    return f"SELECT {_trim(c.expr)} AS v FROM {tbl}{ts} WHERE {c.expr} IS NOT NULL LIMIT {SAMPLE_VALUES}"


def _inspect_sample(
    conn: psycopg.Connection, tbl: str, table: str, c: _Col, n_rows: int, is_base: bool, cfg: ProjectConfig
) -> None:
    """Infer the type and check for PII on a bounded sample of non-null values."""
    block = is_base and n_rows > TABLESAMPLE_ABOVE
    counts = _pattern_counts(conn, _sample_sql(tbl, c, n_rows, block))
    if block and counts["n"] == 0:  # sparse column: the block sample missed every value
        block = False
        counts = _pattern_counts(conn, _sample_sql(tbl, c, n_rows, False))
    c.sample_n = int(counts["n"])
    c.inferred_type, c.date_format = _infer_type(counts)
    values: list[str] = []
    # Dates are never phone numbers, but ISO dates ("2019-01-05") match the phone pattern in
    # privacy.detect_pii, so date columns are judged on their name (and config) only.
    if c.inferred_type not in ("date", "datetime"):
        sample = _sample_sql(tbl, c, n_rows, block)
        values = [r["v"] for r in fetch_all(conn, f"SELECT DISTINCT v FROM ({sample}) s LIMIT {PII_SAMPLE_VALUES}")]
    c.pii_reason = detect_pii(table, c.name, values, cfg.privacy)


def _pattern_counts(conn: psycopg.Connection, sample_sql: str) -> dict[str, Any]:
    bools = ", ".join(f"'{b}'" for b in _BOOL_VALUES)
    parts = [
        "count(*) AS n",
        f"count(*) FILTER (WHERE lower(v) IN ({bools})) AS n_bool",
        f"count(*) FILTER (WHERE v ~ {_lit(_INT_RE)}) AS n_int",
        f"count(*) FILTER (WHERE v ~ {_lit(_NUM_RE)}) AS n_num",
    ]
    parts += [f"count(*) FILTER (WHERE v ~* {_lit(f.regex)}) AS d_{i}" for i, f in enumerate(DATE_FORMATS)]
    parts += [f"count(*) FILTER (WHERE v ~ {_lit(rx)}) AS t_{i}" for i, (_, rx) in enumerate(DATETIME_FORMATS)]
    return fetch_one(conn, f"SELECT {', '.join(parts)} FROM ({sample_sql}) s") or {"n": 0}


def _infer_type(counts: dict[str, Any]) -> tuple[str, str | None]:
    """Pick the first type whose pattern covers >= TYPE_MATCH_RATIO of the sampled values."""
    n = int(counts["n"])
    if n == 0:
        return "empty", None
    need = TYPE_MATCH_RATIO * n
    if counts["n_bool"] >= need:
        return "boolean", None
    dt = [int(counts[f"t_{i}"]) for i in range(len(DATETIME_FORMATS))]
    if sum(dt) >= need:
        best = max(range(len(dt)), key=lambda i: (dt[i], -i))
        return "datetime", DATETIME_FORMATS[best][0]
    d = [int(counts[f"d_{i}"]) for i in range(len(DATE_FORMATS))]
    best = max(range(len(d)), key=lambda i: (d[i], -i))
    if d[best] >= need:
        return "date", DATE_FORMATS[best].fmt
    if counts["n_int"] >= need:
        return "integer", None
    if counts["n_num"] >= need:
        return "float", None
    return "string", None


def _chunks(items: list[_Col], per_item: int) -> list[list[_Col]]:
    size = max(1, _MAX_TARGETS // per_item)
    return [items[i : i + size] for i in range(0, len(items), size)] or [[]]


def _aggregate_pass(conn: psycopg.Connection, tbl: str, cols: list[_Col]) -> None:
    """Counts, lengths and type-aware ranges for every column, one scan per column chunk."""
    for chunk in _chunks(cols, 7):
        if not chunk:
            continue
        parts: list[str] = []
        for c in chunk:
            e, i = c.expr, c.idx
            parts += [
                f"count({e}) AS nn_{i}",
                f"count(DISTINCT {e}) AS nd_{i}",
                f"min(length({e})) AS lmin_{i}",
                f"max(length({e})) AS lmax_{i}",
            ]
            if c.pii_reason is not None:
                continue
            v = _trim(e)
            if c.inferred_type in ("integer", "float"):
                x = f"(CASE WHEN {v} ~ {_lit(_NUM_RE)} THEN {v}::numeric END)"
                parts += [f"min({x})::text AS lo_{i}", f"max({x})::text AS hi_{i}", f"avg({x})::float8 AS mean_{i}"]
            elif c.inferred_type in ("date", "datetime"):
                if c.inferred_type == "date":
                    f = _DATE_BY_FMT[c.date_format]  # type: ignore[index]
                    x = f"(CASE WHEN {v} ~* {_lit(f.regex)} THEN {f.key_sql(v)} END)"
                else:
                    x = f"(CASE WHEN {v} ~ {_lit(_DATETIME_ANY)} THEN {_datetime_key(v)} END)"
                parts += [f'min({x} COLLATE "C") AS lo_{i}', f'max({x} COLLATE "C") AS hi_{i}']
            elif c.inferred_type in ("string", "boolean"):
                parts += [f'min({e} COLLATE "C") AS lo_{i}', f'max({e} COLLATE "C") AS hi_{i}']
        row = fetch_one(conn, f"SELECT {', '.join(parts)} FROM {tbl}") or {}
        for c in chunk:
            i = c.idx
            c.stats = {
                "nn": row.get(f"nn_{i}") or 0,
                "nd": row.get(f"nd_{i}") or 0,
                "lmin": row.get(f"lmin_{i}"),
                "lmax": row.get(f"lmax_{i}"),
                "lo": row.get(f"lo_{i}"),
                "hi": row.get(f"hi_{i}"),
            }
            mean = row.get(f"mean_{i}")
            if mean is not None:
                c.stats["mean"] = float(f"{mean:.6g}")
            if c.inferred_type in ("date", "datetime"):
                c.stats["lo"] = _iso(c.stats["lo"], c.inferred_type)
                c.stats["hi"] = _iso(c.stats["hi"], c.inferred_type)


def _iso(key: str | None, kind: str) -> str | None:
    if key is None:
        return None
    try:
        return date.fromisoformat(key).isoformat() if kind == "date" else datetime.fromisoformat(key).isoformat(sep=" ")
    except ValueError:  # e.g. 30 February: report the normalised text as found
        return key


def _verify_text_extremes(conn: psycopg.Connection, tbl: str, cols: list[_Col], min_cell: int) -> None:
    """Keep lexical min/max of text columns only if each value occurs >= min_cell times."""
    text_cols = [c for c in cols if c.inferred_type in ("string", "boolean") and c.stats.get("lo") is not None]
    for chunk in _chunks(text_cols, 2):
        if not chunk:
            continue
        parts, params = [], []
        for c in chunk:
            parts += [f"count(*) FILTER (WHERE {c.expr} = %s) AS clo_{c.idx}", f"count(*) FILTER (WHERE {c.expr} = %s) AS chi_{c.idx}"]
            params += [c.stats["lo"], c.stats["hi"]]
        row = fetch_one(conn, f"SELECT {', '.join(parts)} FROM {tbl}", params) or {}
        for c in chunk:
            if (row.get(f"clo_{c.idx}") or 0) < min_cell or (row.get(f"chi_{c.idx}") or 0) < min_cell:
                c.stats["lo"] = c.stats["hi"] = None


def _top_values(conn: psycopg.Connection, tbl: str, c: _Col, min_cell: int, top_n: int) -> list[ValueFreq]:
    rows = fetch_all(
        conn,
        f"""SELECT {c.expr} AS value, count(*) AS n FROM {tbl} WHERE {c.expr} IS NOT NULL
            GROUP BY {c.expr} HAVING count(*) >= {int(min_cell)}
            ORDER BY count(*) DESC, {c.expr} COLLATE "C" LIMIT {int(top_n)}""",
    )
    kept, _ = suppress_small_cells([ValueFreq(value=r["value"], count=int(r["n"])) for r in rows], min_cell)
    return kept
