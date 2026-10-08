"""Profile the staged source tables (the White Rabbit scan), computed in SQL.

Every statistic is an aggregate query over the ``text`` tables in the staging schema, so the
scan scales to millions of rows: a few full-table passes per table (counts, ranges, value
frequencies, value shapes) plus a bounded sample of non-null values per column for type
inference and PII detection. The sample is spread over the whole table (row or block sampling),
not taken from its first rows; a column the sample holds few values of is read from its first
values instead. No patient row is ever pulled into Python.

Type inference picks the first type whose pattern covers 95% of the sample. Date layouts include
day-first and month-first dates, Stata and SAS dates ('05mar2019', '05MAR2019'), two-digit years
with the year last ('05/03/19'; there is no year-first layout, which the values cannot always tell
apart from a day-first one: a note says so), dd/mm/yyyy HH:MM timestamps (also with '-' or '.'),
times of day, and day numbers (read as Excel serial days) when a word of the column name says it
holds dates. Partial dates (an unknown day or month written 00 or 97-99, a missing day, a year
alone) and ISO dates count towards every date layout, as the build reads them with any template.
The full scan then classifies every value as the build reads it with the profiled format:
complete dates (which give min/max; impossible dates such as 31/02 do not), partial dates, and
values that do not conform; it also collects the time zone offsets written after the times.

Privacy is enforced here, not only when prompting: top values, value shapes and offsets below
``min_cell_count`` are filtered in SQL (``HAVING``), lexical min/max of text columns are kept
only when those values are themselves frequent enough, and columns detected as direct
identifiers get no values, ranges, means or shapes at all. Patient keys and other
high-cardinality id columns (visit, household) list no values either: their frequencies would be
per-person record counts (their shapes are listed when at least ``min_cell_count`` different
keys share them).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import psycopg

from ..config import ProjectConfig
from ..db import Schemas, fetch_all, fetch_one, qi
from ..models import ColumnProfile, SourceProfile, TableProfile, ValueFreq
from ..privacy import PII_MIN_VALUES, detect_pii, suppress_small_cells
from ..workspace import utcnow

TYPE_MATCH_RATIO = 0.95  # share of sampled non-null values a pattern must match
SAMPLE_VALUES = 20_000  # non-null values inspected per column for type inference
SAMPLE_OVERDRAW = 1.25  # rows sampled per value wanted, so that the sample is not cut short
TABLESAMPLE_ABOVE = 200_000  # tables larger than this are block-sampled; smaller ones row-sampled
SPARSE_SAMPLE = 100  # a sample with fewer non-null values than this: the column's first values are read
PII_SAMPLE_VALUES = 50  # distinct values handed to privacy.detect_pii
KEY_MAX_LISTED_DISTINCT = 50  # other id-named columns with more distinct values list none
MAX_REGEXES = 28  # distinct regular expressions per query (PostgreSQL caches 32 compiled ones)
MAX_SHAPES = 10  # value shapes (and shapes of non-conforming values) listed per column
MAX_SHAPE_LENGTH = 32  # longer values get no shape
PLACEHOLDER_MIN_SHARE = 0.01  # one future date this common in a date column is reported
_MAX_TARGETS = 600  # select-list entries per aggregate query (Postgres caps it at 1664)
# Keys of people (patients, HDSS individuals and their parents or household heads): never listed.
_PERSON_KEY_NAME = re.compile(
    r"(patient|person|individual|indiv|client|subject|participant|member|resident|mother|father|guardian|"
    r"respondent|deceased|child|spouse|head|perm|ext)_?(id|no|num|number|uuid)$|^pid$",
    re.I,
)
# Other keys (visits, households, locations, ODK submissions): not listed when they have many values.
_KEY_NAME = re.compile(
    r"(^|_)(id|uuid|guid|key|pid)(_|$)|[a-z](id|uuid|guid)$|(^|_)(parent_)?key$"
    r"|(visit|encounter|record|episode|household|hh|location|socialgroup|instance|submission|form)_?"
    r"(id|key|no|num|number)$",
    re.I,
)
# Column names that say the column holds dates (a whole word of the name: visit_date, dateofbirth,
# dob; not accommodated): only these are read as day numbers.
_DATE_NAME = re.compile(r"(^|_)date|date\d*(_|$)|(^|_)(dte|dob|dod|doa)(_|$)|(^|_)dt$", re.I)
_NOT_DATE_NAME = re.compile(r"(up|candi|vali|man)date", re.I)

_TRIM_CHARS = "E' \\t\\r\\n'"
_BOOL_VALUES = ("true", "false", "yes", "no", "y", "n", "t", "f")
_INT_RE = r"^[+-]?\d+$"
_INT_READ = r"^[+-]?\d+(\.0*)?$"  # what the build reads as an integer ('5.0' and '5.' too)
_NUM_RE = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d{1,3})?$"

# Building blocks for strict date regexes (month 1-12, day 1-31, 4-digit year).
_Y = r"[1-9]\d{3}"
_YY = r"\d{2}"
_M = r"(0?[1-9]|1[0-2])"
_D = r"(0?[1-9]|[12]\d|3[01])"
_UNK = r"(00|9[7-9])"  # an unknown day or month
_MX = rf"({_M}|{_UNK})"
_MON = "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
# month names as the build reads them: the full English names too
_MONTH_NAME = ("(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t|tember)?|oct(ober)?|"
               "nov(ember)?|dec(ember)?)")
_MONTHS = "ARRAY['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']"
_YMD_PADDED = r"[1-9]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
_HM = r"([01]?\d|2[0-3]):[0-5]\d"
_TZ = r"(Z|[+-]\d{2}(:?\d{2})?)?"
_SEC = r":[0-5]\d(\.\d+)?"
# an offset the build reads ('Z', '+03', '+0300', '+03:00'; at most 14 hours)
_OFFSET = r"(Z|[+-](0\d|1[0-4])(:?[0-5]\d)?)"
# Excel serial day numbers of 1910-01-01 .. 2099-12-31 (days since 1899-12-30)
EXCEL_MIN, EXCEL_MAX = 3654, 73050

# Format notes (ColumnProfile.format_notes); they never quote a value.
EXCEL_NOTE = ("day numbers, read here as Excel serial days (days since 1899-12-30; the range shown is that "
              "reading). Stata and SAS count days from 1960-01-01, Unix from 1970-01-01: check which with the "
              "source, then convert with DATE '1899-12-30' + x::integer (or DATE '1960-01-01' + x::integer); a "
              "::date cast or to_date reads none of them")
TWO_DIGIT_YEAR_NOTE = ("two-digit years, taken to be the last field: the build reads them as the latest such year "
                       "not after the current one (30 is 1930, 19 is 2019); check that this suits the column (old "
                       "birth dates)")
YEAR_FIRST_NOTE = ("no year in the sample is 00 or above 31, so the values read just as well year-first (yy/mm/dd, "
                   "a different date): the year is taken to be last; check the order with the source")
DAY_FIRST_NOTE = ("no day in the sample is above 12, so day-first and month-first read the same values: day-first "
                  "is assumed; check it with the source")


def _lit(pattern: str) -> str:
    """Dollar-quoted SQL literal: immune to quoting and backslash settings."""
    return f"$re${pattern}$re$"


def _trim(expr: str) -> str:
    return f"btrim({expr}, {_TRIM_CHARS})"


def _ymd_key(y: str, m: str, d: str) -> str:
    return f"({y} || '-' || lpad({m}, 2, '0') || '-' || lpad({d}, 2, '0'))"


def _split(sep: str, i: int) -> str:
    return f"split_part(@V, '{sep}', {i})"


def _year(yy: str) -> str:
    """A two-digit year as the build reads it: the latest such year not after the current one."""
    return (f"((CASE WHEN ({yy})::int <= extract(year FROM current_date)::int % 100 THEN 2000 ELSE 1900 END)"
            f" + ({yy})::int)::text")


def _month(name: str) -> str:
    return f"array_position({_MONTHS}, lower({name}))::text"


def _time_key(t: str) -> str:
    """'HH:MI:SS' of a time 'H:MI[:SS[.f]]'."""
    return (f"(lpad(split_part({t}, ':', 1), 2, '0') || ':' || split_part({t}, ':', 2) || ':' || "
            f"coalesce(nullif(left(split_part({t}, ':', 3), 2), ''), '00'))")


@dataclass(frozen=True)
class _DateFormat:
    """A date layout: strftime name, the regexes of a complete date, of its partial forms (an
    unknown day or month) and of the date without its day ('06/1975'), as the format writes them
    (type inference), a SQL sort key over ``@V`` ('YYYY-MM-DD'), and how the full scan reads it:
    ``scan*`` take any one of ``-/._`` for a separator (and full month names), as the build does,
    and ``norm`` (SQL over ``@V``) writes such a value with the format's own separator before the
    key reads it."""

    fmt: str
    body: str
    key: str
    partial: str | None = None
    no_day: str | None = None
    scan: str = ""
    scan_partial: str | None = None
    scan_no_day: str | None = None
    norm: str = "@V"

    @property
    def regex(self) -> str:
        return f"^{self.body}$"

    @property
    def loose(self) -> str:
        """Complete or partial."""
        forms = [p for p in (self.body, self.partial, self.no_day) if p]
        return f"^({'|'.join(forms)})$" if len(forms) > 1 else self.regex

    def key_sql(self, value_sql: str) -> str:
        return self.key.replace("@V", value_sql)

    def scan_key(self, value_sql: str) -> str:
        """The key of a value the full scan matched with ``scan``."""
        return self.key_sql(self.norm.replace("@V", value_sql))


_SEP = "[-/._]"  # the build reads any one of these for a template separator
_S = "§"  # where a layout's separator goes in the patterns below


def _layout(
    fmt: str, sep: str, body: str, key: str, partial: str | None, no_day: str | None = None, infer_no_day: bool = True
) -> _DateFormat:
    """A layout whose patterns write ``_S`` for its separator ``sep``. ``infer_no_day`` False: a
    date without its day is read by the build but not taken as a sign of a date."""
    esc = "\\" + sep if sep == "." else sep

    def own(p: str | None) -> str | None:
        return p.replace(_S, esc) if p else None

    def scan(p: str | None) -> str | None:
        return p.replace(_S, _SEP).replace(_MON, _MONTH_NAME) if p else None

    return _DateFormat(fmt, own(body), key, own(partial), own(no_day) if infer_no_day else None,  # type: ignore[arg-type]
                       scan(body), scan(partial), scan(no_day), f"translate(@V, '-/._', '{sep * 4}')")  # type: ignore[arg-type]


def _dmy(sep: str, fmt: str, day_first: bool, two_digit: bool = False) -> _DateFormat:
    y = _YY if two_digit else _Y
    a, b = (_D, _M) if day_first else (_M, _D)
    d_idx, m_idx = (1, 2) if day_first else (2, 1)
    year = _year(_split(sep, 3)) if two_digit else _split(sep, 3)
    # an unknown day (the month known or not), or an unknown month (the day then does not matter)
    if day_first:
        partial = rf"{_UNK}{_S}{_MX}{_S}{y}|\d{{1,2}}{_S}{_UNK}{_S}{y}"
    else:
        partial = rf"{_MX}{_S}{_UNK}{_S}{y}|{_UNK}{_S}\d{{1,2}}{_S}{y}"
    key = _ymd_key(year, _split(sep, m_idx), _split(sep, d_idx))
    # '06/75' is read without its day too, but it is not taken as a sign of a date
    return _layout(fmt, sep, f"{a}{_S}{b}{_S}{y}", key, partial, f"{_MX}{_S}{y}", infer_no_day=not two_digit)


def _mon(fmt: str, two_digit: bool, sep: str) -> _DateFormat:
    """Day, month name, year: '05-Mar-2019' (sep '-') or the Stata / SAS '05mar2019' (no sep)."""
    y = _YY if two_digit else _Y
    if not sep:  # digits, a month name, digits
        name = "translate(lower(@V), '0123456789', '')"
        year = _year("right(@V, 2)") if two_digit else "right(@V, 4)"
        key = _ymd_key(year, _month(f"left({name}, 3)"), f"left(@V, strpos(lower(@V), {name}) - 1)")
        body, partial = f"{_D}{_MON}{y}", f"{_UNK}{_MON}{y}"
        return _DateFormat(fmt, body, key, partial, None, f"{_D}{_MONTH_NAME}{y}", f"{_UNK}{_MONTH_NAME}{y}")
    year = _year(_split(sep, 3)) if two_digit else _split(sep, 3)
    key = _ymd_key(year, _month(f"left({_split(sep, 2)}, 3)"), _split(sep, 1))
    return _layout(fmt, sep, f"{_D}{_S}{_MON}{_S}{y}", key, f"{_UNK}{_S}{_MON}{_S}{y}", f"{_MON}{_S}{y}",
                   infer_no_day=not two_digit)


# ISO: the build reads a year-first date with any separator under the format's template
# ('YYYY-MM-DD'), and an ISO date ('-' only, also partial) under any other template.
ISO_DATE = _layout(
    "%Y-%m-%d", "-", f"{_Y}{_S}{_M}{_S}{_D}", _ymd_key(_split("-", 1), _split("-", 2), _split("-", 3)),
    rf"{_Y}{_S}{_UNK}{_S}\d{{1,2}}|{_Y}{_S}{_M}{_S}{_UNK}", f"{_Y}{_S}{_MX}",
)
# Order matters on ties: day-first layouts precede month-first ones, so dates whose day and
# month are both <= 12 everywhere are read as dd/mm (the common convention outside the US).
# Two-digit years come last, and only with the year last.
DATE_FORMATS: list[_DateFormat] = [
    ISO_DATE,
    _layout("%Y/%m/%d", "/", f"{_Y}{_S}{_M}{_S}{_D}", _ymd_key(_split("/", 1), _split("/", 2), _split("/", 3)),
            rf"{_Y}{_S}{_UNK}{_S}\d{{1,2}}|{_Y}{_S}{_M}{_S}{_UNK}", f"{_Y}{_S}{_MX}"),
    _dmy("/", "%d/%m/%Y", True),
    _dmy("/", "%m/%d/%Y", False),
    _DateFormat(
        "%Y%m%d",
        r"(18|19|20|21)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])",
        _ymd_key("substr(@V, 1, 4)", "substr(@V, 5, 2)", "substr(@V, 7, 2)"),
        r"(18|19|20|21)\d{2}(00|9[7-9])\d{2}|(18|19|20|21)\d{2}(0[1-9]|1[0-2])(00|9[7-9])",
        None,
        r"(18|19|20|21)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])",
        r"(18|19|20|21)\d{2}(00|9[7-9])\d{2}|(18|19|20|21)\d{2}(0[1-9]|1[0-2])(00|9[7-9])",
    ),
    _dmy("-", "%d-%m-%Y", True),
    _dmy("-", "%m-%d-%Y", False),
    _mon("%d-%b-%Y", False, "-"),
    _dmy(".", "%d.%m.%Y", True),
    _mon("%d%b%Y", False, ""),  # Stata %td and SAS DATE9: 05mar2019, 05MAR2019
    _dmy("/", "%d/%m/%y", True, two_digit=True),
    _dmy("/", "%m/%d/%y", False, two_digit=True),
    _dmy("-", "%d-%m-%y", True, two_digit=True),
    _dmy(".", "%d.%m.%y", True, two_digit=True),
    _mon("%d-%b-%y", True, "-"),
    _mon("%d%b%y", True, ""),  # SAS DATE7: 05MAR19
]
_DATE_BY_FMT = {f.fmt: f for f in DATE_FORMATS}


@dataclass(frozen=True)
class _DatetimeFormat:
    """A timestamp layout: strftime name, the regex of this variant (it picks the reported
    format), the regex of its family (every variant with the same date part and separator), the
    date part, the separator before the time ('' for ISO, which takes ' ' or 'T') and whether the
    format has seconds."""

    fmt: str
    regex: str
    family: str
    date: _DateFormat
    sep: str = ""
    seconds: bool = True


_ISO_DATETIME = f"^{_YMD_PADDED}[ T]{_HM}({_SEC})?{_TZ}$"


def _split_datetime(d: _DateFormat, sep: str, fmt: str, seconds: bool) -> _DatetimeFormat:
    """d, a separator, then HH:MI (seconds False) or HH:MI:SS (True), and maybe an offset (the
    build reads one after the time with any template)."""
    sec, tz = (_SEC if seconds else ""), f"( *{_OFFSET})?"
    return _DatetimeFormat(fmt, f"^{d.body}{sep}{_HM}{sec}{tz}$", f"^{d.body}{sep}{_HM}({_SEC})?{tz}$", d, sep, seconds)


# ISO-like timestamps (the date part zero-padded, an offset or 'Z' allowed) first.
DATETIME_FORMATS: list[_DatetimeFormat] = [
    *(
        _DatetimeFormat(fmt, f"^{_YMD_PADDED}{sep}{_HM}{sec}{_TZ}$", _ISO_DATETIME, ISO_DATE, sep, bool(sec))
        for fmt, sep, sec in [
            ("%Y-%m-%d %H:%M:%S", " ", _SEC), ("%Y-%m-%dT%H:%M:%S", "T", _SEC),
            ("%Y-%m-%d %H:%M", " ", ""), ("%Y-%m-%dT%H:%M", "T", ""),
        ]
    ),
    _split_datetime(_DATE_BY_FMT["%d/%m/%Y"], " ", "%d/%m/%Y %H:%M", False),
    _split_datetime(_DATE_BY_FMT["%d/%m/%Y"], " ", "%d/%m/%Y %H:%M:%S", True),
    _split_datetime(_DATE_BY_FMT["%m/%d/%Y"], " ", "%m/%d/%Y %H:%M", False),
    _split_datetime(_DATE_BY_FMT["%m/%d/%Y"], " ", "%m/%d/%Y %H:%M:%S", True),
    _split_datetime(_DATE_BY_FMT["%d-%m-%Y"], " ", "%d-%m-%Y %H:%M", False),
    _split_datetime(_DATE_BY_FMT["%d-%m-%Y"], " ", "%d-%m-%Y %H:%M:%S", True),
    _split_datetime(_DATE_BY_FMT["%d.%m.%Y"], " ", "%d.%m.%Y %H:%M", False),
    _split_datetime(_DATE_BY_FMT["%d.%m.%Y"], " ", "%d.%m.%Y %H:%M:%S", True),
    _split_datetime(_DATE_BY_FMT["%d%b%Y"], ":", "%d%b%Y:%H:%M:%S", True),  # SAS DATETIME
    _split_datetime(_DATE_BY_FMT["%d%b%Y"], " ", "%d%b%Y %H:%M:%S", True),  # Stata %tc
    _split_datetime(_DATE_BY_FMT["%d%b%Y"], " ", "%d%b%Y %H:%M", False),
]
_FAMILIES = list(dict.fromkeys(f.family for f in DATETIME_FORMATS))
TIME_FORMATS = [("%H:%M", f"^{_HM}$"), ("%H:%M:%S", f"^{_HM}{_SEC}$")]
# A cheap test that a value may be a date, a timestamp or a time (the layouts are tried only then)
_DATE_LIKE = (rf"^([0-9]{{1,4}}[-/.][0-9]{{1,2}}[-/.]|[0-9]{{4}}[-/][0-9]{{1,2}}$|[0-9]{{1,2}}[-/.][0-9]{{4}}$|"
              rf"[0-9]{{4}}$|[0-9]{{8}}$|[0-9]{{1,2}}:[0-9]{{2}}|[0-9]{{0,2}}[-. ]?{_MON})")
# values that settle an order: a first field above 12 is a day, a last field of 00 or above 31 a year
_DAY_ABOVE_12 = r"^(1[3-9]|2\d|3[01])[-/._]"
_MID_ABOVE_12 = r"^\d{1,2}[-/._](1[3-9]|2\d|3[01])[-/._]"
_YEAR_LAST = r"^\d{1,2}[-/._]\d{1,2}[-/._](00|3[2-9]|[4-9]\d)($|[ T])"


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
# How the full scan reads a column of a date-like type
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reading:
    """How the full scan reads a column, as the build reads it with the profiled format: the
    branches that give a complete value (anchored regex, matched case-insensitively -> SQL key over
    ``@V``, the value, ``@D``, its date part, and ``@T``, the time after it, or NULL), tried in
    order; the regex of partial dates (only incomplete forms: a complete date that the branches
    do not read does not conform); whether the key holds a date to check (31/02 is not a date)
    and whether offsets are collected."""

    branches: tuple[tuple[str, str], ...]
    partial: str | None = None
    dated: bool = True
    offsets: bool = True


# What the build reads after a date under a date-only template, and after an ISO date under any
# template: ' ' or 'T', H:MI, seconds (with a fraction), then an offset.
_DATE_TAIL = rf"(( +|T){_HM}(:[0-5]\d([.,]\d+)?)?( *{_OFFSET})?)?"
# The date part of a value (its trailing time and offset removed: one regular expression without
# groups, which PostgreSQL matches far faster than one with them), the time after it and whether
# an offset follows the time. Values without ':' have no time and skip them.
_TIME_AFTER = r"(?: +|[Tt]|:)(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:[.,]\d+)?(?: *(?:[Zz]|[+-]\d{2}(?::?\d{2})?))?$"
_OFFSET_AFTER = r":\d{2}(?:[.,]\d+)? *(?:[Zz]|[+-]\d{2}(?::?\d{2})?)$"


def _date_part(v: str) -> str:
    return f"(CASE WHEN strpos({v}, ':') > 0 THEN regexp_replace({v}, {_lit(_TIME_AFTER)}, '') ELSE {v} END)"


def _time_of(v: str, dp: str) -> str:
    """The time after the date part ``dp`` of ``v`` ('H:MI[:SS]', without a fraction or an
    offset), or NULL for a date alone."""
    t = f"ltrim(substr({v}, length({dp}) + 1), ' Tt:')"
    return f"(CASE WHEN length({dp}) < length({v}) THEN left({t}, length({t}) - length(ltrim({t}, '0123456789:'))) END)"


def _has_offset(v: str) -> str:
    return f"(strpos({v}, ':') > 0 AND {v} ~ {_lit(_OFFSET_AFTER)})"


def _offset_of(v: str) -> str:
    """The offset written after the time of ``v`` ('Z', '+03:00', '+0300', '+03'), or NULL: the
    text after the last sign, or 'Z'."""
    sign = f"nullif(least(nullif(strpos(reverse({v}), '+'), 0), nullif(strpos(reverse({v}), '-'), 0)), 0)"
    return f"(CASE WHEN {_has_offset(v)} THEN CASE WHEN upper(right({v}, 1)) = 'Z' THEN 'Z' ELSE right({v}, {sign}) END END)"


# ISO dates under another layout's template: complete, partial (with the date tail) and the year
# with its month or alone (no tail)
_ISO_ELSEWHERE = (f"^{ISO_DATE.body}{_DATE_TAIL}$", f"({ISO_DATE.partial}){_DATE_TAIL}")
_YEAR_ALONE = rf"{_Y}(-{_MX})?"


def _partial_rx(forms: list[str]) -> str:
    return "^(" + "|".join(forms) + ")$"


def _date_reading(d: _DateFormat) -> _Reading:
    """Complete dates in the layout (a time and an offset may follow) or ISO; partial ones (the
    template's forms with an unknown part or without the day, ISO ones, a year)."""
    branches = [(f"^{d.scan}{_DATE_TAIL}$", d.scan_key("@D"))]
    own = [p for p in (d.scan_partial, d.scan_no_day) if p]
    forms = [f"({'|'.join(own)}){_DATE_TAIL}"] if own else []
    if d is not ISO_DATE:  # the build reads ISO dates (also partial ones) with any template
        branches.append((_ISO_ELSEWHERE[0], ISO_DATE.key_sql("@D")))
        forms.append(_ISO_ELSEWHERE[1])
    return _Reading(tuple(branches), _partial_rx([*forms, _YEAR_ALONE]))


def _datetime_reading(f: _DatetimeFormat) -> _Reading:
    """Timestamps as the template writes them (its separator before the time, seconds only where
    it has them, no fraction; an offset may follow), ISO dates and timestamps (any time after
    them), the template's date alone (midnight); partial dates (the template's forms with an
    unknown part, with or without the time: a timestamp template reads no date without its day).
    An ISO template reads year-first dates with another separator ('2019/03/05') as it reads its
    own forms."""
    stamp = f"(@D || ' ' || coalesce({_time_key('@T')}, '00:00:00'))"
    d = f.date
    sep = " +" if f.sep == " " else f.sep
    seconds = r":[0-5]\d" if f.seconds else ""
    time = f"({sep}{_HM}{seconds}( *{_OFFSET})?)"
    own = (f"^{d.scan}{time}?$", stamp.replace("@D", d.scan_key("@D")))
    iso = (_ISO_ELSEWHERE[0], stamp.replace("@D", ISO_DATE.key_sql("@D")))
    branches = [iso, own] if f.family == _ISO_DATETIME else [own, iso]
    return _Reading(tuple(branches), _partial_rx([f"({d.scan_partial}){time}?", _ISO_ELSEWHERE[1], _YEAR_ALONE]))


def _time_reading(fmt: str) -> _Reading:
    """Times of day: with seconds only when the format has them."""
    rx = f"^{_HM}:[0-5]\\d$" if fmt.endswith("%S") else f"^{_HM}$"
    return _Reading(((rx, _time_key("@V")),), dated=False, offsets=False)


# Excel serial day numbers: the key is NULL (not conforming) outside 1910 .. 2099
_EXCEL_READING = _Reading(((
    r"^\d{4,5}$",
    f"(CASE WHEN (@V)::int BETWEEN {EXCEL_MIN} AND {EXCEL_MAX} "
    "THEN to_char(DATE '1899-12-30' + (@V)::int, 'YYYY-MM-DD') END)",
),), dated=False, offsets=False)


def _valid_date(k: str) -> str:
    """``k`` ('YYYY-MM-DD...') when its day exists in its month, else NULL. The month and day
    are numbers in range (the branch regex checked them), so nothing here can fail."""
    return (f"(CASE WHEN substr({k}, 9, 2) <= '28' OR substr({k}, 9, 2)::int <= extract(day FROM "
            f"make_date(substr({k}, 1, 4)::int, substr({k}, 6, 2)::int, 1) + interval '1 month - 1 day') THEN {k} END)")


# The shape map: digits and the usual separators first (translate() looks each character up in
# order), then the letters. Characters not in it are kept; non-ASCII ones become '?'.
_SHAPE_FROM = "0123456789/-:. " + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SHAPE_TO = "9" * 10 + "/-:. " + "a" * 26 + "A" * 26


def _shape(v: str) -> str:
    """The shape of a value: digits as 9, letters as a / A, other characters kept (non-ASCII ones
    as '?'); NULL for long values."""
    t = f"translate({v}, '{_SHAPE_FROM}', '{_SHAPE_TO}')"
    return (f"(CASE WHEN length({v}) <= {MAX_SHAPE_LENGTH} THEN CASE WHEN octet_length({v}) = length({v}) "
            f"THEN {t} ELSE regexp_replace({t}, '[^ -~]', '?', 'g') END END)")


# --------------------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------------------


@dataclass
class _Col:
    name: str
    expr: str  # SQL expression yielding the raw text value
    declared_type: str | None
    idx: int
    inferred_type: str = "empty"
    date_format: str | None = None
    reading: _Reading | None = None  # date, datetime, time and Excel columns
    notes: list[str] = field(default_factory=list)
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
    manifest_cols = {c["name"]: c for c in (manifest_entry or {}).get("columns", [])}
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
            declared_type=manifest_cols.get(r["column_name"], {}).get("declared_type"),
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
        key = False
        if c.pii_reason is None and nn:
            key = _is_key_column(c.name, nd)
            if key:
                # Frequencies of a patient/visit key are per-person record counts: withhold them.
                s["lo"] = s["hi"] = s["mean"] = None
            elif not (nd == nn and min_cell > 1) and top_n > 0:
                top = _top_values(conn, tbl, c, min_cell, top_n)
            n_hidden = nd - len(top)
        # text shapes only where the listed values leave most of the column unseen
        shapes, odd_shapes = _shapes(conn, tbl, c, min_cell, 2 * sum(v.count for v in top) < nn, key)
        if c.reading is not None and len(shapes) == 1 and shapes[0].count == nn:
            shapes = []  # one shape for every value says no more than the format
        offsets, rare_offsets = _offsets(conn, tbl, c, min_cell)
        if c.inferred_type == "date":
            c.notes += _placeholder_notes(c, top, nn, min_cell)
        na_markers = sum((manifest_cols.get(c.name, {}).get("na_markers") or {}).values())
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
                n_na_markers=na_markers,
                n_nonconforming=int(s.get("nc") or 0),
                nonconforming_shapes=odd_shapes,
                n_partial_dates=int(s.get("np") or 0),
                value_shapes=shapes,
                tz_offsets=offsets,
                n_rare_offsets=rare_offsets,
                format_notes=[] if c.pii_reason else c.notes,
            )
        )
    return TableProfile(name=table, n_rows=n_rows, columns=profiles)


def _is_key_column(name: str, n_distinct: int) -> bool:
    """A patient key, or a high-cardinality visit/household/record key: its values are not listed."""
    if _PERSON_KEY_NAME.search(name):
        return True
    return n_distinct > KEY_MAX_LISTED_DISTINCT and bool(_KEY_NAME.search(name))


def _date_name(name: str) -> bool:
    return bool(_DATE_NAME.search(name)) and not _NOT_DATE_NAME.search(name)


def _sample_sql(tbl: str, c: _Col, n_rows: int, sampled: bool) -> str:
    """Up to about SAMPLE_VALUES non-null values spread over the table: a row sample (BERNOULLI)
    of mid-sized tables, a block sample (SYSTEM) of large ones, the whole column otherwise."""
    ts, limit = "", SAMPLE_VALUES
    if sampled:
        method = "SYSTEM" if n_rows > TABLESAMPLE_ABOVE else "BERNOULLI"
        pct = min(100.0, 100.0 * SAMPLE_OVERDRAW * SAMPLE_VALUES / max(n_rows, 1))
        ts, limit = f" TABLESAMPLE {method} ({pct:.6f}) REPEATABLE (42)", 2 * SAMPLE_VALUES
    return f"SELECT {_trim(c.expr)} AS v FROM {tbl}{ts} WHERE {c.expr} IS NOT NULL LIMIT {limit}"


def _inspect_sample(
    conn: psycopg.Connection, tbl: str, table: str, c: _Col, n_rows: int, is_base: bool, cfg: ProjectConfig
) -> None:
    """Infer the type and check for PII on a bounded sample of non-null values."""
    sampled = is_base and n_rows > SAMPLE_VALUES
    excel = _date_name(c.name)
    counts = _pattern_counts(conn, _sample_sql(tbl, c, n_rows, sampled), excel)
    if sampled and counts["n"] < SPARSE_SAMPLE:  # a sparse column: the sample holds few of its values
        sampled = False
        counts = _pattern_counts(conn, _sample_sql(tbl, c, n_rows, False), excel)
    _infer_type(c, counts)
    values: list[str] = []
    # Dates are never phone numbers, but ISO dates ("2019-01-05") match the phone pattern in
    # privacy.detect_pii, so date columns are judged on their name (and config) only.
    if c.inferred_type not in ("date", "datetime", "time"):
        values = _distinct_values(conn, _sample_sql(tbl, c, n_rows, sampled))
        if sampled and len(values) < PII_MIN_VALUES:  # too few to judge by: the column's first values
            values = _distinct_values(conn, _sample_sql(tbl, c, n_rows, False))
    c.pii_reason = detect_pii(table, c.name, values, cfg.privacy)


def _distinct_values(conn: psycopg.Connection, sample_sql: str) -> list[str]:
    return [r["v"] for r in fetch_all(conn, f"SELECT DISTINCT v FROM ({sample_sql}) s LIMIT {PII_SAMPLE_VALUES}")]


def _regex_groups(items: list[tuple[Any, set[str]]]) -> list[list[Any]]:
    """Pack items (each with the regular expressions it uses) into groups that use at most
    MAX_REGEXES distinct expressions: PostgreSQL caches 32 compiled expressions per session, and
    a query that cycles through more recompiles them for every row (a hundred times slower)."""
    groups: list[tuple[list[Any], set[str]]] = []
    for item, rxs in items:
        for members, used in groups:
            if len(used | rxs) <= MAX_REGEXES:
                members.append(item)
                used |= rxs
                break
        else:
            groups.append(([item], set(rxs)))
    return [members for members, _ in groups]


# Characters a layout cannot do without: a layout is tried on a sample only when they occur in it.
_MARKS = {"/": "strpos(v, '/') > 0", "-": "strpos(v, '-') > 0", ".": "strpos(v, '.') > 0", ":": "strpos(v, ':') > 0",
          "a": "lower(v) <> upper(v)"}  # 'a': a letter (a month name)


def _marks(fmt: str) -> set[str]:
    return {ch for ch in "/-.:" if ch in fmt} | ({"a"} if "%b" in fmt else set())


def _pattern_counts(conn: psycopg.Connection, sample_sql: str, excel: bool = False) -> dict[str, Any]:
    bools = ", ".join(f"'{b}'" for b in _BOOL_VALUES)
    base = [
        "count(*) AS n",
        f"count(*) FILTER (WHERE lower(v) IN ({bools})) AS n_bool",
        f"count(*) FILTER (WHERE v ~ {_lit(_INT_RE)}) AS n_int",
        f"count(*) FILTER (WHERE v ~ {_lit(_NUM_RE)}) AS n_num",
        f"count(*) FILTER (WHERE v ~* {_lit(_DATE_LIKE)}) AS n_date_like",
        *(f"count(*) FILTER (WHERE {test}) AS mark_{j}" for j, test in enumerate(_MARKS.values())),
    ]
    if excel:
        rx = r"^[0-9]{4,5}$"
        base.append(f"count(CASE WHEN v ~ {_lit(rx)} THEN CASE WHEN v::int BETWEEN {EXCEL_MIN} "
                    f"AND {EXCEL_MAX} THEN 1 END END) AS n_excel")
    out: dict[str, Any] = fetch_one(conn, f"SELECT {', '.join(base)} FROM ({sample_sql}) s") or {"n": 0}
    if not out["n"] or out["n_date_like"] < TYPE_MATCH_RATIO * out["n"]:  # not a date or time column
        return out
    present = {ch for j, ch in enumerate(_MARKS) if out[f"mark_{j}"]}
    parts: list[tuple[str, set[str]]] = []

    def count(name: str, rx: str, fmt: str, flag: str = "i") -> None:
        """A pattern count, or 0 without running it when the sample lacks a character it needs."""
        if _marks(fmt) <= present:
            parts.append((f"count(*) FILTER (WHERE v {'~*' if flag == 'i' else '~'} {_lit(rx)}) AS {name}", {flag + rx}))
        else:
            out[name] = 0

    for i, f in enumerate(DATE_FORMATS):
        count(f"dc_{i}", f.regex, f.fmt)
        count(f"dl_{i}", f.loose, f.fmt)
    for i, f in enumerate(DATETIME_FORMATS):
        count(f"t_{i}", f.regex, f.fmt)
    families = {fam: f.fmt for f in DATETIME_FORMATS for fam in [f.family]}
    for i, fam in enumerate(_FAMILIES):
        count(f"tf_{i}", fam, families[fam])
    for i, (fmt, rx) in enumerate(TIME_FORMATS):
        count(f"tm_{i}", rx, fmt, "c")
    for name, rx in [("yr", f"^{_Y}$"), ("day_gt12", _DAY_ABOVE_12), ("mid_gt12", _MID_ABOVE_12),
                     ("year_last", _YEAR_LAST)]:
        count(name, rx, "", "c")
    # The sample is read once; each group of patterns then runs over all of it before the next.
    if parts:
        groups = ", ".join(f"(SELECT {', '.join(g)} FROM s) g{j}" for j, g in enumerate(_regex_groups(parts)))
        out.update(fetch_one(conn, f"WITH s AS MATERIALIZED ({sample_sql}) SELECT * FROM {groups}") or {})
    return out


def _infer_type(c: _Col, counts: dict[str, Any]) -> None:
    """Pick the first type whose pattern covers >= TYPE_MATCH_RATIO of the sampled values (ISO
    and partial dates count for every date layout) and set how the full scan reads it."""
    n = int(counts["n"])
    c.inferred_type, c.date_format, c.reading = "empty", None, None
    if n == 0:
        return
    need = TYPE_MATCH_RATIO * n
    if counts["n_bool"] >= need:
        c.inferred_type = "boolean"
        return
    if "dl_0" in counts and _infer_date_like(c, counts, n, need):
        return
    if counts.get("n_excel") is not None and int(counts["n_excel"]) >= need:
        c.inferred_type, c.reading = "date", _EXCEL_READING
        c.notes.append(EXCEL_NOTE)
        return
    if counts["n_int"] >= need:
        c.inferred_type = "integer"
    elif counts["n_num"] >= need:
        c.inferred_type = "float"
    else:
        c.inferred_type = "string"


def _infer_date_like(c: _Col, counts: dict[str, Any], n: int, need: float) -> bool:
    """Set a timestamp, date or time type when one fits; False when none does."""
    iso = DATE_FORMATS.index(ISO_DATE)
    iso_loose, iso_complete = int(counts[f"dl_{iso}"]), int(counts[f"dc_{iso}"])
    years = int(counts["yr"])  # a year alone: a partial date under every layout
    own = {f.fmt: int(counts[f"dl_{i}"]) for i, f in enumerate(DATE_FORMATS)}
    complete = {f.fmt: int(counts[f"dc_{i}"]) for i, f in enumerate(DATE_FORMATS)}

    def date_fits(d: _DateFormat) -> int:
        """Values a date layout reads: its own forms, ISO dates (also partial) unless it is ISO,
        and years alone."""
        return own[d.fmt] + (0 if d is ISO_DATE else iso_loose) + years

    # timestamps: at least 1 - TYPE_MATCH_RATIO of the values carry a time in the layout's family
    fam = {fam: int(counts[f"tf_{i}"]) for i, fam in enumerate(_FAMILIES)}
    best_dt: _DatetimeFormat | None = None
    for i, f in enumerate(DATETIME_FORMATS):
        timed = fam[f.family]
        fits = timed + date_fits(f.date) + (0 if f.family == _ISO_DATETIME else fam[_ISO_DATETIME])
        if fits < need or timed < n - need or timed == 0:
            continue
        variant = int(counts[f"t_{i}"])
        if best_dt is None or (timed, variant) > (fam[best_dt.family], int(counts[f"t_{DATETIME_FORMATS.index(best_dt)}"])):
            best_dt = f
    if best_dt is not None:
        c.inferred_type, c.date_format, c.reading = "datetime", best_dt.fmt, _datetime_reading(best_dt)
        c.notes += _order_notes(best_dt.date.fmt, counts)
        return True

    best: _DateFormat | None = None
    for d in DATE_FORMATS:
        whole = complete[d.fmt] + (0 if d is ISO_DATE else iso_complete)
        if date_fits(d) < need or 2 * whole < n:  # mostly partial dates: not a date column
            continue
        if best is None or own[d.fmt] > own[best.fmt]:
            best = d
    if best is not None:
        c.inferred_type, c.date_format, c.reading = "date", best.fmt, _date_reading(best)
        c.notes += _order_notes(best.fmt, counts)
        return True
    times = [int(counts[f"tm_{i}"]) for i in range(len(TIME_FORMATS))]
    if sum(times) >= need:
        c.inferred_type = "time"
        c.date_format = TIME_FORMATS[1 if times[1] > times[0] else 0][0]
        c.reading = _time_reading(c.date_format)
        return True
    return False


def _order_notes(fmt: str, counts: dict[str, Any]) -> list[str]:
    """Notes where the sample cannot settle the order of a layout's fields."""
    notes = []
    if "%y" in fmt:
        notes.append(TWO_DIGIT_YEAR_NOTE)
        if "%b" not in fmt and not int(counts["year_last"]):
            notes.append(YEAR_FIRST_NOTE)
    if fmt[:2] == "%d" and fmt[3:5] == "%m" and not int(counts["day_gt12"]) and not int(counts["mid_gt12"]):
        notes.append(DAY_FIRST_NOTE)
    return notes


def _chunks(items: list[_Col], per_item: int) -> list[list[_Col]]:
    size = max(1, _MAX_TARGETS // per_item)
    return [items[i : i + size] for i in range(0, len(items), size)] or [[]]


def _classified(tbl: str, cols: list[_Col], where: str = "") -> str:
    """A subquery over ``tbl`` with, per column i: r_i (raw value), v_i (trimmed); for date-like
    columns k_i (the sort key of a complete, valid value), p_i (1: a partial date) and tz_i (1: an
    offset follows the time); for integer and boolean columns ok_i, for integer and float
    columns x_i (the number). Each layer is fenced (OFFSET 0), so every expression is evaluated
    once per row: the date part, then the time after it, then the key, then its checks."""
    layers: list[list[str]] = [[] for _ in range(5)]

    def add(j: int, name: str, expr: str) -> None:  # computed in layer j, passed through the later ones
        layers[j].append(f"{expr} AS {name}")
        for later in layers[j + 1 :]:
            later.append(name)

    for c in cols:
        i, e = c.idx, c.expr
        v = f"v_{i}"
        add(0, f"r_{i}", e)
        add(0, v, _trim(e))
        if c.pii_reason is not None:
            continue
        if c.reading is not None:
            rd = c.reading
            keys = " ".join(key for _, key in rd.branches)
            if "@D" in keys or "@T" in keys:
                add(1, f"dp_{i}", _date_part(v))
            if "@T" in keys:
                add(2, f"tm_{i}", _time_of(v, f"dp_{i}"))
            if rd.offsets:
                add(1, f"tz_{i}", f"CASE WHEN {_has_offset(v)} THEN 1 END")
            branches = " ".join(
                f"WHEN {v} ~* {_lit(rx)} THEN {key.replace('@T', f'tm_{i}').replace('@D', f'dp_{i}').replace('@V', v)}"
                for rx, key in rd.branches
            )
            add(3, f"rk_{i}", f"CASE {branches} END")
            add(4, f"k_{i}", _valid_date(f"rk_{i}") if rd.dated else f"rk_{i}")
            partial = f"WHEN rk_{i} IS NULL AND {v} ~* {_lit(rd.partial)} THEN 1" if rd.partial else "WHEN false THEN 1"
            add(4, f"p_{i}", f"CASE {partial} END")
        elif c.inferred_type in ("integer", "float"):
            add(1, f"x_{i}", f"CASE WHEN {v} ~ {_lit(_NUM_RE)} THEN {v}::numeric END")
            if c.inferred_type == "integer":
                add(1, f"ok_{i}", f"CASE WHEN {v} ~ {_lit(_INT_READ)} THEN 1 END")
        elif c.inferred_type == "boolean":
            bools = ", ".join(f"'{b}'" for b in _BOOL_VALUES)
            add(1, f"ok_{i}", f"CASE WHEN lower({v}) IN ({bools}) THEN 1 END")
    sql = f"SELECT {', '.join(layers[0])} FROM {tbl}{where} OFFSET 0"
    for j, layer in enumerate(layers[1:], 1):
        sql = f"SELECT {', '.join(layer)} FROM ({sql}) s{j - 1} OFFSET 0"
    return sql


def _nonconforming(c: _Col) -> str | None:
    """SQL over the classified columns: true for a value that does not conform (None: no rule)."""
    i = c.idx
    if c.pii_reason is not None:
        return None
    if c.reading is not None:
        return f"(v_{i} IS NOT NULL AND k_{i} IS NULL AND p_{i} IS NULL)"
    if c.inferred_type in ("integer", "boolean"):
        return f"(v_{i} IS NOT NULL AND ok_{i} IS NULL)"
    if c.inferred_type == "float":
        return f"(v_{i} IS NOT NULL AND x_{i} IS NULL)"
    return None


_REGEX_USE = re.compile(r"(~\*?)\s*\$re\$(.*?)\$re\$|\$re\$(.*?)\$re\$(, '', 'i')?", re.S)


def _col_regexes(c: _Col) -> set[str]:
    """The regular expressions (with their case flag) that ``_classified`` evaluates for ``c``:
    PostgreSQL caches them by pattern and flags."""
    out: set[str] = set()
    for m in _REGEX_USE.finditer(_classified("t", [c])):
        if m.group(2) is not None:
            out.add(("i" if m.group(1) == "~*" else "c") + m.group(2))
        else:
            out.add(("i" if m.group(4) else "c") + m.group(3))
    return out


def _aggregate_pass(conn: psycopg.Connection, tbl: str, cols: list[_Col]) -> None:
    """Counts, lengths, type-aware ranges and the conforming / partial / non-conforming counts of
    every column, one scan per column chunk."""
    chunks = [chunk for group in _regex_groups([(c, _col_regexes(c)) for c in cols]) for chunk in _chunks(group, 12)]
    for chunk in chunks:
        if not chunk:
            continue
        parts: list[str] = []
        for c in chunk:
            i = c.idx
            r = f"r_{i}"
            parts += [
                f"count({r}) AS nn_{i}",
                f"count(DISTINCT {r}) AS nd_{i}",
                f"min(length({r})) AS lmin_{i}",
                f"max(length({r})) AS lmax_{i}",
            ]
            if c.pii_reason is not None:
                continue
            nc = _nonconforming(c)
            if nc is not None:
                parts.append(f"count(*) FILTER (WHERE {nc}) AS nc_{i}")
            if c.reading is not None:
                parts += [f'min(k_{i} COLLATE "C") AS lo_{i}', f'max(k_{i} COLLATE "C") AS hi_{i}',
                          f"count(p_{i}) AS np_{i}"]
                if c.reading.offsets:
                    parts.append(f"count(tz_{i}) AS ntz_{i}")
            elif c.inferred_type in ("integer", "float"):
                x = f"x_{i}"
                parts += [f"min({x})::text AS lo_{i}", f"max({x})::text AS hi_{i}", f"avg({x})::float8 AS mean_{i}"]
            elif c.inferred_type in ("string", "boolean"):
                parts += [f'min({r} COLLATE "C") AS lo_{i}', f'max({r} COLLATE "C") AS hi_{i}']
        row = fetch_one(conn, f"SELECT {', '.join(parts)} FROM ({_classified(tbl, chunk)}) s") or {}
        for c in chunk:
            i = c.idx
            c.stats = {
                "nn": row.get(f"nn_{i}") or 0,
                "nd": row.get(f"nd_{i}") or 0,
                "lmin": row.get(f"lmin_{i}"),
                "lmax": row.get(f"lmax_{i}"),
                "lo": row.get(f"lo_{i}"),
                "hi": row.get(f"hi_{i}"),
                "nc": row.get(f"nc_{i}") or 0,
                "np": row.get(f"np_{i}") or 0,
                "ntz": row.get(f"ntz_{i}") or 0,
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
    except ValueError:  # cannot happen for a valid key: report the normalised text as found
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


def _shapes(
    conn: psycopg.Connection, tbl: str, c: _Col, min_cell: int, values_hidden: bool, key: bool = False
) -> tuple[list[ValueFreq], list[ValueFreq]]:
    """(value shapes, shapes of the non-conforming values), each seen at least ``min_cell``
    times. Value shapes are listed for dates, timestamps and times, and for text whose values
    are mostly not listed (``values_hidden``); non-conforming shapes for every typed column. None
    for identifiers. In a key column (``key``) a shape also needs ``min_cell`` different values:
    the rows of one key are one person's (or visit's) records."""
    if c.pii_reason is not None or not c.stats.get("nn"):
        return [], []
    nc = _nonconforming(c)
    want_values = c.reading is not None or (c.inferred_type == "string" and values_hidden)
    want_odd = nc is not None and int(c.stats.get("nc") or 0) >= min_cell
    if not (want_values or want_odd):
        return [], []
    i, k = c.idx, int(min_cell)
    if want_odd:  # read every value again to tell which ones do not conform
        source, odd = _classified(tbl, [c], f" WHERE {c.expr} IS NOT NULL"), nc
    else:
        source, odd = f"SELECT {_trim(c.expr)} AS v_{i} FROM {tbl} WHERE {c.expr} IS NOT NULL", "false"
    keep = "" if want_values else f" WHERE {odd}"
    having, n_odd = f"count(*) >= {k}", f"count(*) FILTER (WHERE {odd})"
    if key:
        having += f" AND count(DISTINCT v_{i}) >= {k}"
        n_odd = f"CASE WHEN count(DISTINCT v_{i}) FILTER (WHERE {odd}) >= {k} THEN {n_odd} ELSE 0 END"
    rows = fetch_all(
        conn,
        f"""SELECT sh, n, n_odd FROM (
              SELECT {_shape(f'v_{i}')} AS sh, count(*) AS n, {n_odd} AS n_odd
              FROM ({source}) s{keep}
              GROUP BY 1 HAVING {having}) g
            WHERE sh IS NOT NULL ORDER BY n DESC, sh COLLATE "C" LIMIT 500""",
    )
    shapes = [ValueFreq(value=r["sh"], count=int(r["n"])) for r in rows][:MAX_SHAPES] if want_values else []
    odd_rows = sorted((r for r in rows if int(r["n_odd"]) >= min_cell), key=lambda r: (-int(r["n_odd"]), r["sh"]))
    odd_shapes = [ValueFreq(value=r["sh"], count=int(r["n_odd"])) for r in odd_rows][:MAX_SHAPES]
    return shapes, odd_shapes


def _offsets(conn: psycopg.Connection, tbl: str, c: _Col, min_cell: int) -> tuple[list[ValueFreq], int]:
    """The offsets written after the times of a date or timestamp column ('Z', '+03:00'; '+03'
    and '+0300' are read as '+03:00'), each seen at least ``min_cell`` times, and the number of
    values with a rarer offset."""
    if c.pii_reason is not None or c.reading is None or not c.reading.offsets or not c.stats.get("ntz"):
        return [], 0
    off = ("CASE WHEN tz = 'Z' THEN 'Z' ELSE left(tz, 3) || ':' || "
           "coalesce(nullif(substr(replace(tz, ':', ''), 4, 2), ''), '00') END")
    rows = fetch_all(
        conn,
        f"""SELECT {off} AS off, count(*) AS n FROM (
              SELECT {_offset_of('v')} AS tz FROM (
                SELECT {_trim(c.expr)} AS v FROM {tbl} WHERE strpos({c.expr}, ':') > 0) s0 OFFSET 0) s
            WHERE tz IS NOT NULL GROUP BY 1 ORDER BY count(*) DESC, 1""",
    )
    kept, _ = suppress_small_cells([ValueFreq(value=r["off"], count=int(r["n"])) for r in rows], min_cell)
    return kept, sum(int(r["n"]) for r in rows) - sum(v.count for v in kept)


def _read_date(value: str, c: _Col) -> date | None:
    """A listed value of a date column as the build reads it (None: not a complete date)."""
    v = value.strip()
    if c.reading is _EXCEL_READING:
        if v.isdigit() and EXCEL_MIN <= int(v) <= EXCEL_MAX:
            return date(1899, 12, 30) + timedelta(days=int(v))
        return None
    for fmt in (c.date_format, "%Y-%m-%d"):
        if not fmt or "%y" in fmt:  # two-digit years never lie after the current year
            continue
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def _placeholder_notes(c: _Col, top: list[ValueFreq], nn: int, min_cell: int) -> list[str]:
    """A note when one date after the profile date (before 2099, where the build already reads
    exits as open) is very common: usually a "still resident" placeholder for an exit date."""
    today = utcnow().date()
    future = [(v.count, d) for v in top if (d := _read_date(v.value, c)) is not None and today < d < date(2099, 1, 1)]
    if not future:
        return []
    n, d = max(future)
    if n < max(min_cell, PLACEHOLDER_MIN_SHARE * nn):
        return []
    return [f"{n} values ({100.0 * n / nn:.0f}%) are one and the same date after the profile date (see the top "
            "values): if this column ends an episode (residency exit, end of follow-up), that date is probably a "
            "placeholder for 'still there'; map it to NULL so the episode stays open (the build ends open episodes "
            "at cdm_source.source_release_date)"]
