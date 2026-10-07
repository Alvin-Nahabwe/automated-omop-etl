"""Rewrite the hard conversions of spec SQL into the build's tolerant ones.

Source columns are staged as text, so a field expression converts them: ``v.visit_date::date``,
``to_date(i.dob, 'DD/MM/YYYY')``, ``p.qty::numeric``. In PostgreSQL one value that does not convert
fails the whole statement, and with it the whole table. The compiler therefore rewrites each such
conversion in a column or expression field (and in the mapping's WHERE) into a call of the ETL
helper functions (``etl/functions.sql``):

* ``x::date``, ``CAST(x AS date)`` -> ``{etl}.omop_as_date_at('<site>', '<policy>', x)``, and likewise
  for ``timestamp``, ``timestamptz`` (``timestamp with time zone``), ``integer`` (int, int4, int2,
  smallint), ``bigint`` (int8) and ``numeric`` (decimal, float, float4, float8, real, double
  precision: kind "number");
* ``to_date(x, 'T')`` -> ``{etl}.omop_as_date_at('<site>', '<policy>', x, 'T')``, and
  ``to_timestamp(x, 'T')`` -> ``{etl}.omop_as_timestamptz_at(...)`` (it returns timestamptz);
* ``{etl}.try_date(x[, 'T'[, policy]])`` and the other uncounted ``try_*`` functions, which spec
  SQL may call: in a field or the WHERE they are counted like a cast, with the field's policy.

A conversion of an operand that is not a column (or a few cheap operators over columns) is
written as the ``*_once`` form of the date functions (``omop_as_date_once_at``, uncounted
``omop_try_date_once``), which reads the value in plpgsql: PostgreSQL does not inline a SQL
function that references an expensive or volatile value several times, and the SQL reader would
then be planned for every row (``_simple_operand``).

A value that cannot be read becomes NULL and is counted at the site (the CDM field, or 'where');
partial dates follow the policy (``etl.policies.partial_policy``). With ``counted=False`` the
uncounted ``{etl}.try_*`` functions are written instead (for the count queries, which must stay
parallel). The operand keeps its text, so its own type picks the function's overload: text is
parsed, a date, timestamp, number or boolean converts as the cast would.

A type modifier is applied after the conversion: ``numeric(10,2)`` rounds to two decimals (its
precision is dropped, so a large value is no overflow error), ``timestamp(0)`` rounds the seconds.
Floating-point casts (float8, real, double precision) are read as numeric: arithmetic on them is
exact decimal arithmetic, and a later integer conversion rounds halves away from zero (2.5 -> 3),
where float8 would round them to even.

Left as written: casts of constants (literals, typed literals, NULL, CURRENT_DATE: they are
folded when the statement is planned, so EXPLAIN checks them), casts of ``EXTRACT`` / ``date_part``
fields that always fit (``EXTRACT(YEAR FROM d)::int``), array casts (``::int[]``), casts to other
types, to_date / to_timestamp with a template that is not a literal or that the helper functions
do not read (``template_issue``), and conversions whose operand the scan cannot delimit. They are
listed in ``Rewrite.left_hard`` for the validator; a strftime template (``'%d/%m/%Y'``, which
PostgreSQL silently misreads) is also in ``Rewrite.bad_templates``, which the compiler refuses.
The scan is a small SQL lexer (strings, ``E''`` escapes, ``$tag$`` literals, quoted names,
comments), not a parser: an expression it cannot read is left exactly as written and reported,
and nothing here raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Kinds of conversion: the counted front end and the uncounted function of each.
KINDS = ("date", "timestamp", "timestamptz", "integer", "bigint", "number")
DATE_KINDS = frozenset({"date", "timestamp", "timestamptz"})
# CDM datatype -> the kind of its tolerant conversion (other types are text: no conversion).
_DATATYPE_KIND = {
    "integer": "integer", "int": "integer", "int4": "integer",
    "bigint": "bigint", "int8": "bigint",
    "float": "number", "numeric": "number", "double": "number", "real": "number",
    "date": "date",
    "datetime": "timestamp", "timestamp": "timestamp",
}
# Cast target type words -> kind, longest first (multi-word types are matched word by word).
_TYPES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("timestamp", "with", "time", "zone"), "timestamptz"),
    (("timestamp", "without", "time", "zone"), "timestamp"),
    (("double", "precision"), "number"),
    (("timestamptz",), "timestamptz"),
    (("timestamp",), "timestamp"),
    (("date",), "date"),
    (("numeric",), "number"), (("decimal",), "number"), (("float8",), "number"), (("float4",), "number"),
    (("float",), "number"), (("real",), "number"),
    (("integer",), "integer"), (("int",), "integer"), (("int4",), "integer"), (("int2",), "integer"),
    (("smallint",), "integer"),
    (("bigint",), "bigint"), (("int8",), "bigint"),
)
_FUNCTIONS = {"to_date": "date", "to_timestamp": "timestamptz"}
# The uncounted helper functions spec SQL may call (etl/functions.sql) -> their kind.
_TRY_KINDS = {f"try_{k}": k for k in KINDS}
# Functions whose result changes from call to call: a value with one is not inlined.
_VOLATILE_FUNCTIONS = {"random", "nextval", "setval", "clock_timestamp", "timeofday", "setseed", "gen_random_uuid",
                       "uuid_generate_v4", "txid_current", "pg_sleep"}
# The cost (operators and calls) up to which PostgreSQL inlines a function that references its
# argument several times is 10; built-in calls can cost more than one, so the scan keeps a margin.
_MAX_SIMPLE_COST = 6
# Words before '(' that make the parenthesised group an expression, not a function call.
_NOT_FUNC = {
    "and", "or", "not", "in", "is", "when", "then", "else", "case", "like", "ilike", "similar", "between", "on",
    "from", "where", "select", "as", "distinct", "values", "using", "by", "having", "of", "for", "to", "escape",
    "isnull", "notnull", "with", "return", "returning", "into", "end", "do", "if", "elsif", "at", "zone",
}
# Words an operand may follow (besides operators, '(', ',' and the start).
_BOUNDARY_WORDS = _NOT_FUNC | {"over", "filter", "within", "group", "order", "partition", "asc", "desc", "nulls",
                               "first", "last", "limit", "offset", "union", "all", "except", "intersect"}
# Operands that are constants (folded when the statement is planned).
_CONSTANT_WORDS = {"null", "true", "false", "current_date", "current_timestamp", "localtimestamp", "localtime",
                   "current_time", "now"}
# EXTRACT / date_part fields whose value always fits an integer.
_SMALL_FIELDS = {"year", "month", "day", "dow", "doy", "isodow", "isoyear", "week", "quarter", "hour", "minute",
                 "decade", "century", "millennium", "second", "milliseconds", "microseconds", "timezone",
                 "timezone_hour", "timezone_minute"}

_TOKEN = re.compile(
    r"""
     (?P<ws>\s+)
    |(?P<comment>--[^\n]*|/\*.*?\*/)
    |(?P<estr>(?<![\w$])[Ee]'(?:[^'\\]|\\.|'')*')
    |(?P<dollar>\$(?P<tag>(?:[A-Za-z_][A-Za-z0-9_]*)?)\$.*?\$(?P=tag)\$)
    |(?P<str>'(?:[^']|'')*')
    |(?P<qid>"(?:[^"]|"")*")
    |(?P<ph>\{\w+\})
    |(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
    |(?P<id>[^\W\d][\w$]*)
    |(?P<cast>::)
    |(?P<p>[()\[\],.;])
    |(?P<op>(?:(?!--|/\*|::)[^\s\w'"()\[\],.;{])+|\{)
    """,
    re.S | re.X,
)
_MAX_STEPS = 500  # rewrites of one expression (each removes one conversion)


@dataclass
class Tok:
    kind: str
    text: str
    start: int
    end: int


class LexError(ValueError):
    """The text is not SQL the scan can read (an unterminated string, say)."""


def lex(sql: str) -> list[Tok]:
    """The tokens of ``sql``, as PostgreSQL reads them with standard_conforming_strings on: a
    backslash escapes only inside ``E'...'``. Raises ``LexError``."""
    out: list[Tok] = []
    pos = 0
    while pos < len(sql):
        m = _TOKEN.match(sql, pos)
        if not m or m.end() == pos:
            raise LexError(f"cannot read the SQL at {sql[pos:pos + 20]!r}")
        out.append(Tok(m.lastgroup or "op", m.group(), m.start(), m.end()))
        pos = m.end()
    return out


@dataclass
class Conversion:
    """One hard conversion found in an expression."""

    kind: str  # one of KINDS
    operand: str  # the text converted
    text: str  # the whole construct as written
    template: str | None = None  # the to_date / to_timestamp template (a literal), if any
    function: str | None = None  # to_date / to_timestamp; None for a cast
    rewritten: bool = False
    note: str | None = None  # why it was left as written


@dataclass
class Rewrite:
    sql: str
    # conversions left as written that can still fail the table on one bad value
    left_hard: list[str] = field(default_factory=list)
    # counter sites the rewritten SQL can set (the site, when anything was rewritten with counters)
    sites: tuple[str, ...] = ()
    conversions: list[Conversion] = field(default_factory=list)
    # literal templates that can never be right: (template, problem); the compiler refuses them
    bad_templates: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(c.rewritten for c in self.conversions)


# --------------------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------------------

# to_date / to_timestamp template tokens the helper functions read (omop_fmt_spec), longest first
_TEMPLATE_TOKENS = ("TZH:TZM", "MONTH", "YYYY", "HH24", "HH12", "MON", "TZH", "YY", "MM", "MI", "DD", "HH", "SS",
                    "MS", "US", "AM", "PM", "OF")
_TZ_TOKENS = {"TZH:TZM", "TZH", "OF"}


def template_issue(fmt: str) -> str | None:
    """Why the helper functions cannot read a to_date / to_timestamp template (None: they can).
    Mirrors ``omop_fmt_spec`` in etl/functions.sql."""
    if fmt.upper() == "ISO":
        return None
    if "%" in fmt:
        return (f"unsupported date template '{fmt}': a strftime pattern, not a PostgreSQL template "
                "(e.g. 'DD/MM/YYYY' for %d/%m/%Y)")
    tokens: list[str] = []
    i = 0
    while i < len(fmt):
        u = fmt[i:i + 7].upper()
        if u.startswith("FM"):
            i += 2
            continue
        tok = next((t for t in _TEMPLATE_TOKENS if u.startswith(t)), None)
        if tok:
            tokens.append(tok)
            i += len(tok)
        elif fmt[i] == '"':
            j = fmt.find('"', i + 1)
            if j < 0:
                return f"unsupported date template '{fmt}': unterminated quote"
            i = j + 1
        elif fmt[i].isalnum():
            return (f"unsupported date template '{fmt}' at '{fmt[i:]}' (supported: YYYY YY MM DD Mon Month HH24 "
                    "HH12 HH MI SS MS US AM PM TZH:TZM TZH OF FM, separators and \"quoted\" text)")
        else:
            i += 1
    tz = [k for k, t in enumerate(tokens) if t in _TZ_TOKENS]
    if tz and tz[0] != len(tokens) - 1:
        return f"unsupported date template '{fmt}': the offset (TZH, OF) must come last"
    return None


def _literal_text(tok: Tok) -> str | None:
    """The value of a plain '...' string literal (None for any other token)."""
    if tok.kind != "str":
        return None
    return tok.text[1:-1].replace("''", "'")


# --------------------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------------------


def _sig(toks: list[Tok], i: int, step: int) -> int | None:
    """Index of the next significant token from ``i`` (exclusive) in direction ``step``."""
    i += step
    while 0 <= i < len(toks) and toks[i].kind in ("ws", "comment"):
        i += step
    return i if 0 <= i < len(toks) else None


def _match_back(toks: list[Tok], j: int, close: str, open_: str) -> int | None:
    depth = 0
    for k in range(j, -1, -1):
        t = toks[k]
        if t.kind != "p":
            continue
        if t.text == close:
            depth += 1
        elif t.text == open_:
            depth -= 1
            if depth == 0:
                return k
    return None


def _match_forward(toks: list[Tok], i: int) -> int | None:
    """Index of the ')' (or ']') that closes the bracket at ``i``."""
    depth = 0
    for k in range(i, len(toks)):
        t = toks[k]
        if t.kind != "p":
            continue
        if t.text in "([":
            depth += 1
        elif t.text in ")]":
            depth -= 1
            if depth == 0:
                return k
    return None


def _word(t: Tok | None) -> str:
    return t.text.lower() if t is not None and t.kind == "id" else ""


def _chain_start(toks: list[Tok], j: int) -> int | None:
    """Start of the name chain ``a.b.c`` (also ``{src}.t``, ``"A"."B"``) that ends at ``j``; None
    for a field of a parenthesised value (``(r).f``)."""
    s = j
    while True:
        d = _sig(toks, s, -1)
        if d is None or toks[d].text != "." or toks[d].kind != "p":
            return s
        e = _sig(toks, d, -1)
        if e is None or toks[e].kind not in ("id", "qid", "ph"):
            return None
        s = e


_CONST = -1  # _operand_start: the operand is a constant


def _operand_start(toks: list[Tok], j: int) -> int | None:
    """Index of the first token of the operand that ends at token ``j`` (the token before '::');
    ``_CONST`` for a constant; None when the scan cannot delimit it."""
    t = toks[j]
    if t.kind in ("str", "estr", "dollar", "num"):
        return _CONST  # a literal, or a typed literal (DATE '2020-01-01')
    if t.kind == "p" and t.text == ")":
        k = _match_back(toks, j, ")", "(")
        if k is None:
            return None
        b = _sig(toks, k, -1)
        w = _word(toks[b]) if b is not None else ""
        if w in ("over", "filter"):  # f(x) OVER (...) / FILTER (...): the operand is the call
            p = _sig(toks, b, -1) if b is not None else None
            return _operand_start(toks, p) if p is not None else None
        if w == "group":  # f(x) WITHIN GROUP (ORDER BY ...)
            p = _sig(toks, b, -1) if b is not None else None
            q = _sig(toks, p, -1) if p is not None and _word(toks[p]) == "within" else None
            return _operand_start(toks, q) if q is not None else None
        if w in ("exists", "array"):
            return b
        if w in ("any", "all", "some"):
            return None
        if b is not None and (toks[b].kind == "qid" or (toks[b].kind == "id" and w not in _NOT_FUNC)):
            return _chain_start(toks, b)  # a function call
        return k  # a parenthesised expression
    if t.kind == "p" and t.text == "]":
        k = _match_back(toks, j, "]", "[")
        b = _sig(toks, k, -1) if k is not None else None
        if b is None:
            return None
        if _word(toks[b]) == "array":
            return b
        return _operand_start(toks, b)  # a subscript: x[1]
    if t.kind == "id" and t.text.lower() == "end":  # CASE ... END
        depth = 0
        for k in range(j, -1, -1):
            w = _word(toks[k])
            if w == "end":
                depth += 1
            elif w == "case":
                depth -= 1
                if depth == 0:
                    return k
        return None
    if t.kind == "id" and t.text.lower() in _CONSTANT_WORDS:
        return _CONST
    if t.kind in ("id", "qid", "ph"):
        return _chain_start(toks, j)
    return None


def _extend_operand(toks: list[Tok], s: int) -> int | None:
    """``s`` widened over earlier casts (``x::text::date``: the operand is ``x::text``), checked to
    start where an operand can start."""
    while True:
        b = _sig(toks, s, -1)
        if b is None:
            return s
        tb = toks[b]
        if tb.kind == "cast":
            p = _sig(toks, b, -1)
            if p is None:
                return None
            start = _operand_start(toks, p)
            if start is None or start == _CONST:
                return None if start is None else _CONST
            s = start
            continue
        if tb.kind == "op" or (tb.kind == "p" and tb.text in "(,[") or _word(tb) in _BOUNDARY_WORDS:
            return s
        return None  # e.g. the second word of a type name (double precision)


def _cast_type(toks: list[Tok], i: int, stop: int | None = None) -> tuple[str, int, str] | None:
    """For the type after the '::' (or AS) at ``i``: (kind, index of its last token, type modifier
    text such as '(10,2)' or ''); None for a type that is not converted (text, an array type, ...)."""
    idx: list[int] = []
    k = i
    while len(idx) < 4:
        n = _sig(toks, k, 1)
        if n is None or (stop is not None and n >= stop) or toks[n].kind != "id":
            break
        idx.append(n)
        k = n
    words = [toks[n].text.lower() for n in idx]
    for names, kind in _TYPES:
        if len(words) < 1 or words[0] != names[0]:
            continue
        last = idx[0]
        modifier = ""
        rest = list(names[1:])
        n = _sig(toks, last, 1)
        if n is not None and toks[n].text == "(" and toks[n].kind == "p":  # numeric(10,2), timestamp(0)
            close = _match_forward(toks, n)
            if close is None:
                return None
            modifier = "".join(t.text for t in toks[n:close + 1])
            last = close
        # the remaining words of the type (with time zone), after the modifier
        for word in rest:
            n = _sig(toks, last, 1)
            if n is None or _word(toks[n]) != word:
                break
            last = n
        else:
            if names == ("timestamp",):
                n = _sig(toks, last, 1)
                if n is not None and _word(toks[n]) in ("with", "without"):
                    return None  # timestamp(p) with... : not read
            n = _sig(toks, last, 1)
            if n is not None and toks[n].kind == "p" and toks[n].text == "[":
                return None  # an array type
            return kind, last, modifier
    return None


def _args(toks: list[Tok], open_: int, close: int) -> list[tuple[int, int]]:
    """(first, last) significant token indexes of each top-level argument between the brackets
    (first > last for an empty one)."""
    out: list[tuple[int, int]] = []
    depth = 0
    first = open_

    def bounds(a: int, b: int) -> tuple[int, int]:
        s, e = _sig(toks, a, 1), _sig(toks, b, -1)
        return (s, e) if s is not None and e is not None and s <= e else (b, a)

    for k in range(open_ + 1, close):
        t = toks[k]
        if t.kind == "p" and t.text in "([":
            depth += 1
        elif t.kind == "p" and t.text in ")]":
            depth -= 1
        elif t.kind == "p" and t.text == "," and depth == 0:
            out.append(bounds(first, k))
            first = k
    out.append(bounds(first, close))
    return out


def _span(sql: str, toks: list[Tok], a: int, b: int) -> str:
    return sql[toks[a].start:toks[b].end].strip()


def _small_extract(toks: list[Tok], s: int, j: int) -> bool:
    """True when tokens s..j are EXTRACT(<field> FROM ...) or date_part('<field>', ...) of a field
    whose value always fits an integer."""
    w = _word(toks[s])
    n = _sig(toks, s, 1)
    if w not in ("extract", "date_part") or n is None or toks[n].text != "(" or _match_forward(toks, n) != j:
        return False
    f = _sig(toks, n, 1)
    if f is None:
        return False
    name = _literal_text(toks[f]) if w == "date_part" else _word(toks[f])
    return (name or "").lower() in _SMALL_FIELDS


# --------------------------------------------------------------------------------------
# rewriting
# --------------------------------------------------------------------------------------


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _simple_operand(operand: str) -> bool:
    """True when PostgreSQL inlines a date conversion of ``operand`` into the statement, which the
    fast path needs: a column, a literal, or a few cheap operators and built-in calls over them. Not
    an operand with a conversion in it (it becomes a counted, volatile call), an ETL helper call, a
    subquery or a volatile function: PostgreSQL does not inline a function that references such a
    value several times (see the module docstring)."""
    try:
        toks = [t for t in lex(operand) if t.kind not in ("ws", "comment")]
    except LexError:
        return False
    cost = 0
    for k, t in enumerate(toks):
        w = _word(t)
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if (t.kind == "ph" and t.text == "{etl}") or w in ("select", "cast") or w in _FUNCTIONS or w in _VOLATILE_FUNCTIONS:
            return False
        if t.kind == "cast":
            ct = _cast_type(toks, k)
            if ct is not None:
                return False  # rewritten into a counted call
            cost += 2  # an I/O conversion: an output and an input function
        elif t.kind == "op" or w in ("like", "ilike", "similar", "between", "in", "is"):
            cost += 1
        elif t.kind in ("id", "qid") and nxt is not None and nxt.text == "(" and w not in _NOT_FUNC:
            cost += 1
    return cost <= _MAX_SIMPLE_COST


def _date_typed(operand: str) -> bool:
    """True when ``operand`` is a date or time value, not text: one conversion to a date kind
    (``to_date(...)``, ``x::date``, a helper call). Its conversion is an identity overload, which
    references the value once and is always inlined."""
    try:
        toks = [t for t in lex(operand) if t.kind not in ("ws", "comment")]
    except LexError:
        return False
    if not toks:
        return False
    if toks[0].kind == "p" and toks[0].text == "(" and _match_forward(toks, 0) == len(toks) - 1:
        return _date_typed(operand[toks[0].end:toks[-1].start])
    first = 2 if len(toks) > 3 and toks[0].text == "{etl}" and toks[1].text == "." else 0  # the function name
    if len(toks) > first + 1 and toks[first + 1].text == "(" and _match_forward(toks, first + 1) == len(toks) - 1:
        name = _word(toks[first])
        helper = re.fullmatch(r"(omop_as_|omop_try_|try_)(date|timestamp|timestamptz)(_once)?(_at)?", name)
        if helper if first else name in _FUNCTIONS:
            return True
        if not first and name in ("coalesce", "greatest", "least", "nullif"):  # the type of the first argument
            (a, b) = _args(toks, 1, len(toks) - 1)[0]
            return a <= b and _date_typed(operand[toks[a].start:toks[b].end])
    depth, last_cast = 0, None
    for k, t in enumerate(toks):
        if t.kind == "p" and t.text in "([":
            depth += 1
        elif t.kind == "p" and t.text in ")]":
            depth -= 1
        elif t.kind == "cast" and depth == 0:
            last_cast = k
    ct = _cast_type(toks, last_cast) if last_cast is not None else None
    return ct is not None and ct[0] in DATE_KINDS and ct[1] == len(toks) - 1


def _call(kind: str, site: str, policy: str, operand: str, template: str | None, counted: bool) -> str:
    """The tolerant conversion of ``operand`` (SQL text) to ``kind``."""
    once = "_once" if kind in DATE_KINDS and not _simple_operand(operand) and not _date_typed(operand) else ""
    if counted:
        args = [_quote(site), _quote(policy), operand] + ([template] if template is not None else [])
        return f"{{etl}}.omop_as_{kind}{once}_at({', '.join(args)})"
    name = f"omop_try_{kind}_once" if once else f"try_{kind}"
    if kind in DATE_KINDS:
        return f"{{etl}}.{name}({operand}, {template if template is not None else _quote('ISO')}, {_quote(policy)})"
    return f"{{etl}}.{name}({operand})"


def _with_modifier(call: str, kind: str, modifier: str) -> str:
    """``call`` with the cast's type modifier applied: numeric(p, s) rounds to s decimals (the
    precision is dropped, so a large value is not an overflow error); timestamp(p) rounds the
    seconds as the cast does."""
    if not modifier:
        return call
    if kind == "number":
        parts = [x.strip() for x in modifier.strip("()").split(",")]
        scale = parts[1] if len(parts) > 1 and parts[1].isdigit() else "0"
        return f"round({call}, {scale})"
    return f"{call}::{kind}{modifier}"


def _next_rewrite(sql: str, toks: list[Tok], site: str, policy: str, counted: bool, out: Rewrite,
                  seen: set[str | int]) -> str | None:
    """``sql`` with its first rewritable conversion rewritten, or None when there is none left. A
    conversion left as written is recorded once and skipped: ``seen`` holds the text and the start
    of each one recorded (a rewrite inside it changes its text, one before it its start)."""

    def keep(conv: Conversion, start: int) -> None:
        if conv.text in seen or start in seen:
            return
        seen.update((conv.text, start))
        out.conversions.append(conv)
        out.left_hard.append(conv.text)

    for i, t in enumerate(toks):
        low = _word(t)
        # {etl}.try_<kind>(x[, 'T'[, policy]]): counted like a cast, with the field's policy
        if t.kind == "ph" and t.text == "{etl}":
            d = _sig(toks, i, 1)
            n = _sig(toks, d, 1) if d is not None and toks[d].text == "." else None
            kind = _TRY_KINDS.get(_word(toks[n]) if n is not None else "")
            o = _sig(toks, n, 1) if kind and n is not None else None
            if o is None or toks[o].text != "(":
                continue
            close = _match_forward(toks, o)
            if close is None:
                return None
            args = _args(toks, o, close)
            (a0, a1) = args[0]
            fmt = args[1] if len(args) > 1 else None
            if a0 > a1 or len(args) > (3 if kind in DATE_KINDS else 1):
                continue  # not a call of the helper's signature: PostgreSQL reports it
            template = toks[fmt[0]].text if fmt is not None and fmt[0] == fmt[1] else None
            if fmt is not None and (template is None or _literal_text(toks[fmt[0]]) is None):
                continue  # a template that is not a literal: read as written, uncounted
            if not counted:  # already the uncounted form: only the policy changes, once
                pol = args[2] if len(args) > 2 else None
                if kind not in DATE_KINDS or (pol is not None and pol[0] == pol[1]
                                              and _literal_text(toks[pol[0]]) == policy):
                    continue
            operand = _span(sql, toks, a0, a1)
            out.conversions.append(Conversion(kind, operand, _span(sql, toks, i, close),
                                              _literal_text(toks[fmt[0]]) if fmt is not None else None,
                                              function=f"try_{kind}", rewritten=True))
            new = _call(kind, site, policy, operand, template, counted)
            return sql[:t.start] + new + sql[toks[close].end:]
        # to_date(x, 'T') / to_timestamp(x, 'T')
        if low in _FUNCTIONS:
            n = _sig(toks, i, 1)
            if n is None or toks[n].text != "(":
                continue
            close = _match_forward(toks, n)
            if close is None:
                return None
            kind = _FUNCTIONS[low]
            text = _span(sql, toks, i, close)
            args = _args(toks, n, close)
            p = _sig(toks, i, -1)
            if p is not None and toks[p].text == ".":
                keep(Conversion(kind, text, text, function=low, note="a schema-qualified call"), t.start)
                continue
            if len(args) != 2:
                if low == "to_date" or len(args) > 2:
                    keep(Conversion(kind, text, text, function=low, note="not a two-argument call"), t.start)
                continue  # to_timestamp(epoch): cannot fail on a number
            (a0, a1), (b0, b1) = args
            template = _literal_text(toks[b0]) if b0 == b1 else None
            if template is None:
                keep(Conversion(kind, text, text, function=low, note="the template is not a literal"), t.start)
                continue
            issue = template_issue(template)
            if issue:
                if "%" in template and (template, issue) not in out.bad_templates:
                    out.bad_templates.append((template, issue))
                keep(Conversion(kind, _span(sql, toks, a0, a1), text, template, function=low, note=issue), t.start)
                continue
            operand = _span(sql, toks, a0, a1)
            out.conversions.append(Conversion(kind, operand, text, template, function=low, rewritten=True))
            new = _call(kind, site, policy, operand, toks[b0].text, counted)
            return sql[:t.start] + new + sql[toks[close].end:]
        # CAST(x AS type)
        if low == "cast":
            n = _sig(toks, i, 1)
            if n is None or toks[n].text != "(":
                continue
            close = _match_forward(toks, n)
            if close is None:
                return None
            depth, as_at = 0, None
            for k in range(n + 1, close):
                tk = toks[k]
                if tk.kind == "p" and tk.text in "([":
                    depth += 1
                elif tk.kind == "p" and tk.text in ")]":
                    depth -= 1
                elif depth == 0 and _word(tk) == "as":
                    as_at = k
            if as_at is None:
                continue
            ct = _cast_type(toks, as_at, stop=close)
            if ct is None or _sig(toks, ct[1], 1) != close:
                continue  # not one of the converted types
            kind, _, modifier = ct
            first, last = _sig(toks, n, 1), _sig(toks, as_at, -1)
            text = _span(sql, toks, i, close)
            if first is None or last is None or first > last:
                keep(Conversion(kind, text, text, note="no operand"), t.start)
                continue
            operand = _span(sql, toks, first, last)
            if _operand_start(toks, last) == _CONST and first == last:
                continue
            out.conversions.append(Conversion(kind, operand, text, rewritten=True))
            new = _call(kind, site, policy, operand, None, counted)
            new = _with_modifier(new, kind, modifier)
            return sql[:t.start] + new + sql[toks[close].end:]
        # x::type
        if t.kind == "cast":
            ct = _cast_type(toks, i)
            if ct is None:
                continue
            kind, last, modifier = ct
            j = _sig(toks, i, -1)
            s = _operand_start(toks, j) if j is not None else None
            if s is not None and s != _CONST:
                s = _extend_operand(toks, s)
            if s == _CONST:
                continue
            if s is None or j is None:
                text = _span(sql, toks, j if j is not None else i, last)
                keep(Conversion(kind, text, text, note="the operand cannot be delimited"), toks[j if j is not None else i].start)
                continue
            if kind in ("integer", "bigint", "number") and _small_extract(toks, s, j):
                continue  # EXTRACT(YEAR FROM ...)::int always fits
            operand = _span(sql, toks, s, j)
            if _is_whole_call(operand, kind, True) or _is_whole_call(operand, kind, False):
                continue  # the type modifier of a conversion already written: numeric(10,2)
            out.conversions.append(Conversion(kind, operand, _span(sql, toks, s, last), rewritten=True))
            new = _call(kind, site, policy, operand, None, counted)
            new = _with_modifier(new, kind, modifier)
            return sql[:toks[s].start] + new + sql[toks[last].end:]
    return None


def rewrite(expr: str | None, site: str, policy: str, *, counted: bool = True) -> Rewrite:
    """``expr`` with its hard conversions rewritten tolerantly (see the module docstring). Never
    raises: an expression the scan cannot read is returned as written, in ``left_hard``."""
    if not expr or not expr.strip():
        return Rewrite(expr or "")
    out = Rewrite(expr)
    seen: set[str | int] = set()
    sql = expr
    try:
        for _ in range(_MAX_STEPS):
            new = _next_rewrite(sql, lex(sql), site, policy, counted, out, seen)
            if new is None:
                break
            sql = new
        else:
            raise LexError("too many conversions")
    except Exception:  # the expression stays as written, and is reported
        return Rewrite(expr, left_hard=[expr.strip()])
    out.sql = sql
    out.sites = (site,) if counted and out.changed else ()
    return out


_CONVERSION_CALL = re.compile(r"^(omop_as_[a-z]+(_once)?_at|try_[a-z]+|omop_try_[a-z]+_once)$")


def without_conversions(sql: str) -> str:
    """``sql`` with the conversion calls cut out: those a rewrite wrote (``{etl}.omop_as_date_at(...)``,
    ``{etl}.try_date(...)``) and to_date / to_timestamp calls. What is left is converted only by
    the casts around it."""
    try:
        toks = lex(sql)
    except LexError:
        return sql
    cut: list[tuple[int, int]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if _word(t) in _FUNCTIONS:
            o = _sig(toks, i, 1)
            close = _match_forward(toks, o) if o is not None and toks[o].text == "(" else None
            if close is not None:
                p = _sig(toks, i, -1)
                start = toks[_sig(toks, p, -1) or 0].start if p is not None and toks[p].text == "." else t.start
                cut.append((start, toks[close].end))
                i = close + 1
                continue
        if t.kind == "ph" and t.text == "{etl}":
            d, n = _sig(toks, i, 1), None
            n = _sig(toks, d, 1) if d is not None and toks[d].text == "." else None
            o = _sig(toks, n, 1) if n is not None and _CONVERSION_CALL.match(toks[n].text) else None
            close = _match_forward(toks, o) if o is not None and toks[o].text == "(" else None
            if close is not None:
                cut.append((t.start, toks[close].end))
                i = close + 1
                continue
        i += 1
    out, pos = [], 0
    for a, b in cut:
        out.append(sql[pos:a])
        out.append(" ")
        pos = b
    return "".join(out) + sql[pos:]


def kind_of(datatype: str) -> str | None:
    """The conversion kind of a CDM datatype (None: text, not converted)."""
    return _DATATYPE_KIND.get(datatype.strip().lower())


def _is_whole_call(sql: str, kind: str, counted: bool) -> bool:
    """True when ``sql`` is one call of the conversion to ``kind`` and nothing else."""
    try:
        toks = [t for t in lex(sql) if t.kind not in ("ws", "comment")]
    except LexError:
        return False
    names = {f"omop_as_{kind}_at", f"omop_as_{kind}_once_at"} if counted else {f"try_{kind}", f"omop_try_{kind}_once"}
    if len(toks) < 5 or toks[0].text != "{etl}" or toks[1].text != "." or toks[2].text not in names:
        return False
    if toks[3].text != "(":
        return False
    return _match_forward(toks, 3) == len(toks) - 1


def tolerant_value(expr: str, datatype: str, site: str, policy: str, *, counted: bool = True) -> Rewrite:
    """A column or expression field's value: ``expr`` rewritten (``rewrite``) and, for a numeric or
    date CDM type, converted to it tolerantly as well (once: an expression that is already one
    such conversion is not wrapped again). For a text type the caller applies its own cast."""
    r = rewrite(expr, site, policy, counted=counted)
    kind = kind_of(datatype)
    if kind is None:
        return r
    body = r.sql.strip()
    if not _is_whole_call(body, kind, counted):
        operand = f"({body}\n)" if "--" in body else f"({body})"
        body = _call(kind, site, policy, operand, None, counted)
    r.sql = body
    r.sites = (site,) if counted else ()
    return r

