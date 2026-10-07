"""Static safety checks for SQL fragments written by the LLM (spec expressions, joins, filters).

Spec SQL is executed by the build, so a fragment must stay a single expression over the staged
source and vocabulary: no statement separators, no access to server files, other databases or
session settings, no catalog snooping, and no calls of the build's own helper functions
(``{etl}.omop_*``, which count the values the build could not load as written). A quoted name
("set_config", "pg_catalog") is checked as the name it is, and a Unicode-escaped one (U&"...") is
refused. This complements (it does not replace) running the build under a least-privilege role
(``target.etl_role``).
"""

from __future__ import annotations

import re

# Literals ('...', E'...', $tag$...$tag$), quoted identifiers ("...") and comments are removed
# before scanning, so a column called "comment" or a string containing ';' is not a problem. They
# are lexed as PostgreSQL does with standard_conforming_strings on (the default, and pinned for
# spec SQL by etl.session): a backslash escapes only inside E'...', so '\' is a whole string.
# The E must start a token: in name'\' it ends the type name and the string is a plain one.
_STRIP = re.compile(
    r"(?P<dollar>\$(?P<tag>[A-Za-z_]*)\$.*?\$(?P=tag)\$)"
    r"|(?<![\w$])[Ee]'(?:[^'\\]|\\.|'')*'"
    r"|'(?:[^']|'')*'"
    r'|"(?:[^"]|"")*"'
    r"|--[^\n]*"
    r"|/\*.*?\*/",
    re.S,
)
_DENIED_FUNCTIONS = re.compile(
    r"(?<![.\w])(pg_read_file|pg_read_binary_file|pg_ls_\w+|pg_stat_file|lo_import|lo_export|lo_get|lo_put|"
    r"dblink\w*|query_to_xml\w*|table_to_xml\w*|schema_to_xml\w*|database_to_xml\w*|cursor_to_xml\w*|"
    r"query_to_xmlschema|pg_sleep\w*|set_config|pg_terminate_backend|pg_cancel_backend|pg_reload_conf|"
    r"pg_rotate_logfile|pg_promote|pg_switch_wal|pg_create_\w+|pg_drop_\w+|pg_advisory\w*|txid_\w+|"
    r"current_setting|pg_file_\w+|pg_logdir_ls|copy|execute|do)\s*\(",
    re.I,
)
# The build's own helper functions in the ETL schema ({etl}.omop_count, omop_drop, ...): they keep
# the counts of values the build could not load as written, which spec SQL must not change.
_INTERNAL_FUNCTIONS = re.compile(r"(?<![\w$])(omop_\w+)\s*\(", re.I)
# Catalog schemas (also when quoted), and statement keywords that can never appear in a single
# expression. A word right after a '.' is a column (alias.update), not a keyword.
_DENIED_SCHEMAS = re.compile(r"(?<![.\w])(pg_catalog|information_schema|pg_temp\w*|pg_toast)\b", re.I)
_DENIED_WORDS = re.compile(
    r"(?<![.\w])(insert|update|delete|truncate|drop|alter|create|grant|revoke|copy|vacuum)\b",
    re.I,
)
_QUOTED_NAME = re.compile(r'"((?:[^"]|"")*)"')
# The U& prefix of a Unicode-escaped identifier or string (U&"\006Fmop_count"), whose escapes can
# spell any name: such names are refused rather than decoded.
_UNICODE_PREFIX = re.compile(r"(?:^|[^\w$])U&$", re.I)


def strip_literals(sql: str) -> str:
    return _STRIP.sub(" ", sql)


def _unquote_names(sql: str) -> str:
    """``sql`` with string literals and comments removed and quoted identifiers unquoted, so a
    quoted name ("pg_catalog"."set_config"(...)) is seen as the schema or function it names."""
    kept = _STRIP.sub(lambda m: m.group(0) if m.group(0).startswith('"') else " ", sql)
    return _QUOTED_NAME.sub(lambda m: m.group(1).replace('""', '"'), kept)


def _unicode_escapes(sql: str) -> bool:
    return any(
        m.group(0)[0] in "'\"" and _UNICODE_PREFIX.search(sql[max(0, m.start() - 3) : m.start()])
        for m in _STRIP.finditer(sql)
    )


def fragment_issues(fragment: str | None) -> list[str]:
    """Problems that make an LLM-written SQL fragment unsafe to execute (empty when fine)."""
    if not fragment:
        return []
    bare = strip_literals(fragment)
    calls = _unquote_names(fragment)
    issues = []
    if ";" in bare:
        issues.append("contains ';' (only a single expression is allowed)")
    for pattern, text, template in (
        (_DENIED_FUNCTIONS, calls, "calls a function that is not allowed in ETL SQL: {}()"),
        (_INTERNAL_FUNCTIONS, calls, "calls an internal function of the build, which ETL SQL may not call: {}()"),
        (_DENIED_SCHEMAS, calls, "uses a keyword or schema that is not allowed in ETL SQL: {}"),
        (_DENIED_WORDS, bare, "uses a keyword or schema that is not allowed in ETL SQL: {}"),
    ):
        seen: set[str] = set()
        for m in pattern.finditer(text):
            if m.group(1).lower() not in seen:
                seen.add(m.group(1).lower())
                issues.append(template.format(m.group(1)))
    if "$" in fragment and re.search(r"\$\w*\$", fragment):
        issues.append("dollar-quoted strings are not allowed")
    if _unicode_escapes(fragment):
        issues.append("Unicode-escaped names and strings (U&\"...\", U&'...') are not allowed")
    return issues
