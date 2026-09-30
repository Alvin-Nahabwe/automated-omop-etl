"""Static safety checks for SQL fragments written by the LLM (spec expressions, joins, filters).

Spec SQL is executed by the build, so a fragment must stay a single expression over the staged
source and vocabulary: no statement separators, no access to server files, other databases or
session settings, and no catalog snooping. This complements (it does not replace) running the
build under a least-privilege role (``target.etl_role``).
"""

from __future__ import annotations

import re

# Literals ('...', E'...', $tag$...$tag$), quoted identifiers ("...") and comments are removed
# before scanning, so a column called "comment" or a string containing ';' is not a problem.
_STRIP = re.compile(
    r"(?P<dollar>\$(?P<tag>[A-Za-z_]*)\$.*?\$(?P=tag)\$)"
    r"|(?:[Ee])?'(?:[^'\\]|\\.|'')*'"
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
# Statement keywords that can never appear in a single expression, and catalog schemas. A word
# right after a '.' is a column (alias.update), not a keyword.
_DENIED_WORDS = re.compile(
    r"(?<![.\w])(pg_catalog|information_schema|pg_temp\w*|pg_toast|insert|update|delete|truncate|drop|alter|"
    r"create|grant|revoke|copy|vacuum)\b",
    re.I,
)


def strip_literals(sql: str) -> str:
    return _STRIP.sub(" ", sql)


def fragment_issues(fragment: str | None) -> list[str]:
    """Problems that make an LLM-written SQL fragment unsafe to execute (empty when fine)."""
    if not fragment:
        return []
    bare = strip_literals(fragment)
    issues = []
    if ";" in bare:
        issues.append("contains ';' (only a single expression is allowed)")
    for pattern, template in (
        (_DENIED_FUNCTIONS, "calls a function that is not allowed in ETL SQL: {}()"),
        (_DENIED_WORDS, "uses a keyword or schema that is not allowed in ETL SQL: {}"),
    ):
        seen: set[str] = set()
        for m in pattern.finditer(bare):
            if m.group(1).lower() not in seen:
                seen.add(m.group(1).lower())
                issues.append(template.format(m.group(1)))
    if "$" in fragment and re.search(r"\$\w*\$", fragment):
        issues.append("dollar-quoted strings are not allowed")
    return issues
