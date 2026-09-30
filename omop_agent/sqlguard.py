"""Static safety checks for SQL fragments written by the LLM (spec expressions, joins, filters).

Spec SQL is executed by the build, so a fragment must stay a single expression over the staged
source and vocabulary: no statement separators, no access to server files, other databases or
session settings, and no catalog snooping. This complements (it does not replace) running the
build under a least-privilege role (``target.etl_role``).
"""

from __future__ import annotations

import re

# Literals ('...', E'...', $tag$...$tag$), quoted identifiers and comments, removed before scanning.
_STRIP = re.compile(
    r"(?P<dollar>\$(?P<tag>[A-Za-z_]*)\$.*?\$(?P=tag)\$)"
    r"|(?:[Ee])?'(?:[^'\\]|\\.|'')*'"
    r"|--[^\n]*"
    r"|/\*.*?\*/",
    re.S,
)
_DENIED_FUNCTIONS = re.compile(
    r"\b(pg_read_file|pg_read_binary_file|pg_ls_\w+|pg_stat_file|lo_import|lo_export|lo_get|lo_put|"
    r"dblink\w*|query_to_xml\w*|table_to_xml\w*|schema_to_xml\w*|database_to_xml\w*|cursor_to_xml\w*|"
    r"query_to_xmlschema|pg_sleep\w*|set_config|pg_terminate_backend|pg_cancel_backend|pg_reload_conf|"
    r"pg_rotate_logfile|pg_promote|pg_switch_wal|pg_create_\w+|pg_drop_\w+|pg_advisory\w*|txid_\w+|"
    r"current_setting|pg_file_\w+|pg_logdir_ls|copy|execute|do)\s*\(",
    re.I,
)
_DENIED_WORDS = re.compile(
    r"\b(pg_catalog|information_schema|pg_temp\w*|pg_toast|insert|update|delete|truncate|drop|alter|create|"
    r"grant|revoke|copy|vacuum|analyze|listen|notify|prepare|deallocate|lock|comment|security|reset)\b",
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
    if m := _DENIED_FUNCTIONS.search(bare):
        issues.append(f"calls a function that is not allowed in ETL SQL: {m.group(1)}()")
    if m := _DENIED_WORDS.search(bare):
        issues.append(f"uses a keyword or schema that is not allowed in ETL SQL: {m.group(1)}")
    if "$" in fragment and re.search(r"\$\w*\$", fragment):
        issues.append("dollar-quoted strings are not allowed")
    return issues
