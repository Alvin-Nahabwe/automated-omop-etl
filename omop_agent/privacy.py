"""Privacy guard: decides what source information may be shown to the LLM.

Rules (enforced by construction, not by prompt):
  * the LLM only ever sees aggregate metadata - table/column names, types, counts, ranges,
    and value frequencies; never a patient row;
  * values seen fewer than ``min_cell_count`` times are suppressed everywhere they could reach
    the LLM (profile top values, vocabulary terms, repair-agent query results);
  * columns detected as direct identifiers (names, phone numbers, national IDs, addresses,
    GPS) get no values, no min/max, only counts;
  * ``send_values_to_llm: false`` removes all values and keeps only statistics.
"""

from __future__ import annotations

import re
from typing import Iterable

from .config import PrivacyConfig
from .models import ColumnProfile, SourceProfile, TableProfile, ValueFreq

_PHONE = re.compile(r"^\+?\d[\d\s-]{8,14}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def detect_pii(table: str, column: str, sample_values: Iterable[str], cfg: PrivacyConfig) -> str | None:
    """Return a reason string if the column looks like a direct identifier, else None."""
    qualified = f"{table}.{column}".lower()
    if qualified in {c.lower() for c in cfg.pii_columns}:
        return "listed in privacy.pii_columns"
    col = column.lower()
    for pat in cfg.pii_name_patterns:
        if re.search(pat, col):
            return f"column name matches /{pat}/"
    values = [v for v in sample_values if v]
    if len(values) >= 5:
        if sum(bool(_EMAIL.match(v)) for v in values) / len(values) > 0.5:
            return "values look like e-mail addresses"
        if sum(bool(_PHONE.match(v)) for v in values) / len(values) > 0.5 and not all(v.isdigit() and len(v) < 9 for v in values):
            return "values look like phone numbers"
    return None


def suppress_small_cells(values: list[ValueFreq], min_cell_count: int) -> tuple[list[ValueFreq], int]:
    kept = [v for v in values if v.count >= min_cell_count]
    return kept, len(values) - len(kept)


def redact_column(col: ColumnProfile) -> ColumnProfile:
    """Strip every value-bearing statistic from a column profile."""
    return col.model_copy(
        update={"top_values": [], "min_value": None, "max_value": None, "mean": None, "n_suppressed_values": 0}
    )


def llm_view(profile: SourceProfile, cfg: PrivacyConfig) -> SourceProfile:
    """The version of the profile that is allowed into a prompt."""
    tables: list[TableProfile] = []
    for t in profile.tables:
        cols = []
        for c in t.columns:
            if c.is_pii or not cfg.send_values_to_llm:
                cols.append(redact_column(c))
                continue
            kept, n_dropped = suppress_small_cells(c.top_values, profile.min_cell_count)
            cols.append(
                c.model_copy(
                    update={
                        "top_values": kept[: cfg.max_values_per_column],
                        "n_suppressed_values": c.n_suppressed_values + n_dropped,
                    }
                )
            )
        tables.append(t.model_copy(update={"columns": cols}))
    return profile.model_copy(update={"tables": tables})


def profile_prompt_text(profile: SourceProfile, cfg: PrivacyConfig, tables: list[str] | None = None) -> str:
    """Compact, privacy-filtered text rendering of the profile for prompts."""
    view = llm_view(profile, cfg)
    lines: list[str] = []
    for t in view.tables:
        if tables is not None and t.name not in tables:
            continue
        lines.append(f"SOURCE TABLE {t.name} ({t.n_rows} rows)")
        for c in t.columns:
            bits = [c.inferred_type]
            if c.declared_type:
                bits.append(f"declared {c.declared_type}")
            null_pct = 100.0 * c.n_null / c.n_rows if c.n_rows else 0.0
            bits.append(f"{null_pct:.0f}% null")
            bits.append(f"{c.n_distinct} distinct")
            if c.unique:
                bits.append("unique")
            if c.date_format:
                bits.append(f"format {c.date_format}")
            if c.min_value is not None or c.max_value is not None:
                bits.append(f"range [{c.min_value} .. {c.max_value}]")
            if c.is_pii:
                bits.append(f"PII - values withheld ({c.pii_reason})")
            line = f"  - {c.name}: " + ", ".join(bits)
            if c.top_values:
                vals = "; ".join(f"{v.value!r}×{v.count}" for v in c.top_values)
                line += f"\n      top values: {vals}"
                if c.n_suppressed_values:
                    line += f" (+{c.n_suppressed_values} rare values withheld)"
            lines.append(line)
        lines.append("")
    return "\n".join(lines)
