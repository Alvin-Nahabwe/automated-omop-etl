"""Render the source profile as a White-Rabbit-style scan report in Markdown (``profile.md``)."""

from __future__ import annotations

from ..models import ColumnProfile, SourceProfile, ValueFreq
from ..privacy import small_count, suppress_small_cells

TOP_VALUES_SHOWN = 8
_MAX_VALUE_CHARS = 40


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _value(v: str) -> str:
    v = v if len(v) <= _MAX_VALUE_CHARS else v[: _MAX_VALUE_CHARS - 1] + "…"
    return f"`{_cell(v)}`" if "`" not in v else _cell(v)


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "–"


def _range(c: ColumnProfile) -> str:
    if c.is_pii:
        return ""
    if c.min_value is not None or c.max_value is not None:
        text = f"{_value(c.min_value or '')} .. {_value(c.max_value or '')}"
        if c.mean is not None:
            text += f" (mean {c.mean:g})"
        return text
    if c.inferred_type == "string" and c.min_length is not None:
        if c.min_length == c.max_length:
            return f"length {c.min_length}"
        return f"length {c.min_length}–{c.max_length}"
    return ""


def _top(c: ColumnProfile, min_cell_count: int) -> str:
    if c.is_pii:
        return "*withheld*"
    # Defensive: never print a small cell, even from a hand-edited profile.
    kept, dropped = suppress_small_cells(list(c.top_values), min_cell_count)
    shown: list[ValueFreq] = kept[:TOP_VALUES_SHOWN]
    parts = [f"{_value(v.value)}×{v.count}" for v in shown]
    if len(kept) > len(shown):
        parts.append(f"+{len(kept) - len(shown)} more listed")
    hidden = c.n_suppressed_values + dropped
    if hidden:
        parts.append(f"{hidden} not shown")
    return ", ".join(parts)


def _freqs(values: list[ValueFreq], k: int) -> str:
    kept, _ = suppress_small_cells(list(values), k)  # defensive, as for top values
    return ", ".join(f"{_value(v.value)}×{v.count}" for v in kept)


def _rare_offsets(c: ColumnProfile, k: int) -> str:
    """The values whose offset is too rare to name (counted with the listed offsets a stricter
    threshold hides)."""
    hidden = sum(v.count for v in c.tz_offsets) - sum(v.count for v in suppress_small_cells(list(c.tz_offsets), k)[0])
    n = c.n_rare_offsets + hidden
    return f"{small_count(n, k)} with rarer offsets" if n else ""


def _reading(c: ColumnProfile, k: int) -> list[str] | None:
    """One row of the "Reading the values" table, or None when there is nothing to say."""
    if c.is_pii:
        return None
    cells = [
        _freqs(c.value_shapes, k),
        small_count(c.n_nonconforming, k) if c.n_nonconforming else "",
        _freqs(c.nonconforming_shapes, k),
        small_count(c.n_partial_dates, k) if c.n_partial_dates else "",
        ", ".join(x for x in [_freqs(c.tz_offsets, k), _rare_offsets(c, k)] if x),
        small_count(c.n_na_markers, k) if c.n_na_markers else "",
        _cell("; ".join(c.format_notes)),
    ]
    return cells if any(cells) else None


def render_profile_md(profile: SourceProfile) -> str:
    """Scan report: an overview table, then one column table per source table."""
    k = profile.min_cell_count
    n_pii = sum(c.is_pii for t in profile.tables for c in t.columns)
    lines = [
        "# Source data profile",
        "",
        f"Scan of {len(profile.tables)} {profile.source_type} source tables, generated "
        f"{profile.generated_at:%Y-%m-%d %H:%M} UTC. Every figure is an aggregate over the staged copy.",
        "",
        f"> **Small-cell rule:** values seen fewer than {k} times are never listed; the *not shown* "
        f"count says how many distinct values were left out (rare values, or beyond the top list). "
        f"Text min/max are shown only when both values occur at least {k} times. "
        f"Columns flagged **PII** ({n_pii} here) are direct identifiers: they show counts only, "
        "never values, ranges, means or shapes. A shape writes a value with 9 for each digit and a / A for "
        f"each letter; shapes and offsets follow the same rule, and counts below {k} are shown as <{k}.",
        "",
        "## Overview",
        "",
        "| Table | Rows | Columns | PII columns |",
        "|---|---:|---:|---:|",
    ]
    for t in profile.tables:
        lines.append(f"| {_cell(t.name)} | {t.n_rows:,} | {len(t.columns)} | {sum(c.is_pii for c in t.columns)} |")
    for t in profile.tables:
        lines += [
            "",
            f"## {t.name}",
            "",
            f"{t.n_rows:,} rows, {len(t.columns)} columns.",
            "",
            "| Column | Type | Null % | Distinct | Unique | Range | Date format | Top values | PII |",
            "|---|---|---:|---:|---|---|---|---|---|",
        ]
        for c in t.columns:
            typ = c.inferred_type + (f" ({_cell(c.declared_type)})" if c.declared_type else "")
            pii = f"**PII**: {_cell(c.pii_reason or 'flagged')}" if c.is_pii else ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(c.name),
                        typ,
                        _pct(c.n_null, c.n_rows),
                        f"{c.n_distinct:,}",
                        "yes" if c.unique else "",
                        _range(c),
                        f"`{c.date_format}`" if c.date_format else "",
                        _top(c, k),
                        pii,
                    ]
                )
                + " |"
            )
        reading = [(c, row) for c in t.columns if (row := _reading(c, k)) is not None]
        if reading:
            lines += [
                "",
                "Reading the values (whole column; ISO dates fit every date format, partial dates have an unknown "
                "day or month or no day):",
                "",
                "| Column | Shapes | Do not fit | Their shapes | Partial dates | Offsets | NA markers | Notes |",
                "|---|---|---:|---|---:|---|---:|---|",
            ]
            lines += [f"| {_cell(c.name)} | " + " | ".join(row) + " |" for c, row in reading]
    return "\n".join(lines) + "\n"
