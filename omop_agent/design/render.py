"""Render an ``ETLSpec`` as a Rabbit-in-a-Hat style markdown document for review."""

from __future__ import annotations

from ..cdm.spec import DERIVED_ERAS, LOAD_ORDER
from ..etl.compiler import ERA_PAD_DAYS, era_status, key_sources
from ..models import ETLSpec, FieldMapping, TableMapping

# How each era table is built (``etl.compiler.era_status``), for the review views.
_ERA_HOW = {
    "derived": "derived by the build from {source} (OHDSI era logic)",
    "spec": "loaded by the spec's own SQL",
    "off": "not built (derive_eras: false)",
    "no_records": "not built (no {source} records are loaded)",
}


def eras_text(spec: ETLSpec) -> str:
    """One line on how the spec's era tables are built, e.g. 'condition_era derived by the build
    from condition_occurrence (OHDSI era logic); drug_era ...', for the spec page, the spec
    document and the report."""
    status = era_status(spec)
    parts = [f"{era} {_ERA_HOW[how].format(source=DERIVED_ERAS[era])}"
             + (" (it replaces the build's era step)" if how == "spec" and spec.derive_eras else "")
             for era, how in status.items()]
    derived = [era for era, how in status.items() if how == "derived"]
    rule = (f". An era joins the records of one person and concept (for drug eras: ingredient of RxNorm or RxNorm "
            f"Extension) that start at most {ERA_PAD_DAYS} days after the earlier ones ended" if derived else "")
    return "; ".join(parts) + rule


def _cell(text: str | None) -> str:
    """Make text safe for a markdown table cell."""
    if not text:
        return ""
    return " ".join(str(text).split()).replace("|", "\\|")


def _code(text: str | None) -> str:
    if not text:
        return ""
    t = _cell(text)
    return f"`{t}`" if "`" not in t else t


def _source_cell(fm: FieldMapping) -> str:
    if fm.kind == "auto_id":
        return "*generated surrogate key*"
    if fm.kind == "null":
        return "*NULL*"
    return _code(fm.expression)


def _vocab_cell(fm: FieldMapping) -> str:
    if fm.kind == "vocab_code":
        return _cell(f"{fm.vocabulary or '?'} ({fm.concept_role})")
    if fm.kind == "concept_map":
        dom = f" → {fm.target_domain}" if fm.target_domain else ""
        return _cell(f"{fm.vocabulary or '?'}{dom}")
    return ""


def _mapping_md(tm: TableMapping, n: int | None) -> list[str]:
    title = f"### {tm.target_table}" + (f" ({n})" if n else "") + f" ← `{tm.source.table}` ({tm.source.alias})"
    out = [title, ""]
    if tm.rationale:
        out += [tm.rationale.strip(), ""]
    if tm.joins:
        out.append("Joins:")
        out.append("")
        for j in tm.joins:
            out.append(f"- {j.join_type.upper()} JOIN `{j.table}` {j.alias} ON `{' '.join(j.on.split())}`")
        out.append("")
    if tm.where:
        out += [f"Filter: `{' '.join(tm.where.split())}`", ""]
    if (tm.dedup_order or "").strip():
        out += [f"One row per key, best row first by: `{' '.join((tm.dedup_order or '').split())}`", ""]
    out.append("| CDM field | kind | source / expression | vocabulary | rationale |")
    out.append("|---|---|---|---|---|")
    for fm in tm.fields:
        out.append(
            f"| {_cell(fm.target_field)} | {fm.kind} | {_source_cell(fm)} | {_vocab_cell(fm)} | {_cell(fm.rationale)} |"
        )
    out.append("")
    return out


def _also(spec: ETLSpec, table: str) -> str:
    """The other mappings whose keys go into the id map (``compiler.key_sources``)."""
    sources = key_sources(spec, table)
    if not sources:
        return ""
    return "; also from " + ", ".join(f"`{tm.source.table}`.`{' '.join(expr.split())}`" for tm, expr, _ in sources)


def render_spec_md(spec: ETLSpec) -> str:
    """Markdown overview + one field table per table mapping."""
    order = {t: i for i, t in enumerate(LOAD_ORDER)}
    mappings = sorted(
        enumerate(spec.table_mappings), key=lambda it: (order.get(it[1].target_table, len(order)), it[0])
    )
    counts: dict[str, int] = {}
    for tm in spec.table_mappings:
        counts[tm.target_table] = counts.get(tm.target_table, 0) + 1

    out = ["# ETL specification", ""]
    out.append(
        "Generated from `etl_spec.yaml` (the editable source of truth). Each table mapping compiles "
        "to one `INSERT ... SELECT`; see `sql/` for the generated SQL."
    )
    out += ["", "## Overview", ""]
    out.append(f"- **CDM version:** {spec.cdm_version}")
    out.append(f"- **Person key:** `{spec.person_key.table}`.`{' '.join(spec.person_key.expression.split())}`"
               + _also(spec, "person"))
    if spec.visit_key:
        out.append(f"- **Visit key:** `{spec.visit_key.table}`.`{' '.join(spec.visit_key.expression.split())}`"
                   + _also(spec, "visit_occurrence"))
    else:
        out.append("- **Visit key:** none (records are not linked to visits)")
    if spec.observation_period == "from_events":
        out.append(
            f"- **Observation period:** derived from events (first to last event date per person), "
            f"type concept {spec.observation_period_type_concept_id}"
        )
    else:
        out.append(
            "- **Observation period:** loaded by a table mapping, one row per episode. The build merges each "
            "person's overlapping and back-to-back episodes, ends open ones (no end date) at "
            "`cdm_source.source_release_date` (configured, or derived from the exit dates) or at death, drops "
            "episodes that end before they start, and gives persons without a valid episode a period from their "
            "first to their last event"
        )
    if spec.route_by_domain:
        out.append(
            "- **Routing by domain:** after loading, every condition_occurrence, procedure_occurrence, drug_exposure, "
            "device_exposure and measurement record whose standard concept belongs to another event domain moves to "
            "that domain's table (a measurement with a result goes to observation); records with concept 0 stay"
        )
    else:
        out.append("- **Routing by domain:** off (`route_by_domain: false`): records stay in the table that loads them")
    out.append(f"- **Eras:** {eras_text(spec)}")
    out += ["", "| # | source table | → | CDM table | fields |", "|---|---|---|---|---|"]
    for k, (_, tm) in enumerate(mappings, start=1):
        src = tm.source.table + "".join(f" + {j.table}" for j in tm.joins)
        out.append(f"| {k} | {_cell(src)} | → | {tm.target_table} | {len(tm.fields)} |")
    for cs in spec.custom_sql:
        out.append(f"| | custom SQL `{_cell(cs.name)}` | → | {cs.target_table} | |")
    out.append("")

    out += ["## Table mappings", ""]
    seen: dict[str, int] = {}
    for _, tm in mappings:
        seen[tm.target_table] = seen.get(tm.target_table, 0) + 1
        n = seen[tm.target_table] if counts[tm.target_table] > 1 else None
        out += _mapping_md(tm, n)

    if spec.custom_sql:
        out += ["## Custom SQL", ""]
        for cs in spec.custom_sql:
            out += [f"### {cs.name} → {cs.target_table}", ""]
            if cs.rationale:
                out += [cs.rationale.strip(), ""]
            out += ["```sql", cs.sql.strip(), "```", ""]

    out += ["## Assumptions", ""]
    out += [f"- {a}" for a in spec.assumptions] or ["- none recorded"]
    out += ["", "## Open questions", ""]
    out += [f"- {q}" for q in spec.open_questions] or ["- none recorded"]
    out.append("")
    return "\n".join(out)
