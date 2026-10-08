"""Inspection report: one Markdown and one self-contained HTML page per project run.

The report reads every artifact in the workspace. Each one is optional: a missing artifact
renders as "not run", and one that cannot be parsed is listed under "unreadable artifacts".
The overall verdict comes only from recorded results (``dq_results.json``, ``etl_run.json``,
``state.json``), never from a model.

When a database connection is given, the loaded CDM is also characterised (an Achilles-lite
summary). Every count in that summary below ``min_cell_count`` is shown as ``<N`` and its
exact value is never stored.

The report lists source codes and terms, rare ones included, because reviewers need them to
check mappings; it says so at the top. ``build_report(..., shareable=True)`` writes a copy
(``report.shareable.md`` / ``.html``) in which every source term seen fewer than
``min_cell_count`` times is replaced by ``<withheld>``.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Callable, TypeVar

import jinja2
import psycopg
import yaml
from markupsafe import Markup, escape
from pydantic import BaseModel

from ..cdm.spec import DERIVED_ERAS, LOAD_ORDER, load_cdm_spec
from ..config import ProjectConfig
from ..db import Schemas, fetch_all, qi, short_error
from ..design.render import build_rules, eras_text
from ..etl.compiler import derived_era_tables, era_status, key_sources, step_label
from ..etl.policies import ROW_SITE, WHERE_SITE
from ..etl.routing import ROUTE_PREPARE
from ..privacy import apply_overrides, small_count, suppress_small_cells
from ..profiling.render import rare_offsets
from ..models import (
    CheckResult,
    ColumnProfile,
    DQReport,
    ETLRunResult,
    ETLSpec,
    MappingDecision,
    RepairAction,
    SourceAnnotations,
    SourceProfile,
    TableLoadResult,
    ValueFreq,
)
from ..workspace import STAGES, Workspace, utcnow

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

TEMPLATE_PKG = "omop_agent.report"
PASSED, FAILED, INCOMPLETE = "PASSED", "FAILED", "INCOMPLETE"

# Event table -> the concept field its "top concepts" are counted on.
TOP_CONCEPT_FIELDS = {
    "condition_occurrence": "condition_concept_id",
    "drug_exposure": "drug_concept_id",
    "measurement": "measurement_concept_id",
    "observation": "observation_concept_id",
    "visit_occurrence": "visit_concept_id",
}
PIPELINE_STAGES = ["profile", "annotate", "vocab_load", "design", "vocab_map", "build", "quality", "report"]
FAILED_STATUSES = {"failed", "error", "errored"}
UNFINISHED_STATUSES = {"running", "awaiting_review", "pending"}
OK_STATUSES = {"done", "passed", "ok", "success", "succeeded", "complete", "completed", "approved"}
SEVERITIES = ["fatal", "convention", "characterization"]
KAHN_CATEGORIES = ["Conformance", "Completeness", "Plausibility"]
CHECK_STATUSES = ["passed", "failed", "error", "not_applicable"]
FIELD_KINDS = [
    "person_id", "visit_id", "auto_id", "column", "expression", "constant", "vocab_code", "concept_map", "null"
]
# Values not loaded as written (TableLoadResult.value_counts), in the order the report shows them.
VALUE_CATEGORIES = [
    "invalid", "imputed", "partial_null", "defaulted", "required_null", "offset_ignored", "offset_converted",
]
MAX_REVIEW_ROWS = 100
MAX_UNMATCHED_SHOWN = 5
WITHHELD = "<withheld>"
CONFIDENTIAL_NOTICE = (
    "This report contains source codes and terms, including rare ones, needed for mapping review. "
    "Share it only within your data-governance boundary."
)
SHAREABLE_NOTICE = (
    "Source codes and terms seen fewer than {n} times are replaced by \"<withheld>\". Frequent source codes "
    "and terms are still listed. Some tables show counts below {n} as \"<{n}\", but totals, differences and the "
    "data-quality counts are exact, so a small count can often be worked out from them. Check the report against "
    "your data-sharing rules before sharing it."
)


# --------------------------------------------------------------------------------------
# Formatting (also registered as template filters)
# --------------------------------------------------------------------------------------


def fmt_int(n: Any) -> str:
    if n is None or n == "":
        return "—"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def fmt_pct(x: float | None, digits: int = 1) -> str:
    """Percentage that never rounds a partial value to 0% or 100%."""
    if x is None:
        return "—"
    step = 10.0**-digits
    if 0 < x < step / 2:
        return f"<{step:.{digits}f}%"
    if 100 - step / 2 <= x < 100:
        return f">{100 - step:.{digits}f}%"
    return f"{x:.{digits}f}%"


def fmt_secs(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m} min {sec} s" if m < 60 else f"{m // 60} h {m % 60} min"


def _as_utc(v: Any) -> datetime | None:
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(v, datetime):
        return None
    return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


def fmt_dt(v: Any) -> str:
    dt = _as_utc(v)
    if dt is None:
        return "" if v in (None, "") else str(v)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def md_escape(v: Any) -> str:
    """Make a value safe inside a Markdown table cell or list item."""
    if v is None:
        return ""
    s = " ".join(str(v).split())
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def inline_code(v: Any) -> Markup:
    """HTML-escape ``v`` and render its `backtick` spans as <code>."""
    return Markup(re.sub(r"`([^`]+)`", r"<code>\1</code>", str(escape(v))))


def text_bar(width_pct: float | None, size: int = 20) -> str:
    return "█" * int(round((width_pct or 0) * size / 100))


def status_class(status: Any) -> str:
    s = str(status or "").lower()
    if s in OK_STATUSES or s == PASSED.lower():
        return "ok"
    if s in FAILED_STATUSES or s == FAILED.lower():
        return "fail"
    if s in UNFINISHED_STATUSES or s == INCOMPLETE.lower() or s in {"unchecked", "flagged", "skipped"}:
        return "warn"
    return "muted"


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return 100.0 * num / den


def _int(v: Any) -> int | None:
    try:
        return None if v is None or isinstance(v, bool) else int(v)
    except (TypeError, ValueError):
        return None


def _short(v: Any, limit: int = 200) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _kv(d: dict[str, Any]) -> str:
    return "; ".join(f"{k}={_short(v)}" for k, v in d.items() if v not in (None, "", [], {}))


def small_cell(n: int | None, min_cell_count: int) -> dict[str, Any]:
    """A count for display: values in (0, min_cell_count) become ``<N`` and are not kept."""
    if n is None:
        return {"count": None, "display": "—", "suppressed": False}
    if 0 < n < min_cell_count:
        return {"count": None, "display": f"<{min_cell_count}", "suppressed": True}
    return {"count": int(n), "display": fmt_int(n), "suppressed": False}


def _add_bars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set ``bar`` (0-100, relative to the largest shown count). Suppressed cells get no bar."""
    top = max((r["count"] for r in rows if r.get("count")), default=0)
    for r in rows:
        r["bar"] = round(100.0 * r["count"] / top, 1) if top and r.get("count") else 0.0
    return rows


# --------------------------------------------------------------------------------------
# Artifact loading (tolerant: a broken artifact is reported, not fatal)
# --------------------------------------------------------------------------------------


def _load(path: Path, parse: Callable[[str], T], problems: list[str]) -> T | None:
    if not path.exists():
        return None
    try:
        return parse(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        msg = " ".join(str(e).split())
        problems.append(f"{path.name}: {type(e).__name__}: {msg[:300]}")
        return None


def _model_list(cls: type[M]) -> Callable[[str], list[M]]:
    def parse(text: str) -> list[M]:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("expected a JSON list")
        return [cls.model_validate(x) for x in data]

    return parse


def _json_object(text: str) -> dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _spec(text: str) -> ETLSpec:
    return ETLSpec.model_validate(yaml.safe_load(text))


def _read_state(ws: Workspace, problems: list[str]) -> dict[str, dict[str, Any]]:
    st = _load(ws.state_json, _json_object, problems) or {}
    stages = st.get("stages")
    return {k: v for k, v in stages.items() if isinstance(v, dict)} if isinstance(stages, dict) else {}


def _read_audit(ws: Workspace, problems: list[str]) -> list[dict[str, Any]]:
    if not ws.audit_log.exists():
        return []
    events: list[dict[str, Any]] = []
    bad = 0
    for line in ws.audit_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(rec, dict):
            events.append(rec)
        else:
            bad += 1
    if bad:
        problems.append(f"{ws.audit_log.name}: {bad} unreadable line(s) skipped")
    return events


def _dicts(v: Any) -> list[dict[str, Any]]:
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def _strs(v: Any) -> list[str]:
    return [str(x) for x in v] if isinstance(v, list) else []


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def _verdict(
    dq: DQReport | None, etl_run: ETLRunResult | None, stages: dict[str, dict[str, Any]]
) -> tuple[str, list[str], list[str]]:
    """The verdict, its reasons, and the steps that failed without failing the build (route and era
    steps are not blocking, see ``_nonblocking_notes``), which are listed apart whatever the verdict."""
    failures: list[str] = []
    notes = _nonblocking_notes(etl_run)
    if etl_run is not None and etl_run.blocking_errors():
        errs = etl_run.blocking_errors()
        failures.append(f"{len(errs)} ETL step(s) errored: " + ", ".join(s.step for s in errs[:5]))
    if dq is not None and not dq.passed:
        n = len(dq.blocking_failures)
        failures.append(f"data-quality gate failed ({n} blocking problem(s))" if n else "data-quality gate failed")
    for name, info in stages.items():
        if name != "report" and str(info.get("status", "")).lower() in FAILED_STATUSES:
            failures.append(f"stage '{name}' failed" + (f": {_short(info['error'], 160)}" if info.get("error") else ""))
    if failures:
        return FAILED, failures, notes
    unfinished = [
        f"stage '{n}' is {i.get('status')}"
        for n, i in stages.items()
        if n != "report" and str(i.get("status", "")).lower() in UNFINISHED_STATUSES
    ]
    if dq is None:
        reasons = (["the ETL has not run"] if etl_run is None else []) + ["data-quality checks have not run"]
        return INCOMPLETE, reasons + unfinished, notes
    if unfinished:
        return INCOMPLETE, unfinished, notes
    dq_at, etl_at = _as_utc(dq.run_at), _as_utc(etl_run.started_at) if etl_run else None
    if dq_at and etl_at and dq_at < etl_at:
        return INCOMPLETE, ["the data-quality results predate the latest ETL run; re-run the checks"], notes
    reasons = ["no blocking data-quality check failed"]
    if etl_run is not None:
        reasons.append(_steps_completed(etl_run))
    return PASSED, reasons, notes


def _steps_completed(etl_run: ETLRunResult) -> str:
    """How many ETL steps of a run without blocking errors completed, failed and were skipped (a
    step that requires a failed one is skipped: a route step after route_prepare)."""
    n, k = len(etl_run.steps), len(etl_run.errors())
    m = sum(1 for s in etl_run.steps if s.skipped and not s.error)
    if not k and not m:
        return f"all {n} ETL steps completed without error"
    done = f"{n - k - m} of {n} ETL steps completed without error"
    if not m:
        return done + ("; the one that failed does not block the build" if k == 1 else
                       f"; the {k} that failed do not block the build")
    return (done + f"; {k} failed and {m} {'was' if m == 1 else 'were'} skipped because of "
            + ("it" if k == 1 else "them") + ", and none of these steps blocks the build")


def _nonblocking_notes(etl_run: ETLRunResult | None) -> list[str]:
    """One line per step that failed without failing the build (``blocking=False``: the route and
    era steps), saying what the CDM lacks because of it. The error itself is in the ETL run table."""
    notes = []
    for s in etl_run.errors() if etl_run is not None else []:
        if s.blocking:
            continue
        label = step_label(s.step)
        if label == ROUTE_PREPARE:
            effect = "no records were moved to the table of their concept's domain"
        elif s.kind == "route":
            effect = f"the {s.target_table} records whose concept belongs to another domain stay where they were loaded"
        elif s.target_table in DERIVED_ERAS:
            effect = f"{s.target_table} was not built"
        else:
            effect = f"{s.target_table} may be incomplete"
        notes.append(f"{s.step} failed without failing the build: {effect} (see the ETL run)")
    return notes


def _relative(v: Any, root: Path) -> Any:
    """Shorten absolute paths inside the workspace to workspace-relative ones."""
    if isinstance(v, str):
        for prefix in {str(root).rstrip("/") + "/", str(root.resolve()).rstrip("/") + "/"}:
            if v.startswith(prefix):
                return v[len(prefix):]
        return v
    if isinstance(v, list):
        return [_relative(x, root) for x in v]
    return v


def _timeline(stages: dict[str, dict[str, Any]], audit: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    known = PIPELINE_STAGES + [s for s in STAGES if s not in PIPELINE_STAGES]
    if "privacy" in stages:  # only projects that send data to Claude have this review step
        known.insert(known.index("profile") + 1, "privacy")
    extra = sorted((s for s in stages if s not in known), key=lambda s: str(stages[s].get("at", "")))
    rows = []
    for name in known + extra:
        info = stages.get(name)
        if info is None:
            rows.append({"stage": name, "status": "not run", "at": "", "details": ""})
            continue
        rest = {k: v for k, v in info.items() if k not in ("status", "at")}
        status = str(info.get("status", "unknown"))
        rows.append({"stage": name, "status": status, "at": fmt_dt(info.get("at")), "details": _kv(rest)})
    return {"stages": rows, "has_state": bool(stages)}


def review_caveats(cfg: ProjectConfig, ws: Workspace) -> list[str]:
    """What still needs a person's review before the CDM built in ``ws`` is used (see _reviews)."""
    problems: list[str] = []
    decisions = _load(ws.mappings_dir / "decisions.json", _model_list(MappingDecision), problems)
    repairs = _load(ws.repairs_json, _model_list(RepairAction), problems)
    return _reviews(cfg, ws, _read_audit(ws, problems), repairs, decisions)[1]


_PASS_HOW = {
    "approved": "approved",
    "skipped": "not checked (run with --yes)",
    "not_checked": "not checked (single stage command)",
    "stopped": "stopped for review",
}


def _reviews(
    cfg: ProjectConfig, ws: Workspace, audit: list[dict[str, Any]], repairs: list[RepairAction] | None,
    decisions: list[MappingDecision] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The review gates as they stand for the CDM just built, and the caveats that qualify a PASS:
    artifacts nobody approved, or that changed after approval (for example by automatic repairs),
    and unreviewed mappings that were loaded."""
    from ..approvals import GATES, gate_state
    from ..privacy_review import refresh, sends_to_llm

    sends = sends_to_llm(cfg)
    if sends:
        refresh(cfg, ws)
    last_pass: dict[str, str] = {}
    for e in audit:
        if e.get("event") == "review_gate" and e.get("stage"):
            last_pass[str(e["stage"])] = _PASS_HOW.get(str(e.get("outcome")), str(e.get("outcome")))
    applied = [a for a in repairs or [] if not a.reverted]
    rows: list[dict[str, Any]] = []
    caveats: list[str] = []
    for stage, label in GATES.items():
        if stage == "privacy" and not sends:
            rows.append({"gate": label, "state": "not needed", "by": "", "at": "", "note": "no data is sent to Claude",
                         "last_pass": ""})
            continue
        g = gate_state(ws, stage)
        a = g.approval
        rows.append({
            "gate": label,
            "state": g.label,
            "by": a.by if a else "",
            "at": fmt_dt(a.at) if a else "",
            "note": a.note if a else "",
            "last_pass": last_pass.get(stage, ""),
        })
        if not g.ready or g.current:
            continue
        if stage == "privacy":
            caveats.append("data was sent to Claude without an approved privacy review of the current profile")
        elif stage == "design":
            why = f": {len(applied)} automatic repair(s) changed it after review" if g.approval and applied else ""
            caveats.append(f"the ETL design this CDM was built from is {g.label}{why}")
        else:
            caveats.append(f"the code mappings used are {g.label}")
    if decisions is not None and cfg.mapping.use_unchecked_in_etl:
        n = sum(1 for d in decisions if d.status == "UNCHECKED" and d.target_concept_id)
        if n:
            caveats.append(f"{n} mapping(s) nobody reviewed (UNCHECKED) were loaded")
    return rows, caveats


def _freq_text(values: list[ValueFreq], k: int) -> str:
    kept, _ = suppress_small_cells(list(values), k)  # defensive: the profile lists none seen fewer than k times
    return ", ".join(f"{v.value} ×{fmt_int(v.count)}" for v in kept)


def _reading(table: str, c: ColumnProfile, k: int) -> dict[str, Any] | None:
    """How a column's values read, from the profile's scan of the whole column (as in profile.md):
    its date format, value shapes, the values that do not fit its type or format and their shapes,
    partial dates, time zone offsets and missing-value markers staged as NULL. Counts below ``k``
    are shown as ``<k``. None for a direct identifier, and for a column without a date format whose
    values all fit its type (its shapes alone are in profile.md)."""
    if c.is_pii:
        return None
    row = {
        "format": c.date_format or "",
        "shapes": _freq_text(c.value_shapes, k),
        "nonconforming": small_count(c.n_nonconforming, k) if c.n_nonconforming else "",
        "nonconforming_shapes": _freq_text(c.nonconforming_shapes, k),
        "partial": small_count(c.n_partial_dates, k) if c.n_partial_dates else "",
        "offsets": ", ".join(x for x in (_freq_text(c.tz_offsets, k), rare_offsets(c, k)) if x),
        "na_markers": small_count(c.n_na_markers, k) if c.n_na_markers else "",
        "notes": "; ".join(c.format_notes),
    }
    if not any(v for key, v in row.items() if key not in ("shapes", "nonconforming_shapes")):
        return None
    return {"table": table, "column": c.name, "type": c.inferred_type, **row}


def _source_section(profile: SourceProfile, annotations: SourceAnnotations | None) -> dict[str, Any]:
    ann = {t.table: t for t in annotations.tables} if annotations else {}
    k = profile.min_cell_count
    tables, pii, reading = [], [], []
    for t in profile.tables:
        a = ann.get(t.name)
        tables.append(
            {
                "name": t.name,
                "n_rows": t.n_rows,
                "n_columns": len(t.columns),
                "n_pii": sum(c.is_pii for c in t.columns),
                "description": a.description if a else "",
                "grain": a.grain if a else "",
            }
        )
        for c in t.columns:
            if c.is_pii:
                pii.append({"table": t.name, "column": c.name, "reason": c.pii_reason or ""})
            if (row := _reading(t.name, c, k)) is not None:
                reading.append(row)
    return {
        "source_type": profile.source_type,
        "generated_at": fmt_dt(profile.generated_at),
        "min_cell_count": k,
        "n_tables": len(profile.tables),
        "n_rows": sum(t.n_rows for t in profile.tables),
        "n_columns": sum(len(t.columns) for t in profile.tables),
        "tables": tables,
        "pii": pii,
        # date formats, shapes, values that do not fit, partial dates, offsets, NA markers
        "reading": reading,
    }


def _design_section(spec: ETLSpec) -> dict[str, Any]:
    total: Counter[str] = Counter()
    tables = []
    for m in spec.table_mappings:
        kinds = Counter(f.kind for f in m.fields)
        total.update(kinds)
        tables.append(
            {
                "target": m.target_table,
                "source": f"{m.source.table} {m.source.alias}",
                "joins": [f"{j.join_type} join {j.table} {j.alias} on {j.on}" for j in m.joins],
                "where": m.where or "",
                "dedup_order": m.dedup_order or "",
                "n_fields": len(m.fields),
                "kinds": ", ".join(f"{k} {n}" for k, n in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
                "rationale": m.rationale,
            }
        )
    order = FIELD_KINDS + sorted(k for k in total if k not in FIELD_KINDS)

    def also(table: str) -> str:  # the other mappings whose keys go into the id map
        return "; ".join(f"{tm.source.table}: {' '.join(expr.split())}" for tm, expr, _ in key_sources(spec, table))

    return {
        "cdm_version": spec.cdm_version,
        "person_key": f"{spec.person_key.table}: {spec.person_key.expression}",
        "person_key_also": also("person"),
        "visit_key": f"{spec.visit_key.table}: {spec.visit_key.expression}" if spec.visit_key else "",
        "visit_key_also": also("visit_occurrence") if spec.visit_key else "",
        "observation_period": spec.observation_period,
        "route_by_domain": spec.route_by_domain,
        # how each era table is built: derived by the build, loaded by the spec's SQL, or not built
        "eras": eras_text(spec),
        "era_status": era_status(spec),
        # what the build does beyond the mappings (one row per key, periods, routing, eras), as on the spec page
        "rules": [{"label": label, "text": text} for label, text in build_rules(spec)],
        "n_mappings": len(spec.table_mappings),
        "n_fields": sum(total.values()),
        "kinds_total": [{"kind": k, "n": total[k]} for k in order if total[k]],
        "tables": tables,
        "custom_sql": [{"name": c.name, "target": c.target_table, "rationale": c.rationale} for c in spec.custom_sql],
        "assumptions": list(spec.assumptions),
        "open_questions": list(spec.open_questions),
    }


def _term(value: Any, count: int | None, min_cell_count: int, shareable: bool) -> Any:
    """A source code or term for display: withheld in a shareable report when it is rare."""
    if shareable and (count is None or count < min_cell_count):
        return WITHHELD
    return value


def _vocab_section(ev: dict[str, Any], min_cell_count: int, shareable: bool = False) -> dict[str, Any]:
    loaded = [
        {
            "vocabulary_id": str(v.get("vocabulary_id") or ""),
            "version": str(v.get("version") or ""),
            "concepts": _int(v.get("concepts")),
            "standard_concepts": _int(v.get("standard_concepts")),
        }
        for v in _dicts(ev.get("loaded"))
    ]
    required = [
        {
            "vocabulary_id": str(r.get("vocabulary_id") or ""),
            "reason": str(r.get("reason") or ""),
            "present": bool(r.get("present")),
        }
        for r in _dicts(ev.get("required"))
    ]
    coverage = []
    for c in _dicts(ev.get("coverage")):
        records, mapped = _int(c.get("records")), _int(c.get("records_mapped"))
        unmatched = _dicts(c.get("unmatched"))
        coverage.append(
            {
                "target": f"{c.get('target_table', '')}.{c.get('target_field', '')}".strip("."),
                "vocabulary_id": str(c.get("vocabulary_id") or ""),
                "source_table": str(c.get("source_table") or ""),
                "distinct_codes": _int(c.get("distinct_codes")),
                "codes_found": _int(c.get("codes_found")),
                "codes_mapped": _int(c.get("codes_mapped_to_standard")),
                "records": records,
                "records_mapped": mapped,
                "pct": _ratio(mapped, records),
                "error": _short(c["error"], 300) if c.get("error") else "",
                "unmatched": ", ".join(
                    f"{_term(u.get('value'), _int(u.get('count')), min_cell_count, shareable)} "
                    f"({small_cell(_int(u.get('count')), min_cell_count)['display']})"
                    for u in unmatched[:MAX_UNMATCHED_SHOWN]
                )
                + (f" +{len(unmatched) - MAX_UNMATCHED_SHOWN} more" if len(unmatched) > MAX_UNMATCHED_SHOWN else ""),
            }
        )
    return {
        "vocabulary_version": ev.get("vocabulary_version"),
        "loaded": loaded,
        "required": required,
        "n_missing": sum(not r["present"] for r in required),
        "coverage": coverage,
        "issues": _strs(ev.get("issues")),
    }


def _mapping_row(name: str, ds: list[MappingDecision], include_unchecked: bool = True) -> dict[str, Any]:
    status = Counter(d.status for d in ds)
    records = sum(d.frequency for d in ds)
    # Records whose term is actually written to source_to_concept_map (what reaches the CDM):
    # FLAGGED and REJECTED terms never are, UNCHECKED ones only with mapping.use_unchecked_in_etl.
    usable = {"APPROVED", "UNCHECKED"} if include_unchecked else {"APPROVED"}
    mapped = sum(d.frequency for d in ds if d.target_concept_id and d.status in usable)
    return {
        "source_vocabulary_id": name,
        "terms": len(ds),
        "approved": status["APPROVED"],
        "unchecked": status["UNCHECKED"],
        "flagged": status["FLAGGED"],
        "rejected": status["REJECTED"],
        "unmapped": sum(1 for d in ds if not d.target_concept_id),
        "suppressed": sum(1 for d in ds if d.method == "suppressed"),
        "records": records,
        "records_mapped": mapped,
        "pct": _ratio(mapped, records),
    }


def _mapping_section(decisions: list[MappingDecision], cfg: ProjectConfig, shareable: bool = False) -> dict[str, Any]:
    groups: dict[str, list[MappingDecision]] = defaultdict(list)
    for d in decisions:
        groups[d.source_vocabulary_id].append(d)
    pending = sorted(
        (d for d in decisions if d.status in ("UNCHECKED", "FLAGGED")),
        key=lambda d: (d.status != "FLAGGED", -d.frequency, d.source_vocabulary_id, d.source_code),
    )
    mcc = cfg.privacy.min_cell_count
    review = [
        {
            "status": d.status,
            "vocabulary": d.source_vocabulary_id,
            "code": _term(d.source_code, d.frequency, mcc, shareable),
            "name": _term(d.source_name, d.frequency, mcc, shareable),
            "frequency": small_cell(d.frequency, mcc)["display"],
            "concept": (
                f"{d.target_concept_id} {d.target_concept_name or ''}".strip() if d.target_concept_id else "none (0)"
            ),
            "equivalence": d.equivalence,
            "confidence": f"{d.confidence:.2f}",
            "method": d.method,
        }
        for d in pending[:MAX_REVIEW_ROWS]
    ]
    return {
        "vocabularies": [_mapping_row(k, groups[k], cfg.mapping.use_unchecked_in_etl) for k in sorted(groups)],
        "totals": _mapping_row("all", decisions, cfg.mapping.use_unchecked_in_etl),
        "review": review,
        "review_total": len(pending),
        "use_unchecked_in_etl": cfg.mapping.use_unchecked_in_etl,
    }


def _key_stats(s: TableLoadResult, count: Callable[[int], str]) -> str:
    """One row per key (person, death, visit_occurrence): source rows that lost to another row of
    their key, keys an earlier mapping loaded, conflicting keys and keys with a code of several
    concepts, e.g. '30 dup / 45 earlier / 3 conflicting'."""
    st = s.stats
    if s.error or "keys" not in st:
        return ""
    parts = [f"{count(int(st.get('duplicate_rows') or 0))} dup", f"{count(int(st.get('already_loaded_keys') or 0))} "
             "earlier", f"{count(int(st.get('conflict_keys') or 0))} conflicting"]
    if st.get("fan_out_keys"):
        parts.append(f"{count(int(st['fan_out_keys']))} fan-out")
    if st.get("gap_keys"):
        parts.append(f"{count(int(st['gap_keys']))} separate stays")
    return " / ".join(parts)


def _link_stats(s: TableLoadResult, count: Callable[[int], str]) -> str:
    """Visit links of a step's records: linked of those with a visit key, how many name another
    person's visit and how many a visit a visit mapping left out (filtered out, or not loaded)."""
    links = s.stats.get("visit_links") or {}
    if not links:
        return ""
    text = f"{count(int(links.get('linked') or 0))}/{count(int(links.get('with_key') or 0))}"
    extra = [f"{count(int(links[k]))} {label}" for k, label in (("other_person", "other person"),
                                                                 ("filtered", "filtered")) if links.get(k)]
    return ", ".join([text] + extra)


def _period_stats(s: TableLoadResult, count: Callable[[int], str]) -> str:
    """The observation-period merge and fill steps: episodes -> periods, dropped and open episodes,
    the release date, and the persons whose episodes overlap or have an open episode followed by a
    later one (merged: for review), e.g. '560 episodes -> 440 periods (40 dropped: end before start;
    20 open, closed at 2023-12-31 (configured) or at death (3); persons with overlapping episodes:
    12, with an open episode followed by a later one: 2)'; persons the fill gave a period from
    their events."""
    st = s.stats
    if s.error:
        return ""

    def n(key: str) -> int:
        return int(st.get(key) or 0)

    if s.kind == "merge" and "episodes" in st:
        how = "configured" if st.get("configured") else "derived"
        closed = f", closed at {st['release_date']} ({how})" if st.get("release_date") else ""
        if n("open_closed_at_death"):
            closed += f" or at death ({count(n('open_closed_at_death'))})"
        return (f"{count(n('episodes'))} episodes → {count(n('inserted'))} periods "
                f"({count(n('dropped_inverted'))} dropped: end before start; {count(n('open_episodes'))} open{closed}; "
                f"persons with overlapping episodes: {count(n('persons_overlapping'))}, with an open episode followed "
                f"by a later one: {count(n('persons_stale_open'))})")
    if s.kind == "derived" and st.get("persons_without_period") is not None:
        return (f"{count(int(st.get('inserted') or 0))} of {count(int(st['persons_without_period']))} persons without "
                "a period got one from their events")
    return ""


def _moved_stats(s: TableLoadResult, count: Callable[[int], str]) -> str:
    """A route step: the records it moved to the table of their concept's domain (it loads none,
    so it has no retention), e.g. 'moved 12'."""
    if s.kind != "route" or s.error or s.skipped or step_label(s.step) == ROUTE_PREPARE:
        return ""
    return f"moved {count(sum(m.moved_rows for m in s.moved))}"


def _map_stats(s: TableLoadResult, count: Callable[[int], str]) -> str:
    """An id-map step: the new keys it numbered; a pruning step: the keys it retired (they keep
    their ids for when they return). These steps write the ETL schema only."""
    if s.error or s.skipped:
        return ""
    if s.kind == "id_map":
        return f"{count(s.rows_inserted)} new keys"
    return f"{count(s.rows_inserted)} keys retired" if s.kind == "prune" else ""


def _step_table(s: TableLoadResult) -> str:
    """The table a step writes: the CDM table, or for the id-map and pruning steps the ETL id map
    (person_map, visit_map), for route_prepare the ETL table of the largest ids before routing."""
    if s.kind in ("id_map", "prune"):
        return "visit_map" if step_label(s.step).startswith("visit_map") else "person_map"
    return "id_high_water" if s.kind == "route" and step_label(s.step) == ROUTE_PREPARE else s.target_table


def _derived_stats(s: TableLoadResult) -> str:
    """A step that derives its rows from other CDM tables: the eras, and the observation periods
    from the records (the fill is described by ``_period_stats``)."""
    if s.kind != "derived" or s.error or s.skipped or "persons_without_period" in s.stats:
        return ""
    if s.target_table in DERIVED_ERAS:
        return f"derived from {DERIVED_ERAS[s.target_table]}"
    return "derived from each person's first and last record" if s.target_table == "observation_period" else ""


def _value_rows(run: ETLRunResult, count: Callable[[int], str]) -> list[dict[str, Any]]:
    """Values not loaded as written, one row per mapping and field (``TableLoadResult.value_counts``):
    unreadable values (NULL), partial dates completed or left NULL, NULL required values given the
    CDM default, rows dropped for a NULL required value, and time zone offsets ignored or converted.
    The WHERE is the field 'where'. A second mapping into a table is named by its step."""
    cdm = load_cdm_spec()
    rows = []
    for s in run.steps:
        if s.error or not s.value_counts:
            continue
        fields = {f for cat, by_field in s.value_counts.items() if cat != ROW_SITE for f in by_field}
        order = ([f.name for f in cdm.table(s.target_table).fields] if cdm.has_table(s.target_table) else [])
        label = step_label(s.step)
        for f in sorted(fields, key=lambda f: (f != WHERE_SITE, order.index(f) if f in order else len(order), f)):
            counts = {c: int(s.value_counts.get(c, {}).get(f) or 0) for c in VALUE_CATEGORIES}
            if not any(counts.values()):
                continue
            rows.append({
                "table": s.target_table,
                "field": "(filter)" if f == WHERE_SITE else f,
                "step": label if label != s.target_table else "",
                **{c: count(n) if n else "" for c, n in counts.items()},
            })
    return rows


def _retention(s: TableLoadResult) -> float | None:
    """Rows inserted per source row; for a step that keeps one row per key, the source rows kept
    (``retained_rows``, the etlRecordRetention basis): rows that lost to another row of their key,
    or rolled up into one visit, are not lost (the Statistics column counts them)."""
    if "keys" in s.stats and s.retained_rows is not None:
        return _ratio(s.retained_rows, s.source_rows)
    return _ratio(s.rows_inserted, s.source_rows)


def _etl_section(run: ETLRunResult, min_cell_count: int = 5, shareable: bool = False) -> dict[str, Any]:
    def count(n: int) -> str:  # counts in (0, min_cell_count) are withheld in the shareable report
        return small_cell(n, min_cell_count)["display"] if shareable else fmt_int(n)

    steps = [
        {
            "table": _step_table(s),
            "step": s.step,
            # route steps load nothing (see "moved"), id-map and pruning steps write the id maps only
            "rows": None if s.kind in ("route", "id_map", "prune") or s.skipped else s.rows_inserted,
            "source_rows": s.source_rows,
            "retention": _retention(s) if s.error is None and not s.skipped else None,
            # the Statistics column: one row per key, the merge or fill, the records moved, what a derived
            # step reads, or the keys an id-map step numbered or retired
            "one_row_per_key": _key_stats(s, count),
            "periods": _period_stats(s, count),
            "moved": _moved_stats(s, count),
            "derived": _derived_stats(s),
            "id_keys": _map_stats(s, count),
            "visit_links": _link_stats(s, count),
            "seconds": s.seconds,
            "error": _short(s.error, 600) if s.error else "",
            "blocking": s.blocking,  # an error of a step that does not block the build (route, eras) is a warning
            "skipped": _short(s.skipped, 300) if s.skipped else "",
        }
        for s in run.steps
    ]
    # the records the route steps moved to the table of their concept's domain, per destination
    moved = [
        {"from_table": m.from_table, "to_table": m.to_table, "domain_id": m.domain_id, "rows": count(m.moved_rows),
         "dropped": count(m.rows_with_dropped_fields)}
        for s in run.steps if s.kind == "route" and not s.error for m in s.moved
    ]
    start, end = _as_utc(run.started_at), _as_utc(run.finished_at)
    return {
        "started_at": fmt_dt(run.started_at),
        "finished_at": fmt_dt(run.finished_at),
        "duration": (end - start).total_seconds() if start and end else None,
        "steps": steps,
        "total_rows": run.rows_loaded(),  # the merge's periods replace rows already counted
        "derived_rows": run.rows_derived(),  # apart: observation periods and eras the build derived
        "moved": moved,
        "moved_rows": count(run.rows_moved()) if run.rows_moved() else "",
        "value_rows": _value_rows(run, count),  # values not loaded as written, per table and field
        "n_errors": len(run.errors()),
        "n_blocking_errors": len(run.blocking_errors()),
        "ok": run.ok,
    }


def _check_status(r: CheckResult) -> str:
    if r.error:
        return "error"
    if r.not_applicable:
        return "not_applicable"
    return "failed" if r.failed else "passed"


def _violated_pct(r: CheckResult) -> float:
    if r.num_denominator_rows:
        return 100.0 * r.num_violated_rows / r.num_denominator_rows
    return 100.0 * r.pct_violated_rows  # DQD convention: a fraction of the denominator


def _tally(results: list[CheckResult], key: Callable[[CheckResult], str], order: list[str]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in results:
        counts[key(r)][_check_status(r)] += 1
    names = [k for k in order if k in counts] + sorted(k for k in counts if k not in order)
    rows = [{"name": k, **{s: counts[k][s] for s in CHECK_STATUSES}, "total": sum(counts[k].values())} for k in names]
    if rows:
        rows.append({"name": "total", **{s: sum(r[s] for r in rows) for s in CHECK_STATUSES}, "total": len(results)})
    return rows


def _dq_section(
    dq: DQReport, cfg: ProjectConfig, derived: set[str] | frozenset[str] = frozenset(),
    not_built: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The data-quality results. The failures on the era tables the build derived (``derived``) are
    listed apart, under the event table each derives from: they follow the condition and drug
    records, whose own checks come first, and are never sent to the repair agent. An era table
    whose step failed or was skipped (``not_built``: table -> why) fails its checks for that reason,
    which its group says instead."""
    blocking_sev = set(cfg.quality.blocking_severities)

    def is_blocking(r: CheckResult) -> bool:
        if r.not_applicable or not (r.failed or r.error):
            return False
        # quality.checks.gate() writes one line per blocking problem, starting with the check key.
        return r.severity in blocking_sev or any(b == r.key or b.startswith(r.key + " (") for b in dq.blocking_failures)

    def row(r: CheckResult) -> dict[str, Any]:
        return {
            "key": r.key,
            "level": r.check_level,
            "severity": r.severity,
            "category": r.category,
            "violated": r.num_violated_rows,
            "denominator": r.num_denominator_rows,
            "pct": _violated_pct(r),
            "threshold": f"{r.threshold_pct:g}%",
            "description": r.description,
            "blocking": is_blocking(r),
        }

    failed = [r for r in dq.results if r.failed and not r.error and not r.not_applicable]
    failed.sort(key=lambda r: (not is_blocking(r), SEVERITIES.index(r.severity), -_violated_pct(r), r.key))
    eras = [{"table": t, "source": DERIVED_ERAS[t], "failed": [row(r) for r in failed if r.table == t],
             "not_built": (not_built or {}).get(t)}
            for t in DERIVED_ERAS if t in derived]
    return {
        "run_at": fmt_dt(dq.run_at),
        "engine": dq.engine,
        "passed": dq.passed,
        "n_checks": len(dq.results),
        "by_severity": _tally(dq.results, lambda r: r.severity, SEVERITIES),
        "by_category": _tally(dq.results, lambda r: r.category or "(none)", KAHN_CATEGORIES),
        "n_failed": len(failed),
        "failed": [row(r) for r in failed if r.table not in derived],
        # the failures on each era table the build derived, under its event table
        "derived_failed": [g for g in eras if g["failed"]],
        "errors": [
            {"key": r.key, "severity": r.severity, "error": _short(r.error, 600), "blocking": is_blocking(r)}
            for r in dq.results
            if r.error
        ],
        "blocking_failures": [_short(b, 400) for b in dq.blocking_failures],
        "n_blocking": len(dq.blocking_failures) or sum(is_blocking(r) for r in dq.results),
    }


def _repair_events(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair-agent runs from the audit log (including ones that changed nothing or failed)."""
    events = []
    for e in audit:
        kind = e.get("event")
        if kind == "repair":
            detail = f"{e.get('actions', 0)} change(s)"
            if e.get("summary"):
                detail += f": {_short(e['summary'], 400)}"
        elif kind == "repair_skipped":
            detail = f"skipped: {_short(e.get('reason') or '', 400)}"
        elif kind == "repair_error":
            detail = f"error: {_short(e.get('error') or '', 400)}"
        else:
            continue
        events.append(
            {
                "at": fmt_dt(e.get("ts")),
                "iteration": e.get("iteration", ""),
                "outcome": {"repair": "done", "repair_skipped": "skipped", "repair_error": "failed"}[kind],
                "trigger": _short(e.get("trigger") or "", 200),
                "detail": detail,
            }
        )
    return events


def _repairs_section(actions: list[RepairAction] | None, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if actions is None and not events:
        return None
    by_iter: dict[int, list[dict[str, str]]] = defaultdict(list)
    for a in actions or []:
        by_iter[a.iteration].append(
            {
                "target": f"{a.target_table}.{a.target_field}" if a.target_field else f"{a.target_table}.*",
                "change": a.change,
                "rationale": a.rationale,
                "trigger": a.trigger,
            }
        )
    return {
        "has_file": actions is not None,
        "n_actions": len(actions or []),
        "iterations": [{"iteration": i, "actions": by_iter[i]} for i in sorted(by_iter)],
        "events": events,
    }


def _llm_section(cfg: ProjectConfig, ws: Workspace, audit: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    errors = []
    models: list[str] = []
    fields = [
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_read", "cache_read_input_tokens"),
        ("cache_write", "cache_creation_input_tokens"),
    ]
    for e in audit:
        if e.get("event") != "llm_call":
            continue
        name = str(e.get("stage") or "(unknown)")
        row = stages.setdefault(
            name, {"stage": name, "calls": 0, **{f: 0 for f, _ in fields}, "errors": 0, "seconds": 0.0}
        )
        usage = e.get("usage") if isinstance(e.get("usage"), dict) else {}
        row["calls"] += 1
        for f, src in fields:
            row[f] += _int(usage.get(src)) or 0
        try:
            row["seconds"] += float(e.get("seconds") or 0)
        except (TypeError, ValueError):
            pass
        if e.get("error"):
            row["errors"] += 1
            errors.append({"stage": name, "key": str(e.get("key") or ""), "error": _short(e["error"], 300)})
        label = " / ".join(str(x) for x in (e.get("provider"), e.get("model")) if x)
        if label and label not in models:
            models.append(label)
    rows = list(stages.values())
    totals = {"calls": sum(r["calls"] for r in rows), "errors": sum(r["errors"] for r in rows)}
    totals.update({f: sum(r[f] for r in rows) for f, _ in fields})
    totals["seconds"] = sum(r["seconds"] for r in rows)
    calls_dir = ws.root / "llm_calls"
    return {
        "provider": cfg.llm.provider,
        "model": cfg.llm.model if cfg.llm.provider != "none" else "",
        "models_used": models,
        "stages": rows,
        "totals": totals,
        "errors": errors[:20],
        "n_errors": len(errors),
        "n_records": len(list(calls_dir.glob("*.json"))) if calls_dir.is_dir() else 0,
    }


def _next_steps(
    verdict: str,
    spec: dict[str, Any] | None,
    notes: list[str],
    mapping: dict[str, Any] | None,
    etl: dict[str, Any] | None,
    dq: dict[str, Any] | None,
    schemas: Schemas,
    r_dqd_ran: bool,
    verdict_notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if mapping is None:
        steps.append({"text": "Map the source codes (the vocabulary mapping stage has not run).", "done": False})
    else:
        n = mapping["review_total"]
        steps.append(
            {
                "text": f"Review the {n} UNCHECKED / FLAGGED vocabulary mapping(s) in mappings/*.csv: set status "
                "to APPROVED, or correct target_concept_id, then re-import them.",
                "done": n == 0,
            }
        )
    n_q = len(spec["open_questions"]) if spec else 0
    steps.append(
        {
            "text": f"Answer the {n_q} open design question(s) in etl_spec.yaml and the {len(notes)} data-owner "
            "note(s) from the profile interpretation; update the spec where the answers change it.",
            "done": spec is not None and n_q == 0 and not notes,
        }
    )
    n_err = etl["n_blocking_errors"] if etl else 0
    n_block = dq["n_blocking"] if dq else 0
    if verdict == FAILED:
        steps.append(
            {
                "text": f"Fix the {n_err} ETL step error(s) and {n_block} blocking data-quality failure(s) "
                "listed above (edit etl_spec.yaml or the mappings), then rebuild.",
                "done": False,
            }
        )
    if verdict_notes:
        k = len(verdict_notes)
        steps.append(
            {
                "text": f"Fix the cause of the {k} ETL step(s) that failed without failing the build (see the summary "
                "and the ETL run), then rebuild: until then the CDM lacks what they would have done.",
                "done": False,
            }
        )
    steps.append(
        {
            "text": "Run the official OHDSI Achilles and DataQualityDashboard in R: set `quality.use_r_dqd: true` "
            "in the project config (Rscript and the OHDSI R packages must be installed) and re-run the checks.",
            "done": r_dqd_ran,
        }
    )
    steps.append(
        {
            "text": "Register the CDM in ATLAS/WebAPI: add a row to webapi.source (JDBC URL of the target database) "
            f"and webapi.source_daimon rows for CDM (daimon_type 0, schema {schemas.cdm}), Vocabulary "
            f"(1, {schemas.vocab}) and Results (2, {schemas.results}); then refresh WebAPI's source cache.",
            "done": False,
        }
    )
    steps.append(
        {
            "text": "Re-run `omop-agent run` after any edit so the CDM, the checks and this report are regenerated.",
            "done": False,
        }
    )
    return steps


# --------------------------------------------------------------------------------------
# CDM characterisation (Achilles-lite)
# --------------------------------------------------------------------------------------


def _concept_label(concept_id: Any, name: Any) -> str:
    if concept_id is None:
        return "(null)"
    if name:
        return str(name)
    return "No matching concept" if concept_id == 0 else "(not in vocabulary)"


def _release_basis(run: ETLRunResult | None) -> str | None:
    """How the build got the source release date: 'configured' or 'derived' (by the merge step)."""
    merge = next((s for s in (run.steps if run else []) if s.kind == "merge" and not s.error), None)
    if merge is None or not merge.stats.get("release_date"):
        return None
    return "configured" if merge.stats.get("configured") else "derived"


def _eras_not_built(run: ETLRunResult | None) -> dict[str, str]:
    """The era tables whose build step failed or was skipped in ``run``, with why."""
    out: dict[str, str] = {}
    for s in run.steps if run is not None else []:
        if s.kind == "derived" and s.target_table in DERIVED_ERAS and (s.error or s.skipped):
            out[s.target_table] = f"step {s.step} failed" if s.error else f"step {s.step} was skipped because {s.skipped}"
    return out


def _derived_eras(run: ETLRunResult | None, spec: ETLSpec | None) -> set[str]:
    """The era tables the build derived: from the steps of the last build, else from the spec."""
    if run is not None:
        return {s.target_table for s in run.steps if s.kind == "derived" and s.target_table in DERIVED_ERAS}
    return set(derived_era_tables(spec)) if spec is not None else set()


def characterize_cdm(
    conn: psycopg.Connection, schemas: Schemas, min_cell_count: int = 5, release_basis: str | None = None
) -> dict[str, Any]:
    """Summarise the loaded CDM: record counts, demographics, observation time, top concepts.

    Observation time includes the persons with more than one period and the periods that end on
    cdm_source.source_release_date (open episodes the build ended there), with how the build got
    that date (``release_basis``: 'configured' or 'derived', from the merge step; None: unknown).
    Every count in (0, min_cell_count) is returned as ``<min_cell_count`` with ``count=None``;
    distribution statistics over fewer than ``min_cell_count`` records are withheld.
    Queries that fail (e.g. a missing table) are listed in ``errors`` and skipped.
    """
    errors: list[str] = []
    cdm, vocab = qi(schemas.cdm), qi(schemas.vocab)

    def q(what: str, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]] | None:
        try:
            with conn.transaction():
                return fetch_all(conn, sql, params)
        except psycopg.Error as e:
            errors.append(f"{what}: {short_error(e)}")
            return None

    def cell(n: Any) -> dict[str, Any]:
        return small_cell(_int(n), min_cell_count)

    existing = {
        (r["table_schema"], r["table_name"])
        for r in q(
            "list tables",
            "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema IN (%s, %s)",
            (schemas.cdm, schemas.vocab),
        )
        or []
    }
    has_concept = (schemas.vocab, "concept") in existing

    def by_concept(what: str, table: str, field: str, limit: int | None = None) -> list[dict[str, Any]]:
        if has_concept:
            sql = (
                f"SELECT t.{qi(field)} AS concept_id, c.concept_name, COUNT(*) AS n FROM {cdm}.{qi(table)} t "
                f"LEFT JOIN {vocab}.concept c ON c.concept_id = t.{qi(field)} GROUP BY 1, 2 ORDER BY n DESC, 1"
            )
        else:
            sql = (
                f"SELECT {qi(field)} AS concept_id, NULL AS concept_name, COUNT(*) AS n FROM {cdm}.{qi(table)} "
                "GROUP BY 1 ORDER BY n DESC, 1"
            )
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = q(what, sql) or []
        return _add_bars(
            [
                {
                    "concept_id": r["concept_id"],
                    "concept_name": _concept_label(r["concept_id"], r["concept_name"]),
                    **cell(r["n"]),
                }
                for r in rows
            ]
        )

    # record counts, in load order
    spec = load_cdm_spec()
    cdm_tables = {t.name for t in spec.cdm_tables}
    ordered = [t for t in LOAD_ORDER if t in cdm_tables] + sorted(cdm_tables - set(LOAD_ORDER))
    counts: dict[str, int] = {}
    for t in ordered:
        if (schemas.cdm, t) not in existing:
            continue
        rows = q(f"count {t}", f"SELECT COUNT(*) AS n FROM {cdm}.{qi(t)}")
        if rows and rows[0]["n"]:
            counts[t] = int(rows[0]["n"])
    record_counts = _add_bars([{"table": t, **cell(n)} for t, n in counts.items()])

    gender: list[dict[str, Any]] = []
    decades: list[dict[str, Any]] = []
    if counts.get("person"):
        gender = by_concept("persons by gender", "person", "gender_concept_id")
        rows = q(
            "year of birth",
            f"SELECT (year_of_birth / 10) * 10 AS decade, COUNT(*) AS n FROM {cdm}.person "
            "GROUP BY 1 ORDER BY 1 NULLS LAST",
        )
        decades = _add_bars(
            [
                {"decade": f"{r['decade']}s" if r["decade"] is not None else "unknown", **cell(r["n"])}
                for r in rows or []
            ]
        )

    obs: dict[str, Any] | None = None
    if counts.get("observation_period"):
        days = "(observation_period_end_date - observation_period_start_date)"
        rows = q(
            "observation period length",
            f"""SELECT COUNT(*) AS n, COUNT(DISTINCT person_id) AS persons,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY {days}) AS p50,
                       percentile_cont(0.1) WITHIN GROUP (ORDER BY {days}) AS p10,
                       percentile_cont(0.9) WITHIN GROUP (ORDER BY {days}) AS p90
                FROM {cdm}.observation_period""",
        )
        if rows:
            r = rows[0]
            enough = int(r["n"]) >= min_cell_count
            obs = {
                "periods": cell(r["n"])["display"],
                "persons": cell(r["persons"])["display"],
                "median_days": round(r["p50"]) if enough and r["p50"] is not None else None,
                "p10_days": round(r["p10"]) if enough and r["p10"] is not None else None,
                "p90_days": round(r["p90"]) if enough and r["p90"] is not None else None,
                "suppressed": not enough,
            }
            release = (f"(SELECT source_release_date FROM {cdm}.cdm_source LIMIT 1)" if counts.get("cdm_source")
                       else "NULL::date")
            rows = q(
                "observation periods per person",
                f"""SELECT (SELECT COUNT(*) FROM (SELECT person_id FROM {cdm}.observation_period GROUP BY person_id
                                                  HAVING COUNT(*) > 1) m) AS multi,
                           (SELECT COUNT(*) FROM {cdm}.observation_period
                            WHERE observation_period_end_date = {release}) AS at_release,
                           {release}::text AS release_date""",
            )
            if rows:
                r = rows[0]
                obs |= {
                    "multi_period_persons": cell(r["multi"])["display"],
                    "ending_at_release": cell(r["at_release"])["display"],
                    "release_date": r["release_date"],
                    "release_basis": release_basis,
                }

    top = [
        {"table": t, "field": f, "rows": by_concept(f"top concepts in {t}", t, f, limit=10)}
        for t, f in TOP_CONCEPT_FIELDS.items()
        if counts.get(t)
    ]

    vocab_version = None
    if (schemas.vocab, "vocabulary") in existing:
        rows = q(
            "vocabulary version", f"SELECT vocabulary_version FROM {vocab}.vocabulary WHERE vocabulary_id = 'None'"
        )
        vocab_version = rows[0]["vocabulary_version"] if rows else None
    cdm_source = None
    if counts.get("cdm_source"):
        rows = q(
            "cdm_source",
            "SELECT cdm_source_name, cdm_holder, cdm_version, vocabulary_version, cdm_release_date "
            f"FROM {cdm}.cdm_source LIMIT 1",
        )
        cdm_source = {k: ("" if v is None else str(v)) for k, v in rows[0].items()} if rows else None

    return {
        "min_cell_count": min_cell_count,
        "vocabulary_version": vocab_version,
        "cdm_source": cdm_source,
        "record_counts": record_counts,
        "persons": cell(counts.get("person"))["display"] if counts.get("person") else None,
        "gender": gender,
        "birth_decades": decades,
        "observation_period": obs,
        "top_concepts": top,
        "errors": errors,
    }


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def collect_report_data(
    cfg: ProjectConfig,
    ws: Workspace,
    conn: psycopg.Connection | None = None,
    schemas: Schemas | None = None,
    shareable: bool = False,
) -> dict[str, Any]:
    """Gather everything the report shows into one plain dict (the templates' only input).

    With ``shareable``, source codes and terms seen fewer than ``min_cell_count`` times are
    replaced by ``<withheld>`` (mapping review table, unmatched codes).
    """
    problems: list[str] = []
    schemas = schemas or Schemas.from_config(cfg.target)
    mcc = cfg.privacy.min_cell_count

    profile = _load(ws.profile_json, SourceProfile.model_validate_json, problems)
    annotations = _load(ws.annotations_json, SourceAnnotations.model_validate_json, problems)
    spec = _load(ws.spec_yaml, _spec, problems)
    vocab_eval = _load(ws.vocab_eval_json, _json_object, problems)
    decisions = _load(ws.mappings_dir / "decisions.json", _model_list(MappingDecision), problems)
    etl_run = _load(ws.etl_run_json, ETLRunResult.model_validate_json, problems)
    dq = _load(ws.dq_json, DQReport.model_validate_json, problems)
    repairs = _load(ws.repairs_json, _model_list(RepairAction), problems)
    stages = _read_state(ws, problems)
    audit = _read_audit(ws, problems)

    cdm: dict[str, Any] | None = None
    cdm_note = "No database connection was given, so the loaded CDM was not characterised."
    if conn is not None:
        try:
            cdm = characterize_cdm(conn, schemas, mcc, _release_basis(etl_run))
        except psycopg.Error as e:
            cdm_note = f"CDM characterisation failed: {short_error(e)}"

    verdict, reasons, verdict_notes = _verdict(dq, etl_run, stages)
    review_rows, caveats = _reviews(cfg, ws, audit, repairs, decisions)
    design = _design_section(spec) if spec else None
    vocab = _vocab_section(vocab_eval, mcc, shareable) if vocab_eval is not None else None
    mapping = _mapping_section(decisions, cfg, shareable) if decisions is not None else None
    etl = _etl_section(etl_run, mcc, shareable) if etl_run else None
    dq_data = _dq_section(dq, cfg, _derived_eras(etl_run, spec), _eras_not_built(etl_run)) if dq else None
    notes = list(annotations.notes) if annotations else []
    llm = _llm_section(cfg, ws, audit)
    # Done only when the current dq_results.json holds DQD results; an older successful run in the
    # audit log does not count once a later check fell back to the built-in engine.
    r_dqd_ran = dq is not None and dq.engine != "builtin"

    vocab_version = (vocab or {}).get("vocabulary_version") or (cdm or {}).get("vocabulary_version")
    cdm_version = spec.cdm_version if spec else ((cdm or {}).get("cdm_source") or {}).get("cdm_version") or "5.4"
    data = {
        "project": {
            "name": cfg.name,
            "generated_at": fmt_dt(utcnow()),
            "cdm_version": cdm_version,
            "vocabulary_version": vocab_version or "unknown",
            "llm": f"{llm['provider']} / {llm['model']}" if llm["model"] else llm["provider"],
            "llm_used": llm["models_used"],
            "workdir": str(ws.root),
            "min_cell_count": mcc,
            "cdm_schema": schemas.cdm,
            "vocab_schema": schemas.vocab,
        },
        "notice": {
            "shareable": shareable,
            "text": SHAREABLE_NOTICE.format(n=mcc) if shareable else CONFIDENTIAL_NOTICE,
        },
        "verdict": verdict,
        "verdict_reasons": reasons,
        "verdict_caveats": caveats if verdict == PASSED else [],
        "verdict_notes": verdict_notes,  # steps that failed without failing the build
        "reviews": review_rows,
        "timeline": _timeline(stages, audit, ws.root),
        "source": _source_section(apply_overrides(profile, cfg.privacy, annotations), annotations) if profile else None,
        "annotations": {"notes": notes, "n_tables": len(annotations.tables)} if annotations else None,
        "design": design,
        "vocab": vocab,
        "mapping": mapping,
        "etl": etl,
        "dq": dq_data,
        "repairs": _repairs_section(repairs, _repair_events(audit)),
        "cdm": cdm,
        "cdm_note": cdm_note,
        "llm": llm,
        "next_steps": _next_steps(verdict, design, notes, mapping, etl, dq_data, schemas, r_dqd_ran, verdict_notes),
        "problems": problems,
    }
    return _redact_errors(data) if shareable else data


# Report fields that can carry database error text, which may quote record values.
_ERROR_KEYS = {"error", "detail", "details", "cdm_note"}
_ERROR_LIST_KEYS = {"blocking_failures", "verdict_reasons", "reasons", "failures"}


def _redact_errors(obj: Any, key: str | None = None) -> Any:
    """For the shareable report: database error messages with record values removed (the same
    redaction as for prompts; PostgreSQL's DETAIL can quote a whole failing row)."""
    from ..quality.repair import redact_error

    if isinstance(obj, dict):
        return {k: _redact_errors(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        if key in _ERROR_LIST_KEYS:
            return [redact_error(x) if isinstance(x, str) else _redact_errors(x) for x in obj]
        return [_redact_errors(x) for x in obj]
    if isinstance(obj, str) and key in _ERROR_KEYS and obj:
        return redact_error(obj)
    return obj


def _template_source(name: str) -> str | None:
    path = resources.files(TEMPLATE_PKG).joinpath("templates").joinpath(name)
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _env(autoescape: bool) -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FunctionLoader(_template_source),
        autoescape=autoescape,
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(
        num=fmt_int,
        pct=fmt_pct,
        secs=fmt_secs,
        md=md_escape,
        statuscls=status_class,
        textbar=text_bar,
        inline_code=inline_code,
    )
    return env


def render_markdown(data: dict[str, Any]) -> str:
    return _env(autoescape=False).get_template("report.md.j2").render(**data)


def render_html(data: dict[str, Any]) -> str:
    return _env(autoescape=True).get_template("report.html.j2").render(**data)


def _shareable_path(p: Path) -> Path:
    return p.with_name(f"{p.stem}.shareable{p.suffix}")


def build_report(
    cfg: ProjectConfig,
    ws: Workspace,
    conn: psycopg.Connection | None = None,
    schemas: Schemas | None = None,
    shareable: bool = False,
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.html`` into the workspace and return their paths.

    With ``shareable=True`` rare source terms are withheld (see :func:`collect_report_data`) and
    the files are ``report.shareable.md`` / ``report.shareable.html``, so the full review copy is
    kept.
    """
    data = collect_report_data(cfg, ws, conn, schemas, shareable=shareable)
    md_path, html_path = ws.report_md, ws.report_html
    if shareable:
        md_path, html_path = _shareable_path(md_path), _shareable_path(html_path)
    md_path.write_text(render_markdown(data), encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")
    return md_path, html_path
