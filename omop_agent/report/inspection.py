"""Inspection report: one Markdown and one self-contained HTML page per project run.

The report reads every artifact in the workspace. Each one is optional: a missing artifact
renders as "not run", and one that cannot be parsed is listed under "unreadable artifacts".
The overall verdict comes only from recorded results (``dq_results.json``, ``etl_run.json``,
``state.json``), never from a model.

When a database connection is given, the loaded CDM is also characterised (an Achilles-lite
summary). Every count in that summary below ``min_cell_count`` is shown as ``<N`` and its
exact value is never stored.
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

from ..cdm.spec import LOAD_ORDER, load_cdm_spec
from ..config import ProjectConfig
from ..db import Schemas, fetch_all, qi, short_error
from ..models import (
    CheckResult,
    DQReport,
    ETLRunResult,
    ETLSpec,
    MappingDecision,
    RepairAction,
    SourceAnnotations,
    SourceProfile,
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
FIELD_KINDS = ["person_id", "visit_id", "auto_id", "column", "expression", "constant", "vocab_code", "concept_map", "null"]
MAX_REVIEW_ROWS = 100
MAX_UNMATCHED_SHOWN = 5


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
    return "—" if x is None else f"{x:.{digits}f}%"


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
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if etl_run is not None and etl_run.errors():
        errs = etl_run.errors()
        failures.append(f"{len(errs)} ETL step(s) errored: " + ", ".join(s.step for s in errs[:5]))
    if dq is not None and not dq.passed:
        n = len(dq.blocking_failures)
        failures.append(f"data-quality gate failed ({n} blocking problem(s))" if n else "data-quality gate failed")
    for name, info in stages.items():
        if name != "report" and str(info.get("status", "")).lower() in FAILED_STATUSES:
            failures.append(f"stage '{name}' failed" + (f": {_short(info['error'], 160)}" if info.get("error") else ""))
    if failures:
        return FAILED, failures
    if dq is None:
        reasons = ["data-quality checks have not run"]
        if etl_run is None:
            reasons.insert(0, "the ETL has not run")
        return INCOMPLETE, reasons
    unfinished = [
        f"stage '{n}' is {i.get('status')}"
        for n, i in stages.items()
        if n != "report" and str(i.get("status", "")).lower() in UNFINISHED_STATUSES
    ]
    if unfinished:
        return INCOMPLETE, unfinished
    dq_at, etl_at = _as_utc(dq.run_at), _as_utc(etl_run.started_at) if etl_run else None
    if dq_at and etl_at and dq_at < etl_at:
        return INCOMPLETE, ["the data-quality results predate the latest ETL run; re-run the checks"]
    reasons = ["no blocking data-quality check failed"]
    if etl_run is not None:
        reasons.append(f"all {len(etl_run.steps)} ETL steps completed without error")
    return PASSED, reasons


def _timeline(stages: dict[str, dict[str, Any]], audit: list[dict[str, Any]]) -> dict[str, Any]:
    known = PIPELINE_STAGES + [s for s in STAGES if s not in PIPELINE_STAGES]
    extra = sorted((s for s in stages if s not in known), key=lambda s: str(stages[s].get("at", "")))
    rows = []
    for name in known + extra:
        info = stages.get(name)
        if info is None:
            rows.append({"stage": name, "status": "not run", "at": "", "details": ""})
            continue
        rest = {k: v for k, v in info.items() if k not in ("status", "at")}
        rows.append(
            {"stage": name, "status": str(info.get("status", "unknown")), "at": fmt_dt(info.get("at")), "details": _kv(rest)}
        )
    approvals = []
    for e in audit:
        if e.get("event") != "human_approval":
            continue
        approved = e.get("approved")
        rest = {k: v for k, v in e.items() if k not in ("ts", "event", "stage", "approved")}
        approvals.append(
            {
                "at": fmt_dt(e.get("ts")),
                "stage": str(e.get("stage") or ""),
                "decision": "approved" if approved is True else "declined" if approved is False else "recorded",
                "details": _kv(rest),
            }
        )
    return {"stages": rows, "approvals": approvals, "has_state": bool(stages)}


def _source_section(profile: SourceProfile, annotations: SourceAnnotations | None) -> dict[str, Any]:
    ann = {t.table: t for t in annotations.tables} if annotations else {}
    tables, pii, dates = [], [], []
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
            if c.date_format:
                dates.append({"table": t.name, "column": c.name, "format": c.date_format, "type": c.inferred_type})
    return {
        "source_type": profile.source_type,
        "generated_at": fmt_dt(profile.generated_at),
        "min_cell_count": profile.min_cell_count,
        "n_tables": len(profile.tables),
        "n_rows": sum(t.n_rows for t in profile.tables),
        "n_columns": sum(len(t.columns) for t in profile.tables),
        "tables": tables,
        "pii": pii,
        "date_formats": dates,
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
                "n_fields": len(m.fields),
                "kinds": ", ".join(f"{k} {n}" for k, n in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
                "rationale": m.rationale,
            }
        )
    order = FIELD_KINDS + sorted(k for k in total if k not in FIELD_KINDS)
    return {
        "cdm_version": spec.cdm_version,
        "person_key": f"{spec.person_key.table}: {spec.person_key.expression}",
        "visit_key": f"{spec.visit_key.table}: {spec.visit_key.expression}" if spec.visit_key else "",
        "observation_period": spec.observation_period,
        "n_mappings": len(spec.table_mappings),
        "n_fields": sum(total.values()),
        "kinds_total": [{"kind": k, "n": total[k]} for k in order if total[k]],
        "tables": tables,
        "custom_sql": [{"name": c.name, "target": c.target_table, "rationale": c.rationale} for c in spec.custom_sql],
        "assumptions": list(spec.assumptions),
        "open_questions": list(spec.open_questions),
    }


def _vocab_section(ev: dict[str, Any], min_cell_count: int) -> dict[str, Any]:
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
        {"vocabulary_id": str(r.get("vocabulary_id") or ""), "reason": str(r.get("reason") or ""), "present": bool(r.get("present"))}
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
                    f"{u.get('value')} ({small_cell(_int(u.get('count')), min_cell_count)['display']})"
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


def _mapping_row(name: str, ds: list[MappingDecision]) -> dict[str, Any]:
    status = Counter(d.status for d in ds)
    records = sum(d.frequency for d in ds)
    mapped = sum(d.frequency for d in ds if d.target_concept_id and d.status != "REJECTED")
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


def _mapping_section(decisions: list[MappingDecision], cfg: ProjectConfig) -> dict[str, Any]:
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
            "code": d.source_code,
            "name": d.source_name,
            "frequency": small_cell(d.frequency, mcc)["display"],
            "concept": f"{d.target_concept_id} {d.target_concept_name or ''}".strip() if d.target_concept_id else "none (0)",
            "equivalence": d.equivalence,
            "confidence": f"{d.confidence:.2f}",
            "method": d.method,
        }
        for d in pending[:MAX_REVIEW_ROWS]
    ]
    return {
        "vocabularies": [_mapping_row(k, groups[k]) for k in sorted(groups)],
        "totals": _mapping_row("all", decisions),
        "review": review,
        "review_total": len(pending),
        "use_unchecked_in_etl": cfg.mapping.use_unchecked_in_etl,
    }


def _etl_section(run: ETLRunResult) -> dict[str, Any]:
    steps = [
        {
            "table": s.target_table,
            "step": s.step,
            "rows": s.rows_inserted,
            "source_rows": s.source_rows,
            "retention": _ratio(s.rows_inserted, s.source_rows) if s.error is None else None,
            "seconds": s.seconds,
            "error": _short(s.error, 600) if s.error else "",
        }
        for s in run.steps
    ]
    start, end = _as_utc(run.started_at), _as_utc(run.finished_at)
    return {
        "started_at": fmt_dt(run.started_at),
        "finished_at": fmt_dt(run.finished_at),
        "duration": (end - start).total_seconds() if start and end else None,
        "steps": steps,
        "total_rows": sum(s.rows_inserted for s in run.steps if s.error is None),
        "n_errors": len(run.errors()),
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


def _dq_section(dq: DQReport, cfg: ProjectConfig) -> dict[str, Any]:
    blocking_sev = set(cfg.quality.blocking_severities)

    def is_blocking(r: CheckResult) -> bool:
        if r.not_applicable or not (r.failed or r.error):
            return False
        # quality.checks.gate() writes one line per blocking problem, starting with the check key.
        return r.severity in blocking_sev or any(b == r.key or b.startswith(r.key + " (") for b in dq.blocking_failures)

    failed = [r for r in dq.results if r.failed and not r.error and not r.not_applicable]
    failed.sort(key=lambda r: (not is_blocking(r), SEVERITIES.index(r.severity), -_violated_pct(r), r.key))
    return {
        "run_at": fmt_dt(dq.run_at),
        "engine": dq.engine,
        "passed": dq.passed,
        "n_checks": len(dq.results),
        "by_severity": _tally(dq.results, lambda r: r.severity, SEVERITIES),
        "by_category": _tally(dq.results, lambda r: r.category or "(none)", KAHN_CATEGORIES),
        "failed": [
            {
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
            for r in failed
        ],
        "errors": [
            {"key": r.key, "severity": r.severity, "error": _short(r.error, 600), "blocking": is_blocking(r)}
            for r in dq.results
            if r.error
        ],
        "blocking_failures": [_short(b, 400) for b in dq.blocking_failures],
        "n_blocking": len(dq.blocking_failures) or sum(is_blocking(r) for r in dq.results),
    }


def _repairs_section(actions: list[RepairAction]) -> dict[str, Any]:
    by_iter: dict[int, list[dict[str, str]]] = defaultdict(list)
    for a in actions:
        by_iter[a.iteration].append(
            {
                "target": f"{a.target_table}.{a.target_field}" if a.target_field else f"{a.target_table}.*",
                "change": a.change,
                "rationale": a.rationale,
                "trigger": a.trigger,
            }
        )
    return {
        "n_actions": len(actions),
        "iterations": [{"iteration": i, "actions": by_iter[i]} for i in sorted(by_iter)],
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
    cfg: ProjectConfig,
    verdict: str,
    spec: dict[str, Any] | None,
    notes: list[str],
    mapping: dict[str, Any] | None,
    etl: dict[str, Any] | None,
    dq: dict[str, Any] | None,
    schemas: Schemas,
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
    n_err = etl["n_errors"] if etl else 0
    n_block = dq["n_blocking"] if dq else 0
    if verdict == FAILED:
        steps.append(
            {
                "text": f"Fix the {n_err} ETL step error(s) and {n_block} blocking data-quality failure(s) "
                "listed above (edit etl_spec.yaml or the mappings), then rebuild.",
                "done": False,
            }
        )
    steps.append(
        {
            "text": "Run the official OHDSI Achilles and DataQualityDashboard in R: set `quality.use_r_dqd: true` "
            "in the project config (Rscript and the OHDSI R packages must be installed) and re-run the checks.",
            "done": bool(cfg.quality.use_r_dqd and dq and dq["engine"] == "dqd"),
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


def characterize_cdm(conn: psycopg.Connection, schemas: Schemas, min_cell_count: int = 5) -> dict[str, Any]:
    """Summarise the loaded CDM: record counts, demographics, observation time, top concepts.

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
            sql = f"SELECT {qi(field)} AS concept_id, NULL AS concept_name, COUNT(*) AS n FROM {cdm}.{qi(table)} GROUP BY 1 ORDER BY n DESC, 1"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = q(what, sql) or []
        return _add_bars(
            [{"concept_id": r["concept_id"], "concept_name": _concept_label(r["concept_id"], r["concept_name"]), **cell(r["n"])} for r in rows]
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
            f"SELECT (year_of_birth / 10) * 10 AS decade, COUNT(*) AS n FROM {cdm}.person GROUP BY 1 ORDER BY 1 NULLS LAST",
        )
        decades = _add_bars(
            [{"decade": f"{r['decade']}s" if r["decade"] is not None else "unknown", **cell(r["n"])} for r in rows or []]
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

    top = [
        {"table": t, "field": f, "rows": by_concept(f"top concepts in {t}", t, f, limit=10)}
        for t, f in TOP_CONCEPT_FIELDS.items()
        if counts.get(t)
    ]

    vocab_version = None
    if (schemas.vocab, "vocabulary") in existing:
        rows = q("vocabulary version", f"SELECT vocabulary_version FROM {vocab}.vocabulary WHERE vocabulary_id = 'None'")
        vocab_version = rows[0]["vocabulary_version"] if rows else None
    cdm_source = None
    if counts.get("cdm_source"):
        rows = q(
            "cdm_source",
            f"SELECT cdm_source_name, cdm_holder, cdm_version, vocabulary_version, cdm_release_date FROM {cdm}.cdm_source LIMIT 1",
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
    cfg: ProjectConfig, ws: Workspace, conn: psycopg.Connection | None = None, schemas: Schemas | None = None
) -> dict[str, Any]:
    """Gather everything the report shows into one plain dict (the templates' only input)."""
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
            cdm = characterize_cdm(conn, schemas, mcc)
        except psycopg.Error as e:
            cdm_note = f"CDM characterisation failed: {short_error(e)}"

    verdict, reasons = _verdict(dq, etl_run, stages)
    design = _design_section(spec) if spec else None
    vocab = _vocab_section(vocab_eval, mcc) if vocab_eval is not None else None
    mapping = _mapping_section(decisions, cfg) if decisions is not None else None
    etl = _etl_section(etl_run) if etl_run else None
    dq_data = _dq_section(dq, cfg) if dq else None
    notes = list(annotations.notes) if annotations else []
    llm = _llm_section(cfg, ws, audit)

    vocab_version = (vocab or {}).get("vocabulary_version") or (cdm or {}).get("vocabulary_version")
    cdm_version = spec.cdm_version if spec else ((cdm or {}).get("cdm_source") or {}).get("cdm_version") or "5.4"
    return {
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
        "verdict": verdict,
        "verdict_reasons": reasons,
        "timeline": _timeline(stages, audit),
        "source": _source_section(profile, annotations) if profile else None,
        "annotations": {"notes": notes, "n_tables": len(annotations.tables)} if annotations else None,
        "design": design,
        "vocab": vocab,
        "mapping": mapping,
        "etl": etl,
        "dq": dq_data,
        "repairs": _repairs_section(repairs) if repairs is not None else None,
        "cdm": cdm,
        "cdm_note": cdm_note,
        "llm": llm,
        "next_steps": _next_steps(cfg, verdict, design, notes, mapping, etl, dq_data, schemas),
        "problems": problems,
    }


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


def build_report(
    cfg: ProjectConfig, ws: Workspace, conn: psycopg.Connection | None = None, schemas: Schemas | None = None
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.html`` into the workspace and return their paths."""
    data = collect_report_data(cfg, ws, conn, schemas)
    ws.report_md.write_text(render_markdown(data), encoding="utf-8")
    ws.report_html.write_text(render_html(data), encoding="utf-8")
    return ws.report_md, ws.report_html
