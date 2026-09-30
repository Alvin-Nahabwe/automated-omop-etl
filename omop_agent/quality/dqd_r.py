"""Optional: run the official OHDSI Achilles + DataQualityDashboard R packages.

``write_r_script`` renders ``quality/r/run_achilles_dqd.R`` into ``<workdir>/r/`` with the
connection settings of ``target.url`` (host, port, database, user). The password is never
written to disk: the script reads it from ``OMOP_AGENT_DB_PASSWORD``, which ``run_r_dqd`` sets
for the Rscript process only. DQD writes its results as JSON to ``<workdir>/dqd/``;
``parse_dqd_json`` turns them into ``CheckResult``s that feed the same gate as the built-in
checks.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any

from psycopg.conninfo import conninfo_to_dict

from ..config import ProjectConfig
from ..db import normalize_url
from ..models import CheckResult, DQReport
from ..workspace import Workspace, utcnow
from .checks import check_catalog, gate

R_PACKAGES = ("DatabaseConnector", "Achilles", "DataQualityDashboard")
TEMPLATE = "run_achilles_dqd.R"
RESULTS_FILE = "dqd_results.json"
PASSWORD_ENV = "OMOP_AGENT_DB_PASSWORD"

_TOKEN = re.compile(r"\{\{(\w+)\}\}")


def r_script_path(ws: Workspace) -> Path:
    return ws.root / "r" / TEMPLATE


def dqd_output_dir(ws: Workspace) -> Path:
    return ws.root / "dqd"


def dqd_results_path(ws: Workspace) -> Path:
    return dqd_output_dir(ws) / RESULTS_FILE


def r_available(cfg: ProjectConfig) -> bool:
    """True when Rscript is on PATH and DatabaseConnector, Achilles and DQD can be loaded."""
    exe = shutil.which(cfg.quality.rscript)
    if not exe:
        return False
    probe = "for (p in c({})) if (!requireNamespace(p, quietly = TRUE)) quit(status = 1)".format(
        ", ".join(f'"{p}"' for p in R_PACKAGES)
    )
    try:
        proc = subprocess.run([exe, "-e", probe], capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _connection(cfg: ProjectConfig) -> dict[str, str]:
    """libpq connection parameters of the target database (including the password, if any)."""
    params = conninfo_to_dict(normalize_url(cfg.target.url))
    return {k: str(v) for k, v in params.items() if v is not None}


def _r_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    s = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s}"'


def render_r_script(cfg: ProjectConfig, ws: Workspace, run_achilles: bool = True) -> str:
    """The R script text for this project (no password in it)."""
    conn = _connection(cfg)
    host = conn.get("host") or "localhost"
    dbname = conn.get("dbname") or conn.get("user") or "postgres"
    values: dict[str, Any] = {
        "server": f"{host}/{dbname}",
        "port": int(conn.get("port") or 5432),
        "user": conn.get("user") or os.environ.get("PGUSER", ""),
        "cdm_schema": cfg.target.cdm_schema,
        "vocab_schema": cfg.target.vocab_schema,
        "results_schema": cfg.target.results_schema,
        "cdm_source_name": cfg.cdm_source.cdm_source_name,
        "output_folder": str(dqd_output_dir(ws).resolve()),
        "results_file": RESULTS_FILE,
        "small_cell_count": int(cfg.privacy.min_cell_count),
        "run_achilles": run_achilles,
    }
    template = resources.files("omop_agent.quality").joinpath("r").joinpath(TEMPLATE).read_text(encoding="utf-8")

    def repl(m: re.Match[str]) -> str:
        if m.group(1) not in values:
            raise KeyError(f"unknown placeholder {m.group(0)} in {TEMPLATE}")
        return _r_literal(values[m.group(1)])

    return _TOKEN.sub(repl, template)


def write_r_script(cfg: ProjectConfig, ws: Workspace) -> Path:
    """Render the Achilles + DQD script into ``<workdir>/r/run_achilles_dqd.R``."""
    path = r_script_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    dqd_output_dir(ws).mkdir(parents=True, exist_ok=True)
    path.write_text(render_r_script(cfg, ws))
    return path


# --------------------------------------------------------------------------------------
# DQD results JSON
# --------------------------------------------------------------------------------------


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in {"", "NA", "NaN", "NULL"})


def _num(v: Any, default: float = 0.0) -> float:
    if isinstance(v, list):  # jsonlite may box scalars
        v = v[0] if v else None
    if _missing(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _flag(v: Any) -> bool:
    if isinstance(v, list):
        v = v[0] if v else None
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return bool(v) if v is not None else False


def _text(v: Any) -> str:
    if isinstance(v, list):
        v = v[0] if v else None
    return "" if _missing(v) else str(v)


def parse_dqd_json(path: Path, cfg: ProjectConfig) -> list[CheckResult]:
    """Convert a DQD results file (``CheckResults`` list) into ``CheckResult``s.

    Tolerant of missing keys and of R's ``NA`` values.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("CheckResults") if isinstance(data, dict) else data
    catalog = check_catalog()
    out: list[CheckResult] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        name = _text(r.get("checkName")) or "unknown"
        level = _text(r.get("checkLevel")).upper()
        if level not in {"TABLE", "FIELD", "CONCEPT"}:
            level = catalog[name].level if name in catalog else "FIELD"
        info = catalog.get(name)
        severity = _text(r.get("severity")).lower() or (info.severity if info else "convention")
        if severity not in {"fatal", "convention", "characterization"}:
            severity = info.severity if info else "convention"
        concept = _num(r.get("conceptId"), default=float("nan"))
        is_error = _flag(r.get("isError"))
        error = _text(r.get("error")) or ("DQD reported an error" if is_error else "")
        out.append(
            CheckResult(
                check_name=name,
                check_level=level,  # type: ignore[arg-type]
                table=_text(r.get("cdmTableName")).lower(),
                field=_text(r.get("cdmFieldName")).lower() or None,
                concept_id=None if concept != concept else int(concept),
                description=_text(r.get("checkDescription")) or (info.description if info else name),
                category=_text(r.get("category")) or (info.category if info else ""),
                subcategory=_text(r.get("subcategory")) or (info.subcategory if info else ""),
                severity=severity,  # type: ignore[arg-type]
                num_violated_rows=int(_num(r.get("numViolatedRows"))),
                num_denominator_rows=int(_num(r.get("numDenominatorRows"))),
                pct_violated_rows=_num(r.get("pctViolatedRows")),
                threshold_pct=_num(r.get("thresholdValue")),
                failed=_flag(r.get("failed")),
                not_applicable=_flag(r.get("notApplicable")),
                error=error or None,
                sql=_text(r.get("queryText")) or None,
            )
        )
    return out


def run_r_dqd(cfg: ProjectConfig, ws: Workspace, timeout: int = 7200) -> DQReport | None:
    """Run Achilles + DQD through Rscript. Returns None when R is unavailable or the run fails."""
    if not r_available(cfg):
        return None
    script = write_r_script(cfg, ws)
    results = dqd_results_path(ws)
    if results.exists():
        results.unlink()
    env = dict(os.environ)
    password = _connection(cfg).get("password") or os.environ.get("PGPASSWORD", "")
    env[PASSWORD_ENV] = password
    log = dqd_output_dir(ws) / "rscript.log"
    ws.audit("dqd_r_start", script=str(script.relative_to(ws.root)))
    try:
        proc = subprocess.run(
            [shutil.which(cfg.quality.rscript) or cfg.quality.rscript, str(script)],
            cwd=ws.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        returncode: int | str = proc.returncode
    except subprocess.TimeoutExpired:
        returncode = "timeout"
    except OSError as e:
        returncode = f"failed to start: {e}"
    if returncode != 0 or not results.exists():
        ws.audit("dqd_r_failed", returncode=returncode, log=str(log.relative_to(ws.root)))
        return None
    checks = parse_dqd_json(results, cfg)
    passed, blocking = gate(checks, cfg)
    ws.audit("dqd_r_done", checks=len(checks), failed=sum(c.failed for c in checks), passed=passed)
    return DQReport(run_at=utcnow(), engine="dqd", results=checks, passed=passed, blocking_failures=blocking)
