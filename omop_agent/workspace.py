"""On-disk layout of a project's artifacts, plus the audit trail and stage state.

Everything the agent produces lands in ``<workdir>/`` so that each step can be reviewed,
edited by a human, re-run, and traced afterwards::

    work/
      state.json            stage status (lets `omop-agent run` resume)
      audit.jsonl           every LLM call, human approval, SQL error and repair
      profile.json/.md      stage 1  - source scan (aggregates only)
      annotations.json      stage 1b - LLM interpretation of the scan
      etl_spec.yaml/.md     stage 2  - ETL design (human-editable)
      mappings/*.csv        stage 3  - vocabulary mapping review files
      sql/*.sql             stage 4  - generated ETL SQL
      etl_run.json          stage 4  - load results
      etl_tables.json       stage 4  - the tables a build that has not finished may have written
      dq_results.json       stage 5  - quality checks
      repairs.json          stage 5b - changes the repair agent made
      report.md/.html       stage 6  - inspection report
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from .fsutil import atomic_write_text

T = TypeVar("T", bound=BaseModel)

STAGES = ["profile", "annotate", "design", "vocab_map", "build", "quality", "report"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "mappings").mkdir(exist_ok=True)
        (self.root / "sql").mkdir(exist_ok=True)
        self._lock = threading.Lock()

    # ---- paths -------------------------------------------------------------------------
    @property
    def profile_json(self) -> Path:
        return self.root / "profile.json"

    @property
    def profile_md(self) -> Path:
        return self.root / "profile.md"

    @property
    def annotations_json(self) -> Path:
        return self.root / "annotations.json"

    @property
    def spec_yaml(self) -> Path:
        return self.root / "etl_spec.yaml"

    @property
    def spec_md(self) -> Path:
        return self.root / "etl_spec.md"

    @property
    def mappings_dir(self) -> Path:
        return self.root / "mappings"

    @property
    def decisions_json(self) -> Path:
        return self.mappings_dir / "decisions.json"

    @property
    def llm_calls_dir(self) -> Path:
        return self.root / "llm_calls"

    @property
    def sql_dir(self) -> Path:
        return self.root / "sql"

    @property
    def etl_run_json(self) -> Path:
        return self.root / "etl_run.json"

    @property
    def etl_tables_json(self) -> Path:
        """The CDM tables a running build may write (it emptied them first); removed once the build
        saved etl_run.json. A build that was cancelled or crashed leaves it, so the next build
        empties those tables too, whatever its spec writes."""
        return self.root / "etl_tables.json"

    @property
    def dq_json(self) -> Path:
        return self.root / "dq_results.json"

    @property
    def repairs_json(self) -> Path:
        return self.root / "repairs.json"

    @property
    def vocab_eval_json(self) -> Path:
        return self.root / "vocab_evaluation.json"

    @property
    def report_md(self) -> Path:
        return self.root / "report.md"

    @property
    def report_html(self) -> Path:
        return self.root / "report.html"

    @property
    def audit_log(self) -> Path:
        return self.root / "audit.jsonl"

    @property
    def state_json(self) -> Path:
        return self.root / "state.json"

    @property
    def pending_json(self) -> Path:
        """What a stopped run was asked to redo (start over, redraft, map again)."""
        return self.root / "pending_run.json"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def run_lock(self) -> Path:
        """Held by whoever is running the pipeline on this project (see runlock.py)."""
        return self.jobs_dir / "run.lock"

    @property
    def run_holder(self) -> Path:
        return self.jobs_dir / "current"

    # ---- model I/O ---------------------------------------------------------------------
    def save_json(self, path: Path, model: BaseModel | list[BaseModel]) -> Path:
        if isinstance(model, list):
            data: Any = [m.model_dump(mode="json") for m in model]
        else:
            data = model.model_dump(mode="json")
        return atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    def load_json(self, path: Path, cls: type[T]) -> T:
        return cls.model_validate_json(path.read_text())

    def load_json_list(self, path: Path, cls: type[T]) -> list[T]:
        return [cls.model_validate(x) for x in json.loads(path.read_text())]

    def save_yaml(self, path: Path, model: BaseModel, header: str = "") -> Path:
        body = yaml.safe_dump(
            model.model_dump(mode="json", exclude_defaults=False),
            sort_keys=False,
            allow_unicode=True,
            width=110,
        )
        return atomic_write_text(path, (header + "\n" if header else "") + body)

    def load_yaml(self, path: Path, cls: type[T]) -> T:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    # ---- audit trail -------------------------------------------------------------------
    def audit(self, event: str, **fields: Any) -> None:
        record = {"ts": utcnow().isoformat(), "event": event, **fields}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock, self.audit_log.open("a") as fh:
            fh.write(line + "\n")

    def read_audit(self) -> list[dict[str, Any]]:
        if not self.audit_log.exists():
            return []
        return [json.loads(line) for line in self.audit_log.read_text().splitlines() if line.strip()]

    # ---- stage state -------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        if self.state_json.exists():
            return json.loads(self.state_json.read_text())
        return {"stages": {}}

    def mark(self, stage: str, status: str, **extra: Any) -> None:
        st = self.state()
        st["stages"][stage] = {"status": status, "at": utcnow().isoformat(), **extra}
        atomic_write_text(self.state_json, json.dumps(st, indent=2) + "\n")

    def stage_status(self, stage: str) -> str | None:
        return self.state()["stages"].get(stage, {}).get("status")
