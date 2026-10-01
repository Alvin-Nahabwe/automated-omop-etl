"""Projects the web app serves: each sub-directory of the projects directory with an omop_agent.yaml."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from ..config import ProjectConfig, load_config
from ..db import normalize_url
from ..workspace import Workspace

PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CONFIG_NAME = "omop_agent.yaml"


@dataclass
class Project:
    id: str
    dir: Path
    cfg: ProjectConfig | None = None
    error: str | None = None

    @property
    def config_path(self) -> Path:
        return self.dir / CONFIG_NAME

    @property
    def name(self) -> str:
        return self.cfg.name if self.cfg else self.id

    @cached_property
    def ws(self) -> Workspace:
        if self.cfg is None:
            raise RuntimeError(f"project {self.id} has no valid configuration: {self.error}")
        return Workspace(self.cfg.workdir)


def _load(pid: str, d: Path) -> Project:
    try:
        return Project(id=pid, dir=d, cfg=load_config(d / CONFIG_NAME))
    except Exception as e:  # a broken config is shown on the page, not a server error
        return Project(id=pid, dir=d, error=f"{type(e).__name__}: {str(e)[:400]}")


def _written_schemas(cfg: ProjectConfig) -> set[str]:
    t = cfg.target
    return {t.cdm_schema, t.staging_schema, t.etl_schema, t.results_schema}


def _all_schemas(cfg: ProjectConfig) -> set[str]:
    return _written_schemas(cfg) | {cfg.target.vocab_schema}


def _same_database(a: ProjectConfig, b: ProjectConfig) -> bool:
    def key(url: str) -> str:
        return normalize_url(url).split("@")[-1].split("?")[0].lower()  # host:port/dbname

    return key(a.target.url) == key(b.target.url)


def schema_conflicts_for(project_dir: Path, cfg: ProjectConfig) -> list[str]:
    """Other projects (next to ``project_dir``) whose database schemas overlap this one's.

    Two projects that write to the same schemas would overwrite each other's staged data and
    CDM. A vocabulary schema that neither project writes to may be shared.
    """
    project_dir = Path(project_dir).resolve()
    out = []
    for other in list_projects(project_dir.parent):
        if other.cfg is None or other.dir.resolve() == project_dir or not _same_database(cfg, other.cfg):
            continue
        shared = (_written_schemas(cfg) & _all_schemas(other.cfg)) | (_all_schemas(cfg) & _written_schemas(other.cfg))
        if shared:
            out.append(f"{other.id}: {', '.join(sorted(shared))}")
    return out


def list_projects(root: Path) -> list[Project]:
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    return [
        _load(d.name, d)
        for d in sorted(root.iterdir(), key=lambda p: p.name.lower())
        if PROJECT_ID.match(d.name) and d.resolve().parent == root and (d / CONFIG_NAME).is_file()
    ]


def get_project(root: Path, pid: str) -> Project | None:
    if not PROJECT_ID.match(pid):
        return None
    root = Path(root).resolve()
    d = (root / pid).resolve()
    if d.parent != root or not (d / CONFIG_NAME).is_file():
        return None
    return _load(pid, d)
