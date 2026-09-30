"""Project configuration (``omop_agent.yaml``).

A project is one source dataset being converted into one OMOP CDM instance. Paths in the
config file are resolved relative to the file's directory, and ``${VAR}`` / ``${VAR:-default}``
references are expanded from the environment so credentials never need to live in the file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Environment variable {name} is referenced in the config but not set")

    return _ENV_REF.sub(repl, value)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceConfig(_Base):
    """Where the patient-level source data lives."""

    type: Literal["csv", "database"]
    path: Path | None = Field(None, description="Directory of CSV files (type=csv). One file = one table.")
    url: str | None = Field(None, description="SQLAlchemy URL of the source database (type=database).")
    db_schema: str | None = Field(None, alias="schema", description="Source schema (type=database).")
    tables: list[str] | None = Field(None, description="Only these tables; default is all.")
    delimiter: str = ","
    encoding: str = "utf-8"
    skip_ragged_rows: bool = Field(
        False, description="Stage CSV files even if some rows have the wrong number of fields (those rows are left out and counted)."
    )

    @model_validator(mode="after")
    def _check(self) -> "SourceConfig":
        if self.type == "csv" and self.path is None:
            raise ValueError("source.path is required when source.type is 'csv'")
        if self.type == "database" and not self.url:
            raise ValueError("source.url is required when source.type is 'database'")
        return self


class TargetConfig(_Base):
    """The PostgreSQL database that receives the staged source copy and the CDM."""

    url: str = Field(..., description="PostgreSQL URL, e.g. postgresql://user:pass@host:5432/omop")
    cdm_schema: str = "cdm"
    vocab_schema: str = Field("cdm", description="Schema holding the Athena vocabulary tables.")
    staging_schema: str = Field("src", description="Raw copy of the source tables.")
    etl_schema: str = Field("etl", description="Working tables: id maps, source_to_concept_map.")
    results_schema: str = "results"
    etl_role: str | None = Field(
        None,
        description="Optional least-privilege role for running the generated ETL SQL (SET LOCAL ROLE): "
        "SELECT on the staging/vocabulary schemas, INSERT/TRUNCATE on the CDM and ETL schemas.",
    )

    @model_validator(mode="after")
    def _distinct_schemas(self) -> "TargetConfig":
        if len({self.cdm_schema, self.staging_schema, self.etl_schema}) < 3:
            raise ValueError("target.cdm_schema, staging_schema and etl_schema must be different schemas")
        if self.staging_schema in {self.vocab_schema, self.results_schema}:
            raise ValueError("target.staging_schema must not be the vocabulary or results schema")
        return self


class VocabularyConfig(_Base):
    athena_dir: Path | None = Field(None, description="Unzipped Athena download (CONCEPT.csv, ...).")


class LLMConfig(_Base):
    provider: Literal["anthropic", "replay", "none"] = "anthropic"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = 64000
    use_fallbacks: bool = Field(True, description="Opt into server-side refusal fallbacks.")
    replay_dir: Path | None = Field(None, description="Recorded responses used when provider=replay.")
    record_dir: Path | None = Field(None, description="If set, every live response is saved here for replay.")
    max_agent_iterations: int = 25
    agent_max_tokens: int = Field(16000, description="max_tokens per turn of the repair agent (non-streaming).")
    base_url: str | None = Field(None, description="Override the API endpoint (default: the SDK's).")
    api_key_env: list[str] = Field(
        default_factory=lambda: ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        description="Environment variables checked, in order, for the API key. If none is set the "
        "SDK's own resolution applies (ANTHROPIC_AUTH_TOKEN, `ant auth login` profile).",
    )


class PrivacyConfig(_Base):
    """Controls what leaves the machine. Only aggregate metadata is ever sent to the LLM."""

    min_cell_count: int = Field(5, description="Values seen fewer times than this are never shown to the LLM.")
    send_values_to_llm: bool = Field(True, description="If false, the LLM sees column statistics but no values.")
    max_values_per_column: int = 25
    pii_columns: list[str] = Field(default_factory=list, description="Extra 'table.column' entries to treat as PII.")
    pii_name_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(^|_)(first|last|middle|given|family|sur|full|maiden|patient|client|person|individual|member|"
            r"mother|mothers|father|fathers|guardian|kin|spouse|child)_?names?($|_)",
            r"^names?$|(^|_)name_of_",
            r"(^|_)(phone|telephone|tel|mobile|msisdn|cell|cellphone|contact|contact_?(no|number))($|_)",
            r"e_?mail",
            r"(^|_)(national_?id|passport|nhif|ssn|nin|mrn|id_?(no|num|number)|hospital_?(no|num|number)|"
            r"record_?(no|num|number)|file_?(no|num|number))($|_)",
            r"address|street|(^|_)gps|latitude|longitude|(^|_)(lat|lon|lng)($|_)",
            r"next_?of_?kin",
        ]
    )


class MappingConfig(_Base):
    auto_approve_confidence: float = Field(0.9, ge=0, le=1)
    candidates_per_term: int = 10
    llm_batch_size: int = 25
    use_unchecked_in_etl: bool = Field(
        True, description="Load UNCHECKED (not yet human-reviewed) mappings into the ETL; they are flagged in the report."
    )


class QualityConfig(_Base):
    max_repair_iterations: int = 3
    blocking_severities: list[str] = Field(default_factory=lambda: ["fatal"])
    repair_severities: list[str] = Field(default_factory=lambda: ["fatal", "convention"])
    use_r_dqd: bool = Field(False, description="Also run the official R DataQualityDashboard + Achilles.")
    rscript: str = "Rscript"


class CdmSourceConfig(_Base):
    cdm_source_name: str = "OMOP CDM"
    cdm_source_abbreviation: str = "CDM"
    cdm_holder: str = ""
    source_description: str = ""


class ProjectConfig(_Base):
    name: str
    workdir: Path = Path("work")
    source: SourceConfig
    target: TargetConfig
    vocabulary: VocabularyConfig = VocabularyConfig()
    llm: LLMConfig = LLMConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    mapping: MappingConfig = MappingConfig()
    quality: QualityConfig = QualityConfig()
    cdm_source: CdmSourceConfig = CdmSourceConfig()
    config_path: Path | None = Field(None, exclude=True)


def _expand_tree(node: object) -> object:
    if isinstance(node, str):
        return expand_env(node)
    if isinstance(node, list):
        return [_expand_tree(x) for x in node]
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    return node


def _resolve(base: Path, p: Path | None) -> Path | None:
    if p is None:
        return None
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = ProjectConfig.model_validate(_expand_tree(raw))
    base = path.parent
    cfg.config_path = path
    cfg.workdir = _resolve(base, cfg.workdir)  # type: ignore[assignment]
    cfg.source.path = _resolve(base, cfg.source.path)
    cfg.vocabulary.athena_dir = _resolve(base, cfg.vocabulary.athena_dir)
    cfg.llm.replay_dir = _resolve(base, cfg.llm.replay_dir)
    cfg.llm.record_dir = _resolve(base, cfg.llm.record_dir)
    return cfg
