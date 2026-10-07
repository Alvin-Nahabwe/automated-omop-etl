"""Project configuration (``omop_agent.yaml``).

A project is one source dataset being converted into one OMOP CDM instance. Paths in the
config file are resolved relative to the file's directory, and ``${VAR}`` / ``${VAR:-default}``
references are expanded from the environment so credentials never need to live in the file.
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


# Missing-value markers that CSV staging stores as NULL by default (source.na_values).
CSV_NA_VALUES = ("NA", "NULL", "\\N", ".")


class _Base(BaseModel):
    # Values may come from ${VAR} expansion (passwords in URLs): keep them out of error messages.
    model_config = ConfigDict(extra="forbid", populate_by_name=True, hide_input_in_errors=True)


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
    na_values: list[str] | None = Field(
        None,
        description="Values (after trimming, case-sensitive) that staging stores as NULL. Default: NA, NULL, \\N "
        "and '.' for CSV files; none for databases. Set [] to keep them as data.",
    )

    @model_validator(mode="after")
    def _check(self) -> "SourceConfig":
        if self.type == "csv" and self.path is None:
            raise ValueError("source.path is required when source.type is 'csv'")
        if self.type == "database" and not self.url:
            raise ValueError("source.url is required when source.type is 'database'")
        return self

    @property
    def effective_na_values(self) -> list[str]:
        """The configured ``na_values``, else the default for the source type."""
        if self.na_values is not None:
            return list(self.na_values)
        return list(CSV_NA_VALUES) if self.type == "csv" else []


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
        description="Optional least-privilege role for running the generated ETL SQL (SET LOCAL ROLE). It "
        "needs USAGE and SELECT on the staging and vocabulary schemas, USAGE plus SELECT, INSERT and DELETE on "
        "the CDM and ETL schemas, EXECUTE on the functions in the ETL schema (granted to PUBLIC by default), "
        "and the connecting user must be a member of it. Tables are emptied as the connecting user, not this role.",
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
    not_pii_columns: list[str] = Field(
        default_factory=list,
        description="'table.column' entries that the automatic identifier detection flagged by mistake "
        "(e.g. a CD4 cell count). They are treated as ordinary columns. pii_columns wins over this list.",
    )
    pii_name_patterns: list[str] = Field(
        default_factory=list,
        description="Extra regular expressions (matched against lower-case column names) for identifier "
        "columns, on top of the built-in rules in omop_agent/privacy.py.",
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
    blocking_checks: list[str] = Field(
        default_factory=lambda: ["measurePersonCompleteness.observation_period"],
        description="Checks that block the gate whatever their DQD severity: a check name "
        "(e.g. 'etlRecordRetention') or a check key prefix (e.g. 'measurePersonCompleteness."
        "observation_period': every person needs an observation period).",
    )
    min_record_retention_pct: float = Field(
        50.0, ge=0, le=100,
        description="A table mapping that loads less than this percentage of its source rows fails the "
        "fatal etlRecordLoss check (e.g. a broken person key). Smaller losses are reported by "
        "etlRecordRetention (threshold 5%) without blocking.",
    )
    repair_severities: list[str] = Field(default_factory=lambda: ["fatal", "convention"])
    use_r_dqd: bool = Field(False, description="Also run the official R DataQualityDashboard + Achilles.")
    require_r_dqd: bool = Field(
        False,
        description="With use_r_dqd: fail the quality gate when the R run is not possible or fails, "
        "instead of continuing with the built-in checks only.",
    )
    rscript: str = "Rscript"


class CdmSourceConfig(_Base):
    cdm_source_name: str = "OMOP CDM"
    cdm_source_abbreviation: str = "CDM"
    cdm_holder: str = ""
    source_description: str = ""
    source_release_date: date | None = Field(
        None,
        description="The data extract date. Written to cdm_source.source_release_date; observation periods "
        "still open at the extract (NULL exit date) end on it. Default: derived from the residency exit dates "
        "and reported.",
    )

    @field_validator("source_release_date", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        # ``${VAR:-}`` expands to an empty string
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("source_release_date", mode="after")
    @classmethod
    def _not_in_future(cls, v: date | None) -> date | None:
        # DQD checks cdm_release_date (the build date) plausibleTemporalAfter source_release_date, threshold 0
        if v is not None and v > date.today():
            raise ValueError(
                f"cdm_source.source_release_date {v.isoformat()} is in the future; use the date of the data extract"
            )
        return v


class EtlConfig(_Base):
    """How the build reads source values that need interpretation. Spec SQL always runs with
    DateStyle 'ISO, YMD' (``etl.session``): a day-first or month-first text date needs to_date
    with its template, since ``'31/12/2020'::date`` is an error whatever the server's DateStyle."""

    timezone: str | None = Field(
        None,
        description="IANA zone of the site, e.g. Africa/Nairobi. Set it when timestamps carry offsets or 'Z': "
        "they are then converted to this zone. Unset: an offset is ignored and the time is kept as written.",
    )
    partial_dates: Literal["widest", "reject"] = Field(
        "widest",
        description="Unknown day or month (00, 97-99, or a template without DD/MM) outside birth and death dates: "
        "'widest' = start dates take the first day, end dates the last day, single dates the 15th; 'reject' = NULL. "
        "Birth and death dates follow THEMIS.",
    )

    @field_validator("timezone", mode="before")
    @classmethod
    def _blank_zone_is_unset(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError):
            if _have_zone_database():
                raise ValueError(f"etl.timezone {v!r} is not an IANA time zone name (e.g. Africa/Nairobi)") from None
        return v


def _have_zone_database() -> bool:
    """False on a system without tz data, where PostgreSQL is left to check the zone name."""
    try:
        ZoneInfo("Africa/Nairobi")
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


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
    etl: EtlConfig = EtlConfig()
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
