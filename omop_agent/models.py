"""Data contracts shared by every pipeline stage.

These models are the hand-off format between stages and the review surface for humans: each
one is serialised to the workspace (JSON or YAML) so a person can inspect and edit it between
steps. Models that the LLM fills in (``SourceAnnotations``, ``TablePlan``, ``TableMapping``,
``MappingBatchDecision``) are also used directly as structured-output schemas, so keep their
field descriptions precise: they are part of the prompt.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Stage 1: source profile ("White Rabbit")
# --------------------------------------------------------------------------------------


class ValueFreq(BaseModel):
    value: str
    count: int


class ColumnProfile(BaseModel):
    name: str
    declared_type: str | None = Field(None, description="Type reported by the source database, if any.")
    inferred_type: Literal["integer", "float", "boolean", "date", "datetime", "string", "empty"]
    n_rows: int
    n_null: int = Field(..., description="NULL or empty-string values.")
    n_distinct: int
    unique: bool = Field(..., description="Every non-null value is distinct.")
    min_value: str | None = None
    max_value: str | None = None
    mean: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    date_format: str | None = Field(None, description="strftime pattern detected for date-like strings.")
    top_values: list[ValueFreq] = Field(
        default_factory=list,
        description="Most frequent values; values seen fewer than min_cell_count times are removed.",
    )
    n_suppressed_values: int = Field(0, description="Distinct values hidden by small-cell suppression.")
    is_pii: bool = False
    pii_reason: str | None = None


class TableProfile(BaseModel):
    name: str
    n_rows: int
    columns: list[ColumnProfile]

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


class SourceProfile(BaseModel):
    source_type: Literal["csv", "database"]
    generated_at: datetime
    min_cell_count: int
    tables: list[TableProfile]

    def table(self, name: str) -> TableProfile | None:
        return next((t for t in self.tables if t.name == name), None)


# --------------------------------------------------------------------------------------
# Stage 1b: LLM interpretation of the profile
# --------------------------------------------------------------------------------------

SemanticType = Literal[
    "person_identifier",
    "record_identifier",
    "foreign_key",
    "date",
    "datetime",
    "coded",
    "categorical",
    "numeric_measurement",
    "free_text",
    "pii",
    "other",
]


class ColumnAnnotation(BaseModel):
    table: str
    column: str
    semantic_type: SemanticType
    coding_system: str | None = Field(
        None,
        description="For coded columns: the OMOP vocabulary_id if the codes belong to a standard "
        "vocabulary (e.g. 'ICD10', 'LOINC', 'RxNorm'), or 'local' for site-specific codes/labels.",
    )
    description: str = Field(..., description="One sentence on what the column holds.")


class TableAnnotation(BaseModel):
    table: str
    description: str
    grain: str = Field(..., description="What one row represents, e.g. 'one diagnosis per visit'.")
    person_key: str | None = Field(None, description="Column that identifies the patient, if any.")
    candidate_cdm_tables: list[str] = Field(default_factory=list)


class SourceAnnotations(BaseModel):
    tables: list[TableAnnotation]
    columns: list[ColumnAnnotation]
    notes: list[str] = Field(default_factory=list, description="Data issues or questions for the data owner.")


# --------------------------------------------------------------------------------------
# Stage 2: ETL design ("Rabbit in a Hat")
# --------------------------------------------------------------------------------------

FieldKind = Literal[
    "column",
    "expression",
    "constant",
    "concept_map",
    "vocab_code",
    "person_id",
    "visit_id",
    "auto_id",
    "null",
]


class FieldMapping(BaseModel):
    """How one CDM field is populated.

    kinds:
      column       expression is a qualified source column, e.g. ``p.sex``
      expression   any PostgreSQL expression over the source aliases
      constant     expression is a SQL literal, e.g. ``32817`` or ``'1970-01-01'``
      concept_map  local codes/labels mapped through the project's source_to_concept_map:
                   expression = the source code/label, vocabulary = the source_vocabulary_id to
                   create for it (e.g. 'SRC_DRUG'), target_domain = OMOP domain of the targets.
                   Yields the standard target_concept_id (0 when unmapped); with
                   concept_role='source' it yields the map's source_concept_id instead.
      vocab_code   codes that already belong to an OMOP vocabulary: expression = the code as it
                   appears in the vocabulary's concept_code (normalise it in SQL if needed),
                   vocabulary = its vocabulary_id (e.g. 'ICD10'). concept_role='standard' follows
                   'Maps to' to the standard concept; concept_role='source' returns the source
                   concept itself (for *_source_concept_id fields).
      person_id    expression = the source patient key; resolved to person_id via the person
                   id map. Rows whose key has no person are dropped.
      visit_id     expression = the source visit key; resolved via the visit id map. When the CDM
                   field is optional the row is kept with NULL; when it is required (the
                   visit_occurrence primary key) rows without a visit are dropped.
      auto_id      a generated surrogate key (only for the table's own primary key; never for
                   person.person_id or visit_occurrence.visit_occurrence_id, which must come from
                   the id maps so that events link to the right person and visit).
      null         leave the field NULL.
    """

    model_config = ConfigDict(extra="forbid")

    target_field: str
    kind: FieldKind
    expression: str | None = None
    vocabulary: str | None = None
    target_domain: str | None = None
    concept_role: Literal["standard", "source"] = "standard"
    rationale: str = ""


class SourceTableRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    alias: str


class JoinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    alias: str
    join_type: Literal["inner", "left"] = "left"
    on: str = Field(..., description="SQL join condition using the aliases.")


class TableMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_table: str
    source: SourceTableRef
    joins: list[JoinSpec] = Field(default_factory=list)
    where: str | None = Field(None, description="Optional SQL filter over the aliases.")
    fields: list[FieldMapping]
    rationale: str = ""


class KeySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str = Field(..., description="Source table whose rows define the entity.")
    expression: str = Field(..., description="SQL expression of the key, over the bare column names.")


class CustomSQL(BaseModel):
    """Escape hatch for logic that does not fit a single INSERT ... SELECT."""

    model_config = ConfigDict(extra="forbid")

    name: str
    target_table: str
    sql: str = Field(..., description="Complete SQL using {cdm}, {src}, {vocab}, {etl} schema placeholders.")
    rationale: str = ""


class ETLSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    cdm_version: str = "5.4"
    person_key: KeySpec
    visit_key: KeySpec | None = None
    observation_period: Literal["from_events", "table_mapping"] = Field(
        "from_events",
        description="'from_events' derives one period per person from min/max event dates; "
        "'table_mapping' means a TableMapping for observation_period is provided.",
    )
    observation_period_type_concept_id: int = 32817
    table_mappings: list[TableMapping]
    custom_sql: list[CustomSQL] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    def mapping_for(self, target_table: str) -> TableMapping | None:
        return next((m for m in self.table_mappings if m.target_table == target_table), None)


class PlannedTable(BaseModel):
    target_table: str
    source_tables: list[str]
    rationale: str


class TablePlan(BaseModel):
    """First design pass: which source tables feed which CDM tables."""

    person_key: KeySpec
    visit_key: KeySpec | None = None
    observation_period: Literal["from_events", "table_mapping"] = "from_events"
    tables: list[PlannedTable]
    unmapped_source_tables: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Stage 3: vocabulary mapping ("Usagi")
# --------------------------------------------------------------------------------------

Equivalence = Literal["EQUAL", "EQUIVALENT", "WIDER", "NARROWER", "INEXACT", "UNMATCHED"]
MappingStatus = Literal["APPROVED", "UNCHECKED", "FLAGGED", "REJECTED"]


class ConceptCandidate(BaseModel):
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str | None
    concept_code: str
    score: float = 0.0


class SourceTerm(BaseModel):
    source_vocabulary_id: str
    source_code: str
    source_name: str
    frequency: int
    target_domain: str | None = None


class MappingDecision(BaseModel):
    source_vocabulary_id: str
    source_code: str
    source_name: str
    frequency: int
    target_domain: str | None = None
    target_concept_id: int = 0
    target_concept_name: str | None = None
    target_domain_id: str | None = None
    target_vocabulary_id: str | None = None
    equivalence: Equivalence = "UNMATCHED"
    confidence: float = 0.0
    status: MappingStatus = "UNCHECKED"
    method: Literal["exact", "lexical", "llm", "manual", "suppressed"] = "lexical"
    rationale: str = ""
    candidates: list[ConceptCandidate] = Field(default_factory=list)


class TermChoice(BaseModel):
    """LLM output for one source term."""

    source_code: str
    chosen_concept_id: int = Field(..., description="concept_id from the candidate list, or 0 if none fits.")
    equivalence: Equivalence
    confidence: float = Field(..., ge=0, le=1)
    rationale: str
    retry_search_terms: list[str] = Field(
        default_factory=list,
        description="If no candidate fits, up to 3 better search phrases (synonyms, expanded "
        "abbreviations, generic drug names) to search the vocabulary again.",
    )


class MappingBatchDecision(BaseModel):
    choices: list[TermChoice]


# --------------------------------------------------------------------------------------
# Stage 5: data quality ("Achilles / DQD")
# --------------------------------------------------------------------------------------


class CheckResult(BaseModel):
    check_name: str
    check_level: Literal["TABLE", "FIELD", "CONCEPT", "ETL"]
    table: str
    field: str | None = None
    concept_id: int | None = None
    description: str
    category: str = ""
    subcategory: str = ""
    severity: Literal["fatal", "convention", "characterization"] = "convention"
    num_violated_rows: int = 0
    num_denominator_rows: int = 0
    pct_violated_rows: float = Field(0.0, description="Fraction 0..1 of the denominator that violates the check.")
    threshold_pct: float = Field(0.0, description="Failure threshold in percent (0..100), as in DQD.")
    failed: bool = False
    not_applicable: bool = False
    error: str | None = None
    sql: str | None = None

    @property
    def key(self) -> str:
        parts = [self.check_name, self.table, self.field or "", str(self.concept_id or "")]
        return ".".join(p for p in parts if p)


class DQReport(BaseModel):
    run_at: datetime
    engine: Literal["builtin", "dqd"] = "builtin"
    results: list[CheckResult]
    passed: bool
    blocking_failures: list[str] = Field(
        default_factory=list, description="One line per blocking failure: '<check key> (<severity>): <detail>'."
    )

    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.failed or r.error]


# --------------------------------------------------------------------------------------
# ETL execution
# --------------------------------------------------------------------------------------


class TableLoadResult(BaseModel):
    target_table: str
    step: str
    rows_inserted: int = 0
    source_rows: int | None = Field(None, description="Source rows after the mapping's joins and WHERE.")
    retained_rows: int | None = Field(
        None,
        description="Source rows that survive the person/visit id joins (before vocabulary fan-out); "
        "source_rows - retained_rows is the number of records dropped.",
    )
    seconds: float = 0.0
    error: str | None = None
    sql: str | None = None


class ETLRunResult(BaseModel):
    started_at: datetime
    finished_at: datetime | None = None
    steps: list[TableLoadResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.error is None for s in self.steps)

    def errors(self) -> list[TableLoadResult]:
        return [s for s in self.steps if s.error]


class RepairAction(BaseModel):
    iteration: int
    target_table: str
    target_field: str | None = None
    change: str
    rationale: str
    trigger: str = Field(..., description="Check key or ETL error that prompted the change.")
