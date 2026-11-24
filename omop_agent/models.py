"""Data contracts shared by every pipeline stage.

These models are the hand-off format between stages and the review surface for humans: each
one is serialised to the workspace (JSON or YAML) so a person can inspect and edit it between
steps. Models that the LLM fills in (``SourceAnnotations``, ``TablePlan``, ``TableMapping``,
``MappingBatchDecision``) are also used directly as structured-output schemas, so keep their
field descriptions precise: they are part of the prompt.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

# --------------------------------------------------------------------------------------
# Stage 1: source profile ("White Rabbit")
# --------------------------------------------------------------------------------------


class ValueFreq(BaseModel):
    value: str
    count: int


class ColumnProfile(BaseModel):
    name: str
    declared_type: str | None = Field(None, description="Type reported by the source database, if any.")
    inferred_type: Literal["integer", "float", "boolean", "date", "datetime", "time", "string", "empty"]
    n_rows: int
    n_null: int = Field(..., description="NULL or empty-string values (missing-value markers included).")
    n_distinct: int
    unique: bool = Field(..., description="Every non-null value is distinct.")
    min_value: str | None = Field(
        None, description="Smallest value; for dates and times the earliest complete, valid one (ISO)."
    )
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
    # Added with the tolerant date reading; the defaults keep older profile.json files loading.
    n_na_markers: int = Field(
        0, description="Values staged as NULL because they were a missing-value marker (source.na_values); "
        "part of n_null."
    )
    n_nonconforming: int = Field(
        0, description="Non-null values (whole column) that do not fit the inferred type or date format. As in "
        "the build, ISO dates fit every date format, any one of - / . _ may stand for a separator and a time may "
        "follow a date. Impossible dates (31/02) do not fit; partial dates are counted apart."
    )
    nonconforming_shapes: list[ValueFreq] = Field(
        default_factory=list,
        description="Shapes of the non-conforming values (digits as 9, letters as a / A, other characters "
        "kept), each seen at least min_cell_count times.",
    )
    n_partial_dates: int = Field(
        0, description="Dates with an unknown day or month (00, 97-99), without their day ('06/1975'), or "
        "only a year; the build completes or rejects them (etl.partial_dates)."
    )
    value_shapes: list[ValueFreq] = Field(
        default_factory=list,
        description="The most common shapes of the values (e.g. '99/99/9999 99:99'), each seen at least "
        "min_cell_count times. Never for direct identifiers.",
    )
    tz_offsets: list[ValueFreq] = Field(
        default_factory=list,
        description="Time zone offsets written after the times ('Z', '+03:00'), each seen at least "
        "min_cell_count times.",
    )
    n_rare_offsets: int = Field(
        0, description="Values whose offset is seen fewer than min_cell_count times (not in tz_offsets)."
    )
    format_notes: list[str] = Field(
        default_factory=list, description="How to read the values, where the type and format alone do not say."
    )


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
    # Who labelled the column: the model, or the rule-based fallback. Not part of the answer schema.
    origin: SkipJsonSchema[Literal["llm", "rules"]] = "rules"


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
                   id map. Rows whose key has no person are dropped. Every person mapping adds
                   the keys of its rows to the map (people can come from several tables).
      visit_id     expression = the source visit key; resolved via the visit id map. When the CDM
                   field is optional the row is kept with NULL; when it is required (the
                   visit_occurrence primary key) rows without a visit are dropped. A record
                   links only to a visit of its own person, and only when its key gives exactly
                   the text of the visit mapping's key.
      auto_id      a generated surrogate key (only for the table's own primary key; never for
                   person.person_id or visit_occurrence.visit_occurrence_id, which must come from
                   the id maps so that events link to the right person and visit).
      null         leave the field NULL.

    The value of a column or expression is converted to the CDM type, and its casts and
    to_date / to_timestamp calls are read tolerantly (``etl.tolerant``): a value that cannot be
    read becomes NULL and is counted, instead of failing the whole table. When a column,
    expression or constant of a required field yields NULL, the build applies the CDM convention
    and counts it (``etl.policies``): a concept becomes 0, visit_end_date, visit_detail_end_date
    and drug_exposure_end_date take the start date, and a row without any other required value (a
    start date, year_of_birth, death_date) is dropped.
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
    dedup_order: str | None = Field(
        None,
        description="Only for tables that keep one row per key (person and death: one per person; "
        "visit_occurrence: one per visit and person, spanning the earliest start to the latest end of its rows). "
        "A SQL ORDER BY list over the aliases that ranks the source rows of one key, best first, e.g. "
        "\"x.dod::date DESC\" (latest death) or \"(v.visit_type = 'IPD') DESC\". A missing or unreadable "
        "value ranks last, and rows whose required fields are NULL never win. Leave it empty to keep the most "
        "complete row (for death: a date known to the day, then the latest). It ranks this mapping's rows only: "
        "between the mappings of one table, the first in spec order that loads a key wins.",
    )
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
        description="'from_events': one period per person from the first to the last event date. 'table_mapping': "
        "mappings (or custom SQL) load one row per source episode (enrolment, residency, follow-up). Either way the "
        "build merges each person's overlapping and back-to-back periods; with 'table_mapping' it ends open periods "
        "(NULL end) at cdm_source.source_release_date or at death and gives persons without a valid episode a "
        "period from their events.",
    )
    observation_period_type_concept_id: int = 32817
    route_by_domain: bool = Field(
        True,
        description="After loading, move every condition/procedure/drug/device/measurement record whose standard "
        "concept belongs to another event domain (Condition, Procedure, Drug, Device, Measurement, Observation) to "
        "that domain's table. Measured results are never dropped.",
    )
    derive_eras: bool = Field(
        True,
        description="Build condition_era and drug_era with the OHDSI era logic. Spec SQL that loads an era table "
        "replaces the built-in step for that table.",
    )
    table_mappings: list[TableMapping]
    custom_sql: list[CustomSQL] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    def mapping_for(self, target_table: str) -> TableMapping | None:
        return next((m for m in self.table_mappings if m.target_table == target_table), None)


# Fields added to ETLSpec / TableMapping after design approvals were first recorded, with their
# defaults. The design-approval digest (approvals.py) leaves a field out while it holds this
# default, so a spec approved before the field existed keeps its approval. Every new defaulted
# field of ETLSpec or TableMapping must be registered here (tests pin the demo spec's digest).
DIGEST_IGNORED_DEFAULTS: dict[str, dict[str, Any]] = {
    "spec": {"derive_eras": True, "route_by_domain": True},
    "table_mapping": {"dedup_order": None},
}


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
    accepted_domains: list[str] = Field(
        default_factory=list,
        description="Domains a target concept may belong to (the record is moved to its domain's table); "
        "empty: target_domain only.",
    )
    auto_approve: bool = Field(
        True,
        description="False: the fields that use its source vocabulary expect concepts of different domains, so "
        "no concept is approved for it without a reviewer.",
    )
    units: list[ValueFreq] = Field(
        default_factory=list,
        description="Units recorded with this code (measurements/observations), each seen at least "
        "min_cell_count times; they tell the mapper e.g. mass vs molar concentration.",
    )


class MappingDecision(BaseModel):
    source_vocabulary_id: str
    source_code: str
    source_name: str
    frequency: int
    target_domain: str | None = None
    accepted_domains: list[str] = Field(default_factory=list)
    target_concept_id: int = 0
    target_concept_name: str | None = None
    target_domain_id: str | None = None
    target_vocabulary_id: str | None = None
    equivalence: Equivalence = "UNMATCHED"
    confidence: float = 0.0
    status: MappingStatus = "UNCHECKED"
    method: Literal["exact", "memory", "lexical", "llm", "manual", "suppressed"] = "lexical"
    rationale: str = ""
    candidates: list[ConceptCandidate] = Field(default_factory=list)
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    review_note: str = ""


class TermChoice(BaseModel):
    """LLM output for one source term."""

    source_code: str
    chosen_concept_id: int = Field(..., description="concept_id from the candidate list, or 0 if none fits.")
    equivalence: Equivalence
    confidence: float = Field(..., ge=0, le=1)
    rationale: str
    retry_search_terms: list[str] = Field(
        default_factory=list,
        description="If no candidate fits, or you chose the best listed one but a better concept "
        "probably exists, up to 3 search phrases (synonyms, expanded abbreviations, generic drug names, "
        "full test names) to search the vocabulary again.",
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
    step: str | None = Field(
        None, description="ETL checks: the step label without its number, e.g. 'person_2' or 'custom_fix_dates'."
    )
    repairable: bool | None = Field(
        None, description="False: never sent to the repair agent (it cannot fix the cause with the spec tools)."
    )

    @property
    def key(self) -> str:
        parts = [self.check_name, self.table, self.field or "", str(self.concept_id or "")]
        # a second mapping (or a custom step) into the same table gets its own ETL check key
        if self.check_level == "ETL" and self.step and self.step != self.table:
            parts.append(self.step)
        return ".".join(p for p in parts if p)


class DQReport(BaseModel):
    run_at: datetime
    engine: Literal["builtin", "dqd", "builtin+dqd"] = "builtin"
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


class RoutedRows(BaseModel):
    """Records a route step moved to the table of their concept's domain."""

    from_table: str
    to_table: str
    domain_id: str
    moved_rows: int = 0
    rows_with_dropped_fields: int = Field(
        0, description="Moved rows that had a value in a field the destination table has no place for."
    )


class TableLoadResult(BaseModel):
    target_table: str
    step: str
    kind: str | None = Field(
        None, description="id_map | insert | custom | prune | route | merge | derived | metadata (None: older results)."
    )
    blocking: bool = Field(True, description="False: an error is reported but does not fail the build (route, eras).")
    rows_inserted: int = 0
    source_rows: int | None = Field(None, description="Source rows after the mapping's joins and WHERE.")
    id_retained_rows: int | None = Field(
        None, description="Source rows that survive the INNER person/visit id-map joins."
    )
    retained_rows: int | None = Field(
        None,
        description="Source rows that survive the person/visit id joins and the required-field policy, before "
        "vocabulary fan-out and before de-duplication (id_retained_rows minus value_counts['row']['dropped']); "
        "source_rows - retained_rows is the number of records dropped.",
    )
    value_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Values not loaded as written: category (invalid, imputed, partial_null, offset_ignored, "
        "offset_converted, defaulted, required_null) -> field -> rows; value_counts['row']['dropped'] counts the "
        "rows dropped for a NULL required value.",
    )
    stats: dict[str, Any] = Field(
        default_factory=dict, description="Step statistics (one row per key, merge, fill, visit links)."
    )
    moved: list[RoutedRows] = Field(default_factory=list, description="Route steps: the records moved.")
    skipped: str | None = Field(None, description="Why the step was not run (a step it requires failed).")
    seconds: float = 0.0
    error: str | None = None
    repairable: bool = Field(
        True, description="False: the error is not one a change of the spec can fix (a missing privilege), so it is "
        "never sent to the repair agent.",
    )
    sql: str | None = None


class ETLRunResult(BaseModel):
    started_at: datetime
    finished_at: datetime | None = None
    steps: list[TableLoadResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No blocking step failed (errors of non-blocking steps are reported but do not fail the build)."""
        return all(s.error is None or not s.blocking for s in self.steps)

    def errors(self) -> list[TableLoadResult]:
        """Every step that failed, blocking or not (for logs and the report)."""
        return [s for s in self.steps if s.error]

    def blocking_errors(self) -> list[TableLoadResult]:
        """The failed steps that fail the build."""
        return [s for s in self.steps if s.error and s.blocking]

    def rows_loaded(self) -> int:
        """Rows the steps without an error loaded into the CDM from the source. The id-map and pruning
        steps write only the id maps (new and retired keys) and are left out, and so is the
        observation-period merge, whose periods replace the episodes its table's writers loaded, and
        so are the route steps, which only move records already loaded (``rows_moved``) and keep
        their bookkeeping. The rows the build derived from other CDM tables are counted apart
        (``rows_derived``)."""
        return sum(s.rows_inserted for s in self.steps
                   if s.error is None and s.kind not in ("id_map", "prune", "merge", "route", "derived"))

    def rows_moved(self) -> int:
        """Records the route steps moved to the table of their concept's domain (they are counted in
        ``rows_loaded`` by the steps that loaded them)."""
        return sum(m.moved_rows for s in self.steps if s.error is None for m in s.moved)

    def rows_derived(self) -> int:
        """The rows the build derived from other CDM tables (observation periods from events, the fill
        of persons without a period, the eras), apart from ``rows_loaded``."""
        return sum(s.rows_inserted for s in self.steps if s.error is None and s.kind == "derived")


class RepairAction(BaseModel):
    iteration: int
    target_table: str
    target_field: str | None = None
    change: str
    rationale: str
    trigger: str = Field(..., description="Check key or ETL error that prompted the change.")
    reverted: bool = Field(False, description="Rolled back because its build was worse than the one before.")
