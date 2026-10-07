"""What the build does with a required CDM value that is NULL.

A required field mapped as a column, expression or constant can still yield NULL for some rows
(a missing discharge date, a sex code the CASE does not list). Rather than failing the whole
table on the NOT NULL constraint, the compiled INSERT follows the CDM convention for the field
and counts every row it touched (``etl.compiler``, layer 2):

* ``zero``: a required concept becomes 0 ("No matching concept"), counted as ``defaulted``;
* ``start``: an end date the CDM conventions allow to equal the start (``visit_end_date``,
  ``visit_detail_end_date``, ``drug_exposure_end_date``) takes the row's start date, counted as
  ``defaulted``;
* ``open``: ``observation_period_end_date`` means "still observed". Until the build merges
  observation periods and closes open ones at the data extract date, it is left as written, so
  a NULL end fails the load as before;
* ``drop``: every other required field (start and event dates, ``year_of_birth``,
  ``death_date``, ``note_text``, ...): the row is dropped, counted per field as
  ``required_null`` and once per row as ``row.dropped``, so the record retention checks see it.

The policies are derived from the CDM v5.4 specification (``cdm.spec.load_cdm_spec``).

``partial_policy`` says how a date with an unknown day or month (00, 97-99, or a template
without DD / MM) is read for one field, by the conversions the compiler writes (``etl.tolerant``):
birth and death fields follow THEMIS (the birth year and month are kept, a missing birth day is
the 1st, a year-only birth date gives no birth_datetime; a death date takes the last day of the
known period); the start of a DQD start/end pair takes the first day and its end the last day
(the widest interval, which cannot invert a correctly ordered pair); other dates take the 15th.
``etl.partial_dates: reject`` makes every non-THEMIS date NULL instead. Fields that are not dates
reject partial dates: a number computed from one would be a guess.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib import resources
from typing import Literal

from ..cdm.spec import RESOURCE_PKG, CdmField, load_cdm_spec
from ..config import EtlConfig

Policy = Literal["zero", "start", "open", "drop"]

# end date -> the start date it takes when it is NULL (CDM conventions: "If the end date is not
# known, use the start date")
END_FROM_START: dict[tuple[str, str], str] = {
    ("visit_occurrence", "visit_end_date"): "visit_start_date",
    ("visit_detail", "visit_detail_end_date"): "visit_detail_start_date",
    ("drug_exposure", "drug_exposure_end_date"): "drug_exposure_start_date",
}
# NULL means the person is still observed
OPEN_ENDS = frozenset({("observation_period", "observation_period_end_date")})
# The field kinds the policies apply to. The others cannot be NULL (id maps, auto_id), already
# give a required concept 0 (vocab_code, concept_map), or are refused by the validator (null).
POLICY_KINDS = frozenset({"column", "expression", "constant"})
# The counter site of the rows dropped for a NULL required value: value_counts['row']['dropped'].
ROW_SITE = "row"


def _is_concept(f: CdmField) -> bool:
    # fact_relationship.domain_concept_id_1/_2 reference concept too
    return f.is_concept_id or f.fk_table == "concept"


def _required_null_policies() -> dict[str, dict[str, Policy]]:
    out: dict[str, dict[str, Policy]] = {}
    for table in load_cdm_spec().cdm_tables:
        for f in table.fields:
            if not f.required:
                continue
            key = (table.name, f.name)
            policy: Policy
            if key in END_FROM_START:
                policy = "start"
            elif key in OPEN_ENDS:
                policy = "open"
            elif _is_concept(f):
                policy = "zero"
            else:
                policy = "drop"
            out.setdefault(table.name, {})[f.name] = policy
    return out


# CDM table -> required field -> policy
REQUIRED_NULL_POLICY: dict[str, dict[str, Policy]] = _required_null_policies()


def required_null_policy(table: str, field: str) -> Policy | None:
    """The policy of a required field, None for an optional one (it stays NULL)."""
    return REQUIRED_NULL_POLICY.get(table, {}).get(field)


# --------------------------------------------------------------------------------------
# partial dates
# --------------------------------------------------------------------------------------

PartialPolicy = Literal["first", "last", "mid", "birth", "month", "year", "reject"]
# The counter site of the mapping's WHERE.
WHERE_SITE = "where"
# THEMIS conventions for incomplete birth and death dates.
THEMIS_PARTIAL: dict[tuple[str, str], PartialPolicy] = {
    ("person", "year_of_birth"): "year",
    ("person", "month_of_birth"): "month",
    ("person", "day_of_birth"): "reject",
    ("person", "birth_datetime"): "birth",
    ("death", "death_date"): "last",
    ("death", "death_datetime"): "last",
}


@lru_cache(maxsize=1)
def start_end_pairs() -> dict[tuple[str, str], str]:
    """(table, start field) -> its end field, for the DQD start/end pairs (plausibleStartBeforeEnd)
    of the CDM tables, e.g. ("visit_occurrence", "visit_start_date") -> "visit_end_date"."""
    cdm = load_cdm_spec()
    text = resources.files(RESOURCE_PKG).joinpath("DQD_OMOP_CDMv5.4_Field_Level.csv").read_text(encoding="utf-8")
    out: dict[tuple[str, str], str] = {}
    for row in csv.DictReader(text.splitlines()):
        table, start = row["cdmTableName"].lower(), row["cdmFieldName"].lower()
        end = (row.get("plausibleStartBeforeEndFieldName") or "").strip().lower()
        if row.get("plausibleStartBeforeEnd", "").strip() != "Yes" or not end or end == start:
            continue
        if cdm.has_table(table) and cdm.table(table).schema_class == "CDM":
            out[(table, start)] = end
    return out


def partial_policy(table: str, field: str, cfg: EtlConfig | None = None) -> PartialPolicy:
    """How a date with an unknown day or month is read for ``table.field`` (or for the mapping's
    WHERE, ``field == WHERE_SITE``); see the module docstring."""
    key = (table, field)
    if key in THEMIS_PARTIAL:
        return THEMIS_PARTIAL[key]
    if cfg is not None and cfg.partial_dates == "reject":
        return "reject"
    if field == WHERE_SITE:
        return "mid"
    pairs = start_end_pairs()
    if key in pairs:
        return "first"
    if field in {end for (t, _), end in pairs.items() if t == table}:
        return "last"
    spec = load_cdm_spec()
    f = spec.table(table).field(field) if spec.has_table(table) else None
    if f is not None and f.datatype.strip().lower() in ("date", "datetime", "timestamp"):
        return "mid"
    return "reject"
