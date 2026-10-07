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
"""

from __future__ import annotations

from typing import Literal

from ..cdm.spec import CdmField, load_cdm_spec

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
