"""Mechanical fixes to a drafted table mapping, applied before it is validated.

Each fix is one where the right answer follows from the CDM specification alone, so a model
(a small local one especially) does not spend a slow revision call on it. Every fix is noted,
and the notes go into the spec's assumptions for the reviewer:

* a missing primary key that is a plain surrogate key gets kind ``auto_id`` (never
  person.person_id or visit_occurrence.visit_occurrence_id, which come from the id maps);
* the ETL helper functions called as ``etl.try_number(...)`` get the ``{etl}`` placeholder
  (the ETL schema's name is configurable), unless the mapping uses ``etl`` as an alias;
* a foreign key to another CDM table set to the constant 0 (no such row exists) is left NULL,
  as the CDM conventions require for an unknown provider, care site or location.
"""

from __future__ import annotations

import re

from ..cdm.ddl import _DEFINED_FUNCTIONS
from ..cdm.spec import load_cdm_spec
from ..etl.compiler import ID_MAP_KEYS
from ..models import FieldMapping, TableMapping

_BARE_ETL = re.compile(r"(?<![\w.{])etl\s*\.\s*(" + "|".join(map(re.escape, _DEFINED_FUNCTIONS)) + r")\s*\(",
                       re.IGNORECASE) if _DEFINED_FUNCTIONS else None
_ZERO = re.compile(r"^\(?\s*'?0'?\s*\)?(::\s*\w+)?$")


def _placeholders(sql: str | None) -> str | None:
    if not sql or _BARE_ETL is None:
        return sql
    return _BARE_ETL.sub(lambda m: "{etl}." + m.group(1).lower() + "(", sql)


def tidy_mapping(tm: TableMapping) -> tuple[TableMapping, list[str]]:
    """``tm`` with the mechanical fixes applied, and a note for each."""
    cdm = load_cdm_spec()
    tt = tm.target_table
    if not cdm.has_table(tt):
        return tm, []
    table = cdm.table(tt)
    tm = tm.model_copy(deep=True)
    notes: list[str] = []

    mapped = {fm.target_field for fm in tm.fields}
    pk = table.primary_key
    if pk and pk not in mapped and (tt, pk) not in ID_MAP_KEYS and pk != "person_id":
        tm.fields.insert(0, FieldMapping(target_field=pk, kind="auto_id"))
        notes.append(f"{tt}.{pk}: the primary key was not mapped; it is generated (auto_id).")

    aliases = {tm.source.alias.lower()} | {j.alias.lower() for j in tm.joins}
    if "etl" not in aliases:
        fixed: list[str] = []
        for fm in tm.fields:
            new = _placeholders(fm.expression)
            if new != fm.expression:
                fm.expression = new
                fixed.append(fm.target_field)
        for j in tm.joins:
            new = _placeholders(j.on)
            if new != j.on:
                j.on = new  # type: ignore[assignment]
                fixed.append(f"join {j.alias}")
        for attr in ("where", "dedup_order"):
            new = _placeholders(getattr(tm, attr))
            if new != getattr(tm, attr):
                setattr(tm, attr, new)
                fixed.append(attr)
        if fixed:
            notes.append(f"{tt}: helper functions called as etl.<name>(...) now use the {{etl}} schema placeholder "
                         f"({', '.join(fixed)}).")

    for fm in tm.fields:
        f = table.field(fm.target_field)
        if (f is not None and f.is_fk and not f.is_concept_id and not f.required and f.fk_table
                and fm.kind == "constant" and _ZERO.match((fm.expression or "").strip())):
            fm.kind, fm.expression = "null", None
            notes.append(f"{tt}.{f.name}: the constant 0 refers to no {f.fk_table} row; it is left NULL.")
    return tm, notes


__all__ = ["tidy_mapping"]
