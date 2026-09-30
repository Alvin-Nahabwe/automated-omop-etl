"""OMOP CDM v5.4 specification, loaded from the official OHDSI metadata bundled in resources/ohdsi."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

RESOURCE_PKG = "omop_agent.resources.ohdsi"

# Tables the ETL writes, in dependency order (targets referenced by FKs come first).
LOAD_ORDER = [
    "location",
    "care_site",
    "provider",
    "person",
    "visit_occurrence",
    "visit_detail",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "device_exposure",
    "measurement",
    "observation",
    "death",
    "note",
    "specimen",
    "payer_plan_period",
    "cost",
    "observation_period",
    "condition_era",
    "drug_era",
    "dose_era",
    "episode",
    "episode_event",
    "fact_relationship",
    "metadata",
    "cdm_source",
]

# Event tables and the date field that places each record in time (used for observation periods
# and temporal checks).
EVENT_DATE_FIELDS = {
    "visit_occurrence": ("visit_start_date", "visit_end_date"),
    "visit_detail": ("visit_detail_start_date", "visit_detail_end_date"),
    "condition_occurrence": ("condition_start_date", "condition_end_date"),
    "drug_exposure": ("drug_exposure_start_date", "drug_exposure_end_date"),
    "procedure_occurrence": ("procedure_date", "procedure_end_date"),
    "device_exposure": ("device_exposure_start_date", "device_exposure_end_date"),
    "measurement": ("measurement_date", None),
    "observation": ("observation_date", None),
    "death": ("death_date", None),
    "specimen": ("specimen_date", None),
    "note": ("note_date", None),
}

# The OMOP domain each event table's main concept field is expected to belong to.
TABLE_DOMAIN = {
    "visit_occurrence": "Visit",
    "visit_detail": "Visit",
    "condition_occurrence": "Condition",
    "drug_exposure": "Drug",
    "procedure_occurrence": "Procedure",
    "device_exposure": "Device",
    "measurement": "Measurement",
    "observation": "Observation",
    "specimen": "Specimen",
}


def _flag(v: str | None) -> bool:
    return (v or "").strip().lower() in {"true", "yes", "1"}


def _na(v: str | None) -> str | None:
    v = (v or "").strip()
    return None if v in {"", "NA"} else v


@dataclass
class CdmField:
    table: str
    name: str
    required: bool
    datatype: str
    description: str
    conventions: str
    is_pk: bool
    is_fk: bool
    fk_table: str | None
    fk_field: str | None
    fk_domain: str | None
    fk_class: str | None

    @property
    def is_concept_id(self) -> bool:
        return self.name.endswith("_concept_id")


@dataclass
class CdmTable:
    name: str
    schema_class: str  # CDM | VOCAB | RESULTS
    required: bool
    concept_prefix: str | None
    description: str
    conventions: str
    fields: list[CdmField] = field(default_factory=list)

    def field(self, name: str) -> CdmField | None:
        return next((f for f in self.fields if f.name == name), None)

    @property
    def primary_key(self) -> str | None:
        return next((f.name for f in self.fields if f.is_pk), None)

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


class CdmSpec:
    def __init__(self, tables: dict[str, CdmTable]):
        self.tables = tables

    def table(self, name: str) -> CdmTable:
        try:
            return self.tables[name.lower()]
        except KeyError:
            raise KeyError(f"'{name}' is not an OMOP CDM v5.4 table") from None

    def has_table(self, name: str) -> bool:
        return name.lower() in self.tables

    @property
    def cdm_tables(self) -> list[CdmTable]:
        return [t for t in self.tables.values() if t.schema_class == "CDM"]

    @property
    def vocab_tables(self) -> list[CdmTable]:
        return [t for t in self.tables.values() if t.schema_class == "VOCAB"]

    def describe(self, name: str, max_chars: int = 400) -> str:
        """Compact text description of one table, for prompts."""
        t = self.table(name)
        lines = [f"TABLE {t.name}: {t.description[:max_chars]}"]
        if t.conventions:
            lines.append(f"  ETL conventions: {t.conventions[:max_chars]}")
        for f in t.fields:
            flags = []
            if f.required:
                flags.append("required")
            if f.is_pk:
                flags.append("primary key")
            if f.fk_domain:
                flags.append(f"domain={f.fk_domain}")
            elif f.is_fk and f.fk_table:
                flags.append(f"fk->{f.fk_table.lower()}")
            guidance = (f.conventions or f.description or "").replace("\n", " ")[:max_chars]
            lines.append(f"  - {f.name} ({f.datatype}{', ' + ', '.join(flags) if flags else ''}): {guidance}")
        return "\n".join(lines)


def _read_csv(name: str) -> list[dict[str, str]]:
    text = resources.files(RESOURCE_PKG).joinpath(name).read_text(encoding="utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


@lru_cache(maxsize=1)
def load_cdm_spec() -> CdmSpec:
    tables: dict[str, CdmTable] = {}
    for r in _read_csv("OMOP_CDMv5.4_Table_Level.csv"):
        name = r["cdmTableName"].strip().lower()
        tables[name] = CdmTable(
            name=name,
            schema_class=r["schema"].strip().upper(),
            required=_flag(r["isRequired"]),
            concept_prefix=_na(r.get("conceptPrefix")),
            description=(r.get("tableDescription") or "").strip(),
            conventions=(r.get("etlConventions") or "").strip() if _na(r.get("etlConventions")) else "",
        )
    for r in _read_csv("OMOP_CDMv5.4_Field_Level.csv"):
        tname = r["cdmTableName"].strip().lower()
        if tname not in tables:
            continue
        tables[tname].fields.append(
            CdmField(
                table=tname,
                name=r["cdmFieldName"].strip().lower(),
                required=_flag(r["isRequired"]),
                datatype=r["cdmDatatype"].strip(),
                description=(_na(r.get("userGuidance")) or ""),
                conventions=(_na(r.get("etlConventions")) or ""),
                is_pk=_flag(r["isPrimaryKey"]),
                is_fk=_flag(r["isForeignKey"]),
                fk_table=(_na(r.get("fkTableName")) or "").lower() or None,
                fk_field=(_na(r.get("fkFieldName")) or "").lower() or None,
                fk_domain=_na(r.get("fkDomain")),
                fk_class=_na(r.get("fkClass")),
            )
        )
    return CdmSpec(tables)
