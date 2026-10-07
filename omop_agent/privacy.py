"""Privacy guard: decides what source information may be shown to the LLM.

Rules (enforced by construction, not by prompt):
  * the LLM only ever sees aggregate metadata - table/column names, types, counts, ranges,
    and value frequencies; never a patient row;
  * values seen fewer than ``min_cell_count`` times are suppressed everywhere they could reach
    the LLM (profile top values, vocabulary terms, repair-agent query results);
  * columns detected as direct identifiers (names, phone numbers, national and register numbers,
    addresses, GPS, free text) get no values, no min/max, only counts; detection is heuristic, so
    a person reviews what will be sent before the first live call (the "privacy" review gate);
  * ``send_values_to_llm: false`` removes all values and keeps only statistics.
"""

from __future__ import annotations

import re
from typing import Iterable

from .config import PrivacyConfig
from typing import TYPE_CHECKING

from .models import ColumnProfile, SourceProfile, TableProfile, ValueFreq

if TYPE_CHECKING:
    from .models import SourceAnnotations, TableMapping

_DATE_LIKE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
# Kenyan numbers (07.., 01.., +254 7.., also with the leading 0 lost in a spreadsheet) and any
# international number written with a leading '+'.
_PHONE = re.compile(r"^(\+?254|0)?[17]\d{8}$|^\+\d{9,15}$")
_PHONE_SEPARATORS = re.compile(r"[\s\-().]")
# One or more coordinates: "-1.2921 36.8219", "-1.2921,36.8219", ODK geopoints "-1.29 36.82 1650.0 5.0".
_GPS = re.compile(r"^-?\d{1,3}\.\d{3,}[\s,;]+-?\d{1,3}\.\d{3,}([\s,;]+-?\d+(\.\d+)?){0,2}$")
# Column names of keys and codes: their long numbers are ids, not phone numbers...
_ID_LIKE_NAME = re.compile(r"(^|_)(id|uuid|guid|key|no|num|number|code|ref)$|[a-z]id$")
# ...unless the name says it is a phone number ("cell_no", "contact_number"; "contact_id" stays a key).
_PHONE_WORD = re.compile(r"(phone|mobile|cell|contact|(^|_)tel|msisdn).*(no|num|number|nr|ref)$")
# Free text: long values written as sentences. Coded labels (LOINC long names, drug strengths,
# ICD titles) are long too, but have few function words and no sentence punctuation.
FREE_TEXT_MIN_WORDS = 8
FREE_TEXT_MIN_LENGTH = 40
_FUNCTION_WORDS = frozenset(
    "a an the and but or so was were is are be been he she his her him they them their we our it its "
    "had has have with for to at on from that this who when after before because then not no "
    "na ya wa kwa ni la za alikuwa akiwa yeye baada kabla".split()
)
_SENTENCE_BREAK = re.compile(r"[a-z][.!?;:,]\s+[a-zA-Z]")

# Words that make "<word>_name" the name of a thing, not of a person.
_THING = (
    r"drug|medicine|medication|med|test|lab|facility|hospital|clinic|site|village|sublocation|location|loc|"
    r"district|county|subcounty|ward|region|province|division|area|place|estate|school|brand|generic|product|"
    r"item|diagnosis|disease|condition|cause|procedure|vaccine|antigen|unit|file|field|form|variable|event|"
    r"program|programme|study|project|organisation|organization|company|employer|round|team|language|tribe|"
    r"ethnicity|religion|occupation|relationship|method|catchment|cluster|zone|country|nationality|sheet|"
    r"table|column|package|app|device|sample|specimen|batch|kit|supplier|ward|unit"
)
_THING_WORD = re.compile(rf"^({_THING})s?$")
# The parts of a column name that name a person: "name", "first_name", "fname", "patientname"...
_NAME_TOKEN = re.compile(
    r"^(names?|[flmo]name|(first|last|sur|middle|given|family|maiden|nick|full|other|christian|baptismal|"
    r"patient|client|person|individual|member|respondent|mother|mothers|father|fathers|guardian|parent|kin|"
    r"spouse|husband|wife|partner|child|baby|infant|caregiver|caretaker|deceased|informant|household_?head|"
    r"head|owner|contact|witness)names?)$"
)
# "patient_name", "mother_s_name": a person word right before "name" settles it.
_PERSON_WORD = re.compile(
    r"^(first|last|sur|middle|given|family|maiden|nick|full|other|patient|client|person|individual|member|"
    r"respondent|mother|mothers|father|fathers|guardian|parent|kin|spouse|husband|wife|partner|child|baby|"
    r"infant|caregiver|caretaker|deceased|informant|head|owner|witness|s)$"
)


def _person_name(col: str) -> bool:
    """A column name that holds a person's name: a name part not qualified by a thing word
    ("facility_name", "drug_name", "name_of_village" are not people)."""
    tokens = [t for t in col.split("_") if t]
    for i, tok in enumerate(tokens):
        if not _NAME_TOKEN.match(tok):
            continue
        before = tokens[i - 1] if i > 0 else ""
        after = tokens[i + 2] if i + 2 < len(tokens) and tokens[i + 1] == "of" else ""
        if before and _THING_WORD.match(before) and not _PERSON_WORD.match(before):
            continue
        if after and _THING_WORD.match(after):
            continue
        return True
    return False


class _NameCheck:
    """A name rule that is a function rather than a pattern (same ``search`` interface)."""

    def __init__(self, fn):  # type: ignore[no-untyped-def]
        self.fn = fn

    def search(self, col: str) -> bool:
        return bool(self.fn(col))


# Built-in identifier rules: (what the column holds, a check on the lower-case column name).
NAME_RULES: list[tuple[str, "re.Pattern[str] | _NameCheck"]] = [
    ("a person's name", _NameCheck(_person_name)),
    ("a staff member's name", re.compile(
        r"(^|_)(seen_by|attended_by|interviewer|enumerator|field_?worker|clinician|recorded_by|entered_by|"
        r"reviewed_by|signed_by|data_?clerk|user_?name)$"
    )),
    ("a name question of the WHO verbal autopsy form", re.compile(r"(^|_)id10(007|017|018|061|062)($|_)")),
    ("a phone number", re.compile(
        r"(^|_)(phone|telephone|tel|msisdn|cellphone|contact_?(no|num|number|phone))($|_)"
        r"|(^|_)(mobile|cell)(_?(no|num|number|phone|nr))?$|phone"
        r"|(^|_)(emergency_?contact|contact_?(person|details|info))($|_)"
    )),
    ("an e-mail address", re.compile(r"e_?mail")),
    ("a national, health or register number", re.compile(
        r"(^|_)(national_?id|passport|nhif|ssn|nin|mrn|id_?(no|num|number)|kra_?pin|ccc|upi|nupi|huduma|"
        r"birth_?cert(ificate)?_?(no|num|number)?)($|_)"
        r"|(^|_)(hospital|record|file|ccc|opd|ipd|ip|anc|pnc|tb|art|hei|pmtct|register|registration|reg|clinic|"
        r"patient|inpatient|outpatient)_?(no|num|number|nr)($|_)"
    )),
    ("an address", re.compile(
        r"address|street|(^|_)(plot|house|stand|building|flat|apartment|door|gate|room)_?(no|num|number|nr)($|_)"
        r"|(^|_)(po_?box|postal_?(address|code)|post_?code|zip_?code|landmark|directions?)($|_)"
    )),
    ("GPS coordinates", re.compile(
        r"(^|_)gps|latitude|longitude|(^|_)(lat|lon|lng)($|_)|(^|_)long$|(^|_)(geo_?point|geo_?location|"
        r"coordinates?|coords?|lat_?long|lat_?lon|latlng|easting|northing|[xy]_?coord(inate)?s?|altitude)($|_)"
    )),
    ("a next-of-kin contact", re.compile(r"next_?of_?kin")),
    ("free text (notes and narratives can contain names)", re.compile(
        r"(^|_)(narrative|narration|comments?|remarks?|notes?|free_?text|specify|other_?specify|explain|"
        r"explanation)(_?(text|txt|other|en))?$|(^|_)id10476($|_)"
    )),
]
# Numeric columns with at most this many distinct values are treated as codes, so their
# min/max are only shown when those values themselves pass small-cell suppression.
CODE_LIKE_MAX_DISTINCT = 50


def _entries(entries: Iterable[str]) -> set[str]:
    return {e.strip().lower() for e in entries}


def name_rule(column: str, cfg: PrivacyConfig) -> str | None:
    """What an identifier column holds, judged by its name alone (None if the name says nothing)."""
    col = column.lower()
    for label, pat in NAME_RULES:
        if pat.search(col):
            return label
    for pat in cfg.pii_name_patterns:
        if re.search(pat, col):
            return f"matches privacy.pii_name_patterns /{pat}/"
    return None


def detect_pii(table: str, column: str, sample_values: Iterable[str], cfg: PrivacyConfig) -> str | None:
    """Return a reason string if the column looks like a direct identifier, else None.

    ``privacy.pii_columns`` always flags a column and ``privacy.not_pii_columns`` clears a column the
    rules below flagged by mistake. The rules: the column name (names, phone, e-mail, national and
    register numbers, addresses, GPS, free text), then a sample of its values (e-mail addresses,
    phone numbers, coordinates, free text).
    """
    qualified = f"{table}.{column}".lower()
    if qualified in _entries(cfg.pii_columns):
        return "listed in privacy.pii_columns"
    if qualified in _entries(cfg.not_pii_columns):
        return None
    label = name_rule(column, cfg)
    if label:
        return f"the column name suggests {label}"
    values = [v.strip() for v in sample_values if v and v.strip()]
    if len(values) >= 5:
        n = len(values)
        if sum(bool(_EMAIL.match(v)) for v in values) / n > 0.5:
            return "the values look like e-mail addresses"
        if not _ID_LIKE_NAME.search(column.lower()) or _PHONE_WORD.search(column.lower()):
            phones = [v for v in values if not _DATE_LIKE.match(v) and _PHONE.match(_PHONE_SEPARATORS.sub("", v))]
            if len(phones) / n > 0.5:
                return "the values look like phone numbers"
        if sum(bool(_GPS.match(v)) for v in values) / n > 0.5:
            return "the values look like GPS coordinates"
        if sum(_looks_like_text(v) for v in values) / n > 0.5:
            return "the values look like free text (notes and narratives can contain names)"
    return None


def _looks_like_text(v: str) -> bool:
    words = v.split()
    if len(v) < FREE_TEXT_MIN_LENGTH or len(words) < FREE_TEXT_MIN_WORDS or "[" in v:
        return False
    n_function = sum(w.strip(".,;:!?()\"'").lower() in _FUNCTION_WORDS for w in words)
    return n_function >= 3 or bool(_SENTENCE_BREAK.search(v))


def apply_overrides(
    profile: SourceProfile, cfg: PrivacyConfig, annotations: "SourceAnnotations | None" = None
) -> SourceProfile:
    """The profile with the project's current privacy decisions applied.

    The profiler's flags are taken when the source is profiled; this applies what may have changed
    since: ``privacy.pii_columns`` and ``privacy.not_pii_columns``, and columns Claude labelled as
    identifiers or free text when it interpreted the profile (those values are then withheld from
    every later prompt and can never be loaded into the CDM).
    """
    extra = _entries(cfg.pii_columns)
    allowed = _entries(cfg.not_pii_columns)
    labelled: dict[str, str] = {}
    if annotations is not None:
        for ca in annotations.columns:
            # Only Claude's own labels: the rule-based fallback calls any long string free text.
            if ca.origin == "llm" and ca.semantic_type in ("pii", "free_text"):
                what = "an identifier" if ca.semantic_type == "pii" else "free text"
                labelled[f"{ca.table}.{ca.column}".lower()] = f"labelled {what} when Claude interpreted the profile"
    tables: list[TableProfile] = []
    for t in profile.tables:
        cols = []
        for c in t.columns:
            key = f"{t.name}.{c.name}".lower()
            if key in extra and not c.is_pii:
                c = redact_column(c).model_copy(update={"is_pii": True, "pii_reason": "listed in privacy.pii_columns"})
            elif key in allowed and c.is_pii and key not in extra:
                c = c.model_copy(update={"is_pii": False, "pii_reason": None})
            elif key in labelled and not c.is_pii and key not in allowed:
                c = redact_column(c).model_copy(update={"is_pii": True, "pii_reason": labelled[key]})
            cols.append(c)
        tables.append(t.model_copy(update={"columns": cols}))
    return profile.model_copy(update={"tables": tables})


def suppress_small_cells(values: list[ValueFreq], min_cell_count: int) -> tuple[list[ValueFreq], int]:
    kept = [v for v in values if v.count >= min_cell_count]
    return kept, len(values) - len(kept)


def redact_column(col: ColumnProfile) -> ColumnProfile:
    """Strip every value-bearing statistic from a column profile."""
    return col.model_copy(
        update={"top_values": [], "min_value": None, "max_value": None, "mean": None, "n_suppressed_values": 0}
    )


def llm_view(profile: SourceProfile, cfg: PrivacyConfig) -> SourceProfile:
    """The version of the profile that is allowed into a prompt."""
    tables: list[TableProfile] = []
    for t in profile.tables:
        cols = []
        for c in t.columns:
            if c.is_pii or not cfg.send_values_to_llm:
                cols.append(redact_column(c))
                continue
            # The stricter of the threshold the profile was taken with and the current setting.
            kept, n_dropped = suppress_small_cells(c.top_values, max(profile.min_cell_count, cfg.min_cell_count))
            update: dict = {
                "top_values": kept[: cfg.max_values_per_column],
                "n_suppressed_values": c.n_suppressed_values + n_dropped,
            }
            if _range_reveals_rare_value(c, {v.value for v in kept}):
                update.update(min_value=None, max_value=None)
            else:
                update.update(min_value=_coarsen(c, c.min_value), max_value=_coarsen(c, c.max_value))
            cols.append(c.model_copy(update=update))
        tables.append(t.model_copy(update={"columns": cols}))
    return profile.model_copy(update={"tables": tables})


def _coarsen(col: ColumnProfile, value: str | None) -> str | None:
    """Range endpoints are single records: show dates as years and continuous numbers rounded to
    two significant figures, so an extreme value (the oldest birth date, a heaviest weight) is not
    disclosed exactly."""
    if value is None:
        return None
    if col.inferred_type in {"date", "datetime"}:
        m = re.match(r"^(\d{4})", value)
        return m.group(1) if m else None
    if col.inferred_type in {"integer", "float"} and col.n_distinct > CODE_LIKE_MAX_DISTINCT:
        try:
            x = float(value)
        except ValueError:
            return None
        if x == 0:
            return "0"
        rounded = float(f"{x:.2g}")
        return f"~{int(rounded) if rounded.is_integer() else rounded}"
    return value


def _range_reveals_rare_value(col: ColumnProfile, shown: set[str]) -> bool:
    """A code-like numeric column's min/max are individual values: show them only if they are
    themselves frequent enough to be listed (e.g. a sex range [1 .. 9] would reveal a rare 9)."""
    if col.inferred_type not in {"integer", "float", "boolean"} or col.n_distinct > CODE_LIKE_MAX_DISTINCT:
        return False
    return any(v is not None and v not in shown for v in (col.min_value, col.max_value))


def profile_prompt_text(profile: SourceProfile, cfg: PrivacyConfig, tables: list[str] | None = None) -> str:
    """Compact, privacy-filtered text rendering of the profile for prompts."""
    view = llm_view(profile, cfg)
    lines: list[str] = []
    for t in view.tables:
        if tables is not None and t.name not in tables:
            continue
        lines.append(f"SOURCE TABLE {t.name} ({t.n_rows} rows)")
        for c in t.columns:
            bits: list[str] = [c.inferred_type]
            if c.declared_type:
                bits.append(f"declared {c.declared_type}")
            null_pct = 100.0 * c.n_null / c.n_rows if c.n_rows else 0.0
            bits.append(f"{null_pct:.0f}% null")
            bits.append(f"{c.n_distinct} distinct")
            if c.unique:
                bits.append("unique")
            if c.date_format:
                bits.append(f"format {c.date_format} (PostgreSQL: to_date(x, '{_pg_template(c.date_format)}'))")
            if c.min_value is not None or c.max_value is not None:
                bits.append(f"range [{c.min_value} .. {c.max_value}]")
            if c.is_pii:
                bits.append(f"PII - values withheld ({c.pii_reason})")
            line = f"  - {c.name}: " + ", ".join(bits)
            if c.top_values:
                vals = "; ".join(f"{v.value!r}×{v.count}" for v in c.top_values)
                line += f"\n      top values: {vals}"
                if c.n_suppressed_values:
                    line += f" (+{c.n_suppressed_values} values not listed: rare or beyond the top {cfg.max_values_per_column})"
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def _pg_template(fmt: str) -> str:
    from .profiling.profiler import strftime_to_pg

    try:
        return strftime_to_pg(fmt)
    except Exception:  # an unknown directive: show the strftime pattern only
        return fmt


def pii_columns(profile: SourceProfile | None, cfg: PrivacyConfig) -> dict[str, set[str]]:
    """Direct-identifier columns per source table (from the profile and ``privacy.pii_columns``)."""
    out: dict[str, set[str]] = {}
    if profile is not None:
        for t in profile.tables:
            cols = {c.name.lower() for c in t.columns if c.is_pii}
            if cols:
                out[t.name.lower()] = cols
    for entry in cfg.pii_columns:
        if "." in entry:
            table, col = entry.lower().rsplit(".", 1)
            out.setdefault(table, set()).add(col)
    return out


_ALIAS_COL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


def pii_references(tm: "TableMapping", profile: SourceProfile | None, cfg: PrivacyConfig) -> list[str]:
    """``table.column`` identifiers that a table mapping reads (direct identifiers must never be
    loaded into the CDM or sent for vocabulary mapping)."""
    from .sqlguard import strip_literals

    pii = pii_columns(profile, cfg)
    if not pii:
        return []
    aliases = {tm.source.alias.lower(): tm.source.table.lower()}
    aliases.update({j.alias.lower(): j.table.lower() for j in tm.joins})
    texts = [f.expression for f in tm.fields] + [tm.where, tm.dedup_order] + [j.on for j in tm.joins]
    hits: list[str] = []
    for text in texts:
        for alias, col in _ALIAS_COL.findall(strip_literals(text or "")):
            table = aliases.get(alias.lower())
            if table and col.lower() in pii.get(table.split(".")[-1], set()):
                ref = f"{table}.{col.lower()}"
                if ref not in hits:
                    hits.append(ref)
    return hits
