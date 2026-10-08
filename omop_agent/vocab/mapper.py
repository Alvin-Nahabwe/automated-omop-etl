"""Map local source codes and labels to standard concepts (the Usagi step).

Terms come from every ``concept_map`` field of the ETL spec, grouped by the field's
``source_vocabulary_id`` (e.g. ``SRC_DRUG``). A field whose expression reads a direct identifier
(a PII column of the profile, ``privacy.pii_columns`` or a PII-like column name) is skipped, so
names or phone numbers are never collected as terms. For each source vocabulary:

1. terms seen fewer than ``privacy.min_cell_count`` times are *suppressed*: they get lexical
   candidates for the reviewer, are never sent to the LLM, and stay unmapped;
2. a term equal to the name or a synonym of exactly one standard concept in the target
   domain is an *exact* match. With an LLM it is still sent, as the top candidate, so that
   related terms are mapped consistently (a live run mapped NEG to the LOINC answer "Neg"
   through the shortcut while the model mapped POS to SNOMED "Positive"). The exact match is
   used directly (method ``exact``) only when no LLM answer is available;
3. the terms go to the LLM in batches, each with its top lexical candidates. The model
   picks one concept id (or 0) and may suggest better search phrases. A choice marked
   UNMATCHED is kept unmapped whatever id came with it;
4. terms left unmapped with suggested phrases are searched again and sent once more;
5. every chosen id is checked against the vocabulary. An id that is not a valid standard
   concept of the target domain is rejected, and the term is FLAGGED for review.

A term of the main concept field of a table the build routes by domain (``etl.routing``) may
also map to a concept of the table's other accepted domains (``SourceTerm.accepted_domains``: a
diagnosis list holds observations and procedures too); the record then moves to the table of
the concept's domain. Its target domain is searched as for any term, and each other accepted
domain adds its best few candidates on top of the target domain's (one retrieval serves all of
them). The exact-match and lexical shortcuts pick only concepts of the target domain, and a
model's choice of another domain is never approved automatically. Neither is any concept of a
source vocabulary whose fields expect different domains (``SourceTerm.auto_approve``).

Without an LLM (provider ``none``, no replay recording, or ``privacy.send_values_to_llm: false``,
which keeps every source value out of prompts) the best lexical candidate is kept as an
UNCHECKED suggestion when its score is at least ``LEXICAL_MIN_SCORE``.

Lexical search retrieves concepts with pg_trgm (or ILIKE when pg_trgm is missing). Small
domains, such as units, are scored exhaustively. Every candidate is scored by the same Python
function, so scores do not depend on the retrieval path (see :func:`text_score`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import psycopg
from rapidfuzz import fuzz

from ..config import PrivacyConfig, ProjectConfig
from ..db import Schemas, execute, fetch_all, short_error
from ..etl.routing import VocabularyDomains, vocabulary_domains
from ..llm import LLMClient, LLMError, LLMUnavailable
from ..models import (
    ConceptCandidate,
    ETLSpec,
    FieldMapping,
    MappingBatchDecision,
    MappingDecision,
    SourceProfile,
    SourceTerm,
    TableMapping,
    TermChoice,
    ValueFreq,
)
from ..privacy import detect_pii, pii_columns, pii_references
from ..sqlguard import strip_literals
from ..workspace import Workspace
from .evaluate import paren, query_rows, source_from_sql
from .loader import has_trigram

STAGE = "vocab_map"
LEXICAL_MIN_SCORE = 0.85  # lexical fallback threshold when no LLM is available
DECISION_CANDIDATES = 5  # candidates kept on each decision for the reviewer
EXHAUSTIVE_DOMAIN_SIZE = 3000  # domains up to this many standard concepts are scored in full
TRGM_WORD_THRESHOLD = 0.3  # pg_trgm.word_similarity_threshold used to retrieve candidates
# Candidates are first retrieved at this higher threshold, which the trigram index answers much faster on
# a full vocabulary; a domain whose pool that does not fill is retrieved again at TRGM_WORD_THRESHOLD
# (the pools are the same either way: see _Searcher._trigram). Drug names usually fill their pools at
# this threshold; for most other terms some accepted domain does not, and both retrievals run.
TRGM_FIRST_THRESHOLD = 0.5
POOL_SIZE = 300  # rows retrieved by trigram/ILIKE search before Python re-scoring
ALSO_ACCEPTED_CANDIDATES = 3  # candidates added from each other accepted domain of a term
MAX_RETRY_PHRASES = 3
LLM_DISABLED_BY_PRIVACY = "LLM mapping is disabled by privacy.send_values_to_llm"

MAPPER_SYSTEM = """\
You are a terminology mapper for an OMOP CDM conversion, working the way an experienced Usagi \
reviewer does. For each source term you receive its local code or label, how often it occurs, \
the OMOP domain its target should belong to, and a ranked list of candidate standard concepts \
found by a lexical search of the loaded vocabulary. Choose the single best candidate for each term.

How to choose:
- Pick a concept_id from the term's candidate list, or 0 when none of them means the same \
thing as the source term. A wrong mapping is worse than no mapping: it silently changes \
analyses, while an unmapped term stays visible and a reviewer can fix it.
- Prefer candidates that are EQUAL or EQUIVALENT in meaning. Use WIDER when the concept is \
more general than the term, NARROWER when it is more specific, and INEXACT when it is related \
but not the same. Use UNMATCHED with concept_id 0.
- For drug products, map to the clinical drug (ingredients + strength + dose form) when the \
term gives a strength or form and a matching clinical drug is listed. Never pick a different \
strength. Fall back to the clinical drug form or the ingredient only when no matching \
clinical drug is listed, and mark it WIDER. RxNorm uses US names (acetaminophen for \
paracetamol, for example).
- Local abbreviations are common: drug short names, lab test codes, unit spellings, result \
flags. Read them with clinical knowledge, but map only when the meaning is clear.
- The score is lexical similarity, not a probability. Judge meaning yourself.
- Give a calibrated confidence from 0 to 1: 0.95 or more only for an unambiguous match, less \
when you are expanding an abbreviation or choosing between close options.
- When no candidate fits, propose up to 3 better search phrases in retry_search_terms, such \
as expanded abbreviations, generic drug names or full test names. The vocabulary will be \
searched again with them. Do the same when you pick the best listed candidate but expect a \
better one exists (give it a lower confidence): your choice is kept unless the new search \
finds something better.
- When units recorded with a lab test or observation are listed, choose the concept whose \
property matches them: g/dL, mg/dL and g/L are mass concentrations ([Mass/volume]), mmol/L and \
umol/L are molar ([Moles/volume]), and so on. Prefer a concept that names the specimen (e.g. \
in Blood) over a generic one when the term implies it.
- Some terms also accept concepts of other domains ("also accepted"), for example a \
diagnosis list that holds observations and procedures; each candidate then shows its domain. \
Prefer a concept of the target domain. Choose one of another accepted domain only when the term \
clearly means that kind of thing (an ICD-10 Z code for antenatal care is an observation, not a \
condition): the record then moves to that domain's table, and a reviewer checks the choice.
- Keep each rationale to one or two sentences.

Return exactly one choice per source term, with source_code copied exactly as given."""


# --------------------------------------------------------------------------------------
# term collection
# --------------------------------------------------------------------------------------


_QUALIFIED = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
# an identifier that is neither qualified (a.b), a qualifier, nor a function name
_BARE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)(?![\w.]|\s*\()")


def _field_pii(tm: TableMapping, fm: FieldMapping, profile: SourceProfile | None, privacy: PrivacyConfig) -> list[str]:
    """``table.column`` direct identifiers read by one field's expression.

    Uses :func:`omop_agent.privacy.pii_references` on the field alone (the mapping's WHERE,
    dedup_order and join conditions select rows but do not become terms), and also catches
    unqualified column names and PII-like names when no profile is available.
    """
    only = tm.model_copy(
        update={"fields": [fm], "where": None, "dedup_order": None,
                "joins": [j.model_copy(update={"on": ""}) for j in tm.joins]}
    )
    hits = list(pii_references(only, profile, privacy))
    tables = [tm.source.table] + [j.table for j in tm.joins]
    aliases = {tm.source.alias.lower(): tm.source.table}
    aliases.update({j.alias.lower(): j.table for j in tm.joins})
    text = strip_literals(fm.expression or "")
    refs = [(aliases.get(a.lower()), c) for a, c in _QUALIFIED.findall(text)]
    refs += [(t, c) for c in _BARE.findall(text) for t in tables]
    pii = pii_columns(profile, privacy)
    for table, col in refs:
        if table is None:
            continue
        name = table.split(".")[-1].lower()
        if col.lower() in pii.get(name, set()) or detect_pii(name, col, [], privacy) is not None:
            ref = f"{table.lower()}.{col.lower()}"
            if ref not in hits:
                hits.append(ref)
    return hits


def _pii_fields(
    spec: ETLSpec, profile: SourceProfile | None, privacy: PrivacyConfig | None
) -> dict[tuple[int, int], str]:
    """(mapping index, field index) -> description of each concept_map field that reads PII."""
    privacy = privacy or PrivacyConfig()
    out: dict[tuple[int, int], str] = {}
    for i, tm in enumerate(spec.table_mappings):
        for j, fm in enumerate(tm.fields):
            if fm.kind == "concept_map" and (fm.expression or "").strip():
                if refs := _field_pii(tm, fm, profile, privacy):
                    out[(i, j)] = f"{tm.target_table}.{fm.target_field} reads {', '.join(refs)}"
    return out


UNIT_CONTEXT_FIELDS = {"measurement_concept_id", "observation_concept_id"}
MAX_UNITS_PER_TERM = 3


def _units_by_code(
    conn: psycopg.Connection,
    schemas: Schemas,
    tm: TableMapping,
    fm: FieldMapping,
    expr: str,
    privacy: PrivacyConfig | None,
    cfg: ProjectConfig | None = None,
) -> dict[str, list[ValueFreq]]:
    """Units recorded with each code of a measurement/observation concept field (the mapper needs
    them to tell, e.g., Glucose [Mass/volume] from [Moles/volume]). Only (code, unit) pairs seen at
    least ``min_cell_count`` times are returned, like any value that can reach the LLM."""
    if fm.target_field not in UNIT_CONTEXT_FIELDS:
        return {}
    unit = next(
        (f for f in tm.fields if f.target_field == "unit_source_value" and f.expression and f.kind in {"column", "expression"}),
        None,
    ) or next((f for f in tm.fields if f.target_field == "unit_concept_id" and f.expression and f.kind == "concept_map"), None)
    if unit is None:
        return {}
    k = (privacy or PrivacyConfig()).min_cell_count
    sql = (
        f"SELECT code, unit, COUNT(*) AS n FROM (\nSELECT {paren(expr)}::text AS code, {paren(unit.expression or '')}::text AS unit\n"
        f"{source_from_sql(tm, cfg.etl if cfg else None)}\n) q\nWHERE code IS NOT NULL AND unit IS NOT NULL AND btrim(unit) <> ''\n"
        f"GROUP BY code, unit HAVING COUNT(*) >= {int(k)} ORDER BY code, n DESC"
    )
    try:
        rows = query_rows(conn, schemas.render(sql), cfg)
    except psycopg.Error:
        return {}  # context is optional: never block mapping on it
    out: dict[str, list[ValueFreq]] = {}
    for r in rows:
        units = out.setdefault(r["code"], [])
        if len(units) < MAX_UNITS_PER_TERM:
            units.append(ValueFreq(value=r["unit"], count=int(r["n"])))
    return out


def pii_concept_fields(
    spec: ETLSpec, profile: SourceProfile | None = None, privacy: PrivacyConfig | None = None
) -> list[str]:
    """``concept_map`` fields that :func:`collect_terms` skips because they read a direct identifier.

    One line per field: ``'<target_table>.<target_field> reads <table.column>[, ...]'``. PII
    columns come from the profile (``is_pii``), ``privacy.pii_columns`` and the PII name
    patterns (``privacy`` defaults to :class:`PrivacyConfig`).
    """
    return list(_pii_fields(spec, profile, privacy).values())


def source_vocabulary_domains(
    spec: ETLSpec, profile: SourceProfile | None = None, privacy: PrivacyConfig | None = None
) -> dict[str, VocabularyDomains]:
    """Per source vocabulary of the ``concept_map`` fields that :func:`collect_terms` reads, the
    target domain and the domains its targets may belong to (``etl.routing.vocabulary_domains``)."""
    return vocabulary_domains(spec, skip=set(_pii_fields(spec, profile, privacy)))


def collect_terms(
    conn: psycopg.Connection,
    schemas: Schemas,
    spec: ETLSpec,
    profile: SourceProfile | None = None,
    privacy: PrivacyConfig | None = None,
    ws: Workspace | None = None,
    *,
    cfg: ProjectConfig | None = None,
) -> list[SourceTerm]:
    """Distinct values (with frequencies) of every ``concept_map`` expression in the spec.

    Values of fields that share a source vocabulary are merged. A term's frequency is the
    largest count from any single read of the source, never a sum: two fields that read the
    same column (e.g. ``drug_concept_id`` and ``drug_source_concept_id``) would otherwise
    double-count the same records and lift a rare value over the small-cell threshold.
    Sorted by (source_vocabulary_id, -frequency, code).

    A field whose expression reads a direct identifier is skipped, so its values can never
    reach the LLM or the mapping files: see :func:`pii_concept_fields` for the list (also
    audited as ``vocab_pii_fields_skipped`` when ``ws`` is given). ``privacy`` defaults to
    :class:`PrivacyConfig`, i.e. the default PII name patterns. The source is read under the build's
    session settings (``cfg``: time zone and imputation cap; defaults without it).

    Every term of a source vocabulary gets its target domain and accepted domains
    (:func:`source_vocabulary_domains`), merged over every field that uses the vocabulary, also
    those whose read of the source repeats another one. When those fields expect concepts of
    different domains the audit records ``vocab_domain_conflict``.
    """
    skip = _pii_fields(spec, profile, privacy)
    if skip and ws is not None:
        ws.audit("vocab_pii_fields_skipped", fields=list(skip.values()))
    domains = vocabulary_domains(spec, skip=set(skip))
    for vocab, vd in sorted(domains.items()):
        if vd.conflict and ws is not None:
            ws.audit("vocab_domain_conflict", source_vocabulary_id=vocab, fields=vd.fields,
                     accepted_domains=vd.accepted, detail=vd.conflict)
    merged: dict[tuple[str, str], SourceTerm] = {}
    seen_reads: set[tuple[str, str]] = set()
    for i, tm in enumerate(spec.table_mappings):
        for j, fm in enumerate(tm.fields):
            expr, vocab = (fm.expression or "").strip(), (fm.vocabulary or "").strip()
            if fm.kind != "concept_map" or not expr or not vocab or (i, j) in skip:
                continue
            sql = (
                f"SELECT code, COUNT(*) AS n FROM (\nSELECT {paren(expr)}::text AS code\n"
                f"{source_from_sql(tm, cfg.etl if cfg else None)}\n) q\n"
                "WHERE code IS NOT NULL AND btrim(code) <> '' GROUP BY code"
            )
            if (vocab, sql) in seen_reads:  # the same records, read again for another field
                continue
            seen_reads.add((vocab, sql))
            try:
                rows = query_rows(conn, schemas.render(sql), cfg)
            except psycopg.Error as e:
                raise ValueError(
                    f"{tm.target_table}.{fm.target_field}: could not read source codes: {short_error(e)}"
                ) from e
            units = _units_by_code(conn, schemas, tm, fm, expr, privacy, cfg)
            vd = domains[vocab]
            for r in rows:
                key = (vocab, r["code"])
                if key in merged:
                    t = merged[key]
                    t.frequency = max(t.frequency, int(r["n"]))
                    t.units = t.units or units.get(r["code"], [])
                else:
                    merged[key] = SourceTerm(
                        source_vocabulary_id=vocab,
                        source_code=r["code"],
                        source_name=r["code"],
                        frequency=int(r["n"]),
                        target_domain=vd.target_domain,
                        accepted_domains=list(vd.accepted),
                        auto_approve=vd.auto_approve,
                        units=units.get(r["code"], []),
                    )
    return sorted(merged.values(), key=lambda t: (t.source_vocabulary_id, -t.frequency, t.source_code))


# --------------------------------------------------------------------------------------
# lexical scoring
# --------------------------------------------------------------------------------------

_JOINER = re.compile(r"(?<=[a-z])['’-](?=[a-z])")  # co-trimoxazole -> cotrimoxazole
_NUM_UNIT = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)(?=[a-z])")  # 500mg -> 500 mg
_NON_ALNUM = re.compile(r"[^a-z0-9.]+")
_STRAY_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_NUMBER = re.compile(r"^\d+(\.\d+)?$")
_STOPWORDS = frozenset("a an and at by for in of on or per the to with".split())


def normalize(text: str) -> str:
    """Lowercase, split numbers from units, drop punctuation (hyphens inside words are joined)."""
    s = _JOINER.sub("", text.lower())
    s = _NUM_UNIT.sub(r"\1 ", s)
    s = _STRAY_DOT.sub(" ", _NON_ALNUM.sub(" ", s))
    return " ".join(s.split())


def _weight(tok: str) -> float:
    if tok in _STOPWORDS:
        return 0.5
    if _NUMBER.match(tok):
        return 3.0  # strengths matter: 250 vs 500 must not look alike
    return float(min(len(tok), 10))


def _singular(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


@lru_cache(maxsize=200_000)
def _token_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if _NUMBER.match(a) or _NUMBER.match(b):
        return 0.0
    a, b = _singular(a), _singular(b)
    if a == b:
        return 1.0
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 3 and long_.startswith(short):
        return 0.9  # abbreviation: caps/capsule, tabs/tablet, pos/positive, temp/temperature
    if len(short) < 4:
        return 0.0  # short tokens: only exact or prefix matches count
    r = fuzz.ratio(a, b) / 100
    return r if r >= 0.75 else 0.0


def _coverage(src: Sequence[str], dst: Sequence[str]) -> float:
    total = sum(_weight(t) for t in src)
    got = sum(_weight(t) * max(_token_sim(t, d) for d in dst) for t in src)
    return got / total if total else 0.0


def text_score(query: str, name: str) -> float:
    """Similarity of two *normalized* strings in 0..1.

    Fuzzy token matching gives recall (how much of the query the name covers) and precision
    (how much of the name the query covers). They are combined as an F2 score, so recall counts
    four times as much, plus a small character-level term (token_set_ratio) that orders weak
    matches. Unlike token_set_ratio alone, a subset does not score 1.0: 'paracetamol 500 mg'
    ranks 'paracetamol 500 mg oral tablet' above the bare ingredient 'paracetamol'.
    """
    if not query or not name:
        return 0.0
    if query == name:
        return 1.0
    q, n = query.split(), name.split()
    r, p = _coverage(q, n), _coverage(n, q)
    f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
    return min(1.0, 0.9 * f2 + 0.1 * fuzz.token_set_ratio(query, name) / 100)


# --------------------------------------------------------------------------------------
# candidate search
# --------------------------------------------------------------------------------------

_CONCEPT_COLS = (
    "c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id, c.concept_class_id, "
    "c.standard_concept, c.concept_code"
)


@dataclass
class _Hit:
    candidate: ConceptCandidate
    names: list[str] = field(default_factory=list)  # concept name first, then synonyms


def _by_name(h: _Hit, q: str) -> _Hit:
    """``h``, a concept of another accepted domain, scored by its names only: a concept code equal to
    a short local code ('TB', 'CR': HCPCS modifiers in Observation) says nothing of its meaning, so
    such a match must not rank above the target domain's concepts (``_Searcher.search``)."""
    if not h.names:
        return h
    score = round(max(text_score(q, normalize(n)) for n in h.names), 3)
    return h if score >= h.candidate.score else _Hit(h.candidate.model_copy(update={"score": score}), h.names)


def _exact_key(text: str) -> str:
    return " ".join(text.lower().split())


class _Searcher:
    """Lexical concept search with per-run caches (domain pools, repeated queries)."""

    def __init__(self, conn: psycopg.Connection, schemas: Schemas):
        self.conn = conn
        self.schemas = schemas
        self.v = self.schemas.render("{vocab}")
        try:
            with conn.transaction():
                self.trigram = has_trigram(conn)
        except psycopg.Error:
            self.trigram = False
        self._pools: dict[str | None, list[dict] | None] = {}
        self._cache: dict[tuple[str, str | None, int], list[_Hit]] = {}
        self._domains: dict[str, str] | None = None

    def domain(self, domain: str | None) -> str | None:
        """The vocabulary's spelling of ``domain`` ('drug' -> 'Drug'); None means any domain."""
        domain = (domain or "").strip()
        if not domain:
            return None
        if self._domains is None:
            try:
                rows = self._rows(f"SELECT domain_id FROM {self.v}.domain")
            except psycopg.Error:
                rows = []
            self._domains = {r["domain_id"].lower(): r["domain_id"] for r in rows}
        return self._domains.get(domain.lower(), domain)

    # ---- retrieval ---------------------------------------------------------------------
    def _rows(self, sql: str, params: dict | None = None, setup: str | None = None) -> list[dict]:
        with self.conn.transaction(force_rollback=True):
            if setup:
                execute(self.conn, setup)
            return fetch_all(self.conn, sql, params)

    def _domain_sql(self, domain: str | None) -> str:
        return "" if domain is None else "AND c.domain_id = %(domain)s"

    def _pool(self, domain: str | None) -> list[dict] | None:
        """All standard concepts of a small domain (with synonyms), or None when it is large."""
        if domain not in self._pools:
            where = f"c.standard_concept = 'S' AND c.invalid_reason IS NULL {self._domain_sql(domain)}"
            params = {"domain": domain, "n": EXHAUSTIVE_DOMAIN_SIZE + 1}
            n = self._rows(
                f"SELECT COUNT(*) AS n FROM (SELECT 1 FROM {self.v}.concept c WHERE {where} LIMIT %(n)s) x", params
            )[0]["n"]
            self._pools[domain] = None
            if n <= EXHAUSTIVE_DOMAIN_SIZE:
                self._pools[domain] = self._rows(
                    f"""SELECT {_CONCEPT_COLS},
                           ARRAY(SELECT s.concept_synonym_name FROM {self.v}.concept_synonym s
                                 WHERE s.concept_id = c.concept_id) AS synonyms
                        FROM {self.v}.concept c WHERE {where}""",
                    params,
                )
        return self._pools[domain]

    def _trigram(self, q: str, codes: list[str], domains: list[str | None]) -> list[dict]:
        """The trigram pools of ``domains`` (one None: any domain): per domain the ``POOL_SIZE``
        concepts of highest word similarity (ties by concept_id). They are first retrieved at
        ``TRGM_FIRST_THRESHOLD``, which on a full vocabulary matches far fewer names and synonyms
        than ``TRGM_WORD_THRESHOLD``; the domains whose pool that leaves short are retrieved again at
        ``TRGM_WORD_THRESHOLD``. The pools are those of a retrieval at ``TRGM_WORD_THRESHOLD``: a pool
        that fills at the higher threshold holds the concepts of highest similarity, all at or above
        it, ranked as in the lower one."""

        def fetch(wanted: list[str | None], threshold: float) -> list[dict]:
            if len(wanted) == 1:
                return self._trigram_rows(q, codes, wanted[0], threshold)
            return self._trigram_rows_each(q, codes, [d for d in wanted if d is not None], threshold)

        rows = fetch(domains, TRGM_FIRST_THRESHOLD)
        full = {d for d in domains if sum(d is None or r["domain_id"] == d for r in rows) >= POOL_SIZE}
        short = [d for d in domains if d not in full]
        if not short:
            return rows
        kept = [r for r in rows if r["domain_id"] in full]
        return kept + fetch(short, TRGM_WORD_THRESHOLD)

    def _trigram_rows(self, q: str, codes: list[str], domain: str | None,
                      threshold: float = TRGM_WORD_THRESHOLD) -> list[dict]:
        dom = self._domain_sql(domain)
        sql = f"""
SELECT {_CONCEPT_COLS},
       ARRAY(SELECT s.concept_synonym_name FROM {self.v}.concept_synonym s WHERE s.concept_id = c.concept_id) AS synonyms
FROM (
    SELECT c.concept_id, word_similarity(%(q)s, lower(c.concept_name)) AS ws
    FROM {self.v}.concept c
    WHERE %(q)s <%% lower(c.concept_name) AND c.standard_concept = 'S' AND c.invalid_reason IS NULL {dom}
    UNION ALL
    SELECT s.concept_id, word_similarity(%(q)s, lower(s.concept_synonym_name))
    FROM {self.v}.concept_synonym s
    WHERE %(q)s <%% lower(s.concept_synonym_name)
    UNION ALL
    SELECT c.concept_id, 1.0 FROM {self.v}.concept c WHERE c.concept_code = ANY(%(codes)s)
) h
JOIN {self.v}.concept c ON c.concept_id = h.concept_id
WHERE c.standard_concept = 'S' AND c.invalid_reason IS NULL {dom}
GROUP BY {_CONCEPT_COLS}
ORDER BY max(h.ws) DESC, c.concept_id
LIMIT %(pool)s"""
        setup = f"SET LOCAL pg_trgm.word_similarity_threshold = {float(threshold)}"
        return self._rows(sql, {"q": q, "codes": codes, "domain": domain, "pool": POOL_SIZE}, setup)

    def _trigram_rows_each(self, q: str, codes: list[str], domains: list[str],
                           threshold: float = TRGM_WORD_THRESHOLD) -> list[dict]:
        """``_trigram_rows`` for several domains in one query: per domain the same ``POOL_SIZE``
        rows as a query of that domain alone (ranked alike, ties by concept_id), but the trigram
        scans of the names and synonyms, which do not depend on the domain, run once. The concept's
        columns come from the one join of the candidates with the concept table (a second join
        would read the whole table when it has no index on concept_id)."""
        top = _CONCEPT_COLS.replace("c.", "top.")
        sql = f"""
SELECT {top},
       ARRAY(SELECT s.concept_synonym_name FROM {self.v}.concept_synonym s WHERE s.concept_id = top.concept_id)
           AS synonyms
FROM (
    SELECT {_CONCEPT_COLS},
           ROW_NUMBER() OVER (PARTITION BY c.domain_id ORDER BY max(h.ws) DESC, c.concept_id) AS rn
    FROM (
        SELECT c.concept_id, word_similarity(%(q)s, lower(c.concept_name)) AS ws
        FROM {self.v}.concept c
        WHERE %(q)s <%% lower(c.concept_name) AND c.standard_concept = 'S' AND c.invalid_reason IS NULL
          AND c.domain_id = ANY(%(domains)s)
        UNION ALL
        SELECT s.concept_id, word_similarity(%(q)s, lower(s.concept_synonym_name))
        FROM {self.v}.concept_synonym s
        WHERE %(q)s <%% lower(s.concept_synonym_name)
        UNION ALL
        SELECT c.concept_id, 1.0 FROM {self.v}.concept c WHERE c.concept_code = ANY(%(codes)s)
    ) h
    JOIN {self.v}.concept c ON c.concept_id = h.concept_id
    WHERE c.standard_concept = 'S' AND c.invalid_reason IS NULL AND c.domain_id = ANY(%(domains)s)
    GROUP BY {_CONCEPT_COLS}
) top
WHERE top.rn <= %(pool)s"""
        setup = f"SET LOCAL pg_trgm.word_similarity_threshold = {float(threshold)}"
        return self._rows(sql, {"q": q, "codes": codes, "domains": domains, "pool": POOL_SIZE}, setup)

    def _ilike_rows(self, q: str, codes: list[str], domain: str | None) -> list[dict]:
        tokens = list(dict.fromkeys(t for t in q.split() if len(t) >= 3))
        params: dict = {"codes": codes, "domain": domain, "pool": POOL_SIZE}
        conds = ["c.concept_code = ANY(%(codes)s)"]
        order = ""
        if tokens:
            params["pats"] = [f"%{t}%" for t in tokens]
            conds += [
                "c.concept_name ILIKE ANY(%(pats)s)",
                f"EXISTS (SELECT 1 FROM {self.v}.concept_synonym s WHERE s.concept_id = c.concept_id"
                " AND s.concept_synonym_name ILIKE ANY(%(pats)s))",
            ]
            for i, t in enumerate(tokens):
                params[f"p{i}"] = f"%{t}%"
            order = "(" + " + ".join(f"(c.concept_name ILIKE %(p{i})s)::int" for i in range(len(tokens))) + ") DESC, "
        sql = f"""
SELECT {_CONCEPT_COLS},
       ARRAY(SELECT s.concept_synonym_name FROM {self.v}.concept_synonym s WHERE s.concept_id = c.concept_id) AS synonyms
FROM {self.v}.concept c
WHERE c.standard_concept = 'S' AND c.invalid_reason IS NULL {self._domain_sql(domain)}
  AND ({" OR ".join(conds)})
ORDER BY {order}c.concept_id
LIMIT %(pool)s"""
        return self._rows(sql, params)

    # ---- search ------------------------------------------------------------------------
    def search(self, text: str, domain: str | None, limit: int = 10, also: Sequence[str] = ()) -> list[_Hit]:
        """The ``limit`` best concepts of ``domain`` (None: any domain) for ``text``, plus the
        ``ALSO_ACCEPTED_CANDIDATES`` best of each other accepted domain (``also``), each domain
        ranked as if searched on its own (so the target domain's candidates are those of a search
        without ``also``), best first; on a tie the target domain's concept comes first. The
        domains too large to score in full are retrieved together (``_trigram``)."""
        prefer = self.domain(domain)
        others = [d for d in dict.fromkeys(self.domain(a) for a in also) if d is not None and d != prefer]
        if prefer is None or not others:
            return self._search_each(text, [(prefer, limit)])[0]
        found = self._search_each(text, [(prefer, limit)] + [(d, ALSO_ACCEPTED_CANDIDATES) for d in others])
        q = normalize(text.strip())
        merged = found[0] + [_by_name(h, q) for hits in found[1:] for h in hits]
        merged.sort(key=lambda h: (-h.candidate.score, h.candidate.domain_id != prefer, h.candidate.concept_id))
        return merged

    def _search_each(self, text: str, wanted: list[tuple[str | None, int]]) -> list[list[_Hit]]:
        """For each (domain, limit) of ``wanted`` (domains spelled by ``domain``), the ``limit`` best
        concepts of that domain for ``text``. A small domain is scored in full; the large ones
        are retrieved by trigram (``_trigram``: one query for all of them, and one more for those
        whose pool its higher threshold leaves short) or ILIKE, ``POOL_SIZE`` rows each, and
        re-scored in Python."""
        raw = text.strip()
        q = normalize(raw)
        codes = list(dict.fromkeys(c for c in (raw, raw.upper(), raw.lower()) if c))
        rows_of: dict[str | None, list[dict]] = {}
        large: list[str | None] = []
        for domain, limit in wanted:
            if (text, domain, limit) in self._cache or domain in rows_of or domain in large:
                continue
            pool = self._pool(domain)
            if pool is not None:
                rows_of[domain] = pool
            else:
                large.append(domain)
        if large:
            fetched: list[dict] = []
            if self.trigram and q:
                try:
                    fetched = self._trigram(q, codes, large)
                except psycopg.Error:
                    self.trigram = False  # pg_trgm not reachable on the search_path: use ILIKE
            if self.trigram:
                for d in large:
                    rows_of[d] = [r for r in fetched if d is None or r["domain_id"] == d]
            else:
                for d in large:
                    rows_of[d] = self._ilike_rows(q, codes, d)
        out = []
        for domain, limit in wanted:
            key = (text, domain, limit)
            if key not in self._cache:
                self._cache[key] = self._scored(rows_of[domain], q, raw)[:limit]
            out.append(self._cache[key])
        return out

    @staticmethod
    def _scored(rows: list[dict], q: str, raw: str) -> list[_Hit]:
        """``rows`` as hits scored against the normalised text ``q`` (and the code ``raw``), best first."""
        hits = []
        for r in rows:
            names = [r["concept_name"], *(s for s in (r.get("synonyms") or []) if s)]
            score = max(text_score(q, normalize(n)) for n in names)
            if r["concept_code"] == raw:
                score = max(score, 0.95)
            elif r["concept_code"].lower() == raw.lower():
                score = max(score, 0.9)
            cand = ConceptCandidate(
                concept_id=r["concept_id"],
                concept_name=r["concept_name"],
                domain_id=r["domain_id"],
                vocabulary_id=r["vocabulary_id"],
                concept_class_id=r["concept_class_id"],
                standard_concept=r["standard_concept"],
                concept_code=r["concept_code"],
                score=round(score, 3),
            )
            hits.append(_Hit(cand, names))
        hits.sort(key=lambda h: (-h.candidate.score, h.candidate.concept_id))
        return hits

    def lookup(self, concept_id: int) -> dict | None:
        rows = self._rows(
            f"SELECT {_CONCEPT_COLS}, c.invalid_reason FROM {self.v}.concept c WHERE c.concept_id = %(id)s",
            {"id": concept_id},
        )
        return rows[0] if rows else None


def search_candidates(
    conn: psycopg.Connection, schemas: Schemas, text: str, domain: str | None, limit: int = 10,
    also: Sequence[str] = (),
) -> list[ConceptCandidate]:
    """Standard, valid concepts that lexically match ``text`` (names, synonyms, exact codes).

    Optionally restricted to ``domain``; ``also`` adds the best few of other accepted domains
    (``SourceTerm.accepted_domains``). Scores are in 0..1, best first. An exact concept_code
    match scores at least 0.95 (0.9 when only the case differs).
    """
    return [h.candidate for h in _Searcher(conn, schemas).search(text, domain, limit, also)]


def check_concept(
    conn: psycopg.Connection, schemas: Schemas, concept_id: int, domain: str | Sequence[str] | None = None
) -> tuple[ConceptCandidate | None, str | None]:
    """(candidate, None) if ``concept_id`` is a valid standard concept (of ``domain``, or of one
    of the domains it lists), else (None, reason)."""
    row = _Searcher(conn, schemas).lookup(concept_id)
    return _validate_row(row, concept_id, domain)


def accepted_domains(d: SourceTerm | MappingDecision) -> list[str]:
    """The domains a target of the term may belong to: its accepted domains, else its target
    domain alone ([]: any domain)."""
    return list(d.accepted_domains) or ([d.target_domain] if (d.target_domain or "").strip() else [])


def _in_target_domain(t: SourceTerm | MappingDecision, c: ConceptCandidate) -> bool:
    """``c`` belongs to the term's target domain (any concept does when it has none)."""
    return not (t.target_domain or "").strip() or c.domain_id.lower() == (t.target_domain or "").strip().lower()


def _validate_row(
    row: dict | None, concept_id: int, domain: str | Sequence[str] | None
) -> tuple[ConceptCandidate | None, str | None]:
    if row is None:
        return None, f"concept_id {concept_id} does not exist in the vocabulary"
    if row["standard_concept"] != "S":
        return None, f"concept_id {concept_id} ({row['concept_name']}) is not a standard concept"
    if row["invalid_reason"]:
        return None, f"concept_id {concept_id} ({row['concept_name']}) is no longer valid"
    domains = [d.strip() for d in ([domain] if isinstance(domain, str) else domain or []) if d and d.strip()]
    if domains and row["domain_id"].lower() not in {d.lower() for d in domains}:
        expected = domains[0] if len(domains) == 1 else ", ".join(domains[:-1]) + " or " + domains[-1]
        return None, f"concept_id {concept_id} ({row['concept_name']}) is in domain {row['domain_id']}, not {expected}"
    return (
        ConceptCandidate(
            concept_id=row["concept_id"],
            concept_name=row["concept_name"],
            domain_id=row["domain_id"],
            vocabulary_id=row["vocabulary_id"],
            concept_class_id=row["concept_class_id"],
            standard_concept=row["standard_concept"],
            concept_code=row["concept_code"],
        ),
        None,
    )


# --------------------------------------------------------------------------------------
# mapping
# --------------------------------------------------------------------------------------


def _chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    size = max(1, size)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _shown(t: SourceTerm, hits: list[_Hit], k: int) -> list[_Hit]:
    """The candidates of ``t`` shown to the model or kept for the reviewer: the first ``k`` of its
    target domain (of any domain without one) and, on top of them, the
    ``ALSO_ACCEPTED_CANDIDATES`` best of each other accepted domain, in the order of ``hits``. So
    the other domains never take the target domain's places."""
    taken: dict[str, int] = {}
    out = []
    for h in hits:
        group = "" if _in_target_domain(t, h.candidate) else h.candidate.domain_id.lower()
        if taken.get(group, 0) < (ALSO_ACCEPTED_CANDIDATES if group else k):
            taken[group] = taken.get(group, 0) + 1
            out.append(h)
    return out


def _decision(t: SourceTerm, hits: list[_Hit], **kw) -> MappingDecision:
    return MappingDecision(
        source_vocabulary_id=t.source_vocabulary_id,
        source_code=t.source_code,
        source_name=t.source_name,
        frequency=t.frequency,
        target_domain=t.target_domain,
        accepted_domains=list(t.accepted_domains),
        candidates=[h.candidate for h in _shown(t, hits, DECISION_CANDIDATES)],
        **kw,
    )


MIXED_DOMAINS_NOTE = (" (The fields that use this source vocabulary expect concepts of different domains, so a reviewer "
                  "checks every code.)")


def _target(c: ConceptCandidate) -> dict:
    return dict(
        target_concept_id=c.concept_id,
        target_concept_name=c.concept_name,
        target_domain_id=c.domain_id,
        target_vocabulary_id=c.vocabulary_id,
    )


def _exact_match(t: SourceTerm, hits: list[_Hit]) -> MappingDecision | None:
    """The one concept of the target domain whose name or synonym equals the term (concepts of
    the term's other accepted domains are never picked without a reviewer or the model)."""
    key = _exact_key(t.source_name)
    matches = [(h, n) for h in hits if _in_target_domain(t, h.candidate) for n in h.names if _exact_key(n) == key]
    ids = {h.candidate.concept_id for h, _ in matches}
    if len(ids) != 1:
        return None
    h, name = matches[0]
    how = "the concept name" if name == h.names[0] else f"the synonym '{name}'"
    return _decision(
        t,
        hits,
        **_target(h.candidate),
        equivalence="EQUAL",
        confidence=1.0,
        status="APPROVED" if t.auto_approve else "UNCHECKED",
        method="exact",
        rationale=f"The source term equals {how} of the only matching standard concept."
        + ("" if t.auto_approve else MIXED_DOMAINS_NOTE),
    )


def _lexical(t: SourceTerm, hits: list[_Hit], why: str) -> MappingDecision:
    """The best lexical match of the target domain as an UNCHECKED suggestion (no LLM)."""
    top = next((h.candidate for h in hits if _in_target_domain(t, h.candidate)), None)
    if top is not None and top.score >= LEXICAL_MIN_SCORE:
        return _decision(
            t,
            hits,
            **_target(top),
            equivalence="EQUIVALENT" if top.score >= 0.95 else "INEXACT",
            confidence=top.score,
            status="UNCHECKED",
            method="lexical",
            rationale=f"No LLM review ({why}). Best lexical match, score {top.score:.2f}; check it before approving.",
        )
    return _decision(
        t,
        hits,
        status="UNCHECKED",
        method="lexical",
        rationale=f"No LLM review ({why}) and no lexical match scored {LEXICAL_MIN_SCORE} or more; map it manually.",
    )


def _format_term(i: int, t: SourceTerm, hits: list[_Hit], k: int, phrases: list[str] | None) -> list[str]:
    lines = [f"{i}. source_code: {json.dumps(t.source_code, ensure_ascii=False)}"]
    if t.source_name != t.source_code:
        lines.append(f"   description: {json.dumps(t.source_name, ensure_ascii=False)}")
    lines.append(f"   frequency: {t.frequency}")
    others = [d for d in t.accepted_domains if d.lower() != (t.target_domain or "").lower()]
    if others:
        lines.append(f"   target domain: {t.target_domain} (also accepted: {', '.join(others)}; the record is moved to "
                     "the table of the chosen concept's domain)")
    else:
        lines.append(f"   target domain: {t.target_domain or 'any'}")
    if t.units:
        lines.append("   units recorded with it: " + "; ".join(f"{json.dumps(u.value, ensure_ascii=False)} ×{u.count}" for u in t.units))
    if phrases:
        lines.append(f"   searched again for: {'; '.join(phrases)}")
    shown = _shown(t, hits, k)
    if others or len({h.candidate.domain_id for h in shown}) > 1:  # other domains accepted: show each candidate's
        lines.append("   candidates (concept_id | name | domain | class | vocabulary | score):")
        for h in shown:
            c = h.candidate
            lines.append(f"   {c.concept_id} | {c.concept_name} | {c.domain_id} | {c.concept_class_id} | "
                         f"{c.vocabulary_id} | {c.score:.2f}")
    elif shown:
        lines.append("   candidates (concept_id | name | class | vocabulary | score):")
        for h in shown:
            c = h.candidate
            lines.append(
                f"   {c.concept_id} | {c.concept_name} | {c.concept_class_id} | {c.vocabulary_id} | {c.score:.2f}"
            )
    else:
        lines.append("   candidates: none found")
    return lines


def build_prompt(
    source_vocabulary_id: str,
    batch: Sequence[SourceTerm],
    hits: dict[str, list[_Hit]],
    k: int,
    retry_phrases: dict[str, list[str]] | None = None,
) -> str:
    """The user prompt for one batch. Only aggregate information: codes/labels and counts."""
    domains = sorted({t.target_domain or "any" for t in batch})
    lines = [f"Source vocabulary: {source_vocabulary_id}", f"Target domain: {', '.join(domains)}", ""]
    if retry_phrases is not None:
        lines += [
            "Second search. In the previous round no candidate fitted these terms, or you expected a better "
            "concept than the one you picked, so the vocabulary was searched again with the phrases you "
            "suggested. Choose from the updated candidates, or 0 if none fits (a term you had already mapped "
            "then keeps its first choice). There will be no further search.",
            "",
        ]
    else:
        lines += [
            f"Choose the best standard concept for each of these {len(batch)} source terms, or 0 if none fits.",
            "",
        ]
    for i, t in enumerate(batch, 1):
        phrases = retry_phrases.get(t.source_code) if retry_phrases else None
        lines += _format_term(i, t, hits[t.source_code], k, phrases) + [""]
    lines.append("Return one choice per source term, copying source_code exactly.")
    return "\n".join(lines)


def _unmatched(ch: TermChoice) -> bool:
    """The model found no fitting concept: id 0, or equivalence UNMATCHED whatever id came with it."""
    return ch.chosen_concept_id == 0 or ch.equivalence == "UNMATCHED"


def _match_choices(batch: Sequence[SourceTerm], choices: list[TermChoice]) -> dict[str, TermChoice]:
    exact = {t.source_code for t in batch}
    loose = {_exact_key(t.source_code): t.source_code for t in batch}
    out: dict[str, TermChoice] = {}
    for ch in choices:
        code = ch.source_code if ch.source_code in exact else loose.get(_exact_key(ch.source_code))
        if code is not None and code not in out:
            out[code] = ch
    return out


class _VocabMapper:
    """Maps the terms of one source vocabulary."""

    def __init__(self, searcher: _Searcher, llm: LLMClient, cfg: ProjectConfig, svid: str):
        self.s = searcher
        self.llm = llm
        self.cfg = cfg
        self.svid = svid
        self.k = max(1, cfg.mapping.candidates_per_term)
        self.hits: dict[str, list[_Hit]] = {}
        self.stats = {"llm_calls": 0, "llm_unavailable": 0, "retried": 0, "rejected_ids": 0}
        # privacy.send_values_to_llm: false keeps every source value out of prompts
        self.llm_disabled = None if cfg.privacy.send_values_to_llm else LLM_DISABLED_BY_PRIVACY

    def _call(self, key: str, prompt: str) -> MappingBatchDecision | str:
        """The model's answer, or the reason there is none."""
        try:
            out = self.llm.structured(
                stage=STAGE, key=key, system=MAPPER_SYSTEM, prompt=prompt, output=MappingBatchDecision
            )
            self.stats["llm_calls"] += 1
            return out
        except LLMUnavailable as e:
            self.stats["llm_unavailable"] += 1
            return f"LLM unavailable: {str(e)[:200]}"
        except LLMError as e:
            self.stats["llm_unavailable"] += 1
            return f"LLM answer unusable: {str(e)[:200]}"

    def _apply(self, t: SourceTerm, ch: TermChoice | None) -> MappingDecision:
        hits = self.hits[t.source_code]
        if ch is None:
            return _decision(
                t,
                hits,
                status="UNCHECKED",
                method="llm",
                rationale="The model returned no choice for this term; map it manually.",
            )
        cid = ch.chosen_concept_id
        if _unmatched(ch):
            note = f" (Marked UNMATCHED, so concept_id {cid} was not used.)" if cid else ""
            return _decision(
                t,
                hits,
                status="UNCHECKED",
                method="llm",
                equivalence="UNMATCHED",
                confidence=ch.confidence,
                rationale=ch.rationale + note,
            )
        cand = next((h.candidate for h in hits if h.candidate.concept_id == cid), None)
        if cand is None:
            row = self.s.lookup(cid)
            cand, why = _validate_row(row, cid, accepted_domains(t))
            if cand is None:
                self.stats["rejected_ids"] += 1
                return _decision(
                    t,
                    hits,
                    status="FLAGGED",
                    method="llm",
                    equivalence="UNMATCHED",
                    confidence=0.0,
                    rationale=f"Rejected the model's choice: {why}. Model rationale: {ch.rationale}",
                )
        # a concept of another accepted domain moves the record to another table: a reviewer decides
        in_domain = _in_target_domain(t, cand)
        auto = ch.confidence >= self.cfg.mapping.auto_approve_confidence and in_domain and t.auto_approve
        note = "" if in_domain else (f" (A {cand.domain_id} concept, not {t.target_domain}: the record moves to the "
                                     f"table of the {cand.domain_id} domain; review it.)")
        note += "" if t.auto_approve else MIXED_DOMAINS_NOTE
        return _decision(
            t,
            hits,
            **_target(cand),
            equivalence=ch.equivalence,
            confidence=ch.confidence,
            status="APPROVED" if auto else "UNCHECKED",
            method="llm",
            rationale=ch.rationale + note,
        )

    def run(self, terms: list[SourceTerm]) -> dict[str, MappingDecision]:
        min_cell = self.cfg.privacy.min_cell_count
        ordered = sorted(terms, key=lambda t: (-t.frequency, t.source_code))
        out: dict[str, MappingDecision] = {}
        exact: dict[str, MappingDecision] = {}  # used when the LLM gives no answer
        pending: list[SourceTerm] = []
        for t in ordered:
            hits = self.s.search(t.source_name, t.target_domain, self.k, t.accepted_domains)
            self.hits[t.source_code] = hits
            if t.frequency < min_cell:
                out[t.source_code] = _decision(
                    t,
                    hits,
                    status="UNCHECKED",
                    method="suppressed",
                    rationale="Below the small-cell threshold; not sent to the LLM. Review it manually.",
                )
            else:
                if (d := _exact_match(t, hits)) is not None:
                    exact[t.source_code] = d
                pending.append(t)
        if self.llm_disabled is not None:
            for t in pending:
                out[t.source_code] = exact.get(t.source_code) or _lexical(t, self.hits[t.source_code], self.llm_disabled)
            return out

        # round 1: every pending term, in batches
        retry: list[SourceTerm] = []
        phrases: dict[str, list[str]] = {}
        for n, batch in enumerate(_chunks(pending, self.cfg.mapping.llm_batch_size), 1):
            ans = self._call(f"{self.svid}:batch{n}", build_prompt(self.svid, batch, self.hits, self.k))
            if isinstance(ans, str):
                for t in batch:
                    out[t.source_code] = exact.get(t.source_code) or _lexical(t, self.hits[t.source_code], ans)
                continue
            choices = _match_choices(batch, ans.choices)
            for t in batch:
                ch = choices.get(t.source_code)
                if ch is None and t.source_code in exact:
                    out[t.source_code] = exact[t.source_code]
                    continue
                out[t.source_code] = self._apply(t, ch)
                suggested = [p.strip() for p in (ch.retry_search_terms if ch else []) if p.strip()]
                # search again when nothing fitted, or when the model settled for a listed concept
                # below the auto-approve bar and named a better one to look for
                wants_better = ch is not None and ch.confidence < self.cfg.mapping.auto_approve_confidence
                if ch is not None and suggested and (_unmatched(ch) or wants_better):
                    retry.append(t)
                    phrases[t.source_code] = suggested[:MAX_RETRY_PHRASES]

        # round 2: search again with the model's phrases, then one more call per batch
        for t in retry:
            merged = {h.candidate.concept_id: h for h in self.hits[t.source_code]}
            for p in phrases[t.source_code]:
                for h in self.s.search(p, t.target_domain, self.k, t.accepted_domains):
                    old = merged.get(h.candidate.concept_id)
                    if old is None or h.candidate.score > old.candidate.score:
                        merged[h.candidate.concept_id] = h
            self.hits[t.source_code] = sorted(  # as _Searcher.search: on a tie the target domain first
                merged.values(),
                key=lambda h, t=t: (-h.candidate.score, not _in_target_domain(t, h.candidate), h.candidate.concept_id),
            )
            out[t.source_code].candidates = [h.candidate for h in _shown(t, self.hits[t.source_code],
                                                                         DECISION_CANDIDATES)]
        self.stats["retried"] = len(retry)
        for n, batch in enumerate(_chunks(retry, self.cfg.mapping.llm_batch_size), 1):
            ans = self._call(f"{self.svid}:retry{n}", build_prompt(self.svid, batch, self.hits, self.k, phrases))
            if isinstance(ans, str):
                continue  # keep the first-round answer (unmapped)
            choices = _match_choices(batch, ans.choices)
            for t in batch:
                if (ch := choices.get(t.source_code)) is None:
                    continue
                if _unmatched(ch) and out[t.source_code].target_concept_id != 0:
                    continue  # nothing better found: keep the first-round concept
                keep_candidates = out[t.source_code].candidates
                out[t.source_code] = self._apply(t, ch)
                out[t.source_code].candidates = out[t.source_code].candidates or keep_candidates
        return out


def map_terms(
    conn: psycopg.Connection,
    schemas: Schemas,
    terms: list[SourceTerm],
    llm: LLMClient,
    cfg: ProjectConfig,
    ws: Workspace | None = None,
) -> list[MappingDecision]:
    """Map every term to a standard concept (or 0). Returns decisions in the order of ``terms``.

    LLM calls are named ``vocab_map/<source_vocabulary_id>:batch<n>`` and ``...:retry<n>``.
    Only terms seen at least ``privacy.min_cell_count`` times ever appear in a prompt. With
    ``privacy.send_values_to_llm: false`` the LLM is never called: terms that are not exact
    matches get the lexical fallback (method ``lexical``), and the audit records why.
    """
    searcher = _Searcher(conn, schemas)
    by_vocab: dict[str, list[SourceTerm]] = {}
    for t in terms:
        by_vocab.setdefault(t.source_vocabulary_id, []).append(t)
    decided: dict[tuple[str, str], MappingDecision] = {}
    for svid in sorted(by_vocab):
        vm = _VocabMapper(searcher, llm, cfg, svid)
        result = vm.run(by_vocab[svid])
        decided.update({(svid, code): d for code, d in result.items()})
        if ws is not None:
            ds = list(result.values())
            ws.audit(
                "vocab_mapped",
                source_vocabulary_id=svid,
                terms=len(ds),
                methods={m: sum(d.method == m for d in ds) for m in ("exact", "llm", "lexical", "suppressed")},
                statuses={s: sum(d.status == s for d in ds) for s in ("APPROVED", "UNCHECKED", "FLAGGED")},
                mapped=sum(d.target_concept_id != 0 for d in ds),
                llm_disabled=vm.llm_disabled,
                **vm.stats,
            )
    return [decided[(t.source_vocabulary_id, t.source_code)] for t in terms]


# --------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------


def _usable(d: MappingDecision, include_unchecked: bool) -> bool:
    ok = {"APPROVED", "UNCHECKED"} if include_unchecked else {"APPROVED"}
    return d.target_concept_id != 0 and d.status in ok


def write_source_to_concept_map(
    conn: psycopg.Connection, schemas: Schemas, decisions: list[MappingDecision], include_unchecked: bool = True
) -> int:
    """Replace the rows of the decisions' source vocabularies in ``{etl}.source_to_concept_map``.

    Writes decisions with a target and status APPROVED (and UNCHECKED when
    ``include_unchecked``); FLAGGED and REJECTED decisions are never written. Returns the number
    of rows inserted. Does not commit: the caller commits.
    """
    vocabs = sorted({d.source_vocabulary_id for d in decisions})
    if not vocabs:
        return 0
    execute(
        conn, schemas.render("DELETE FROM {etl}.source_to_concept_map WHERE source_vocabulary_id = ANY(%s)"), (vocabs,)
    )
    sql = schemas.render("""INSERT INTO {etl}.source_to_concept_map
               (source_code, source_concept_id, source_vocabulary_id, source_code_description,
                target_concept_id, target_vocabulary_id, invalid_reason)
           VALUES (%s, 0, %s, %s, %s,
                   LEFT(COALESCE(%s, (SELECT vocabulary_id FROM {vocab}.concept WHERE concept_id = %s)), 20), NULL)
           ON CONFLICT DO NOTHING""")
    n = 0
    with conn.cursor() as cur:
        for d in decisions:
            if _usable(d, include_unchecked):
                cur.execute(
                    sql,
                    (
                        d.source_code[:255],
                        d.source_vocabulary_id,
                        d.source_name[:255],
                        d.target_concept_id,
                        d.target_vocabulary_id,
                        d.target_concept_id,
                    ),
                )
                n += max(cur.rowcount, 0)
    return n


def mapping_summary(decisions: list[MappingDecision], include_unchecked: bool = True) -> list[dict]:
    """Per source vocabulary: term counts by outcome, and records covered by a usable mapping.

    Outcomes are exclusive: approved / unchecked (have a target), flagged, rejected, unmapped
    (no target). ``records_mapped`` counts records whose term would be written to
    source_to_concept_map.
    """
    rows: dict[str, dict] = {}
    for d in decisions:
        r = rows.setdefault(
            d.source_vocabulary_id,
            dict(
                source_vocabulary_id=d.source_vocabulary_id,
                terms=0,
                approved=0,
                unchecked=0,
                flagged=0,
                rejected=0,
                unmapped=0,
                records=0,
                records_mapped=0,
            ),
        )
        r["terms"] += 1
        r["records"] += d.frequency
        if d.status == "FLAGGED":
            r["flagged"] += 1
        elif d.status == "REJECTED":
            r["rejected"] += 1
        elif d.target_concept_id == 0:
            r["unmapped"] += 1
        elif d.status == "APPROVED":
            r["approved"] += 1
        else:
            r["unchecked"] += 1
        if _usable(d, include_unchecked):
            r["records_mapped"] += d.frequency
    for r in rows.values():
        r["pct_records_mapped"] = round(100.0 * r["records_mapped"] / r["records"], 1) if r["records"] else 0.0
    return [rows[k] for k in sorted(rows)]
