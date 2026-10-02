"""Load an Athena vocabulary download into ``vocab_schema``.

Athena files are tab-delimited with a header row, unquoted, with ``YYYYMMDD`` dates, which
PostgreSQL parses natively. They are streamed with COPY in CSV mode using a quote character
that never occurs (backspace), the usual OHDSI approach, so quotes inside concept names
survive.
"""

from __future__ import annotations

import re
from pathlib import Path

import psycopg

from ..cdm.ddl import create_cdm_tables
from ..db import Schemas, execute, fetch_all, qi, scalar
from ..workspace import Workspace

VOCAB_FILES = {
    "concept": "CONCEPT.csv",
    "vocabulary": "VOCABULARY.csv",
    "domain": "DOMAIN.csv",
    "concept_class": "CONCEPT_CLASS.csv",
    "relationship": "RELATIONSHIP.csv",
    "concept_relationship": "CONCEPT_RELATIONSHIP.csv",
    "concept_synonym": "CONCEPT_SYNONYM.csv",
    "concept_ancestor": "CONCEPT_ANCESTOR.csv",
    "drug_strength": "DRUG_STRENGTH.csv",
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_concept_code ON {vocab}.concept (concept_code, vocabulary_id)",
    "CREATE INDEX IF NOT EXISTS idx_concept_domain ON {vocab}.concept (domain_id, standard_concept)",
    "CREATE INDEX IF NOT EXISTS idx_concept_rel_1 ON {vocab}.concept_relationship (concept_id_1, relationship_id)",
    "CREATE INDEX IF NOT EXISTS idx_concept_syn ON {vocab}.concept_synonym (concept_id)",
    "CREATE INDEX IF NOT EXISTS idx_concept_anc_d ON {vocab}.concept_ancestor (descendant_concept_id)",
    # The lookups by id that the official OHDSI DDL indexes (OMOPCDM_postgresql_5.4_indices.sql)
    # and that ATLAS, the mapping review and the coverage evaluation rely on.
    "CREATE INDEX IF NOT EXISTS idx_concept_concept_id ON {vocab}.concept (concept_id)",
    "CREATE INDEX IF NOT EXISTS idx_concept_relationship_id_2 ON {vocab}.concept_relationship (concept_id_2)",
    "CREATE INDEX IF NOT EXISTS idx_concept_ancestor_id_1 ON {vocab}.concept_ancestor (ancestor_concept_id)",
    "CREATE INDEX IF NOT EXISTS idx_drug_strength_id_1 ON {vocab}.drug_strength (drug_concept_id)",
    "CREATE INDEX IF NOT EXISTS idx_drug_strength_id_2 ON {vocab}.drug_strength (ingredient_concept_id)",
]


_INDEX_NAME = re.compile(r"CREATE INDEX IF NOT EXISTS (\w+) ON")


def ensure_indexes(conn: psycopg.Connection, schemas: Schemas) -> list[str]:
    """Create any missing vocabulary indexes (a vocabulary loaded by an older version lacks some).
    Idempotent; on a full Athena vocabulary the first run takes a few minutes. Returns the indexes
    that are missing but could not be created (the vocabulary tables belong to another role)."""
    have = {
        r["indexname"]
        for r in fetch_all(conn, "SELECT indexname FROM pg_indexes WHERE schemaname = %s", (schemas.vocab,))
    }
    stmts = INDEXES + (TRGM_INDEXES if ensure_trigram(conn) else [])
    skipped: list[str] = []
    for stmt in stmts:
        m = _INDEX_NAME.search(stmt)
        if m and m.group(1) in have:
            continue
        try:
            with conn.transaction():
                execute(conn, schemas.render(stmt))
        except psycopg.errors.InsufficientPrivilege:
            skipped.append(m.group(1) if m else stmt)
    conn.commit()
    return skipped
TRGM_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_concept_name_trgm ON {vocab}.concept USING gin (lower(concept_name) gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_concept_syn_trgm ON {vocab}.concept_synonym USING gin (lower(concept_synonym_name) gin_trgm_ops)",
]


def _find(athena_dir: Path, filename: str) -> Path | None:
    for cand in (athena_dir / filename, athena_dir / filename.lower()):
        if cand.exists():
            return cand
    return None


def has_trigram(conn: psycopg.Connection) -> bool:
    return bool(scalar(conn, "SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'"))


def ensure_trigram(conn: psycopg.Connection) -> bool:
    """Try to enable pg_trgm (used for fast fuzzy concept search). Returns availability."""
    if has_trigram(conn):
        return True
    try:
        with conn.transaction():
            execute(conn, "CREATE EXTENSION IF NOT EXISTS pg_trgm")
        return True
    except psycopg.Error:
        return False


def vocabulary_version(conn: psycopg.Connection, schemas: Schemas) -> str | None:
    try:
        with conn.transaction():
            return scalar(
                conn, f"SELECT vocabulary_version FROM {qi(schemas.vocab)}.vocabulary WHERE vocabulary_id = 'None'"
            )
    except psycopg.Error:
        return None


def loaded_vocabularies(conn: psycopg.Connection, schemas: Schemas) -> set[tuple[str, str]]:
    """(vocabulary_id, vocabulary_version) of every vocabulary currently loaded."""
    try:
        with conn.transaction():
            rows = fetch_all(conn, f"SELECT vocabulary_id, vocabulary_version FROM {qi(schemas.vocab)}.vocabulary")
    except psycopg.Error:
        return set()
    return {(r["vocabulary_id"], r["vocabulary_version"] or "") for r in rows}


def vocab_loaded(conn: psycopg.Connection, schemas: Schemas) -> bool:
    try:
        with conn.transaction():
            return bool(scalar(conn, f"SELECT 1 FROM {qi(schemas.vocab)}.concept LIMIT 1"))
    except psycopg.Error:
        return False


def load_athena(
    conn: psycopg.Connection,
    schemas: Schemas,
    athena_dir: Path,
    ws: Workspace | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Load the Athena CSVs found in ``athena_dir``. Returns rows loaded per table.

    Skips loading only when the loaded vocabularies are exactly the ones in the download (same
    vocabulary ids and versions), unless ``force`` is set. Comparing the release version alone
    is not enough: a second bundle from the same Athena release that adds, say, ICD10 or
    RxNorm Extension has the same release version.
    """
    athena_dir = Path(athena_dir)
    if not _find(athena_dir, "CONCEPT.csv"):
        raise FileNotFoundError(f"No CONCEPT.csv in {athena_dir}: is this an unzipped Athena download?")
    create_cdm_tables(conn, schemas, include_vocab=True)

    new_version = None
    bundle: set[tuple[str, str]] = set()
    vfile = _find(athena_dir, "VOCABULARY.csv")
    if vfile:
        for line in vfile.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            parts = line.split("\t")
            if parts and parts[0]:
                bundle.add((parts[0], parts[3] if len(parts) > 3 else ""))
            if parts and parts[0] == "None" and len(parts) > 3:
                new_version = parts[3]
    if not force and bundle and vocab_loaded(conn, schemas) and loaded_vocabularies(conn, schemas) == bundle:
        missing = ensure_indexes(conn, schemas)
        if ws:
            ws.audit("vocab_load_skipped", version=new_version, vocabularies=len(bundle), indexes_not_created=missing)
        return {}

    counts: dict[str, int] = {}
    for table, filename in VOCAB_FILES.items():
        path = _find(athena_dir, filename)
        target = f"{qi(schemas.vocab)}.{qi(table)}"
        execute(conn, f"TRUNCATE TABLE {target}")
        if path is None:
            counts[table] = 0
            continue
        with path.open("rb") as fh, conn.cursor() as cur:
            header = fh.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
            cols = ", ".join(qi(h.strip().lower()) for h in header)
            with cur.copy(
                f"COPY {target} ({cols}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', QUOTE E'\\b', NULL '')"
            ) as cp:
                while chunk := fh.read(1 << 20):
                    cp.write(chunk)
        counts[table] = int(scalar(conn, f"SELECT COUNT(*) FROM {target}") or 0)

    for stmt in INDEXES:
        execute(conn, schemas.render(stmt))
    if ensure_trigram(conn):
        for stmt in TRGM_INDEXES:
            execute(conn, schemas.render(stmt))
    execute(conn, f"ANALYZE {qi(schemas.vocab)}.concept")
    conn.commit()

    if _find(athena_dir, "CONCEPT_CPT4.csv"):
        msg = "CONCEPT_CPT4.csv present: run the Athena cpt4.jar with a UMLS API key to add CPT4 names."
        if ws:
            ws.audit("vocab_warning", message=msg)
    if ws:
        ws.audit("vocab_loaded", version=new_version, counts=counts, source=str(athena_dir))
    return counts


def vocabulary_inventory(conn: psycopg.Connection, schemas: Schemas) -> list[dict]:
    """vocabulary_id, version and concept counts for every loaded vocabulary."""
    return fetch_all(
        conn,
        f"""SELECT v.vocabulary_id, v.vocabulary_version,
                   COUNT(c.concept_id) AS concepts,
                   COUNT(c.concept_id) FILTER (WHERE c.standard_concept = 'S') AS standard_concepts
            FROM {qi(schemas.vocab)}.vocabulary v
            LEFT JOIN {qi(schemas.vocab)}.concept c ON c.vocabulary_id = v.vocabulary_id
            GROUP BY v.vocabulary_id, v.vocabulary_version
            ORDER BY v.vocabulary_id""",
    )
