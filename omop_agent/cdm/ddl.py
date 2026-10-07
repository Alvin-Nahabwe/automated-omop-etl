"""Create the CDM, vocabulary and ETL working tables in PostgreSQL.

Uses the official OHDSI v5.4 PostgreSQL DDL. Vocabulary tables go to ``vocab_schema`` and all
other tables to ``cdm_schema`` (they may be the same schema).
"""

from __future__ import annotations

import re
from importlib import resources

import psycopg

from ..db import Schemas, ensure_schema, execute, qi, scalar, table_exists
from .spec import RESOURCE_PKG, load_cdm_spec

_CREATE = re.compile(r"CREATE TABLE @cdmDatabaseSchema\.(\w+)\s*\((.*?)\);", re.S | re.I)
_PK = re.compile(r"ALTER TABLE @cdmDatabaseSchema\.(\w+)\s+ADD CONSTRAINT (\w+) PRIMARY KEY \((\w+)\);", re.I)


def _ddl_text(name: str) -> str:
    return resources.files(RESOURCE_PKG).joinpath(name).read_text(encoding="utf-8")


def cdm_create_statements(schemas: Schemas) -> dict[str, str]:
    """Map table name -> CREATE TABLE statement in the right schema."""
    spec = load_cdm_spec()
    out: dict[str, str] = {}
    for m in _CREATE.finditer(_ddl_text("OMOPCDM_postgresql_5.4_ddl.sql")):
        table, body = m.group(1).lower(), m.group(2)
        schema = schemas.vocab if spec.has_table(table) and spec.table(table).schema_class == "VOCAB" else schemas.cdm
        out[table] = f"CREATE TABLE IF NOT EXISTS {qi(schema)}.{qi(table)} ({body});"
    return out


def schema_for(table: str, schemas: Schemas) -> str:
    spec = load_cdm_spec()
    if spec.has_table(table) and spec.table(table).schema_class == "VOCAB":
        return schemas.vocab
    return schemas.cdm


def create_cdm_tables(
    conn: psycopg.Connection,
    schemas: Schemas,
    include_vocab: bool = True,
    drop_existing: bool = False,
) -> list[str]:
    """Create any missing CDM tables. Returns the tables created."""
    spec = load_cdm_spec()
    for s in {schemas.cdm, schemas.vocab}:
        ensure_schema(conn, s)
    created = []
    for table, stmt in cdm_create_statements(schemas).items():
        is_vocab = spec.has_table(table) and spec.table(table).schema_class == "VOCAB"
        if is_vocab and not include_vocab:
            continue
        schema = schemas.vocab if is_vocab else schemas.cdm
        if drop_existing and not is_vocab:
            execute(conn, f"DROP TABLE IF EXISTS {qi(schema)}.{qi(table)} CASCADE")
        if not table_exists(conn, schema, table):
            execute(conn, stmt)
            created.append(table)
    return created


def add_primary_keys(conn: psycopg.Connection, schemas: Schemas, tables: list[str] | None = None) -> None:
    """Add the official primary keys (after loading; duplicate keys raise)."""
    for m in _PK.finditer(_ddl_text("OMOPCDM_postgresql_5.4_primary_keys.sql")):
        table, cname, col = m.group(1).lower(), m.group(2), m.group(3)
        if tables is not None and table not in tables:
            continue
        schema = schema_for(table, schemas)
        exists = scalar(
            conn,
            "SELECT 1 FROM pg_constraint c JOIN pg_namespace n ON n.oid = c.connamespace "
            "WHERE n.nspname = %s AND c.conname = %s",
            (schema, cname),
        )
        if exists:
            continue
        execute(conn, f"ALTER TABLE {qi(schema)}.{qi(table)} ADD CONSTRAINT {qi(cname)} PRIMARY KEY ({qi(col)})")


# The id maps give each source key its CDM id. A key that did not become a row is moved to the
# matching *_retired table with its id, and gets that id back when it returns, so ids are never
# reused or lost. id_high_water keeps the largest id a table had before records were moved out of
# it, and build_facts holds values that one build step derives for later steps (both emptied or
# rewritten by the steps that use them).
ETL_TABLES = {
    "person_map": """
        CREATE TABLE IF NOT EXISTS {etl}.person_map (
            source_value text PRIMARY KEY,
            person_id integer NOT NULL UNIQUE
        )""",
    "visit_map": """
        CREATE TABLE IF NOT EXISTS {etl}.visit_map (
            source_value text PRIMARY KEY,
            visit_occurrence_id integer NOT NULL UNIQUE,
            person_source_value text
        )""",
    "person_map_retired": """
        CREATE TABLE IF NOT EXISTS {etl}.person_map_retired (
            source_value text PRIMARY KEY,
            person_id integer NOT NULL UNIQUE
        )""",
    "visit_map_retired": """
        CREATE TABLE IF NOT EXISTS {etl}.visit_map_retired (
            source_value text PRIMARY KEY,
            visit_occurrence_id integer NOT NULL UNIQUE,
            person_source_value text
        )""",
    "id_high_water": """
        CREATE TABLE IF NOT EXISTS {etl}.id_high_water (
            table_name text PRIMARY KEY,
            max_id bigint NOT NULL
        )""",
    "build_facts": """
        CREATE TABLE IF NOT EXISTS {etl}.build_facts (
            name text NOT NULL,
            value text
        )""",
    "source_to_concept_map": """
        CREATE TABLE IF NOT EXISTS {etl}.source_to_concept_map (
            source_code varchar(255) NOT NULL,
            source_concept_id integer NOT NULL DEFAULT 0,
            source_vocabulary_id varchar(50) NOT NULL,
            source_code_description varchar(255),
            target_concept_id integer NOT NULL,
            target_vocabulary_id varchar(20),
            valid_start_date date NOT NULL DEFAULT DATE '1970-01-01',
            valid_end_date date NOT NULL DEFAULT DATE '2099-12-31',
            invalid_reason varchar(1),
            PRIMARY KEY (source_vocabulary_id, source_code, target_concept_id)
        )""",
}


def create_etl_tables(conn: psycopg.Connection, schemas: Schemas, role: str | None = None) -> list[str]:
    """Create any missing ETL working tables. Returns the tables created.

    With ``role`` (``target.etl_role``), the role gets SELECT, INSERT and DELETE on the tables this
    call creates, so a table that a new version adds does not fail the next build run under the
    role. Grants on existing tables are left to the administrator."""
    ensure_schema(conn, schemas.etl)
    created = []
    for table, stmt in ETL_TABLES.items():
        if not table_exists(conn, schemas.etl, table):
            execute(conn, schemas.render(stmt))
            created.append(table)
    # a role that does not exist fails every step with a clear error when the build sets it
    if role and created and scalar(conn, "SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)):
        for table in created:
            execute(conn, f"GRANT SELECT, INSERT, DELETE ON {qi(schemas.etl)}.{qi(table)} TO {qi(role)}")
    return created


def truncate_cdm_tables(conn: psycopg.Connection, schemas: Schemas, tables: list[str]) -> None:
    for t in tables:
        if table_exists(conn, schemas.cdm, t):
            execute(conn, f"TRUNCATE TABLE {qi(schemas.cdm)}.{qi(t)}")
