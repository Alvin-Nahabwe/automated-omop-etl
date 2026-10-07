"""Shared fixtures.

Tests marked ``postgres`` need a PostgreSQL database; point ``OMOP_AGENT_TEST_DB_URL`` at one
(the role must be allowed to create schemas). Tests marked ``slow`` (scale measurements) run only
with ``OMOP_AGENT_SLOW_TESTS=1``. Each test gets its own uniquely named schemas,
which are dropped afterwards, so tests can share one database.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from omop_agent.config import ProjectConfig
from omop_agent.db import Schemas, connect, execute, qi

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
FIXTURE_VOCAB = REPO / "omop_agent" / "demo" / "fixtures" / "vocab"
DB_URL = os.environ.get("OMOP_AGENT_TEST_DB_URL")


def pytest_collection_modifyitems(config, items):
    skip_pg = pytest.mark.skip(reason="set OMOP_AGENT_TEST_DB_URL to run PostgreSQL tests")
    skip_slow = pytest.mark.skip(reason="set OMOP_AGENT_SLOW_TESTS=1 to run the slow scale tests")
    for item in items:
        if "postgres" in item.keywords and not DB_URL:
            item.add_marker(skip_pg)
        if "slow" in item.keywords and not os.environ.get("OMOP_AGENT_SLOW_TESTS"):
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def demo_data_dir(tmp_path_factory) -> Path:
    from omop_agent.demo.synth import generate

    d = tmp_path_factory.mktemp("demo_data")
    generate(d, n_people=400, seed=7)
    return d


def _new_schemas() -> Schemas:
    tag = uuid.uuid4().hex[:8]
    return Schemas(cdm=f"t{tag}_cdm", vocab=f"t{tag}_vocab", src=f"t{tag}_src", etl=f"t{tag}_etl", results=f"t{tag}_res")


def _drop(schemas: Schemas) -> None:
    conn = connect(DB_URL, autocommit=True)  # type: ignore[arg-type]
    for s in {schemas.cdm, schemas.vocab, schemas.src, schemas.etl, schemas.results}:
        execute(conn, f"DROP SCHEMA IF EXISTS {qi(s)} CASCADE")
    conn.close()


@pytest.fixture
def pg():
    """(connection, schemas) with fresh, empty schemas."""
    schemas = _new_schemas()
    conn = connect(DB_URL)  # type: ignore[arg-type]
    try:
        yield conn, schemas
    finally:
        conn.rollback()
        conn.close()
        _drop(schemas)


def make_config(tmp_path: Path, data_dir: Path, schemas: Schemas, **llm) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "name": "test",
            "workdir": str(tmp_path / "work"),
            "source": {"type": "csv", "path": str(data_dir)},
            "target": {
                "url": DB_URL or "postgresql://unused",
                "cdm_schema": schemas.cdm,
                "vocab_schema": schemas.vocab,
                "staging_schema": schemas.src,
                "etl_schema": schemas.etl,
                "results_schema": schemas.results,
            },
            "vocabulary": {"athena_dir": str(FIXTURE_VOCAB)},
            "llm": llm or {"provider": "none"},
        }
    )


@pytest.fixture
def staged(pg, tmp_path, demo_data_dir):
    """Demo CSVs staged + fixture vocabulary loaded + CDM/ETL tables created.

    Yields (conn, schemas, cfg).
    """
    from omop_agent.cdm.ddl import create_cdm_tables, create_etl_tables
    from omop_agent.profiling.loader import stage_source
    from omop_agent.vocab.loader import load_athena

    conn, schemas = pg
    cfg = make_config(tmp_path, demo_data_dir, schemas)
    stage_source(cfg, conn, schemas)
    load_athena(conn, schemas, FIXTURE_VOCAB)
    create_cdm_tables(conn, schemas, include_vocab=False)
    create_etl_tables(conn, schemas)
    conn.commit()
    yield conn, schemas, cfg
