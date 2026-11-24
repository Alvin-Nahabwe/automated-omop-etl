"""Self-contained demo: synthetic HDSS + clinic data, synthetic vocabulary, recorded LLM answers."""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

from .synth import generate

FIXTURES = Path(str(resources.files("omop_agent.demo").joinpath("fixtures")))
FIXTURE_VOCAB = FIXTURES / "vocab"
FIXTURE_REPLAY = FIXTURES / "replay"


def create_demo_project(
    root: str | Path,
    db_url: str,
    provider: str = "replay",
    n_people: int = 1500,
    seed: int = 42,
    schema_prefix: str = "demo",
    llm_url: str = "http://127.0.0.1:8080/v1",
    llm_model: str = "gpt-oss-120b",
) -> Path:
    """Generate demo data and write ``omop_agent.yaml``. Returns the config path.

    The recorded answers in ``fixtures/replay`` were written for the default ``n_people`` and
    ``seed``. With ``provider='openai_compatible'`` (a local model server at ``llm_url``) or
    ``provider='anthropic'`` (Claude) the model is called live, and its answers are recorded under
    ``recordings/``.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    generate(root / "data", n_people=n_people, seed=seed)
    if provider == "replay":
        llm = f"  provider: replay\n  replay_dir: {FIXTURE_REPLAY}\n"
    elif provider == "openai_compatible":
        llm = (
            f"  provider: openai_compatible\n  base_url: {llm_url}\n  model: {llm_model}\n"
            "  record_dir: ./recordings\n"
        )
    elif provider == "anthropic":
        llm = "  provider: anthropic\n  model: claude-opus-5-5\n  effort: high\n  record_dir: ./recordings\n"
    else:
        llm = "  provider: none\n"
    p = schema_prefix
    # Keep a password that came from the environment out of the file: write the reference.
    url = "${OMOP_AGENT_DB_URL}" if db_url and os.environ.get("OMOP_AGENT_DB_URL") == db_url else db_url
    cfg = root / "omop_agent.yaml"
    cfg.write_text(
        f"""# Demo project: synthetic data only. See omop_agent/demo/fixtures/vocab/README.md.
name: omop-agent-demo
workdir: ./work

source:
  type: csv
  path: ./data

target:
  url: {json.dumps(url)}
  cdm_schema: {p}_cdm
  vocab_schema: {p}_vocab
  staging_schema: {p}_src
  etl_schema: {p}_etl
  results_schema: {p}_results

vocabulary:
  athena_dir: {FIXTURE_VOCAB}

llm:
{llm}
privacy:
  min_cell_count: 5

quality:
  max_repair_iterations: 3
  use_r_dqd: ${{OMOP_AGENT_USE_R_DQD:-false}}
  require_r_dqd: ${{OMOP_AGENT_REQUIRE_R_DQD:-false}}

etl:
  timezone: Africa/Nairobi

cdm_source:
  cdm_source_name: Synthetic HDSS demo
  cdm_source_abbreviation: DEMO
  cdm_holder: omop-agent
  source_description: Synthetic HDSS registry linked to clinic records (no real people).
  source_release_date: 2023-12-31   # the end of the synthetic data: open residency episodes end on it
"""
    )
    return cfg
