# omop-agent: an AI agent for automated OMOP ETL

`omop-agent` automates the OHDSI ETL pipeline for converting patient-level source data (CSV
exports or a relational database) into the **OMOP Common Data Model v5.4** on PostgreSQL.

It follows the same steps a data team takes by hand with White Rabbit, Rabbit-in-a-Hat,
Athena, Usagi, R/SQL, Achilles and the Data Quality Dashboard. Each step is either automated
in code or drafted by Claude and then checked in code. The agent stops for human review where
judgement matters, and it keeps fixing its own ETL until the data-quality gate passes.

| OHDSI pipeline step | Tool you would use | What `omop-agent` does |
|---|---|---|
| Summarize the source data | White Rabbit | Stages the source into Postgres and profiles every column in SQL (types, date formats, null %, frequencies, ranges). Claude then interprets the scan: grain, person keys, coding systems, data issues. |
| Create the ETL design | Rabbit-in-a-Hat | Claude drafts a table/field-level `etl_spec.yaml`, which code validates against the CDM spec, the source profile and the vocabulary, and `EXPLAIN`s. **You review it.** |
| Download vocabulary | Athena | Loads your Athena download (COPY with indexes and trigram search). |
| Evaluate vocabulary | manual | Checks that the required vocabularies are present and measures code coverage (e.g. what % of ICD-10 records reach a standard concept). |
| Map source codes | Usagi | Lexical and trigram candidate search, then Claude picks the best standard concept (only from real candidates) with a confidence score, re-searching after expanding abbreviations. Output is Usagi-style CSVs. **You review them.** |
| Implement the ETL | R + PostgreSQL | Compiles the spec into plain, readable SQL (`sql/*.sql`) and loads the CDM. |
| Perform quality assessment | Achilles, DQD | Runs the DQD field and table checks in Postgres using DQD's own thresholds and severities. Optionally runs the official R Achilles + DQD too. |
| Pass? No → fix | you | A **repair agent** (Claude with tools) diagnoses failing checks or SQL errors, inspects aggregates, and patches the spec. The CDM is rebuilt and re-checked. |
| Inspection report | manual | `report.html` / `report.md`: verdict, mapping coverage, DQ results, the repairs made, CDM characterization, open questions. |
| Analysis | R, ATLAS | The CDM is ready for ATLAS/HADES; the report lists the registration steps. |

See [docs/architecture.md](docs/architecture.md) for the design.

## Quick start: the offline demo

The demo uses a synthetic population (an HDSS register linked to clinic visits, ICD-10
diagnoses, prescriptions, labs and verbal-autopsy deaths) and a small synthetic vocabulary. It
runs with recorded LLM answers, so it needs no API key and no Athena download. You only need
PostgreSQL.

```bash
pip install -e .
omop-agent demo ./omop-demo --db-url postgresql://user:pass@localhost:5432/omop
open ./omop-demo/work/report.html
```

The recorded design contains two realistic bugs. One is an outpatient `visit_end_date` that is
NULL; the other is ICD-10 codes written without the dot. The demo shows the loop at work: the
load fails, the repair agent fixes the spec, then a quality check fails and the agent fixes
the code normalisation, and the rebuilt CDM passes. To run the same demo against Claude live,
use `--provider anthropic` (it needs an API key, see below).

## Using it on your data

```bash
pip install -e .
export ANTHROPIC_API_KEY=...          # or CLAUDE_API_KEY
omop-agent init my-project --source ./exports --db-url 'postgresql://${PGUSER}:${PGPASSWORD}@db:5432/omop' --athena-dir ./athena
cd my-project
omop-agent run                        # stops for review after the design and after the mapping
```

What `run` does, and the stage commands you can re-run on their own after editing an artifact:

```text
omop-agent profile          # stage source -> src.*, write profile.md (scan report)
omop-agent annotate         # Claude interprets the profile -> annotations.json
omop-agent vocab-load       # load Athena CSVs
omop-agent design           # Claude drafts etl_spec.yaml / etl_spec.md   <- review & edit
omop-agent map              # evaluate vocabulary + map codes -> mappings/*.csv  <- review & edit
omop-agent apply-mappings   # load your CSV edits into source_to_concept_map
omop-agent build            # compile + load + DQ checks + repair loop
omop-agent check            # DQ checks only
omop-agent report           # report.md / report.html
omop-agent status
```

Everything lands in `work/`: the profile, spec, mappings, generated SQL, DQ results,
repairs, `audit.jsonl`, and `llm_calls/` (every prompt and answer, verbatim).

### Configuration (`omop_agent.yaml`)

`omop-agent init` writes a commented template. The main settings:

* `source`: `type: csv` with `path`, or `type: database` with a SQLAlchemy `url` and `schema`.
* `target`: the PostgreSQL URL and the schema names (`cdm`, `vocab`, `src` staging, `etl` working tables, `results`).
* `vocabulary.athena_dir`: an unzipped Athena download.
* `llm`: `provider` (`anthropic` | `replay` | `none`), `model` (default `claude-opus-5-5`), `effort`, `record_dir`/`replay_dir`.
* `privacy`: `min_cell_count` (default 5), `send_values_to_llm`, extra `pii_columns`.
* `mapping.auto_approve_confidence`, `quality.max_repair_iterations`, `quality.blocking_severities`, `quality.use_r_dqd`.

`${VAR}` references are read from the environment, so credentials stay out of the file.

## Privacy and safety

Health data is sensitive, so these rules are enforced in code rather than left to prompts:

* **No patient rows reach the LLM.** Claude sees table and column names, types, counts,
  ranges and frequent values only.
* **Small-cell suppression.** A value seen fewer than `min_cell_count` times is never shown to
  the LLM. That covers the profile, vocabulary terms and repair-agent queries. Rare codes are
  left for human mapping.
* **Direct identifiers are detected and withheld.** Names, phone numbers, national IDs,
  addresses and GPS columns show no values at all. You can list extra identifier columns in
  `privacy.pii_columns`.
* **The repair agent can only run aggregates.** Each query must return a count column `n`.
  Groups below the threshold are withheld, and queries that mention identifier columns are
  refused. All queries run in read-only transactions.
* **Nothing Claude produces is trusted blindly.** Concept ids are checked against the loaded
  vocabulary, SQL is `EXPLAIN`ed before it's accepted, and DQ verdicts come from SQL.
* **Everything is auditable.** `audit.jsonl` holds the full trail, `llm_calls/` holds every
  prompt and answer, and `repairs.json` records every automatic change with its rationale.

`provider: none` runs every deterministic stage (staging, profiling, lexical mapping, build,
checks, report) with no LLM at all. In that mode you write `etl_spec.yaml` yourself.

## How Claude is used

The agent calls Claude through the official `anthropic` Python SDK (`omop_agent/llm/`):

* **Structured outputs**: annotation, design and mapping answers are parsed straight into
  pydantic models (`client.beta.messages.stream(..., output_format=Model)`).
* **Tool use**: the repair agent runs on the SDK tool runner with six tools (read the mapping,
  the CDM spec and the profile; run an aggregate query; update a field; replace a mapping).
* **Adaptive thinking** at configurable `effort`, **prompt caching** of the system prompts,
  and **server-side refusal fallbacks** (`fallbacks: "default"`), which you can turn off with
  `llm.use_fallbacks: false`.
* **Record and replay**: set `llm.record_dir` to capture a live run, then `provider: replay`
  reruns it deterministically (useful for audits, CI and teaching).

## Development

```bash
pip install -e ".[dev]"
export OMOP_AGENT_TEST_DB_URL=postgresql://omop:omop@localhost:5432/omop_test   # PostgreSQL tests
pytest -q
```

Tests that need PostgreSQL are skipped when `OMOP_AGENT_TEST_DB_URL` is unset. CI runs the
suite plus the offline demo against a `postgres:16` service.

## Limitations and roadmap

* The target database is PostgreSQL only. Source databases can be anything SQLAlchemy reads.
* Records are not yet re-routed by domain: a source code that maps to a concept in another
  domain stays in the table the spec chose, and DQD flags it.
* The eras (`condition_era`, `drug_era`), `visit_detail`, `note`/NLP and `cost` tables are not
  generated automatically; add them as table mappings or `custom_sql` in the spec.
* The built-in checks cover DQD's table and field levels. Concept-level checks (plausible
  gender, plausible units) need the official R DQD (`quality.use_r_dqd: true`).
* The demo vocabulary and the recorded LLM answers are synthetic and hand-written. Never load
  the fixture vocabulary into a real CDM.

## Licence and attribution

The OMOP CDM v5.4 DDL and field metadata, and the Data Quality Dashboard check metadata, come
from [OHDSI/CommonDataModel](https://github.com/OHDSI/CommonDataModel) and
[OHDSI/DataQualityDashboard](https://github.com/OHDSI/DataQualityDashboard) (Apache License
2.0). See `omop_agent/resources/ohdsi/NOTICE`.
