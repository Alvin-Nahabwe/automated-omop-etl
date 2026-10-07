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
build ends those visits on their start date (the CDM convention) and reports it as
`etlDefaultedValue`, and the repair agent writes that fallback into the spec; then the ICD-10
codes fail `standardConceptRecordCompleteness`, the agent fixes the code normalisation, and the
rebuilt CDM passes. To run the same demo against Claude live, use `--provider anthropic` (it
needs an API key, see below).

## The review web app and the Docker bundle

For a team, install the bundle in `deploy/` on the server that holds the data: PostgreSQL 16
and a small web app, started with one `docker compose up`. Reviewers then work in a browser
instead of editing YAML and CSV files:

* **ETL design**: read Claude's draft table by table, see what changed since it was last
  approved (including changes made by the repair agent), and approve it.
* **Code mappings**: each local code with how often it occurs, Claude's proposed concept and
  reasoning, and the other candidates. Search the vocabulary, pick a concept, approve, flag for
  discussion or reject, with a note. Then approve the mappings as a whole.
* **Runs**: an operator starts the pipeline in the background and watches its log. A run
  pauses at each review step until the current design or mappings are approved.
* **Report and audit trail**: the inspection report, and every approval and edit with the
  reviewer's name.

Accounts are local and managed on the server (`omop-agent users add NAME --role
viewer|reviewer|operator`). The app writes to the same files as the command line, so both can
be used on the same project. Setup, HTTPS and backups: [docs/deploy.md](docs/deploy.md).
Without Docker: `pip install -e ".[web]"` and `omop-agent serve --projects DIR`.

## Using it on your data

```bash
pip install -e .
export ANTHROPIC_API_KEY=...          # or CLAUDE_API_KEY
omop-agent init my-project --source ./exports --db-url 'postgresql://${PGUSER}:${PGPASSWORD}@db:5432/omop' --athena-dir ./athena
cd my-project
omop-agent run                        # stops for review after the design and after the mapping
```

Re-running `omop-agent run` resumes. It reuses the staged source, the annotations, your edited
`etl_spec.yaml` and your reviewed mappings. `--redesign` redrafts the spec (after backing up
the current one as `etl_spec.<n>.yaml`), `--remap` maps every code again, and `--fresh` starts
over. You can also run the stages one at a time:

```text
omop-agent profile          # stage source -> src.*, write profile.md (scan report)
omop-agent annotate         # Claude interprets the profile -> annotations.json
omop-agent vocab-download   # fetch + unpack the bundle from the link in $ATHENA_DOWNLOAD_URL
omop-agent vocab-load       # load Athena CSVs
omop-agent design           # Claude drafts etl_spec.yaml / etl_spec.md   <- review & edit
omop-agent map              # evaluate vocabulary + map codes -> mappings/*.csv  <- review & edit
omop-agent apply-mappings   # load your CSV edits into source_to_concept_map
omop-agent build            # compile + load + DQ checks + repair loop
omop-agent check            # DQ checks only
omop-agent report [--shareable]   # report.md / report.html
omop-agent status           # stages and review gates
omop-agent approve design --by NAME     # record a review (also: vocab_map)
omop-agent run --recorded-approvals     # stop at a gate unless its current artifact was approved
```

Everything lands in `work/`: the profile, spec, mappings, generated SQL, DQ results,
repairs, `audit.jsonl`, and `llm_calls/` (every prompt and answer, verbatim).

### Getting the vocabulary from Athena

1. Log in at [athena.ohdsi.org](https://athena.ohdsi.org). On the **Download** tab, keep the
   pre-selected infrastructure vocabularies and add the ones your source codes need. For
   African HDSS or EMR data that usually means **ICD10** (WHO, not only ICD10CM), **SNOMED**,
   **LOINC**, **UCUM** and **RxNorm**, plus **RxNorm Extension** for drug products outside
   the US.
2. Athena e-mails a download link when the bundle is ready. The link gives access to your
   licensed download, so keep it out of the config file and chat: put it in the
   `ATHENA_DOWNLOAD_URL` environment variable.
3. Run `omop-agent vocab-download` (streams and unpacks into `vocabulary.athena_dir`; the link is
   never printed), then `omop-agent vocab-load`. Athena may send the zip gzip-encoded; the
   download command handles that. A full bundle is about 2 GB unpacked and takes several
   minutes to load.

`omop-agent map` reports which vocabularies your spec needs but the download lacks (for
example WHO `ICD10` when only `ICD10CM` was selected).

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

* **No patient rows reach the LLM.** Claude sees table and column names, types, counts and
  frequent values. Ranges are coarsened: dates to years, continuous numbers to two
  significant figures. Code-like ranges whose end is a rare value are withheld.
* **Small-cell suppression.** A value seen fewer than `min_cell_count` times is never shown to
  the LLM. That covers the profile, vocabulary terms and repair-agent results. Rare codes are
  left for human mapping.
* **Direct identifiers are detected and withheld.** Column-name rules cover names (including
  `deceased_name`, `hh_head_name`, `fname`, "Mother's Name" and the WHO verbal autopsy name
  questions), staff names, phone numbers, e-mail, national, health and register numbers (CCC,
  UPI, Huduma, OPD/IP numbers), addresses (plot and house numbers), GPS (including ODK
  geopoints) and free text (narratives, comments, "other, specify"). Value rules catch e-mail
  addresses, Kenyan and international phone numbers, coordinates and free text. Columns Claude
  itself labels as identifiers or free text are withheld from then on. Values of patient and
  parent keys are never listed, nor are those of household, visit and other record keys with
  more than 50 distinct values. Withheld columns can't be used in the ETL spec or sent for
  mapping; the validator rejects them. List extra identifier columns in `privacy.pii_columns`,
  and columns flagged by mistake (a CD4 count) in `privacy.not_pii_columns` (their values are
  collected the next time the source is profiled).
* **A person checks what will be sent before Claude is first called.** Detection is
  heuristic, so a project that uses Claude stops for a *privacy review*: the withheld columns
  and the profile exactly as Claude will see it (`work/privacy_review.md`, or the Privacy tab of
  the web app). Nothing goes to Claude until a reviewer approves it, and a re-staged source or a
  changed privacy setting needs a new approval.
* **The repair agent can only count.** It can't write SQL to read data. Its `count_records`
  tool takes a table, up to three grouping columns and typed filters, and the tool builds the
  query itself with bound parameters. It returns `COUNT(*)` per group, withholds groups below
  the threshold, and refuses identifiers, patient keys and record ids. Database error messages
  are redacted before the model sees them.
* **LLM-written SQL is checked before it runs.** Spec expressions must be single expressions
  over the staged source and vocabulary. Statement separators and file, network or session
  functions are rejected. They are `EXPLAIN`ed before they are accepted. For defence in depth,
  set `target.etl_role` to a least-privilege role and the build runs under it. Grant it these
  privileges once `omop-agent vocab-load` has created the tables (shown for the default
  schemas; add your vocabulary schema if it is not `cdm`):

  ```sql
  GRANT USAGE ON SCHEMA src, cdm, etl TO omop_etl;
  GRANT SELECT ON ALL TABLES IN SCHEMA src TO omop_etl;
  GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA cdm, etl TO omop_etl;
  GRANT omop_etl TO omop;  -- the user in target.url
  ```

  ETL tables that the build creates later (a new version can add some) are granted to the
  role automatically. The role also needs EXECUTE on the helper functions the build installs in
  the ETL schema (PUBLIC has it by default). A new version of those functions is installed by
  the first build that runs as their owner.
* **Nothing Claude produces is trusted blindly.** Concept ids must exist in your vocabulary
  as valid standard concepts in the right domain. Person and visit ids always come from the id
  maps, so events can't attach to the wrong patient. The repair agent may not filter out
  records, add inner joins or blank fields; a repair that makes results worse, or loads fewer
  records, is rolled back; and DQ verdicts come from SQL.
* **A PASS means the data is there.** Besides fatal DQD failures, the gate blocks when a table
  mapping loads less than half of its source rows (`etlRecordLoss`) and when persons have no
  observation period (`quality.blocking_checks`). A required value that comes out NULL follows
  the CDM convention and is counted, never hidden: a concept becomes 0 and a visit or drug end
  date the start date (`etlDefaultedValue`), and a row without a start date, birth year or death
  date is dropped (`etlRequiredFieldNull`, and a loss for the retention checks). A CDM built
  from a design or mappings that nobody approved, or that automatic repairs changed after
  approval, is reported as "PASSED, not fully reviewed".
* **Source files are staged exactly or not at all.** CSVs are decoded strictly: set
  `source.encoding` for Latin-1 or cp1252 exports. Rows with the wrong number of fields stop
  the run with their line numbers, unless you set `source.skip_ragged_rows`.
* **Everything is auditable.** `audit.jsonl` holds the full trail. `llm_calls/` holds every
  prompt, every tool result and the agent's intermediate messages, verbatim. `repairs.json`
  records every automatic change with its rationale.
* **Reports are marked confidential.** The inspection report lists source terms, rare ones
  included, because reviewers need them. Every report is written with a shareable copy
  (`report.shareable.*`) in which rare terms are withheld and database error messages lose the
  record values they can quote.
* `privacy.send_values_to_llm: false` goes further. Claude then sees statistics only, code
  mapping is lexical only, and the repair agent can only group by concept ids.

`provider: none` runs every deterministic stage (staging, profiling, lexical mapping, build,
checks, report) with no LLM at all. In that mode you write `etl_spec.yaml` yourself.

## How Claude is used

The agent calls Claude through the official `anthropic` Python SDK (`omop_agent/llm/`):

* **Structured outputs**: annotation, design and mapping answers are parsed straight into
  pydantic models (`client.beta.messages.stream(..., output_format=Model)`).
* **Tool use**: the repair agent runs on the SDK tool runner with six tools (read the mapping,
  the CDM spec and the profile; count records; update a field; replace a mapping). Tool input
  errors go back to the model with details, and usage is summed over every turn.
* **Adaptive thinking** at configurable `effort`, **prompt caching** of the system prompts
  (including the design context shared by every per-table call),
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
suite plus the offline demo against a `postgres:16` service, and builds the Docker bundle and
runs the demo inside it.

## Limitations and roadmap

* **Residual privacy risk in the repair agent.** `count_records` returns exact counts for groups
  of `min_cell_count` or more. By comparing the totals of several filters it chooses (for
  example "all records" against "all but one value"), a model could infer a small count, and
  learn whether some rare value matches a filter, without ever seeing the value itself.
  Closing this fully needs noisy or rounded counts. Until then, if that residual is not
  acceptable, set `quality.max_repair_iterations: 0` (no repair agent) or
  `privacy.send_values_to_llm: false`.
* Site-specific concepts loaded into the vocabulary schema (ids of 2,000,000,000 and above)
  are treated as reference data and are not suppressed.

* The target database is PostgreSQL only. Source databases can be anything SQLAlchemy reads.
* Person and visit ids are kept stable across re-runs and are never given to another key.
  People and visits that a load filtered out are retired from the id maps with their ids
  (`etl.person_map_retired`, `etl.visit_map_retired`) and get the same ids back when they return.
  Versions before the retired tables deleted such keys: after an upgrade, a deleted key that
  returns gets a new id, and the id of a deleted key above the largest id still in the map can
  be given to a new key once.
* Records are not yet re-routed by domain: a source code that maps to a concept in another
  domain stays in the table the spec chose, and DQD flags it.
* The eras (`condition_era`, `drug_era`), `visit_detail`, `note`/NLP and `cost` tables are not
  generated automatically; add them as table mappings or `custom_sql` in the spec.
* The built-in checks cover most of DQD's table and field checks (19 of 27 check types, and
  they agree with DQD wherever both run). Concept-level checks (plausible gender, plausible
  units), the schema checks and Achilles need the official R route: the R image
  (`Dockerfile.r`, see docs/deploy.md), or R installed with `deploy/r/install_r_dqd.sh`, and
  `quality.use_r_dqd: true`. It runs once, on the final CDM, and adds about 2–3 minutes.
* The demo vocabulary and the recorded LLM answers are synthetic and hand-written. Never load
  the fixture vocabulary into a real CDM.
* The web app has local accounts only (no single sign-on or two-factor login yet). Keep it on
  the internal network or VPN, behind HTTPS. The ETL design is reviewed in the browser but
  edited on the server (`etl_spec.yaml`).
* Identifier detection is rule-based (column names and value patterns), which is why the
  privacy review exists: a person still has to look.

## Licence and attribution

The OMOP CDM v5.4 DDL and field metadata, and the Data Quality Dashboard check metadata, come
from [OHDSI/CommonDataModel](https://github.com/OHDSI/CommonDataModel) and
[OHDSI/DataQualityDashboard](https://github.com/OHDSI/DataQualityDashboard) (Apache License
2.0). See `omop_agent/resources/ohdsi/NOTICE`.
