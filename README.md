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
| Implement the ETL | R + PostgreSQL | Compiles the spec into plain, readable SQL (`sql/*.sql`) and loads the CDM, with the OHDSI conventions built in: one row per person and visit, merged observation periods, records in the table of their domain, condition and drug eras. |
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
NULL; the other is ICD-10 codes written without the dot. The first CDM already passes the gate,
but with convention-level failures from these two bugs, which the repair agent fixes one per
iteration: the build ends those visits on their start date (the CDM convention) and reports it as
`etlDefaultedValue`, and the agent writes that fallback into the spec; then the ICD-10 codes
fail `standardConceptRecordCompleteness`, and the agent fixes the code normalisation. The final
report has fewer failed checks. To run the same demo against Claude live, use
`--provider anthropic` (it needs an API key, see below).

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
  `na_values` lists the missing-value markers staged as NULL (default `NA`, `NULL`, `\N` and `.`
  for CSV files, none for databases; `[]` keeps them as data).
* `target`: the PostgreSQL URL and the schema names (`cdm`, `vocab`, `src` staging, `etl` working
  tables, `results`); `etl_role` is an optional least-privilege role for the build (see "Privacy
  and safety").
* `vocabulary.athena_dir`: an unzipped Athena download.
* `llm`: `provider` (`anthropic` | `replay` | `none`), `model` (default `claude-opus-5-5`), `effort`, `record_dir`/`replay_dir`.
* `privacy`: `min_cell_count` (default 5), `send_values_to_llm`, extra `pii_columns`.
* `mapping.auto_approve_confidence`, `quality.max_repair_iterations`, `quality.blocking_severities`, `quality.use_r_dqd`.
* `cdm_source`: the `cdm_source` row; `source_release_date` is the date of the data extract.
  Observation periods still open there (no exit date) end on it. Unset, the build derives it from
  the exit dates and reports it (`etlSourceReleaseDate`); a derived date after the day of the build
  (placeholder or mistyped exit dates) fails the gate until it is set.
* `etl.timezone`: the site's time zone, e.g. `Africa/Nairobi`. Times written with an offset or `Z`
  (ODK submission times, for example) are converted to it. Unset, an offset is ignored and the
  time is kept as written; a spec that reads a column with `Z`, `+00:00` or several different
  offsets then fails validation, because its times would not be on one clock.
* `etl.partial_dates`: how a date with an unknown day or month (`00`, `97`-`99`, or no day, as in
  `06/1975`) is read, except birth and death dates, which follow the THEMIS rules. `widest` (the
  default): a start date takes the first day, an end date the last day, any other date the 15th.
  `reject`: the date is left NULL.

`${VAR}` references are read from the environment, so credentials stay out of the file.

### What the build does beyond the mappings

The spec says how each source table fills a CDM table. The build applies these OHDSI conventions
to every spec, and reports what each one did, in the inspection report and as checks (see
[docs/architecture.md](docs/architecture.md)):

* **People and visits from several tables.** Every person and visit_occurrence mapping adds the
  keys of its rows to the id map (prefix the keys of a source that numbers its own, e.g.
  `'EMR:' || patient_no`). `person` and `death` keep one row per person and `visit_occurrence` one
  row per visit and person (several rows of a visit span its earliest start to its latest end); a
  mapping's `dedup_order` chooses the row (a missing value ranks last), and duplicates and
  conflicting rows are reported (`etlDuplicateRecords`, `etlKeyConflict`), never resolved
  silently. A visit key shared by two people still fails `isPrimaryKey`, and one coarser than a
  visit (rows that are separate stays) fails `etlVisitGrain`. A record links only to a visit of
  its own person.
* **Stable ids.** Person and visit ids are kept across re-runs and are never given to another key.
  People and visits that a load filtered out are retired from the id maps with their ids
  (`etl.person_map_retired`, `etl.visit_map_retired`) and get the same ids back when they return.
* **Values that cannot be read.** A value a field cannot convert (a date such as 31/02/2019, text
  in a number) becomes NULL and is counted per field (`etlInvalidValue`), instead of failing the
  whole table. A required value that is NULL follows the CDM convention and is counted: a concept
  becomes 0 and a visit, visit detail or drug end date the start date (`etlDefaultedValue`); a row
  without a start date, birth year or death date is dropped (`etlRequiredFieldNull`). Partial
  dates follow `etl.partial_dates` and time zone offsets `etl.timezone`. The report lists them per
  field ("Values not loaded as written").
* **Observation periods.** With `observation_period: table_mapping`, residency or follow-up
  episodes are loaded one row per episode, with no end date while an episode is still open. The
  build merges each person's overlapping and back-to-back episodes, ends open ones at the source
  release date or at death (real exits are kept), drops and counts episodes that end before they
  start (`etlObservationPeriodMerge`), and gives persons without a valid episode a period from
  their first to their last record.
* **Records in the table of their domain** (`route_by_domain`, on by default). A condition,
  procedure, drug, device or measurement record whose standard concept is a condition,
  procedure, drug, device, measurement or observation of another table is moved to that table
  after loading; a measurement with a result whose concept is not a measurement or an
  observation goes to `observation`, which keeps the result. Moves are reported
  (`etlDomainRouting` and the report's "Records moved" table).
* **Eras** (`derive_eras`, on by default). `condition_era` and `drug_era` are derived with the
  OHDSI era logic; drug eras per standard RxNorm or RxNorm Extension ingredient, so vaccines
  without an ingredient get none (`measureDrugEraCoverage`). Spec SQL that loads an era table
  replaces the build's step for that table, and the validator warns about it.

The steps that move records and derive eras do not block the build: when one fails, the CDM is
still checked, and the report names the step and what the CDM lacks because of it.

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
  as valid standard concepts in the right domain (for the main concept of a condition,
  procedure, drug, device or measurement code list also another domain the table accepts, whose
  records the build moves to their own table; such a choice is never approved automatically).
  Person and visit ids always come from the id
  maps, so events can't attach to the wrong patient, and a record links only to a visit of its
  own person (checked after the load by `fkVisitPerson`). The repair agent may not filter out
  records, add inner joins or blank fields, nor change who exists in the CDM (person keys, the
  visit key, new person, visit or death mappings, their filters): those are left to a person. An
  edit is dry-run first, and refused when it would make a required value whose NULL drops the row
  (a start date, birth year, death date) or a record's visit key NULL for rows where it was not,
  or change which observation-period episodes are open. A repair that
  makes results worse, or loads fewer records, is rolled back; and DQ verdicts come from SQL.
* **A PASS means the data is there.** The gate passes only when no blocking build step failed, no
  check of a blocking severity (`quality.blocking_severities`, default `fatal`) failed, and no
  check listed in `quality.blocking_checks` failed (by default: persons without an observation
  period). Among the fatal checks: every table mapping keeps at least
  `quality.min_record_retention_pct` (default 50%) of its source rows (`etlRecordLoss`), a
  record's visit belongs to its own person (`fkVisitPerson`), a person has at most one death
  (`measureDuplicateDeaths`), and at most 100 - `min_record_retention_pct` percent of a mapping's
  records with a visit key name a key that no visit mapping reads (`etlVisitLinkLoss`). Values
  that could not be read are counted per field, and a required value that comes out NULL follows
  the CDM convention and is counted, never hidden: a concept becomes 0, a visit or drug end date
  the start date (`etlDefaultedValue`), an observation period end means the episode is still open,
  and a row without a start date, birth year or death date is dropped (`etlRequiredFieldNull`; the
  retention checks count it as lost). The steps that move records by domain and derive the eras do
  not block: when one fails, the report lists it apart, with what the CDM lacks. A CDM built from
  a design or mappings that nobody approved, or that automatic repairs changed after approval, is
  reported as "PASSED, not fully reviewed".
* **Source files are staged exactly or not at all.** CSVs are decoded strictly: set
  `source.encoding` for Latin-1 or cp1252 exports. Rows with the wrong number of fields stop
  the run with their line numbers, unless you set `source.skip_ragged_rows`.
* **Everything is auditable.** `audit.jsonl` holds the full trail. `llm_calls/` holds every
  prompt, every tool result and the agent's intermediate messages, verbatim. `repairs.json`
  records every automatic change with its rationale.
* **Reports are marked confidential.** The inspection report lists source terms, rare ones
  included, because reviewers need them. Every report is written with a shareable copy
  (`report.shareable.*`) in which rare terms are withheld and database error messages lose the
  record values they can quote. Some of its tables show counts below `min_cell_count` as `<k`,
  but totals, differences and the data-quality counts are exact, so a small count can often be
  worked out from them.
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

`tests/test_hdss.py` builds a small HDSS dataset (`tests/fixtures/hdss/`, about 35 people) in
which every source row is one named case (`tests/hdss_cases.py`): duplicate and corrected
registrations, partial and invalid dates, residency episodes that overlap, touch, stay open or end
after the extract date, clinic patients who are not HDSS residents, ward stays, diagnoses whose
codes belong to other domains, drug courses, and deaths from two sources. It checks the CDM row by
row, and the check results and the gate. `tests/perf/scale.py` is not part of the suite: it
measures the build on synthetic sources of a million people and a full Athena vocabulary (see
"Measured performance" in `docs/architecture.md`).

## Upgrading

Updating to the version with the conventions above ("What the build does beyond the mappings")
changes the CDM that an unchanged spec builds. The first build of a project after the update says
so: one log line, and an
`upgrade_notice` event in `audit.jsonl` that lists the new behaviours. Existing files keep
loading (`omop_agent.yaml`, `etl_spec.yaml`, `etl_run.json`, `dq_results.json`, `profile.json`,
`decisions.json`), and approved designs and mappings stay approved: the new spec fields
(`route_by_domain`, `derive_eras`, `dedup_order`) are left out of the design's approval digest
while they hold their defaults. The report and the web app's design page show what the build now
does with every spec. What changes:

**Database and permissions**

* The build creates four ETL tables (`person_map_retired`, `visit_map_retired`, `id_high_water`,
  `build_facts`) and grants them to `target.etl_role`. It installs helper functions in the ETL
  schema with a version stamp; a new version must be installed once by their owner (a build that
  runs as another user stops with a message saying so). Installing drops EXECUTE grants made by
  hand to other roles: the new functions are executable by PUBLIC (PostgreSQL's default) and
  granted to `etl_role`.
* A step that fails for a missing privilege is not sent to the repair agent; its error names the
  GRANT to run.
* Spec SQL may no longer call functions named `omop_*` (the build's helper functions in the ETL
  schema, which keep its counts) or use Unicode-escaped names and strings (`U&"..."`,
  `U&'...'`): a spec that does fails validation.
* Spec SQL runs with DateStyle `ISO, YMD` and `standard_conforming_strings` on, and with the
  time zone `etl.timezone` when it is set (the server's zone otherwise), in the build as in the
  validator and the repair agent's counts; the files in `sql/`
  start with these settings as comments. In custom SQL, a hard cast of a day-first text date
  with a four-digit year (`'31/12/2020'::date`) now fails whatever the server's DateStyle, and
  one with a two-digit year (`'31/12/20'::date`) is read year-first (2031-12-20) without an
  error; use `to_date(x, 'DD/MM/YYYY')` or `to_date(x, 'DD/MM/YY')`. The validator reports such a
  cast of a column whose profile shows a day-first format.
* Keys that a load filters out are retired with their ids instead of deleted (see "Stable ids").
  Older versions deleted them: a deleted key that returns gets a new id once, and the id of a
  deleted key above the largest id still in the map can be given to a new key once.

**Staging and profiling**

* CSV staging stores `NA`, `NULL`, `\N` and `.` as NULL the next time the source is staged
  (counted per column in `staging_manifest.json` and the `source_staged` audit event);
  `source.na_values: []` keeps them as data.
* `profile.json` gains value shapes, the counts of values that do not fit a column's type, of
  partial dates and of NA markers, time zone offsets and format notes. An old profile loads and
  gives the same prompt text, so an approved privacy review stays approved until the source is
  profiled again; a new profile shows more, so it needs a new privacy review. Minimum and maximum
  dates use complete, valid dates only. Type inference reads a sample spread over the table and
  knows more layouts (Stata and SAS dates, `dd/mm/yyyy HH:MM`, two-digit years, times of day,
  Excel day numbers), so a column profiled again can change type.
* New validator errors: a column with `Z`, `+00:00` or several offsets read while `etl.timezone`
  is unset; a date conversion of a column of day numbers; an ISO cast of a column profiled as
  day-first or month-first.

**Reading values**

* A value that a field's conversion cannot read becomes NULL and is counted (`etlInvalidValue`)
  instead of failing the table's step. A row whose `WHERE` cannot be decided (an unreadable value
  in it) is left out and counted as dropped by the retention checks.
* A NULL required value follows the CDM convention (`etlDefaultedValue`, `etlRequiredFieldNull`)
  instead of failing the step on a NOT NULL constraint; rows dropped this way count as lost for
  `etlRecordRetention` and the fatal `etlRecordLoss`.
* Dates: a separator in a template matches any one separator (as `to_date` does); `06/1975` under
  `DD/MM/YYYY` is a partial date, completed by `etl.partial_dates` (`etlPartialDate`); ISO dates
  are read under any template. Times with an offset are converted to `etl.timezone`, or keep
  their wall clock when it is unset; `to_timestamp` keeps the instant only with a `TZH` / `OF`
  template. Floating-point casts are read as `numeric` (a later integer cast rounds halves away
  from zero).

**People, visits and keys**

* Every person and visit_occurrence mapping adds its keys to the id maps (steps `person_map_2`,
  `visit_map_2`, ...).
* `person`, `death` and `visit_occurrence` load one row per key, where several rows of a key used
  to fail `isPrimaryKey` or load twice; duplicates and conflicts are counted (`etlDuplicateRecords`,
  `etlKeyConflict`, `etlKeyFanOut`). `NULLS LAST` is added to every `dedup_order` item without a
  NULLS clause, and a death date known to the day wins over one completed from a year or month.
* A record whose visit key names another person's visit now loads without a visit (counted in
  `etlVisitLinkage`).
* New fatal checks: `etlVisitGrain` (more than 5% of a mapping's visits rolled up from rows that
  are separate stays), `etlVisitLinkLoss`, `measureDuplicateDeaths` and `fkVisitPerson`.
* `etlKeyConflict` fails for person keys whose best-ranked rows still disagree on sex or birth
  year: set a `dedup_order`, or fix the source. Disagreements with the row an earlier mapping
  loaded are counted in its description.
* The repair agent may not change who exists in the CDM (person_id expressions, the person and
  visit keys, new person, visit or death mappings, their filters), nor make a record's visit key
  NULL; it may still fix the person key of a mapping that cannot run at all. It is no longer
  given `isPrimaryKey` of person and visit_occurrence or `etlDuplicateRecords.person`.

**Observation periods**

* A NULL `observation_period_end_date` now means the episode is open (stored as `9999-12-31` until
  the merge); kind `null` on that field is a validator error (write kind `constant` with
  `NULL::date`). The `observation_period_merge` step ends open episodes at the source release
  date or at death (one that starts after either keeps one day), and merges overlapping and
  back-to-back episodes, so `measureObservationPeriodOverlap` no longer fails for them. With
  `table_mapping`, persons without a valid episode get a period from their records.
* `cdm_source.source_release_date` is the configured date, or else the one the merge derives from
  the exit dates (from the record dates with `from_events`); it is the build date only when no
  observation periods are built. `etlSourceReleaseDate` fails while the date is derived and open
  episodes end on it (fatal when the derived date is after the day of the build), or when a
  configured date is older than the data. `etl.build_facts` is emptied at the start of every
  build.
* New checks: `etlObservationPeriodMerge` (episodes dropped because they end before they start,
  or treated as open because their exit date could not be read; convention, 5%, sent to the
  repair agent), with a fatal `etlRecordLoss` copy when more than
  100 - `quality.min_record_retention_pct` percent of the episodes are dropped; and
  `withinObservationPeriod` (records outside every period of their person; characterization,
  5%).
* `from_events` together with an `observation_period` mapping or custom SQL is a validator error.
* `measurePersonCompleteness.observation_period` still blocks, but is not sent to the repair agent
  once the build derived periods; the repair agent may not change which episodes are open, and
  the repair loop compares builds by the valid episodes loaded plus the periods the fill gave.

**Routing by domain**

* Records move to the table of their standard concept's domain (`route_by_domain: true`). A spec
  that routed records by hand (a second mapping of the same codes with a filter on
  `{vocab}.concept.domain_id`) now loads them twice: the validator warns, and every build reports
  such a mapping as a failed `etlHandRouting` (convention, not sent to the repair agent). Remove
  the hand-made routing, or set `route_by_domain: false`.
* Code mapping accepts a concept of another domain for the main concept of a condition,
  procedure, drug, device or measurement code list (such a choice is never approved
  automatically), and keeps up to 3 candidates of each other accepted domain beside the 5 of the
  target domain. A source vocabulary whose fields expect concepts of different domains gets a
  validator warning (`vocab_domain_conflict` in the audit), and none of its concepts is approved
  automatically.
* `load_vocabulary` creates the vocabulary indexes also on a vocabulary loaded by someone else.

**Eras**

* `condition_era` and `drug_era` are derived by default (`derive_eras: true`). Spec SQL that loads
  an era table replaces the build's step, runs at the era table's place and gets a validator
  warning; the repair agent may not add a mapping into an era table, and is not given the checks
  of an era table (built by the build, or by such SQL) while `derive_eras` is on. The design no
  longer plans era tables.
* New checks: `measureConditionEraCompleteness`, `measureDrugEraCompleteness` and
  `measureDrugEraCoverage`; with the era tables' own checks they add about 70 results to a
  typical build.

**Build order, results and the report**

* Custom SQL for a table without a mapping runs at that table's place in the load order, not after
  everything else: custom SQL that read a later table (observation SQL that reads `death`, say)
  now runs before that table is loaded. Move such SQL to the later table, or split it.
* The route and era steps do not block the build: their errors are reported (`etlStepError`,
  convention, not sent to the repair agent) and the checks still run.
* A second mapping into a table gets its own ETL check key (`etlRecordRetention.measurement.measurement_2`).
* Rows the build derives (observation periods, eras) and records it moves are counted apart from
  the rows loaded, in the log, the audit totals and the report. The report's ETL table gains
  "Statistics" and "Visit links" columns; "Date formats detected" becomes "Reading the values";
  and the report has new tables "Values not loaded as written" and "Records moved to the table of
  their concept's domain".

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
* Left out on purpose:
  * `dose_era` (OHDSI publishes no script for it), and post-processing of the built eras by spec
    SQL (spec SQL replaces the era step instead).
  * Moving records out of `observation` (DQD has no domain check there), moving records with
    concept 0 by their source concept, moving concepts of other domains (values, units: `fkDomain`
    still reports them), and the value of a 'Maps to value' relationship (ICD-10 "history of"
    codes become observations without a value).
  * Links to moved records are not repointed: moved records get new ids. The build loads `cost`,
    `fact_relationship` and `episode_event` after moving records, so spec SQL for them sees the
    final ids.
  * A visit map keyed by person and visit number: when visit numbers restart for each person, the
    visit key must include the person key (e.g. `v.individual_id || ':' || v.visit_no`).
  * Joining observation periods a few days apart: periods merge only when they overlap or are back
    to back (as DQD defines it).
  * A birth date from the earliest record when only the year of birth is known: `birth_datetime`
    is left NULL instead.
  * Other markers of an unknown day or month (such as `15` or `UN`): only `00` and `97`-`99` are
    read as unknown.
  * Reconciling the records of each source table one by one, design guidance for HDSS events
    beyond residency (migrations, births and pregnancies, mother-child links), de-duplication of
    `location` and `care_site`, and moving `specimen` records by domain.
* `visit_detail`, `note`/NLP and `cost` are not generated automatically; add them as table
  mappings or `custom_sql` in the spec. Custom SQL that writes event records and an era table in
  one step runs at the era table's place, so its event records are not moved by domain and the
  observation periods do not read them: split it in two.
* The rules that recognise hand-made routing see only a literal event-domain name next to
  `domain_id` in a `WHERE` or inner-join condition, and the same codes of the same rows in two
  mappings of one source table.
* Reading values tolerantly costs time: an ISO date or a numeric template such as `DD/MM/YYYY`
  takes 1-3 s per million values, ODK timestamps with offsets about 6 s, and other templates
  (`DD/MM/YY`, `DDMonYYYY`, `HH12 AM`) about 13 s, against 0.2-0.5 s for a plain cast. A field
  that converts the same unreadable value twice (`COALESCE(a::date, b::date)`) counts it twice,
  a template that is not a literal (written in hand-made SQL) is read at about 150 µs a value,
  and a cast of a `jsonb` value (`(j->'x')::int`) fails validation: cast the text instead
  (`(j->>'x')::int`). An unreadable value that a `COALESCE` replaces is still counted: in an
  optional date field it fails `etlInvalidValue` at convention severity although the field has a
  value (for a required field the check sees that the build had no NULL to complete or drop).
* An exit date that exists but cannot be read is loaded as an open episode (it runs to the release
  date or to death); `etlObservationPeriodMerge` counts such episodes.
* Some steps and checks added for HDSS data were not yet timed on a full HDSS (about 1M people and
  14M records): one row per key, the era steps and checks, and the route steps with their final
  SQL. The observation-period merge took about 6 s for 900,000 episodes. Profiling a date-heavy
  table takes about twice as long as before.
* The profile's split between values that fit and values that do not can differ from the build's
  reader in rare cases: one-digit minutes (`7:5`) count as not fitting, a two-digit year-first date
  (`19/03/05`) is read day-first, day numbers are read as Excel serial days (Stata, SAS and Unix
  count from other dates; the note says so), and ratios such as `1:16` are profiled as times of
  day. Offsets that are each too rare to list are counted but not checked against `etl.timezone`.
* The built-in checks cover most of DQD's table and field checks (20 of 27 check types, and
  they agree with DQD wherever both run). Concept-level checks (plausible gender, plausible
  units), the schema checks and Achilles need the official R route: the R image
  (`Dockerfile.r`, see docs/deploy.md), or R installed with `deploy/r/install_r_dqd.sh`, and
  `quality.use_r_dqd: true`. It runs once, on the final CDM, and adds about 2–3 minutes. The
  checks added for the HDSS conventions (`etl*`, `measureDrugEraCompleteness`,
  `measureDrugEraCoverage`, `fkVisitPerson`, `measureDuplicateDeaths`, `withinObservationPeriod`)
  are the tool's own; DQD in R does not run them.
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
2.0). See `omop_agent/resources/ohdsi/NOTICE`. The tests compare the eras with the OHDSI era
scripts of [OHDSI/ETL-Synthea](https://github.com/OHDSI/ETL-Synthea) (Apache License 2.0),
translated to PostgreSQL; see `tests/fixtures/NOTICE`.
