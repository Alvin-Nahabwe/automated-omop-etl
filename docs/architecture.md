# Architecture

`omop-agent` automates the OHDSI ETL pipeline that data teams (for example at APHRC) run by
hand with White Rabbit, Rabbit-in-a-Hat, Athena, Usagi, SQL/R, Achilles and the Data Quality
Dashboard. The same stages remain, but each is either automated deterministically or drafted
by Claude and checked by code. A human can review and edit every hand-off between stages.

```mermaid
flowchart LR
    SRC[(Source data<br/>CSV / database)] --> STAGE[Stage to Postgres<br/><i>text copy</i>]
    STAGE --> PROF[Profile<br/><i>White Rabbit</i>]
    PROF -->|privacy review| ANN[Interpret profile<br/><b>Claude</b>]
    ATH[Athena download] --> VLOAD[Load vocabulary]
    ANN --> DES[Design ETL<br/><i>Rabbit-in-a-Hat</i><br/><b>Claude</b> + validator]
    VLOAD --> DES
    DES -->|human review| VEVAL[Evaluate vocabulary<br/>coverage]
    VEVAL --> MAP[Map source codes<br/><i>Usagi</i><br/>lexical search + <b>Claude</b> rerank]
    MAP -->|human review| BUILD[Compile + run SQL<br/><i>R / PostgreSQL</i>]
    BUILD --> DQ[Quality checks<br/><i>DQD rules, Achilles-lite</i>]
    DQ -->|fail| REP[Repair agent<br/><b>Claude</b> + tools]
    REP --> BUILD
    DQ -->|after repairs| RDQD[Official DQD + Achilles<br/><i>R, optional</i>]
    RDQD -->|pass| CDM[(OMOP CDM)]
    CDM --> REPORT[Inspection report]
    CDM --> ATLAS[ATLAS / R analysis]
```

## Design principles

1. **Claude drafts, code verifies.** Every LLM output is a pydantic model validated on
   arrival. Concept ids are checked against the loaded vocabulary, since the model can only
   choose among candidates or ids that exist. Generated SQL is `EXPLAIN`ed before it runs.
   Data-quality results come from SQL, never from the model.
2. **Patient data never goes to the LLM.** Prompts contain aggregate metadata only: names,
   types, counts, coarsened ranges and frequent values. Values seen fewer than
   `min_cell_count` times, and every value of a detected direct identifier (names, phones,
   e-mail, ID numbers, addresses, GPS, free text), are withheld by construction
   (`omop_agent/privacy.py`). Detection uses column-name rules (including common local ones
   such as `nupi`, `ccc_no` and the WHO verbal-autopsy name fields) and value patterns; a
   column Claude labels as an identifier or free text is withheld from then on. Before the
   first call a reviewer approves the profile exactly as Claude will see it (the privacy
   review, `privacy_review.py`). The repair agent can
   only count records through a tool that builds the query itself, withholds small groups and
   refuses identifiers. Database errors are redacted before they reach the model. Every prompt
   and tool result is stored verbatim in `work/llm_calls/` for audit.
3. **Human in the loop at the right places.** What is sent to Claude, the ETL spec
   (`etl_spec.yaml`) and the vocabulary mappings (`mappings/*.csv`, Usagi-like columns) are
   each reviewed before the pipeline moves on, unless `--yes` is passed. Spec and mappings are
   plain files; edits are picked up on the next step.
4. **Deterministic replays.** Every LLM call has a `(stage, key)` name. With
   `llm.record_dir` a live run is recorded; with `provider: replay` it runs offline. The demo
   and the test-suite use hand-written replays.
5. **Everything is audited.** `work/audit.jsonl` records stage transitions, LLM calls (with
   token usage), each review gate and its outcome, SQL errors and each repair made to the
   spec.

## Stages and modules

| Stage | Module | Output |
|---|---|---|
| Stage source | `profiling/loader.py` | `src.*` text tables, `staging_manifest.json` |
| Profile | `profiling/profiler.py`, `profiling/render.py` | `profile.json`, `profile.md` |
| Privacy review | `privacy.py`, `privacy_review.py` | `privacy_review.json`, `privacy_review.md` |
| Interpret | `profiling/annotate.py` | `annotations.json` |
| Load vocabulary | `vocab/loader.py` | vocabulary tables |
| Design | `design/designer.py`, `design/validate.py`, `design/render.py` | `etl_spec.yaml`, `etl_spec.md` |
| Evaluate vocabulary | `vocab/evaluate.py` | `vocab_evaluation.json` |
| Map codes | `vocab/mapper.py`, `vocab/review.py` | `mappings/*.csv`, `etl.source_to_concept_map` |
| Build | `etl/compiler.py`, `etl/runner.py` | `sql/*.sql`, CDM tables, `etl_run.json` |
| Quality | `quality/checks.py`, `quality/dqd_r.py` | `dq_results.json` |
| Repair | `quality/repair.py` | updated `etl_spec.yaml`, `repairs.json` |
| Report | `report/inspection.py` | `report.md`, `report.html` |
| Orchestration | `pipeline.py`, `cli.py` | `state.json` |

## The ETL spec

`ETLSpec` (`omop_agent/models.py`) is the hand-off between design and build, in the role of
a Rabbit-in-a-Hat document, and it compiles directly to SQL. Each `TableMapping` becomes one
`INSERT INTO cdm.<table> SELECT ... FROM src.<table> <alias> [JOIN ...] [WHERE ...]`. Each
`FieldMapping.kind` compiles to:

| kind | SQL |
|---|---|
| `column`, `expression`, `constant` | the expression, cast to the CDM column type |
| `person_id` | `JOIN etl.person_map pm ON pm.source_value = (expr)::text`: unmatched rows are dropped (and counted). Required for every `person_id` field, including `person.person_id` |
| `visit_id` | `LEFT JOIN etl.visit_map` (INNER when the field is required) |
| `auto_id` | `max(existing id) + ROW_NUMBER() OVER ()`, so several mappings can feed one table |
| `vocab_code` | `LEFT JOIN vocab.concept` on code+vocabulary, then `Maps to` for `concept_role: standard` |
| `concept_map` | `LEFT JOIN etl.source_to_concept_map` on code+`source_vocabulary_id` |
| `null` | `NULL` |

The id maps are kept across re-runs, so a person keeps the same `person_id`, and no id is
given to another key. After `person` and `visit_occurrence` load, map entries for people or
visits that did not become a row (filtered out by the mappings, or a failed load) are retired
to `etl.person_map_retired` / `etl.visit_map_retired`, so events can't point at them; a retired
key that returns gets its old id back, and new keys are numbered after the largest id in the
map or retired. (Versions before the retired tables deleted such keys, so the first build after
an upgrade can give a deleted key's id to a new key once.) The visit pruning step also records
the person each loaded visit belongs to (`visit_map.person_source_value`). The build
re-validates the spec first, because it may have been edited by hand. Retention is measured
before vocabulary fan-out (`TableLoadResult.retained_rows`), so one-to-many mappings can't hide
dropped records.

Concept lookups yield `0` when the source value exists but has no mapping. When the source
value is NULL they yield NULL, or 0 if the CDM field is required. `varchar(n)` targets are
truncated to `n` characters. The `cdm_source` row is generated from `cdm_source` in the config;
`cdm_version_concept_id` is looked up in the vocabulary (CDM vocabulary, code `CDM v5.4`), or
0 if that vocabulary isn't loaded. Placeholders `{cdm}`, `{vocab}`, `{src}`, `{etl}`, `{results}` in
expressions and custom SQL resolve to the configured schemas.

## Quality gate and repair loop

The built-in checks re-implement the Data Quality Dashboard field- and table-level checks in
PostgreSQL. Names, thresholds and severities are read from DQD's own v5.4 metadata (bundled in
`resources/ohdsi`), so results line up with DQD. An extra `etlRecordRetention` check compares
source rows with rows kept after the person/visit joins for each table mapping. DQD checks the
built-in engine doesn't implement are listed as not applicable, and so is each empty table, as a
single summary row. The run passes when no check with a
blocking severity (default `fatal`) fails, no check listed in `quality.blocking_checks` fails
(by default `measurePersonCompleteness` on `observation_period`, because analyses in ATLAS
need every person to have one) and no ETL step errored. A table mapping that keeps fewer than
`quality.min_record_retention_pct` (default 50%) of its source rows fails the fatal
`etlRecordLoss` check, so a build that silently drops most records cannot pass. Otherwise the
repair agent gets the failures and a set of tools:

* `get_table_mapping`, `get_cdm_table`, `get_source_profile`: read context;
* `count_records`: counts only. The tool builds `SELECT g…, COUNT(*) AS n FROM <table>
  WHERE <typed filters> GROUP BY g HAVING COUNT(*) >= min_cell_count` itself, with bound
  parameters. Identifiers, patient keys and record ids can't be grouped or filtered.
  Vocabulary tables are reference data and are not suppressed;
* `update_field_mapping`, `replace_table_mapping`: change the spec. A change is applied only
  if it passes the SQL guard (`sqlguard.py`) and the PII and person-link rules, and the table
  still compiles and `EXPLAIN`s cleanly. An edit identical to the current mapping is a no-op.
  The repair agent may not make a check pass by removing data: adding or changing a `WHERE`
  filter, adding an inner join, changing the source table, blanking a field or removing a
  field mapping is rejected, and the agent is told to leave such failures to a person.

The ETL is rebuilt and re-checked, up to `quality.max_repair_iterations` times. Each build is
scored: losing data (a CDM table keeping under 95% of the rows an earlier build loaded, or
checks that applied before becoming not applicable) ranks below any number of failing checks,
then ETL errors, then blocking and repairable failures. An iteration that makes the score worse
is rolled back to the best spec, rebuilt, and its repairs are marked `reverted` in
`repairs.json`. `etl_spec.before_repair.yaml` keeps the spec as it was before automatic
repairs, and every change is recorded in `repairs.json` and the audit log.

With `quality.use_r_dqd: true` and R available (the `Dockerfile.r` image, or
`deploy/r/install_r_dqd.sh`), the official Achilles and DataQualityDashboard packages run once,
on the CDM the repair loop settled on, through a generated R script
(`quality/r/run_achilles_dqd.R`). Their results replace the built-in ones for the checks DQD
ran and feed the same gate; the report's engine reads `builtin+dqd`. If R is missing or fails,
the run says so and falls back to the built-in checks, unless `quality.require_r_dqd` is set,
in which case the run fails. Vocabulary coverage is measured again on the final build.

## Review gates, the web app and deployment

The pipeline stops three times for people:

* **privacy**: before the first call to Claude (only when `llm.provider` is `anthropic`). The
  reviewer sees the withheld columns and the profile text exactly as it will be sent. The
  digest covers that text and the privacy and LLM settings, so a new source extract or a
  changed setting needs a new approval;
* **design**: after the ETL design;
* **vocab_map**: after code mapping.

How a gate decides depends on how the pipeline is run:

* `omop-agent run` asks at the terminal; `--yes` skips all gates (for demos and CI), and the
  report then says the run was not fully reviewed.
* `omop-agent run --recorded-approvals`, and every run started from the web app, continue only
  when a reviewer approved what the gate shows *as it is now*. `approvals.py` stores each
  approval in `work/approvals.json` with a sha256 digest. For the design, the digest covers the
  validated spec, so formatting and comments don't count. For the mappings, it covers each
  term's target concept and status. A hand edit, a redraft, a repair, a changed mapping or a
  new source code changes the digest, and the run stops with status `waiting_review`. A copy of
  what was approved is kept under `work/approvals/`, so the web app can show what changed.

A stopped run is continued by a plain run. What the stopped run was asked to redo (start over,
redraft the design, map again) is kept in `work/pending_run.json` until that step has run, so
stopping at the privacy review, which comes before that work, doesn't lose the request.

Every gate writes a `review_gate` audit event (approved, skipped, stopped or not checked). The
report's Reviews table is built from those events and the approvals, and a PASSED verdict
carries a "Not fully reviewed" caveat when a gate was skipped, the repair agent changed the
design after it was approved, or mappings nobody reviewed (`UNCHECKED`) were loaded.

The web app (`omop_agent/web/`, FastAPI with server-rendered Jinja pages) is an interface to
the same files the command line uses, not a separate system:

* Mapping edits change cells of `mappings/<vocabulary>.csv` (`vocab/review.py:update_review_row`,
  under a file lock), exactly as a spreadsheet edit would. `import_review` applies them on the
  next run. Each edit records `reviewed_by`, `reviewed_at` and `review_note`, which are carried
  into `decisions.json`.
* Runs are separate processes (`web/worker.py`) that hold `work/jobs/run.lock`, one per
  project. Their records and logs are in `work/jobs/`. Mapping edits are refused while a run is
  in progress, and forms carry the state the reviewer saw (expected concept and status, gate
  digest), so nobody approves or overwrites a change they did not see.
* Accounts (`web/users.py`) are local, have three roles (viewer, reviewer, operator) and are
  managed with `omop-agent users` on the server. Sessions are signed cookies (strict SameSite,
  `Secure` behind HTTPS) that end when the account changes. Every form has a CSRF token. Pages
  load no external assets and no inline scripts, under a strict Content-Security-Policy.

`deploy/` packages it for one server: an image with the CLI and the app (non-root), and
PostgreSQL 16 with `pg_trgm` and an application role that owns its database but is not a
superuser. The database is not published outside the compose network. See `docs/deploy.md`.
