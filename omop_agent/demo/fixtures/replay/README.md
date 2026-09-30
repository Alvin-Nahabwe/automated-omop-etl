# Recorded LLM answers for the offline demo

The files here are **hand-written**, not captured from a live Claude run. They let
`omop-agent demo` and CI run the whole pipeline deterministically with no API key. They are
written for the default demo dataset (`n_people=1500`, `seed=42`) and the synthetic fixture
vocabulary.

Layout: `<stage>/<key>.json`. `key` is the call name used in the code, with `:` written as
`_`.

| stage | keys | what it stands in for |
|---|---|---|
| `annotate` | `source` | interpreting the source profile |
| `design` | `plan`, `table_<cdm table>` | the ETL design. The design deliberately contains two bugs (see below). |
| `vocab_map` | `<SRC_VOCAB>_batch<n>`, `<SRC_VOCAB>_retry<n>` | Usagi-style concept choices; abbreviations first trigger a re-search |
| `repair` | `iteration_<n>` | the repair agent's tool calls, which are re-executed against the live tools |

The two design bugs are there so the demo exercises the repair loop:

1. `visit_occurrence.visit_end_date` uses the discharge date, which is empty for outpatient
   visits. The load fails with a NOT NULL error.
2. ICD-10 codes are looked up without normalising codes written without the dot (`J189`), so
   about 13% of conditions get concept 0. This fails DQD's `standardConceptRecordCompleteness`.

To record real answers instead, run a project with `llm.provider: anthropic` and
`llm.record_dir: ./recordings`, then point `llm.replay_dir` at that folder.
