"""Scale measurements of the build, on synthetic HDSS-like sources and a full Athena vocabulary.

Not part of the test suite (pytest does not collect this file). It prints the timings recorded in
docs/architecture.md ("Measured performance"). Run it against a scratch database:

    python tests/perf/scale.py --db-url postgresql://omop:omop@localhost:5432/omop_test --vocab athena2

``--vocab`` names a schema that holds a full Athena download (CONCEPT, CONCEPT_RELATIONSHIP,
CONCEPT_ANCESTOR, CONCEPT_SYNONYM with the trigram indexes of ``vocab.loader.ensure_indexes``). The
script only reads it. It builds into its own schemas (``--prefix``, default ``p_perf``) and drops
them at the end unless ``--keep``. Phases (``--phase``, default all):

* ``events``: 1M persons (5% second registrations, half of them with another sex), 3M visit rows
  (10% of the visits in two ward rows), 3M diagnoses with ICD-10 codes of the vocabulary (some map
  to Observation, Procedure and Measurement concepts, so they are routed) and 2M prescriptions of
  2,000 local drug names mapped to standard Drug concepts (2% to Device concepts): one row per key
  (layer 4), the id maps, visit links, routing and the eras, then the checks;
* ``merge``: the same persons with 5M residency episodes (overlaps, back-to-back moves, gaps, open
  and inverted episodes), a death for one person in 17 and 2M records dated from 2002 to 2023: the
  observation_period insert, merge and fill, the checks (withinObservationPeriod), then the profile
  stage over these sources;
* ``search``: the per-term candidate search of the vocab_map stage (``mapper._Searcher``) for
  local drug, lab and diagnosis names, with the other accepted domains of each.

A vocabulary loaded by an older version, or by someone else, may lack indexes that
``load_vocabulary`` creates (``vocab.loader.ensure_indexes``), such as the one on
``concept (concept_id)``. With ``--indexed-copy`` the script copies the concept table into
``<prefix>_vocab`` (the other vocabulary tables are views of ``--vocab``) and creates the missing
indexes there, so it measures the vocabulary as a build of this version would index it (about 2 GB
more disk).

The sources are written with SQL (deterministic, no random()), as text columns like staging does.
At the default size the schemas take about 2 GB at the peak (the events phase); the merge phase
starts after the events phase's sources and CDM rows are dropped. All phases take about 20 minutes
with the vocabulary in the operating system's cache.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omop_agent.cdm.ddl import create_etl_tables  # noqa: E402
from omop_agent.config import ProjectConfig  # noqa: E402
from omop_agent.db import Schemas, connect, execute, qi, scalar  # noqa: E402
from omop_agent.vocab.loader import ensure_indexes  # noqa: E402
from omop_agent.etl import runner  # noqa: E402
from omop_agent.models import ETLSpec, FieldMapping, TableMapping  # noqa: E402
from omop_agent.quality import checks  # noqa: E402
from omop_agent.vocab import mapper  # noqa: E402
from omop_agent.workspace import Workspace  # noqa: E402


def F(target: str, kind: str, expr: str | None = None, **kw) -> FieldMapping:  # noqa: N802
    return FieldMapping(target_field=target, kind=kind, expression=expr, **kw)


def person_mapping() -> TableMapping:
    return TableMapping(
        target_table="person", source={"table": "individuals", "alias": "i"}, dedup_order="i.reg_date::date DESC",
        fields=[
            F("person_id", "person_id", "i.individual_id"),
            F("gender_concept_id", "expression", "CASE i.sex WHEN '1' THEN 8507 WHEN '2' THEN 8532 END"),
            F("year_of_birth", "expression", "EXTRACT(YEAR FROM to_date(i.dob, 'DD/MM/YYYY'))::int"),
            F("month_of_birth", "expression", "EXTRACT(MONTH FROM to_date(i.dob, 'DD/MM/YYYY'))::int"),
            F("day_of_birth", "expression", "EXTRACT(DAY FROM to_date(i.dob, 'DD/MM/YYYY'))::int"),
            F("birth_datetime", "expression", "to_date(i.dob, 'DD/MM/YYYY')::timestamp"),
            F("race_concept_id", "constant", "0"),
            F("ethnicity_concept_id", "constant", "0"),
            F("person_source_value", "column", "i.individual_id"),
            F("gender_source_value", "column", "i.sex"),
        ])


def events_spec() -> ETLSpec:
    visits = TableMapping(
        target_table="visit_occurrence", source={"table": "clinic_visits", "alias": "v"},
        where="v.visit_type <> 'EMERG'",
        fields=[
            F("visit_occurrence_id", "visit_id", "v.visit_id"),
            F("person_id", "person_id", "v.individual_id"),
            F("visit_concept_id", "expression", "CASE v.visit_type WHEN 'OPD' THEN 9202 WHEN 'IPD' THEN 9201 END"),
            F("visit_start_date", "expression", "v.visit_date::date"),
            F("visit_start_datetime", "expression", "v.submitted_at::timestamp"),
            F("visit_end_date", "expression", "v.discharge_date::date"),
            F("visit_type_concept_id", "constant", "32817"),
            F("visit_source_value", "column", "v.visit_type"),
        ])
    conditions = TableMapping(
        target_table="condition_occurrence", source={"table": "diagnoses", "alias": "d"},
        fields=[
            F("condition_occurrence_id", "auto_id"),
            F("person_id", "person_id", "d.individual_id"),
            F("condition_concept_id", "vocab_code", "d.icd10", vocabulary="ICD10"),
            F("condition_start_date", "expression", "d.dx_date::date"),
            F("condition_type_concept_id", "constant", "32817"),
            F("visit_occurrence_id", "visit_id", "d.visit_ref"),
            F("condition_source_value", "column", "d.icd10"),
            F("condition_source_concept_id", "vocab_code", "d.icd10", vocabulary="ICD10", concept_role="source"),
        ])
    drugs = TableMapping(
        target_table="drug_exposure", source={"table": "prescriptions", "alias": "p"},
        fields=[
            F("drug_exposure_id", "auto_id"),
            F("person_id", "person_id", "p.individual_id"),
            F("drug_concept_id", "concept_map", "p.drug", vocabulary="SRC_DRUG", target_domain="Drug"),
            F("drug_exposure_start_date", "expression", "p.date_prescribed::date"),
            F("drug_exposure_end_date", "expression", "p.date_prescribed::date + GREATEST(COALESCE(p.days::int, 1), 1) - 1"),
            F("drug_type_concept_id", "constant", "32817"),
            F("quantity", "expression", "p.qty::numeric"),
            F("days_supply", "expression", "p.days::int"),
            F("visit_occurrence_id", "visit_id", "p.visit_ref"),
            F("drug_source_value", "column", "p.drug"),
        ])
    return ETLSpec(person_key={"table": "individuals", "expression": "individual_id"},
                   visit_key={"table": "clinic_visits", "expression": "visit_id"},
                   observation_period="table_mapping", table_mappings=[person_mapping(), visits, conditions, drugs])


def merge_spec() -> ETLSpec:
    periods = TableMapping(
        target_table="observation_period", source={"table": "residency_episodes", "alias": "r"},
        fields=[
            F("observation_period_id", "auto_id"),
            F("person_id", "person_id", "r.individual_id"),
            F("observation_period_start_date", "expression", "to_date(r.start_date, 'DD/MM/YYYY')"),
            F("observation_period_end_date", "expression", "to_date(r.end_date, 'DD/MM/YYYY')"),
            F("period_type_concept_id", "constant", "32817"),
        ])
    deaths = TableMapping(
        target_table="death", source={"table": "deaths", "alias": "x"},
        fields=[
            F("person_id", "person_id", "x.individual_id"),
            F("death_date", "expression", "x.dod::date"),
            F("death_type_concept_id", "constant", "32817"),
        ])
    records = TableMapping(  # records in and out of the periods: the fill and withinObservationPeriod
        target_table="condition_occurrence", source={"table": "diagnoses", "alias": "d"},
        fields=[
            F("condition_occurrence_id", "auto_id"),
            F("person_id", "person_id", "d.individual_id"),
            F("condition_concept_id", "constant", "0"),
            F("condition_start_date", "expression", "d.dx_date::date"),
            F("condition_type_concept_id", "constant", "32817"),
        ])
    return ETLSpec(person_key={"table": "individuals", "expression": "individual_id"},
                   observation_period="table_mapping", table_mappings=[person_mapping(), periods, records, deaths])


# ---- sources -------------------------------------------------------------------------------

def key(expr: str) -> str:
    return f"'KH' || lpad(({expr})::text, 7, '0')"


def write_individuals(conn, src: str, n: int) -> None:
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.individuals AS
SELECT {key('p')} AS individual_id,
       CASE WHEN p % 50 = 0 THEN '9' ELSE (1 + p % 2)::text END AS sex,
       CASE WHEN p % 100 = 1 THEN '00/' || lpad((1 + p % 12)::text, 2, '0') || '/' || (1940 + p % 80)
            ELSE to_char(DATE '1940-01-01' + (p::bigint * 7919 % 29000)::int, 'DD/MM/YYYY') END AS dob,
       to_char(DATE '2005-01-01' + p % 5000, 'YYYY-MM-DD') AS reg_date
FROM generate_series(1, {n}) p
UNION ALL  -- 5% second registrations: every other one corrects the sex
SELECT {key('p')}, CASE WHEN p % 40 = 0 THEN (2 - p % 2)::text ELSE (1 + p % 2)::text END,
       to_char(DATE '1940-01-01' + (p::bigint * 7919 % 29000)::int, 'DD/MM/YYYY'),
       to_char(DATE '2015-01-01' + p % 2000, 'YYYY-MM-DD')
FROM generate_series(20, {n}, 20) p""")


def write_events(conn, src: str, vocab: str, n: int, visits: int, diagnoses: int, prescriptions: int) -> dict:
    # visit v belongs to person 1 + v * 7919 mod n; one visit in 10 is an admission with two ward rows
    person_of = f"1 + (v::bigint * 7919 % {n})"
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.clinic_visits AS
SELECT v::text AS visit_id, {key(person_of)} AS individual_id,
       (DATE '2012-01-01' + (v::bigint * 31 % 4000)::int)::text AS visit_date,
       CASE WHEN v % 3 = 0 THEN NULL
            WHEN v % 10 = 0 THEN (DATE '2012-01-01' + (v::bigint * 31 % 4000)::int + 2)::text
            ELSE (DATE '2012-01-01' + (v::bigint * 31 % 4000)::int)::text END AS discharge_date,
       CASE WHEN v % 50 = 1 THEN 'EMERG' WHEN v % 10 = 0 THEN 'IPD' ELSE 'OPD' END AS visit_type,
       CASE WHEN v % 4 = 0 THEN to_char(DATE '2012-01-01' + (v::bigint * 31 % 4000)::int, 'YYYY-MM-DD')
                                || 'T09:30:00+03:00' END AS submitted_at
FROM generate_series(1, {visits}) v
UNION ALL  -- the second ward row of the admissions
SELECT v::text, {key(person_of)}, (DATE '2012-01-01' + (v::bigint * 31 % 4000)::int + 2)::text,
       (DATE '2012-01-01' + (v::bigint * 31 % 4000)::int + 5)::text, 'IPD', NULL
FROM generate_series(10, {visits}, 10) v""")
    # ICD-10 codes of the vocabulary with a standard target; every third code, in code order
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.perf_icd10 AS
SELECT ROW_NUMBER() OVER (ORDER BY concept_code) AS k, concept_code
FROM (SELECT DISTINCT c.concept_code FROM {vocab}.concept c
      JOIN {vocab}.concept_relationship r ON r.concept_id_1 = c.concept_id AND r.relationship_id = 'Maps to'
                                           AND r.invalid_reason IS NULL
      WHERE c.vocabulary_id = 'ICD10' AND c.concept_id % 3 = 0) x""")
    n_codes = int(scalar(conn, f"SELECT COUNT(*) FROM {src}.perf_icd10"))
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.diagnoses AS
SELECT d::text AS diag_id, {key(person_of)} AS individual_id, v::text AS visit_ref,
       (DATE '2012-01-01' + (v::bigint * 31 % 4000)::int)::text AS dx_date, c.concept_code AS icd10
FROM (SELECT d, 1 + (d - 1) % {visits} AS v, 1 + (d::bigint * 104729 % {n_codes}) AS k
      FROM generate_series(1, {diagnoses}) d) x
JOIN {src}.perf_icd10 c ON c.k = x.k""")
    # 2,000 local drug names: 1,960 standard Drug concepts (RxNorm and RxNorm Extension, clinical and
    # branded drugs), 40 Device concepts (routed to device_exposure)
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.perf_drugs AS
SELECT ROW_NUMBER() OVER (ORDER BY concept_id) AS k, concept_id, vocabulary_id
FROM (SELECT concept_id, vocabulary_id FROM {vocab}.concept
      WHERE domain_id = 'Drug' AND standard_concept = 'S' AND invalid_reason IS NULL
        AND concept_class_id IN ('Clinical Drug', 'Branded Drug', 'Quant Clinical Drug')
        AND vocabulary_id IN ('RxNorm', 'RxNorm Extension') AND concept_id % 101 = 0
      ORDER BY concept_id LIMIT 1960) d
UNION ALL
SELECT 1960 + ROW_NUMBER() OVER (ORDER BY concept_id), concept_id, vocabulary_id
FROM (SELECT concept_id, vocabulary_id FROM {vocab}.concept
      WHERE domain_id = 'Device' AND standard_concept = 'S' AND invalid_reason IS NULL AND concept_id % 7 = 0
      ORDER BY concept_id LIMIT 40) x""")
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.prescriptions AS
SELECT r::text AS rx_id, {key(person_of)} AS individual_id, v::text AS visit_ref,
       (DATE '2012-01-01' + (v::bigint * 31 % 4000)::int)::text AS date_prescribed,
       'DRUG' || lpad((1 + r::bigint * 7 % 2000)::text, 4, '0') AS drug,
       (ARRAY['3', '5', '7', '14', '30', '90'])[1 + r % 6] AS days, (10 + r % 50)::text AS qty
FROM (SELECT r, 1 + (r::bigint * 3 - 1) % {visits} AS v FROM generate_series(1, {prescriptions}) r) x""")
    for t in ("individuals", "clinic_visits", "diagnoses", "prescriptions"):
        execute(conn, f"ANALYZE {src}.{t}")
    return {"icd10_codes": n_codes}


def write_episodes(conn, src: str, n: int) -> None:
    # 1-9 consecutive episodes per person, 5 on average, with gaps of -10..+30 days (overlaps,
    # back-to-back moves, gaps); one person in 10 still resident (no exit), 0.5% inverted rows
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.residency_episodes AS
WITH ep AS (
    SELECT p, k, n_ep, DATE '2002-01-01' + p % 3000 AS base,
           40 + (p::bigint * k * 7919 % 700)::int AS len, (p::bigint * k * 104729 % 41)::int - 10 AS gap
    FROM (SELECT p, 1 + p % 9 AS n_ep FROM generate_series(1, {n}) p) pp, LATERAL generate_series(1, pp.n_ep) k
), placed AS (
    SELECT p, k, n_ep, len, base + COALESCE(SUM(len + gap) OVER (PARTITION BY p ORDER BY k
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)::int AS s
    FROM ep
)
SELECT {key('p')} AS individual_id, to_char(s, 'DD/MM/YYYY') AS start_date,
       CASE WHEN k = n_ep AND p % 10 = 0 THEN NULL
            WHEN (p + k) % 200 = 0 THEN to_char(s - 5, 'DD/MM/YYYY')
            ELSE to_char(s + len, 'DD/MM/YYYY') END AS end_date
FROM placed""")
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.deaths AS
SELECT {key('p')} AS individual_id, (DATE '2004-01-01' + (p::bigint * 31 % 6000)::int)::text AS dod
FROM generate_series(1, {n}, 17) p""")
    execute(conn, f"""
CREATE UNLOGGED TABLE {src}.diagnoses AS
SELECT {key('1 + d::bigint * 7919 % ' + str(n))} AS individual_id,
       (DATE '2002-01-01' + (d::bigint * 31 % 8000)::int)::text AS dx_date
FROM generate_series(1, {2 * n}) d""")
    for t in ("residency_episodes", "deaths", "diagnoses"):
        execute(conn, f"ANALYZE {src}.{t}")


# ---- timing --------------------------------------------------------------------------------

class Timers:
    """The count queries and ANALYZEs of the runner, and every check query, timed by patching."""

    def __init__(self) -> None:
        self.counts: dict[str, float] = {}
        self.analyze: dict[str, float] = {}
        self.checks: dict[str, float] = {}

    @contextlib.contextmanager
    def patched(self):
        counts, analyze, evaluate = runner._counts, runner._analyze, checks._evaluate

        def timed_counts(conn, step, res):
            t = time.time()
            try:
                return counts(conn, step, res)
            finally:
                self.counts[step.name] = time.time() - t

        def timed_analyze(conn, relations):
            t = time.time()
            try:
                return analyze(conn, relations)
            finally:
                for r in relations:
                    self.analyze[r] = self.analyze.get(r, 0.0) + (time.time() - t) / max(len(relations), 1)

        def timed_evaluate(ctx, res, sql):
            t = time.time()
            try:
                return evaluate(ctx, res, sql)
            finally:
                self.checks[res.key] = time.time() - t

        runner._counts, runner._analyze, checks._evaluate = timed_counts, timed_analyze, timed_evaluate
        try:
            yield self
        finally:
            runner._counts, runner._analyze, checks._evaluate = counts, analyze, evaluate


def schema_mb(conn, schemas: Schemas) -> float:
    names = sorted({schemas.cdm, schemas.src, schemas.etl})
    return float(scalar(conn, """
        SELECT COALESCE(SUM(pg_total_relation_size(c.oid)), 0) / 1048576.0 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = ANY(%s) AND c.relkind IN ('r', 'm')""",
                         (names,)))


def build(conn, schemas: Schemas, cfg: ProjectConfig, ws: Workspace, spec: ETLSpec, title: str) -> None:
    timers = Timers()
    with timers.patched():
        t = time.time()
        result = runner.run_etl(conn, schemas, spec, cfg, ws)
        build_s = time.time() - t
        t = time.time()
        report = checks.run_builtin_checks(conn, schemas, cfg, spec, result)
        checks_s = time.time() - t
    print(f"\n### {title}: build {build_s:.1f} s, checks {checks_s:.1f} s ({len(report.results)} results), "
          f"schemas {schema_mb(conn, schemas):.0f} MB")
    print("| step | rows | seconds | of which counts |\n|---|---:|---:|---:|")
    for s in result.steps:
        rows = s.rows_inserted if not s.moved else sum(m.moved_rows for m in s.moved)
        note = f" ERROR {s.error[:120]}" if s.error else ""
        print(f"| {s.step} | {rows:,} | {s.seconds or 0:.1f} | {timers.counts.get(s.step, 0):.1f} |{note}")
    print(f"ANALYZE in total: {sum(timers.analyze.values()):.1f} s")
    slow = sorted(timers.checks.items(), key=lambda kv: -kv[1])[:12]
    print("slowest checks: " + "; ".join(f"{k} {v:.1f} s" for k, v in slow))
    for s in result.steps:
        if s.stats and s.kind in ("insert", "merge", "derived"):
            print(f"  {s.step} stats: {s.stats}")


def profile(conn, schemas: Schemas, cfg: ProjectConfig) -> None:
    """The profile stage over the sources of the merge phase (the 5M-row residency table has two
    day-first date columns)."""
    from omop_agent.profiling import profiler

    seconds: dict[str, float] = {}
    original = profiler._profile_table

    def timed(conn_, schema, table, *rest):
        t = time.time()
        try:
            return original(conn_, schema, table, *rest)
        finally:
            seconds[table] = time.time() - t

    profiler._profile_table = timed
    try:
        prof = profiler.profile_staging(conn, schemas, cfg)
    finally:
        profiler._profile_table = original
    conn.rollback()
    print("\n### profile")
    for tp in prof.tables:
        print(f"{tp.name}: {tp.n_rows:,} rows, {len(tp.columns)} columns, {seconds.get(tp.name, 0):.1f} s")


# ---- candidate search ----------------------------------------------------------------------

SEARCH_TERMS = {
    "Drug": ["paracetamol 500mg tab", "amoxicillin 250mg caps", "coartem", "artemether lumefantrine 20/120",
             "ors sachets", "zinc sulphate 20mg", "metformin 500", "amlodipine 5mg", "cotrimoxazole 960",
             "ceftriaxone 1g inj", "quinine injection", "tenofovir lamivudine dolutegravir", "ferrous sulphate",
             "folic acid 5mg", "ibuprofen 400mg", "albendazole 400", "metronidazole 200mg", "salbutamol inhaler",
             "prednisolone 5mg", "hydrochlorothiazide", "nifedipine 20mg", "insulin soluble",
             "benzathine penicillin", "oxytocin 10 iu", "magnesium sulphate inj", "gentamicin 80mg",
             "chlorpheniramine", "diclofenac 50", "omeprazole 20mg", "ciprofloxacin 500mg"],
    "Measurement": ["haemoglobin", "malaria rdt", "blood glucose random", "hiv rapid test", "cd4 count",
                    "viral load", "serum creatinine", "urinalysis protein", "blood pressure systolic", "weight",
                    "height", "temperature", "muac", "hba1c", "white cell count", "platelet count",
                    "sputum afb", "genexpert mtb", "alt", "potassium", "sodium", "cholesterol total",
                    "pulse rate", "respiratory rate", "oxygen saturation", "hepatitis b surface antigen",
                    "syphilis rpr", "stool microscopy", "widal", "haematocrit"],
    "Condition": ["malaria", "pneumonia", "urti", "diarrhoea with dehydration", "hypertension",
                  "type 2 diabetes", "hiv positive", "pulmonary tb", "anaemia", "gastroenteritis",
                  "acute malnutrition", "asthma", "urinary tract infection", "skin infection", "typhoid fever",
                  "otitis media", "conjunctivitis", "epilepsy", "heart failure", "stroke", "road traffic injury",
                  "burns", "dog bite", "pregnancy", "abortion incomplete", "pre-eclampsia", "neonatal sepsis",
                  "measles", "worms", "dental caries"],
}


def search(conn, schemas: Schemas) -> None:
    from omop_agent.cdm.spec import ACCEPTED_DOMAINS

    print("\n### per-term candidate search")
    print("| domain | terms | other accepted domains | median ms | p90 ms | max ms | first term ms |\n"
          "|---|---:|---|---:|---:|---:|---:|")
    tables = {"Drug": "drug_exposure", "Measurement": "measurement", "Condition": "condition_occurrence"}
    for domain, terms in SEARCH_TERMS.items():
        also = [d for d in ACCEPTED_DOMAINS.get(tables[domain], ()) if d != domain]
        searcher = mapper._Searcher(conn, schemas)  # a new run: no cached pools or queries
        times = []
        for t in terms:
            started = time.time()
            searcher.search(t, domain, 10, also)
            times.append((time.time() - started) * 1000)
        q = statistics.quantiles(times, n=10)
        print(f"| {domain} | {len(terms)} | {', '.join(also) or '-'} | {statistics.median(times):.0f} | {q[-1]:.0f} | "
              f"{max(times):.0f} | {times[0]:.0f} |")


def indexed_copy(conn, vocab: str, target: str) -> None:
    """``target``: a copy of ``vocab``.concept with the indexes of ``ensure_indexes``, and views of the
    other vocabulary tables (their own indexes serve the views)."""
    t = time.time()
    execute(conn, f"CREATE SCHEMA {qi(target)}")
    execute(conn, f"CREATE TABLE {qi(target)}.concept AS SELECT * FROM {qi(vocab)}.concept")
    for table in ("concept_relationship", "concept_ancestor", "concept_synonym", "drug_strength", "vocabulary",
                  "domain", "concept_class", "relationship", "source_to_concept_map"):
        execute(conn, f"CREATE VIEW {qi(target)}.{table} AS SELECT * FROM {qi(vocab)}.{table}")
    conn.commit()
    ensure_indexes(conn, Schemas(cdm=target, vocab=target, src=target, etl=target, results=target))
    execute(conn, f"ANALYZE {qi(target)}.concept")
    conn.commit()
    print(f"indexed copy of {vocab}.concept in {target} ({time.time() - t:.0f} s)")


# ---- main ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--vocab", default="athena2", help="schema with a full Athena vocabulary (read only)")
    ap.add_argument("--prefix", default="p_perf")
    ap.add_argument("--persons", type=int, default=1_000_000)
    ap.add_argument("--phase", choices=["all", "events", "merge", "search"], default="all")
    ap.add_argument("--indexed-copy", action="store_true",
                    help="measure on a copy of the concept table with the indexes load_vocabulary creates")
    ap.add_argument("--keep", action="store_true", help="keep the schemas")
    args = ap.parse_args()

    n = args.persons
    vocab = f"{args.prefix}_vocab" if args.indexed_copy else args.vocab
    schemas = Schemas(cdm=f"{args.prefix}_cdm", vocab=vocab, src=f"{args.prefix}_src", etl=f"{args.prefix}_etl",
                      results=f"{args.prefix}_res")
    conn = connect(args.db_url)
    print(f"PostgreSQL {scalar(conn, 'SHOW server_version')}, work_mem {scalar(conn, 'SHOW work_mem')}, "
          f"shared_buffers {scalar(conn, 'SHOW shared_buffers')}, vocabulary {args.vocab}: "
          f"{scalar(conn, f'SELECT COUNT(*) FROM {qi(args.vocab)}.concept'):,} concepts")
    work = Path(tempfile.mkdtemp(prefix="omop_perf_"))
    cfg = ProjectConfig.model_validate({
        "name": "perf", "workdir": str(work), "source": {"type": "csv", "path": str(work)},
        "target": {"url": args.db_url, "cdm_schema": schemas.cdm, "vocab_schema": schemas.vocab,
                   "staging_schema": schemas.src, "etl_schema": schemas.etl, "results_schema": schemas.results},
        "llm": {"provider": "none"}, "cdm_source": {"source_release_date": "2023-12-31"},
        "etl": {"timezone": "Africa/Nairobi"},
    })
    ws = Workspace(cfg.workdir)

    def drop() -> None:
        for s in (schemas.cdm, schemas.src, schemas.etl, schemas.results) + ((vocab,) if args.indexed_copy else ()):
            execute(conn, f"DROP SCHEMA IF EXISTS {qi(s)} CASCADE")
        conn.commit()

    try:
        drop()
        if args.indexed_copy:
            indexed_copy(conn, args.vocab, vocab)
        if args.phase in ("all", "events", "merge"):
            execute(conn, f"CREATE SCHEMA {qi(schemas.src)}")
            t = time.time()
            write_individuals(conn, schemas.src, n)
            conn.commit()
            create_etl_tables(conn, schemas)
            conn.commit()
            print(f"individuals: {scalar(conn, f'SELECT COUNT(*) FROM {schemas.src}.individuals'):,} rows "
                  f"({time.time() - t:.0f} s)")
        if args.phase in ("all", "events"):
            t = time.time()
            facts = write_events(conn, schemas.src, qi(vocab), n, visits=int(n * 2.7), diagnoses=3 * n,
                                 prescriptions=2 * n)
            execute(conn, f"""
                INSERT INTO {schemas.etl}.source_to_concept_map (source_code, source_vocabulary_id, target_concept_id,
                                                                 target_vocabulary_id)
                SELECT 'DRUG' || lpad(k::text, 4, '0'), 'SRC_DRUG', concept_id, vocabulary_id FROM {schemas.src}.perf_drugs""")
            conn.commit()
            sizes = {t_: scalar(conn, f"SELECT COUNT(*) FROM {schemas.src}.{t_}")
                     for t_ in ("clinic_visits", "diagnoses", "prescriptions")}
            print(f"sources written in {time.time() - t:.0f} s: {sizes}, {facts}")
            build(conn, schemas, cfg, ws, events_spec(), f"events: {n:,} persons")
            for t_ in ("clinic_visits", "diagnoses", "prescriptions", "perf_icd10", "perf_drugs"):
                execute(conn, f"DROP TABLE {schemas.src}.{t_}")
            conn.commit()
        if args.phase in ("all", "merge"):
            t = time.time()
            write_episodes(conn, schemas.src, n)
            conn.commit()
            print(f"residency_episodes: {scalar(conn, f'SELECT COUNT(*) FROM {schemas.src}.residency_episodes'):,} rows "
                  f"({time.time() - t:.0f} s)")
            build(conn, schemas, cfg, ws, merge_spec(), f"merge: {n:,} persons")
            profile(conn, schemas, cfg)
        if args.phase in ("all", "search"):
            search(conn, schemas)
    finally:
        conn.rollback()
        if not args.keep:
            drop()
        conn.close()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
