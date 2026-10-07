"""Observation periods: open ends, the merge of each person's episodes, the source release date,
the fill for persons without a valid episode, and the checks, validator rules, prompts and report
text around them."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from omop_agent.cdm.ddl import create_cdm_tables, create_etl_tables
from omop_agent.config import CdmSourceConfig
from omop_agent.db import execute, fetch_all, scalar
from omop_agent.design.designer import DESIGN_SYSTEM, _normalise_plan, _plan_prompt
from omop_agent.design.render import render_spec_md
from omop_agent.design.validate import validate_spec
from omop_agent.etl.compiler import _compile, compile_spec, release_rank
from omop_agent.etl.runner import _run_step, run_etl
from omop_agent.models import CustomSQL, ETLRunResult, KeySpec, PlannedTable, TableLoadResult, TablePlan
from omop_agent.pipeline import _BuildHistory
from omop_agent.quality.checks import etl_checks, gate, repairable, run_builtin_checks
from omop_agent.quality.repair import (
    REPAIR_SYSTEM,
    RepairState,
    _check_summary,
    build_repair_prompt,
    build_repair_tools,
    observation_period_end_issues,
    validate_mapping,
)
from omop_agent.report.inspection import _etl_section, _release_basis, characterize_cdm
from tests.test_etl import S, count, fixed_ref, load_ref, set_field, step
from tests.test_quality import NOW, _profile, plain_config

OPEN_FROM = date(2099, 1, 1)
DQD_OVERLAP = """SELECT COUNT(DISTINCT a.person_id) FROM {cdm}.observation_period a
    JOIN {cdm}.observation_period b ON a.person_id = b.person_id AND a.observation_period_id <> b.observation_period_id
    WHERE (a.observation_period_start_date <= b.observation_period_end_date
           AND a.observation_period_end_date >= b.observation_period_start_date)
       OR a.observation_period_end_date + 1 = b.observation_period_start_date
       OR b.observation_period_end_date + 1 = a.observation_period_start_date"""


def d(text: str) -> date:
    return date.fromisoformat(text)


def merge_step(release: date | None = None, spec=None, schemas=S):
    cdm_source = CdmSourceConfig(source_release_date=release) if release else None
    return step(compile_spec(spec or load_ref(), schemas, cdm_source), "observation_period_merge")


def from_events(spec=None):
    spec = (spec or load_ref()).model_copy(update={"observation_period": "from_events"}, deep=True)
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "observation_period"]
    return spec


# --------------------------------------------------------------------------------------
# the merge as defined, in Python: the reference the database results are compared with
# --------------------------------------------------------------------------------------


def derived_release(episodes: list[tuple]) -> date | None:
    """The k-th latest exit of the valid closed episodes; with too few, the k-th latest start, else
    the earliest start."""
    if not episodes:
        return None
    exits = sorted((e for _, _, s, e, _ in episodes if e < OPEN_FROM and s <= e), reverse=True)
    if len(exits) >= release_rank(len(exits)):
        return exits[release_rank(len(exits)) - 1]
    starts = sorted((s for _, _, s, _, _ in episodes), reverse=True)
    k = max(5, math.ceil(len(starts) / 10_000))
    return starts[k - 1] if len(starts) >= k else min(starts)


def reference(episodes: list[tuple], deaths: list[tuple], release: date | None = None):
    """Each person's periods [(start, end, type)] and the release date, from episodes
    (id, person, start, end, type) and deaths (person, date)."""
    release = release or derived_release(episodes)
    death: dict[int, date] = {}
    for p, dd in deaths:
        death[p] = min(dd, death.get(p, dd))
    closed = []
    for id_, p, s, e, t in episodes:
        if e >= OPEN_FROM:  # open: at death before the release date, else at the release date (one day after it,
            dd = death.get(p)  # or after death)
            e = s if dd is not None and dd < s else dd if dd is not None and dd < release else max(release, s)
        if s <= e:
            closed.append((p, s, e, id_, t))
    out: dict[int, list[tuple]] = {}
    for p, s, e, _, t in sorted(closed):
        periods = out.setdefault(p, [])
        if periods and s <= periods[-1][1] + timedelta(days=1):  # overlapping or back to back
            ps, pe, pt = periods[-1]
            periods[-1] = (ps, max(pe, e), pt)
        else:
            periods.append((s, e, t))
    return out, release


def loaded_periods(conn, schemas) -> list[dict]:
    return fetch_all(conn, f"""SELECT observation_period_id AS id, person_id AS p, observation_period_start_date AS s,
                                      observation_period_end_date AS e, period_type_concept_id AS t
                               FROM {schemas.cdm}.observation_period ORDER BY observation_period_id""")


def by_person(rows: list[dict]) -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = {}
    for r in rows:
        out.setdefault(r["p"], []).append((r["s"], r["e"], r["t"]))
    return out


@pytest.fixture
def cdm(pg):
    """Empty CDM and ETL tables: episodes go straight into observation_period and the merge step
    runs on them alone."""
    conn, schemas = pg
    create_cdm_tables(conn, schemas, include_vocab=False)
    create_etl_tables(conn, schemas)
    conn.commit()
    return conn, schemas


def load(conn, schemas, episodes: list[tuple], deaths: list[tuple] = ()) -> None:
    execute(conn, f"DELETE FROM {schemas.cdm}.observation_period")
    execute(conn, f"DELETE FROM {schemas.cdm}.death")
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO {schemas.cdm}.observation_period VALUES (%s, %s, %s, %s, %s)", episodes)
        cur.executemany(f"INSERT INTO {schemas.cdm}.death (person_id, death_date, death_type_concept_id) "
                        "VALUES (%s, %s, 32817)", deaths)
    conn.commit()


def run_merge(conn, schemas, release: date | None = None) -> TableLoadResult:
    res = _run_step(conn, merge_step(release, schemas=schemas), None)
    conn.commit()
    return res


# the hand cases of the plan (ed_merge.sql): (id, person, start, end, type)
CASES = [
    (1, 1, d("2015-01-01"), d("2016-06-30"), 32817), (2, 1, d("2016-07-01"), d("2018-12-31"), 32817),
    (3, 1, d("2018-06-01"), d("2020-12-31"), 32817),  # back to back, then an overlap: one period
    (4, 2, d("2015-01-01"), d("2015-12-31"), 32817), (5, 2, d("2016-01-02"), d("2016-12-31"), 32817),  # a day apart
    (9, 4, d("2018-01-01"), d("9999-12-31"), 32817), (10, 4, d("2016-01-01"), d("2017-12-31"), 32880),  # then open
    (11, 5, d("2019-05-01"), d("2019-01-01"), 32817),  # ends before it starts: dropped, no period
    (18, 9, d("2030-01-01"), d("9999-12-31"), 32817),  # open, starts after the release date: one day
    (19, 10, d("2020-01-01"), d("9999-09-09"), 32817),  # a placeholder exit is open too
    (20, 11, d("2016-01-01"), d("9999-12-31"), 32817),  # open, died before the release date
    (21, 12, d("2019-01-01"), d("9999-12-31"), 32817),  # open, starts after the person's death: one day
    (30, 14, d("2010-01-01"), d("9999-12-31"), 32817), (31, 14, d("2014-01-01"), d("2016-03-31"), 32817),  # stale open
    (32, 15, d("2012-01-01"), d("2025-05-05"), 32817),  # a mistyped future exit: kept as written
    (33, 16, d("2012-01-01"), d("2021-06-30"), 32817), (34, 17, d("2013-01-01"), d("2021-08-31"), 32817),
    (35, 18, d("2013-01-01"), d("2021-09-30"), 32817),
]
CASE_DEATHS = [(11, d("2018-03-03")), (12, d("2018-01-01")), (12, d("2018-02-02"))]


# --------------------------------------------------------------------------------------
# compiler (no database)
# --------------------------------------------------------------------------------------


def test_op_end_is_open_when_null():
    spec = load_ref()
    c = _compile(spec.mapping_for("observation_period"), spec)
    assert c.required["observation_period_end_date"] == "open"
    assert ("COALESCE(v.observation_period_end_date, DATE '9999-12-31') AS observation_period_end_date,"
            "  -- NULL = still open (ended by the observation_period_merge step)") in c.insert
    # open ends are not counted by the insert (the merge counts them); unreadable ends are
    assert "omop_derived" not in c.insert and "observation_period_end_date" in c.count_sites
    assert "'observation_period_end_date'" not in c.insert.split("-- 2)")[1].split("-- 1)")[0]


def test_merge_step_sql():
    m = merge_step()
    assert (m.kind, m.target_table, m.returns, m.blocking, m.requires) == ("merge", "observation_period", "row", True, None)
    assert m.source_count_sql is None and m.retained_count_sql is None and m.count_sites == ()
    forget, merge = m.statements
    assert forget == "DELETE FROM etl.build_facts WHERE name = 'source_release_date';"
    # the release date is computed once, from the data only: never from the build date
    assert merge.count("bounds AS (") == 1 and "FROM bounds" in merge and "CROSS JOIN bounds b" in merge
    assert "CURRENT_DATE" not in merge and "now()" not in merge.lower()
    assert "OFFSET (SELECT GREATEST(5, CEIL(COUNT(*) / 10000.0))::int - 1 FROM moved WHERE NOT is_open AND s <= e)" in merge
    assert "observation_period_end_date >= DATE '2099-01-01' AS is_open" in merge
    assert "INSERT INTO etl.build_facts (name, value)" in merge
    # death (a target of the spec) ends open episodes; without death there is no death join
    assert "FROM cdm.death GROUP BY person_id) de" in merge
    assert "WHEN de.death_date < m.s THEN m.s" in merge  # an open episode that starts after death: one day
    # the persons are counted once (a sort of every episode)
    assert merge.count("COUNT(DISTINCT person_id) AS n FROM moved") == 1 and merge.count("FROM persons") == 2
    no_death = load_ref().model_copy(deep=True)
    no_death.table_mappings = [tm for tm in no_death.table_mappings if tm.target_table != "death"]
    merge = merge_step(spec=no_death).statements[1]
    assert "cdm.death" not in merge and "false AS at_death" in merge
    # a configured date replaces the derivation
    merge = merge_step(date(2023, 12, 31)).statements[1]
    assert "SELECT DATE '2023-12-31' AS release_date, true AS configured" in merge and "OFFSET" not in merge
    assert "(configured)" in merge
    # periods derived from the records are never open: a far-future record date is a typo, not an open end
    merge = step(compile_spec(from_events(), S), "observation_period_merge").statements[1]
    assert "false AS is_open" in merge and "DATE '2099-01-01'" not in merge and "cdm.death" not in merge
    # a spec writer of observation_period keeps them open, also with from_events (the validator refuses that)
    mixed = load_ref().model_copy(update={"observation_period": "from_events"}, deep=True)
    assert "observation_period_end_date >= DATE '2099-01-01' AS is_open" in step(
        compile_spec(mixed, S), "observation_period_merge").statements[1]


def test_merge_placement():
    # custom-only death and specimen loads run before the merge, which follows the last OP writer
    spec = load_ref().model_copy(deep=True)
    spec.table_mappings = [tm for tm in spec.table_mappings if tm.target_table != "death"]
    spec.custom_sql = [CustomSQL(name="deaths", target_table="death", sql="SELECT 1"),
                       CustomSQL(name="specimens", target_table="specimen", sql="SELECT 2"),
                       CustomSQL(name="more periods", target_table="observation_period", sql="SELECT 3")]
    steps = compile_spec(spec, S)
    labels = [s.label for s in steps]
    assert labels[labels.index("custom_deaths"):] == [
        "custom_deaths", "custom_specimens", "observation_period", "custom_more_periods", "observation_period_merge",
        "observation_period_fill", "cdm_source"]
    assert "cdm.death" in step(steps, "observation_period_merge").sql
    # from_events: the derived periods, then the merge (it renumbers them and derives the release
    # date); no fill, every person with a record has a period already
    labels = [s.label for s in compile_spec(from_events(), S)]
    assert labels[-3:] == ["observation_period", "observation_period_merge", "cdm_source"]
    # the fill reads every loaded event table, only for persons without a period, and returns its row
    fill = step(compile_spec(load_ref(), S), "observation_period_fill")
    assert (fill.kind, fill.returns, fill.blocking, fill.requires) == ("derived", "row", True, None)
    assert "AND NOT EXISTS (SELECT 1 FROM cdm.observation_period op WHERE op.person_id = ev.person_id)" in fill.sql
    assert "SELECT person_id, measurement_date AS event_date FROM cdm.measurement" in fill.sql
    assert "AS persons_without_period;" in fill.sql and "LEAST" not in fill.sql  # not clipped
    assert "NOT EXISTS" not in step(compile_spec(from_events(), S), "observation_period").sql


def test_cdm_source_release_date():
    cs = compile_spec(load_ref(), S)[-1]
    assert ("COALESCE((SELECT MAX(value::date) FROM etl.build_facts WHERE name = 'source_release_date'),"
            " CURRENT_DATE),") in cs.sql
    configured = compile_spec(load_ref(), S, CdmSourceConfig(source_release_date=date(2023, 12, 31)))[-1]
    assert "    DATE '2023-12-31',\n    CURRENT_DATE," in configured.sql and "build_facts" not in configured.sql


def test_release_rank():
    assert [release_rank(n) for n in (0, 4, 5, 50_000, 50_001, 1_000_000)] == [5, 5, 5, 5, 6, 100]


# --------------------------------------------------------------------------------------
# the merge on PostgreSQL
# --------------------------------------------------------------------------------------


@pytest.mark.postgres
def test_merge_edge_cases(cdm):
    conn, schemas = cdm
    load(conn, schemas, CASES, CASE_DEATHS)
    res = run_merge(conn, schemas)
    assert res.error is None, res.error
    st = res.stats
    # the 5th latest valid closed exit: the 2025-05-05 typo does not move it
    assert {k: st[k] for k in ("release_date", "configured", "closed_episodes", "fact_rows")} == {
        "release_date": "2020-12-31", "configured": False, "closed_episodes": 11, "fact_rows": 1}
    assert {k: st[k] for k in ("episodes", "dropped_inverted", "open_episodes", "open_closed_at_death",
                               "open_starting_after_death", "open_starting_after_release", "exits_after_release",
                               "ending_from_release", "persons_overlapping", "persons_stale_open", "persons",
                               "persons_without_period", "inserted")} == {
        "episodes": 18, "dropped_inverted": 1, "open_episodes": 6, "open_closed_at_death": 1,
        "open_starting_after_death": 1, "open_starting_after_release": 1, "exits_after_release": 4,
        "ending_from_release": 9, "persons_overlapping": 2, "persons_stale_open": 1, "persons": 13,
        "persons_without_period": 1, "inserted": 13}
    assert res.rows_inserted == 13 and res.kind == "merge"
    periods = by_person(loaded_periods(conn, schemas))
    expected, release = reference(CASES, CASE_DEATHS)
    assert release == d("2020-12-31") and periods == expected
    assert periods[1] == [(d("2015-01-01"), d("2020-12-31"), 32817)]
    assert periods[2] == [(d("2015-01-01"), d("2015-12-31"), 32817), (d("2016-01-02"), d("2016-12-31"), 32817)]
    assert periods[4] == [(d("2016-01-01"), d("2020-12-31"), 32880)]  # the type of the first episode
    assert 5 not in periods  # left for the fill
    assert periods[9] == [(d("2030-01-01"), d("2030-01-01"), 32817)]
    assert periods[11] == [(d("2016-01-01"), d("2018-03-03"), 32817)]
    assert periods[12] == [(d("2019-01-01"), d("2019-01-01"), 32817)]  # not years after the death
    assert periods[15] == [(d("2012-01-01"), d("2025-05-05"), 32817)]
    # ids 1..N in person and start order; the DQD overlap predicate finds nobody; one fact
    rows = loaded_periods(conn, schemas)
    assert [r["id"] for r in rows] == list(range(1, 14)) and rows == sorted(rows, key=lambda r: (r["p"], r["s"]))
    assert scalar(conn, DQD_OVERLAP.format(cdm=schemas.cdm)) == 0
    assert fetch_all(conn, f"SELECT name, value FROM {schemas.etl}.build_facts") == [
        {"name": "source_release_date", "value": "2020-12-31"}]
    # run again on its own output: nothing changes, the fact is replaced
    again = run_merge(conn, schemas)
    assert again.stats["dropped_inverted"] == 0 and by_person(loaded_periods(conn, schemas)) == expected
    assert scalar(conn, f"SELECT COUNT(*) FROM {schemas.etl}.build_facts") == 1


@pytest.mark.postgres
def test_configured_release_date_does_not_clip_or_drop(cdm):
    conn, schemas = cdm
    load(conn, schemas, CASES, CASE_DEATHS)
    res = run_merge(conn, schemas, date(2019, 6, 30))
    st = res.stats
    assert (st["release_date"], st["configured"]) == ("2019-06-30", True)
    periods = by_person(loaded_periods(conn, schemas))
    assert periods == reference(CASES, CASE_DEATHS, date(2019, 6, 30))[0]
    # real exits after the date are kept as written, open episodes that start after it keep one day
    assert periods[1] == [(d("2015-01-01"), d("2020-12-31"), 32817)]
    assert periods[15] == [(d("2012-01-01"), d("2025-05-05"), 32817)]
    assert periods[10] == [(d("2020-01-01"), d("2020-01-01"), 32817)]
    assert periods[12] == [(d("2019-01-01"), d("2019-01-01"), 32817)]
    assert periods[11] == [(d("2016-01-01"), d("2018-03-03"), 32817)]
    # nothing is dropped for the release date: only the inverted episode
    assert st["dropped_inverted"] == 1 and sum(len(v) for v in periods.values()) == st["inserted"] == 13
    assert (st["open_starting_after_release"], st["exits_after_release"]) == (2, 5)
    # the configured date is stale: reported, not repairable
    checks = {r.key: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[res]))}
    late = checks["etlSourceReleaseDate.observation_period.observation_period_merge"]
    assert (late.failed, late.num_violated_rows, late.num_denominator_rows, late.repairable) == (True, 7, 18, False)
    assert "stale" in late.description and "2019-06-30" in late.description


@pytest.mark.postgres
def test_release_date_robust(cdm):
    conn, schemas = cdm
    exits = [d("2020-12-31") - timedelta(days=i) for i in range(10)]
    episodes = [(i + 1, i + 1, d("2015-01-01"), e, 32817) for i, e in enumerate(exits)]
    episodes += [(50, 50, d("2015-01-01"), d("2030-05-05"), 32817),  # a mistyped exit, far in the future
                 (51, 51, d("2031-01-01"), d("2030-01-01"), 32817),  # junk, inverted, dated in the future
                 (52, 52, d("2016-01-01"), d("9999-12-31"), 32817)]  # still open
    load(conn, schemas, episodes)
    res = run_merge(conn, schemas)
    # the 5th latest valid exit (the typo is one of the five): the junk row does not move it
    assert res.stats["release_date"] == "2020-12-28" == str(exits[3])
    assert by_person(loaded_periods(conn, schemas))[52] == [(d("2016-01-01"), d("2020-12-28"), 32817)]
    assert by_person(loaded_periods(conn, schemas))[50] == [(d("2015-01-01"), d("2030-05-05"), 32817)]
    # it does not depend on the day of the build: the statement reads no clock
    assert "CURRENT_DATE" not in merge_step().sql and "now(" not in merge_step().sql.lower()
    # an open episode ended on a derived date is reported: set the configured date
    checks = {r.check_name: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[res]))}
    assert checks["etlSourceReleaseDate"].failed and checks["etlSourceReleaseDate"].num_violated_rows == 1
    assert "derived: the 5th latest exit date" in checks["etlSourceReleaseDate"].description
    # with fewer than five closed episodes, the 5th latest start; with fewer than five episodes, the earliest
    few = [(1, 1, d("2010-01-01"), d("2011-01-01"), 32817)] + [
        (i, i, d("2012-01-01") + timedelta(days=i), d("9999-12-31"), 32817) for i in range(2, 7)]
    load(conn, schemas, few)
    assert run_merge(conn, schemas).stats["release_date"] == str(derived_release(few)) == "2012-01-03"
    load(conn, schemas, few[:3])
    assert run_merge(conn, schemas).stats["release_date"] == "2010-01-01"
    load(conn, schemas, [])
    empty = run_merge(conn, schemas)
    assert (empty.stats["release_date"], empty.stats["fact_rows"], empty.rows_inserted) == (None, 0, 0)


@pytest.mark.postgres
def test_from_events_periods_are_never_open(cdm):
    conn, schemas = cdm
    periods = [(i, i, d("2015-01-01"), d("2020-12-31") - timedelta(days=i), 32817) for i in range(1, 11)]
    periods.append((11, 11, d("2016-01-01"), d("2109-03-01"), 32817))  # a record dated 2109 (2019 mistyped)
    load(conn, schemas, periods)
    res = _run_step(conn, merge_step(spec=from_events(), schemas=schemas), None)
    conn.commit()
    assert (res.stats["open_episodes"], res.stats["release_date"]) == (0, "2020-12-27")
    # kept as derived (plausibleValueHigh reports the typo), not cut to the release date
    assert by_person(loaded_periods(conn, schemas))[11] == [(d("2016-01-01"), d("2109-03-01"), 32817)]
    checks = {r.check_name: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[res]))}
    assert not checks["etlSourceReleaseDate"].failed and not checks["etlObservationPeriodMerge"].failed


@pytest.mark.postgres
def test_merge_property(cdm):
    """2,000 random persons with 1-6 episodes: overlaps, adjacency, gaps, inverted rows, open and
    placeholder ends, deaths. The periods are exactly those of the reference merge, never overlap or
    touch (DQD), and are numbered 1..N."""
    conn, schemas = cdm
    rng = random.Random(20261007)
    episodes, deaths, n = [], [], 0
    for p in range(1, 2001):
        start = d("2010-01-01") + timedelta(days=rng.randrange(4000))
        for _ in range(rng.randint(1, 6)):
            n += 1
            u = rng.random()
            if u < 0.04:
                end = start - timedelta(days=rng.randint(1, 20))  # inverted
            elif u < 0.12:
                end = d("9999-12-31") if rng.random() < 0.8 else d("2999-01-01")  # open, or a placeholder
            else:
                end = start + timedelta(days=rng.randrange(400))
            episodes.append((n, p, start, end, rng.choice([32817, 32880])))
            # the next episode starts up to 30 days before the end of this one (an overlap), the day after
            # it (back to back), or a few days later (a gap)
            length = (end - start).days if start <= end < OPEN_FROM else rng.randrange(400)
            start += timedelta(days=length + rng.randint(-30, 5))
        if rng.random() < 0.05:
            deaths.append((p, d("2010-01-01") + timedelta(days=rng.randrange(5000))))
    rng.shuffle(episodes)
    load(conn, schemas, episodes, deaths)
    res = run_merge(conn, schemas)
    assert res.error is None
    expected, release = reference(episodes, deaths)
    assert res.stats["release_date"] == str(release)
    rows = loaded_periods(conn, schemas)
    assert by_person(rows) == expected
    assert [r["id"] for r in rows] == list(range(1, len(rows) + 1))
    assert scalar(conn, DQD_OVERLAP.format(cdm=schemas.cdm)) == 0
    assert res.stats["episodes"] == len(episodes)
    assert res.stats["dropped_inverted"] == sum(1 for _, _, s, e, _ in episodes if e < OPEN_FROM and s > e)
    assert res.stats["persons_without_period"] == len({p for _, p, _, _, _ in episodes} - set(expected))
    # covered person-days are those of the valid episodes, closed (no day lost or added)
    days = {(p, s + timedelta(days=i)) for p, ps in expected.items() for s, e, _ in ps for i in range((e - s).days + 1)}
    assert len(days) == scalar(conn, f"""SELECT SUM(observation_period_end_date - observation_period_start_date + 1)
                                         FROM {schemas.cdm}.observation_period""")


# --------------------------------------------------------------------------------------
# builds of the demo data
# --------------------------------------------------------------------------------------


def residency(conn, src: str) -> list[tuple]:
    return [tuple(r.values()) for r in fetch_all(
        conn, f"SELECT individual_id, start_date, start_event, end_date, end_event FROM {src}.residency_episodes "
              "ORDER BY individual_id, start_date")]


def set_residency(conn, src: str, rows: list[tuple]) -> None:
    execute(conn, f"DELETE FROM {src}.residency_episodes")
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO {src}.residency_episodes VALUES (%s, %s, %s, %s, %s)", rows)
    conn.commit()


def multi_episodes(rows: list[tuple]) -> tuple[list[tuple], dict[str, int]]:
    """The demo's one episode per person as an HDSS has them (the obsperiod design's e6): internal
    moves (back-to-back episodes), overlapping episodes, out-migrations with a 60-day gap, still
    resident (no exit date), and junk rows that end before they start."""
    out, kinds = [], {}
    for i, (pid, s, se, e, ee) in enumerate(rows):
        s_, e_ = d(s), d(e)
        mid = s_ + (e_ - s_) / 2
        k = i % 10
        if k == 0 and (e_ - s_).days > 10:
            out += [(pid, s, se, mid.isoformat(), "EXT"), (pid, (mid + timedelta(1)).isoformat(), "ENT", e, ee)]
        elif k == 1 and (e_ - s_).days > 100:
            out += [(pid, s, se, (mid + timedelta(30)).isoformat(), "EXT"),
                    (pid, (mid - timedelta(30)).isoformat(), "ENT", e, ee)]
        elif k == 2 and (e_ - s_).days > 100:
            out += [(pid, s, se, (mid - timedelta(30)).isoformat(), "OMG"),
                    (pid, (mid + timedelta(30)).isoformat(), "IMG", e, ee)]
        elif k == 3 and ee == "OBE":
            out += [(pid, s, se, None, None)]
        elif k == 4:
            out += [(pid, s, se, e, ee), (pid, e, "ENT", s, "EXT")]
        else:
            out += [(pid, s, se, e, ee)]
            k = -1
        kinds[k] = kinds.get(k, 0) + 1
    return out, kinds


def periods_by_key(conn, schemas) -> dict[str, list[tuple]]:
    rows = fetch_all(conn, f"""SELECT pm.source_value AS p, op.observation_period_start_date AS s,
                                      op.observation_period_end_date AS e, op.period_type_concept_id AS t
                               FROM {schemas.cdm}.observation_period op
                               JOIN {schemas.etl}.person_map pm ON pm.person_id = op.person_id
                               ORDER BY op.observation_period_id""")
    return by_person(rows)


@pytest.mark.postgres
def test_multi_episode_residency_merge(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    rows, kinds = multi_episodes(residency(conn, src))
    assert set(kinds) == {-1, 0, 1, 2, 3, 4}
    set_residency(conn, src, rows)
    spec = fixed_ref()
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok, [(e.step, e.error) for e in result.errors()]
    by = {s.step.split("_", 1)[1]: s for s in result.steps}
    episodes = [(i, p, d(s), d(e) if e else d("9999-12-31"), 32817) for i, (p, s, _, e, _) in enumerate(rows)]
    deaths = [(r["individual_id"], d(r["dod"])) for r in fetch_all(conn, f"SELECT individual_id, dod FROM {src}.deaths")]
    expected, release = reference(episodes, deaths)
    st = by["observation_period_merge"].stats
    valid = sum(1 for _, _, s, e, _ in episodes if s <= e)
    assert (by["observation_period"].rows_inserted, st["episodes"]) == (len(rows), len(rows))
    assert (st["dropped_inverted"], st["open_episodes"]) == (len(rows) - valid, kinds[3])
    assert st["release_date"] == str(release) == "2023-12-31" and st["configured"] is False
    assert st["inserted"] == sum(len(v) for v in expected.values()) < valid
    assert st["persons_overlapping"] > 0 and st["persons_without_period"] == 0
    # exactly the reference periods: the covered person-days of the valid episodes, nothing more
    assert periods_by_key(conn, schemas) == expected
    assert scalar(conn, DQD_OVERLAP.format(cdm=schemas.cdm)) == 0
    assert by["observation_period_fill"].stats == {"inserted": 0, "persons_without_period": 0}
    assert scalar(conn, f"SELECT source_release_date FROM {schemas.cdm}.cdm_source") == release
    # the checks: the gate passes; the dropped junk rows are under 5%; the derived date is reported
    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    keys = [r.key for r in report.results]
    assert len(keys) == len(set(keys))  # the merge step has its own keys
    by_key = {r.key: r for r in report.results}
    assert report.passed, report.blocking_failures
    assert not by_key["measureObservationPeriodOverlap.observation_period"].failed
    merged = by_key["etlObservationPeriodMerge.observation_period.observation_period_merge"]
    assert (merged.num_violated_rows, merged.num_denominator_rows, merged.failed) == (
        len(rows) - valid, len(rows), (len(rows) - valid) * 100 > 5 * len(rows))
    assert f"Open episodes (no exit date): {kinds[3]} of {len(rows)}" in merged.description
    late = by_key["etlSourceReleaseDate.observation_period.observation_period_merge"]
    assert late.failed and not repairable(late, {"fatal", "convention"})
    assert late.num_violated_rows == kinds[3] - st["open_closed_at_death"] - st["open_starting_after_death"]
    assert "etlRecordRetention.observation_period.observation_period_merge" not in by_key
    # the build history counts the valid episodes the merge took (and the fill's periods), not the episodes and
    # the periods: a repair that fixes an episode may join two periods
    assert _BuildHistory._rows(result)["observation_period"] == valid == st["episodes"] - st["dropped_inverted"]


@pytest.mark.postgres
def test_null_start_dropped_not_aborted(staged):
    conn, schemas, cfg = staged
    src = schemas.src
    rows = residency(conn, src)
    # four episodes without a readable start, three still open
    rows = ([(p, None, se, e, ee) for p, _, se, e, ee in rows[:3]] + [(rows[3][0], "garbage", *rows[3][2:])]
            + [(p, s, se, None, None) for p, s, se, _, _ in rows[4:7]] + rows[7:])
    set_residency(conn, src, rows)
    result = run_etl(conn, schemas, fixed_ref(), cfg, None)
    assert result.ok
    by = {s.step.split("_", 1)[1]: s for s in result.steps}
    op = by["observation_period"]
    assert op.error is None and op.rows_inserted == len(rows) - 4
    assert op.value_counts == {"required_null": {"observation_period_start_date": 4}, "row": {"dropped": 4},
                               "invalid": {"observation_period_start_date": 1}}
    assert by["observation_period_merge"].stats["open_episodes"] == 3
    # the four persons get a period from their records (those with records)
    assert by["observation_period_fill"].stats["persons_without_period"] == 4
    assert count(conn, schemas, "observation_period", "observation_period_end_date >= DATE '2099-01-01'") == 0


@pytest.mark.postgres
def test_fill_for_persons_without_episodes(staged):
    conn, schemas, cfg = staged
    src, cdm = schemas.src, schemas.cdm
    a, b = [r["individual_id"] for r in fetch_all(
        conn, f"SELECT individual_id FROM {src}.clinic_visits GROUP BY 1 ORDER BY 1 LIMIT 2")]
    execute(conn, f"DELETE FROM {src}.residency_episodes WHERE individual_id = %s", (a,))  # clinic records only
    execute(conn, f"UPDATE {src}.residency_episodes SET start_date = end_date, end_date = start_date "
                  "WHERE individual_id = %s", (b,))  # only an episode that ends before it starts
    execute(conn, f"INSERT INTO {src}.individuals (individual_id, sex, dob) VALUES ('KH999999', '1', '01/01/1980')")
    conn.commit()
    spec = fixed_ref()
    result = run_etl(conn, schemas, spec, cfg)
    assert result.ok
    by = {s.step.split("_", 1)[1]: s for s in result.steps}
    assert by["observation_period_merge"].stats["persons_without_period"] == 1  # b
    assert by["observation_period_fill"].stats == {"inserted": 2, "persons_without_period": 3}
    assert by["observation_period_fill"].rows_inserted == 2
    periods = periods_by_key(conn, schemas)
    for key in (a, b):  # from the first to the last record, not clipped
        span = fetch_all(conn, f"""SELECT MIN(x) AS s, MAX(x) AS e FROM (
            SELECT visit_start_date AS x, person_id FROM {cdm}.visit_occurrence
            UNION ALL SELECT visit_end_date, person_id FROM {cdm}.visit_occurrence
            UNION ALL SELECT condition_start_date, person_id FROM {cdm}.condition_occurrence
            UNION ALL SELECT condition_end_date, person_id FROM {cdm}.condition_occurrence
            UNION ALL SELECT drug_exposure_start_date, person_id FROM {cdm}.drug_exposure
            UNION ALL SELECT drug_exposure_end_date, person_id FROM {cdm}.drug_exposure
            UNION ALL SELECT measurement_date, person_id FROM {cdm}.measurement
            UNION ALL SELECT death_date, person_id FROM {cdm}.death) ev
            JOIN {schemas.etl}.person_map pm ON pm.person_id = ev.person_id WHERE pm.source_value = %s""", (key,))[0]
        assert periods[key] == [(span["s"], span["e"], 32817)]
    assert "KH999999" not in periods
    ids = [r["observation_period_id"] for r in fetch_all(
        conn, f"SELECT observation_period_id FROM {cdm}.observation_period ORDER BY 1")]
    assert ids == list(range(1, len(ids) + 1))
    # the person without any record still has no period: the blocking person completeness fails and says why
    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    by_key = {r.key: r for r in report.results}
    pc = by_key["measurePersonCompleteness.observation_period"]
    assert pc.failed and pc.num_violated_rows == 1 and "no valid episode and no record" in pc.description
    # no spec change can give them a period: it blocks, but the repair agent is not sent after it
    assert pc.repairable is False and not repairable(pc, {"fatal", "convention"})
    assert not report.passed and any("measurePersonCompleteness.observation_period" in b for b in report.blocking_failures)
    merged = by_key["etlObservationPeriodMerge.observation_period.observation_period_merge"]
    assert merged.num_violated_rows == 1 and not merged.failed
    assert "Persons without a valid episode: 1 of" in merged.description
    assert "one from their first to their last event: 2 of 3 persons" in merged.description
    assert by_key["etlSourceReleaseDate.observation_period.observation_period_merge"].num_violated_rows == 0
    # the repair prompt withholds the small counts of the description
    _, text = _check_summary(merged, 5)
    assert "2 of 3 persons" not in text and "<5 of <5 persons" in text


@pytest.mark.postgres
def test_within_observation_period_check(staged):
    conn, schemas, cfg = staged
    src, cdm = schemas.src, schemas.cdm
    # the residency of the first 40 people ends 30 days after it starts: their later records fall outside
    execute(conn, f"""UPDATE {src}.residency_episodes SET end_date = (start_date::date + 30)::text
                      WHERE individual_id IN (SELECT individual_id FROM {src}.residency_episodes ORDER BY 1 LIMIT 40)""")
    conn.commit()
    spec = fixed_ref()
    result = run_etl(conn, schemas, spec, cfg)
    report = run_builtin_checks(conn, schemas, cfg, spec, result)
    within = {r.key: r for r in report.results if r.check_name == "withinObservationPeriod"}
    assert set(within) == {f"withinObservationPeriod.{t}.{f}" for t, f in [
        ("visit_occurrence", "visit_start_date"), ("condition_occurrence", "condition_start_date"),
        ("drug_exposure", "drug_exposure_start_date"), ("measurement", "measurement_date"),
        ("death", "death_date")]}
    for key, r in within.items():
        _, table, fld = key.split(".")
        outside = scalar(conn, f"""SELECT COUNT(*) FROM {cdm}.{table} t WHERE NOT EXISTS (
            SELECT 1 FROM {cdm}.observation_period op WHERE op.person_id = t.person_id
              AND t.{fld} >= op.observation_period_start_date AND t.{fld} <= op.observation_period_end_date)""")
        assert (r.num_violated_rows, r.num_denominator_rows) == (outside, count(conn, schemas, table)), key
        assert r.severity == "characterization" and r.check_level == "FIELD" and r.error is None
    assert within["withinObservationPeriod.visit_occurrence.visit_start_date"].num_violated_rows > 0
    assert report.passed  # characterization: reported, never blocking


@pytest.mark.postgres
def test_repair_may_not_open_or_close_episodes(staged):
    """The repair agent may not change which episodes are open to make a check pass: making the
    exits of episodes that end before they start NULL opens them (they would run to the release
    date), and a fallback onto the start makes open episodes one day long. Both are refused, by the
    static rule and by the dry run (which also sees forms the rule does not know); a placeholder exit
    date may become NULL."""
    conn, schemas, cfg = staged
    src = schemas.src
    rows = [(p, e, se, s, ee) if i % 10 == 0 else (p, s, se, None, None) if i % 10 == 1
            else (p, s, se, "1900-01-01", ee) if i % 10 == 2 else (p, s, se, e, ee)
            for i, (p, s, se, e, ee) in enumerate(residency(conn, src))]  # swapped, open, a placeholder exit
    set_residency(conn, src, rows)
    spec = fixed_ref()
    assert run_etl(conn, schemas, spec, cfg).ok  # the build before the repair fills the id maps
    old = spec.mapping_for("observation_period")

    def edit(expression: str, kind: str = "expression") -> tuple[str | None, list[str]]:
        new = set_field(spec, "observation_period", "observation_period_end_date", kind=kind,
                        expression=expression).mapping_for("observation_period")
        return validate_mapping(new, spec, conn, schemas, cfg, old=old), observation_period_end_issues(old, new)

    guard, static = edit("CASE WHEN r.end_date::date >= r.start_date::date THEN r.end_date::date END")
    assert guard is not None and guard.startswith("it would turn ")
    assert "closed episodes of observation_period into open ones that run to the release date" in guard
    assert static and "turns episodes that end before they start into open ones" in static[0]
    guard, static = edit("NULLIF(r.end_date::date, r.end_date::date)")  # a form the static rule does not know
    assert guard is not None and guard.startswith("it would turn ") and static == []
    for expression in ("GREATEST(r.start_date::date, r.end_date::date)",
                       "COALESCE(r.end_date::date, r.start_date::date)"):
        guard, static = edit(expression)
        assert guard is not None and guard.startswith("it would close ") and "on their start date" in guard
        assert static and "falls back to the start date" in static[0]
    # a placeholder exit date (one value) may become NULL; an end left as it was is not judged again
    assert edit("NULLIF(r.end_date, '1900-01-01')::date") == (None, [])
    assert edit("r.end_date::date") == (None, [])
    # through the agent's tool: refused, and the spec is unchanged
    state = RepairState(spec=spec, conn=conn, schemas=schemas, cfg=cfg, profile=_profile(), iteration=1, actions=[],
                        trigger="etlObservationPeriodMerge.observation_period.observation_period_merge")
    update = {t.name: t for t in build_repair_tools(state)}["update_field_mapping"].invoke
    out = update({"target_table": "observation_period", "target_field": "observation_period_end_date",
                  "kind": "expression", "expression": "GREATEST(r.start_date::date, r.end_date::date)",
                  "rationale": "episodes that end before they start"})
    assert out.startswith("REJECTED: observation_period.observation_period_end_date: an end that falls back")
    assert "may not change how long people are observed" in out and state.spec == spec and state.actions == []


@pytest.mark.postgres
def test_characterization_shows_the_release_date(staged):
    conn, schemas, cfg = staged
    rows, _ = multi_episodes(residency(conn, schemas.src))
    set_residency(conn, schemas.src, rows)
    result = run_etl(conn, schemas, fixed_ref(), cfg)
    assert result.ok and _release_basis(result) == "derived"
    assert _release_basis(None) is None
    configured = result.model_copy(deep=True)
    next(s for s in configured.steps if s.kind == "merge").stats["configured"] = True
    assert _release_basis(configured) == "configured"
    cdm = schemas.cdm
    multi = scalar(conn, f"""SELECT COUNT(*) FROM (SELECT person_id FROM {cdm}.observation_period GROUP BY person_id
                                                    HAVING COUNT(*) > 1) m""")
    ending = scalar(conn, f"""SELECT COUNT(*) FROM {cdm}.observation_period
                              WHERE observation_period_end_date = (SELECT source_release_date FROM {cdm}.cdm_source)""")
    assert multi >= 5 and ending >= 5  # the out-migrations with a gap; the open episodes
    obs = characterize_cdm(conn, schemas, 5, _release_basis(result))["observation_period"]
    assert (obs["multi_period_persons"], obs["ending_at_release"]) == (f"{multi:,}", f"{ending:,}")
    assert (obs["release_date"], obs["release_basis"]) == ("2023-12-31", "derived")
    # small counts are withheld
    obs = characterize_cdm(conn, schemas, 10**6)["observation_period"]
    assert (obs["multi_period_persons"], obs["ending_at_release"], obs["release_basis"]) == (
        "<1000000", "<1000000", None)


# --------------------------------------------------------------------------------------
# checks, history, report, prompts and the validator (no database)
# --------------------------------------------------------------------------------------


def merge_result(**stats) -> TableLoadResult:
    base = {"inserted": 90, "episodes": 100, "dropped_inverted": 2, "open_episodes": 10, "open_closed_at_death": 1,
            "open_starting_after_release": 0, "exits_after_release": 4, "persons_overlapping": 3,
            "persons_stale_open": 1, "persons": 80, "persons_without_period": 1, "release_date": "2023-12-31",
            "configured": False, "closed_episodes": 88, "fact_rows": 1}
    base.update(stats)
    return TableLoadResult(target_table="observation_period", step="12_observation_period_merge", kind="merge",
                           rows_inserted=base["inserted"], stats=base)


def test_merge_checks(tmp_path):
    fill = TableLoadResult(target_table="observation_period", step="13_observation_period_fill", kind="derived",
                           rows_inserted=7, stats={"inserted": 7, "persons_without_period": 9})
    checks = {r.key: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[merge_result(), fill]))}
    merged = checks["etlObservationPeriodMerge.observation_period.observation_period_merge"]
    assert (merged.severity, merged.threshold_pct, merged.failed, merged.num_violated_rows,
            merged.num_denominator_rows, merged.step) == ("convention", 5.0, False, 2, 100, "observation_period_merge")
    assert repairable(merged.model_copy(update={"failed": True}), {"convention"})
    text = merged.description
    for part in ("2 of 100 episodes", "Open episodes (no exit date): 10 of 100", "2023-12-31 (derived: the 5th latest exit",
                 "or at death (1 of 10)", "overlapping episodes: 3 of 80", "a later one: 1 of 80",
                 "Persons without a valid episode: 1 of 80", "7 of 9 persons"):
        assert part in text, part
    assert "etlRecordLoss.observation_period.observation_period_merge" not in checks
    late = checks["etlSourceReleaseDate.observation_period.observation_period_merge"]
    # the open episodes ended at death do not depend on the derived date
    assert (late.failed, late.num_violated_rows, late.severity, late.repairable) == (True, 9, "convention", False)
    assert "Set cdm_source.source_release_date" in late.description
    # exit dates the mapping could not read were loaded as open: counted as episodes the merge could not use
    op = TableLoadResult(target_table="observation_period", step="11_observation_period", kind="insert",
                         rows_inserted=100, value_counts={"invalid": {"observation_period_end_date": 4}})
    checks = {r.check_name: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[op, merge_result(), fill]))}
    merged = checks["etlObservationPeriodMerge"]
    assert (merged.num_violated_rows, merged.failed) == (6, True)
    assert "treated as still open because their exit date could not be read (etlInvalidValue)" in merged.description
    assert "4 of 100 episodes" in merged.description and "2 of 100 episodes" in merged.description
    # a derived date after the day of the build (placeholder or mistyped exits): fatal, for a person to fix
    future = {r.check_name: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[merge_result(
        release_date="2050-12-31", open_episodes=0, open_closed_at_death=0, ending_from_release=40)]))}
    late = future["etlSourceReleaseDate"]
    assert (late.severity, late.failed, late.num_violated_rows, late.repairable) == ("fatal", True, 40, False)
    assert "after the day of the build (2024-01-01)" in late.description and "placeholders" in late.description
    passed, blocking = gate(list(future.values()), plain_config(tmp_path))
    assert not passed and blocking[0].startswith("etlSourceReleaseDate")
    # most episodes end before they start: the dates are read wrongly, fatal
    bad = merge_result(dropped_inverted=60, inserted=30)
    checks = {r.key: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[bad]), min_retention_pct=50)}
    loss = checks["etlRecordLoss.observation_period.observation_period_merge"]
    assert (loss.severity, loss.failed, loss.threshold_pct, loss.num_violated_rows) == ("fatal", True, 50.0, 60)
    assert checks["etlObservationPeriodMerge.observation_period.observation_period_merge"].failed
    passed, blocking = gate(list(checks.values()), plain_config(tmp_path))
    assert not passed and any(b.startswith("etlRecordLoss.observation_period.observation_period_merge") for b in blocking)
    # configured and nobody after it: passes; no open episodes and derived: passes
    ok = {r.check_name: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[merge_result(
        configured=True, exits_after_release=0)]))}
    assert not ok["etlSourceReleaseDate"].failed and "(configured" in ok["etlObservationPeriodMerge"].description
    ok = {r.check_name: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[merge_result(open_episodes=0)]))}
    assert not ok["etlSourceReleaseDate"].failed
    # no episodes at all (from_events without records)
    none = {r.check_name: r for r in etl_checks(ETLRunResult(started_at=NOW, steps=[merge_result(
        episodes=0, inserted=0, dropped_inverted=0, open_episodes=0, release_date=None, closed_episodes=0)]))}
    assert not any(r.failed for r in none.values()) and "no observation period episodes" in none[
        "etlSourceReleaseDate"].description
    # a merge that failed is a fatal step error, never sent to the repair agent
    failed = merge_result().model_copy(update={"error": "boom", "stats": {}})
    (err,) = etl_checks(ETLRunResult(started_at=NOW, steps=[failed]))
    assert (err.check_name, err.severity, err.repairable) == ("etlStepError", "fatal", False)


def test_build_history_counts_the_final_periods():
    steps = [
        TableLoadResult(target_table="person", step="03_person", kind="insert", rows_inserted=80),
        TableLoadResult(target_table="observation_period", step="11_observation_period", kind="insert",
                        rows_inserted=100),
        merge_result(),
        TableLoadResult(target_table="observation_period", step="13_observation_period_fill", kind="derived",
                        rows_inserted=7, stats={"inserted": 7, "persons_without_period": 9}),
        TableLoadResult(target_table="cdm_source", step="14_cdm_source", kind="metadata", rows_inserted=1),
    ]
    run = ETLRunResult(started_at=NOW, steps=steps)
    # the valid episodes the merge took (100 - 2 dropped) and the fill's periods
    assert _BuildHistory._rows(run) == {"person": 80, "observation_period": 105, "cdm_source": 1}
    assert run.rows_loaded() == 80 + 100 + 7 + 1 and run.rows_derived() == 7  # the merge replaces counted rows
    # from_events: the derived periods are what the merge took
    ev = [TableLoadResult(target_table="observation_period", step="11_observation_period", kind="derived",
                          rows_inserted=80), merge_result(inserted=80, episodes=80, dropped_inverted=0)]
    assert _BuildHistory._rows(ETLRunResult(started_at=NOW, steps=ev)) == {"observation_period": 80}
    # without a merge (older results), every step counts
    old = [TableLoadResult(target_table="observation_period", step="11_observation_period", kind="derived",
                           rows_inserted=80)]
    assert _BuildHistory._rows(ETLRunResult(started_at=NOW, steps=old)) == {"observation_period": 80}
    # a repair that turns most episodes into junk loses periods
    history = _BuildHistory()
    history.add(run, None)
    worse = ETLRunResult(started_at=NOW, steps=[steps[0], steps[1], merge_result(inserted=40, dropped_inverted=60),
                                                steps[3], steps[4]])
    assert history.losses(worse, None) == ["observation_period loaded 47 rows, 105 before"]
    # a repair that fixes the exit dates of middle episodes joins periods (fewer of them): no loss
    history = _BuildHistory()
    history.add(ETLRunResult(started_at=NOW, steps=[merge_result(episodes=600, dropped_inverted=100, inserted=300)]),
                None)
    fixed = ETLRunResult(started_at=NOW, steps=[merge_result(episodes=600, dropped_inverted=0, inserted=200)])
    assert history.losses(fixed, None) == []


def test_report_shows_the_merge():
    run = ETLRunResult(started_at=NOW, steps=[
        merge_result(),
        TableLoadResult(target_table="observation_period", step="13_observation_period_fill", kind="derived",
                        rows_inserted=7, stats={"inserted": 7, "persons_without_period": 9})])
    steps = _etl_section(run)["steps"]
    assert steps[0]["periods"] == ("100 episodes → 90 periods (2 dropped: end before start; 10 open, closed at "
                                   "2023-12-31 (derived) or at death (1); persons with overlapping episodes: 3, "
                                   "with an open episode followed by a later one: 1)")
    assert steps[1]["periods"] == "7 of 9 persons without a period got one from their events"
    shared = _etl_section(run, 5, shareable=True)["steps"]
    assert shared[0]["periods"].startswith("100 episodes → 90 periods (<5 dropped")
    assert shared[0]["periods"].endswith("or at death (<5); persons with overlapping episodes: <5, with an open "
                                         "episode followed by a later one: <5)")
    assert _etl_section(run)["total_rows"] == 7 and _etl_section(run)["derived_rows"] == 7


def test_prompts_and_texts(tmp_path):
    assert "end NULL when the episode is still open" in DESIGN_SYSTEM
    assert "Never give observation_period_end_date a fallback" in DESIGN_SYSTEM
    assert "do not merge, de-duplicate or filter episodes yourself" in DESIGN_SYSTEM
    assert "one row per episode" in _plan_prompt("", "")
    assert "observation_period_merge, observation_period_fill" in REPAIR_SYSTEM
    cfg = plain_config(tmp_path)
    prompt = build_repair_prompt(load_ref(), 1, [], [], cfg)
    assert "never write merging, filters or a fallback into the observation_period end date" in prompt
    assert "Observation periods: from_events" in build_repair_prompt(from_events(), 1, [], [], cfg)
    md = render_spec_md(load_ref())
    assert "ends open ones (no end date) at `cdm_source.source_release_date`" in md


def test_normalise_plan_follows_the_planned_tables():
    def plan(strategy: str, tables: list[str]) -> TablePlan:
        return TablePlan(person_key=KeySpec(table="individuals", expression="individual_id"),
                         observation_period=strategy,  # type: ignore[arg-type]
                         tables=[PlannedTable(target_table=t, source_tables=["x"], rationale="") for t in tables])

    assert _normalise_plan(plan("from_events", ["person", "observation_period"])).observation_period == "table_mapping"
    assert _normalise_plan(plan("table_mapping", ["person"])).observation_period == "from_events"
    assert _normalise_plan(plan("from_events", ["person"])).observation_period == "from_events"
    switched = _normalise_plan(plan("from_events", ["person", "observation_period"]))
    assert any("table_mapping" in a for a in switched.assumptions)


def test_validator_observation_period_rules():
    def op_issues(spec) -> list[tuple[str, str | None, str]]:
        return [(i.severity, i.target_field, i.message) for i in validate_spec(spec)
                if i.target_table == "observation_period"]

    assert op_issues(load_ref()) == []
    # from_events and a mapping: an error (the fill covers persons without an episode)
    spec = load_ref().model_copy(update={"observation_period": "from_events"})
    (issue,) = op_issues(spec)
    assert issue[0] == "error" and "use 'table_mapping'" in issue[2]
    # an end with a fallback: the build date, the start date, a placeholder, a constant
    for expression, kind, why in [
        ("COALESCE(r.end_date::date, CURRENT_DATE)", "expression", "the build date"),
        ("COALESCE(r.end_date::date, now()::date)", "expression", "the build date"),
        ("COALESCE(r.end_date::date, r.start_date::date)", "expression", "falls back to the start date"),
        ("CASE WHEN r.end_date IS NULL THEN r.start_date::date ELSE r.end_date::date END", "expression",
         "falls back to the start date"),
        ("COALESCE(r.end_date::date, '2023-12-31'::date)", "expression", "a placeholder date"),
        ("CASE WHEN r.end_date IS NULL THEN DATE '2023-12-31' ELSE r.end_date::date END", "expression",
         "a placeholder date"),
        ("CASE WHEN r.end_date IS NULL THEN '2023-12-31'::date ELSE r.end_date::date END", "expression",
         "a placeholder date"),
        ("GREATEST(r.start_date::date, r.end_date::date)", "expression", "falls back to the start date"),
        ("DATE '2023-12-31'", "constant", "a constant end date"),
    ]:
        issues = op_issues(set_field(load_ref(), "observation_period", "observation_period_end_date", kind=kind,
                                     expression=expression))
        assert [(s, f) for s, f, _ in issues] == [("warning", "observation_period_end_date")], expression
        assert why in issues[0][2] and "leave the end NULL" in issues[0][2], expression
    # an end made NULL when it is before the start: episodes that end before they start become open
    for expression in ("CASE WHEN r.end_date::date < r.start_date::date THEN NULL ELSE r.end_date::date END",
                       "CASE WHEN r.end_date::date >= r.start_date::date THEN r.end_date::date END"):
        (issue,) = op_issues(set_field(load_ref(), "observation_period", "observation_period_end_date",
                                       expression=expression))
        assert issue[0] == "warning" and "turns episodes that end before they start into open ones" in issue[2]
        assert "falls back to the start date" not in issue[2]
    # making a placeholder NULL is fine, in a CASE condition too, and so is a NULL constant
    for kind, expression in [("expression", "NULLIF(r.end_date, '9999-12-31')::date"),
                             ("expression", "CASE WHEN r.end_date = '9999-12-31' THEN NULL ELSE r.end_date::date END"),
                             ("expression", "CASE WHEN r.end_event = 'OBE' THEN NULL ELSE r.end_date::date END"),
                             ("constant", "NULL::date")]:
        ok = set_field(load_ref(), "observation_period", "observation_period_end_date", kind=kind,
                       expression=expression)
        assert op_issues(ok) == [], expression
    # kind null is refused like any required field's, and says how to load episodes that are all open
    (issue,) = op_issues(set_field(load_ref(), "observation_period", "observation_period_end_date", kind="null",
                                   expression=None))
    assert issue[0] == "error" and "use kind constant with expression NULL::date" in issue[2]
    # a correlated subquery that merges episodes
    merging = set_field(load_ref(), "observation_period", "observation_period_end_date", expression=(
        "(SELECT MAX(x.end_date::date) FROM {src}.residency_episodes x WHERE x.individual_id = r.individual_id)"))
    (issue,) = op_issues(merging)
    assert issue[0] == "warning" and "let the build merge" in issue[2]
