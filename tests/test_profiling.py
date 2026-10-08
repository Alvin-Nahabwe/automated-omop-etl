from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest
from conftest import make_config

from omop_agent.db import ensure_schema, execute, fetch_all, qi, scalar
from omop_agent.demo.synth import FIRST_NAMES, LAST_NAMES, RARE_DRUGS
from omop_agent.llm import NullLLM, ReplayLLM
from omop_agent.llm.base import AgentResult, LLMClient
from omop_agent.llm.replay import save_recording
from omop_agent.models import (
    ColumnAnnotation,
    ColumnProfile,
    SourceAnnotations,
    SourceProfile,
    TableAnnotation,
    TableProfile,
    ValueFreq,
)
from omop_agent.profiling.annotate import ANNOTATE_SYSTEM, annotate, heuristic_annotations
from omop_agent.profiling.profiler import profile_staging, strftime_to_pg
from omop_agent.profiling.render import render_profile_md
from omop_agent.workspace import Workspace

DEMO_TABLES = ["clinic_visits", "deaths", "diagnoses", "individuals", "lab_results", "prescriptions", "residency_episodes"]


@pytest.fixture
def demo(staged):
    conn, schemas, cfg = staged
    return conn, schemas, cfg, profile_staging(conn, schemas, cfg)


def col(profile: SourceProfile, table: str, column: str) -> ColumnProfile:
    t = profile.table(table)
    assert t is not None, table
    c = t.column(column)
    assert c is not None, f"{table}.{column}"
    return c


# --------------------------------------------------------------------------------------
# Profiler on the demo data
# --------------------------------------------------------------------------------------


@pytest.mark.postgres
def test_profile_shape_and_types(demo):
    conn, schemas, cfg, profile = demo
    assert [t.name for t in profile.tables] == DEMO_TABLES
    assert profile.source_type == "csv"
    assert profile.min_cell_count == cfg.privacy.min_cell_count == 5
    assert profile.generated_at.tzinfo is not None
    assert profile.table("individuals").n_rows == 400

    dob = col(profile, "individuals", "dob")
    assert (dob.inferred_type, dob.date_format) == ("date", "%d/%m/%Y")
    # chronological (not lexical) range, converted to ISO
    assert "1940-01-01" <= dob.min_value < "1950-01-01"
    assert "2020-01-01" <= dob.max_value <= "2022-12-31"

    vd = col(profile, "clinic_visits", "visit_date")
    assert (vd.inferred_type, vd.date_format) == ("date", "%Y-%m-%d")
    assert "2015-01-01" <= vd.min_value <= vd.max_value <= "2023-12-31"

    qty = col(profile, "prescriptions", "qty")
    assert qty.inferred_type == "integer" and qty.mean is not None
    assert float(qty.min_value) <= qty.mean <= float(qty.max_value)

    vid = col(profile, "clinic_visits", "visit_id")
    assert vid.unique and vid.inferred_type == "string" and vid.n_distinct == vid.n_rows
    assert vid.top_values == [] and vid.n_suppressed_values == vid.n_distinct

    result = col(profile, "lab_results", "result")
    assert result.inferred_type == "string"  # numbers mixed with POS/NEG
    assert {"POS", "NEG"} <= {v.value for v in result.top_values}
    assert {"HB", "MRDT"} <= {v.value for v in col(profile, "lab_results", "test").top_values}


@pytest.mark.postgres
def test_discharge_date_nulls(demo):
    conn, schemas, cfg, profile = demo
    c = col(profile, "clinic_visits", "discharge_date")
    non_ipd = scalar(conn, f"SELECT count(*) FROM {schemas.src}.clinic_visits WHERE visit_type <> 'IPD'")
    assert c.n_null == non_ipd
    assert 0.6 < c.n_null / c.n_rows < 0.99
    assert c.inferred_type == "date" and not c.is_pii  # ISO dates must not look like phone numbers


@pytest.mark.postgres
def test_pii_columns_have_no_values(demo):
    conn, schemas, cfg, profile = demo
    for name in ("first_name", "last_name", "phone"):
        c = col(profile, "individuals", name)
        assert c.is_pii and c.pii_reason
        assert c.top_values == [] and c.min_value is None and c.max_value is None and c.mean is None
        assert c.n_distinct > 0  # counts are still reported
    for name in ("individual_id", "sex", "dob", "village", "hh_id"):
        assert not col(profile, "individuals", name).is_pii, name

    dumped = profile.model_dump_json()
    phones = [r["phone"] for r in fetch_all(conn, f"SELECT phone FROM {schemas.src}.individuals")]
    for value in [*FIRST_NAMES, *LAST_NAMES, *phones]:
        assert value not in dumped


@pytest.mark.postgres
def test_key_columns_list_no_values(demo):
    conn, schemas, cfg, profile = demo
    for table in ("clinic_visits", "diagnoses", "lab_results", "prescriptions", "deaths"):
        c = col(profile, table, "individual_id")
        assert c.top_values == [] and c.min_value is None and c.max_value is None
        assert c.n_suppressed_values == c.n_distinct and not c.is_pii
    assert col(profile, "individuals", "hh_id").top_values == []  # 124 households


@pytest.mark.postgres
def test_sex_values(demo):
    conn, schemas, cfg, profile = demo
    sex = col(profile, "individuals", "sex")
    values = {v.value for v in sex.top_values}
    assert {"1", "2"} <= values
    n9 = scalar(conn, f"SELECT count(*) FROM {schemas.src}.individuals WHERE sex = '9'")
    assert ("9" in values) == (n9 >= cfg.privacy.min_cell_count)
    assert sex.n_distinct == 3 and sex.n_suppressed_values == 3 - len(values)


@pytest.mark.postgres
def test_no_small_cells_anywhere(demo):
    conn, schemas, cfg, profile = demo
    k = profile.min_cell_count
    for t in profile.tables:
        for c in t.columns:
            assert all(v.count >= k for v in c.top_values), f"{t.name}.{c.name}"
            assert c.n_suppressed_values == (0 if c.is_pii else c.n_distinct - len(c.top_values))
            if c.inferred_type == "string":
                for v in (c.min_value, c.max_value):
                    if v is not None:
                        n = scalar(conn, f"SELECT count(*) FROM {schemas.src}.{qi(t.name)} WHERE {qi(c.name)} = %s", (v,))
                        assert n >= k, f"{t.name}.{c.name} min/max {v!r} seen {n} times"


@pytest.mark.postgres
def test_rare_drugs_suppressed(demo):
    conn, schemas, cfg, profile = demo
    for drug in RARE_DRUGS:  # premise: the demo writes them fewer than 5 times
        n = scalar(conn, f"SELECT count(*) FROM {schemas.src}.prescriptions WHERE drug = %s", (drug,))
        assert 0 < n < cfg.privacy.min_cell_count
    c = col(profile, "prescriptions", "drug")
    shown = {v.value for v in c.top_values}
    assert "Paracetamol 500mg" in shown
    assert not any(d.startswith(("Quinine", "Ceftriaxone")) for d in shown)
    assert c.n_suppressed_values >= len(RARE_DRUGS)
    assert c.min_value != c.max_value and c.min_value not in RARE_DRUGS and c.max_value not in RARE_DRUGS
    md = render_profile_md(profile)
    assert "Quinine" not in md and "Ceftriaxone" not in md


@pytest.mark.postgres
def test_manifest_order_and_declared_types(staged):
    conn, schemas, cfg = staged
    manifest = {
        "prescriptions": {"columns": [{"name": "qty", "source_name": "Qty", "declared_type": "INTEGER"}]},
        "individuals": {"columns": []},
        "not_staged": {"columns": []},
    }
    profile = profile_staging(conn, schemas, cfg, manifest)
    names = [t.name for t in profile.tables]
    assert names[:2] == ["prescriptions", "individuals"]
    assert sorted(names) == DEMO_TABLES
    assert col(profile, "prescriptions", "qty").declared_type == "INTEGER"
    assert col(profile, "prescriptions", "drug").declared_type is None


@pytest.mark.postgres
def test_render_profile_md(demo):
    conn, schemas, cfg, profile = demo
    md = render_profile_md(profile)
    assert md.startswith("# Source data profile")
    assert "fewer than 5 times" in md
    assert "| individuals | 400 | 8 | 3 |" in md
    for t in DEMO_TABLES:
        assert f"## {t}" in md
    assert "`%d/%m/%Y`" in md and "**PII**" in md and "*withheld*" in md
    assert "`Paracetamol 500mg`×" in md
    for name in [*FIRST_NAMES, *LAST_NAMES]:
        assert name not in md


# --------------------------------------------------------------------------------------
# Type inference on hand-made columns
# --------------------------------------------------------------------------------------

TYPE_CASES = {
    # column: (values cycled over the rows, expected type, expected format)
    "flag": (["Y", "N", "yes", "No", "t", "F"], "boolean", None),
    "zero_one": (["0", "1"], "integer", None),
    "amount": (["1.5", "2", "-3.25", "10"], "float", None),
    "ts": (["2020-01-02 10:11:12", "2021-06-30 23:59:01", "2019-12-31 7:05:00"], "datetime", "%Y-%m-%d %H:%M:%S"),
    "ts_t": (["2020-01-02T10:11", "2020-03-04T08:00"], "datetime", "%Y-%m-%dT%H:%M"),
    "us_date": (["12/31/2019", "01/15/2020", "3/5/2018"], "date", "%m/%d/%Y"),
    "eu_date": (["31/12/2019", "01/01/2020", "5/3/2018"], "date", "%d/%m/%Y"),
    "ambiguous": (["01/02/2020", "03/04/2020"], "date", "%d/%m/%Y"),
    "compact": (["20190105", "20201231", "19991101"], "date", "%Y%m%d"),
    "dashed": (["31-12-2019", "01-01-2020"], "date", "%d-%m-%Y"),
    "mon": (["05-Jan-2019", "17-MAR-2020", "1-dec-2018"], "date", "%d-%b-%Y"),
    "dotted": (["31.12.2019", "01.01.2020"], "date", "%d.%m.%Y"),
    "code": (["A1", "B2", "C3"], "string", None),
    "empty": ([None], "empty", None),
}


@pytest.mark.postgres
def test_type_inference(pg, tmp_path):
    conn, schemas = pg
    cfg = make_config(tmp_path, tmp_path, schemas)
    ensure_schema(conn, schemas.src)
    names = [*TYPE_CASES, "bad_date", "mostly_int"]
    execute(conn, f"CREATE TABLE {schemas.src}.cases ({', '.join(f'{qi(n)} text' for n in names)})")
    n_rows = 60
    rows = []
    for i in range(n_rows):
        row = [vals[i % len(vals)] for vals, _, _ in TYPE_CASES.values()]
        row.append("31/02/2025" if i == 0 else f"{1 + i % 28:02d}/{1 + i % 12:02d}/20{10 + i % 10}")
        row.append("n/a" if i == 0 else str(i))
        rows.append(row)
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO {schemas.src}.cases VALUES ({', '.join(['%s'] * len(names))})", rows)

    t = profile_staging(conn, schemas, cfg).table("cases")
    assert t is not None and t.n_rows == n_rows
    for name, (_, typ, fmt) in TYPE_CASES.items():
        c = t.column(name)
        assert (c.inferred_type, c.date_format) == (typ, fmt), name

    assert (t.column("eu_date").min_value, t.column("eu_date").max_value) == ("2018-03-05", "2020-01-01")
    assert (t.column("us_date").min_value, t.column("us_date").max_value) == ("2018-03-05", "2020-01-15")
    assert (t.column("mon").min_value, t.column("mon").max_value) == ("2018-12-01", "2020-03-17")
    assert (t.column("compact").min_value, t.column("compact").max_value) == ("1999-11-01", "2020-12-31")
    assert (t.column("ts").min_value, t.column("ts").max_value) == ("2019-12-31 07:05:00", "2021-06-30 23:59:01")
    assert t.column("amount").mean == pytest.approx((1.5 + 2 - 3.25 + 10) / 4)
    empty = t.column("empty")
    assert (empty.n_null, empty.n_distinct, empty.unique, empty.top_values) == (n_rows, 0, False, [])
    # an impossible date among valid dd/mm/yyyy values: still a date column, the scan does not
    # fail, and the impossible date is counted as non-conforming instead of being the maximum
    bad = t.column("bad_date")
    assert (bad.inferred_type, bad.date_format) == ("date", "%d/%m/%Y")
    assert (bad.min_value, bad.max_value) == ("2010-03-23", "2019-12-04")
    assert (bad.n_nonconforming, bad.n_partial_dates) == (1, 0)
    mostly = t.column("mostly_int")
    assert mostly.inferred_type == "integer" and (mostly.min_value, mostly.max_value) == ("1", "59")

    # the reported strftime format parses the column in PostgreSQL
    for name in ("eu_date", "us_date", "compact", "mon", "dotted", "dashed"):
        tmpl = strftime_to_pg(t.column(name).date_format)
        lo = scalar(conn, f"SELECT min(to_date({qi(name)}, %s))::text FROM {schemas.src}.cases", (tmpl,))
        assert lo == t.column(name).min_value, name


def _layout_rows(n: int) -> list[list[str | None]]:
    """Rows of the layouts table: each column cycles through valid values, with named exceptions."""
    mons = "jan feb mar apr may jun jul aug sep oct nov dec".split()
    rows = []
    for i in range(n):
        d, m, y = 1 + i % 28, 1 + i % 12, 2010 + i % 10
        if i < 20:
            dob = "00/06/1975"  # unknown day
        elif i < 25:
            dob = "06/1975"  # no day
        elif i == 25:
            dob = "99/99/1980"  # unknown day and month
        elif i == 26:
            dob = "31/02/1980"  # no such date
        elif i < 33:
            dob = "unknown"
        else:
            dob = f"{d:02d}/{m:02d}/{1950 + i % 50}"
        if i < 60:
            odk = f"{y}-{m:02d}-{d:02d}T23:30:00Z"
        elif i < 62:
            odk = f"{y}-{m:02d}-{d:02d}T10:00:00-05:00"  # rare: only counted
        elif i == 62:
            odk = f"{y}-{m:02d}-{d:02d}T10:00:00+0300"
        else:
            odk = f"{y}-{m:02d}-{d:02d}T10:00:00+03:00"
        # timestamps with seconds, some without (the build reads none of those) and some with an
        # offset (it reads those)
        hms = f"{d:02d}/{m:02d}/{y} {i % 24:02d}:{i % 60:02d}" + ("" if i % 10 < 3 else ":30")
        hms += "+03:00" if i % 10 == 9 else ""
        # dates followed by a time: with an offset, after a 'T', with seconds (all read), at hour 25 (not)
        dmy_t = f"{d:02d}/{m:02d}/{y}" + {0: " 10:30+03:00", 1: "T10:30", 2: " 25:00", 3: " 10:30:15"}.get(i % 100, "")
        rows.append([
            f"{d:02d}{mons[m - 1]}{y}",  # Stata %td
            f"{d:02d}{mons[m - 1].upper()}{y}",  # SAS DATE9
            f"{d:02d}{mons[m - 1].upper()}{y}:{i % 24:02d}:{i % 60:02d}:00",  # SAS DATETIME
            f"{d:02d}/{m:02d}/{y} {i % 24}:{i % 60:02d}",
            f"{d:02d}/{m:02d}/{y % 100:02d}",
            str(43000 + i),  # Excel serial days, in a column named like a date ...
            str(43000 + i),  # ... and in ones that are not
            str(43000 + i),
            f"{i % 24}:{i % 60:02d}",
            dob,
            f"{y}-{m:02d}-{d:02d}" if i % 5 == 0 else f"{d:02d}/{m:02d}/{y}",  # dd/mm with ISO dates
            odk,
            "2050-12-31" if i % 4 == 0 else f"{y}-{m:02d}-{d:02d}",  # "still resident" exits
            "n/a" if i % 30 == 0 else str(i),
            f"{1 + i % 12:02d}/{1 + i % 12:02d}/{y}",  # day and month <= 12 everywhere
            f"{1 + i % 12:02d}/{1 + i % 12:02d}/{y} 10:{i % 60:02d}",  # the same with a time
            hms,
            f"{d:02d}-{m:02d}-{y} {i % 24:02d}:{i % 60:02d}",  # dd-mm-yyyy HH:MM
            dmy_t,
            # birth dates: 10% a year alone, 10% without the day
            "1975" if i % 10 == 0 else f"{m:02d}.1975" if i % 10 == 5 else f"{d:02d}.{m:02d}.{1950 + i % 50}",
            f"{d:02d}/{m:02d}/{(1950 + i % 50) % 100:02d}",  # two-digit years above 31: the year is last
            f"{i % 24}:{i % 60:02d}" if i % 2 else f"{i % 24}:{i % 60:02d}:30",  # times, half with seconds
            f"{i}.0" if i % 100 == 1 else str(i),  # integers, some written '5.0' (the build reads them)
            # ISO timestamps; a few with '/': read with the format's own 'T', not with a space
            f"{y}/{m:02d}/{d:02d}" + (" " if i % 50 == 7 else "T") + "10:30:00" if i % 50 in (7, 27)
            else f"{y}-{m:02d}-{d:02d}T10:30:00",
        ])
    return rows


LAYOUT_COLUMNS = ["stata", "sas", "sas_dt", "dmy_hm", "yy", "excel_date", "excel_n", "dt_code", "tod", "dob", "mixed",
                  "odk", "end_date", "qty", "ambiguous", "ambiguous_ts", "hms_mix", "dash_hm", "dmy_t", "birth_mix",
                  "yy_old", "tod_mix", "counts", "iso_slash"]


@pytest.mark.postgres
def test_profiler_layouts_and_nonconforming(pg, tmp_path):
    from omop_agent.cdm.ddl import create_etl_tables
    from omop_agent.profiling.profiler import (
        DAY_FIRST_NOTE,
        EXCEL_NOTE,
        SAMPLE_VALUES,
        TWO_DIGIT_YEAR_NOTE,
        YEAR_FIRST_NOTE,
    )

    conn, schemas = pg
    cfg = make_config(tmp_path, tmp_path, schemas)
    ensure_schema(conn, schemas.src)
    execute(conn, f"CREATE TABLE {schemas.src}.layouts ({', '.join(f'{qi(c)} text' for c in LAYOUT_COLUMNS)})")
    n = 200
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO {schemas.src}.layouts VALUES ({', '.join(['%s'] * len(LAYOUT_COLUMNS))})",
                        _layout_rows(n))
    # a table larger than the sample whose values change late in the file: the sample is spread
    # over the table (its first 20,000 rows hold only dates)
    late = SAMPLE_VALUES + 10_000
    execute(conn, f"""CREATE TABLE {schemas.src}.late AS SELECT CASE WHEN g <= {SAMPLE_VALUES + 1000}
                      THEN to_char(DATE '2015-01-01' + g % 2000, 'YYYY-MM-DD') ELSE 'pending' END AS outcome_date
                      FROM generate_series(1, {late}) g""")
    t = profile_staging(conn, schemas, cfg).table("layouts")
    c = {x.name: x for x in t.columns}
    expected = {
        "stata": ("date", "%d%b%Y"), "sas": ("date", "%d%b%Y"), "sas_dt": ("datetime", "%d%b%Y:%H:%M:%S"),
        "dmy_hm": ("datetime", "%d/%m/%Y %H:%M"), "yy": ("date", "%d/%m/%y"), "excel_date": ("date", None),
        "excel_n": ("integer", None), "dt_code": ("integer", None), "tod": ("time", "%H:%M"),
        "dob": ("date", "%d/%m/%Y"), "mixed": ("date", "%d/%m/%Y"), "odk": ("datetime", "%Y-%m-%dT%H:%M:%S"),
        "end_date": ("date", "%Y-%m-%d"), "qty": ("integer", None), "ambiguous": ("date", "%d/%m/%Y"),
        "ambiguous_ts": ("datetime", "%d/%m/%Y %H:%M"), "hms_mix": ("datetime", "%d/%m/%Y %H:%M:%S"),
        "dash_hm": ("datetime", "%d-%m-%Y %H:%M"), "dmy_t": ("date", "%d/%m/%Y"), "birth_mix": ("date", "%d.%m.%Y"),
        "yy_old": ("date", "%d/%m/%y"), "tod_mix": ("time", "%H:%M"), "counts": ("integer", None),
        "iso_slash": ("datetime", "%Y-%m-%dT%H:%M:%S"),
    }
    assert {k: (x.inferred_type, x.date_format) for k, x in c.items()} == expected
    days = [date(2010 + i % 10, 1 + i % 12, 1 + i % 28) for i in range(n)]
    times = [datetime(d.year, d.month, d.day, i % 24, i % 60) for i, d in enumerate(days)]
    for name in ("stata", "sas", "yy"):
        assert (c[name].min_value, c[name].max_value) == (min(days).isoformat(), max(days).isoformat()), name
    for name in ("sas_dt", "dmy_hm", "dash_hm"):
        assert (c[name].min_value, c[name].max_value) == (str(min(times)), str(max(times))), name
    assert (c["tod"].min_value, c["tod"].max_value) == ("00:00:00", "23:59:00")
    # notes where the sample cannot settle the order of the fields
    assert c["yy"].format_notes == [TWO_DIGIT_YEAR_NOTE, YEAR_FIRST_NOTE]  # every year is 10..19
    assert c["yy_old"].format_notes == [TWO_DIGIT_YEAR_NOTE]
    assert c["ambiguous"].format_notes == [DAY_FIRST_NOTE] and c["ambiguous_ts"].format_notes == [DAY_FIRST_NOTE]
    assert c["excel_date"].format_notes == [EXCEL_NOTE] and c["dt_code"].format_notes == []
    excel = (date(1899, 12, 30) + timedelta(days=43000), date(1899, 12, 30) + timedelta(days=43000 + n - 1))
    assert (c["excel_date"].min_value, c["excel_date"].max_value) == tuple(d.isoformat() for d in excel)

    # partial dates are counted apart; min/max cover complete, valid dates only (DTB16)
    dob = c["dob"]
    assert (dob.n_partial_dates, dob.n_nonconforming) == (26, 7)
    assert dob.nonconforming_shapes == [ValueFreq(value="aaaaaaa", count=6)]  # 31/02/1980 is a small cell
    assert dob.min_value >= "1950-01-01" and dob.max_value <= "1999-12-31"
    # a year alone, and a month and year, are partial dates in type inference too
    assert (c["birth_mix"].n_partial_dates, c["birth_mix"].n_nonconforming) == (40, 0)
    # ISO dates fit a day-first column (the build reads them with any template)
    mixed = c["mixed"]
    assert mixed.n_nonconforming == 0 and (mixed.min_value, mixed.max_value) == (min(days).isoformat(),
                                                                               max(days).isoformat())
    assert {v.value for v in mixed.value_shapes} == {"99/99/9999", "9999-99-99"}
    # a time without the seconds the format has (or with seconds it lacks) is not read; an offset
    # after it is (and is counted)
    hms = c["hms_mix"]
    assert (hms.n_nonconforming, hms.n_partial_dates) == (60, 0)
    assert hms.tz_offsets == [ValueFreq(value="+03:00", count=20)]
    assert c["tod_mix"].n_nonconforming == 100
    dmy_t = c["dmy_t"]  # a date-only layout reads a time after the date, with an offset
    assert (dmy_t.n_nonconforming, dmy_t.n_partial_dates, dmy_t.tz_offsets, dmy_t.n_rare_offsets) == (2, 0, [], 2)
    two_digit_hours = sum(i % 24 >= 10 for i in range(n))
    assert c["dmy_hm"].value_shapes == [ValueFreq(value="99/99/9999 99:99", count=two_digit_hours),
                                        ValueFreq(value="99/99/9999 9:99", count=n - two_digit_hours)]
    assert c["stata"].value_shapes == []  # one shape for every value says no more than the format
    qty = c["qty"]
    assert (qty.n_nonconforming, qty.nonconforming_shapes) == (7, [ValueFreq(value="a/a", count=7)])
    assert c["counts"].n_nonconforming == 0  # '101.0' is read as an integer
    assert (c["iso_slash"].n_nonconforming, c["iso_slash"].n_partial_dates) == (4, 0)
    # offsets: '+0300' is '+03:00'; the two '-05:00' are below the small-cell threshold and only counted
    odk = c["odk"]
    assert odk.tz_offsets == [ValueFreq(value="+03:00", count=138), ValueFreq(value="Z", count=60)]
    assert (odk.n_rare_offsets, odk.format_notes) == (2, [])
    assert "-05:00" not in odk.model_dump_json()
    (note,) = c["end_date"].format_notes
    assert note.startswith("50 values (25%) are one and the same date after the profile date") and "2050" not in note
    late_col = profile_staging(conn, schemas, cfg).table("late").column("outcome_date")
    assert late_col.inferred_type == "string"
    assert late_col.value_shapes == [ValueFreq(value="9999-99-99", count=SAMPLE_VALUES + 1000),
                                     ValueFreq(value="aaaaaaa", count=9000)]

    # the build reads every value as the profile says: complete, partial or not at all, and the
    # same earliest and latest values
    create_etl_tables(conn, schemas)
    etl = schemas.etl
    for name, x in c.items():
        if x.inferred_type not in ("date", "datetime") or x.date_format is None:
            continue
        fn = "try_date" if x.inferred_type == "date" else "try_timestamp"
        tmpl = strftime_to_pg(x.date_format)
        got = fetch_all(conn, f"""SELECT
            min({etl}.{fn}({qi(name)}, %(t)s))::text AS lo, max({etl}.{fn}({qi(name)}, %(t)s))::text AS hi,
            count(*) FILTER (WHERE {etl}.{fn}({qi(name)}, %(t)s, 'first') IS NULL AND {qi(name)} IS NOT NULL) AS nc,
            count(*) FILTER (WHERE {etl}.{fn}({qi(name)}, %(t)s, 'first') IS NOT NULL
                             AND {etl}.{fn}({qi(name)}, %(t)s) IS NULL) AS np
            FROM {schemas.src}.layouts""", {"t": tmpl})[0]
        assert got == {"lo": x.min_value, "hi": x.max_value, "nc": x.n_nonconforming, "np": x.n_partial_dates}, name
    lo = scalar(conn, f"SELECT min(DATE '1899-12-30' + excel_date::integer)::text FROM {schemas.src}.layouts")
    assert lo == c["excel_date"].min_value


def test_strftime_to_pg():
    assert strftime_to_pg("%d/%m/%Y") == "DD/MM/YYYY"
    assert strftime_to_pg("%Y-%m-%dT%H:%M:%S") == 'YYYY-MM-DD"T"HH24:MI:SS'
    assert strftime_to_pg("%d-%b-%Y") == "DD-Mon-YYYY"
    with pytest.raises(ValueError):
        strftime_to_pg("%Q")


# --------------------------------------------------------------------------------------
# Annotation
# --------------------------------------------------------------------------------------


@pytest.mark.postgres
def test_heuristic_annotations_demo(demo):
    conn, schemas, cfg, profile = demo
    ann = heuristic_annotations(profile)
    tables = {t.table: t for t in ann.tables}
    cols = {(c.table, c.column): c for c in ann.columns}
    assert len(cols) == sum(len(t.columns) for t in profile.tables)
    assert tables["individuals"].person_key == "individual_id"
    assert tables["lab_results"].person_key == "individual_id"
    icd = cols[("diagnoses", "icd10")]
    assert (icd.semantic_type, icd.coding_system) == ("coded", "ICD10")
    assert cols[("individuals", "first_name")].semantic_type == "pii"
    assert cols[("individuals", "dob")].semantic_type == "date"
    assert cols[("individuals", "sex")].semantic_type == "categorical"
    assert cols[("clinic_visits", "visit_id")].semantic_type == "record_identifier"
    assert cols[("diagnoses", "visit_id")].semantic_type == "foreign_key"
    assert cols[("prescriptions", "qty")].semantic_type == "numeric_measurement"
    assert cols[("lab_results", "result")].semantic_type == "numeric_measurement"
    expected = {
        "individuals": "person",
        "residency_episodes": "observation_period",
        "clinic_visits": "visit_occurrence",
        "diagnoses": "condition_occurrence",
        "prescriptions": "drug_exposure",
        "lab_results": "measurement",
        "deaths": "death",
    }
    for table, cdm in expected.items():
        assert tables[table].candidate_cdm_tables[0] == cdm, table
    assert any("%d/%m/%Y" in n and "DD/MM/YYYY" in n for n in ann.notes)
    assert any("discharge_date" in n for n in ann.notes)


def _recorded_answer() -> dict:
    return SourceAnnotations(
        tables=[
            TableAnnotation(
                table="individuals",
                description="HDSS register.",
                grain="one row per individual",
                person_key="individual_id",
                candidate_cdm_tables=["Person", "cdm.location", "not_a_cdm_table"],
            ),
            TableAnnotation(table="diagnoses", description="Diagnoses.", grain="one per visit", person_key="no_such_column"),
            TableAnnotation(table="ghost", description="Not in the source.", grain="?"),
        ],
        columns=[
            ColumnAnnotation(table="individuals", column="individual_id", semantic_type="person_identifier", description="Id."),
            ColumnAnnotation(table="individuals", column="first_name", semantic_type="categorical", coding_system="local", description="Name."),
            ColumnAnnotation(table="diagnoses", column="icd10", semantic_type="coded", coding_system="ICD10", description="ICD-10 code."),
            ColumnAnnotation(table="individuals", column="no_such_column", semantic_type="other", description="?"),
            ColumnAnnotation(table="ghost", column="x", semantic_type="other", description="?"),
        ],
        notes=["dob is dd/mm/yyyy.", "Sex code 9 needs confirming."],
    ).model_dump(mode="json")


@pytest.mark.postgres
def test_annotate_with_replay(demo, tmp_path):
    conn, schemas, cfg, profile = demo
    save_recording(tmp_path / "replay", "annotate", "source", {"kind": "structured", "output": _recorded_answer()})
    ws = Workspace(tmp_path / "work")
    ann = annotate(profile, cfg, ReplayLLM(tmp_path / "replay", ws))

    assert [t.table for t in ann.tables] == [t.name for t in profile.tables]  # ghost dropped, rest filled
    tables = {t.table: t for t in ann.tables}
    assert tables["individuals"].description == "HDSS register."
    assert tables["individuals"].candidate_cdm_tables == ["person", "location"]
    assert tables["diagnoses"].person_key is None
    assert tables["prescriptions"].candidate_cdm_tables == ["drug_exposure"]  # from heuristics

    cols = {(c.table, c.column): c for c in ann.columns}
    assert len(cols) == len(ann.columns) == sum(len(t.columns) for t in profile.tables)
    assert ("ghost", "x") not in cols and ("individuals", "no_such_column") not in cols
    fn = cols[("individuals", "first_name")]
    assert (fn.semantic_type, fn.coding_system, fn.description) == ("pii", None, "Name.")
    assert cols[("individuals", "phone")].semantic_type == "pii"  # filled by heuristics
    assert cols[("diagnoses", "icd10")].description == "ICD-10 code."
    assert ann.notes == ["dob is dd/mm/yyyy.", "Sex code 9 needs confirming."]

    # the prompt that went to the "model" is on record and carries no identifying values
    [call_file] = (ws.root / "llm_calls").glob("*.json")
    call = json.loads(call_file.read_text())
    assert (call["stage"], call["key"], call["system"]) == ("annotate", "source", ANNOTATE_SYSTEM)
    assert "SOURCE TABLE individuals" in call["prompt"] and "'Paracetamol 500mg'" in call["prompt"]
    for value in [*FIRST_NAMES, *LAST_NAMES, *RARE_DRUGS, "KH000001"]:
        assert value not in call["prompt"]


@pytest.mark.postgres
def test_annotate_without_llm_uses_heuristics(demo):
    conn, schemas, cfg, profile = demo
    assert annotate(profile, cfg, NullLLM()) == heuristic_annotations(profile)


# --------------------------------------------------------------------------------------
# Pure Python: privacy filtering of a hand-built profile
# --------------------------------------------------------------------------------------


class CapturingLLM(LLMClient):
    provider = "test"

    def __init__(self, answer: SourceAnnotations):
        super().__init__()
        self.answer = answer
        self.prompts: list[str] = []

    def structured(self, *, stage, key, system, prompt, output):  # type: ignore[override]
        self.prompts.append(prompt)
        return self.answer

    def run_agent(self, **kwargs) -> AgentResult:  # type: ignore[override]
        raise NotImplementedError


def _hand_profile() -> SourceProfile:
    def c(name: str, **kw) -> ColumnProfile:
        base = dict(name=name, inferred_type="string", n_rows=100, n_null=0, n_distinct=10, unique=False)
        return ColumnProfile(**{**base, **kw})

    return SourceProfile(
        source_type="csv",
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        min_cell_count=5,
        tables=[
            TableProfile(
                name="patients",
                n_rows=100,
                columns=[
                    c("patient_id", n_distinct=100, unique=True, n_suppressed_values=100),
                    # a PII column that (wrongly) still carries values must not leak them
                    c("first_name", is_pii=True, pii_reason="name", min_value="Aisha", max_value="Zawadi",
                      top_values=[ValueFreq(value="Aisha", count=40)]),
                    c("dx_code", n_distinct=4,
                      top_values=[ValueFreq(value="B54", count=60), ValueFreq(value="I10", count=30),
                                  ValueFreq(value="A09", count=3), ValueFreq(value="Q99_RARE", count=1)]),
                    c("visit_date", inferred_type="date", date_format="%d.%m.%Y", n_null=30, n_distinct=70,
                      min_value="2019-01-01", max_value="2020-12-31"),
                ],
            )
        ],
    )


def test_privacy_filtering_hand_built_profile():
    profile = _hand_profile()
    md = render_profile_md(profile)
    assert "`B54`×60" in md and "`I10`×30" in md
    for leaked in ("Aisha", "Zawadi", "Q99_RARE", "`A09`×3"):
        assert leaked not in md
    assert "2 not shown" in md  # the two small cells are counted, not listed

    answer = SourceAnnotations(
        tables=[TableAnnotation(table="patients", description="Patients.", grain="one per patient", person_key="patient_id")],
        columns=[ColumnAnnotation(table="patients", column="first_name", semantic_type="other", description="x")],
    )
    llm = CapturingLLM(answer)
    ann = annotate(profile, make_config_stub(), llm)
    [prompt] = llm.prompts
    assert "B54" in prompt and "patients" in prompt
    for leaked in ("Aisha", "Zawadi", "Q99_RARE", "'A09'"):
        assert leaked not in prompt
    cols = {c.column: c for c in ann.columns}
    assert cols["first_name"].semantic_type == "pii"
    assert cols["dx_code"].semantic_type == "coded" and cols["dx_code"].coding_system == "ICD10"
    assert cols["visit_date"].semantic_type == "date"

    heur = heuristic_annotations(profile)
    assert heur.tables[0].person_key == "patient_id"
    assert heur.tables[0].candidate_cdm_tables[0] == "person"
    assert any("%d.%m.%Y" in n and "DD.MM.YYYY" in n for n in heur.notes)
    assert any("visit_date" in n and "30%" in n for n in heur.notes)


def make_config_stub():
    from omop_agent.config import ProjectConfig

    return ProjectConfig.model_validate(
        {"name": "t", "source": {"type": "csv", "path": "."}, "target": {"url": "postgresql://unused"}}
    )
