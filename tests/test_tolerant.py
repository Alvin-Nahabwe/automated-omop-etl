"""Tolerant conversion: the rewrite of spec SQL (no database) and the helper functions of
etl/functions.sql (PostgreSQL)."""

from __future__ import annotations

import random
import re
import time
from datetime import date, timedelta

import pytest

from omop_agent.cdm.ddl import create_etl_tables
from omop_agent.config import EtlConfig
from omop_agent.db import execute, fetch_all, ql, scalar
from omop_agent.etl.policies import partial_policy, start_end_pairs
from omop_agent.etl.tolerant import (
    kind_of,
    lex,
    rewrite,
    template_issue,
    tolerant_value,
    without_conversions,
)

# --------------------------------------------------------------------------------------
# the rewrite (no database)
# --------------------------------------------------------------------------------------

_E = "{etl}"


@pytest.mark.parametrize(
    "expr, counted, uncounted",
    [
        ("v.visit_date::date", f"{_E}.omop_as_date_at('f', 'mid', v.visit_date)", f"{_E}.try_date(v.visit_date, 'ISO', 'mid')"),
        ("to_date(i.dob, 'DD/MM/YYYY')", f"{_E}.omop_as_date_at('f', 'mid', i.dob, 'DD/MM/YYYY')",
         f"{_E}.try_date(i.dob, 'DD/MM/YYYY', 'mid')"),
        # to_timestamp(x, 'T') returns timestamptz, and keeps that type
        ("to_timestamp(v.t, 'DD/MM/YYYY HH24:MI')", f"{_E}.omop_as_timestamptz_at('f', 'mid', v.t, 'DD/MM/YYYY HH24:MI')",
         f"{_E}.try_timestamptz(v.t, 'DD/MM/YYYY HH24:MI', 'mid')"),
        ("v.t::timestamptz AT TIME ZONE 'Africa/Nairobi'",
         f"{_E}.omop_as_timestamptz_at('f', 'mid', v.t) AT TIME ZONE 'Africa/Nairobi'", None),
        ("v.t::timestamp with time zone", f"{_E}.omop_as_timestamptz_at('f', 'mid', v.t)", None),
        ("CAST(v.d AS timestamp without time zone)", f"{_E}.omop_as_timestamp_at('f', 'mid', v.d)", None),
        ("CAST(v.x::text AS integer)", f"{_E}.omop_as_integer_at('f', 'mid', v.x::text)", f"{_E}.try_integer(v.x::text)"),
        ("COALESCE(v.discharge_date::date, v.visit_date::date)",
         f"COALESCE({_E}.omop_as_date_at('f', 'mid', v.discharge_date), {_E}.omop_as_date_at('f', 'mid', v.visit_date))",
         None),
        # the inner conversion first, then the outer one on its result
        ("EXTRACT(YEAR FROM to_date(i.dob, 'DD/MM/YYYY'))::int",
         f"EXTRACT(YEAR FROM {_E}.omop_as_date_at('f', 'mid', i.dob, 'DD/MM/YYYY'))::int", None),
        ("to_date(i.dob, 'DD/MM/YYYY')::timestamp",
         f"{_E}.omop_as_timestamp_at('f', 'mid', {_E}.omop_as_date_at('f', 'mid', i.dob, 'DD/MM/YYYY'))", None),
        ("CASE WHEN r.result ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN r.result::numeric END",
         f"CASE WHEN r.result ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN {_E}.omop_as_number_at('f', 'mid', r.result) END", None),
        ("NULLIF(trim(r.value), '')::numeric", f"{_E}.omop_as_number_at('f', 'mid', NULLIF(trim(r.value), ''))", None),
        ("(r.value)::numeric(10,2)", f"round({_E}.omop_as_number_at('f', 'mid', (r.value)), 2)", None),
        ("v.x::text::date", f"{_E}.omop_as_date_at('f', 'mid', v.x::text)", None),
        ("CASE WHEN v.a IS NULL THEN v.b ELSE v.a END::date",
         f"{_E}.omop_as_date_at('f', 'mid', CASE WHEN v.a IS NULL THEN v.b ELSE v.a END)", None),
        ('"V"."Visit Date"::date', f'{_E}.omop_as_date_at(\'f\', \'mid\', "V"."Visit Date")', None),
        ("arr[1]::int", f"{_E}.omop_as_integer_at('f', 'mid', arr[1])", None),
        ("-v.n::int", f"-{_E}.omop_as_integer_at('f', 'mid', v.n)", None),
        ("v.qty::double precision * 2", f"{_E}.omop_as_number_at('f', 'mid', v.qty) * 2", None),
        ("x :: date", f"{_E}.omop_as_date_at('f', 'mid', x)", None),
        ("v.a::int BETWEEN 1 AND v.b::int",
         f"{_E}.omop_as_integer_at('f', 'mid', v.a) BETWEEN 1 AND {_E}.omop_as_integer_at('f', 'mid', v.b)", None),
        # a comparison cast to a number: the boolean overload
        ("(v.flag = 'Y')::int", f"{_E}.omop_as_integer_at('f', 'mid', (v.flag = 'Y'))", f"{_E}.try_integer((v.flag = 'Y'))"),
        # the groups after OVER / FILTER / WITHIN GROUP / EXISTS belong to the operand
        ("count(*) OVER (PARTITION BY v.p)::int", f"{_E}.omop_as_integer_at('f', 'mid', count(*) OVER (PARTITION BY v.p))",
         None),
        ("sum(v.x) FILTER (WHERE v.y > 0)::numeric",
         f"{_E}.omop_as_number_at('f', 'mid', sum(v.x) FILTER (WHERE v.y > 0))", None),
        ("percentile_cont(0.5) WITHIN GROUP (ORDER BY v.x)::int",
         f"{_E}.omop_as_integer_at('f', 'mid', percentile_cont(0.5) WITHIN GROUP (ORDER BY v.x))", None),
        ("EXISTS (SELECT 1 FROM {src}.t t WHERE t.a = v.a)::int",
         f"{_E}.omop_as_integer_at('f', 'mid', EXISTS (SELECT 1 FROM {{src}}.t t WHERE t.a = v.a))", None),
        # strings: a backslash escapes only in E'...'; casts inside literals and comments stay
        ("E'\\\\'::text || v.d::date", f"E'\\\\'::text || {_E}.omop_as_date_at('f', 'mid', v.d)", None),
        ("'\\' || v.d::date::text", f"'\\' || {_E}.omop_as_date_at('f', 'mid', v.d)::text", None),
        ("$t$ x::date $t$ || v.d::date", f"$t$ x::date $t$ || {_E}.omop_as_date_at('f', 'mid', v.d)", None),
        ("x.code || '::date'", "x.code || '::date'", None),
        ("-- note ::date\n v.d::date", f"-- note ::date\n {_E}.omop_as_date_at('f', 'mid', v.d)", None),
        # left as they are: constants, array types, other types, EXTRACT fields that always fit
        ("'2019-01-01'::date", "'2019-01-01'::date", None),
        ("DATE '2019-01-01'::timestamp", "DATE '2019-01-01'::timestamp", None),
        ("NULL::date", "NULL::date", None),
        ("CURRENT_DATE::timestamp", "CURRENT_DATE::timestamp", None),
        ("string_to_array(v.a, ',')::int[]", "string_to_array(v.a, ',')::int[]", None),
        ("CAST(x AS int[])", "CAST(x AS int[])", None),
        ("v.a::text", "v.a::text", None),
        ("to_timestamp(v.epoch_secs)", "to_timestamp(v.epoch_secs)", None),
        ("date_part('year', v.d::date)::int", f"date_part('year', {_E}.omop_as_date_at('f', 'mid', v.d))::int", None),
    ],
)
def test_rewrite_casts(expr, counted, uncounted):
    r = rewrite(expr, "f", "mid")
    assert r.sql == counted
    assert r.left_hard == [] and r.bad_templates == []
    assert r.sites == (("f",) if r.changed else ())
    u = rewrite(expr, "f", "mid", counted=False)
    if uncounted is not None:
        assert u.sql == uncounted
    assert "omop_as_" not in u.sql and u.sites == ()


@pytest.mark.parametrize(
    "expr, hard",
    [
        ("to_date(v.d, v.fmt)", "to_date(v.d, v.fmt)"),  # the template is not a literal
        ("to_date(v.d, 'WW/YYYY')", "to_date(v.d, 'WW/YYYY')"),  # a template the helpers do not read
        ("pg_catalog.to_date(v.d, 'YYYY')", "to_date(v.d, 'YYYY')"),
        ("to_date(v.d, 'DD/MM/YYYY', 1)", "to_date(v.d, 'DD/MM/YYYY', 1)"),
        ("(r).f::date", "f::date"),  # a field of a parenthesised value
    ],
)
def test_conversions_left_as_written_are_reported(expr, hard):
    r = rewrite(expr, "f", "mid")
    assert r.sql == expr and r.left_hard == [hard] and not r.changed and r.sites == ()


def test_unreadable_sql_is_left_as_written_and_never_raises():
    for expr in ("'unterminated", "x::date || 'open", "E'\\", "\"open"):
        r = rewrite(expr, "f", "mid")
        assert (r.sql, r.left_hard, r.sites) == (expr, [expr.strip()], ())
    # whatever the input, rewrite returns (a fuzz over SQL-ish fragments)
    rng = random.Random(3)
    parts = ["v.d", "::date", "::int", "(", ")", "'", "E'", "\\", "$$", "--", "\n", "/*", "*/", ",", "CAST", "AS",
             "to_date", "OVER", "CASE", "END", "[", "]", "::numeric(1,", '"', "{etl}", ".", " "]
    for _ in range(2000):
        expr = "".join(rng.choice(parts) for _ in range(rng.randint(1, 12)))
        r = rewrite(expr, "f", "mid")
        assert isinstance(r.sql, str)


def test_strftime_templates_are_refused():
    r = rewrite("to_date(v.d, '%d/%m/%Y')", "f", "mid")
    assert r.sql == "to_date(v.d, '%d/%m/%Y')" and r.left_hard
    assert r.bad_templates and "strftime" in r.bad_templates[0][1]


def test_template_issue():
    for ok in ("DD/MM/YYYY", "ISO", "YYYY-MM-DD\"T\"HH24:MI:SS", "FMDD/FMMM/YYYY", "DD-Mon-YYYY", "DD Month YYYY",
               "DD/MM/YY HH12:MI AM", "YYYY-MM-DD HH24:MI:SS.US TZH:TZM", "YYYYMMDD"):
        assert template_issue(ok) is None, ok
    assert "strftime" in template_issue("%Y-%m-%d")
    assert "at 'WW" in template_issue("WW/YYYY")
    assert "unterminated" in template_issue('YYYY"T')
    assert "must come last" in template_issue("TZH YYYY-MM-DD")


def test_tolerant_value_converts_once():
    def tv(expr, datatype, policy="first"):
        return tolerant_value(expr, datatype, "f", policy).sql

    # an expression that is already the conversion is not wrapped again
    assert tv("v.visit_date::date", "date") == f"{_E}.omop_as_date_at('f', 'first', v.visit_date)"
    assert tv("to_date(i.dob, 'DD/MM/YYYY')", "date") == f"{_E}.omop_as_date_at('f', 'first', i.dob, 'DD/MM/YYYY')"
    assert tv("p.qty::numeric", "float") == f"{_E}.omop_as_number_at('f', 'first', p.qty)"
    # any other value is converted to the CDM type (its own type picks the overload)
    assert tv("v.visit_date", "date") == f"{_E}.omop_as_date_at('f', 'first', (v.visit_date))"
    assert tv("CASE i.sex WHEN '1' THEN 8507 END", "integer") == (
        f"{_E}.omop_as_integer_at('f', 'first', (CASE i.sex WHEN '1' THEN 8507 END))")
    assert tv("v.x -- note", "integer") == f"{_E}.omop_as_integer_at('f', 'first', (v.x -- note\n))"
    assert tv("v.d::date", "datetime") == (
        f"{_E}.omop_as_timestamp_at('f', 'first', ({_E}.omop_as_date_at('f', 'first', v.d)))")
    # a text field: only the conversions inside it
    r = tolerant_value("v.d::date::text", "varchar(50)", "f", "first")
    assert r.sql == f"{_E}.omop_as_date_at('f', 'first', v.d)::text" and r.sites == ("f",)
    r = tolerant_value("i.individual_id", "varchar(50)", "f", "first")
    assert (r.sql, r.sites) == ("i.individual_id", ())
    assert tolerant_value("v.visit_date", "date", "f", "first", counted=False).sql == (
        f"{_E}.try_date((v.visit_date), 'ISO', 'first')")
    assert [kind_of(t) for t in ("integer", "bigint", "float", "date", "datetime", "varchar(50)", "varchar(max)")] == [
        "integer", "bigint", "number", "date", "timestamp", None, None]


def test_expression_operands_are_read_once():
    """A date conversion of an operand PostgreSQL would not inline the reader for (a volatile or
    costly value) is the *_once form; a column, a few cheap calls or a date value is not."""
    r = rewrite("to_date(lpad(v.d::int::text, 8, '0'), 'DDMMYYYY')", "f", "mid")
    assert r.sql == (f"{_E}.omop_as_date_once_at('f', 'mid', lpad({_E}.omop_as_integer_at('f', 'mid', v.d)::text, "
                     "8, '0'), 'DDMMYYYY')")
    assert rewrite("to_date(lpad(v.d::int::text, 8, '0'), 'DDMMYYYY')", "f", "mid", counted=False).sql == (
        f"{_E}.omop_try_date_once(lpad({_E}.try_integer(v.d)::text, 8, '0'), 'DDMMYYYY', 'mid')")
    assert tolerant_value("CASE WHEN v.type::int = 2 THEN v.discharge_date END", "date", "f", "last").sql == (
        f"{_E}.omop_as_date_once_at('f', 'last', (CASE WHEN {_E}.omop_as_integer_at('f', 'last', v.type) = 2 "
        "THEN v.discharge_date END))")
    assert tolerant_value("random()::text", "date", "f", "mid").sql.startswith(f"{_E}.omop_as_date_once_at(")
    expensive = "replace(replace(replace(replace(replace(replace(replace(v.d, 'a', ''), 'b', ''), 'c', ''), 'd', ''), " \
                "'e', ''), 'f', ''), 'g', '')::date"
    assert rewrite(expensive, "f", "mid").sql.startswith(f"{_E}.omop_as_date_once_at(")
    for cheap in ("COALESCE(v.a, v.b)::date", "to_date(left(v.d, 10), 'DD/MM/YYYY')", "NULLIF(v.d, '')::date",
                  "(v.a || '-01')::date"):
        assert "_once" not in rewrite(cheap, "f", "mid").sql, cheap
    # a date value converts through its identity overload, which is always inlined
    assert "_once" not in tolerant_value("to_date(i.dob, 'DD/MM/YYYY')", "datetime", "f", "mid").sql
    assert "_once" not in tolerant_value("COALESCE(v.b::date, v.a::date)", "date", "f", "mid").sql
    # numbers have no template to read: their text overload is cheap whatever the operand
    assert rewrite("(v.a || v.b)::numeric", "f", "mid").sql == f"{_E}.omop_as_number_at('f', 'mid', (v.a || v.b))"


def test_try_functions_are_counted_in_fields():
    """{etl}.try_* in a field or the WHERE is counted like a cast, with the field's policy."""
    def rw(expr, counted=True):
        return rewrite(expr, "f", "last", counted=counted)

    assert rw("{etl}.try_date(r.test_date)").sql == f"{_E}.omop_as_date_at('f', 'last', r.test_date)"
    r = rw("{etl}.try_date(x.dod, 'DD/MM/YYYY', 'first')")
    assert r.sql == f"{_E}.omop_as_date_at('f', 'last', x.dod, 'DD/MM/YYYY')" and r.sites == ("f",)
    assert r.conversions[0].function == "try_date" and r.conversions[0].template == "DD/MM/YYYY"
    assert rw("{etl}.try_number(r.v) * 2").sql == f"{_E}.omop_as_number_at('f', 'last', r.v) * 2"
    # the count queries keep the uncounted function, with the field's policy
    assert rw("{etl}.try_date(x.dod, 'DD/MM/YYYY', 'first')", False).sql == (
        f"{_E}.try_date(x.dod, 'DD/MM/YYYY', 'last')")
    assert rw("{etl}.try_number(r.v)", False).sql == f"{_E}.try_number(r.v)"
    # a template that is not a literal is read as written
    assert rw("{etl}.try_date(x.dod, x.fmt)").sql == "{etl}.try_date(x.dod, x.fmt)"
    assert tolerant_value("{etl}.try_date(r.test_date)", "date", "f", "mid").sql == (
        f"{_E}.omop_as_date_at('f', 'mid', r.test_date)")


def test_a_conversion_left_as_written_is_reported_once():
    r = rewrite("to_date(v.x::int::text, v.fmt)", "f", "mid")
    assert r.sql == f"to_date({_E}.omop_as_integer_at('f', 'mid', v.x)::text, v.fmt)"
    assert r.left_hard == ["to_date(v.x::int::text, v.fmt)"]
    r = rewrite("COALESCE(to_date(v.a, v.fmt), to_date(v.b, v.fmt))::timestamp", "f", "mid")
    assert r.sql.startswith(f"{_E}.omop_as_timestamp_at('f', 'mid', COALESCE(to_date(v.a, v.fmt)")
    assert r.left_hard == ["to_date(v.a, v.fmt)", "to_date(v.b, v.fmt)"]


def test_without_conversions():
    assert without_conversions(f"{_E}.omop_as_date_at('f', 'mid', i.dob, 'DD/MM/YYYY') + v.x").strip() == "+ v.x"
    assert without_conversions(f"{_E}.omop_as_date_once_at('f', 'mid', v.a || v.b) + v.x").strip() == "+ v.x"
    assert without_conversions(f"COALESCE({_E}.try_date(v.a, 'ISO', 'mid'), v.b)") == "COALESCE( , v.b)"
    assert without_conversions("v.a::date") == "v.a::date"


def test_lexer():
    kinds = [(t.kind, t.text) for t in lex("E'a\\'b' || 'c\\' || $x$ ' $x$ -- c\n\"q\"\"r\"::date") if t.kind != "ws"]
    assert kinds == [("estr", "E'a\\'b'"), ("op", "||"), ("str", "'c\\'"), ("op", "||"), ("dollar", "$x$ ' $x$"),
                     ("comment", "-- c"), ("qid", '"q""r"'), ("cast", "::"), ("id", "date")]


def test_field_policies():
    # THEMIS birth and death dates
    assert [partial_policy("person", f) for f in ("year_of_birth", "month_of_birth", "day_of_birth", "birth_datetime")] \
        == ["year", "month", "reject", "birth"]
    assert partial_policy("death", "death_date") == partial_policy("death", "death_datetime") == "last"
    # the widest interval for the DQD start / end pairs, the 15th for other dates
    assert partial_policy("visit_occurrence", "visit_start_date") == "first"
    assert partial_policy("visit_occurrence", "visit_end_datetime") == "last"
    assert partial_policy("procedure_occurrence", "procedure_date") == "first"
    assert partial_policy("procedure_occurrence", "procedure_end_date") == "last"
    assert partial_policy("observation_period", "observation_period_start_date") == "first"
    assert partial_policy("observation_period", "observation_period_end_date") == "last"
    assert partial_policy("measurement", "measurement_date") == "mid"
    assert partial_policy("visit_occurrence", "where") == "mid"
    # a number computed from a partial date would be a guess
    assert partial_policy("drug_exposure", "days_supply") == "reject"
    assert partial_policy("not_a_table", "x") == "reject"
    # partial_dates: reject leaves only the THEMIS rules
    rej = EtlConfig(partial_dates="reject")
    assert partial_policy("visit_occurrence", "visit_start_date", rej) == "reject"
    assert partial_policy("measurement", "measurement_date", rej) == "reject"
    assert partial_policy("person", "birth_datetime", rej) == "birth"
    pairs = start_end_pairs()
    assert pairs[("visit_occurrence", "visit_start_date")] == "visit_end_date"
    assert ("cdm_source", "source_release_date") not in pairs and ("concept", "valid_start_date") not in pairs


# --------------------------------------------------------------------------------------
# the helper functions (PostgreSQL)
# --------------------------------------------------------------------------------------


@pytest.fixture
def funcs(pg):
    """(conn, etl schema) with the helper functions installed and the build's DateStyle."""
    conn, schemas = pg
    create_etl_tables(conn, schemas)
    conn.commit()
    execute(conn, "SET datestyle = 'ISO, YMD'")
    execute(conn, "SET timezone = 'Africa/Nairobi'")
    return conn, schemas.etl


# (function, value, template, policy, expected); the site time zone is Africa/Nairobi
PARSE_CASES = [
    ("date", "2019-03-05", "ISO", "reject", "2019-03-05"), ("date", "2019-3-5", "ISO", "reject", "2019-03-05"),
    ("date", "2019-02-30", "ISO", "reject", None), ("date", "2020-02-29", "ISO", "reject", "2020-02-29"),
    ("date", "2019-02-29", "ISO", "reject", None), ("date", "NA", "ISO", "reject", None),
    ("date", "today", "ISO", "reject", None), ("date", "2019-03-05abc", "ISO", "reject", None),
    ("date", " 2019-03-05 ", "ISO", "reject", "2019-03-05"), ("date", "2019-03-05 10:30", "ISO", "reject", "2019-03-05"),
    ("date", "2020-01-01T23:30:00Z", "ISO", "reject", "2020-01-02"),
    ("date", "1975-06-00", "ISO", "mid", "1975-06-15"), ("date", "1975-00-00", "ISO", "last", "1975-12-31"),
    ("date", "1975-06-00", "ISO", "reject", None),
    ("date", "05/03/2019", "DD/MM/YYYY", "reject", "2019-03-05"), ("date", "5/3/2019", "DD/MM/YYYY", "reject", "2019-03-05"),
    ("date", "31/02/2019", "DD/MM/YYYY", "reject", None), ("date", "00/06/1975", "DD/MM/YYYY", "mid", "1975-06-15"),
    ("date", "00/06/1975", "DD/MM/YYYY", "first", "1975-06-01"), ("date", "00/06/1975", "DD/MM/YYYY", "last", "1975-06-30"),
    ("date", "00/02/2020", "DD/MM/YYYY", "last", "2020-02-29"), ("date", "99/99/1975", "DD/MM/YYYY", "last", "1975-12-31"),
    # THEMIS: a year alone gives no birth date; a month without a day the 1st
    ("date", "00/00/1975", "DD/MM/YYYY", "birth", None), ("date", "00/06/1975", "DD/MM/YYYY", "birth", "1975-06-01"),
    ("date", "00/06/1975", "DD/MM/YYYY", "month", "1975-06-01"), ("date", "00/00/1975", "DD/MM/YYYY", "month", None),
    ("date", "00/00/1975", "DD/MM/YYYY", "year", "1975-01-01"),
    ("date", "15/00/1975", "DD/MM/YYYY", "mid", "1975-06-15"), ("date", "00/00/0000", "DD/MM/YYYY", "mid", None),
    ("date", "99/99/9999", "DD/MM/YYYY", "mid", None), ("date", "31/12/9999", "DD/MM/YYYY", "reject", "9999-12-31"),
    ("date", "9999-12-31", "ISO", "reject", "9999-12-31"), ("date", "01/06/75", "DD/MM/YYYY", "reject", None),
    ("date", "01/06/1975abc", "DD/MM/YYYY", "reject", None), ("date", "1975-06-01", "DD/MM/YYYY", "reject", "1975-06-01"),
    ("date", "15/06/19750", "DD/MM/YYYY", "reject", None), ("date", "32/01/2019", "DD/MM/YYYY", "reject", None),
    ("date", "01/13/2019", "DD/MM/YYYY", "reject", None), ("date", "03/04/2019", "MM/DD/YYYY", "reject", "2019-03-04"),
    ("date", "2019.03.04", "YYYY.MM.DD", "reject", "2019-03-04"), ("date", "05/03/49", "DD/MM/YY", "reject", "1949-03-05"),
    ("date", "05/03/19", "DD/MM/YY", "reject", "2019-03-05"), ("date", "05-Mar-2019", "DD-Mon-YYYY", "reject", "2019-03-05"),
    ("date", "05-MAR-2019", "DD-Mon-YYYY", "reject", "2019-03-05"),
    ("date", "05-March-2019", "DD-Month-YYYY", "reject", "2019-03-05"), ("date", "05-Marx-2019", "DD-Mon-YYYY", "reject", None),
    ("date", "05mar2019", "DDMonYYYY", "reject", "2019-03-05"), ("date", "05MAR2019", "DDMONYYYY", "reject", "2019-03-05"),
    ("date", "06/1975", "MM/YYYY", "mid", "1975-06-15"), ("date", "06/1975", "MM/YYYY", "birth", "1975-06-01"),
    ("date", "1975", "YYYY", "mid", "1975-06-15"), ("date", "1975", "YYYY", "reject", None),
    ("date", "20190305", "YYYYMMDD", "reject", "2019-03-05"), ("date", "20190230", "YYYYMMDD", "reject", None),
    ("date", "2019035", "YYYYMMDD", "reject", None),
    ("ts", "05/03/2019 14:30", "DD/MM/YYYY HH24:MI", "reject", "2019-03-05 14:30:00"),
    ("ts", "05/03/2019 24:30", "DD/MM/YYYY HH24:MI", "reject", None),
    ("ts", "05/03/2019 02:30 PM", "DD/MM/YYYY HH12:MI AM", "reject", "2019-03-05 14:30:00"),
    ("ts", "2020-01-01T23:30:00Z", "ISO", "reject", "2020-01-02 02:30:00"),
    ("ts", "2020-01-01T23:30:00+03:00", "ISO", "reject", "2020-01-01 23:30:00"),
    ("ts", "2020-01-01T23:30:00.123+0300", "ISO", "reject", "2020-01-01 23:30:00.123"),
    ("ts", "2020-01-01 23:30", "ISO", "reject", "2020-01-01 23:30:00"),
    ("ts", "2020-01-01 23:30:59", "ISO", "reject", "2020-01-01 23:30:59"),
    ("ts", "2020-01-01 23:60:00", "ISO", "reject", None), ("ts", "2020-01-01", "ISO", "reject", "2020-01-01 00:00:00"),
    ("ts", "2019-03-05T14:30:00", 'YYYY-MM-DD"T"HH24:MI:SS', "reject", "2019-03-05 14:30:00"),
    ("ts", "15/03/2019 10:30 -03:30", "DD/MM/YYYY HH24:MI TZH:TZM", "reject", "2019-03-15 17:00:00"),
    ("ts", "150320191030", "DDMMYYYYHH24MI", "reject", "2019-03-15 10:30:00"),
    ("num", "1.5", None, None, "1.5"), ("num", "12", None, None, "12"), ("num", "-3", None, None, "-3"),
    ("num", "1e3", None, None, "1000"), ("num", " 12 ", None, None, "12"), ("num", "NaN", None, None, None),
    ("num", "Infinity", None, None, None), ("num", "0x1F", None, None, None), ("num", "1_000", None, None, None),
    ("num", "1,5", None, None, None), ("num", ".", None, None, None), ("num", "-", None, None, None),
    ("num", ".5", None, None, "0.5"),
    ("int", "12", None, None, "12"), ("int", "12.0", None, None, "12"), ("int", "12.5", None, None, None),
    ("int", "99999999999", None, None, None), ("int", "-5", None, None, "-5"), ("int", "POS", None, None, None),
    # beyond the dates design's cases: a trailing time under a date template, a date alone under a
    # time template, the ISO partial forms, year-first and month-name forms under ISO, words
    ("date", "15/06/2019 10:30", "DD/MM/YYYY", "reject", "2019-06-15"),
    ("ts", "15/06/2019 10:30", "DD/MM/YYYY", "reject", "2019-06-15 10:30:00"),
    ("ts", "15/06/2019", "DD/MM/YYYY HH24:MI", "reject", "2019-06-15 00:00:00"),
    ("date", "1975-06", "ISO", "first", "1975-06-01"), ("date", "1975-06", "ISO", "reject", None),
    ("date", "1975", "ISO", "last", "1975-12-31"),
    ("date", "19750601", "ISO", "reject", "1975-06-01"), ("date", "2019/03/05", "ISO", "reject", "2019-03-05"),
    ("date", "5 Mar 2019", "ISO", "reject", "2019-03-05"), ("date", "March 5, 2019", "ISO", "reject", "2019-03-05"),
    ("date", "2019-Mar-05", "ISO", "reject", "2019-03-05"),
    ("date", "05-Mar-19", "ISO", "reject", None),  # a two-digit year: ambiguous under ISO
    ("date", "05/03/2019", "ISO", "reject", None),  # day-first: not read under ISO, YMD
    ("date", "epoch", "ISO", "reject", None), ("date", "now", "ISO", "reject", None),
    ("date", "infinity", "ISO", "reject", None), ("date", "J2451187", "ISO", "reject", None),
    ("date", "2019-03-05", "YYYY-MM-DD", "reject", "2019-03-05"),
    ("ts", "2019-03-05 10:11:12", "YYYY-MM-DD HH24:MI:SS", "reject", "2019-03-05 10:11:12"),
    ("date", "3/4/2019", "MM/DD/YYYY", "reject", "2019-03-04"),
    # a value without its day under a template with one: a partial date ('06/1975'); not for a
    # template whose numbers touch (six digits are no DDMMYYYY)
    ("date", "06/1975", "DD/MM/YYYY", "mid", "1975-06-15"), ("date", "6/1975", "DD/MM/YYYY", "birth", "1975-06-01"),
    ("date", "06/1975", "DD/MM/YYYY", "reject", None), ("date", "13/1975", "DD/MM/YYYY", "mid", None),
    ("date", "06/1975", "MM/DD/YYYY", "first", "1975-06-01"), ("date", "Jun-1975", "DD-Mon-YYYY", "last", "1975-06-30"),
    ("date", "061975", "DDMMYYYY", "mid", None),
    # a separator matches any one separator, as in to_date
    ("date", "05-03-2019", "DD/MM/YYYY", "reject", "2019-03-05"), ("date", "05.03.2019", "DD/MM/YYYY", "reject", "2019-03-05"),
    ("date", "2019/03/05", "YYYY-MM-DD", "reject", "2019-03-05"),
    ("ts", "05/03/2019 10.30", "DD/MM/YYYY HH24:MI", "reject", "2019-03-05 10:30:00"),
    # ISO with fractional seconds or an offset (the SQL fast path), and malformed ISO values
    ("ts", "2020-01-01T23:30:00.5", "ISO", "reject", "2020-01-01 23:30:00.5"),
    ("ts", "2020-01-01 23:30Z", "ISO", "reject", "2020-01-02 02:30:00"),
    ("ts", "2020-01-01T23:30:00+03", "ISO", "reject", "2020-01-01 23:30:00"),
    ("ts", "2020-02-30T23:30:00Z", "ISO", "reject", None), ("ts", "2020-01-01T23:30:00+15:00", "ISO", "reject", None),
    ("ts", "2019-01-1T123:30", "ISO", "reject", None), ("ts", "2019-01-05 10:3:", "ISO", "reject", None),
    ("ts", "2019-01-1012 :30", "ISO", "reject", None), ("ts", "2019-01-051 0:30", "ISO", "reject", None),
    ("ts", "2019-01-305 0:30", "ISO", "reject", None), ("ts", "2019-01-5 110:30", "ISO", "reject", None),
    ("ts", "2019-01-05T10:0:", "ISO", "reject", None), ("ts", "2019-01-0T190:30:59", "ISO", "reject", None),
]


def _try(fn: str, etl: str) -> str:
    return {"date": f"{etl}.try_date(%s, %s, %s)::text", "ts": f"{etl}.try_timestamp(%s, %s, %s)::text",
            "num": f"{etl}.try_number(%s::text)::text", "int": f"{etl}.try_integer(%s::text)::text"}[fn]


@pytest.mark.postgres
def test_parse_functions_cases(funcs):
    conn, etl = funcs
    execute(conn, "SELECT set_config('omop_agent.timezone', 'Africa/Nairobi', false)")
    failures = []
    for fn, x, fmt, pol, expected in PARSE_CASES:
        params = (x, fmt, pol) if fn in ("date", "ts") else (x,)
        got = scalar(conn, "SELECT " + _try(fn, etl), params)
        if got != expected:
            failures.append((fn, x, fmt, pol, expected, got))
        # the counted front end gives the same value
        if fn in ("date", "ts"):
            front = f"omop_as_{'date' if fn == 'date' else 'timestamp'}_at"
            counted = scalar(conn, f"SELECT {etl}.{front}('f', %s, %s, %s)::text", (pol, x, fmt))
            if counted != expected:
                failures.append(("counted", fn, x, fmt, pol, expected, counted))
    assert failures == []
    conn.rollback()


@pytest.mark.postgres
def test_offsets_follow_the_site_time_zone(funcs):
    conn, etl = funcs
    values = ["2020-01-01T23:30:00Z", "2020-01-01T23:30:00+03:00", "2020-01-01 10:00"]

    def read(fn: str) -> list:
        return [scalar(conn, f"SELECT {etl}.{fn}(%s)::text", (v,)) for v in values]

    # unset: the wall clock as written
    execute(conn, "SELECT set_config('omop_agent.timezone', '', false)")
    assert read("try_timestamp") == ["2020-01-01 23:30:00", "2020-01-01 23:30:00", "2020-01-01 10:00:00"]
    assert read("try_date") == ["2020-01-01", "2020-01-01", "2020-01-01"]
    # set: converted to the zone
    execute(conn, "SELECT set_config('omop_agent.timezone', 'Africa/Nairobi', false)")
    assert read("try_timestamp") == ["2020-01-02 02:30:00", "2020-01-01 23:30:00", "2020-01-01 10:00:00"]
    assert read("try_date") == ["2020-01-02", "2020-01-01", "2020-01-01"]
    # a timestamptz keeps its instant; without an offset it is in the session zone
    execute(conn, "SET timezone = 'UTC'")
    assert read("try_timestamptz") == ["2020-01-01 23:30:00+00", "2020-01-01 20:30:00+00", "2020-01-01 10:00:00+00"]

    # to_timestamp honours an offset only with a TZH / OF template; with another template the
    # offset is handled as for a timestamp (ignored when no site zone is set, as PostgreSQL does)
    def tstz(x: str, fmt: str) -> str:
        return scalar(conn, f"SELECT {etl}.try_timestamptz(%s, %s)::text", (x, fmt))

    execute(conn, "SELECT set_config('omop_agent.timezone', '', false)")
    assert tstz("05/03/2019 10:30:00+03:00", "DD/MM/YYYY HH24:MI:SS") == "2019-03-05 10:30:00+00"
    assert tstz("2019-03-05T01:30:00.000+03:00", 'YYYY-MM-DD"T"HH24:MI:SS') == "2019-03-05 01:30:00+00"
    assert tstz("05/03/2019 10:30 +03:00", "DD/MM/YYYY HH24:MI TZH:TZM") == "2019-03-05 07:30:00+00"
    assert tstz("2019-03-05T01:30:00.000+03:00", "ISO") == "2019-03-04 22:30:00+00"  # ::timestamptz
    execute(conn, "SELECT set_config('omop_agent.timezone', 'Africa/Nairobi', false)")
    execute(conn, "SET timezone = 'Africa/Nairobi'")
    assert tstz("05/03/2019 10:30:00Z", "DD/MM/YYYY HH24:MI:SS") == "2019-03-05 13:30:00+03"
    conn.rollback()


@pytest.mark.postgres
def test_counters(funcs):
    conn, etl = funcs

    def counters(site: str) -> dict[str, int]:
        cats = ("invalid", "imputed", "partial_null", "offset_ignored", "offset_converted")
        rows = fetch_all(conn, "SELECT c, current_setting('omop_agent.' || c || '__' || %s, true) AS v "
                               "FROM unnest(%s::text[]) c", (site, list(cats)))
        return {r["c"]: int(r["v"]) for r in rows if r["v"]}

    execute(conn, "SELECT set_config('omop_agent.timezone', '', true)")
    fetch_all(conn, f"SELECT {etl}.omop_as_date_at('f1', 'mid', x, 'DD/MM/YYYY') FROM unnest(%s::text[]) x",
              (["05/03/2019", "00/06/1975", "NA", "31/02/2019", None, "  "],))
    fetch_all(conn, f"SELECT {etl}.omop_as_date_at('f2', 'reject', x, 'DD/MM/YYYY') FROM unnest(%s::text[]) x",
              (["00/06/1975", "00/00/1975"],))
    fetch_all(conn, f"SELECT {etl}.omop_as_integer_at('f3', 'reject', x) FROM unnest(%s::text[]) x",
              (["1", "x", "", " ", None, "99999999999"],))
    # unset: an offset that is not the session zone's (+03:00 here) is ignored and counted
    fetch_all(conn, f"SELECT {etl}.omop_as_timestamp_at('f4', 'reject', x) FROM unnest(%s::text[]) x",
              (["2020-01-01T23:30:00Z", "2020-01-01T23:30:00+03:00", "2020-01-01 23:30", "01/01/2020 23:30Z"],))
    fetch_all(conn, f"SELECT {etl}.omop_as_number_at('f5', 'reject', x) FROM unnest(%s::numeric[]) x",
              (["1.5", "NaN", None],))
    fetch_all(conn, f"SELECT {etl}.omop_as_integer_at('f6', 'reject', x) FROM unnest(%s::numeric[]) x",
              (["1.4", "2.6", "NaN", "1e20", None],))
    # blank and NULL values are missing, not invalid; a month or year kept is not imputed
    assert counters("f1") == {"invalid": 2, "imputed": 1}
    assert counters("f2") == {"partial_null": 2}
    assert counters("f3") == {"invalid": 2}
    assert counters("f4") == {"offset_ignored": 1, "invalid": 1}
    assert counters("f5") == {"invalid": 1} and counters("f6") == {"invalid": 2}
    fetch_all(conn, f"SELECT {etl}.omop_as_date_at('f7', 'year', '00/00/1975', 'DD/MM/YYYY')")
    assert counters("f7") == {}
    execute(conn, "SELECT set_config('omop_agent.timezone', 'Africa/Nairobi', true)")
    # set: an offset that is not the zone's own is converted and counted; the zone's own changes nothing
    fetch_all(conn, f"SELECT {etl}.omop_as_timestamp_at('f8', 'reject', x) FROM unnest(%s::text[]) x",
              (["2020-01-01T23:30:00Z", "2020-01-01 23:30", "2020-01-01T23:30:00.250+03:00"],))
    assert counters("f8") == {"offset_converted": 1}
    fetch_all(conn, f"SELECT {etl}.omop_as_timestamp_at('f9', 'reject', x, 'DD/MM/YYYY HH24:MI') "
                    "FROM unnest(%s::text[]) x", (["01/01/2020 23:30Z", "01/01/2020 23:30 +03:00"],))
    assert counters("f9") == {"offset_converted": 1}
    # the uncounted functions count nothing
    fetch_all(conn, f"SELECT {etl}.try_date(x, 'DD/MM/YYYY', 'mid') FROM unnest(%s::text[]) x", (["NA", "00/06/1975"],))
    assert counters("") == {}
    conn.rollback()


@pytest.mark.postgres
def test_imputation_cap(funcs):
    conn, etl = funcs

    def d(x: str, pol: str) -> str:
        return scalar(conn, f"SELECT {etl}.try_date(%s, 'DD/MM/YYYY', %s)::text", (x, pol))

    execute(conn, "SELECT set_config('omop_agent.imputation_cap', '2023-12-31', true)")
    assert d("00/00/2023", "last") == "2023-12-31" and d("00/12/2023", "mid") == "2023-12-15"
    # the known part of the date wins over the cap: no date before the year it is in
    assert d("00/00/2024", "last") == "2024-01-01" and d("00/00/2022", "last") == "2022-12-31"
    execute(conn, "SELECT set_config('omop_agent.imputation_cap', '2023-06-10', true)")
    assert d("00/06/2023", "last") == "2023-06-10" and d("00/06/2023", "mid") == "2023-06-10"
    assert d("00/06/2023", "first") == "2023-06-01"
    # without a cap: the current date (in the session's zone, which can be a day off the host's)
    execute(conn, "SELECT set_config('omop_agent.imputation_cap', '', true)")
    today = scalar(conn, "SELECT current_date::text")
    assert d(f"00/00/{today[:4]}", "last") == today
    conn.rollback()


@pytest.mark.postgres
def test_overloads(funcs):
    conn, etl = funcs
    row = fetch_all(conn, f"""SELECT
        {etl}.omop_as_date_at('s', 'mid', '2019-01-01') AS unknown_literal,
        {etl}.omop_as_integer_at('s', 'mid', EXTRACT(YEAR FROM date '2019-01-01')) AS from_numeric,
        {etl}.omop_as_date_at('s', 'mid', timestamptz '2020-01-01 23:30+00') AS from_timestamptz,
        pg_typeof({etl}.omop_as_timestamp_at('s', 'mid', {etl}.omop_as_date_at('s', 'mid', '05/03/2019', 'DD/MM/YYYY')))::text
          AS ts_of_date,
        {etl}.omop_as_integer_at('s', 'mid', ('Y' = 'Y')) AS from_boolean,
        {etl}.omop_as_number_at('s', 'mid', 5) AS from_integer,
        {etl}.omop_as_date_at('s', 'mid', '2019-03-05'::varchar) AS from_varchar,
        {etl}.omop_as_integer_at('s', 'mid', 5::smallint) AS from_smallint,
        {etl}.omop_as_bigint_at('s', 'mid', 5) AS bigint_from_integer,
        pg_typeof({etl}.omop_as_timestamptz_at('s', 'mid', '2020-01-01'))::text AS tstz,
        {etl}.try_date(date '2019-01-02') AS try_from_date, {etl}.try_integer(3::bigint) AS try_from_bigint,
        {etl}.try_integer(NULL) AS try_null""")[0]
    assert row == {
        "unknown_literal": date(2019, 1, 1), "from_numeric": 2019, "from_timestamptz": date(2020, 1, 2),
        "ts_of_date": "timestamp without time zone", "from_boolean": 1, "from_integer": 5,
        "from_varchar": date(2019, 3, 5), "from_smallint": 5, "bigint_from_integer": 5,
        "tstz": "timestamp with time zone", "try_from_date": date(2019, 1, 2), "try_from_bigint": 3, "try_null": None,
    }
    conn.rollback()


ADVERSARIAL = [
    "2019-03-15T1:30:000", "2019-3-15T10:30:00", "2019-03-15 10:3:00", "201-03-15T10:30:000", "2019-03-1510:30:00",
    "2019-03-15 23:59:59", "2019-03-15T24:00:00", "2019-02-29 10:00", "--- :", "1//2019", "/1/2019", "1/1/",
    "99999999999999999999", "12345678901234567890123456789012345", "1e999999", "-0", "+", "  ", "",
    "2019-03-15T10:30:00+25:00", "2019-03-15T10:30:00+0", "ДД/ММ/ГГГГ", "2019-03-15T10:30:60", "٣/٣/٢٠١٩",
    "9" * 2000, "1e999", "0001-01-01", "9999-12-31 23:59:59.999999", "15/03/2019 10:30 +14:59",
    "15/03/2019 10:30 -15:00", "2019-03-05\r", "2019-03-05 10:00:00 EAT", "J2451187", "allballs", "00/00/9999",
    "1-1-1", "0000-01-01", "20190305", "201903051", "05-xyz-2019", "1503201910", "15032019", "9999-99-99",
    "2019-03-05T10:30:00.1234567890123Z", "00", "0/0/0", "Feb 30 2019", "31 Sept 2019", "99/99/99",
    "12/12/1212 12:12:12.1212121 PM +14", "\x00", "'", "\\", "NaN", "Infinity", "-Infinity",
    # malformed ISO values of the fast path's lengths (16 and 19)
    "2019-01-1T123:30", "2019-01-05 10:3:", "2019-01-1012 :30", "2019-01-051 0:30", "2019-01-305 0:30",
    "2019-01-5 110:30", "2019-01-05 1::30", "2019-01-0T190:30:59", "2019-01-5T030:59", "2019-01-05T10:0:",
]


def _mutations(rng: random.Random, n: int) -> list[str]:
    """Values one or two edits away from well-formed dates and timestamps in common layouts."""
    seeds = ["2019-03-05", "2019-03-05 10:30", "2019-03-05T10:30:59", "2019-03-05T10:30:59.123+03:00",
             "05/03/2019", "05/03/2019 10:30", "05-Mar-2019", "20190305", "05/03/19 02:30 PM", "1975-06", "06/1975"]
    chars = "0123456789-/:. TZ+,aM"
    out = []
    for _ in range(n):
        v = list(rng.choice(seeds))
        for _ in range(rng.randint(1, 2)):
            i = rng.randrange(len(v))
            op = rng.randrange(4)
            if op == 0:
                del v[i]
            elif op == 1:
                v.insert(i, rng.choice(chars))
            elif op == 2:
                v[i] = rng.choice(chars)
            else:  # a character moved: the length stays, the separators shift
                v.insert(rng.randrange(len(v)), v.pop(i))
            if not v:
                break
        out.append("".join(v))
    # and every single character moved in the ISO layouts the SQL fast paths read
    for seed in ("2019-03-05 10:30", "2019-03-05T10:30:59", "2019-03-05T10:30:59Z", "05/03/2019 10:30"):
        for i in range(len(seed)):
            for j in range(len(seed)):
                v = list(seed)
                v.insert(j, v.pop(i))
                out.append("".join(v))
    return out
TEMPLATES = ["ISO", "DD/MM/YYYY", "MM/DD/YYYY", "DD/MM/YY", "DD-Mon-YYYY", "DDMMYYYY", "YYYYMMDD", "MM/YYYY", "YYYY",
             "DD/MM/YYYY HH24:MI", "DD/MM/YYYY HH12:MI AM", "YYYY-MM-DD HH24:MI:SS TZH:TZM", 'YYYY-MM-DD"T"HH24:MI:SS']


@pytest.mark.postgres
def test_parse_functions_never_raise(funcs):
    conn, etl = funcs
    rng = random.Random(11)
    alphabet = "0123456789/-:. TZ+aAmMpP"
    values = ADVERSARIAL + ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24))) for _ in range(500)]
    values = [v.replace("\x00", "") for v in values]  # PostgreSQL text cannot hold NUL
    near = _mutations(rng, 1500)  # near misses of well-formed values: they reach the fast paths' casts
    for tz in ("", "Africa/Nairobi"):
        execute(conn, "SELECT set_config('omop_agent.timezone', %s, false)", (tz,))
        for fmt in TEMPLATES:
            for pol in ("first", "last", "mid", "birth", "month", "year", "reject"):
                # one statement per template and policy over every value: an error would raise here. The
                # template is a literal, as in the compiled SQL (it is then read once, when planning)
                f, p = ql(fmt), ql(pol)
                fetch_all(conn, f"""SELECT {etl}.try_date(x, {f}, {p}), {etl}.try_timestamp(x, {f}, {p}),
                                           {etl}.try_timestamptz(x, {f}, {p}),
                                           {etl}.omop_as_date_at('f', {p}, x, {f}),
                                           {etl}.omop_as_timestamp_at('f', {p}, x, {f}),
                                           {etl}.omop_as_timestamptz_at('f', {p}, x, {f})
                                    FROM unnest(%s::text[]) x""", (values,))
            f = ql(fmt)
            fetch_all(conn, f"""SELECT {etl}.try_timestamp(x, {f}, 'mid'), {etl}.omop_as_timestamptz_at('f', 'mid', x, {f}),
                                       {etl}.omop_try_date_once(x, {f}, 'mid'),
                                       {etl}.omop_as_timestamptz_once_at('f', 'mid', x, {f})
                                FROM unnest(%s::text[]) x""", (near if tz else values,))
        fetch_all(conn, f"""SELECT {etl}.try_integer(x), {etl}.try_bigint(x), {etl}.try_number(x),
                                   {etl}.omop_as_integer_at('f', 'r', x), {etl}.omop_as_bigint_at('f', 'r', x),
                                   {etl}.omop_as_number_at('f', 'r', x)
                            FROM unnest(%s::text[]) x""", (values + near,))
    numbers = ["NaN", "Infinity", "-Infinity", "1e30", "-1e30", "2147483647.5", "-2147483648.5", "0", "1.5"]
    fetch_all(conn, f"""SELECT {etl}.try_integer(x), {etl}.try_bigint(x), {etl}.try_number(x),
                               {etl}.try_integer(x::float8), {etl}.try_number(x::float8),
                               {etl}.omop_as_integer_at('f', 'r', x), {etl}.omop_as_bigint_at('f', 'r', x::float8),
                               {etl}.omop_as_number_at('f', 'r', x::float8)
                        FROM unnest(%s::numeric[]) x""", (numbers,))
    conn.rollback()


@pytest.mark.postgres
def test_readers_are_inlined(funcs):
    """A column's conversion is inlined whole, its template read when the statement is planned; an
    expression's (the *_once form) is one plpgsql call, with the template read when planned too."""
    conn, etl = funcs
    execute(conn, "CREATE TEMP TABLE t (x text, k text)")

    def calls(select: str) -> set[str]:
        plan = "\n".join(r["QUERY PLAN"] for r in fetch_all(conn, f"EXPLAIN VERBOSE SELECT {select} FROM t"))
        return set(re.findall(r"\b((?:omop|try)_[a-z_]+)\(", plan))

    column = calls(f"{etl}.omop_as_date_at('f', 'mid', x, 'DD/MM/YYYY'), {etl}.try_timestamp(left(x, 16), 'ISO', 'mid')")
    assert column <= {"omop_walk_ts_at", "omop_walk_ts", "omop_counted"}, column
    once = calls(f"{etl}.omop_as_date_once_at('f', 'mid', CASE WHEN {etl}.omop_as_integer_at('g', 'mid', k) = 2 "
                 "THEN x END, 'DD/MM/YYYY')")
    assert "omop_read_ts_once_at" in once and not once & {"omop_fast_layout", "omop_fmt_rx", "omop_read_ts_at"}, once
    conn.rollback()


def _formats(d: date, fmt: str) -> str:
    return {
        "DD/MM/YYYY": d.strftime("%d/%m/%Y"), "MM/DD/YYYY": d.strftime("%m/%d/%Y"), "YYYY.MM.DD": d.strftime("%Y.%m.%d"),
        "DD-Mon-YYYY": d.strftime("%d-%b-%Y"), "ISO": d.isoformat(), "DD/MM/YYYY HH24:MI": d.strftime("%d/%m/%Y 13:05"),
    }[fmt]


@pytest.mark.postgres
def test_fast_paths_agree_with_the_parser(funcs):
    conn, etl = funcs
    rng = random.Random(5)
    days = [date(1900, 1, 1) + timedelta(days=rng.randrange(0, 200 * 365)) for _ in range(300)]
    days += [date(2000, 2, 29), date(2019, 12, 31), date(1, 1, 1), date(9999, 12, 31)]
    for fmt in ("DD/MM/YYYY", "MM/DD/YYYY", "YYYY.MM.DD", "DD-Mon-YYYY", "ISO", "DD/MM/YYYY HH24:MI"):
        # unpadded variants too, and their invalid neighbours (day 31 / 30 of short months)
        values = [_formats(d, fmt) for d in days] + [_formats(d, fmt).lstrip("0") for d in days]
        values += ["31/04/2019", "30/02/2019", "04/31/2019", "2019.02.30", "31-Apr-2019", "2019-04-31", "31/04/2019 10:00"]
        lay = scalar(conn, f"SELECT {etl}.omop_fast_layout(%s)", (fmt,))
        assert lay is not None, fmt
        rows = fetch_all(conn, f"""SELECT x, {etl}.omop_fast(x, %s) AS fast, {etl}.omop_walk_ts(x, {etl}.omop_fmt_rx(%s), 'reject') AS slow
                                   FROM unnest(%s::text[]) x""", (lay, fmt, values))
        # whatever the fast path reads, the parser reads the same; the rest goes to the parser
        assert [r for r in rows if r["fast"] is not None and r["fast"] != r["slow"]] == [], fmt
        assert sum(r["fast"] is not None for r in rows) >= len(days), fmt
    conn.rollback()


@pytest.mark.postgres
def test_uncounted_functions_are_parallel_safe(funcs):
    conn, etl = funcs
    rows = fetch_all(conn, """SELECT p.proname, p.proparallel, p.provolatile FROM pg_proc p
                              JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = %s""", (etl,))
    by_name: dict[str, set[str]] = {}
    for r in rows:
        by_name.setdefault(r["proname"], set()).add(r["proparallel"])
    for name in ("try_date", "try_timestamp", "try_timestamptz", "try_integer", "try_bigint", "try_number",
                 "omop_walk", "omop_walk_ts", "omop_walk_tstz", "omop_fast", "omop_fmt_rx", "omop_fast_layout",
                 "omop_read_ts", "omop_read_tstz", "omop_read_ts_once", "omop_read_tstz_once", "omop_iso_local",
                 "omop_try_date_once", "omop_try_timestamp_once", "omop_try_timestamptz_once"):
        assert by_name.get(name) == {"s"}, name
    # and they run in parallel workers (no exception blocks: a worker cannot start a subtransaction)
    execute(conn, f"CREATE TABLE {etl}.pv AS SELECT CASE WHEN g % 3 = 0 THEN '00/06/1975' WHEN g % 3 = 1 THEN 'x' "
                  "ELSE '05/03/2019' END AS v FROM generate_series(1, 30000) g")
    execute(conn, f"ANALYZE {etl}.pv")
    for setting in ("SET LOCAL max_parallel_workers_per_gather = 2", "SET LOCAL parallel_setup_cost = 0",
                    "SET LOCAL parallel_tuple_cost = 0", "SET LOCAL min_parallel_table_scan_size = 0",
                    "SET LOCAL debug_parallel_query = on"):
        execute(conn, setting)
    query = f"SELECT count(*) FROM {etl}.pv WHERE {etl}.try_date(v, 'DD/MM/YYYY', 'mid') IS NOT NULL AND {etl}.try_integer(v) IS NULL"
    plan = "\n".join(r["QUERY PLAN"] for r in fetch_all(conn, "EXPLAIN " + query))
    assert "Gather" in plan
    assert scalar(conn, query) == 20000
    conn.rollback()


def _timed(conn, sql: str) -> tuple[int, float]:
    started = time.time()
    n = scalar(conn, sql)
    return n, time.time() - started


@pytest.mark.postgres
@pytest.mark.slow
def test_slow_template_scale(funcs):
    """1M values per case, as an INSERT reads them (no parallel workers): a non-ISO timestamp template
    (to_timestamp(x, 'DD/MM/YYYY HH24:MI'), a fast layout) in less than 5 s; ODK timestamps (ISO with
    milliseconds and an offset) in less than 10 s; a template only the plpgsql parser reads
    ('DD/MM/YY') and an operand that is an expression (the *_once form) in less than 30 s."""
    conn, etl = funcs
    execute(conn, "SELECT set_config('omop_agent.timezone', 'Africa/Nairobi', true)")
    execute(conn, """CREATE TEMP TABLE big AS SELECT
        to_char(timestamp '2010-01-01' + (g % 3650) * interval '1 day' + (g % 1440) * interval '1 minute',
                'DD/MM/YYYY HH24:MI') AS dt,
        to_char(timestamp '2010-01-01' + (g % 3650) * interval '1 day' + (g % 1440) * interval '1 minute',
                'YYYY-MM-DD"T"HH24:MI:SS.MS') || '+03:00' AS odk,
        to_char(date '2010-01-01' + (g % 3650), 'DD/MM/YY') AS yy, (g % 3)::text AS kind
        FROM generate_series(1, 1000000) g""")
    execute(conn, "SET LOCAL max_parallel_workers_per_gather = 0")  # as in an INSERT
    cases = {
        "DD/MM/YYYY HH24:MI": (f"{etl}.omop_as_timestamptz_at('f', 'mid', dt, 'DD/MM/YYYY HH24:MI')", 5),
        "ODK": (f"{etl}.omop_as_timestamp_at('f', 'mid', odk)", 10),
        "DD/MM/YY": (f"{etl}.omop_as_date_at('f', 'mid', yy, 'DD/MM/YY')", 30),
        # as the compiler writes a date field whose expression has a conversion in it
        "expression": (tolerant_value("CASE WHEN kind::int < 3 THEN left(odk, 10) END", "date", "f", "mid").sql
                       .replace("{etl}", etl), 30),
    }
    for name, (value, limit) in cases.items():
        n, seconds = _timed(conn, f"SELECT count(x) FROM (SELECT {value} x FROM big OFFSET 0) s")
        print(f"\n1M values, {name}: {seconds:.2f} s")
        assert n == 1_000_000 and seconds < limit, (name, seconds)
    conn.rollback()
