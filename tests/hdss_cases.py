"""The named cases of the mini HDSS fixture (``tests/fixtures/hdss/``) and what the build must make
of each. ``tests/test_hdss.py`` builds the fixture once with ``tests/fixtures/hdss_etl_spec.yaml``
and the configuration below, and checks every case.

Each source row stands for one case. People are named by their source key: ``H01``..``H30`` in the
HDSS register (``individuals.csv``), ``EMR:P001``.. for clinic patients who are not HDSS residents
(``emr_patients.csv``). Visits are named by their key in the visit map (``1001``.. outpatient,
``IPD:1001``.. inpatient admissions).

``CASES`` maps a case to one or more ``(sql, rows)`` checks: the query, with ``{cdm}`` and ``{etl}``
for the schemas and the views ``pk`` (key, person_id) and ``vk`` (key, visit_occurrence_id) over the
id maps, must return exactly ``rows``.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

# the project settings of the fixture (the rest is set by the test: database, schemas, paths)
CONFIG = {
    "cdm_source": {"source_release_date": "2023-12-31"},
    "etl": {"timezone": "Africa/Nairobi"},
}
RELEASE = date(2023, 12, 31)

D = date


def _person(key: str) -> str:
    return (f"SELECT p.gender_concept_id, p.year_of_birth, p.month_of_birth, p.day_of_birth, p.birth_datetime "
            f"FROM {{cdm}}.person p JOIN pk USING (person_id) WHERE pk.key = '{key}'")


def _periods(key: str) -> str:
    return (f"SELECT op.observation_period_start_date, op.observation_period_end_date "
            f"FROM {{cdm}}.observation_period op JOIN pk USING (person_id) WHERE pk.key = '{key}' "
            "ORDER BY op.observation_period_start_date")


def _visit(key: str) -> str:
    return (f"SELECT pk.key, v.visit_concept_id, v.visit_start_date, v.visit_start_datetime, v.visit_end_date "
            f"FROM {{cdm}}.visit_occurrence v JOIN vk USING (visit_occurrence_id) JOIN pk USING (person_id) "
            f"WHERE vk.key = '{key}'")


def _conditions(source_value: str) -> str:
    """The condition, observation and procedure records of a diagnosis code: where it ended up."""
    return (
        "SELECT t, key, concept_id, d, visit FROM ("
        " SELECT 'condition' AS t, pk.key, c.condition_concept_id AS concept_id, c.condition_start_date AS d,"
        "        vk.key AS visit, c.condition_source_value AS sv"
        " FROM {cdm}.condition_occurrence c JOIN pk USING (person_id)"
        " LEFT JOIN vk ON vk.visit_occurrence_id = c.visit_occurrence_id"
        " UNION ALL"
        " SELECT 'observation', pk.key, o.observation_concept_id, o.observation_date, vk.key, o.observation_source_value"
        " FROM {cdm}.observation o JOIN pk USING (person_id)"
        " LEFT JOIN vk ON vk.visit_occurrence_id = o.visit_occurrence_id"
        " UNION ALL"
        " SELECT 'procedure', pk.key, x.procedure_concept_id, x.procedure_date, vk.key, x.procedure_source_value"
        " FROM {cdm}.procedure_occurrence x JOIN pk USING (person_id)"
        " LEFT JOIN vk ON vk.visit_occurrence_id = x.visit_occurrence_id"
        f") r WHERE sv = '{source_value}' ORDER BY t, key, d"
    )


def _diagnosis(key: str, day: str) -> str:
    """The condition record of one diagnosis (person, date): its concept and visit."""
    return (f"SELECT c.condition_concept_id, vk.key FROM {{cdm}}.condition_occurrence c JOIN pk USING (person_id) "
            f"LEFT JOIN vk ON vk.visit_occurrence_id = c.visit_occurrence_id "
            f"WHERE pk.key = '{key}' AND c.condition_start_date = DATE '{day}'")


def _drug_eras(key: str) -> str:
    return (f"SELECT e.drug_concept_id, e.drug_era_start_date, e.drug_era_end_date, e.drug_exposure_count, e.gap_days "
            f"FROM {{cdm}}.drug_era e JOIN pk USING (person_id) WHERE pk.key = '{key}' ORDER BY 1, 2")


def _death(key: str) -> str:
    return (f"SELECT d.death_date, d.cause_concept_id, d.cause_source_value "
            f"FROM {{cdm}}.death d JOIN pk USING (person_id) WHERE pk.key = '{key}'")


def _gone(key: str) -> str:
    """A person who was not loaded: no person row, the key retired, no record of any kind."""
    tables = ["visit_occurrence", "condition_occurrence", "drug_exposure", "measurement", "observation",
              "observation_period", "death"]
    records = " + ".join(f"(SELECT COUNT(*) FROM {{cdm}}.{t} x WHERE x.person_id = r.person_id)" for t in tables)
    return (f"SELECT (SELECT COUNT(*) FROM {{cdm}}.person WHERE person_source_value = '{key}'), {records} "
            f"FROM {{etl}}.person_map_retired r WHERE r.source_value = '{key}'")


# Concepts of the fixture vocabulary (omop_agent/demo/fixtures/vocab) and of hdss_vocab.sql
MALE, FEMALE = 8507, 8532
OUTPATIENT, INPATIENT = 9202, 9201
MALARIA, PNEUMONIA, HIV, TB, URI = 2000001001, 2000001002, 2000001006, 2000001007, 2000001009
NORMAL_PREGNANCY, PREGNANT, IUD_INSERTION, PREGNANCY_TEST = 2000100001, 2000100002, 2000100003, 2000100004
ARTEMETHER, LUMEFANTRINE, AMOXICILLIN, ACETAMINOPHEN, ARTESUNATE = (
    2000003101, 2000003102, 2000003104, 2000003103, 2000100020)
HEMOGLOBIN, MALARIA_RDT, POSITIVE, NEGATIVE, G_DL = 2000004001, 2000004002, 2000006001, 2000006002, 2000005001

CASES: dict[str, list[tuple[str, list[tuple]]]] = {
    # ---- people: individuals.csv (individual_id, sex, dob DD/MM/YYYY, reg_date) ------------------
    "H01 plain": [(_person("H01"), [(MALE, 1980, 4, 12, datetime(1980, 4, 12))])],
    "H02 exact duplicate row: one person": [
        ("SELECT COUNT(*) FROM {cdm}.person WHERE person_source_value = 'H02'", [(1,)])],
    "H03 re-registered with the sex corrected: the latest registration wins (dedup_order)": [
        (_person("H03"), [(FEMALE, 1992, 8, 15, datetime(1992, 8, 15))])],
    "H04 dob 00/06/1975: year and month kept, no day, birth_datetime on the 1st (THEMIS)": [
        (_person("H04"), [(FEMALE, 1975, 6, None, datetime(1975, 6, 1))])],
    "H05 dob 00/00/1980: the year only": [(_person("H05"), [(FEMALE, 1980, None, None, None)])],
    "H06 dob NA (staged as NULL): the person is dropped, with every record": [(_gone("H06"), [(0, 0)])],
    "H07 dob 31/02/1990 (invalid): the person is dropped": [(_gone("H07"), [(0, 0)])],
    "H08 sex 9 (the CASE has no ELSE): gender 0": [(_person("H08"), [(0, 1985, 10, 20, datetime(1985, 10, 20))])],
    # ---- people: emr_patients.csv (clinic patients) ------------------------------------------------
    "E1-E4 not in the HDSS: persons EMR:<patient_no>, periods from their clinic records": [
        ("SELECT pk.key, p.gender_concept_id, p.year_of_birth, op.observation_period_start_date, "
         "op.observation_period_end_date FROM {cdm}.person p JOIN pk USING (person_id) "
         "JOIN {cdm}.observation_period op USING (person_id) WHERE pk.key LIKE 'EMR:%' ORDER BY 1", [
             ("EMR:P001", FEMALE, 1995, D(2020, 5, 5), D(2020, 5, 9)),  # to the end of a 5-day prescription
             ("EMR:P002", MALE, 1988, D(2020, 6, 6), D(2020, 6, 6)),
             ("EMR:P003", FEMALE, 2001, D(2021, 7, 7), D(2021, 7, 7)),
             ("EMR:P004", MALE, 1979, D(2022, 8, 8), D(2022, 8, 8))])],
    "E5 linked to H01: no new person, its clinic visit is H01's": [
        ("SELECT (SELECT COUNT(*) FROM {etl}.person_map WHERE source_value = 'EMR:P005'), "
         "(SELECT COUNT(*) FROM {cdm}.person)", [(0, 32)]),
        (_visit("1018"), [("H01", OUTPATIENT, D(2022, 2, 2), datetime(2022, 2, 2, 9), D(2022, 2, 2))])],
    # ---- residency_episodes.csv (start_date, end_date DD/MM/YYYY): observation periods -------------
    "H09 back-to-back internal move: one period": [(_periods("H09"), [(D(2010, 1, 1), RELEASE)])],
    "H10 overlapping episodes: one period": [(_periods("H10"), [(D(2010, 1, 1), D(2020, 12, 31))])],
    "H11 60-day out-migration with a clinic visit in the gap: two periods, the visit stays": [
        (_periods("H11"), [(D(2012, 1, 1), D(2018, 3, 31)), (D(2018, 5, 31), RELEASE)]),
        (_visit("1007"), [("H11", OUTPATIENT, D(2018, 4, 15), datetime(2018, 4, 15, 14), D(2018, 4, 15))])],
    "H12 open episode (no exit date): ends at the release date": [(_periods("H12"), [(D(2011, 1, 1), RELEASE)])],
    "H13 open episode and a death: ends at the death (the register's date)": [
        (_periods("H13"), [(D(2005, 1, 1), D(2021, 6, 10))])],
    "H14 inverted junk episode and a valid one: the valid one": [(_periods("H14"), [(D(2016, 1, 1), RELEASE)])],
    "H15 inverted episode only: a period from the records (fill)": [
        (_periods("H15"), [(D(2019, 3, 10), D(2019, 3, 10))])],
    "H16 exit 31/12/9999 (a 'still resident' sentinel): open": [(_periods("H16"), [(D(2011, 1, 1), RELEASE)])],
    "H17 open episode followed by a later closed one (stale open): one period": [
        (_periods("H17"), [(D(2010, 1, 1), RELEASE)])],
    "H18 partial entry 00/03/2015 and exit 00/05/2018: the widest interval": [
        (_periods("H18"), [(D(2015, 3, 1), D(2018, 5, 31))])],
    "H19 no entry date: the episode is dropped, a period from the records (fill)": [
        (_periods("H19"), [(D(2020, 2, 2), D(2020, 2, 2))])],
    "H20 exit after the configured release date: kept as written": [
        (_periods("H20"), [(D(2015, 1, 1), D(2024, 3, 15))])],
    "H23 open episode and a VA death 00/00/2019: ends on 2019-12-31": [
        (_periods("H23"), [(D(2000, 1, 1), D(2019, 12, 31))])],
    "IMG entries: one in-migration observation each, a partial entry date on the 15th": [
        ("SELECT pk.key, o.observation_date FROM {cdm}.observation o JOIN pk USING (person_id) "
         "WHERE o.observation_concept_id = 2000100005 ORDER BY 1", [
             ("H08", D(2016, 1, 1)), ("H10", D(2016, 6, 1)), ("H11", D(2018, 5, 31)), ("H14", D(2016, 1, 1)),
             ("H17", D(2014, 1, 1)), ("H18", D(2015, 3, 15))])],
    # ---- clinic_visits.csv: outpatient visits -----------------------------------------------------
    "1012 no discharge date: the visit ends on its start date": [
        (_visit("1012"), [("H08", OUTPATIENT, D(2020, 9, 9), datetime(2020, 9, 9, 8), D(2020, 9, 9))])],
    "1010 visit date 2019-13-01 (invalid): the visit is dropped, its key retired": [
        ("SELECT (SELECT COUNT(*) FROM vk WHERE key = '1010'), "
         "(SELECT person_source_value FROM {etl}.visit_map_retired WHERE source_value = '1010')", [(0, "H12")])],
    "1011 EMERG visit: left out by the mapping's WHERE, its key retired": [
        ("SELECT (SELECT COUNT(*) FROM vk WHERE key = '1011'), "
         "(SELECT COUNT(*) FROM {etl}.visit_map_retired WHERE source_value = '1011')", [(0, 1)])],
    "1006 visit of a person who was not loaded (H06): dropped": [
        ("SELECT COUNT(*) FROM {etl}.visit_map_retired WHERE source_value = '1006'", [(1,)])],
    "1013 no visit date, submitted 2020-01-01T23:30:00Z: 2020-01-02 02:30 in Africa/Nairobi": [
        (_visit("1013"), [("H09", OUTPATIENT, D(2020, 1, 2), datetime(2020, 1, 2, 2, 30), D(2020, 1, 2))])],
    "1021 submitted with +03:00 (the zone's own offset): as written": [
        (_visit("1021"), [("H10", OUTPATIENT, D(2020, 3, 5), datetime(2020, 3, 5, 10), D(2020, 3, 5))])],
    # ---- ipd_wards.csv: inpatient admissions ----------------------------------------------------
    "A1 admission 1001 in two ward rows: one inpatient visit over both": [
        (_visit("IPD:1001"), [("H02", INPATIENT, D(2021, 2, 1), None, D(2021, 2, 7))])],
    "admission 1001 and outpatient visit 1001 (another person): the IPD: prefix keeps them apart": [
        ("SELECT vk.key, pk.key, v.visit_concept_id FROM {cdm}.visit_occurrence v JOIN vk USING (visit_occurrence_id) "
         "JOIN pk USING (person_id) WHERE vk.key IN ('1001', 'IPD:1001') ORDER BY 1",
         [("1001", "H01", OUTPATIENT), ("IPD:1001", "H02", INPATIENT)])],
    # ---- diagnoses.csv (ICD-10) ---------------------------------------------------------------
    "B54: a malaria condition, linked to its visit": [
        (_conditions("B54"), [
            ("condition", "EMR:P001", MALARIA, D(2020, 5, 5), "1014"),
            ("condition", "H01", MALARIA, D(2021, 3, 1), "1001"),
            ("condition", "H01", MALARIA, D(2021, 3, 20), "1002"),
            ("condition", "H10", MALARIA, D(2019, 9, 9), "IPD:1002"),
            ("condition", "H10", MALARIA, D(2019, 9, 10), None),
            ("condition", "H19", MALARIA, D(2020, 2, 2), "1009")])],
    "Z34.9 (an Observation concept): moved to observation": [
        (_conditions("Z34.9"), [("observation", "H03", NORMAL_PREGNANCY, D(2018, 2, 2), "1005")])],
    "O98.7 (Maps to a condition and an observation): one record in each table": [
        (_conditions("O98.7"), [("condition", "H04", HIV, D(2017, 5, 5), "1029"),
                                ("observation", "H04", PREGNANT, D(2017, 5, 5), "1029")])],
    "Z30.1 (a Procedure concept): moved to procedure_occurrence": [
        (_conditions("Z30.1"), [("procedure", "H05", IUD_INSERTION, D(2016, 6, 6), "1030")])],
    "R99 (not in the vocabulary): stays in condition_occurrence with concept 0": [
        (_conditions("R99"), [("condition", "H08", 0, D(2020, 9, 9), "1012")])],
    "D08 visit key of another person's visit (1001 is H01's): no visit": [(_diagnosis("H09", "2020-01-02"), [(URI, None)])],
    "D09 visit the visit mapping filtered out (EMERG 1011): no visit": [(_diagnosis("H12", "2020-08-08"), [(URI, None)])],
    "D19 visit dropped for its invalid date (1010): no visit": [(_diagnosis("H12", "2019-12-01"), [(URI, None)])],
    "D10 IPD key written ipd:1002 (no visit mapping reads it): no visit": [
        (_diagnosis("H10", "2019-09-10"), [(MALARIA, None)])],
    "D18 IPD:1002: linked to the admission": [(_diagnosis("H10", "2019-09-09"), [(MALARIA, "IPD:1002")])],
    "H01 malaria on 1 and 20 March 2021: one condition era": [
        ("SELECT e.condition_concept_id, e.condition_era_start_date, e.condition_era_end_date, "
         "e.condition_occurrence_count FROM {cdm}.condition_era e JOIN pk USING (person_id) WHERE pk.key = 'H01'",
         [(MALARIA, D(2021, 3, 1), D(2021, 3, 21), 2)])],
    "H12 two URI diagnoses eight months apart: two condition eras": [
        ("SELECT e.condition_era_start_date, e.condition_occurrence_count FROM {cdm}.condition_era e "
         "JOIN pk USING (person_id) WHERE pk.key = 'H12' ORDER BY 1", [(D(2019, 12, 1), 1), (D(2020, 8, 8), 1)])],
    # ---- prescriptions.csv -------------------------------------------------------------------
    "H01 AL on 1 and 20 March, then 40 days after: two eras per ingredient (artemether, lumefantrine)": [
        (_drug_eras("H01"), [
            (ARTEMETHER, D(2021, 3, 1), D(2021, 3, 22), 2, 17), (ARTEMETHER, D(2021, 5, 1), D(2021, 5, 3), 1, 0),
            (LUMEFANTRINE, D(2021, 3, 1), D(2021, 3, 22), 2, 17), (LUMEFANTRINE, D(2021, 5, 1), D(2021, 5, 3), 1, 0)])],
    "H03 measles vaccine (no ingredient): an exposure without an era": [
        ("SELECT d.drug_concept_id, (SELECT COUNT(*) FROM {cdm}.drug_era e WHERE e.person_id = d.person_id) "
         "FROM {cdm}.drug_exposure d JOIN pk USING (person_id) WHERE pk.key = 'H03'", [(2000100030, 0)])],
    "H04 days -3: loaded as written, no era": [
        ("SELECT d.days_supply, d.drug_exposure_end_date, (SELECT COUNT(*) FROM {cdm}.drug_era e "
         "WHERE e.person_id = d.person_id) FROM {cdm}.drug_exposure d JOIN pk USING (person_id) WHERE pk.key = 'H04'",
         [(-3, D(2017, 5, 5), 0)])],
    "H05 days NaN (invalid): days_supply NULL, the exposure ends on its start, one era": [
        ("SELECT d.days_supply, d.drug_exposure_end_date FROM {cdm}.drug_exposure d JOIN pk USING (person_id) "
         "WHERE pk.key = 'H05'", [(None, D(2016, 6, 6))]),
        (_drug_eras("H05"), [(AMOXICILLIN, D(2016, 6, 6), D(2016, 6, 6), 1, 0)])],
    "H08 artesunate (an RxNorm Extension ingredient): an era": [
        (_drug_eras("H08"), [(ARTESUNATE, D(2020, 9, 9), D(2020, 9, 9), 1, 0)])],
    # ---- lab_results.csv ---------------------------------------------------------------------
    "numeric results: value_as_number; POS / NEG: value_as_concept_id": [
        ("SELECT pk.key, m.measurement_concept_id, m.value_as_number, m.value_as_concept_id, m.unit_concept_id "
         "FROM {cdm}.measurement m JOIN pk USING (person_id) ORDER BY 1, 2", [
             ("EMR:P002", HEMOGLOBIN, Decimal("9.8"), None, G_DL),
             ("H01", HEMOGLOBIN, Decimal("11.5"), None, G_DL),
             ("H01", MALARIA_RDT, None, POSITIVE, None),
             ("H02", MALARIA_RDT, None, NEGATIVE, None),
             ("H09", HEMOGLOBIN, Decimal("13.1"), None, G_DL)])],
    "PREG (an Observation concept) with result POS: moved to observation, the result kept": [
        ("SELECT pk.key, o.observation_concept_id, o.value_as_concept_id, o.value_source_value, vk.key "
         "FROM {cdm}.observation o JOIN pk USING (person_id) LEFT JOIN vk USING (visit_occurrence_id) "
         "WHERE o.observation_source_value = 'PREG'", [("H03", PREGNANCY_TEST, POSITIVE, "POS", "1005")])],
    # ---- deaths.csv (register) and va_deaths.csv (verbal autopsies) -----------------------------
    "H13 register death and a VA death two days later: the register's": [
        (_death("H13"), [(D(2021, 6, 10), None, None)])],
    "H21 two VA rows, the verified one earlier: the verified one (dedup_order)": [
        (_death("H21"), [(D(2022, 5, 5), HIV, "HIV")]), (_periods("H21"), [(D(2000, 1, 1), D(2022, 5, 5))])],
    # the verified row has no date: rows without a required value are dropped before one row per key is
    # chosen, so a row that cannot be loaded never wins over one that can
    "H22 a verified VA row without a date and an unverified dated one: the dated one": [
        (_death("H22"), [(D(2021, 3, 3), HIV, "HIV/TB")])],
    "H22 cause HIV/TB mapped to two concepts: one death, the smaller concept": [
        ("SELECT COUNT(*) FROM {cdm}.death d JOIN pk USING (person_id) WHERE pk.key = 'H22'", [(1,)])],
    "H23 VA date 00/00/2019: 2019-12-31 (THEMIS)": [(_death("H23"), [(D(2019, 12, 31), TB, "TB")])],
    "one death per person": [
        ("SELECT pk.key FROM {cdm}.death d JOIN pk USING (person_id) ORDER BY 1",
         [("H13",), ("H21",), ("H22",), ("H23",), ("H29",)])],
}

# what the build's steps report (TableLoadResult), by step label
STEP_COUNTS = {
    # label: (source_rows, id_retained_rows, retained_rows, rows_inserted)
    "person": (32, 32, 30, 28),
    "person_2": (4, 4, 4, 4),
    "visit_occurrence": (29, 28, 27, 27),
    "visit_occurrence_2": (3, 3, 3, 2),
    "condition_occurrence": (19, 18, 18, 19),  # O98.7 maps to two concepts
    "drug_exposure": (10, 9, 9, 9),
    "measurement": (7, 6, 6, 6),
    "observation": (6, 6, 6, 6),
    "death": (3, 2, 2, 2),
    "death_2": (6, 6, 5, 3),
    "observation_period": (35, 33, 32, 32),
}
VALUE_COUNTS = {
    "person": {"defaulted": {"gender_concept_id": 1},
               "invalid": {"year_of_birth": 1, "month_of_birth": 1, "day_of_birth": 1, "birth_datetime": 1},
               "required_null": {"year_of_birth": 2},
               "partial_null": {"month_of_birth": 1, "day_of_birth": 2, "birth_datetime": 1},
               "imputed": {"birth_datetime": 1}, "row": {"dropped": 2}},
    "person_2": {},
    "visit_occurrence": {"invalid": {"visit_start_date": 1}, "required_null": {"visit_start_date": 1},
                         "offset_converted": {"visit_start_date": 1, "visit_start_datetime": 1},
                         "defaulted": {"visit_end_date": 1}, "row": {"dropped": 1}},
    "visit_occurrence_2": {},
    "condition_occurrence": {},
    # days 'NaN': days_supply, and the end date expression that reads it (and falls back)
    "drug_exposure": {"invalid": {"drug_exposure_end_date": 1, "days_supply": 1}},
    "measurement": {},
    "observation": {"imputed": {"observation_date": 1}},
    "death": {},
    "death_2": {"imputed": {"death_date": 1}, "required_null": {"death_date": 1}, "row": {"dropped": 1}},
    "observation_period": {"imputed": {"observation_period_start_date": 1, "observation_period_end_date": 1},
                           "required_null": {"observation_period_start_date": 1}, "row": {"dropped": 1}},
}
KEY_STATS = {  # the one-row-per-key statistics
    "person": {"inserted": 28, "candidate_rows": 30, "source_rows_kept": 30, "keys": 28, "duplicate_rows": 2,
               "already_loaded_keys": 0, "conflict_keys": 1, "undecided_keys": 0, "fan_out_keys": 0,
               "dedup_order": True},
    "visit_occurrence_2": {"inserted": 2, "candidate_rows": 3, "source_rows_kept": 3, "keys": 2, "duplicate_rows": 1,
                           "already_loaded_keys": 0, "conflict_keys": 0, "fan_out_keys": 0, "gap_keys": 0,
                           "dedup_order": False},
    # H13 is in the register; H21's rows disagree; H22's cause has two concepts
    "death_2": {"inserted": 3, "candidate_rows": 6, "source_rows_kept": 5, "keys": 4, "duplicate_rows": 1,
                "already_loaded_keys": 1, "conflict_keys": 2, "undecided_keys": 0, "fan_out_keys": 1,
                "dedup_order": True},
}
VISIT_LINKS = {  # D08 another person's visit; D09, D19 visits the visit mapping left out; D10 an unknown key
    "condition_occurrence": {"with_key": 18, "linked": 14, "other_person": 1, "filtered": 2},
    "drug_exposure": {"with_key": 9, "linked": 9, "other_person": 0, "filtered": 0},
    "measurement": {"with_key": 6, "linked": 6, "other_person": 0, "filtered": 0},
}
MOVED = {  # route step label: (to_table, domain_id, moved_rows)
    "route_condition_occurrence": [("observation", "Observation", 2), ("procedure_occurrence", "Procedure", 1)],
    "route_drug_exposure": [],
    "route_measurement": [("observation", "Observation", 1)],
}
MERGE_STATS = {
    "inserted": 27, "episodes": 32, "dropped_inverted": 2, "open_episodes": 19, "open_closed_at_death": 4,
    "open_starting_after_death": 0, "open_starting_after_release": 0, "exits_after_release": 1,
    "ending_from_release": 16, "persons_overlapping": 2, "persons_stale_open": 1, "persons": 27,
    "persons_without_period": 1, "release_date": "2023-12-31", "configured": True, "closed_episodes": 11,
    "fact_rows": 1,
}
FILL_STATS = {"inserted": 6, "persons_without_period": 6}  # H15, H19 and E1-E4
DERIVED_ROWS = {"condition_era": 13, "drug_era": 8}

# every failed check of the build (key -> severity); none blocks, so the gate passes
FAILED_CHECKS = {
    # the design of the fixture: drops, duplicates and conflicts it holds on purpose
    "etlRecordRetention.person": "convention",  # H06, H07
    "etlRequiredFieldNull.person.year_of_birth": "convention",
    "etlDuplicateRecords.person": "convention",  # H02, H03
    "etlRecordRetention.visit_occurrence": "convention",  # 1006 (H06), 1010
    "etlRecordRetention.condition_occurrence": "convention",  # D11 (H06)
    "etlVisitLinkage.condition_occurrence.visit_occurrence_id": "convention",  # D08, D10
    "etlRecordRetention.drug_exposure": "convention",  # R08 (H06)
    # days 'NaN': the end date's expression falls back to one day, so no end date is NULL (characterization)
    "etlInvalidValue.drug_exposure.drug_exposure_end_date": "characterization",
    "etlInvalidValue.drug_exposure.days_supply": "characterization",
    "etlRecordRetention.measurement": "convention",  # L06 (H06)
    "etlRecordRetention.death": "convention",  # H06
    "etlRecordRetention.death.death_2": "convention",  # VA4 without a date (verified, so it would win)
    "etlRequiredFieldNull.death.death_date.death_2": "convention",
    "etlKeyFanOut.death.death_2": "convention",  # HIV/TB
    "etlRecordRetention.observation_period": "convention",  # H06, H07, H19
    "etlObservationPeriodMerge.observation_period.observation_period_merge": "convention",  # H14, H15 inverted
    "etlSourceReleaseDate.observation_period.observation_period_merge": "convention",  # H20
    # DQD checks
    "measurePersonCompleteness.procedure_occurrence": "convention",  # only the routed Z30.1
    "measureDrugEraCoverage.drug_era": "convention",  # the vaccine and the negative days
    "standardConceptRecordCompleteness.condition_occurrence.condition_concept_id": "convention",  # R99
    "sourceValueCompleteness.condition_occurrence.condition_source_value": "convention",  # R99
    "sourceConceptRecordCompleteness.drug_exposure.drug_source_concept_id": "convention",  # mapped as 0
    "plausibleValueLow.drug_exposure.days_supply": "characterization",  # -3
    "withinObservationPeriod.condition_occurrence.condition_start_date": "characterization",  # H11's gap
    "withinObservationPeriod.condition_era.condition_era_start_date": "characterization",
}
