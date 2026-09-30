import pytest

from omop_agent.db import fetch_all, row_count, scalar
from omop_agent.profiling.loader import sanitize_identifier

pytestmark = pytest.mark.postgres


def test_sanitize_identifier():
    used: set[str] = set()
    assert sanitize_identifier("Patient ID", used) == "patient_id"
    assert sanitize_identifier("patient-id", used) == "patient_id_2"
    assert sanitize_identifier("1st visit") == "c_1st_visit"


def test_staging_and_vocab(staged):
    conn, schemas, cfg = staged
    assert row_count(conn, schemas.src, "individuals") == 400
    # empty strings are loaded as NULL
    n_null = scalar(conn, f"SELECT COUNT(*) FROM {schemas.src}.clinic_visits WHERE discharge_date IS NULL")
    n_opd = scalar(conn, f"SELECT COUNT(*) FROM {schemas.src}.clinic_visits WHERE visit_type <> 'IPD'")
    assert n_null == n_opd
    rows = fetch_all(conn, f"SELECT concept_name, valid_start_date FROM {schemas.vocab}.concept WHERE concept_id = 8507")
    assert rows[0]["concept_name"] == "MALE"
    assert str(rows[0]["valid_start_date"]) == "1970-01-01"
    maps = scalar(
        conn,
        f"""SELECT c2.concept_name FROM {schemas.vocab}.concept c1
            JOIN {schemas.vocab}.concept_relationship r ON r.concept_id_1 = c1.concept_id AND r.relationship_id = 'Maps to'
            JOIN {schemas.vocab}.concept c2 ON c2.concept_id = r.concept_id_2
            WHERE c1.vocabulary_id = 'ICD10' AND c1.concept_code = 'B54'""",
    )
    assert maps == "Malaria"
