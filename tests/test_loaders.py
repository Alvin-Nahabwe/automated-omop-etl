from dataclasses import replace
from pathlib import Path

import pytest
from conftest import make_config

from omop_agent.db import execute, fetch_all, qi, row_count, scalar, table_exists
from omop_agent.profiling.loader import MANIFEST, SourceDataError, sanitize_identifier, stage_csv, stage_source
from omop_agent.workspace import Workspace

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


# --------------------------------------------------------------------------------------
# strict CSV staging
# --------------------------------------------------------------------------------------

RAGGED_CSV = (
    "id,dx_text,dx_date\n"
    "1,Malaria,2020-01-01\n"
    "2,Fever, unspecified,2020-01-02\n"  # line 3: an unquoted comma shifts the date
    '3,"Cough,\npersistent",2020-01-03\n'  # lines 4-5: one quoted record over two lines
    "4,Headache\n"  # line 6: a field is missing
    "\n"  # a blank line is skipped, not counted as ragged
)


def _data_dir(tmp_path: Path, files: dict[str, bytes]) -> Path:
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    for name, content in files.items():
        (d / name).write_bytes(content)
    return d


def test_ragged_rows_stop_staging_without_quoting_the_data(pg, tmp_path):
    conn, schemas = pg
    cfg = make_config(tmp_path, _data_dir(tmp_path, {"diagnoses.csv": RAGGED_CSV.encode()}), schemas)
    with pytest.raises(SourceDataError) as exc:
        stage_source(cfg, conn, schemas)
    msg = str(exc.value)
    assert "diagnoses.csv" in msg and "2 row(s)" in msg and "1 with more, 1 with fewer" in msg
    assert "line(s) 3, 6" in msg and "source.delimiter" in msg and "quot" in msg
    for value in ("Fever", "unspecified", "Headache", "2020-01-02"):
        assert value not in msg
    # the failed staging was rolled back
    assert not table_exists(conn, schemas.src, "diagnoses")


def test_ragged_rows_are_left_out_when_not_strict(pg, tmp_path):
    conn, schemas = pg
    cfg = make_config(tmp_path, _data_dir(tmp_path, {"diagnoses.csv": RAGGED_CSV.encode()}), schemas)
    ws = Workspace(tmp_path / "work")
    manifest = stage_source(cfg, conn, schemas, ws, strict=False)
    m = manifest["diagnoses"]
    assert (m["rows"], m["ragged_rows"], m["ragged_too_many_fields"], m["ragged_too_few_fields"]) == (2, 2, 1, 1)
    assert m["ragged_example_lines"] == [3, 6] and m["blank_lines"] == 1
    rows = fetch_all(conn, f"SELECT id, dx_text, dx_date FROM {schemas.src}.diagnoses ORDER BY id")
    assert rows == [
        {"id": "1", "dx_text": "Malaria", "dx_date": "2020-01-01"},
        {"id": "3", "dx_text": "Cough,\npersistent", "dx_date": "2020-01-03"},
    ]
    saved = (ws.root / MANIFEST).read_text()
    assert "Fever" not in saved and "Headache" not in saved
    event = next(e for e in ws.read_audit() if e["event"] == "source_staged")
    assert event["ragged_rows_skipped"] == {"diagnoses": 2}


def test_undecodable_file_names_the_encoding_setting(pg, tmp_path):
    conn, schemas = pg
    body = "".join(f"{i},Malaria\n" for i in range(1, 5000))  # past the first decoding chunk
    content = ("id,dx_text\n" + body + "5000,Fièvre\n").encode("latin-1")
    cfg = make_config(tmp_path, _data_dir(tmp_path, {"diagnoses.csv": content}), schemas)
    with pytest.raises(SourceDataError) as exc:
        stage_source(cfg, conn, schemas)
    msg = str(exc.value)
    assert "diagnoses.csv" in msg and "utf-8" in msg and "line 5001" in msg
    assert "source.encoding" in msg and "latin-1" in msg and "Fi" not in msg

    cfg.source.encoding = "latin-1"
    manifest = stage_source(cfg, conn, schemas)
    assert manifest["diagnoses"]["rows"] == 5000
    assert scalar(conn, f"SELECT dx_text FROM {schemas.src}.diagnoses WHERE id = '5000'") == "Fièvre"

    cfg.source.encoding = "latin-42"
    with pytest.raises(SourceDataError, match="not a known text encoding"):
        stage_source(cfg, conn, schemas)


def test_utf8_byte_order_mark_is_not_part_of_the_header(pg, tmp_path):
    conn, schemas = pg
    content = "id,dx_text\n1,Fièvre\n".encode("utf-8-sig")
    cfg = make_config(tmp_path, _data_dir(tmp_path, {"diagnoses.csv": content}), schemas)
    manifest = stage_source(cfg, conn, schemas)
    assert manifest["diagnoses"]["columns"][0]["source_name"] == "id"
    assert scalar(conn, f"SELECT dx_text FROM {schemas.src}.diagnoses WHERE id = '1'") == "Fièvre"


@pytest.mark.parametrize("role", ["cdm", "vocab"])
def test_staging_refuses_the_cdm_and_vocabulary_schemas(pg, tmp_path, role):
    conn, schemas = pg
    other = getattr(schemas, role)
    execute(conn, f"CREATE SCHEMA {qi(other)}")
    execute(conn, f"CREATE TABLE {qi(other)}.person (person_id int)")
    execute(conn, f"INSERT INTO {qi(other)}.person VALUES (1)")
    conn.commit()
    cfg = make_config(tmp_path, _data_dir(tmp_path, {"person.csv": b"person_id\n2\n"}), schemas)
    clash = replace(schemas, src=other)
    with pytest.raises(SourceDataError, match=f"also the {role} schema"):
        stage_source(cfg, conn, clash)
    with pytest.raises(SourceDataError):
        stage_csv(cfg, conn, clash)
    assert scalar(conn, f"SELECT person_id FROM {qi(other)}.person") == 1  # not dropped


def test_reload_when_the_bundle_adds_vocabularies(staged, tmp_path):
    """Same Athena release, more vocabularies: must reload, not skip."""
    import shutil

    from omop_agent.demo import FIXTURE_VOCAB
    from omop_agent.vocab.loader import load_athena

    conn, schemas, cfg = staged
    assert load_athena(conn, schemas, FIXTURE_VOCAB) == {}  # identical bundle: skipped
    bigger = tmp_path / "vocab"
    shutil.copytree(FIXTURE_VOCAB, bigger)
    with (bigger / "VOCABULARY.csv").open("a") as fh:
        fh.write("RxNorm Extension\tRxNorm Extension (synthetic fixture)\tomop-agent demo fixture\tfixture-2026-09\t2000009999\n")
    counts = load_athena(conn, schemas, bigger)
    assert counts["vocabulary"] == 10


# --------------------------------------------------------------------------------------
# missing-value markers (source.na_values)
# --------------------------------------------------------------------------------------

NA_CSV = (
    "id,dob,result,code\n"
    "1,NA,POS,NA\n"
    "2, NA ,.,na\n"  # trimmed before the comparison; the comparison is case-sensitive
    "3,25/05/1998,NULL,A1\n"
    "4,\\N,7.5,.5\n"
)


def test_na_values_csv_only(pg, tmp_path):
    conn, schemas = pg
    cfg = make_config(tmp_path, _data_dir(tmp_path, {"labs.csv": NA_CSV.encode()}), schemas)
    ws = Workspace(tmp_path / "work")
    manifest = stage_source(cfg, conn, schemas, ws)
    rows = fetch_all(conn, f"SELECT id, dob, result, code FROM {schemas.src}.labs ORDER BY id")
    assert rows == [
        {"id": "1", "dob": None, "result": "POS", "code": None},
        {"id": "2", "dob": None, "result": None, "code": "na"},
        {"id": "3", "dob": "25/05/1998", "result": None, "code": "A1"},
        {"id": "4", "dob": None, "result": "7.5", "code": ".5"},
    ]
    markers = {c["name"]: c["na_markers"] for c in manifest["labs"]["columns"]}
    assert markers == {"id": {}, "dob": {"NA": 2, "\\N": 1}, "result": {".": 1, "NULL": 1}, "code": {"NA": 1}}
    event = next(e for e in ws.read_audit() if e["event"] == "source_staged")
    assert event["na_markers_as_null"] == {"labs": 6}
    # the profile counts them as NULL, and says how many were markers
    from omop_agent.profiling.profiler import profile_staging

    dob = profile_staging(conn, schemas, cfg, manifest).table("labs").column("dob")
    assert (dob.n_null, dob.n_na_markers) == (3, 3)

    # set [] to keep them as data
    cfg.source.na_values = []
    manifest = stage_source(cfg, conn, schemas)
    assert scalar(conn, f"SELECT count(*) FROM {schemas.src}.labs WHERE dob = 'NA' OR result = '.'") == 2
    assert all(not c["na_markers"] for c in manifest["labs"]["columns"])

    # a database source keeps them unless na_values is set: 'NA' may be a real value there (Namibia)
    from conftest import DB_URL

    src = schemas.etl  # any other test schema serves as the source database schema
    execute(conn, f"CREATE SCHEMA {qi(src)}")
    execute(conn, f"CREATE TABLE {qi(src)}.countries (code text, n integer)")
    execute(conn, f"INSERT INTO {qi(src)}.countries VALUES ('NA', 1), ('KE', NULL), ('.', 3)")
    conn.commit()
    db_cfg = cfg.model_copy(update={"source": cfg.source.model_validate({
        "type": "database", "url": DB_URL.replace("postgresql://", "postgresql+psycopg://", 1), "db_schema": src})})
    assert db_cfg.source.effective_na_values == []
    manifest = stage_source(db_cfg, conn, schemas)
    assert scalar(conn, f"SELECT count(*) FROM {schemas.src}.countries WHERE code IN ('NA', '.')") == 2
    assert all(not c["na_markers"] for c in manifest["countries"]["columns"])
    db_cfg.source.na_values = ["NA"]
    manifest = stage_source(db_cfg, conn, schemas)
    assert fetch_all(conn, f"SELECT code FROM {schemas.src}.countries ORDER BY n") == [
        {"code": None}, {"code": "."}, {"code": "KE"}]
    assert {c["name"]: c["na_markers"] for c in manifest["countries"]["columns"]} == {"code": {"NA": 1}, "n": {}}
