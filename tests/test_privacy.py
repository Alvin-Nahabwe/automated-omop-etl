"""Identifier detection, the privacy overrides and the privacy review gate."""

from __future__ import annotations

import json

import pytest

from omop_agent.approvals import gate_state, record_approval, recorded_approver
from omop_agent.config import PrivacyConfig, ProjectConfig
from omop_agent.models import ColumnAnnotation, ColumnProfile, SourceAnnotations, SourceProfile, TableProfile, ValueFreq
from omop_agent.pipeline import Pipeline, PipelineStopped
from omop_agent.privacy import apply_overrides, detect_pii, llm_view
from omop_agent.config import LLMConfig
from omop_agent.privacy_review import REVIEW_MD, build_review, refresh, review_path
from omop_agent.workspace import Workspace, utcnow

NAMES = ["deceased_name", "respondent_name", "informant_name", "hh_head_name", "interviewer_name", "fname", "lname",
         "othername", "mother_s_name", "patient_s_name", "name_of_mother", "chv_name", "provider_name", "clinician",
         "seen_by", "surname", "first_name", "name", "patientname", "clientname", "mothersname", "guardianname",
         "clinic_patient_name", "facility_client_name", "household_head_name", "name_of_patient"]
OTHER_IDENTIFIERS = ["id10007", "id10017", "id10018", "ccc_no", "upi", "birth_cert_no", "huduma_no", "opd_no", "ip_no",
                     "plot_no", "house_no", "geopoint", "coordinates", "latlong", "hh_long", "x_coord", "easting",
                     "phone", "mobile", "email", "national_id", "va_narrative", "comments", "other_specify", "id10476",
                     "cell", "cell_no", "cell_number", "patientphone", "emergency_contact", "contact_person"]
CLINICAL = ["cd4_cell_count", "cell_count", "tb_contact", "contact_screened", "mobile_clinic", "name_of_facility",
            "drug_name", "test_name", "village_name", "facility_name", "sub_location_name", "cause_of_death",
            "long_term_illness", "house_type", "sex", "dob", "individual_id", "notes_count", "respondent_relationship",
            "brand_name", "unit_name", "file_name", "contacts_traced", "contact_tracing_status"]


@pytest.mark.parametrize("column", NAMES + OTHER_IDENTIFIERS)
def test_identifier_columns_are_detected_by_name(column):
    assert detect_pii("t", column, [], PrivacyConfig()), column


@pytest.mark.parametrize("column", CLINICAL)
def test_clinical_columns_are_not_flagged(column):
    assert detect_pii("t", column, [], PrivacyConfig()) is None, column


def test_identifiers_are_detected_by_value():
    cfg = PrivacyConfig()
    phones = ["0712345678", "0722 333 444", "+254711222333", "712345678", "0101234567"]
    assert "phone" in detect_pii("t", "contact", phones, cfg)
    assert detect_pii("t", "individual_id", phones, cfg) is None  # an id column: long numbers are ids
    assert detect_pii("t", "ts", [str(1_700_000_000 + i) for i in range(5)], cfg) is None  # epoch seconds
    gps = ["-1.3000 36.8000 1650.0 5.0", "-1.3001 36.8001 1650.0 5.0", "-1.29,36.80", "-1.2921 36.8219", "-1.33 36.83"]
    assert "GPS" in detect_pii("t", "loc", gps, cfg)
    stories = ["The deceased had fever for three days and was taken to hospital",
               "She complained of chest pain and difficulty breathing before she died",
               "He was coughing for two weeks then he collapsed at home one night",
               "Mother said the child had diarrhoea and vomiting for two days", "x"]
    assert "free text" in detect_pii("t", "story", stories, cfg)
    drugs = ["Paracetamol 500mg", "Amoxicillin 250mg caps", "ORS sachets", "Zinc 20mg", "Metformin 500mg"]
    assert detect_pii("t", "drug", drugs, cfg) is None
    # Long coded labels are not free text: LOINC names, drug strengths, ICD titles.
    labels = ["Glucose [Mass/volume] in Serum or Plasma --fasting", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation",
              "Amoxicillin 250 MG / Clavulanate 62.5 MG Oral Suspension 100 ML bottle",
              "Artemether 20 MG / Lumefantrine 120 MG Oral Tablet 24 tablets pack",
              "Other and unspecified disorders of the circulatory system in diseases classified elsewhere"]
    assert detect_pii("t", "test_description", labels, cfg) is None
    # A key-like name does not hide phone numbers when the name says phone.
    assert "phone" in detect_pii("t", "alt_cell_ref", phones, cfg)
    assert detect_pii("t", "contact_id", ["123456789", "712345678", "100000001", "100000002", "100000003"], cfg) is None


def test_allow_list_and_explicit_list():
    cfg = PrivacyConfig(not_pii_columns=["t.tb_contact_phone"], pii_columns=["t.weight", "t.tb_contact_phone"])
    assert detect_pii("t", "tb_contact_phone", [], cfg) == "listed in privacy.pii_columns"  # pii_columns wins
    cfg = PrivacyConfig(not_pii_columns=["T.Mobile"])
    assert detect_pii("t", "mobile", [], cfg) is None
    assert detect_pii("t", "weight", [], PrivacyConfig(pii_columns=["t.weight"])) == "listed in privacy.pii_columns"


def _profile() -> SourceProfile:
    def col(name, values=(), pii=None):
        return ColumnProfile(
            name=name, inferred_type="string", n_rows=100, n_null=0, n_distinct=len(values) or 50, unique=False,
            top_values=[ValueFreq(value=v, count=20) for v in values], is_pii=pii is not None, pii_reason=pii,
        )

    return SourceProfile(
        generated_at=utcnow(), min_cell_count=5, source_type="csv",
        tables=[TableProfile(name="va", n_rows=100, columns=[
            col("informant", ["Kamau Mwangi", "Mary Achieng"]),
            col("cause", ["Malaria", "Pneumonia"]),
            col("cd4_count", pii="the column name suggests a phone number"),
            col("story", ["fever then cough"]),
        ])],
    )


def _labels(origin="llm") -> SourceAnnotations:
    return SourceAnnotations(tables=[], columns=[
        ColumnAnnotation(table="va", column="informant", semantic_type="pii", description="name", origin=origin),
        ColumnAnnotation(table="va", column="story", semantic_type="free_text", description="narrative", origin=origin),
    ])


def test_overrides_and_claudes_labels_are_applied():
    annotations = _labels()
    cfg = PrivacyConfig(not_pii_columns=["va.cd4_count"])
    profile = apply_overrides(_profile(), cfg, annotations)
    cols = {c.name: c for c in profile.tables[0].columns}
    assert cols["informant"].is_pii and "labelled an identifier" in cols["informant"].pii_reason
    assert cols["informant"].top_values == []
    assert cols["story"].is_pii and "free text" in cols["story"].pii_reason
    assert not cols["cd4_count"].is_pii and not cols["cause"].is_pii
    text = "\n".join(str(c.top_values) for c in llm_view(profile, cfg).tables[0].columns)
    assert "Kamau" not in text and "Malaria" in text

    # The rule-based fallback calls any long string free text: its labels withhold nothing.
    cols = {c.name: c for c in apply_overrides(_profile(), cfg, _labels("rules")).tables[0].columns}
    assert not cols["informant"].is_pii and not cols["story"].is_pii


def test_label_origin_is_not_part_of_claudes_answer_schema():
    from omop_agent.profiling.annotate import heuristic_annotations, reconcile

    assert "origin" not in str(SourceAnnotations.model_json_schema())
    profile = _profile()
    answer = SourceAnnotations(tables=[], columns=[
        ColumnAnnotation(table="va", column="story", semantic_type="free_text", description="narrative"),
    ])
    origins = {c.column: c.origin for c in reconcile(answer, profile).columns}
    assert origins["story"] == "llm" and origins["cause"] == "rules"  # filled in by the fallback
    assert {c.origin for c in heuristic_annotations(profile).columns} == {"rules"}


def test_current_setting_suppresses_values_a_laxer_profile_kept():
    profile = _profile()  # profiled with min_cell_count 5; every value seen 20 times
    view = llm_view(profile, PrivacyConfig(min_cell_count=25))
    assert all(c.top_values == [] for c in view.tables[0].columns)


def _project(tmp_path, provider="anthropic", **privacy) -> ProjectConfig:
    llm = {"provider": provider, **({"replay_dir": str(tmp_path / "replay")} if provider == "replay" else {})}
    return ProjectConfig.model_validate({
        "name": "p", "workdir": str(tmp_path / "work"), "source": {"type": "csv", "path": str(tmp_path)},
        "target": {"url": "postgresql://nobody@127.0.0.1:1/none"}, "llm": llm, "privacy": privacy,
    })


def test_privacy_review_and_gate(tmp_path):
    cfg = _project(tmp_path)
    ws = Workspace(cfg.workdir)
    assert refresh(cfg, ws) is None  # nothing profiled yet
    ws.save_json(ws.profile_json, _profile())
    assert refresh(cfg, ws) == review_path(ws)
    review = build_review(cfg, ws)
    assert [w["column"] for w in review["withheld"]] == ["cd4_count"]
    assert "Kamau Mwangi" in review["profile_text"]  # the reviewer sees exactly what would be sent
    assert "Withheld columns (1)" in (ws.root / REVIEW_MD).read_text()

    # A pipeline step that would send data stops until the review is approved.
    p = Pipeline(cfg, approver=recorded_approver(ws), log=lambda m: None)
    with pytest.raises(PipelineStopped) as e:
        p._privacy_gate()
    assert e.value.stage == "privacy" and "approve privacy" in str(e.value)
    record_approval(ws, "privacy", "rita")
    Pipeline(cfg, approver=recorded_approver(ws), log=lambda m: None)._privacy_gate()

    # Withholding another column changes what is sent, so the approval no longer holds.
    cfg2 = _project(tmp_path, pii_columns=["va.informant"])
    refresh(cfg2, ws)
    assert not gate_state(ws, "privacy").current
    with pytest.raises(PipelineStopped):
        Pipeline(cfg2, approver=recorded_approver(ws), log=lambda m: None)._privacy_gate()


def test_claudes_later_labels_keep_the_approval(tmp_path):
    cfg = _project(tmp_path)
    ws = Workspace(cfg.workdir)
    ws.save_json(ws.profile_json, _profile())
    refresh(cfg, ws)
    record_approval(ws, "privacy", "rita")
    ws.save_json(ws.annotations_json, _labels())  # Claude labels two more columns: they are withheld from now on
    refresh(cfg, ws)
    review = build_review(cfg, ws)
    assert [w["column"] for w in review["also_withheld"]] == ["informant", "story"]
    assert "Kamau Mwangi" in review["profile_text"]  # what was approved and sent first
    assert gate_state(ws, "privacy").current  # withholding more needs no new approval
    assert "Also withheld from later prompts (2)" in (ws.root / REVIEW_MD).read_text()
    Pipeline(cfg, approver=recorded_approver(ws), log=lambda m: None)._privacy_gate()
    assert ws.stage_status("privacy") == "done"


def test_nothing_is_sent_without_a_profile(tmp_path):
    cfg = _project(tmp_path)
    with pytest.raises(RuntimeError, match="Profile the source first"):
        Pipeline(cfg, approver=recorded_approver(Workspace(cfg.workdir)), log=lambda m: None)._privacy_gate()


def test_no_privacy_review_without_claude(tmp_path):
    for provider in ("none", "replay"):
        cfg = _project(tmp_path, provider=provider)
        ws = Workspace(cfg.workdir)
        ws.save_json(ws.profile_json, _profile())
        assert refresh(cfg, ws) is None
        Pipeline(cfg, approver=recorded_approver(ws), log=lambda m: None)._privacy_gate()  # does not stop


def test_privacy_review_follows_where_the_model_runs(tmp_path):
    from omop_agent.privacy_review import sends_to_llm

    def cfg(**llm):
        return _project(tmp_path, provider="openai_compatible").model_copy(
            update={"llm": LLMConfig(provider="openai_compatible", **llm)}
        )

    # a model on this machine or the institution's private network: prompts stay on site
    for url in ("http://127.0.0.1:8080/v1", "http://localhost:11434/v1", "http://10.20.0.5:8000/v1",
                "http://192.168.1.40/v1", "http://llm.internal/v1", "http://[::1]:8080/v1", "http://llm:8080/v1"):
        assert not sends_to_llm(cfg(base_url=url)), url
    assert not sends_to_llm(cfg())  # the default URL is local
    # a public address needs the review unless it is declared on premises
    assert sends_to_llm(cfg(base_url="https://llm.example.org/v1"))
    assert not sends_to_llm(cfg(base_url="https://llm.example.org/v1", on_premises=True))
    assert sends_to_llm(cfg(base_url="http://127.0.0.1:8080/v1", on_premises=False))
    ws = Workspace(tmp_path / "work")
    ws.save_json(ws.profile_json, _profile())
    local = cfg()
    assert refresh(local, ws) is None
    Pipeline(local, approver=recorded_approver(ws), log=lambda m: None)._privacy_gate()  # does not stop


def test_stage_commands_need_a_recorded_privacy_review(tmp_path):
    from typer.testing import CliRunner

    from omop_agent.cli import app

    cfg_path = tmp_path / "omop_agent.yaml"
    cfg_path.write_text(
        "name: p\nworkdir: ./work\nsource: {type: csv, path: .}\n"
        "target: {url: 'postgresql://nobody@127.0.0.1:1/none'}\nllm: {provider: anthropic}\n"
    )
    ws = Workspace(tmp_path / "work")
    ws.save_json(ws.profile_json, _profile())
    result = CliRunner().invoke(app, ["annotate", "-c", str(cfg_path)])
    assert result.exit_code == 2 and "Waiting for a reviewer" in result.output
    result = CliRunner().invoke(app, ["approve", "privacy", "--by", "rita", "-c", str(cfg_path)])
    assert result.exit_code == 0 and "Privacy review approved by rita" in result.output
    assert gate_state(ws, "privacy").current


# --------------------------------------------------------------------------------------
# value shapes, offsets and the counts of values that do not fit
# --------------------------------------------------------------------------------------

# The prompt text of _profile() before the profile had shapes and offsets: an old profile.json
# must give exactly this text, so an approved privacy review stays approved after an upgrade.
OLD_PROFILE_TEXT = """SOURCE TABLE va (100 rows)
  - informant: string, 0% null, 2 distinct
      top values: 'Kamau Mwangi'×20; 'Mary Achieng'×20
  - cause: string, 0% null, 2 distinct
      top values: 'Malaria'×20; 'Pneumonia'×20
  - cd4_count: string, 0% null, 50 distinct, PII - values withheld (the column name suggests a phone number)
  - story: string, 0% null, 1 distinct
      top values: 'fever then cough'×20
"""


def test_old_profiles_keep_their_text_and_review(tmp_path):
    from omop_agent.privacy import profile_prompt_text

    new_fields = {"n_na_markers", "n_nonconforming", "nonconforming_shapes", "n_partial_dates", "value_shapes",
                  "tz_offsets", "n_rare_offsets", "format_notes"}
    old = _profile().model_dump(mode="json")
    for t in old["tables"]:
        for c in t["columns"]:
            for f in new_fields:
                c.pop(f)
    profile = SourceProfile.model_validate(old)  # a profile.json written before these fields existed
    assert profile_prompt_text(profile, PrivacyConfig()) == OLD_PROFILE_TEXT

    cfg = _project(tmp_path)
    ws = Workspace(cfg.workdir)
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.profile_json.write_text(json.dumps(old))
    refresh(cfg, ws)
    record_approval(ws, "privacy", "rita")
    ws.save_json(ws.profile_json, profile)  # the same profile, saved by this version
    refresh(cfg, ws)
    assert gate_state(ws, "privacy").current
    # profiled again with shapes: what is sent changes, so the review needs a new approval
    profile.tables[0].columns[1].value_shapes = [ValueFreq(value="Aaaaaaa", count=40)]
    ws.save_json(ws.profile_json, profile)
    refresh(cfg, ws)
    assert not gate_state(ws, "privacy").current
    assert "value shapes (9 digit, a/A letter): 'Aaaaaaa'×40" in build_review(cfg, ws)["profile_text"]


def _shaped_profile() -> SourceProfile:
    def col(name, **kw):
        base = dict(name=name, inferred_type="string", n_rows=100, n_null=0, n_distinct=60, unique=False)
        return ColumnProfile(**{**base, **kw})

    return SourceProfile(
        generated_at=utcnow(), min_cell_count=5, source_type="csv",
        tables=[TableProfile(name="visits", n_rows=100, columns=[
            col("visit_ref", value_shapes=[ValueFreq(value="A9999", count=80), ValueFreq(value="AAA-99", count=6)]),
            col("submitted", inferred_type="datetime", date_format="%Y-%m-%dT%H:%M:%S", n_nonconforming=3,
                n_partial_dates=12, n_na_markers=4, tz_offsets=[ValueFreq(value="+03:00", count=70),
                                                                ValueFreq(value="Z", count=6)],
                nonconforming_shapes=[ValueFreq(value="aaaaaaa", count=6)], format_notes=["a note"]),
            # an identifier that (wrongly) still carries shapes must not leak them
            col("phone", is_pii=True, pii_reason="phone", value_shapes=[ValueFreq(value="9999999999", count=90)],
                format_notes=["a note on the phone"]),
        ])],
    )


def test_shapes_and_offsets_follow_the_small_cell_rule():
    from omop_agent.privacy import profile_prompt_text
    from omop_agent.profiling.render import render_profile_md

    profile = _shaped_profile()
    text = profile_prompt_text(profile, PrivacyConfig())
    assert "value shapes (9 digit, a/A letter): 'A9999'×80; 'AAA-99'×6" in text
    assert "time zone offsets '+03:00'×70; 'Z'×6" in text
    assert "<5 values do not fit the format %Y-%m-%dT%H:%M:%S" in text  # 3: a small count
    assert "12 partial dates" in text and "<5 missing-value markers" in text and "note: a note" in text
    assert "shapes of the values that do not fit: 'aaaaaaa'×6" in text
    phone_line = next(line for line in text.splitlines() if line.startswith("  - phone:"))
    assert "PII" in phone_line and "9999999999" not in text and "a note on the phone" not in text
    # a stricter current threshold hides what the profile kept
    strict = profile_prompt_text(profile, PrivacyConfig(min_cell_count=10))
    for gone in ("'AAA-99'", "'Z'", "'aaaaaaa'"):
        assert gone not in strict
    assert "'A9999'×80" in strict and "<10 values do not fit" in strict
    # without values: counts and notes only
    bare = profile_prompt_text(profile, PrivacyConfig(send_values_to_llm=False))
    assert "value shapes" not in bare and "offsets" not in bare and "shapes of the values" not in bare
    assert "note: a note" in bare and "12 partial dates" in bare
    # profile.md shows the same, under the same rule
    md = render_profile_md(profile)
    assert "`A9999`×80, `AAA-99`×6" in md and "`+03:00`×70, `Z`×6" in md and "| <5 |" in md
    assert "9999999999" not in md and "a note on the phone" not in md
    # a column withheld later (privacy.pii_columns) loses its shapes and offsets too
    hidden = apply_overrides(profile, PrivacyConfig(pii_columns=["visits.submitted"]))
    sub = hidden.tables[0].columns[1]
    assert sub.is_pii and sub.tz_offsets == [] and sub.value_shapes == [] and sub.nonconforming_shapes == []
    assert "+03:00" not in profile_prompt_text(hidden, PrivacyConfig())


@pytest.mark.postgres
def test_profiled_shapes_never_leak_rare_or_identifier_values(pg, tmp_path):
    from conftest import make_config

    from omop_agent.db import ensure_schema, execute
    from omop_agent.privacy import profile_prompt_text
    from omop_agent.profiling.profiler import profile_staging
    from omop_agent.profiling.render import render_profile_md

    conn, schemas = pg
    cfg = make_config(tmp_path, tmp_path, schemas)
    ensure_schema(conn, schemas.src)
    execute(conn, f"""CREATE TABLE {schemas.src}.visits AS SELECT
        'V' || lpad(g::text, 5, '0') AS visit_ref,
        CASE WHEN g = 1 THEN 'Special-Ward-7' ELSE 'OPD-' || (g % 90) END AS ward_code,
        '07' || lpad((g * 7919 % 100000000)::text, 8, '0') AS phone,
        to_char(TIMESTAMP '2020-01-01' + g * interval '1 hour', 'YYYY-MM-DD"T"HH24:MI:SS')
          || CASE WHEN g = 2 THEN '+01:00' WHEN g % 3 = 0 THEN 'Z' ELSE '+03:00' END AS submitted,
        CASE WHEN g = 3 THEN '1/1' ELSE g::text END AS qty,
        CASE WHEN g <= 7 THEN 'TEMP_A1' ELSE 'KMR' || lpad((g / 3)::text, 5, '0') END AS individual_id
        FROM generate_series(1, 400) g""")
    profile = profile_staging(conn, schemas, cfg)
    k = profile.min_cell_count
    cols = {c.name: c for c in profile.table("visits").columns}
    assert cols["phone"].is_pii
    assert (cols["phone"].value_shapes, cols["phone"].nonconforming_shapes, cols["phone"].format_notes) == ([], [], [])
    for c in cols.values():
        assert all(v.count >= k for v in [*c.value_shapes, *c.nonconforming_shapes, *c.tz_offsets]), c.name
    assert cols["visit_ref"].value_shapes == [ValueFreq(value="A99999", count=400)]
    assert {v.value for v in cols["submitted"].tz_offsets} == {"+03:00", "Z"}
    assert cols["qty"].n_nonconforming == 1 and cols["qty"].nonconforming_shapes == []
    # a key's shape is listed only when enough different keys have it: the 7 rows of 'TEMP_A1' are
    # one person's records
    assert (cols["individual_id"].top_values, cols["individual_id"].value_shapes) == (
        [], [ValueFreq(value="AAA99999", count=393)])
    # the rare values (a ward code, an offset, a fraction, a key) and their shapes appear nowhere;
    # the value with the rare offset is counted as "<k"
    prompt = profile_prompt_text(profile, cfg.privacy)
    texts = [profile.model_dump_json(), prompt, render_profile_md(profile)]
    for leaked in ("Special", "Aaaaaaa-Aaaa-9", "+01:00", "1/1", "9/9", "TEMP", "AAAA_A9"):
        assert all(leaked not in t for t in texts), leaked
    assert cols["submitted"].n_rare_offsets == 1
    assert f"<{k} values with a time zone offset seen fewer than {k} times" in prompt
    assert f"<{k} with rarer offsets" in texts[2]
    # nor do the identifier's values or shape
    phones = [r["phone"] for r in conn.execute(f"SELECT phone FROM {schemas.src}.visits").fetchall()]
    assert all(p not in t for p in phones for t in texts)
    assert all("9999999999" not in t for t in texts)


@pytest.mark.postgres
def test_sparse_identifier_columns_are_still_detected(pg, tmp_path):
    """A mostly empty phone or e-mail column in a large table, under a name the rules miss: the
    spread sample holds too few of its values, so the profiler reads the column's first values to
    judge it (household phones repeat, and would otherwise be listed)."""
    from conftest import make_config

    from omop_agent.db import ensure_schema, execute
    from omop_agent.profiling.profiler import profile_staging

    conn, schemas = pg
    cfg = make_config(tmp_path, tmp_path, schemas)
    ensure_schema(conn, schemas.src)
    execute(conn, f"""CREATE TABLE {schemas.src}.individuals AS SELECT 'P' || g AS perm_id,
        CASE WHEN g % 3750 = 0 THEN '07' || lpad((g / 18750)::text, 8, '0') END AS hh_contact,
        CASE WHEN g % 3750 = 0 THEN 'house' || (g / 18750) || '@mail.example.org' END AS hh_alt
        FROM generate_series(1, 150000) g""")
    cols = {c.name: c for c in profile_staging(conn, schemas, cfg).table("individuals").columns}
    for name, reason in [("hh_contact", "phone numbers"), ("hh_alt", "e-mail addresses")]:
        c = cols[name]
        assert (c.n_rows - c.n_null, c.is_pii, c.top_values) == (40, True, []), name
        assert reason in (c.pii_reason or ""), name
