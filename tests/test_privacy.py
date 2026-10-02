"""Identifier detection, the privacy overrides and the privacy review gate."""

from __future__ import annotations

import pytest

from omop_agent.approvals import gate_state, record_approval, recorded_approver
from omop_agent.config import PrivacyConfig, ProjectConfig
from omop_agent.models import ColumnAnnotation, ColumnProfile, SourceAnnotations, SourceProfile, TableProfile, ValueFreq
from omop_agent.pipeline import Pipeline, PipelineStopped
from omop_agent.privacy import apply_overrides, detect_pii, llm_view
from omop_agent.privacy_review import REVIEW_MD, build_review, refresh, review_path
from omop_agent.workspace import Workspace, utcnow

NAMES = ["deceased_name", "respondent_name", "informant_name", "hh_head_name", "interviewer_name", "fname", "lname",
         "othername", "mother_s_name", "patient_s_name", "name_of_mother", "chv_name", "provider_name", "clinician",
         "seen_by", "surname", "first_name", "name"]
OTHER_IDENTIFIERS = ["id10007", "id10017", "id10018", "ccc_no", "upi", "birth_cert_no", "huduma_no", "opd_no", "ip_no",
                     "plot_no", "house_no", "geopoint", "coordinates", "latlong", "hh_long", "x_coord", "easting",
                     "phone", "mobile", "email", "national_id", "va_narrative", "comments", "other_specify", "id10476"]
CLINICAL = ["cd4_cell_count", "cell_count", "tb_contact", "contact_screened", "mobile_clinic", "name_of_facility",
            "drug_name", "test_name", "village_name", "facility_name", "sub_location_name", "cause_of_death",
            "long_term_illness", "house_type", "sex", "dob", "individual_id", "notes_count", "respondent_relationship"]


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


def test_overrides_and_claudes_labels_are_applied():
    annotations = SourceAnnotations(tables=[], columns=[
        ColumnAnnotation(table="va", column="informant", semantic_type="pii", description="informant's name"),
        ColumnAnnotation(table="va", column="story", semantic_type="free_text", description="narrative"),
    ])
    cfg = PrivacyConfig(not_pii_columns=["va.cd4_count"])
    profile = apply_overrides(_profile(), cfg, annotations)
    cols = {c.name: c for c in profile.tables[0].columns}
    assert cols["informant"].is_pii and "labelled an identifier" in cols["informant"].pii_reason
    assert cols["informant"].top_values == []
    assert cols["story"].is_pii and "free text" in cols["story"].pii_reason
    assert not cols["cd4_count"].is_pii and not cols["cause"].is_pii
    text = "\n".join(str(c.top_values) for c in llm_view(profile, cfg).tables[0].columns)
    assert "Kamau" not in text and "Malaria" in text


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


def test_no_privacy_review_without_claude(tmp_path):
    for provider in ("none", "replay"):
        cfg = _project(tmp_path, provider=provider)
        ws = Workspace(cfg.workdir)
        ws.save_json(ws.profile_json, _profile())
        assert refresh(cfg, ws) is None
        Pipeline(cfg, approver=recorded_approver(ws), log=lambda m: None)._privacy_gate()  # does not stop


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
