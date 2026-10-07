"""The review web app: sign-in, protections, mapping review, review gates and background runs."""

from __future__ import annotations

import csv
import os
import re
import shutil
import signal
import time
import uuid
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from omop_agent.approvals import gate_state, record_approval  # noqa: E402
from omop_agent.config import load_config  # noqa: E402
from omop_agent.db import connect, execute, qi  # noqa: E402
from omop_agent.models import ConceptCandidate, MappingDecision  # noqa: E402
from omop_agent.vocab.review import export_review, import_review  # noqa: E402
from omop_agent.web.app import create_app, term_anchor  # noqa: E402
from omop_agent.web.jobs import JobBusy, JobStore  # noqa: E402
from omop_agent.web.security import REPORT_CSP  # noqa: E402
from omop_agent.web.users import UserError, UserStore  # noqa: E402
from omop_agent.workspace import Workspace  # noqa: E402

SPEC = Path(__file__).parent / "fixtures" / "demo_etl_spec.yaml"
DB_URL = os.environ.get("OMOP_AGENT_TEST_DB_URL")
PASSWORDS = {"olive": "operator-password", "rita": "reviewer-password", "vera": "viewer-password"}
UNREACHABLE_DB = "postgresql://nobody@127.0.0.1:1/none"


def _candidate(cid: int, name: str) -> ConceptCandidate:
    return ConceptCandidate(
        concept_id=cid, concept_name=name, domain_id="Drug", vocabulary_id="RxNorm",
        concept_class_id="Clinical Drug", standard_concept="S", concept_code=str(cid),
    )


def _decisions() -> list[MappingDecision]:
    return [
        MappingDecision(
            source_vocabulary_id="SRC_DRUG", source_code="Paracetamol 500mg", source_name="Paracetamol 500mg",
            frequency=900, target_domain="Drug", target_concept_id=11, target_concept_name="acetaminophen 500 MG",
            equivalence="EQUAL", confidence=0.85, status="UNCHECKED", method="llm", rationale="Strength matches.",
            candidates=[_candidate(11, "acetaminophen 500 MG"), _candidate(12, "acetaminophen 500 MG Oral Tablet")],
        ),
        MappingDecision(
            source_vocabulary_id="SRC_DRUG", source_code="TLD (TDF/3TC/DTG)", source_name="TLD (TDF/3TC/DTG)",
            frequency=287, target_domain="Drug", status="UNCHECKED", method="llm",
        ),
    ]


def _write_project(root: Path, pid: str = "demo", db_url: str = UNREACHABLE_DB, prefix: str = "t") -> Path:
    d = root / pid
    (d / "data").mkdir(parents=True)
    (d / "omop_agent.yaml").write_text(
        f"""name: Test project
workdir: ./work
source: {{type: csv, path: ./data}}
target:
  url: "{db_url}"
  cdm_schema: {prefix}_cdm
  vocab_schema: {prefix}_vocab
  staging_schema: {prefix}_src
  etl_schema: {prefix}_etl
  results_schema: {prefix}_results
llm: {{provider: none}}
"""
    )
    return d


@pytest.fixture
def site(tmp_path):
    """A projects directory with one project (spec and mappings, no database) and three accounts."""
    projects, data = tmp_path / "projects", tmp_path / "data"
    d = _write_project(projects)
    ws = Workspace(d / "work")
    shutil.copy(SPEC, ws.spec_yaml)
    export_review(ws, _decisions())
    users = UserStore(data)
    users.add("olive", PASSWORDS["olive"], "operator", "Olive O.")
    users.add("rita", PASSWORDS["rita"], "reviewer")
    users.add("vera", PASSWORDS["vera"], "viewer")
    app = create_app(projects, data, secret_key="x" * 40)
    return TestClient(app), ws, users, data


def _csrf(client: TestClient, path: str = "/login") -> str:
    html = client.get(path).text
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def login(client: TestClient, name: str, password: str | None = None):
    token = _csrf(client)
    return client.post(
        "/login", data={"username": name, "password": password or PASSWORDS[name], "csrf_token": token, "next": "/"},
        follow_redirects=False,
    )


def _decide(client, ws, code, action, concept_id=None, note="", expected=None):
    d = next(x for x in import_review(ws) if x.source_code == code)
    exp_target, exp_status = expected or (d.target_concept_id, d.status)
    data = {
        "csrf_token": _csrf(client, "/p/demo/mappings?filter=all"), "vocab_id": "SRC_DRUG", "code": code,
        "action": action, "note": note, "expected_target": str(exp_target), "expected_status": exp_status,
        "back_filter": "all",
    }
    if concept_id is not None:
        data["concept_id"] = str(concept_id)
    return client.post("/p/demo/mappings/decide", data=data, follow_redirects=False)


# ---- accounts ------------------------------------------------------------------------


def test_user_store(tmp_path):
    users = UserStore(tmp_path)
    u = users.add("Alice", "long-enough-pw", "reviewer", "Alice N.")
    assert u.name == "alice" and u.can("viewer") and u.can("reviewer") and not u.can("operator")
    assert oct((tmp_path / "users.json").stat().st_mode & 0o777) == "0o600"
    assert "long-enough-pw" not in (tmp_path / "users.json").read_text()
    assert users.authenticate("ALICE", "long-enough-pw").name == "alice"
    assert users.authenticate("alice", "wrong-password") is None
    assert users.authenticate("nobody", "long-enough-pw") is None
    with pytest.raises(UserError):
        users.add("bob", "short", "viewer")
    with pytest.raises(UserError):
        users.add("bob", "long-enough-pw", "admin")
    with pytest.raises(UserError):
        users.add("alice", "long-enough-pw", "viewer")
    with pytest.raises(UserError):
        users.add("../x", "long-enough-pw", "viewer")
    epoch = users.get("alice").epoch
    users.set_active("alice", False)
    assert users.get("alice").epoch == epoch + 1
    assert users.authenticate("alice", "long-enough-pw") is None
    assert not users.get("alice").can("viewer")


# ---- sign-in and protections ---------------------------------------------------------


def test_pages_need_sign_in_and_send_security_headers(site):
    client, *_ = site
    r = client.get("/p/demo/mappings", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=%2Fp%2Fdemo%2Fmappings"
    for k in ("content-security-policy", "x-frame-options", "x-content-type-options", "referrer-policy"):
        assert k in r.headers
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/healthz").text == "ok"
    page = client.get("/login")
    assert page.headers["cache-control"] == "no-store"
    assert "<script>" not in page.text and " style=" not in page.text  # nothing inline, so the CSP can forbid it


def test_sign_in_and_out(site):
    client, *_ = site
    r = login(client, "olive", "wrong password!")
    assert r.status_code == 401 and "Wrong user name or password" in r.text
    r = login(client, "olive")
    assert r.status_code == 303 and r.headers["location"] == "/"
    home = client.get("/")
    assert home.status_code == 200 and "Test project" in home.text and "Olive O." in home.text
    token = _csrf(client, "/")
    client.post("/logout", data={"csrf_token": token}, follow_redirects=False)
    assert client.get("/", follow_redirects=False).status_code == 303


def test_repeated_failures_lock_the_account_out(site):
    client, *_ = site
    for _ in range(5):
        assert login(client, "rita", "not the password").status_code == 401
    r = login(client, "rita")
    assert r.status_code == 429 and "Too many failed sign-ins" in r.text


def test_open_redirects_are_refused(site):
    client, *_ = site
    token = _csrf(client)
    r = client.post(
        "/login", data={"username": "olive", "password": PASSWORDS["olive"], "csrf_token": token, "next": "//evil.example/"},
        follow_redirects=False,
    )
    assert r.headers["location"] == "/"


def test_forms_need_the_csrf_token(site):
    client, ws, *_ = site
    login(client, "rita")
    r = client.post("/p/demo/mappings/approve", data={"note": "x"})
    assert r.status_code == 403
    r = client.post("/login", data={"username": "rita", "password": PASSWORDS["rita"], "csrf_token": "forged"})
    assert r.status_code == 403
    assert not gate_state(ws, "vocab_map").approval


def test_sessions_end_when_the_account_changes(site):
    client, _, users, _ = site
    login(client, "rita")
    assert client.get("/", follow_redirects=False).status_code == 200
    users.set_role("rita", "viewer")
    assert client.get("/", follow_redirects=False).status_code == 303


def test_unknown_and_malformed_projects_are_not_found(site):
    client, *_ = site
    login(client, "vera")
    assert client.get("/p/nope").status_code == 404
    assert client.get("/p/..%2F..%2Fetc").status_code == 404
    assert client.get("/p/.hidden").status_code == 404


def test_roles(site):
    client, ws, *_ = site
    login(client, "vera")
    page = client.get("/p/demo/mappings?filter=all").text
    assert "Paracetamol 500mg" in page and "Approve selection" not in page
    assert _decide(client, ws, "Paracetamol 500mg", "approve").status_code == 403
    token = _csrf(client, "/p/demo")
    assert client.post("/p/demo/runs", data={"csrf_token": token, "kind": "continue"}).status_code == 403
    assert "Continue the pipeline" not in client.get("/p/demo").text

    client.cookies.clear()
    login(client, "rita")
    token = _csrf(client, "/p/demo")
    assert client.post("/p/demo/runs", data={"csrf_token": token, "kind": "continue"}).status_code == 403


# ---- mapping review ------------------------------------------------------------------


def test_reviewer_approves_flags_and_rejects(site):
    client, ws, *_ = site
    login(client, "rita")
    page = client.get("/p/demo/mappings").text
    assert "Paracetamol 500mg" in page and "Strength matches." in page

    # Pick the other listed candidate (no database needed for a listed one).
    r = _decide(client, ws, "Paracetamol 500mg", "approve", concept_id=12, note="tablet form")
    assert r.status_code == 303 and r.headers["location"].endswith("#" + term_anchor("SRC_DRUG", "Paracetamol 500mg"))
    para = next(d for d in import_review(ws) if d.source_code == "Paracetamol 500mg")
    assert (para.target_concept_id, para.status, para.method, para.reviewed_by, para.review_note) == (
        12, "APPROVED", "manual", "rita", "tablet form",
    )
    with (ws.mappings_dir / "SRC_DRUG.csv").open(newline="") as fh:
        row = next(r for r in csv.DictReader(fh) if r["source_code"] == "Paracetamol 500mg")
    assert row["reviewed_by"] == "rita" and row["target_concept_id"] == "12"

    _decide(client, ws, "TLD (TDF/3TC/DTG)", "flag", note="ask pharmacy")
    tld = next(d for d in import_review(ws) if d.source_code.startswith("TLD"))
    assert tld.status == "FLAGGED" and tld.review_note == "ask pharmacy"
    _decide(client, ws, "TLD (TDF/3TC/DTG)", "reject")
    assert next(d for d in import_review(ws) if d.source_code.startswith("TLD")).status == "REJECTED"

    events = [e for e in ws.read_audit() if e["event"] == "mapping_reviewed"]
    assert [e["action"] for e in events] == ["approve", "flag", "reject"]
    assert events[0]["before"] == {"target_concept_id": 11, "status": "UNCHECKED"}
    assert events[0]["after"] == {"target_concept_id": 12, "status": "APPROVED"} and events[0]["by"] == "rita"


def test_unlisted_concepts_need_the_database_and_stale_pages_are_refused(site):
    client, ws, *_ = site
    login(client, "rita")
    r = _decide(client, ws, "Paracetamol 500mg", "approve", concept_id=999)
    assert r.status_code == 303
    assert "cannot be reached" in client.get(r.headers["location"]).text
    assert next(d for d in import_review(ws) if d.source_code == "Paracetamol 500mg").status == "UNCHECKED"

    _decide(client, ws, "Paracetamol 500mg", "approve")  # someone else approves it first
    r = _decide(client, ws, "Paracetamol 500mg", "reject", expected=(11, "UNCHECKED"))
    assert "was changed by rita after you opened the page" in client.get(r.headers["location"]).text
    assert next(d for d in import_review(ws) if d.source_code == "Paracetamol 500mg").status == "APPROVED"


def test_mapping_gate_approves_only_what_the_reviewer_saw(site):
    client, ws, *_ = site
    login(client, "rita")
    page = client.get("/p/demo/mappings").text
    digest = re.search(r'name="expected_digest" value="([^"]+)"', page).group(1)
    _decide(client, ws, "TLD (TDF/3TC/DTG)", "flag")  # the mappings change after the page was loaded
    token = _csrf(client, "/p/demo/mappings")
    r = client.post("/p/demo/mappings/approve", data={"csrf_token": token, "expected_digest": digest})
    assert "changed after you opened the page" in r.text
    assert not gate_state(ws, "vocab_map").current

    page = client.get("/p/demo/mappings").text
    digest = re.search(r'name="expected_digest" value="([^"]+)"', page).group(1)
    client.post("/p/demo/mappings/approve", data={"csrf_token": token, "expected_digest": digest, "note": "ok"})
    g = gate_state(ws, "vocab_map")
    assert g.current and g.approval.by == "rita" and g.approval.note == "ok"


def test_spec_page_shows_the_design_and_changes_since_approval(site):
    client, ws, *_ = site
    login(client, "rita")
    page = client.get("/p/demo/spec").text
    assert "Approve the ETL design" in page and "condition_occurrence" in page
    record_approval(ws, "design", "rita")
    ws.spec_yaml.write_text(ws.spec_yaml.read_text().replace("Sex code 9", "Sex code 9 (edited)"))
    page = client.get("/p/demo/spec").text
    assert "changed since approval" in page and "Sex code 9 (edited)" in page and 'class="add"' in page


def test_spec_page_shows_key_sources_and_dedup_order(site):
    from omop_agent.models import ETLSpec, FieldMapping, TableMapping

    client, ws, *_ = site
    login(client, "rita")
    spec = ETLSpec.model_validate(yaml.safe_load(ws.spec_yaml.read_text()))
    spec.mapping_for("death").dedup_order = "x.dod::date DESC"
    spec.table_mappings.append(TableMapping(
        target_table="person", source={"table": "emr_patients", "alias": "e"}, fields=[
            FieldMapping(target_field="person_id", kind="person_id", expression="'EMR:' || e.patient_no")]))
    ws.save_yaml(ws.spec_yaml, spec)
    page = client.get("/p/demo/spec").text
    assert "also <code>emr_patients</code> keyed by <code>&#39;EMR:&#39; || e.patient_no</code>" in page
    assert "one row per key, the best row first by <code>x.dod::date DESC</code>" in page


def test_report_is_served_with_a_strict_policy(site):
    client, ws, *_ = site
    login(client, "vera")
    assert client.get("/p/demo/report").status_code == 404
    ws.report_html.write_text("<html><body>report</body></html>")
    r = client.get("/p/demo/report")
    assert r.status_code == 200 and r.headers["content-security-policy"] == REPORT_CSP


# ---- runs ----------------------------------------------------------------------------


def _wait(store: JobStore, job_id: str, timeout: float = 60) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        status = store.status(store.get(job_id))
        if status not in ("queued", "running"):
            return status
        time.sleep(0.3)
    raise TimeoutError(job_id)


def test_a_run_started_from_the_app_reports_its_failure(site):
    client, ws, *_ = site
    login(client, "olive")
    token = _csrf(client, "/p/demo")
    r = client.post("/p/demo/runs", data={"csrf_token": token, "kind": "fresh"}, follow_redirects=False)
    assert "confirmation box" in client.get(r.headers["location"]).text  # destructive kinds need a tick

    r = client.post("/p/demo/runs", data={"csrf_token": token, "kind": "continue"}, follow_redirects=False)
    job_id = r.headers["location"].rsplit("/", 1)[-1]
    store = JobStore(ws)
    assert _wait(store, job_id) == "failed"  # the database in this project does not exist
    job = store.get(job_id)
    assert job.by == "olive" and job.message
    page = client.get(f"/p/demo/runs/{job_id}").text
    assert "failed" in page and "Run " in page
    assert {"web_run_started", "web_run_finished"} <= {e["event"] for e in ws.read_audit()}


def test_one_run_at_a_time(site):
    _, ws, *_ = site
    store = JobStore(ws)
    cfg = ws.root.parent / "omop_agent.yaml"
    job = store.start("continue", "olive", cfg, command=["python3", "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(JobBusy):
            store.start("continue", "olive", cfg)
        assert store.active().id == job.id
        assert store.get("../../etc/passwd") is None
    finally:
        os.killpg(job.pid, signal.SIGTERM)


# ---- the whole review cycle on PostgreSQL --------------------------------------------


def _drop(prefix: str) -> None:
    conn = connect(DB_URL, autocommit=True)  # type: ignore[arg-type]
    for s in ("cdm", "vocab", "src", "etl", "results"):
        execute(conn, f"DROP SCHEMA IF EXISTS {qi(f'{prefix}_{s}')} CASCADE")
    conn.close()


@pytest.mark.postgres
def test_review_cycle_with_recorded_approvals(tmp_path):
    from omop_agent.demo import create_demo_project
    from omop_agent.web.jobs import Job
    from omop_agent.web.worker import run_job
    from omop_agent.workspace import utcnow

    prefix = f"t{uuid.uuid4().hex[:8]}web"
    projects, data = tmp_path / "projects", tmp_path / "data"
    cfg_path = create_demo_project(projects / "demo", DB_URL, provider="replay", schema_prefix=prefix)  # type: ignore[arg-type]
    try:
        ws = Workspace(load_config(cfg_path).workdir)
        store = JobStore(ws)

        def run(kind: str = "continue") -> Job:
            job = Job(id=f"{utcnow():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}", kind=kind, by="olive",
                      config_path=str(cfg_path), created_at=utcnow())
            store.save(job)
            return run_job(store.path(job.id))

        first = run()
        assert (first.status, first.stage) == ("waiting_review", "design")
        record_approval(ws, "design", "rita")
        second = run()
        assert (second.status, second.stage) == ("waiting_review", "vocab_map")

        users = UserStore(data)
        users.add("rita", PASSWORDS["rita"], "reviewer")
        client = TestClient(create_app(projects, data, secret_key="y" * 40))
        login(client, "rita")
        d = next(x for x in import_review(ws) if x.source_vocabulary_id == "SRC_LAB" and x.source_code == "HB")
        anchor = term_anchor(d.source_vocabulary_id, d.source_code)
        page = client.get(f"/p/demo/mappings?filter=all&vocab=SRC_LAB&term={anchor}&q=hemoglobin").text
        assert "Hemoglobin" in page  # the vocabulary search works against the database
        r = client.post("/p/demo/mappings/decide", data={
            "csrf_token": _csrf(client, "/p/demo/mappings"), "vocab_id": "SRC_LAB", "code": "HB", "action": "approve",
            "concept_id": "2000004001", "expected_target": str(d.target_concept_id), "expected_status": d.status,
        }, follow_redirects=False)
        assert "approved" in client.get(r.headers["location"]).text
        page = client.get("/p/demo/mappings").text
        digest = re.search(r'name="expected_digest" value="([^"]+)"', page).group(1)
        client.post("/p/demo/mappings/approve", data={"csrf_token": _csrf(client, "/p/demo/mappings"), "expected_digest": digest})
        assert gate_state(ws, "vocab_map").current

        third = run()
        assert third.status == "passed", third.message
        hb = next(x for x in import_review(ws) if x.source_code == "HB")
        assert (hb.status, hb.reviewed_by) == ("APPROVED", "rita")
        assert client.get("/p/demo/report").status_code == 200
    finally:
        _drop(prefix)


# ---- fixes from review ---------------------------------------------------------------


def test_a_broken_spec_file_does_not_break_the_pages(site):
    client, ws, *_ = site
    record_approval(ws, "design", "rita")
    ws.spec_yaml.write_text("table_mappings: [unclosed\n  - : :")
    login(client, "vera")
    assert client.get("/").status_code == 200
    assert client.get("/p/demo").status_code == 200
    page = client.get("/p/demo/spec")
    assert page.status_code == 200 and "has errors" in page.text and "changed since approval" in page.text


def test_projects_sharing_schemas_are_flagged_and_not_run(site, tmp_path):
    from omop_agent.web.projects import schema_conflicts_for

    client, ws, *_ = site
    projects = ws.root.parent.parent
    other = _write_project(projects, "other")  # same URL and the same schema names as "demo"
    third = _write_project(projects, "third", prefix="third")
    cfg = load_config(projects / "demo" / "omop_agent.yaml")
    assert schema_conflicts_for(projects / "demo", cfg) == ["other: t_cdm, t_etl, t_results, t_src"]
    assert schema_conflicts_for(third, load_config(third / "omop_agent.yaml")) == []
    login(client, "olive")
    assert "shares database schemas with another project" in client.get("/p/demo").text
    token = _csrf(client, "/p/demo")
    r = client.post("/p/demo/runs", data={"csrf_token": token, "kind": "continue"}, follow_redirects=False)
    job_id = r.headers["location"].rsplit("/", 1)[-1]
    store = JobStore(ws)
    assert _wait(store, job_id) == "failed"
    assert "another project uses" in store.get(job_id).message
    assert other.exists()


def test_mapping_edits_wait_for_a_run_started_elsewhere(site):
    from omop_agent.runlock import run_lock

    client, ws, *_ = site
    login(client, "rita")
    with run_lock(ws, "command line, pid 1"):
        assert "A run started outside this app is in progress" in client.get("/p/demo").text
        assert "edits are paused" in client.get("/p/demo/mappings").text
        r = _decide(client, ws, "TLD (TDF/3TC/DTG)", "reject")
        assert "Mapping edits are paused" in client.get(r.headers["location"]).text
    assert next(d for d in import_review(ws) if d.source_code.startswith("TLD")).status == "UNCHECKED"
    _decide(client, ws, "TLD (TDF/3TC/DTG)", "reject")
    assert next(d for d in import_review(ws) if d.source_code.startswith("TLD")).status == "REJECTED"


def test_command_line_stages_refuse_to_run_during_another_run(site):
    from typer.testing import CliRunner

    from omop_agent.cli import app as cli
    from omop_agent.runlock import run_lock

    _, ws, *_ = site
    cfg = str(ws.root.parent / "omop_agent.yaml")
    with run_lock(ws, "web run 20261001T000000-abcdef"):
        result = CliRunner().invoke(cli, ["map", "-c", cfg])
    assert result.exit_code == 3 and "another run of this project is in progress" in result.output


def test_sign_in_input_is_bounded_and_the_log_is_private(site):
    client, _, users, data = site
    r = login(client, "x" * 100_000, "whatever-password")
    assert r.status_code == 401
    log = (data / "access.jsonl").read_text()
    assert "xxxxxxxxxx" not in log and "(not a user name)" in log
    assert oct((data / "access.jsonl").stat().st_mode & 0o777) == "0o600"
    assert login(client, "olive", "p" * 5000).status_code == 401


def test_signing_out_ends_the_users_other_sessions(site):
    client, _, users, data = site
    other = TestClient(client.app)
    login(client, "rita")
    login(other, "rita")
    assert other.get("/", follow_redirects=False).status_code == 200
    client.post("/logout", data={"csrf_token": _csrf(client, "/")})
    assert other.get("/", follow_redirects=False).status_code == 303


def test_one_persons_typos_do_not_lock_others_out():
    from omop_agent.web.security import LoginThrottle

    t = LoginThrottle()
    for _ in range(5):
        t.failure(*t.keys("rita", "10.0.0.5"))
    assert t.blocked(*t.keys("rita", "10.0.0.5"))
    assert not t.blocked(*t.keys("rita", "10.0.0.9"))  # rita from her own machine
    assert not t.blocked(*t.keys("olive", "10.0.0.5"))  # others on the shared machine
    for i in range(30):
        t.failure(*t.keys(f"user{i}", "10.0.0.66"))
    assert t.blocked(*t.keys("olive", "10.0.0.66"))  # one machine trying many accounts


def test_worker_waits_briefly_for_the_run_lock(site):
    import threading

    from omop_agent.runlock import run_lock
    from omop_agent.web.jobs import Job
    from omop_agent.web.worker import run_job
    from omop_agent.workspace import utcnow

    _, ws, *_ = site
    store = JobStore(ws)
    job = Job(id=f"{utcnow():%Y%m%dT%H%M%S}-abc123", kind="continue", by="olive",
              config_path=str(ws.root.parent / "omop_agent.yaml"), created_at=utcnow())
    store.save(job)
    held, release = threading.Event(), threading.Event()

    def hold():
        with run_lock(ws, "someone checking"):
            held.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    held.wait(5)
    threading.Timer(1.0, release.set).start()
    done = run_job(store.path(job.id))
    t.join()
    assert "Another run" not in done.message  # it waited, then ran (and failed on the missing database)


def test_config_errors_do_not_echo_secrets(tmp_path, monkeypatch):
    from omop_agent.web.projects import list_projects

    monkeypatch.setenv("SECRET_URL", "postgresql://omop:Sup3rS3cret@db/omop")
    d = tmp_path / "p" / "bad"
    d.mkdir(parents=True)
    (d / "omop_agent.yaml").write_text("name: bad\ntarget: {url: ${SECRET_URL}, cdm_schema: [not, a, name]}\n")
    [proj] = list_projects(tmp_path / "p")
    assert proj.error and "Sup3rS3cret" not in proj.error


def test_reviewer_credit_follows_the_edit(site):
    client, ws, *_ = site
    login(client, "rita")
    _decide(client, ws, "Paracetamol 500mg", "approve")
    first = next(d for d in import_review(ws) if d.source_code == "Paracetamol 500mg")
    export_review(ws, import_review(ws))  # a pipeline run folds the edit into decisions.json
    time.sleep(1.1)
    _decide(client, ws, "Paracetamol 500mg", "flag")
    second = next(d for d in import_review(ws) if d.source_code == "Paracetamol 500mg")
    assert second.reviewed_by == "rita" and second.reviewed_at > first.reviewed_at
    export_review(ws, import_review(ws))
    # A spreadsheet edit that changes the status without a new stamp is not credited to rita.
    path = ws.mappings_dir / "SRC_DRUG.csv"
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["source_code"] == "Paracetamol 500mg":
            r["status"] = "REJECTED"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    third = next(d for d in import_review(ws) if d.source_code == "Paracetamol 500mg")
    assert third.status == "REJECTED" and third.reviewed_by is None


def test_a_row_deleted_from_the_csv_can_still_be_reviewed(site):
    client, ws, *_ = site
    path = ws.mappings_dir / "SRC_DRUG.csv"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(line for line in lines if not line.startswith("UNCHECKED,SRC_DRUG,TLD")))
    login(client, "rita")
    r = _decide(client, ws, "TLD (TDF/3TC/DTG)", "reject")
    assert r.status_code == 303
    assert next(d for d in import_review(ws) if d.source_code.startswith("TLD")).status == "REJECTED"


def test_record_approval_refuses_a_stale_digest(site):
    from omop_agent.approvals import ApprovalConflict, artifact_digest

    _, ws, *_ = site
    digest = artifact_digest(ws, "vocab_map")
    from omop_agent.vocab.review import update_review_row

    update_review_row(ws, "SRC_DRUG", "TLD (TDF/3TC/DTG)", {"status": "FLAGGED"})
    with pytest.raises(ApprovalConflict):
        record_approval(ws, "vocab_map", "rita", expected_digest=digest)
    a = record_approval(ws, "vocab_map", "rita", expected_digest=artifact_digest(ws, "vocab_map"))
    assert gate_state(ws, "vocab_map").current and a.by == "rita"


def test_privacy_review_page(site):
    from omop_agent.models import ColumnProfile, SourceProfile, TableProfile, ValueFreq
    from omop_agent.workspace import utcnow

    client, ws, *_ = site
    login(client, "rita")
    assert "Not needed" in client.get("/p/demo/privacy").text  # this project's provider is none
    cfg_path = ws.root.parent / "omop_agent.yaml"
    cfg_path.write_text(cfg_path.read_text().replace("llm: {provider: none}", "llm: {provider: anthropic}"))
    ws.save_json(ws.profile_json, SourceProfile(source_type="csv", generated_at=utcnow(), min_cell_count=5, tables=[
        TableProfile(name="va", n_rows=100, columns=[
            ColumnProfile(name="deceased_name", inferred_type="string", n_rows=100, n_null=0, n_distinct=80,
                          unique=False, is_pii=True, pii_reason="the column name suggests a person's name"),
            ColumnProfile(name="cause", inferred_type="string", n_rows=100, n_null=0, n_distinct=2, unique=False,
                          top_values=[ValueFreq(value="Malaria", count=60)]),
        ])]))
    page = client.get("/p/demo/privacy").text
    assert "deceased_name" in page and "a person&#39;s name" in page and "Malaria" in page
    assert "Check what is sent to Claude" in client.get("/p/demo").text
    digest = re.search(r'name="expected_digest" value="([^"]+)"', page).group(1)
    client.post("/p/demo/privacy/approve", data={"csrf_token": _csrf(client, "/p/demo/privacy"),
                                                  "expected_digest": digest, "note": "checked the VA columns"})
    g = gate_state(ws, "privacy")
    assert g.current and g.approval.by == "rita"
    client.cookies.clear()
    login(client, "vera")
    assert "Approve the privacy review" not in client.get("/p/demo/privacy").text
