"""The review web app (FastAPI, server-rendered pages, no JavaScript needed).

Screens: projects; a project's stages, review gates and runs; the ETL design with an approve
button and a diff against the approved version; the code mappings, with Claude's reasoning,
the candidates and a vocabulary search, and approve / flag / reject per term; run logs; the
inspection report; the audit trail.

Everything a reviewer does is written to the same files the command line uses (the mapping
CSVs, ``approvals.json``) and to the project's audit log with the reviewer's name.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from ..approvals import (
    GATE_PHRASES,
    GATES,
    ApprovalConflict,
    approved_snapshot,
    gate_state,
    mapping_changes,
    record_approval,
)
from ..db import Schemas, connect
from ..design.render import build_rules
from ..etl.compiler import key_sources
from ..models import ConceptCandidate, DQReport, ETLSpec, MappingDecision
from ..runlock import RunInProgress, run_lock
from ..vocab.review import import_review, update_review_row
from ..workspace import utcnow
from .jobs import KIND_LABELS, KINDS, JobBusy, JobStore
from .projects import Project, get_project, list_projects, schema_conflicts_for
from .security import (
    APP_CSP,
    BASE_HEADERS,
    REPORT_CSP,
    LoginThrottle,
    csrf_ok,
    csrf_token,
    load_secret_key,
    safe_next,
)
from .users import ROLES, USERNAME, User, UserStore

HERE = Path(__file__).parent
STAGE_ORDER = ["profile", "annotate", "vocab_load", "design", "vocab_map", "build", "quality", "report"]
STAGE_LABELS = {
    "profile": "Profile the source",
    "annotate": "Interpret the profile",
    "vocab_load": "Load the vocabulary",
    "design": "Design the ETL",
    "vocab_map": "Map source codes",
    "build": "Build the CDM",
    "quality": "Quality checks",
    "report": "Inspection report",
}
MAPPING_FILTERS = {
    "review": "Needs review",
    "unmapped": "No concept",
    "approved": "Approved",
    "rejected": "Rejected",
    "all": "All",
}
PAGE_SIZE = 100
SEARCH_MIN_SCORE = 0.1  # weaker lexical matches are noise in the reviewer's list
NOTE_MAX = 1000
PASSWORD_MAX = 1024
# Audit fields too large or too raw to show in the browser.
AUDIT_HIDDEN = {"system", "prompt", "response", "transcript", "tool_calls", "sql"}


class LoginRequired(Exception):
    def __init__(self, next_path: str):
        self.next_path = next_path


def _fmt_dt(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _status_cls(status: str | None) -> str:
    s = (status or "").lower()
    if s in {"done", "passed", "approved", "running"}:
        return "ok"
    if s in {"failed", "failed_quality", "rejected", "interrupted", "cancelled"}:
        return "fail"
    if s in {"awaiting_review", "waiting_review", "changed since approval", "awaiting review", "flagged", "unchecked", "queued"}:
        return "warn"
    return "muted"


def term_anchor(svid: str, code: str) -> str:
    return "t-" + hashlib.sha1(f"{svid}\x00{code}".encode()).hexdigest()[:12]


def create_app(
    projects_dir: Path,
    data_dir: Path,
    secure_cookies: bool = False,
    session_hours: float = 8,
    secret_key: str | None = None,
) -> FastAPI:
    projects_dir, data_dir = Path(projects_dir), Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    users = UserStore(data_dir)
    throttle = LoginThrottle()
    session_s = int(session_hours * 3600)

    env = Environment(loader=FileSystemLoader(str(HERE / "templates")), autoescape=True)
    env.filters["dt"] = _fmt_dt
    env.filters["statuscls"] = _status_cls
    env.globals.update(STAGE_LABELS=STAGE_LABELS, GATES=GATES, KIND_LABELS=KIND_LABELS, ROLES=ROLES)
    templates = Jinja2Templates(env=env)

    app = FastAPI(title="omop-agent", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        SessionMiddleware,
        secret_key=load_secret_key(data_dir, secret_key),
        session_cookie="omop_agent_session",
        max_age=session_s,
        same_site="strict",
        https_only=secure_cookies,
    )

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response: Response = await call_next(request)
        for k, v in BASE_HEADERS.items():
            response.headers.setdefault(k, v)
        response.headers.setdefault("Content-Security-Policy", APP_CSP)
        if not request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-store")
        if secure_cookies:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response

    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # ---- helpers -----------------------------------------------------------------------
    def access_log(event: str, **fields: Any) -> None:
        record = {"ts": utcnow().isoformat(), "event": event, **fields}
        fd = os.open(data_dir / "access.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def current_user(request: Request) -> User | None:
        s = request.session
        name = s.get("user")
        if not name:
            return None
        user = users.get(name)
        if (
            user is None
            or not user.active
            or s.get("epoch") != user.epoch
            or time.time() - float(s.get("iat", 0)) > session_s
        ):
            for k in ("user", "epoch", "iat"):
                s.pop(k, None)
            return None
        return user

    def need(request: Request, role: str = "viewer") -> User:
        user = current_user(request)
        if user is None:
            path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
            raise LoginRequired(path if request.method == "GET" else "/")
        if not user.can(role):
            raise HTTPException(403, f"This needs the {role} role; you are signed in as a {user.role}.")
        return user

    async def form_of(request: Request) -> dict[str, str]:
        form = await request.form()
        data = {k: v for k, v in form.items() if isinstance(v, str)}
        if not csrf_ok(request.session, data.get("csrf_token")):
            raise HTTPException(403, "This form has expired or did not come from this site. Reload the page and try again.")
        return data

    def flash(request: Request, kind: str, message: str) -> None:
        request.session.setdefault("flash", []).append([kind, message])

    def render(request: Request, name: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
        ctx.setdefault("user", current_user(request))
        ctx["csrf"] = csrf_token(request.session)
        ctx["flashes"] = request.session.pop("flash", [])
        return templates.TemplateResponse(request, name, ctx, status_code=status_code)

    def project_or_404(pid: str) -> Project:
        proj = get_project(projects_dir, pid)
        if proj is None:
            raise HTTPException(404, "No such project.")
        return proj

    def configured(proj: Project) -> Project:
        if proj.cfg is None:
            raise HTTPException(409, f"The project's omop_agent.yaml could not be read: {proj.error}")
        return proj

    @contextmanager
    def vocab_conn(proj: Project) -> Iterator[Any]:
        """A read-only connection to the project's database, or None if it is not reachable."""
        try:
            conn = connect(proj.cfg.target.url, connect_timeout=5)  # type: ignore[union-attr]
            conn.read_only = True
        except Exception:
            yield None
            return
        try:
            yield conn
        finally:
            conn.close()

    def spec_domains(proj: Project) -> dict | None:
        """Per source vocabulary, the domains the project's current spec accepts (``import_review``
        refreshes the decisions with them first, as the command line does, so an edit is checked
        against the spec as it is now); None without a readable spec."""
        from ..models import SourceAnnotations, SourceProfile
        from ..privacy import apply_overrides
        from ..vocab.mapper import source_vocabulary_domains

        ws = proj.ws
        if not ws.spec_yaml.exists():
            return None
        try:
            spec = ws.load_yaml(ws.spec_yaml, ETLSpec)
            profile = None
            if ws.profile_json.exists():
                ann = ws.load_json(ws.annotations_json, SourceAnnotations) if ws.annotations_json.exists() else None
                profile = apply_overrides(ws.load_json(ws.profile_json, SourceProfile), proj.cfg.privacy, ann)  # type: ignore[union-attr]
            return source_vocabulary_domains(spec, profile, proj.cfg.privacy)  # type: ignore[union-attr]
        except (OSError, ValueError, yaml.YAMLError):  # pydantic's ValidationError is a ValueError
            return None

    def redirect(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    # ---- errors ------------------------------------------------------------------------
    @app.exception_handler(LoginRequired)
    async def login_required(request: Request, exc: LoginRequired):
        return redirect("/login?" + urlencode({"next": exc.next_path}))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        titles = {403: "Not allowed", 404: "Not found", 409: "Project not usable"}
        return await run_in_threadpool(
            render, request, "error.html.j2", status_code=exc.status_code,
            title=titles.get(exc.status_code, "Something went wrong"), message=str(exc.detail),
        )

    # ---- sign in / out -----------------------------------------------------------------
    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return PlainTextResponse("ok")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(HERE / "static" / "favicon.svg", media_type="image/svg+xml")

    @app.get("/login")
    def login_page(request: Request, next: str = "/"):
        if current_user(request):
            return redirect(safe_next(next))
        return render(request, "login.html.j2", next=safe_next(next), no_users=not users.list())

    @app.post("/login")
    async def login(request: Request):
        form = await form_of(request)
        return await run_in_threadpool(_login, request, form)

    def _login(request: Request, form: dict[str, str]):
        typed = form.get("username", "").strip().lower()
        name = typed if USERNAME.match(typed) else ""  # anything else is never logged or echoed
        password = form.get("password", "")
        ip = request.client.host if request.client else "?"
        keys = throttle.keys(name or "?", ip)
        nxt = safe_next(form.get("next"))
        if throttle.blocked(*keys):
            access_log("login_blocked", user=name or "(not a user name)", ip=ip)
            return render(
                request, "login.html.j2", status_code=429, next=nxt, username=name,
                error="Too many failed sign-ins. Wait 15 minutes and try again.",
            )
        user = users.authenticate(name or "-", password[:PASSWORD_MAX])
        if user is None or len(password) > PASSWORD_MAX:
            throttle.failure(*keys)
            access_log("login_failed", user=name or "(not a user name)", ip=ip)
            return render(
                request, "login.html.j2", status_code=401, next=nxt, username=name,
                error="Wrong user name or password.",
            )
        throttle.success(*keys)
        request.session.clear()
        request.session.update({"user": user.name, "epoch": user.epoch, "iat": time.time()})
        csrf_token(request.session)
        access_log("login", user=user.name, ip=ip)
        return redirect(nxt)

    @app.post("/logout")
    async def logout(request: Request):
        await form_of(request)
        return await run_in_threadpool(_logout, request)

    def _logout(request: Request):
        user = current_user(request)
        request.session.clear()
        if user:
            users.sign_out_everywhere(user.name)  # a copied session cookie stops working too
            access_log("logout", user=user.name, ip=request.client.host if request.client else "?")
        return redirect("/login")

    # ---- projects ----------------------------------------------------------------------
    def summary(proj: Project) -> dict[str, Any]:
        out: dict[str, Any] = {"project": proj}
        if proj.cfg is None:
            return out
        ws = proj.ws
        stages = ws.state().get("stages", {})
        out["stages"] = stages
        from ..privacy_review import refresh, sends_to_llm

        out["sends_data"] = sends_to_llm(proj.cfg)
        if out["sends_data"]:
            try:
                refresh(proj.cfg, ws)
            except (OSError, ValueError):  # an unreadable profile: the privacy page says why
                pass
        out["gates"] = {g: gate_state(ws, g) for g in GATES}
        jobs = JobStore(ws)
        active = jobs.active()
        out["active_job"] = active
        out["last_job"] = next(iter(jobs.list(limit=1)), None)
        out["has_report"] = ws.report_html.exists()
        out["dq"] = None
        if ws.dq_json.exists():
            try:
                out["dq"] = ws.load_json(ws.dq_json, DQReport)
            except (ValidationError, ValueError):
                pass
        return out

    @app.get("/")
    def home(request: Request):
        user = need(request)
        return render(
            request, "projects.html.j2", user=user, projects=[summary(p) for p in list_projects(projects_dir)],
            projects_dir=str(projects_dir),
        )

    @app.get("/p/{pid}")
    def project_page(request: Request, pid: str):
        user = need(request)
        proj = project_or_404(pid)
        ctx = summary(proj)
        if proj.cfg is not None:
            jobs = JobStore(proj.ws)
            ctx["jobs"] = [(j, jobs.status(j)) for j in jobs.list(limit=15)]
            ctx["counts"] = _mapping_counts(proj)
            order = STAGE_ORDER + [s for s in ctx["stages"] if s not in STAGE_ORDER]
            ctx["stage_rows"] = [(s, ctx["stages"].get(s)) for s in order]
            ctx["conflicts"] = schema_conflicts_for(proj.dir, proj.cfg)
            holder = jobs.holder()
            ctx["other_run"] = holder if holder is not None and ctx["active_job"] is None else None
        return render(request, "project.html.j2", user=user, tab="overview", **ctx)

    def _mapping_counts(proj: Project) -> dict[str, int] | None:
        if not proj.ws.decisions_json.exists():
            return None
        return _counts(import_review(proj.ws))

    # ---- privacy review ----------------------------------------------------------------
    @app.get("/p/{pid}/privacy")
    def privacy_page(request: Request, pid: str):
        from ..privacy_review import build_review, refresh, sends_to_llm

        user = need(request)
        proj = configured(project_or_404(pid))
        ws = proj.ws
        sends = sends_to_llm(proj.cfg)  # type: ignore[arg-type]
        review, problem = None, None
        try:
            if sends and refresh(proj.cfg, ws) is not None:  # type: ignore[arg-type]
                review = build_review(proj.cfg, ws)  # type: ignore[arg-type]
        except (OSError, ValueError) as e:
            problem = f"The profile or the annotations could not be read ({type(e).__name__}). Profile the source again."
        return render(
            request, "privacy.html.j2", user=user, project=proj, tab="privacy", sends=sends, review=review,
            problem=problem, gate=gate_state(ws, "privacy"),
        )

    @app.post("/p/{pid}/privacy/approve")
    async def approve_privacy(request: Request, pid: str):
        form = await form_of(request)
        return await run_in_threadpool(_approve_gate, request, pid, "privacy", form, f"/p/{pid}/privacy")

    # ---- ETL design --------------------------------------------------------------------
    @app.get("/p/{pid}/spec")
    def spec_page(request: Request, pid: str):
        user = need(request)
        proj = configured(project_or_404(pid))
        ws = proj.ws
        ctx: dict[str, Any] = {"project": proj, "gate": gate_state(ws, "design"), "spec": None, "spec_error": None,
                               "key_sources": {"person": [], "visit": []}, "rules": []}
        if ws.spec_yaml.exists():
            text = ws.spec_yaml.read_text()
            ctx["spec_text"] = text
            try:
                ctx["spec"] = ETLSpec.model_validate(yaml.safe_load(text))
            except (ValidationError, yaml.YAMLError) as e:
                ctx["spec_error"] = str(e)[:3000]
            else:  # the other mappings whose keys go into the person and visit id maps
                ctx["key_sources"] = {
                    what: [{"table": tm.source.table, "expression": expr} for tm, expr, _ in key_sources(ctx["spec"], t)]
                    for what, t in (("person", "person"), ("visit", "visit_occurrence"))
                }
                ctx["rules"] = build_rules(ctx["spec"])  # one row per key, periods, routing, eras
            approved = approved_snapshot(ws, "design")
            if approved is not None and not ctx["gate"].current:
                ctx["diff"] = list(
                    difflib.unified_diff(
                        approved.splitlines(), text.splitlines(), "approved", "current", lineterm="", n=2
                    )
                )
        issues_path = ws.root / "design_issues.json"
        ctx["issues"] = _json_list(issues_path)
        ctx["repairs"] = _json_list(ws.repairs_json)
        return render(request, "spec.html.j2", user=user, tab="spec", **ctx)

    @app.post("/p/{pid}/spec/approve")
    async def approve_spec(request: Request, pid: str):
        form = await form_of(request)
        return await run_in_threadpool(_approve_gate, request, pid, "design", form, f"/p/{pid}/spec")

    def _approve_gate(request: Request, pid: str, stage: str, form: dict, back: str):
        user = need(request, "reviewer")
        proj = configured(project_or_404(pid))
        try:  # approve only what the reviewer saw
            record_approval(
                proj.ws, stage, user.name, form.get("note", "")[:NOTE_MAX], expected_digest=form.get("expected_digest", "")
            )
        except ApprovalConflict:
            flash(request, "error", f"The {GATE_PHRASES[stage]} changed after you opened the page. Review it again.")
            return redirect(back)
        except FileNotFoundError as e:
            flash(request, "error", str(e))
            return redirect(back)
        msg = f"{GATES[stage]} approved."
        if user.can("operator"):
            msg += " You can now continue the pipeline from the project page."
        else:
            msg += " An operator can now continue the pipeline."
        flash(request, "ok", msg)
        return redirect(back)

    # ---- code mappings -----------------------------------------------------------------
    @app.get("/p/{pid}/mappings")
    def mappings_page(
        request: Request, pid: str, filter: str = "review", vocab: str = "", page: int = 1, term: str = "", q: str = ""
    ):
        user = need(request)
        proj = configured(project_or_404(pid))
        ws = proj.ws
        if not ws.decisions_json.exists():
            return render(request, "mappings.html.j2", user=user, project=proj, decisions=None, tab="mappings")
        filter = filter if filter in MAPPING_FILTERS else "review"
        schemas = Schemas.from_config(proj.cfg.target)  # type: ignore[union-attr]
        search_results = None
        search_error = None
        domains = spec_domains(proj)
        with vocab_conn(proj) as conn:
            decisions = import_review(ws, conn, schemas, domains=domains)
            vocabs = sorted({d.source_vocabulary_id for d in decisions})
            in_vocab = [d for d in decisions if not vocab or d.source_vocabulary_id == vocab]
            counts = _counts(in_vocab)
            rows = sorted(
                (d for d in in_vocab if _matches(d, filter)),
                key=lambda d: (d.source_vocabulary_id, -d.frequency, d.source_code),
            )
            pages = max(1, -(-len(rows) // PAGE_SIZE))
            page = min(max(1, page), pages)
            shown = rows[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
            open_term = next((d for d in shown if term_anchor(d.source_vocabulary_id, d.source_code) == term), None)
            if open_term is not None and q.strip():
                if conn is None:
                    search_error = "The vocabulary database cannot be reached, so it cannot be searched."
                else:
                    from ..vocab.mapper import search_candidates

                    try:
                        found = search_candidates(conn, schemas, q.strip()[:200], open_term.target_domain, limit=15,
                                                  also=open_term.accepted_domains)
                        search_results = [c for c in found if c.score >= SEARCH_MIN_SCORE]
                    except Exception as e:  # e.g. the vocabulary is not loaded yet
                        search_error = f"Search failed: {type(e).__name__}"
        gate = gate_state(ws, "vocab_map")
        return render(
            request, "mappings.html.j2", user=user, project=proj, decisions=shown, total=len(rows), counts=counts,
            vocabs=vocabs, vocab=vocab, filter=filter, filters=MAPPING_FILTERS, page=page, pages=pages,
            open_term=open_term, q=q, search_results=search_results, search_error=search_error, gate=gate,
            changes=mapping_changes(ws) if gate.approval and not gate.current else [], busy=JobStore(ws).busy(),
            choices=lambda d: _choices(d, search_results if d is open_term else None),
            db_ok=conn is not None, anchor=term_anchor, tab="mappings", include_unchecked=proj.cfg.mapping.use_unchecked_in_etl,  # type: ignore[union-attr]
        )

    @app.post("/p/{pid}/mappings/decide")
    async def decide(request: Request, pid: str):
        form = await form_of(request)
        return await run_in_threadpool(_decide, request, pid, form)

    def _decide(request: Request, pid: str, form: dict[str, str]):
        user = need(request, "reviewer")
        proj = configured(project_or_404(pid))
        ws = proj.ws
        svid, code = form.get("vocab_id", ""), form.get("code", "")
        back_q = {k: form.get(f"back_{k}", "") for k in ("filter", "vocab", "page")}
        back = f"/p/{pid}/mappings?" + urlencode({k: v for k, v in back_q.items() if v}) + "#" + term_anchor(svid, code)
        action = form.get("action", "")
        if action not in {"approve", "flag", "reject"}:
            raise HTTPException(400, "Unknown action.")
        paused = "A pipeline run is in progress. Mapping edits are paused until it finishes."
        if JobStore(ws).active() is not None:
            flash(request, "error", paused)
            return redirect(back)
        try:
            # Hold the run lock while checking and saving, so no run starts in between and
            # overwrites the edit (runs from the command line take the same lock).
            with run_lock(ws, f"mapping edit by {user.name}", wait_s=2):
                return _save_decision(request, proj, user, form, svid, code, action, back)
        except RunInProgress:
            flash(request, "error", paused)
            return redirect(back)

    def _save_decision(request, proj: Project, user: User, form: dict, svid: str, code: str, action: str, back: str):
        ws = proj.ws
        schemas = Schemas.from_config(proj.cfg.target)  # type: ignore[union-attr]
        domains = spec_domains(proj)
        with vocab_conn(proj) as conn:
            decisions = import_review(ws, conn, schemas, domains=domains)
            d = next((x for x in decisions if x.source_vocabulary_id == svid and x.source_code == code), None)
            if d is None:
                raise HTTPException(404, "That source code is not in the mappings.")
            if form.get("expected_target") != str(d.target_concept_id) or form.get("expected_status") != d.status:
                who = f" by {d.reviewed_by}" if d.reviewed_by else ""
                flash(request, "error", f"'{code}' was changed{who} after you opened the page. Check it again before saving.")
                return redirect(back)
            changes: dict[str, str] = {
                "reviewed_by": user.name,
                "reviewed_at": utcnow().isoformat(timespec="seconds"),
                "review_note": form.get("note", "").strip()[:NOTE_MAX],
            }
            target = d.target_concept_id
            if action == "approve":
                try:
                    target = int(form.get("concept_id", str(d.target_concept_id)))
                except ValueError:
                    raise HTTPException(400, "The selected concept is not a number.")
                if target != d.target_concept_id:
                    if target == 0:
                        changes.update(target_concept_id="0", target_concept_name="", target_domain_id="", target_vocabulary_id="")
                    else:
                        concept = next((c for c in d.candidates if c.concept_id == target), None)
                        if conn is not None:
                            from ..vocab.mapper import accepted_domains, check_concept

                            concept, why = check_concept(conn, schemas, target, accepted_domains(d))
                            if concept is None:
                                flash(request, "error", f"Not saved: {why}.")
                                return redirect(back)
                        elif concept is None:
                            flash(request, "error", "Not saved: the vocabulary database cannot be reached to check that concept.")
                            return redirect(back)
                        changes.update(
                            target_concept_id=str(concept.concept_id), target_concept_name=concept.concept_name,
                            target_domain_id=concept.domain_id, target_vocabulary_id=concept.vocabulary_id,
                        )
                changes["status"] = "APPROVED"
            else:
                changes["status"] = {"flag": "FLAGGED", "reject": "REJECTED"}[action]
        try:
            update_review_row(ws, svid, code, changes)
        except KeyError:
            flash(request, "error", f"'{code}' is missing from the review files; run the mapping step again.")
            return redirect(back)
        ws.audit(
            "mapping_reviewed", by=user.name, source_vocabulary_id=svid, source_code=code, action=action,
            before={"target_concept_id": d.target_concept_id, "status": d.status},
            after={"target_concept_id": target, "status": changes["status"]}, note=changes["review_note"],
        )
        labels = {"APPROVED": "approved", "FLAGGED": "flagged for discussion", "REJECTED": "rejected"}
        flash(request, "ok", f"'{code}' {labels[changes['status']]}.")
        return redirect(back)

    @app.post("/p/{pid}/mappings/approve")
    async def approve_mappings(request: Request, pid: str):
        form = await form_of(request)
        return await run_in_threadpool(_approve_gate, request, pid, "vocab_map", form, f"/p/{pid}/mappings")

    # ---- runs --------------------------------------------------------------------------
    @app.post("/p/{pid}/runs")
    async def start_run(request: Request, pid: str):
        form = await form_of(request)
        return await run_in_threadpool(_start_run, request, pid, form)

    def _start_run(request: Request, pid: str, form: dict[str, str]):
        user = need(request, "operator")
        proj = configured(project_or_404(pid))
        kind = form.get("kind", "continue")
        if kind not in KINDS:
            raise HTTPException(400, "Unknown kind of run.")
        if kind != "continue" and form.get("confirm") != "yes":
            flash(request, "error", f"Tick the confirmation box to {KIND_LABELS[kind].lower()}.")
            return redirect(f"/p/{pid}")
        try:
            job = JobStore(proj.ws).start(kind, user.name, proj.config_path)
        except JobBusy as e:
            flash(request, "error", str(e).capitalize() + ".")
            return redirect(f"/p/{pid}")
        return redirect(f"/p/{pid}/runs/{job.id}")

    @app.get("/p/{pid}/runs/{job_id}")
    def run_page(request: Request, pid: str, job_id: str):
        user = need(request)
        proj = configured(project_or_404(pid))
        jobs = JobStore(proj.ws)
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No such run.")
        status = jobs.status(job)
        return render(
            request, "run.html.j2", user=user, project=proj, job=job, status=status,
            log=jobs.log_tail(job_id), live=status in ("queued", "running"), tab="overview",
        )

    @app.post("/p/{pid}/runs/{job_id}/stop")
    async def stop_run(request: Request, pid: str, job_id: str):
        await form_of(request)
        return await run_in_threadpool(_stop_run, request, pid, job_id)

    def _stop_run(request: Request, pid: str, job_id: str):
        user = need(request, "operator")
        proj = configured(project_or_404(pid))
        if JobStore(proj.ws).cancel(job_id):
            proj.ws.audit("web_run_stop_requested", job=job_id, by=user.name)
            flash(request, "ok", "Asked the run to stop.")
        else:
            flash(request, "error", "That run is not running.")
        return redirect(f"/p/{pid}/runs/{job_id}")

    # ---- report and audit --------------------------------------------------------------
    @app.get("/p/{pid}/report")
    def report(request: Request, pid: str, version: str = "full"):
        need(request)
        proj = configured(project_or_404(pid))
        path = proj.ws.root / ("report.shareable.html" if version == "shareable" else "report.html")
        if not path.exists():
            raise HTTPException(404, "There is no inspection report yet: it is written at the end of a run.")
        return FileResponse(path, media_type="text/html", headers={"Content-Security-Policy": REPORT_CSP})

    @app.get("/p/{pid}/audit")
    def audit(request: Request, pid: str):
        user = need(request)
        proj = configured(project_or_404(pid))
        events = []
        for e in reversed(proj.ws.read_audit()[-400:]):
            details = {k: v for k, v in e.items() if k not in {"ts", "event"} | AUDIT_HIDDEN}
            text = json.dumps(details, ensure_ascii=False, default=str)
            events.append({"ts": e.get("ts"), "event": e.get("event"), "by": e.get("by"), "details": text[:400]})
        return render(request, "audit.html.j2", user=user, project=proj, events=events, tab="audit")

    return app


def _json_list(path: Path) -> list:
    """A JSON list from ``path``; empty when missing or unreadable (a page must still render)."""
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _choices(d: MappingDecision, results: list[ConceptCandidate] | None) -> list[ConceptCandidate]:
    """Concepts a reviewer can pick for ``d``: search results, then the candidates, then the current one."""
    seen: set[int] = set()
    out: list[ConceptCandidate] = []
    for c in [*(results or []), *d.candidates]:
        if c.concept_id not in seen:
            seen.add(c.concept_id)
            out.append(c)
    if d.target_concept_id and d.target_concept_id not in seen:
        out.append(
            ConceptCandidate(
                concept_id=d.target_concept_id, concept_name=d.target_concept_name or "",
                domain_id=d.target_domain_id or "", vocabulary_id=d.target_vocabulary_id or "",
                concept_class_id="", standard_concept="S", concept_code="",
            )
        )
    return out


def _matches(d: MappingDecision, filter: str) -> bool:
    if filter == "review":
        return d.status in ("UNCHECKED", "FLAGGED")
    if filter == "unmapped":
        return d.target_concept_id == 0 and d.status != "REJECTED"
    if filter == "approved":
        return d.status == "APPROVED"
    if filter == "rejected":
        return d.status == "REJECTED"
    return True


def _counts(decisions: list[MappingDecision]) -> dict[str, int]:
    return {f: sum(1 for d in decisions if _matches(d, f)) for f in MAPPING_FILTERS}
