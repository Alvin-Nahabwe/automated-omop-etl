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
from starlette.middleware.sessions import SessionMiddleware

from ..approvals import GATE_PHRASES, GATES, approved_snapshot, gate_state, mapping_changes, record_approval
from ..db import Schemas, connect
from ..models import ConceptCandidate, DQReport, ETLSpec, MappingDecision
from ..vocab.review import import_review, update_review_row
from ..workspace import utcnow
from .jobs import KIND_LABELS, KINDS, JobBusy, JobStore
from .projects import Project, get_project, list_projects
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
from .users import ROLES, User, UserStore

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
        with (data_dir / "access.jsonl").open("a") as fh:
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

    def redirect(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    # ---- errors ------------------------------------------------------------------------
    @app.exception_handler(LoginRequired)
    async def login_required(request: Request, exc: LoginRequired):
        return redirect("/login?" + urlencode({"next": exc.next_path}))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        titles = {403: "Not allowed", 404: "Not found", 409: "Project not usable"}
        return render(
            request, "error.html.j2", status_code=exc.status_code,
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
    async def login_page(request: Request, next: str = "/"):
        if current_user(request):
            return redirect(safe_next(next))
        return render(request, "login.html.j2", next=safe_next(next), no_users=not users.list())

    @app.post("/login")
    async def login(request: Request):
        form = await form_of(request)
        name = form.get("username", "").strip().lower()
        ip = request.client.host if request.client else "?"
        keys = (f"user:{name}", f"ip:{ip}")
        nxt = safe_next(form.get("next"))
        if throttle.blocked(*keys):
            access_log("login_blocked", user=name, ip=ip)
            return render(
                request, "login.html.j2", status_code=429, next=nxt, username=name,
                error="Too many failed sign-ins. Wait 15 minutes and try again.",
            )
        user = users.authenticate(name, form.get("password", ""))
        if user is None:
            throttle.failure(*keys)
            access_log("login_failed", user=name, ip=ip)
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
        user = current_user(request)
        request.session.clear()
        if user:
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
    async def home(request: Request):
        user = need(request)
        return render(
            request, "projects.html.j2", user=user, projects=[summary(p) for p in list_projects(projects_dir)],
            projects_dir=str(projects_dir),
        )

    @app.get("/p/{pid}")
    async def project_page(request: Request, pid: str):
        user = need(request)
        proj = project_or_404(pid)
        ctx = summary(proj)
        if proj.cfg is not None:
            jobs = JobStore(proj.ws)
            ctx["jobs"] = [(j, jobs.status(j)) for j in jobs.list(limit=15)]
            ctx["counts"] = _mapping_counts(proj)
            order = STAGE_ORDER + [s for s in ctx["stages"] if s not in STAGE_ORDER]
            ctx["stage_rows"] = [(s, ctx["stages"].get(s)) for s in order]
        return render(request, "project.html.j2", user=user, tab="overview", **ctx)

    def _mapping_counts(proj: Project) -> dict[str, int] | None:
        if not proj.ws.decisions_json.exists():
            return None
        return _counts(import_review(proj.ws))

    # ---- ETL design --------------------------------------------------------------------
    @app.get("/p/{pid}/spec")
    async def spec_page(request: Request, pid: str):
        user = need(request)
        proj = configured(project_or_404(pid))
        ws = proj.ws
        ctx: dict[str, Any] = {"project": proj, "gate": gate_state(ws, "design"), "spec": None, "spec_error": None}
        if ws.spec_yaml.exists():
            text = ws.spec_yaml.read_text()
            ctx["spec_text"] = text
            try:
                ctx["spec"] = ETLSpec.model_validate(yaml.safe_load(text))
            except (ValidationError, yaml.YAMLError) as e:
                ctx["spec_error"] = str(e)[:3000]
            approved = approved_snapshot(ws, "design")
            if approved is not None and not ctx["gate"].current:
                ctx["diff"] = list(
                    difflib.unified_diff(
                        approved.splitlines(), text.splitlines(), "approved", "current", lineterm="", n=2
                    )
                )
        issues_path = ws.root / "design_issues.json"
        ctx["issues"] = json.loads(issues_path.read_text()) if issues_path.exists() else []
        ctx["repairs"] = json.loads(ws.repairs_json.read_text()) if ws.repairs_json.exists() else []
        return render(request, "spec.html.j2", user=user, tab="spec", **ctx)

    @app.post("/p/{pid}/spec/approve")
    async def approve_spec(request: Request, pid: str):
        user = need(request, "reviewer")
        form = await form_of(request)
        proj = configured(project_or_404(pid))
        return _approve_gate(request, proj, "design", user, form, f"/p/{pid}/spec")

    def _approve_gate(request: Request, proj: Project, stage: str, user: User, form: dict, back: str):
        # Approve only what the reviewer saw: refuse if it changed after the page was loaded.
        if form.get("expected_digest") != gate_state(proj.ws, stage).digest:
            flash(request, "error", f"The {GATE_PHRASES[stage]} changed after you opened the page. Review it again.")
            return redirect(back)
        try:
            record_approval(proj.ws, stage, user.name, form.get("note", "")[:NOTE_MAX])
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
    async def mappings_page(
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
        with vocab_conn(proj) as conn:
            decisions = import_review(ws, conn, schemas) if conn is not None else import_review(ws)
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
                        found = search_candidates(conn, schemas, q.strip()[:200], open_term.target_domain, limit=15)
                        search_results = [c for c in found if c.score >= SEARCH_MIN_SCORE]
                    except Exception as e:  # e.g. the vocabulary is not loaded yet
                        search_error = f"Search failed: {type(e).__name__}"
        gate = gate_state(ws, "vocab_map")
        busy = JobStore(ws).active() is not None
        return render(
            request, "mappings.html.j2", user=user, project=proj, decisions=shown, total=len(rows), counts=counts,
            vocabs=vocabs, vocab=vocab, filter=filter, filters=MAPPING_FILTERS, page=page, pages=pages,
            open_term=open_term, q=q, search_results=search_results, search_error=search_error, gate=gate,
            changes=mapping_changes(ws) if gate.approval and not gate.current else [], busy=busy,
            choices=lambda d: _choices(d, search_results if d is open_term else None),
            db_ok=conn is not None, anchor=term_anchor, tab="mappings", include_unchecked=proj.cfg.mapping.use_unchecked_in_etl,  # type: ignore[union-attr]
        )

    @app.post("/p/{pid}/mappings/decide")
    async def decide(request: Request, pid: str):
        user = need(request, "reviewer")
        form = await form_of(request)
        proj = configured(project_or_404(pid))
        ws = proj.ws
        svid, code = form.get("vocab_id", ""), form.get("code", "")
        back_q = {k: form.get(f"back_{k}", "") for k in ("filter", "vocab", "page")}
        back = f"/p/{pid}/mappings?" + urlencode({k: v for k, v in back_q.items() if v}) + "#" + term_anchor(svid, code)
        if JobStore(ws).active() is not None:
            flash(request, "error", "A pipeline run is in progress. Mapping edits are paused until it finishes.")
            return redirect(back)
        action = form.get("action", "")
        if action not in {"approve", "flag", "reject"}:
            raise HTTPException(400, "Unknown action.")
        schemas = Schemas.from_config(proj.cfg.target)  # type: ignore[union-attr]
        with vocab_conn(proj) as conn:
            decisions = import_review(ws, conn, schemas) if conn is not None else import_review(ws)
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
                            from ..vocab.mapper import check_concept

                            concept, why = check_concept(conn, schemas, target, d.target_domain)
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
        update_review_row(ws, svid, code, changes)
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
        user = need(request, "reviewer")
        form = await form_of(request)
        proj = configured(project_or_404(pid))
        return _approve_gate(request, proj, "vocab_map", user, form, f"/p/{pid}/mappings")

    # ---- runs --------------------------------------------------------------------------
    @app.post("/p/{pid}/runs")
    async def start_run(request: Request, pid: str):
        user = need(request, "operator")
        form = await form_of(request)
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
    async def run_page(request: Request, pid: str, job_id: str):
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
        user = need(request, "operator")
        await form_of(request)
        proj = configured(project_or_404(pid))
        if JobStore(proj.ws).cancel(job_id):
            proj.ws.audit("web_run_stop_requested", job=job_id, by=user.name)
            flash(request, "ok", "Asked the run to stop.")
        else:
            flash(request, "error", "That run is not running.")
        return redirect(f"/p/{pid}/runs/{job_id}")

    # ---- report and audit --------------------------------------------------------------
    @app.get("/p/{pid}/report")
    async def report(request: Request, pid: str, version: str = "full"):
        need(request)
        proj = configured(project_or_404(pid))
        path = proj.ws.root / ("report.shareable.html" if version == "shareable" else "report.html")
        if not path.exists():
            raise HTTPException(404, "There is no inspection report yet: it is written at the end of a run.")
        return FileResponse(path, media_type="text/html", headers={"Content-Security-Policy": REPORT_CSP})

    @app.get("/p/{pid}/audit")
    async def audit(request: Request, pid: str):
        user = need(request)
        proj = configured(project_or_404(pid))
        events = []
        for e in reversed(proj.ws.read_audit()[-400:]):
            details = {k: v for k, v in e.items() if k not in {"ts", "event"} | AUDIT_HIDDEN}
            text = json.dumps(details, ensure_ascii=False, default=str)
            events.append({"ts": e.get("ts"), "event": e.get("event"), "by": e.get("by"), "details": text[:400]})
        return render(request, "audit.html.j2", user=user, project=proj, events=events, tab="audit")

    return app


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
