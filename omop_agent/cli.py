"""Command-line interface: ``omop-agent <command>``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="AI agent for automated OMOP CDM ETL: profile -> design -> map vocabularies -> build -> check/repair -> report.",
)
console = Console()

ConfigOpt = typer.Option(Path("omop_agent.yaml"), "--config", "-c", help="Project config file.")


def _pipeline(config: Path, yes: bool = True, recorded: bool = False):
    from .approvals import recorded_approver
    from .pipeline import Pipeline

    cfg = load_config(config)

    def approver(stage: str, message: str, paths: list[str]) -> bool:
        if yes:
            return True
        console.print(f"\n[bold yellow]Review gate: {stage}[/] {message}")
        for p in paths:
            console.print(f"  • {p}")
        return typer.confirm("Continue with these artifacts?", default=True)

    p = Pipeline(cfg, approver=approver, log=lambda m: console.print(m, highlight=False))
    if recorded:
        p.approver = recorded_approver(p.ws)
    return p


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Where to write omop_agent.yaml."),
    name: str = typer.Option("my-omop-project", help="Project name."),
    source_type: str = typer.Option("csv", help="csv or database."),
    source: str = typer.Option("./data", help="CSV directory, or SQLAlchemy URL for a database source."),
    db_url: Optional[str] = typer.Option(
        None,
        help="Target PostgreSQL URL (${VAR} / ${VAR:-default} are read from the environment). Default: "
        "${OMOP_AGENT_DB_URL} when that variable is set (as in the Docker bundle), else a local server.",
    ),
    athena_dir: str = typer.Option("./vocab", help="Unzipped Athena vocabulary download."),
) -> None:
    """Write a starter omop_agent.yaml."""
    import os

    if db_url is None:
        db_url = (
            "${OMOP_AGENT_DB_URL}" if os.environ.get("OMOP_AGENT_DB_URL")
            else "postgresql://${PGUSER:-postgres}:${PGPASSWORD:-}@localhost:5432/omop"
        )
    source_type = source_type.strip().lower()
    if source_type not in {"csv", "database"}:
        raise typer.BadParameter("--source-type must be 'csv' or 'database'")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "omop_agent.yaml"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists")

    def q(value: str) -> str:  # a YAML-safe (JSON) double-quoted scalar
        return json.dumps(value, ensure_ascii=False)

    src = (
        f"  type: csv\n  path: {q(source)}\n"
        if source_type == "csv"
        else f"  type: database\n  url: {q(source)}\n  schema: public\n"
    )
    path.write_text(
        f"""# omop-agent project configuration. ${{VAR}} and ${{VAR:-default}} are read from the environment.
name: {q(name)}
workdir: ./work

source:
{src}
target:
  url: {q(db_url)}
  cdm_schema: cdm
  vocab_schema: cdm
  staging_schema: src
  etl_schema: etl
  results_schema: results
  # etl_role: omop_etl        # optional least-privilege role for running the generated SQL

vocabulary:
  athena_dir: {q(athena_dir)}

llm:
  provider: anthropic          # anthropic | replay | none
  model: claude-opus-5-5
  effort: high
  # record_dir: ./recordings   # save live answers for offline replay

privacy:
  min_cell_count: 5            # values seen fewer times never reach the LLM
  send_values_to_llm: true
  pii_columns: []              # extra "table.column" identifiers to withhold

mapping:
  auto_approve_confidence: 0.9
  use_unchecked_in_etl: true

quality:
  max_repair_iterations: 3
  blocking_severities: [fatal]
  use_r_dqd: false             # also run the official Achilles + DataQualityDashboard (needs R)

cdm_source:
  cdm_source_name: {q(name)}
  cdm_source_abbreviation: {q(name[:25])}
  cdm_holder: ""
  source_description: ""
"""
    )
    load_config(path)  # fail now, not on the first command, if the file is not valid
    console.print(f"Wrote {path}")


@app.command()
def run(
    config: Path = ConfigOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't stop at review gates."),
    fresh: bool = typer.Option(False, help="Start over: re-stage, re-profile, redraft and remap everything."),
    redesign: bool = typer.Option(False, help="Redraft the ETL spec (the current one is backed up)."),
    remap: bool = typer.Option(False, help="Map every source code again instead of keeping reviewed decisions."),
    recorded_approvals: bool = typer.Option(
        False,
        "--recorded-approvals",
        help="Don't ask at review gates: continue only where a reviewer has approved the current "
        "artifacts (in the web app or with `omop-agent approve`), otherwise stop.",
    ),
) -> None:
    """Run the whole pipeline (stops for review after design and mapping unless --yes).

    Re-running resumes: existing artifacts, including a hand-edited etl_spec.yaml and reviewed
    mappings, are kept unless --fresh / --redesign / --remap say otherwise."""
    from .pipeline import PipelineStopped, SpecInvalid

    if yes and recorded_approvals:
        raise typer.BadParameter("use either --yes or --recorded-approvals, not both")
    p = _pipeline(config, yes, recorded=recorded_approvals)
    try:
        report = p.run_all(fresh=fresh, redesign=redesign, remap=remap)
    except PipelineStopped as e:
        console.print(f"[yellow]{e}[/]")
        raise typer.Exit(2)
    except SpecInvalid as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"\n[bold {'green' if report.passed else 'red'}]{'PASSED' if report.passed else 'FAILED'}[/] -> {p.ws.report_html}")
    raise typer.Exit(0 if report.passed else 1)


@app.command()
def profile(config: Path = ConfigOpt) -> None:
    """Stage the source data and write the scan report (profile.md)."""
    p = _pipeline(config)
    try:
        p.stage_and_profile()
    finally:
        p.close()


@app.command()
def annotate(config: Path = ConfigOpt) -> None:
    """Ask the LLM to interpret the profile (annotations.json)."""
    p = _pipeline(config)
    try:
        p.annotate()
    finally:
        p.close()


@app.command("vocab-load")
def vocab_load(config: Path = ConfigOpt, force: bool = typer.Option(False, help="Reload even if the version matches.")) -> None:
    """Load the Athena vocabulary download."""
    p = _pipeline(config)
    try:
        p.load_vocabulary(force=force)
    finally:
        p.close()


@app.command("vocab-download")
def vocab_download(
    config: Path = ConfigOpt,
    url_env: str = typer.Option("ATHENA_DOWNLOAD_URL", help="Environment variable holding Athena's download link."),
    dest: Optional[Path] = typer.Option(None, help="Where to unpack (default: vocabulary.athena_dir from the config)."),
) -> None:
    """Download and unpack the Athena bundle from the link Athena e-mailed you (the link is never printed)."""
    from .vocab.download import fetch_athena

    cfg = load_config(config)
    target = dest or cfg.vocabulary.athena_dir
    if target is None:
        raise typer.BadParameter("set vocabulary.athena_dir in the config or pass --dest")
    try:
        fetch_athena(Path(target), url_env, log=lambda m: console.print(m, highlight=False))
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"Next: omop-agent vocab-load -c {config}")


@app.command()
def design(config: Path = ConfigOpt) -> None:
    """Draft the ETL specification (etl_spec.yaml / etl_spec.md)."""
    p = _pipeline(config)
    try:
        p.design()
    finally:
        p.close()


@app.command("map")
def map_codes(
    config: Path = ConfigOpt,
    remap: bool = typer.Option(False, help="Map every term again instead of keeping earlier/reviewed decisions."),
) -> None:
    """Evaluate the vocabulary and map source codes (mappings/*.csv)."""
    p = _pipeline(config)
    try:
        p.map_vocabulary(remap=remap)
    finally:
        p.close()


@app.command("apply-mappings")
def apply_mappings(config: Path = ConfigOpt) -> None:
    """Load reviewed mappings/*.csv edits into source_to_concept_map."""
    p = _pipeline(config)
    try:
        n = p.apply_reviewed_mappings()
        console.print(f"{n} mappings written")
    finally:
        p.close()


@app.command()
def build(
    config: Path = ConfigOpt,
    repair: bool = typer.Option(True, help="Run quality checks and the repair loop after building."),
) -> None:
    """Compile the spec to SQL and load the CDM."""
    from .pipeline import SpecInvalid

    p = _pipeline(config)
    try:
        if repair:
            report = p.build_check_repair()
            raise typer.Exit(0 if report.passed else 1)
        result = p.build()
        raise typer.Exit(0 if result.ok else 1)
    except SpecInvalid as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    finally:
        p.close()


@app.command()
def check(config: Path = ConfigOpt) -> None:
    """Run the data-quality checks on the loaded CDM."""
    p = _pipeline(config)
    try:
        report = p.check()
        table = Table("check", "severity", "violated", "%", "threshold")
        for r in report.failed()[:40]:
            table.add_row(r.key, r.severity, str(r.num_violated_rows), f"{100 * r.pct_violated_rows:.1f}", f"{r.threshold_pct:g}")
        if report.failed():
            console.print(table)
        raise typer.Exit(0 if report.passed else 1)
    finally:
        p.close()


@app.command()
def report(
    config: Path = ConfigOpt,
    shareable: bool = typer.Option(False, help="Also write report.shareable.* with rare source terms withheld."),
) -> None:
    """Write the inspection report (report.md / report.html)."""
    p = _pipeline(config)
    try:
        p.report(shareable=shareable)
    finally:
        p.close()


@app.command()
def status(config: Path = ConfigOpt) -> None:
    """Show stage status and the review gates."""
    cfg = load_config(config)
    from .approvals import GATES, gate_state
    from .workspace import Workspace

    ws = Workspace(cfg.workdir)
    table = Table("stage", "status", "at")
    for stage, info in ws.state().get("stages", {}).items():
        table.add_row(stage, info.get("status", ""), info.get("at", ""))
    console.print(table)
    gates = Table("review gate", "state", "approved by", "at")
    for stage in GATES:
        g = gate_state(ws, stage)
        a = g.approval
        gates.add_row(stage, g.label, a.by if a else "", a.at.isoformat(timespec="seconds") if a else "")
    console.print(gates)


@app.command()
def approve(
    stage: str = typer.Argument(..., help="design (the ETL spec) or vocab_map (the code mappings)."),
    by: str = typer.Option(..., "--by", help="Reviewer's name, recorded with the approval."),
    note: str = typer.Option("", help="Optional comment kept with the approval."),
    config: Path = ConfigOpt,
) -> None:
    """Record that a reviewer approved the current ETL spec or code mappings.

    `omop-agent run --recorded-approvals` (and runs started from the web app) continue past a
    review gate only while what was approved is unchanged."""
    from .approvals import GATES, record_approval
    from .workspace import Workspace

    stage = {"mappings": "vocab_map", "spec": "design"}.get(stage, stage)
    if stage not in GATES:
        raise typer.BadParameter(f"stage must be one of: {', '.join(GATES)}")
    ws = Workspace(load_config(config).workdir)
    try:
        a = record_approval(ws, stage, by, note)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"{GATES[stage]} approved by {a.by} at {a.at.isoformat(timespec='seconds')}")


@app.command()
def demo(
    directory: Path = typer.Argument(Path("omop-demo"), help="Where to create the demo project."),
    db_url: str = typer.Option(..., "--db-url", envvar="OMOP_AGENT_DB_URL", help="PostgreSQL URL (schemas demo_* are created)."),
    provider: str = typer.Option("replay", help="replay (offline, recorded answers) or anthropic (live Claude)."),
    people: int = typer.Option(1500, help="Synthetic population size."),
    schema_prefix: str = typer.Option("demo", help="Schemas are named <prefix>_cdm, <prefix>_vocab, ..."),
    run_pipeline: bool = typer.Option(True, "--run/--no-run", help="Run the pipeline after creating the project."),
) -> None:
    """Create and run a demo on synthetic HDSS + clinic data with a synthetic vocabulary."""
    from .demo import create_demo_project

    cfg_path = create_demo_project(directory, db_url, provider=provider, n_people=people, schema_prefix=schema_prefix)
    console.print(f"Demo project written to {directory} (config {cfg_path})")
    if run_pipeline:
        p = _pipeline(cfg_path, yes=True)
        report = p.run_all(fresh=True)
        console.print(f"\n[bold {'green' if report.passed else 'red'}]{'PASSED' if report.passed else 'FAILED'}[/] -> {p.ws.report_html}")
        raise typer.Exit(0 if report.passed else 1)


DataDirOpt = typer.Option(
    Path("omop-agent-data"), "--data-dir", envvar="OMOP_AGENT_DATA",
    help="Where the web app keeps its accounts, session key and sign-in log.",
)


@app.command()
def serve(
    projects: Path = typer.Option(
        Path("projects"), "--projects", envvar="OMOP_AGENT_PROJECTS",
        help="Directory with one sub-directory (containing omop_agent.yaml) per project.",
    ),
    data_dir: Path = DataDirOpt,
    host: str = typer.Option("127.0.0.1", help="Address to listen on. Put a TLS reverse proxy in front for network use."),
    port: int = typer.Option(8000, help="Port to listen on."),
    secure_cookies: bool = typer.Option(
        False, "--secure-cookies/--no-secure-cookies", envvar="OMOP_AGENT_SECURE_COOKIES",
        help="Send the session cookie over HTTPS only (turn on when served behind HTTPS).",
    ),
    session_hours: float = typer.Option(8, envvar="OMOP_AGENT_SESSION_HOURS", help="Sign users out after this long."),
    forwarded_allow_ips: str = typer.Option(
        "127.0.0.1", envvar="FORWARDED_ALLOW_IPS",
        help="Reverse proxies trusted to report the client address (X-Forwarded-For).",
    ),
) -> None:
    """Start the review web app: review the ETL design and code mappings, run the pipeline, read reports."""
    import os

    try:
        import uvicorn

        from .web.app import create_app
    except ImportError as e:
        console.print(f"[red]The web app needs extra packages ({e.name}): pip install 'omop-agent[web]'[/]")
        raise typer.Exit(1)
    from .web.users import UserStore

    web = create_app(
        projects, data_dir, secure_cookies=secure_cookies, session_hours=session_hours,
        secret_key=os.environ.get("OMOP_AGENT_WEB_SECRET"),
    )
    if not UserStore(data_dir).list():
        console.print(f"[yellow]No accounts yet. Create one: omop-agent users add NAME --role operator --data-dir {data_dir}[/]")
    console.print(f"Serving projects from {projects.resolve()} on http://{host}:{port}")
    uvicorn.run(
        web, host=host, port=port, proxy_headers=True, forwarded_allow_ips=forwarded_allow_ips,
        server_header=False, log_level="info",
    )


users_app = typer.Typer(no_args_is_help=True, help="Manage the web app's accounts (run on the server).")
app.add_typer(users_app, name="users")


def _password(password_stdin: bool) -> str:
    import sys

    if password_stdin:
        return sys.stdin.readline().rstrip("\n")
    return typer.prompt("Password", hide_input=True, confirmation_prompt=True)


def _users(data_dir: Path):
    from .web.users import UserStore

    return UserStore(data_dir)


def _user_change(fn, *args) -> None:
    from .web.users import UserError

    try:
        user = fn(*args)
    except UserError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    if user is not None:
        console.print(f"{user.name}: {user.role}{'' if user.active else ' (disabled)'}")


@users_app.command("add")
def users_add(
    name: str = typer.Argument(..., help="Sign-in name (lower case)."),
    role: str = typer.Option("reviewer", help="viewer, reviewer or operator."),
    full_name: str = typer.Option("", help="Name shown in the app and on approvals."),
    password_stdin: bool = typer.Option(False, help="Read the password from standard input instead of prompting."),
    data_dir: Path = DataDirOpt,
) -> None:
    """Create an account."""
    _user_change(_users(data_dir).add, name, _password(password_stdin), role, full_name)


@users_app.command("list")
def users_list(data_dir: Path = DataDirOpt) -> None:
    """List the accounts."""
    table = Table("name", "full name", "role", "active", "created")
    for u in _users(data_dir).list():
        table.add_row(u.name, u.full_name, u.role, "yes" if u.active else "no", u.created_at)
    console.print(table)


@users_app.command("passwd")
def users_passwd(
    name: str,
    password_stdin: bool = typer.Option(False, help="Read the password from standard input instead of prompting."),
    data_dir: Path = DataDirOpt,
) -> None:
    """Set a new password (signs the account out everywhere)."""
    _user_change(_users(data_dir).set_password, name, _password(password_stdin))


@users_app.command("set-role")
def users_set_role(name: str, role: str = typer.Argument(..., help="viewer, reviewer or operator."), data_dir: Path = DataDirOpt) -> None:
    """Change an account's role (signs it out everywhere)."""
    _user_change(_users(data_dir).set_role, name, role)


@users_app.command("disable")
def users_disable(name: str, data_dir: Path = DataDirOpt) -> None:
    """Disable an account (signs it out at once; its approvals stay in the audit trail)."""
    _user_change(_users(data_dir).set_active, name, False)


@users_app.command("enable")
def users_enable(name: str, data_dir: Path = DataDirOpt) -> None:
    """Re-enable a disabled account."""
    _user_change(_users(data_dir).set_active, name, True)


@app.command("show-spec")
def show_spec(config: Path = ConfigOpt) -> None:
    """Print the current ETL spec summary."""
    cfg = load_config(config)
    from .models import ETLSpec
    from .workspace import Workspace

    ws = Workspace(cfg.workdir)
    spec = ws.load_yaml(ws.spec_yaml, ETLSpec)
    console.print_json(json.dumps({m.target_table: m.source.table for m in spec.table_mappings}))


def main(argv: Optional[list[str]] = None) -> None:
    app(argv)


if __name__ == "__main__":
    main()
