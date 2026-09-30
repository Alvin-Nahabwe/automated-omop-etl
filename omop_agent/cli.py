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


def _pipeline(config: Path, yes: bool = True):
    from .pipeline import Pipeline

    cfg = load_config(config)

    def approver(stage: str, message: str, paths: list[str]) -> bool:
        if yes:
            return True
        console.print(f"\n[bold yellow]Review gate: {stage}[/] {message}")
        for p in paths:
            console.print(f"  • {p}")
        return typer.confirm("Continue with these artifacts?", default=True)

    return Pipeline(cfg, approver=approver, log=lambda m: console.print(m, highlight=False))


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Where to write omop_agent.yaml."),
    name: str = typer.Option("my-omop-project", help="Project name."),
    source_type: str = typer.Option("csv", help="csv or database."),
    source: str = typer.Option("./data", help="CSV directory, or SQLAlchemy URL for a database source."),
    db_url: str = typer.Option("postgresql://${PGUSER}:${PGPASSWORD}@localhost:5432/omop", help="Target PostgreSQL URL."),
    athena_dir: str = typer.Option("./vocab", help="Unzipped Athena vocabulary download."),
) -> None:
    """Write a starter omop_agent.yaml."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "omop_agent.yaml"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists")
    src = f"  type: csv\n  path: {source}\n" if source_type == "csv" else f"  type: database\n  url: {source}\n  schema: public\n"
    path.write_text(
        f"""# omop-agent project configuration. ${{VAR}} and ${{VAR:-default}} are read from the environment.
name: {name}
workdir: ./work

source:
{src}
target:
  url: {db_url}
  cdm_schema: cdm
  vocab_schema: cdm
  staging_schema: src
  etl_schema: etl
  results_schema: results

vocabulary:
  athena_dir: {athena_dir}

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
  cdm_source_name: {name}
  cdm_source_abbreviation: {name[:25]}
  cdm_holder: ""
  source_description: ""
"""
    )
    console.print(f"Wrote {path}")


@app.command()
def run(
    config: Path = ConfigOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't stop at review gates."),
    skip_staging: bool = typer.Option(False, help="Reuse the staged source and profile."),
) -> None:
    """Run the whole pipeline (stops for review after design and mapping unless --yes)."""
    from .pipeline import PipelineStopped

    p = _pipeline(config, yes)
    try:
        report = p.run_all(skip_staging=skip_staging)
    except PipelineStopped as e:
        console.print(f"[yellow]{e}[/]")
        raise typer.Exit(2)
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
    p = _pipeline(config)
    try:
        if repair:
            report = p.build_check_repair()
            raise typer.Exit(0 if report.passed else 1)
        result = p.build()
        raise typer.Exit(0 if result.ok else 1)
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
def report(config: Path = ConfigOpt) -> None:
    """Write the inspection report (report.md / report.html)."""
    p = _pipeline(config)
    try:
        p.report()
    finally:
        p.close()


@app.command()
def status(config: Path = ConfigOpt) -> None:
    """Show stage status."""
    cfg = load_config(config)
    from .workspace import Workspace

    ws = Workspace(cfg.workdir)
    table = Table("stage", "status", "at")
    for stage, info in ws.state().get("stages", {}).items():
        table.add_row(stage, info.get("status", ""), info.get("at", ""))
    console.print(table)


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
        report = p.run_all()
        console.print(f"\n[bold {'green' if report.passed else 'red'}]{'PASSED' if report.passed else 'FAILED'}[/] -> {p.ws.report_html}")


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
