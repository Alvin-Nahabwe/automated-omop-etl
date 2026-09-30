"""The orchestrator: runs the ETL pipeline stage by stage, with review gates and the repair loop.

::

    stage source ─ profile ─ interpret ─ load vocabulary ─ design ─┐
                                                                  ▼  (review etl_spec.yaml)
                          evaluate vocabulary ─ map source codes ─┐
                                                                  ▼  (review mappings/*.csv)
                              build ─ quality ──fail──▶ repair ─▶ build ...
                                         └─pass──▶ report

Each stage reads the artifacts of the previous ones from the workspace, so any stage can be
re-run on its own (``omop-agent <stage>``) after a human edits an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import psycopg

from .cdm.ddl import create_cdm_tables, create_etl_tables
from .config import ProjectConfig
from .db import Schemas, connect
from .llm import LLMClient, make_llm
from .models import DQReport, ETLRunResult, ETLSpec, MappingDecision, SourceAnnotations, SourceProfile
from .workspace import Workspace

# Called at review gates with (stage, message, artifact paths); returns True to continue.
Approver = Callable[[str, str, list[str]], bool]


def auto_approve(stage: str, message: str, paths: list[str]) -> bool:
    return True


class PipelineStopped(Exception):
    """Raised when a reviewer declines to continue at a gate."""


@dataclass
class Pipeline:
    cfg: ProjectConfig
    approver: Approver = auto_approve
    log: Callable[[str], None] = print
    ws: Workspace = field(init=False)
    schemas: Schemas = field(init=False)
    llm: LLMClient = field(init=False)
    _conn: psycopg.Connection | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.ws = Workspace(self.cfg.workdir)
        self.schemas = Schemas.from_config(self.cfg.target)
        self.llm = make_llm(self.cfg.llm, self.ws)

    # ---- plumbing ----------------------------------------------------------------------
    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = connect(self.cfg.target.url)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    def _gate(self, stage: str, message: str, paths: list[str]) -> None:
        ok = self.approver(stage, message, paths)
        self.ws.audit("human_approval", stage=stage, approved=ok, artifacts=paths)
        if not ok:
            self.ws.mark(stage, "awaiting_review")
            raise PipelineStopped(f"Stopped for review at '{stage}'. Edit {', '.join(paths)} and re-run.")

    def _run_stage(self, stage: str, fn: Callable[[], object]) -> object:
        self.log(f"▶ {stage}")
        self.ws.mark(stage, "running")
        try:
            result = fn()
        except Exception as e:
            self.ws.mark(stage, "failed", error=str(e)[:500])
            self.ws.audit("stage_failed", stage=stage, error=str(e)[:2000])
            raise
        self.ws.mark(stage, "done")
        return result

    # ---- artifacts ---------------------------------------------------------------------
    def load_profile(self) -> SourceProfile:
        return self.ws.load_json(self.ws.profile_json, SourceProfile)

    def load_annotations(self) -> SourceAnnotations | None:
        p = self.ws.annotations_json
        return self.ws.load_json(p, SourceAnnotations) if p.exists() else None

    def load_spec(self) -> ETLSpec:
        return self.ws.load_yaml(self.ws.spec_yaml, ETLSpec)

    # ---- stages ------------------------------------------------------------------------
    def stage_and_profile(self) -> SourceProfile:
        from .profiling.loader import stage_source
        from .profiling.profiler import profile_staging
        from .profiling.render import render_profile_md

        def run() -> SourceProfile:
            manifest = stage_source(self.cfg, self.conn, self.schemas, self.ws)
            self.log(f"  staged {len(manifest)} tables into schema '{self.schemas.src}'")
            profile = profile_staging(self.conn, self.schemas, self.cfg, manifest)
            self.ws.save_json(self.ws.profile_json, profile)
            self.ws.profile_md.write_text(render_profile_md(profile))
            n_pii = sum(c.is_pii for t in profile.tables for c in t.columns)
            self.log(f"  profiled {len(profile.tables)} tables; {n_pii} identifier columns withheld from the LLM")
            return profile

        return self._run_stage("profile", run)  # type: ignore[return-value]

    def annotate(self) -> SourceAnnotations:
        from .profiling.annotate import annotate

        def run() -> SourceAnnotations:
            ann = annotate(self.load_profile(), self.cfg, self.llm)
            self.ws.save_json(self.ws.annotations_json, ann)
            for note in ann.notes:
                self.log(f"  note: {note}")
            return ann

        return self._run_stage("annotate", run)  # type: ignore[return-value]

    def load_vocabulary(self, force: bool = False) -> None:
        from .vocab.loader import load_athena, vocab_loaded

        def run() -> None:
            create_cdm_tables(self.conn, self.schemas, include_vocab=True)
            create_etl_tables(self.conn, self.schemas)
            self.conn.commit()
            if self.cfg.vocabulary.athena_dir:
                counts = load_athena(self.conn, self.schemas, self.cfg.vocabulary.athena_dir, self.ws, force=force)
                self.log(f"  vocabulary: {counts.get('concept', 'already loaded')} concepts")
            elif not vocab_loaded(self.conn, self.schemas):
                raise RuntimeError(
                    "No vocabulary loaded. Download vocabularies from https://athena.ohdsi.org and set "
                    "vocabulary.athena_dir, or load them into the vocab schema yourself."
                )

        self._run_stage("vocab_load", run)

    def design(self) -> ETLSpec:
        from .design.designer import design_etl

        def run() -> ETLSpec:
            spec = design_etl(
                self.load_profile(), self.load_annotations(), self.cfg, self.llm, self.conn, self.schemas, self.ws
            )
            self.log(f"  designed {len(spec.table_mappings)} table mappings -> {self.ws.spec_yaml.name}")
            return spec

        spec = self._run_stage("design", run)
        self._gate(
            "design",
            "Review the ETL design (etl_spec.md). Edit etl_spec.yaml to change it.",
            [str(self.ws.spec_md), str(self.ws.spec_yaml)],
        )
        return spec  # type: ignore[return-value]

    def map_vocabulary(self, remap: bool = False) -> list[MappingDecision]:
        """Evaluate the vocabulary and map source codes.

        Decisions from an earlier run, including a reviewer's CSV edits, are kept for terms that
        are still present; only new terms are mapped. ``remap=True`` maps everything again.
        """
        from .vocab.evaluate import evaluate_vocabulary, render_vocab_evaluation_md
        from .vocab.mapper import collect_terms, map_terms, mapping_summary, write_source_to_concept_map
        from .vocab.review import export_review, import_review

        def run() -> list[MappingDecision]:
            spec = self.load_spec()
            ev = evaluate_vocabulary(self.conn, self.schemas, spec, self.cfg)
            self.ws.save_json(self.ws.vocab_eval_json, ev)
            (self.ws.root / "vocab_evaluation.md").write_text(render_vocab_evaluation_md(ev))
            for issue in ev.issues:
                self.log(f"  vocabulary: {issue}")
            terms = collect_terms(self.conn, self.schemas, spec)
            prior: dict[tuple[str, str], MappingDecision] = {}
            if not remap and self.ws.decisions_json.exists():
                prior = {(d.source_vocabulary_id, d.source_code): d for d in import_review(self.ws, self.conn, self.schemas)}
            freq = {(t.source_vocabulary_id, t.source_code): t.frequency for t in terms}
            kept = [d.model_copy(update={"frequency": freq[k]}) for k, d in prior.items() if k in freq]
            new_terms = [t for t in terms if (t.source_vocabulary_id, t.source_code) not in prior]
            if kept:
                self.log(f"  kept {len(kept)} earlier/reviewed decisions; mapping {len(new_terms)} new terms")
            decisions = kept + (map_terms(self.conn, self.schemas, new_terms, self.llm, self.cfg, self.ws) if new_terms else [])
            decisions.sort(key=lambda d: (d.source_vocabulary_id, -d.frequency, d.source_code))
            export_review(self.ws, decisions)
            n = write_source_to_concept_map(
                self.conn, self.schemas, decisions, include_unchecked=self.cfg.mapping.use_unchecked_in_etl
            )
            self.conn.commit()
            for row in mapping_summary(decisions):
                self.log(
                    f"  {row['source_vocabulary_id']}: {row['terms']} terms, {row['approved']} approved, "
                    f"{row['unchecked']} unchecked, {row['unmapped']} unmapped"
                )
            self.log(f"  wrote {n} rows to {self.schemas.etl}.source_to_concept_map")
            return decisions

        decisions = self._run_stage("vocab_map", run)
        self._gate(
            "vocab_map",
            "Review the code mappings in mappings/*.csv (set status/target_concept_id), then continue.",
            [str(self.ws.mappings_dir)],
        )
        return decisions  # type: ignore[return-value]

    def apply_reviewed_mappings(self) -> int:
        """Re-import edited mapping CSVs into source_to_concept_map."""
        from .vocab.mapper import write_source_to_concept_map
        from .vocab.review import import_review

        decisions = import_review(self.ws, self.conn, self.schemas)
        n = write_source_to_concept_map(
            self.conn, self.schemas, decisions, include_unchecked=self.cfg.mapping.use_unchecked_in_etl
        )
        self.conn.commit()
        self.ws.audit("mappings_reimported", rows=n)
        return n

    def build(self, spec: ETLSpec | None = None) -> ETLRunResult:
        from .etl.runner import run_etl

        def run() -> ETLRunResult:
            result = run_etl(self.conn, self.schemas, spec or self.load_spec(), self.cfg, self.ws)
            loaded = sum(s.rows_inserted for s in result.steps if s.error is None)
            self.log(f"  {len(result.steps)} steps, {loaded} rows loaded, {len(result.errors())} errors")
            for s in result.errors():
                self.log(f"  ✗ {s.step}: {s.error}")
            return result

        return self._run_stage("build", run)  # type: ignore[return-value]

    def check(self, spec: ETLSpec | None = None, etl_result: ETLRunResult | None = None) -> DQReport:
        from .quality.checks import gate, run_builtin_checks
        from .quality.dqd_r import r_available, run_r_dqd

        def run() -> DQReport:
            spec_ = spec or self.load_spec()
            etl = etl_result or (
                self.ws.load_json(self.ws.etl_run_json, ETLRunResult) if self.ws.etl_run_json.exists() else None
            )
            report = run_builtin_checks(self.conn, self.schemas, self.cfg, spec_, etl)
            if self.cfg.quality.use_r_dqd:
                if r_available(self.cfg):
                    r_report = run_r_dqd(self.cfg, self.ws)
                    if r_report is not None:
                        results = report.results + r_report.results
                        passed, blocking = gate(results, self.cfg, etl)
                        report = report.model_copy(
                            update={"results": results, "passed": passed, "blocking_failures": blocking}
                        )
                else:
                    self.log("  R DataQualityDashboard not available; built-in checks only")
            self.ws.save_json(self.ws.dq_json, report)
            failed = report.failed()
            self.log(
                f"  {len(report.results)} checks, {len(failed)} failed "
                f"({len(report.blocking_failures)} blocking) -> {'PASS' if report.passed else 'FAIL'}"
            )
            return report

        return self._run_stage("quality", run)  # type: ignore[return-value]

    def build_check_repair(self) -> DQReport:
        """Build, check, and let the repair agent fix the spec until the gate passes."""
        from .quality.repair import repair_spec

        spec = self.load_spec()
        profile = self.load_profile() if self.ws.profile_json.exists() else None
        max_iter = self.cfg.quality.max_repair_iterations
        iteration = 0
        while True:
            etl = self.build(spec)
            if etl.ok:
                report = self.check(spec, etl)
                if report.passed and not self._repairable(report):
                    return report
            else:
                report = None
            if iteration >= max_iter:
                self.log(f"  stopping after {max_iter} repair iterations")
                return report if report is not None else self.check(spec, etl)
            iteration += 1
            self.log(f"▶ repair (iteration {iteration})")
            spec, actions = repair_spec(
                spec, iteration=iteration, dq=report, etl_result=etl, llm=self.llm, conn=self.conn,
                schemas=self.schemas, cfg=self.cfg, ws=self.ws, profile=profile,
            )
            for a in actions:
                self.log(f"  ✎ {a.target_table}.{a.target_field or '*'}: {a.change}")
            if not actions:
                self.log("  repair agent made no changes")
                return report if report is not None else self.check(spec, etl)

    def _repairable(self, report: DQReport) -> bool:
        sev = set(self.cfg.quality.repair_severities)
        return any((r.failed or r.error) and r.severity in sev and not r.not_applicable for r in report.results)

    def report(self) -> None:
        from .report.inspection import build_report

        def run() -> None:
            md, html = build_report(self.cfg, self.ws, self.conn, self.schemas)
            self.log(f"  inspection report: {md} and {html}")

        self._run_stage("report", run)

    # ---- everything --------------------------------------------------------------------
    def run_all(self, skip_staging: bool = False) -> DQReport:
        self.ws.audit(
            "pipeline_start", project=self.cfg.name, llm=self.cfg.llm.provider, model=self.cfg.llm.model
        )
        try:
            if not skip_staging or not self.ws.profile_json.exists():
                self.stage_and_profile()
            self.annotate()
            self.load_vocabulary()
            self.design()
            self.map_vocabulary()
            report = self.build_check_repair()
            self.report()
            self.ws.audit("pipeline_end", passed=report.passed)
            return report
        finally:
            self.close()
