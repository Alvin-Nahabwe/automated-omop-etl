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

import json
from dataclasses import dataclass, field
from typing import Callable

import psycopg

from .cdm.ddl import create_cdm_tables, create_etl_tables
from .config import ProjectConfig
from .db import Schemas, connect
from .fsutil import atomic_write_text
from .llm import LLMClient, make_llm
from .models import DQReport, ETLRunResult, ETLSpec, MappingDecision, SourceAnnotations, SourceProfile, TableLoadResult
from .workspace import Workspace

# Called at review gates with (stage, message, artifact paths); returns True to continue. An
# approver may have a ``mode`` attribute, recorded with each gate: "interactive" (asked at the
# terminal), "recorded" (passes only with a recorded approval of the artifact as it is now),
# "skip" (--yes: gates are not checked) or "stage" (a single stage command: the design and mapping
# gates are not checked there, the privacy gate needs a recorded approval).
Approver = Callable[[str, str, list[str]], bool]


def auto_approve(stage: str, message: str, paths: list[str]) -> bool:
    return True


auto_approve.mode = "skip"  # type: ignore[attr-defined]


class PipelineStopped(Exception):
    """Raised when a reviewer declines to continue at a gate."""

    def __init__(self, message: str, stage: str | None = None):
        super().__init__(message)
        self.stage = stage


class SpecInvalid(Exception):
    """Raised when the ETL spec on disk has validation errors (e.g. after a hand edit)."""


@dataclass
class Pipeline:
    cfg: ProjectConfig
    approver: Approver = auto_approve
    log: Callable[[str], None] = print
    ws: Workspace = field(init=False)
    schemas: Schemas = field(init=False)
    llm: LLMClient = field(init=False)
    _conn: psycopg.Connection | None = field(default=None, init=False)
    _privacy_checked: bool = field(default=False, init=False)

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
        mode = getattr(self.approver, "mode", "custom")
        if mode == "stage":
            mode = "recorded" if stage == "privacy" else "stage"
        ok = self.approver(stage, message, paths)
        outcome = "stopped" if not ok else {"skip": "skipped", "stage": "not_checked"}.get(mode, "approved")
        self.ws.audit("review_gate", stage=stage, outcome=outcome, mode=mode, artifacts=paths)
        if outcome == "skipped" and stage == "privacy":
            self.log("  ! privacy review skipped (--yes): data goes to the model without a recorded review")
        if not ok:
            self.ws.mark(stage, "awaiting_review")
            if mode == "recorded":
                from .approvals import GATE_PHRASES

                raise PipelineStopped(
                    f"Waiting for a reviewer to approve the {GATE_PHRASES.get(stage, stage)} (in the web app, or "
                    f"`omop-agent approve {stage} --by NAME` after checking {', '.join(paths)}).",
                    stage,
                )
            raise PipelineStopped(f"Stopped for review at '{stage}'. Edit {', '.join(paths)} and re-run.", stage)

    def _privacy_gate(self) -> None:
        """Before any step that sends source data to the model: the privacy review must be approved."""
        from .privacy_review import REVIEW_MD, refresh, sends_to_llm

        if self._privacy_checked:
            return
        if not sends_to_llm(self.cfg):
            if self.ws.stage_status("privacy") == "awaiting_review":
                self.ws.mark("privacy", "done", note="prompts do not leave your own servers")
            return
        if not self.ws.profile_json.exists() or refresh(self.cfg, self.ws) is None:
            raise RuntimeError(
                "Profile the source first (`omop-agent profile`): the privacy review is built from the "
                "profile and must be approved before anything is sent to the model."
            )
        self._gate(
            "privacy",
            "Check what will be sent to the model: the withheld columns and the profile exactly as the model will see it.",
            [str(self.ws.root / REVIEW_MD)],
        )
        if self.ws.stage_status("privacy") != "done":
            self.ws.mark("privacy", "done")
        self._privacy_checked = True

    # A run that stops at a review gate is continued by a plain run, so what the operator asked
    # for (start over, redraft, map again) is kept here until the step has been done.
    def _pending(self) -> dict[str, bool]:
        try:
            raw = json.loads(self.ws.pending_json.read_text()) if self.ws.pending_json.exists() else {}
        except ValueError:
            raw = {}
        return {k: bool(v) for k, v in raw.items() if k in PENDING_STEPS and v}

    def _save_pending(self, flags: dict[str, bool]) -> None:
        flags = {k: True for k, v in flags.items() if v}
        if flags:
            atomic_write_text(self.ws.pending_json, json.dumps(flags, indent=2) + "\n")
        elif self.ws.pending_json.exists():
            self.ws.pending_json.unlink()

    def _done_pending(self, step: str) -> None:
        flags = self._pending()
        if flags.pop(step, False):
            self._save_pending(flags)

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
        """The profile with the current privacy decisions applied (privacy.pii_columns /
        not_pii_columns, and columns the model labelled as identifiers or free text)."""
        from .privacy import apply_overrides

        profile = self.ws.load_json(self.ws.profile_json, SourceProfile)
        return apply_overrides(profile, self.cfg.privacy, self.load_annotations())

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
            manifest = stage_source(self.cfg, self.conn, self.schemas, self.ws, strict=not self.cfg.source.skip_ragged_rows)
            self.log(f"  staged {len(manifest)} tables into schema '{self.schemas.src}'")
            profile = profile_staging(self.conn, self.schemas, self.cfg, manifest)
            self.ws.save_json(self.ws.profile_json, profile)
            self.ws.profile_md.write_text(render_profile_md(profile))
            n_pii = sum(c.is_pii for t in profile.tables for c in t.columns)
            self.log(f"  profiled {len(profile.tables)} tables; {n_pii} identifier columns withheld from the LLM")
            return profile

        profile = self._run_stage("profile", run)
        self._privacy_checked = False  # a new profile needs its own review
        self._done_pending("profile")
        return profile  # type: ignore[return-value]

    def annotate(self) -> SourceAnnotations:
        from .profiling.annotate import annotate

        self._privacy_gate()

        def run() -> SourceAnnotations:
            ann = annotate(self.load_profile(), self.cfg, self.llm)
            self.ws.save_json(self.ws.annotations_json, ann)
            for note in ann.notes:
                self.log(f"  note: {note}")
            return ann

        ann = self._run_stage("annotate", run)
        self._done_pending("annotate")
        return ann  # type: ignore[return-value]

    def load_vocabulary(self, force: bool = False) -> None:
        from .vocab.loader import ensure_indexes, load_athena, vocab_loaded

        def run() -> None:
            create_cdm_tables(self.conn, self.schemas, include_vocab=True)
            create_etl_tables(self.conn, self.schemas, self.cfg.target.etl_role)
            self.conn.commit()
            if self.cfg.vocabulary.athena_dir:
                counts = load_athena(self.conn, self.schemas, self.cfg.vocabulary.athena_dir, self.ws, force=force)
                self.log(f"  vocabulary: {counts.get('concept', 'already loaded')} concepts")
            elif not vocab_loaded(self.conn, self.schemas):
                raise RuntimeError(
                    "No vocabulary loaded. Download vocabularies from https://athena.ohdsi.org and set "
                    "vocabulary.athena_dir, or load them into the vocab schema yourself."
                )
            else:  # loaded by someone else: add the indexes the mapping and the build rely on
                skipped = ensure_indexes(self.conn, self.schemas)
                if skipped:
                    self.log(f"  vocabulary indexes not created (the vocabulary tables belong to another role, are "
                             f"views or are read-only): {', '.join(skipped)}"
                             + ("; the build finds the concepts to route by domain with a join of the concept table"
                                if "idx_concept_concept_id" in skipped else ""))
                    self.ws.audit("vocab_indexes_not_created", indexes=skipped)

        self._run_stage("vocab_load", run)

    def design(self, redraft: bool = True) -> ETLSpec:
        """Draft the ETL spec, or (``redraft=False``) keep the existing, possibly hand-edited one.

        Redrafting over an existing spec first backs it up as ``etl_spec.<n>.yaml``.
        """
        from .design.designer import design_etl
        from .design.render import render_spec_md

        if not redraft and self.ws.spec_yaml.exists():
            spec = self.load_spec()  # validates the (possibly hand-edited) YAML
            self.ws.spec_md.write_text(render_spec_md(spec))
            self.log(f"▶ design: using the existing {self.ws.spec_yaml.name} (pass --redesign to redraft)")
            self.ws.audit("design_reused", tables=[m.target_table for m in spec.table_mappings])
        else:
            self._privacy_gate()

            def run() -> ETLSpec:
                if self.ws.spec_yaml.exists():
                    n = 1
                    while (backup := self.ws.root / f"etl_spec.{n}.yaml").exists():
                        n += 1
                    backup.write_text(self.ws.spec_yaml.read_text())
                    self.log(f"  previous spec saved as {backup.name}")
                    self.ws.audit("spec_backup", path=backup.name)
                spec = design_etl(
                    self.load_profile(), self.load_annotations(), self.cfg, self.llm, self.conn, self.schemas, self.ws
                )
                self.log(f"  designed {len(spec.table_mappings)} table mappings -> {self.ws.spec_yaml.name}")
                return spec

            spec = self._run_stage("design", run)  # type: ignore[assignment]
            self._done_pending("redesign")
        self._gate(
            "design",
            "Review the ETL design (etl_spec.md). Edit etl_spec.yaml to change it.",
            [str(self.ws.spec_md), str(self.ws.spec_yaml)],
        )
        self.ws.mark("design", "done")
        return spec

    def map_vocabulary(self, remap: bool = False) -> list[MappingDecision]:
        """Evaluate the vocabulary and map source codes.

        Decisions from an earlier run, including a reviewer's CSV edits, are kept for terms that
        are still present; only new terms are mapped. ``remap=True`` maps everything again.
        """
        from .vocab.evaluate import evaluate_vocabulary, render_vocab_evaluation_md
        from .vocab.mapper import (
            collect_terms,
            map_terms,
            mapping_summary,
            pii_concept_fields,
            source_vocabulary_domains,
            write_source_to_concept_map,
        )
        from .vocab.review import export_review, import_review

        self._privacy_gate()

        def run() -> list[MappingDecision]:
            spec = self.load_spec()
            # the mapping's WHERE is read with the ETL helper functions (try_*): install this
            # version's, as the vocabulary load does (a project mapped again after an upgrade)
            create_etl_tables(self.conn, self.schemas, self.cfg.target.etl_role)
            self.conn.commit()
            ev = evaluate_vocabulary(self.conn, self.schemas, spec, self.cfg)
            self.ws.save_json(self.ws.vocab_eval_json, ev)
            (self.ws.root / "vocab_evaluation.md").write_text(render_vocab_evaluation_md(ev))
            for issue in ev.issues:
                self.log(f"  vocabulary: {issue}")
            profile = self.load_profile() if self.ws.profile_json.exists() else None
            for f in pii_concept_fields(spec, profile, self.cfg.privacy):
                self.log(f"  skipped {f}: direct identifiers are never mapped or loaded")
            terms = collect_terms(self.conn, self.schemas, spec, profile, self.cfg.privacy, self.ws, cfg=self.cfg)
            prior: dict[tuple[str, str], MappingDecision] = {}
            if not remap and self.ws.decisions_json.exists():
                # the domains the spec accepts now, before the reviewer's edits are checked against them
                domains = source_vocabulary_domains(spec, profile, self.cfg.privacy)
                prior = {(d.source_vocabulary_id, d.source_code): d
                         for d in import_review(self.ws, self.conn, self.schemas, domains=domains)}
            by_key = {(t.source_vocabulary_id, t.source_code): t for t in terms}
            kept = [d.model_copy(update={"frequency": by_key[k].frequency, "target_domain": by_key[k].target_domain,
                                         "accepted_domains": list(by_key[k].accepted_domains)})
                    for k, d in prior.items() if k in by_key]
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
        self._done_pending("remap")
        self._gate(
            "vocab_map",
            "Review the code mappings in mappings/*.csv (set status/target_concept_id), then continue.",
            [str(self.ws.mappings_dir)],
        )
        return decisions  # type: ignore[return-value]

    def apply_reviewed_mappings(self) -> int:
        """Re-import edited mapping CSVs into source_to_concept_map. The edits are checked against
        the domains the current spec accepts for each source vocabulary."""
        from .vocab.mapper import source_vocabulary_domains, write_source_to_concept_map
        from .vocab.review import import_review

        domains = None
        if self.ws.spec_yaml.exists():
            profile = self.load_profile() if self.ws.profile_json.exists() else None
            domains = source_vocabulary_domains(self.load_spec(), profile, self.cfg.privacy)
        decisions = import_review(self.ws, self.conn, self.schemas, domains=domains)
        n = write_source_to_concept_map(
            self.conn, self.schemas, decisions, include_unchecked=self.cfg.mapping.use_unchecked_in_etl
        )
        self.conn.commit()
        self.ws.audit("mappings_reimported", rows=n)
        return n

    def validated_spec(self) -> ETLSpec:
        """Load etl_spec.yaml and validate it again: it may have been edited by hand since it was
        designed. Errors (unsafe SQL, identifiers loaded into the CDM, events not linked through
        the id maps, unknown fields, ...) stop the build; warnings are logged."""
        from .design.validate import format_issues, validate_spec

        spec = self.load_spec()
        profile = self.load_profile() if self.ws.profile_json.exists() else None
        issues = validate_spec(spec, profile, self.conn, self.schemas, privacy=self.cfg.privacy, cfg=self.cfg)
        self.conn.rollback()
        errors = [i for i in issues if i.severity == "error"]
        for i in issues:
            if i.severity != "error":
                self.log(f"  spec warning: {i.message}")
        if errors:
            self.ws.audit("spec_invalid", errors=[str(i) for i in errors])
            raise SpecInvalid(f"{self.ws.spec_yaml.name} has {len(errors)} error(s):\n{format_issues(errors)}")
        return spec

    def build(self, spec: ETLSpec | None = None) -> ETLRunResult:
        from .etl.runner import UPGRADE_NOTICE, run_etl, upgrade_notice

        def run() -> ETLRunResult:
            # a spec the validator refuses builds nothing: the notice waits for a build that runs
            valid = spec or self.validated_spec()
            behaviours = upgrade_notice(self.ws)
            if behaviours:  # the first build since the update: say once what builds now do
                self.log("  note: the first build of this project with the new version, which changes the CDM it "
                         f"builds: {'; '.join(behaviours)} (README.md, \"Upgrading\")")
                self.ws.audit("upgrade_notice", notice=UPGRADE_NOTICE, behaviours=behaviours)
            result = run_etl(self.conn, self.schemas, valid, self.cfg, self.ws)
            derived, moved = result.rows_derived(), result.rows_moved()
            n_errors, n_blocking = len(result.errors()), len(result.blocking_errors())
            self.log(f"  {len(result.steps)} steps, {result.rows_loaded()} rows loaded"
                     + (f", {derived} derived by the build (observation periods, eras)" if derived else "")
                     + (f", {moved} records moved to the table of their concept's domain" if moved else "")
                     + f", {n_errors} errors"
                     + (f" ({n_errors - n_blocking} in steps that do not block the build)" if n_errors > n_blocking
                        else ""))
            for s in result.errors():
                self.log(f"  ✗ {s.step}{'' if s.blocking else ' (not blocking)'}: {s.error}")
            return result

        return self._run_stage("build", run)  # type: ignore[return-value]

    def check(
        self, spec: ETLSpec | None = None, etl_result: ETLRunResult | None = None, include_r: bool = True
    ) -> DQReport:
        """Run the data-quality checks: the built-in engine, plus the official R Achilles and
        DataQualityDashboard when ``quality.use_r_dqd`` is set and ``include_r`` (the repair loop
        checks with the built-in engine only and runs R once, on the final CDM)."""
        from .quality.checks import gate, run_builtin_checks
        from .quality.dqd_r import merge_results, r_available, r_failure_summary, r_missing_result, run_r_dqd

        def run() -> DQReport:
            spec_ = spec or self.load_spec()
            etl = etl_result or (
                self.ws.load_json(self.ws.etl_run_json, ETLRunResult) if self.ws.etl_run_json.exists() else None
            )
            from .etl.runner import OLDER_BUILD_NOTE, built_by_older_version

            if built_by_older_version(etl):
                self.log(f"  note: {OLDER_BUILD_NOTE}")
            report = run_builtin_checks(self.conn, self.schemas, self.cfg, spec_, etl)
            if self.cfg.quality.use_r_dqd and include_r:
                problem = None
                if not r_available(self.cfg):
                    problem = (
                        f"R, or one of the R packages DatabaseConnector, Achilles and DataQualityDashboard, "
                        f"is not installed (quality.rscript: {self.cfg.quality.rscript})"
                    )
                    self.log(f"  ✗ {problem}; built-in checks only")
                    r_report = None
                else:
                    r_report = run_r_dqd(self.cfg, self.ws)
                    if r_report is None:
                        problem = f"the R run failed: {r_failure_summary(self.ws)}"
                        self.log(
                            "  ✗ R Achilles/DataQualityDashboard failed; built-in checks only "
                            f"(see {self.ws.root / 'dqd' / 'rscript.log'}): {r_failure_summary(self.ws)}"
                        )
                if problem is not None:
                    self.ws.audit("dqd_r_fallback_builtin", reason=problem[:500])
                    if self.cfg.quality.require_r_dqd:
                        results = report.results + [r_missing_result(problem)]
                        passed, blocking = gate(results, self.cfg, etl)
                        report = report.model_copy(
                            update={"results": results, "passed": passed, "blocking_failures": blocking}
                        )
                if r_report is not None:
                    results = merge_results(report.results, r_report.results)
                    passed, blocking = gate(results, self.cfg, etl)
                    self.log(
                        f"  R DataQualityDashboard: {len(r_report.results)} checks, "
                        f"{len(r_report.failed())} failed; Achilles results in {self.schemas.results}"
                    )
                    report = report.model_copy(
                        update={
                            "engine": "builtin+dqd",
                            "results": results,
                            "passed": passed,
                            "blocking_failures": blocking,
                        }
                    )
            self.ws.save_json(self.ws.dq_json, report)
            failed = report.failed()
            self.log(
                f"  {len(report.results)} checks, {len(failed)} failed "
                f"({len(report.blocking_failures)} blocking) -> {'PASS' if report.passed else 'FAIL'}"
            )
            return report

        return self._run_stage("quality", run)  # type: ignore[return-value]

    def build_check_repair(self) -> DQReport:
        """Build, check, and let the repair agent fix the spec until the gate passes.

        A repair iteration that makes the result worse (an ETL error or more failing checks) is
        rolled back: the last best spec is restored, rebuilt and re-checked.
        """
        from .quality.repair import repair_spec

        if self.ws.repairs_json.exists():  # repairs.json describes this build session only
            self.ws.repairs_json.replace(self.ws.root / "repairs.previous.json")
        if self.ws.spec_yaml.exists():  # the spec exactly as reviewed, before any automatic repair
            (self.ws.root / "etl_spec.before_repair.yaml").write_text(self.ws.spec_yaml.read_text())
        spec = self.validated_spec()
        profile = self.load_profile() if self.ws.profile_json.exists() else None
        max_iter = self.cfg.quality.max_repair_iterations
        iteration = 0
        best: tuple[tuple, ETLSpec, int] | None = None  # (score, spec, iteration)
        seen = _BuildHistory()
        while True:
            etl = self.build(spec)
            report = self.check(spec, etl, include_r=False) if etl.ok else None
            losses = seen.losses(etl, report) if iteration > 0 else []
            score = self._score(etl, report, losses, spec)
            if best is not None and score < best[0]:
                why = f" ({'; '.join(losses[:3])})" if losses else ""
                self.log(f"  repair iteration {iteration} made the result worse{why}; restoring the previous spec")
                self.ws.audit("repair_reverted", iteration=iteration, reasons=losses[:20])
                _mark_reverted(self.ws, after_iteration=best[2])
                spec = best[1]
                self.ws.save_yaml(self.ws.spec_yaml, spec, header=SPEC_HEADER)
                from .design.render import render_spec_md

                self.ws.spec_md.write_text(render_spec_md(spec))
                etl, report = self.build(spec), None
                break
            if best is None or score >= best[0]:
                best = (score, spec, iteration)
            seen.add(etl, report)
            if report is not None and report.passed and not self._repairable(report, etl, spec):
                break
            if iteration >= max_iter:
                self.log(f"  stopping after {max_iter} repair iterations")
                break
            iteration += 1
            self._privacy_gate()
            self.log(f"▶ repair (iteration {iteration})")
            spec, actions = repair_spec(
                spec, iteration=iteration, dq=report, etl_result=etl, llm=self.llm, conn=self.conn,
                schemas=self.schemas, cfg=self.cfg, ws=self.ws, profile=profile,
            )
            for a in actions:
                self.log(f"  ✎ {a.target_table}.{a.target_field or '*'}: {a.change}")
            if not actions:
                self.log("  repair agent made no changes")
                break
        # The final CDM: the official R checks (if enabled) run once, here, not in every iteration.
        if report is None or (self.cfg.quality.use_r_dqd and etl.ok):
            report = self.check(spec, etl)
        self._refresh_vocab_evaluation(spec)
        return report

    def _refresh_vocab_evaluation(self, spec: ETLSpec) -> None:
        """Re-measure vocabulary coverage for the spec that built the CDM: repairs and hand edits
        change the vocab_code expressions after the mapping stage measured them."""
        from .vocab.evaluate import evaluate_vocabulary, render_vocab_evaluation_md

        if not self.ws.vocab_eval_json.exists():
            return
        try:
            ev = evaluate_vocabulary(self.conn, self.schemas, spec, self.cfg)
        except psycopg.Error as e:
            self.conn.rollback()
            self.log(f"  could not re-measure vocabulary coverage: {e.__class__.__name__}")
            return
        self.ws.save_json(self.ws.vocab_eval_json, ev)
        (self.ws.root / "vocab_evaluation.md").write_text(render_vocab_evaluation_md(ev))

    def _score(
        self, etl: ETLRunResult, report: DQReport | None, losses: list[str] | None = None, spec: ETLSpec | None = None
    ) -> tuple:
        """Higher is better: no records or data lost compared with earlier builds, then gate passed,
        fewer blocking ETL errors, fewer blocking and repairable failures. Losing data ranks below
        any failing check, so a repair that drops records or blanks fields to pass is rolled back."""
        from .quality.checks import derived_tables, repairable

        no_loss = not losses
        if report is None:
            return (no_loss, False, -len(etl.blocking_errors()), -10**9, -10**9)
        sev = set(self.cfg.quality.repair_severities)
        derived = derived_tables(etl, spec)
        n_repairable = sum(repairable(r, sev, derived=derived) for r in report.results)
        return (no_loss, report.passed, -len(etl.blocking_errors()), -len(report.blocking_failures), -n_repairable)

    def _repairable(self, report: DQReport, etl: ETLRunResult | None = None, spec: ETLSpec | None = None) -> bool:
        """True when ``report`` has a failure the repair agent should be given (``checks.repairable``;
        not the checks of the era tables the build of ``etl`` derived, or that ``spec`` loads itself
        while the build derives eras)."""
        from .quality.checks import derived_tables, repairable

        sev = set(self.cfg.quality.repair_severities)
        derived = derived_tables(etl, spec)
        return any(repairable(r, sev, derived=derived) for r in report.results)

    def report(self, shareable: bool = True) -> None:
        """Write the inspection report, and (by default) the shareable copy beside it, so the two
        always describe the same build."""
        from .report.inspection import build_report

        def run() -> None:
            md, html = build_report(self.cfg, self.ws, self.conn, self.schemas)
            self.log(f"  inspection report: {md} and {html}")
            if shareable:
                md, html = build_report(self.cfg, self.ws, self.conn, self.schemas, shareable=True)
                self.log(f"  shareable version (rare source terms and database error values withheld): {md}")

        self._run_stage("report", run)

    # ---- everything --------------------------------------------------------------------
    def run_all(self, fresh: bool = False, redesign: bool = False, remap: bool = False) -> DQReport:
        """Run every stage. Existing artifacts are reused unless ``fresh`` is set: the staged
        source and profile, the annotations, the (possibly hand-edited) ETL spec unless
        ``redesign``, and earlier or reviewed mapping decisions unless ``remap``."""
        self.ws.audit(
            "pipeline_start", project=self.cfg.name, llm=self.cfg.llm.provider, model=self.cfg.llm.model,
            fresh=fresh, redesign=redesign, remap=remap,
        )
        try:
            asked = {"profile": fresh, "annotate": fresh, "redesign": fresh or redesign, "remap": fresh or remap}
            earlier = self._pending()
            if earlier:
                self.log(f"  continuing an earlier request: {', '.join(sorted(earlier))}")
            todo = {k: v or earlier.get(k, False) for k, v in asked.items()}
            self._save_pending(todo)
            if todo["profile"] or not self.ws.profile_json.exists():
                self.stage_and_profile()
            else:
                self.log("▶ profile: reusing the staged source and profile.json")
            self._privacy_gate()
            if todo["annotate"] or not self.ws.annotations_json.exists():
                self.annotate()
            self.load_vocabulary()
            self.design(redraft=todo["redesign"] or not self.ws.spec_yaml.exists())
            self.map_vocabulary(remap=todo["remap"])
            report = self.build_check_repair()
            self.report()
            self.ws.audit("pipeline_end", passed=report.passed)
            return report
        finally:
            self.close()


PENDING_STEPS = ("profile", "annotate", "redesign", "remap")

SPEC_HEADER = "# ETL specification (restored by the repair loop). Edit and re-run `omop-agent build`."

# A table that keeps less than this share of the rows an earlier build loaded has lost data.
LOSS_TOLERANCE = 0.95


class _BuildHistory:
    """What earlier builds of one repair session loaded, to spot a repair that loses data."""

    def __init__(self) -> None:
        self.rows: dict[str, int] = {}
        self.links: dict[str, tuple[int, int]] = {}  # table -> (records with a visit key, linked to a visit)
        self.applicable: set[str] = set()
        self.route_only: set[str] = set()  # tables that received records moved by domain and nothing else

    @staticmethod
    def _rows(etl: ETLRunResult) -> dict[str, int]:
        """Rows loaded per CDM table (the id-map and pruning steps write no CDM table and are left out).
        When its periods were merged, observation_period counts the valid episodes the merge took
        (episodes, or event periods, less those dropped because they end before they start). Not the
        merged periods: a repair that fixes an episode's dates can join two periods into one, and must
        not count as a loss. Nor the fill's periods: a repair that inverts a person's only episode
        would get that person a period from their first to their last record instead, and the count
        must still see the episode lost (a person the fill covered who gets a valid episode adds one,
        and leaving the fill makes no loss). The route
        steps move records between tables after they were counted here, so moves change no total. The
        eras the build derives are left out: they follow the condition and drug records counted here."""
        from .cdm.spec import load_cdm_spec

        cdm = load_cdm_spec()
        out: dict[str, int] = {}
        merge = next((s for s in etl.steps if s.kind == "merge" and not s.error and "episodes" in s.stats), None)
        for s in etl.steps:
            if s.error or not s.target_table or not cdm.has_table(s.target_table) or _is_era_step(s):
                continue
            n = int(s.rows_inserted or 0)
            if merge is not None and s.target_table == "observation_period":
                if s is merge:
                    n = int(s.stats.get("episodes") or 0) - int(s.stats.get("dropped_inverted") or 0)
                else:
                    continue  # the episodes (or event periods) the merge took, counted above; the fill's
                    # periods are no measure of what was loaded (they make up for lost episodes)
            out[s.target_table] = out.get(s.target_table, 0) + n
        return out

    @staticmethod
    def _links(etl: ETLRunResult) -> dict[str, tuple[int, int]]:
        """Per CDM table, the records with a visit key and those linked to a visit of their person
        (``stats['visit_links']``): a repair that makes visit keys NULL or breaks them loses links."""
        out: dict[str, tuple[int, int]] = {}
        for s in etl.steps:
            links = s.stats.get("visit_links") if not s.error and s.target_table else None
            if links:
                with_key, linked = out.get(s.target_table, (0, 0))  # type: ignore[arg-type]
                out[s.target_table] = (with_key + int(links.get("with_key") or 0),  # type: ignore[index]
                                       linked + int(links.get("linked") or 0))
        return out

    @staticmethod
    def _route_only(etl: ETLRunResult) -> set[str]:
        """Tables that received records moved by domain but that no mapping or custom SQL loaded."""
        moved = {m.to_table for s in etl.steps if not s.error for m in s.moved}
        return moved - {s.target_table for s in etl.steps if s.kind in (None, "insert", "custom")}

    def add(self, etl: ETLRunResult, report: DQReport | None) -> None:
        self.route_only |= self._route_only(etl)
        for table, n in self._rows(etl).items():
            self.rows[table] = max(self.rows.get(table, 0), n)
        for table, (with_key, linked) in self._links(etl).items():
            before = self.links.get(table, (0, 0))
            self.links[table] = (max(before[0], with_key), max(before[1], linked))
        if report is not None:
            self.applicable |= {r.key for r in report.results if not r.not_applicable and r.check_level != "ETL"}

    def losses(self, etl: ETLRunResult, report: DQReport | None) -> list[str]:
        out: list[str] = []
        now = self._rows(etl)
        for table, before in sorted(self.rows.items()):
            n = now.get(table, 0)
            if before > 0 and n < LOSS_TOLERANCE * before:
                out.append(f"{table} loaded {n} rows, {before} before")
        links = self._links(etl)
        for table, (with_key, linked) in sorted(self.links.items()):
            now_key, now_linked = links.get(table, (0, 0))
            if with_key > 0 and now_key < LOSS_TOLERANCE * with_key:
                out.append(f"{table} has {now_key} records with a visit key, {with_key} before")
            elif linked > 0 and now_linked < LOSS_TOLERANCE * linked:
                out.append(f"{table} has {now_linked} records linked to their visit, {linked} before")
        if report is not None:
            # a table that only received records moved by domain empties when a repair maps their codes
            # to concepts of the loaded tables' domains: that is not data lost; nor is an era table the
            # build derives that empties (its condition or drug records are tracked themselves)
            written = {s.target_table for s in etl.steps if s.kind in (None, "insert", "custom")}
            exempt = ((self.route_only | self._route_only(etl)) - written) | {
                s.target_table for s in etl.steps if _is_era_step(s)}
            blanked = sorted(r.key for r in report.results
                             if r.not_applicable and r.key in self.applicable and r.table not in exempt)
            if blanked:
                out.append(f"{len(blanked)} check(s) no longer apply, e.g. {blanked[0]} (a field or table emptied)")
        return out


def _is_era_step(s: TableLoadResult) -> bool:
    """A step that derived an era table (``cdm.spec.DERIVED_ERAS``) from the loaded records."""
    from .cdm.spec import DERIVED_ERAS

    return s.kind == "derived" and s.target_table in DERIVED_ERAS


def _mark_reverted(ws: Workspace, after_iteration: int) -> None:
    """Mark the repairs made after ``after_iteration`` as rolled back in repairs.json."""
    from .models import RepairAction

    if not ws.repairs_json.exists():
        return
    actions = ws.load_json_list(ws.repairs_json, RepairAction)
    ws.save_json(
        ws.repairs_json,
        [a.model_copy(update={"reverted": True}) if a.iteration > after_iteration else a for a in actions],
    )
