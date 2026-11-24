# Project operating instructions

## Project facts

Rules below refer to these by name.

| Name used below | Value for this project |
|---|---|
| **Project** | `omop-agent`: an AI agent that automates the OHDSI ETL pipeline, converting patient-level source data into the OMOP CDM v5.4 on PostgreSQL (`README.md`, `docs/architecture.md`). It handles health data. |
| **Deferred register** | `docs/deferred.md` |
| **Decision records** | `docs/architecture.md`, in the section for the stage the decision changes |
| **Test database** | PostgreSQL 16 with `pg_trgm`, reached through the `OMOP_AGENT_TEST_DB_URL` environment variable. CI uses `postgresql://omop:omop@localhost:5432/omop_test` (`.github/workflows/ci.yml`). With the variable unset the PostgreSQL tests are skipped, and a skipped test is not a passed one. |
| **Narrowest checks** | One test: `pytest -q tests/test_core.py::test_expand_env`; one file: `pytest -q tests/test_core.py`. The project configures no linter and no type checker (`pyproject.toml`, `.github/workflows/ci.yml`); do not report either as run. |
| **Full suite** | `pytest -q` with `OMOP_AGENT_TEST_DB_URL` set (about 3 minutes; the only skip is the slow scale test in `tests/test_tolerant.py`, run with `OMOP_AGENT_SLOW_TESTS=1`), then the offline demo ("Verification commands" below). CI also builds the Docker bundle and the R image and runs the demo inside each (the `docker` and `docker-r` jobs); those run on push. |
| **Live verification** | The offline demo for the pipeline, the build, the quality gate and the report; a browser session for the web app; the CI job steps for the Docker bundle and the R image. Commands are under "Verification commands" below. |
| **Expensive-to-reverse choices** | Anything a deployed project already holds on disk or in its database: the `etl_spec.yaml` and `omop_agent.yaml` schemas; the files under `work/` that a later run reloads; the approval digests (`omop_agent/approvals.py`: a changed digest un-approves existing designs and mappings); the id maps and the ETL-schema tables (ids are kept across builds); what the build does with an unchanged spec (README, "Upgrading"). Also: what may reach the LLM, the web role model, the CDM version, and the target database. |
| **Access-gating surfaces** | (1) **The LLM boundary:** `omop_agent/privacy.py`, `omop_agent/privacy_review.py`, the repair agent's tools and error redaction in `omop_agent/quality/repair.py`, and any code that builds a prompt or a tool result. (2) **Review gates:** `omop_agent/approvals.py` and the gate decisions in `omop_agent/pipeline.py`. (3) **SQL written by the model or the spec:** `omop_agent/sqlguard.py` and its callers. (4) **Web:** the `need(request, role)` check in `omop_agent/web/app.py`, accounts and roles in `omop_agent/web/users.py`, sessions, CSRF and sign-in limits in `omop_agent/web/security.py`. (5) **Database roles and grants:** `deploy/postgres-init/10-omop-role.sh` and the grants the build makes to `target.etl_role`. |

## Verification commands

Run on 2026-11-24 against a local PostgreSQL 16; each printed what is shown.

The offline demo (recorded LLM answers, no API key):

```bash
WORK=$(mktemp -d)
OMOP_AGENT_DB_URL="$OMOP_AGENT_TEST_DB_URL" omop-agent demo "$WORK/projects/demo"
```

It ends with `PASSED -> ` and the path of `work/report.html`, and exits 0. Then read the artifact
the change affects under `$WORK/projects/demo/work/`: `report.md`, `dq_results.json`, `sql/`,
`etl_spec.yaml`, `audit.jsonl`.

The web app, on the demo project:

```bash
printf 'local-check-1234\n' | omop-agent users add check --role operator --password-stdin --data-dir "$WORK/data"
omop-agent serve --projects "$WORK/projects" --data-dir "$WORK/data" --port 8765
```

The first prints `check: operator`. With the server up, `curl -fsS http://127.0.0.1:8765/healthz`
prints `ok`, and `/p/demo` answers 303 (to the sign-in page) until you sign in. Those two show the
app started, not that a change works: drive the changed page in a browser, signed in at the role
the change concerns (`viewer`, `reviewer` or `operator`).

The Docker bundle and the R image: the steps of the `docker` and `docker-r` jobs in
`.github/workflows/ci.yml`. Not run here.

A run against Claude live (`--provider anthropic`) needs an API key and is not part of routine
verification.

---

# Part A: how the session is run

## A1. How work is organised: one session by default; lanes only when asked

- **By default the main session does the work** and stays the single controller. It may dispatch a
  subagent where the work needs one: an independent review (always required), a scoped build that
  runs while the session does something else, or a read across many files whose conclusion is all
  that is needed. Subagents are authorised for those without asking each time.
- **Lanes, a coordinator and multi-agent workflows run only when the user asks for them** in that
  piece of work's request. When asked: a coordinator (the main session) maps the lanes, rules
  across them, merges, and runs the Full suite on every merged commit. Each lane has its own
  worktree, branch and test data, and a lead that is the controller of its lane; a lead does not
  build or review except a change smaller than its brief, still independently reviewed. How wide
  to run is established from a map of every task's files and interfaces, never assumed, and never
  wider than the machine can verify.
- **True either way:** every change gets an independent review before it counts as done, and the
  Full suite runs on what is merged. A helper agent's output is pulled from a file, not awaited
  (B4).

## A2. Models: the lightest tier that does the work cleanly

The main-loop model is the user's setting; leave it. This section acts through dispatch. Where a
skill's or a workflow reference's model advice (including advice to omit the model) disagrees with
this section, this section wins.

- **Name the model on every dispatch** (`haiku`, `sonnet`, `opus` or `fable`), with effort set as
  its tier says. Omitted, the model falls to the agent definition's pinned model or the session's
  and matches this section only by luck. A fork ignores the model and runs the session's own; when
  the work belongs on another model, use a fresh agent.
- **Haiku** (effort low to medium): **read-only lookups only**: finding files and symbols,
  summarising grep output and logs. No edits, no reviews, nothing that needs judgment. Dispatch it
  as `general-purpose` or a custom agent that loads the project rules, never as the built-in
  Explore or Plan, which start without CLAUDE.md. A `general-purpose` agent can still edit, so its
  brief says read-only. One narrow question per dispatch. Check what it returns before relying on it.
- **Sonnet** (medium to high; higher only when asked): well-scoped tasks, fixing bugs whose cause
  is known, polished documents, unit tests, fan-out across files on a clear spec, and routine
  reviews.
- **Opus** (medium; high for hard bugs; on a task its tests can verify, low first and then high,
  before anything higher or an escalation): complex work requiring careful judgment, every review that
  is not routine, and a lane lead when the user has asked for lanes.
- **Fable** (medium to high; higher only after high fails): **advisor** (a second opinion on a
  decision the controller cannot settle, asked as a question), **coordinator** (the main session
  when the user runs it on Fable, or a coordinator dispatched when the user asks for lanes),
  **escalations** (stuck work, below), and building four kinds of work: a multi-hour autonomous
  migration, an outage investigation whose root cause is ambiguous, formal verification, a novel
  algorithm. Brief it with the outcome and constraints, not a step list. It does not review or
  lead a lane by default.
- **Reviews: routine on Sonnet, the rest on Opus.** A review is routine only when it passes the
  test in B4 ("When a review is routine"). In doubt, Opus. All other reviewing, auditing or
  adversarial verifying is judgment: a lighter checker misses the plausible-looking errors it is
  there to catch. Running your own checks before a claim is not a review; every agent does that.
- **Diagnosing a bug whose cause is unknown is judgment** and goes to Opus (an outage investigation
  whose root cause is ambiguous may go to Fable, above); the settled fix can go to a lighter model.
- **A task that fits both sides is split** where it can be: settle the judgment on Opus, hand the
  settled execution to Sonnet. One that cannot be split goes to Opus. Judgment given to a lighter
  model fails quietly, as an answer that looks right; execution given to a heavier one costs only
  tokens.
- **A Haiku or Sonnet brief says to stop and report, not decide, anything that turns out to need
  careful judgment**: a bug whose fix turns out to be a design choice, or a routine review of a
  change that turns out not to be routine.
- **Stuck work moves up.** Stuck means the agent (or the controller working inline) reports it
  cannot finish, or makes no headway: the same failure survives two distinct fixes, a cause stays
  unknown after two distinct hypotheses are tested, or a reviewer is still undecided after two
  distinct checks. Repeating an approach does not count. Reviewers who each reach a verdict but
  disagree are not stuck; the controller settles the split. Every brief says to stop and report
  once stuck. Haiku's stuck work goes to Sonnet, Sonnet's to Opus, and Opus's to Fable: a fresh
  agent briefed with the goal, what was tried and why it did not settle it, not the step list that
  failed. A block only the user can clear (missing access, a decision that is theirs) goes to the
  user. If Fable is stuck too, stop and bring it to the user. If Fable is unavailable, continue
  with Opus and Sonnet: stuck Opus work goes to the user, and the four kinds Fable builds go to
  Opus.

**Session suggestions.** Suggest only, in one line, before an expensive multi-turn loop; the
switch is the user's.

- A routine, well-scoped edit or lookup on Fable, or on Opus at high effort or above:
  `/effort low` or `/model sonnet`.
- A multi-file feature: Plan mode, or `/model opusplan`.
- The session itself stuck: `/effort high` first, then a stronger model.
- A model switch discards the prompt cache: suggest it right after `/clear` or `/compact`, not
  mid-session on a large warm cache.

## A3. Verifying a change versus re-reading a file

Both hold:

- **Do not re-read a file to confirm a write landed.** Edit and Write fail loudly; a clean result
  is the confirmation.
- **Do verify behaviour before claiming anything works** (C5): run the command, drive the app,
  observe the output, in the message that makes the claim.

The first is about whether bytes reached disk, the second about whether the system does what you
said it does. Skipping the first is waste; skipping the second is how unverified work gets
reported as done.

## A4. Token and context hygiene

This lowers no bar set elsewhere: the evidence rules and the reviews stand.

- **Read surgically.** Go to the symbol or line range before the whole file. Never pull large
  logs, lockfiles, build artifacts or raw CSV/JSON into context; query them with `jq`, `grep`,
  `python3`, `head` or `tail`. Here that covers the demo's `work/` files, `dq_results.json`,
  `audit.jsonl`, the vocabulary fixtures and the 42 KB `README.md`.
- **Keep output concise.** Report paths, line numbers, exact commands and pass/fail evidence; no
  narrative recap, no code block that is already on disk.
- **Prefer a CLI to an MCP server** where both exist (`gh`, `git`), except where a rule or the
  session names the tool.
- **Protect the prompt cache.** Batch related edits and their verification commands into one turn.
- **Narrowest check while working.** After an edit, run the narrowest relevant test and report its
  output; do not repeat a check nothing has invalidated, and do not restyle working code unasked.
  C5, the independent review and the Full suite on what is merged are unchanged.
- **Isolate noisy work.** Log triage, dependency audits and multi-file searches go to a subagent
  on the model that kind of work needs, so only the conclusion reaches the main context.
- **Stay scoped.** When a multi-file change is ambiguous, give a short plan or ask before editing.

## Compact Instructions

When compacting this conversation (auto-compact or `/compact`), preserve:

1. The active task's goal, acceptance criteria and unresolved blockers.
2. Exact paths modified, key symbol and API signatures, and decisions made.
3. The latest failing test or compiler output and the exact verification commands to re-run.

Drop raw exploration logs, superseded diffs and verbose tool output.

---

# Part B: operating rules, organised by moment

## B0. How to use Part B

**These rules are organised by moment, not by topic.** The question is never "am I violating
something?", which requires holding the whole document in mind. The question is **"which
checkpoint am I at, and what fires here?"** Go to that section and do what it says.

> **An obligation that names no observable trigger will not fire. Adding emphasis does not change
> this.** If you cannot say what event makes a rule fire, it is a wish, not a rule.

At the start of a work session, scan the session's skill list; if the work is executing a written
plan, read the plan.

Maintaining these rules:

1. Any amendment states which checkpoint it binds to. One that binds to none is rejected.
2. Before adding anything, check whether an existing principle already covers it.
3. Re-read Part B end to end at every phase exit (C6).
4. Prefer converting a rule into a script, hook or test over restating it. Mechanism beats prose.
5. Record the incident that prompted a new rule, in the same commit that adds the rule.

## B1. Checkpoints

| # | Checkpoint | Fires when |
|---|---|---|
| **C1** | Task start | You begin any non-trivial task |
| **C2** | Before dispatch | You are about to hand work to a subagent |
| **C3** | Mid-execution | One of the B2 conditions occurs, not on a schedule |
| **C4** | On receiving work | A subagent reports, or you finish a chunk yourself |
| **C5** | Before claiming done | You are about to say complete, passing, working or fixed |
| **C6** | Phase exit | A plan or phase is finished, a branch is ready to merge, or work on the default branch is about to be pushed |
| **C7** | Before committing | You are about to run `git commit` |

### C1. Task start

- Scan the session's skill list and invoke the fitting skills **before** executing. The always-on
  tier applies by default, each skill at its own moment: `rigor-loop:rigor-loop` here;
  `superpowers:brainstorming` here, before feature or behaviour work (not when executing an
  approved plan); `superpowers:test-driven-development` before implementation code;
  `superpowers:systematic-debugging` at its B2 trigger; `code-review` at C4;
  `superpowers:verification-before-completion` at C5; `security-review` at C6. Skills for a
  particular area of work apply when the task enters that area.
- **Work touching an Access-gating surface or data access gets the preventive security pass here**,
  before code exists (B7.1), whatever an orchestration skill reports.
- **A skill or agent this file names that does not resolve in the session's list fails silently.**
  A name may resolve under another prefix (`rigor-loop` and `what-if-oracle` appear under
  `anthropic-skills:` in cloud sessions); use the form the session lists. When one is missing
  altogether, say so, do the job the rule gives it by direct analysis, and record it under B6. One
  standing row in the Deferred register per missing name is enough.
- **A process or orchestration skill never discharges the always-on tier.** Skills that decide how
  work is sequenced and reviewed supply no domain expertise, and their reviews check a diff
  against a brief after code exists. Symptom to catch: "the review loop already covers this."
- Work the user asked to run as lanes: map every task's files and interfaces before any split
  (A1). Without that request, the work is not split into lanes.
- State the task's **single responsibility and what is explicitly not its job**, and **what it
  interacts with, including layers it does not call directly** (B3).
- Do **not** try to predict your mid-task triggers. C3 replaces prediction with observation.

### C2. Before dispatch

Everything in B4. The dispatch must carry the process, not just the task.

### C3. Mid-execution

The conditions in B2. Binds the controller and every subagent.

### C4. On receiving work

- `git status --porcelain --untracked-files=all`. Untracked residue is a finding: a reviewer
  handed `git diff` never sees an untracked file.
- Verify the claims yourself; a report is a set of assertions (B5).
- **Apply the `code-review` lens, then the `simplify` lens to what you are keeping.** A
  spec-compliance verdict ("does this diff match its brief?") discharges neither. The discharge is
  a separate, explicit verdict per lens, from the controller invoking the skills or from a
  dispatched reviewer applying both by direct analysis. A lens that finds nothing says so plainly.
- Assess **reach** (B3): what does this change affect beyond the files it edited?
- Security here is the preventive pass (B7.1), not the diff scan. The diff scan is at C6.

### C5. Before claiming done

- **Live end-to-end verification is the definition of done**: observed working by the project's
  Live verification method, not merely building or passing tests. State plainly what was and was
  not verified live. If one way of running it fails, try another before concluding none is
  available.
- Run the verification command **in this message**. A previous run is not evidence about the
  current state.
- **A test run counts only with the Test database reachable.** Report the skip count and what was
  skipped; "all passed" over skipped PostgreSQL tests is a false claim.
- Never declare work sound if a mandated process was bypassed. Say what was skipped.

### C6. Phase exit

- Run `security-review` over the **whole branch diff**. The skill diffs against `origin/HEAD`,
  which a fresh clone may lack: check with `git rev-parse --abbrev-ref origin/HEAD` and set it
  with `git remote set-head origin -a`. Per task it re-reviews reviewed code at compounding cost;
  the per-task security work is the preventive pass (B7.1).
- Review the Deferred register in full. No phase exits with an open row targeted at that phase
  unless it is re-targeted with justification.
- Re-read Part B end to end.
- Re-audit skill invocation from the session transcripts (Claude Code keeps them as JSONL under
  `~/.claude/projects/`): count actual skill and agent calls, and compare the skills bound to a
  checkpoint against the ones that are not. A commit-log grep measures how often a name was
  discussed, and recollection does not substitute. Confirm every skill name this file uses still
  resolves.
- When the branch is ready to merge, `superpowers:finishing-a-development-branch`.

### C7. Before committing

- **Check the branch.** On `main` or `master`, implementation work is not committed without the
  user's explicit consent; offer a branch or worktree first.
- **Read the message as someone who was not present.** It describes the change, not the doing of
  the change.
- **Qualify every identifier on first use**: which file, which register, which plan.
- **Every load-bearing claim carries a file and line, or a command and its output.**
- **Never cite a path under a gitignored directory.** If the evidence matters, commit it somewhere
  it survives.
- Session narrative goes to the Deferred register or a gitignored workspace, never into permanent
  history.

## B2. Mid-execution triggers

Each is an observable condition, not a prediction. When one occurs, stop and do the action before
continuing. This applies to subagents exactly as it applies to the controller.

| When you notice… | Stop and… |
|---|---|
| You are about to work around something that failed | Investigate it first (B5): `superpowers:systematic-debugging`. One failed attempt is where investigating starts, not where it concludes. |
| You are about to write error handling, a fallback, a `catch` or a default value | Dispatch `pr-review-toolkit:silent-failure-hunter` (an agent, not a skill). It is aimed at the silent fail-open shape. |
| You are about to write or say *cannot*, *too costly*, *not applicable*, *unsupported* | Treat it as a factual claim needing evidence (B5). |
| You are touching a file the brief did not name | Ask whether this is still the task's scope. If it is scope creep, stop. If the brief was wrong, say so. |
| An assumption in the brief turns out to be false | Report it. Do not silently correct and continue; the brief's author needs to know. |
| You are about to write anything on an Access-gating surface | `ecc:security-review` applies now, not at the end. It is a checklist: derive this task's obligations and put them in the brief (B7.1). |
| You reach one of the Expensive-to-reverse choices | `what-if-oracle:what-if-oracle`. Mandatory at these forks. |
| The change is reaching further than the task described | Assess reach (B3) and report the divergence. |
| You are about to reuse a previously derived conclusion | Re-derive it, or state that you are reusing it and why that is valid now (B5). |
| You have just produced an enumeration that will be treated as complete: a file list, a test matrix, a task sequence, a set of proof obligations | Run B3's decomposition obligation before going further. Do not skip it because the list feels thorough; that feeling is the failure mode. |
| You have finished a design artifact (a spec, a decision record, a plan, a decomposition) and are about to show it to the user | Dispatch a **fresh agent that did not write it**, on Opus, briefed to attack the decomposition ("what is missing, and what lens would reveal it?"), not to review the content. Self-review does not catch this class. |
| You are about to hand over anything another person or agent will act on without re-deriving it: a plan, a brief, a list of commands, a runbook | The five hand-over checks below. |
| You are about to record an obligation as discharged, in a report, a commit message, a coverage map or a register closure | State the obligation's **property** in your own words. State what the test actually **causes to happen**. Ask whether the second entails the first: could the property be violated by another mechanism while this test still passes? If yes, name the gap; do not close it. |
| You are editing an environment file, a container or deployment config, a provisioning script, or any secret-bearing config (here: `deploy/`, `Dockerfile`, `Dockerfile.r`, `.github/workflows/`, any `omop_agent.yaml`) | `ecc:security-review` (secrets); for `deploy/compose.yaml` also `ecc:docker-patterns`. Never put a connection string on a command line: `omop-agent` reads `OMOP_AGENT_DB_URL`, and `omop_agent.yaml` takes `${VAR}` references (`omop_agent/config.py`). |
| You are about to write or edit HTML, CSS or client-side JS (here: `omop_agent/web/templates/`, `omop_agent/web/static/`, `omop_agent/report/templates/`) | `modern-web-guidance:modern-web-guidance` first. |
| You are about to defer, substitute or park anything | B6. |

**The five hand-over checks.**

1. **Execute the claims.** Any statement about how a library, database or tool behaves is run
   before the brief ships, and the brief cites the command and its output. If it cannot be run,
   mark it as an assumption for the implementer to check.
2. **Pair reads and writes.** Every value the plan says is read has a named writer, and every
   value written a named reader, verified by grep, not memory.
3. **Commands for a person are complete.** No placeholder and no unset name. A value that exists
   only on another machine is read there by a command into a variable, never guessed. What a
   command must print is shown as expected output.
4. **Machine facts come from the command that reads them.** Names that exist on this machine
   (containers, volumes, networks, databases, schemas) and local ports are pasted into the brief
   from that command's output; name only what it lists.
5. **A runbook command shows a run or says "Not run".** This covers commands added to
   `README.md` and `docs/deploy.md`.

## B3. Design and change

**Designing something new.** State in writing: (a) this unit's single responsibility and what is
explicitly not its job; (b) how it interacts with every layer it touches, including the ones it
does not call directly. Most defects are interaction defects: components verified in isolation
compose into a broken system. Architectural decisions get an entry in Decision records.

**Changing something that exists: determine its reach.** Before and after:

- What else reads, writes or depends on this?
- Do the parts I changed still form a consistent whole, or is each merely locally correct?
- What did this change make possible that was not possible before?
- What now happens on the paths I did not touch?

The fourth is the one that does not get asked. Its failure class is a control whose enforcement
point moved while its trigger stayed behind. It is invisible in any single task's diff, so no
diff-scoped review can catch it; only whole-tree analysis can, which is why C6 runs at branch
granularity. Where the project can, enumerate from disk the callers of guards that must be wired
and the read-sites of values that gate behaviour, in a test that fails when one has none.

**Decomposing something into parts: check the relations, not the elements.** A list's correctness
is entirely in its relations; every element can be right while the list is wrong. Whenever you
produce one:

1. **Name the property that generated it.**
2. **Generate it again from a different property and diff.** Vary the search **root**, not only
   the predicate: three questions asked of the same subtree are one lens. Ask "what is the root I
   keep searching, and what lives outside it?"
3. **State what connects the items**, not only what they are.
4. **For each verification item, state the claim it proves**, then check it proves that claim and
   not a neighbouring one.
5. **Run `what-if-oracle:what-if-oracle` on the decomposition itself**, asking "what if this ships
   exactly as specified?" This is a separate invocation from the B2 fork trigger, which explores
   only a decision you have already named.

**Where a list must be complete, generate it; do not write it by hand.**

**Agreement between two checks is not corroboration when both use the same lens.**

## B4. Delegation

Subagents are bound by these rules, not exempt. The controller does not get to launder a shortcut
by delegating it.

- **One controller by default; lanes only when asked** (A1). The main session alone puts questions
  to the user; a lane lead does not.
- **Models** as in A2.
- **When a review is routine.** The change is an ordinary one Sonnet would build (well-scoped,
  clear spec, cause known) and touches none of: an Access-gating surface, an Expensive-to-reverse
  choice, secret-bearing config or a container or deployment config, a test recorded as
  discharging an obligation, this file or a hook or settings file that sets rules, or a design
  artifact. An audit or an adversarial verification is never routine. A change to data access,
  error handling or a fallback is therefore routine when it touches none of those; C1's skills,
  B2's triggers and B7.1's verification still apply to it, whichever model reviews. The dispatch
  says why the review is routine, requires C4's two lens verdicts, and tells the reviewer to stop
  and report if it finds the change is not routine; that review then goes to Opus.
- **Every dispatch carries the process.** The prompt requires the subagent to scan the session's
  skill list and invoke the fitting skill before writing code, and to name what it invoked in its
  report. A brief and constraints alone tell it what to build while exempting it from how this
  project builds.
- **The dispatch states B2 explicitly.** A subagent cannot honour triggers it was never given.
- **The dispatch names which altitude owns what**: whether security review, live verification and
  reach assessment sit with the controller or the subagent. Left unsaid, each assumes the other
  did it.
- **A helper's output is pulled, not awaited.** A subagent's completion notification routes to the
  controller, not to the peer that launched it, so an agent that launches a helper and waits will
  wait forever. The dispatch names the file each helper writes its result to; once the expected
  time has passed, read that file or proceed without it. Any dispatch that permits helpers says
  this.
- **Nothing is written outside assigned scope**, not even as a head start for a later task.
- **Everything written is committed or declared**, including probe scripts and uncommitted tests.
- **Reports are evidence, not assurances.** State what was run and what the output was.
- **The controller does not stage or commit while a subagent is running** in the same worktree.
  `git add -A` is repo-wide and sweeps in-progress work into a commit whose message describes
  something else. If it happens, `git reset --mixed HEAD~1` restores the index and HEAD without
  touching file contents.

## B5. Evidence

**An assertion is a hypothesis until checked, wherever it came from.** The ranking of danger runs
opposite to intuition:

| Source of claim | Danger | Why |
|---|---|---|
| Inherited from a document | Moderate | Visibly second-hand, so doubt comes naturally. |
| Generated by you to explain an obstacle | **Highest** | Arrives already feeling like an observation. |
| Derived by you earlier and reused | High | Was true when derived; silently ages. |

- **One error message is a symptom, not a diagnosis.**
- **Effort claims are claims.** "Too costly" or "large refactor" needs the same grounding as any
  other. Do not assert a cost you have not scoped.
- **Look at the thing itself.** Read the tool, skill or config definition before theorising.
- **Escalate with findings, not verdicts**: "I ran X, got Y, cause is Z, here are the options",
  not "X doesn't work here."
- **Cite the file and line, or the command and its output**, for load-bearing claims.

## B6. Recording what is not done

Every conscious decision to not do something now (a deferral, a parked decision, an accepted risk,
a known bug, or a triggered skill not run) gets a row in the Deferred register **in the same
commit that creates it**, with why and a concrete pick-up trigger. "Later" is not a trigger. Rows
are closed, never deleted; closure cites the commit and states what was verified live. A todo is
not sufficient: todos are session-local and take the obligation with them.

**A register row is not a substitute for trying.** Opening a row on "I cannot" grounds requires
evidence of the attempt (the command and its output, or the source read), exactly as closing one
does. "I have chosen not to" needs only justification and a trigger.

## B7. Standing principles

These have no single checkpoint; they bind everywhere.

1. **Security is foundational.** Authorisation and data isolation are verified on every change
   touching an Access-gating surface or data access, not deferred. Here data isolation includes
   the LLM boundary: no patient row, small cell or direct identifier reaches a prompt, a tool
   result or an error message sent to the model. Two halves, both required:

   | | Instrument | Fires | Catches |
   |---|---|---|---|
   | **Preventive** | `ecc:security-review`, a checklist: threat-model this task, derive what it must prove, put the obligations in the brief | Every such task, at C1, before code exists | Design defects, including fail-opens |
   | **Detective** | `security-review`, a scan of the diff | C6, over the whole branch | What got built, as opposed to what was intended |

   Derive the preventive obligations fresh per task. Reusing an earlier task's threat table is the
   B2 "reusing a previously derived conclusion" trigger.
2. **Honesty over completion.** Label unverified work unverified. Surface assumptions and gaps.
3. **A process decision is judged at the moment it was made, not by how it turned out.** "It was
   adequate" is survivorship reasoning.

---

## Rules that are prose only here

Nothing in this repository enforces the rules below; each depends on being remembered until the
mechanism in the right-hand column exists (B0, item 4).

| Rule | What would enforce it |
|---|---|
| Skills and agents named in C1, C4, C6, B2, B3 and B7 | The plugins that supply them: `superpowers`, `rigor-loop`, `what-if-oracle`, `ecc`, `pr-review-toolkit`, `modern-web-guidance`, plus the built-in `code-review`, `simplify` and `security-review`. Row 1 of the Deferred register lists the names that did not resolve in a cloud session. |
| Hand-over checks 3 and 4 (B2) | A Stop hook and a PreToolUse hook on agent dispatches. |
| Hand-over check 5 (B2) | A lint test in CI over the command blocks of `README.md` and `docs/deploy.md`. |
| Reach: guards and their triggers (B3) | A test that enumerates from disk the web routes and their `need(request, role)` checks, and the callers of `sqlguard.fragment_issues`, and fails when one has none. |
| C6 skill re-audit | A command that counts skill and agent calls in the session transcripts. |
| Rule history (B0, item 5) | A provenance file, one entry per rule; create it when the first rule is added to this file. |
| Lane work (A1) | A team charter with roles, merge rules and report formats; A1 carries the summary only, so write the charter before the first lane run. |
