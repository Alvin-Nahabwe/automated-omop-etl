# Deferred register

Work this project has consciously not done yet: deferrals, parked decisions, accepted risks, known
bugs, and skills a rule called for that were not run. The rules for this file are in `CLAUDE.md`,
section B6. In short:

- A row is added in the same commit that creates the deferral, with why and a concrete pick-up
  trigger. "Later" is not a trigger.
- A row opened because something could not be done cites the attempt: the command and its output,
  or the source read.
- Rows are closed, never deleted. Closing a row cites the commit and states what was verified live.

Product limits that users should know about are in the README, under "Limitations and roadmap".

| # | Opened | What is not done | Why | Pick-up trigger | Status |
|---|---|---|---|---|---|
| 1 | 2026-11-24 | Four names that `CLAUDE.md` gives a job to are not run in cloud sessions: the skills `ecc:security-review` (B2, B7.1), `ecc:docker-patterns` (B2) and `modern-web-guidance:modern-web-guidance` (B2), and the agent `pr-review-toolkit:silent-failure-hunter` (B2). Their jobs are still owed and are done as `CLAUDE.md` C1 says: a checklist derived in writing in the brief, an agent's job given to a fresh agent. | None of the four is in the skill list or the agent types offered to the Claude Code cloud session that installed `CLAUDE.md` on 2026-11-24. That list has `rigor-loop` and `what-if-oracle` only under the `anthropic-skills:` prefix. | The `ecc`, `pr-review-toolkit` or `modern-web-guidance` plugin appears in a session's skill list; or `CLAUDE.md` is changed to name an equivalent that does resolve. Check at every phase exit (`CLAUDE.md` C6, "Confirm every skill name this file uses still resolves"). | Open |
