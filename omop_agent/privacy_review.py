"""The privacy review: what this project will send to Claude, for a person to approve first.

Identifier detection is heuristic, so before the first live call a reviewer checks the profile as
Claude will see it and the columns that are withheld, and approves it (the ``privacy`` review
gate, in the web app or with ``omop-agent approve privacy``). The review is written to
``<workdir>/privacy_review.json`` (what the approval's digest covers) and ``privacy_review.md``.
It is rebuilt from the current profile and privacy settings whenever it is shown or checked, so a
re-staged source or a changed setting needs a fresh approval. Columns that Claude later labels as
identifiers or free text are withheld from then on; they are listed, but they only remove data
from what was approved, so they don't need a new approval (``also_withheld`` is left out of the
digest).

Projects that do not call Claude (``llm.provider`` ``none`` or ``replay``) need no privacy review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .fsutil import atomic_write_text
from .models import SourceAnnotations, SourceProfile
from .privacy import apply_overrides, profile_prompt_text
from .workspace import Workspace

REVIEW_JSON = "privacy_review.json"
REVIEW_MD = "privacy_review.md"

LATER_SENDS = [
    "The profile below (column statistics and frequent values), when Claude interprets the source, "
    "drafts the ETL design and repairs it.",
    "Source codes and labels of the fields being mapped to concepts, each seen at least "
    "{k} times, with their counts and the units recorded with them.",
    "Aggregate counts (never below {k}) that the repair agent asks for, and database error "
    "messages with the values in them removed.",
    "Never: patient rows, values of the withheld columns below, or values seen fewer than {k} times.",
]


def sends_to_llm(cfg: ProjectConfig) -> bool:
    """True when this project's settings send data to the Claude API."""
    return cfg.llm.provider == "anthropic"


def review_path(ws: Workspace) -> Path:
    return ws.root / REVIEW_JSON


def current_profile(cfg: ProjectConfig, ws: Workspace, with_labels: bool = True) -> SourceProfile | None:
    """The profile with the current privacy settings (and, with ``with_labels``, Claude's
    identifier labels) applied."""
    if not ws.profile_json.exists():
        return None
    profile = ws.load_json(ws.profile_json, SourceProfile)
    annotations = None
    if with_labels and ws.annotations_json.exists():
        annotations = ws.load_json(ws.annotations_json, SourceAnnotations)
    return apply_overrides(profile, cfg.privacy, annotations)


def _withheld(profile: SourceProfile) -> list[dict[str, Any]]:
    return [
        {"table": t.name, "column": c.name, "reason": c.pii_reason or "identifier", "n_distinct": c.n_distinct}
        for t in profile.tables
        for c in t.columns
        if c.is_pii
    ]


def build_review(cfg: ProjectConfig, ws: Workspace) -> dict[str, Any] | None:
    """The review content, or None before the source has been profiled."""
    profile = current_profile(cfg, ws, with_labels=False)
    if profile is None:
        return None
    k = cfg.privacy.min_cell_count
    withheld = _withheld(profile)
    labelled = current_profile(cfg, ws)
    before = {(w["table"], w["column"]) for w in withheld}
    also = [w for w in _withheld(labelled) if (w["table"], w["column"]) not in before] if labelled else []
    return {
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
        "settings": {
            "min_cell_count": k,
            "send_values_to_llm": cfg.privacy.send_values_to_llm,
            "max_values_per_column": cfg.privacy.max_values_per_column,
            "pii_columns": sorted(cfg.privacy.pii_columns),
            "not_pii_columns": sorted(cfg.privacy.not_pii_columns),
        },
        "sent": [s.format(k=k) for s in LATER_SENDS],
        "tables": [{"name": t.name, "rows": t.n_rows, "columns": len(t.columns)} for t in profile.tables],
        "withheld": withheld,
        "also_withheld": also,
        "profile_text": profile_prompt_text(profile, cfg.privacy),
    }


def digest_payload(review: dict[str, Any]) -> dict[str, Any]:
    """The part of the review an approval covers (later labels only withhold more)."""
    return {k: v for k, v in review.items() if k != "also_withheld"}


def render_md(review: dict[str, Any]) -> str:
    s = review["settings"]
    lines = [
        "# Privacy review: what this project sends to Claude",
        "",
        f"Provider `{review['provider']}`, model `{review['model']}`. Values seen fewer than "
        f"{s['min_cell_count']} times are never sent; values are sent: {'yes' if s['send_values_to_llm'] else 'no'}.",
        "",
        "## What is sent, and when",
        "",
        *[f"* {x}" for x in review["sent"]],
        "",
        f"## Withheld columns ({len(review['withheld'])}): counts only, never values",
        "",
        "| Table | Column | Why |",
        "|---|---|---|",
        *[f"| {w['table']} | {w['column']} | {w['reason']} |" for w in review["withheld"]],
        "",
        *(
            [
                f"Also withheld from later prompts ({len(review['also_withheld'])}), because Claude labelled them "
                "as identifiers or free text: " + ", ".join(f"{w['table']}.{w['column']}" for w in review["also_withheld"]),
                "",
            ]
            if review.get("also_withheld")
            else []
        ),
        "Check the columns that are NOT in this list too. To withhold another column, add it to "
        "`privacy.pii_columns`. To release a column flagged by mistake, add it to `privacy.not_pii_columns`; "
        "its values were never collected, so they are sent only after the source is profiled again.",
        "",
        "## The profile exactly as Claude will see it",
        "",
        "```text",
        review["profile_text"].rstrip(),
        "```",
        "",
        "Approve it in the web app (Privacy review) or with `omop-agent approve privacy --by NAME`.",
        "",
    ]
    return "\n".join(lines)


def refresh(cfg: ProjectConfig, ws: Workspace) -> Path | None:
    """Write the review files from the current profile and settings. Returns the JSON path, or None
    when there is no profile yet (or when the project does not send data to Claude)."""
    path = review_path(ws)
    review = build_review(cfg, ws) if sends_to_llm(cfg) else None
    if review is None:
        for p in (path, ws.root / REVIEW_MD):
            p.unlink(missing_ok=True)
        return None
    atomic_write_text(path, json.dumps(review, indent=1, sort_keys=True) + "\n")
    atomic_write_text(ws.root / REVIEW_MD, render_md(review))
    return path
