"""Replay recorded LLM answers so the pipeline runs deterministically without an API key.

Recordings live at ``<replay_dir>/<stage>/<key>.json``::

    {"kind": "structured", "output": {...}}                       # structured(...)
    {"kind": "agent", "tool_calls": [{"name": ..., "input": {...}}], "final_text": "..."}

For agent recordings the tool calls are re-executed against the live tools, so their side
effects (for example a repair to the ETL spec) really happen during the replay.

Set ``llm.record_dir`` while running against Claude to capture recordings of a live run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..workspace import Workspace
from .base import AgentResult, AgentTool, LLMClient, LLMUnavailable, T, safe_key


def recording_path(root: Path, stage: str, key: str) -> Path:
    return root / safe_key(stage) / f"{safe_key(key)}.json"


def save_recording(root: Path, stage: str, key: str, payload: dict[str, Any]) -> Path:
    path = recording_path(root, stage, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


class ReplayMissing(LLMUnavailable):
    pass


class ReplayLLM(LLMClient):
    provider = "replay"

    def __init__(self, replay_dir: Path, workspace: Workspace | None = None):
        super().__init__(workspace)
        self.replay_dir = Path(replay_dir)
        self.model = f"replay:{self.replay_dir.name}"

    def _load(self, stage: str, key: str, kind: str) -> dict[str, Any]:
        path = recording_path(self.replay_dir, stage, key)
        if not path.exists():
            raise ReplayMissing(f"no recording for {stage}/{key} at {path}")
        data = json.loads(path.read_text())
        if data.get("kind") != kind:
            raise ReplayMissing(f"recording {path} is a {data.get('kind')} call, expected {kind}")
        return data

    def structured(self, *, stage: str, key: str, system: str, prompt: str, output: type[T]) -> T:
        started = time.time()
        data = self._load(stage, key, "structured")
        parsed = output.model_validate(data["output"])
        self._log_call(
            stage=stage, key=key, kind="structured", system=system, prompt=prompt,
            response=data["output"], started=started,
        )
        return parsed

    def run_agent(
        self,
        *,
        stage: str,
        key: str,
        system: str,
        prompt: str,
        tools: list[AgentTool],
        max_iterations: int | None = None,
    ) -> AgentResult:
        started = time.time()
        data = self._load(stage, key, "agent")
        by_name = {t.name: t for t in tools}
        calls = []
        for call in data.get("tool_calls", []):
            tool = by_name.get(call["name"])
            if tool is None:
                result = f"Error: Tool '{call['name']}' not found"
            else:
                try:
                    result = tool.invoke(call.get("input", {}))
                except Exception as e:  # mirror AnthropicLLM: errors go back to the model as text
                    result = f"ERROR: {type(e).__name__}: {e}"
            calls.append({"name": call["name"], "input": call.get("input", {}), "result": result})
        text = data.get("final_text", "")
        self._log_call(
            stage=stage, key=key, kind="agent", system=system, prompt=prompt, response=text,
            started=started, tool_calls=calls,
        )
        return AgentResult(final_text=text, tool_calls=calls)
