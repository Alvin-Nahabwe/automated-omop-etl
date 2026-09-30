"""Provider-independent LLM interface used by every pipeline stage.

Stages ask for exactly two things:

* ``structured(...)`` - one request whose answer must validate against a pydantic model
  (source annotations, table plans, field mappings, vocabulary choices);
* ``run_agent(...)`` - a tool-using loop in which the model calls functions we control
  (the data-quality repair agent).

Every call carries a ``stage`` and a ``key``. Together they name the call in the audit trail
and let ``ReplayLLM`` return a recorded answer, which is how the demo and tests run offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from ..workspace import Workspace

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """No model is configured (provider 'none') or no recording exists for this call."""


class LLMError(RuntimeError):
    """The model answered but the answer is unusable (refusal, truncation, invalid output)."""


@dataclass
class AgentTool:
    """A function the agent may call. ``fn`` receives the validated input model, returns text."""

    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any], str]

    def invoke(self, raw_input: dict[str, Any]) -> str:
        return self.fn(self.input_model.model_validate(raw_input))


@dataclass
class AgentResult:
    final_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")[:120] or "call"


class LLMClient(ABC):
    provider: str = "abstract"
    model: str = ""

    def __init__(self, workspace: Workspace | None = None):
        self.workspace = workspace
        self._call_no = 0

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def structured(self, *, stage: str, key: str, system: str, prompt: str, output: type[T]) -> T: ...

    @abstractmethod
    def run_agent(
        self,
        *,
        stage: str,
        key: str,
        system: str,
        prompt: str,
        tools: list[AgentTool],
        max_iterations: int | None = None,
    ) -> AgentResult: ...

    # ---- transparency log: exactly what was sent and received -------------------------
    def _log_call(
        self,
        *,
        stage: str,
        key: str,
        kind: str,
        system: str,
        prompt: str,
        response: Any,
        started: float,
        usage: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        if self.workspace is None:
            return
        self._call_no += 1
        calls_dir = self.workspace.root / "llm_calls"
        calls_dir.mkdir(exist_ok=True)
        record = {
            "provider": self.provider,
            "model": self.model,
            "stage": stage,
            "key": key,
            "kind": kind,
            "system": system,
            "prompt": prompt,
            "response": response,
            "tool_calls": tool_calls or [],
            "usage": usage or {},
            "error": error,
        }
        n = len(list(calls_dir.glob("*.json"))) + 1
        path = calls_dir / f"{n:04d}_{safe_key(stage)}_{safe_key(key)}.json"
        path.write_text(json.dumps(record, indent=2, default=str, ensure_ascii=False))
        self.workspace.audit(
            "llm_call",
            stage=stage,
            key=key,
            provider=self.provider,
            model=self.model,
            kind=kind,
            prompt_sha256=hashlib.sha256((system + "\n" + prompt).encode()).hexdigest(),
            prompt_chars=len(system) + len(prompt),
            seconds=round(time.time() - started, 2),
            usage=usage or {},
            n_tool_calls=len(tool_calls or []),
            error=error,
            record=str(path.relative_to(self.workspace.root)),
        )


class NullLLM(LLMClient):
    """provider: none - every call raises, and stages fall back to deterministic heuristics."""

    provider = "none"

    @property
    def available(self) -> bool:
        return False

    def structured(self, *, stage: str, key: str, system: str, prompt: str, output: type[T]) -> T:
        raise LLMUnavailable("LLM provider is 'none'")

    def run_agent(self, *, stage, key, system, prompt, tools, max_iterations=None) -> AgentResult:  # type: ignore[override]
        raise LLMUnavailable("LLM provider is 'none'")
