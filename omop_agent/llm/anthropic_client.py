"""Claude via the official Anthropic Python SDK."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import anthropic
from anthropic import beta_tool

from ..config import LLMConfig
from ..workspace import Workspace
from .base import AgentResult, AgentTool, LLMClient, LLMError, T
from .replay import save_recording

# Server-side refusal fallback ("default" routes each refusal category to Anthropic's
# recommended fallback model). Disable with llm.use_fallbacks: false.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _usage(msg: Any) -> dict[str, Any]:
    u = getattr(msg, "usage", None)
    if u is None:
        return {}
    return {
        k: getattr(u, k, None)
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    }


class AnthropicLLM(LLMClient):
    provider = "anthropic"

    def __init__(self, cfg: LLMConfig, workspace: Workspace | None = None, client: anthropic.Anthropic | None = None):
        super().__init__(workspace)
        self.cfg = cfg
        self.model = cfg.model
        if client is None:
            api_key = next((os.environ[v] for v in cfg.api_key_env if os.environ.get(v)), None)
            # With no key in the listed variables, the SDK falls back to ANTHROPIC_AUTH_TOKEN or
            # an `ant auth login` profile.
            kwargs: dict[str, Any] = {"base_url": cfg.base_url} if cfg.base_url else {}
            if api_key:
                kwargs["api_key"] = api_key
            client = anthropic.Anthropic(**kwargs)
        self.client = client
        self.record_dir: Path | None = cfg.record_dir

    def _common(self, system: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.cfg.effort},
        }
        if self.cfg.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    @staticmethod
    def _check_stop(msg: Any, stage: str, key: str) -> None:
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"{stage}/{key}: model declined the request (category={category})")
        if msg.stop_reason == "max_tokens":
            raise LLMError(f"{stage}/{key}: response truncated at max_tokens; raise llm.max_tokens")

    def structured(self, *, stage: str, key: str, system: str, prompt: str, output: type[T]) -> T:
        started = time.time()
        kwargs = self._common(system)
        try:
            # Streaming keeps long structured answers clear of HTTP timeouts.
            with self.client.beta.messages.stream(
                max_tokens=self.cfg.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                output_format=output,
                **kwargs,
            ) as stream:
                msg = stream.get_final_message()
            self._check_stop(msg, stage, key)
            parsed = msg.parsed_output
            if parsed is None:
                raise LLMError(f"{stage}/{key}: no structured output in response")
        except (anthropic.APIError, LLMError) as e:
            self._log_call(
                stage=stage, key=key, kind="structured", system=system, prompt=prompt,
                response=None, started=started, error=str(e),
            )
            raise
        self._log_call(
            stage=stage, key=key, kind="structured", system=system, prompt=prompt,
            response=parsed.model_dump(mode="json"), started=started, usage=_usage(msg),
        )
        if self.record_dir:
            save_recording(self.record_dir, stage, key, {"kind": "structured", "output": parsed.model_dump(mode="json")})
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
        calls: list[dict[str, Any]] = []

        def wrap(tool: AgentTool):
            def _fn(**kwargs: Any) -> str:
                result = tool.invoke(kwargs)
                calls.append({"name": tool.name, "input": kwargs, "result": result[:4000]})
                return result

            return beta_tool(_fn, name=tool.name, description=tool.description, input_schema=tool.input_model)

        runner = self.client.beta.messages.tool_runner(
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            tools=[wrap(t) for t in tools],
            max_iterations=max_iterations or self.cfg.max_agent_iterations,
            **self._common(system),
        )
        try:
            final = runner.until_done()
            self._check_stop(final, stage, key)
        except (anthropic.APIError, LLMError) as e:
            self._log_call(
                stage=stage, key=key, kind="agent", system=system, prompt=prompt, response=None,
                started=started, tool_calls=calls, error=str(e),
            )
            raise
        text = "\n".join(b.text for b in final.content if b.type == "text").strip()
        self._log_call(
            stage=stage, key=key, kind="agent", system=system, prompt=prompt, response=text,
            started=started, usage=_usage(final), tool_calls=calls,
        )
        if self.record_dir:
            save_recording(
                self.record_dir, stage, key,
                {"kind": "agent", "tool_calls": [{"name": c["name"], "input": c["input"]} for c in calls], "final_text": text},
            )
        return AgentResult(final_text=text, tool_calls=calls)
