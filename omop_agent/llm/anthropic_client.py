"""Claude via the official Anthropic Python SDK."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import anthropic
import pydantic
from anthropic import beta_tool

from ..config import LLMConfig
from ..workspace import Workspace
from .base import AgentResult, AgentTool, LLMClient, LLMError, LLMUnavailable, T
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
    def _check_stop(msg: Any, stage: str, key: str, max_tokens_setting: str = "llm.max_tokens") -> None:
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"{stage}/{key}: model declined the request (category={category})")
        if msg.stop_reason == "max_tokens":
            raise LLMError(f"{stage}/{key}: response truncated at max_tokens; raise {max_tokens_setting}")

    @staticmethod
    def _as_llm_error(e: Exception, stage: str, key: str) -> LLMError | LLMUnavailable:
        """Map SDK/validation failures to the interface's errors (callers handle only these)."""
        if isinstance(e, (LLMError, LLMUnavailable)):
            return e
        if isinstance(e, TypeError) and "authentication" in str(e).lower():
            return LLMUnavailable(
                "No Anthropic credentials found: set ANTHROPIC_API_KEY (or CLAUDE_API_KEY), or run `ant auth login`."
            )
        if isinstance(e, anthropic.AuthenticationError):
            return LLMUnavailable(f"Anthropic API rejected the credentials: {e}")
        if isinstance(e, pydantic.ValidationError):
            return LLMError(f"{stage}/{key}: the structured answer did not validate (truncated or refused mid-answer?): {e}")
        return LLMError(f"{stage}/{key}: {type(e).__name__}: {e}")

    def structured(self, *, stage: str, key: str, system: str, prompt: str, output: type[T]) -> T:
        started = time.time()
        kwargs = self._common(system)
        msg = None
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
        except (anthropic.APIError, LLMError, LLMUnavailable, pydantic.ValidationError, TypeError) as e:
            err = self._as_llm_error(e, stage, key)
            self._log_call(
                stage=stage, key=key, kind="structured", system=system, prompt=prompt,
                response=None, started=started, usage=_usage(msg) if msg is not None else None, error=str(err),
            )
            raise err from e
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
        transcript: list[dict[str, Any]] = []
        usage: dict[str, int] = {}

        def wrap(tool: AgentTool):
            def _fn(**kwargs: Any) -> str:
                try:
                    result = tool.invoke(kwargs)
                except pydantic.ValidationError as e:
                    # Give the model the details (the SDK would only say "Invalid arguments").
                    result = f"REJECTED: invalid arguments for {tool.name}: {e}"
                except Exception as e:  # a tool bug must not abort the run; the model sees the error
                    result = f"ERROR: {type(e).__name__}: {e}"
                calls.append({"name": tool.name, "input": kwargs, "result": result})
                return result

            return beta_tool(_fn, name=tool.name, description=tool.description, input_schema=tool.input_model)

        limit = max_iterations or self.cfg.max_agent_iterations
        final = None
        error: LLMError | LLMUnavailable | None = None
        try:
            runner = self.client.beta.messages.tool_runner(
                max_tokens=self.cfg.agent_max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=[wrap(t) for t in tools],
                max_iterations=limit,
                **self._common(system),
            )
            for message in runner:
                final = message
                for k, v in _usage(message).items():
                    usage[k] = usage.get(k, 0) + (v or 0)
                texts = [b.text for b in message.content if b.type == "text" and b.text.strip()]
                uses = [b.name for b in message.content if b.type == "tool_use"]
                transcript.append({"stop_reason": message.stop_reason, "text": texts, "tool_uses": uses})
            if final is None:
                raise LLMError(f"{stage}/{key}: the agent returned no message")
            self._check_stop(final, stage, key, "llm.agent_max_tokens")
        except (anthropic.APIError, LLMError, LLMUnavailable, pydantic.ValidationError, TypeError) as e:
            error = self._as_llm_error(e, stage, key)
        text = ""
        if final is not None:
            text = "\n".join(b.text for b in final.content if b.type == "text").strip()
            if error is None and final.stop_reason == "tool_use":
                text = (text + "\n" if text else "") + f"(stopped: iteration limit of {limit} reached)"
                if self.workspace:
                    self.workspace.audit("agent_iteration_limit", stage=stage, key=key, limit=limit)
        self._log_call(
            stage=stage, key=key, kind="agent", system=system, prompt=prompt, response=text,
            started=started, usage=usage, tool_calls=calls, transcript=transcript,
            error=str(error) if error else None,
        )
        if error is not None:
            # Tool calls already made (e.g. applied repairs) stay visible to the caller via the log.
            raise error
        if self.record_dir:
            save_recording(
                self.record_dir, stage, key,
                {"kind": "agent", "tool_calls": [{"name": c["name"], "input": c["input"]} for c in calls], "final_text": text},
            )
        return AgentResult(final_text=text, tool_calls=calls)
