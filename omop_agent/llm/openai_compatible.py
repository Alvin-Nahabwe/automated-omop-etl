"""Self-hosted open-weight models behind an OpenAI-compatible chat-completions server.

This works with llama.cpp's ``llama-server`` (start it with ``--jinja`` so tool calls work),
ik_llama.cpp, Ollama (``/v1``), vLLM and SGLang. Nothing here is specific to one model.

Two things make a smaller local model dependable on our tasks:

* constrained decoding: ``structured`` sends the answer's JSON Schema as ``response_format``,
  so the server can only produce JSON of that shape (llama.cpp compiles it to a grammar);
* validate and retry: an answer that still fails the pydantic model (a value out of range, a
  missing field the grammar could not express) goes back to the model with the validation
  errors, up to ``llm.structured_retries`` times.

The repair agent's tool loop is driven here (the server only proposes tool calls), so every
call goes through ``AgentTool.invoke`` with the same validation and error reporting as with
Claude. Reasoning models (gpt-oss, Qwen thinking) may put their reasoning in a separate field
or in ``<think>`` tags; only the final answer is used.

Keep the system prompt identical across calls: llama-server reuses the cached prefix, which
saves most of the (slow, on CPU) prompt processing.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pydantic

from ..config import LLMConfig
from ..workspace import Workspace
from .base import AgentResult, AgentTool, LLMClient, LLMError, LLMUnavailable, T
from .replay import save_recording

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def endpoint_is_private(base_url: str | None) -> bool:
    """True for a loopback or private-network address (the model runs on our own machines).

    That is an IP address in a loopback, private or link-local range, ``localhost``, a name under
    ``.internal``, ``.local`` or ``.localhost``, or a name without dots (a Docker service such as
    ``llm``, or a host on the local network): none of these is reachable on the public internet.
    """
    host = (urlparse(base_url or "").hostname or "").lower().rstrip(".")
    if not host:
        return False
    if "." not in host and ":" not in host:
        return True
    if host.endswith((".localhost", ".internal", ".local")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def schema_name(model: type[pydantic.BaseModel]) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", model.__name__)[:64] or "answer"


def extract_json(text: str) -> str:
    """The JSON part of a model's answer: drop reasoning tags and code fences, keep the outer object."""
    text = _THINK.sub("", text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if text and text[0] not in "{[":
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start >= 0:
            text = text[start:]
    if text and text[0] in "{[":
        close = "}" if text[0] == "{" else "]"
        end = text.rfind(close)
        if end >= 0:
            text = text[: end + 1]
    return text


def _usage(body: dict[str, Any]) -> dict[str, int]:
    u = body.get("usage") or {}
    return {k: int(u[k]) for k in ("prompt_tokens", "completion_tokens") if isinstance(u.get(k), (int, float))}


def _add_usage(total: dict[str, int], more: dict[str, int]) -> None:
    for k, v in more.items():
        total[k] = total.get(k, 0) + v


class OpenAICompatibleLLM(LLMClient):
    provider = "openai_compatible"

    def __init__(self, cfg: LLMConfig, workspace: Workspace | None = None, client: httpx.Client | None = None):
        super().__init__(workspace)
        self.cfg = cfg
        self.model = cfg.model
        self.base_url = (cfg.base_url or "").rstrip("/")
        api_key = next((os.environ[v] for v in cfg.api_key_env if os.environ.get(v)), None)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_seconds, connect=30.0),
            # A local server must not be reached through the outbound proxy.
            trust_env=not endpoint_is_private(self.base_url),
        )
        self.record_dir: Path | None = cfg.record_dir

    # ---- HTTP ----------------------------------------------------------------------------
    def _post(self, payload: dict[str, Any], stage: str, key: str) -> dict[str, Any]:
        body = {"model": self.cfg.model, **self.cfg.extra_body, **payload}
        if self.cfg.temperature is not None:
            body.setdefault("temperature", self.cfg.temperature)
        if self.cfg.reasoning_effort:
            body.setdefault("reasoning_effort", self.cfg.reasoning_effort)
        url = f"{self.base_url}/chat/completions"
        try:
            r = self.client.post(url, json=body)
        except httpx.ConnectError as e:
            raise LLMUnavailable(
                f"cannot reach the model server at {self.base_url} ({e}). Start it (e.g. llama-server "
                "--jinja ...) or set llm.base_url."
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError(f"{stage}/{key}: the model server did not answer within llm.timeout_seconds ({e})") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{stage}/{key}: {type(e).__name__}: {e}") from e
        if r.status_code in (401, 403):
            raise LLMUnavailable(f"the model server at {self.base_url} refused the credentials (HTTP {r.status_code})")
        if r.status_code >= 400:
            raise LLMError(f"{stage}/{key}: the model server answered HTTP {r.status_code}: {r.text[:500]}")
        try:
            data = r.json()
            data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{stage}/{key}: unexpected response from the model server: {r.text[:500]}") from e
        return data

    def _response_format(self, output: type[pydantic.BaseModel]) -> dict[str, Any] | None:
        mode = self.cfg.structured_output
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": schema_name(output), "schema": output.model_json_schema(), "strict": True},
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def schema_accepted(self, output: type[pydantic.BaseModel]) -> tuple[bool, str]:
        """Whether the server can constrain an answer to this schema (asks for one token)."""
        fmt = self._response_format(output)
        if fmt is None or fmt["type"] != "json_schema":
            return True, f"not checked (llm.structured_output: {self.cfg.structured_output})"
        try:
            self._post(
                {"messages": [{"role": "user", "content": "{}"}], "max_tokens": 1, "response_format": fmt},
                "llm_check", output.__name__,
            )
        except LLMError as e:
            return False, f"{e} (try llm.structured_output: json_object)"
        return True, "accepted"

    def _schema_instruction(self, output: type[pydantic.BaseModel]) -> str:
        return (
            "\n\nAnswer with one JSON object only (no prose, no code fences) that validates against "
            f"this JSON Schema:\n{json.dumps(output.model_json_schema(), separators=(',', ':'))}"
        )

    # ---- structured answers --------------------------------------------------------------
    def structured(self, *, stage: str, key: str, system: str, prompt: str, output: type[T]) -> T:
        started = time.time()
        usage: dict[str, int] = {}
        transcript: list[dict[str, Any]] = []
        user = prompt if self.cfg.structured_output == "json_schema" else prompt + self._schema_instruction(output)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload: dict[str, Any] = {"max_tokens": self.cfg.max_tokens}
        fmt = self._response_format(output)
        if fmt is not None:
            payload["response_format"] = fmt
        parsed: T | None = None
        error: LLMError | LLMUnavailable | None = None
        for attempt in range(self.cfg.structured_retries + 1):
            try:
                data = self._post({**payload, "messages": messages}, stage, key)
            except (LLMError, LLMUnavailable) as e:
                error = e
                break
            _add_usage(usage, _usage(data))
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            transcript.append({"attempt": attempt + 1, "finish_reason": choice.get("finish_reason"), "content": content})
            if choice.get("finish_reason") == "length":
                error = LLMError(f"{stage}/{key}: response truncated at max_tokens; raise llm.max_tokens")
                break
            try:
                parsed = output.model_validate_json(extract_json(content))
                error = None
                break
            except pydantic.ValidationError as e:
                error = LLMError(f"{stage}/{key}: the structured answer did not validate after {attempt + 1} attempt(s): {e}")
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "That answer does not validate against the schema:\n"
                        f"{e}\nReply with the corrected JSON object only.",
                    },
                ]
        if parsed is None:
            err = error or LLMError(f"{stage}/{key}: no structured output in response")
            self._log_call(
                stage=stage, key=key, kind="structured", system=system, prompt=prompt, response=None,
                started=started, usage=usage, transcript=transcript, error=str(err),
            )
            raise err
        self._log_call(
            stage=stage, key=key, kind="structured", system=system, prompt=prompt,
            response=parsed.model_dump(mode="json"), started=started, usage=usage,
            transcript=transcript if len(transcript) > 1 else None,
        )
        if self.record_dir:
            save_recording(self.record_dir, stage, key, {"kind": "structured", "output": parsed.model_dump(mode="json")})
        return parsed

    # ---- tool-using agent ----------------------------------------------------------------
    @staticmethod
    def _tool_spec(tool: AgentTool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_model.model_json_schema(),
            },
        }

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
        by_name = {t.name: t for t in tools}
        specs = [self._tool_spec(t) for t in tools]
        calls: list[dict[str, Any]] = []
        transcript: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        limit = max_iterations or self.cfg.max_agent_iterations
        text = ""
        error: LLMError | LLMUnavailable | None = None
        stopped_at_limit = False
        try:
            for i in range(limit):
                data = self._post(
                    {"messages": messages, "tools": specs, "tool_choice": "auto", "max_tokens": self.cfg.agent_max_tokens},
                    stage, key,
                )
                _add_usage(usage, _usage(data))
                choice = data["choices"][0]
                msg = choice["message"]
                content = _THINK.sub("", msg.get("content") or "").strip()
                tool_calls = msg.get("tool_calls") or []
                transcript.append({
                    "finish_reason": choice.get("finish_reason"),
                    "text": [content] if content else [],
                    "tool_uses": [(tc.get("function") or {}).get("name") for tc in tool_calls],
                })
                if not tool_calls:
                    if choice.get("finish_reason") == "length":
                        raise LLMError(f"{stage}/{key}: response truncated at max_tokens; raise llm.agent_max_tokens")
                    text = content
                    break
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                for n, tc in enumerate(tool_calls):
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    raw = fn.get("arguments")
                    try:
                        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be a JSON object")
                    except ValueError as e:
                        args, result = {"_raw": raw}, f"REJECTED: arguments for {name} are not valid JSON: {e}"
                    else:
                        tool = by_name.get(name)
                        if tool is None:
                            result = f"REJECTED: no tool named {name!r}; the tools are: {', '.join(by_name)}"
                        else:
                            try:
                                result = tool.invoke(args)
                            except pydantic.ValidationError as e:
                                result = f"REJECTED: invalid arguments for {name}: {e}"
                            except Exception as e:  # a tool bug must not abort the run; the model sees the error
                                result = f"ERROR: {type(e).__name__}: {e}"
                    calls.append({"name": name, "input": args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{i}_{n}", "content": result})
                text = content
            else:
                stopped_at_limit = True
        except (LLMError, LLMUnavailable) as e:
            error = e
        if stopped_at_limit:
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
