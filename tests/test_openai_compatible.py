"""The OpenAI-compatible client for self-hosted open-weight models (llama.cpp, Ollama, vLLM)."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from omop_agent.config import LLMConfig
from omop_agent.llm import AgentTool, LLMError, LLMUnavailable, make_llm
from omop_agent.llm.openai_compatible import OpenAICompatibleLLM, endpoint_is_private, extract_json
from omop_agent.workspace import Workspace


class Answer(BaseModel):
    value: int = Field(ge=0)
    label: str = ""


class Server:
    """A scripted chat-completions endpoint: answers in order, records every request."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        self.requests.append(json.loads(request.content))
        a = self.answers.pop(0)
        if isinstance(a, httpx.Response):
            return a
        return httpx.Response(200, json=a)


def reply(content="", tool_calls=None, finish="stop", prompt_tokens=100, completion_tokens=20):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def call(name, args, id_="c1"):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args) if isinstance(args, dict) else args}}


def llm_for(server, ws=None, **cfg) -> OpenAICompatibleLLM:
    cfg.setdefault("base_url", "http://127.0.0.1:8080/v1")
    return OpenAICompatibleLLM(LLMConfig(**cfg), ws, client=httpx.Client(transport=httpx.MockTransport(server)))


def test_defaults_and_factory():
    cfg = LLMConfig()
    assert (cfg.provider, cfg.model, cfg.base_url, cfg.max_tokens) == (
        "openai_compatible", "gpt-oss-120b", "http://127.0.0.1:8080/v1", 16384,
    )
    assert cfg.api_key_env == ["OMOP_AGENT_LLM_API_KEY"]  # never the Anthropic keys
    assert isinstance(make_llm(cfg), OpenAICompatibleLLM)
    explicit = LLMConfig(model="qwen3.6-35b-a3b", base_url="http://127.0.0.1:11434/v1", max_tokens=4096)
    assert (explicit.model, explicit.base_url, explicit.max_tokens) == ("qwen3.6-35b-a3b", "http://127.0.0.1:11434/v1", 4096)


def test_structured_sends_the_schema_and_records(tmp_path):
    server = Server(reply('{"value": 7, "label": "ok"}'))
    ws = Workspace(tmp_path / "w")
    llm = llm_for(server, ws, model="gpt-oss-120b", reasoning_effort="high", temperature=0.2,
                  record_dir=tmp_path / "rec", extra_body={"top_k": 20})
    out = llm.structured(stage="design", key="plan", system="sys", prompt="hello", output=Answer)
    assert out == Answer(value=7, label="ok")
    body = server.requests[0]
    assert body["model"] == "gpt-oss-120b" and body["max_tokens"] == 16384
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == Answer.model_json_schema()
    assert (body["reasoning_effort"], body["temperature"], body["top_k"]) == ("high", 0.2, 20)
    record = json.loads(next((ws.root / "llm_calls").glob("*.json")).read_text())
    assert record["provider"] == "openai_compatible" and record["response"] == {"value": 7, "label": "ok"}
    assert record["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    saved = json.loads((tmp_path / "rec" / "design" / "plan.json").read_text())
    assert saved == {"kind": "structured", "output": {"value": 7, "label": "ok"}}


def test_invalid_answers_are_sent_back_with_the_errors(tmp_path):
    server = Server(reply('{"value": -3}'), reply('<think>fix it</think>```json\n{"value": 3}\n```'))
    ws = Workspace(tmp_path / "w")
    out = llm_for(server, ws).structured(stage="s", key="k", system="", prompt="p", output=Answer)
    assert out.value == 3
    retry = server.requests[1]["messages"]
    assert retry[2] == {"role": "assistant", "content": '{"value": -3}'}
    assert "does not validate" in retry[3]["content"] and "greater than or equal to 0" in retry[3]["content"]
    record = json.loads(next((ws.root / "llm_calls").glob("*.json")).read_text())
    assert record["usage"]["prompt_tokens"] == 200 and len(record["transcript"]) == 2


def test_gives_up_after_the_configured_retries(tmp_path):
    server = Server(reply("no json here"), reply('{"value": "x"}'))
    ws = Workspace(tmp_path / "w")
    with pytest.raises(LLMError, match="did not validate after 2 attempt"):
        llm_for(server, ws, structured_retries=1).structured(stage="s", key="k", system="", prompt="p", output=Answer)
    assert ws.read_audit()[-1]["error"]  # the failed call is logged


def test_prompt_mode_puts_the_schema_in_the_prompt():
    server = Server(reply('Sure! {"value": 1}'))
    out = llm_for(server, structured_output="prompt").structured(stage="s", key="k", system="", prompt="p", output=Answer)
    assert out.value == 1
    body = server.requests[0]
    assert "response_format" not in body and '"value"' in body["messages"][1]["content"]


def test_truncation_and_server_errors():
    with pytest.raises(LLMError, match="raise llm.max_tokens"):
        llm_for(Server(reply('{"val', finish="length"))).structured(stage="s", key="k", system="", prompt="p", output=Answer)
    with pytest.raises(LLMError, match="HTTP 500"):
        llm_for(Server(httpx.Response(500, text="boom"))).structured(stage="s", key="k", system="", prompt="p", output=Answer)
    with pytest.raises(LLMUnavailable, match="refused the credentials"):
        llm_for(Server(httpx.Response(401))).structured(stage="s", key="k", system="", prompt="p", output=Answer)

    def down(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailable, match="cannot reach the model server"):
        llm_for(down).structured(stage="s", key="k", system="", prompt="p", output=Answer)


def test_api_key_comes_from_the_configured_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-never-sent")
    monkeypatch.setenv("OMOP_AGENT_LLM_API_KEY", "local-key")
    llm = OpenAICompatibleLLM(LLMConfig())
    assert llm.client.headers["Authorization"] == "Bearer local-key"
    monkeypatch.delenv("OMOP_AGENT_LLM_API_KEY")
    assert "Authorization" not in OpenAICompatibleLLM(LLMConfig()).client.headers


def test_agent_runs_the_tool_loop(tmp_path):
    server = Server(
        reply("", [call("double", {"value": 4}), call("double", {"value": -1}, "c2")], finish="tool_calls"),
        reply("", [call("nope", {}), call("double", "{not json", "c4")], finish="tool_calls"),
        reply("<think>done</think>Fixed: doubled 4."),
    )
    ws = Workspace(tmp_path / "w")
    seen: list[int] = []

    def double(a: Answer) -> str:
        seen.append(a.value)
        return str(a.value * 2)

    tool = AgentTool("double", "double a value", Answer, double)
    res = llm_for(server, ws, agent_max_tokens=8000).run_agent(
        stage="repair", key="iteration:1", system="s", prompt="p", tools=[tool],
    )
    assert res.final_text == "Fixed: doubled 4." and seen == [4]
    results = [c["result"] for c in res.tool_calls]
    assert results[0] == "8" and results[1].startswith("REJECTED: invalid arguments")
    assert results[2].startswith("REJECTED: no tool named 'nope'") and "not valid JSON" in results[3]
    first = server.requests[0]
    assert first["tools"][0]["function"]["name"] == "double" and first["max_tokens"] == 8000
    assert first["tools"][0]["function"]["parameters"] == Answer.model_json_schema()
    second = server.requests[1]["messages"]
    assert second[2]["role"] == "assistant" and len(second[2]["tool_calls"]) == 2
    assert second[3] == {"role": "tool", "tool_call_id": "c1", "content": "8"}
    record = json.loads(next((ws.root / "llm_calls").glob("*.json")).read_text())
    assert record["usage"]["prompt_tokens"] == 300 and len(record["transcript"]) == 3


def test_agent_stops_at_the_iteration_limit(tmp_path):
    loop = reply("", [call("double", {"value": 1})], finish="tool_calls")
    ws = Workspace(tmp_path / "w")
    tool = AgentTool("double", "d", Answer, lambda a: "2")
    res = llm_for(Server(loop, loop), ws).run_agent(
        stage="repair", key="k", system="s", prompt="p", tools=[tool], max_iterations=2,
    )
    assert "iteration limit of 2 reached" in res.final_text and len(res.tool_calls) == 2
    assert any(e["event"] == "agent_iteration_limit" for e in ws.read_audit())


def test_agent_error_is_raised_after_logging(tmp_path):
    ws = Workspace(tmp_path / "w")
    tool = AgentTool("double", "d", Answer, lambda a: "2")
    server = Server(reply("", [call("double", {"value": 1})], finish="tool_calls"), httpx.Response(500, text="oom"))
    with pytest.raises(LLMError, match="HTTP 500"):
        llm_for(server, ws).run_agent(stage="repair", key="k", system="s", prompt="p", tools=[tool])
    record = json.loads(next((ws.root / "llm_calls").glob("*.json")).read_text())
    assert len(record["tool_calls"]) == 1 and "HTTP 500" in record["error"]


def test_helpers():
    assert extract_json('<think>{"a": 1}</think> Here: {"value": 2} thanks') == '{"value": 2}'
    assert extract_json('```json\n{"value": 2}\n```') == '{"value": 2}'
    assert endpoint_is_private("http://127.0.0.1:8080/v1") and endpoint_is_private("http://172.16.4.2/v1")
    assert not endpoint_is_private("https://api.example.com/v1") and not endpoint_is_private(None)


def test_llm_check_reports_each_check(tmp_path):
    from omop_agent.llm.check import EXPECTED, PIPELINE_SCHEMAS, run_checks

    choices = [{"source_code": c, "chosen_concept_id": min(ids), "equivalence": "EQUAL" if min(ids) else "UNMATCHED",
                "confidence": 0.9, "rationale": "r"} for c, ids in EXPECTED.items()]
    schema_ok = [reply("{") for _ in PIPELINE_SCHEMAS]
    server = Server(
        *schema_ok,
        reply(json.dumps({"choices": choices})),
        reply("", [call("count_records", {"table": "condition_occurrence"})], finish="tool_calls"),
        reply("There are 412 records."),
    )
    report = run_checks(llm_for(server))
    assert report.ok, report.checks
    assert [c.name for c in report.checks][-2:] == ["structured answer (code mapping)", "tool call round trip"]
    assert all(r["max_tokens"] == 1 for r in server.requests[: len(PIPELINE_SCHEMAS)])

    # a server that cannot compile a schema stops the check there, with the fix in the detail
    refused = Server(httpx.Response(400, text="failed to parse grammar"), *schema_ok[1:])
    report = run_checks(llm_for(refused))
    assert not report.ok and "structured_output: json_object" in report.checks[0].detail
    assert len(report.checks) == len(PIPELINE_SCHEMAS)
