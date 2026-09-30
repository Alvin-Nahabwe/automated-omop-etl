from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from omop_agent.config import LLMConfig, expand_env, load_config
from omop_agent.llm import AgentTool, LLMError, LLMUnavailable, NullLLM, ReplayLLM, make_llm
from omop_agent.llm.anthropic_client import FALLBACK_BETA, AnthropicLLM
from omop_agent.llm.replay import recording_path, save_recording
from omop_agent.privacy import detect_pii, llm_view, profile_prompt_text
from omop_agent.config import PrivacyConfig
from omop_agent.models import ColumnProfile, SourceProfile, TableProfile, ValueFreq
from omop_agent.workspace import Workspace


class Answer(BaseModel):
    value: int


def test_expand_env(monkeypatch):
    monkeypatch.setenv("PGPASS", "s3cret")
    assert expand_env("postgresql://u:${PGPASS}@h/db") == "postgresql://u:s3cret@h/db"
    assert expand_env("${NOT_SET_X:-fallback}") == "fallback"
    with pytest.raises(ValueError):
        expand_env("${NOT_SET_Y}")


def test_load_config_resolves_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DBURL", "postgresql://x@y/z")
    (tmp_path / "omop_agent.yaml").write_text(
        "name: t\nsource: {type: csv, path: ./data}\ntarget: {url: '${DBURL}'}\nvocabulary: {athena_dir: ./vocab}\n"
    )
    cfg = load_config(tmp_path / "omop_agent.yaml")
    assert cfg.target.url == "postgresql://x@y/z"
    assert cfg.source.path == (tmp_path / "data").resolve()
    assert cfg.workdir == (tmp_path / "work").resolve()
    assert cfg.llm.model == "claude-opus-5-5"


def test_replay_structured_and_agent(tmp_path):
    ws = Workspace(tmp_path / "work")
    save_recording(tmp_path / "rec", "stage", "a:b", {"kind": "structured", "output": {"value": 3}})
    save_recording(
        tmp_path / "rec", "stage", "agent",
        {"kind": "agent", "tool_calls": [{"name": "add", "input": {"value": 2}}, {"name": "nope", "input": {}}], "final_text": "done"},
    )
    assert recording_path(tmp_path / "rec", "stage", "a:b").name == "a_b.json"
    llm = ReplayLLM(tmp_path / "rec", ws)
    assert llm.structured(stage="stage", key="a:b", system="s", prompt="p", output=Answer).value == 3

    seen = []
    tool = AgentTool("add", "add", Answer, lambda a: seen.append(a.value) or f"added {a.value}")
    res = llm.run_agent(stage="stage", key="agent", system="s", prompt="p", tools=[tool])
    assert seen == [2] and res.final_text == "done"
    assert "not found" in res.tool_calls[1]["result"]
    with pytest.raises(LLMUnavailable):
        llm.structured(stage="stage", key="missing", system="s", prompt="p", output=Answer)
    # every call is logged verbatim for audit
    records = sorted((ws.root / "llm_calls").glob("*.json"))
    assert len(records) == 2
    assert json.loads(records[0].read_text())["prompt"] == "p"
    assert [e["event"] for e in ws.read_audit()] == ["llm_call", "llm_call"]


def test_null_llm():
    llm = make_llm(LLMConfig(provider="none"))
    assert isinstance(llm, NullLLM) and not llm.available
    with pytest.raises(LLMUnavailable):
        llm.structured(stage="s", key="k", system="", prompt="", output=Answer)


class _FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def _fake_client(msg, calls):
    def stream(**kwargs):
        calls.append(kwargs)
        return _FakeStream(msg)

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))


def test_anthropic_structured_request_shape(tmp_path):
    msg = SimpleNamespace(
        stop_reason="end_turn", parsed_output=Answer(value=7), stop_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )
    calls: list[dict] = []
    cfg = LLMConfig(record_dir=tmp_path / "rec")
    llm = AnthropicLLM(cfg, Workspace(tmp_path / "w"), client=_fake_client(msg, calls))  # type: ignore[arg-type]
    out = llm.structured(stage="design", key="plan", system="sys", prompt="hello", output=Answer)
    assert out.value == 7
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}
    assert kw["output_format"] is Answer
    assert kw["fallbacks"] == "default" and kw["betas"] == [FALLBACK_BETA]
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    # the live answer was recorded for replay
    assert ReplayLLM(tmp_path / "rec").structured(stage="design", key="plan", system="", prompt="", output=Answer).value == 7


def test_anthropic_refusal_raises(tmp_path):
    msg = SimpleNamespace(stop_reason="refusal", parsed_output=None, stop_details=SimpleNamespace(category="bio"), usage=None)
    llm = AnthropicLLM(LLMConfig(use_fallbacks=False), None, client=_fake_client(msg, []))  # type: ignore[arg-type]
    with pytest.raises(LLMError, match="declined"):
        llm.structured(stage="s", key="k", system="", prompt="", output=Answer)


def test_api_key_env_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-test")
    llm = AnthropicLLM(LLMConfig())
    assert llm.client.api_key == "sk-ant-test"


def _col(name, values=None, **kw):
    return ColumnProfile(
        name=name, inferred_type="string", n_rows=100, n_null=0, n_distinct=len(values or []), unique=False,
        top_values=[ValueFreq(value=v, count=c) for v, c in (values or [])], **kw,
    )


def test_privacy_view():
    cfg = PrivacyConfig()
    assert detect_pii("people", "first_name", [], cfg)
    assert detect_pii("people", "contact", ["0712345678"] * 6, cfg)
    assert detect_pii("people", "sex", ["1", "2"], cfg) is None
    profile = SourceProfile(
        source_type="csv", generated_at="2026-01-01T00:00:00Z", min_cell_count=5,
        tables=[TableProfile(name="t", n_rows=100, columns=[
            _col("drug", [("Paracetamol", 50), ("Rare drug", 2)]),
            _col("phone", [("0712", 9)], is_pii=True, pii_reason="phone", min_value="0712"),
        ])],
    )
    view = llm_view(profile, cfg)
    drug, phone = view.tables[0].columns
    assert [v.value for v in drug.top_values] == ["Paracetamol"] and drug.n_suppressed_values == 1
    assert phone.top_values == [] and phone.min_value is None
    text = profile_prompt_text(profile, cfg)
    assert "Rare drug" not in text and "0712" not in text and "Paracetamol" in text
    assert "values withheld" in text
    no_values = profile_prompt_text(profile, PrivacyConfig(send_values_to_llm=False))
    assert "Paracetamol" not in no_values
