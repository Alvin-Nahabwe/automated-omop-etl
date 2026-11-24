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
    assert (cfg.llm.provider, cfg.llm.model) == ("openai_compatible", "gpt-oss-120b")


def test_config_etl_settings_and_release_date(tmp_path, monkeypatch):
    from datetime import date, timedelta

    from pydantic import ValidationError

    from omop_agent.config import CdmSourceConfig, EtlConfig, ProjectConfig, SourceConfig

    monkeypatch.delenv("RELEASE_DATE", raising=False)
    (tmp_path / "omop_agent.yaml").write_text(
        "name: t\nsource: {type: csv, path: ./data}\ntarget: {url: 'postgresql://x@y/z'}\n"
        "cdm_source: {source_release_date: '${RELEASE_DATE:-}'}\netl: {timezone: Africa/Nairobi}\n"
    )
    cfg = load_config(tmp_path / "omop_agent.yaml")
    assert cfg.cdm_source.source_release_date is None  # an unset ${VAR:-} means "derive it"
    assert cfg.etl == EtlConfig(timezone="Africa/Nairobi", partial_dates="widest")
    monkeypatch.setenv("RELEASE_DATE", "2023-12-31")
    assert load_config(tmp_path / "omop_agent.yaml").cdm_source.source_release_date == date(2023, 12, 31)
    # defaults: no zone (offsets ignored, wall clock kept), widest partial dates
    plain = ProjectConfig.model_validate({"name": "t", "source": {"type": "csv", "path": "."},
                                          "target": {"url": "postgresql://x@y/z"}})
    assert plain.etl.timezone is None and plain.etl.partial_dates == "widest"
    with pytest.raises(ValidationError, match="not an IANA time zone"):
        EtlConfig(timezone="Mars/Olympus_Mons")
    with pytest.raises(ValidationError):
        EtlConfig(partial_dates="middle")
    # a release date after today would fail DQD's plausibleTemporalAfter on cdm_release_date
    with pytest.raises(ValidationError, match="in the future"):
        CdmSourceConfig(source_release_date=date.today() + timedelta(days=1))
    assert CdmSourceConfig(source_release_date=date.today()).source_release_date == date.today()

    # missing-value markers: a CSV default, none for databases, or exactly the configured list
    csv_src = SourceConfig(type="csv", path=tmp_path)
    assert csv_src.effective_na_values == ["NA", "NULL", "\\N", "."]
    assert SourceConfig(type="database", url="postgresql://x@y/z").effective_na_values == []
    assert SourceConfig(type="csv", path=tmp_path, na_values=[]).effective_na_values == []
    assert SourceConfig(type="csv", path=tmp_path, na_values=["-99"]).effective_na_values == ["-99"]


def test_new_spec_fields_default_to_the_new_behaviour():
    from omop_agent.models import ETLSpec, MappingDecision, SourceTerm, TableMapping

    spec = ETLSpec.model_validate({"person_key": {"table": "p", "expression": "id"}, "table_mappings": []})
    assert spec.route_by_domain is True and spec.derive_eras is True
    tm = TableMapping.model_validate({"target_table": "death", "source": {"table": "d", "alias": "d"}, "fields": []})
    assert tm.dedup_order is None
    assert "dedup_order" in TableMapping.model_json_schema()["properties"]
    term = SourceTerm(source_vocabulary_id="V", source_code="c", source_name="c", frequency=1)
    decision = MappingDecision(source_vocabulary_id="V", source_code="c", source_name="c", frequency=1)
    assert term.accepted_domains == [] and decision.accepted_domains == []


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
    cfg = LLMConfig(provider="anthropic", record_dir=tmp_path / "rec")
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
    llm = AnthropicLLM(LLMConfig(provider="anthropic", use_fallbacks=False), None, client=_fake_client(msg, []))  # type: ignore[arg-type]
    with pytest.raises(LLMError, match="declined"):
        llm.structured(stage="s", key="k", system="", prompt="", output=Answer)


def test_api_key_env_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-test")
    llm = AnthropicLLM(LLMConfig(provider="anthropic"))
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


def test_privacy_dates_are_not_phones_and_code_ranges_hide_rare_values():
    cfg = PrivacyConfig()
    assert detect_pii("visits", "visit_date", ["2019-01-05"] * 6, cfg) is None
    assert detect_pii("people", "dob", ["19/05/1942"] * 6, cfg) is None
    sex = ColumnProfile(
        name="sex", inferred_type="integer", n_rows=100, n_null=0, n_distinct=3, unique=False,
        min_value="1", max_value="9",
        top_values=[ValueFreq(value="1", count=50), ValueFreq(value="2", count=48), ValueFreq(value="9", count=2)],
    )
    profile = SourceProfile(
        source_type="csv", generated_at="2026-01-01T00:00:00Z", min_cell_count=5,
        tables=[TableProfile(name="people", n_rows=100, columns=[sex])],
    )
    view = llm_view(profile, cfg).tables[0].columns[0]
    assert view.min_value is None and view.max_value is None
    assert [v.value for v in view.top_values] == ["1", "2"]


class _Stream(_FakeStream):
    def __init__(self, exc):
        self.exc = exc

    def get_final_message(self):
        raise self.exc


def test_structured_validation_error_becomes_llm_error(tmp_path):
    try:
        Answer.model_validate({"value": "not-a-number"})
    except Exception as e:  # a real pydantic.ValidationError
        exc = e
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: _Stream(exc))))
    ws = Workspace(tmp_path / "w")
    llm = AnthropicLLM(LLMConfig(provider="anthropic"), ws, client=client)  # type: ignore[arg-type]
    with pytest.raises(LLMError, match="did not validate"):
        llm.structured(stage="s", key="k", system="", prompt="p", output=Answer)
    assert ws.read_audit()[-1]["error"]  # the failed call is still logged


def test_missing_credentials_become_llm_unavailable():
    exc = TypeError('"Could not resolve authentication method. Expected either api_key or auth_token to be set."')
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: _Stream(exc))))
    llm = AnthropicLLM(LLMConfig(provider="anthropic"), None, client=client)  # type: ignore[arg-type]
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        llm.structured(stage="s", key="k", system="", prompt="p", output=Answer)


def _msg(stop_reason, text="", usage=10):
    content = [SimpleNamespace(type="text", text=text)] if text else []
    return SimpleNamespace(
        stop_reason=stop_reason, content=content, stop_details=None,
        usage=SimpleNamespace(input_tokens=usage, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


def test_agent_sums_usage_reports_limit_and_explains_bad_tool_input(tmp_path):
    seen: dict = {}

    def tool_runner(**kwargs):
        seen.update(kwargs)
        tool = kwargs["tools"][0]
        # the model sends an invalid argument; the tool must answer with details, not raise
        seen["bad"] = tool.call({"value": "oops"})
        seen["good"] = tool.call({"value": 4})
        return iter([_msg("tool_use"), _msg("tool_use"), _msg("tool_use", "still working")])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner)))
    ws = Workspace(tmp_path / "w")
    llm = AnthropicLLM(LLMConfig(provider="anthropic", agent_max_tokens=20000), ws, client=client)  # type: ignore[arg-type]
    tool = AgentTool("double", "double a value", Answer, lambda a: str(a.value * 2))
    res = llm.run_agent(stage="repair", key="iteration:1", system="s", prompt="p", tools=[tool], max_iterations=3)
    assert seen["max_tokens"] == 20000
    assert seen["bad"].startswith("REJECTED: invalid arguments") and "value" in seen["bad"]
    assert seen["good"] == "8"
    assert "iteration limit of 3 reached" in res.final_text
    record = json.loads(sorted((ws.root / "llm_calls").glob("*.json"))[0].read_text())
    assert record["usage"]["input_tokens"] == 30  # summed over the three turns
    assert len(record["transcript"]) == 3 and len(record["tool_calls"]) == 2
    assert any(e["event"] == "agent_iteration_limit" for e in ws.read_audit())
