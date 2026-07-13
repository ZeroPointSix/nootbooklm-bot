from types import SimpleNamespace

import pytest

from agent.llm_caller import AgentRuntime, _safe_tool_output
from notebooklm_tool import ToolDefinition


class Streamer:
    def __init__(self):
        self.items = []

    def append(self, **kwargs):
        self.items.append(kwargs)


class Provider:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [ToolDefinition("notebook_list", "list", {"type": "object"})]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"content": [{"type": "text", "text": "notebooks"}]}


def event_call(name="notebook_list", arguments="{}"):
    item = SimpleNamespace(
        type="function_call",
        id="item1",
        call_id="call1",
        name=name,
        arguments=arguments,
    )
    return SimpleNamespace(type="response.output_item.done", item=item)


class Responses:
    def __init__(self, rounds):
        self.rounds = iter(rounds)

    def create(self, **_kwargs):
        return next(self.rounds)


def runtime(rounds, max_rounds=8):
    provider = Provider()
    llm = SimpleNamespace(responses=Responses(rounds))
    return (
        AgentRuntime(provider, api_key="test", llm=llm, max_tool_rounds=max_rounds),
        provider,
    )


def test_dynamic_tool_call_is_forwarded_and_result_returned_to_model():
    rt, provider = runtime([[event_call()], []])
    prompts = [{"role": "user", "content": "列出 notebooks"}]
    rt.run(Streamer(), prompts)
    assert provider.calls == [("notebook_list", {})]
    assert prompts[-1]["type"] == "function_call_output"


def test_unknown_tool_is_not_forwarded():
    rt, provider = runtime([[event_call("delete_everything")], []])
    prompts = []
    rt.run(Streamer(), prompts)
    assert provider.calls == []
    assert "UNKNOWN_TOOL" in prompts[-1]["output"]


def test_invalid_json_is_not_forwarded():
    rt, provider = runtime([[event_call(arguments="{bad")], []])
    prompts = []
    rt.run(Streamer(), prompts)
    assert provider.calls == []
    assert "INVALID_ARGUMENTS" in prompts[-1]["output"]


def test_tool_loop_is_bounded():
    rt, _ = runtime([[event_call()], [event_call()]], max_rounds=1)
    with pytest.raises(RuntimeError, match="安全上限"):
        rt.run(Streamer(), [])


def test_sensitive_tool_result_fields_are_redacted_recursively():
    output = _safe_tool_output(
        {"data": {"cookie": "secret", "safe": "ok"}, "access_token": "secret"}
    )
    assert "secret" not in output
    assert output.count("[REDACTED]") == 2
