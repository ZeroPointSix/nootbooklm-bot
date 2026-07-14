from types import SimpleNamespace

import pytest

from agent.llm_caller import (
    AgentRuntime,
    _safe_tool_output,
    format_error_message,
    format_tool_failure_message,
)
from notebooklm_tool import NotebookToolError, ToolDefinition


class Streamer:
    def __init__(self):
        self.items = []

    def append(self, **kwargs):
        self.items.append(kwargs)


class Provider:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def list_tools(self):
        return [ToolDefinition("notebook_list", "list", {"type": "object"})]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.error:
            raise self.error
        return {"content": [{"type": "text", "text": "notebooks"}]}


def chunk_text(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))]
    )


def chunk_call(name="notebook_list", arguments="{}"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="call1",
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                )
            )
        ]
    )


class Completions:
    def __init__(self, rounds):
        self.rounds = iter(rounds)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.rounds)


def runtime(rounds, max_rounds=8, provider=None):
    provider = provider or Provider()
    completions = Completions(rounds)
    llm = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return (
        AgentRuntime(provider, api_key="test", llm=llm, max_tool_rounds=max_rounds),
        provider,
        completions,
    )


def test_chat_completion_tool_call_is_forwarded_and_result_returned_to_model():
    rt, provider, completions = runtime([[chunk_call()], []])
    prompts = [{"role": "user", "content": "列出 notebooks"}]
    rt.run(Streamer(), prompts)
    assert provider.calls == [("notebook_list", {})]
    assert completions.requests[0]["tools"][0]["function"]["name"] == "notebook_list"
    assert prompts[-1]["role"] == "tool"
    assert prompts[-1]["tool_call_id"] == "call1"


def test_chat_completion_text_delta_is_streamed():
    rt, provider, _ = runtime([[chunk_text("OK")]])
    streamer = Streamer()
    rt.run(streamer, [{"role": "user", "content": "hi"}])
    assert provider.calls == []
    assert streamer.items == [{"markdown_text": "OK"}]


def test_pre_tool_text_is_suppressed_when_tool_call_is_present():
    rt, provider, _ = runtime([[chunk_text("我先看看"), chunk_call()], []])
    streamer = Streamer()
    rt.run(streamer, [{"role": "user", "content": "列出 notebooks"}])
    assert provider.calls == [("notebook_list", {})]
    assert [item for item in streamer.items if "markdown_text" in item] == []


def test_tool_failure_stops_model_continuation_with_single_clear_error():
    provider = Provider(
        error=NotebookToolError("SOURCE_NOT_READY", "来源还没有处于可读取状态")
    )
    rt, _, completions = runtime(
        [[chunk_text("我先看看"), chunk_call()], [chunk_text("不应该继续生成")]],
        provider=provider,
    )
    streamer = Streamer()

    rt.run(streamer, [{"role": "user", "content": "读取来源"}])

    markdowns = [
        item["markdown_text"] for item in streamer.items if "markdown_text" in item
    ]
    assert len(completions.requests) == 1
    assert len(markdowns) == 1
    assert "SOURCE_NOT_READY" in markdowns[0]
    assert "我先看看" not in markdowns[0]
    assert "不应该继续生成" not in markdowns[0]


def test_unknown_tool_is_not_forwarded():
    rt, provider, _ = runtime([[chunk_call("delete_everything")], []])
    prompts = []
    rt.run(Streamer(), prompts)
    assert provider.calls == []
    assert "UNKNOWN_TOOL" in prompts[-1]["content"]


def test_invalid_json_is_not_forwarded():
    rt, provider, _ = runtime([[chunk_call(arguments="{bad")], []])
    prompts = []
    rt.run(Streamer(), prompts)
    assert provider.calls == []
    assert "INVALID_ARGUMENTS" in prompts[-1]["content"]


def test_tool_loop_is_bounded():
    rt, _, _ = runtime([[chunk_call()], [chunk_call()]], max_rounds=1)
    with pytest.raises(RuntimeError, match="安全上限"):
        rt.run(Streamer(), [])


def test_sensitive_tool_result_fields_are_redacted_recursively():
    output = _safe_tool_output(
        {"data": {"cookie": "secret", "safe": "ok"}, "access_token": "secret"}
    )
    assert "secret" not in output
    assert output.count("[REDACTED]") == 2


def test_error_message_shows_distinct_code_and_action():
    message = format_error_message(RuntimeError("工具调用轮数超过安全上限"))
    assert "AGENT_RUNTIME_ERROR" in message
    assert "建议动作" in message


def test_source_not_ready_action_recommends_wait_only():
    message = format_tool_failure_message(
        "source_read", "SOURCE_NOT_READY", "来源仍在处理"
    )
    assert "source_wait" in message
    assert "重新添加" not in message


def test_source_processing_failed_action_recommends_readd_only():
    message = format_tool_failure_message(
        "source_read", "SOURCE_PROCESSING_FAILED", "来源处理失败"
    )
    assert "重新添加" in message
    assert "source_wait" not in message
