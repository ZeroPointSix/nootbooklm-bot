from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Protocol

import openai
from openai.types.responses import ResponseInputParam
from slack_sdk.models.messages.chunk import TaskUpdateChunk
from slack_sdk.web.chat_stream import ChatStream

from config import Settings
from notebooklm_mcp import MCPError, ToolDefinition
from notebooklm_tool import (
    NotebookToolError,
    NotebookToolProvider,
    build_notebook_provider,
)

SYSTEM_PROMPT = """你是 Slack 中的 NotebookLM 研究助手。
NotebookLM 的任何操作都必须通过已提供的内置 NotebookLM 工具真实执行，绝不能虚构 Notebook、来源或结果。
不要请求、显示或复述 Cookie、Token、密码、Storage State 或内部错误堆栈。
写操作必须使用用户明确指定或工具确认的 Notebook；不确定时先询问。
工具失败时明确说明失败及安全的恢复建议。回答适合 Slack 阅读并保持简洁。"""


class Streamer(Protocol):
    def append(self, **kwargs: Any) -> Any: ...


SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "cookies",
    "token",
    "access_token",
    "refresh_token",
    "storage_state",
    "password",
}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _safe_tool_output(result: dict[str, Any]) -> str:
    encoded = json.dumps(_redact(result), ensure_ascii=False, default=str)
    return encoded[:100_000]


class AgentRuntime:
    def __init__(
        self,
        notebook: NotebookToolProvider,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tool_rounds: int = 8,
        llm: Any | None = None,
    ):
        self.notebook = notebook
        self.mcp = notebook
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.llm = llm or openai.OpenAI(api_key=api_key)

    def run(self, streamer: Streamer, prompts: ResponseInputParam) -> None:
        tools = self.notebook.list_tools()
        allowed = {tool.name: tool for tool in tools}
        for round_number in range(self.max_tool_rounds + 1):
            tool_calls = self._stream_once(streamer, prompts, tools)
            if not tool_calls:
                return
            if round_number == self.max_tool_rounds:
                raise RuntimeError("工具调用轮数超过安全上限")
            for call in tool_calls:
                self._execute_call(streamer, prompts, allowed, call)

    def _stream_once(
        self,
        streamer: Streamer,
        prompts: ResponseInputParam,
        tools: list[ToolDefinition],
    ) -> list[Any]:
        calls = []
        response = self.llm.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompts,
            tools=[tool.as_openai_tool() for tool in tools],
            stream=True,
        )
        for event in response:
            if event.type == "response.output_text.delta":
                streamer.append(markdown_text=str(event.delta))
            elif (
                event.type == "response.output_item.done"
                and event.item.type == "function_call"
            ):
                calls.append(event.item)
                streamer.append(
                    chunks=[
                        TaskUpdateChunk(
                            id=event.item.call_id,
                            title=f"正在执行 NotebookLM 工具：{event.item.name}",
                            status="in_progress",
                        )
                    ]
                )
        return calls

    def _execute_call(
        self,
        streamer: Streamer,
        prompts: ResponseInputParam,
        allowed: dict[str, ToolDefinition],
        call: Any,
    ) -> None:
        prompts.append(
            {
                "id": call.id,
                "call_id": call.call_id,
                "type": "function_call",
                "name": call.name,
                "arguments": call.arguments,
            }
        )
        try:
            if call.name not in allowed:
                raise NotebookToolError("UNKNOWN_TOOL", "模型请求了未注册的工具")
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError
            result = self.notebook.call_tool(call.name, arguments)
            output = _safe_tool_output(result)
            status = "complete"
            title = f"NotebookLM 工具已完成：{call.name}"
        except (json.JSONDecodeError, ValueError):
            output = json.dumps(
                {"error": "INVALID_ARGUMENTS", "message": "工具参数无效"}
            )
            status = "error"
            title = "NotebookLM 工具参数无效"
        except (MCPError, NotebookToolError) as exc:
            output = json.dumps(
                {"error": exc.code, "message": str(exc)}, ensure_ascii=False
            )
            status = "error"
            title = str(exc)
        prompts.append(
            {"type": "function_call_output", "call_id": call.call_id, "output": output}
        )
        streamer.append(
            chunks=[TaskUpdateChunk(id=call.call_id, title=title, status=status)]
        )


def call_llm(
    streamer: ChatStream,
    prompts: ResponseInputParam,
    *,
    runtime: AgentRuntime | None = None,
) -> None:
    (runtime or get_runtime()).run(streamer, prompts)


@lru_cache(maxsize=1)
def get_runtime() -> AgentRuntime:
    settings = Settings.from_env()
    settings.validate_bot()
    notebook = build_notebook_provider(settings)
    return AgentRuntime(
        notebook,
        api_key=settings.openai_api_key or "",
        model=settings.openai_model,
        max_tool_rounds=settings.max_tool_rounds,
    )
