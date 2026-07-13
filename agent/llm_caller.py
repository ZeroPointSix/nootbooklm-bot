from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Protocol

import openai
from openai.types.responses import ResponseInputParam
from slack_sdk.models.messages.chunk import TaskUpdateChunk
from slack_sdk.web.chat_stream import ChatStream

from notebooklm_mcp import MCPClient, MCPError, ToolDefinition
from notebooklm_mcp.shared import get_shared_mcp_client
from config import Settings

SYSTEM_PROMPT = """你是 Slack 中的 NotebookLM 研究助手。
NotebookLM 的任何操作都必须通过已提供的 MCP 工具真实执行，绝不能虚构 Notebook、来源或结果。
不要请求、显示或复述 Cookie、Token、密码、Storage State 或内部错误堆栈。
Google/NotebookLM 登录只通过 Slack 斜杠命令 /notebook login 完成；如果工具返回 authenticated=false，提醒用户执行 /notebook login，不要尝试清理、重置或自行设置认证。
写操作必须使用用户明确指定或工具确认的 Notebook；不确定时先询问。
工具失败时明确说明失败及安全的恢复建议。回答适合 Slack 阅读并保持简洁。"""

LLM_HIDDEN_TOOLS = {"setup_auth", "re_auth", "cleanup_data"}


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


def _llm_visible_tools(tools: list[ToolDefinition]) -> list[ToolDefinition]:
    return [tool for tool in tools if tool.name not in LLM_HIDDEN_TOOLS]


def _as_chat_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _as_chat_messages(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in prompts:
        role = item.get("role")
        if role in {"user", "assistant", "system"}:
            content = item.get("content", "")
            messages.append({"role": role, "content": str(content)})
    return messages


def _choice_delta(event: Any) -> Any:
    choices = getattr(event, "choices", None) or []
    if not choices:
        return None
    return getattr(choices[0], "delta", None)


def _delta_tool_calls(delta: Any) -> list[Any]:
    if delta is None:
        return []
    return getattr(delta, "tool_calls", None) or []



class AgentRuntime:
    def __init__(
        self,
        mcp: MCPClient,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        max_tool_rounds: int = 8,
        llm: Any | None = None,
    ):
        self.mcp = mcp
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        if llm is not None:
            self.llm = llm
        else:
            client_kwargs: dict[str, str] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.llm = openai.OpenAI(**client_kwargs)

    def run(self, streamer: Streamer, prompts: ResponseInputParam) -> None:
        tools = _llm_visible_tools(self.mcp.list_tools())
        allowed = {tool.name: tool for tool in tools}
        if hasattr(self.llm, "chat"):
            self._run_chat(streamer, prompts, tools, allowed)
            return
        for round_number in range(self.max_tool_rounds + 1):
            tool_calls = self._stream_once(streamer, prompts, tools)
            if not tool_calls:
                return
            if round_number == self.max_tool_rounds:
                raise RuntimeError("工具调用轮数超过安全上限")
            for call in tool_calls:
                self._execute_call(streamer, prompts, allowed, call)

    def _run_chat(
        self,
        streamer: Streamer,
        prompts: list[dict[str, Any]],
        tools: list[ToolDefinition],
        allowed: dict[str, ToolDefinition],
    ) -> None:
        messages = _as_chat_messages(prompts)
        for round_number in range(self.max_tool_rounds + 1):
            tool_calls = self._stream_chat_once(streamer, messages, tools)
            if not tool_calls:
                return
            if round_number == self.max_tool_rounds:
                raise RuntimeError("工具调用轮数超过安全上限")
            assistant_tool_calls = []
            for call in tool_calls:
                assistant_tool_calls.append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                )
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": assistant_tool_calls}
            )
            for call in tool_calls:
                output, status, title = self._call_tool_safely(allowed, call["name"], call["arguments"])
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": output}
                )
                streamer.append(
                    chunks=[TaskUpdateChunk(id=call["id"], title=title, status=status)]
                )

    def _stream_chat_once(
        self,
        streamer: Streamer,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
    ) -> list[dict[str, str]]:
        calls: dict[int, dict[str, str]] = {}
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[_as_chat_tool(tool) for tool in tools],
            stream=True,
        )
        for event in response:
            delta = _choice_delta(event)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                streamer.append(markdown_text=str(content))
            for tool_call in _delta_tool_calls(delta):
                index = int(getattr(tool_call, "index", 0) or 0)
                current = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if getattr(tool_call, "id", None):
                    current["id"] = str(tool_call.id)
                function = getattr(tool_call, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        current["name"] = str(function.name)
                    if getattr(function, "arguments", None):
                        current["arguments"] += str(function.arguments)
        ordered = [call for _, call in sorted(calls.items()) if call["id"] and call["name"]]
        for call in ordered:
            streamer.append(
                chunks=[
                    TaskUpdateChunk(
                        id=call["id"],
                        title=f"正在执行 NotebookLM 工具：{call['name']}",
                        status="in_progress",
                    )
                ]
            )
        return ordered

    def _call_tool_safely(
        self,
        allowed: dict[str, ToolDefinition],
        name: str,
        arguments_json: str,
    ) -> tuple[str, str, str]:
        try:
            if name not in allowed:
                raise MCPError("UNKNOWN_TOOL", "模型请求了未注册的工具")
            arguments = json.loads(arguments_json or "{}")
            if not isinstance(arguments, dict):
                raise ValueError
            result = self.mcp.call_tool(name, arguments)
            return _safe_tool_output(result), "complete", f"NotebookLM 工具已完成：{name}"
        except (json.JSONDecodeError, ValueError):
            return (
                json.dumps({"error": "INVALID_ARGUMENTS", "message": "工具参数无效"}),
                "error",
                "NotebookLM 工具参数无效",
            )
        except MCPError as exc:
            return (
                json.dumps({"error": exc.code, "message": str(exc)}, ensure_ascii=False),
                "error",
                str(exc),
            )

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
        output, status, title = self._call_tool_safely(
            allowed, call.name, call.arguments
        )
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
    return AgentRuntime(
        get_shared_mcp_client(),
        api_key=settings.llm_api_key or "",
        model=settings.llm_model,
        base_url=settings.llm_api_url,
        max_tool_rounds=settings.max_tool_rounds,
    )
