from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import openai
from slack_sdk.models.messages.chunk import TaskUpdateChunk
from slack_sdk.web.chat_stream import ChatStream

from config import Settings
from notebooklm_tool import (
    NotebookToolError,
    NotebookToolProvider,
    ToolDefinition,
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


def _chat_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


@dataclass
class ChatToolCall:
    call_id: str
    name: str
    arguments: str


def _get_nested(value: Any, *attrs: str) -> Any:
    current = value
    for attr in attrs:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(attr)
        else:
            current = getattr(current, attr, None)
    return current


def _error_text(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, openai.AuthenticationError):
        return ("LLM_AUTH_ERROR", "LLM 网关认证失败，请检查 OpenAI 兼容 API Key 配置。")
    if isinstance(exc, openai.PermissionDeniedError):
        return ("LLM_PERMISSION_DENIED", "LLM 网关拒绝访问，请检查账号权限或模型权限。")
    if isinstance(exc, openai.RateLimitError):
        return ("LLM_RATE_LIMIT", "LLM 网关限流，请稍后重试或切换可用额度。")
    if isinstance(exc, openai.APITimeoutError):
        return ("LLM_TIMEOUT", "LLM 网关请求超时，NotebookLM 登录状态未必异常。")
    if isinstance(exc, openai.APIConnectionError):
        return (
            "LLM_CONNECTION_ERROR",
            "无法连接 LLM 网关，请检查网络或 OPENAI_BASE_URL。",
        )
    if isinstance(exc, openai.BadRequestError):
        return (
            "LLM_BAD_REQUEST",
            f"LLM 网关不接受当前 OpenAI 兼容请求格式：{exc.message}",
        )
    if isinstance(exc, openai.APIStatusError):
        body = _redact(getattr(exc, "body", None))
        detail = (
            json.dumps(body, ensure_ascii=False, default=str) if body else exc.message
        )
        return (
            "LLM_STATUS_ERROR",
            f"LLM 网关返回 HTTP {exc.status_code}：{detail[:500]}",
        )
    if isinstance(exc, NotebookToolError):
        return (f"NOTEBOOKLM_{exc.code}", str(exc))
    if isinstance(exc, RuntimeError):
        return ("AGENT_RUNTIME_ERROR", str(exc))
    return ("AGENT_UNKNOWN_ERROR", f"代理运行异常：{type(exc).__name__}")


def format_error_message(exc: Exception) -> str:
    code, detail = _error_text(exc)
    return (
        ":warning: NotebookLM 请求失败\n"
        f"• 错误类型：`{code}`\n"
        f"• 具体原因：{detail}\n"
        "• 建议动作：如果 `/notebook status` 仍为 ready，请优先检查 LLM 网关；"
        "如果状态不是 ready，再重新执行 `/notebook login`。"
    )


def format_tool_failure_message(tool_name: str, code: str, detail: str) -> str:
    actions = {
        "SOURCE_NOT_READY": "请先执行 source_wait，等待来源处理完成后再读。",
        "SOURCE_PROCESSING_FAILED": "请删除该来源后重新添加，确认处理成功后再读取。",
        "SOURCE_READ_UNSUPPORTED": "请先改用摘要读取，或重新添加一个 NotebookLM 支持全文读取的来源。",
        "LOGIN_EXPIRED": "请重新执行 /notebook login 后再试。",
        "TIMEOUT": "请稍后重试；如果持续超时，再检查 NotebookLM 上游状态。",
        "NOTEBOOKLM_UPSTREAM_CHANGED": "请检查 NotebookLM 上游页面或 SDK 兼容性。",
    }
    action = actions.get(code, "请根据错误类型处理后重试。")
    return (
        ":warning: NotebookLM 工具调用失败\n"
        f"• 工具：`{tool_name}`\n"
        f"• 错误类型：`{code}`\n"
        f"• 具体原因：{detail}\n"
        f"• 建议动作：{action}"
    )


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
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.llm = llm or openai.OpenAI(api_key=api_key)

    def run(self, streamer: Streamer, prompts: list[dict[str, Any]]) -> None:
        tools = self.notebook.list_tools()
        allowed = {tool.name: tool for tool in tools}
        for round_number in range(self.max_tool_rounds + 1):
            tool_calls = self._stream_once(streamer, prompts, tools)
            if not tool_calls:
                return
            if round_number == self.max_tool_rounds:
                raise RuntimeError("工具调用轮数超过安全上限")
            for call in tool_calls:
                if not self._execute_call(streamer, prompts, allowed, call):
                    return

    def _stream_once(
        self,
        streamer: Streamer,
        prompts: list[dict[str, Any]],
        tools: list[ToolDefinition],
    ) -> list[ChatToolCall]:
        calls: dict[int, dict[str, str]] = {}
        text_chunks: list[str] = []
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *prompts],
            tools=[_chat_tool(tool) for tool in tools],
            tool_choice="auto",
            stream=True,
        )
        for chunk in response:
            choices = _get_nested(chunk, "choices") or []
            if not choices:
                continue
            delta = _get_nested(choices[0], "delta")
            content = _get_nested(delta, "content")
            if content:
                text_chunks.append(str(content))
            for tool_delta in _get_nested(delta, "tool_calls") or []:
                index = _get_nested(tool_delta, "index")
                if index is None:
                    index = len(calls)
                call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                call_id = _get_nested(tool_delta, "id")
                name = _get_nested(tool_delta, "function", "name")
                arguments = _get_nested(tool_delta, "function", "arguments")
                if call_id:
                    call["id"] = str(call_id)
                if name:
                    call["name"] = str(name)
                if arguments:
                    call["arguments"] += str(arguments)
        tool_calls = [
            ChatToolCall(
                call_id=call["id"],
                name=call["name"],
                arguments=call["arguments"],
            )
            for _, call in sorted(calls.items())
            if call["id"] and call["name"]
        ]
        if tool_calls:
            prompts.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                streamer.append(
                    chunks=[
                        TaskUpdateChunk(
                            id=call.call_id,
                            title=f"正在执行 NotebookLM 工具：{call.name}",
                            status="in_progress",
                        )
                    ]
                )
        else:
            for text in text_chunks:
                streamer.append(markdown_text=text)
        return tool_calls

    def _execute_call(
        self,
        streamer: Streamer,
        prompts: list[dict[str, Any]],
        allowed: dict[str, ToolDefinition],
        call: ChatToolCall,
    ) -> bool:
        failure_message = None
        try:
            if call.name not in allowed:
                raise NotebookToolError("UNKNOWN_TOOL", "模型请求了未注册的工具")
            arguments = json.loads(call.arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError
            result = self.notebook.call_tool(call.name, arguments)
            output = _safe_tool_output(result)
            status = "complete"
            title = f"NotebookLM 工具已完成：{call.name}"
        except (json.JSONDecodeError, ValueError):
            output = json.dumps(
                {"error": "INVALID_ARGUMENTS", "message": "工具参数无效"},
                ensure_ascii=False,
            )
            status = "error"
            title = "NotebookLM 工具参数无效"
            failure_message = format_tool_failure_message(
                call.name, "INVALID_ARGUMENTS", "工具参数无效"
            )
        except NotebookToolError as exc:
            payload = {"error": exc.code, "message": str(exc)}
            details = _redact(getattr(exc, "details", {}) or {})
            if details:
                payload["details"] = details
            output = json.dumps(payload, ensure_ascii=False, default=str)
            status = "error"
            title = str(exc)
            failure_message = format_tool_failure_message(call.name, exc.code, str(exc))
        prompts.append(
            {"role": "tool", "tool_call_id": call.call_id, "content": output}
        )
        streamer.append(
            chunks=[TaskUpdateChunk(id=call.call_id, title=title, status=status)]
        )
        if failure_message:
            streamer.append(markdown_text=failure_message)
            return False
        return True


def call_llm(
    streamer: ChatStream,
    prompts: list[dict[str, Any]],
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
