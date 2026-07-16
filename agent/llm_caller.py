from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any, Protocol

from slack_sdk.models.messages.chunk import TaskUpdateChunk
from slack_sdk.web.chat_stream import ChatStream

from config import Settings
from notebooklm_tool import (
    NotebookToolError,
    NotebookToolProvider,
    build_notebook_provider,
)

from pi_agent.agent_core import (
    Agent,
    AgentEvent,
    AgentTool,
    AssistantMessage,
    Model,
    TextContent,
    UserMessage,
)
from pi_agent.pi_ai import create_agent_stream_fn, create_default_registry

SYSTEM_PROMPT = """你是 Slack 中的 NotebookLM 研究助手。
NotebookLM 的任何操作都必须通过已提供的内置 NotebookLM 工具真实执行，绝不能虚构 Notebook、来源或结果。
不要请求、显示或复述 Cookie、Token、密码、Storage State 或内部错误堆栈。
写操作必须使用用户明确指定或工具确认的 Notebook；不确定时先询问。
工具失败时明确说明失败及安全的恢复建议。回答适合 Slack 阅读并保持简洁。"""


# NOTE: pi-agent 的 Streamer 协议要求 append 方法支持 markdown_text 和 chunks
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


TOOL_TASK_COPY = {
    "notebook_list": {
        "in_progress": "正在获取 Notebook 列表...",
        "complete": "已获取 Notebook 列表",
        "error": "获取 Notebook 列表失败",
    },
    "notebook_create": {
        "in_progress": "正在创建 Notebook...",
        "complete": "已创建 Notebook",
        "error": "创建 Notebook 失败",
    },
    "notebook_describe": {
        "in_progress": "正在查看 Notebook...",
        "complete": "已查看 Notebook",
        "error": "查看 Notebook 失败",
    },
    "source_list": {
        "in_progress": "正在获取来源列表...",
        "complete": "已获取来源列表",
        "error": "获取来源列表失败",
    },
    "source_read": {
        "in_progress": "正在读取来源...",
        "complete": "已读取来源",
        "error": "读取来源失败",
    },
    "source_wait": {
        "in_progress": "正在等待来源处理完成...",
        "complete": "来源已处理完成",
        "error": "等待来源处理失败",
    },
    "chat_ask": {
        "in_progress": "正在向 NotebookLM 提问...",
        "complete": "已收到 NotebookLM 回答",
        "error": "NotebookLM 问答失败",
    },
    "server_info": {
        "in_progress": "正在检查 NotebookLM 服务...",
        "complete": "NotebookLM 服务检查完成",
        "error": "NotebookLM 服务检查失败",
    },
}

PREFIX_TOOL_TASK_COPY = [
    (
        "notebook_",
        {
            "in_progress": "正在处理 Notebook...",
            "complete": "Notebook 操作已完成",
            "error": "Notebook 操作失败",
        },
    ),
    (
        "source_",
        {
            "in_progress": "正在处理来源...",
            "complete": "来源操作已完成",
            "error": "来源操作失败",
        },
    ),
    (
        "studio_",
        {
            "in_progress": "正在处理 Studio 产物...",
            "complete": "Studio 产物操作已完成",
            "error": "Studio 产物操作失败",
        },
    ),
    (
        "research_",
        {
            "in_progress": "正在处理研究任务...",
            "complete": "研究任务操作已完成",
            "error": "研究任务操作失败",
        },
    ),
    (
        "share_",
        {
            "in_progress": "正在处理共享设置...",
            "complete": "共享设置已更新",
            "error": "共享设置更新失败",
        },
    ),
]

FALLBACK_TOOL_TASK_COPY = {
    "in_progress": "正在处理 NotebookLM 请求...",
    "complete": "NotebookLM 请求已完成",
    "error": "NotebookLM 请求需要处理",
}


def _tool_task_copy(tool_name: str) -> dict[str, str]:
    if tool_name in TOOL_TASK_COPY:
        return TOOL_TASK_COPY[tool_name]
    for prefix, copy in PREFIX_TOOL_TASK_COPY:
        if tool_name.startswith(prefix):
            return copy
    return FALLBACK_TOOL_TASK_COPY


def _tool_task_title(tool_name: str, status: str, detail: object | None = None) -> str:
    title = _tool_task_copy(tool_name).get(status, FALLBACK_TOOL_TASK_COPY[status])
    if detail is None:
        return title
    return f"{title}: {_compact_detail(detail)}"


def _compact_detail(detail: object, max_length: int = 96) -> str:
    text = str(detail).strip()
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3].rstrip()}..."


def _tool_result_detail(result: dict[str, Any]) -> object | None:
    for key in (
        "message",
        "title",
        "notebook_title",
        "source_title",
        "artifact_title",
        "poll_task_id",
    ):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("notebooks", "sources", "items", "artifacts"):
        value = result.get(key)
        if isinstance(value, list):
            return f"{len(value)} 项"
    return None


def _tool_result_title(tool_name: str, result: dict[str, Any]) -> str:
    if result.get("error") is not None:
        detail = result.get("message") or result["error"]
        return _tool_task_title(tool_name, "error", detail)
    return _tool_task_title(tool_name, "complete", _tool_result_detail(result))


def _error_text(exc: Exception) -> tuple[str, str]:
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
    """
    基于 pi-agent 的 AgentRuntime 实现。
    使用 pi_agent.Agent 替代原有的 OpenAI 直接调用逻辑。
    """

    def __init__(
        self,
        notebook: NotebookToolProvider,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tool_rounds: int = 8,
        llm: Any | None = None,  # 兼容旧接口，忽略此参数
    ):
        self.notebook = notebook
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.api_key = api_key

        # 初始化 pi-agent 的 Provider Registry
        self._registry = create_default_registry()
        self._stream_fn = create_agent_stream_fn(self._registry)

        # 配置 pi-agent 的 Model
        # pi-agent 支持 OpenAI、Anthropic 等多种 provider
        # 这里使用 openai 兼容模式
        self._pi_model = Model(
            id=model,
            provider="openai",
            api="openai",  # 使用 OpenAI Completions 兼容 API
        )

    def run(self, streamer: Streamer, prompts: list[dict[str, Any]]) -> None:
        """
        运行 agent 处理一轮对话。
        将同步的 prompts 转换为 pi-agent 的消息格式并执行。
        """
        # 将 prompts 转换为 pi-agent 格式
        messages = self._convert_prompts_to_messages(prompts)

        # 创建 pi-agent Agent 实例
        agent = Agent(
            stream_fn=self._stream_fn,
            session_id="notebooklm-slack-agent",
        )
        agent.set_model(self._pi_model)
        agent.set_system_prompt(SYSTEM_PROMPT)
        agent.set_tools(self._build_pi_tools())

        # 设置事件监听器用于流式输出
        self._setup_streamer_listener(agent, streamer)

        # 同步运行 async 代码
        try:
            asyncio.run(self._run_agent_async(agent, messages, streamer))
        except RuntimeError as exc:
            if "already running" in str(exc).lower():
                # 如果已经在事件循环中，使用 run_until_complete
                loop = asyncio.get_event_loop()
                loop.run_until_complete(
                    self._run_agent_async(agent, messages, streamer)
                )
            else:
                raise

    def _convert_prompts_to_messages(self, prompts: list[dict[str, Any]]) -> list:
        """将 OpenAI 格式的 prompts 转换为 pi-agent 格式的消息。"""
        messages = []
        for prompt in prompts:
            role = prompt.get("role")
            content = prompt.get("content")
            tool_calls = prompt.get("tool_calls")

            if role == "system":
                # system prompt 单独处理，这里跳过
                continue
            elif role == "user":
                if content:
                    messages.append(UserMessage(content=content))
            elif role == "assistant":
                # assistant 消息包含 tool_calls 或文本内容
                if tool_calls:
                    # pi-agent 通过 tool_calls 在 AssistantMessage 中表示
                    assistant_content = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        assistant_content.append(
                            {
                                "type": "toolCall",
                                "id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "arguments": json.loads(func.get("arguments", "{}")),
                            }
                        )
                    messages.append(
                        AssistantMessage(
                            content=assistant_content,
                            api="openai",
                            provider="openai",
                            model=self.model,
                        )
                    )
                elif content:
                    messages.append(
                        AssistantMessage(
                            content=[TextContent(text=content)],
                            api="openai",
                            provider="openai",
                            model=self.model,
                        )
                    )
            elif role == "tool":
                # tool 结果消息
                tool_call_id = prompt.get("tool_call_id")
                content = prompt.get("content", "")
                messages.append(
                    {
                        "role": "toolResult",
                        "tool_call_id": tool_call_id,
                        "tool_name": "",  # 从 tool_call_id 推断或从上下文获取
                        "content": [TextContent(text=content)],
                        "is_error": "error" in json.loads(content)
                        if content.startswith("{")
                        else False,
                    }
                )
        return messages

    def _build_pi_tools(self) -> list[AgentTool]:
        """将 NotebookLM 工具转换为 pi-agent AgentTool 格式。"""
        from pi_agent.agent_core import AgentToolResult, TextContent

        tools = self.notebook.list_tools()
        pi_tools = []

        for tool in tools:
            tool_name = tool.name

            async def execute(
                tool_call_id: str,
                params: dict[str, Any],
                abort_event: asyncio.Event | None = None,
                on_update=None,
                _tool_name: str = tool_name,
            ) -> AgentToolResult:
                # 实际调用 NotebookLM 工具
                # 在线程池中运行同步的 call_tool 以避免阻塞
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.notebook.call_tool, _tool_name, params
                )
                # 转换为 pi-agent 的 AgentToolResult
                return AgentToolResult(
                    content=[TextContent(text=_safe_tool_output(result))],
                    details=result,
                )

            pi_tools.append(
                AgentTool(
                    name=tool.name,
                    label=tool.name.replace("_", " ").title(),
                    description=tool.description,
                    execute=execute,
                )
            )

        return pi_tools

    def _setup_streamer_listener(self, agent: Agent, streamer: Streamer) -> None:
        """设置事件监听器将 pi-agent 事件转发到 Slack streamer。"""

        def on_agent_event(event: AgentEvent) -> None:
            event_type = event.get("type")

            if event_type == "toolcall_start":
                # 工具调用开始
                tool_name = event.get("tool_call", {}).get("name", "")
                call_id = event.get("tool_call", {}).get("id", "")
                if tool_name:
                    streamer.append(
                        chunks=[
                            TaskUpdateChunk(
                                id=call_id,
                                title=_tool_task_title(tool_name, "in_progress"),
                                status="in_progress",
                            )
                        ]
                    )

            elif event_type == "toolcall_end":
                # 工具调用结束
                tool_call = event.get("tool_call", {})
                tool_name = tool_call.get("name", "")
                call_id = tool_call.get("id", "")
                if tool_name:
                    # 这里需要从结果中获取状态，暂时标记为 complete
                    streamer.append(
                        chunks=[
                            TaskUpdateChunk(
                                id=call_id,
                                title=_tool_task_title(tool_name, "complete"),
                                status="complete",
                            )
                        ]
                    )

            elif event_type == "text_delta":
                # 文本增量输出
                delta = event.get("delta", "")
                if delta:
                    streamer.append(markdown_text=delta)

            elif event_type == "done":
                # 对话结束
                message = event.get("message")
                if (
                    message
                    and hasattr(message, "error_message")
                    and message.error_message
                ):
                    streamer.append(
                        markdown_text=format_error_message(
                            RuntimeError(message.error_message)
                        )
                    )

        agent.subscribe(on_agent_event)

    async def _run_agent_async(
        self, agent: Agent, messages: list, streamer: Streamer
    ) -> None:
        """异步运行 agent。"""
        if messages:
            await agent.prompt(messages)
        else:
            await agent.continue_()


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


def call_llm(
    streamer: ChatStream,
    prompts: list[dict[str, Any]],
    *,
    runtime: AgentRuntime | None = None,
) -> None:
    (runtime or get_runtime()).run(streamer, prompts)
