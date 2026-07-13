from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from notebooklm_mcp import MCPClient, MCPError, ToolDefinition

DANGEROUS_MCP_TOOLS = {"cleanup_data", "re_auth", "setup_auth"}
HEALTH_TOOL_NAMES = ("notebook_health", "get_health")


class NotebookToolError(RuntimeError):
    """A safe, normalized NotebookLM tool failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class NotebookToolProvider(Protocol):
    def list_tools(self) -> list[ToolDefinition]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...

    def reconnect(self) -> None: ...


def _object_schema(
    properties: dict[str, Any] | None = None,
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": list(required),
        "additionalProperties": False,
    }


def _text_property(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


def _check(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status, "message": message}


LOCAL_TOOLS = [
    ToolDefinition(
        "notebook_health",
        "检查内置 NotebookLM 工具的 profile、登录态和能力准备度。",
        _object_schema(),
    ),
    ToolDefinition(
        "notebook_list",
        "列出当前账号可用的 NotebookLM 笔记本。",
        _object_schema(),
    ),
    ToolDefinition(
        "notebook_create",
        "创建新的 NotebookLM 笔记本。",
        _object_schema({"title": _text_property("笔记本标题")}, required=("title",)),
    ),
    ToolDefinition(
        "notebook_select",
        "选择后续操作默认使用的 NotebookLM 笔记本。",
        _object_schema(
            {"notebook_id": _text_property("NotebookLM 笔记本 ID")},
            required=("notebook_id",),
        ),
    ),
    ToolDefinition(
        "notebook_get",
        "读取指定 NotebookLM 笔记本的摘要状态。",
        _object_schema({"notebook_id": _text_property("NotebookLM 笔记本 ID")}),
    ),
    ToolDefinition(
        "notebook_add_source",
        "向 NotebookLM 笔记本添加 URL 或文本来源。",
        _object_schema(
            {
                "source": _text_property("URL 或文本内容"),
                "source_type": {
                    "type": "string",
                    "description": "来源类型",
                    "enum": ["url", "text"],
                },
                "notebook_id": _text_property(
                    "NotebookLM 笔记本 ID；省略时使用默认选择"
                ),
            },
            required=("source", "source_type"),
        ),
    ),
    ToolDefinition(
        "notebook_ask",
        "向 NotebookLM 笔记本提问并返回答案。",
        _object_schema(
            {
                "question": _text_property("要询问 NotebookLM 的问题"),
                "notebook_id": _text_property(
                    "NotebookLM 笔记本 ID；省略时使用默认选择"
                ),
            },
            required=("question",),
        ),
    ),
]


class LocalNotebookToolProvider:
    """Built-in NotebookLM tool facade owned by this repository."""

    def __init__(self, profile_path: str):
        self.profile_path = Path(profile_path)

    def reconnect(self) -> None:
        return None

    def list_tools(self) -> list[ToolDefinition]:
        return list(LOCAL_TOOLS)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "notebook_health":
            return self.health()
        allowed = {tool.name for tool in LOCAL_TOOLS}
        if name not in allowed:
            raise NotebookToolError(
                "UNKNOWN_TOOL", "模型请求了未注册的 NotebookLM 工具"
            )
        health = self.health()
        if not health.get("authenticated"):
            raise NotebookToolError(
                "NOTEBOOK_LOGIN_REQUIRED",
                "NotebookLM 登录态不可用，请先执行 /notebook login。",
            )
        raise NotebookToolError(
            "NOTEBOOK_BROWSER_AUTOMATION_PENDING",
            "内置 NotebookLM tool 已接管接口和健全检查；具体浏览器自动化动作仍需接入。"
            "过渡期间可显式设置 NOTEBOOKLM_BACKEND=mcp 使用旧后端。",
        )

    def health(self) -> dict[str, Any]:
        checks: list[dict[str, str]] = []
        if not self.profile_path.is_file():
            checks.append(
                _check("profile_file", "failed", "未找到默认账号 storage_state")
            )
            return self._health(
                False, checks, "login_required", "需要先执行 /notebook login"
            )

        checks.append(_check("profile_file", "ok", "默认账号 storage_state 存在"))
        try:
            state = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checks.append(
                _check(
                    "storage_state", "failed", "storage_state 无法读取或不是合法 JSON"
                )
            )
            return self._health(
                False, checks, "profile_invalid", "登录态文件损坏，请重新登录"
            )

        cookies = state.get("cookies")
        origins = state.get("origins")
        if not isinstance(cookies, list) or not isinstance(origins, list):
            checks.append(
                _check("storage_state", "failed", "storage_state 缺少 cookies/origins")
            )
            return self._health(
                False, checks, "profile_invalid", "登录态格式无效，请重新登录"
            )
        checks.append(_check("storage_state", "ok", "storage_state 结构有效"))

        google_cookies = [
            item
            for item in cookies
            if isinstance(item, dict)
            and isinstance(item.get("domain"), str)
            and item["domain"].endswith(".google.com")
        ]
        if google_cookies:
            checks.append(_check("google_session", "ok", "检测到 Google 会话 cookie"))
        else:
            checks.append(
                _check("google_session", "failed", "未检测到 Google 会话 cookie")
            )

        notebooklm_origins = [
            item
            for item in origins
            if isinstance(item, dict)
            and isinstance(item.get("origin"), str)
            and "notebooklm.google" in item["origin"]
        ]
        if notebooklm_origins:
            checks.append(
                _check("notebooklm_origin", "ok", "检测到 NotebookLM origin state")
            )
        else:
            checks.append(
                _check(
                    "notebooklm_origin",
                    "warning",
                    "未检测到 NotebookLM origin state；首次访问时可能仍需页面确认",
                )
            )

        authenticated = bool(google_cookies)
        stage = "ready" if authenticated else "login_required"
        summary = (
            "内置工具可读取默认账号登录态"
            if authenticated
            else "需要重新登录 NotebookLM"
        )
        return self._health(authenticated, checks, stage, summary)

    def _health(
        self,
        ready: bool,
        checks: list[dict[str, str]],
        stage: str,
        summary: str,
    ) -> dict[str, Any]:
        return {
            "backend": "local",
            "ready": ready,
            "authenticated": ready,
            "stage": stage,
            "summary": summary,
            "profile_path": str(self.profile_path),
            "checks": checks,
            "capabilities": {
                "readiness_probe": "profile_state",
                "browser_automation": "pending",
                "mcp_required": False,
            },
        }


class McpNotebookToolProvider:
    """Compatibility provider for deployments that still need the old MCP backend."""

    def __init__(self, client: MCPClient):
        self.client = client

    def reconnect(self) -> None:
        self.client.reconnect()

    def list_tools(self) -> list[ToolDefinition]:
        return [
            tool
            for tool in self.client.list_tools()
            if tool.name not in DANGEROUS_MCP_TOOLS
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in DANGEROUS_MCP_TOOLS:
            raise NotebookToolError(
                "UNSAFE_TOOL_BLOCKED", "认证清理类 MCP 工具不会暴露给模型"
            )
        try:
            return self.client.call_tool(name, arguments)
        except MCPError as exc:
            raise NotebookToolError(exc.code, str(exc)) from exc

    def health(self) -> dict[str, Any]:
        try:
            tools = self.list_tools()
        except MCPError as exc:
            return {
                "backend": "mcp",
                "ready": False,
                "authenticated": False,
                "stage": "mcp_unavailable",
                "summary": str(exc),
                "checks": [_check("mcp_tools", "failed", str(exc))],
            }

        checks = [_check("mcp_tools", "ok", f"发现 {len(tools)} 个可用业务工具")]
        tool_names = {tool.name for tool in tools}
        for health_tool in HEALTH_TOOL_NAMES:
            if health_tool not in tool_names:
                continue
            try:
                result = self.call_tool(health_tool, {})
            except NotebookToolError as exc:
                checks.append(_check("notebooklm_health", "failed", str(exc)))
                return self._mcp_health(False, False, "health_failed", str(exc), checks)
            payload = _extract_health_payload(result)
            authenticated = bool(
                payload.get("authenticated", payload.get("ready", False))
            )
            ready = bool(payload.get("ready", authenticated))
            checks.append(
                _check(
                    "notebooklm_health",
                    "ok" if ready else "failed",
                    "MCP health 探针通过" if ready else "MCP health 未确认登录态",
                )
            )
            return self._mcp_health(
                ready,
                authenticated,
                "ready" if ready else "login_required",
                "MCP 后端已通过 health 探针" if ready else "MCP 后端未通过登录态验证",
                checks,
            )

        checks.append(
            _check(
                "notebooklm_health", "warning", "MCP 未提供 health 工具，仅验证工具发现"
            )
        )
        return self._mcp_health(
            True, True, "tools_discovered", "MCP 工具发现正常", checks
        )

    @staticmethod
    def _mcp_health(
        ready: bool,
        authenticated: bool,
        stage: str,
        summary: str,
        checks: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "backend": "mcp",
            "ready": ready,
            "authenticated": authenticated,
            "stage": stage,
            "summary": summary,
            "checks": checks,
            "capabilities": {
                "readiness_probe": "mcp_health",
                "browser_automation": "external_mcp",
                "mcp_required": True,
            },
        }


def _extract_health_payload(result: dict[str, Any]) -> dict[str, Any]:
    if any(key in result for key in ("ready", "authenticated", "status")):
        return result
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            try:
                payload = json.loads(item["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def build_notebook_provider(settings: Any) -> NotebookToolProvider:
    if settings.notebooklm_backend == "mcp":
        return McpNotebookToolProvider(
            MCPClient(
                transport=settings.mcp_transport,
                command=settings.mcp_command,
                url=settings.mcp_url,
                timeout=settings.mcp_timeout_seconds,
            )
        )
    return LocalNotebookToolProvider(settings.profile_path)
