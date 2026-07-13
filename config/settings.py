from __future__ import annotations

import os
import shlex
from dataclasses import dataclass


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    slack_bot_token: str | None
    slack_app_token: str | None
    openai_api_key: str | None
    openai_model: str
    notebooklm_backend: str
    mcp_transport: str
    mcp_command: tuple[str, ...]
    mcp_url: str | None
    mcp_timeout_seconds: float
    max_tool_rounds: int
    auth_base_url: str | None
    auth_session_ttl_seconds: int
    auth_internal_token: str | None
    auth_browser_viewer_url: str | None
    profile_path: str
    auth_session_db_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
            slack_app_token=os.getenv("SLACK_APP_TOKEN"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            notebooklm_backend=os.getenv("NOTEBOOKLM_BACKEND", "local").strip().lower(),
            mcp_transport=os.getenv("NOTEBOOKLM_MCP_TRANSPORT", "stdio"),
            mcp_command=tuple(
                shlex.split(os.getenv("NOTEBOOKLM_MCP_COMMAND", "notebooklm-mcp"))
            ),
            mcp_url=os.getenv("NOTEBOOKLM_MCP_URL"),
            mcp_timeout_seconds=float(
                os.getenv("NOTEBOOKLM_MCP_TIMEOUT_SECONDS", "30")
            ),
            max_tool_rounds=int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "8")),
            auth_base_url=os.getenv("AUTH_BASE_URL"),
            auth_session_ttl_seconds=int(os.getenv("AUTH_SESSION_TTL_SECONDS", "600")),
            auth_internal_token=os.getenv("AUTH_INTERNAL_TOKEN"),
            auth_browser_viewer_url=os.getenv("AUTH_BROWSER_VIEWER_URL"),
            profile_path=os.getenv(
                "NOTEBOOKLM_PROFILE_PATH",
                ".notebooklm/profiles/default/storage_state.json",
            ),
            auth_session_db_path=os.getenv(
                "AUTH_SESSION_DB_PATH", ".notebooklm/login-sessions.db"
            ),
        )

    def validate_bot(self) -> None:
        missing = [
            name
            for name, value in (
                ("SLACK_BOT_TOKEN", self.slack_bot_token),
                ("SLACK_APP_TOKEN", self.slack_app_token),
                ("OPENAI_API_KEY", self.openai_api_key),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(f"缺少必需配置：{', '.join(missing)}")
        self._validate_notebook_backend()
        if not 1 <= self.max_tool_rounds <= 32:
            raise ConfigurationError("AGENT_MAX_TOOL_ROUNDS 必须在 1 到 32 之间")

    def validate_auth(self) -> None:
        if not self.auth_base_url or not self.auth_base_url.startswith("https://"):
            raise ConfigurationError("AUTH_BASE_URL 必须是 HTTPS URL")
        if not self.auth_internal_token or len(self.auth_internal_token) < 32:
            raise ConfigurationError("AUTH_INTERNAL_TOKEN 至少需要 32 个字符")
        if (
            not self.auth_browser_viewer_url
            or not self.auth_browser_viewer_url.startswith("https://")
        ):
            raise ConfigurationError("AUTH_BROWSER_VIEWER_URL 必须是 HTTPS URL")
        if not 60 <= self.auth_session_ttl_seconds <= 3600:
            raise ConfigurationError("AUTH_SESSION_TTL_SECONDS 必须在 60 到 3600 之间")
        self._validate_notebook_backend()

    def _validate_notebook_backend(self) -> None:
        if self.notebooklm_backend not in {"local", "mcp"}:
            raise ConfigurationError("NOTEBOOKLM_BACKEND 必须是 local 或 mcp")
        if self.notebooklm_backend != "mcp":
            return
        if self.mcp_transport not in {"stdio", "http"}:
            raise ConfigurationError("NOTEBOOKLM_MCP_TRANSPORT 必须是 stdio 或 http")
        if self.mcp_transport == "stdio" and not self.mcp_command:
            raise ConfigurationError("stdio 模式需要 NOTEBOOKLM_MCP_COMMAND")
        if self.mcp_transport == "http" and not self.mcp_url:
            raise ConfigurationError("http 模式需要 NOTEBOOKLM_MCP_URL")
