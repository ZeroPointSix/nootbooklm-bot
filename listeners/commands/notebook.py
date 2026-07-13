from __future__ import annotations

from typing import Any
from urllib.parse import quote

from auth import ProfileManager, SQLiteLoginSessionStore
from config import Settings
from notebooklm_tool import build_notebook_provider

_settings = Settings.from_env()
_sessions = SQLiteLoginSessionStore(
    _settings.auth_session_db_path, _settings.auth_session_ttl_seconds
)
_profiles = ProfileManager(_settings.profile_path)


def notebook_command(ack, command: dict, respond) -> None:
    ack()
    parts = str(command.get("text", "")).strip().split()
    action = parts[0].lower() if parts else "help"
    if action == "help":
        respond(
            "可用命令：/notebook login、/notebook login cancel、"
            "/notebook status、/notebook logout confirm"
        )
        return
    if action == "status":
        respond(_status_text())
        return
    if action == "login" and len(parts) > 1 and parts[1].lower() == "cancel":
        session = _sessions.cancel_active()
        respond("已取消登录会话。" if session else "当前没有登录会话。")
        return
    if action == "login":
        if not _settings.auth_base_url:
            respond("登录服务尚未配置，请联系管理员设置 AUTH_BASE_URL。")
            return
        try:
            session, token = _sessions.create(
                team_id=command["team_id"],
                channel_id=command["channel_id"],
                thread_ts=command.get("thread_ts"),
                user_id=command["user_id"],
            )
        except ValueError as exc:
            respond(str(exc))
            return
        url = f"{_settings.auth_base_url.rstrip('/')}/auth/notebooklm/{quote(token)}"
        respond(
            "请在 10 分钟内打开以下一次性 HTTPS 链接完成登录。"
            "请勿转发该链接：\n"
            f"{url}\n会话：{session.session_id}"
        )
        return
    if action == "logout":
        if len(parts) < 2 or parts[1].lower() != "confirm":
            respond("退出会删除默认账号登录态。请执行 /notebook logout confirm 确认。")
            return
        _sessions.cancel_active()
        _profiles.logout()
        respond("NotebookLM 默认账号登录态已清除。")
        return
    respond("未知子命令。执行 /notebook help 查看帮助。")


def _status_text() -> str:
    active = _sessions.active()
    try:
        health = build_notebook_provider(_settings).health()
    except Exception as exc:
        health = {
            "backend": "native",
            "ready": False,
            "summary": "NotebookLM 健全检查失败",
            "checks": [
                {"name": "provider", "status": "failed", "message": str(exc)},
            ],
        }

    lines = [
        f"NotebookLM 健全状态：{'ready' if health.get('ready') else 'not_ready'}",
        f"后端：{health.get('backend', 'unknown')}",
        f"摘要：{health.get('summary', '无摘要')}",
    ]
    if active:
        lines.append(f"登录会话：{active.status.value}")
    for item in health.get("checks", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- {item.get('name', 'check')}: "
            f"{_label(item.get('status'))} - {item.get('message', '')}"
        )
    return "\n".join(lines)


def _label(status: Any) -> str:
    return {
        "ok": "通过",
        "warning": "警告",
        "failed": "失败",
        "skipped": "跳过",
    }.get(str(status), "未知")


def get_session_store() -> SQLiteLoginSessionStore:
    return _sessions


def get_profile_manager() -> ProfileManager:
    return _profiles
