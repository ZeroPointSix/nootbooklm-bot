from types import SimpleNamespace

from listeners.commands import notebook


class Store:
    def active(self):
        return None


class Profiles:
    def __init__(self, exists):
        self._exists = exists

    def exists(self):
        return self._exists


def run_status(monkeypatch, *, authenticated, profile_exists=False, error=None):
    replies = []
    monkeypatch.setattr(notebook, "_sessions", Store())
    monkeypatch.setattr(notebook, "_profiles", Profiles(profile_exists))
    monkeypatch.setattr(notebook, "get_mcp_auth_status", lambda: (authenticated, error))
    notebook.notebook_command(
        ack=lambda: None,
        command={"text": "status"},
        respond=replies.append,
    )
    return replies[0]


def test_status_uses_mcp_online_authenticated_state(monkeypatch):
    message = run_status(monkeypatch, authenticated=True)
    assert "在线验证通过" in message
    assert "authenticated=true" in message


def test_status_reports_mcp_authentication_failure_even_with_local_profile(monkeypatch):
    message = run_status(monkeypatch, authenticated=False, profile_exists=True)
    assert "在线验证未通过" in message
    assert "/notebook login" in message


def test_status_separates_local_profile_from_unknown_mcp_state(monkeypatch):
    message = run_status(
        monkeypatch, authenticated=None, profile_exists=True, error="NotebookLM MCP 服务不可用"
    )
    assert "已保存本地登录态" in message
    assert "MCP 在线验证失败" in message


def test_active_login_session_short_circuits_online_probe(monkeypatch):
    replies = []
    monkeypatch.setattr(
        notebook,
        "_sessions",
        SimpleNamespace(active=lambda: SimpleNamespace(status=SimpleNamespace(value="browser_started"))),
    )
    monkeypatch.setattr(notebook, "get_mcp_auth_status", lambda: (_ for _ in ()).throw(AssertionError))
    notebook.notebook_command(
        ack=lambda: None,
        command={"text": "status"},
        respond=replies.append,
    )
    assert replies == ["NotebookLM 登录状态：browser_started"]
