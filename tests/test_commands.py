from types import SimpleNamespace

from listeners.commands import notebook


class Store:
    def __init__(self, active=None):
        self._active = active

    def active(self):
        return self._active


class FakeProvider:
    def __init__(self, health):
        self.health_result = health

    def health(self):
        return self.health_result


def run_status(monkeypatch, *, health=None, health_error=None, active=None):
    replies = []
    monkeypatch.setattr(notebook, "_sessions", Store(active=active))

    if health_error is not None:

        def raise_health(_settings):
            raise health_error

        monkeypatch.setattr(notebook, "build_notebook_provider", raise_health)
    else:

        def provider_factory(_settings):
            return FakeProvider(health or {"ready": False, "backend": "native"})

        monkeypatch.setattr(notebook, "build_notebook_provider", provider_factory)

    notebook.notebook_command(
        ack=lambda: None,
        command={"text": "status"},
        respond=replies.append,
    )
    return replies[0]


def test_status_reports_ready_native_health(monkeypatch):
    message = run_status(
        monkeypatch,
        health={
            "ready": True,
            "backend": "native",
            "summary": "NotebookLM 已就绪",
            "checks": [{"name": "storage_state", "status": "ok", "message": "ok"}],
        },
    )
    assert "ready" in message
    assert "native" in message
    assert "NotebookLM 已就绪" in message


def test_status_reports_not_ready_native_health(monkeypatch):
    message = run_status(
        monkeypatch,
        health={
            "ready": False,
            "backend": "native",
            "summary": "需要先执行 /notebook login",
            "checks": [
                {
                    "name": "profile_file",
                    "status": "failed",
                    "message": "未找到默认账号 storage_state",
                }
            ],
        },
    )
    assert "not_ready" in message
    assert "/notebook login" in message


def test_status_reports_provider_failure(monkeypatch):
    message = run_status(monkeypatch, health_error=RuntimeError("provider unavailable"))
    assert "not_ready" in message
    assert "provider unavailable" in message


def test_active_login_session_is_reported(monkeypatch):
    message = run_status(
        monkeypatch,
        active=SimpleNamespace(status=SimpleNamespace(value="browser_started")),
        health={"ready": False, "backend": "native", "summary": "login in progress"},
    )
    assert "browser_started" in message
