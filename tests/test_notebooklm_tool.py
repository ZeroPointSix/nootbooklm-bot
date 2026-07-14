import json

import pytest

from notebooklm_tool import LocalNotebookToolProvider, NotebookToolError


EXPECTED_TOOL_NAMES = [
    "notebook_list",
    "notebook_create",
    "notebook_describe",
    "notebook_rename",
    "notebook_delete",
    "source_list",
    "source_read",
    "source_rename",
    "source_delete",
    "source_wait",
    "source_add",
    "source_add_and_wait",
    "source_upload_bytes",
    "source_add_drive_file",
    "chat_ask",
    "chat_configure",
    "suggest_prompts",
    "note_save",
    "studio_list",
    "studio_generate",
    "studio_status",
    "studio_get_prompt",
    "studio_download",
    "studio_rename",
    "studio_retry",
    "studio_delete",
    "research_start",
    "research_status",
    "research_import",
    "research_cancel",
    "share_status",
    "share_set_access",
    "share_set_user",
    "share_remove_user",
    "server_info",
]


class FakeBackend:
    def __init__(self, *, probe_result=None, probe_error=None):
        self.calls = []
        self.reconnects = 0
        self.probes = 0
        self.probe_result = probe_result or {
            "ok": True,
            "probe": "notebooks.list",
            "notebook_count": 0,
        }
        self.probe_error = probe_error

    def reconnect(self):
        self.reconnects += 1

    def probe(self):
        self.probes += 1
        if self.probe_error:
            raise self.probe_error
        return self.probe_result

    def invoke(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return {"tool_name": tool_name, "arguments": arguments}


def storage_state(path, *, google=True, notebooklm=True):
    cookies = []
    origins = []
    if google:
        cookies.append({"domain": ".google.com", "name": "SID", "value": "secret"})
    if notebooklm:
        origins.append({"origin": "https://notebooklm.google.com", "localStorage": []})
    path.write_text(
        json.dumps({"cookies": cookies, "origins": origins}), encoding="utf-8"
    )


def test_local_provider_exposes_35_native_tools(tmp_path):
    provider = LocalNotebookToolProvider(str(tmp_path / "storage_state.json"))
    assert [tool.name for tool in provider.list_tools()] == EXPECTED_TOOL_NAMES
    assert len(provider.list_tools()) == 35


def test_local_health_requires_profile(tmp_path):
    provider = LocalNotebookToolProvider(str(tmp_path / "storage_state.json"))
    health = provider.health()
    assert health["backend"] == "native"
    assert health["ready"] is False
    assert health["stage"] == "login_required"


def test_local_health_accepts_google_storage_state(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path)
    backend = FakeBackend(probe_result={"ok": True, "probe": "notebooks.list"})
    health = LocalNotebookToolProvider(str(path), backend=backend).health()
    assert health["ready"] is True
    assert health["authenticated"] is True
    assert health["capabilities"]["tool_count"] == 35
    assert health["capabilities"]["readiness_probe"] == "profile_state+notebooks.list"
    assert health["capabilities"]["external_protocol_required"] is False
    assert health["capabilities"]["bridge"] is False
    assert backend.probes == 1


def test_local_health_warns_when_notebooklm_origin_is_missing(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path, notebooklm=False)
    health = LocalNotebookToolProvider(str(path), backend=FakeBackend()).health()
    assert health["ready"] is True
    assert any(item["status"] == "warning" for item in health["checks"])


def test_local_health_rejects_stale_storage_when_online_probe_fails(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path)
    backend = FakeBackend(probe_error=RuntimeError("expired"))

    health = LocalNotebookToolProvider(str(path), backend=backend).health()

    assert health["ready"] is False
    assert health["authenticated"] is False
    assert health["stage"] == "online_probe_failed"
    assert any(
        item["name"] == "notebooklm_online" and item["status"] == "failed"
        for item in health["checks"]
    )


def test_local_business_tools_require_login(tmp_path):
    provider = LocalNotebookToolProvider(str(tmp_path / "storage_state.json"))
    with pytest.raises(NotebookToolError, match="/notebook login") as caught:
        provider.call_tool("notebook_list", {})
    assert caught.value.code == "NOTEBOOK_LOGIN_REQUIRED"


def test_local_tool_calls_are_dispatched_to_native_backend(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path)
    backend = FakeBackend()
    provider = LocalNotebookToolProvider(str(path), backend=backend)

    result = provider.call_tool("notebook_list", {"limit": 1})

    assert result == {
        "ok": True,
        "tool": "notebook_list",
        "result": {"tool_name": "notebook_list", "arguments": {"limit": 1}},
    }
    assert backend.calls == [("notebook_list", {"limit": 1})]


def test_new_source_tools_are_dispatched_to_native_backend(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path)
    backend = FakeBackend()
    provider = LocalNotebookToolProvider(str(path), backend=backend)

    provider.call_tool(
        "source_add_drive_file",
        {"notebook": "Research", "document_id": "drive-doc", "title": "Doc"},
    )
    provider.call_tool(
        "source_upload_bytes",
        {"notebook": "Research", "bytes_base64": "SGk=", "filename": "hi.txt"},
    )

    assert backend.calls == [
        (
            "source_add_drive_file",
            {"notebook": "Research", "document_id": "drive-doc", "title": "Doc"},
        ),
        (
            "source_upload_bytes",
            {"notebook": "Research", "bytes_base64": "SGk=", "filename": "hi.txt"},
        ),
    ]


def test_server_info_reports_health_without_login(tmp_path):
    provider = LocalNotebookToolProvider(str(tmp_path / "storage_state.json"))

    result = provider.call_tool("server_info", {})

    assert result["ok"] is True
    assert result["tool"] == "server_info"
    assert result["result"]["tool_count"] == 35
    assert result["result"]["auth"]["authenticated"] is False


def test_destructive_tools_require_confirmation_before_backend_call(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path)
    backend = FakeBackend()
    provider = LocalNotebookToolProvider(str(path), backend=backend)

    preview = provider.call_tool("notebook_delete", {"notebook": "Research"})

    assert preview["needs_confirmation"] is True
    assert backend.calls == []

    provider.call_tool("notebook_delete", {"notebook": "Research", "confirm": True})
    assert backend.calls == [
        ("notebook_delete", {"notebook": "Research", "confirm": True})
    ]


def test_unknown_tools_are_rejected(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path)
    provider = LocalNotebookToolProvider(str(path), backend=FakeBackend())
    with pytest.raises(NotebookToolError) as caught:
        provider.call_tool("notebook_health", {"notebook": "Research"})
    assert caught.value.code == "UNKNOWN_TOOL"
