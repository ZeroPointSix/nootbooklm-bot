import json

import pytest

from notebooklm_tool import LocalNotebookToolProvider, NotebookToolError


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


def test_local_provider_exposes_built_in_tools(tmp_path):
    provider = LocalNotebookToolProvider(str(tmp_path / "storage_state.json"))
    assert [tool.name for tool in provider.list_tools()] == [
        "notebook_health",
        "notebook_list",
        "notebook_create",
        "notebook_select",
        "notebook_get",
        "notebook_add_source",
        "notebook_ask",
    ]


def test_local_health_requires_profile(tmp_path):
    provider = LocalNotebookToolProvider(str(tmp_path / "storage_state.json"))
    health = provider.health()
    assert health["backend"] == "local"
    assert health["ready"] is False
    assert health["stage"] == "login_required"


def test_local_health_accepts_google_storage_state(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path)
    health = LocalNotebookToolProvider(str(path)).health()
    assert health["ready"] is True
    assert health["authenticated"] is True
    assert health["capabilities"]["mcp_required"] is False


def test_local_health_warns_when_notebooklm_origin_is_missing(tmp_path):
    path = tmp_path / "storage_state.json"
    storage_state(path, notebooklm=False)
    health = LocalNotebookToolProvider(str(path)).health()
    assert health["ready"] is True
    assert any(item["status"] == "warning" for item in health["checks"])


def test_local_business_tools_require_login(tmp_path):
    provider = LocalNotebookToolProvider(str(tmp_path / "storage_state.json"))
    with pytest.raises(NotebookToolError, match="/notebook login") as caught:
        provider.call_tool("notebook_list", {})
    assert caught.value.code == "NOTEBOOK_LOGIN_REQUIRED"
