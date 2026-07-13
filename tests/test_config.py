import pytest

from config.settings import ConfigurationError, Settings


def test_bot_validation_rejects_missing_secrets(monkeypatch):
    for key in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigurationError, match="SLACK_BOT_TOKEN"):
        Settings.from_env().validate_bot()


def test_bot_validation_rejects_invalid_notebook_backend(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NOTEBOOKLM_BACKEND", "unknown")
    with pytest.raises(ConfigurationError, match="NOTEBOOKLM_BACKEND"):
        Settings.from_env().validate_bot()


def test_local_backend_does_not_require_mcp_url(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NOTEBOOKLM_BACKEND", "local")
    monkeypatch.setenv("NOTEBOOKLM_MCP_TRANSPORT", "http")
    monkeypatch.delenv("NOTEBOOKLM_MCP_URL", raising=False)
    Settings.from_env().validate_bot()


def test_mcp_backend_requires_mcp_url(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NOTEBOOKLM_BACKEND", "mcp")
    monkeypatch.setenv("NOTEBOOKLM_MCP_TRANSPORT", "http")
    monkeypatch.delenv("NOTEBOOKLM_MCP_URL", raising=False)
    with pytest.raises(ConfigurationError, match="NOTEBOOKLM_MCP_URL"):
        Settings.from_env().validate_bot()


def test_auth_validation_requires_https_and_long_internal_token(monkeypatch):
    monkeypatch.setenv("AUTH_BASE_URL", "http://unsafe.example")
    monkeypatch.setenv("AUTH_INTERNAL_TOKEN", "short")
    with pytest.raises(ConfigurationError, match="HTTPS"):
        Settings.from_env().validate_auth()
