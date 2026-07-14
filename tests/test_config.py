import pytest

from config.settings import ConfigurationError, Settings


def test_bot_validation_rejects_missing_secrets(monkeypatch):
    for key in (
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigurationError, match="SLACK_BOT_TOKEN"):
        Settings.from_env().validate_bot()


def test_bot_validation_has_no_backend_switch(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings.from_env()
    settings.validate_bot()
    assert not hasattr(settings, "notebooklm_backend")


def test_bot_validation_rejects_invalid_tool_rounds(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AGENT_MAX_TOOL_ROUNDS", "33")
    with pytest.raises(ConfigurationError, match="AGENT_MAX_TOOL_ROUNDS"):
        Settings.from_env().validate_bot()


def test_auth_validation_requires_https_and_long_internal_token(monkeypatch):
    monkeypatch.setenv("AUTH_BASE_URL", "http://unsafe.example")
    monkeypatch.setenv("AUTH_INTERNAL_TOKEN", "short")
    with pytest.raises(ConfigurationError, match="HTTPS"):
        Settings.from_env().validate_auth()
