import pytest

from config.settings import ConfigurationError, Settings


def test_bot_validation_rejects_missing_secrets(monkeypatch):
    for key in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigurationError, match="SLACK_BOT_TOKEN"):
        Settings.from_env().validate_bot()


def test_auth_validation_requires_https_and_long_internal_token(monkeypatch):
    monkeypatch.setenv("AUTH_BASE_URL", "http://unsafe.example")
    monkeypatch.setenv("AUTH_INTERNAL_TOKEN", "short")
    with pytest.raises(ConfigurationError, match="HTTPS"):
        Settings.from_env().validate_auth()
