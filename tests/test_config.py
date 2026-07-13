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


def test_llm_settings_support_legacy_openai_env(monkeypatch):
    for key in (
        "LLM_API_KEY",
        "LLM_API_URL",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    settings = Settings.from_env()
    assert settings.llm_api_key == "legacy-key"
    assert settings.llm_model == "legacy-model"
    assert settings.llm_api_url == "https://legacy.example/v1"


def test_llm_settings_prefer_new_env_over_legacy(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "new-key")
    monkeypatch.setenv("LLM_API_URL", "https://new.example/v1")
    monkeypatch.setenv("LLM_MODEL", "new-model")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    settings = Settings.from_env()
    assert settings.llm_api_key == "new-key"
    assert settings.llm_model == "new-model"
    assert settings.llm_api_url == "https://new.example/v1"


def test_auth_validation_requires_https_and_long_internal_token(monkeypatch):
    monkeypatch.setenv("AUTH_BASE_URL", "http://unsafe.example")
    monkeypatch.setenv("AUTH_INTERNAL_TOKEN", "short")
    with pytest.raises(ConfigurationError, match="HTTPS"):
        Settings.from_env().validate_auth()
