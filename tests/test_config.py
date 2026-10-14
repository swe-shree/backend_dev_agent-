import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_ENVIRONMENT", "test")
    monkeypatch.setenv("AGENT_LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
