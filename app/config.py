from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables prefixed with AGENT_ (or a .env file)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENT_", extra="ignore")

    app_name: str = "Backend Dev Agent"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Repository analysis. Client-supplied repository paths must resolve inside this root.
    allowed_repo_root: Path | None = None
    max_file_bytes: int = Field(default=1_000_000, gt=0)
    max_files: int = Field(default=5_000, gt=0)
    max_search_results: int = Field(default=200, gt=0)
    git_timeout_seconds: float = Field(default=10.0, gt=0)
    max_git_output_bytes: int = Field(default=200_000, gt=0)

    # Gemini. The key uses the bare GEMINI_API_KEY name, not the AGENT_ prefix. SecretStr keeps it out of reprs and logs.
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    # gemini-2.5-flash is no longer available to new users, so the default is a current Flash-class model.
    gemini_model: str = Field(default="gemini-3.5-flash-lite", min_length=1)
    gemini_timeout_seconds: float = Field(default=30.0, gt=0)
    gemini_max_prompt_chars: int = Field(default=4000, gt=0)
    gemini_max_output_tokens: int = Field(default=512, gt=0)
    # Thinking tokens are billed as output. None uses the model's default. Some models reject 0 (for example gemini-3.5-flash-lite).
    gemini_thinking_budget: int | None = Field(default=None, ge=0)
    # Optional overrides for the built-in price table, in USD per 1M text tokens.
    gemini_input_price_per_million_usd: float | None = Field(default=None, ge=0)
    gemini_output_price_per_million_usd: float | None = Field(default=None, ge=0)

    # Bug analysis. These bound what is sent to Gemini in a single task.
    agent_context_max_files: int = Field(default=6, gt=0)
    agent_context_max_chars: int = Field(default=12_000, gt=0)
    agent_context_window_lines: int = Field(default=12, ge=0)
    agent_context_max_terms: int = Field(default=5, gt=0)
    agent_max_matches_per_term: int = Field(default=30, gt=0)
    agent_analysis_max_output_tokens: int = Field(default=1024, gt=0)

    # Fix proposals. These bound the single patch-generation call and the patch it may return.
    agent_fix_max_output_tokens: int = Field(default=1536, gt=0)
    agent_fix_max_context_chars: int = Field(default=8_000, gt=0)
    agent_fix_max_files: int = Field(default=3, gt=0)
    agent_fix_max_hunks_per_file: int = Field(default=5, gt=0)
    agent_fix_max_changed_lines: int = Field(default=120, gt=0)
    agent_fix_max_patch_chars: int = Field(default=6_000, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
