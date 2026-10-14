"""Shared access rules: the configured allowed root and the analysis limits derived from settings."""

from pathlib import Path

from app.analysis.models import AnalysisLimits
from app.analysis.paths import resolve_repo_path
from app.config import Settings


class RepoRootUnavailableError(RuntimeError):
    """AGENT_ALLOWED_REPO_ROOT is unset or does not point at an existing directory."""


def allowed_root(settings: Settings) -> Path:
    configured = settings.allowed_repo_root
    if configured is None:
        raise RepoRootUnavailableError("Repository analysis is not configured. Set AGENT_ALLOWED_REPO_ROOT.")
    try:
        root = configured.resolve(strict=True)
    except OSError as exc:
        raise RepoRootUnavailableError("Configured AGENT_ALLOWED_REPO_ROOT does not exist.") from exc
    if not root.is_dir():
        raise RepoRootUnavailableError("Configured AGENT_ALLOWED_REPO_ROOT is not a directory.")
    return root


def resolve_request_path(requested: str, settings: Settings) -> Path:
    """Raises RepoRootUnavailableError or PathValidationError."""
    return resolve_repo_path(requested, allowed_root(settings))


def analysis_limits(settings: Settings) -> AnalysisLimits:
    return AnalysisLimits(
        max_file_bytes=settings.max_file_bytes,
        max_files=settings.max_files,
        max_search_results=settings.max_search_results,
        git_timeout_seconds=settings.git_timeout_seconds,
        max_git_output_bytes=settings.max_git_output_bytes,
    )
