"""Validates client-supplied repository paths against the configured allowed root."""

from pathlib import Path


class PathValidationError(ValueError):
    """A client-supplied path is empty, malformed, outside the allowed root, or not a directory."""


def resolve_repo_path(requested: str, allowed_root: Path) -> Path:
    """Resolve a requested path and confirm it is an existing directory inside allowed_root.

    allowed_root must already be resolved (absolute, symlinks followed).
    Relative paths are interpreted relative to allowed_root. Symlinks are followed before
    the containment check, so a link pointing outside the root is rejected.
    """
    if not requested.strip() or "\x00" in requested:
        raise PathValidationError("Path is empty or contains invalid characters.")

    try:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = allowed_root / candidate
        resolved = candidate.resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise PathValidationError("Path could not be resolved.") from exc

    if not resolved.is_relative_to(allowed_root):
        raise PathValidationError("Path is outside the allowed repository root.")
    if not resolved.exists():
        raise PathValidationError("Path does not exist.")
    if not resolved.is_dir():
        raise PathValidationError("Path is not a directory.")
    return resolved
