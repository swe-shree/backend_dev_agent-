"""Walks a repository, pruning excluded directories and skipping unsafe or oversized files."""

import os
from pathlib import Path

from app.analysis.models import AnalysisLimits, DiscoveryResult, SourceFile
from app.analysis.rules import is_excluded_dir, is_sensitive_file


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _is_virtualenv(directory: Path) -> bool:
    return (directory / "pyvenv.cfg").is_file()


def discover_files(root: Path, limits: AnalysisLimits) -> DiscoveryResult:
    """List readable candidate files under root, in deterministic order.

    root must be a resolved directory. Links are never followed, so nothing outside root is reached.
    """
    result = DiscoveryResult()

    for current_str, dirnames, filenames in os.walk(root):
        current = Path(current_str)

        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            directory = current / name
            if _is_link(directory):
                result.skipped_links += 1
            elif is_excluded_dir(name) or _is_virtualenv(directory):
                result.excluded_dirs += 1
            else:
                kept_dirs.append(name)
        dirnames[:] = kept_dirs

        for name in sorted(filenames):
            path = current / name
            if _is_link(path):
                result.skipped_links += 1
                continue
            if is_sensitive_file(name):
                result.skipped_sensitive += 1
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > limits.max_file_bytes:
                result.skipped_oversize += 1
                continue
            if len(result.files) >= limits.max_files:
                result.truncated = True
                return result
            result.files.append(
                SourceFile(path=path, relative_path=path.relative_to(root).as_posix(), size=size)
            )

    return result
