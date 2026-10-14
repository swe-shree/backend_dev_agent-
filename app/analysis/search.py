"""Deterministic literal text search over permitted files. Queries are never treated as regular expressions."""

import re
from collections.abc import Iterable
from pathlib import Path

from app.analysis.models import SearchMatch, SearchResult, SourceFile

SNIPPET_MAX_CHARS = 200
BINARY_SNIFF_BYTES = 8192

# Masks values assigned to secret-like names so snippets do not leak credentials.
# The keyword may sit inside a longer name (API_TOKEN, db_password, client_secret).
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([A-Za-z0-9_-]*(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|access[_-]?key)[A-Za-z0-9_-]*)(\s*[:=]\s*)(\S+)"
)


def mask_secret_values(text: str) -> str:
    """Replace the value of any secret-like assignment with [REDACTED]."""
    return _SECRET_ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)


def read_text_lines(path: Path, max_bytes: int) -> list[str] | None:
    """Read up to max_bytes as lines. Returns None for unreadable or binary files."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes)
    except OSError:
        return None
    if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
        return None
    return raw.decode("utf-8", errors="replace").split("\n")


def _snippet(line: str) -> str:
    return mask_secret_values(line.strip())[:SNIPPET_MAX_CHARS]


def search_files(
    files: Iterable[SourceFile],
    query: str,
    *,
    extensions: frozenset[str] | None,
    case_sensitive: bool,
    max_results: int,
    max_file_bytes: int,
) -> SearchResult:
    """Find lines containing query. extensions are lowercase, including the dot (e.g. ".py")."""
    needle = query if case_sensitive else query.lower()
    result = SearchResult()

    for source in files:
        if extensions is not None and source.path.suffix.lower() not in extensions:
            continue
        try:
            with source.path.open("rb") as handle:
                raw = handle.read(max_file_bytes)
        except OSError:
            continue
        if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
            result.files_skipped_binary += 1
            continue

        result.files_scanned += 1
        text = raw.decode("utf-8", errors="replace")
        # Split on newlines only so line numbers match what editors show.
        for number, line in enumerate(text.split("\n"), start=1):
            haystack = line if case_sensitive else line.lower()
            if needle not in haystack:
                continue
            if len(result.matches) >= max_results:
                result.truncated = True
                return result
            result.matches.append(
                SearchMatch(path=source.relative_path, line=number, snippet=_snippet(line))
            )

    return result
