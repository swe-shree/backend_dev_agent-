"""Validates a bug report and classifies it into a task plan. Deterministic. No model calls."""

import re
from dataclasses import dataclass
from typing import Literal

MIN_DESCRIPTION_CHARS = 10
MAX_DESCRIPTION_CHARS = 2000
FALLBACK_TERM_LIMIT = 3
MIN_TERM_CHARS = 3

# A bug report must say something failed or behaved unexpectedly.
_BUG_KEYWORDS = (
    "bug", "error", "exception", "fail", "broken", "crash", "wrong", "unexpected",
    "incorrect", "invalid", "traceback", "regression", "not work", "doesn't", "does not",
    "raise", "returns", "instead of",
)
_STOPWORDS = frozenset(
    {
        "this", "that", "with", "when", "from", "into", "they", "them", "there", "should",
        "would", "could", "about", "after", "before", "because", "still", "some", "only",
        "code", "function", "method", "value", "values", "gets", "giving", "fine", "work",
        "works", "working", "does", "doesn't", "not", "the", "and", "for", "but", "its",
    }
)
_QUOTED = re.compile(r"\"([^\"\n]{3,80})\"|`([^`\n]{3,80})`")
_IDENTIFIER = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)(\()?")
_WORD = re.compile(r"[A-Za-z]{4,}")


class TaskValidationError(ValueError):
    """The bug report is too short, too long, or not a bug report."""


@dataclass(frozen=True)
class TaskPlan:
    kind: Literal["bug_analysis"]
    description: str
    terms: list[str]


def validate_description(raw: str) -> str:
    cleaned = "".join(ch for ch in raw if ch in "\n\t" or ch.isprintable()).strip()
    if len(cleaned) < MIN_DESCRIPTION_CHARS:
        raise TaskValidationError(f"Describe the bug in at least {MIN_DESCRIPTION_CHARS} characters.")
    if len(cleaned) > MAX_DESCRIPTION_CHARS:
        raise TaskValidationError(f"Bug description must be at most {MAX_DESCRIPTION_CHARS} characters.")
    return cleaned


def _is_code_like(token: str, called: bool) -> bool:
    if len(token) < MIN_TERM_CHARS:
        return False
    return (
        called
        or "_" in token
        or "." in token
        or re.search(r"[a-z][A-Z]", token) is not None
        or token.isupper()
        or token.endswith(("Error", "Exception"))
    )


def extract_terms(text: str) -> list[str]:
    """Quoted strings first, then code-like identifiers. Plain English words are a last resort."""
    found: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip().rstrip(".")
        key = term.lower()
        if len(term) >= MIN_TERM_CHARS and key not in seen:
            seen.add(key)
            found.append(term)

    for match in _QUOTED.finditer(text):
        add(match.group(1) or match.group(2))
    for match in _IDENTIFIER.finditer(text):
        token = match.group(1).rstrip(".")
        if _is_code_like(token, called=match.group(2) is not None):
            add(token)

    if not found:
        for word in _WORD.findall(text):
            if word.lower() in _STOPWORDS or any(k.strip() in word.lower() for k in _BUG_KEYWORDS):
                continue
            add(word)
            if len(found) >= FALLBACK_TERM_LIMIT:
                break
    return found


def classify_task(description: str, max_terms: int) -> TaskPlan:
    lowered = description.lower()
    if not any(keyword in lowered for keyword in _BUG_KEYWORDS):
        raise TaskValidationError(
            "This endpoint analyzes bug reports. Describe the failure, error, or unexpected behavior."
        )
    return TaskPlan(kind="bug_analysis", description=description, terms=extract_terms(description)[:max_terms])
