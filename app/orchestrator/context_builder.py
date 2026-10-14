"""Selects a small, bounded set of code excerpts for the model.

Only files discovered by the analysis rules are read. Secret-like values are masked. Excerpts are
line windows around deterministic search matches, merged and deduplicated, under hard limits.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from app.analysis.models import DiscoveryResult
from app.analysis.search import mask_secret_values, read_text_lines, search_files

MAX_LINE_CHARS = 300
MAX_BLOCKS_PER_FILE = 4


@dataclass(frozen=True)
class ContextLimits:
    max_files: int
    max_chars: int
    window_lines: int
    max_matches_per_term: int


@dataclass(frozen=True)
class ContextBlock:
    path: str
    start_line: int
    end_line: int
    text: str
    is_test: bool


@dataclass
class ContextBundle:
    blocks: list[ContextBlock] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    matched_terms: dict[str, list[str]] = field(default_factory=dict)
    total_matches: int = 0
    truncated: bool = False

    @property
    def characters(self) -> int:
        return sum(len(block.text) for block in self.blocks)


@dataclass
class _Hit:
    terms: set[str] = field(default_factory=set)
    lines: set[int] = field(default_factory=set)


def is_test_path(relative_path: str) -> bool:
    parts = PurePosixPath(relative_path).parts
    name = parts[-1]
    return name.startswith("test_") or name.endswith("_test.py") or any(p in ("tests", "test") for p in parts[:-1])


def _windows(lines: Iterable[int], radius: int, total: int) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent line ranges around each match line (1-based, inclusive)."""
    merged: list[tuple[int, int]] = []
    for line in sorted(lines):
        start, end = max(1, line - radius), min(total, line + radius)
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _render(lines: list[str], start: int, end: int) -> str:
    rendered = []
    for number in range(start, end + 1):
        line = mask_secret_values(lines[number - 1].rstrip("\r"))[:MAX_LINE_CHARS]
        rendered.append(f"{number:>5}: {line}")
    return "\n".join(rendered)


class _BlockBudget:
    """Character budget and duplicate check shared by every excerpt block in one bundle."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.used = 0
        self._seen: set[str] = set()

    def is_duplicate(self, content: str) -> bool:
        digest = hashlib.sha256(" ".join(content.split()).encode("utf-8")).hexdigest()
        if digest in self._seen:
            return True
        self._seen.add(digest)
        return False


def _add_blocks(
    bundle: ContextBundle,
    path: str,
    lines: list[str],
    ranges: list[tuple[int, int]],
    budget: _BlockBudget,
) -> tuple[int, bool]:
    """Append rendered blocks for one file. Returns (blocks added, budget exhausted)."""
    added = 0
    for start, end in ranges:
        content = "\n".join(lines[start - 1 : end])
        if budget.is_duplicate(content):
            continue
        text = _render(lines, start, end)
        if budget.used + len(text) > budget.max_chars:
            bundle.truncated = True
            return added, True
        budget.used += len(text)
        bundle.blocks.append(ContextBlock(path=path, start_line=start, end_line=end, text=text, is_test=is_test_path(path)))
        added += 1
    return added, False


def build_context(
    discovery: DiscoveryResult,
    terms: list[str],
    changed_paths: set[str],
    limits: ContextLimits,
    max_file_bytes: int,
) -> ContextBundle:
    bundle = ContextBundle()
    hits: dict[str, _Hit] = {}

    # Deterministic search first. Each term is searched once across the discovered files.
    for term in terms:
        result = search_files(
            discovery.files,
            term,
            extensions=None,
            case_sensitive=False,
            max_results=limits.max_matches_per_term,
            max_file_bytes=max_file_bytes,
        )
        bundle.total_matches += len(result.matches)
        for match in result.matches:
            hit = hits.setdefault(match.path, _Hit())
            hit.terms.add(term)
            hit.lines.add(match.line)
    if not hits:
        return bundle

    # Files matching more terms come first, then changed files, then path order for stability.
    ranked = sorted(hits, key=lambda p: (-len(hits[p].terms), p not in changed_paths, p))
    sources = {source.relative_path: source for source in discovery.files}
    budget = _BlockBudget(limits.max_chars)

    for path in ranked:
        if len(bundle.files) >= limits.max_files:
            bundle.truncated = True
            break
        lines = read_text_lines(sources[path].path, max_file_bytes)
        if not lines:
            continue

        ranges = _windows(hits[path].lines, limits.window_lines, len(lines))[:MAX_BLOCKS_PER_FILE]
        added, stop = _add_blocks(bundle, path, lines, ranges, budget)
        if added:
            bundle.files.append(path)
            bundle.matched_terms[path] = sorted(hits[path].terms)
        if stop:
            break

    return bundle


def build_evidence_context(
    discovery: DiscoveryResult,
    targets: dict[str, set[int]],
    *,
    max_files: int,
    max_chars: int,
    window_lines: int,
    max_file_bytes: int,
) -> ContextBundle:
    """Excerpts around exact line numbers chosen by the caller (for example, located evidence)."""
    bundle = ContextBundle()
    sources = {source.relative_path: source for source in discovery.files}
    budget = _BlockBudget(max_chars)

    for path, numbers in targets.items():
        if len(bundle.files) >= max_files:
            bundle.truncated = True
            break
        source = sources.get(path)
        lines = read_text_lines(source.path, max_file_bytes) if source else None
        if not lines:
            continue

        valid = {n for n in numbers if 1 <= n <= len(lines)}
        ranges = _windows(valid, window_lines, len(lines))[:MAX_BLOCKS_PER_FILE]
        added, stop = _add_blocks(bundle, path, lines, ranges, budget)
        if added:
            bundle.files.append(path)
        if stop:
            break

    return bundle
