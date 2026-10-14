"""Structural and context checks for a proposed unified diff.

Nothing is written, applied or executed. Each hunk's old side is compared with the current file text in
memory. Python results are parsed with ast, which builds a syntax tree without running the code.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.analysis.rules import is_excluded_dir, is_sensitive_file
from app.analysis.search import read_text_lines

_DIFF_GIT = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")
MAX_PATH_CHARS = 300
# Lines that describe operations this tool does not propose: creation, deletion, rename, copy, mode and binary changes.
_UNSUPPORTED_PREFIXES = (
    "new file mode",
    "deleted file mode",
    "old mode",
    "new mode",
    "rename from",
    "rename to",
    "copy from",
    "copy to",
    "similarity index",
    "dissimilarity index",
    "Binary files",
    "GIT binary patch",
)


@dataclass(frozen=True)
class PatchLimits:
    max_files: int
    max_hunks_per_file: int
    max_changed_lines: int
    max_patch_chars: int


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    # Each entry keeps its diff prefix: " " context, "-" removed, "+" added.
    lines: tuple[str, ...]

    @property
    def old_block(self) -> list[str]:
        return [line[1:] for line in self.lines if not line.startswith("+")]

    @property
    def new_block(self) -> list[str]:
        return [line[1:] for line in self.lines if not line.startswith("-")]

    @property
    def additions(self) -> int:
        return sum(1 for line in self.lines if line.startswith("+"))

    @property
    def deletions(self) -> int:
        return sum(1 for line in self.lines if line.startswith("-"))


@dataclass
class FilePatch:
    path: str
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def additions(self) -> int:
        return sum(hunk.additions for hunk in self.hunks)

    @property
    def deletions(self) -> int:
        return sum(hunk.deletions for hunk in self.hunks)


@dataclass
class PatchValidation:
    files: list[FilePatch] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    python_files_parse: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.files)


def _show(text: str) -> str:
    """Short, single-line form of model-supplied text for use in messages."""
    return text.replace("\n", " ")[:80]


def _header_path(raw: str, prefix: str) -> str | None:
    """Path from a ---/+++ header. None means /dev/null, which marks file creation or deletion."""
    name = raw.split("\t", 1)[0].strip()
    if name == "/dev/null":
        return None
    return name[len(prefix) :] if name.startswith(prefix) else name


def parse_unified_diff(diff_text: str, limits: PatchLimits) -> tuple[list[FilePatch], list[str]]:
    """Syntax only: headers, hunk headers, line counts and prefixes. Returns (files, problems)."""
    if len(diff_text) > limits.max_patch_chars:
        return [], [f"Patch is longer than {limits.max_patch_chars} characters."]
    if "\x00" in diff_text:
        return [], ["Patch contains NUL characters."]

    lines = diff_text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()

    files: list[FilePatch] = []
    current: FilePatch | None = None
    previous_end = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        where = f"line {i + 1}"

        if line.startswith(_UNSUPPORTED_PREFIXES):
            return [], [f"Unsupported change at {where}. Only edits to existing text files are allowed."]

        if line.startswith("diff --git "):
            match = _DIFF_GIT.match(line)
            if match is None:
                return [], [f"Malformed diff --git header at {where}."]
            if match.group(1) != match.group(2):
                return [], [f"Renames and copies are not supported (at {where})."]
            i += 1
            continue

        if line.startswith("index "):
            i += 1
            continue

        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                return [], [f"Missing +++ header after the --- header at {where}."]
            old_path = _header_path(line[4:], "a/")
            new_path = _header_path(lines[i + 1][4:], "b/")
            if old_path is None or new_path is None:
                return [], [f"File creation and deletion are not supported (at {where})."]
            if old_path != new_path:
                return [], [f"Renames and copies are not supported (at {where})."]
            current = FilePatch(path=new_path)
            files.append(current)
            if len(files) > limits.max_files:
                return [], [f"Patch changes more than {limits.max_files} file(s)."]
            previous_end = 0
            i += 2
            continue

        if line.startswith("@@"):
            if current is None:
                return [], [f"A hunk appears before any file header (at {where})."]
            match = _HUNK_HEADER.match(line)
            if match is None:
                return [], [f"Malformed hunk header at {where}."]
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) is not None else 1
            new_count = int(match.group(4)) if match.group(4) is not None else 1
            if old_count and old_start < 1:
                return [], [f"Malformed hunk header at {where}."]
            if old_start < previous_end:
                return [], [f"Hunks must be in ascending order and must not overlap (at {where})."]
            if len(current.hunks) >= limits.max_hunks_per_file:
                return [], [f"A file has more than {limits.max_hunks_per_file} hunk(s)."]

            i += 1
            body: list[str] = []
            old_left, new_left = old_count, new_count
            while old_left > 0 or new_left > 0:
                if i >= len(lines):
                    found_old, found_new = old_count - old_left, new_count - new_left
                    return [], [
                        f"A hunk in {current.path} ends before its line counts are reached "
                        f"(header expects {old_count} old and {new_count} new lines, found {found_old} and {found_new})."
                    ]
                body_line = lines[i]
                if body_line.startswith("\\"):
                    i += 1
                    continue
                tag = body_line[:1]
                if tag == " ":
                    old_left -= 1
                    new_left -= 1
                elif tag == "-":
                    old_left -= 1
                elif tag == "+":
                    new_left -= 1
                else:
                    return [], [f"Invalid line prefix inside a hunk at line {i + 1}."]
                if old_left < 0 or new_left < 0:
                    return [], [f"Hunk line counts do not match its header in {current.path}."]
                body.append(body_line)
                i += 1
            while i < len(lines) and lines[i].startswith("\\"):
                i += 1

            if not any(entry.startswith(("+", "-")) for entry in body):
                return [], [f"A hunk in {current.path} contains no changes."]
            current.hunks.append(Hunk(old_start=old_start, old_count=old_count, lines=tuple(body)))
            previous_end = old_start + old_count
            continue

        return [], [f"Unexpected line outside a hunk at {where}."]

    if not files:
        return [], ["The patch contains no file changes."]
    if any(not file.hunks for file in files):
        return [], ["A file in the patch has no hunks."]
    return files, []


def path_problem(path: str, root: Path, discovered: set[str], context: set[str]) -> str | None:
    """Returns a problem description, or None when the path may be patched."""
    if not path or len(path) > MAX_PATH_CHARS:
        return "A target path is empty or too long."
    if "\\" in path or path.startswith("/") or _DRIVE_LETTER.match(path) or "\x00" in path:
        return f"Target path must be relative with forward slashes: {_show(path)}"
    segments = path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        return f"Target path contains traversal or empty segments: {_show(path)}"
    if any(is_excluded_dir(segment) for segment in segments[:-1]) or is_sensitive_file(segments[-1]):
        return f"Target path is excluded or sensitive: {_show(path)}"
    if path not in discovered:
        return f"Target file is not in the analyzed set (excluded, sensitive, linked, too large, or missing): {_show(path)}"
    if path not in context:
        return f"Target file was not in the code context sent to the model: {_show(path)}"
    if not (root / path).resolve().is_relative_to(root):
        return f"Target path resolves outside the repository root: {_show(path)}"
    return None


def _apply_hunks(current: list[str], hunks: list[Hunk]) -> list[str] | None:
    """Checks every hunk's old side against the current text. Returns the patched lines, or None on mismatch."""
    output: list[str] = []
    cursor = 0
    for hunk in hunks:
        start = hunk.old_start - 1 if hunk.old_count else hunk.old_start
        block = hunk.old_block
        if start < cursor or start + len(block) > len(current):
            return None
        if current[start : start + len(block)] != block:
            return None
        output.extend(current[cursor:start])
        output.extend(hunk.new_block)
        cursor = start + len(block)
    output.extend(current[cursor:])
    return output


def validate_patch(
    diff_text: str,
    *,
    root: Path,
    sources: dict[str, Path],
    context_files: set[str],
    limits: PatchLimits,
    max_file_bytes: int,
) -> PatchValidation:
    """Checks structure, target paths, size limits and hunk context. sources maps analyzed paths to files on disk."""
    files, problems = parse_unified_diff(diff_text, limits)
    if problems:
        return PatchValidation(problems=problems)

    discovered = set(sources)
    for file in files:
        problem = path_problem(file.path, root, discovered, context_files)
        if problem:
            problems.append(problem)
    if problems:
        return PatchValidation(problems=problems)

    changed = sum(file.additions + file.deletions for file in files)
    if changed > limits.max_changed_lines:
        return PatchValidation(problems=[f"Patch changes {changed} lines; the limit is {limits.max_changed_lines}."])

    result = PatchValidation(files=files)
    for file in files:
        lines = read_text_lines(sources[file.path], max_file_bytes)
        if lines is None:
            result.problems.append(f"{file.path} could not be read as text, so the patch cannot be checked.")
            continue
        patched = _apply_hunks([line.rstrip("\r") for line in lines], file.hunks)
        if patched is None:
            result.problems.append(f"A hunk in {file.path} does not match the current file contents.")
            continue
        if file.path.endswith(".py"):
            try:
                ast.parse("\n".join(patched))
            except (SyntaxError, ValueError):
                result.problems.append(f"The patched {file.path} would not parse as Python.")
                continue
            result.python_files_parse.append(file.path)

    if result.problems:
        result.files = []
        result.python_files_parse = []
    return result
