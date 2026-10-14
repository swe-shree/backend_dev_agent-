"""Read-only Git inspection.

Only `rev-parse`, `status` and `diff` are executed. Arguments are passed as a list with shell=False,
each call has a timeout, and output is capped. GIT_OPTIONAL_LOCKS=0 stops `status` from refreshing
the index on disk, and --no-textconv stops diff from running configured external programs.
"""

import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from app.analysis.models import AnalysisLimits, GitDiffResult, GitStatusEntry, GitStatusResult
from app.analysis.rules import git_exclude_pathspecs

GIT_EXECUTABLE = "git"
_GIT_ENV_OVERRIDES = {"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "GIT_PAGER": "cat"}


class GitUnavailableError(RuntimeError):
    """The git executable could not be started."""


class GitTimeoutError(RuntimeError):
    """A git command exceeded the configured timeout."""


@dataclass(frozen=True)
class _GitOutput:
    returncode: int
    stdout: str
    truncated: bool


def _run_git(repo: Path, args: list[str], limits: AnalysisLimits) -> _GitOutput:
    argv = [GIT_EXECUTABLE, "-C", str(repo), *args]
    env = {**os.environ, **_GIT_ENV_OVERRIDES}
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise GitUnavailableError("git executable not found.") from exc

    expired = threading.Event()

    def _expire() -> None:
        expired.set()
        process.kill()

    timer = threading.Timer(limits.git_timeout_seconds, _expire)
    timer.start()
    try:
        data = process.stdout.read(limits.max_git_output_bytes + 1)
        truncated = len(data) > limits.max_git_output_bytes
        if truncated:
            data = data[: limits.max_git_output_bytes]
            process.kill()
    finally:
        timer.cancel()
        process.stdout.close()
        returncode = process.wait()

    if expired.is_set():
        raise GitTimeoutError(f"Git command timed out after {limits.git_timeout_seconds} seconds.")
    return _GitOutput(returncode=returncode, stdout=data.decode("utf-8", errors="replace"), truncated=truncated)


def _work_tree_state(repo: Path, limits: AnalysisLimits) -> tuple[bool, bool]:
    """Return (git_available, is_work_tree)."""
    try:
        output = _run_git(repo, ["rev-parse", "--is-inside-work-tree"], limits)
    except GitUnavailableError:
        return False, False
    return True, output.returncode == 0 and output.stdout.strip() == "true"


def is_git_repository(repo: Path, limits: AnalysisLimits) -> bool:
    _, is_work_tree = _work_tree_state(repo, limits)
    return is_work_tree


def get_status(repo: Path, limits: AnalysisLimits) -> GitStatusResult:
    available, is_work_tree = _work_tree_state(repo, limits)
    if not available:
        return GitStatusResult(is_git_repo=False, git_available=False, message="git executable not found.")
    if not is_work_tree:
        return GitStatusResult(is_git_repo=False, git_available=True, message="Not a Git working tree.")

    branch_output = _run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"], limits)
    branch = branch_output.stdout.strip() if branch_output.returncode == 0 else None

    output = _run_git(
        repo,
        ["status", "--porcelain=v1", "-z", "--untracked-files=normal", "--", ".", *git_exclude_pathspecs()],
        limits,
    )
    if output.returncode != 0 and not output.truncated:
        return GitStatusResult(is_git_repo=True, git_available=True, branch=branch, message="git status failed.")

    return GitStatusResult(
        is_git_repo=True,
        git_available=True,
        branch=branch,
        entries=_parse_status(output),
        truncated=output.truncated,
    )


def _parse_status(output: _GitOutput) -> list[GitStatusEntry]:
    """Parse `--porcelain=v1 -z` output. Renames and copies are followed by an extra original-path token."""
    tokens = output.stdout.split("\0")
    if output.truncated:
        # The last token may be cut mid-path when output was capped.
        tokens = tokens[:-1]

    entries: list[GitStatusEntry] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if len(token) < 4:
            continue
        code, path = token[:2], token[3:]
        if "R" in code or "C" in code:
            index += 1
        entries.append(GitStatusEntry(status=code, path=path))
    return entries


def get_diff(repo: Path, limits: AnalysisLimits, staged: bool) -> GitDiffResult:
    available, is_work_tree = _work_tree_state(repo, limits)
    if not available:
        return GitDiffResult(is_git_repo=False, git_available=False, staged=staged, message="git executable not found.")
    if not is_work_tree:
        return GitDiffResult(is_git_repo=False, git_available=True, staged=staged, message="Not a Git working tree.")

    args = ["diff", "--no-ext-diff", "--no-textconv", "--no-color"]
    if staged:
        args.append("--cached")
    args += ["--", ".", *git_exclude_pathspecs()]

    output = _run_git(repo, args, limits)
    # A truncated run was killed on purpose, so its exit code says nothing about success.
    if output.returncode != 0 and not output.truncated:
        return GitDiffResult(is_git_repo=True, git_available=True, staged=staged, message="git diff failed.")
    return GitDiffResult(
        is_git_repo=True,
        git_available=True,
        staged=staged,
        diff=output.stdout,
        truncated=output.truncated,
    )
