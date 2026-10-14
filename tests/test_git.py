import hashlib
import subprocess
import sys
import time

import pytest

from app.analysis import git as git_module
from app.analysis.discovery import discover_files
from app.analysis.git import GitTimeoutError, get_diff, get_status, is_git_repository
from app.analysis.models import AnalysisLimits
from app.analysis.search import search_files
from tests.conftest import requires_git, run_git, write_file

LIMITS = AnalysisLimits()


def _snapshot(root):
    """Hash every file under root, including .git internals, to detect any modification."""
    digests = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def test_non_git_directory_is_reported_as_such(sample_repo):
    assert is_git_repository(sample_repo, LIMITS) is False

    status = get_status(sample_repo, LIMITS)
    assert status.is_git_repo is False
    assert status.git_available is True
    assert status.entries == []

    diff = get_diff(sample_repo, LIMITS, staged=False)
    assert diff.is_git_repo is False
    assert diff.diff == ""


@requires_git
def test_git_repository_is_detected(git_repo):
    assert is_git_repository(git_repo, LIMITS) is True


@requires_git
def test_status_reports_modified_staged_and_untracked_files(git_repo):
    status = get_status(git_repo, LIMITS)

    assert status.is_git_repo is True
    assert status.branch == "main"
    entries = {entry.path: entry.status for entry in status.entries}
    assert entries["app/main.py"] == " M"
    assert entries["app/util.py"] == "M "
    assert entries["app/new_module.py"] == "??"


@requires_git
def test_status_never_lists_sensitive_files(git_repo):
    status = get_status(git_repo, LIMITS)

    paths = {entry.path for entry in status.entries}
    assert ".env" not in paths
    assert "certs/server.key" not in paths


@requires_git
def test_unstaged_diff_contains_change_and_no_secrets(git_repo):
    diff = get_diff(git_repo, LIMITS, staged=False)

    assert diff.is_git_repo is True
    assert "+    return 'changed'" in diff.diff
    assert "AGENT_SECRET" not in diff.diff
    assert "rotated-supersecret" not in diff.diff
    assert "util.py" not in diff.diff  # util.py is staged, so it is not in the unstaged diff


@requires_git
def test_staged_diff_contains_only_staged_changes(git_repo):
    diff = get_diff(git_repo, LIMITS, staged=True)

    assert diff.staged is True
    assert "+VALUE = 43" in diff.diff
    assert "return 'changed'" not in diff.diff
    assert "AGENT_SECRET" not in diff.diff


@requires_git
def test_git_output_is_size_limited(git_repo):
    limits = AnalysisLimits(max_git_output_bytes=40)

    diff = get_diff(git_repo, limits, staged=False)

    assert diff.truncated is True
    assert len(diff.diff.encode("utf-8")) <= 40


@requires_git
def test_git_command_is_killed_after_timeout(git_repo, monkeypatch):
    real_popen = subprocess.Popen

    def hanging_popen(argv, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(git_module.subprocess, "Popen", hanging_popen)
    limits = AnalysisLimits(git_timeout_seconds=0.5)

    started = time.monotonic()
    with pytest.raises(GitTimeoutError):
        get_status(git_repo, limits)
    assert time.monotonic() - started < 15


def test_missing_git_executable_is_reported_not_raised(sample_repo, monkeypatch):
    monkeypatch.setattr(git_module, "GIT_EXECUTABLE", "git-executable-that-does-not-exist-xyz")

    status = get_status(sample_repo, LIMITS)

    assert status.git_available is False
    assert status.is_git_repo is False
    assert is_git_repository(sample_repo, LIMITS) is False


@requires_git
def test_repository_path_with_shell_metacharacters_is_handled_literally(workspace, tmp_path):
    weird = workspace / "evil; echo pwned & rem $(echo pwned)"
    weird.mkdir()
    write_file(weird / "a.txt", "hello\n")
    run_git(weird, "init", "-b", "main")
    run_git(weird, "add", "-A")
    run_git(weird, "commit", "-m", "init")

    status = get_status(weird, LIMITS)

    assert status.is_git_repo is True
    assert status.branch == "main"
    assert not (tmp_path / "cwd" / "pwned").exists()


@requires_git
def test_analysis_does_not_modify_repository_or_git_state(git_repo):
    before = _snapshot(git_repo)

    discovery = discover_files(git_repo.resolve(), LIMITS)
    search_files(
        discovery.files,
        "TODO",
        extensions=None,
        case_sensitive=False,
        max_results=LIMITS.max_search_results,
        max_file_bytes=LIMITS.max_file_bytes,
    )
    is_git_repository(git_repo, LIMITS)
    get_status(git_repo, LIMITS)
    get_diff(git_repo, LIMITS, staged=False)
    get_diff(git_repo, LIMITS, staged=True)

    assert _snapshot(git_repo) == before
