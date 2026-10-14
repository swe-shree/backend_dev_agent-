import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import get_settings

GIT_AVAILABLE = shutil.which("git") is not None
requires_git = pytest.mark.skipif(not GIT_AVAILABLE, reason="git executable not available")


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    # Run from an empty directory so a developer's .env in the project root is never loaded.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    for name in ("AGENT_ALLOWED_REPO_ROOT", "AGENT_MAX_FILE_BYTES", "AGENT_MAX_FILES", "AGENT_MAX_SEARCH_RESULTS"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def workspace(tmp_path) -> Path:
    """Allowed root for the analysis. Anything under tmp_path/outside is off-limits."""
    root = tmp_path / "allowed"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    write_file(tmp_path / "outside" / "private.txt", "TODO outside the allowed root\n")
    return root


@pytest.fixture
def sample_repo(workspace) -> Path:
    repo = workspace / "sample"
    write_file(repo / "app" / "main.py", "def handler():\n    return 'hello'\n\nTODO_marker = 1\n")
    write_file(repo / "app" / "util.py", "# helper\nVALUE = 42\n")
    write_file(repo / "README.md", "# Sample\nSearchable TODO here\n")
    write_file(repo / "config.txt", "password = hunter2\n")
    write_file(repo / ".env", "AGENT_SECRET=supersecret\n")
    write_file(repo / "certs" / "server.key", "PRIVATE KEY MATERIAL\n")
    write_file(repo / "node_modules" / "pkg" / "index.js", "TODO in node_modules\n")
    write_file(repo / ".venv" / "pyvenv.cfg", "home = python\n")
    write_file(repo / ".venv" / "lib" / "site.py", "TODO in venv\n")
    write_file(repo / "big.txt", "x" * 2000 + "\n")
    (repo / "blob.bin").write_bytes(b"TODO\x00binary")
    return repo


BILLING_SOURCE = """API_TOKEN = "abc123-secret"  # token used by the export job
TAX_RATE = 0.2


def compute_total(items):
    total = 0
    for item in items:
        total += item["price"] * item["qty"]
    return total_with_tax  # BUG: undefined name
"""

DUPLICATED_SOURCE = "def compute_total(items):\n    return sum(items)\n" + "\n" * 30 + (
    "def compute_total(items):\n    return sum(items)\n"
)


@pytest.fixture
def bug_repo(workspace) -> Path:
    """A small repository with one planted bug (undefined name on billing.py line 9) and some secrets to keep out."""
    repo = workspace / "bugrepo"
    write_file(repo / "app" / "billing.py", BILLING_SOURCE)
    write_file(repo / "app" / "dup.py", DUPLICATED_SOURCE)
    write_file(repo / "tests" / "test_billing.py", "from app.billing import compute_total\n\n\ndef test_total():\n    assert compute_total([]) == 0\n")
    write_file(repo / "notes.md", "Unrelated notes.\n")
    write_file(repo / ".env", "compute_total=ENV-LEAK-VALUE\n")
    write_file(repo / "node_modules" / "lib" / "index.js", "function compute_total() {}\n")
    return repo


@pytest.fixture
def git_repo(sample_repo) -> Path:
    """sample_repo as a Git repository with tracked files, staged, unstaged and untracked changes.

    .env and certs/server.key are committed on purpose, so tests can confirm that Git output
    never exposes them even when they are tracked and modified.
    """
    run_git(sample_repo, "init", "-b", "main")
    run_git(sample_repo, "add", "-A")
    run_git(sample_repo, "commit", "-m", "initial")

    write_file(sample_repo / "app" / "main.py", "def handler():\n    return 'changed'\n\nTODO_marker = 2\n")
    write_file(sample_repo / "app" / "util.py", "# helper\nVALUE = 43\n")
    run_git(sample_repo, "add", "app/util.py")
    write_file(sample_repo / "app" / "new_module.py", "NEW = True\n")
    write_file(sample_repo / ".env", "AGENT_SECRET=rotated-supersecret\n")
    return sample_repo
