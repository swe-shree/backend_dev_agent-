import os

import pytest

from app.analysis.discovery import discover_files
from app.analysis.models import AnalysisLimits
from app.analysis.rules import git_exclude_pathspecs, is_sensitive_file
from tests.conftest import write_file

DEFAULT_LIMITS = AnalysisLimits()


def _relative_paths(result):
    return [source.relative_path for source in result.files]


def test_discovers_only_permitted_files(sample_repo):
    result = discover_files(sample_repo.resolve(), DEFAULT_LIMITS)

    assert set(_relative_paths(result)) == {
        "app/main.py",
        "app/util.py",
        "README.md",
        "config.txt",
        "big.txt",
        "blob.bin",
    }


def test_excluded_directories_are_pruned(sample_repo):
    result = discover_files(sample_repo.resolve(), DEFAULT_LIMITS)

    paths = _relative_paths(result)
    assert not any(p.startswith("node_modules/") for p in paths)
    assert not any(p.startswith(".venv/") for p in paths)
    assert result.excluded_dirs == 2


def test_sensitive_files_are_excluded_and_counted(sample_repo):
    result = discover_files(sample_repo.resolve(), DEFAULT_LIMITS)

    paths = _relative_paths(result)
    assert ".env" not in paths
    assert "certs/server.key" not in paths
    assert result.skipped_sensitive == 2


def test_virtualenv_detected_by_pyvenv_cfg_even_with_other_name(sample_repo):
    env_dir = sample_repo / "tooling" / "py311"
    write_file(env_dir / "pyvenv.cfg", "home = x\n")
    write_file(env_dir / "lib.py", "TODO\n")

    result = discover_files(sample_repo.resolve(), DEFAULT_LIMITS)

    assert not any(p.startswith("tooling/") for p in _relative_paths(result))


def test_oversized_files_are_skipped(sample_repo):
    limits = AnalysisLimits(max_file_bytes=1000)

    result = discover_files(sample_repo.resolve(), limits)

    assert "big.txt" not in _relative_paths(result)
    assert result.skipped_oversize == 1


def test_file_count_limit_truncates(sample_repo):
    limits = AnalysisLimits(max_files=2)

    result = discover_files(sample_repo.resolve(), limits)

    assert len(result.files) == 2
    assert result.truncated is True


def test_results_are_deterministic_and_posix_relative(sample_repo):
    first = _relative_paths(discover_files(sample_repo.resolve(), DEFAULT_LIMITS))
    second = _relative_paths(discover_files(sample_repo.resolve(), DEFAULT_LIMITS))

    assert first == second
    assert all("\\" not in p for p in first)


def test_symlinked_files_and_directories_are_not_followed(sample_repo, tmp_path):
    outside = tmp_path / "outside"
    try:
        os.symlink(outside / "private.txt", sample_repo / "linked.txt")
        os.symlink(outside, sample_repo / "linked_dir", target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks requires extra privileges on this system")

    result = discover_files(sample_repo.resolve(), DEFAULT_LIMITS)

    paths = _relative_paths(result)
    assert "linked.txt" not in paths
    assert not any(p.startswith("linked_dir/") for p in paths)
    assert result.skipped_links == 2


def test_sensitive_name_rules():
    for name in [".env", ".env.production", "server.pem", "id_rsa", "credentials.json", "client_secret.json"]:
        assert is_sensitive_file(name), name
    # Deliberately conservative: ".env.*" also matches templates such as ".env.example".
    for name in ["main.py", "environment.py", "README.md"]:
        assert not is_sensitive_file(name), name


def test_git_pathspecs_cover_sensitive_patterns_and_excluded_dirs():
    specs = git_exclude_pathspecs()

    assert ":(exclude,glob)**/.env" in specs
    assert ":(exclude,glob)**/*.key" in specs
    assert ":(exclude,glob)**/node_modules/**" in specs
