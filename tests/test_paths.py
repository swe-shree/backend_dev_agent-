import os

import pytest

from app.analysis.paths import PathValidationError, resolve_repo_path


def test_relative_path_inside_root_is_accepted(workspace, sample_repo):
    assert resolve_repo_path("sample", workspace) == sample_repo.resolve()


def test_absolute_path_inside_root_is_accepted(workspace, sample_repo):
    assert resolve_repo_path(str(sample_repo), workspace) == sample_repo.resolve()


def test_root_itself_is_accepted(workspace):
    assert resolve_repo_path(".", workspace) == workspace.resolve()


@pytest.mark.parametrize(
    "requested",
    [
        "..",
        "../outside",
        "sample/../../outside",
        "sample/app/../../../outside",
    ],
)
def test_traversal_outside_root_is_rejected(workspace, requested):
    with pytest.raises(PathValidationError, match="outside the allowed repository root"):
        resolve_repo_path(requested, workspace)


def test_absolute_path_outside_root_is_rejected(workspace, tmp_path):
    with pytest.raises(PathValidationError, match="outside the allowed repository root"):
        resolve_repo_path(str(tmp_path / "outside"), workspace)


def test_symlink_pointing_outside_root_is_rejected(workspace, tmp_path):
    link = workspace / "escape"
    try:
        os.symlink(tmp_path / "outside", link, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks requires extra privileges on this system")

    with pytest.raises(PathValidationError, match="outside the allowed repository root"):
        resolve_repo_path("escape", workspace)


def test_missing_path_is_rejected(workspace):
    with pytest.raises(PathValidationError, match="does not exist"):
        resolve_repo_path("no-such-repo", workspace)


def test_file_is_not_a_directory(workspace, sample_repo):
    with pytest.raises(PathValidationError, match="not a directory"):
        resolve_repo_path("sample/README.md", workspace)


@pytest.mark.parametrize("requested", ["", "   ", "sample\x00", "\x00"])
def test_empty_or_nul_paths_are_rejected(workspace, requested):
    with pytest.raises(PathValidationError, match="invalid characters"):
        resolve_repo_path(requested, workspace)


@pytest.mark.parametrize(
    "requested",
    [
        "sample; echo pwned",
        "sample && del important",
        "sample | whoami",
        "$(whoami)",
        "--output=pwned.txt",
    ],
)
def test_shell_metacharacters_are_treated_as_literal_names(workspace, requested):
    with pytest.raises(PathValidationError, match="does not exist"):
        resolve_repo_path(requested, workspace)
    assert not (workspace / "pwned.txt").exists()
