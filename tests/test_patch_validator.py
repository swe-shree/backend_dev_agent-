import pytest

from app.analysis.discovery import discover_files
from app.analysis.models import AnalysisLimits
from app.fixes.patch_validator import PatchLimits, parse_unified_diff, path_problem, validate_patch

LIMITS = PatchLimits(max_files=3, max_hunks_per_file=5, max_changed_lines=120, max_patch_chars=6000)

# Replaces the planted bug on app/billing.py line 9. Context lines 7-8 must match the file exactly.
GOOD_DIFF = (
    "--- a/app/billing.py\n"
    "+++ b/app/billing.py\n"
    "@@ -7,3 +7,3 @@\n"
    "     for item in items:\n"
    '         total += item["price"] * item["qty"]\n'
    "-    return total_with_tax  # BUG: undefined name\n"
    "+    return total\n"
)


@pytest.fixture
def sources(bug_repo):
    return {source.relative_path: source.path for source in discover_files(bug_repo, AnalysisLimits()).files}


def _check(diff, bug_repo, sources, limits=LIMITS, context=frozenset({"app/billing.py"})):
    return validate_patch(
        diff,
        root=bug_repo.resolve(),
        sources=sources,
        context_files=set(context),
        limits=limits,
        max_file_bytes=AnalysisLimits().max_file_bytes,
    )


def test_valid_patch_passes_all_checks(bug_repo, sources):
    result = _check(GOOD_DIFF, bug_repo, sources)

    assert result.ok
    assert [f.path for f in result.files] == ["app/billing.py"]
    assert (result.files[0].additions, result.files[0].deletions) == (1, 1)
    assert result.python_files_parse == ["app/billing.py"]


def test_crlf_line_endings_are_accepted(bug_repo, sources):
    assert _check(GOOD_DIFF.replace("\n", "\r\n"), bug_repo, sources).ok


def test_no_newline_marker_is_accepted(bug_repo, sources):
    assert _check(GOOD_DIFF + "\\ No newline at end of file\n", bug_repo, sources).ok


@pytest.mark.parametrize("diff", ["", "   \n", "\n\n"])
def test_empty_patches_are_rejected(bug_repo, sources, diff):
    result = _check(diff, bug_repo, sources)

    assert not result.ok
    assert result.problems


def test_nul_characters_are_rejected(bug_repo, sources):
    assert not _check(GOOD_DIFF + "\x00", bug_repo, sources).ok


def test_malformed_hunk_header_is_rejected(bug_repo, sources):
    diff = GOOD_DIFF.replace("@@ -7,3 +7,3 @@", "@@ nonsense @@")

    result = _check(diff, bug_repo, sources)

    assert any("Malformed hunk header" in p for p in result.problems)


def test_hunk_that_ends_early_is_rejected(bug_repo, sources):
    diff = GOOD_DIFF.replace("@@ -7,3 +7,3 @@", "@@ -7,4 +7,3 @@")

    result = _check(diff, bug_repo, sources)

    assert any("ends before its line counts" in p for p in result.problems)


def test_extra_lines_beyond_header_counts_are_rejected(bug_repo, sources):
    # The header allows one old line, but the body has two context lines on the old side.
    diff = GOOD_DIFF.replace("@@ -7,3 +7,3 @@", "@@ -7,1 +7,2 @@")

    result = _check(diff, bug_repo, sources)

    assert any("line counts do not match" in p for p in result.problems)


def test_invalid_hunk_line_prefix_is_rejected(bug_repo, sources):
    diff = GOOD_DIFF.replace("+    return total\n", "*    return total\n")

    assert any("Invalid line prefix" in p for p in _check(diff, bug_repo, sources).problems)


def test_content_outside_hunks_is_rejected(bug_repo, sources):
    result = _check(GOOD_DIFF + "stray prose from the model\n", bug_repo, sources)

    assert any("outside a hunk" in p for p in result.problems)


def test_missing_plus_header_is_rejected(bug_repo, sources):
    diff = GOOD_DIFF.replace("+++ b/app/billing.py\n", "")

    assert any("Missing +++" in p for p in _check(diff, bug_repo, sources).problems)


@pytest.mark.parametrize(
    "diff",
    [
        "--- /dev/null\n+++ b/app/new.py\n@@ -0,0 +1 @@\n+x = 1\n",
        "--- a/app/billing.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n",
    ],
)
def test_file_creation_and_deletion_are_rejected(bug_repo, sources, diff):
    result = _check(diff, bug_repo, sources)

    assert any("creation and deletion are not supported" in p for p in result.problems)


def test_binary_change_is_rejected(bug_repo, sources):
    diff = (
        "diff --git a/app/billing.py b/app/billing.py\n"
        "Binary files a/app/billing.py and b/app/billing.py differ\n"
    )

    assert any("Unsupported change" in p for p in _check(diff, bug_repo, sources).problems)


def test_git_binary_patch_is_rejected(bug_repo, sources):
    diff = "diff --git a/app/billing.py b/app/billing.py\nGIT binary patch\nliteral 3\n"

    assert any("Unsupported change" in p for p in _check(diff, bug_repo, sources).problems)


def test_rename_is_rejected(bug_repo, sources):
    diff = (
        "diff --git a/app/billing.py b/app/other.py\n"
        "rename from app/billing.py\n"
        "rename to app/other.py\n"
    )

    assert any("Renames and copies" in p for p in _check(diff, bug_repo, sources).problems)


def test_mismatched_old_and_new_paths_are_rejected(bug_repo, sources):
    diff = GOOD_DIFF.replace("+++ b/app/billing.py", "+++ b/app/other.py")

    assert any("Renames and copies" in p for p in _check(diff, bug_repo, sources).problems)


@pytest.mark.parametrize(
    "target",
    ["../outside.txt", "app/../../outside.txt", "/etc/passwd", "C:/Windows/system.py", "app\\billing.py", "./app/billing.py"],
)
def test_traversal_and_absolute_targets_are_rejected(bug_repo, sources, target):
    diff = f"--- a/{target}\n+++ b/{target}\n@@ -1 +1 @@\n-a\n+b\n"

    assert not _check(diff, bug_repo, sources).ok


@pytest.mark.parametrize("target", [".env", "node_modules/lib/index.js", "certs/server.key", "app/.venv/site.py"])
def test_excluded_and_sensitive_targets_are_rejected(bug_repo, sources, target):
    diff = f"--- a/{target}\n+++ b/{target}\n@@ -1 +1 @@\n-a\n+b\n"

    result = _check(diff, bug_repo, sources, context=frozenset({target}))

    assert not result.ok


def test_env_file_content_never_appears_in_problem_messages(bug_repo, sources):
    diff = "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-compute_total=ENV-LEAK-VALUE\n+x=1\n"

    result = _check(diff, bug_repo, sources, context=frozenset({".env"}))

    assert not result.ok
    assert "ENV-LEAK-VALUE" not in " ".join(result.problems)


def test_path_problem_flags_excluded_and_sensitive_names_directly(bug_repo):
    root = bug_repo.resolve()

    assert path_problem(".env", root, {"app/billing.py"}, {"app/billing.py"}) is not None
    assert path_problem("node_modules/x.js", root, {"node_modules/x.js"}, {"node_modules/x.js"}) is not None
    assert path_problem("app/billing.py", root, {"app/billing.py"}, {"app/billing.py"}) is None


def test_file_outside_the_model_context_is_rejected(bug_repo, sources):
    diff = "--- a/app/dup.py\n+++ b/app/dup.py\n@@ -1 +1 @@\n-x\n+y\n"

    result = _check(diff, bug_repo, sources, context=frozenset({"app/billing.py"}))

    assert any("not in the code context" in p for p in result.problems)


def test_hunk_that_does_not_match_current_file_is_rejected(bug_repo, sources):
    diff = GOOD_DIFF.replace("     for item in items:", "     for thing in things:")

    result = _check(diff, bug_repo, sources)

    assert any("does not match the current file" in p for p in result.problems)


def test_overlapping_hunks_are_rejected(bug_repo, sources):
    diff = (
        "--- a/app/billing.py\n"
        "+++ b/app/billing.py\n"
        "@@ -7,3 +7,3 @@\n"
        "     for item in items:\n"
        '         total += item["price"] * item["qty"]\n'
        "-    return total_with_tax  # BUG: undefined name\n"
        "+    return total\n"
        "@@ -8,2 +8,2 @@\n"
        '         total += item["price"] * item["qty"]\n'
        "-    return total_with_tax  # BUG: undefined name\n"
        "+    return total\n"
    )

    result = _check(diff, bug_repo, sources)

    assert any("ascending order" in p for p in result.problems)


def test_hunk_without_changes_is_rejected(bug_repo, sources):
    diff = (
        "--- a/app/billing.py\n"
        "+++ b/app/billing.py\n"
        "@@ -7,2 +7,2 @@\n"
        "     for item in items:\n"
        '         total += item["price"] * item["qty"]\n'
    )

    assert any("contains no changes" in p for p in _check(diff, bug_repo, sources).problems)


def test_python_that_would_not_parse_is_rejected(bug_repo, sources):
    diff = GOOD_DIFF.replace("+    return total\n", "+    return (total\n")

    result = _check(diff, bug_repo, sources)

    assert any("would not parse as Python" in p for p in result.problems)


def test_changed_line_limit_is_enforced(bug_repo, sources):
    tight = PatchLimits(max_files=3, max_hunks_per_file=5, max_changed_lines=1, max_patch_chars=6000)

    result = _check(GOOD_DIFF, bug_repo, sources, limits=tight)

    assert any("changes 2 lines" in p for p in result.problems)


def test_file_count_limit_is_enforced():
    two_files = (
        "--- a/app/a.py\n+++ b/app/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/app/b.py\n+++ b/app/b.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    limits = PatchLimits(max_files=1, max_hunks_per_file=5, max_changed_lines=120, max_patch_chars=6000)

    files, problems = parse_unified_diff(two_files, limits)

    assert files == []
    assert any("more than 1 file" in p for p in problems)


def test_hunk_count_limit_is_enforced():
    two_hunks = (
        "--- a/app/billing.py\n+++ b/app/billing.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
        "@@ -5 +5 @@\n-c\n+d\n"
    )
    limits = PatchLimits(max_files=3, max_hunks_per_file=1, max_changed_lines=120, max_patch_chars=6000)

    _, problems = parse_unified_diff(two_hunks, limits)

    assert any("more than 1 hunk" in p for p in problems)


def test_patch_length_limit_is_enforced(bug_repo, sources):
    short = PatchLimits(max_files=3, max_hunks_per_file=5, max_changed_lines=120, max_patch_chars=40)

    result = _check(GOOD_DIFF, bug_repo, sources, limits=short)

    assert any("longer than 40 characters" in p for p in result.problems)


@pytest.mark.parametrize(
    "line",
    ["deleted file mode 100644", "new file mode 100644", "old mode 100644", "similarity index 90%"],
)
def test_mode_and_similarity_changes_are_rejected(bug_repo, sources, line):
    diff = f"diff --git a/app/billing.py b/app/billing.py\n{line}\n" + GOOD_DIFF

    assert any("Unsupported change" in p for p in _check(diff, bug_repo, sources).problems)
