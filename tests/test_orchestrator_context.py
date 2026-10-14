from app.analysis.discovery import discover_files
from app.analysis.models import AnalysisLimits
from app.orchestrator.context_builder import ContextLimits, build_context, is_test_path
from tests.conftest import write_file

LIMITS = AnalysisLimits()


def _limits(**overrides):
    values = {"max_files": 6, "max_chars": 12_000, "window_lines": 2, "max_matches_per_term": 30}
    values.update(overrides)
    return ContextLimits(**values)


def _build(repo, terms, changed=frozenset(), **overrides):
    discovery = discover_files(repo.resolve(), LIMITS)
    return build_context(discovery, terms, set(changed), _limits(**overrides), LIMITS.max_file_bytes)


def test_test_files_are_recognised():
    assert is_test_path("tests/test_billing.py")
    assert is_test_path("app/billing_test.py")
    assert is_test_path("tests/helpers.py")
    assert not is_test_path("app/billing.py")
    assert not is_test_path("app/latest.py")


def test_only_matching_files_are_included(bug_repo):
    bundle = _build(bug_repo, ["compute_total"])

    assert set(bundle.files) == {"app/billing.py", "app/dup.py", "tests/test_billing.py"}
    assert "notes.md" not in bundle.files
    assert bundle.total_matches >= 3


def test_sensitive_and_excluded_content_never_reaches_context(bug_repo):
    bundle = _build(bug_repo, ["compute_total"])

    everything = "\n".join(block.text for block in bundle.blocks)
    assert "ENV-LEAK-VALUE" not in everything
    assert "function compute_total() {}" not in everything
    assert all(not block.path.startswith("node_modules") for block in bundle.blocks)


def test_secret_values_are_masked_in_excerpts(bug_repo):
    bundle = _build(bug_repo, ["compute_total", "API_TOKEN"])

    everything = "\n".join(block.text for block in bundle.blocks)
    assert "abc123-secret" not in everything
    assert "API_TOKEN" in everything
    assert "[REDACTED]" in everything


def test_excerpts_carry_line_numbers(bug_repo):
    bundle = _build(bug_repo, ["compute_total"], window_lines=0)

    billing = [b for b in bundle.blocks if b.path == "app/billing.py"]
    assert billing
    assert any("    5: def compute_total(items):" in b.text for b in billing)
    assert all(b.start_line <= b.end_line for b in billing)


def test_matches_across_terms_rank_first(bug_repo):
    # Only billing.py contains both terms, so it must come before files matching one term.
    write_file(bug_repo / "app" / "other.py", "def helper():\n    return TAX_RATE\n")
    bundle = _build(bug_repo, ["compute_total", "API_TOKEN"])

    assert bundle.files[0] == "app/billing.py"
    assert bundle.matched_terms["app/billing.py"] == ["API_TOKEN", "compute_total"]


def test_changed_files_rank_ahead_of_equal_matches(bug_repo):
    bundle = _build(bug_repo, ["compute_total"], changed={"app/dup.py"})

    assert bundle.files.index("app/dup.py") < bundle.files.index("app/billing.py")


def test_identical_excerpts_are_deduplicated_across_files(bug_repo):
    # billing.py and dup.py both contain "def compute_total(items):" as a one-line excerpt; it is sent once.
    bundle = _build(bug_repo, ["compute_total"], window_lines=0)

    assert sum("def compute_total(items):" in b.text for b in bundle.blocks) == 1


def test_test_role_is_marked(bug_repo):
    bundle = _build(bug_repo, ["compute_total"])

    assert any(b.is_test for b in bundle.blocks if b.path == "tests/test_billing.py")
    assert not any(b.is_test for b in bundle.blocks if b.path == "app/billing.py")


def test_file_limit_is_enforced_and_reported(bug_repo):
    bundle = _build(bug_repo, ["compute_total"], max_files=1)

    assert len(bundle.files) == 1
    assert bundle.truncated is True


def test_character_limit_is_enforced_and_reported(bug_repo):
    bundle = _build(bug_repo, ["compute_total"], max_chars=200)

    assert bundle.characters <= 200
    assert bundle.truncated is True


def test_no_matches_gives_empty_bundle(bug_repo):
    bundle = _build(bug_repo, ["no_such_symbol_anywhere"])

    assert bundle.blocks == []
    assert bundle.files == []
    assert bundle.truncated is False


def test_no_terms_gives_empty_bundle(bug_repo):
    assert _build(bug_repo, []).blocks == []
