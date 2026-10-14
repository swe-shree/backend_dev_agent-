from app.analysis.discovery import discover_files
from app.analysis.models import AnalysisLimits
from app.analysis.search import SNIPPET_MAX_CHARS, search_files
from tests.conftest import write_file

LIMITS = AnalysisLimits()


def _search(repo, query, **overrides):
    options = {
        "extensions": None,
        "case_sensitive": False,
        "max_results": LIMITS.max_search_results,
        "max_file_bytes": LIMITS.max_file_bytes,
    }
    options.update(overrides)
    discovery = discover_files(repo.resolve(), LIMITS)
    return search_files(discovery.files, query, **options)


def _locations(result):
    return {(m.path, m.line) for m in result.matches}


def test_finds_matches_with_correct_line_numbers(sample_repo):
    result = _search(sample_repo, "TODO")

    # Excluded directories, sensitive files and binary files are not searched.
    assert _locations(result) == {("README.md", 2), ("app/main.py", 4)}
    assert result.files_skipped_binary == 1


def test_default_search_is_case_insensitive(sample_repo):
    assert _locations(_search(sample_repo, "todo")) == {("README.md", 2), ("app/main.py", 4)}


def test_case_sensitive_search(sample_repo):
    assert _search(sample_repo, "todo", case_sensitive=True).matches == []


def test_extension_filter(sample_repo):
    result = _search(sample_repo, "TODO", extensions=frozenset({".md"}))

    assert _locations(result) == {("README.md", 2)}


def test_result_limit_sets_truncated(sample_repo):
    result = _search(sample_repo, "TODO", max_results=1)

    assert len(result.matches) == 1
    assert result.truncated is True


def test_no_truncation_when_all_matches_fit(sample_repo):
    result = _search(sample_repo, "TODO", max_results=10)

    assert result.truncated is False


def test_secret_file_contents_are_never_returned(sample_repo):
    for query in ["supersecret", "AGENT_SECRET", "PRIVATE KEY", "node_modules", "venv"]:
        assert _search(sample_repo, query).matches == [], query


def test_secret_values_in_non_sensitive_files_are_redacted(sample_repo):
    result = _search(sample_repo, "hunter2")

    # The line is found, but the credential value itself is masked in the snippet.
    assert _locations(result) == {("config.txt", 1)}
    snippet = result.matches[0].snippet
    assert "hunter2" not in snippet
    assert "[REDACTED]" in snippet


def test_secret_keyword_inside_longer_name_is_masked(sample_repo):
    write_file(sample_repo / "app" / "export.py", 'API_TOKEN = "abc123-secret"\nclient_secret: xyz789\n')

    result = _search(sample_repo, "API_TOKEN")

    snippet = result.matches[0].snippet
    assert "abc123-secret" not in snippet
    assert "[REDACTED]" in snippet


def test_query_is_literal_not_a_regex(sample_repo):
    assert _search(sample_repo, "(.*)").matches == []
    assert _search(sample_repo, "handler()").matches != []


def test_shell_like_query_is_literal(sample_repo):
    assert _search(sample_repo, "$(whoami)").matches == []
    assert _search(sample_repo, "; rm -rf /").matches == []


def test_snippets_are_bounded(sample_repo):
    long_line = "needle " + "x" * 1000 + "\n"
    (sample_repo / "long.txt").write_text(long_line, encoding="utf-8")

    result = _search(sample_repo, "needle")

    assert len(result.matches) == 1
    assert len(result.matches[0].snippet) <= SNIPPET_MAX_CHARS
