import os

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from tests.conftest import requires_git, run_git, write_file


@pytest.fixture
def make_client(monkeypatch):
    def _make(root=None, **limits):
        if root is None:
            monkeypatch.delenv("AGENT_ALLOWED_REPO_ROOT", raising=False)
        else:
            monkeypatch.setenv("AGENT_ALLOWED_REPO_ROOT", str(root))
        for key, value in limits.items():
            monkeypatch.setenv(f"AGENT_{key.upper()}", str(value))
        get_settings.cache_clear()
        return TestClient(create_app())

    return _make


def test_endpoints_are_unavailable_without_allowed_root(make_client, sample_repo):
    client = make_client(root=None)

    response = client.post("/repo/summary", json={"path": "sample"})

    assert response.status_code == 503
    assert "AGENT_ALLOWED_REPO_ROOT" in response.json()["detail"]


def test_missing_allowed_root_is_unavailable(make_client, tmp_path):
    client = make_client(root=tmp_path / "does-not-exist")

    response = client.post("/repo/summary", json={"path": "."})

    assert response.status_code == 503


def test_summary_reports_counts_without_leaking_paths(make_client, workspace, sample_repo):
    client = make_client(root=workspace)

    response = client.post("/repo/summary", json={"path": "sample"})

    assert response.status_code == 200
    body = response.json()
    assert body["repo_name"] == "sample"
    assert body["files_included"] == 6
    assert body["files_skipped_sensitive"] == 2
    assert body["directories_excluded"] == 2
    assert body["is_git_repo"] is False
    assert str(workspace) not in response.text
    assert ".env" not in response.text
    assert "server.key" not in response.text


def test_search_returns_locations_and_redacted_snippets(make_client, workspace, sample_repo):
    client = make_client(root=workspace)

    response = client.post("/repo/search", json={"path": "sample", "query": "TODO"})

    assert response.status_code == 200
    locations = {(m["path"], m["line"]) for m in response.json()["matches"]}
    assert locations == {("README.md", 2), ("app/main.py", 4)}


def test_search_normalizes_extensions(make_client, workspace, sample_repo):
    client = make_client(root=workspace)

    response = client.post(
        "/repo/search", json={"path": "sample", "query": "TODO", "extensions": [" .PY "]}
    )

    assert response.status_code == 200
    assert [m["path"] for m in response.json()["matches"]] == ["app/main.py"]


@pytest.mark.parametrize("extensions", [["py"], ["../etc"], [".py; rm"], ["*.py"]])
def test_search_rejects_malformed_extensions(make_client, workspace, sample_repo, extensions):
    client = make_client(root=workspace)

    response = client.post(
        "/repo/search", json={"path": "sample", "query": "TODO", "extensions": extensions}
    )

    assert response.status_code == 422


def test_search_result_count_is_capped_by_configuration(make_client, workspace, sample_repo):
    client = make_client(root=workspace, max_search_results=1)

    response = client.post("/repo/search", json={"path": "sample", "query": "TODO", "max_results": 500})

    body = response.json()
    assert len(body["matches"]) == 1
    assert body["truncated"] is True


def test_search_rejects_empty_or_oversized_query(make_client, workspace, sample_repo):
    client = make_client(root=workspace)

    assert client.post("/repo/search", json={"path": "sample", "query": ""}).status_code == 422
    assert client.post("/repo/search", json={"path": "sample", "query": "x" * 201}).status_code == 422


def test_shell_like_query_is_treated_as_text(make_client, workspace, sample_repo):
    client = make_client(root=workspace)

    response = client.post("/repo/search", json={"path": "sample", "query": "; rm -rf /"})

    assert response.status_code == 200
    assert response.json()["matches"] == []


@pytest.mark.parametrize(
    "path",
    [
        "..",
        "../outside",
        "sample/../../outside",
        "sample; echo pwned",
        "$(whoami)",
        "--output=pwned.txt",
        "sample\x00",
    ],
)
def test_invalid_or_malicious_paths_are_rejected(make_client, workspace, sample_repo, tmp_path, path):
    client = make_client(root=workspace)

    for endpoint in ["/repo/summary", "/repo/search", "/repo/git/status", "/repo/git/diff"]:
        body = {"path": path, "query": "TODO"} if endpoint == "/repo/search" else {"path": path}
        response = client.post(endpoint, json=body)
        assert response.status_code == 400, (endpoint, response.text)

    assert not (workspace / "pwned.txt").exists()
    assert not (tmp_path / "cwd" / "pwned.txt").exists()


def test_absolute_path_outside_root_is_rejected(make_client, workspace, tmp_path):
    client = make_client(root=workspace)

    response = client.post("/repo/summary", json={"path": str(tmp_path / "outside")})

    assert response.status_code == 400
    assert "outside the allowed repository root" in response.json()["detail"]


def test_symlink_escape_is_rejected(make_client, workspace, tmp_path):
    try:
        os.symlink(tmp_path / "outside", workspace / "escape", target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks requires extra privileges on this system")
    client = make_client(root=workspace)

    response = client.post("/repo/search", json={"path": "escape", "query": "TODO"})

    assert response.status_code == 400


def test_outside_file_contents_are_never_returned(make_client, workspace, sample_repo):
    client = make_client(root=workspace)

    response = client.post("/repo/search", json={"path": "..", "query": "TODO outside"})

    assert response.status_code == 400
    assert "TODO outside the allowed root" not in response.text


def test_git_endpoints_report_non_git_directory(make_client, workspace, sample_repo):
    client = make_client(root=workspace)

    status = client.post("/repo/git/status", json={"path": "sample"}).json()
    diff = client.post("/repo/git/diff", json={"path": "sample"}).json()

    assert status["is_git_repo"] is False
    assert status["git_available"] is True
    assert diff["is_git_repo"] is False


@requires_git
def test_git_endpoints_return_status_and_diff(make_client, workspace, git_repo):
    client = make_client(root=workspace)

    status = client.post("/repo/git/status", json={"path": "sample"}).json()
    unstaged = client.post("/repo/git/diff", json={"path": "sample"}).json()
    staged = client.post("/repo/git/diff", json={"path": "sample", "staged": True}).json()

    assert status["is_git_repo"] is True
    assert status["branch"] == "main"
    assert {"path": "app/main.py", "status": " M"} in status["entries"]
    assert ".env" not in [e["path"] for e in status["entries"]]
    assert "return 'changed'" in unstaged["diff"]
    assert "AGENT_SECRET" not in unstaged["diff"]
    assert "+VALUE = 43" in staged["diff"]
    assert staged["staged"] is True


@requires_git
def test_git_endpoint_output_is_capped(make_client, workspace, git_repo):
    client = make_client(root=workspace, max_git_output_bytes=40)

    diff = client.post("/repo/git/diff", json={"path": "sample"}).json()

    assert diff["truncated"] is True
    assert len(diff["diff"].encode("utf-8")) <= 40


def test_fresh_repo_response_contains_no_absolute_paths(make_client, workspace, sample_repo):
    client = make_client(root=workspace)

    for endpoint in ["/repo/summary", "/repo/git/status"]:
        response = client.post(endpoint, json={"path": "sample"})
        assert str(workspace) not in response.text
