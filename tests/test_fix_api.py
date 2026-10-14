import json

import pytest
from fastapi.testclient import TestClient

from app.api import llm as llm_api
from app.api.llm import get_llm_or_error
from app.config import get_settings
from app.llm.gemini_client import GeminiError
from app.llm.models import LLMErrorKind
from app.main import create_app
from tests.test_llm_client import _api_error, _response
from tests.test_patch_generator import BUG, GOOD_DIFF, _analysis, _fix_json, _llm, _snapshot


@pytest.fixture
def make_client(monkeypatch):
    def _make(root=None, llm=None):
        if root is None:
            monkeypatch.delenv("AGENT_ALLOWED_REPO_ROOT", raising=False)
        else:
            monkeypatch.setenv("AGENT_ALLOWED_REPO_ROOT", str(root))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        get_settings.cache_clear()
        llm_api._client_singleton.cache_clear()
        app = create_app()
        if llm is not None:
            app.dependency_overrides[get_llm_or_error] = lambda: llm
        return TestClient(app)

    return _make


def _body(analysis=None, path="bugrepo", description=BUG):
    report = analysis if analysis is not None else _analysis()
    return {"path": path, "bug_description": description, "analysis": report.model_dump(mode="json")}


def test_fix_proposal_is_unavailable_without_allowed_root(make_client, bug_repo):
    client = make_client(root=None)

    response = client.post("/agent/fix-proposal", json=_body())

    assert response.status_code == 503


def test_successful_proposal_returns_expected_shape(make_client, workspace, bug_repo):
    llm, sdk = _llm(_response(text=_fix_json()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/fix-proposal", json=_body())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "proposed"
    assert set(body) >= {
        "status", "repository", "root_cause_summary", "explanation", "proposed_patch", "affected_files",
        "verification_steps", "warnings", "limitations", "llm",
    }
    assert body["proposed_patch"]["status"] == "proposed_unverified"
    assert body["proposed_patch"]["checks"]["correctness"] == "unverified"
    assert body["proposed_patch"]["unified_diff"] == GOOD_DIFF
    assert body["llm"]["usage"]["input_tokens"] == 1000
    assert body["llm"]["cost"]["status"] == "estimated"
    assert sdk.models.calls == 1


def test_response_does_not_leak_absolute_paths_or_secrets(make_client, workspace, bug_repo):
    llm, _ = _llm(_response(text=_fix_json()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/fix-proposal", json=_body())

    assert str(workspace) not in response.text
    assert "ENV-LEAK-VALUE" not in response.text
    assert "abc123-secret" not in response.text


@pytest.mark.parametrize("path", ["..", "../outside", "bugrepo; echo pwned", "$(whoami)"])
def test_invalid_paths_are_rejected_before_any_call(make_client, workspace, bug_repo, path):
    llm, sdk = _llm(_response(text=_fix_json()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/fix-proposal", json=_body(path=path))

    assert response.status_code == 400
    assert sdk.models.calls == 0


def test_non_bug_description_is_rejected_with_422(make_client, workspace, bug_repo):
    llm, sdk = _llm(_response(text=_fix_json()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/fix-proposal", json=_body(description="Please summarize the README file."))

    assert response.status_code == 422
    assert sdk.models.calls == 0


def test_missing_analysis_is_rejected_by_schema(make_client, workspace, bug_repo):
    client = make_client(root=workspace, llm=_llm(_response(text=_fix_json()))[0])

    response = client.post("/agent/fix-proposal", json={"path": "bugrepo", "bug_description": BUG})

    assert response.status_code == 422


def test_too_many_hypotheses_are_rejected_by_schema(make_client, workspace, bug_repo):
    client = make_client(root=workspace, llm=_llm(_response(text=_fix_json()))[0])
    hypothesis = _analysis().hypotheses[0].model_dump(mode="json")
    body = _body()
    body["analysis"]["hypotheses"] = [hypothesis] * 6

    response = client.post("/agent/fix-proposal", json=body)

    assert response.status_code == 422


def test_overlong_quote_is_rejected_by_schema(make_client, workspace, bug_repo):
    client = make_client(root=workspace, llm=_llm(_response(text=_fix_json()))[0])
    body = _body()
    body["analysis"]["hypotheses"][0]["evidence"][0]["quote"] = "x" * 301

    response = client.post("/agent/fix-proposal", json=body)

    assert response.status_code == 422


def test_missing_model_configuration_returns_status_in_body(make_client, workspace, bug_repo):
    client = make_client(root=workspace, llm=GeminiError(LLMErrorKind.NOT_CONFIGURED, "GEMINI_API_KEY is not set."))

    response = client.post("/agent/fix-proposal", json=_body())

    assert response.status_code == 200
    assert response.json()["status"] == "llm_unavailable"
    assert response.json()["llm"]["called"] is False


def test_provider_error_is_reported_in_body(make_client, workspace, bug_repo):
    llm, _ = _llm(_api_error(429, "RESOURCE_EXHAUSTED", "Too many requests"))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/fix-proposal", json=_body())

    assert response.status_code == 200
    assert response.json()["status"] == "llm_error"
    assert response.json()["llm"]["error"] == "rate_limited"


def test_rejected_patch_is_reported_without_returning_the_diff(make_client, workspace, bug_repo):
    bad = json.dumps({"root_cause_summary": "x", "unified_diff": "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-a\n+b\n"})
    llm, _ = _llm(_response(text=bad))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/fix-proposal", json=_body())

    assert response.json()["status"] == "patch_rejected"
    assert response.json()["proposed_patch"] is None
    assert "+b" not in response.text


def test_endpoint_does_not_modify_repository(make_client, workspace, bug_repo):
    before = _snapshot(bug_repo)
    llm, _ = _llm(_response(text=_fix_json()))
    client = make_client(root=workspace, llm=llm)

    client.post("/agent/fix-proposal", json=_body())

    assert _snapshot(bug_repo) == before
