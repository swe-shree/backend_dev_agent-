import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app.api import llm as llm_api
from app.api.llm import get_llm_or_error
from app.config import get_settings
from app.llm.gemini_client import GeminiClient, GeminiConfig, GeminiError
from app.llm.models import LLMErrorKind
from app.llm.token_tracker import TokenTracker
from app.main import create_app
from tests.test_llm_client import FakeSDK, _api_error, _response

BUG = "compute_total returns total_with_tax which is undefined, so the call fails with NameError"


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


def _llm(outcome):
    sdk = FakeSDK(outcome)
    client = GeminiClient(GeminiConfig(model="gemini-3.5-flash-lite", timeout_seconds=5, thinking_budget=None), sdk, TokenTracker())
    return client, sdk


def _findings():
    return json.dumps(
        {
            "root_cause_hypotheses": [
                {
                    "statement": "compute_total returns an undefined name.",
                    "confidence": "high",
                    "evidence": [{"file": "app/billing.py", "line": 9, "quote": "return total_with_tax  # BUG: undefined name"}],
                }
            ],
            "verification_steps": ["Run pytest tests/test_billing.py"],
            "limitations": [],
        }
    )


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_analyze_is_unavailable_without_allowed_root(make_client, bug_repo):
    client = make_client(root=None)

    response = client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": BUG})

    assert response.status_code == 503


def test_successful_analysis_returns_report_shape(make_client, workspace, bug_repo):
    llm, sdk = _llm(_response(text=_findings()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": BUG})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert set(body) >= {
        "status", "repository", "search_terms", "confirmed_facts", "hypotheses", "affected_files",
        "verification_steps", "limitations", "context", "llm",
    }
    assert body["hypotheses"][0]["evidence"][0]["verified"] is True
    assert body["llm"]["called"] is True
    assert body["llm"]["usage"]["input_tokens"] == 1000
    assert sdk.models.calls == 1


def test_response_does_not_leak_absolute_paths_or_secrets(make_client, workspace, bug_repo):
    llm, _ = _llm(_response(text=_findings()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": BUG})

    assert str(workspace) not in response.text
    assert "ENV-LEAK-VALUE" not in response.text
    assert "abc123-secret" not in response.text


@pytest.mark.parametrize("path", ["..", "../outside", "bugrepo/../../outside", "bugrepo; echo pwned", "$(whoami)"])
def test_invalid_paths_are_rejected(make_client, workspace, bug_repo, tmp_path, path):
    llm, sdk = _llm(_response(text=_findings()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/analyze", json={"path": path, "bug_description": BUG})

    assert response.status_code == 400
    assert sdk.models.calls == 0


def test_absolute_path_outside_root_is_rejected(make_client, workspace, bug_repo, tmp_path):
    llm, _ = _llm(_response(text=_findings()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/analyze", json={"path": str(tmp_path / "outside"), "bug_description": BUG})

    assert response.status_code == 400


@pytest.mark.parametrize("description", ["short", "Write me a poem about rivers and mountains."])
def test_invalid_descriptions_are_rejected(make_client, workspace, bug_repo, description):
    llm, sdk = _llm(_response(text=_findings()))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": description})

    assert response.status_code == 422
    assert sdk.models.calls == 0


def test_overlong_description_is_rejected_by_schema(make_client, workspace, bug_repo):
    client = make_client(root=workspace, llm=_llm(_response(text=_findings()))[0])

    response = client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": "x" * 2001})

    assert response.status_code == 422


def test_missing_context_returns_200_without_model_call(make_client, workspace, bug_repo):
    llm, sdk = _llm(_response(text=_findings()))
    client = make_client(root=workspace, llm=llm)

    response = client.post(
        "/agent/analyze",
        json={"path": "bugrepo", "bug_description": "crash in NoSuchSymbolAnywhere when it runs"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_context"
    assert sdk.models.calls == 0


def test_analysis_works_without_gemini_key_and_reports_why(make_client, workspace, bug_repo):
    client = make_client(root=workspace, llm=None)

    response = client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": BUG})

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "llm_unavailable"
    assert body["llm"]["error"] == "not_configured"
    assert body["confirmed_facts"]


def test_provider_error_is_reported_in_body_with_evidence(make_client, workspace, bug_repo):
    llm, _ = _llm(_api_error(429, "RESOURCE_EXHAUSTED", "Too many requests"))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": BUG})

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "llm_error"
    assert body["llm"]["error"] == "rate_limited"
    assert body["confirmed_facts"]


def test_malformed_model_output_is_reported_in_body(make_client, workspace, bug_repo):
    llm, _ = _llm(_response(text="not json"))
    client = make_client(root=workspace, llm=llm)

    response = client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": BUG})

    assert response.json()["status"] == "invalid_model_output"
    assert response.json()["hypotheses"] == []


def test_endpoint_does_not_modify_repository(make_client, workspace, bug_repo):
    before = _snapshot(bug_repo)
    llm, _ = _llm(_response(text=_findings()))
    client = make_client(root=workspace, llm=llm)

    client.post("/agent/analyze", json={"path": "bugrepo", "bug_description": BUG})

    assert _snapshot(bug_repo) == before
