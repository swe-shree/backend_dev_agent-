import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

from app.api.llm import get_gemini_client
from app.config import get_settings
from app.llm.gemini_client import GeminiClient, GeminiConfig
from app.llm.token_tracker import TokenTracker, get_tracker
from app.main import create_app
from tests.test_llm_client import FakeSDK, _api_error, _response


@pytest.fixture
def llm_env(monkeypatch):
    """Reset LLM singletons around each test so counts and clients never leak between tests."""
    from app.api import llm as llm_api

    get_tracker.cache_clear()
    llm_api._client_singleton.cache_clear()
    yield monkeypatch
    get_tracker.cache_clear()
    llm_api._client_singleton.cache_clear()


def _client_with(outcome, tracker):
    return GeminiClient(
        GeminiConfig(model="gemini-2.5-flash", timeout_seconds=5, thinking_budget=0),
        FakeSDK(outcome),
        tracker,
    )


def _app_with_client(client):
    app = create_app()
    app.dependency_overrides[get_gemini_client] = lambda: client
    return TestClient(app)


def test_health_works_without_gemini_key(llm_env):
    llm_env.delenv("GEMINI_API_KEY", raising=False)
    llm_env.delenv("AGENT_ALLOWED_REPO_ROOT", raising=False)
    get_settings.cache_clear()

    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200


def test_llm_test_returns_503_when_key_missing(llm_env):
    llm_env.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post("/llm/test", json={"prompt": "hi"})

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "not_configured"


def test_llm_test_success_returns_text_and_usage(llm_env):
    tracker = TokenTracker()
    client = _app_with_client(_client_with(_response(text="Pong"), tracker))

    response = client.post("/llm/test", json={"prompt": "Ping", "max_output_tokens": 16})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Pong"
    assert body["model"] == "gemini-2.5-flash"
    assert body["usage"]["input_tokens"] == 1000
    assert body["cost"]["status"] == "estimated"


@pytest.mark.parametrize(
    "failure,status,error",
    [
        (_api_error(401, "UNAUTHENTICATED", "bad key"), 502, "auth"),
        (_api_error(429, "RESOURCE_EXHAUSTED", "Too many requests"), 429, "rate_limited"),
        (_api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded for metric"), 429, "quota_exhausted"),
        (errors.ServerError(500, {"error": {"code": 500, "message": "boom", "status": "INTERNAL"}}), 502, "service_error"),
        (httpx.ReadTimeout("slow"), 504, "timeout"),
        (types.GenerateContentResponse(candidates=[]), 502, "invalid_response"),
    ],
)
def test_llm_test_maps_failures_to_http_status(llm_env, failure, status, error):
    tracker = TokenTracker()
    client = _app_with_client(_client_with(failure, tracker))

    response = client.post("/llm/test", json={"prompt": "hi"})

    assert response.status_code == status
    assert response.json()["detail"]["error"] == error


def test_llm_test_rejects_overlong_prompt(llm_env):
    llm_env.setenv("AGENT_GEMINI_MAX_PROMPT_CHARS", "10")
    get_settings.cache_clear()
    client = _app_with_client(_client_with(_response(), TokenTracker()))

    response = client.post("/llm/test", json={"prompt": "x" * 11})

    assert response.status_code == 422


def test_llm_test_rejects_output_token_request_above_ceiling(llm_env):
    llm_env.setenv("AGENT_GEMINI_MAX_OUTPUT_TOKENS", "100")
    get_settings.cache_clear()
    client = _app_with_client(_client_with(_response(), TokenTracker()))

    response = client.post("/llm/test", json={"prompt": "hi", "max_output_tokens": 101})

    assert response.status_code == 422


def test_llm_test_rejects_empty_prompt(llm_env):
    client = _app_with_client(_client_with(_response(), TokenTracker()))

    assert client.post("/llm/test", json={"prompt": ""}).status_code == 422


def test_prompt_is_passed_only_as_given(llm_env):
    sdk = FakeSDK(_response())
    client = GeminiClient(GeminiConfig(model="gemini-2.5-flash", timeout_seconds=5, thinking_budget=0), sdk, TokenTracker())
    app = create_app()
    app.dependency_overrides[get_gemini_client] = lambda: client

    TestClient(app).post("/llm/test", json={"prompt": "exact text"})

    assert sdk.models.last_kwargs["contents"] == "exact text"


def test_usage_endpoint_counts_requests_including_failures(llm_env):
    tracker = TokenTracker()
    get_tracker.cache_clear()
    llm_env.setattr("app.api.llm.get_tracker", lambda: tracker)
    ok = _app_with_client(_client_with(_response(), tracker))
    failing = _app_with_client(_client_with(_api_error(429, "RESOURCE_EXHAUSTED", "Too many"), tracker))

    ok.post("/llm/test", json={"prompt": "hi"})
    failing.post("/llm/test", json={"prompt": "hi"})
    summary = ok.get("/llm/usage").json()

    assert summary["total_requests"] == 2
    assert summary["successful_requests"] == 1
    assert summary["failed_requests"] == {"rate_limited": 1}
