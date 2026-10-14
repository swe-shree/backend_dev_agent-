import httpx
import pytest
from google.genai import errors, types

from app.config import Settings
from app.llm.gemini_client import GeminiClient, GeminiConfig, GeminiError, build_gemini_client, build_http_options
from app.llm.models import LLMErrorKind
from app.llm.token_tracker import TokenTracker


class FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0
        self.last_kwargs = None

    def generate_content(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class FakeSDK:
    def __init__(self, outcome):
        self.models = FakeModels(outcome)


def _response(text="Hello there", thoughts=None, usage=True):
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000,
            candidates_token_count=500,
            thoughts_token_count=thoughts,
            total_token_count=1500 + (thoughts or 0),
        )
        if usage
        else None,
    )


def _client(outcome, tracker=None, **config):
    options = {
        "model": "gemini-2.5-flash",
        "timeout_seconds": 5,
        "thinking_budget": 0,
    }
    options.update(config)
    sdk = FakeSDK(outcome)
    tracker = tracker or TokenTracker()
    return GeminiClient(GeminiConfig(**options), sdk, tracker), sdk, tracker


def _api_error(code, status, message):
    return errors.ClientError(code, {"error": {"code": code, "message": message, "status": status}})


def test_missing_api_key_is_reported_as_not_configured():
    settings = Settings(_env_file=None, gemini_api_key=None)

    with pytest.raises(GeminiError) as info:
        build_gemini_client(settings, TokenTracker())

    assert info.value.kind == LLMErrorKind.NOT_CONFIGURED


def test_blank_api_key_is_treated_as_missing():
    settings = Settings(_env_file=None, gemini_api_key="   ")

    with pytest.raises(GeminiError) as info:
        build_gemini_client(settings, TokenTracker())

    assert info.value.kind == LLMErrorKind.NOT_CONFIGURED


def test_default_model_is_current_flash_class_and_thinking_uses_model_default():
    settings = Settings(_env_file=None)

    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.gemini_thinking_budget is None


def test_api_key_is_secret_and_not_shown_in_settings_repr():
    settings = Settings(_env_file=None, gemini_api_key="super-secret-value")

    assert "super-secret-value" not in repr(settings)
    assert settings.gemini_api_key.get_secret_value() == "super-secret-value"


def test_gemini_key_env_var_is_read_without_agent_prefix(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key.get_secret_value() == "from-env"


def test_successful_response_is_validated_and_costed():
    client, sdk, tracker = _client(_response(thoughts=100))

    result = client.generate("Say hello", max_output_tokens=64)

    assert result.text == "Hello there"
    assert result.finish_reason == "STOP"
    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 500
    assert result.usage.thoughts_tokens == 100
    # 1000 * 0.30/M + (500 + 100 thinking) * 2.50/M = 0.0003 + 0.0015
    assert result.cost.status == "estimated"
    assert result.cost.estimated_usd == pytest.approx(0.0018)


def test_request_carries_model_and_output_limit():
    client, sdk, _ = _client(_response())

    client.generate("hi", max_output_tokens=32)

    assert sdk.models.last_kwargs["model"] == "gemini-2.5-flash"
    assert sdk.models.last_kwargs["config"].max_output_tokens == 32
    assert sdk.models.last_kwargs["config"].thinking_config.thinking_budget == 0


def test_thought_parts_are_excluded_from_text():
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="internal reasoning", thought=True), types.Part(text="Answer")],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    client, _, _ = _client(response)

    assert client.generate("hi", max_output_tokens=8).text == "Answer"


def test_missing_usage_marks_cost_unknown_instead_of_inventing_it():
    client, _, _ = _client(_response(usage=False))

    result = client.generate("hi", max_output_tokens=8)

    assert result.usage.input_tokens is None
    assert result.cost.status == "unknown"
    assert result.cost.estimated_usd is None


def test_unpriced_model_marks_cost_unknown():
    client, _, _ = _client(_response(), model="some-unknown-model")

    result = client.generate("hi", max_output_tokens=8)

    assert result.cost.status == "unknown"
    assert "No pricing" in result.cost.reason


def test_configured_price_overrides_builtin_table():
    client, _, _ = _client(
        _response(),
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=2.0,
    )

    result = client.generate("hi", max_output_tokens=8)

    # 1000 * 1.0/M + 500 * 2.0/M = 0.002
    assert result.cost.estimated_usd == pytest.approx(0.002)


def test_empty_candidates_are_malformed_output():
    client, _, _ = _client(types.GenerateContentResponse(candidates=[]))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.INVALID_RESPONSE


def test_candidate_without_text_is_malformed_output():
    client, _, _ = _client(_response(text=None))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.INVALID_RESPONSE
    assert "no text" in info.value.message


@pytest.mark.parametrize("code,status", [(401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED")])
def test_auth_failures(code, status):
    client, _, _ = _client(_api_error(code, status, "bad credentials"))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.AUTH


def test_invalid_api_key_400_is_auth_failure():
    client, _, _ = _client(_api_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.AUTH


def test_rate_limit_is_classified():
    client, _, _ = _client(_api_error(429, "RESOURCE_EXHAUSTED", "Too many requests per minute"))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.RATE_LIMITED


def test_quota_exhaustion_is_classified_separately():
    client, _, _ = _client(_api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded for metric generate_content"))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.QUOTA_EXHAUSTED


def test_service_error_is_classified():
    client, _, _ = _client(errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.SERVICE_ERROR


def test_timeout_is_classified():
    client, _, _ = _client(httpx.ReadTimeout("timed out"))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.TIMEOUT


def test_connection_failure_is_service_error():
    client, _, _ = _client(httpx.ConnectError("refused"))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert info.value.kind == LLMErrorKind.SERVICE_ERROR


@pytest.mark.parametrize(
    "failure",
    [_api_error(429, "RESOURCE_EXHAUSTED", "Too many requests"), httpx.ReadTimeout("slow")],
)
def test_failures_are_not_retried(failure):
    client, sdk, _ = _client(failure)

    with pytest.raises(GeminiError):
        client.generate("hi", max_output_tokens=8)

    assert sdk.models.calls == 1


def test_sdk_retries_are_disabled_and_timeout_is_set():
    options = build_http_options(12.5)

    assert options.retry_options.attempts == 1
    assert options.timeout == 12500


def test_error_messages_never_echo_provider_text():
    secret_like = "AIzaSyD-fake-key-should-not-appear"
    client, _, _ = _client(_api_error(401, "UNAUTHENTICATED", f"key {secret_like} invalid"))

    with pytest.raises(GeminiError) as info:
        client.generate("hi", max_output_tokens=8)

    assert secret_like not in info.value.message
