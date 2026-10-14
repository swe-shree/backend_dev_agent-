"""Gemini client built on the official Google Gen AI SDK.

Failures become GeminiError with a fixed, non-sensitive message. Provider text and the API key are
never placed in errors, logs or responses. Retries are disabled in the SDK, and this client never retries.
"""

import time
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors, types

from app.config import Settings
from app.llm.models import GeminiResult, LLMErrorKind, ModelPricing, UsageInfo
from app.llm.token_tracker import TokenTracker, estimate_cost, resolve_pricing


class GeminiError(Exception):
    def __init__(self, kind: LLMErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class GeminiConfig:
    model: str
    timeout_seconds: float
    thinking_budget: int | None
    input_price_per_million_usd: float | None = None
    output_price_per_million_usd: float | None = None


def build_http_options(timeout_seconds: float) -> types.HttpOptions:
    # attempts=1 turns off the SDK's built-in retries. Any retry has to be added deliberately, with limits.
    return types.HttpOptions(
        timeout=int(timeout_seconds * 1000),
        retry_options=types.HttpRetryOptions(attempts=1),
    )


def map_api_error(exc: errors.APIError) -> GeminiError:
    code = exc.code or 0
    status = (exc.status or "").upper()
    message = (exc.message or "").lower()

    if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED") or (code == 400 and "api key" in message):
        return GeminiError(LLMErrorKind.AUTH, "Gemini rejected the configured API key.")
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        if "quota" in message:
            return GeminiError(LLMErrorKind.QUOTA_EXHAUSTED, "Gemini quota is exhausted for this key or project.")
        return GeminiError(LLMErrorKind.RATE_LIMITED, "Gemini rate limit reached. Try again later.")
    if code == 404:
        return GeminiError(
            LLMErrorKind.INVALID_REQUEST,
            "The configured Gemini model was not found or is unavailable to this key. Check AGENT_GEMINI_MODEL.",
        )
    if code == 400:
        return GeminiError(LLMErrorKind.INVALID_REQUEST, "Gemini rejected the request.")
    return GeminiError(LLMErrorKind.SERVICE_ERROR, "The Gemini service returned an error.")


def _finish_reason_name(candidate: types.Candidate) -> str | None:
    reason = candidate.finish_reason
    if reason is None:
        return None
    return str(getattr(reason, "value", reason))


def _extract_text(candidate: types.Candidate) -> str:
    parts = candidate.content.parts if candidate.content and candidate.content.parts else []
    return "".join(part.text for part in parts if part.text and not part.thought)


def _usage_from(metadata: types.GenerateContentResponseUsageMetadata | None) -> UsageInfo:
    if metadata is None:
        return UsageInfo()
    return UsageInfo(
        input_tokens=metadata.prompt_token_count,
        output_tokens=metadata.candidates_token_count,
        thoughts_tokens=metadata.thoughts_token_count,
        total_tokens=metadata.total_token_count,
    )


class GeminiClient:
    def __init__(self, config: GeminiConfig, sdk_client, tracker: TokenTracker) -> None:
        self._config = config
        self._sdk = sdk_client
        self._tracker = tracker

    @property
    def model(self) -> str:
        return self._config.model

    def generate(self, prompt: str, *, max_output_tokens: int, json_output: bool = False) -> GeminiResult:
        started = time.monotonic()
        pricing = resolve_pricing(
            self._config.model,
            self._config.input_price_per_million_usd,
            self._config.output_price_per_million_usd,
        )
        try:
            result = self._call(prompt, max_output_tokens=max_output_tokens, pricing=pricing, json_output=json_output)
        except GeminiError as error:
            self._tracker.record(
                model=self._config.model,
                outcome=error.kind.value,
                duration_ms=int((time.monotonic() - started) * 1000),
                usage=UsageInfo(),
                cost=estimate_cost(UsageInfo(), pricing),
            )
            raise

        self._tracker.record(
            model=result.model,
            outcome="success",
            duration_ms=int((time.monotonic() - started) * 1000),
            usage=result.usage,
            cost=result.cost,
        )
        return result

    def _call(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        pricing: ModelPricing | None,
        json_output: bool,
    ) -> GeminiResult:
        thinking = (
            types.ThinkingConfig(thinking_budget=self._config.thinking_budget)
            if self._config.thinking_budget is not None
            else None
        )
        config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            thinking_config=thinking,
            response_mime_type="application/json" if json_output else None,
        )

        try:
            response = self._sdk.models.generate_content(model=self._config.model, contents=prompt, config=config)
        except errors.APIError as exc:
            raise map_api_error(exc) from None
        except httpx.TimeoutException:
            raise GeminiError(LLMErrorKind.TIMEOUT, "The Gemini request timed out.") from None
        except (httpx.TransportError, OSError):
            raise GeminiError(LLMErrorKind.SERVICE_ERROR, "Could not reach the Gemini service.") from None

        candidates = response.candidates or []
        if not candidates:
            raise GeminiError(
                LLMErrorKind.INVALID_RESPONSE,
                "Gemini returned no candidates. The prompt may have been blocked.",
            )
        candidate = candidates[0]
        finish_reason = _finish_reason_name(candidate)
        text = _extract_text(candidate)
        if not text:
            raise GeminiError(
                LLMErrorKind.INVALID_RESPONSE,
                f"Gemini returned no text (finish reason: {finish_reason or 'unknown'}).",
            )

        usage = _usage_from(response.usage_metadata)
        # Pydantic validates the result, including that text is non-empty and token counts are non-negative.
        try:
            return GeminiResult(
                text=text,
                model=self._config.model,
                finish_reason=finish_reason,
                usage=usage,
                cost=estimate_cost(usage, pricing),
            )
        except ValueError:
            raise GeminiError(LLMErrorKind.INVALID_RESPONSE, "Gemini returned a response that failed validation.") from None


def build_gemini_client(settings: Settings, tracker: TokenTracker) -> GeminiClient:
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    if not key.strip():
        raise GeminiError(LLMErrorKind.NOT_CONFIGURED, "GEMINI_API_KEY is not set.")

    sdk = genai.Client(api_key=key.strip(), http_options=build_http_options(settings.gemini_timeout_seconds))
    config = GeminiConfig(
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
        thinking_budget=settings.gemini_thinking_budget,
        input_price_per_million_usd=settings.gemini_input_price_per_million_usd,
        output_price_per_million_usd=settings.gemini_output_price_per_million_usd,
    )
    return GeminiClient(config, sdk, tracker)
