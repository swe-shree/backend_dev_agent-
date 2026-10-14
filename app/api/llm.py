from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.gemini_client import GeminiClient, GeminiError, build_gemini_client
from app.llm.models import GeminiResult, LLMErrorKind, UsageSummary
from app.llm.token_tracker import get_tracker

router = APIRouter(prefix="/llm", tags=["llm"])

_STATUS_BY_KIND: dict[LLMErrorKind, int] = {
    LLMErrorKind.NOT_CONFIGURED: 503,
    LLMErrorKind.AUTH: 502,
    LLMErrorKind.RATE_LIMITED: 429,
    LLMErrorKind.QUOTA_EXHAUSTED: 429,
    LLMErrorKind.INVALID_REQUEST: 400,
    LLMErrorKind.SERVICE_ERROR: 502,
    LLMErrorKind.TIMEOUT: 504,
    LLMErrorKind.INVALID_RESPONSE: 502,
}


class LLMTestRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Short developer-supplied prompt. Repository contents are never added.")
    max_output_tokens: int = Field(default=256, ge=1)


@lru_cache
def _client_singleton() -> GeminiClient:
    return build_gemini_client(get_settings(), get_tracker())


def get_gemini_client() -> GeminiClient:
    try:
        return _client_singleton()
    except GeminiError as exc:
        raise HTTPException(status_code=_STATUS_BY_KIND[exc.kind], detail={"error": exc.kind.value, "message": exc.message}) from exc


def get_llm_or_error() -> GeminiClient | GeminiError:
    """Lets callers that can work without a model (such as bug analysis) see the configuration error instead of failing."""
    try:
        return _client_singleton()
    except GeminiError as exc:
        return exc


@router.post("/test", response_model=GeminiResult)
def llm_test(request: LLMTestRequest, client: GeminiClient = Depends(get_gemini_client)) -> GeminiResult:
    settings = get_settings()
    if len(request.prompt) > settings.gemini_max_prompt_chars:
        raise HTTPException(status_code=422, detail=f"Prompt exceeds {settings.gemini_max_prompt_chars} characters.")
    if request.max_output_tokens > settings.gemini_max_output_tokens:
        raise HTTPException(status_code=422, detail=f"max_output_tokens must be at most {settings.gemini_max_output_tokens}.")

    try:
        return client.generate(request.prompt, max_output_tokens=request.max_output_tokens)
    except GeminiError as exc:
        raise HTTPException(
            status_code=_STATUS_BY_KIND[exc.kind],
            detail={"error": exc.kind.value, "message": exc.message},
        ) from exc


@router.get("/usage", response_model=UsageSummary)
def llm_usage() -> UsageSummary:
    return get_tracker().summary()
