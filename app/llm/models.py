from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"


class LLMErrorKind(StrEnum):
    NOT_CONFIGURED = "not_configured"
    AUTH = "auth"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    INVALID_REQUEST = "invalid_request"
    SERVICE_ERROR = "service_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"


class UsageInfo(BaseModel):
    """Token counts reported by the API. Any field is None when the API did not report it."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    thoughts_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ModelPricing(BaseModel):
    """USD per 1M text tokens. Thinking tokens are billed at the output rate."""

    input_per_million_usd: float = Field(ge=0)
    output_per_million_usd: float = Field(ge=0)
    verified_on: str | None = None
    source_url: str | None = PRICING_SOURCE_URL


class CostEstimate(BaseModel):
    estimated_usd: float | None = Field(default=None, ge=0)
    status: Literal["estimated", "unknown"]
    reason: str | None = None


class GeminiResult(BaseModel):
    text: str = Field(min_length=1)
    model: str
    finish_reason: str | None = None
    usage: UsageInfo
    cost: CostEstimate


class UsageSummary(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: dict[str, int]
    input_tokens: int
    output_tokens: int
    estimated_usd: float
    requests_with_unknown_cost: int
