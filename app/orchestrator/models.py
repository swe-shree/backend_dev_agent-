from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.llm.models import CostEstimate, UsageInfo

Confidence = Literal["low", "medium", "high"]
AnalysisStatus = Literal["completed", "insufficient_context", "llm_unavailable", "llm_error", "invalid_model_output"]


# Shape the model is asked to return. Validated strictly before anything is reported.
class ModelEvidence(BaseModel):
    file: str = Field(max_length=300)
    line: int | None = None
    quote: str = Field(default="", max_length=300)


class ModelHypothesis(BaseModel):
    statement: str = Field(min_length=1, max_length=1000)
    confidence: Confidence = "low"
    evidence: list[ModelEvidence] = Field(default_factory=list, max_length=5)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class ModelFindings(BaseModel):
    root_cause_hypotheses: list[ModelHypothesis] = Field(default_factory=list, max_length=3)
    verification_steps: list[str] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=8)


# What the report returns to the caller.
class EvidenceRef(BaseModel):
    file: str
    line: int | None
    quote: str
    # True only when the quote appears verbatim in the excerpts that were sent to the model.
    verified: bool


class Hypothesis(BaseModel):
    """A model-proposed explanation. Always a hypothesis, never a confirmed fact."""

    statement: str
    confidence: Confidence
    evidence: list[EvidenceRef]


class ContextSummary(BaseModel):
    files: list[str]
    blocks: int
    characters: int
    terms_searched: list[str]
    matched_terms: dict[str, list[str]]
    total_matches: int
    truncated: bool


class LLMSummary(BaseModel):
    called: bool
    model: str | None = None
    usage: UsageInfo | None = None
    cost: CostEstimate | None = None
    error: str | None = None


class AnalysisReport(BaseModel):
    status: AnalysisStatus
    repository: str
    search_terms: list[str]
    confirmed_facts: list[str]
    hypotheses: list[Hypothesis]
    affected_files: list[str]
    verification_steps: list[str]
    limitations: list[str]
    context: ContextSummary
    llm: LLMSummary
