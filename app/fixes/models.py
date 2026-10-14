"""Schemas for fix proposals. A proposal is a draft patch, never an applied change or a verified fix."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.orchestrator.models import AnalysisReport, LLMSummary
from app.orchestrator.task_router import MAX_DESCRIPTION_CHARS

FixStatus = Literal[
    "proposed",
    "no_patch",
    "patch_rejected",
    "insufficient_evidence",
    "llm_unavailable",
    "llm_error",
    "invalid_model_output",
]

# Bounds on the analysis report accepted from the client. The report is re-checked against the repository anyway.
MAX_ANALYSIS_HYPOTHESES = 5
MAX_ANALYSIS_EVIDENCE = 5
MAX_QUOTE_CHARS = 300


class FixProposalRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1024, description="Repository directory, inside AGENT_ALLOWED_REPO_ROOT.")
    bug_description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_CHARS)
    analysis: AnalysisReport = Field(description="Report returned by POST /agent/analyze for the same path.")

    @model_validator(mode="after")
    def _bound_analysis(self) -> "FixProposalRequest":
        hypotheses = self.analysis.hypotheses
        if len(hypotheses) > MAX_ANALYSIS_HYPOTHESES:
            raise ValueError(f"analysis.hypotheses may contain at most {MAX_ANALYSIS_HYPOTHESES} items.")
        for hypothesis in hypotheses:
            if len(hypothesis.evidence) > MAX_ANALYSIS_EVIDENCE:
                raise ValueError(f"Each hypothesis may cite at most {MAX_ANALYSIS_EVIDENCE} evidence items.")
            for item in hypothesis.evidence:
                if len(item.quote) > MAX_QUOTE_CHARS:
                    raise ValueError(f"Evidence quotes may be at most {MAX_QUOTE_CHARS} characters.")
        return self


# Shape the model is asked to return. Validated strictly before anything is reported.
class ModelFix(BaseModel):
    root_cause_summary: str = Field(default="", max_length=1000)
    explanation: str = Field(default="", max_length=2000)
    unified_diff: str = Field(default="", max_length=20_000)
    verification_steps: list[str] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=8)


class PatchFile(BaseModel):
    path: str
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    hunks: int = Field(ge=1)


class PatchChecks(BaseModel):
    # Checks that ran. None of them prove the fix is correct.
    hunks_match_current_files: Literal[True] = True
    python_files_parse: list[str] = Field(default_factory=list, description="Python files whose patched text parsed with ast.")
    tests_run: Literal[False] = False
    correctness: Literal["unverified"] = "unverified"


class ProposedPatch(BaseModel):
    status: Literal["proposed_unverified"] = "proposed_unverified"
    unified_diff: str
    files: list[PatchFile]
    checks: PatchChecks


class FixProposal(BaseModel):
    status: FixStatus
    repository: str
    root_cause_summary: str | None = None
    explanation: str | None = None
    proposed_patch: ProposedPatch | None = None
    affected_files: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    context_files: list[str] = Field(default_factory=list)
    context_characters: int = 0
    context_truncated: bool = False
    llm: LLMSummary
