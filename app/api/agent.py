from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.analysis.access import RepoRootUnavailableError
from app.analysis.paths import PathValidationError
from app.api.llm import get_llm_or_error
from app.config import get_settings
from app.fixes.models import FixProposal, FixProposalRequest
from app.fixes.patch_generator import PatchProposalEngine
from app.llm.gemini_client import GeminiClient, GeminiError
from app.orchestrator.engine import AnalysisEngine
from app.orchestrator.models import AnalysisReport
from app.orchestrator.task_router import MAX_DESCRIPTION_CHARS, TaskValidationError

router = APIRouter(prefix="/agent", tags=["agent"])


class AnalyzeRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1024, description="Repository directory, inside AGENT_ALLOWED_REPO_ROOT.")
    bug_description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_CHARS)


@router.post("/analyze", response_model=AnalysisReport)
def analyze(
    request: AnalyzeRequest,
    llm: GeminiClient | GeminiError = Depends(get_llm_or_error),
) -> AnalysisReport:
    engine = AnalysisEngine(get_settings(), llm)
    try:
        return engine.analyze(request.path, request.bug_description)
    except TaskValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RepoRootUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/fix-proposal", response_model=FixProposal)
def fix_proposal(
    request: FixProposalRequest,
    llm: GeminiClient | GeminiError = Depends(get_llm_or_error),
) -> FixProposal:
    """Drafts a patch from a completed analysis. The repository is never modified and the patch is not applied."""
    engine = PatchProposalEngine(get_settings(), llm)
    try:
        return engine.propose(request.path, request.bug_description, request.analysis)
    except TaskValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RepoRootUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
