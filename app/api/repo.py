import re
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.analysis.access import RepoRootUnavailableError, analysis_limits, resolve_request_path
from app.analysis.discovery import discover_files
from app.analysis.git import GitTimeoutError, get_diff, get_status, is_git_repository
from app.analysis.models import AnalysisLimits
from app.analysis.paths import PathValidationError
from app.analysis.search import search_files
from app.config import get_settings

router = APIRouter(prefix="/repo", tags=["repository"])

_EXTENSION_PATTERN = re.compile(r"\.[a-z0-9_+-]{1,16}")
_SUMMARY_EXTENSION_LIMIT = 20


class RepoRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1024, description="Repository directory, relative to AGENT_ALLOWED_REPO_ROOT or absolute inside it.")


class RepoSummaryResponse(BaseModel):
    repo_name: str
    is_git_repo: bool
    files_included: int
    total_bytes: int
    files_skipped_sensitive: int
    files_skipped_oversize: int
    links_skipped: int
    directories_excluded: int
    truncated: bool
    extensions: dict[str, int]


class SearchRequest(RepoRequest):
    query: str = Field(min_length=1, max_length=200)
    extensions: list[str] | None = Field(default=None, max_length=20, description="Extensions to include, e.g. [\".py\"].")
    case_sensitive: bool = False
    max_results: int = Field(default=50, ge=1, le=1000)

    @field_validator("extensions")
    @classmethod
    def _normalize_extensions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        for extension in value:
            candidate = extension.strip().lower()
            if not _EXTENSION_PATTERN.fullmatch(candidate):
                raise ValueError(f"Invalid extension: {candidate!r}")
            normalized.append(candidate)
        return normalized


class SearchMatchModel(BaseModel):
    path: str
    line: int
    snippet: str


class SearchResponse(BaseModel):
    query: str
    matches: list[SearchMatchModel]
    files_scanned: int
    files_skipped_binary: int
    truncated: bool


class GitStatusEntryModel(BaseModel):
    status: str
    path: str


class GitStatusResponse(BaseModel):
    is_git_repo: bool
    git_available: bool
    branch: str | None
    entries: list[GitStatusEntryModel]
    truncated: bool
    message: str | None


class GitDiffRequest(RepoRequest):
    staged: bool = Field(default=False, description="Show staged changes instead of unstaged.")


class GitDiffResponse(BaseModel):
    is_git_repo: bool
    git_available: bool
    staged: bool
    diff: str
    truncated: bool
    message: str | None


def _limits() -> AnalysisLimits:
    return analysis_limits(get_settings())


def _resolve_repo(path: str) -> Path:
    try:
        return resolve_request_path(path, get_settings())
    except RepoRootUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _git_timeout(exc: GitTimeoutError) -> HTTPException:
    return HTTPException(status_code=504, detail=str(exc))


@router.post("/summary", response_model=RepoSummaryResponse)
def repo_summary(request: RepoRequest) -> RepoSummaryResponse:
    repo = _resolve_repo(request.path)
    limits = _limits()
    discovery = discover_files(repo, limits)

    extensions = Counter(source.path.suffix.lower() or "(none)" for source in discovery.files)
    try:
        is_git = is_git_repository(repo, limits)
    except GitTimeoutError as exc:
        raise _git_timeout(exc) from exc

    return RepoSummaryResponse(
        repo_name=repo.name or "repository",
        is_git_repo=is_git,
        files_included=len(discovery.files),
        total_bytes=sum(source.size for source in discovery.files),
        files_skipped_sensitive=discovery.skipped_sensitive,
        files_skipped_oversize=discovery.skipped_oversize,
        links_skipped=discovery.skipped_links,
        directories_excluded=discovery.excluded_dirs,
        truncated=discovery.truncated,
        extensions=dict(extensions.most_common(_SUMMARY_EXTENSION_LIMIT)),
    )


@router.post("/search", response_model=SearchResponse)
def repo_search(request: SearchRequest) -> SearchResponse:
    repo = _resolve_repo(request.path)
    limits = _limits()
    discovery = discover_files(repo, limits)

    result = search_files(
        discovery.files,
        request.query,
        extensions=frozenset(request.extensions) if request.extensions else None,
        case_sensitive=request.case_sensitive,
        max_results=min(request.max_results, limits.max_search_results),
        max_file_bytes=limits.max_file_bytes,
    )
    return SearchResponse(
        query=request.query,
        matches=[SearchMatchModel(path=m.path, line=m.line, snippet=m.snippet) for m in result.matches],
        files_scanned=result.files_scanned,
        files_skipped_binary=result.files_skipped_binary,
        truncated=result.truncated or discovery.truncated,
    )


@router.post("/git/status", response_model=GitStatusResponse)
def repo_git_status(request: RepoRequest) -> GitStatusResponse:
    repo = _resolve_repo(request.path)
    try:
        result = get_status(repo, _limits())
    except GitTimeoutError as exc:
        raise _git_timeout(exc) from exc
    return GitStatusResponse(
        is_git_repo=result.is_git_repo,
        git_available=result.git_available,
        branch=result.branch,
        entries=[GitStatusEntryModel(status=e.status, path=e.path) for e in result.entries],
        truncated=result.truncated,
        message=result.message,
    )


@router.post("/git/diff", response_model=GitDiffResponse)
def repo_git_diff(request: GitDiffRequest) -> GitDiffResponse:
    repo = _resolve_repo(request.path)
    try:
        result = get_diff(repo, _limits(), staged=request.staged)
    except GitTimeoutError as exc:
        raise _git_timeout(exc) from exc
    return GitDiffResponse(
        is_git_repo=result.is_git_repo,
        git_available=result.git_available,
        staged=result.staged,
        diff=result.diff,
        truncated=result.truncated,
        message=result.message,
    )
