from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnalysisLimits:
    max_file_bytes: int = 1_000_000
    max_files: int = 5_000
    max_search_results: int = 200
    git_timeout_seconds: float = 10.0
    max_git_output_bytes: int = 200_000


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    size: int


@dataclass
class DiscoveryResult:
    files: list[SourceFile] = field(default_factory=list)
    skipped_sensitive: int = 0
    skipped_oversize: int = 0
    skipped_links: int = 0
    excluded_dirs: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class SearchMatch:
    path: str
    line: int
    snippet: str


@dataclass
class SearchResult:
    matches: list[SearchMatch] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped_binary: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class GitStatusEntry:
    status: str
    path: str


@dataclass
class GitStatusResult:
    is_git_repo: bool
    git_available: bool
    branch: str | None = None
    entries: list[GitStatusEntry] = field(default_factory=list)
    truncated: bool = False
    message: str | None = None


@dataclass
class GitDiffResult:
    is_git_repo: bool
    git_available: bool
    staged: bool
    diff: str = ""
    truncated: bool = False
    message: str | None = None
