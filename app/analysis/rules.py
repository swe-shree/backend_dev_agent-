"""Deterministic rules describing which paths the repository analysis may touch."""

import fnmatch

# Lowercase names. Matched against each directory name, at any depth.
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".idea",
        ".vscode",
        ".next",
        ".nuxt",
        ".gradle",
        "build",
        "dist",
        "target",
        "coverage",
        "htmlcov",
    }
)

# Lowercase glob patterns matched against each file name. Errs on the side of exclusion.
SENSITIVE_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "*credential*",
    "*secret*",
    "service-account*.json",
    "*.tfstate",
)


def is_excluded_dir(name: str) -> bool:
    return name.lower() in EXCLUDED_DIR_NAMES


def is_sensitive_file(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in SENSITIVE_FILE_PATTERNS)


def git_exclude_pathspecs() -> list[str]:
    """Git pathspec magic that removes sensitive files and excluded directories from git output."""
    specs = [f":(exclude,glob)**/{pattern}" for pattern in SENSITIVE_FILE_PATTERNS]
    specs += [f":(exclude,glob)**/{name}/**" for name in sorted(EXCLUDED_DIR_NAMES)]
    return specs
