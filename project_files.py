"""Safe, bounded access to explicitly requested files in the project tree."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_PROJECT_FILE_BYTES = 256_000
MAX_PROJECT_FILE_CONTEXT_CHARS = 8_000


class ProjectFileError(ValueError):
    """A requested project file could not be read safely."""


def read_project_file(relative_path: str, *, project_root: str | Path | None = None) -> str:
    """Read a bounded UTF-8 file while ensuring its resolved path stays in root."""
    raw_path = str(relative_path or "").strip().replace("\\", "/")
    if not raw_path or Path(raw_path).is_absolute() or re.match(r"^[A-Za-z]:", raw_path):
        raise ProjectFileError("Use a relative path inside the project directory.")
    if any(part.lower() == ".env" or part.lower().startswith(".env.") for part in raw_path.split("/")):
        raise ProjectFileError("Credential configuration files cannot be read through chat.")
    root = Path(project_root or PROJECT_ROOT).resolve()
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ProjectFileError("The requested path is outside the project directory.") from None
    if not candidate.is_file():
        raise ProjectFileError("The requested project file does not exist or is not a regular file.")
    try:
        size = candidate.stat().st_size
        if size > MAX_PROJECT_FILE_BYTES:
            raise ProjectFileError(f"The requested file exceeds the {MAX_PROJECT_FILE_BYTES}-byte read limit.")
        return candidate.read_text(encoding="utf-8-sig")
    except ProjectFileError:
        raise
    except (OSError, UnicodeError):
        raise ProjectFileError("The requested project file could not be read as UTF-8 text.") from None


def requirement_entries(text: str) -> list[dict[str, str]]:
    """Return package names and exact requirement lines from requirements text."""
    entries: list[dict[str, str]] = []
    for line in str(text or "").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement or requirement.startswith(("-", "--")):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(.*)$", requirement)
        if match:
            entries.append({"name": match.group(1), "requirement": requirement})
    return entries


def find_requirement(text: str, package_name: str) -> str | None:
    """Find an exact requirement line using PEP 503-style name normalization."""
    normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
    wanted = normalize(str(package_name or "").strip())
    for entry in requirement_entries(text):
        if normalize(entry["name"]) == wanted:
            return entry["requirement"]
    return None


def requested_project_paths(prompt: str) -> list[str]:
    """Find file-like relative paths explicitly mentioned in a user request."""
    pattern = r"(?<![\w.-])((?:[A-Za-z]:)?[/\\]?(?:\.\.?[/\\])?(?:[\w.-]+[/\\])*[\w.-]+\.(?:txt|md|py|toml|json|ya?ml|cfg|ini|rst|csv))(?=$|[\s,;:!?]|\.(?=\s|$))"
    found = []
    for match in re.finditer(pattern, str(prompt or ""), re.IGNORECASE):
        path = match.group(1).replace("\\", "/")
        if path not in found:
            found.append(path)
    return found[:3]

