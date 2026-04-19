"""Workspace skills tools."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from langchain.tools import tool
import yaml

from ..tabular.storage import DEFAULT_ROOT_DIR

SKILL_FILENAME = "SKILL.md"
DEFAULT_MAX_FILES = 20
DEFAULT_MAX_CHARS_PER_FILE = 8000
IGNORED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9-]{1,64}$")


@dataclass(frozen=True)
class SkillFile:
    """Parsed skill file content plus stable path metadata."""

    path: Path
    relative_path: str
    skill_root: Path
    frontmatter: dict[str, Any]
    body: str


def _resolve_search_path(
    *,
    root_dir: Path,
    path: str,
) -> Path:
    """Resolve one search path inside the configured workspace root."""
    cleaned_path = path.strip() or "."
    candidate_path = Path(cleaned_path).expanduser().resolve() if cleaned_path.startswith("/") else (root_dir / cleaned_path).resolve()
    try:
        candidate_path.relative_to(root_dir)
    except ValueError as error:
        raise ValueError(f"Path outside root: {path}") from error
    return candidate_path


def _iter_skill_paths(search_path: Path) -> list[Path]:
    """Return stable SKILL.md paths under one rooted search path."""
    if search_path.is_file():
        return [search_path] if search_path.name == SKILL_FILENAME else []

    discovered_paths: list[Path] = []
    for current_root, dir_names, file_names in os.walk(search_path):
        dir_names[:] = sorted(name for name in dir_names if name not in IGNORED_DIR_NAMES)
        for file_name in sorted(file_names):
            if file_name == SKILL_FILENAME:
                discovered_paths.append(Path(current_root) / file_name)
    return discovered_paths


def _parse_skill_file(
    path: Path,
    *,
    root_dir: Path,
) -> SkillFile:
    """Read one skill file and return parsed frontmatter plus markdown body."""
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    return SkillFile(
        path=path,
        relative_path=str(path.relative_to(root_dir)),
        skill_root=path.parent,
        frontmatter=frontmatter,
        body=body,
    )


def _discover_skill_files(
    search_path: Path,
    *,
    root_dir: Path,
) -> list[SkillFile]:
    """Return parsed skill files under one rooted search path."""
    return [
        _parse_skill_file(
            path,
            root_dir=root_dir,
        )
        for path in _iter_skill_paths(search_path)
    ]


def _split_frontmatter(
    text: str,
) -> tuple[
    dict[str, Any],
    str,
]:
    """Return parsed YAML frontmatter plus the remaining markdown body."""
    if not text.startswith("---\n"):
        return {}, text

    end_marker = text.find("\n---\n", 4)
    if end_marker == -1:
        return {}, text

    frontmatter_text = text[4:end_marker]
    body = text[end_marker + 5 :]
    try:
        parsed_frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        return {}, body
    if not isinstance(parsed_frontmatter, dict):
        return {}, body
    return parsed_frontmatter, body


def _iter_resource_paths(skill_root: Path, resource_dir_name: str) -> list[Path]:
    """Return stable file paths under one optional skill resource directory."""
    resource_root = skill_root / resource_dir_name
    if not resource_root.is_dir():
        return []

    discovered_paths: list[Path] = []
    for current_root, dir_names, file_names in os.walk(resource_root):
        dir_names[:] = sorted(name for name in dir_names if name not in IGNORED_DIR_NAMES)
        for file_name in sorted(file_names):
            discovered_paths.append(Path(current_root) / file_name)
    return discovered_paths


def _skill_metadata(
    skill_file: SkillFile,
) -> tuple[
    dict[str, Any] | None,
    str | None,
]:
    """Build one standards-aligned metadata payload for a skill file."""
    skill_name = skill_file.frontmatter.get("name")
    description = skill_file.frontmatter.get("description")

    if not isinstance(skill_name, str) or not skill_name.strip():
        return None, f"{skill_file.relative_path}: missing required frontmatter field 'name'"
    if not SKILL_NAME_PATTERN.fullmatch(skill_name) or skill_name.startswith("-") or skill_name.endswith("-") or "--" in skill_name:
        return None, f"{skill_file.relative_path}: invalid skill name '{skill_name}'"
    if skill_file.skill_root.name != skill_name:
        return None, f"{skill_file.relative_path}: skill name '{skill_name}' must match parent directory '{skill_file.skill_root.name}'"
    if not isinstance(description, str) or not description.strip():
        return None, f"{skill_file.relative_path}: missing required frontmatter field 'description'"

    return {
        "path": skill_file.relative_path,
        "name": skill_name,
        "description": description,
    }, None


def _read_text_file(
    path: Path,
    *,
    root_dir: Path,
    max_chars: int,
) -> dict[str, Any]:
    """Return bounded UTF-8 text content for one file."""
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root_dir)),
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def _load_resource_group(
    paths: list[Path],
    *,
    root_dir: Path,
    max_chars_per_file: int,
) -> list[dict[str, Any]]:
    """Load bounded text payloads for one resource group."""
    return [
        _read_text_file(
            path,
            root_dir=root_dir,
            max_chars=max_chars_per_file,
        )
        for path in paths
    ]


def _load_skill(
    skill_file: SkillFile,
    *,
    root_dir: Path,
    max_chars_per_file: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load one skill with instructions plus scripts and references."""
    metadata, error_message = _skill_metadata(skill_file)
    if metadata is None:
        return None, error_message

    scripts = _iter_resource_paths(skill_file.skill_root, "scripts")
    references = _iter_resource_paths(skill_file.skill_root, "references")
    return {
        **metadata,
        "skill_path": str(skill_file.path),
        "instructions": skill_file.body[:max_chars_per_file],
        "instructions_truncated": len(skill_file.body) > max_chars_per_file,
        "scripts": _load_resource_group(
            scripts,
            root_dir=root_dir,
            max_chars_per_file=max_chars_per_file,
        ),
        "references": _load_resource_group(
            references,
            root_dir=root_dir,
            max_chars_per_file=max_chars_per_file,
        ),
    }, None


def _matching_skill_files(
    skill_files: list[SkillFile],
    names: list[str],
) -> tuple[list[SkillFile], list[str]]:
    """Return selected skill files plus diagnostics for unknown names."""
    if not names:
        return skill_files, []

    files_by_name: dict[str, SkillFile] = {}
    diagnostics: list[str] = []
    for skill_file in skill_files:
        metadata, error_message = _skill_metadata(skill_file)
        if metadata is not None:
            files_by_name[metadata["name"]] = skill_file
        elif error_message is not None:
            diagnostics.append(error_message)

    selected_files: list[SkillFile] = []
    for name in names:
        matched_file = files_by_name.get(name)
        if matched_file is None:
            diagnostics.append(f"Skill not found: {name}")
            continue
        selected_files.append(matched_file)
    return selected_files, diagnostics


def _result_payload(
    skills: list[dict[str, Any]],
    diagnostics: list[str],
) -> dict[str, Any]:
    """Build the shared result shape for skills tools."""
    result: dict[str, Any] = {
        "status": "ok",
        "skills": skills,
    }
    if diagnostics:
        result["diagnostics"] = diagnostics
    return result


@tool(parse_docstring=True)
def list_skills(
    path: str = ".",
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """List skills from the workspace.

    Args:
        path: Relative directory or file path inside the workspace root. Defaults to the workspace root.
        max_files: Maximum number of skill files to return.
    """
    search_path = _resolve_search_path(
        root_dir=DEFAULT_ROOT_DIR,
        path=path,
    )
    skill_files = _discover_skill_files(search_path, root_dir=DEFAULT_ROOT_DIR)
    bounded_files = skill_files[: max(0, max_files)]
    skills: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for skill_file in bounded_files:
        payload, error_message = _skill_metadata(skill_file)
        if payload is not None:
            skills.append(payload)
        elif error_message is not None:
            diagnostics.append(error_message)
    return _result_payload(skills, diagnostics)


@tool(parse_docstring=True)
def load_skills(
    path: str = ".",
    skill_names: list[str] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_chars_per_file: int = DEFAULT_MAX_CHARS_PER_FILE,
) -> dict[str, Any]:
    """Load skills from the workspace, including instructions, scripts, and references.

    Args:
        path: Relative directory or file path inside the workspace root. Defaults to the workspace root.
        skill_names: Optional skill names to load. When omitted, loads all discovered skills up to max_files.
        max_files: Maximum number of skills to load.
        max_chars_per_file: Maximum number of characters to load from each text file.
    """
    search_path = _resolve_search_path(
        root_dir=DEFAULT_ROOT_DIR,
        path=path,
    )
    skill_files = _discover_skill_files(search_path, root_dir=DEFAULT_ROOT_DIR)
    selected_files, diagnostics = _matching_skill_files(
        skill_files,
        names=skill_names or [],
    )
    bounded_files = (selected_files or skill_files)[: max(0, max_files)]
    skills: list[dict[str, Any]] = []
    for skill_file in bounded_files:
        payload, error_message = _load_skill(
            skill_file,
            root_dir=DEFAULT_ROOT_DIR,
            max_chars_per_file=max(0, max_chars_per_file),
        )
        if payload is not None:
            skills.append(payload)
        elif error_message is not None:
            diagnostics.append(error_message)
    return _result_payload(skills, diagnostics)
