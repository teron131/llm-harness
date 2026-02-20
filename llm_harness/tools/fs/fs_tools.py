"""Minimal filesystem tools for tool-calling.

Sandbox-oriented with root_dir + traversal protection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from langchain_core.tools import tool

PATH_TRAVERSAL_ERROR = "Path traversal not allowed"
PATH_OUTSIDE_ROOT_ERROR = "Path outside root"


@dataclass(frozen=True)
class SandboxFS:
    root_dir: Path

    def resolve(self, user_path: str) -> Path:
        cleaned_path = user_path.strip()
        if not cleaned_path:
            raise ValueError("Empty path")
        if cleaned_path.startswith("~"):
            raise ValueError(PATH_TRAVERSAL_ERROR)

        virtual_path = cleaned_path if cleaned_path.startswith("/") else f"/{cleaned_path}"
        if ".." in virtual_path:
            raise ValueError(PATH_TRAVERSAL_ERROR)

        resolved_path = (self.root_dir / virtual_path.lstrip("/")).resolve()
        try:
            resolved_path.relative_to(self.root_dir)
        except ValueError as e:
            raise ValueError(PATH_OUTSIDE_ROOT_ERROR) from e
        return resolved_path


def make_fs_tools(*, root_dir: str | Path):
    fs = SandboxFS(Path(root_dir).resolve())

    def _resolve_existing_file(path: str) -> Path:
        file_path = fs.resolve(path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path

    @tool
    def fs_read_text(path: str) -> str:
        """Read a UTF-8 text file from the sandboxed workspace.

        Args:
            path: File path relative to the sandbox root (or virtual absolute like "/foo.txt").
        """

        return _resolve_existing_file(path).read_text(encoding="utf-8")

    @tool
    def fs_write_text(path: str, text: str) -> str:
        """Write a UTF-8 text file into the sandboxed workspace.

        Creates parent directories as needed.

        Args:
            path: File path relative to the sandbox root (or virtual absolute like "/out.txt").
            text: Full file contents.
        """

        fp = fs.resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text, encoding="utf-8")
        return f"Wrote {path}"

    @tool
    def fs_edit_with_ed(path: str, script: str) -> str:
        """Edit a file by running an `ed` script against it.

        This is well-suited for LLM-generated, line-oriented patches (including multi-line inserts).

        Args:
            path: File path relative to the sandbox root (or virtual absolute like "/foo.txt").
            script: An ed program (commands + replacement lines). Prefer ending with `wq`.
        """

        fp = _resolve_existing_file(path)

        ed_path = shutil.which("ed")
        if not ed_path:
            raise RuntimeError("`ed` not found on PATH")

        proc = subprocess.run(  # noqa: S603
            [ed_path, "-s", str(fp)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ed failed (code={proc.returncode}): {err}")

        return f"Edited {path}"

    return [fs_read_text, fs_write_text, fs_edit_with_ed]


__all__ = [
    "make_fs_tools",
]
