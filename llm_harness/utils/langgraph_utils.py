"""Helpers for rendering artifact files from LangGraph graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_GRAPH_DIR = Path("artifacts/graphs")


def write_langgraph_artifacts(
    graph: Any,
    *,
    filename_stem: str = "langgraph",
    output_dir: str | Path | None = None,
    root_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write Mermaid and PNG artifacts for any compiled LangGraph graph."""
    resolved_root_dir = Path.cwd() if root_dir is None else Path(root_dir).expanduser().resolve()
    resolved_output_dir = resolved_root_dir / DEFAULT_GRAPH_DIR if output_dir is None else Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    graph_spec = graph.get_graph()
    mermaid_path = resolved_output_dir / f"{filename_stem}.mmd"
    png_path = resolved_output_dir / f"{filename_stem}.png"
    if overwrite or not mermaid_path.exists():
        mermaid_path.write_text(graph_spec.draw_mermaid(), encoding="utf-8")
    if overwrite or not png_path.exists():
        png_path.write_bytes(graph_spec.draw_mermaid_png())
    return {
        "mermaid_path": str(mermaid_path),
        "png_path": str(png_path),
    }
