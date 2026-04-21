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
) -> dict[str, str]:
    """Write Mermaid and PNG artifacts for any compiled LangGraph graph."""
    resolved_output_dir = (Path(output_dir) if output_dir is not None else DEFAULT_GRAPH_DIR).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    graph_spec = graph.get_graph()
    mermaid_path = resolved_output_dir / f"{filename_stem}.mmd"
    png_path = resolved_output_dir / f"{filename_stem}.png"
    if not mermaid_path.exists():
        mermaid_path.write_text(graph_spec.draw_mermaid(), encoding="utf-8")
    if not png_path.exists():
        png_path.write_bytes(graph_spec.draw_mermaid_png())
    return {
        "mermaid_path": str(mermaid_path),
        "png_path": str(png_path),
    }
