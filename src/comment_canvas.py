

import base64
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from src.pdf_comments import PdfComment


COMPONENT_DIR = Path(__file__).resolve().parents[1] / "components" / "pdf_comments"

_comment_canvas = components.declare_component(
    "pdf_comment_canvas", path=str(COMPONENT_DIR)
)


def comment_canvas(
    page_image: bytes,
    comments: list[PdfComment],
    *,
    draft_text: str,
    draft_color: str,
    draft_highlighted: bool,
    key: str,
) -> dict[str, Any] | None:
    """Display a page and return one drag-to-place or marker-move event."""
    markers = [
        {
            "id": comment.comment_id,
            "text": comment.text,
            "x": comment.x,
            "y": comment.y,
            "highlighted": comment.highlighted,
            "color": comment.color,
        }
        for comment in comments
    ]
    return _comment_canvas(
        image="data:image/png;base64," + base64.b64encode(page_image).decode("ascii"),
        comments=markers,
        draft_text=draft_text,
        draft_color=draft_color,
        draft_highlighted=draft_highlighted,
        key=key,
        default=None,
    )
