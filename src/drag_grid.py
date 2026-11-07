"""Drag-and-drop page preview - built in-house, no external service.

A tiny Streamlit custom component (raw postMessage protocol, no npm build)
wrapped around vendored SortableJS (MIT license, lives in
components/rearrange_grid/, loads from disk - fully offline).

What the component does:
    - shows page thumbnails as cards you can DRAG with the mouse to reorder
      them, and CLICK to select (the red-framed tools target);

    - sends one event back to Python whenever the user acts.
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional

import streamlit.components.v1 as components

COMPONENT_DIR = Path(__file__).resolve().parents[1] / "components" / "rearrange_grid"

_drag_component = components.declare_component(
    "rearrange_grid", path=str(COMPONENT_DIR)
)


def drag_grid(
    page_thumbnails: List[bytes],
    page_numbers: List[int],
    selected: int,
    per_row: int,
    key: str,
) -> Optional[Dict]:
    """Show draggable page cards and return the user's last event.

    Returns one of:
        {"type": "select", "page": n}          - a card was clicked
        {"type": "order",  "order": [n1, n2]}  - cards were drag-reordered
        None                                   - nothing new since the last rerun

    Call with a fresh `key` after an event was consumed, otherwise Streamlit
    replays the stored value on every rerun.
    """
    items = [
        {
            "number": int(number),
            "image": "data:image/png;base64,"
            + base64.b64encode(png).decode("ascii"),
        }
        for number, png in zip(page_numbers, page_thumbnails)
    ]
    return _drag_component(
        items=items,
        selected=int(selected),
        per_row=int(per_row),
        key=key,
        default=None,
    )
