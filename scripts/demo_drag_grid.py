"""Throwaway visual test for the drag_grid component (port 8502 only).

Builds a synthetic 6-page PDF (no upload needed) and mirrors the app logic:
click = select, drag = reorder. Shows the last event received by Python.
"""

import sys
from pathlib import Path

import streamlit as st

# Bootstrap: the demo lives in scripts/, so put the project root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drag_grid import drag_grid
from src.page_view import render_page_with_boxes

import fitz  # PyMuPDF


def make_pdf() -> bytes:
    doc = fitz.open()
    for i in range(1, 7):
        page = doc.new_page()
        page.insert_text((72, 200), f"Demo page {i}", fontsize=40)
    return doc.tobytes()


@st.cache_data
def build_thumbnails(pdf: bytes) -> list:
    return [render_page_with_boxes(pdf, n, dpi=100) for n in range(1, 7)]


st.set_page_config(page_title="drag grid demo", layout="wide")

pdf = make_pdf()
render_pool = build_thumbnails(pdf)

st.session_state.setdefault("selected", 1)
st.session_state.setdefault("clock", 0)
st.session_state.setdefault("order", list(range(1, 7)))

order = st.session_state["order"]

if st.button("↻ rerun"):
    st.rerun()

event = drag_grid(
    page_thumbnails=[render_pool[number - 1] for number in order],
    page_numbers=order,
    selected=st.session_state["selected"],
    per_row=3,
    key=f"demo_grid_{st.session_state['clock']}",
)

st.write("Current order:", st.session_state["order"])
st.write("Selected:", st.session_state["selected"])
st.write("Last event received:", st.session_state.get("last_event"))

if isinstance(event, dict):
    st.session_state["clock"] += 1
    st.session_state["last_event"] = event
    if event.get("type") == "select":
        st.session_state["selected"] = int(event["page"])
        st.rerun()
    elif event.get("type") == "order":
        st.session_state["order"] = [int(n) for n in event["order"]]
        st.rerun()
