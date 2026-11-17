"""AI Document Learning Assistant - entry point.

Stage 7A: the app now has the three-tab structure the user asked for:

    1. Upload     - PDF upload + step-by-step preparation + document metrics
    2. Edit PDF   - the editing workspace (replace / delete / rotate; add
                    page, merge and "OCR & save as editable text" come in
                    Stage 7B/7C) + the document quality check
    3. Ask        - the RAG: question answering with citations, plus
                    find-in-document with real page-image highlighting

Underlying pipeline (unchanged): upload PDF -> extract pages (Stage 2, OCR
fallback Stage 5) -> chunk with positions (Stage 3A) -> embed with MiniLM
(Stage 3B) -> semantic search (Stage 3C) -> grounded answer with citations
(Stage 4C) via local Ollama (Step 4B). Edits always happen on a working copy;
the AI refreshes from it automatically.

Run with:  streamlit run app.py
"""

import difflib
import re
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="AI DLA — Document Learning Assistant",
    page_icon="📚",
    layout="wide",
)

from src.chunking import chunk_pages
from src.comment_canvas import comment_canvas
from src.document_search import find_word
from src.drag_grid import drag_grid
from src.embeddings import embed_chunks
from src.llm_providers import DEFAULT_MODELS, SUPPORTED_PROVIDERS, ask_openai_compatible
from src.ollama import ask_ollama
from src.ocr import ocr_page, page_needs_ocr
from src.page_view import render_page_with_boxes, render_selectable_page_html
from src.pdf_font_estimation import estimate_scanned_page_fonts
from src.pdf_comments import (
    add_pdf_comment,
    delete_pdf_comment,
    list_pdf_comments,
    move_pdf_comment,
    update_pdf_comment,
)
from src.pdf_editor import (
    apply_page_order,
    delete_pages,
    insert_blank_page,
    insert_pages_from,
    move_page,
    replace_text,
    rotate_page,
    save_text_layer,
)
from src.pdf_loader import extract_pdf_pages
from src.qa import DEFAULT_MIN_SCORE, answer_question
from src.quality import quality_report
from src.retrieval import search_chunks

st.title("Your PDF workspace")
st.write(
    "Prepare a document, refine its pages, then ask questions with answers "
    "grounded in its contents. Processing stays on this computer."
)

# Reason explanations for "not found" bundles (see src/qa.py).
REASON_HELP = {
    "no_passages": "The document had no searchable text (empty or scanned-only pages).",
    "low_relevance": (
        "No passage was similar enough to your question, and none contained "
        "enough of the question's key terms, so no answer was attempted."
    ),
    "model_no_support": (
        "The passages looked plausible, and even after an explicit second "
        "check the local model confirmed they do not contain the answer."
    ),
    "ungrounded_answer": (
        "The model's reply contained values that appear nowhere in the "
        "retrieved passages, so it was discarded for safety."
    ),
}

# Tools dropdown entry that opens the pencil-driven comment flow.
COMMENT_TOOL = "Add a comment"


@st.cache_data  # heavy extraction runs once per document copy, not per click
def cached_extract_pages(pdf_bytes: bytes):
    """Pipeline step 1-2 (read pages; OCR runs inside for scanned pages)."""
    return extract_pdf_pages(pdf_bytes)


@st.cache_data  # the slowest step: turning texts into search vectors
def cached_embed_chunks(chunks):
    """Pipeline step 3 (MiniLM embeddings, CPU)."""
    return embed_chunks(chunks)


@st.cache_data
def cached_quality_report(pdf_bytes: bytes):
    """Quality inspection runs once per document copy, on demand only."""
    return quality_report(pdf_bytes)


@st.cache_data(show_spinner=False)
def cached_scanned_font_estimate(pdf_bytes: bytes, page_number: int):
    """Estimate local font matches for one scanned page on demand."""
    return estimate_scanned_page_fonts(pdf_bytes, page_number, top_n=8)


tab_upload, tab_edit, tab_ask = st.tabs(
    ["1️⃣ Upload", "2️⃣ Edit PDF", "3️⃣ Ask"]
)


def _materialize_pending_page_order() -> None:
    """Fold a pending drag-grid page order into the working-copy bytes.

    While the user drags pages in the preview grid, a drop ONLY stores a
    `page_order` mapping in session state - the working-copy bytes stay
    untouched, so text extraction, embeddings and every page-thumbnail cache
    stay hot and the rearrange feels instant. The order is folded into the
    real bytes on the next FULL app rerun (any tool click, download, tab
    switch...). Grid-only fragment reruns never reach this, which is exactly
    what makes rearranging fast.
    """
    pending = st.session_state.get("page_order")
    if not pending:
        return
    try:
        result = apply_page_order(st.session_state.working_pdf, pending)
    except ValueError:
        # The page count changed (e.g. a page was removed) while an order was
        # still pending - drop the stale mapping instead of failing the run.
        st.session_state.pop("page_order", None)
        return
    st.session_state.working_pdf = result.pdf_bytes
    st.session_state.pop("page_order", None)

# --- Tab 1: Upload + preparation ----------------------------------------------
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Only local processing: the PDF is parsed inside this app and is "
        "not sent to any external service.",
    )

    if uploaded_file is None:
        st.info("Upload a PDF to get started. Then move to the Edit tab if "
                "you want to change something, or straight to Ask.")

# The pipeline powers all three tabs, so it runs OUTSIDE any tab container.
pages = None
chunks = None
embedded_chunks = None
working_pdf = None
is_edited = False

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    # --- Working copy bookkeeping (Stage 6, Feature B) ---------------------------
    # The UPLOADED bytes are never edited. Every edit happens on a separate
    # "working copy" kept in session memory. The whole pipeline (text, chunks,
    # search, Q&A, find) always reads from that working copy, so an edit flows
    # into the AI automatically on the next page refresh.
    if st.session_state.get("original_bytes") != pdf_bytes:
        st.session_state.original_bytes = pdf_bytes
        st.session_state.working_pdf = pdf_bytes
        # A different document invalidates any pending drag-grid order.
        st.session_state.pop("page_order", None)
    # Fold a pending drag-grid reorder into the real working copy ONCE, on the
    # first full rerun after the rearrange (grid-only reruns skip this).
    _materialize_pending_page_order()
    working_pdf = st.session_state.working_pdf
    is_edited = working_pdf != pdf_bytes

    try:
        # Stage 5D (part a): visible, step-by-step preparation instead of a
        # silent freeze. Each step is cached separately, so later reruns skip it.
        with tab_upload:
            with st.status("Preparing your document...", expanded=True) as prepare:
                st.write("1/3 Reading every page (OCR starts on scanned pages)...")
                pages = cached_extract_pages(working_pdf)
                st.write(f"{len(pages)} pages read. Building searchable chunks...")
                chunks = chunk_pages(uploaded_file.name, pages)
                st.write(f"{len(chunks)} chunks created. Generating embeddings...")
                embedded_chunks = cached_embed_chunks(chunks)
                prepare.update(
                    label="Document ready", state="complete", expanded=False
                )

    except ValueError as error:
        st.error(f"Could not process this document: {error}")
        st.stop()

    with tab_upload:
        # --- Summary metrics ----------------------------------------------------
        pages_without_text = [p.page_number for p in pages if not p.has_text]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("File", uploaded_file.name + (" (edited copy)" if is_edited else ""))
        col2.metric("Pages", len(pages))
        col3.metric("Searchable chunks", len(embedded_chunks))
        col4.metric("Text characters", sum(len(p.text) for p in pages))

        if not embedded_chunks:
            st.warning(
                "No native text layer was found. This document is (at least "
                "partly) a scan. Scanned pages are read via OCR automatically - "
                "run the quality check in the Edit tab to see how well."
            )
        elif pages_without_text:
            st.caption(f"Pages without extractable text: {pages_without_text}")

# --- Tab 2: Edit PDF (Stage 6 workspace + Stage 5D quality check) -------------
with tab_edit:
    if uploaded_file is None:
        st.info("Upload a PDF in the Upload tab first.")
    else:
        page_count_total = max(len(pages), 1)
        st.session_state["page_count_total"] = page_count_total

        # --- Preview grid: the whole tab = preview cards + tools ---------------
        tool_page = st.session_state.get("tool_page", 1)
        if tool_page > page_count_total:  # e.g. page was deleted
            st.session_state.tool_page = page_count_total
            tool_page = page_count_total

        @st.cache_data
        def cached_selected_page_preview(
            pdf_bytes: bytes,
            page_number: int,
            include_annotations: bool = True,
        ):
            """Render the selected page sharply for its on-demand preview."""
            return render_page_with_boxes(
                pdf_bytes,
                page_number,
                dpi=220,
                include_annotations=include_annotations,
            )

        @st.cache_data
        def cached_selectable_page_preview(pdf_bytes: bytes, page_number: int) -> str:
            """Render browser-safe preview HTML with a selectable text overlay."""
            return render_selectable_page_html(pdf_bytes, page_number, dpi=180)

        def dismiss_page_preview():
            st.session_state.pop("preview_page", None)
            st.session_state.pop("preview_original_pdf", None)

        @st.dialog(
            "Page preview",
            width="large",
            on_dismiss=dismiss_page_preview,
        )
        def show_page_preview(page_number: int):
            st.markdown(
                """
                <style>
                div[data-st-overlay-root="true"] {
                    backdrop-filter: blur(5px);
                    background: rgba(15, 23, 42, 0.22);
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            original_pdf = st.session_state.get("preview_original_pdf")
            if original_pdf is not None:
                change_summary = st.session_state.get("preview_change_summary", "")
                st.markdown("**OCR edit review**")
                st.caption(
                    "Edited areas have a subtle outline on the page. Review the "
                    "before-and-after text changes alongside the original."
                )
                if change_summary:
                    with st.expander("Before / after text changes", expanded=True):
                        st.code(change_summary, language="diff")
                original_column, edited_column = st.columns(2)
                previews = (
                    (original_column, original_pdf, "Original"),
                    (edited_column, working_pdf, "OCR edited"),
                )
                for column, pdf_bytes, label in previews:
                    with column:
                        st.markdown(f"**{label}**")
                        st.iframe(
                            cached_selectable_page_preview(pdf_bytes, page_number),
                            height=650,
                            tab_index=0,
                            alt=f"{label} PDF page {page_number}",
                        )
            else:
                st.iframe(
                    cached_selectable_page_preview(working_pdf, page_number),
                    height=750,
                    tab_index=0,
                    alt=f"PDF page {page_number}",
                )

        preview_page = st.session_state.get("preview_page")
        if preview_page is not None:
            show_page_preview(int(preview_page))

        def close_comment_workspace():
            """Dialog ✕ handler: stays closed until the tool is reselected."""
            page = st.session_state.get("tool_page", 1)
            st.session_state[f"comment_workspace_open_{page}"] = False

        @st.dialog(
            "Add a comment",
            width="large",
            on_dismiss=close_comment_workspace,
        )
        def show_comment_workspace(page_number: int):
            """Big page preview for the pencil-driven comment flow."""
            st.caption(
                "Click the ✏️ pencil on the page, type the comment, then drag "
                "on the page to drop it. Drag a dropped comment to move it, or "
                "use its ✏️ / 🗑 icons to edit the text or delete it. Click ✕ "
                "at the top right to finish."
            )
            editor_key = f"comment_editor_open_{page_number}"
            draft_key = f"comment_text_{page_number}"
            edit_id_key = f"comment_edit_id_{page_number}"
            highlight_key = f"comment_highlight_{page_number}"
            size_key = f"comment_size_{page_number}"
            comment_text = ""
            # Deferring the text clear to the top of the dialog body keeps the
            # assignment before the text_area is instantiated (Streamlit
            # forbids rewriting a widget key afterwards).
            if st.session_state.pop(f"{draft_key}_clear", False):
                st.session_state[draft_key] = ""
            # Editing an existing comment pre-fills its text in the same box.
            prefill = st.session_state.pop(f"{draft_key}_prefill", None)
            if prefill is not None:
                st.session_state[draft_key] = prefill
            # Editing also restores that comment's stored style so the
            # toggles reflect what the comment currently looks like.
            style_prefill = st.session_state.pop(
                f"{draft_key}_style_prefill", None
            )
            if isinstance(style_prefill, dict):
                st.session_state[highlight_key] = bool(
                    style_prefill.get("highlighted", False)
                )
                st.session_state[size_key] = float(
                    style_prefill.get("font_size", 11.0)
                )
            editing_id = st.session_state.get(edit_id_key)
            if st.session_state.get(editor_key, False):
                comment_text = st.text_area(
                    "Comment",
                    key=draft_key,
                    height=80,
                    placeholder="Type a comment to place on this page...",
                )
                # Optional style controls: plain text is the default; the
                # highlight and the size only change when asked for here.
                style_row = st.columns([2, 1, 1, 3])
                with style_row[0]:
                    highlighted = st.checkbox(
                        "Highlight",
                        key=highlight_key,
                        help=(
                            "Off: plain text, no highlight or border. "
                            "On: a soft colour fill behind the text."
                        ),
                    )
                with style_row[1]:
                    if st.button("A−", key=f"comment_size_down_{page_number}"):
                        st.session_state[size_key] = max(
                            4.0, float(st.session_state.get(size_key, 11.0)) - 1.0
                        )
                with style_row[2]:
                    if st.button("A+", key=f"comment_size_up_{page_number}"):
                        st.session_state[size_key] = min(
                            72.0, float(st.session_state.get(size_key, 11.0)) + 1.0
                        )
                with style_row[3]:
                    st.caption(
                        f"Text size: {float(st.session_state.get(size_key, 11.0)):g} pt "
                        "(4–72)"
                    )
                size = float(st.session_state.get(size_key, 11.0))
                action_columns = st.columns(2)
                with action_columns[0]:
                    if st.button("Cancel", key=f"comment_cancel_{page_number}"):
                        st.session_state[editor_key] = False
                        st.session_state.pop(edit_id_key, None)
                        st.session_state[f"{draft_key}_clear"] = True
                        st.rerun()
                with action_columns[1]:
                    if (
                        editing_id
                        and st.button(
                            "Save changes", key=f"comment_save_{page_number}"
                        )
                    ):
                        try:
                            with st.spinner("Updating comment..."):
                                st.session_state.working_pdf = update_pdf_comment(
                                    working_pdf,
                                    editing_id,
                                    comment_text,
                                    highlighted=highlighted,
                                    font_size=size,
                                )
                        except ValueError as comment_error:
                            st.error(f"Comment problem: {comment_error}")
                        else:
                            st.session_state[editor_key] = False
                            st.session_state.pop(edit_id_key, None)
                            st.session_state[f"{draft_key}_clear"] = True
                            st.session_state.edit_note = "Comment updated."
                            st.rerun()
            page_comments = list_pdf_comments(working_pdf, int(page_number))
            comment_event = comment_canvas(
                cached_selected_page_preview(
                    working_pdf, int(page_number), include_annotations=False
                ),
                page_comments,
                draft_text=comment_text.strip(),
                draft_color="#FFD54F",
                draft_highlighted=bool(
                    st.session_state.get(highlight_key, False)
                ),
                draft_font_size=float(
                    st.session_state.get(size_key, 11.0)
                ),
                key=(
                    "comment_canvas_"
                    f"{st.session_state.get('comment_canvas_clock', 0)}"
                ),
            )

            with st.expander(
                f"Existing comments ({len(page_comments)})", expanded=False
            ):
                if not page_comments:
                    st.caption("No comments on this page yet.")
                for comment in page_comments:
                    comment_column, delete_column = st.columns([5, 1])
                    with comment_column:
                        st.write(f"**{comment.text}**")
                        st.caption(
                            f"Page position: {comment.x:.0%}, {comment.y:.0%}"
                        )
                    with delete_column:
                        if st.button(
                            "Delete",
                            key=f"delete_comment_{comment.comment_id}",
                        ):
                            try:
                                with st.spinner("Deleting comment..."):
                                    st.session_state.working_pdf = delete_pdf_comment(
                                        working_pdf, comment.comment_id
                                    )
                            except ValueError as comment_error:
                                st.error(f"Comment problem: {comment_error}")
                            else:
                                st.session_state.edit_note = "Comment deleted."
                                st.rerun()

            if isinstance(comment_event, dict):
                st.session_state["comment_canvas_clock"] = (
                    st.session_state.get("comment_canvas_clock", 0) + 1
                )
                try:
                    if comment_event.get("type") == "pencil":
                        # Pencil toggles the comment box. Closing happens only
                        # via the deferred _clear flag, never by touching the
                        # text_area key directly.
                        opening = not st.session_state.get(editor_key, False)
                        st.session_state[editor_key] = opening
                        if not opening:
                            st.session_state.pop(edit_id_key, None)
                            st.session_state[f"{draft_key}_clear"] = True
                        st.rerun()
                    elif comment_event.get("type") == "edit":
                        target = next(
                            (
                                comment
                                for comment in page_comments
                                if comment.comment_id
                                == comment_event.get("id")
                            ),
                            None,
                        )
                        if target is not None:
                            # Defer the pre-fill so the text_area (already
                            # instantiated this run) picks it up next run.
                            st.session_state[editor_key] = True
                            st.session_state[edit_id_key] = target.comment_id
                            st.session_state[f"{draft_key}_prefill"] = (
                                target.text
                            )
                            st.session_state[f"{draft_key}_style_prefill"] = {
                                "highlighted": target.highlighted,
                                "font_size": target.font_size,
                            }
                            st.rerun()
                    elif comment_event.get("type") == "delete":
                        with st.spinner("Deleting comment..."):
                            st.session_state.working_pdf = delete_pdf_comment(
                                working_pdf, comment_event.get("id", "")
                            )
                        # A deletion can remove a comment being edited.
                        st.session_state.pop(edit_id_key, None)
                        st.session_state.edit_note = "Comment deleted."
                        st.rerun()
                    elif comment_event.get("type") == "place":
                        updated_pdf = None
                        if comment_text.strip():
                            active_edit_id = st.session_state.get(edit_id_key)
                            with st.spinner("Saving comment..."):
                                if active_edit_id:
                                    # Dropping while editing applies the new text
                                    # at the drop point (text + relocation).
                                    updated_pdf = update_pdf_comment(
                                        working_pdf,
                                        active_edit_id,
                                        comment_text,
                                        comment_event.get("x"),
                                        comment_event.get("y"),
                                        highlighted=bool(
                                            st.session_state.get(
                                                highlight_key, False
                                            )
                                        ),
                                        font_size=float(
                                            st.session_state.get(size_key, 11.0)
                                        ),
                                    )
                                    st.session_state.edit_note = (
                                        f"Comment updated on page {page_number}."
                                    )
                                else:
                                    updated_pdf = add_pdf_comment(
                                        working_pdf,
                                        int(page_number),
                                        comment_event.get("x"),
                                        comment_event.get("y"),
                                        comment_text,
                                        highlighted=bool(
                                            st.session_state.get(
                                                highlight_key, False
                                            )
                                        ),
                                        color="#FFD54F",
                                        font_size=float(
                                            st.session_state.get(size_key, 11.0)
                                        ),
                                    )
                                    st.session_state.edit_note = (
                                        f"Comment added to page {page_number}. "
                                        "Drag the comment box to reposition it."
                                    )
                            st.session_state[editor_key] = False
                            st.session_state.pop(edit_id_key, None)
                            st.session_state[f"{draft_key}_clear"] = True
                        else:
                            # Defensive: the canvas only enables drop mode
                            # with text, so this path is unlikely in practice.
                            st.error("Type a comment before dropping it on the page.")
                        if updated_pdf is not None:
                            st.session_state.working_pdf = updated_pdf
                            st.rerun()
                    elif comment_event.get("type") == "move":
                        with st.spinner("Moving comment..."):
                            updated_pdf = move_pdf_comment(
                                working_pdf,
                                str(comment_event.get("id", "")),
                                comment_event.get("x"),
                                comment_event.get("y"),
                            )
                        st.session_state.edit_note = (
                            f"Comment moved on page {page_number}."
                        )
                        st.session_state.working_pdf = updated_pdf
                        st.rerun()
                except ValueError as comment_error:
                    st.error(f"Comment problem: {comment_error}")

        @st.cache_data  # note the dpi in the key: quality follows the zoom level
        def cached_page_card(
            pdf_bytes: bytes, page_number: int, selected: bool, card_dpi: int
        ):
            """One preview card; redraws only when copy/selection/dpi change."""
            return render_page_card(pdf_bytes, page_number, selected=selected, dpi=card_dpi)

        # --- Two-section workspace layout ---------------------------------------------
        # LEFT column (~70%): the PDF workspace - page grid, edit toast,
        # download and quality check.
        # RIGHT column (~30%): the tools for the page selected in the grid.
        # The CSS below draws the vertical divider line on the tools
        # column so the two sections read as split panes.
        st.markdown(
            """
            <style>
            [data-testid="stColumn"]:has(.workspace-divider-anchor) {
                border-left: 1.5px solid rgba(130, 130, 130, 0.45);
                padding-left: 18px;
            }
            .workspace-divider-anchor { display: none; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        workspace_col, tools_col = st.columns([7, 3], gap="medium")
        with tools_col:
            st.markdown(
                '<span class="workspace-divider-anchor"></span>',
                unsafe_allow_html=True,
            )
            st.subheader("🧰 Tools")
            edit_action = st.selectbox(
                f"Tool for the red-framed page {tool_page}",
                [
                    "Replace text",
                    "Extract & edit OCR text",
                    COMMENT_TOOL,
                    "Insert blank page after this one",
                    "Insert pages from another PDF (merge/combine)",
                ],
            )

            if edit_action != COMMENT_TOOL:
                # Picking another tool forgets the comment-workspace
                # activation, so selecting the comment tool again auto-
                # opens its big dialog fresh.
                st.session_state.pop("active_comment_tool", None)

            old_text = new_text = ""
            if edit_action == "Replace text":
                col_old, col_new = st.columns(2)
                with col_old:
                    old_text = st.text_input("Text to find")
                with col_new:
                    new_text = st.text_input("Replace it with (empty = delete the text)")
            elif edit_action == "Insert pages from another PDF (merge/combine)":
                other_file = st.file_uploader(
                    "Choose the other PDF",
                    type=["pdf"],
                    key="other_pdf_for_insert",
                )
                other_position = st.radio(
                    "Where do its pages go?",
                    ["Insert before this page", "Append at end of document"],
                    index=1,
                )

            if edit_action == "Extract & edit OCR text":
                # Two-step flow: (1) OCR reads the page, (2) the user edits
                # and saves the result as a real text layer (Feature D v1).
                scratch_key = f"ocr_scratch_page_{int(tool_page)}"
                scanned_page = page_needs_ocr(working_pdf, int(tool_page))
                if not scanned_page:
                    st.info(
                        "This page does not appear to be a scanned image "
                        "with little or no selectable text. Use OCR for "
                        "scanned pages and Replace text for native text."
                    )
                elif st.button("1. Read page with OCR"):
                    with st.spinner("Reading the scanned page image with OCR..."):
                        try:
                            ocr_result = ocr_page(working_pdf, int(tool_page))
                        except (RuntimeError, ValueError) as engine_error:
                            st.error(f"OCR problem: {engine_error}")
                            st.stop()
                    st.session_state[scratch_key] = ocr_result.text
                    st.session_state[f"{scratch_key}_words"] = ocr_result.words
                    st.session_state.pop(f"{scratch_key}_font_estimate", None)
                    st.session_state.pop(f"{scratch_key}_font_choice", None)
                    st.rerun()

                if scanned_page and scratch_key in st.session_state:
                    saved_text = st.text_area(
                        "2. Check / edit the extracted text",
                        value=st.session_state[scratch_key],
                        height=220,
                        key=f"ocr_edit_page_{int(tool_page)}",
                    )
                    st.caption(
                        "Edited areas will have a subtle outline; the original "
                        "is preserved in the side-by-side preview."
                    )
                    font_estimate_key = f"{scratch_key}_font_estimate"
                    if st.button(
                        "Estimate local font matches",
                        key=f"{scratch_key}_estimate_fonts",
                    ):
                        with st.spinner(
                            "Comparing OCR lines with locally installed fonts..."
                        ):
                            try:
                                st.session_state[font_estimate_key] = (
                                    cached_scanned_font_estimate(
                                        working_pdf, int(tool_page)
                                    )
                                )
                            except (ValueError, RuntimeError) as font_error:
                                st.error(f"Font estimation problem: {font_error}")

                    selected_font_path = None
                    estimate = st.session_state.get(font_estimate_key)
                    if estimate is not None:
                        st.caption(
                            f"{estimate.score_description} "
                            f"Compared {estimate.candidate_count} local font "
                            f"faces. Page alignment: {estimate.alignment}."
                        )
                        for line_number, line_estimate in enumerate(
                            estimate.lines, start=1
                        ):
                            best_match = line_estimate.matches[0]
                            st.write(
                                f"Line {line_number}: “{line_estimate.text}”"
                            )
                            st.caption(
                                f"Top match: {best_match.family_name} "
                                f"{best_match.style_name} — similarity "
                                f"{best_match.confidence_score:.3f}; "
                                f"estimated size "
                                f"{best_match.estimated_size_points:.1f} pt; "
                                f"average interword spacing "
                                f"{line_estimate.average_interword_spacing_points!s} pt."
                            )
                        if estimate.line_spacing_points:
                            st.caption(
                                "Estimated line spacing (pt): "
                                + ", ".join(
                                    f"{spacing:.1f}"
                                    for spacing in estimate.line_spacing_points
                                )
                            )

                        font_options = {"Use default (Helvetica)": None}
                        for line_estimate in estimate.lines:
                            for match in line_estimate.matches:
                                path = Path(match.font_path)
                                if path.suffix.lower() not in {".ttf", ".otf"}:
                                    continue
                                label = (
                                    f"{match.family_name} — {match.style_name} "
                                    f"(similarity {match.confidence_score:.3f}; "
                                    f"{path.name})"
                                )
                                font_options.setdefault(label, match.font_path)

                        if len(font_options) > 1:
                            font_choice = st.selectbox(
                                "Manual replacement font (optional)",
                                options=list(font_options),
                                key=f"{scratch_key}_font_choice",
                            )
                            selected_font_path = font_options[font_choice]
                            st.caption(
                                "Your choice is applied to changed OCR text only; "
                                "it does not confirm the document's original font. "
                                "Review the marked side-by-side preview after saving."
                            )
                        else:
                            st.info(
                                "No installable TTF/OTF candidate is available "
                                "for manual replacement; the default font will be used."
                            )

                    if st.button("3. Replace scanned text", type="primary"):
                        original_ocr_text = st.session_state[scratch_key]
                        with st.spinner(
                            "Correcting scanned-page text and preparing the preview..."
                        ):
                            try:
                                result = save_text_layer(
                                    working_pdf,
                                    int(tool_page),
                                    saved_text,
                                    ocr_words=st.session_state.get(
                                        f"{scratch_key}_words"
                                    ),
                                    font_path=selected_font_path,
                                )
                                cached_selected_page_preview(
                                    result.pdf_bytes, int(tool_page)
                                )
                                change_summary = "\n".join(
                                    difflib.unified_diff(
                                        original_ocr_text.splitlines(),
                                        saved_text.splitlines(),
                                        fromfile="OCR original",
                                        tofile="Edited text",
                                        lineterm="",
                                    )
                                )
                            except (ValueError, RuntimeError) as layer_error:
                                st.error(f"Save problem: {layer_error}")
                                st.stop()
                        st.session_state.working_pdf = result.pdf_bytes
                        st.session_state.preview_original_pdf = working_pdf
                        st.session_state.preview_change_summary = (
                            change_summary or "No text changes detected."
                        )
                        st.session_state.preview_page = int(tool_page)
                        st.session_state.edit_note = (
                            "Your edited text was applied. Review the page preview "
                            "to confirm it looks right."
                        )
                        st.session_state.pop(scratch_key, None)
                        st.session_state.pop(f"{scratch_key}_words", None)
                        st.rerun()
            elif edit_action == COMMENT_TOOL:
                st.caption(
                    "Click the ✏️ pencil on the page preview to add a "
                    "comment to this page."
                )
            else:
                if st.button("Apply edit", type="primary"):
                    result = None
                    spinner_text = (
                        "Matching the original font and applying the replacement..."
                        if edit_action == "Replace text"
                        else "Editing the working copy..."
                    )
                    with st.spinner(spinner_text):
                        try:
                            if edit_action == "Replace text":
                                if not old_text.strip():
                                    st.warning("Type the text you want to find first.")
                                else:
                                    result = replace_text(
                                        working_pdf, int(tool_page), old_text, new_text
                                    )
                            elif edit_action == "Insert blank page after this one":
                                result = insert_blank_page(working_pdf, int(tool_page))
                            elif (
                                edit_action
                                == "Insert pages from another PDF (merge/combine)"
                            ):
                                if other_file is None:
                                    st.warning("Choose the other PDF first.")
                                else:
                                    before = (
                                        int(tool_page)
                                        if other_position == "Insert before this page"
                                        else None
                                    )
                                    result = insert_pages_from(
                                        working_pdf, other_file.read(), before
                                    )
                        except (ValueError, RuntimeError) as edit_error:
                            st.error(f"Edit problem: {edit_error}")
                            st.stop()

                    if result is not None:
                        st.session_state.working_pdf = result.pdf_bytes
                        st.session_state.edit_note = (
                            f"{result.changes} change(s) applied on page "
                            f"{int(tool_page)} ({result.origin}). "
                        )
                        st.rerun()  # rerun refreshes the pipeline from the new working copy
            # --- "Add a comment" tool: opens the large comment workspace dialog
            # (same presentation as the click preview) where the page preview
            # carries a pencil button. Pencil opens the comment box; the finished
            # comment is dragged to any position on the page and embedded in the
            # working copy.
            if edit_action == COMMENT_TOOL:
                tool_active_key = "active_comment_tool"
                workspace_key = f"comment_workspace_open_{tool_page}"
                # Fresh tool selection auto-opens the workspace; a dismissed dialog
                # keeps it closed until the tool is selected again.
                if st.session_state.get(tool_active_key) != tool_page:
                    st.session_state[tool_active_key] = tool_page
                    st.session_state[workspace_key] = True
                if st.session_state.get(workspace_key, False):
                    show_comment_workspace(int(tool_page))
                elif st.button(
                    "Open comment workspace",
                    key=f"comment_open_{tool_page}",
                    type="primary",
                ):
                    st.session_state[workspace_key] = True
                    st.rerun()

        with workspace_col:
            st.radio(
                "Pages per row",
                [1, 2, 3],
                index=2,  # default: 3 cards per row
                horizontal=True,
                key="per_row",  # remembered across clicks
                label_visibility="collapsed",
            )
            # Natural page scroll (no inner box): the cards take over the page,
            # with the user-chosen number of cards per row. Fewer cards per row
            # = bigger display = higher render DPI, so nothing looks compressed.
            # NOTE: the grid runs inside a @st.fragment so grid events (select /
            # drag reorder) rerun ONLY this block instead of the whole app. That
            # keeps the heavy pipeline (text extraction, embeddings, tool panel)
            # completely out of every grid interaction.
            @st.fragment
            def pages_grid():
                st.subheader("Pages")
                working_pdf_grid = st.session_state.working_pdf
                page_count = int(st.session_state.get("page_count_total", 1))
                tool_page = st.session_state.get("tool_page", 1)
                per_row = max(1, int(st.session_state.get("per_row", 3)))
                DPI_PER_ROW = {1: 200, 2: 150, 3: 120}
                card_dpi = DPI_PER_ROW.get(per_row, 120)

                @st.cache_data
                def cached_page_preview(pdf_bytes: bytes, page_number: int, card_dpi: int):
                    """One plain page render: input to the drag grid."""
                    return render_page_with_boxes(pdf_bytes, page_number, dpi=card_dpi)

                preview_limit = st.session_state.get("preview_limit", 24)
                visible_count = min(page_count, preview_limit)
                preview_numbers = list(range(1, visible_count + 1))

                # What the grid currently displays: the physical page number shown
                # at each displayed position. `page_order` holds this mapping while
                # a drag rearrange has not yet been folded into the working copy,
                # so the thumbnails below stay cache HITS (working copy bytes did
                # not change) and the drop is instant.
                order_map = st.session_state.get("page_order") or list(
                    range(1, page_count + 1)
                )

                # Component key version: every consumed event bumps this counter, so
                # the component gets a fresh state and events never replay forever.
                grid_key = f"drag_grid_{st.session_state.get('grid_clock', 0)}"
                drag_event = drag_grid(
                    page_thumbnails=[
                        cached_page_preview(
                            working_pdf_grid, order_map[number - 1], card_dpi
                        )
                        for number in preview_numbers
                    ],
                    page_numbers=preview_numbers,
                    selected=tool_page,
                    per_row=per_row,
                    key=grid_key,
                )

                if isinstance(drag_event, dict):
                    st.session_state["grid_clock"] = (
                        st.session_state.get("grid_clock", 0) + 1
                    )
                    if drag_event.get("type") == "select":
                        selected_page = int(drag_event["page"])
                        st.session_state.tool_page = selected_page
                        st.session_state.preview_page = selected_page
                        st.rerun()
                    elif drag_event.get("type") == "remove":
                        removed_page = int(drag_event.get("page", 0))
                        # A pending grid order must become real bytes BEFORE the
                        # deletion, so `removed_page` points at the right page.
                        _materialize_pending_page_order()
                        working_pdf_grid = st.session_state.working_pdf
                        try:
                            with st.spinner(f"Removing page {removed_page}..."):
                                result = delete_pages(working_pdf_grid, [removed_page])
                        except ValueError as page_error:
                            st.error(f"Error removing page : {page_error}")
                        else:
                            selected_page = int(st.session_state.get("tool_page", 1))
                            if removed_page < selected_page:
                                selected_page -= 1
                            elif removed_page == selected_page:
                                selected_page = min(removed_page, page_count - 1)
                            st.session_state.tool_page = max(1, selected_page)
                            st.session_state.working_pdf = result.pdf_bytes
                            st.session_state.pop("page_order", None)
                            st.session_state.pop("preview_page", None)
                            st.session_state.pop("preview_original_pdf", None)
                            st.session_state.edit_note = (
                                f"Page {removed_page} removed successfully."
                            )
                            st.rerun()
                    elif drag_event.get("type") == "rotate":
                        rotated_page = int(drag_event.get("page", 0))
                        _materialize_pending_page_order()
                        working_pdf_grid = st.session_state.working_pdf
                        try:
                            with st.spinner(f"Rotating page {rotated_page}..."):
                                result = rotate_page(working_pdf_grid, rotated_page, 90)
                        except ValueError as page_error:
                            st.error(f"Could not rotate page: {page_error}")
                        else:
                            st.session_state.tool_page = rotated_page
                            st.session_state.working_pdf = result.pdf_bytes
                            st.session_state.edit_note = (
                                f"Page {rotated_page} rotated 90° clockwise. "
                                "Successfully."
                            )
                            st.rerun()
                    elif drag_event.get("type") == "order":
                        dropped = [int(n) for n in drag_event.get("order", [])]
                        visible_set = set(preview_numbers)
                        if (
                            sorted(dropped) == sorted(visible_set)
                            and dropped != preview_numbers
                        ):
                            # The user reordered the visible pages among themselves;
                            # pages beyond the preview limit keep their relative
                            # order right after them, so nothing is ever lost.
                            # FAST PATH: only the DISPLAY mapping changes here.
                            # The working-copy bytes are folded on the next full
                            # rerun by _materialize_pending_page_order().
                            tail = [
                                n
                                for n in range(1, page_count + 1)
                                if n not in visible_set
                            ]
                            st.session_state.page_order = [
                                order_map[n - 1] for n in dropped + tail
                            ]
                            st.toast("Pages rearranged by drag and drop.", icon="✅")
                            st.rerun(scope="fragment")

                if page_count > preview_limit:
                    st.button(
                        f"⬇ {visible_count} of {page_count} pages - show more",
                        on_click=lambda: st.session_state.update(
                            preview_limit=st.session_state.get("preview_limit", 24) + 24
                        ),
                    )

            pages_grid()
            if st.session_state.get("edit_note"):
                note = st.session_state.pop("edit_note")
                # Single transient toast - floats above everything (including the
                # open comment dialog) and disappears on its own, so no duplicate
                # success banner lingers in the page flow below.
                st.toast(note, icon="✅")

            fallback_download_name = re.sub(
                r'[<>:"/\\|?*\x00-\x1f]',
                "_",
                f"working-copy-{uploaded_file.name}",
            ).strip().rstrip(".")
            if not fallback_download_name:
                fallback_download_name = "working-copy-document.pdf"

            def _clean_pdf_name(raw_name: str) -> str:
                cleaned = re.sub(
                    r'[<>:"/\\|?*\x00-\x1f]', "_", raw_name
                ).strip().rstrip(".")
                if not cleaned:
                    cleaned = fallback_download_name
                if not cleaned.lower().endswith(".pdf"):
                    cleaned = f"{cleaned}.pdf"
                return cleaned

            # `live=True` commits the name while the user types (after a short
            # pause) instead of waiting for Enter or a click outside the box.
            # Without it, typing a new name and clicking "Download" right away
            # let the button finish its rerun with the OLD committed name, so the
            # downloaded file kept the previous filename (reported bug).
            download_name = _clean_pdf_name(
                st.text_input(
                    "Download filename",
                    value=fallback_download_name,
                    live=True,
                    help="Choose the name for the downloaded PDF.",
                    key="working_copy_filename",
                )
            )

            def _announce_download():
                # Read the LATEST committed widget value (not the closure value),
                # so the toast names the file the user actually downloads.
                latest = _clean_pdf_name(
                    st.session_state.get("working_copy_filename", "")
                )
                st.session_state.edit_note = f"Downloaded {latest} successfully."

            st.download_button(
                "⬇️ Download working copy",
                data=st.session_state.working_pdf,
                file_name=download_name,
                mime="application/pdf",
                # The download runs client-side; this callback re-arms the shared
                # note so the next run pops it as a toast + success confirmation.
                on_click=_announce_download,
            )

            # --- Document quality check (kept as a collapsed tool) -----------------
            with st.expander("🩺 Quality check"):
                if st.button("Run quality check"):
                    with st.spinner("Checking every page..."):
                        try:
                            report = cached_quality_report(working_pdf)
                        except ValueError as error:
                            st.error(f"Could not analyze this document: {error}")
                            st.stop()

                    confidence_label = (
                        f"{report.average_confidence}"
                        if report.average_confidence is not None
                        else "— (no OCR needed)"
                    )
                    top1, top2, top3, top4 = st.columns(4)
                    top1.metric("Native text pages", report.native_pages)
                    top2.metric("Scanned (OCR) pages", report.scanned_pages)
                    top3.metric("Unreadable pages", report.empty_pages)
                    top4.metric("Avg OCR confidence", confidence_label)

                    st.dataframe(
                        [
                            {
                                "Page": page.page_number,
                                "Type": page.page_type,
                                "Native chars": page.native_chars,
                                "OCR chars": page.ocr_chars,
                                "OCR confidence": page.ocr_confidence,
                            }
                            for page in report.pages
                        ],
                        hide_index=True,
                    )

                    if report.empty_pages:
                        st.warning(
                            f"{report.empty_pages} page(s) are effectively unreadable - "
                            "the AI cannot answer questions from those pages. Editing "
                            "or re-scanning them will help."
                        )


# --- Tab 3: Ask (RAG + find + transparency) --------------------------------------
with tab_ask:
    if uploaded_file is None:
        st.info("Upload a PDF in the Upload tab first.")
    elif not embedded_chunks:
        st.warning(
            "Nothing is searchable yet - check the Edit tab's quality report. "
            "Scanned pages may still have been read via OCR."
        )
    else:
        if pages_without_text:
            st.caption(f"Pages without extractable text: {pages_without_text}")

        # --- Answer engine (local Ollama or the user's own API key) -----------
        # Phase 1.3: the local Ollama may live behind the secure Cloudflare
        # Tunnel + auth proxy (OLLAMA_BASE_URL / OLLAMA_API_TOKEN secrets); or
        # the user can answer with their own hosted API key (BYOK).
        st.subheader("❓ Ask about the document")
        engine = st.radio(
            "Answer engine",
            ["Local Ollama model", "Bring your own API key (BYOK)"],
            horizontal=True,
            help="Local Ollama: your machine's private model. BYOK: a hosted "
            "provider you already pay for - the key stays in memory for this "
            "session only and is never stored or logged.",
        )

        if engine.startswith("Bring"):
            provider_name = st.selectbox("Provider", list(SUPPORTED_PROVIDERS))
            default_base = SUPPORTED_PROVIDERS[provider_name]
            base_url = st.text_input(
                "API base URL",
                value=default_base,
                disabled=bool(default_base),
                help="Pre-filled for known providers; paste your own URL for "
                "'Custom (OpenAI-compatible)'.",
            )
            api_key = st.text_input("API key", type="password")
            model_name = st.text_input(
                "Model name",
                value=DEFAULT_MODELS.get(provider_name, ""),
                placeholder="e.g. gpt-4o-mini",
            )

        question = st.text_input(
            "Your question",
            placeholder="e.g. How often should the filter be cleaned?",
        )

        if st.button("Ask", type="primary"):
            if not question.strip():
                st.warning("Please type a question first.")
            else:
                if engine.startswith("Bring"):
                    try:
                        ask_fn = ask_openai_compatible(
                            api_key=api_key,
                            model=model_name,
                            base_url=base_url,
                        )
                    except ValueError as setup_error:
                        st.warning(str(setup_error))
                        st.stop()
                else:
                    ask_fn = ask_ollama

                with st.spinner("Searching the document and thinking..."):
                    try:
                        bundle = answer_question(
                            question, embedded_chunks, ask=ask_fn
                        )
                    except RuntimeError as server_error:
                        # ollama/BYOK connection problems carry friendly
                        # messages already.
                        st.error(f"Model problem: {server_error}")
                        st.stop()

                if bundle.not_found:
                    hint = REASON_HELP.get(bundle.reason, "")
                    st.warning(f"**{bundle.answer}**\n\nReason: {hint}")

                    # Stage 5D (part c): SHOW why it was refused instead of
                    # leaving the user guessing. Closest passages + scores.
                    closest = search_chunks(question, embedded_chunks, top_k=3)
                    if closest:
                        st.info(
                            f"The best-matching passage scored "
                            f"{closest[0].score:.2f} similarity; answers need "
                            f"similarity **{DEFAULT_MIN_SCORE:.2f}** or passages "
                            f"containing the question's key terms."
                        )
                        with st.expander("🧠 See the closest passages the AI found"):
                            for number, result in enumerate(closest, start=1):
                                st.markdown(
                                    f"[{number}] page {result.page_number} — "
                                    f"similarity **{result.score:.2f}**"
                                )
                                st.text(result.text)
                else:
                    st.success(bundle.answer)

                    st.subheader("📎 Sources")
                    st.caption(
                        "These passages are EXACTLY what the model read before "
                        "answering (Stage 5D transparency); citations are "
                        "attached by our code, never by the model."
                    )
                    for number, source in enumerate(bundle.sources, start=1):
                        with st.expander(
                            f"[{number}] {source.document_id} - page "
                            f"{source.page_number} "
                            f"(similarity {source.score:.2f})"
                        ):
                            st.markdown(
                                f"**Characters {source.start_char}-"
                                f"{source.end_char}**"
                            )
                            st.text(source.text)

        # --- Find in document (works on scanned pages too - Stage 5E) -----------
        st.subheader("🔍 Find in document")
        find_query = st.text_input(
            "Word or phrase",
            placeholder="e.g. pressure gauge",
            help="Case-insensitive. Works on scanned pages as well (OCR) - "
            "first search on a scanned page can take a few seconds per page.",
        )

        if st.button("Find"):
            if not find_query.strip():
                st.warning("Please type a word or phrase to search for.")
            else:
                with st.spinner("Searching every page (scanned pages go through OCR)..."):
                    try:
                        found = find_word(working_pdf, find_query)
                    except RuntimeError as engine_error:
                        st.error(f"Search problem: {engine_error}")
                        st.stop()

                if not found:
                    st.warning(f"Nothing found for '{find_query}' in this document.")
                else:
                    st.success(f"{len(found)} match(es) for '{find_query}':")

                    # Group matches by page so each page is drawn ONCE with all
                    # of its highlights (Stage 5.5: real highlighting).
                    matches_by_page: dict[int, list] = {}
                    for item in found:
                        matches_by_page.setdefault(item.page_number, []).append(item)

                    for page_number, items in sorted(matches_by_page.items()):
                        with st.expander(
                            f"📄 Page {page_number} — {len(items)} match(es)"
                        ):
                            try:
                                png = render_page_with_boxes(
                                    working_pdf,
                                    page_number,
                                    boxes=[
                                        (item.left, item.top, item.width, item.height)
                                        for item in items
                                    ],
                                )
                                st.image(
                                    png,
                                    caption=f"Matches highlighted on page {page_number}",
                                )
                            except ValueError as render_error:
                                st.error(f"Could not draw this page: {render_error}")

                            for index, item in enumerate(items, start=1):
                                source_label = (
                                    f"scanned page (OCR)"
                                    f"{f', confidence {item.confidence}' if item.confidence else ''}"
                                    if item.origin == "ocr"
                                    else "native text"
                                )
                                st.markdown(
                                    f"{index}. **…{item.cover_text}…** — {source_label}"
                                )

        # --- Page-by-page text preview (kept from Stage 2) ------------------------
        with st.expander("🗂 Document text, page by page"):
            for page in pages:
                if page.has_text:
                    st.markdown(
                        f"**Page {page.page_number}** — {len(page.text)} characters"
                    )
                    st.text(page.text)
                else:
                    st.markdown(f"*Page {page.page_number}: no extractable text*")
