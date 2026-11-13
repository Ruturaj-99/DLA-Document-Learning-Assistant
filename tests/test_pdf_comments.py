"""Tests for comments embedded in PDF working copies."""

import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_comments import (  # noqa: E402
    _raw_display_rect,
    add_pdf_comment,
    delete_pdf_comment,
    list_pdf_comments,
    move_pdf_comment,
    update_pdf_comment,
)
from src.pdf_editor import apply_page_order  # noqa: E402


def _build_pdf(rotated: bool = False) -> bytes:
    document = pymupdf.open()
    first = document.new_page(width=612, height=792)
    first.insert_text((72, 200), "first page")
    document.new_page(width=612, height=792).insert_text((72, 200), "second page")
    if rotated:
        document[0].set_rotation(90)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_add_comment_embeds_text_position_and_marker_highlight():
    original = _build_pdf()
    result = add_pdf_comment(
        original,
        1,
        0.25,
        0.4,
        "Check this paragraph",
        highlighted=True,
        color="#39A6FF",
    )

    comments = list_pdf_comments(result)
    assert len(comments) == 1
    assert comments[0].page_number == 1
    assert comments[0].text == "Check this paragraph"
    assert comments[0].x == pytest.approx(0.25, abs=1e-4)
    assert comments[0].y == pytest.approx(0.4, abs=1e-4)
    assert comments[0].highlighted is True
    assert comments[0].color == "#39A6FF"
    document = pymupdf.open(stream=result, filetype="pdf")
    try:
        page = document[0]
        annotation = next(page.annots())
        assert annotation.type[1] == "FreeText"
        assert annotation.info["content"] == "Check this paragraph"
        assert annotation.colors["stroke"] == pytest.approx(
            (0.224, 0.651, 1.0), abs=0.005
        )
        pixmap = page.get_pixmap()
        highlight_pixel = pixmap.pixel(
            round(annotation.rect.x1 - 5), round(annotation.rect.y0 + 3)
        )
        assert highlight_pixel[2] > 220
    finally:
        document.close()
    assert list_pdf_comments(original) == []


def test_unhighlighted_comment_uses_neutral_marker_color():
    pdf_bytes = add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Neutral marker")

    comment = list_pdf_comments(pdf_bytes)[0]

    assert comment.highlighted is False
    assert comment.color == "#D9D9D9"


def test_add_comment_defaults_to_plain_text_no_fill():
    """Default comments are plain text: no highlight fill, 11 pt."""
    pdf_bytes = add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Plain comment")

    comment = list_pdf_comments(pdf_bytes)[0]
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        annotation = next(document[0].annots())
        assert comment.font_size == pytest.approx(11.0)
        assert not annotation.colors["fill"]  # no highlight fill
    finally:
        document.close()


def test_font_size_round_trips_and_scales_the_text():
    pdf_bytes = add_pdf_comment(
        _build_pdf(), 1, 0.2, 0.3, "Big text", font_size=24
    )

    comment = list_pdf_comments(pdf_bytes)[0]
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        annotation = next(document[0].annots())
        assert comment.font_size == pytest.approx(24.0)
        assert _raw_display_rect(document[0], annotation).height > 24
    finally:
        document.close()


def test_font_size_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="between 4 and 72"):
        add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Too big", font_size=73)
    with pytest.raises(ValueError, match="between 4 and 72"):
        add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Too small", font_size=3)


def test_update_can_restyle_highlight_and_size():
    original = add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Restyle me")
    comment_id = list_pdf_comments(original)[0].comment_id

    # A v1-style no-size comment keeps its stored look unless restyled.
    updated = update_pdf_comment(
        original, comment_id, "Restyle me", highlighted=True, font_size=18
    )
    styled = list_pdf_comments(updated)[0]
    assert styled.highlighted is True
    assert styled.font_size == pytest.approx(18.0)

    # None keeps the just-stored style.
    kept = update_pdf_comment(updated, comment_id, "Restyle me")
    assert list_pdf_comments(kept)[0].font_size == pytest.approx(18.0)
    assert list_pdf_comments(kept)[0].highlighted is True


def test_legacy_v1_comment_still_reads_with_default_size():
    """Comments written before per-comment sizes parse as 11 pt."""
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 200), "legacy")
    annot = page.add_freetext_annot(
        pymupdf.Rect(72, 200, 200, 220), "Legacy comment", fontname="Helv"
    )
    annot.set_info(
        title="AI DLA comment",
        subject="ai-dla-comment:v1:abcd1234abcd1234abcd1234abcd1234:1:FFD54F",
    )
    annot.update()
    pdf_bytes = document.tobytes()
    document.close()

    comment = list_pdf_comments(pdf_bytes)[0]

    assert comment.comment_id == "abcd1234abcd1234abcd1234abcd1234"
    assert comment.highlighted is True
    assert comment.color == "#FFD54F"
    assert comment.font_size == pytest.approx(11.0)


def test_update_with_invalid_font_size_is_rejected():
    original = add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Sized")
    comment_id = list_pdf_comments(original)[0].comment_id

    with pytest.raises(ValueError, match="between 4 and 72"):
        update_pdf_comment(original, comment_id, "Sized", font_size=100)


def test_move_comment_changes_saved_position_and_keeps_comment():
    pdf_bytes = add_pdf_comment(
        _build_pdf(),
        1,
        0.2,
        0.3,
        "Move this note",
        highlighted=True,
        color="#FFD54F",
    )
    comment = list_pdf_comments(pdf_bytes)[0]

    moved_pdf = move_pdf_comment(pdf_bytes, comment.comment_id, 0.65, 0.7)

    moved = list_pdf_comments(moved_pdf)[0]
    assert moved.comment_id == comment.comment_id
    assert moved.text == "Move this note"
    assert moved.x == pytest.approx(0.65, abs=1e-4)
    assert moved.y == pytest.approx(0.7, abs=1e-4)
    assert moved.highlighted is True
    assert moved.color == "#FFD54F"
    assert (comment.x, comment.y) == pytest.approx((0.2, 0.3), abs=1e-4)


def test_delete_comment_removes_only_selected_comment():
    pdf_bytes = add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Keep")
    pdf_bytes = add_pdf_comment(pdf_bytes, 2, 0.4, 0.5, "Remove")
    comments = list_pdf_comments(pdf_bytes)

    result = delete_pdf_comment(pdf_bytes, comments[1].comment_id)

    remaining = list_pdf_comments(result)
    assert [(comment.page_number, comment.text) for comment in remaining] == [
        (1, "Keep")
    ]


def test_comments_follow_pages_when_reordered():
    pdf_bytes = add_pdf_comment(_build_pdf(), 1, 0.3, 0.4, "First page note")
    pdf_bytes = apply_page_order(pdf_bytes, [2, 1]).pdf_bytes

    comments = list_pdf_comments(pdf_bytes)

    assert len(comments) == 1
    assert comments[0].page_number == 2
    assert comments[0].text == "First page note"


def test_position_round_trips_on_rotated_page():
    pdf_bytes = add_pdf_comment(_build_pdf(rotated=True), 1, 0.2, 0.3, "Rotated note")

    comment = list_pdf_comments(pdf_bytes, page_number=1)[0]

    assert comment.x == pytest.approx(0.2, abs=1e-4)
    assert comment.y == pytest.approx(0.3, abs=1e-4)
    moved_pdf = move_pdf_comment(
        pdf_bytes, comment.comment_id, 0.65, 0.7
    )
    moved = list_pdf_comments(moved_pdf, page_number=1)[0]
    assert moved.x == pytest.approx(0.65, abs=1e-4)
    assert moved.y == pytest.approx(0.7, abs=1e-4)


def test_invalid_comment_input_is_rejected():
    original = _build_pdf()
    with pytest.raises(ValueError, match="Enter comment text"):
        add_pdf_comment(original, 1, 0.2, 0.3, "  ")
    with pytest.raises(ValueError, match="within the page"):
        add_pdf_comment(original, 1, 1.1, 0.3, "Outside")
    with pytest.raises(ValueError, match="hex color"):
        add_pdf_comment(original, 1, 0.2, 0.3, "Bad color", color="blue")
    with pytest.raises(ValueError, match="no longer exists"):
        move_pdf_comment(original, "missing", 0.5, 0.5)


def test_update_keeps_identity_and_changes_text_in_place():
    pdf_bytes = add_pdf_comment(
        _build_pdf(),
        1,
        0.2,
        0.3,
        "Old text",
        highlighted=True,
        color="#FFD54F",
    )
    comment = list_pdf_comments(pdf_bytes)[0]

    updated_pdf = update_pdf_comment(pdf_bytes, comment.comment_id, "New text")

    [updated] = list_pdf_comments(updated_pdf)
    assert updated.comment_id == comment.comment_id
    assert updated.text == "New text"
    assert (updated.x, updated.y) == pytest.approx((comment.x, comment.y), abs=1e-4)
    assert updated.highlighted is True
    assert updated.color == "#FFD54F"


def test_update_with_position_relocates_the_comment():
    pdf_bytes = add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Old text")
    comment = list_pdf_comments(pdf_bytes)[0]

    moved_pdf = update_pdf_comment(
        pdf_bytes, comment.comment_id, "Relocated", 0.6, 0.4
    )

    [moved] = list_pdf_comments(moved_pdf)
    assert moved.comment_id == comment.comment_id
    assert moved.text == "Relocated"
    assert (moved.x, moved.y) == pytest.approx((0.6, 0.4), abs=1e-4)


def test_update_rejects_bad_text_and_missing_comment():
    pdf_bytes = add_pdf_comment(_build_pdf(), 1, 0.2, 0.3, "Keep")
    comment = list_pdf_comments(pdf_bytes)[0]
    with pytest.raises(ValueError, match="Enter comment text"):
        update_pdf_comment(pdf_bytes, comment.comment_id, "  ")
    with pytest.raises(ValueError, match="no longer exists"):
        update_pdf_comment(pdf_bytes, "missing", "Any text")


def test_duplicate_drop_of_same_text_merges_into_single_layer():
    """A doubled place gesture must not stack the same comment twice."""
    pdf_bytes = add_pdf_comment(
        _build_pdf(),
        1,
        0.35,
        0.30,
        "Add some comment and incorporated with the same",
        highlighted=True,
        color="#FFD54F",
    )

    # Same text, virtually the same anchor (e.g. the event fired twice).
    merged = add_pdf_comment(pdf_bytes, 1, 0.351, 0.301, "Add some comment and incorporated with the same")
    comments = list_pdf_comments(merged)
    assert len(comments) == 1
    assert comments[0].x == pytest.approx(0.35, abs=1e-4)

    # The same text at a clearly different anchor is still a distinct comment.
    second = add_pdf_comment(merged, 1, 0.6, 0.6, "Add some comment and incorporated with the same")
    assert len(list_pdf_comments(second)) == 2
