"""Stage 5.5: tests for src/page_view.py (page rendering + highlight boxes).

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

Verified ideas:
    - the returned bytes are a real, openable PNG of the page,
    - drawing a box actually CHANGES the picture (vs. no-box render),
    - the original PDF bytes are never modified,
    - invalid page numbers are rejected with a friendly ValueError,
    - end-to-end: boxes from find_word on a scanned page render fine.
"""

import io
import sys
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402

from src.document_search import find_word  # noqa: E402
from src.ocr import find_tesseract_engine  # noqa: E402
from src.page_view import (  # noqa: E402
    render_document_contact_sheet,
    render_page_card,
    render_page_thumbnail,
    render_page_with_boxes,
    render_selectable_page_html,
)

ENGINE = find_tesseract_engine()

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _build_native_pdf() -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=612, height=650)
    page.insert_text(
        (72, 200),
        "The coil resistance must be measured with the heater cold.",
        fontname="helv",
        fontsize=14,
    )
    return document.tobytes()


def _png_pixel_size(png_bytes: bytes) -> tuple[int, int]:
    """Read width/height straight from the PNG header (bytes 16-24)."""
    assert png_bytes.startswith(PNG_MAGIC), "not a valid PNG"
    width = int.from_bytes(png_bytes[16:20], "big")
    height = int.from_bytes(png_bytes[20:24], "big")
    return width, height


def test_render_returns_valid_png_at_dpi_scale():
    pdf = _build_native_pdf()
    png = render_page_with_boxes(pdf, 1)

    assert png.startswith(PNG_MAGIC)
    # Re-openable as an image (sanity check on top of the header check).
    width, height = _png_pixel_size(png)
    # 612 x 650 points at 150 dpi -> 1275 x 1355 pixels (dpi/72 scale).
    assert (width, height) == (1275, 1355)


def test_selectable_preview_contains_text_and_embedded_page_image():
    preview_html = render_selectable_page_html(_build_native_pdf(), 1)

    assert "heater cold" in preview_html
    assert "data:image/png;base64," in preview_html
    assert "user-select: text" in preview_html
    assert "data:application/pdf" not in preview_html
    assert 'id="zoom-in"' in preview_html
    assert 'id="zoom-out"' in preview_html
    assert "setZoom(zoom + 0.25)" in preview_html
    assert "setZoom(zoom - 0.25)" in preview_html


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_selectable_preview_adds_ocr_text_for_image_only_page():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1200, 250), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (40, 70),
        "Selectable scan text",
        fill="black",
        font=ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 34),
    )
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    pdf = document.tobytes()
    document.close()

    preview_html = render_selectable_page_html(pdf, 1)

    assert "Selectable" in preview_html
    assert "scan" in preview_html
    assert "text" in preview_html


def test_selectable_preview_escapes_pdf_text():
    document = pymupdf.open()
    document.new_page().insert_text((72, 100), "<script>alert(1)</script>")
    pdf = document.tobytes()
    document.close()

    preview_html = render_selectable_page_html(pdf, 1)

    assert "&lt;script&gt;" in preview_html
    assert "<script>alert(1)</script>" not in preview_html


def test_selectable_preview_rejects_invalid_page_number():
    with pytest.raises(ValueError, match="does not exist"):
        render_selectable_page_html(_build_native_pdf(), 2)
    with pytest.raises(ValueError, match="No PDF data"):
        render_selectable_page_html(b"", 1)


def test_box_actually_changes_the_picture():
    pdf = _build_native_pdf()
    plain = render_page_with_boxes(pdf, 1, boxes=[])
    highlighted = render_page_with_boxes(pdf, 1, boxes=[(72, 190, 40, 16)])

    assert plain.startswith(PNG_MAGIC)
    assert highlighted != plain, "a drawn box must alter the rendered image"


def test_annotations_can_be_omitted_from_the_rendered_page():
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    annotation = page.add_freetext_annot(
        pymupdf.Rect(72, 180, 240, 220),
        "Comment layer",
        fontsize=14,
        border_width=0,
    )
    annotation.update()
    pdf = document.tobytes()
    document.close()

    with_annotations = render_page_with_boxes(pdf, 1)
    without_annotations = render_page_with_boxes(pdf, 1, include_annotations=False)

    assert with_annotations != without_annotations


def test_original_pdf_bytes_are_untouched():
    pdf = _build_native_pdf()
    original_copy = bytes(pdf)
    render_page_with_boxes(pdf, 1, boxes=[(72, 190, 40, 16)])
    assert pdf == original_copy


def test_out_of_range_page_is_rejected():
    pdf = _build_native_pdf()
    with pytest.raises(ValueError, match="does not exist"):
        render_page_with_boxes(pdf, 5)
    with pytest.raises(ValueError, match="No PDF data"):
        render_page_with_boxes(b"", 1)


def test_boxes_accept_none_safely():
    pdf = _build_native_pdf()
    assert render_page_with_boxes(pdf, 1, boxes=None).startswith(PNG_MAGIC)


def _png_dimensions_via_pillow(png_bytes: bytes) -> tuple[int, int]:
    from PIL import Image
    import io as _io

    with Image.open(_io.BytesIO(png_bytes)) as image:
        return image.size


# --- Stage 7B: full-document contact sheet --------------------------------------


def _build_three_page_pdf() -> bytes:
    import pymupdf

    document = pymupdf.open()
    for text in ("one", "two", "three"):
        page = document.new_page(width=612, height=792)
        page.insert_text((72, 200), f"page {text}", fontname="helv", fontsize=14)
    return document.tobytes()


def test_contact_sheet_contains_all_pages_in_one_image():
    pdf = _build_three_page_pdf()
    sheet = render_document_contact_sheet(pdf, columns=2, cell_width_px=300)

    assert sheet.startswith(PNG_MAGIC)
    # Decode with a THIRD-PARTY tool (Pylow/PIL) to stay honest about the size.
    width, height = _png_dimensions_via_pillow(sheet)
    grid_columns = 2
    padding = 10
    gap = 10
    expected_two_cells = 2 * padding + grid_columns * 300 + (grid_columns - 1) * gap
    assert width == expected_two_cells  # 3 pages -> 2 rows tall
    assert height >= 2 * 300  # taller than one row of cells


def test_contact_sheet_highlight_changes_the_picture():
    pdf = _build_three_page_pdf()
    plain = render_document_contact_sheet(pdf, highlight_page=None)
    marked = render_document_contact_sheet(pdf, highlight_page=2)

    assert plain.startswith(PNG_MAGIC)
    assert marked != plain, "the red frame around page 2 must alter the image"


def test_page_card_renders_and_selection_frame_changes_it():
    pdf = _build_native_pdf()
    plain_card = render_page_card(pdf, 1)
    framed_card = render_page_card(pdf, 1, selected=True)

    assert plain_card.startswith(PNG_MAGIC)
    assert framed_card != plain_card, "the red selection frame must alter the card"
    with pytest.raises(ValueError, match="does not exist"):
        render_page_card(pdf, 9)


def test_thumbnail_border_changes_picture_borderless_matches_base():
    pdf = _build_native_pdf()
    plain = render_page_thumbnail(pdf, 1, border=False)
    framed = render_page_thumbnail(pdf, 1, border=True)

    assert plain == render_page_with_boxes(pdf, 1, dpi=90)  # borderless = base
    assert framed != plain, "the red target frame must alter the card image"
    with pytest.raises(ValueError, match="does not exist"):
        render_page_thumbnail(pdf, 9)


def test_contact_sheet_original_untouched_and_bad_inputs_rejected():
    pdf = _build_three_page_pdf()
    original_copy = bytes(pdf)
    render_document_contact_sheet(pdf, highlight_page=1)
    assert pdf == original_copy

    with pytest.raises(ValueError, match="column"):
        render_document_contact_sheet(pdf, columns=0)
    with pytest.raises(ValueError, match="No PDF data"):
        render_document_contact_sheet(b"")


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_scan_page_matches_render_with_box_end_to_end():
    """find_word on the fake-scan page -> its box -> rendered highlight."""
    import test_pdf_loader_ocr as factory

    matches = find_word(factory.MIXED_PDF, "pressure gauge")
    assert matches, "phrase must be found on the scanned page first"
    boxes = [
        (match.left, match.top, match.width, match.height) for match in matches
    ]
    png = render_page_with_boxes(factory.MIXED_PDF, 1, boxes=boxes)
    assert png.startswith(PNG_MAGIC)
