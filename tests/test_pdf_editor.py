"""Stage 6A: tests for src/pdf_editor.py (edit on a separate working copy).

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

Key ideas under test:
    - the original bytes are NEVER modified (copies only),
    - page structure changes (delete / rotate) via pypdf,
    - text replacement on native pages (real redact + re-insert),
    - text replacement on scanned pages (white-out OCR words + real text
      layer on top) - only when the Tesseract engine is installed.
"""

import io
import sys
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader  # noqa: E402

from src.ocr import find_tesseract_engine, page_needs_ocr  # noqa: E402
from src.pdf_editor import (  # noqa: E402
    apply_page_order,
    delete_pages,
    insert_blank_page,
    insert_pages_from,
    move_page,
    replace_text,
    rotate_page,
    save_text_layer,
)
from src.pdf_loader import extract_pdf_pages  # noqa: E402

ENGINE = find_tesseract_engine()


def _build_two_page_native_pdf() -> bytes:
    """Two pages with REAL vector text (no OCR involved anywhere)."""
    import pymupdf

    document = pymupdf.open()
    first = document.new_page(width=612, height=792)
    first.insert_text((72, 200), "alpha beta gamma", fontname="helv", fontsize=14)
    second = document.new_page(width=612, height=792)
    second.insert_text((72, 200), "delta epsilon", fontname="helv", fontsize=14)
    return document.tobytes()


def _build_image_only_pdf() -> bytes:
    import pymupdf
    from PIL import Image

    image_buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(image_buffer, format="PNG")
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    return document.tobytes()


def test_page_needs_ocr_distinguishes_native_text_from_image_page():
    assert page_needs_ocr(_build_two_page_native_pdf(), 1) is False
    assert page_needs_ocr(_build_image_only_pdf(), 1) is True


def test_save_text_layer_rejects_native_text_page():
    with pytest.raises(ValueError, match="not appear to be a scanned/image-only"):
        save_text_layer(_build_two_page_native_pdf(), 1, "Corrected text")


def test_delete_pages_removes_only_chosen_pages():
    original = _build_two_page_native_pdf()
    result = delete_pages(original, [1])

    assert result.changes == 1 and result.origin == "structure"
    remaining = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert len(remaining) == 1
    assert "delta" in remaining[0].text
    # THE golden rule: the original bytes still hold BOTH pages.
    assert len(extract_pdf_pages(original, enable_ocr=False)) == 2


def test_deleting_every_page_is_rejected():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="at least one page"):
        delete_pages(original, [1, 2])


def test_delete_out_of_range_page_is_rejected():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="does not exist"):
        delete_pages(original, [7])


def test_rotate_page_sets_rotation_flag():
    original = _build_two_page_native_pdf()
    result = rotate_page(original, 2, 90)

    assert result.changes == 1
    reader = PdfReader(io.BytesIO(result.pdf_bytes))
    assert reader.pages[1].rotation == 90
    # Original untouched: its page 2 still has rotation 0.
    original_reader = PdfReader(io.BytesIO(original))
    assert original_reader.pages[1].rotation == 0


def test_replace_text_native_page_updates_text_layer():
    original = _build_two_page_native_pdf()
    result = replace_text(original, 1, "gamma", "omega")

    assert result.changes == 1 and result.origin == "native text"
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert "omega" in pages[0].text
    assert "gamma" not in pages[0].text
    # Original untouched.
    assert "gamma" in extract_pdf_pages(original, enable_ocr=False)[0].text


def test_replace_text_native_page_preserves_font_size_weight_and_color():
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text(
        (72, 200),
        "Original",
        fontname="hebo",
        fontsize=22,
        color=(0.8, 0.1, 0.2),
    )
    original = document.tobytes()
    document.close()

    result = replace_text(original, 1, "Original", "Replacement")

    with pymupdf.open(stream=result.pdf_bytes, filetype="pdf") as edited:
        spans = [
            span
            for block in edited[0].get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
            if "Replacement" in span["text"]
        ]
    assert len(spans) == 1
    assert spans[0]["size"] == pytest.approx(22, abs=0.1)
    assert spans[0]["flags"] & 16
    assert spans[0]["color"] == 0xCC1A33


def test_replace_text_native_page_reuses_registered_font_resource():
    import pymupdf

    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    if not font_path.is_file():
        pytest.skip("Arial TTF is not installed on this machine.")
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text(
        (72, 200),
        "Original",
        fontname="source_font",
        fontfile=str(font_path),
        fontsize=19,
    )
    original = document.tobytes()
    document.close()

    result = replace_text(original, 1, "Original", "Replacement")

    with pymupdf.open(stream=result.pdf_bytes, filetype="pdf") as edited:
        spans = [
            span
            for block in edited[0].get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
            if "Replacement" in span["text"]
        ]
    assert len(spans) == 1
    assert "Arial" in spans[0]["font"]
    assert spans[0]["size"] == pytest.approx(19, abs=0.1)


def test_replace_text_with_empty_string_deletes_word():
    original = _build_two_page_native_pdf()
    result = replace_text(original, 1, "gamma", "")

    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert "gamma" not in pages[0].text
    assert "alpha" in pages[0].text  # the rest survived


def _build_letterhead_pdf() -> bytes:
    """One line of letterhead: 'RUTURAJ HEAVY' printed right before
    'ELECTRICALS LTD' - the layout from the overlap bug report."""
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    first = "RUTURAJ HEAVY"
    # Start the second span exactly where the first one ends (+8pt of space).
    neighbour_x = 72 + pymupdf.Font("hebo").text_length(first, 22) + 8
    page.insert_text((72, 200), first, fontname="hebo", fontsize=22)
    page.insert_text(
        (neighbour_x, 200), "ELECTRICALS LTD", fontname="helv", fontsize=22
    )
    payload = document.tobytes()
    document.close()
    return payload


def _page_spans(pdf_bytes: bytes) -> list[dict]:
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        return [
            span
            for block in document[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
        ]


def test_replace_text_native_page_shrinks_instead_of_overlapping_next_word():
    """A longer replacement must end BEFORE the neighbouring word, never over it."""
    original = _build_letterhead_pdf()
    long_replacement = (
        "RUTURAJ HEAVY ENGINEERING WORKS AND INDUSTRIES PRIVATE LIMITED"
    )

    result = replace_text(original, 1, "RUTURAJ HEAVY", long_replacement)

    spans = _page_spans(result.pdf_bytes)
    new_span = next(s for s in spans if "ENGINEERING WORKS" in s["text"])
    neighbour = next(s for s in spans if "ELECTRICALS" in s["text"])
    new_bbox, neighbour_bbox = new_span["bbox"], neighbour["bbox"]

    # No horizontal collision with the word printed next to it.
    assert new_bbox[2] <= neighbour_bbox[0] + 0.5
    # It shrank to fit the gap (instead of growing over the neighbour)...
    assert 4.0 <= new_span["size"] < 22
    # ...and the neighbour itself is untouched.
    assert neighbour["size"] == pytest.approx(22, abs=0.1)
    assert new_bbox[1] < neighbour_bbox[3] and new_bbox[3] > neighbour_bbox[1]


def test_replace_text_native_page_keeps_original_size_when_it_fits():
    """When the new text fits, its font size matches the surrounding text."""
    original = _build_letterhead_pdf()

    result = replace_text(original, 1, "RUTURAJ HEAVY", "RUTURAJ CORP")

    spans = _page_spans(result.pdf_bytes)
    new_span = next(s for s in spans if "RUTURAJ CORP" in s["text"])
    neighbour = next(s for s in spans if "ELECTRICALS" in s["text"])

    assert new_span["size"] == pytest.approx(22, abs=0.1)  # same as neighbour
    assert new_span["flags"] & 16  # still bold, like the original
    assert new_span["bbox"][2] <= neighbour["bbox"][0] + 0.5  # no overlap


def test_replace_text_native_page_rejects_text_that_cannot_fit():
    original = _build_letterhead_pdf()

    with pytest.raises(ValueError, match="overlap|shorter"):
        replace_text(original, 1, "RUTURAJ HEAVY", "X" * 150)

    # The original page was never modified by the rejected edit.
    assert "RUTURAJ HEAVY" in extract_pdf_pages(original, enable_ocr=False)[0].text


def test_missing_text_on_native_page_is_rejected():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="was not found"):
        replace_text(original, 1, "hedgehog", "anything")


def test_empty_search_text_is_rejected():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="find"):
        replace_text(original, 1, "   ", "anything")


# --- Scanned-page editing (needs the OCR engine) -------------------------------


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_replace_text_on_scanned_page_creates_real_text():
    """The word lives in the PAGE 1 IMAGE; the edit must white it out and put
    real text on top. After the edit, plain extraction (NO OCR) already sees
    the new word - the page now behaves like native text."""
    import test_pdf_loader_ocr as factory

    result = replace_text(factory.MIXED_PDF, 1, "gauge", "valve")

    assert result.origin == "scanned page (OCR)"
    assert result.changes == 1
    edited_pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert "valve" in edited_pages[0].text
    # Page 2 (native) survived untouched.
    assert "heater" in edited_pages[1].text


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_missing_text_on_scanned_page_is_rejected():
    import test_pdf_loader_ocr as factory

    with pytest.raises(ValueError, match="was not found"):
        replace_text(factory.MIXED_PDF, 1, "hedgehog", "anything")


# --- Scan-backed pages: page image + invisible OCR text layer -----------------


def _build_scan_backed_pdf() -> bytes:
    """A scanned page WITH an invisible-style OCR text layer on top.

    The words exist twice: burned into the page image (what the eye sees)
    and as a real text layer (what search/the AI reads) - the exact situation
    where a plain redaction left the old text visible and the replacement
    overlapped it.
    """
    import pymupdf
    from PIL import Image, ImageDraw

    image_buffer = io.BytesIO()
    image = Image.new("RGB", (612, 792), "white")
    ImageDraw.Draw(image).text(
        (72, 186), "alpha beta gamma", fill="black"
    )  # burned into the image, baseline ~= 200
    image.save(image_buffer, format="PNG")
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    page.insert_text((72, 200), "alpha beta gamma", fontname="helv", fontsize=14)
    return document.tobytes()


def _dark_pixel_count(pdf_bytes: bytes, rect) -> int:
    """Count dark pixels inside `rect` (pymupdf top-left coordinates)."""
    import pymupdf

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = document[0]
        pixmap = page.get_pixmap(dpi=72)  # 1 px = 1 pt, same top-left origin
        x0 = max(0, int(rect.x0))
        x1 = min(pixmap.width, int(rect.x1))
        y0 = max(0, int(rect.y0))
        y1 = min(pixmap.height, int(rect.y1))
        count = 0
        for y in range(y0, y1):
            for x in range(x0, x1):
                if sum(pixmap.pixel(x, y)[:3]) < 384:  # darker than mid-grey
                    count += 1
        return count
    finally:
        document.close()


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_replace_text_on_scan_backed_page_erases_burned_in_text():
    """Deleting a word on a scan-backed page must ALSO erase the pixels of
    the word burned into the image - the old complaint was that the original
    text stayed visible and the replacement overlapped it."""
    import pymupdf

    original = _build_scan_backed_pdf()
    burned_region = pymupdf.Rect(138, 182, 190, 208)
    before = _dark_pixel_count(original, burned_region)
    assert before > 0  # the burned-in word is really there

    result = replace_text(original, 1, "gamma", "")

    assert result.origin == "scan-backed page (OCR layer)"
    after = _dark_pixel_count(result.pdf_bytes, burned_region)
    assert after < before * 0.25  # image area is whited out now
    # The searchable layer lost the word too.
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert "gamma" not in pages[0].text


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_replace_text_on_scan_backed_page_swaps_in_real_text():
    original = _build_scan_backed_pdf()

    result = replace_text(original, 1, "gamma", "omega")

    assert result.origin == "scan-backed page (OCR layer)"
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert "omega" in pages[0].text
    assert "gamma" not in pages[0].text


def test_replace_text_spanning_two_lines_inserts_once():
    """A phrase that wraps onto a second line is ONE match: search_for
    returns one rect per line, and the replacement must be written once -
    not once per line rect (which duplicated it)."""
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 200), "alpha beta", fontname="helv", fontsize=14)
    page.insert_text((72, 220), "gamma delta", fontname="helv", fontsize=14)
    original = document.tobytes()
    document.close()

    result = replace_text(original, 1, "alpha beta gamma", "omega")

    assert result.changes == 1
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    text = pages[0].text
    assert text.count("omega") == 1
    assert "alpha" not in text and "beta" not in text and "gamma" not in text


# --- Stage 7B: insert blank page / insert pages from another PDF / text layer --


def test_apply_page_order_reorders_the_whole_document():
    original = _build_two_page_native_pdf()
    result = apply_page_order(original, [2, 1])

    assert result.changes == 2 and result.origin == "structure"
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert "delta" in pages[0].text and "alpha" in pages[1].text


def test_apply_page_order_rejects_incomplete_orders():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="exactly once"):
        apply_page_order(original, [1, 1])
    with pytest.raises(ValueError, match="exactly once"):
        apply_page_order(original, [1])


def test_move_page_reorders_pages():
    original = _build_two_page_native_pdf()
    result = move_page(original, 1, 2)  # page 1 -> position 2

    assert result.changes == 1 and result.origin == "structure"
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert "delta" in pages[0].text  # old page 2 is now first
    assert "alpha" in pages[1].text  # old page 1 is now second
    # Original untouched.
    assert "alpha" in extract_pdf_pages(original, enable_ocr=False)[0].text


def test_move_page_to_same_position_is_a_noop():
    original = _build_two_page_native_pdf()
    result = move_page(original, 2, 2)
    assert result.changes == 1
    # A "move to the same place" still round-trips through pypdf, so raw
    # bytes can differ in trivial metadata (xref offsets). The REAL
    # invariant is: page order and content are exactly preserved.
    pages_after = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    pages_before = extract_pdf_pages(original, enable_ocr=False)
    assert [p.text for p in pages_after] == [p.text for p in pages_before]
    assert len(pages_after) == len(pages_before)


def test_move_page_invalid_positions_are_rejected():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="does not exist"):
        move_page(original, 1, 5)  # new position out of range
    with pytest.raises(ValueError, match="does not exist"):
        move_page(original, 9, 1)  # page itself out of range


def _build_one_page_other_pdf() -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 200), "omega zeta", fontname="helv", fontsize=14)
    return document.tobytes()


def test_insert_blank_page_in_the_middle():
    original = _build_two_page_native_pdf()
    result = insert_blank_page(original, after_page=1)

    assert result.changes == 1 and result.origin == "structure"
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert len(pages) == 3
    assert pages[1].text.strip() == ""  # new page 2 is blank
    assert "delta" in pages[2].text  # the old page 2 moved to position 3
    # Original untouched.
    assert len(extract_pdf_pages(original, enable_ocr=False)) == 2


def test_insert_blank_page_at_the_start():
    original = _build_two_page_native_pdf()
    result = insert_blank_page(original, after_page=0)

    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert len(pages) == 3
    assert pages[0].text.strip() == ""  # new FIRST page is blank
    assert "alpha" in pages[1].text


def test_insert_blank_page_invalid_position_is_rejected():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="Cannot insert"):
        insert_blank_page(original, after_page=9)


def test_insert_pages_from_other_pdf_before_chosen_page():
    original = _build_two_page_native_pdf()
    result = insert_pages_from(original, _build_one_page_other_pdf(), 2)

    assert result.changes == 1
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert len(pages) == 3
    assert "omega" in pages[1].text  # the other PDF's page went to position 2
    assert "delta" in pages[2].text
    assert len(extract_pdf_pages(original, enable_ocr=False)) == 2


def test_insert_pages_from_appended_at_end_by_default():
    original = _build_two_page_native_pdf()
    result = insert_pages_from(original, _build_one_page_other_pdf(), None)

    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert len(pages) == 3
    assert "omega" in pages[2].text  # last page


def test_insert_pages_from_invalid_other_pdf_is_rejected():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="Could not open"):
        insert_pages_from(original, b"this is not a pdf", 1)


def test_save_text_layer_empty_text_is_rejected():
    original = _build_two_page_native_pdf()
    with pytest.raises(ValueError, match="text to place"):
        save_text_layer(original, 1, "   ")


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_save_text_layer_on_scan_becomes_ai_readable():
    """After saving, plain extraction (NO OCR) sees the new text layer -"""
    """meaning Ask answers from the edited text, not the machine guess."""
    import pymupdf
    import test_pdf_loader_ocr as factory

    result = save_text_layer(
        factory.MIXED_PDF, 1, "Checked by the manual author: pressure reads OK."
    )

    assert result.changes == 1
    pages = extract_pdf_pages(result.pdf_bytes, enable_ocr=False)
    assert "pressure reads OK" in pages[0].text
    original_document = pymupdf.open(
        stream=factory.MIXED_PDF, filetype="pdf"
    )
    edited_document = pymupdf.open(stream=result.pdf_bytes, filetype="pdf")
    assert (
        original_document[0].get_pixmap().samples
        != edited_document[0].get_pixmap().samples
    )
    original_document.close()
    edited_document.close()


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_save_text_layer_visibly_replaces_ocr_text_in_place():
    """Edited text is visible over the scan in the detected text location."""
    import pymupdf
    import test_pdf_loader_ocr as factory
    from src.ocr import ocr_page

    original_document = pymupdf.open(
        stream=factory.MIXED_PDF, filetype="pdf"
    )
    original_page = original_document[0]
    original_pixels = original_page.get_pixmap().samples
    ocr_result = ocr_page(factory.MIXED_PDF, 1)

    result = save_text_layer(
        factory.MIXED_PDF,
        1,
        ocr_result.text.replace("pressure", "temperature"),
        ocr_words=ocr_result.words,
    )
    edited_document = pymupdf.open(stream=result.pdf_bytes, filetype="pdf")
    edited_page = edited_document[0]
    edited_pixmap = edited_page.get_pixmap()
    assert edited_pixmap.samples != original_pixels
    edit_markers = [
        drawing
        for drawing in edited_page.get_drawings()
        if drawing["fill"] == (1.0, 1.0, 1.0)
    ]
    assert edit_markers
    assert any(
        drawing["color"] == pytest.approx((0.45, 0.5, 0.55))
        and drawing["width"] == 0.5
        for drawing in edit_markers
    )
    from PIL import Image

    edited_image = Image.frombytes(
        "RGB", (edited_pixmap.width, edited_pixmap.height), edited_pixmap.samples
    )
    old_line_right = max(
        word.left + word.width for word in ocr_result.words
    )
    old_line_top = min(word.top for word in ocr_result.words)
    assert edited_image.getpixel((old_line_right - 5, old_line_top - 2)) == (
        255,
        255,
        255,
    )
    extracted_text = " ".join(edited_page.get_text().lower().split())
    assert "temperature gauge" in extracted_text
    from src.document_search import find_word

    assert find_word(result.pdf_bytes, "temperature gauge")

    text_spans = [
        span
        for block in edited_page.get_text("dict")["blocks"]
        if "lines" in block
        for line in block["lines"]
        for span in line["spans"]
    ]
    assert text_spans
    detected_top = min(word.top for word in ocr_result.words)
    detected_bottom = max(word.top + word.height for word in ocr_result.words)
    assert min(span["bbox"][1] for span in text_spans) >= detected_top - 5
    assert max(span["bbox"][3] for span in text_spans) <= detected_bottom + 5

    edited_document.close()
    original_document.close()


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_save_text_layer_embeds_a_manually_selected_local_font():
    import pymupdf
    import test_pdf_loader_ocr as factory
    from src.ocr import ocr_page

    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    if not font_path.is_file():
        pytest.skip("Arial TTF is not installed on this machine.")
    ocr_result = ocr_page(factory.MIXED_PDF, 1)

    result = save_text_layer(
        factory.MIXED_PDF,
        1,
        ocr_result.text.replace("pressure", "temperature"),
        ocr_words=ocr_result.words,
        font_path=font_path,
    )

    document = pymupdf.open(stream=result.pdf_bytes, filetype="pdf")
    spans = [
        span
        for block in document[0].get_text("dict")["blocks"]
        if "lines" in block
        for line in block["lines"]
        for span in line["spans"]
    ]
    replacements = [span for span in spans if "temperature" in span["text"]]
    assert replacements
    assert all("Arial" in span["font"] for span in replacements)
    document.close()


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_save_text_layer_preserves_multiple_ocr_line_positions():
    import io
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont
    from src.ocr import ocr_page

    image = Image.new("RGB", (1200, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 34)
    draw.text((40, 70), "First line pressure gauge", fill="black", font=font)
    draw.text((40, 170), "Second line before shift", fill="black", font=font)
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    scan_pdf = document.tobytes()
    document.close()

    ocr_result = ocr_page(scan_pdf, 1)
    assert len({word.line_number for word in ocr_result.words}) == 2
    edited = save_text_layer(
        scan_pdf,
        1,
        "First line temperature gauge\nSecond line after shift",
        ocr_words=ocr_result.words,
    )
    edited_document = pymupdf.open(stream=edited.pdf_bytes, filetype="pdf")
    edited_page = edited_document[0]
    extracted_text = " ".join(edited_page.get_text().lower().split())
    assert "temperature gauge" in extracted_text
    assert "second line after shift" in extracted_text

    spans = [
        span
        for block in edited_page.get_text("dict")["blocks"]
        if "lines" in block
        for line in block["lines"]
        for span in line["spans"]
    ]
    assert len(spans) == 2
    assert spans[0]["bbox"][1] < spans[1]["bbox"][1]
    edited_line = next(
        span
        for span in spans
        if span["text"] == "First line temperature gauge"
    )
    next_line_top = min(
        word.top for word in ocr_result.words if word.line_number == 1
    )
    assert edited_line["bbox"][3] < next_line_top
    edited_document.close()


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_save_text_layer_keeps_unchanged_lines_and_marks_an_added_line():
    import io
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont
    from src.ocr import ocr_page

    image = Image.new("RGB", (1200, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 34)
    draw.text((40, 70), "First line pressure gauge", fill="black", font=font)
    draw.text((40, 170), "Second line before shift", fill="black", font=font)
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    scan_pdf = document.tobytes()
    document.close()

    ocr_result = ocr_page(scan_pdf, 1)
    edited = save_text_layer(
        scan_pdf,
        1,
        "First line pressure gauge\nAdded note for point three\n"
        "Second line before shift",
        ocr_words=ocr_result.words,
    )
    edited_document = pymupdf.open(stream=edited.pdf_bytes, filetype="pdf")
    edited_page = edited_document[0]
    extracted_text = " ".join(edited_page.get_text().lower().split())
    assert "first line pressure gauge" in extracted_text
    assert "added note for point three" in extracted_text
    assert "second line before shift" in extracted_text
    spans = [
        span
        for block in edited_page.get_text("dict")["blocks"]
        if "lines" in block
        for line in block["lines"]
        for span in line["spans"]
    ]
    added_line = next(
        span for span in spans if span["text"] == "Added note for point three"
    )
    second_line_top = min(
        word.top for word in ocr_result.words if word.line_number == 1
    )
    assert added_line["bbox"][3] < second_line_top

    pixmap = edited_page.get_pixmap()
    from PIL import Image as PILImage

    rendered = PILImage.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    first_line = [
        word for word in ocr_result.words if word.line_number == 0
    ]
    assert any(
        80 < red < 150 and 100 < green < 160 and 110 < blue < 180
        for red, green, blue in rendered.crop(
            (
                min(word.left for word in first_line) - 3,
                min(word.top for word in first_line) - 3,
                max(word.left + word.width for word in first_line) + 20,
                max(word.top + word.height for word in first_line) + 25,
            )
        ).getdata()
    )
    edited_document.close()
