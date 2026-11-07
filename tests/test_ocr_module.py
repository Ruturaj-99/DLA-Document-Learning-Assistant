"""Stage 5B: tests for src/ocr.py (OCR for scanned PDF pages).

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

How we test OCR without a real scanner: a PHYSICAL-PAGE fake is built in
code - text is drawn on an image (PIL), and that image is embedded as the
whole page content of a PDF (pymupdf). From the pipeline's point of view
that page IS a scan: native text extraction finds nothing, OCR must work.

Engine-dependent tests skip politely when Tesseract is absent (never fail,
never download).
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ocr  # noqa: E402
from src.pdf_loader import PageContent, extract_pdf_pages  # noqa: E402

# --- pure logic: no engine needed, always run -------------------------------


def test_needs_ocr_flags_nearly_textless_pages():
    assert ocr.needs_ocr("") is True
    assert ocr.needs_ocr("   ") is True
    assert ocr.needs_ocr("junk") is True  # a stray fragment below the threshold
    assert ocr.needs_ocr("A full sentence that clearly contains real text.") is False


def test_find_pages_needing_ocr_returns_page_numbers():
    pages = [
        PageContent(1, "Real native text here, plenty of characters."),
        PageContent(2, ""),
        PageContent(3, "junk"),
    ]
    assert ocr.find_pages_needing_ocr(pages) == [2, 3]


# --- engine-dependent: build a fake scan and read it ------------------------

ENGINE = ocr.find_tesseract_engine()
pytestmark = pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)

SCAN_TEXT = "Inspect the pressure gauge before every shift."

pytest.importorskip("pymupdf", reason="pymupdf unavailable")
import io  # noqa: E402

import pymupdf  # noqa: E402
from PIL import Image as PILImage, ImageDraw, ImageFont  # noqa: E402


def build_fake_scan_pdf(texts: list[str]) -> bytes:
    """One image-only page per text: the pipeline sees it as a scanner output."""
    document = pymupdf.open()
    for text in texts:
        image = PILImage.new("RGB", (1200, 250), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 34)
        draw.text((40, 70), text, fill="black", font=font)
        page = document.new_page(width=612, height=792)  # US Letter in points
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        page.insert_image(page.rect, stream=buffer.getvalue())
    return document.tobytes()


SAMPLE_SCAN_PDF = build_fake_scan_pdf([SCAN_TEXT])
NATIVE_PDF = None  # built lazily from the existing Stage 2 test factory


def get_native_text_pdf() -> bytes:
    """A REAL text PDF (from the Stage 2 test factory), for contrast."""
    global NATIVE_PDF
    if NATIVE_PDF is None:
        import test_pdf_loader as factory

        NATIVE_PDF = factory.build_test_pdf(["Native text about heaters and coils."])
    return NATIVE_PDF


def test_visual_recognition_returns_text_and_pdf_coordinates():
    result = ocr.ocr_page(SAMPLE_SCAN_PDF, page_number=1)
    assert result.page_number == 1
    for word in ["inspect", "pressure", "gauge", "shift"]:
        assert word in result.text.lower(), f"missing OCR text: {word}"
    assert result.words, "OCR must produce word boxes"
    for word in result.words:
        assert word.confidence > 0
        assert word.height > 0 and word.width > 0

    # Coordinates must be in PDF points INSIDE the page (612x792 was set above).
    for word in result.words:
        assert 0 <= word.left <= 612
        assert 0 <= word.top <= 792
    # OCR text feeds straight into the standard PageContent lineage.
    standard = result.to_page_content()
    assert standard.page_number == 1 and standard.text == result.text


def test_render_scales_word_boxes_with_dpi():
    """Boxes are stored in PDF POINTS, so the same word must occupy the same
    physical space at any DPI - that is the whole point of the conversion.
    (In raw pixels the box doubles from 100->200 dpi, then shrinks back
    while converting to points; net result: constant physical size.)
    """
    low = ocr.ocr_page(SAMPLE_SCAN_PDF, page_number=1, dpi=100)
    high = ocr.ocr_page(SAMPLE_SCAN_PDF, page_number=1, dpi=200)
    word_low = next(w for w in low.words if w.text.lower() == "gauge")
    word_high = next(w for w in high.words if w.text.lower() == "gauge")
    ratio = word_high.width / word_low.width
    assert 0.8 < ratio < 1.2, f"physical size should be DPI-independent: {ratio}"


def test_detection_finds_scan_page_in_native_pdf_but_ocr_keeps_text_pdf_alone():
    # The Stage 2 factory PDF has REAL text: extraction finds a full sentence.
    native_pages = extract_pdf_pages(get_native_text_pdf())
    assert ocr.find_pages_needing_ocr(native_pages) == []

    # The fake scan: pure native extraction (OCR off) finds nothing -> flagged.
    scan_pages = extract_pdf_pages(SAMPLE_SCAN_PDF, enable_ocr=False)
    assert ocr.find_pages_needing_ocr(scan_pages) == [1]


def test_ocr_rejects_a_page_with_selectable_native_text():
    with pytest.raises(ValueError, match="scanned image"):
        ocr.ocr_page(get_native_text_pdf(), page_number=1)
