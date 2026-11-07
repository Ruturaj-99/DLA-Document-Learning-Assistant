"""Stage 5C: tests for the OCR fallback in src/pdf_loader.py.

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

The central artifact is the MIXED document: page 1 is a fake scan (image
only, built in code), page 2 carries REAL native text (written with pymupdf's
text API). A correct fallback must:
    - recognize the scan page via OCR,
    - leave the native page exactly as extracted by pypdf,
    - harmlessly degrade when the engine is missing or disabled.
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ocr as ocr_module  # noqa: E402
from src.pdf_loader import extract_pdf_pages  # noqa: E402

ENGINE = ocr_module.find_tesseract_engine()

pytest.importorskip("pymupdf", reason="pymupdf unavailable")
import io  # noqa: E402

import pymupdf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SCAN_PAGE_TEXT = "Inspect the pressure gauge before every shift."
NATIVE_PAGE_TEXT = "This page holds genuine native text about heater coils."


def build_mixed_pdf() -> bytes:
    """Page 1 = image only (fake scan), page 2 = normal native text."""
    document = pymupdf.open()

    # Fake scan page: rasterize drawn text, embed the image as whole content.
    image = Image.new("RGB", (1200, 250), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 34)
    draw.text((40, 70), SCAN_PAGE_TEXT, fill="black", font=font)
    scan_page = document.new_page(width=612, height=792)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    scan_page.insert_image(scan_page.rect, stream=buffer.getvalue())

    # Native page: real vector text in the PDF's content stream.
    text_page = document.new_page(width=612, height=792)
    text_page.insert_text((72, 200), NATIVE_PAGE_TEXT, fontname="helv", fontsize=16)

    return document.tobytes()


MIXED_PDF = build_mixed_pdf() if ENGINE else None


def test_ocr_fallback_recognizes_only_the_scanned_page():
    pages = extract_pdf_pages(MIXED_PDF, enable_ocr=True)
    assert [p.page_number for p in pages] == [1, 2]
    # Page 1: native extraction finds nothing -> OCR text arrives instead.
    for word in ["inspect", "pressure", "gauge"]:
        assert word in pages[0].text.lower(), f"missing OCR text: {word}"
    # Page 2: native text passes through untouched.
    assert "heater coils" in pages[1].text


def test_enable_ocr_false_keeps_pure_native_behaviour():
    pages = extract_pdf_pages(MIXED_PDF, enable_ocr=False)
    assert pages[0].text == ""  # scan page stays textless, like Stage 2
    assert "heater coils" in pages[1].text


def broken_page(*args, **kwargs):
    raise RuntimeError("engine exploded")


def test_missing_engine_degrades_gracefully_without_crash():
    with mock.patch.object(ocr_module, "is_ocr_available", return_value=False):
        pages = extract_pdf_pages(MIXED_PDF, enable_ocr=True)
        assert pages[0].text == ""  # warned, not crashed
        assert "heater coils" in pages[1].text


def test_failing_single_page_does_not_kill_extraction():
    with mock.patch.object(ocr_module, "ocr_page", side_effect=broken_page):
        pages = extract_pdf_pages(MIXED_PDF, enable_ocr=True)
        assert pages[0].text == ""  # the failing page yields nothing...
        assert "heater coils" in pages[1].text  # ...but page 2 is untouched


def test_native_only_pdfs_skip_ocr_entirely():
    """A clearly-text PDF (>= 20 chars per page) must never wake OCR up."""
    import test_pdf_loader as factory

    native_pdf = factory.build_test_pdf(["Hello native world, plenty of characters."])
    with mock.patch.object(ocr_module, "ocr_page", side_effect=broken_page) as ocr_spy:
        pages = extract_pdf_pages(native_pdf, enable_ocr=True)
        ocr_spy.assert_not_called()  # OCR never even woke up
    assert "native world" in pages[0].text
