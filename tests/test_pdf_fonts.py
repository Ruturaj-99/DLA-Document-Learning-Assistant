"""Tests for embedded-font detection in text-based PDFs."""

import io
import sys
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_fonts import detect_embedded_fonts  # noqa: E402


def _available_test_font() -> Path | None:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _build_pdf_with_font(font_name: str, font_file: str | None = None) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 100),
        "Embedded font inspection sample",
        fontname=font_name,
        fontfile=font_file,
        fontsize=13,
    )
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_detects_embedded_font_resource_and_text_style():
    font_path = _available_test_font()
    if font_path is None:
        pytest.skip("No system TrueType font is available for the embedded-font test.")

    report = detect_embedded_fonts(
        _build_pdf_with_font("TestEmbeddedFont", str(font_path))
    )

    assert report.page_count == 1
    page = report.pages[0]
    assert page.page_number == 1
    assert page.has_text
    assert page.text_spans
    assert page.text_spans[0].text.replace("\u00a0", " ") == (
        "Embedded font inspection sample"
    )
    assert page.text_spans[0].font_name
    assert page.text_spans[0].font_size == 13
    assert len(page.text_spans[0].bounds) == 4
    assert page.embedded_fonts
    assert page.embedded_fonts[0].embedded
    assert page.embedded_fonts[0].extension.lower() != "n/a"


def test_standard_base14_font_is_not_reported_as_embedded():
    report = detect_embedded_fonts(_build_pdf_with_font("helv"))

    page = report.pages[0]
    assert page.has_text
    assert page.font_resources
    assert not page.embedded_fonts
    assert any(not font.embedded for font in page.font_resources)
    assert page.text_spans[0].font_name == "Helvetica"


def test_scanned_image_page_has_no_extracted_text_spans():
    image_buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(image_buffer, format="PNG")

    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    pdf_bytes = document.tobytes()
    document.close()

    report = detect_embedded_fonts(pdf_bytes)

    assert report.page_count == 1
    assert report.pages[0].has_text is False
    assert report.pages[0].text_spans == ()


def test_empty_or_invalid_pdf_is_rejected():
    with pytest.raises(ValueError, match="No PDF data"):
        detect_embedded_fonts(b"")

    with pytest.raises(ValueError, match="Could not open"):
        detect_embedded_fonts(b"not a PDF")
