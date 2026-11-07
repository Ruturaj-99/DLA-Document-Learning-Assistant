"""Tests for local font estimation on OCR-recognized scanned PDF pages."""

import io
import sys
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr import find_tesseract_engine  # noqa: E402
from src.pdf_font_estimation import (  # noqa: E402
    discover_local_font_faces,
    estimate_scanned_page_fonts,
)

ENGINE = find_tesseract_engine()
pytestmark = pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)


def _available_fonts() -> list[Path]:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    return [path for path in candidates if path.is_file()]


def _build_scan(font_path: Path, text_lines: list[str] | None = None) -> bytes:
    lines = text_lines or ["Inspect the pressure gauge before every shift."]
    image = Image.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 34)
    for index, text in enumerate(lines):
        draw.text((40, 50 + index * 100), text, fill="black", font=font)

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_estimation_ranks_source_font_highest_with_similarity_score():
    fonts = _available_fonts()
    if len(fonts) < 2:
        pytest.skip("Two local fonts are needed to compare candidate rankings.")
    source_font = fonts[0]
    pdf_bytes = _build_scan(source_font)
    original_pdf_bytes = bytes(pdf_bytes)

    estimate = estimate_scanned_page_fonts(
        pdf_bytes,
        1,
        candidate_font_paths=fonts[:2],
        top_n=2,
    )

    assert estimate.page_number == 1
    assert estimate.candidate_count == 2
    assert "not a probability" in estimate.score_description
    assert len(estimate.lines) == 1
    assert estimate.lines[0].matches[0].family_name == ImageFont.truetype(
        str(source_font), 16
    ).getname()[0]
    assert 0 < estimate.lines[0].matches[0].confidence_score <= 1
    assert estimate.lines[0].matches[0].glyph_shape_similarity > 0
    assert estimate.lines[0].matches[0].estimated_size_points > 0
    assert estimate.lines[0].average_interword_spacing_points is not None
    assert estimate.alignment == "single line"
    assert pdf_bytes == original_pdf_bytes


def test_estimation_reports_line_spacing_and_alignment():
    fonts = _available_fonts()
    if not fonts:
        pytest.skip("No system font is available for the scan fixture.")
    pdf_bytes = _build_scan(
        fonts[0],
        ["First line of text", "Second line of text", "Third line of text"],
    )

    estimate = estimate_scanned_page_fonts(
        pdf_bytes,
        1,
        candidate_font_paths=fonts[:1],
    )

    assert len(estimate.lines) == 3
    assert estimate.alignment == "left aligned"
    assert len(estimate.line_spacing_points) == 2
    assert all(spacing > 0 for spacing in estimate.line_spacing_points)


def test_default_candidates_are_discovered_from_local_system_fonts():
    fonts = discover_local_font_faces()
    if not fonts:
        pytest.skip("No supported local font files were discovered.")

    assert all(face.family_name and face.style_name for face in fonts)
    assert all(Path(face.path).is_file() for face in fonts)


def test_estimation_rejects_bad_options_and_page_numbers():
    fonts = _available_fonts()
    if not fonts:
        pytest.skip("No system font is available for the scan fixture.")
    pdf_bytes = _build_scan(fonts[0])

    with pytest.raises(ValueError, match="top_n"):
        estimate_scanned_page_fonts(pdf_bytes, 1, top_n=0)
    with pytest.raises(ValueError, match="dpi"):
        estimate_scanned_page_fonts(pdf_bytes, 1, dpi=0)
    with pytest.raises(ValueError, match="does not exist"):
        estimate_scanned_page_fonts(
            pdf_bytes, 2, candidate_font_paths=fonts[:1]
        )
    with pytest.raises(ValueError, match="font file"):
        estimate_scanned_page_fonts(
            pdf_bytes,
            1,
            candidate_font_paths=[Path("missing-font-file.ttf")],
        )


def test_blank_scanned_page_reports_no_text_to_estimate():
    image_buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(image_buffer, format="PNG")
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_buffer.getvalue())
    pdf_bytes = document.tobytes()
    document.close()
    fonts = _available_fonts()
    if not fonts:
        pytest.skip("No system font is available for font estimation.")

    with pytest.raises(ValueError, match="did not find any text lines"):
        estimate_scanned_page_fonts(
            pdf_bytes, 1, candidate_font_paths=fonts[:1]
        )
