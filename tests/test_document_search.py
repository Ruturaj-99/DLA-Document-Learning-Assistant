"""Stage 5E: tests for src/document_search.py (find in document, OCR-aware).

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

The mixed document (1 fake-scan page + 1 native page) is reused from the
Stage 5C test factory. Tests grouped:
    - pure-native search: works WITHOUT the OCR engine at all,
    - scan-page search: needs the engine, skips politely when absent,
    - honesty: absent words return []; blank queries fail fast.
"""

import sys
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.document_search import find_word  # noqa: E402
from src.ocr import find_tesseract_engine  # noqa: E402

ENGINE = find_tesseract_engine()


def _build_native_only_pdf() -> bytes:
    """One page with REAL vector text (pymupdf's text API = native)."""
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text(
        (72, 200),
        "The coil resistance must be measured with the heater cold.",
        fontname="helv",
        fontsize=14,
    )
    return document.tobytes()


def test_blank_query_fails_fast_without_any_ocr():
    with pytest.raises(ValueError, match="word or phrase"):
        find_word(b"any bytes", "   ")


def test_native_page_search_does_not_need_ocr_engine():
    """Search a page with REAL text - no OCR engine involved at all."""
    matches = find_word(_build_native_only_pdf(), "HEATER")  # case-insensitive
    assert matches, "expected a native match for 'heater'"
    match = matches[0]
    assert match.origin == "native"
    assert match.confidence is None  # native words have no OCR noise
    assert "heater" in match.cover_text.lower()
    # Box inside the US Letter page (612 x 792 points).
    assert 0 <= match.left <= 612 and 0 <= match.top <= 792
    assert match.height > 0


def test_absent_word_returns_empty_list_on_native_page():
    assert find_word(_build_native_only_pdf(), "hedgehog") == []


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_scan_page_search_finds_phrase_from_image():
    """The phrase exists ONLY inside the embedded picture of page 1."""
    import test_pdf_loader_ocr as factory

    matches = find_word(factory.MIXED_PDF, "pressure gauge")
    scan_matches = [m for m in matches if m.page_number == 1]
    assert scan_matches, "the phrase lives on the scanned page and must be found"
    match = scan_matches[0]
    assert match.origin == "ocr"
    assert match.confidence is not None and match.confidence > 0
    assert "gauge" in match.cover_text.lower()
    assert 0 <= match.left <= 612 and 0 <= match.top <= 792


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_absent_word_returns_empty_list_on_scan_page():
    import test_pdf_loader_ocr as factory

    assert find_word(factory.MIXED_PDF, "hedgehog warranty") == []
