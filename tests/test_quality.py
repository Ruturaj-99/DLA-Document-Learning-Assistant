"""Stage 5D: tests for src/quality.py (document quality report).

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

Verified ideas:
    - native pages are labelled "native text" with native_chars > 0,
    - scanned pages get OCR'd: characters + average confidence,
    - a blank scanned page is honestly reported as "OCR found almost nothing",
    - the summary counts (native / scanned / empty) add up correctly,
    - average confidence is None when no OCR page exists.
"""

import sys
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr import find_tesseract_engine  # noqa: E402
from src.quality import quality_report  # noqa: E402

ENGINE = find_tesseract_engine()


def _native_two_page_pdf() -> bytes:
    import pymupdf

    document = pymupdf.open()
    first = document.new_page(width=612, height=792)
    first.insert_text(
        (72, 200),
        "This native page explains the heater coil maintenance procedure "
        "in genuine vector text with plenty of characters.",
        fontname="helv",
        fontsize=12,
    )
    second = document.new_page(width=612, height=792)
    second.insert_text(
        (72, 200),
        "Page two continues with the filter replacement schedule and the "
        "recommended inspection intervals for the pressure gauge.",
        fontname="helv",
        fontsize=12,
    )
    return document.tobytes()


def _blank_scanned_pdf() -> bytes:
    """One page holding just a flat grey image - nothing readable at all."""
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.draw_rect((0, 0, 612, 792), color=(0.8, 0.8, 0.8), fill=(0.8, 0.8, 0.8))
    pix = page.get_pixmap(dpi=100)
    scan_bytes = pix.tobytes("png")

    document2 = pymupdf.open()
    page2 = document2.new_page(width=612, height=792)
    page2.insert_image((0, 0, 612, 792), stream=scan_bytes)
    return document2.tobytes()


def test_all_native_pages_reported_as_native():
    pdf = _native_two_page_pdf()
    report = quality_report(pdf)

    assert len(report.pages) == 2
    for page in report.pages:
        assert page.page_type == "native text"
        assert page.native_chars > 0
        assert page.ocr_chars == 0
        assert page.ocr_confidence is None
    assert report.native_pages == 2
    assert report.scanned_pages == 0
    assert report.empty_pages == 0
    assert report.average_confidence is None  # no OCR anywhere


def test_report_counts_pages_even_without_ocr_engine():
    """The report NEVER crashes without the engine - it says so instead."""
    pdf = _native_two_page_pdf()
    report = quality_report(pdf)  # native pages need no engine at all
    assert report.native_pages == len(report.pages)


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_real_scan_page_gets_confidence_and_characters():
    """The factory's fake-scan page 1 holds readable image text."""
    import test_pdf_loader_ocr as factory

    report = quality_report(factory.MIXED_PDF)

    scan_report = report.pages[0]
    assert "scan" in scan_report.page_type
    assert scan_report.native_chars == 0
    assert scan_report.ocr_chars > 0
    assert scan_report.ocr_confidence is not None and scan_report.ocr_confidence > 0
    # Page 2 is native.
    assert report.pages[1].page_type == "native text"
    assert report.scanned_pages == 1 and report.native_pages == 1
    assert report.average_confidence is not None


@pytest.mark.skipif(
    ENGINE is None, reason="Tesseract engine not installed on this machine"
)
def test_blank_scan_is_honestly_reported_unreadable():
    report = quality_report(_blank_scanned_pdf())

    page = report.pages[0]
    assert "almost nothing" in page.page_type
    assert page.ocr_chars < 20
    assert report.empty_pages == 1 and report.scanned_pages == 1
