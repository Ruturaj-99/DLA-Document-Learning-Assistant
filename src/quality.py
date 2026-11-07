"""Document quality report - "what does the AI actually see?".

Stage 5D (part b) of the AI Document Learning Assistant.

The user cannot look inside the pipeline, so pages can hide problems: a
scanned page may OCR badly (low confidence), a page may be completely
unreadable, or the OCR engine may simply be missing. This report shows,
page by page:

    - page type: native text / scanned (OCR'd) / scanned but NO engine / empty
    - how many characters came from the PDF text layer vs. from OCR
    - the average OCR confidence per scanned page (0-100, tesseract's own
      word confidences are averaged)

Honesty rules: nothing is guessed. A scanned page with no installed engine
is REPORTED as such, and confidence is None when there is no OCR at all.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.ocr import OCR_DPI, is_ocr_available, needs_ocr, ocr_page
from src.pdf_loader import extract_pdf_pages


@dataclass
class PageReport:
    """What one page looks like from the AI's point of view."""

    page_number: int  # 1-based
    page_type: str  # honest label, see module docstring
    native_chars: int  # characters from the PDF's own text layer
    ocr_chars: int  # characters OCR managed to read (0 for native pages)
    ocr_confidence: Optional[float]  # average word confidence, OCR only


@dataclass
class DocumentReport:
    """All pages + a few summary numbers for the top of the UI."""

    pages: List[PageReport]
    native_pages: int
    scanned_pages: int
    empty_pages: int
    average_confidence: Optional[float]  # None when no OCR anywhere


def _average_word_confidence(words) -> Optional[float]:
    """Plain average of the OCR word confidences; None when there are none."""
    scores = [word.confidence for word in words if word.confidence is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 1)


def quality_report(pdf_bytes: bytes, dpi: int = OCR_DPI) -> DocumentReport:
    """Inspect every page of the document and describe what the AI can read.

    Native extraction runs WITHOUT OCR (that is our detector). Scanned
    pages are then OCR'd on the spot to measure quality.

    Raises:
        ValueError: from `extract_pdf_pages` for invalid PDFs.
    """
    pages = extract_pdf_pages(pdf_bytes, enable_ocr=False)

    page_reports: List[PageReport] = []
    scanned = 0
    empty = 0
    native = 0
    confidence_sum = 0.0
    confidence_count = 0

    for page in pages:
        native_chars = len(page.text.strip())

        if needs_ocr(page.text):
            if is_ocr_available():
                ocr_result = ocr_page(pdf_bytes, page.page_number, dpi=dpi)
                ocr_chars = len(ocr_result.text.strip())
                confidence = _average_word_confidence(ocr_result.words)

                # Decide between "OCR rescued this page" and "still unreadable".
                if ocr_chars < 20:
                    page_type = "scanned - OCR found almost nothing"
                    empty += 1  # effectively unusable for the AI
                else:
                    page_type = "scanned (read via OCR)"
                scanned += 1
            else:
                ocr_chars = 0
                confidence = None
                page_type = "scanned - OCR engine missing"
                scanned += 1
        elif native_chars > 0:
            ocr_chars = 0
            confidence = None
            page_type = "native text"
            native += 1
        else:
            ocr_chars = 0
            confidence = None
            page_type = "no recognizable text"
            empty += 1

        if confidence is not None:
            confidence_sum += confidence
            confidence_count += 1

        page_reports.append(
            PageReport(
                page_number=page.page_number,
                page_type=page_type,
                native_chars=native_chars,
                ocr_chars=ocr_chars,
                ocr_confidence=confidence,
            )
        )

    average_confidence = (
        round(confidence_sum / confidence_count, 1) if confidence_count else None
    )

    return DocumentReport(
        pages=page_reports,
        native_pages=native,
        scanned_pages=scanned,
        empty_pages=empty,
        average_confidence=average_confidence,
    )
