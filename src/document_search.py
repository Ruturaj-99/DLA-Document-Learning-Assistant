

from dataclasses import dataclass
from typing import Optional

import pymupdf

from src.ocr import OCR_DPI, is_ocr_available, needs_ocr, ocr_page, page_needs_ocr
from src.pdf_loader import extract_pdf_pages


@dataclass
class WordBox:
    """A word and its place on the page (PDF points, origin = top-left)."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: Optional[float] = None  # None for native words (they are exact)


@dataclass
class SearchMatch:
    """One found occurrence of the searched word or phrase."""

    page_number: int  # 1-based
    cover_text: str  # a few words around the match (the "snippet")
    left: int
    top: int
    width: int
    height: int
    origin: str  # "native" or "ocr" - WHERE the words came from
    confidence: Optional[float] = None  # average OCR confidence, OCR-only


def _match_tokens(
    words: list[WordBox], tokens: list[str]
) -> list[list[WordBox]]:
    """Find every place where `tokens` appear in sequence (case-insensitive).

    Receives ALREADY-lowercased tokens (the caller does the lowering once).
    Called a "sliding window": move a window of `len(tokens)` words across
    the document's words; wherever the words match, keep that group.
    """
    matches: list[list[WordBox]] = []
    window_size = len(tokens)
    lowered = [word.text.casefold() for word in words]
    for start in range(len(words) - window_size + 1):
        if lowered[start : start + window_size] == tokens:
            matches.append(words[start : start + window_size])
    return matches


def _union_box(matched_words: list[WordBox]) -> tuple[int, int, int, int]:
    """One rectangle around several words (their outer edges joined)."""
    left = min(word.left for word in matched_words)
    top = min(word.top for word in matched_words)
    right = max(word.left + word.width for word in matched_words)
    bottom = max(word.top + word.height for word in matched_words)
    return left, top, right - left, bottom - top


def _native_words(pdf_bytes: bytes, page_number: int) -> list[WordBox]:
    """Words + boxes of a native text page, straight from the PDF."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page = document[page_number - 1]
        raw = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, no)
    return [
        WordBox(
            text=word,
            left=round(x0),
            top=round(y0),
            width=round(x1 - x0),
            height=round(y1 - y0),
        )
        for (x0, y0, x1, y1, word, _block, _line, _no) in raw
    ]


def _snippet(words: list[WordBox], matched: list[WordBox]) -> str:
    """A few words around the match (max ~5 before, 5 after)."""
    if matched[0] not in words:  # safety net; should never happen
        return " ".join(word.text for word in matched)
    first_index = words.index(matched[0])
    last_index = words.index(matched[-1])
    window = words[max(0, first_index - 5) : last_index + 6]
    return " ".join(word.text for word in window)


def find_word(
    pdf_bytes: bytes, query: str, dpi: int = OCR_DPI
) -> list[SearchMatch]:
    """Search a word/phrase on all pages, native AND scanned.

    Args:
        pdf_bytes: the raw original PDF (never modified).
        query: a word or phrase, e.g. "pressure gauge". Case-insensitive.
        dpi: OCR rendering sharpness for scanned pages.

    Returns:
        SearchMatch items ordered by page. EMPTY list means honestly
        "not found" - not an error.

    Raises:
        ValueError: for an empty/blank query.
        RuntimeError: only if a scanned page needs OCR and the engine is
            missing (the message explains how to fix it).
    """
    if not query.strip():
        raise ValueError("Please type a word or phrase to search for.")
    tokens = query.split()
    lowered_tokens = [token.casefold() for token in tokens]

    # Native extraction WITHOUT OCR: this is only our page map/detector.
    pages = extract_pdf_pages(pdf_bytes, enable_ocr=False)

    matches: list[SearchMatch] = []
    ocr_warned_missing = False

    for page in pages:
        if needs_ocr(page.text) and page_needs_ocr(pdf_bytes, page.page_number):
            # Scanned page: words must come from OCR.
            if not is_ocr_available():
                if not ocr_warned_missing:
                    print(
                        "Warning: cannot search scanned pages - OCR engine "
                        "(Tesseract) is not installed."
                    )
                    ocr_warned_missing = True
                continue
            ocr_result = ocr_page(pdf_bytes, page.page_number, dpi=dpi)
            words: list[WordBox] = [
                WordBox(
                    text=word.text,
                    left=word.left,
                    top=word.top,
                    width=word.width,
                    height=word.height,
                    confidence=word.confidence,
                )
                for word in ocr_result.words
            ]
            origin = "ocr"
        else:
            # Native page: exact words straight from the PDF.
            words = _native_words(pdf_bytes, page.page_number)
            origin = "native"

        for matched in _match_tokens(words, lowered_tokens):
            left, top, width, height = _union_box(matched)
            confidences = [
                word.confidence for word in matched if word.confidence is not None
            ]
            matches.append(
                SearchMatch(
                    page_number=page.page_number,
                    cover_text=_snippet(words, matched),
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    origin=origin,
                    confidence=(
                        round(sum(confidences) / len(confidences), 1)
                        if confidences
                        else None
                    ),
                )
            )

    return matches
