"""OCR for scanned PDF pages.

Stage 5B of the AI Document Learning Assistant.

In simple words: some PDF pages contain no real text at all - they are
PICTURES of text (a scanner's output). Recognition happens in three steps:

    1. pymupdf renders the page to a sharp image (it also knows where every
       image sits on the page),
    2. pytesseract reads the words off that image and reports, for every
       word, WHERE on the image it sits (left/top/width/height in pixels),
    3. we convert those pixel boxes into PDF points (the PDF's own measuring
       grid), so the boxes line up with the real page - ready for the later
       highlight feature.

Boundaries of this module (Stage 5B only):
    - it recognizes single pages and finds pages that need OCR,
    - it does NOT yet decide WHEN pdf_loader should call it (that is Stage 5C),
    - it never downloads anything; if the engine is missing, we say so.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from src.pdf_loader import PageContent

# Rendering sharpness. 200 gives clean recognition on most scans while
# staying fast on a CPU-only machine (300 would be sharper but slower).
OCR_DPI = 200

# A native page counts as "textless" (-> OCR needed) when extraction returns
# fewer printable characters than this. Title pages often hold a few words,
# so we use a small threshold rather than 0.
MIN_NATIVE_TEXT_CHARS = 20

_engine_path: str | None = None


def find_tesseract_engine() -> str | None:
    """Return the tesseract.exe path, or None when not installed.

    Checks PATH first, then the standard UB-Mannheim install locations.
    The result is remembered: the engine never moves mid-run.
    """
    global _engine_path
    if _engine_path is not None:
        return _engine_path

    found = shutil.which("tesseract")
    if not found:
        for candidate in [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe",
        ]:
            if candidate.exists():
                found = str(candidate)
                break
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        _engine_path = found
    return found


def is_ocr_available() -> bool:
    """True when the Tesseract engine is installed and runnable."""
    return find_tesseract_engine() is not None


@dataclass
class OcrWord:
    """One recognized word and its place on the PDF page.

    Coordinates are PDF points (72 per inch) with the origin at the page's
    top-left corner - the same space pymupdf uses, so highlight rectangles
    later need no extra conversion.
    """

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    line_number: int = 0


@dataclass
class OcrPage:
    """The full OCR outcome for one PDF page."""

    page_number: int  # 1-based, consistent with the rest of the project
    text: str  # all recognized words, in reading order
    words: list[OcrWord]

    def to_page_content(self) -> PageContent:
        """Convert into the standard page type the rest of the project uses."""
        return PageContent(page_number=self.page_number, text=self.text)


def needs_ocr(native_text: str) -> bool:
    """Decide from already-extracted text whether a page looks "textless".

    A scanned page yields an empty string or stray fragments from normal
    extraction; a real text page yields full sentences.
    """
    return len(native_text.strip()) < MIN_NATIVE_TEXT_CHARS


def find_pages_needing_ocr(pages: list[PageContent]) -> list[int]:
    """Return the 1-based page numbers that should run through OCR."""
    return [page.page_number for page in pages if needs_ocr(page.text)]


def page_needs_ocr(pdf_bytes: bytes, page_number: int) -> bool:
    """Check whether one page is image-based and lacks enough native text."""
    if not pdf_bytes:
        raise ValueError("The document is empty - nothing to inspect.")
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"Page {page_number} does not exist - this document has "
                f"{document.page_count} page(s)."
            )
        page = document[page_number - 1]
        return needs_ocr(page.get_text()) and bool(page.get_images(full=True))


def render_pdf_page(pdf_bytes: bytes, page_number: int, dpi: int = OCR_DPI) -> Image.Image:
    """Render one PDF page to a PIL image (that's what OCR reads).

    Args:
        pdf_bytes: the raw original PDF (never modified).
        page_number: 1-based.
        dpi: rendering sharpness; word boxes scale with this.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page = document[page_number - 1]  # pymupdf counts from 0
        pixmap = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def ocr_page(pdf_bytes: bytes, page_number: int, dpi: int = OCR_DPI) -> OcrPage:
    """OCR a scanned/image-only page: text plus word boxes in PDF points.

    Args:
        pdf_bytes: the raw original PDF (never modified).
        page_number: 1-based.
        dpi: rendering sharpness; word boxes scale with this.

    Raises:
        RuntimeError: when the Tesseract engine is not installed.
        ValueError: for empty documents or page numbers below 1.
    """
    if not page_needs_ocr(pdf_bytes, page_number):
        raise ValueError(
            "OCR is only for pages with a scanned image and little or no "
            "selectable text. Use Replace text to edit native text."
        )
    if not find_tesseract_engine():
        raise RuntimeError(
            "OCR engine (Tesseract) is not installed. Please install Tesseract."
        )

    image = render_pdf_page(pdf_bytes, page_number, dpi)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # pixels -> PDF points: page rendered at `dpi`, PDF grid uses 72/inch.
    scale = 72.0 / dpi

    words: list[OcrWord] = []
    lines_by_key: dict[tuple, list[OcrWord]] = {}
    # Remember each line's vertical position so we can sort lines visually.
    first_top_in_points: dict[tuple, int] = {}

    for index, raw_word in enumerate(data["text"]):
        word = raw_word.strip()
        if not word or float(data["conf"][index]) <= 0:
            continue  # skip placeholders and unreliable detections
        top = round(int(data["top"][index]) * scale)
        line_key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        ocr_word = OcrWord(
            text=word,
            left=round(int(data["left"][index]) * scale),
            top=top,
            width=round(int(data["width"][index]) * scale),
            height=round(int(data["height"][index]) * scale),
            confidence=float(data["conf"][index]),
        )
        words.append(ocr_word)
        lines_by_key.setdefault(line_key, []).append(ocr_word)
        first_top_in_points.setdefault(line_key, top)

    # Reading order: lines sorted by their vertical position, words joined
    # left-to-right; preserving line breaks lets edits retain the scan layout.
    ordered_lines = sorted(lines_by_key, key=lambda key: first_top_in_points[key])
    text_lines = []
    for line_number, key in enumerate(ordered_lines):
        line_words = sorted(lines_by_key[key], key=lambda word: word.left)
        for word in line_words:
            word.line_number = line_number
        text_lines.append(" ".join(word.text for word in line_words))
    text = "\n".join(text_lines).strip()

    return OcrPage(page_number=page_number, text=text, words=words)
