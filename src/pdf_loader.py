"""PDF text extraction.

Stage 2 of the AI Document Learning Assistant.

The goal here is simple: turn the raw bytes of an uploaded PDF into a list
of pages, each with a clear page number (1-based, like a human counts pages)
and the text found on that page.

This module has NO knowledge of Streamlit, LLMs, or vector stores. That
separation makes it easy to test in isolation and reuse later.
"""

import io
from dataclasses import dataclass

import pymupdf
from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass
class PageContent:
    """The text extracted from one page of a PDF.

    Attributes:
        page_number: Human-facing page number, starting at 1.
        text: The extracted text, empty string if the page has none
            (for example a scanned image without an OCR text layer).
    """

    page_number: int
    text: str

    @property
    def has_text(self) -> bool:
        """True when this page actually yielded extractable text."""
        return self.text != ""


def extract_pdf_pages(pdf_bytes: bytes, enable_ocr: bool = True) -> list[PageContent]:
    """Extract text from PDF bytes, grouped by page.

    Args:
        pdf_bytes: The raw content of a PDF file (as read from an upload).
        enable_ocr: When True (default), pages whose native extraction yields
            (almost) no text get a second chance through OCR (see src/ocr.py).
            Set False to keep Stage 2 pure-native behaviour.

    Returns:
        A list of PageContent, one entry per PDF page, in page order.

    Raises:
        ValueError: If the data is not a readable PDF (corrupt, encrypted,
            or empty). The message is written so it can be shown to an
            end user directly.
    """
    if not pdf_bytes:
        raise ValueError(
            "The file is empty. Please upload a non-empty PDF document."
        )

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except (PdfReadError, ValueError, EOFError) as error:
        raise ValueError(
            "This file could not be read as a PDF. It may be damaged or "
            "not a PDF at all."
        ) from error

    if reader.is_encrypted:
        raise ValueError(
            "This PDF is password-protected. Password-protected documents "
            "are not supported."
        )

    pages: list[PageContent] = []
    for index, page in enumerate(reader.pages):
        page_number = index + 1  # PDF pages are 0-based internally; users count from 1
        try:
            text = page.extract_text() or ""
        except Exception as error:  # a single bad page must not kill the extraction
            try:
                with pymupdf.open(stream=pdf_bytes, filetype="pdf") as fallback_document:
                    text = fallback_document[page_number - 1].get_text("text") or ""
            except Exception as fallback_error:
                text = ""
                print(
                    f"Warning: could not extract text from page {page_number} "
                    f"with pypdf ({error}) or PyMuPDF ({fallback_error})."
                )
            else:
                if text.strip():
                    print(
                        f"Warning: pypdf could not extract page {page_number} "
                        f"({error}); recovered text with PyMuPDF."
                    )
                else:
                    print(
                        f"Warning: could not extract text from page {page_number} "
                        f"with pypdf ({error}); PyMuPDF returned no text."
                    )
        pages.append(PageContent(page_number=page_number, text=text.strip()))

    if enable_ocr:
        _apply_ocr_fallback(pdf_bytes, pages)

    return pages


def _apply_ocr_fallback(pdf_bytes: bytes, pages: list[PageContent]) -> None:
    """Second-chance pass: recognize scanned (textless) pages via OCR.

    Kept fail-safe by design: a missing engine or a single failing page only
    produces a warning and leaves that page textless - extraction never
    breaks because of OCR.

    NOTE: the import is lazy ON PURPOSE: src.ocr imports PageContent from
    this module, so a top-level import would create a circular import loop.
    """
    from src import ocr as ocr_module  # lazy import (see note above)

    if not ocr_module.is_ocr_available():
        print("OCR skipped: Tesseract engine is not installed. "
              "Scanned pages will stay unreadable until it is installed.")
        return

    for page in pages:
        if not ocr_module.needs_ocr(page.text) or not ocr_module.page_needs_ocr(
            pdf_bytes, page.page_number
        ):
            continue
        try:
            result = ocr_module.ocr_page(pdf_bytes, page.page_number)
        except Exception as error:
            print(f"Warning: OCR failed on page {page.page_number}: {error}")
            continue
        # Never downgrade: only accept OCR text when it actually found MORE
        # content than the native extraction did. (A "nearly empty" page with
        # a few native characters keeps those characters instead of losing
        # them to an OCR run that saw a blank page.)
        if result.text and len(result.text) > len(page.text):
            # PageContent is a plain dataclass; replacing .text keeps page
            # numbers intact and everything downstream (chunking, embeddings)
            # sees OCR text exactly like native text.
            page.text = result.text


def summarize_document(pdf_bytes: bytes, file_name: str) -> dict:
    """Return a small summary dict, handy for the UI and for logging.

    Raises the same ValueError as extract_pdf_pages() for invalid PDFs.
    """
    pages = extract_pdf_pages(pdf_bytes)
    return {
        "file_name": file_name,
        "page_count": len(pages),
        "pages": pages,
        "chars_total": sum(len(p.text) for p in pages),
        "pages_without_text": [p.page_number for p in pages if not p.has_text],
    }
