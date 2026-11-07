"""Inspect fonts embedded in text-based PDF pages using PyMuPDF.

This module reports PDF font resources separately from extracted text spans.
An embedded font resource means the PDF contains a font program; it does not
prove that a font is visually identical to any replacement font.
"""

from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class PdfFontResource:
    """A font resource referenced by one PDF page."""

    xref: int
    resource_name: str
    base_font_name: str
    font_type: str
    extension: str
    encoding: str
    embedded: bool


@dataclass(frozen=True)
class PdfTextSpan:
    """An extracted text run and its PDF font metrics."""

    text: str
    font_name: str
    font_size: float
    flags: int
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class PdfPageFonts:
    """Font resources and text runs for one 1-based PDF page."""

    page_number: int
    has_text: bool
    font_resources: tuple[PdfFontResource, ...]
    text_spans: tuple[PdfTextSpan, ...]

    @property
    def embedded_fonts(self) -> tuple[PdfFontResource, ...]:
        """Return font resources whose font programs are embedded."""
        return tuple(font for font in self.font_resources if font.embedded)


@dataclass(frozen=True)
class PdfFontReport:
    """Per-page font inspection result for a PDF."""

    page_count: int
    pages: tuple[PdfPageFonts, ...]

    @property
    def embedded_fonts(self) -> tuple[PdfFontResource, ...]:
        """Return embedded font resources, preserving their page context."""
        return tuple(
            font
            for page in self.pages
            for font in page.embedded_fonts
        )


def detect_embedded_fonts(pdf_bytes: bytes) -> PdfFontReport:
    """Inspect embedded font resources and extracted text spans in a PDF.

    The PDF is opened read-only and its bytes are never modified. Standard
    PDF fonts without a font program are reported with ``embedded=False``.
    Scanned pages normally have no text spans, even if they contain images.

    Raises:
        ValueError: if the input is empty or cannot be opened as a PDF.
    """
    if not pdf_bytes:
        raise ValueError("No PDF data was given for font detection.")

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError(f"Could not open this PDF for font detection: {error}") from error

    try:
        pages: list[PdfPageFonts] = []
        for page_index, page in enumerate(document):
            resources = tuple(
                PdfFontResource(
                    xref=int(font[0]),
                    extension=str(font[1]),
                    font_type=str(font[2]),
                    base_font_name=str(font[3]),
                    resource_name=str(font[4]),
                    encoding=str(font[5]),
                    embedded=bool(font[1]) and str(font[1]).lower() != "n/a",
                )
                for font in page.get_fonts(full=True)
            )
            spans = tuple(
                PdfTextSpan(
                    text=str(span["text"]),
                    font_name=str(span["font"]),
                    font_size=float(span["size"]),
                    flags=int(span["flags"]),
                    bounds=tuple(float(value) for value in span["bbox"]),
                )
                for block in page.get_text("dict")["blocks"]
                if block.get("type") == 0
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            )
            pages.append(
                PdfPageFonts(
                    page_number=page_index + 1,
                    has_text=bool(spans),
                    font_resources=resources,
                    text_spans=spans,
                )
            )

        return PdfFontReport(page_count=document.page_count, pages=tuple(pages))
    finally:
        document.close()
