"""Stage 2: tests for src/pdf_loader.py.

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

The test PDF is generated in code (not checked in as a binary file), so the
tests are self-contained and easy to understand: a minimal but valid PDF is
assembled from raw bytes, and we control exactly what text is on each page.
"""

import sys
from pathlib import Path

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_loader import extract_pdf_pages, summarize_document  # noqa: E402


def _escape(text: str) -> str:
    """Escape characters that have special meaning inside a PDF string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_test_pdf(page_texts: list[str]) -> bytes:
    """Build a tiny, valid PDF containing the given page texts.

    Object numbering plan:
        1 = catalog (root of the document)
        2 = pages tree (lists all pages)
        3 = font (Helvetica, available in every PDF reader)
        4, 5, 6, ... = one page object + one content stream per page

    For each page we add a content stream that draws the text with the font.
    Byte offsets for the xref table are computed as we append, so the result
    is a valid PDF that pypdf can read.
    """
    objects: list[dict] = []  # each entry: {"num": n, "body": str}

    font_num = 3  # object 3 is the shared Helvetica font
    objects.append({"num": 1, "body": ""})  # reserved: catalog (filled below)
    objects.append({"num": 2, "body": ""})  # reserved: pages tree (filled below)
    objects.append(
        {"num": font_num, "body": "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    )

    kids = []
    for text in page_texts:
        page_obj = len(objects) + 1
        content_obj = page_obj + 1
        kids.append(f"{page_obj} 0 R")

        stream = "BT /F1 12 Tf 72 720 Td (" + _escape(text) + ") Tj ET"
        objects.append(
            {
                "num": page_obj,
                "body": (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Contents {content_obj} 0 R /Resources << /Font << /F1 {font_num} 0 R >> >>"
                ),
            }
        )
        objects.append(
            {
                "num": content_obj,
                "body": f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream",
            }
        )

    kids_ref = "[" + " ".join(kids) + "]"
    objects[0]["body"] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[1]["body"] = f"<< /Type /Pages /Kids {kids_ref} /Count {len(page_texts)} >>"

    # Serialize everything and remember byte offsets for the xref table.
    out = bytearray()
    out += b"%PDF-1.4\n"
    offsets = {}
    for obj in objects:
        offsets[obj["num"]] = len(out)
        out += f"{obj['num']} 0 obj\n{obj['body']}\nendobj\n".encode("latin-1")

    xref_offset = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for obj in objects:
        out += f"{offsets[obj['num']]:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


SAMPLE_PDF = build_test_pdf(["First page text about transformers.", "Second page: attention scores."])


def test_page_numbers_are_one_based():
    pages = extract_pdf_pages(SAMPLE_PDF)
    assert [p.page_number for p in pages] == [1, 2]


def test_extracts_expected_text():
    pages = extract_pdf_pages(SAMPLE_PDF)
    assert pages[0].text == "First page text about transformers."
    assert pages[1].text == "Second page: attention scores."


def test_pymupdf_recovers_text_when_pypdf_extraction_fails(monkeypatch, capsys):
    import pymupdf
    from pypdf._page import PageObject

    def fail_extraction(*args, **kwargs):
        raise RuntimeError("Reached maximum number of tokens per stream.")

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "Fallback extraction works.")
    fallback_pdf = document.tobytes()
    document.close()
    monkeypatch.setattr(PageObject, "extract_text", fail_extraction)

    pages = extract_pdf_pages(fallback_pdf, enable_ocr=False)

    assert pages[0].text == "Fallback extraction works."
    assert "recovered text with PyMuPDF" in capsys.readouterr().out


def test_empty_pdf_bytes_raise_value_error():
    import pytest

    with pytest.raises(ValueError):
        extract_pdf_pages(b"")


def test_not_a_pdf_raises_value_error():
    import pytest

    with pytest.raises(ValueError) as error:
        extract_pdf_pages(b"this is a text file, not a pdf")
    assert "could not be read as a PDF" in str(error.value)


def test_page_without_text_yields_empty_string():
    pages = extract_pdf_pages(build_test_pdf(["Real content", ""]))
    assert pages[0].has_text is True
    assert pages[1].page_number == 2
    assert pages[1].text == ""
    assert pages[1].has_text is False


def test_summarize_document_reports_counts():
    summary = summarize_document(SAMPLE_PDF, "sample.pdf")
    assert summary["file_name"] == "sample.pdf"
    assert summary["page_count"] == 2
    assert summary["pages_without_text"] == []
    assert summary["chars_total"] > 0
