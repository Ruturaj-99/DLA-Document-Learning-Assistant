"""Stage 5F: real-PDF checks that table-of-contents pages stop winning.

These tests use the 40-page evaluation PDF from the DLA exercise. They are
skipped automatically when the file is not present, so the normal test suite
still runs on any machine.

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/test_retrieval_toc_pdf.py -v
"""

import sys
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_pages  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.pdf_loader import extract_pdf_pages  # noqa: E402
from src.retrieval import search_chunks  # noqa: E402

PDF_PATH = Path(r"C:\Users\Electrical6\Downloads\PDF TESTING FOR LLM.pdf")
TOC_PAGE = 4

Q4 = "What are the two construction types of the float and board gauge, and how do their applications differ?"
Q5 = "What is the specified accuracy of the float and board gauge?"
Q6 = "What is the specified float size for the float and board gauge?"
Q7 = "What is the least count and process connection specified for the reflex level gauge?"

pytestmark = pytest.mark.skipif(
    not PDF_PATH.exists(), reason=f"evaluation PDF not found at {PDF_PATH}"
)


@pytest.fixture(scope="module")
def embedded_pdf():
    pages = extract_pdf_pages(PDF_PATH.read_bytes())
    chunks = chunk_pages(PDF_PATH.name, pages)
    return embed_chunks(chunks)


@pytest.mark.parametrize("question", [Q4, Q5, Q6, Q7], ids=["Q4", "Q5", "Q6", "Q7"])
def test_technical_questions_do_not_return_contents_page(embedded_pdf, question):
    results = search_chunks(question, embedded_pdf, top_k=5)
    pages = [result.page_number for result in results]
    assert pages, "retrieval returned no passages"
    assert TOC_PAGE not in pages, (
        f"contents page leaked into the top-5 (pages={pages})"
    )


def test_q4_returns_the_float_and_board_datasheet(embedded_pdf):
    results = search_chunks(Q4, embedded_pdf, top_k=5)
    pages = [result.page_number for result in results]
    assert 6 in pages, f"expected the float and board datasheet on page 6, got {pages}"


def test_q5_returns_the_float_and_board_accuracy_page(embedded_pdf):
    results = search_chunks(Q5, embedded_pdf, top_k=5)
    pages = [result.page_number for result in results]
    assert 9 in pages, f"expected the float and board accuracy page 9, got {pages}"


def test_q6_returns_the_float_and_board_float_size_page(embedded_pdf):
    results = search_chunks(Q6, embedded_pdf, top_k=5)
    pages = [result.page_number for result in results]
    assert 9 in pages, f"expected the float and board float-size page 9, got {pages}"


def test_explicit_toc_question_still_returns_the_contents_page(embedded_pdf):
    results = search_chunks(
        "Which page has the reflex level gauge datasheet?", embedded_pdf, top_k=5
    )
    assert results
    assert results[0].page_number == TOC_PAGE, (
        f"contents page should rank first, got "
        f"{[(r.page_number, round(r.score, 3)) for r in results]}"
    )
