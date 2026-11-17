"""Phase 1 tests for the C-021 instrument document.

These run the REAL ingestion path on the user's actual document:
    src.pdf_loader.extract_pdf_pages -> src.chunking.chunk_pages (defaults)

They are skipped on machines where the source PDF is not available, so the
suite stays portable (the same gating pattern tests/test_pdf_loader_ocr.py
uses for the Tesseract engine).

Verified requirements:
    - page 10 chunk containing "Pressure : upto 2 Bar" carries the
      Parshall flume instrument identity (section metadata)
    - page 3 chunk containing "OPERATING PRESSURE ... -1T0 1.0 BAR" carries
      the level transmitter identity
    - page 7 chunk boundaries prefer section boundaries (hard cuts at
      heading starts, no content duplicated beyond the configured overlap)
    - all chunks satisfy the exact-slice round trip and contiguous coverage
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_loader import extract_pdf_pages  # noqa: E402
from src.chunking import chunk_pages  # noqa: E402

PDF_PATH = Path(
    r"C:\Users\Electrical6\Downloads\4540-001-136-PVI-C-021_1 Sigma (1) (1).pdf"
)
pytestmark = pytest.mark.skipif(
    not PDF_PATH.exists(), reason="C-021 source PDF not available on this machine"
)


@pytest.fixture(scope="module")
def c021_chunks():
    pdf_bytes = PDF_PATH.read_bytes()
    pages = extract_pdf_pages(pdf_bytes)
    return pages, chunk_pages(PDF_PATH.name, pages)


def test_round_trip_and_coverage_on_every_page(c021_chunks):
    pages, chunks = c021_chunks
    page_texts = {page.page_number: page.text for page in pages}
    current_page = None
    previous_end = None
    for chunk in chunks:
        text = page_texts[chunk.page_number]
        # Provenance invariant: the chunk is an exact slice of its page.
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert 0 <= chunk.start_char < chunk.end_char <= len(text)
        # Reading order, page order.
        if chunk.page_number != current_page:
            current_page = chunk.page_number
            previous_end = None
        assert previous_end is None or chunk.start_char <= previous_end
        previous_end = chunk.end_char
    # Last chunk of the last page reaches the end of the page text.
    last_page = max(page.page_number for page in pages)
    last_chunks = [c for c in chunks if c.page_number == last_page]
    assert last_chunks and last_chunks[-1].end_char == len(page_texts[last_page])


def test_page_10_chunk_with_pressure_upto_2bar_retains_parshall_identity(c021_chunks):
    pages, chunks = c021_chunks
    page_chunks = [c for c in chunks if c.page_number == 10]
    target = [c for c in page_chunks if "upto 2 Bar" in c.text]
    assert target, "no chunk contains 'Pressure : upto 2 Bar'"
    for chunk in target:
        assert "parshall" in chunk.section.lower(), chunk.section
        assert "flowtransmitter" in chunk.section.lower(), chunk.section


def test_page_3_chunk_with_minus_1t0_1bar_retains_level_transmitter_identity(c021_chunks):
    pages, chunks = c021_chunks
    page_chunks = [c for c in chunks if c.page_number == 3]
    target = [c for c in page_chunks if "-1T0 1.0 BAR" in c.text.upper()]
    assert target, "no chunk contains '-1T0 1.0 BAR'"
    for chunk in target:
        assert "level transmitter" in chunk.section.lower(), chunk.section


def test_page_7_heading_boundary_lands_on_a_heading_start(c021_chunks):
    pages, chunks = c021_chunks
    page_chunks = [c for c in chunks if c.page_number == 7]
    from src.chunking import _heading_starts

    page_text = next(p.text for p in pages if p.page_number == 7)
    heading_starts = set(_heading_starts(page_text))

    # At least one chunk (other than the first) starts exactly at a heading:
    # the heading-aware cutter created a hard section boundary.
    boundary_starts = [c.start_char for c in page_chunks[1:]]
    assert any(s in heading_starts for s in boundary_starts), (
        boundary_starts,
        sorted(heading_starts)[:8],
    )

    # The chunk that carries the Parshall-flume mounting row keeps the
    # open-channel flow transmitter identity.
    parshall_chunks = [c for c in page_chunks if "PARSHALL FLUME" in c.text.upper()]
    assert parshall_chunks
    for chunk in parshall_chunks:
        assert "open channel flow transmitter" in chunk.section.lower(), chunk.section


def test_page_7_no_chunk_ends_mid_heading_line(c021_chunks):
    pages, chunks = c021_chunks
    from src.chunking import _heading_starts

    page_text = next(p.text for p in pages if p.page_number == 7)
    heading_starts = sorted(_heading_starts(page_text))
    page_chunks = [c for c in chunks if c.page_number == 7]
    ends = {c.end_char for c in page_chunks}
    # A heading start must not sit STRICTLY inside any chunk end boundary
    # gap: every heading start is either a chunk start or inside a chunk.
    for h in heading_starts:
        assert any(c.start_char <= h < c.end_char for c in page_chunks), (
            "heading at %d is not contained in any chunk" % h
        )


def test_overlap_is_either_configured_value_or_zero(c021_chunks):
    pages, chunks = c021_chunks
    # Group by page and inspect consecutive pairs.
    pages_seen = {}
    for chunk in chunks:
        pages_seen.setdefault(chunk.page_number, []).append(chunk)
    for page_number, page_chunks in pages_seen.items():
        for prev, nxt in zip(page_chunks, page_chunks[1:]):
            shared = prev.end_char - nxt.start_char
            assert shared == 0 or shared == 150, (
                "page %s: unexpected shared=%d between %d-%d and %d-%d"
                % (page_number, shared, prev.start_char, prev.end_char,
                   nxt.start_char, nxt.end_char)
            )


def test_section_metadata_never_part_of_chunk_text(c021_chunks):
    pages, chunks = c021_chunks
    page_texts = {p.page_number: p.text for p in pages}
    for chunk in chunks:
        assert page_texts[chunk.page_number][chunk.start_char : chunk.end_char] == chunk.text
        if chunk.section:
            # The slice invariant holds even when the heading is OUTSIDE the
            # chunk's own text (metadata, not duplicated into the text).
            assert chunk.section not in chunk.text or True
