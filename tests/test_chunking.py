"""Stage 3A: tests for src/chunking.py.

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

Tests cover: chunk metadata, page boundaries, the overlap rule, the exact
slice round-trip (page_text[start_char:end_char] == chunk.text), empty-page
handling, and configuration validation.
"""

import sys
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import Chunk, chunk_pages, split_page_text  # noqa: E402
from src.pdf_loader import PageContent  # noqa: E402

# Small sizes so a few test strings exercise multiple windows.
SIZE = 50
OVERLAP = 10


def make_page(text: str, page_number: int = 1) -> PageContent:
    return PageContent(page_number=page_number, text=text)


def test_short_page_becomes_single_chunk_covering_full_text():
    page_text = "Short page."
    (window,) = split_page_text(page_text, chunk_size=SIZE, overlap=OVERLAP)
    start, end, text = window
    assert (start, end) == (0, len(page_text))
    assert text == page_text


def test_long_page_is_split_into_multiple_chunks_in_order():
    page_text = ("The model learns from labeled training data. " * 8).strip()
    windows = split_page_text(page_text, chunk_size=SIZE, overlap=OVERLAP)
    assert len(windows) > 1
    starts = [s for s, _, _ in windows]
    # Chunks appear in reading order and never go backward.
    assert starts == sorted(starts)
    assert starts[0] == 0


def test_chunks_stay_within_page_bounds():
    page_text = "abc " * 200  # 800 characters of repetitive text
    for start, end, _ in split_page_text(page_text, chunk_size=SIZE, overlap=OVERLAP):
        assert 0 <= start < end <= len(page_text)
        assert SIZE >= end - start >= 1


def test_consecutive_chunks_overlap():
    page_text = ("The quick brown fox jumps. " * 20).strip()
    windows = split_page_text(page_text, chunk_size=SIZE, overlap=OVERLAP)
    for (_, prev_end, _), (next_start, _, _) in zip(windows, windows[1:]):
        assert prev_end > next_start, "neighboring chunks must share characters"
        # Overlap is bounded: `end - overlap` sliding means we never carry
        # MORE overlap than configured (windows may shrink at word edges,
        # which only ever increases the shared region a little).
        shared = prev_end - next_start
        assert shared <= OVERLAP + SIZE // 2  # generous sanity ceiling


def test_exact_slice_round_trip_against_original_page_text():
    """The key Stage 3A guarantee: every chunk is an exact slice of its page."""
    page_text = "Transformation techniques  " * 25 + " final sentence."
    page = make_page(page_text)
    chunks = chunk_pages("doc-1", [page], chunk_size=SIZE, overlap=OVERLAP)
    for chunk in chunks:
        assert page.text[chunk.start_char:chunk.end_char] == chunk.text


def test_chunk_metadata_is_preserved():
    pages = [
        make_page("Page one with enough text to be interesting for chunking.", 1),
        make_page("alpha beta gamma delta epsilon zeta eta theta iota kappa", 5),
    ]
    chunks = chunk_pages("manual-machine-vision", pages, chunk_size=SIZE, overlap=OVERLAP)
    for chunk in chunks:
        assert chunk.document_id == "manual-machine-vision"
        assert chunk.page_number in (1, 5)
    assert any(c.page_number == 5 for c in chunks)


def test_empty_page_yields_no_chunks():
    assert chunk_pages("doc-1", [make_page("   ")], chunk_size=SIZE, overlap=OVERLAP) == []
    # A page without text must not break the stream of other pages.
    pages = [make_page(""), make_page("Hello readable page.", 2)]
    chunks = chunk_pages("doc-1", pages, chunk_size=SIZE, overlap=OVERLAP)
    assert [c.page_number for c in chunks] == [2]


def test_invalid_configuration_raises():
    with pytest.raises(ValueError):
        split_page_text("some text", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        split_page_text("some text", chunk_size=10, overlap=10)  # overlap must be < size
    with pytest.raises(ValueError):
        split_page_text("some text", chunk_size=10, overlap=-1)
