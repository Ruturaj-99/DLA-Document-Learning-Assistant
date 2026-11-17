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

from src.chunking import (  # noqa: E402
    Chunk,
    chunk_pages,
    detect_page_section,
    heading_class,
    split_page_text,
)
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


# --- Section heading detection ------------------------------------------------


def test_real_heading_lines_are_classified():
    from src.chunking import heading_class

    # Title class: instrument/document level headings.
    assert heading_class("DATASHEET FOR LEVEL TRANSMITTER (CHP RUN-OFF BHEL DOC NO.") == "title"
    assert heading_class("OPEN CHANNEL / PARSHALL FLUME FLOWTRANSMITTER") == "title"
    assert heading_class("ULTRASONIC LEVEL TRANSMITTER") == "title"
    # Subsection class: numbered sub-sections and block titles.
    assert heading_class("1.00.00 |SITE CONDITIONS") == "subsection"
    assert heading_class(".02.04 [SENSOR (LEVEL MEASURING UNIT)") == "subsection"
    assert heading_class("TRANSMITTER - ULFM") == "subsection"
    assert heading_class("DECODING SHEET") == "subsection"
    # Not headings at all: spec rows, footers, prose, block labels.
    assert heading_class("Pressure : upto 2 Bar") is None  # label : value row
    assert heading_class("8 (OPERATING PRESSURE -1T0 1.0 BAR") is None  # indexed row
    assert heading_class("10 OUTPUT 4..20mA /22,8...26,4VDC / HART (2wires)") is None
    assert heading_class("MOC : ABS") is None  # colon + all caps, but a row
    assert heading_class("A | PBT") is None  # single-letter decoding code
    assert heading_class("4540-001-136-PVI-C-021/01/CAT-I/Page: 6 of 11") is None
    assert heading_class("Page 1 of 1") is None
    assert heading_class("50f11") is None
    assert heading_class("The heater must be drained before transport.") is None
    assert heading_class("SENSOR") is None  # single-word block label


def test_detect_page_section_returns_topmost_title():
    page_text = (
        "4540-001-136-PVI-C-021/01/CAT-/Page: 2 of 11\n"
        "DATASHEET FOR LEVEL TRANSMITTER (CHP RUN-OFF BHEL DOC NO. - PE-V0\n"
        "1.00.00 |SITE CONDITIONS\n"
        "1 [AMBIENT TEMPERATURE(MAX/MIN) 55 CENTIGRADE\n"
    )
    section = detect_page_section(page_text)
    assert "LEVEL TRANSMITTER" in section


def test_late_all_caps_lines_do_not_hijack_the_section():
    """A note near the end of the page must not become the section of the
    last chunks - the page's topmost title wins (requirement: never assign
    unrelated headings)."""
    title = "DATASHEET FOR LEVEL TRANSMITTER (CHP RUN-OFF BHEL DOC NO.\n"
    body = "the sensor is a two wire loop powered device with hart support\n" * 10
    note = "NOTE - ELECTRONIC MODULE WITH DISPLAY FROM BM TECHNOLOGY, ITALY\n"
    chunks = chunk_pages("doc", [make_page(title + body + note)], chunk_size=200, overlap=30)
    assert chunks
    assert {chunk.section for chunk in chunks} == {title.strip()}


def test_pages_without_heading_get_empty_section():
    page_text = "plain prose without any heading lines at all " * 10
    chunks = chunk_pages("doc-1", [make_page(page_text)], chunk_size=100, overlap=20)
    assert all(chunk.section == "" for chunk in chunks)


# --- Heading-aware chunk boundaries -------------------------------------------


def test_split_cuts_at_heading_start_and_heading_leads_next_chunk():
    heading = "OPEN CHANNEL / PARSHALL FLUME FLOWTRANSMITTER"
    before = "sensor measures the channel flow with an ultrasonic pulse beam " * 8
    after = "the transmitter is mounted on top of the parshall flume directly " * 8
    text = before + "\n" + heading + "\n" + after
    heading_start = len(before) + 1

    windows = split_page_text(text, chunk_size=300, overlap=40)

    # Some window must END exactly where the heading starts...
    assert any(end == heading_start for _, end, _ in windows)
    # ...and the following window must BEGIN with the heading (hard cut,
    # no overlap pulled across the section boundary).
    cuts = [(start, text_piece) for start, _, text_piece in windows if start == heading_start]
    assert cuts and cuts[0][1].startswith(heading)


def test_heading_cut_duplicates_nothing_and_discards_nothing():
    heading = "TECHNICAL PARTICULARS"
    body_a = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 6
    body_b = "one two three four five six seven eight nine ten " * 6
    text = body_a + heading + "\n" + body_b
    windows = split_page_text(text, chunk_size=200, overlap=30)

    # Full coverage, in order, no gaps: windows[0] starts at 0, starts
    # never move backward, and the last window reaches the end.
    assert windows[0][0] == 0
    starts = [start for start, _, _ in windows]
    assert starts == sorted(starts)
    assert windows[-1][1] == len(text)
    for (_, prev_end, _), (next_start, _, _) in zip(windows, windows[1:]):
        # Every character is kept: the next window never skips ahead.
        assert next_start <= prev_end
        # Shared content is either the configured overlap or ZERO (a hard
        # heading cut) - never anything in between or duplicated beyond.
        shared = prev_end - next_start
        assert shared in (0, 30)


def test_chunk_pages_attaches_section_metadata_without_touching_text():
    heading = "DATASHEET FOR OPEN CHANNEL FLOW TRANSMITTER (CHP RUN-OFF"
    body = "the flow transmitter is mounted on top of the parshall flume " * 20
    page = make_page(heading + "\n" + body)
    chunks = chunk_pages("c021", [page], chunk_size=200, overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        # Section is metadata, present on every chunk of the page...
        assert chunk.section == heading
        # ...and the exact-slice invariant is untouched.
        assert page.text[chunk.start_char : chunk.end_char] == chunk.text
        # Chunks cut from the middle of the table still carry the identity.
        if "parshall flume" in chunk.text:
            assert "OPEN CHANNEL FLOW TRANSMITTER" in chunk.section


def test_headingless_pages_keep_the_previous_split_behaviour():
    """Auto-detection must not change splits when no heading is present."""
    page_text = "the quick brown fox jumps over the lazy dog. " * 30
    auto = split_page_text(page_text, chunk_size=120, overlap=25)
    manual = split_page_text(page_text, chunk_size=120, overlap=25, heading_starts=[])
    assert auto == manual
