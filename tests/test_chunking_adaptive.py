"""Stage 5G: adaptive (semantic) chunking tests for src/chunking.py.

The old chunker cut every `chunk_size` characters; the new one groups
paragraphs, headings and table rows into semantic blocks and packs those
into chunks. These tests pin the behaviour that matters downstream:

    - chunk boundaries land at natural edges, never mid-sentence-line
    - a heading leads the chunk that contains its content (sticky)
    - table rows stay together and keep their heading
    - full page coverage with no gaps and exact-slice round trips
    - overlap is exactly the configured value or 0 (hard heading cut)
    - every chunk stays within the configured character budget

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/test_chunking_adaptive.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import (  # noqa: E402
    _semantic_blocks,
    split_page_text,
)
from src.pdf_loader import PageContent  # noqa: E402


def windows_of(text: str, chunk_size: int, overlap: int, heading_starts=None):
    return split_page_text(
        text, chunk_size=chunk_size, overlap=overlap,
        heading_starts=heading_starts,
    )


def assert_universal_invariants(text: str, windows, chunk_size: int, overlap: int):
    """The guarantees every chunking mode must satisfy."""
    # Full coverage: first window starts at 0, the last reaches the end,
    # starts never move backward, and no character is skipped.
    assert windows, "a non-empty page must produce chunks"
    assert windows[0][0] == 0
    assert windows[-1][1] == len(text)
    starts = [start for start, _, _ in windows]
    assert starts == sorted(starts)
    for (_, prev_end, _), (next_start, _, _) in zip(windows, windows[1:]):
        assert next_start <= prev_end, "coverage gap between chunks"
    for start, end, chunk_text in windows:
        # Exact slice round trip and budget bounds.
        assert text[start:end] == chunk_text
        assert 0 <= start < end <= len(text)
        assert end - start <= chunk_size + overlap  # merged-tail allowance
        # No mid-line cut at the END of a chunk: a chunk ends after a
        # line break / space, or at the very end of the page.
        if end < len(text):
            assert text[end - 1].isspace() or text[end] == " ", (
                f"chunk {start}:{end} ends mid-line at {text[end - 5:end]!r}"
            )
        # Shared content is exactly the configured overlap or zero
        # (hard cut at a heading boundary) - never partial duplication.
    for (_, prev_end, _), (next_start, _, _) in zip(windows, windows[1:]):
        shared = prev_end - next_start
        assert shared in (0, overlap), f"unexpected shared={shared}"


def test_paragraphs_are_kept_intact():
    """A short paragraph is never cut mid-thought: whole paragraphs travel
    together, and a chunk boundary falls between paragraphs."""
    paragraph_a = "The blue valve feeds the condenser through a manual bypass. " * 2
    paragraph_b = "Every shift must log the discharge pressure reading twice. " * 2
    paragraph_c = "Replace the gasket whenever the housing is opened. " * 2
    text = paragraph_a + "\n\n" + paragraph_b + "\n\n" + paragraph_c
    windows = windows_of(text, chunk_size=300, overlap=40)

    assert_universal_invariants(text, windows, 300, 40)
    # Each paragraph fits inside one chunk, so every boundary must fall
    # on the "\n" line ends between paragraphs, not inside a paragraph.
    ends = [end for _, end, _ in windows[:-1]]
    for end in ends:
        assert text[end - 1] == "\n", f"chunk ends mid-paragraph at {end}"


def test_heading_leads_its_own_chunk():
    heading = "TECHNICAL PARTICULARS OF LEVEL TRANSMITTER"
    heading_start = len("Introduction prose line one.\nIntroduction prose line two.\n")
    rows = "\n".join(f"Range : {value} meter" for value in ("0-2", "0-5", "0-10"))
    text = (
        "Introduction prose line one.\n"
        "Introduction prose line two.\n"
        + heading
        + "\n"
        + rows
        + "\n"
    )

    windows = windows_of(text, chunk_size=200, overlap=30)
    assert_universal_invariants(text, windows, 200, 30)

    # The heading starts a chunk (sticky), and the chunk that contains
    # the heading also contains the rows that follow it.
    heading_chunks = [w for w in windows if w[0] == heading_start]
    assert heading_chunks, "no chunk starts exactly at the heading"
    assert heading_chunks[0][2].startswith(heading)
    assert "Range : 0-2" in heading_chunks[0][2]
    # No chunk ENDS with the heading (a heading never trails a chunk).
    for _, end, chunk_text in windows:
        stripped = chunk_text.rstrip()
        if stripped.endswith(heading):
            assert end == len(text), "heading trails a chunk that has a successor"


def test_table_rows_stay_together_and_with_their_heading():
    heading = "SENSOR SPECIFICATIONS"
    rows = "\n".join(
        f"{index} Parameter{index} : value-{index} unit" for index in range(1, 13)
    )
    text = heading + "\n" + rows + "\n"
    windows = windows_of(text, chunk_size=280, overlap=30)
    assert_universal_invariants(text, windows, 280, 30)

    # No chunk cuts a row in half: every chunk end falls after a newline
    # (already checked) and every row appears complete in one chunk.
    for index in range(1, 13):
        row = f"{index} Parameter{index} : value-{index} unit"
        assert any(row in chunk for _, _, chunk in windows), f"row {index} was split"


def test_blocks_tile_the_page_text():
    text = (
        "HEADING ONE\n"
        "first paragraph line\n"
        "second paragraph line\n"
        "\n"
        "1 Label : value\n"
        "2 Label : value\n"
        "\n"
        "HEADING TWO\n"
        "closing paragraph\n"
    )
    blocks = _semantic_blocks(text, use_headings=True)
    assert blocks[0][0] == 0
    assert blocks[-1][1] == len(text)
    for (_, prev_end, _), (next_start, _, _) in zip(blocks, blocks[1:]):
        assert prev_end == next_start, "semantic blocks must tile the text"
    kinds = [kind for _, _, kind in blocks]
    assert kinds == ["heading", "prose", "row", "heading", "prose"]


def test_long_paragraph_is_split_at_word_edges_not_mid_word():
    text = ("The compressor surge margin degrades slowly over time. " * 30).strip()
    windows = windows_of(text, chunk_size=120, overlap=20)
    assert_universal_invariants(text, windows, 120, 20)
    assert len(windows) > 3


def test_headingless_page_behaviour_unchanged_by_heading_starts_argument():
    """heading_starts=[] only disables stickiness; on a page without
    headings it must produce identical windows."""
    text = "plain prose without any heading lines at all " * 40
    assert windows_of(text, 120, 25) == windows_of(
        text, 120, 25, heading_starts=[]
    )


def test_heading_stickiness_can_be_disabled():
    heading = "APPENDIX SECTION"
    body = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 8
    heading_start = len(body) + 1  # body, newline, then the heading line
    text = body + "\n" + heading + "\n" + body
    auto = windows_of(text, 200, 30)
    disabled = windows_of(text, 200, 30, heading_starts=[])
    # With stickiness enabled, some chunk starts exactly at the heading;
    # with it disabled, headings are treated as ordinary content.
    assert any(start == heading_start for start, _, _ in auto)
    assert any(chunk_text.startswith(heading) for _, _, chunk_text in auto)
    assert not any(
        chunk_text.startswith(heading) for _, _, chunk_text in disabled[1:]
    )


def test_realistic_datasheet_page_chunking():
    """A C-021-style page: heading, numbered subsection, spec rows."""
    text = (
        "DATASHEET FOR LEVEL TRANSMITTER (CHP RUN-OFF BHEL DOC NO. - PE-V0\n"
        "1.00.00 |TECHNICAL PARTICULARS\n"
        "1 [AMBIENT TEMPERATURE(MAX/MIN) 55 CENTIGRADE\n"
        "2 [PROCESS TEMPERATURE 90 CENTIGRADE\n"
        "3 [CALIBRATION 0-5 METER WKW\n"
        "4 [MATERIAL OF WETTED PARTS SS316\n"
        "Pressure : upto 2 Bar\n"
        "Output : 4-20 mA two wire\n"
    )
    page = PageContent(page_number=7, text=text)
    from src.chunking import chunk_pages

    chunks = chunk_pages("c021-test", [page])
    assert chunks
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.section.startswith("DATASHEET FOR LEVEL TRANSMITTER")
    # The heading and its rows share a chunk (or at least the heading
    # chunk reaches into the rows).
    heading_chunks = [c for c in chunks if "1.00.00 |TECHNICAL" in c.text]
    assert heading_chunks
    assert "AMBIENT TEMPERATURE" in heading_chunks[0].text


def test_very_long_single_line_never_exceeds_budget():
    text = "word " * 500  # one 2500-character line, no newlines
    windows = windows_of(text, chunk_size=100, overlap=20)
    assert_universal_invariants(text, windows, 100, 20)
