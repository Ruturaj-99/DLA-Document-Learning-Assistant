"""Text chunking with source metadata.

Stage 3A of the AI Document Learning Assistant.

Long pages are split into smaller, overlapping chunks so that later stages
can search for precise passages. Every chunk remembers exactly where it
came from:

    document_id  - which uploaded document it belongs to
    page_number  - 1-based page inside that document
    text         - the chunk text itself
    start_char   - index of the first chunk character in the PAGE text
    end_char     - index just past the last character (Python slice style)
    section      - the page's primary section heading (e.g. the instrument
                   a datasheet page describes), or "" when no heading was
                   detected. METADATA ONLY: it is never part of `text`.

The two offsets are guaranteed to be true slices: writing
`page_text[start_char:end_char]` always reproduces `chunk.text`.
That round-trip guarantee is what lets a later stage verify a cited
passage against the original page text before visualizing it.

Two chunking behaviours keep section context intact:

    1. Section detection - every page is scanned for heading-like lines
       (short, mostly-uppercase lines such as "DATASHEET FOR LEVEL
       TRANSMITTER ..."). The topmost title-class heading becomes every
       chunk's `section`, so a chunk cut from the middle of a spec table
       still carries the instrument it belongs to.

Adaptive segmentation (Stage 5G) replaces the old "cut every 1200
characters" behaviour:

    2. Semantic blocks - each page is first divided into natural blocks:
       a paragraph (consecutive prose lines), a heading, or a run of
       tabular lines (row-index rows and "Label : value" rows). Blank
       lines end a block, and a heading always begins one.
    3. Structure-aware packing - blocks are packed into chunks up to a
       SOFT target (75% of the chunk budget), so a chunk boundary lands
       at the start of the next paragraph / table row / heading instead
       of mid-thought. Consecutive headings and the page's leading lines
       are merged so no tiny fragments are emitted, and a single long
       paragraph is still split (at line, then sentence/space edges)
       when it alone exceeds the budget.
    4. Sticky headings - a heading never trails a chunk: closing a chunk
       right before a heading makes the NEXT chunk start with that
       heading, so "TECHNICAL PARTICULARS" stays glued to the rows that
       follow it. Across such a heading boundary the chunks meet hard
       (no overlap); every other boundary repeats up to `overlap`
       characters of lead-in context, which also keeps facts that sit
       near the end of one chunk inside the embedding window of the next.

The 1200-character budget and 150-character overlap are unchanged as
UPPER BOUNDS; typical chunks are now shorter and end at natural edges.

Everything else about the pipeline (embeddings, retrieval, the LLM
prompt) is untouched by this module.
"""

import re
from dataclasses import dataclass

from src.pdf_loader import PageContent

# Sensible defaults; can be overridden per call for different document types.
DEFAULT_CHUNK_SIZE = 1200  # characters per chunk
DEFAULT_OVERLAP = 150  # characters repeated between neighboring chunks

# --- Section heading detection -----------------------------------------------
# Conservative on purpose: it is far better to MISS a heading (chunk gets
# section "") than to assign an unrelated one (e.g. a spec row or a page
# footer). The rules below were tuned against real datasheet pages such as
# the C-021 instrument document, where spec rows look like
#     "8 (OPERATING PRESSURE -1T0 1.0 BAR"   (row index + value)
#     "Pressure : upto 2 Bar"                (label : value)
# and headings look like
#     "DATASHEET FOR LEVEL TRANSMITTER (CHP RUN-OFF ... DATED - 04.11.2025"
#     "OPEN CHANNEL / PARSHALL FLUME FLOWTRANSMITTER"
#     "1.00.00 |SITE CONDITIONS"

_HEADING_MAX_CHARS = 160  # titles may carry doc numbers/rev lines
_HEADING_MIN_CHARS = 4  # single words like "SENSOR" are block labels, not sections
_MIN_TITLE_WORDS = 3  # title-class headings name a thing with >= 3 words
_UPPER_RATIO = 0.6  # heading text is mostly uppercase

# A spec row, not a heading: "10 OUTPUT ...", "13| zero and Span ...",
# "1 [AMBIENT TEMPERATURE ...".
_ROW_INDEX_PATTERN = re.compile(r"^\d+[\s\[|:]")
# A numbered sub-section: "3.00.00 |TECHNICAL PARTICULARS", ".02.04 [SENSOR ...".
_SECTION_NUMBER_PATTERN = re.compile(r"^\.?\d+\.\d+(?:\.\d+)?")
# Words = runs of letters with at least 2 characters (so single-letter
# decoding codes like "A | PBT" never qualify).
_WORD_PATTERN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def heading_class(line: str) -> str | None:
    """Classify one line: "title", "subsection", or None (not a heading).

    "title"      - instrument/document level, e.g. "DATASHEET FOR LEVEL
                   TRANSMITTER (CHP RUN-OFF ...)". Used as chunk identity.
    "subsection" - numbered sub-sections ("1.00.00 |SITE CONDITIONS") and
                   block titles ("TRANSMITTER - ULFM", "DECODING SHEET").
                   Valid split points, but never chunk identity on their own.
    """
    line = line.strip()
    if not _HEADING_MIN_CHARS <= len(line) <= _HEADING_MAX_CHARS:
        return None
    if ":" in line:
        return None  # "Pressure : upto 2 Bar" and friends are spec rows
    if _ROW_INDEX_PATTERN.match(line):
        return None  # "8 (OPERATING PRESSURE ..." is a table row
    words = _WORD_PATTERN.findall(line)
    if len(words) < 2:
        return None  # single tokens ("SENSOR") are block labels
    alphabetic = [character for character in line if character.isalpha()]
    uppercase = [character for character in alphabetic if character.isupper()]
    if not alphabetic or len(uppercase) / len(alphabetic) < _UPPER_RATIO:
        return None  # normal prose / mixed-case lines
    if _SECTION_NUMBER_PATTERN.match(line):
        return "subsection"
    if len(words) >= _MIN_TITLE_WORDS:
        return "title"
    return "subsection"


def _heading_starts(text: str) -> list[int]:
    """Character offsets where a heading line begins in `text`.

    The offset points at the first non-whitespace character of the line,
    so a boundary cut makes the next chunk start with the heading itself.
    """
    starts: list[int] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")  # keep the real terminator length
        stripped = line.strip()
        if stripped and heading_class(stripped) is not None:
            starts.append(offset + (len(line) - len(line.lstrip())))
        offset += len(raw_line)
    return starts


def detect_page_section(text: str) -> str:
    """The page's primary section heading, or "" when none is detected.

    The TOPMOST title-class heading wins. This is what makes a datasheet
    page's whole table carry its instrument name (e.g. every chunk of the
    Parshall flume page knows it belongs to the flume transmitter), and it
    keeps later all-caps lines (notes, tag tables) from hijacking the
    identity of chunks near the end of the page.
    """
    for raw_line in text.splitlines(keepends=True):
        stripped = raw_line.strip()
        if stripped and heading_class(stripped) == "title":
            return stripped
    return ""


@dataclass
class Chunk:
    """One searchable passage, anchored to its source location.

    Invariants:
        - text == page_text[start_char:end_char]  (exact slice)
        - 0 <= start_char < end_char <= len(page_text)
        - section is metadata about the page, NEVER part of text
          ("" when the page has no detectable section heading)
    """

    document_id: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    section: str = ""


def _line_kind(stripped: str, use_headings: bool) -> str:
    """Classify one non-blank line: "heading", "row", or "prose"."""
    if use_headings and heading_class(stripped) is not None:
        return "heading"
    if _ROW_INDEX_PATTERN.match(stripped) or ":" in stripped:
        return "row"
    return "prose"


def _semantic_blocks(
    text: str, use_headings: bool
) -> list[tuple[int, int, str]]:
    """Divide a page into contiguous (start, end, kind) semantic blocks.

    A block is a paragraph (consecutive prose lines), a single heading
    line, or a run of tabular lines (row-index rows and "Label : value"
    rows). Blank lines end the current block. The returned blocks TILE
    the text exactly: the first block starts at offset 0, blank separator
    lines attach to the block before them, and the last block ends at
    len(text) - so packing the blocks can never lose or skip characters.
    """
    blocks: list[tuple[int, int, str]] = []
    cur_start = cur_end = None
    cur_kind: str | None = None
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_end = offset + len(raw_line)
        stripped = raw_line.strip()
        if not stripped:
            if cur_kind is not None:  # blank line ends the current block
                blocks.append((cur_start, cur_end, cur_kind))
                cur_start = cur_end = None
                cur_kind = None
        else:
            kind = _line_kind(stripped, use_headings)
            if cur_kind == kind and kind in ("prose", "row"):
                cur_end = line_end  # extend the paragraph / table run
            else:
                if cur_kind is not None:
                    blocks.append((cur_start, cur_end, cur_kind))
                cur_start, cur_end, cur_kind = offset, line_end, kind
        offset = line_end
    if cur_kind is not None:
        blocks.append((cur_start, cur_end, cur_kind))
    if not blocks:
        return []

    # Tile: leading blanks join the first block, blank runs between
    # blocks join the preceding one, trailing text joins the last.
    tiled: list[tuple[int, int, str]] = []
    for index, (start, _end, kind) in enumerate(blocks):
        end = (
            blocks[index + 1][0] if index + 1 < len(blocks) else len(text)
        )
        tiled.append((0 if index == 0 else start, end, kind))
    return tiled


def _split_long_line(
    text: str, start: int, end: int, piece_cap: int
) -> list[tuple[int, int]]:
    """Split a single line longer than piece_cap at space edges.

    Last resort: a hard cut for one unbreakable word longer than the
    whole budget (never possible for real text, but the invariant
    "piece <= piece_cap" must hold unconditionally).
    """
    pieces: list[tuple[int, int]] = []
    piece_start = start
    while end - piece_start > piece_cap:
        limit = piece_start + piece_cap
        space = text.rfind(" ", piece_start, limit)
        cut = space + 1 if space != -1 else limit
        pieces.append((piece_start, cut))
        piece_start = cut
    if end > piece_start:
        pieces.append((piece_start, end))
    return pieces


def _split_oversized(
    text: str, start: int, end: int, piece_cap: int, overlap: int
) -> list[tuple[int, int]]:
    """Split one block longer than piece_cap into pieces <= piece_cap.

    Cuts first at line ends (never mid-row or mid-sentence-line), then,
    for a single line still too long, at spaces. A trailing piece no
    longer than `overlap` merges back into the previous piece (when it
    still fits the cap) so the next chunk never opens with a sliver.
    """
    pieces: list[tuple[int, int]] = []
    piece_start = start
    piece_length = 0
    offset = start
    while offset < end:
        newline = text.find("\n", offset, end)
        line_end = end if newline == -1 else newline + 1
        line_length = line_end - offset
        if piece_length and piece_length + line_length > piece_cap:
            pieces.append((piece_start, offset))
            piece_start, piece_length = offset, 0
            line_length = line_end - offset
        if line_length > piece_cap:
            pieces.extend(_split_long_line(text, offset, line_end, piece_cap))
            piece_start, piece_length = line_end, 0
        else:
            piece_length += line_length
        offset = line_end
    if piece_length:
        pieces.append((piece_start, end))
    if len(pieces) >= 2:
        prev_start, prev_end = pieces[-2]
        last_start, last_end = pieces[-1]
        if (
            last_end - last_start <= overlap
            and prev_end - prev_start + last_end - last_start <= piece_cap
        ):
            pieces[-2:] = [(prev_start, last_end)]
    return pieces


def _pack_blocks(
    text: str,
    blocks: list[tuple[int, int, str]],
    chunk_size: int,
    overlap: int,
    sticky_headings: bool,
) -> list[tuple[int, int, str]]:
    """Pack semantic blocks into (start, end, text) chunk windows.

    Packing rules, in order:
      - a block longer than chunk_size is pre-split into pieces;
      - a heading never trails a chunk: when a heading arrives, the
        current chunk closes (hard cut) so the NEXT chunk starts with
        the heading, and consecutive headings merge together;
      - a heading-only chunk absorbs the block (or first piece) that
        follows, so a heading is never emitted alone;
      - content packs up to the SOFT target (75% of the budget) and a
        chunk shorter than chunk_size // 4 absorbs the next piece;
      - when a chunk closes, the next one repeats exactly `overlap`
        characters of lead-in - or none (hard join) when that would
        overflow the budget or the previous chunk is shorter than
        `overlap` itself;
      - a trailing window no longer than max(overlap, chunk_size // 6)
        merges into the previous chunk (<= chunk_size + overlap).
    """
    soft = max(1, int(chunk_size * 0.75))
    floor = max(1, chunk_size // 4)
    # Split pieces leave room for the overlap lead-in, so a chunk that
    # opens with a lead-in never exceeds chunk_size.
    piece_cap = max(1, chunk_size - overlap)

    pieces: list[tuple[int, int, bool]] = []  # (start, end, is_heading)
    for block_start, block_end, kind in blocks:
        is_heading = kind == "heading" and sticky_headings
        if block_end - block_start > piece_cap:
            for piece_start, piece_end in _split_oversized(
                text, block_start, block_end, piece_cap, overlap
            ):
                pieces.append((piece_start, piece_end, is_heading))
        else:
            pieces.append((block_start, block_end, is_heading))

    windows: list[tuple[int, int, str]] = []
    cur_start = cur_end = None
    cur_heading_only = False

    for piece_start, piece_end, is_heading in pieces:
        if cur_start is None:
            cur_start, cur_end = piece_start, piece_end
            cur_heading_only = is_heading
            continue

        cur_length = cur_end - cur_start
        piece_length = piece_end - piece_start

        if is_heading:
            if cur_heading_only:
                cur_end = piece_end  # consecutive headings merge
            else:
                # Sticky: close so the heading leads the NEXT chunk.
                windows.append((cur_start, cur_end, text[cur_start:cur_end]))
                cur_start, cur_end = piece_start, piece_end
                cur_heading_only = True
            continue

        if cur_heading_only:
            if cur_length + piece_length <= chunk_size:
                cur_end = piece_end  # the heading absorbs what follows
                cur_heading_only = False
            else:
                windows.append((cur_start, cur_end, text[cur_start:cur_end]))
                cur_start, cur_end = piece_start, piece_end
                cur_heading_only = is_heading  # False here
            continue

        if (cur_length < floor and cur_length + piece_length <= chunk_size) or (
            cur_length + piece_length <= soft
        ):
            cur_end = piece_end
            continue

        # Close the chunk; the next one opens with an overlap lead-in.
        windows.append((cur_start, cur_end, text[cur_start:cur_end]))
        if piece_start - overlap >= cur_start:
            cur_start = piece_start - overlap
        else:
            cur_start = piece_start  # hard join (shared = 0)
        cur_end = piece_end

    if cur_start is not None:
        if (
            windows
            and cur_end - cur_start <= max(overlap, chunk_size // 6)
            and cur_end - windows[-1][0] <= chunk_size + overlap
        ):
            prev_start = windows[-1][0]
            windows[-1] = (prev_start, cur_end, text[prev_start:cur_end])
        else:
            windows.append((cur_start, cur_end, text[cur_start:cur_end]))

    return windows


def split_page_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    heading_starts: list[int] | None = None,
) -> list[tuple[int, int, str]]:
    """Split one page's text into (start_char, end_char, text) windows.

    Adaptive segmentation (Stage 5G): the page is first divided into
    semantic blocks - paragraphs, heading lines and runs of tabular rows
    - which are then packed into chunks up to a soft target of 75% of
    `chunk_size`. Chunk boundaries therefore land at the start of the
    next paragraph / table row / heading instead of mid-thought, while
    `chunk_size` remains a hard upper bound (except a merged trailing
    window, which may reach chunk_size + overlap).

    Sticky headings: a heading never trails a chunk. When a heading is
    reached, the current chunk closes so the next chunk STARTS with the
    heading; the two chunks meet hard (no overlap) there. Every other
    boundary repeats exactly `overlap` characters of lead-in, so a fact
    near the end of one chunk stays inside the embedding window of the
    next - or no lead-in at all when that would overflow the budget.

    Pass `heading_starts=[]` to disable heading stickiness entirely;
    omit the argument (or pass a non-empty list) to enable it.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1 character.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    if not text.strip():
        return []  # nothing searchable on this page (e.g. an empty or image-only page)

    sticky_headings = heading_starts is None or bool(heading_starts)
    blocks = _semantic_blocks(text, sticky_headings)
    if not blocks:
        return []
    return _pack_blocks(text, blocks, chunk_size, overlap, sticky_headings)


def chunk_pages(
    document_id: str,
    pages: list[PageContent],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Turn extracted pages into anchored chunks ready for later stages.

    Args:
        document_id: A stable identifier for the uploaded document
            (e.g. its filename or a generated id - decided later).
        pages: The output of src.pdf_loader.extract_pdf_pages().
        chunk_size: Maximum chunk length in characters.
        overlap: Number of characters neighboring chunks share.

    Returns:
        Chunks in reading order (page order, then left-to-right within the
        page). Pages without text simply contribute no chunks. Every chunk
        carries the page's primary section heading in `.section` ("" when
        none was detected).
    """
    chunks: list[Chunk] = []
    for page in pages:
        section = detect_page_section(page.text)
        heading_starts = _heading_starts(page.text)
        for start, end, text in split_page_text(
            page.text, chunk_size, overlap, heading_starts
        ):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    page_number=page.page_number,
                    text=text,
                    start_char=start,
                    end_char=end,
                    section=section,
                )
            )
    return chunks
