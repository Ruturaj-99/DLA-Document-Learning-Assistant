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

The two offsets are guaranteed to be true slices: writing
`page_text[start_char:end_char]` always reproduces `chunk.text`.
That round-trip guarantee is what lets a later stage verify a cited
passage against the original page text before visualizing it.
"""

from dataclasses import dataclass

from src.pdf_loader import PageContent

# Sensible defaults; can be overridden per call for different document types.
DEFAULT_CHUNK_SIZE = 1200  # characters per chunk
DEFAULT_OVERLAP = 150  # characters repeated between neighboring chunks


@dataclass
class Chunk:
    """One searchable passage, anchored to its source location.

    Invariants:
        - text == page_text[start_char:end_char]  (exact slice)
        - 0 <= start_char < end_char <= len(page_text)
    """

    document_id: str
    page_number: int
    text: str
    start_char: int
    end_char: int


def split_page_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[tuple[int, int, str]]:
    """Split one page's text into (start_char, end_char, text) windows.

    The loop keeps a sliding window of `chunk_size` characters and moves it
    forward by `chunk_size - overlap` each step (condensed windows move at
    least one character, never backward). When a window would cut a word in
    half (not at the very end of the text), we back the window's end off to
    the previous space so chunks break at word boundaries.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1 character.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    if not text.strip():
        return []  # nothing searchable on this page (e.g. an empty or image-only page)

    windows: list[tuple[int, int, str]] = []
    text_length = len(text)
    start = 0

    while start < text_length:
        # One-block pages become a single window covering the whole text.
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # Avoid splitting a word: move `end` back to the last space
            # within the current window, if any.
            space_index = text.rfind(" ", start, end)
            if space_index > start:
                end = space_index + 1  # keep the space as part of the chunk

        windows.append((start, end, text[start:end]))

        if end >= text_length:
            break  # the page is covered; no more windows needed

        # Slide forward. The max(...) guarantees progress even when the
        # word-boundary hack above shortened the window.
        start = max(end - overlap, start + 1)

    return windows


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
        page). Pages without text simply contribute no chunks.
    """
    chunks: list[Chunk] = []
    for page in pages:
        for start, end, text in split_page_text(page.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    page_number=page.page_number,
                    text=text,
                    start_char=start,
                    end_char=end,
                )
            )
    return chunks
