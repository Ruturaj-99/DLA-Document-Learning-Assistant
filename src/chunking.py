from dataclasses import dataclass

from src.pdf_loader import PageContent


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
