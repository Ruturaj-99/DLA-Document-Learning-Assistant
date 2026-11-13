"""Benchmark: where the 'Rearranging pages...' seconds actually go.

Run from the project root:
    .venv\\Scripts\\python scripts\\bench_rearrange.py
    .venv\\Scripts\\python scripts\\bench_rearrange.py --embed   (also time embeddings)

It builds a synthetic 24-page native PDF (like a typical document), then times
every stage that a page rearrange touches, comparing the OLD flow (rewrite the
PDF bytes -> everything re-computes) with the NEW fast path (bytes untouched ->
only the display mapping changes, thumbnails are cache hits).
"""

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf

from src.chunking import chunk_pages
from src.pdf_editor import apply_page_order
from src.page_view import render_page_with_boxes
from src.pdf_loader import extract_pdf_pages

PAGES = 24
CARD_DPI = 120
REVERSED_ORDER = list(range(PAGES, 0, -1))


def build_synthetic_pdf(pages: int) -> bytes:
    """A plain native PDF with one text line per page (no OCR involved)."""
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Page {index + 1} content line", fontsize=14)
    payload = document.tobytes()
    document.close()
    return payload


def timed(label: str, function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    seconds = time.perf_counter() - start
    print(f"  {label:<52} {seconds:8.2f} s")
    return value, seconds


def main() -> None:
    run_embeddings = "--embed" in sys.argv
    print(f"Benchmark on a {PAGES}-page native PDF (card DPI {CARD_DPI})\n")

    pdf = build_synthetic_pdf(PAGES)

    print("OLD flow (drop rewrites the working copy -> every cache misses):")
    _, t_reorder = timed("1. apply_page_order (pypdf rewrite)", apply_page_order, pdf, REVERSED_ORDER)
    reordered = apply_page_order(pdf, REVERSED_ORDER).pdf_bytes
    _, t_extract = timed("2. re-extract every page", extract_pdf_pages, reordered, False)
    _, t_chunk = timed("3. re-chunk", chunk_pages, "bench.pdf", extract_pdf_pages(reordered, False))
    if run_embeddings:
        from src.embeddings import embed_chunks

        _, t_embed = timed(
            "4. re-embed every chunk (CPU sentence-transformers)",
            embed_chunks,
            chunk_pages("bench.pdf", extract_pdf_pages(reordered, False)),
        )
    else:
        t_embed = 0.0
        print("  4. re-embed every chunk                           (skipped: pass --embed)")
    _, t_thumbs = timed(
        f"5. re-render all {PAGES} preview cards",
        lambda: [render_page_with_boxes(reordered, number, dpi=CARD_DPI) for number in range(1, PAGES + 1)],
    )
    old_total = t_reorder + t_extract + t_chunk + t_embed + t_thumbs
    print(f"  {'OLD total (what one drop costs today)':<52} {old_total:8.2f} s\n")

    print("NEW fast path (drop only reorders the display mapping):")
    _, t_map = timed("1. update session page_order (pure Python)", list, REVERSED_ORDER)
    # Thumbnails are rendered from the UNCHANGED bytes at mapped pages ->
    # identical calls to what the app already rendered -> st.cache_data hits.
    _, t_hit = timed(
        f"2. all {PAGES} preview cards (simulated cache hits)",
        lambda: [render_page_with_boxes(pdf, number, dpi=CARD_DPI) for number in range(1, PAGES + 1)],
    )
    print("     (in the app these calls return instantly from st.cache_data;")
    print("      the number above is the one-time cost of the FIRST render)")
    new_total = t_map
    print(f"  {'NEW cost per drop':<52} {new_total:8.2f} s  (+ cached thumbnails)")


if __name__ == "__main__":
    main()
