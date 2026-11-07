"""Performance benchmark: a SYNTHETIC 200-page PDF through the whole pipeline.

This is a tool, not a test and not a demo: all page content is generated
here (repeating placeholder sentences), so no "real document" is claimed.

Run from the project root with:
    .venv\\Scripts\\python scripts/perf_test_200_pages.py

It reports the real time for each stage:
    PDF extraction -> chunking -> embedding -> one semantic search -> one RAG answer
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_pages  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.pdf_loader import extract_pdf_pages  # noqa: E402
from src.qa import answer_question  # noqa: E402
from src.ollama import ask_ollama, ollama_available  # noqa: E402

PAGE_COUNT = 200

# A small pool of synthetic "manual" sentences. Content repeats on purpose;
# this measures SPEED, not uniqueness.
SENTENCES = [
    "The two-speed fan switches automatically in cold weather.",
    "Clean the filter with warm water every three months.",
    "To reset the unit hold the power button for five seconds.",
    "The display flashes twice when the timer is armed.",
    "Annual maintenance requires checking the hose seals.",
    "Do not obstruct the air intake during operation.",
]

# --- 1. Build the synthetic PDF -------------------------------------------
def escape(text):
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_synthetic_pdf(page_count: int) -> bytes:
    objects = [
        {"body": "<< /Type /Catalog /Pages 2 0 R >>"},   # object 1
        {"body": None},                                  # object 2: pages tree
        {"body": "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"},  # 3
    ]
    page_texts = []
    for page_number in range(1, page_count + 1):
        text = (
            " ".join(SENTENCES) + f" Page marker {page_number} of {page_count}."
        )
        page_texts.append(text)
        page_obj = len(objects) + 1
        content_obj = page_obj + 1
        stream = "BT /F1 12 Tf 72 720 Td (" + escape(text) + ") Tj ET"
        objects.append({"body": (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_obj} 0 R /Resources << /Font << /F1 3 0 R >> >>"
        )})
        objects.append({"body": (
            f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"
        )})
    objects[1]["body"] = (
        f"<< /Type /Pages /Kids [{ ' '.join(f'{i} 0 R' for i in range(4, len(objects) + 1, 2)) }]"
        f" /Count {page_count} >>"
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj_num, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{obj_num} 0 obj\n{obj['body']}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


# --- 2. Measure every stage ---------------------------------------------------
def main() -> None:
    print(f"Building a synthetic {PAGE_COUNT}-page PDF in memory...")
    pdf_bytes = build_synthetic_pdf(PAGE_COUNT)
    pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
    print(f"PDF size: {pdf_size_mb:.1f} MB (synthetic, generated locally)")

    t0 = time.perf_counter()
    pages = extract_pdf_pages(pdf_bytes)
    t_extract = time.perf_counter() - t0
    chars = sum(len(p.text) for p in pages)
    print(f"1. Extraction : {t_extract:6.2f} s  ({len(pages)} pages, {chars:,} characters)")

    t0 = time.perf_counter()
    chunks = chunk_pages("synthetic-manual", pages)
    t_chunk = time.perf_counter() - t0
    print(f"2. Chunking   : {t_chunk:6.2f} s  ({len(chunks)} chunks)")

    t0 = time.perf_counter()
    embedded = embed_chunks(chunks)
    t_embed = time.perf_counter() - t0
    print(f"3. Embedding  : {t_embed:6.2f} s  ({len(embedded)} vectors)")

    if ollama_available():
        t0 = time.perf_counter()
        bundle = answer_question(
            "How often is the filter cleaned and with what?",
            embedded,
            ask=ask_ollama,
        )
        t_search_and_answer = time.perf_counter() - t0
        print(f"4. Search + answer: {t_search_and_answer:6.2f} s")
        print("   Question: How often is the filter cleaned and with what?")
        print(f"   Answer: {bundle.answer[:200]}")
        if bundle.sources:
            top = bundle.sources[0]
            print(f"   Top source: page {top.page_number}, score {top.score:.3f}")
    else:
        print("4. Search + answer: SKIPPED (Ollama or llama3.2:3b not available)")


if __name__ == "__main__":
    main()
