"""Semantic search (retrieval).

Stage 3C of the AI Document Learning Assistant.

In simple words: when the user asks a question, we turn the question into
the same kind of number list (vector) as the document chunks. Then we measure
how similar the question is to every chunk. The chunks with the best scores
are the most likely places where the answer lives.

This module is the last piece before an LLM would ever see anything:
question in -> ranked list of passages out, each one remembering exactly
where it came from (document, page, character positions).

IMPORTANT: this is still a "simple search over a list" implementation -
there is NO vector database yet. Sorting a Python list keeps our tests
honest and shows exactly how ranking works. A later stage may swap the
storage underneath without touching this interface.

Table-of-contents handling (Stage 5F):
    A table-of-contents page is short and repeats the exact instrument
    names a user is likely to type, so it can outrank the datasheet that
    actually contains the value. We detect TOC-shaped chunks and:
        - hide them from normal technical queries, so the real
          specification pages win;
        - keep them (and boost them) when the question explicitly asks
          for the contents/index, so the TOC stays searchable on demand;
        - also boost dotted-leader contents entries ("Step 2 ... 9"),
          which the strict detector cannot see (Stage 5L) - boost only,
          they are never hidden from technical questions;
        - fall back to the unfiltered list if a document consists only
          of a TOC, instead of returning nothing.
    Detection is text-based on purpose: no chunk size, overlap, embedding,
    extraction or score-threshold changes are needed.

Hybrid retrieval (Stage 5J):
    One retriever is not enough. Vector search matches MEANING but blurs
    exact values; BM25 keyword search matches EXACT words, part numbers
    and numeric values but misses paraphrases. We run BOTH over the same
    (Stage 5F-filtered) candidate list and combine the two rankings with
    weighted Reciprocal Rank Fusion: every chunk earns
    weight / (RRF_K + rank) from each list it appears in, and the sum
    decides the final order. BM25's job here is to pull exact-term
    chunks UP into the ranking, while the vector ranking still leads -
    on real PDFs BM25 alone is noisier than vector search.
    Fusion is document-agnostic: no hardcoded terms or per-document
    rules.

Boilerplate demotion (Stage 5K):
    Repeated header/footer lines (copyright notices, version footers)
    make many near-identical chunks cluster in BOTH retriever rankings
    and swarm the fused top-k ahead of the real specification passage.
    A normalized line unchanged across >= BOILERPLATE_LINE_THRESHOLD
    candidate chunks is page furniture; each chunk's fused score is
    scaled by 1 - BOILERPLATE_PENALTY * (its repeated-character share).
    Demotion is skipped when the question explicitly asks for the TOC.

    The `score` on each SearchResult KEEPS its cosine-similarity meaning
    (plus the Stage 5F intent boost for TOC questions), so the QA score
    gate and the UI keep working exactly as before; the fused ranking
    decides WHICH passages are returned and in what order. Returned
    scores are monotonic non-increasing to stay consistent with that
    order.
"""

import re
from dataclasses import dataclass

import numpy as np

from src.bm25 import BM25Index
from src.embeddings import EmbeddedChunk, cosine_similarity, embed_question

# --- Table-of-contents detection ------------------------------------------------
# A TOC page pairs a "PARTICULARS ... PAGE NO." header with numbered entries
# that end in a page number (or page range). Technical datasheet pages also
# contain words like "TECHNICAL PARTICULARS" and long dotted section-number
# lists, but they do NOT pair that header with page-numbered entries - so the
# detector stays conservative and technical pages are not filtered.
_TOC_HEADING_PATTERN = re.compile(
    r"\b(?:table\s+of\s+contents|contents|particulars)\b", re.IGNORECASE
)
_TOC_PAGE_HEADER_PATTERN = re.compile(
    r"\bpage\s*(?:no\.?|number)\b", re.IGNORECASE
)
_TOC_ENTRY_PATTERN = re.compile(
    r"^\s*\d+[.)]?\s+\S.{2,}?\s+\d+(?:\s*(?:-|to)\s*\d+)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Questions that explicitly want the contents/index. These keep the TOC
# searchable; everything else treats TOC chunks as navigation noise.
_TOC_QUESTION_PATTERN = re.compile(
    r"(?:"
    r"\btable\s+of\s+contents\b|"
    r"\bcontents?\s+(?:page|list|table)\b|"
    r"\btoc\b|"
    r"\bindex\b|"
    r"\b(?:list|show|what\s+are|which)\b[^?.]{0,40}\b(?:sections?|chapters?|contents?)\b|"
    r"\b(?:which|what)\s+page\b|"
    r"\bpage\s+(?:number|no\.?)\b|"
    r"\bdocument\s+(?:map|outline)\b"
    r")",
    re.IGNORECASE,
)

# Added to TOC chunks only when the question explicitly asks for the TOC.
# This is an intent boost, not a similarity-threshold change: it keeps a
# low-scoring contents page from being hidden by the normal 0.25 QA gate.
TOC_INTENT_BOOST = 0.35

# Reciprocal Rank Fusion (Stage 5J): a chunk ranked r-th in a list
# contributes 1/(RRF_K + r), weighted per retriever. 60 is the standard
# RRF constant from the RRF paper. The vector ranking carries the larger
# weight because on real PDFs BM25's ranking is noisier (questions share
# generic vocabulary with boilerplate), while BM25's job is to pull the
# exact term/number chunks up - not to lead the ordering. Both weights
# are corpus-level constants, not document-specific rules.
RRF_K = 60
RRF_VECTOR_WEIGHT = 0.7
RRF_BM25_WEIGHT = 0.3

# Boilerplate demotion (Stage 5K): real PDFs repeat the same header/footer
# lines (copyright notices, "Datasheet v1.0 ...", drawing titles) on many
# pages. Those chunks match generic question words, cluster together in BOTH
# retriever rankings, and can swarm the fused top-k ahead of the actual
# specification passage. Fix: a normalized line that appears UNCHANGED in
# BOILERPLATE_LINE_THRESHOLD or more candidate chunks is page furniture; the
# share of such repeated characters in a chunk scales its fused score by
# (1 - BOILERPLATE_PENALTY * share). Chunks built from real content keep
# factor 1.0; hybrid chunks (one repeated header line + spec rows) are only
# mildly demoted because their repeated-character share is small.
# Purely statistical: no line list is tuned to any document.
BOILERPLATE_LINE_THRESHOLD = 4
BOILERPLATE_PENALTY = 0.3


def is_table_of_contents(text: str) -> bool:
    """True when `text` looks like a table of contents / index page.

    Conservative by design: it requires a TOC-style header AND at least
    three numbered entries ending in a page number. A datasheet page with
    "TECHNICAL PARTICULARS", dotted section numbers (".01.00") and a
    "Page 1 of 2" footer therefore does NOT match.
    """
    if not text or not text.strip():
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False

    joined = "\n".join(lines)
    if len(_TOC_ENTRY_PATTERN.findall(joined)) < 3:
        return False

    has_toc_heading = bool(_TOC_HEADING_PATTERN.search(joined))
    has_page_header = bool(_TOC_PAGE_HEADER_PATTERN.search(joined))
    if has_toc_heading and has_page_header:
        return True

    # An explicit "Table of contents" heading is enough when entries exist.
    return bool(
        re.search(r"\btable\s+of\s+contents\b", joined, re.IGNORECASE)
    )


def question_requests_toc(question: str) -> bool:
    """True when the user explicitly asks for the contents/index/navigation."""
    if not question or not question.strip():
        return False
    return bool(_TOC_QUESTION_PATTERN.search(question))


# Some manuals format the table of contents as dotted-leader entries
# ("Step 2: Adding Stations ............ 9") with no "PAGE NO." column and
# sometimes not even a contents heading in the same chunk - a style the
# conservative is_table_of_contents detector cannot see (measured on the
# ODAMS manual: all three page-2 chunks stayed invisible, so a TOC question
# never received the Stage 5F intent boost and the 0.25 gate rejected it).
# This detector is deliberately used ONLY for the intent boost, not for
# hiding chunks from technical questions: these chunks are harmless in
# normal rankings (they never dominated a fused top-k in the benchmark),
# and filtering them would shift the BM25 statistics of the remaining
# candidates and reorder unrelated questions.
_DOTTED_TOC_ENTRY_PATTERN = re.compile(
    r"^\s*\S.{2,}?\s\.{4,}\s*\d+(?:\s*(?:-|to)\s*\d+)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def is_dotted_contents(text: str) -> bool:
    """True when a chunk is made of dotted-leader contents entries.

    Conservative: needs at least BOILERPLATE-free plain lines and at least
    three "text .... page" entry lines. Prose, tables and figure captions
    contain no runs of four or more dots followed by a trailing page
    number, so they never match.
    """
    if not text or not text.strip():
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    return len(_DOTTED_TOC_ENTRY_PATTERN.findall(text)) >= 3


def _normalize_line(line: str) -> str:
    """Lowercase, punctuation-free, whitespace-collapsed form of one line."""
    return " ".join("".join(ch.lower() for ch in line if ch.isalnum()).split())


def boilerplate_factors(texts: list[str]) -> list[float]:
    """Demotion factor per text, 1.0 for clean chunks, <1.0 for boilerplate.

    A normalized line repeated UNCHANGED in BOILERPLATE_LINE_THRESHOLD or
    more different texts is page furniture (copyright notices, datasheet
    version footers, drawing titles). Each text's factor is
    1 - BOILERPLATE_PENALTY * (share of its characters that sit on such
    repeated lines), so a chunk made purely of repeated headers is scaled
    the most, a spec chunk with one inherited header line barely at all,
    and a chunk with no repeated lines not at all.
    """
    if len(texts) < BOILERPLATE_LINE_THRESHOLD:
        return [1.0] * len(texts)

    line_documents: dict[str, int] = {}
    parsed_lines: list[list[str]] = []
    for text in texts:
        lines = [ln for ln in (_normalize_line(l) for l in text.splitlines()) if ln]
        parsed_lines.append(lines)
        for line in set(lines):
            line_documents[line] = line_documents.get(line, 0) + 1

    repeated = {
        line
        for line, count in line_documents.items()
        if count >= BOILERPLATE_LINE_THRESHOLD
    }
    if not repeated:
        return [1.0] * len(texts)

    factors: list[float] = []
    for lines in parsed_lines:
        if not lines:
            factors.append(1.0)
            continue
        repeated_chars = sum(len(line) for line in lines if line in repeated)
        total_chars = sum(len(line) for line in lines)
        share = repeated_chars / total_chars
        factors.append(1.0 - BOILERPLATE_PENALTY * share)
    return factors


@dataclass
class SearchResult:
    """One chunk from the search results, plus its similarity score.

    Attributes:
        score: cosine similarity between the question and the chunk,
            roughly -1..1, with ~0.5+ meaning "clearly related meaning".
        All other fields are copied 1:1 from the source chunk.
    """

    document_id: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    score: float


def search_chunks(
    question: str, embedded_chunks: list[EmbeddedChunk], top_k: int = 5
) -> list[SearchResult]:
    """Find the top_k chunks most relevant to the question (hybrid search).

    Args:
        question: what the user wants to know, e.g. "How do I reset the
            smoke detector?".
        embedded_chunks: the document chunks with their vectors
            (output of embeddings.embed_chunks()).
        top_k: how many results to return. Larger values are capped at the
            number of chunks available (we never invent results).

    Returns:
        SearchResult items, best first, ranked by Reciprocal Rank Fusion
        of a vector (meaning) ranking and a BM25 (exact keyword) ranking,
        with repeated-header boilerplate chunks demoted (Stage 5K).
        `score` carries the cosine similarity (plus the Stage 5F TOC
        intent boost when the question asks for the contents/index) so
        the QA score gate keeps its meaning; returned scores are
        monotonic non-increasing. An empty question or empty chunk list
        returns an empty list.

        TOC behaviour: table-of-contents chunks are omitted from normal
        technical queries (BOTH retrievers see the filtered list). If the
        question explicitly asks for the contents/index, TOC chunks are
        kept, outrank everything else in the fusion, and get the intent
        boost on their displayed score (capped at 1.0). If a document
        contains nothing but TOC chunks, the unfiltered list is returned
        rather than an empty result.

    Raises:
        ValueError: if top_k is smaller than 1.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1 (you asked how many results to show).")
    if not question.strip():
        return []
    if not embedded_chunks:
        return []

    wants_toc = question_requests_toc(question)

    # --- Stage 5F: TOC filtering applies to BOTH retrievers ----------------
    candidates = list(embedded_chunks)
    if not wants_toc:
        # Normal technical question: hide navigation pages so the actual
        # datasheet/specification passages fill the top-k. If EVERY chunk
        # is a TOC (e.g. a contents-only upload), keep the original list
        # rather than answering "no passages".
        without_toc = [
            chunk for chunk in candidates if not is_table_of_contents(chunk.text)
        ]
        if without_toc:
            candidates = without_toc

    # --- 1. Vector ranking (meaning) ----------------------------------------
    question_vector = np.asarray(embed_question(question))
    vector_scores = [
        cosine_similarity(question_vector, chunk.vector) for chunk in candidates
    ]
    vector_order = sorted(
        range(len(candidates)), key=lambda index: (-vector_scores[index], index)
    )
    vector_rank = {index: rank for rank, index in enumerate(vector_order, start=1)}

    # --- 2. BM25 ranking (exact words, part numbers, values) ----------------
    bm25_index = BM25Index([chunk.text for chunk in candidates])
    bm25_order = bm25_index.ranked(question)
    bm25_rank = {index: rank for rank, index in enumerate(bm25_order, start=1)}

    # --- 3. Reciprocal Rank Fusion ------------------------------------------
    fused: dict[int, float] = {}
    for index in range(len(candidates)):
        score = 0.0
        if index in vector_rank:
            score += RRF_VECTOR_WEIGHT / (RRF_K + vector_rank[index])
        if index in bm25_rank:
            score += RRF_BM25_WEIGHT / (RRF_K + bm25_rank[index])
        fused[index] = score

    if wants_toc:
        # Explicit contents/index question: every contents chunk outranks
        # every other chunk (the bonus exceeds the best fusion score a
        # non-contents chunk can reach, RRF_VECTOR_WEIGHT + RRF_BM25_WEIGHT
        # over rank 1); among contents chunks the fused ranking decides.
        # This keeps the TOC searchable on demand. is_dotted_contents adds
        # dotted-leader contents pages that the strict detector cannot see
        # (Stage 5L). Boilerplate demotion is skipped here: when the user
        # asks for navigation pages, repeated header lines ARE the
        # requested content.
        intent_bonus = (RRF_VECTOR_WEIGHT + RRF_BM25_WEIGHT) / (RRF_K + 1)
        for index, chunk in enumerate(candidates):
            if is_table_of_contents(chunk.text) or is_dotted_contents(chunk.text):
                fused[index] += intent_bonus
    else:
        # Stage 5K: repeated header/footer lines make boilerplate chunks
        # cluster in both retriever rankings and swarm the fused top-k.
        # Scale each chunk's fused score by its repeated-character share.
        factors = boilerplate_factors([chunk.text for chunk in candidates])
        for index, factor in enumerate(factors):
            if factor < 1.0:
                fused[index] *= factor

    final_order = sorted(
        range(len(candidates)), key=lambda index: (-fused[index], index)
    )[:top_k]

    # --- 4. Results: cosine meaning preserved, order from fusion ------------
    results: list[SearchResult] = []
    running_max = float("inf")
    for index in final_order:
        chunk = candidates[index]
        score = vector_scores[index]
        if wants_toc and (
            is_table_of_contents(chunk.text) or is_dotted_contents(chunk.text)
        ):
            score = min(1.0, score + TOC_INTENT_BOOST)
        score = min(score, running_max)  # stay consistent with fused order
        running_max = score
        results.append(
            SearchResult(
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                score=score,
            )
        )
    return results
