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
"""

from dataclasses import dataclass

import numpy as np

from src.embeddings import EmbeddedChunk, cosine_similarity, embed_question


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
    """Find the top_k chunks most similar to the question.

    Args:
        question: what the user wants to know, e.g. "How do I reset the
            smoke detector?".
        embedded_chunks: the document chunks with their vectors
            (output of embeddings.embed_chunks()).
        top_k: how many results to return. Larger values are capped at the
            number of chunks available (we never invent results).

    Returns:
        SearchResult items sorted from most similar to least similar.
        An empty question or empty chunk list returns an empty list.

    Raises:
        ValueError: if top_k is smaller than 1.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1 (you asked how many results to show).")
    if not question.strip():
        return []
    if not embedded_chunks:
        return []

    question_vector = np.asarray(embed_question(question))

    scored: list[SearchResult] = []
    for chunk in embedded_chunks:
        score = cosine_similarity(question_vector, chunk.vector)
        scored.append(
            SearchResult(
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                score=score,
            )
        )

    # Sort by score, best first. reverse=True means "biggest number on top".
    scored.sort(key=lambda result: result.score, reverse=True)
    return scored[:top_k]
