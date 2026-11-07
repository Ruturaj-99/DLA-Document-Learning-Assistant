"""Stage 3B: tests for src/embeddings.py.

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v

Notes:
    - The FIRST run downloads the all-MiniLM-L6-v2 model (~90 MB) and can
      take a few minutes; later runs are fast (model is cached locally).
    - No LLM, no vector database, no internet API is used in the logic.
"""

import math
import sys
from pathlib import Path

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import Chunk  # noqa: E402
from src.embeddings import (  # noqa: E402
    EXPECTED_DIMENSIONS,
    cosine_similarity,
    embed_chunks,
    embed_question,
    embed_texts,
)

EXAMPLE_CHUNKS = [
    Chunk("doc-1", 3, "The heating system has a two-speed fan.", 0, 40),
    Chunk("doc-1", 4, "How do I reset the device? Press power for 5 seconds.", 10, 62),
    Chunk("doc-2", 1, "Completely unrelated gardening text about roses.", 0, 40),
]


def test_vector_count_matches_chunk_count():
    embedded = embed_chunks(EXAMPLE_CHUNKS)
    assert len(embedded) == len(EXAMPLE_CHUNKS)


def test_vectors_are_valid_float_lists():
    embedded = embed_chunks(EXAMPLE_CHUNKS)
    for item in embedded:
        assert len(item.vector) == EXPECTED_DIMENSIONS
        for value in item.vector:
            assert math.isfinite(value)  # no NaN / infinity
        # A real embedding is a direction, not all zeros.
        assert any(abs(value) > 1e-6 for value in item.vector)


def test_all_source_metadata_is_preserved():
    embedded = embed_chunks(EXAMPLE_CHUNKS)
    for original, result in zip(EXAMPLE_CHUNKS, embedded):
        assert result.document_id == original.document_id
        assert result.page_number == original.page_number
        assert result.text == original.text
        assert result.start_char == original.start_char
        assert result.end_char == original.end_char


def test_related_sentences_score_higher_than_unrelated():
    """Relative comparison, NOT a fixed magic number."""
    base = "How do I reset the smoke detector?"
    related = "To reset the smoke detector, hold the button for five seconds."
    unrelated = "Tonight the local football team plays in the stadium."

    vectors = embed_texts([base, related, unrelated])
    related_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])

    assert related_score > unrelated_score, (
        f"related ({related_score:.3f}) must beat unrelated ({unrelated_score:.3f})"
    )


def test_question_embedding_matches_its_own_passage_best():
    """Same check from the question side, like the final app will use it."""
    question = embed_question("How do I reset the smoke detector?")
    passages = embed_texts(
        [
            "To reset the smoke detector, hold the button for five seconds.",
            "The roses in the garden bloom in spring.",
            "Wipe the printer head with a soft cloth.",
        ]
    )
    scores = [cosine_similarity(question, p) for p in passages]
    assert scores[0] == max(scores)


def test_empty_input_is_handled():
    assert embed_chunks([]) == []
    assert embed_texts([]) == []


def test_identical_texts_embed_identically():
    """Determinism: same input text twice -> equal vectors."""
    a, b = embed_texts(["Determinism test sentence.", "Determinism test sentence."])
    assert a == b
