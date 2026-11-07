"""Stage 3C: tests for src/retrieval.py.

Run from the project root with:
    .venv\\Scripts\\python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

# Make "src" importable when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import Chunk  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.retrieval import search_chunks  # noqa: E402

SAMPLE_CHUNKS = [
    Chunk("manual-heater", 1, "The heater has a two-speed fan for warm air.", 0, 50),
    Chunk("manual-heater", 2, "To reset the heater, hold the power button for five seconds.", 55, 115),
    Chunk("garden-notes", 3, "Roses need sunlight and regular watering in spring.", 0, 50),
    Chunk("manual-heater", 5, "The filter should be cleaned every three months.", 120, 170),
]
EMBEDDED = embed_chunks(SAMPLE_CHUNKS)  # one time for the whole test file


def test_returns_requested_number_of_results():
    results = search_chunks("How do I reset the heater?", EMBEDDED, top_k=2)
    assert len(results) == 2


def test_results_ranked_by_similarity():
    results = search_chunks("How do I reset the heater?", EMBEDDED, top_k=3)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True), "highest score must come first"


def test_relevant_passage_ranks_above_unrelated():
    results = search_chunks("How do I reset the heater?", EMBEDDED, top_k=3)
    assert any("reset the heater" in result.text for result in results[:1]), (
        f"top result should be the reset passage, got: {results[0].text}"
    )


def test_scores_are_in_a_sane_range():
    results = search_chunks("Roses in the garden", EMBEDDED, top_k=4)
    for result in results:
        assert -1.0 <= result.score <= 1.0


def test_metadata_is_preserved():
    results = search_chunks("clean the filter", EMBEDDED, top_k=4)
    matching = [r for r in results if "filter" in r.text]
    assert matching, "the filter passage should appear in results"
    result = matching[0]
    assert result.document_id == "manual-heater"
    assert result.page_number == 5
    assert result.start_char == 120
    assert result.end_char == 170


def test_top_k_larger_than_collection_returns_all():
    assert len(EMBEDDED) == 4
    results = search_chunks("anything", EMBEDDED, top_k=100)
    assert len(results) == 4


def test_empty_chunks_returns_empty_list():
    assert search_chunks("How do I reset the heater?", [], top_k=5) == []


def test_empty_question_returns_empty_list():
    assert search_chunks("   ", EMBEDDED, top_k=5) == []


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        search_chunks("question", EMBEDDED, top_k=0)
    with pytest.raises(ValueError):
        search_chunks("question", EMBEDDED, top_k=-3)
