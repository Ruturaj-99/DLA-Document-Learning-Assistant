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
from src.retrieval import (  # noqa: E402
    TOC_INTENT_BOOST,
    is_dotted_contents,
    is_table_of_contents,
    question_requests_toc,
    search_chunks,
)

SAMPLE_CHUNKS = [
    Chunk("manual-heater", 1, "The heater has a two-speed fan for warm air.", 0, 50),
    Chunk("manual-heater", 2, "To reset the heater, hold the power button for five seconds.", 55, 115),
    Chunk("garden-notes", 3, "Roses need sunlight and regular watering in spring.", 0, 50),
    Chunk("manual-heater", 5, "The filter should be cleaned every three months.", 120, 170),
]
EMBEDDED = embed_chunks(SAMPLE_CHUNKS)  # one time for the whole test file

# --- Stage 5F: table-of-contents fixtures -------------------------------------
TOC_TEXT = (
    "SL. NO. PARTICULARS PAGE NO.\n"
    "1 DATASHEET OF HEATER 3\n"
    "2 DRAWING OF HEATER 4 TO 6\n"
    "3 MAINTENANCE INSTRUCTIONS 7\n"
    "Page: 2 of 20"
)
SPEC_TEXT = (
    "TECHNICAL PARTICULARS\n"
    "1.00.00\n"
    ".01.00\n"
    ".02.00\n"
    "ACCURACY plus/minus 2%\n"
    "LEAST COUNT 5MM\n"
    "PROCESS CONNECTION 25 NB ANSI B 16.5 150 # RF FLANGED\n"
    "Page 1 of 2"
)
TOC_CHUNKS = [
    Chunk("manual-heater", 4, TOC_TEXT, 0, len(TOC_TEXT)),
    Chunk("manual-heater", 2, "To reset the heater, hold the power button for five seconds.", 200, 260),
    Chunk("manual-heater", 1, "The heater has a two-speed fan for warm air.", 270, 320),
    Chunk("manual-heater", 5, "The filter should be cleaned every three months.", 330, 380),
]
TOC_EMBEDDED = embed_chunks(TOC_CHUNKS)


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


# --- Stage 5F: table-of-contents detection -----------------------------------


def test_detects_table_of_contents_page():
    assert is_table_of_contents(TOC_TEXT) is True


def test_does_not_flag_technical_specification_page():
    assert is_table_of_contents(SPEC_TEXT) is False


def test_does_not_flag_ordinary_prose():
    assert is_table_of_contents(
        "To reset the heater, hold the power button for five seconds."
    ) is False


def test_does_not_flag_empty_text():
    assert is_table_of_contents("") is False
    assert is_table_of_contents("   \n  \n") is False


@pytest.mark.parametrize(
    "question",
    [
        "What is listed in the table of contents?",
        "Show me the contents page.",
        "List the sections in this document.",
        "Which page has the reflex level gauge datasheet?",
    ],
)
def test_explicit_toc_questions_are_recognised(question):
    assert question_requests_toc(question) is True


# --- Stage 5L: dotted-leader contents pages -------------------------------------
#
# Some manuals (measured: ODAMS page 2) format the table of contents as
# "Step 2: Adding Stations ............ 9" entries with no PAGE NO. column
# and no heading in most chunks. is_table_of_contents cannot see them, so
# TOC questions never got the intent boost and the QA gate rejected them.

DOTTED_TOC_TEXT = (
    "Table Contents\n"
    "Summary of Steps for Registration ................................................ 3\n"
    "Section A: Detailed Steps for Registration .................................... 4\n"
    "Step 1: Industry Registration ......................................................... 4\n"
    "Step 2: Adding Stations, Analyzer Details ................................. 9\n"
    "Section B: Miscellaneous Information .......................................... 19\n"
)


def test_dotted_contents_entries_are_detected():
    assert is_dotted_contents(DOTTED_TOC_TEXT) is True


def test_dotted_detector_rejects_prose_and_plain_tables():
    prose = (
        "Wait for the dots to appear on the screen.... then press 1\n"
        "The second line of prose explains the procedure in detail.\n"
        "A third line with an ellipsis... followed by more text here 12\n"
        "The fourth line finishes the paragraph as usual."
    )
    assert is_dotted_contents(prose) is False
    # the strict detector must NOT classify it as a TOC either
    assert is_table_of_contents(prose) is False


def test_dotted_contents_chunk_is_boosted_for_toc_questions():
    chunks = embed_chunks(
        [
            Chunk("manual", 2, "Technical specifications: accuracy plus/minus 2%.", 300, 350),
            Chunk("manual", 1, DOTTED_TOC_TEXT, 0, len(DOTTED_TOC_TEXT)),
        ]
    )
    results = search_chunks(
        "Which sections are listed in the table of contents?", chunks, top_k=5
    )
    assert results[0].page_number == 1
    # The boosted score must clear the QA gate (0.25) even when the raw
    # cosine is low - that is the whole point of the dotted-contents fix.
    assert results[0].score >= 0.25, results[0].score


def test_dotted_contents_chunk_is_not_hidden_from_technical_questions():
    """Boost-only: dotted contents pages stay visible for normal questions."""
    chunks = embed_chunks(
        [
            Chunk("manual", 2, "Technical specifications: accuracy plus/minus 2%.", 300, 350),
            Chunk("manual", 1, DOTTED_TOC_TEXT, 0, len(DOTTED_TOC_TEXT)),
        ]
    )
    results = search_chunks("What is the specified accuracy?", chunks, top_k=5)
    assert any(result.page_number == 1 for result in results), (
        "dotted contents chunks are never hidden (filtering unchanged)"
    )
    # ...and the strict detector still does not classify them as TOC:
    assert not is_table_of_contents(DOTTED_TOC_TEXT)


@pytest.mark.parametrize(
    "question",
    [
        "How do I reset the heater?",
        "What is the specified accuracy of the float and board gauge?",
        "What is the specified float size for the float and board gauge?",
        "What is the least count and process connection specified for the reflex level gauge?",
        "What are the optional transmitter's output signal, supply voltage and maximum load impedance?",
    ],
)
def test_technical_questions_are_not_toc_requests(question):
    assert question_requests_toc(question) is False


def test_normal_question_excludes_toc_chunks():
    results = search_chunks("How do I reset the heater?", TOC_EMBEDDED, top_k=10)
    assert results, "non-TOC chunks should still be returned"
    assert all(result.page_number != 4 for result in results), (
        "the contents page must not appear for a normal technical question"
    )
    assert all(result.text != TOC_TEXT for result in results)


def test_explicit_toc_question_keeps_toc_searchable():
    results = search_chunks(
        "What is listed in the table of contents?", TOC_EMBEDDED, top_k=4
    )
    assert results
    assert results[0].page_number == 4, (
        f"the contents page should rank first, got: "
        f"{[(r.page_number, r.score) for r in results]}"
    )


def test_explicit_toc_question_applies_bounded_intent_boost():
    from src.embeddings import cosine_similarity, embed_question

    question = "What is listed in the table of contents?"
    question_vector = embed_question(question)
    toc_chunk = next(
        chunk for chunk in TOC_EMBEDDED if is_table_of_contents(chunk.text)
    )
    raw_score = cosine_similarity(question_vector, toc_chunk.vector)

    results = search_chunks(question, TOC_EMBEDDED, top_k=4)
    boosted = next(result for result in results if result.page_number == 4)
    assert boosted.score == pytest.approx(min(1.0, raw_score + TOC_INTENT_BOOST))
    assert boosted.score <= 1.0

    # Only TOC chunks are boosted; ordinary passages keep their raw score.
    heater = next(result for result in results if result.page_number != 4)
    heater_chunk = next(
        chunk for chunk in TOC_EMBEDDED if chunk.start_char == heater.start_char
    )
    assert heater.score == pytest.approx(
        cosine_similarity(question_vector, heater_chunk.vector)
    )


def test_only_toc_chunks_fall_back_to_unfiltered_list():
    only_toc = embed_chunks([Chunk("contents-only", 1, TOC_TEXT, 0, len(TOC_TEXT))])
    results = search_chunks("How do I reset the heater?", only_toc, top_k=5)
    assert len(results) == 1
    assert results[0].page_number == 1
